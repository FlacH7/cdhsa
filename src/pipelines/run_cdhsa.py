"""
src.pipelines.run_cdhsa - Orchestrator for the full CD-HSA pipeline
===================================================================

Provides a high-level interface that runs all steps (A through D)
with sensible defaults, plus a dataclass for configuration.

Usage (programmatic, original API preserved)::::

    from src.pipelines.run_cdhsa import CDHSAConfig, run_cdhsa, CDHSAResult
    cfg = CDHSAConfig(fixed_rank=10, a6_n_null=100, bc_n_perm=5000)
    result = run_cdhsa(X, L, cfg)
    print(result.summary())

Usage (CLI - builds Hankel matrices from super-subjects)::::

    # 60 subjects partidos en 3 super-sujetos de 20 cada uno
    python -m src.pipelines.run_cdhsa \\
        --session session1 \\
        --tasks eyesclosed eyesopen \\
        --n-super-subjects 3 \\
        --total-subjects 60 \\
        --t-start 0 --t-end 300 \\
        --L 10 \\
        --l-freq 1.0 --h-freq 40.0 \\
        --hankel-depth 250 \\
        --fixed-rank 10 --a6-n-null 100 --bc-n-perm 5000

Pipeline interno
-----------------
Se particionan los ``total_subjects`` sujetos en ``n_super_subjects``
super-sujetos de tamano ``total_subjects // n_super_subjects`` cada uno.
Cada super-sujeto es la concatenacion temporal de sus sujetos
individuales (via ``load_super_subject_eeg``).

Para cada par (super-sujeto, condicion)::::

    1. Cargar y concatenar EEG  -> load_super_subject_eeg()
    2. Filtro pasa-banda        -> extract_filtered_data_matrix()
       (X_filtered: n_channels x n_times, centrada a media cero)
    3. Matriz de Hankel         -> _build_multivariate_hankel()
       (H: (n_channels * depth) x (n_times - depth + 1))
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import mne

# =====================================================================
# CDHSAConfig
# =====================================================================

@dataclass
class CDHSAConfig:
    """Configuration for the full CD-HSA pipeline."""
    fixed_rank: int = 10
    rank_method: str = "fixed"
    rmax: int = 20
    n_blocks: int = 4
    repro_threshold: float = 0.80
    repro_strategy: str = "consecutive"
    max_common: int = 30
    prevalence_quantile: float = 0.10
    # --- rank_method='variance': criterio X% de varianza explicada ---
    var_explained: float = 0.99
    var_max_rank: int = 50
    var_min_rank: int = 1
    a6_max_common: int = 0
    a6_n_folds: int = 5
    a6_n_null: int = 100
    a6_alpha: float = 0.05
    a6_seed: int = 1234
    a6_null_type: str = "haar"        # 'haar' | 'hankel' (Seccion 7 del paper)
    a6_hankel_row_block: int = 0      # filas de X por canal; 0 = rotacion densa
    bc_blocks: list | None = None
    bc_energy_metric: str = "log_absolute"
    bc_geometry_metric: str = "adjusted"   # ||U^T W_k||²/j (paper, v4)
    bc_n_perm: int = 5000
    bc_seed: int = 20260812
    bc_alpha: float = 0.05
    bc_condition_names: list[str] = field(default_factory=list)
    bc_rank_outcome: str = "selected"     # 'selected' | 'effective' (Remark 3.6)
    effective_rank: bool = False          # estimador de rango efectivo X% (A1-A5)
    tangent_blocks: list | None = None
    tangent_blocks_mode: str = "cumulative"  # 'omnibus' | 'cumulative' (paper 3.4)
    tangent_n_perm: int = 5000
    tangent_seed: int = 9999
    tangent_alpha: float = 0.05
    d_max_specific: int = 10
    d_rank_adaptive: bool = False     # seleccion Haar de r_c (Def. 3.13)
    d_n_null_specific: int = 100
    d_alpha_specific: float = 0.05
    d_loso: bool = False               # calibracion LOSO (Def. 3.15)
    d_loso_n_perm: int = 1000
    d_residual_rank_method: str = "local_gap"
    d_residual_rank_threshold: float = 0.1
    d_fixed_residual_rank: int = 5
    run_replica_consistency: bool = False  # Framework 2 (Seccion 4.5)
    skip_bc: bool = False
    skip_tangent: bool = False
    skip_d: bool = False
    seed: int = 42


# =====================================================================
# CDHSAResult
# =====================================================================

class CDHSAResult:
    """Container for all CD-HSA results."""

    def __init__(self, config, R, A6=None, BC=None, G=None, D=None,
                 D_loso=None, REPR=None):
        self.config = config
        self.R = R
        self.A6 = A6
        self.BC = BC
        self.G = G
        self.D = D
        self.D_loso = D_loso
        self.REPR = REPR

    def summary(self) -> str:
        lines = []
        lines.append("CD-HSA Results Summary")
        lines.append("=" * 60)
        R = self.R
        lines.append(f"  Subjects: {R['S']}, Conditions: {R['C']}, "
                     f"Channels: {R['p']}, d = {R['d']}")
        lines.append(f"  Common directions estimated: {len(R['lambda_'])}")
        if R.get('rank_method') == 'variance':
            rk = np.asarray(R['rank'])
            lines.append(
                f"  Local rank (variance X={R['var_explained'] * 100:.1f}%): "
                f"min={int(rk.min())}, median={int(np.median(rk))}, "
                f"max={int(rk.max())}, "
                f"censored={int(np.sum(R['var_censored']))}/{rk.size}"
            )
        names = (self.config.bc_condition_names
                 or [f"Condition {c + 1}" for c in range(R['C'])])
        if self.A6 is not None:
            lines.append(f"\n  A6 Common rank r0 = {self.A6['r0']} "
                         f"(null={self.config.a6_null_type})")
            if self.A6['r0'] > 0:
                prev = R.get('prevalence')
                for j in range(self.A6['r0']):
                    pi_txt = ("" if prev is None or j >= len(prev)
                              else f"  pi={prev[j]:.4f}")
                    lines.append(
                        f"    j={j+1}: lambda={self.A6['lambda0'][j]:.4f}{pi_txt}"
                    )
        if self.BC is not None:
            lines.append(f"\n  B/C Condition tests (alpha={self.config.bc_alpha}):")
            rk_sum = self.BC.get('summary', {})
            if 'rank_F' in rk_sum:
                deg = rk_sum.get('rank_degenerate')
                note = (" (degenerate: fixed-rank config, Remark 3.6)"
                        if deg else "")
                lines.append(
                    f"    Rank test: F={float(np.atleast_1d(rk_sum['rank_F'])[0]):.2f}"
                    f"  p={float(np.atleast_1d(rk_sum['rank_p_maxF'])[0]):.4f}{note}"
                )
            for k, name in enumerate(self.BC['block_names']):
                sig_e = "*" if self.BC['sig_energy_maxF'][k] else ""
                sig_g = "*" if self.BC['sig_geometry_maxF'][k] else ""
                eta_e = eta_g = ""
                if 'energy_partial_eta_sq' in rk_sum:
                    eta_e = f" (eta2={rk_sum['energy_partial_eta_sq'][k]:.2f})"
                    eta_g = f" (eta2={rk_sum['geometry_partial_eta_sq'][k]:.2f})"
                lines.append(
                    f"    {name}: energy_F="
                    f"{self.BC['summary']['energy_F'][k]:.2f}"
                    f"{sig_e}{eta_e}  geom_F="
                    f"{self.BC['summary']['geometry_F'][k]:.2f}"
                    f"{sig_g}{eta_g}"
                )
        if self.G is not None:
            lines.append(f"\n  Tangent geometry test:")
            for k in range(len(self.G['T_obs'])):
                sig = "*" if self.G['sig_maxT'][k] else ""
                idx = np.atleast_1d(self.G['blocks'][k])
                blabel = (f"block {k+1} [{int(idx[0])}..{int(idx[-1])}]"
                          if idx.size > 1 else f"block {k+1}")
                lines.append(
                    f"    {blabel}: T={self.G['T_obs'][k]:.4f}"
                    f"  p_maxT={self.G['p_maxT'][k]:.4f}{sig}"
                )
        if self.D is not None:
            rank_note = ("adaptive-Haar"
                         if self.D.get('rank_adaptive')
                         else f"cap={self.D.get('max_specific')}")
            lines.append(f"\n  D Condition-specific modes (rank: {rank_note}):")
            for c in range(self.D['C']):
                rc = self.D['r_specific'][c]
                pc = self.D['prevalence_contrast'][c]
                cname = names[c] if c < len(names) else f"Condition {c+1}"
                lines.append(
                    f"    {cname}: {rc} modes,"
                    f"  prevalence contrast = {pc:.4f}"
                )
        if self.D_loso is not None:
            lines.append(
                f"\n  D LOSO prevalence contrast "
                f"(Def. 3.15, n_perm={self.D_loso['n_perm']}):"
            )
            for c in range(len(self.D_loso['delta_loso'])):
                dlt = self.D_loso['delta_loso'][c]
                pv = self.D_loso['p_loso'][c]
                sig = "*" if self.D_loso['sig_loso'][c] else ""
                cname = names[c] if c < len(names) else f"Condition {c+1}"
                lines.append(
                    f"    {cname}: Delta_loso={dlt:.4f}  p={pv:.4f}{sig}"
                )
        if self.REPR is not None:
            S = self.REPR['S']
            lines.append(f"\n  Replica consistency (Framework 2, >=4/5 of {S}):")
            bb = self.REPR['backbone']
            lines.append(
                "    Backbone overlap: "
                + ", ".join(f"{v:.3f}" for v in bb['overlap'])
                + f" (ref random {bb['ref_random']:.3f})"
                + f" -> {bb['n_above_ref']}/{S}"
            )
            en = self.REPR.get('energy')
            if en is not None:
                if 'n_sign_consistent' in en:
                    ok = en['consistent_4of5']
                    lines.append(
                        f"    Energy shifts (C=2): consistent in "
                        f"{int(np.sum(ok))}/{len(ok)} directions"
                    )
                elif 'n_dominant_match' in en:
                    ok = en['consistent_4of5']
                    lines.append(
                        f"    Energy dominant condition: consistent in "
                        f"{int(np.sum(ok))}/{len(ok)} directions"
                    )
            de = self.REPR.get('deformation')
            if de is not None:
                med = np.median(de['T_replica'], axis=0)
                txt = ", ".join(f"{v:.3g}" for v in med[:5])
                txt += "..." if len(med) > 5 else ""
                lines.append(f"    Deformation median T^(s) per block: {txt}")
                cs = de.get('cosine_similarity')
                if cs is not None:
                    txt2 = ", ".join(f"{v:.2f}" for v in cs[:5])
                    txt2 += "..." if len(cs) > 5 else ""
                    lines.append(f"    Deformation direction cosine: {txt2}")
            di = self.REPR.get('discriminability')
            if di is not None:
                lines.append(
                    "    Discriminability Delta^(s): "
                    + ", ".join(f"{v:.3f}" for v in di['delta_replica'])
                    + f" -> {di['n_positive']}/{S}"
                )
        return "\n".join(lines)


# =====================================================================
# run_cdhsa (original)
# =====================================================================

def run_cdhsa(
    X: list[list[NDArray[np.floating]]],
    L: int,
    config: CDHSAConfig | None = None,
    col_masks: list | None = None,
) -> CDHSAResult:
    """Run the full CD-HSA pipeline.

    Parameters
    ----------
    X : list of list of arrays, shape (p, T)
        X[s][c] is the data matrix (e.g. Hankel) for subject s,
        condition c, with shape (p, T).
    L : int
        Second-level embedding depth (L_bh of the paper).
    config : CDHSAConfig, optional
    col_masks : list of list of arrays or None
        Máscaras de columnas de nivel 2 por grabación (ver
        ``a_common_subspace.level2_boundary_mask``): excluyen las
        ventanas que cruzan fronteras de concatenación entre sujetos.
        Se aplican en A1-A5, en la energía del Step B y en el nulo
        Hankel-preservante. None = sin enmascarar.

    Returns
    -------
    result : CDHSAResult
    """
    from src.cdhsa.a_common_subspace import cdhsa_A1_A5
    from src.cdhsa.a6_common_rank import cdhsa_A6_common_rank
    from src.cdhsa.b_energy import cdhsa_BC_condition_tests
    from src.cdhsa.c_geometry import cdhsa_tangent_geometry_test
    from src.cdhsa.d_condition_specific import (
        cdhsa_D_condition_specific_modes,
        cdhsa_D_prevalence_loso,
    )
    from src.cdhsa.replica_consistency import cdhsa_replica_consistency

    if config is None:
        config = CDHSAConfig()

    R = cdhsa_A1_A5(
        X, L,
        rank_method=config.rank_method,
        fixed_rank=config.fixed_rank,
        rmax=config.rmax,
        n_blocks=config.n_blocks,
        repro_threshold=config.repro_threshold,
        repro_strategy=config.repro_strategy,
        max_common=config.max_common,
        prevalence_quantile=config.prevalence_quantile,
        var_explained=config.var_explained,
        var_max_rank=config.var_max_rank,
        var_min_rank=config.var_min_rank,
        col_masks=col_masks,
        effective_rank=config.effective_rank,
    )

    # A6: a6_max_common = 0 => todos los candidatos de A1-A5
    # (evita censurar r0 en un tope arbitrario, p.ej. 20).
    a6_max = (config.a6_max_common if config.a6_max_common > 0
             else min(config.max_common, len(R['lambda_'])))
    a6_opts = {
        'max_common': a6_max,
        'n_folds': config.a6_n_folds,
        'n_null': config.a6_n_null,
        'alpha': config.a6_alpha,
        'seed': config.a6_seed,
        'null_type': config.a6_null_type,
    }
    if config.a6_null_type == 'hankel':
        a6_opts['X'] = X
        a6_opts['L_hankel'] = L
        a6_opts['hankel_row_block'] = config.a6_hankel_row_block
        a6_opts['col_masks'] = col_masks
    A6 = cdhsa_A6_common_rank(R, opts=a6_opts)

    BC = None
    G = None
    D = None
    D_loso = None
    REPR = None

    if not config.skip_bc and A6['r0'] >= 1:
        bc_opts = {
            'blocks': config.bc_blocks,
            'energy_metric': config.bc_energy_metric,
            'geometry_metric': config.bc_geometry_metric,
            'n_perm': config.bc_n_perm,
            'seed': config.bc_seed,
            'alpha': config.bc_alpha,
            'rank_outcome': config.bc_rank_outcome,
        }
        if config.bc_condition_names:
            bc_opts['condition_names'] = config.bc_condition_names
        BC = cdhsa_BC_condition_tests(X, L, R, A6, opts=bc_opts)

    if not config.skip_tangent and A6['r0'] >= 1:
        t_blocks = config.tangent_blocks
        if t_blocks is None:
            min_rank = int(np.min(R['rank']))
            if config.tangent_blocks_mode == 'cumulative':
                # Bloques acumulativos B_k = {1..k}, k = 1..kmax
                # (paper 3.4). kmax = min(r0, min r_sc): el bloque no
                # puede ser mas ancho que el rango local minimo.
                kmax = min(A6['r0'], min_rank)
                t_blocks = [np.arange(1, k + 1, dtype=int)
                            for k in range(1, kmax + 1)]
            else:
                # Omnibus {1..r0}, recortado al rango local minimo si
                # hace falta (rangos ragged, p.ej. variance).
                if A6['r0'] > min_rank:
                    print(f"  [WARN] tangent omnibus {A6['r0']} > min rango "
                          f"local {min_rank}: bloque recortado a {min_rank}.")
                t_blocks = [np.arange(1, min(A6['r0'], min_rank) + 1,
                                      dtype=int)]
        G = cdhsa_tangent_geometry_test(R, A6, blocks=t_blocks, opts={
            'n_perm': config.tangent_n_perm,
            'seed': config.tangent_seed,
            'alpha': config.tangent_alpha,
        })

    if not config.skip_d and A6['r0'] >= 1:
        D = cdhsa_D_condition_specific_modes(X, L, R, A6, opts={
            'max_specific': config.d_max_specific,
            'residual_rank_method': config.d_residual_rank_method,
            'residual_rank_threshold': config.d_residual_rank_threshold,
            'fixed_residual_rank': config.d_fixed_residual_rank,
            'prevalence_quantile': config.prevalence_quantile,
            'rank_adaptive': config.d_rank_adaptive,
            'n_null_specific': config.d_n_null_specific,
            'alpha_specific': config.d_alpha_specific,
        })
        if config.d_loso:
            if R['S'] < 2:
                print("  [WARN] d_loso omitido: S < 2 (no hay sujetos que "
                      "dejar fuera).")
            else:
                D_loso = cdhsa_D_prevalence_loso(D, opts={
                    'n_perm': config.d_loso_n_perm,
                    'alpha': config.bc_alpha,
                })

    if config.run_replica_consistency and A6['r0'] >= 1:
        if R['S'] < 2:
            print("  [WARN] replica_consistency omitido: Framework 2 "
                  "requiere S >= 2.")
        else:
            REPR = cdhsa_replica_consistency(R, A6, BC=BC, G=G, D=D)

    return CDHSAResult(config=config, R=R, A6=A6, BC=BC, G=G, D=D,
                       D_loso=D_loso, REPR=REPR)


# =====================================================================
# Construccion de matrices de Hankel desde super-sujetos EEG
# =====================================================================

def build_hankel_from_eeg(
    *,
    session: str,
    tasks: list[str],
    n_super_subjects: int,
    total_subjects: int = 60,
    subject_start_offset: int = 1,
    db_path: str | Path | None = None,
    t_start: float | None = None,
    t_stop: float | None = None,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    hankel_depth: int | None = None,
    verbose: bool | str | None = None,
) -> tuple[list[list[NDArray[np.floating]]], dict]:
    """Construir matrices de Hankel para cada par (super-sujeto, condicion).

    Particiona ``total_subjects`` en ``n_super_subjects`` super-sujetos
    de tamano ``total_subjects // n_super_subjects``.  Cada super-sujeto
    es la concatenacion temporal de sus sujetos individuales.

    Se usa una interseccion GLOBAL de canales entre todos los
    super-sujetos y condiciones para garantizar que todas las
    matrices de Hankel tengan el mismo numero de filas ``p``.

    Pipeline por (super-sujeto, condicion)::::

        0. Cargar todos los raws -> interseccion global de canales
        1. Pick canales comunes  -> raw.pick(common_channels)
        2. Filtro pasa-banda    -> extract_filtered_data_matrix()
           X_filtered: (n_channels, n_times), centrada a media cero
        3. Matriz de Hankel     -> _build_multivariate_hankel()
           H: (n_channels * depth, n_times - depth + 1)

    Returns
    -------
    X : list[list[NDArray]]
        X[s][c] = Hankel para super-sujeto s, condicion c.
    info : dict
        Metadata completa de la construccion. Incluye
        ``member_n_times[s][c]`` (muestras de cada sujeto miembro tras
        recorte+filtro) para el enmascaramiento de fronteras de nivel 2
        (ver ``a_common_subspace.level2_boundary_mask``).
    """
    from src.latent_space_extraction.super_subject_eeg import (
        load_super_subject_eeg,
        resolve_super_subject_subject_ids,
    )
    from src.latent_space_extraction.eeg_preprocessing import (
        extract_filtered_data_matrix,
    )
    from src.latent_space_extraction.hankel_dmd_extractor import (
        _build_multivariate_hankel,
        _auto_embedding_depth,
    )

    if db_path is None:
        try:
            from src.utils.config import DB_TEST_RETEST_GEDAI_PATH
            db_path = DB_TEST_RETEST_GEDAI_PATH
        except ImportError:
            raise ValueError(
                "--db-path es obligatorio si src.utils.config "
                "no define DB_TEST_RETEST_GEDAI_PATH."
            )

    if total_subjects % n_super_subjects != 0:
        raise ValueError(
            f"total_subjects ({total_subjects}) no es divisible "
            f"por n_super_subjects ({n_super_subjects})."
        )
    subjects_per = total_subjects // n_super_subjects

    S = n_super_subjects
    C = len(tasks)

    print(f"  Particion: {total_subjects} subjects -> "
          f"{S} super-sujetos de {subjects_per} cada uno")
    print(f"  Super-sujeto 1 = subs {subject_start_offset}"
          f"..{subject_start_offset + subjects_per - 1}")
    print(f"  Super-sujeto {S} = subs "
          f"{subject_start_offset + (S-1)*subjects_per}"
          f"..{subject_start_offset + S*subjects_per - 1}")
    print()

    # =================================================================
    # PASADA 0: Cargar todos los raws y calcular interseccion global
    # =================================================================
    print("  PASADA 0: Cargando raws para calcular interseccion global de canales...")
    sys.stdout.flush()

    raws_store: dict[tuple[int, int], "mne.io.Raw"] = {}  # (ss_idx, c_idx) -> raw
    raws_member_lengths: dict[tuple[int, int], list[int]] = {}
    ss_member_ids: dict[int, list[int]] = {}
    load_errors: list[tuple[int, int, str]] = []

    t0_load = time.time()
    for ss_id in range(1, S + 1):
        member_ids = resolve_super_subject_subject_ids(
            super_subject_id=ss_id,
            subjects_per_super_subject=subjects_per,
            subject_start_offset=subject_start_offset,
        )
        ss_member_ids[ss_id] = member_ids

        for c_idx, task in enumerate(tasks):
            tag = (f"  [{ss_id}/{S}] (subs {member_ids[0]}..{member_ids[-1]})"
                   f"/{session}/{task}")
            print(f"{tag} ...", end=" ")
            sys.stdout.flush()

            try:
                raw, member_lengths = load_super_subject_eeg(
                    super_subject_id=ss_id,
                    session=session,
                    task=task,
                    subjects_per_super_subject=subjects_per,
                    subject_start_offset=subject_start_offset,
                    db_path=db_path,
                    t_start=t_start,
                    t_stop=t_stop,
                    preload=True,
                    verbose=False,
                    return_member_lengths=True,
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"SKIP ({exc})")
                load_errors.append((ss_id - 1, c_idx, str(exc)))
                continue

            raws_store[(ss_id - 1, c_idx)] = raw
            raws_member_lengths[(ss_id - 1, c_idx)] = member_lengths
            print(f"OK  ch={len(raw.ch_names)} dur={raw.times[-1]:.0f}s")
            sys.stdout.flush()

    print(f"  Carga completa en {time.time() - t0_load:.1f}s")

    if not raws_store:
        raise RuntimeError("No se pudo cargar ningun raw.")

    # --- Interseccion global de canales ---
    all_ch_names: list[list[str]] = [
        raws_store[key].ch_names for key in sorted(raws_store)
    ]
    # Preservar orden del primer raw
    global_channels = list(all_ch_names[0])
    for ch_list in all_ch_names[1:]:
        global_channels = [ch for ch in global_channels if ch in ch_list]

    n_ch_before = {len(cl) for cl in all_ch_names}
    print(f"  Canales por raw antes: {n_ch_before}")
    print(f"  Interseccion global  : {len(global_channels)} canales")
    if len(global_channels) < min(n_ch_before):
        print(f"  (se descartan {min(n_ch_before) - len(global_channels)} canales)")
    print()

    # =================================================================
    # PASADA 1: Pick canales comunes + filtro + Hankel
    # =================================================================
    print("  PASADA 1: Construyendo matrices de Hankel...")
    sys.stdout.flush()

    X: list[list[NDArray[np.floating]]] = []
    shapes: list[list[tuple[int, int] | None]] = []
    sfreqs: list[float] = []
    depths_used: list[int] = []
    n_channels_list: list[int] = []
    n_times_filtered_list: list[int] = []
    skipped: list[tuple[int, int, str]] = list(load_errors)
    super_subject_ids_list: list[list[int]] = []
    durations: list[float] = []
    member_n_times_all: list[list[list[int] | None]] = []

    t0_global = time.time()

    for ss_id in range(1, S + 1):
        ss_label = f"super_subject-{ss_id:02d}"
        X_s: list[NDArray[np.floating]] = []
        shapes_s: list[tuple[int, int] | None] = []
        members_s: list[list[int] | None] = []
        member_ids = ss_member_ids[ss_id]
        super_subject_ids_list.append(member_ids)

        for c_idx, task in enumerate(tasks):
            tag = (f"[{ss_id}/{S}] {ss_label} "
                   f"(subs {member_ids[0]}..{member_ids[-1]})"
                   f"/{session}/{task}")

            key = (ss_id - 1, c_idx)
            if key not in raws_store:
                print(f"  {tag} SKIP (error en carga)")
                X_s.append(np.empty((0, 0)))
                shapes_s.append(None)
                members_s.append(None)
                continue

            raw = raws_store.pop(key)  # liberar memoria
            # Longitudes de los miembros (para la mascara de fronteras)
            member_n_times_sc = raws_member_lengths.pop(key, None)
            print(f"  {tag} ...", end=" ")
            sys.stdout.flush()

            # --- Pick canales comunes ---
            raw.pick(global_channels)
            sfreq = float(raw.info["sfreq"])
            duration_s = raw.times[-1]
            n_ch_raw = len(raw.ch_names)

            # --- Filtro pasa-banda ---
            X_filtered, _raw_filt, sfreq = extract_filtered_data_matrix(
                raw, l_freq=l_freq, h_freq=h_freq, verbose=False,
            )
            del raw, _raw_filt  # liberar memoria
            n_ch, n_times = X_filtered.shape

            # --- Construir matriz de Hankel ---
            if hankel_depth is None:
                depth = _auto_embedding_depth(sfreq, n_times)
            else:
                depth = int(hankel_depth)

            if depth >= n_times:
                print(f"SKIP (depth={depth} >= n_times={n_times})")
                X_s.append(np.empty((0, 0)))
                shapes_s.append(None)
                members_s.append(None)
                skipped.append((ss_id - 1, c_idx,
                    f"depth={depth} >= n_times={n_times}"))
                continue

            # Coherencia de las longitudes de miembros con el stream filtrado
            if member_n_times_sc is not None and sum(member_n_times_sc) != n_times:
                print(f"\n  [WARN] sum(member_n_times)={sum(member_n_times_sc)} "
                      f"!= n_times={n_times}: la mascara de fronteras NO "
                      f"se usara para (s={ss_id - 1}, c={c_idx}).")
                member_n_times_sc = None

            H = _build_multivariate_hankel(X_filtered, depth)
            del X_filtered  # liberar memoria

            X_s.append(H)
            shapes_s.append(H.shape)
            members_s.append(member_n_times_sc)
            sfreqs.append(sfreq)
            depths_used.append(depth)
            n_channels_list.append(n_ch)
            n_times_filtered_list.append(n_times)
            durations.append(duration_s)

            print(f"OK  ch={n_ch} T={n_times} "
                  f"dur={duration_s:.0f}s "
                  f"depth={depth} -> H={H.shape} "
                  f"(miembros: {len(member_n_times_sc) if member_n_times_sc else 1})")
            sys.stdout.flush()

        X.append(X_s)
        shapes.append(shapes_s)
        member_n_times_all.append(members_s)

    # Liberar cualquier raw residual
    raws_store.clear()

    elapsed = time.time() - t0_global

    info = {
        "session": session,
        "tasks": tasks,
        "n_super_subjects": S,
        "total_subjects": total_subjects,
        "subjects_per_super_subject": subjects_per,
        "subject_start_offset": subject_start_offset,
        "super_subject_ids": super_subject_ids_list,
        "S": S,
        "C": C,
        "l_freq": l_freq,
        "h_freq": h_freq,
        "hankel_depth_requested": hankel_depth,
        "global_channels": global_channels,
        "n_global_channels": len(global_channels),
        "shapes": shapes,
        "sfreqs": sfreqs,
        "depths_used": depths_used,
        "n_channels": n_channels_list,
        "n_times_filtered": n_times_filtered_list,
        "durations": durations,
        "skipped": skipped,
        "member_n_times": member_n_times_all,
        "elapsed_load": time.time() - t0_load,
        "elapsed_build": elapsed,
    }
    if sfreqs:
        info["sfreq_common"] = (sfreqs[0] if len(set(sfreqs)) == 1
                                 else None)
        info["depth_common"] = (depths_used[0]
                                if len(set(depths_used)) == 1
                                else None)
        info["n_channels_common"] = (n_channels_list[0]
                                     if len(set(n_channels_list)) == 1
                                     else None)

    return X, info


def build_hankel_single_ss(
    *,
    super_subject_id: int,
    session: str,
    tasks: list[str],
    subjects_per_super_subject: int = 20,
    subject_start_offset: int = 1,
    db_path: str | Path | None = None,
    t_start: float | None = None,
    t_stop: float | None = None,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    hankel_depth: int | None = None,
    verbose: bool | str | None = None,
    global_channels: list[str] | None = None,
) -> tuple[list[list[NDArray[np.floating]]], dict]:
    """Construir matrices de Hankel para un UNICO super-sujeto (S=1).

    Carga el super-sujeto indicado para cada condicion, construye
    la Hankel y retorna X con S=1.

    Parameters
    ----------
    global_channels : list[str] | None
        Si se proporciona, se hace ``raw.pick(global_channels)``
        despues de cargar los datos, antes de filtrar y construir
        la Hankel.  Esto garantiza una dimension ``d`` consistente
        cuando se procesan multiples super-sujetos por separado.
    """
    from src.latent_space_extraction.super_subject_eeg import (
        load_super_subject_eeg,
        resolve_super_subject_subject_ids,
    )
    from src.latent_space_extraction.eeg_preprocessing import (
        extract_filtered_data_matrix,
    )
    from src.latent_space_extraction.hankel_dmd_extractor import (
        _build_multivariate_hankel,
        _auto_embedding_depth,
    )

    if db_path is None:
        try:
            from src.utils.config import DB_TEST_RETEST_GEDAI_PATH
            db_path = DB_TEST_RETEST_GEDAI_PATH
        except ImportError:
            raise ValueError(
                "--db-path es obligatorio si src.utils.config "
                "no define DB_TEST_RETEST_GEDAI_PATH."
            )

    member_ids = resolve_super_subject_subject_ids(
        super_subject_id=super_subject_id,
        subjects_per_super_subject=subjects_per_super_subject,
        subject_start_offset=subject_start_offset,
    )

    C = len(tasks)
    print(f"  Super-sujeto {super_subject_id}: "
          f"subs {member_ids[0]}..{member_ids[-1]} ({len(member_ids)})")
    print(f"  Condiciones: {', '.join(tasks)}")
    print()

    X_s: list[NDArray[np.floating]] = []
    shapes_s: list[tuple[int, int] | None] = []
    sfreqs: list[float] = []
    depths_used: list[int] = []
    n_channels_list: list[int] = []
    n_times_filtered_list: list[int] = []
    durations: list[float] = []
    skipped: list[tuple[int, int, str]] = []
    ch_names: list[str] = []
    member_n_times_list: list[list[int] | None] = []

    t0 = time.time()

    for c_idx, task in enumerate(tasks):
        tag = f"  [{c_idx+1}/{C}] {task} ..."
        print(tag, end=" ")
        sys.stdout.flush()

        try:
            raw, member_n_times = load_super_subject_eeg(
                super_subject_id=super_subject_id,
                session=session,
                task=task,
                subjects_per_super_subject=subjects_per_super_subject,
                subject_start_offset=subject_start_offset,
                db_path=db_path,
                t_start=t_start,
                t_stop=t_stop,
                preload=True,
                verbose=False,
                return_member_lengths=True,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"SKIP ({exc})")
            X_s.append(np.empty((0, 0)))
            shapes_s.append(None)
            member_n_times_list.append(None)
            skipped.append((0, c_idx, str(exc)))
            continue

        # Restringir a canales globales si se proporcionan
        if global_channels is not None:
            available = [c for c in global_channels if c in raw.ch_names]
            missing = set(global_channels) - set(raw.ch_names)
            if missing:
                print(f"WARN {len(missing)} canales globales faltantes")
            raw.pick(available)

        sfreq = float(raw.info["sfreq"])
        duration_s = raw.times[-1]
        n_ch = len(raw.ch_names)
        if not ch_names:
            ch_names = list(raw.ch_names)

        X_filtered, _raw_filt, sfreq = extract_filtered_data_matrix(
            raw, l_freq=l_freq, h_freq=h_freq, verbose=False,
        )
        del raw, _raw_filt
        n_ch, n_times = X_filtered.shape

        if hankel_depth is None:
            depth = _auto_embedding_depth(sfreq, n_times)
        else:
            depth = int(hankel_depth)

        if depth >= n_times:
            print(f"SKIP (depth={depth} >= n_times={n_times})")
            X_s.append(np.empty((0, 0)))
            shapes_s.append(None)
            member_n_times_list.append(None)
            skipped.append((0, c_idx,
                f"depth={depth} >= n_times={n_times}"))
            continue

        # Coherencia de las longitudes de miembros con el stream filtrado
        if member_n_times is not None and sum(member_n_times) != n_times:
            print(f"\n  [WARN] sum(member_n_times)={sum(member_n_times)} "
                  f"!= n_times={n_times}: la mascara de fronteras NO "
                  f"se usara para (c={c_idx}).")
            member_n_times = None

        H = _build_multivariate_hankel(X_filtered, depth)
        del X_filtered

        X_s.append(H)
        shapes_s.append(H.shape)
        member_n_times_list.append(member_n_times)
        sfreqs.append(sfreq)
        depths_used.append(depth)
        n_channels_list.append(n_ch)
        n_times_filtered_list.append(n_times)
        durations.append(duration_s)

        print(f"OK  ch={n_ch} T={n_times} dur={duration_s:.0f}s "
              f"depth={depth} -> H={H.shape}")
        sys.stdout.flush()

    elapsed = time.time() - t0

    info = {
        "mode": "single_super_subject",
        "super_subject_id": super_subject_id,
        "member_ids": member_ids,
        "session": session,
        "tasks": tasks,
        "S": 1,
        "C": C,
        "subjects_per_super_subject": subjects_per_super_subject,
        "subject_start_offset": subject_start_offset,
        "global_channels": ch_names,
        "l_freq": l_freq,
        "h_freq": h_freq,
        "hankel_depth_requested": hankel_depth,
        "shapes": [shapes_s],
        "sfreqs": sfreqs,
        "depths_used": depths_used,
        "n_channels": n_channels_list,
        "n_times_filtered": n_times_filtered_list,
        "durations": durations,
        "skipped": skipped,
        "member_n_times": [member_n_times_list],
        "elapsed_build": elapsed,
    }
    if sfreqs:
        info["sfreq_common"] = sfreqs[0]
        info["depth_common"] = depths_used[0]
        info["n_channels_common"] = n_channels_list[0]

    return [X_s], info


def characterize_hankel_matrices(
    X: list[list[NDArray[np.floating]]],
    info: dict,
) -> str:
    """Resumen de las matrices de Hankel construidas.

    Para cada (super-sujeto, condicion) valido calcula:
    - Forma, rango numerico (SVD parcial), compresion, top-5 sing.vals
    """
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  CARACTERIZACION DE MATRICES DE HANKEL")
    lines.append("=" * 70)

    S = info["S"]
    C = info["C"]
    tasks = info["tasks"]
    subs_per = info["subjects_per_super_subject"]

    lines.append(f"  Super-sujetos     : {S} "
                 f"({subs_per} subjects cada uno)")
    lines.append(f"  Total subjects    : {info['total_subjects']}")
    lines.append(f"  Condiciones       : {C}  ({', '.join(tasks)})")
    lines.append(f"  Filtro            : {info['l_freq']}-{info['h_freq']} Hz")
    lines.append(f"  Depth pedido      : {info['hankel_depth_requested']}")
    if info.get("sfreq_common") is not None:
        lines.append(f"  sfreq             : {info['sfreq_common']:.2f} Hz")
    if info.get("depth_common") is not None:
        lines.append(f"  Depth real        : {info['depth_common']}")
    if info.get("n_channels_common") is not None:
        lines.append(f"  Canales           : {info['n_channels_common']}")
    lines.append(f"  Saltados          : {len(info['skipped'])}")
    lines.append(f"  Tiempo construccion: {info['elapsed_build']:.1f} s")

    header = (f"  {'SS':<6} {'Miembros':<20} {'Cond':<15} "
              f"{'Forma H':<28} {'Rank':>6} {'Comp.':>8}  "
              f"{'Top-5 sing.vals'}")
    lines.append("")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    n_valid = 0
    for s in range(S):
        member_ids = info["super_subject_ids"][s]
        members_str = f"{member_ids[0]}..{member_ids[-1]}"
        ss_label = f"SS-{s+1}"

        for c in range(C):
            H = X[s][c]
            if H.size == 0:
                lines.append(
                    f"  {ss_label:<6} {members_str:<20} "
                    f"{tasks[c]:<15} {'(vacio)':<28}"
                )
                continue

            n_valid += 1
            p, T_samples = H.shape

            k_svd = min(p, T_samples, 50)
            try:
                from scipy.sparse.linalg import svds
                svals = svds(H, k=k_svd,
                             return_singular_vectors=False)
                svals = np.sort(svals)[::-1]
                rank_est = int(np.sum(svals > svals[0] * 1e-6))
            except Exception:
                rank_est = -1
                svals = np.array([])

            comp = (p / rank_est if rank_est > 0 else float("inf"))
            top5 = ", ".join(f"{v:.1f}" for v in svals[:5])
            lines.append(
                f"  {ss_label:<6} {members_str:<20} "
                f"{tasks[c]:<15} {str(H.shape):<28} "
                f"{rank_est:>6} {comp:>7.2f}x  {top5}"
            )

    lines.append("")
    lines.append(f"  Total matrices validas: {n_valid} / {S * C}")

    return "\n".join(lines)


# =====================================================================
# CLI
# =====================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CD-HSA: construye matrices de Hankel desde super-sujetos "
            "del dataset Gedai y ejecuta el analisis CD-HSA."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Fuente de datos (super-sujetos) ---
    # Modo 1: --n-super-subjects (multi-SS, particion automatica)
    # Modo 2: --super-subject-id (single-SS, para batch por SS)
    mode_ss = parser.add_mutually_exclusive_group(required=True)
    mode_ss.add_argument(
        "--n-super-subjects", type=int,
        help=("Cantidad de super-sujetos a armar. "
              "Los total_subjects se reparten equitativamente "
              "(ej: 60/3 = 20 subjects por super-sujeto)."),
    )
    mode_ss.add_argument(
        "--super-subject-id", type=int,
        help=("ID de un unico super-sujeto (1-indexed). "
              "Usado por el batch runner para correr CD-HSA "
              "con S=1 sobre un solo grupo de sujetos."),
    )

    parser.add_argument(
        "--session", type=str, required=True,
        help="Session ID (ej: session1)",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", required=True,
        help="Condiciones/tareas (ej: eyesclosed eyesopen)",
    )
    parser.add_argument(
        "--total-subjects", type=int, default=60,
        help="Total de sujetos individuales disponibles. Default: 60",
    )
    parser.add_argument(
        "--subjects-per-super-subject", type=int, default=20,
        help="Sujetos por super-sujeto (solo con --super-subject-id). Default: 20",
    )
    parser.add_argument(
        "--subject-start-offset", type=int, default=1,
        help="Indice del primer sujeto. Default: 1",
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Raiz del dataset Gedai",
    )
    parser.add_argument("--t-start", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help=("Directorio raiz para guardar resultados. "
              "Default: BASE_RESULTS_PATH de src.utils.config"),
    )

    # --- CDHSA ---
    parser.add_argument(
        "--L", type=int, required=True,
        help=("Profundidad del embedding Hankel de SEGUNDO nivel (el L_bh "
              "del paper): construye la matriz de nivel 2 "
              "(p*L, K) sobre la que viven W0/W_specific. NO es la "
              "dimension de un subespacio."),
    )

    # --- Preprocesamiento ---
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=40.0)

    # --- Hankel ---
    parser.add_argument(
        "--hankel-depth", type=int, default=None,
        help="Profundidad Hankel. None = auto",
    )

    # --- Config CDHSA ---
    g = parser.add_argument_group("Parametros CDHSA")
    g.add_argument("--fixed-rank", type=int, default=10)
    g.add_argument("--rank-method", type=str, default="fixed",
                   choices=["fixed", "reproducibility", "variance"],
                   help="Metodo de seleccion del rango local. "
                        "'variance': menor r que explica --var-explained "
                        "de la energia Hankel (por grabacion; revive el "
                        "test de rango del Step B).")
    g.add_argument("--var-explained", type=float, default=0.99,
                   help="Fraccion de varianza para rank-method=variance "
                        "(X%%, p.ej. 0.99). Default: 0.99")
    g.add_argument("--var-max-rank", type=int, default=50,
                   help="Tope de candidatos para rank-method=variance. "
                        "Default: 50")
    g.add_argument("--var-min-rank", type=int, default=1,
                   help="Piso de rango local para rank-method=variance. "
                        "Default: 1")
    g.add_argument("--a6-n-null", type=int, default=100)
    g.add_argument("--a6-max-common", type=int, default=0,
                   help="Candidatos a testear en A6. 0 = todos los de "
                        "A1-A5 (evita censurar r0).")
    g.add_argument("--a6-null-type", type=str, default="haar",
                   choices=["haar", "hankel"],
                   help="Nulo del criterio dual. 'hankel' = rotacion "
                        "de canales que preserva la estructura de "
                        "retardo (Seccion 7 del paper; mas caro).")
    g.add_argument("--a6-hankel-row-block", type=int, default=0,
                   help="Filas de X por canal para el nulo hankel "
                        "(= hankel_depth del primer nivel; el pipeline "
                        "lo autodetecta desde la info de construccion).")
    g.add_argument("--bc-n-perm", type=int, default=5000)
    g.add_argument("--effective-rank", action="store_true",
                   help="Calcular ademas un rango efectivo X%% por "
                        "grabacion (A1-A5) y guardarlo en "
                        "R['rank_effective']; habilita --rank-outcome "
                        "effective para el test de rango (Remark 3.6).")
    g.add_argument("--rank-outcome", type=str, default="selected",
                   choices=["selected", "effective"],
                   help="Resultado del test de rango del Step B: el rango "
                        "primario ('selected') o el efectivo X%% "
                        "('effective'; requiere --effective-rank).")
    g.add_argument("--no-boundary-mask", action="store_true",
                   help="Desactivar el enmascaramiento de columnas de "
                        "nivel 2 que cruzan fronteras de concatenacion "
                        "entre sujetos (paper: se descartan; a lo sumo "
                        "L+depth-2 por frontera). Default: activado.")
    g.add_argument("--tangent-blocks-mode", type=str, default="cumulative",
                   choices=["omnibus", "cumulative"],
                   help="Bloques del test tangente: 'cumulative' = "
                        "B_k={1..k}, k=1..min(r0, r) con max-T sobre "
                        "la familia (paper 3.4, default); 'omnibus' = "
                        "un solo bloque {1..r0}.")
    g.add_argument("--d-max-specific", type=int, default=10,
                   help="Maximos modos especificos por condicion (Step D). "
                        "Default: 10")
    g.add_argument("--d-rank-adaptive", action="store_true",
                   help="Seleccionar r_c por test Haar consecutivo (Def. "
                        "3.13) en vez del tope fijo.")
    g.add_argument("--d-n-null-specific", type=int, default=100,
                   help="Replicas del nulo Haar de Step D. Default: 100")
    g.add_argument("--d-alpha-specific", type=float, default=0.05)
    g.add_argument("--d-loso", action="store_true",
                   help="Calibracion LOSO del contraste de prevalencia "
                        "(Def. 3.15 del paper).")
    g.add_argument("--d-loso-n-perm", type=int, default=1000,
                   help="Permutaciones del nulo LOSO. Default: 1000")
    g.add_argument("--run-replica-consistency", action="store_true",
                   help="Consistencia por replica (Framework 2, Seccion "
                        "4.5 del paper, criterio >=4/5).")
    g.add_argument("--skip-bc", action="store_true")
    g.add_argument("--skip-tangent", action="store_true")
    g.add_argument("--skip-d", action="store_true")

    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument(
        "--no-save", action="store_true",
        help="No guardar resultados a disco (solo imprimir)",
    )

    return parser.parse_args(argv)


# =====================================================================
# Guardado de resultados
# =====================================================================

def _json_safe(obj: Any) -> Any:
    """Convertir un obj a algo serializable por json."""
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "shape": list(obj.shape),
                "dtype": str(obj.dtype)}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _rank_tag(rank_method: str, fixed_rank: int,
              var_explained: float, var_max_rank: int) -> str:
    """Etiqueta de rango para el nombre del directorio de salida.

    Debe mantenerse sincronizada con
    ``run_batch_cdhsa.CDHSABatchRunner._rank_tag``.
    """
    if rank_method == "variance":
        return f"var{round(var_explained * 100)}c{var_max_rank}"
    if rank_method == "reproducibility":
        return "repro"
    return f"fr{fixed_rank}"


def _resolve_out_dir(args: argparse.Namespace) -> Path:
    """Resolver el directorio de salida y crear la subcarpeta.

    Estructura::

        {BASE_RESULTS_PATH}/cdhsa/{session}/
            nSS{n}_L{L}_{ranktag}_a6n{a6}_bcn{bc}/
            {l_freq}-{h_freq}Hz_depth{d}/
    """
    base = args.out_dir
    if base is None:
        try:
            from src.utils.config import BASE_RESULTS_PATH
            base = str(BASE_RESULTS_PATH)
        except ImportError:
            raise ValueError(
                "--out-dir es obligatorio si src.utils.config "
                "no define BASE_RESULTS_PATH."
            )

    t_start_tag = f"{args.t_start}s" if args.t_start is not None else "any"
    t_end_tag = f"{args.t_end}s" if args.t_end is not None else "any"

    if args.super_subject_id is not None:
        ss_label = f"SS{args.super_subject_id}"
    else:
        ss_label = f"nSS{args.n_super_subjects}"

    rtag = _rank_tag(args.rank_method, args.fixed_rank,
                     args.var_explained, args.var_max_rank)

    out_dir = Path(
        f"{base}/cdhsa/{args.session}"
        f"/{ss_label}_L{args.L}"
        f"_{rtag}_a6n{args.a6_n_null}_bcn{args.bc_n_perm}"
        f"/{args.l_freq}-{args.h_freq}Hz"
        f"_depth{args.hankel_depth or 'auto'}"
        f"/from{t_start_tag}_to{t_end_tag}"
        f"_{'_'.join(args.tasks)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _compute_col_masks(
    X: list[list[NDArray[np.floating]]],
    hankel_info: dict,
    L: int,
) -> list[list[NDArray[np.bool_] | None]] | None:
    """
    Computa las máscaras de columnas de nivel 2 desde la metadata.

    Usa ``hankel_info['member_n_times'][s][c]`` (muestras por sujeto
    miembro) y el depth del primer nivel, que se recupera como
    ``depth = T_total - T_nivel1 + 1`` (no hace falta confiar en los
    registros planos de depths_used). Una grabación con un solo
    miembro no tiene fronteras internas -> máscara None (sin coste).

    Returns
    -------
    masks : list of list of arrays or None, o None si no hay metadata
        de miembros (resultados pre-v4 o pipelines sin concatenación).
    """
    from src.cdhsa.a_common_subspace import level2_boundary_mask

    member_info = hankel_info.get('member_n_times')
    if not member_info:
        print("  [WARN] hankel_info sin member_n_times: no se pueden "
              "enmascarar las fronteras de concatenacion (ejecucion "
              "pre-v4 o datos de un solo sujeto por grabacion).")
        return None

    S = len(X)
    C = len(X[0])
    if len(member_info) != S:
        print(f"  [WARN] member_n_times tiene {len(member_info)} filas "
              f"pero S={S}: sin enmascaramiento.")
        return None

    masks: list[list[NDArray[np.bool_] | None]] = []
    n_masked = 0
    for s in range(S):
        row: list[NDArray[np.bool_] | None] = []
        for c in range(C):
            members = member_info[s][c] if c < len(member_info[s]) else None
            if (members is None or len(members) <= 1
                    or X[s][c].size == 0):
                row.append(None)
                continue
            T_total = int(sum(members))
            T1 = int(X[s][c].shape[1])
            depth = T_total - T1 + 1
            if depth < 1:
                print(f"  [WARN] depth inconsistente en (s={s}, c={c}): "
                      f"T_total={T_total}, T1={T1}: sin mascara.")
                row.append(None)
                continue
            mask = level2_boundary_mask(members, depth, L)
            row.append(mask)
            n_masked += int(np.sum(~mask))
        masks.append(row)

    print(f"  [Fronteras] {n_masked} columnas de nivel 2 seran "
          f"descartadas (ventanas que cruzan sujetos).")
    return masks


def save_results(
    out_dir: Path,
    X: list[list[NDArray[np.floating]]],
    hankel_info: dict,
    characterization: str,
    result: CDHSAResult,
    cfg: CDHSAConfig,
    L: int,
) -> None:
    """Guardar todos los resultados en out_dir.

    Archivos creados::

        hankel_info.json       Metadata de construccion de Hankel
        config.json            Configuracion CDHSA usada
        characterization.txt   Tabla de caracterizacion de matrices
        cdhsa_summary.txt      Resumen textual del CD-HSA
        hankel_matrices.npz    Matrices de Hankel (X[s][c])
        cdhsa_arrays.npz       Arrays numericos del resultado CD-HSA
    """
    print(f"\n  Guardando resultados en: {out_dir}")
    sys.stdout.flush()

    # --- 1. hankel_info.json ---
    hankel_info_json = _json_safe(hankel_info)
    with open(out_dir / "hankel_info.json", "w") as f:
        json.dump(hankel_info_json, f, indent=2, default=str)
    print("    [OK] hankel_info.json")

    # --- 2. config.json ---
    cfg_dict = _json_safe(asdict(cfg))
    cfg_dict["L"] = L
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)
    print("    [OK] config.json")

    # --- 3. characterization.txt ---
    with open(out_dir / "characterization.txt", "w") as f:
        f.write(characterization)
    print("    [OK] characterization.txt")

    # --- 4. cdhsa_summary.txt ---
    summary_text = result.summary()
    with open(out_dir / "cdhsa_summary.txt", "w") as f:
        f.write(summary_text)
    print("    [OK] cdhsa_summary.txt")

    # --- 5. hankel_matrices.npz ---
    save_dict: dict[str, NDArray] = {}
    for s in range(len(X)):
        for c in range(len(X[s])):
            key = f"H_ss{s+1}_c{c+1}"
            H = X[s][c]
            if H.size > 0:
                save_dict[key] = H
            else:
                save_dict[key] = np.array([])
    np.savez_compressed(out_dir / "hankel_matrices.npz", **save_dict)
    print(f"    [OK] hankel_matrices.npz  ({len(save_dict)} matrices)")

    # --- 6. cdhsa_arrays.npz ---
    arrays_dict: dict[str, NDArray] = {}
    _save_result_arrays(result.R, "R", arrays_dict)
    if result.A6 is not None:
        _save_result_arrays(result.A6, "A6", arrays_dict)
    if result.BC is not None:
        _save_result_arrays(result.BC, "BC", arrays_dict)
    if result.G is not None:
        _save_result_arrays(result.G, "G", arrays_dict)
    if result.D is not None:
        _save_result_arrays(result.D, "D", arrays_dict)
    if result.D_loso is not None:
        _save_result_arrays(result.D_loso, "D_loso", arrays_dict)
    if result.REPR is not None:
        _save_result_arrays(result.REPR, "REPR", arrays_dict)

    # --- 7. mascaras de fronteras (colmask_ss{s}_c{c}) ---
    # Se guardan aparte porque el aplanador generico no preserva la
    # estructura (lista de listas de arrays). extract_mode_indices las
    # usa para poner alpha=0 en las columnas que cruzan sujetos.
    col_masks = result.R.get('col_masks')
    if col_masks is not None:
        n_masks = 0
        for s in range(len(col_masks)):
            for c in range(len(col_masks[s])):
                m = col_masks[s][c]
                if m is None:
                    continue
                arrays_dict[f'colmask_ss{s + 1}_c{c + 1}'] = np.asarray(m)
                n_masks += 1
        if n_masks:
            print(f"    [OK] {n_masks} mascaras de fronteras en "
                  f"cdhsa_arrays.npz (colmask_ss*_c*)")

    np.savez_compressed(out_dir / "cdhsa_arrays.npz", **arrays_dict)
    print(f"    [OK] cdhsa_arrays.npz  ({len(arrays_dict)} arrays)")

    print(f"  Listo. {len(list(out_dir.iterdir()))} archivos en {out_dir}")


def _save_result_arrays(
    d: dict, prefix: str, out: dict[str, NDArray]
) -> None:
    """Extraer arrays numericos de un dict de resultados y meterlos en out."""
    for k, v in d.items():
        key = f"{prefix}__{k}"
        if isinstance(v, np.ndarray):
            out[key] = v
        elif isinstance(v, dict):
            _save_result_arrays(v, key, out)
        elif isinstance(v, (list, tuple)) and len(v) > 0:
            # Intentar convertir lista de arrays a stack
            try:
                arr = np.array(v)
                if arr.dtype.kind in ("f", "i", "u", "b"):
                    out[key] = arr
                elif arr.dtype.kind == "O":
                    # Ragged (p.ej. r_c distinto por condicion en Step D):
                    # guardar cada elemento con clave indexada
                    # (D__W_specific__0, D__W_specific__1, ...).
                    for i, item in enumerate(v):
                        if isinstance(item, np.ndarray):
                            out[f"{key}__{i}"] = item
            except (ValueError, TypeError):
                pass


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    verbose = "INFO" if args.verbose else None

    # 0. Resolver directorio de salida
    if not args.no_save:
        out_dir = _resolve_out_dir(args)
    else:
        out_dir = None

    # 1. Construir matrices de Hankel
    single_mode = args.super_subject_id is not None

    if single_mode:
        print("=" * 70)
        print("  CONSTRUYENDO MATRICES DE HANKEL (SINGLE SUPER-SUBJECT)")
        print("=" * 70)

        X, hankel_info = build_hankel_single_ss(
            super_subject_id=args.super_subject_id,
            session=args.session,
            tasks=args.tasks,
            subjects_per_super_subject=args.subjects_per_super_subject,
            subject_start_offset=args.subject_start_offset,
            db_path=args.db_path,
            t_start=args.t_start,
            t_stop=args.t_end,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            hankel_depth=args.hankel_depth,
            verbose=verbose,
        )
    else:
        print("=" * 70)
        print("  CONSTRUYENDO MATRICES DE HANKEL DESDE SUPER-SUJETOS")
        print("=" * 70)

        X, hankel_info = build_hankel_from_eeg(
            session=args.session,
            tasks=args.tasks,
            n_super_subjects=args.n_super_subjects,
            total_subjects=args.total_subjects,
            subject_start_offset=args.subject_start_offset,
            db_path=args.db_path,
            t_start=args.t_start,
            t_stop=args.t_end,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            hankel_depth=args.hankel_depth,
            verbose=verbose,
        )

    # 2. Caracterizar
    characterization = characterize_hankel_matrices(X, hankel_info)
    print(characterization)

    n_valid = sum(
        1 for s in range(len(X)) for c in range(len(X[s]))
        if X[s][c].size > 0
    )
    if n_valid == 0:
        print("\n[ERROR] No se construyeron matrices validas.")
        return 1

    # 3. Configurar y ejecutar CD-HSA
    cfg = CDHSAConfig(
        fixed_rank=args.fixed_rank,
        rank_method=args.rank_method,
        var_explained=args.var_explained,
        var_max_rank=args.var_max_rank,
        var_min_rank=args.var_min_rank,
        a6_n_null=args.a6_n_null,
        a6_max_common=args.a6_max_common,
        a6_null_type=args.a6_null_type,
        a6_hankel_row_block=args.a6_hankel_row_block,
        bc_n_perm=args.bc_n_perm,
        bc_condition_names=list(args.tasks),
        bc_rank_outcome=args.rank_outcome,
        effective_rank=args.effective_rank,
        tangent_blocks_mode=args.tangent_blocks_mode,
        d_max_specific=args.d_max_specific,
        d_rank_adaptive=args.d_rank_adaptive,
        d_n_null_specific=args.d_n_null_specific,
        d_alpha_specific=args.d_alpha_specific,
        d_loso=args.d_loso,
        d_loso_n_perm=args.d_loso_n_perm,
        run_replica_consistency=args.run_replica_consistency,
        skip_bc=args.skip_bc,
        skip_tangent=args.skip_tangent,
        skip_d=args.skip_d,
    )

    # Nulo Hankel-preservante: autodetectar las filas de X por canal
    # (= depth del primer nivel, layout canal-major de la Hankel de
    # _build_multivariate_hankel) para la rotacion I_{L1} (x) Q fiel al
    # paper (Seccion 7).
    if cfg.a6_null_type == 'hankel' and cfg.a6_hankel_row_block == 0:
        depth_common = hankel_info.get('depth_common')
        if depth_common:
            cfg.a6_hankel_row_block = int(depth_common)
            print(f"  [a6-null=hankel] row_block autodetectado: "
                  f"{cfg.a6_hankel_row_block} (canales: "
                  f"{hankel_info.get('n_channels_common')})")
        else:
            print("  [WARN] a6-null-type=hankel sin row_block: no se pudo "
                  "autodetectar el depth del primer nivel; se usara "
                  "rotacion densa (legacy).")

    # --- Mascara de fronteras de concatenacion (paper, default ON) ---
    col_masks = None
    if not args.no_boundary_mask:
        col_masks = _compute_col_masks(X, hankel_info, args.L)
    S = len(X)
    print("\n" + "=" * 70)
    print("  EJECUTANDO CD-HSA")
    print("=" * 70)
    print(f"  Modo                  : {'single-SS' if single_mode else 'multi-SS'}")
    if single_mode:
        print(f"  Super-sujeto ID        : {args.super_subject_id}")
        print(f"  Subjects por SS       : {args.subjects_per_super_subject}")
    else:
        print(f"  Super-sujetos (S)     : {args.n_super_subjects}")
        print(f"  Subjects por SS       : {args.total_subjects // args.n_super_subjects}")
    print(f"  S (matrices)          : {S}")
    print(f"  Condiciones (C)       : {len(args.tasks)}")
    print(f"  L (embedding nivel 2) : {args.L}")
    if col_masks is not None:
        n_drop = sum(
            int(np.sum(~np.asarray(m).astype(bool)))
            for row in col_masks for m in row if m is not None
        )
        print(f"  Mascara de fronteras  : ON ({n_drop} columnas de nivel 2 "
              f"descartadas)")
    else:
        print("  Mascara de fronteras  : OFF (--no-boundary-mask o sin "
              "member_n_times)")
    if cfg.rank_method == 'variance':
        print(f"  rank_method           : variance (X={cfg.var_explained*100:.1f}%, "
              f"cap={cfg.var_max_rank}, piso={cfg.var_min_rank})")
    else:
        print(f"  rank_method           : {cfg.rank_method}")
        print(f"  fixed_rank            : {cfg.fixed_rank}")
    if cfg.effective_rank:
        print(f"  Rango efectivo X%     : ON (rank_outcome={cfg.bc_rank_outcome})")
    print(f"  a6_n_null             : {cfg.a6_n_null} (null={cfg.a6_null_type})")
    print(f"  bc_n_perm             : {cfg.bc_n_perm}")
    print(f"  tangent_blocks        : {cfg.tangent_blocks_mode}")
    print(f"  d_max_specific       : {cfg.d_max_specific} "
          f"(adaptive={cfg.d_rank_adaptive}, loso={cfg.d_loso})")
    if cfg.run_replica_consistency:
        print(f"  replica_consistency  : True (>=4/5)")
    if out_dir is not None:
        print(f"  Out dir               : {out_dir}")
    print("")
    sys.stdout.flush()

    result = run_cdhsa(X, args.L, cfg, col_masks=col_masks)

    # 4. Resultados
    print("")
    print(result.summary())

    # 5. Guardar
    if out_dir is not None:
        save_results(
            out_dir=out_dir,
            X=X,
            hankel_info=hankel_info,
            characterization=characterization,
            result=result,
            cfg=cfg,
            L=args.L,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
