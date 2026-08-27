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
    a6_max_common: int = 0
    a6_n_folds: int = 5
    a6_n_null: int = 100
    a6_alpha: float = 0.05
    a6_seed: int = 1234
    bc_blocks: list | None = None
    bc_energy_metric: str = "log_absolute"
    bc_geometry_metric: str = "adjusted"
    bc_n_perm: int = 5000
    bc_seed: int = 20260812
    bc_alpha: float = 0.05
    bc_condition_names: list[str] = field(default_factory=list)
    tangent_blocks: list | None = None
    tangent_n_perm: int = 5000
    tangent_seed: int = 9999
    tangent_alpha: float = 0.05
    d_max_specific: int = 10
    d_residual_rank_method: str = "local_gap"
    d_residual_rank_threshold: float = 0.1
    d_fixed_residual_rank: int = 5
    skip_bc: bool = False
    skip_tangent: bool = False
    skip_d: bool = False
    seed: int = 42


# =====================================================================
# CDHSAResult
# =====================================================================

class CDHSAResult:
    """Container for all CD-HSA results."""

    def __init__(self, config, R, A6=None, BC=None, G=None, D=None):
        self.config = config
        self.R = R
        self.A6 = A6
        self.BC = BC
        self.G = G
        self.D = D

    def summary(self) -> str:
        lines = []
        lines.append("CD-HSA Results Summary")
        lines.append("=" * 60)
        R = self.R
        lines.append(f"  Subjects: {R['S']}, Conditions: {R['C']}, "
                     f"Channels: {R['p']}, d = {R['d']}")
        lines.append(f"  Common directions estimated: {len(R['lambda_'])}")
        if self.A6 is not None:
            lines.append(f"\n  A6 Common rank r0 = {self.A6['r0']}")
            if self.A6['r0'] > 0:
                for j in range(self.A6['r0']):
                    lines.append(
                        f"    j={j+1}: lambda={self.A6['lambda0'][j]:.4f}"
                    )
        if self.BC is not None:
            lines.append(f"\n  B/C Condition tests (alpha={self.config.bc_alpha}):")
            for k, name in enumerate(self.BC['block_names']):
                sig_e = "*" if self.BC['sig_energy_maxF'][k] else ""
                sig_g = "*" if self.BC['sig_geometry_maxF'][k] else ""
                lines.append(
                    f"    {name}: energy_F="
                    f"{self.BC['summary']['energy_F'][k]:.2f}"
                    f"{sig_e}  geom_F="
                    f"{self.BC['summary']['geometry_F'][k]:.2f}"
                    f"{sig_g}"
                )
        if self.G is not None:
            lines.append(f"\n  Tangent geometry test:")
            for k in range(len(self.G['T_obs'])):
                sig = "*" if self.G['sig_maxT'][k] else ""
                lines.append(
                    f"    block {k+1}: T={self.G['T_obs'][k]:.4f}"
                    f"  p_maxT={self.G['p_maxT'][k]:.4f}{sig}"
                )
        if self.D is not None:
            lines.append(f"\n  D Condition-specific modes:")
            for c in range(self.D['C']):
                rc = self.D['r_specific'][c]
                pc = self.D['prevalence_contrast'][c]
                lines.append(
                    f"    Condition {c+1}: {rc} modes,"
                    f"  prevalence contrast = {pc:.4f}"
                )
        return "\n".join(lines)


# =====================================================================
# run_cdhsa (original)
# =====================================================================

def run_cdhsa(
    X: list[list[NDArray[np.floating]]],
    L: int,
    config: CDHSAConfig | None = None,
) -> CDHSAResult:
    """Run the full CD-HSA pipeline.

    Parameters
    ----------
    X : list of list of arrays, shape (p, T)
        X[s][c] is the data matrix (e.g. Hankel) for subject s,
        condition c, with shape (p, T).
    L : int
        Subspace dimension to analyse.
    config : CDHSAConfig, optional

    Returns
    -------
    result : CDHSAResult
    """
    from src.cdhsa.a_common_subspace import cdhsa_A1_A5
    from src.cdhsa.a6_common_rank import cdhsa_A6_common_rank
    from src.cdhsa.b_energy import cdhsa_BC_condition_tests
    from src.cdhsa.c_geometry import cdhsa_tangent_geometry_test
    from src.cdhsa.d_condition_specific import cdhsa_D_condition_specific_modes

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
    )

    a6_max = (config.a6_max_common if config.a6_max_common > 0
             else min(20, len(R['lambda_'])))
    A6 = cdhsa_A6_common_rank(R, opts={
        'max_common': a6_max,
        'n_folds': config.a6_n_folds,
        'n_null': config.a6_n_null,
        'alpha': config.a6_alpha,
        'seed': config.a6_seed,
    })

    BC = None
    G = None
    D = None

    if not config.skip_bc and A6['r0'] >= 1:
        bc_opts = {
            'blocks': config.bc_blocks,
            'energy_metric': config.bc_energy_metric,
            'geometry_metric': config.bc_geometry_metric,
            'n_perm': config.bc_n_perm,
            'seed': config.bc_seed,
            'alpha': config.bc_alpha,
        }
        if config.bc_condition_names:
            bc_opts['condition_names'] = config.bc_condition_names
        BC = cdhsa_BC_condition_tests(X, L, R, A6, opts=bc_opts)

    if not config.skip_tangent and A6['r0'] >= 1:
        t_blocks = config.tangent_blocks
        if t_blocks is None:
            t_blocks = [np.arange(1, A6['r0'] + 1, dtype=int)]
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
        })

    return CDHSAResult(config=config, R=R, A6=A6, BC=BC, G=G, D=D)


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
        Metadata completa de la construccion.
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
                raw = load_super_subject_eeg(
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
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"SKIP ({exc})")
                load_errors.append((ss_id - 1, c_idx, str(exc)))
                continue

            raws_store[(ss_id - 1, c_idx)] = raw
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

    t0_global = time.time()

    for ss_id in range(1, S + 1):
        ss_label = f"super_subject-{ss_id:02d}"
        X_s: list[NDArray[np.floating]] = []
        shapes_s: list[tuple[int, int] | None] = []
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
                continue

            raw = raws_store.pop(key)  # liberar memoria
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
                skipped.append((ss_id - 1, c_idx,
                    f"depth={depth} >= n_times={n_times}"))
                continue

            H = _build_multivariate_hankel(X_filtered, depth)
            del X_filtered  # liberar memoria

            X_s.append(H)
            shapes_s.append(H.shape)
            sfreqs.append(sfreq)
            depths_used.append(depth)
            n_channels_list.append(n_ch)
            n_times_filtered_list.append(n_times)
            durations.append(duration_s)

            print(f"OK  ch={n_ch} T={n_times} "
                  f"dur={duration_s:.0f}s "
                  f"depth={depth} -> H={H.shape}")
            sys.stdout.flush()

        X.append(X_s)
        shapes.append(shapes_s)

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

    t0 = time.time()

    for c_idx, task in enumerate(tasks):
        tag = f"  [{c_idx+1}/{C}] {task} ..."
        print(tag, end=" ")
        sys.stdout.flush()

        try:
            raw = load_super_subject_eeg(
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
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"SKIP ({exc})")
            X_s.append(np.empty((0, 0)))
            shapes_s.append(None)
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
            skipped.append((0, c_idx,
                f"depth={depth} >= n_times={n_times}"))
            continue

        H = _build_multivariate_hankel(X_filtered, depth)
        del X_filtered

        X_s.append(H)
        shapes_s.append(H.shape)
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
        help="Dimension del subespacio para CD-HSA",
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
                   choices=["fixed", "reproducibility"])
    g.add_argument("--a6-n-null", type=int, default=100)
    g.add_argument("--bc-n-perm", type=int, default=5000)
    g.add_argument("--d-max-specific", type=int, default=10,
                   help="Maximos modos especificos por condicion (Step D). "
                   "Default: 10")
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


def _resolve_out_dir(args: argparse.Namespace) -> Path:
    """Resolver el directorio de salida y crear la subcarpeta.

    Estructura::

        {BASE_RESULTS_PATH}/cdhsa/{session}/
            nSS{n}_L{L}_fr{fr}_a6n{a6}_bcn{bc}/
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

    out_dir = Path(
        f"{base}/cdhsa/{args.session}"
        f"/{ss_label}_L{args.L}"
        f"_fr{args.fixed_rank}_a6n{args.a6_n_null}_bcn{args.bc_n_perm}"
        f"/{args.l_freq}-{args.h_freq}Hz"
        f"_depth{args.hankel_depth or 'auto'}"
        f"/from{t_start_tag}_to{t_end_tag}"
        f"_{'_'.join(args.tasks)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


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
        a6_n_null=args.a6_n_null,
        bc_n_perm=args.bc_n_perm,
        bc_condition_names=list(args.tasks),
        d_max_specific=args.d_max_specific,
        skip_bc=args.skip_bc,
        skip_tangent=args.skip_tangent,
        skip_d=args.skip_d,
    )

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
    print(f"  L (subespacio)        : {args.L}")
    print(f"  fixed_rank            : {cfg.fixed_rank}")
    print(f"  a6_n_null             : {cfg.a6_n_null}")
    print(f"  bc_n_perm             : {cfg.bc_n_perm}")
    if out_dir is not None:
        print(f"  Out dir               : {out_dir}")
    print("")
    sys.stdout.flush()

    result = run_cdhsa(X, args.L, cfg)

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
