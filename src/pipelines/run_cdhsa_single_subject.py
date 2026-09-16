"""
src.pipelines.run_cdhsa_single_subject - CD-HSA over individual subjects
========================================================================

Dedicated CLI pipeline that builds Hankel matrices from **individual
subjects** (S = n_subjects, C = tasks) instead of super-subjects, and
then runs the memory-optimized CD-HSA core (``run_cdhsa_v2``).

A separate CLI is provided (rather than a third mode inside
``run_cdhsa.py``) because the original CLI already has a mutually
exclusive super-subject group (``--n-super-subjects`` /
``--super-subject-id``); adding a third mode there would pollute the
existing CLI.

The CD-HSA core is agnostic: it only receives ``X[s][c]`` (Hankel
matrix of subject s under condition c).  The ONLY thing that changes
with respect to ``run_cdhsa.py`` is how ``X`` is built: each ``s`` is a
single subject (no time-axis concatenation).

Usage (CLI)::::

    # 60 sujetos individuales (S=60) con 2 tareas (C=2)
    python -m src.pipelines.run_cdhsa_single_subject \\
        --session session1 \\
        --n-subjects 60 \\
        --tasks eyesclosed music \\
        --t-start 100 --t-end 200 \\
        --L 10 --hankel-depth 10 \\
        --fixed-rank 20 --a6-n-null 500 --bc-n-perm 5000

    # Lista explicita de sujetos
    python -m src.pipelines.run_cdhsa_single_subject \\
        --session session1 \\
        --subject-ids 1 2 3 5 8 \\
        --tasks eyesclosed music \\
        --L 10

Pipeline interno
-----------------
Para cada par (sujeto, condicion)::::

    1. Cargar EEG individual   -> load_single_subject_eeg()
    2. Pick canales comunes    -> raw.pick(global_channels)
       (interseccion GLOBAL sobre todos los sujetos y condiciones)
    3. Filtro pasa-banda       -> extract_filtered_data_matrix()
       (X_filtered: n_channels x n_times, centrada a media cero)
    4. Matriz de Hankel        -> _build_multivariate_hankel()
       (H: (n_channels * depth) x (n_times - depth + 1))

Despues: characterize_hankel_matrices() -> run_cdhsa_v2() -> save_results()
(ambas reutilizadas sin modificar de src.pipelines.run_cdhsa /
src.pipelines.run_cdhsa_v2).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import mne

from src.pipelines.run_cdhsa import (
    CDHSAConfig,
    characterize_hankel_matrices,
    save_results,
)
from src.pipelines.run_cdhsa_v2 import run_cdhsa_v2


# =====================================================================
# Construccion de matrices de Hankel desde sujetos individuales
# =====================================================================

def build_hankel_from_single_subjects(
    *,
    session: str,
    tasks: list[str],
    subject_ids: list[int] | None = None,
    n_subjects: int = 60,
    subject_start_offset: int = 1,
    db_path: str | Path | None = None,
    t_start: float | None = None,
    t_stop: float | None = None,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    hankel_depth: int | None = None,
    verbose: bool | str | None = None,
) -> tuple[list[list[NDArray[np.floating]]], dict]:
    """Construir matrices de Hankel para cada par (sujeto, condicion).

    Cada entrada ``s`` de ``X`` es UN sujeto individual (sin
    concatenacion temporal).  Se usa una interseccion GLOBAL de canales
    entre todos los sujetos y condiciones para garantizar que todas las
    matrices de Hankel tengan el mismo numero de filas ``p``.

    Pipeline por (sujeto, condicion)::::

        0. Cargar todos los raws -> interseccion global de canales
        1. Pick canales comunes  -> raw.pick(global_channels)
        2. Filtro pasa-banda    -> extract_filtered_data_matrix()
           X_filtered: (n_channels, n_times), centrada a media cero
        3. Matriz de Hankel     -> _build_multivariate_hankel()
           H: (n_channels * depth, n_times - depth + 1)

    Parameters
    ----------
    session : str
        Session ID (ej: ``"session1"``).
    tasks : list[str]
        Condiciones/tareas (C = len(tasks)).
    subject_ids : list[int] | None
        Lista explicita de sujetos.  Si es ``None``, se auto-resuelve
        como ``range(subject_start_offset, subject_start_offset + n_subjects)``.
    n_subjects : int, default 60
        Numero de sujetos para la auto-resolucion.
    subject_start_offset : int, default 1
        Indice del primer sujeto (auto-resolucion).
    db_path : str | Path | None
        Raiz del dataset Gedai.  Default: ``DB_TEST_RETEST_GEDAI_PATH``.
    t_start, t_stop : float | None
        Ventana temporal POR SUJETO (segundos).
    l_freq, h_freq : float
        Filtro pasa-banda (Hz).
    hankel_depth : int | None
        Profundidad Hankel.  ``None`` = auto (``_auto_embedding_depth``).
    verbose : bool | str | None
        Nivel de verbosidad MNE.

    Returns
    -------
    X : list[list[NDArray]]
        X[s][c] = Hankel para el sujeto s, condicion c.
    info : dict
        Metadata completa de la construccion (compatible con
        ``characterize_hankel_matrices`` de ``src.pipelines.run_cdhsa``).
    """
    from src.latent_space_extraction.single_subject_eeg import (
        load_single_subject_eeg,
        resolve_single_subject_ids,
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

    subject_ids = resolve_single_subject_ids(
        n_subjects=n_subjects,
        subject_start_offset=subject_start_offset,
        subject_ids=subject_ids,
    )

    S = len(subject_ids)
    C = len(tasks)

    print(f"  Modo single-subject: {S} sujetos individuales "
          f"(subs {subject_ids[0]}..{subject_ids[-1]}), "
          f"{C} condiciones")
    print()

    # =================================================================
    # PASADA 0: Cargar todos los raws y calcular interseccion global
    # =================================================================
    print("  PASADA 0: Cargando raws para calcular interseccion global de canales...")
    sys.stdout.flush()

    raws_store: dict[tuple[int, int], "mne.io.Raw"] = {}  # (s_idx, c_idx) -> raw
    load_errors: list[tuple[int, int, str]] = []

    t0_load = time.time()
    for s_idx, subject_id in enumerate(subject_ids):
        subj_label = f"sub-{subject_id:02d}"
        for c_idx, task in enumerate(tasks):
            tag = f"  [{s_idx + 1}/{S}] {subj_label}/{session}/{task}"
            print(f"{tag} ...", end=" ")
            sys.stdout.flush()

            try:
                raw = load_single_subject_eeg(
                    subject_id=subject_id,
                    session=session,
                    task=task,
                    db_path=db_path,
                    t_start=t_start,
                    t_stop=t_stop,
                    preload=True,
                    verbose=False,
                )
            except Exception as exc:  # tolerancia a sujetos faltantes/corruptos
                print(f"SKIP ({exc})")
                load_errors.append((s_idx, c_idx, str(exc)))
                continue

            raws_store[(s_idx, c_idx)] = raw
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

    for s_idx, subject_id in enumerate(subject_ids):
        subj_label = f"sub-{subject_id:02d}"
        X_s: list[NDArray[np.floating]] = []
        shapes_s: list[tuple[int, int] | None] = []
        # Cada sujeto individual es su propio "super-sujeto" de tamano 1
        super_subject_ids_list.append([subject_id])

        for c_idx, task in enumerate(tasks):
            tag = f"[{s_idx + 1}/{S}] {subj_label}/{session}/{task}"

            key = (s_idx, c_idx)
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
                skipped.append((s_idx, c_idx,
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
        "mode": "single_subject",
        "session": session,
        "tasks": tasks,
        "subject_ids": list(subject_ids),
        "subject_start_offset": subject_start_offset,
        # Claves requeridas por characterize_hankel_matrices():
        "S": S,
        "C": C,
        "subjects_per_super_subject": 1,
        "total_subjects": len(subject_ids),
        "super_subject_ids": super_subject_ids_list,
        "l_freq": l_freq,
        "h_freq": h_freq,
        "hankel_depth_requested": hankel_depth,
        "skipped": skipped,
        "elapsed_build": elapsed,
        # Metadata adicional del modo single-subject:
        "global_channels": global_channels,
        "n_global_channels": len(global_channels),
        "shapes": shapes,
        "sfreqs": sfreqs,
        "depths_used": depths_used,
        "n_channels": n_channels_list,
        "n_times_filtered": n_times_filtered_list,
        "durations": durations,
        "elapsed_load": time.time() - t0_load,
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


# =====================================================================
# CLI
# =====================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CD-HSA sobre sujetos individuales: construye matrices de "
            "Hankel por sujeto (S = n_subjects, sin concatenar) del "
            "dataset Gedai y ejecuta el analisis CD-HSA (v2)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Fuente de datos (sujetos individuales) ---
    # --n-subjects N (auto: range(offset, offset+N))  XOR
    # --subject-ids i1 i2 ... (lista explicita)
    # Si no se da ninguno: default implicito n_subjects=60.
    mode_subj = parser.add_mutually_exclusive_group()
    mode_subj.add_argument(
        "--n-subjects", type=int, default=None,
        help=("Cantidad de sujetos individuales (S). "
              "Se toman range(subject_start_offset, offset + N). "
              "Default: 60 si no se da --subject-ids."),
    )
    mode_subj.add_argument(
        "--subject-ids", type=int, nargs="+", default=None,
        help=("Lista explicita de sujetos individuales "
              "(ej: --subject-ids 1 2 3 5 8)."),
    )

    parser.add_argument(
        "--session", type=str, required=True,
        help="Session ID (ej: session1)",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", required=True,
        help="Condiciones/tareas (ej: eyesclosed music)",
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
# Directorio de salida
# =====================================================================

def _resolve_out_dir(args: argparse.Namespace) -> Path:
    """Resolver el directorio de salida y crear la subcarpeta.

    Espejo de ``run_cdhsa._resolve_out_dir`` pero con la etiqueta
    ``nSub{N}`` (N = numero de sujetos individuales).

    Estructura::

        {BASE_RESULTS_PATH}/cdhsa/{session}/
            nSub{N}_L{L}_fr{fr}_a6n{a6}_bcn{bc}/
            {l_freq}-{h_freq}Hz_depth{d}/
            from{t0}s_to{t1}s_{tasks}
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

    if args.subject_ids is not None:
        n_subjects = len(args.subject_ids)
    else:
        n_subjects = args.n_subjects if args.n_subjects is not None else 60

    subj_label = f"nSub{n_subjects}"

    out_dir = Path(
        f"{base}/cdhsa/{args.session}"
        f"/{subj_label}_L{args.L}"
        f"_fr{args.fixed_rank}_a6n{args.a6_n_null}_bcn{args.bc_n_perm}"
        f"/{args.l_freq}-{args.h_freq}Hz"
        f"_depth{args.hankel_depth or 'auto'}"
        f"/from{t_start_tag}_to{t_end_tag}"
        f"_{'_'.join(args.tasks)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# =====================================================================
# MAIN
# =====================================================================

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    verbose = "INFO" if args.verbose else None

    # 0. Resolver directorio de salida
    if not args.no_save:
        out_dir = _resolve_out_dir(args)
    else:
        out_dir = None

    # 1. Construir matrices de Hankel (sujetos individuales)
    print("=" * 70)
    print("  CONSTRUYENDO MATRICES DE HANKEL (SINGLE-SUBJECT)")
    print("=" * 70)

    X, hankel_info = build_hankel_from_single_subjects(
        session=args.session,
        tasks=args.tasks,
        subject_ids=args.subject_ids,
        n_subjects=args.n_subjects if args.n_subjects is not None else 60,
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

    # 3. Configurar y ejecutar CD-HSA (v2: memory-optimized)
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
    print("  EJECUTANDO CD-HSA (v2: memory-optimized)")
    print("=" * 70)
    print(f"  Modo                  : single-subject")
    print(f"  Sujetos (S)           : {S}")
    print(f"  Condiciones (C)       : {len(args.tasks)}")
    print(f"  L (subespacio)        : {args.L}")
    print(f"  fixed_rank            : {cfg.fixed_rank}")
    print(f"  a6_n_null             : {cfg.a6_n_null}")
    print(f"  bc_n_perm             : {cfg.bc_n_perm}")
    if out_dir is not None:
        print(f"  Out dir               : {out_dir}")
    print("")
    sys.stdout.flush()

    result = run_cdhsa_v2(X, args.L, cfg)

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
