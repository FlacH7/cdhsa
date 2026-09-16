#!/usr/bin/env python3
"""
run_cdhsa_ludovico.py - CDHSA pipeline for the Ludovico_01 dataset
=====================================================================

Treats each CSV file in the Ludovico_01 dataset as a different
"task/condition" for a **single subject**, then runs CDHSA to find
condition-specific residual modes (Step D).

Key differences from the standard test_retest_gedai pipeline:
  - S=1 (single subject), C=N_csv (conditions)
  - A6 is bypassed with a fixed rank (no cross-validation possible
    with S<2)
  - Steps B/C and tangent geometry are skipped (require S>=2 for
    within-subject permutation tests)
  - Step D is the primary output

The script does **NOT** modify any existing pipeline files.
All new code lives here.

Usage (CLI)::

    python run_cdhsa_ludovico.py \
        --db-path /path/to/DB_LUDOVICO_01_PATH \
        --out-dir /path/to/results/ludovico \
        --L 10 --hankel-depth 10 --fixed-rank 15 \
        --no-filter

Usage (programmatic)::

    from run_cdhsa_ludovico import run_ludovico_pipeline
    result = run_ludovico_pipeline(
        db_path="/path/to/DB_LUDOVICO_01_PATH",
        out_dir="/path/to/results/ludovico",
        L=10, hankel_depth=10, fixed_rank=15,
    )

Data loading
--------------
Uses ``src.latent_space_extraction.ludovico_01_eeg`` to load each CSV
as an MNE RawArray (channel x time, centred to zero mean).

Hankel construction
--------------------
A two-stage block-Hankel is built, matching the original pipeline:
  1. Stage-1 Hankel with ``hankel_depth`` on the raw (p, T) data
     -> H1 of shape (p * hankel_depth, T - hankel_depth + 1)
  2. Stage-2 Hankel with ``L`` inside CDHSA (a_common_subspace)
     -> H2 of shape (p * hankel_depth * L, ...)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


# =====================================================================
# Config
# =====================================================================

@dataclass
class LudovicoCDHSAConfig:
    """Configuration for the Ludovico_01 CDHSA pipeline."""
    L: int = 10
    hankel_depth: int = 10
    fixed_rank: int = 15
    d_max_specific: int = 10
    residual_rank_method: str = "local_gap"
    residual_rank_threshold: float = 0.1
    fixed_residual_rank: int = 5
    prevalence_quantile: float = 0.10
    sfreq: float = 1000.0
    apply_filter: bool = False
    l_freq: float = 1.0
    h_freq: float = 40.0
    t_start: float | None = None
    t_stop: float | None = None
    subjects: list[str] | None = None  # explicit list, None = auto-detect
    seed: int = 42


# =====================================================================
# Data loading
# =====================================================================

def load_ludovico_as_single_subject(
    db_path: str | Path,
    config: LudovicoCDHSAConfig | None = None,
    verbose: bool = True,
) -> tuple[list[NDArray[np.floating]], dict]:
    """Load all Ludovico_01 CSVs as conditions for a single subject.

    Each CSV file becomes one "condition" (task).  The result is
    conceptually X[0][c] for c = 0 .. C-1.

    Parameters
    ----------
    db_path : str or Path
        Root directory containing the CSV files.
    config : LudovicoCDHSAConfig, optional
    verbose : bool

    Returns
    -------
    X_raw : list of NDArray
        X_raw[c] shape (p, T) for each condition c.
    info : dict
        Metadata about the loaded data.
    """
    from src.latent_space_extraction.ludovico_01_eeg import (
        load_ludovico_01_from_ids,
        list_ludovico_01_subjects,
    )

    if config is None:
        config = LudovicoCDHSAConfig()

    db_path = Path(db_path)
    if not db_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {db_path}")

    # Resolve which subjects (CSV files) to load
    if config.subjects is not None:
        subjects = list(config.subjects)
    else:
        subjects = list_ludovico_01_subjects(db_path)

    if not subjects:
        raise FileNotFoundError(
            f"No CSV files found in {db_path}. "
            f"Expected files like data_XX_YY_ZZ.csv"
        )

    C = len(subjects)
    X_raw: list[NDArray[np.floating]] = []
    shapes: list[tuple[int, int]] = []
    ch_names: list[str] | None = None
    durations: list[float] = []

    if verbose:
        print(f"  Found {C} CSV files in {db_path}")
        print()

    t0 = time.time()

    for c_idx, subj_name in enumerate(subjects):
        tag = f"  [{c_idx + 1}/{C}] {subj_name}"
        if verbose:
            print(f"{tag} ...", end=" ", flush=True)

        try:
            raw = load_ludovico_01_from_ids(
                subject=subj_name,
                db_path=db_path,
                sfreq=config.sfreq,
                t_start=config.t_start,
                t_stop=config.t_stop,
                preload=True,
                verbose=False,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"SKIP ({exc})")
            X_raw.append(np.empty((0, 0)))
            shapes.append((0, 0))
            durations.append(0.0)
            continue

        # Optional bandpass filter (via MNE)
        if config.apply_filter:
            raw.filter(
                l_freq=config.l_freq,
                h_freq=config.h_freq,
                verbose=False,
            )

        data = raw.get_data().astype(np.float64)  # (p, T)

        if ch_names is None:
            ch_names = list(raw.ch_names)

        X_raw.append(data)
        shapes.append(data.shape)
        durations.append(raw.times[-1] if len(raw.times) > 0 else 0.0)

        if verbose:
            p, T = data.shape
            dur = T / config.sfreq
            print(
                f"OK  ch={p} T={T} "
                f"dur={dur:.2f}s (sfreq={config.sfreq:.0f})"
            )

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Load completed in {elapsed:.1f}s")

    # Validate channel consistency
    p = None
    valid_indices = []
    for c_idx, (x, sh) in enumerate(zip(X_raw, shapes)):
        if x.size == 0:
            continue
        if p is None:
            p = sh[0]
        elif sh[0] != p:
            print(
                f"  [WARN] {subjects[c_idx]} has {sh[0]} channels, "
                f"expected {p}. Skipping."
            )
            continue
        valid_indices.append(c_idx)

    if p is None or len(valid_indices) == 0:
        raise RuntimeError("No valid data loaded.")

    # Keep only valid conditions
    X_raw = [X_raw[i] for i in valid_indices]
    shapes = [shapes[i] for i in valid_indices]
    subjects_valid = [subjects[i] for i in valid_indices]
    durations = [durations[i] for i in valid_indices]

    info = {
        "dataset": "ludovico_01",
        "db_path": str(db_path),
        "subjects": subjects_valid,
        "S": 1,
        "C": len(subjects_valid),
        "p": p,
        "sfreq": config.sfreq,
        "t_start": config.t_start,
        "t_stop": config.t_stop,
        "apply_filter": config.apply_filter,
        "l_freq": config.l_freq,
        "h_freq": config.h_freq,
        "shapes": [shapes],  # wrapped in [s] for S=1
        "durations": durations,
        "elapsed_load": elapsed,
        "skipped_original": C - len(valid_indices),
        "ch_names": ch_names,
    }

    return X_raw, info


# =====================================================================
# Hankel construction
# =====================================================================

def build_hankel_ludovico(
    X_raw: list[NDArray[np.floating]],
    hankel_depth: int | None = None,
    verbose: bool = True,
) -> tuple[list[list[NDArray[np.floating]]], dict]:
    """Build stage-1 Hankel matrices from raw data.

    Each X_raw[c] of shape (p, T) becomes a Hankel H_c of shape
    (p * depth, T - depth + 1).  The result is wrapped as X[0][c]
    for compatibility with the CDHSA X[s][c] convention (S=1).

    Parameters
    ----------
    X_raw : list of NDArray, shape (p, T) each
    hankel_depth : int or None
        Embedding depth.  None = auto (min(10, T // 10)).

    Returns
    -------
    X : list[list[NDArray]]
        X[0][c] = Hankel matrix for condition c.
    info : dict
    """
    from src.cdhsa.a_common_subspace import build_block_hankel

    C = len(X_raw)
    p = X_raw[0].shape[0] if X_raw[0].size > 0 else 0

    X: list[list[NDArray[np.floating]]] = [[]]  # S = 1
    hankel_shapes: list[tuple[int, int] | None] = []
    depths_used: list[int] = []
    n_times_filtered: list[int] = []
    skipped: list[tuple[int, int, str]] = []

    t0 = time.time()

    if verbose:
        print(f"  Building stage-1 Hankel (C={C} conditions)...")
        sys.stdout.flush()

    for c in range(C):
        data = X_raw[c]
        if data.size == 0:
            X[0].append(np.empty((0, 0)))
            hankel_shapes.append(None)
            depths_used.append(0)
            n_times_filtered.append(0)
            skipped.append((0, c, "empty input"))
            continue

        _, T = data.shape

        # Resolve depth
        if hankel_depth is None:
            depth = min(10, T // 10)
            depth = max(depth, 2)
        else:
            depth = int(hankel_depth)

        if depth >= T:
            if verbose:
                print(f"  [{c + 1}/{C}] SKIP (depth={depth} >= T={T})")
            X[0].append(np.empty((0, 0)))
            hankel_shapes.append(None)
            depths_used.append(depth)
            n_times_filtered.append(T)
            skipped.append((0, c, f"depth={depth} >= T={T}"))
            continue

        H = build_block_hankel(data, depth)
        X[0].append(H)
        hankel_shapes.append(H.shape)
        depths_used.append(depth)
        n_times_filtered.append(T)

        if verbose:
            print(
                f"  [{c + 1}/{C}] {data.shape} -> "
                f"depth={depth} -> H={H.shape}"
            )
        sys.stdout.flush()

    elapsed = time.time() - t0

    info = {
        "mode": "ludovico_single_subject",
        "S": 1,
        "C": C,
        "p": p,
        "n_channels": p,
        "hankel_depth_requested": hankel_depth,
        "hankel_shapes": [hankel_shapes],
        "depths_used": depths_used,
        "n_times_filtered": n_times_filtered,
        "skipped": skipped,
        "elapsed_build": elapsed,
    }

    # Common values if all are the same
    if depths_used:
        info["depth_common"] = (
            depths_used[0] if len(set(depths_used)) == 1 else None
        )
    # p_hankel: row dimension of the Stage-1 Hankel = n_channels * depth
    # This is what CDHSASpecificModesDynamics (Stage 2) checks against
    if info["depth_common"] is not None:
        info["p_hankel"] = p * info["depth_common"]
        info["hankel_depth"] = info["depth_common"]

    return X, info


# =====================================================================
# Characterization
# =====================================================================

def characterize_hankel_ludovico(
    X: list[list[NDArray[np.floating]]],
    load_info: dict,
    hankel_info: dict,
) -> str:
    """Textual characterization of the Hankel matrices."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  CARACTERIZACION DE MATRICES DE HANKEL (LUDOVICO_01)")
    lines.append("=" * 70)

    S = hankel_info["S"]
    C = hankel_info["C"]
    subjects = load_info["subjects"]

    lines.append(f"  Sujetos (S)     : {S}")
    lines.append(f"  Condiciones (C) : {C}")
    lines.append(f"  Canales (p)     : {hankel_info['p']}")
    lines.append(f"  Depth pedido    : {hankel_info['hankel_depth_requested']}")
    if hankel_info.get("depth_common") is not None:
        lines.append(f"  Depth real      : {hankel_info['depth_common']}")
    lines.append(f"  Filtro          : {load_info.get('apply_filter', False)}")
    lines.append(f"  Saltados        : {len(hankel_info['skipped'])}")
    lines.append(f"  Tiempo construccion: {hankel_info['elapsed_build']:.1f} s")

    header = (
        f"  {'#':<4} {'Condicion':<20} "
        f"{'Forma H':<28} {'Rank':>6} {'Comp.':>8}  "
        f"{'Top-5 sing.vals'}"
    )
    lines.append("")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    n_valid = 0
    for c in range(C):
        H = X[0][c]
        name = subjects[c] if c < len(subjects) else f"c{c + 1}"

        if H.size == 0:
            lines.append(
                f"  {c + 1:<4} {name:<20} "
                f"{'(vacio)':<28}"
            )
            continue

        n_valid += 1
        m, n = H.shape

        k_svd = min(m, n, 50)
        try:
            from scipy.sparse.linalg import svds
            svals = svds(H, k=k_svd, return_singular_vectors=False)
            svals = np.sort(svals)[::-1]
            rank_est = int(np.sum(svals > svals[0] * 1e-6))
        except Exception:
            rank_est = -1
            svals = np.array([])

        comp = m / rank_est if rank_est > 0 else float("inf")
        top5 = ", ".join(f"{v:.1f}" for v in svals[:5])
        lines.append(
            f"  {c + 1:<4} {name:<20} "
            f"{str(H.shape):<28} "
            f"{rank_est:>6} {comp:>7.2f}x  {top5}"
        )

    lines.append("")
    lines.append(f"  Total matrices validas: {n_valid} / {S * C}")

    return "\n".join(lines)


# =====================================================================
# CDHSA pipeline for S=1
# =====================================================================

def run_cdhsa_ludovico_core(
    X: list[list[NDArray[np.floating]]],
    L: int,
    config: LudovicoCDHSAConfig | None = None,
    verbose: bool = True,
) -> dict:
    """Run CDHSA on Ludovico_01 data (S=1, C=N).

    Pipeline
    --------
    1. **A1-A5**: Common subspace estimation (works with S=1).
       Builds block-Hankel from the stage-1 Hankel matrices,
       estimates local ranks, and finds common population directions.

    2. **A6 bypass**: With S=1, cross-validation is not possible
       (``crossvalidate_common_rank`` requires S >= 2).  Instead,
       we directly take the top ``fixed_rank`` columns of W as W0.

    3. **B/C skipped**: Within-subject permutation tests require
       S >= 2.  With S=1 there is no cross-subject structure to test.

    4. **Tangent skipped**: Same reason as B/C.

    5. **D**: Condition-specific residual modes.  After removing
       W0, the residual bases are pooled within each condition.
       With S=1 this is simply the single subject's residual, but
       the singular values and modes are still informative.

    Parameters
    ----------
    X : list[list[NDArray]]
        X[0][c] = stage-1 Hankel for condition c, shape (p*depth, K).
    L : int
        CDHSA embedding depth (second stage).
    config : LudovicoCDHSAConfig

    Returns
    -------
    result : dict with keys 'R', 'A6', 'BC', 'G', 'D'
    """
    if config is None:
        config = LudovicoCDHSAConfig()

    from src.cdhsa.a_common_subspace import cdhsa_A1_A5
    from src.cdhsa.d_condition_specific import cdhsa_D_condition_specific_modes

    S = len(X)
    C = len(X[0])

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  EJECUTANDO CD-HSA (LUDOVICO_01)")
        print(f"{'=' * 70}")
        print(f"  Modo              : single-subject (S=1)")
        print(f"  S (matrices)      : {S}")
        print(f"  C (condiciones)   : {C}")
        print(f"  L (subespacio)    : {L}")
        print(f"  fixed_rank        : {config.fixed_rank}")
        print(f"  d_max_specific    : {config.d_max_specific}")
        print(f"  A6                : BYPASSED (fixed r0, no CV)")
        print(f"  B/C               : SKIPPED (S=1)")
        print(f"  Tangent           : SKIPPED (S=1)")
        print(f"  D                 : ACTIVE (goal)")
        print("")
        sys.stdout.flush()

    # =================================================================
    # A1-A5: Common subspace
    # =================================================================
    if verbose:
        print("  [A1-A5] Estimating common subspace...")
        sys.stdout.flush()

    t0 = time.time()
    R = cdhsa_A1_A5(
        X, L,
        rank_method="fixed",
        fixed_rank=config.fixed_rank,
        prevalence_quantile=config.prevalence_quantile,
    )
    t_a15 = time.time() - t0

    if verbose:
        print(f"  [A1-A5] Done in {t_a15:.1f}s")
        print(f"  [A1-A5] p={R['p']}, d=p*L={R['d']}, N=S*C={R['N']}")
        print(f"  [A1-A5] Common directions estimated: {len(R['lambda_'])}")
        for j in range(min(len(R['lambda_']), 15)):
            print(f"           j={j + 1:2d}: lambda={R['lambda_'][j]:.6f}")
        print()
        sys.stdout.flush()

    # =================================================================
    # A6 bypass: fixed rank (no CV with S=1)
    # =================================================================
    qmax = min(config.fixed_rank, R['W'].shape[1], R['W'].shape[0])
    r0 = qmax
    W0 = R['W'][:, :r0].copy()
    lambda0 = R['lambda_'][:r0].copy()

    A6 = {
        'r0': r0,
        'W0': W0,
        'lambda0': lambda0,
        'bypassed': True,
        'bypass_reason': 'S=1, cross-validation requires S>=2',
        'r_values': np.arange(1, r0 + 1, dtype=int),
        'lambda_observed': lambda0,
    }

    if verbose:
        print(f"  [A6]    BYPASSED -> fixed r0 = {r0}")
        print(f"  [A6]    W0 shape: {W0.shape}")
        for j in range(min(r0, 10)):
            print(
                f"           j={j + 1:2d}: "
                f"lambda0={lambda0[j]:.6f}"
            )
        print()
        sys.stdout.flush()

    # =================================================================
    # B/C: SKIPPED
    # =================================================================
    BC = None
    if verbose:
        print(
            "  [B/C]   SKIPPED -- within-subject permutation tests "
            "require S >= 2"
        )
        print()

    # =================================================================
    # Tangent: SKIPPED
    # =================================================================
    G = None
    if verbose:
        print(
            "  [TANG]  SKIPPED -- tangent geometry test requires S >= 2"
        )
        print()

    # =================================================================
    # D: Condition-specific modes (THE GOAL)
    # =================================================================
    if verbose:
        print("  [D]     Computing condition-specific residual modes...")
        sys.stdout.flush()

    t0 = time.time()
    D = cdhsa_D_condition_specific_modes(X, L, R, A6, opts={
        'max_specific': config.d_max_specific,
        'residual_rank_method': config.residual_rank_method,
        'residual_rank_threshold': config.residual_rank_threshold,
        'fixed_residual_rank': config.fixed_residual_rank,
        'prevalence_quantile': config.prevalence_quantile,
    })
    t_d = time.time() - t0

    if verbose:
        print(f"  [D]     Done in {t_d:.1f}s")
        print()
        sys.stdout.flush()

    return {
        'R': R,
        'A6': A6,
        'BC': BC,
        'G': G,
        'D': D,
    }


# =====================================================================
# Summary
# =====================================================================

def build_summary(result: dict, load_info: dict) -> str:
    """Build a human-readable summary of the CDHSA results."""
    lines = []
    lines.append("CD-HSA Results Summary (Ludovico_01)")
    lines.append("=" * 60)

    R = result['R']
    A6 = result['A6']
    D = result['D']
    subjects = load_info.get('subjects', [])

    lines.append(f"  Dataset     : ludovico_01")
    lines.append(f"  S           : {R['S']} (single subject)")
    lines.append(f"  C           : {R['C']} (conditions)")
    lines.append(f"  Channels    : {R['p']}")
    lines.append(f"  d = p * L   : {R['d']}")
    lines.append(f"  L           : {R['L_used']}")
    lines.append(f"  Common dir. : {len(R['lambda_'])}")

    lines.append(f"\n  A6 (bypassed, fixed): r0 = {A6['r0']}")
    for j in range(min(A6['r0'], 15)):
        lines.append(
            f"    j={j + 1:2d}: lambda0 = {A6['lambda0'][j]:.6f}"
        )

    lines.append(f"\n  B/C  : SKIPPED (S=1)")
    lines.append(f"  Tang : SKIPPED (S=1)")

    if D is not None:
        lines.append(f"\n  D Condition-specific modes:")
        for c in range(D['C']):
            rc = D['r_specific'][c]
            pc = D['prevalence_contrast'][c]
            mean_own = D['mean_own_alignment'][c]
            name = (
                subjects[c]
                if c < len(subjects)
                else f"Condition {c + 1}"
            )
            lines.append(f"    {name}:")
            lines.append(f"      Modes found     : {rc}")
            lines.append(f"      Mean own align. : {mean_own:.6f}")
            lines.append(
                f"      Prevalence contr.: {pc:.6f}"
            )
            lines.append(
                f"      Residual rank   : {D['residual_rank'][0, c]}"
            )

            # Top singular values
            if c < len(D['lambda_specific']):
                sv = D['lambda_specific'][c]
                if len(sv) > 0:
                    top_sv = ", ".join(
                        f"{v:.4f}" for v in sv[:5]
                    )
                    lines.append(f"      Top SVs         : {top_sv}")

            # Top mode info
            if c < len(D['W_specific']):
                Wc = D['W_specific'][c]
                if Wc.shape[1] > 0:
                    # Frobenius norm of each mode
                    mode_norms = np.linalg.norm(Wc, axis=0)
                    top_norms = ", ".join(
                        f"{v:.4f}" for v in mode_norms[:5]
                    )
                    lines.append(
                        f"      Mode norms      : {top_norms}"
                    )

    return "\n".join(lines)


# =====================================================================
# Save results
# =====================================================================

def _json_safe(obj: Any) -> Any:
    """Convert an object to something JSON-serializable."""
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
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


def _ensure_stacked_d_arrays(
    D: dict, arrays_dict: dict[str, NDArray]
) -> None:
    """Ensure D__W_specific and D__lambda_specific are in arrays_dict.

    CDHSASpecificModesDynamics (Stage 2) loads::

        npz_data["D__W_specific"]       # 3D (C, p*L, r_max) or 2D (p*L, sum_r)
        npz_data["D__lambda_specific"] # 2D (C, r_max) or 1D (sum_r,)
        npz_data["D__r_specific"]       # 1D (C,)

    The generic ``_save_arrays_recursive`` may have failed to stack them
    (different r_c per condition → np.array() raises ValueError).
    This function pads to uniform shape and saves the stacked versions.
    """
    W_list: list[NDArray] = D.get("W_specific", [])
    lam_list: list[NDArray] = D.get("lambda_specific", [])
    r_specific: NDArray = D["r_specific"]
    C = len(r_specific)

    if len(W_list) == 0:
        return

    # --- W_specific: pad to (C, p, r_max) ---
    if "D__W_specific" not in arrays_dict:
        p = W_list[0].shape[0]
        r_max = int(r_specific.max())
        W_padded = np.zeros((C, p, r_max), dtype=np.float64)
        for c in range(C):
            if c < len(W_list) and W_list[c].size > 0:
                rc = int(r_specific[c])
                W_padded[c, :, :rc] = W_list[c][:, :rc]
        arrays_dict["D__W_specific"] = W_padded

    # --- lambda_specific: pad to (C, r_max) ---
    if "D__lambda_specific" not in arrays_dict:
        r_max = int(r_specific.max())
        lam_padded = np.zeros((C, r_max), dtype=np.float64)
        for c in range(C):
            if c < len(lam_list) and len(lam_list[c]) > 0:
                rc = int(r_specific[c])
                lam_padded[c, :rc] = lam_list[c][:rc]
        arrays_dict["D__lambda_specific"] = lam_padded

    # --- r_specific should already be saved, but ensure it ---
    if "D__r_specific" not in arrays_dict:
        arrays_dict["D__r_specific"] = r_specific


def _save_arrays_recursive(
    d: dict, prefix: str, out: dict[str, NDArray]
) -> None:
    """Recursively extract arrays from a result dict.

    Lists of variable-sized arrays (e.g. W_specific, lambda_specific)
    are saved element-by-element with indexed keys like
    ``D__W_specific_c0``, ``D__W_specific_c1``, etc.
    """
    for k, v in d.items():
        key = f"{prefix}__{k}"
        if isinstance(v, np.ndarray):
            out[key] = v
        elif isinstance(v, dict):
            _save_arrays_recursive(v, key, out)
        elif isinstance(v, (list, tuple)) and len(v) > 0:
            if all(isinstance(item, np.ndarray) for item in v):
                # List of arrays — save each element separately
                # (they may have different shapes)
                for idx, item in enumerate(v):
                    out[f"{key}_c{idx}"] = item
            else:
                # Mixed list — try to stack, skip on failure
                try:
                    arr = np.array(v)
                    if arr.dtype.kind in ("f", "i", "u", "b"):
                        out[key] = arr
                except (ValueError, TypeError):
                    # List of lists of arrays (e.g. U_residual)
                    for idx, item in enumerate(v):
                        if isinstance(item, np.ndarray):
                            out[f"{key}_c{idx}"] = item
                        elif isinstance(item, (list, tuple)):
                            for idx2, sub in enumerate(item):
                                if isinstance(sub, np.ndarray):
                                    out[f"{key}_c{idx}_c{idx2}"] = sub


def save_results_ludovico(
    out_dir: Path,
    X: list[list[NDArray[np.floating]]],
    load_info: dict,
    hankel_info: dict,
    characterization: str,
    result: dict,
    config: LudovicoCDHSAConfig,
) -> None:
    """Save all results to out_dir.

    Files created::

        load_info.json         Metadata from data loading
        hankel_info.json       Metadata from Hankel construction
        config.json            Pipeline configuration used
        characterization.txt   Hankel matrix characterization
        cdhsa_summary.txt      Human-readable CDHSA results
        hankel_matrices.npz    Stage-1 Hankel matrices X[0][c]
        cdhsa_arrays.npz       Numerical arrays from CDHSA results
        condition_specific_modes.npz  Clean Step D output: W_specific
                                 and lambda_specific per condition
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Saving results in: {out_dir}")
    sys.stdout.flush()

    # --- 1. load_info.json ---
    with open(out_dir / "load_info.json", "w") as f:
        json.dump(_json_safe(load_info), f, indent=2, default=str)
    print("    [OK] load_info.json")

    # --- 2. hankel_info.json ---
    with open(out_dir / "hankel_info.json", "w") as f:
        json.dump(_json_safe(hankel_info), f, indent=2, default=str)
    print("    [OK] hankel_info.json")

    # --- 3. config.json ---
    cfg_dict = _json_safe(asdict(config))
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)
    print("    [OK] config.json")

    # --- 4. characterization.txt ---
    with open(out_dir / "characterization.txt", "w") as f:
        f.write(characterization)
    print("    [OK] characterization.txt")

    # --- 5. cdhsa_summary.txt ---
    summary_text = build_summary(result, load_info)
    with open(out_dir / "cdhsa_summary.txt", "w") as f:
        f.write(summary_text)
    print("    [OK] cdhsa_summary.txt")

    # --- 6. hankel_matrices.npz ---
    save_dict: dict[str, NDArray] = {}
    C = len(X[0])
    for c in range(C):
        key = f"H_c{c + 1}"
        H = X[0][c]
        if H.size > 0:
            save_dict[key] = H
        else:
            save_dict[key] = np.array([])
    np.savez_compressed(out_dir / "hankel_matrices.npz", **save_dict)
    print(f"    [OK] hankel_matrices.npz  ({len(save_dict)} matrices)")

    # --- 7. cdhsa_arrays.npz ---
    arrays_dict: dict[str, NDArray] = {}
    _save_arrays_recursive(result['R'], "R", arrays_dict)
    _save_arrays_recursive(result['A6'], "A6", arrays_dict)
    if result['D'] is not None:
        _save_arrays_recursive(result['D'], "D", arrays_dict)

        # Ensure D__W_specific and D__lambda_specific exist as stacked
        # arrays that CDHSASpecificModesDynamics (Stage 2) can load.
        # _save_arrays_recursive may save them as D__W_specific_c0, c1,
        # ... if the list elements have different shapes (np.array()
        # fails silently).  We pad to uniform shape here.
        _ensure_stacked_d_arrays(result['D'], arrays_dict)

    np.savez_compressed(out_dir / "cdhsa_arrays.npz", **arrays_dict)
    print(f"    [OK] cdhsa_arrays.npz  ({len(arrays_dict)} arrays)")

    # --- 8. condition_specific_modes.npz (clean D output) ---
    if result["D"] is not None:
        D = result["D"]
        d_dict: dict[str, NDArray] = {}
        for c in range(D["C"]):
            name = load_info.get("subjects", [f"c{c+1}"])[c]
            if c < len(D["W_specific"]) and D["W_specific"][c].size > 0:
                d_dict[f"W_specific_{name}"] = D["W_specific"][c]
            if c < len(D["lambda_specific"]) and len(D["lambda_specific"][c]) > 0:
                d_dict[f"lambda_specific_{name}"] = D["lambda_specific"][c]
        d_dict["r_specific"] = D["r_specific"]
        d_dict["prevalence_contrast"] = D["prevalence_contrast"]
        d_dict["alignment_specific"] = D["alignment_specific"]
        d_dict["residual_rank"] = D["residual_rank"]
        d_dict["mean_own_alignment"] = D["mean_own_alignment"]
        if D["alignment_cross"].size > 0:
            d_dict["alignment_cross"] = D["alignment_cross"]
            d_dict["mean_cross_alignment"] = D["mean_cross_alignment"]
        np.savez_compressed(out_dir / "condition_specific_modes.npz", **d_dict)
        print(f"    [OK] condition_specific_modes.npz ({len(d_dict)} arrays)")

    print(f"  Done. {len(list(out_dir.iterdir()))} files in {out_dir}")


# =====================================================================
# High-level entry point
# =====================================================================

def run_ludovico_pipeline(
    db_path: str | Path,
    out_dir: str | Path,
    config: LudovicoCDHSAConfig | None = None,
    verbose: bool = True,
) -> dict:
    """End-to-end CDHSA pipeline for Ludovico_01.

    Parameters
    ----------
    db_path : str or Path
        Path to the Ludovico_01 dataset root.
    out_dir : str or Path
        Directory to save results.
    config : LudovicoCDHSAConfig, optional
    verbose : bool

    Returns
    -------
    result : dict
        The CDHSA result dict with keys R, A6, BC, G, D.
    """
    if config is None:
        config = LudovicoCDHSAConfig()

    # 1. Load
    print("=" * 70)
    print("  LUDOVICO_01 CDHSA PIPELINE")
    print("=" * 70)
    print("\n--- Loading data ---")
    X_raw, load_info = load_ludovico_as_single_subject(
        db_path, config, verbose=verbose
    )

    # 2. Hankel
    print("\n--- Building Hankel matrices ---")
    X, hankel_info = build_hankel_ludovico(
        X_raw, hankel_depth=config.hankel_depth, verbose=verbose
    )

    # 3. Characterize
    characterization = characterize_hankel_ludovico(X, load_info, hankel_info)
    print(characterization)

    # Validate
    n_valid = sum(
        1 for c in range(len(X[0])) if X[0][c].size > 0
    )
    if n_valid == 0:
        print("\n[ERROR] No valid Hankel matrices constructed.")
        return {'R': None, 'A6': None, 'BC': None, 'G': None, 'D': None}

    # 4. CDHSA
    result = run_cdhsa_ludovico_core(
        X, config.L, config, verbose=verbose
    )

    # 5. Summary
    summary = build_summary(result, load_info)
    print(summary)

    # 6. Save
    save_results_ludovico(
        out_dir=out_dir,
        X=X,
        load_info=load_info,
        hankel_info=hankel_info,
        characterization=characterization,
        result=result,
        config=config,
    )

    return result


# =====================================================================
# CLI
# =====================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CD-HSA pipeline for the Ludovico_01 dataset. "
            "Treats each CSV as a condition for a single subject "
            "and extracts condition-specific modes (Step D)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--db-path", type=str, default=None,
        help=(
            "Path to the Ludovico_01 dataset root. "
            "If omitted, uses DB_LUDOVICO_01_PATH from src.utils.config."
        ),
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help=(
            "Output directory for results. "
            "If omitted, uses BASE_RESULTS_PATH from src.utils.config."
        ),
    )

    # CDHSA parameters
    parser.add_argument("--L", type=int, default=10,
                       help="CDHSA subspace dimension (default: 10)")
    parser.add_argument("--hankel-depth", type=int, default=10,
                       help="Stage-1 Hankel depth (default: 10)")
    parser.add_argument("--fixed-rank", type=int, default=15,
                       help="Fixed rank for A1-A5 and A6 bypass (default: 15)")
    parser.add_argument("--d-max-specific", type=int, default=10,
                       help="Max condition-specific modes per condition (default: 10)")
    parser.add_argument("--residual-rank-method", type=str, default="local_gap",
                       choices=["local_gap", "local_threshold", "fixed"])
    parser.add_argument("--prevalence-quantile", type=float, default=0.10)

    # Preprocessing
    parser.add_argument("--no-filter", action="store_true",
                       help="Skip bandpass filter (default: skip for non-EEG data)")
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--sfreq", type=float, default=1000.0)

    # Time window
    parser.add_argument("--t-start", type=float, default=None)
    parser.add_argument("--t-stop", type=float, default=None)

    # Subject selection
    parser.add_argument(
        "--subjects", type=str, nargs="+", default=None,
        help="Explicit list of CSV names (without .csv). Default: auto-detect.",
    )

    return parser.parse_args(argv)


def _resolve_db_path(args: argparse.Namespace) -> Path:
    """Resolve the dataset path from args or config."""
    if args.db_path is not None:
        return Path(args.db_path)
    try:
        from src.utils.config import DB_LUDOVICO_01_PATH
        return Path(DB_LUDOVICO_01_PATH)
    except (ImportError, AttributeError):
        raise ValueError(
            "--db-path is required when DB_LUDOVICO_01_PATH is not in config."
        )


def _resolve_out_dir(args: argparse.Namespace) -> Path:
    """Resolve the output directory from args or config."""
    if args.out_dir is not None:
        return Path(args.out_dir)
    try:
        from src.utils.config import BASE_RESULTS_PATH
        base = Path(BASE_RESULTS_PATH)
    except (ImportError, AttributeError):
        base = Path("./results")
    return base / "ludovico_01" / "cdhsa"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    db_path = _resolve_db_path(args)
    out_dir = _resolve_out_dir(args)

    config = LudovicoCDHSAConfig(
        L=args.L,
        hankel_depth=args.hankel_depth,
        fixed_rank=args.fixed_rank,
        d_max_specific=args.d_max_specific,
        residual_rank_method=args.residual_rank_method,
        prevalence_quantile=args.prevalence_quantile,
        sfreq=args.sfreq,
        apply_filter=not args.no_filter,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        t_start=args.t_start,
        t_stop=args.t_stop,
        subjects=args.subjects,
    )

    try:
        run_ludovico_pipeline(
            db_path=db_path,
            out_dir=out_dir,
            config=config,
        )
        return 0
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
