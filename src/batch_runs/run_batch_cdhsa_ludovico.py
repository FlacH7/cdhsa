#!/usr/bin/env python3
"""
run_batch_cdhsa_ludovico.py
==========================
Batch executor for the CD-HSA pipeline on the **Ludovico_01** dataset.

Reads parameters from ``cdhsa_batch_params_ludovico.json`` and delegates
to ``run_cdhsa_ludovico.run_ludovico_pipeline``.

Unlike the standard batch runner (which handles multi-SS / per-SS modes),
this runner is much simpler because Ludovico_01 has a single
configuration: S=1, C=N_csv_files.

After the pipeline completes, it extracts the top-N condition-specific
mode indices and saves a ``mode_map.json`` to ``params_dir`` (mirrors
the post-processing done by the standard batch runner).

Usage::

    # Default (reads cdhsa_batch_params_ludovico.json in same directory)
    python run_batch_cdhsa_ludovico.py

    # Custom JSON
    python run_batch_cdhsa_ludovico.py --params-json /path/to/params.json

    # Override db-path and out-dir via CLI
    python run_batch_cdhsa_ludovico.py --db-path /data/ludovico --out-dir /results/ludovico
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the directory containing the scripts is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_cdhsa_ludovico")

# ---------------------------------------------------------------------------
# Default JSON path
# ---------------------------------------------------------------------------
DEFAULT_PARAMS_JSON = _SCRIPT_DIR / "cdhsa_batch_params_ludovico.json"


# ===================================================================
# JSON loading + validation
# ===================================================================


def _load_params(json_path: Path) -> dict:
    """Load and validate the Ludovico batch JSON parameters."""
    if not json_path.exists():
        logger.error("JSON not found: %s", json_path)
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as fh:
        params = json.load(fh)

    # Minimal validation
    if "cdhsa_params" not in params:
        logger.error("Missing 'cdhsa_params' in JSON.")
        sys.exit(1)

    cdhsa = params["cdhsa_params"]
    if "L" not in cdhsa:
        logger.error("'cdhsa_params.L' is required.")
        sys.exit(1)

    return params


# ===================================================================
# Banner
# ===================================================================


def _print_banner(
    params: dict,
    db_path: Path,
    out_dir: Path,
    params_dir: Path,
):
    cdhsa = params["cdhsa_params"]
    pre = params.get("preprocessing", {})

    logger.info("=" * 70)
    logger.info("  BATCH CD-HSA -- LUDOVICO_01")
    logger.info("=" * 70)
    logger.info("  Experiment  : %s", params.get("experiment_label", "N/A"))
    logger.info("  Dataset     : ludovico_01")
    logger.info("  DB path     : %s", db_path)
    logger.info("  Out dir     : %s", out_dir)
    logger.info("  Params dir  : %s", params_dir)
    logger.info("  ---")
    logger.info("  Mode        : S=1 (single subject), C=auto (all CSVs)")
    logger.info("  A6          : BYPASSED (fixed rank, no CV)")
    logger.info("  B/C         : SKIPPED")
    logger.info("  Tangent     : SKIPPED")
    logger.info("  D           : ACTIVE (goal)")
    logger.info("  ---")
    logger.info("  CDHSA params:")
    logger.info("    L                : %d", cdhsa.get("L", 10))
    logger.info("    hankel_depth     : %s", cdhsa.get("hankel_depth", 10))
    logger.info("    fixed_rank       : %d", cdhsa.get("fixed_rank", 15))
    logger.info("    d_max_specific   : %d", cdhsa.get("d_max_specific", 10))
    logger.info("    residual_rank_m  : %s", cdhsa.get("residual_rank_method", "local_gap"))
    logger.info("    prevalence_quant : %.2f", cdhsa.get("prevalence_quantile", 0.10))
    logger.info("  ---")
    logger.info("  Preprocessing:")
    logger.info("    sfreq            : %.0f", pre.get("sfreq", 1000.0))
    logger.info("    apply_filter     : %s", pre.get("apply_filter", False))
    logger.info("    l_freq - h_freq   : %.1f - %.1f Hz",
                pre.get("l_freq", 1.0), pre.get("h_freq", 40.0))
    logger.info("    t_start          : %s", pre.get("t_start"))
    logger.info("    t_stop           : %s", pre.get("t_stop"))
    logger.info("  ---")
    logger.info("  Post-processing:")
    top_n = params.get("execution", {}).get("mode_extract_top_n", 4)
    logger.info("    mode_extract_top_n : %d", top_n)
    logger.info("=" * 70)


# ===================================================================
# Path resolution
# ===================================================================


def _resolve_db_path(params: dict, cli_db_path: str | None) -> Path:
    """Resolve the dataset path from CLI, JSON, or config."""
    if cli_db_path is not None:
        return Path(cli_db_path)

    ds_cfg = params.get("dataset", {})
    env_var = ds_cfg.get("db_path_env_var")
    if env_var and env_var in os.environ:
        return Path(os.environ[env_var])

    # Try config module
    try:
        from src.utils.config import DB_LUDOVICO_01_PATH
        return Path(DB_LUDOVICO_01_PATH)
    except (ImportError, AttributeError):
        pass

    # Try direct JSON path
    if "db_path" in ds_cfg:
        return Path(ds_cfg["db_path"])

    raise ValueError(
        "Cannot resolve dataset path. Provide --db-path or set "
        "DB_LUDOVICO_01_PATH in src.utils.config or as env var."
    )


def _resolve_out_dir(params: dict, cli_out_dir: str | None) -> Path:
    """Resolve the output directory."""
    if cli_out_dir is not None:
        return Path(cli_out_dir)

    try:
        from src.utils.config import BASE_RESULTS_PATH
        base = Path(BASE_RESULTS_PATH)
    except (ImportError, AttributeError):
        base = Path("./results")

    label = params.get("experiment_label", "ludovico_cdhsa")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / "ludovico_01" / "cdhsa" / f"{label}_{ts}"


def _resolve_params_dir(params: dict, cli_params_dir: str | None) -> Path:
    """Resolve the directory where mode_map.json will be saved.

    Mirrors the logic in the standard batch runner:
      1. CLI override  (--params-dir)
      2. BASE_PARAMS_FILE from src.utils.config
      3. Fallback: ./params
    """
    if cli_params_dir is not None:
        return Path(cli_params_dir)

    try:
        from src.utils.config import BASE_PARAMS_FILE
        return Path(BASE_PARAMS_FILE)
    except (ImportError, AttributeError):
        pass

    return Path("./params")


# ===================================================================
# Mode-map extraction  (replicates src.cdhsa.extract_mode_indices)
# ===================================================================


def _extract_mode_indices(
    results_dir: Path,
    top_n: int = 4,
    cdhsa_params: dict | None = None,
) -> dict[str, Any]:
    """Extract the top-N condition-specific mode indices from CDHSA results.

    This function replicates what ``src.cdhsa.extract_mode_indices`` does
    in the standard pipeline.  It reads the ``cdhsa_arrays.npz`` and
    ``condition_specific_modes.npz`` produced by ``run_cdhsa_ludovico.py``
    and produces a JSON-serializable mode map whose structure is
    **exactly** what ``CDHSASpecificModesDynamics`` (Stage 2) expects.

    Expected by Stage2 (fields actually read)::

        mode_map["conditions"]           # DICT {name: {info}}
        mode_map["conditions"][name]["mode_indices_in_W"]
        mode_map["conditions"][name]["top_n_actual"]
        mode_map["metadata"]["pipeline_config"]["L"]
        mode_map["metadata"]["source_dir"]
        mode_map["metadata"]["p_hankel"]
        mode_map["common_subspace"]

    Parameters
    ----------
    results_dir : Path
        Directory containing ``cdhsa_arrays.npz``,
        ``condition_specific_modes.npz``, ``load_info.json``,
        ``config.json``, and ``hankel_info.json``.
    top_n : int
        Number of top modes to report per condition.
    cdhsa_params : dict | None
        CDHSA parameters used (for embedding in metadata).

    Returns
    -------
    mode_map : dict
        JSON-serializable dictionary compatible with
        ``CDHSASpecificModesDynamics``.
    """
    cdhsa_params = cdhsa_params or {}

    # --- Load condition names from load_info.json ---
    condition_names: list[str] = []
    load_info_path = results_dir / "load_info.json"
    if load_info_path.exists():
        with open(load_info_path, "r") as f:
            load_info = json.load(f)
        condition_names = list(load_info.get("subjects", []))

    # --- Load D arrays from condition_specific_modes.npz ---
    csm_path = results_dir / "condition_specific_modes.npz"
    if not csm_path.exists():
        raise FileNotFoundError(
            f"condition_specific_modes.npz not found in {results_dir}"
        )

    csm = np.load(csm_path, allow_pickle=False)
    r_specific = csm["r_specific"]  # shape (C,)
    n_conditions = len(r_specific)

    if len(condition_names) == 0:
        condition_names = [f"condition_{c}" for c in range(n_conditions)]

    prevalence_contrast = csm["prevalence_contrast"]  # shape (C,)
    alignment_specific = csm["alignment_specific"]  # shape (1, C) for S=1
    if alignment_specific.ndim == 2:
        alignment_per_cond = alignment_specific[0, :]
    else:
        alignment_per_cond = alignment_specific

    # --- Resolve p_hankel from hankel_info.json ---
    p_hankel: int | None = None
    hankel_info_path = results_dir / "hankel_info.json"
    if hankel_info_path.exists():
        with open(hankel_info_path, "r") as f:
            hankel_info = json.load(f)
        # p_hankel = n_channels * hankel_depth (rows of the Stage-1 Hankel)
        p_hankel = hankel_info.get("p_hankel")
        if p_hankel is None:
            n_ch = hankel_info.get("n_channels", 0)
            depth = hankel_info.get("hankel_depth", 0)
            if n_ch and depth:
                p_hankel = n_ch * depth

    # --- Build conditions dict (MUST be a dict, NOT a list) ---
    conditions_dict: dict[str, dict] = {}
    for c in range(n_conditions):
        name = condition_names[c] if c < len(condition_names) else f"condition_{c}"
        r_c = int(r_specific[c])
        actual_top = min(top_n, r_c)

        # Mode indices: top-N by singular value (already sorted by SVD)
        mode_indices_in_W = list(range(actual_top))

        # Corresponding singular values
        lambda_key = f"lambda_specific_{name}"
        lambda_top: list[float | None] = [None] * actual_top
        if lambda_key in csm:
            lambdas = csm[lambda_key]
            lambda_top = [round(float(lambdas[j]), 8) for j in range(actual_top)]

        conditions_dict[name] = {
            "mode_indices_in_W": mode_indices_in_W,
            "top_n_actual": actual_top,
            "lambda_specific": lambda_top,
            "r_specific": int(r_c),
            "alignment_specific": round(float(alignment_per_cond[c]), 8),
            "prevalence_contrast": round(float(prevalence_contrast[c]), 8),
        }

    # --- Resolve L from cdhsa_params or config.json ---
    L = cdhsa_params.get("L")
    if L is None:
        config_path = results_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                cfg = json.load(f)
            L = cfg.get("L")
    if L is not None:
        L = int(L)

    # --- A6 info (bypassed for ludovico, but key must exist) ---
    common_subspace: dict = {
        "r0": None,
        "lambda0": None,
        "bypass_reason": "S=1 (ludovico_01): A6 requires S>=2 for CV",
    }

    # --- Assemble the mode_map in the exact structure Stage2 expects ---
    mode_map = {
        # Stage2 reads: mode_map["conditions"]  (must be DICT)
        "conditions": conditions_dict,
        # Stage2 reads: mode_map["metadata"]["pipeline_config"]["L"]
        # Stage2 reads: mode_map["metadata"]["source_dir"]
        # Stage2 reads: mode_map["metadata"]["p_hankel"]
        "metadata": {
            "pipeline_config": {
                "L": L,
                "hankel_depth": cdhsa_params.get("hankel_depth"),
                "fixed_rank": cdhsa_params.get("fixed_rank"),
                "d_max_specific": cdhsa_params.get("d_max_specific"),
            },
            "source_dir": str(results_dir),
            "p_hankel": p_hankel,
            "dataset": "ludovico_01",
            "n_conditions": n_conditions,
            "condition_names": condition_names,
            "steps_run": ["A1-A5", "A6_bypass", "D"],
            "steps_skipped": ["B/C", "tangent"],
        },
        # Stage2 reads: mode_map.get("common_subspace", {})
        "common_subspace": common_subspace,
        # Extra info (not read by Stage2, but useful for bookkeeping)
        "top_n_requested": top_n,
        "generated_at": datetime.now().isoformat(),
    }

    return mode_map


def _run_mode_extraction(
    params: dict,
    out_dir: Path,
    params_dir: Path,
) -> None:
    """Extract mode indices and save mode_map.json to params_dir.

    Mirrors the standard batch runner's ``_run_mode_extraction()``
    but works inline (no subprocess) because
    ``src.cdhsa.extract_mode_indices`` is not available.

    The output JSON is saved under ``params_dir`` with the same naming
    convention as the standard pipeline::

        mode_map_{top_n}_modes_{session}_{tasks}_{tw}.json

    For Ludovico there is no session or time-window concept, so those
    tags are replaced with ``ludovico_01`` and ``full`` respectively.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("  EXTRAYENDO INDICES DE MODOS ESPECIFICOS (mode_map.json)")
    logger.info("=" * 70)

    if not params_dir:
        logger.warning(
            "No se definio params_dir; no se puede guardar mode_map.json"
        )
        return

    params_dir.mkdir(parents=True, exist_ok=True)

    top_n = params.get("execution", {}).get("mode_extract_top_n", 4)

    # Verify the results directory has the required files
    npz_path = out_dir / "cdhsa_arrays.npz"
    csm_path = out_dir / "condition_specific_modes.npz"
    if not csm_path.exists():
        logger.warning(
            "SKIP: condition_specific_modes.npz not found in %s", out_dir
        )
        return

    # Build a descriptive output filename (matching standard convention)
    # Standard: mode_map_{top_n}_modes_{session}_{tasks}_{tw}.json
    # For Ludovico: session=ludovico_01, tasks=all, tw=full
    json_name = (
        f"mode_map_{top_n}_modes_ludovico_01_all_conditions_full.json"
    )
    json_out = params_dir / json_name

    logger.info("  EXTRACT | %s -> %s", out_dir, json_name)

    try:
        mode_map = _extract_mode_indices(
            out_dir,
            top_n=top_n,
            cdhsa_params=params.get("cdhsa_params", {}),
        )

        # Add experiment label (bookkeeping, not read by Stage2)
        mode_map["experiment_label"] = params.get(
            "experiment_label", "ludovico_cdhsa"
        )

        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(mode_map, f, indent=2, default=str)

        logger.info("  [OK] %s", json_out)
        logger.info("")
        logger.info(
            "  Extraccion de modos completada. "
            "JSON guardado en: %s", params_dir
        )

        # Also copy the batch params JSON used for this run into params_dir
        # (useful for reproducibility — same as the standard pipeline's pattern)
        batch_params_copy = params_dir / "cdhsa_batch_params_ludovico_used.json"
        with open(batch_params_copy, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, default=str, ensure_ascii=False)
        logger.info("  [OK] Batch params copy: %s", batch_params_copy)

    except Exception as exc:
        logger.error("  [FAIL] Error extrayendo modos: %s", exc)
        import traceback
        traceback.print_exc()


# ===================================================================
# Run
# ===================================================================


def run_batch(
    params: dict,
    db_path: Path,
    out_dir: Path,
    params_dir: Path,
) -> int:
    """Run the Ludovico CDHSA pipeline as specified in params.

    After the pipeline completes successfully, extracts mode indices
    and saves mode_map.json to params_dir (mirrors standard pipeline).
    """
    from src.pipelines.run_cdhsa_ludovico import LudovicoCDHSAConfig, run_ludovico_pipeline

    cdhsa = params["cdhsa_params"]
    pre = params.get("preprocessing", {})
    cond = params.get("conditions", {})

    config = LudovicoCDHSAConfig(
        L=cdhsa.get("L", 10),
        hankel_depth=cdhsa.get("hankel_depth", 10),
        fixed_rank=cdhsa.get("fixed_rank", 15),
        d_max_specific=cdhsa.get("d_max_specific", 10),
        residual_rank_method=cdhsa.get("residual_rank_method", "local_gap"),
        prevalence_quantile=cdhsa.get("prevalence_quantile", 0.10),
        sfreq=pre.get("sfreq", 1000.0),
        apply_filter=pre.get("apply_filter", False),
        l_freq=pre.get("l_freq", 1.0),
        h_freq=pre.get("h_freq", 40.0),
        t_start=pre.get("t_start"),
        t_stop=pre.get("t_stop"),
        subjects=cond.get("subjects"),
    )

    t0 = time.time()

    try:
        run_ludovico_pipeline(
            db_path=db_path,
            out_dir=out_dir,
            config=config,
        )
        elapsed = time.time() - t0
        logger.info("")
        logger.info("PIPELINE COMPLETED in %.1f s", elapsed)
        logger.info("Results saved in: %s", out_dir)

        # --- Post-processing: extract mode_map.json ---
        _run_mode_extraction(params, out_dir, params_dir)

        return 0
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error("")
        logger.error("BATCH FAILED after %.1f s: %s", elapsed, exc)
        import traceback
        traceback.print_exc()
        return 1


# ===================================================================
# Entry point
# ===================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch runner for CD-HSA on Ludovico_01. "
            "Reads config from JSON, runs pipeline, and extracts mode_map.json."
        ),
    )
    parser.add_argument(
        "--params-json", type=str, default=None,
        help=f"Path to JSON params. Default: {DEFAULT_PARAMS_JSON}",
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Override dataset path.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--params-dir", type=str, default=None,
        help=(
            "Override directory for mode_map.json output. "
            "Default: BASE_PARAMS_FILE from config, or ./params"
        ),
    )
    args = parser.parse_args()

    json_path = Path(args.params_json) if args.params_json else DEFAULT_PARAMS_JSON
    params = _load_params(json_path)
    logger.info("Params loaded from: %s", json_path)

    db_path = _resolve_db_path(params, args.db_path)
    out_dir = _resolve_out_dir(params, args.out_dir)
    params_dir = _resolve_params_dir(params, args.params_dir)

    _print_banner(params, db_path, out_dir, params_dir)

    return run_batch(params, db_path, out_dir, params_dir)


if __name__ == "__main__":
    sys.exit(main())
