#!/usr/bin/env python3
"""
latent_space_plots_from_eeg_gedai_super_subject.py
====================================================
Latent-space extraction + diagnostic plots for **super-subject** EEG
recordings from the test-retest Gedai dataset (EEGLAB ``.set``/``.fdt``),
**without the Kramers-Moyal / IgA potential part**.

This is a trimmed sibling of
``test_iga_from_eeg_latent_test_retest_gedai_super_subject.py`` (repo
``latent-space-and-potential-reconstruction``).  A super-subject is the
time-axis concatenation of *N* individual subjects' EEG recordings sharing
the same ``session`` and ``task``; the concatenated raw is processed as a
single recording by the 3-stage latent pipeline (Stage 0 load ->
Stage 1 embedding -> Stage 2 dynamics -> Stage 3 selection).

What is KEPT (everything up to, but not including, the KM coefficients)
-----------------------------------------------------------------------
* Stage 0: load of the concatenated raw + PSD of the original channels
  + exploratory plots (channel topographies, correlation matrix).
* Stage 1-3: latent-space extraction via ``extract_latent_space``
  (with cache), including the ``cdhsa_specific_modes`` Stage-2 strategy
  that projects the block-Hankel onto the condition-specific CD-HSA
  modes read from ``mode_map.json`` + ``cdhsa_arrays.npz``.
* ALL pre-KM plots:
  - Stage 1 (Hankel): singular values, variance explained.
  - Stage 2 (dynamics): PCA variance, ICLabel summary, pre/post-ICA PSDs,
    Hankel PCA singular values, FastICA convergence, DMD eigenvalues /
    frequency-damping, Diffusion-Maps spectra and kernel diagnostics,
    CD-HSA plots (eigenvalue spectrum, mode structure, projection power).
  - Stage 3 (Markov, when applicable): tau heatmaps, ranked taus,
    selection vs distribution, transition matrix.
  - PSD of the latent space and **channel influence on each latent
    dimension** (``channel_influence.png`` + ``channel_influence_data.npz``
    + spectral-contribution metrics) — the headline figure of this
    pipeline.
  - Latent trajectory, latent time series, zoom, outlier-cleaned series.
  - Chapman-Kolmogorov Markovianity test (with its plots).
  - Pipeline overview: energy budget + flowchart.

What is CUT (from the KM coefficients onwards)
----------------------------------------------
Bandwidth optimisation, KM coefficient estimation, empirical densities,
KM component plots, 1D/multidimensional IgA potential reconstruction,
potential 2D/slice/streamline/residual/non-conservative plots,
high-dimensional sub-job splitting and per-job compositing, and the
``methods_comparison`` post-processing matrix of the old batch runner.

Path / cache compatibility
--------------------------
The output directory, cache file and spec-hash logic are **byte-identical**
to the full pipeline, so caches and results produced by either pipeline
are interchangeable::

    Output: {out_dir}/test_retest_gedai_super_subject/super_subject-{id}/
             {session}/{latent_dim}_latent_dim_{spec_label}_{spec_hash}/
             from{t_start}s_to_{t_end}s_{task}
    Cache : {BASE_CACHE_PATH}/cache_eeg_test_retest_gedai_super_subject/
            super_subject-{id}/{session}/
            task_{task}_latent_dim_{latent_dim}_{spec_label}_{spec_hash}/
            from{t_start}s_to_{t_end}s.npz

Usage
-----
From the project root (normally dispatched by ``run_batch_cdhsa.py``,
which auto-injects ``mode_map_path``/``npz_path``)::

    python -m src.pipelines.latent_space_plots_from_eeg_gedai_super_subject \\
        --super-subject 1 --session session1 --task eyesclosed \\
        --t-start 100 --t-end 200 \\
        --stage1-embedding hankel --stage1-params '{"depth": 10}' \\
        --stage2-dynamics cdhsa_specific_modes \\
        --stage2-params '{"mode_map_path": "params/mode_map_...json",
                          "npz_path": "results/cdhsa/.../cdhsa_arrays.npz"}' \\
        --stage3-selection top_n --latent-dim 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Ensure package is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.latent_space_extraction.extract_latent_subspace import (
    extract_latent_space,
    map_legacy_scoring_method,
)

from src.latent_space_extraction.super_subject_eeg import load_super_subject_eeg

from src.latent_space_extraction.data_analysis_tools import outliers_cleaning

from src.latent_space_extraction.ck_test import chapman_kolmogorov_test

# --- Plotting module (src.plotters) ---
from src.plotters import (
    # Stage 0
    plot_channel_topographies,
    plot_channel_correlation_matrix,
    # Stage 1
    plot_hankel_singular_values,
    plot_hankel_variance_explained,
    # Stage 2
    plot_pca_variance_explained,
    plot_pre_ica_component_psds,
    plot_icalabel_summary,
    plot_post_ica_component_psds,
    plot_hankel_pca_singular_values,
    plot_fastica_convergence,
    plot_dmd_eigenvalue_unit_circle,
    plot_dmd_frequency_damping,
    plot_diffusion_eigenvalue_spectrum,
    plot_diffusion_kernel_diagnostics,
    plot_diffusion_2d_components,
    # Stage 3
    plot_markov_tau_heatmap,
    plot_markov_tau_ranked,
    plot_markov_selection_vs_distribution,
    plot_markov_transition_matrix,
    # Latent detail
    plot_latent_timeseries_zoom,
    # Pipeline overview
    plot_pipeline_energy_budget,
    plot_pipeline_flowchart,
    # Style setup
    setup_plotting_style,
)

from src.plotters.trajectory_plots import plot_latent_trajectory

# --- Power-spectral-density analysis module ---
from src.spectral_analysis.psd_analysis import (
    compute_and_plot_raw_psd,
    compute_and_plot_latent_psd,
    compute_channel_influence_on_latent,
)

from src.utils.config import (
    BASE_RESULTS_PATH,
    BASE_CACHE_PATH,
    DB_TEST_RETEST_GEDAI_PATH,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Latent-space extraction + diagnostic plots (NO KM / NO "
            "potential) for SUPER-SUBJECT test-retest EEG (concatenation "
            "of N subjects) preprocessed with Gedai (EEGLAB .set/.fdt)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- Super-subject source ----
    parser.add_argument(
        "--super-subject", type=int, required=True,
        help="1-indexed super-subject identifier (1, 2, 3, ...).",
    )
    parser.add_argument(
        "--subject-ids", type=str, default=None,
        help=(
            "JSON list of subject indices that compose the super-subject, "
            "e.g. '[1,2,...,20]'. When omitted, the pool is built from "
            "--subjects-per-super-subject and --subject-start-offset."
        ),
    )
    parser.add_argument(
        "--subjects-per-super-subject", type=int, default=20,
        help="Subjects per super-subject (auto-resolution). Default: 20.",
    )
    parser.add_argument(
        "--subject-start-offset", type=int, default=1,
        help=(
            "Index of the first subject in the dataset (typically 1 for "
            "sub-01).  Used by auto-resolution. Default: 1."
        ),
    )
    # ---- Session / task ----
    parser.add_argument(
        "--session", type=str, required=True,
        help="Session ID (e.g. session1)",
    )
    parser.add_argument(
        "--task", type=str, required=True,
        help="Task label (eyesclosed, eyesopen, mathematic, memory, music)",
    )
    parser.add_argument(
        "--channel-intersection", type=str, default=None,
        help=(
            "JSON list of channel names representing the global channel "
            "intersection across all super-subjects. When provided, the "
            "concatenated raw is restricted to exactly these channels before "
            "any processing. This ensures CD-HSA mode dimensionality "
            "consistency across super-subjects with different channel counts."
        ),
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Override Gedai dataset root path",
    )
    parser.add_argument(
        "--t-start", type=float, default=None,
        help="Per-subject start time (s) of the EEG segment to analyze.",
    )
    parser.add_argument(
        "--t-end", type=float, default=None,
        help="Per-subject end time (s) of the EEG segment to analyze.",
    )
    # ---- Cache / persistence ----
    parser.add_argument(
        "--cache-file", type=str, default=None,
        help="Path to cache file for the latent space.",
    )
    parser.add_argument(
        "--ignore-cache", action="store_true",
        help="Ignore an existing cache file and force recomputation.",
    )
    # ---- Latent-space extraction params ----
    parser.add_argument(
        "--latent-dim", type=int, default=2,
        help="Dimensionality of the latent subspace (default: 2)",
    )
    # ---- NEW 3-stage pipeline API ----
    parser.add_argument(
        "--stage1-embedding", type=str, default=None,
        choices=["none", "hankel"],
        help="Stage 1 embedding. If omitted, the legacy "
             "--scoring-method mapping is used.",
    )
    parser.add_argument(
        "--stage1-params", type=str, default=None,
        help='JSON dict of Stage-1 params, e.g. \'{"depth": 250}\'',
    )
    parser.add_argument(
        "--stage2-dynamics", type=str, default=None,
        choices=["pca_ica", "pca", "dmd", "diffusion_maps", "cdhsa_specific_modes"],
        help="Stage 2 dynamics (new API).",
    )
    parser.add_argument(
        "--stage2-params", type=str, default=None,
        help=(
            'JSON dict of Stage-2 params. For cdhsa_specific_modes: '
            '{"mode_map_path": "...", "npz_path": "..."} '
            '(condition is auto-injected from --task).'
        ),
    )
    parser.add_argument(
        "--stage3-selection", type=str, default=None,
        choices=["top_n", "markov_fastest", "markov_slowest"],
        help="Stage 3 selection (new API).",
    )
    parser.add_argument(
        "--stage3-params", type=str, default=None,
        help='JSON dict of Stage-3 params, e.g. \'{"n_bins": 10}\'',
    )
    # ---- Legacy scoring API (mapped onto the new stages) ----
    parser.add_argument(
        "--scoring-method", type=str, default="hankel_dmd",
        choices=["markov", "markov_inverted", "conservative", "weighted",
                 "sequential", "pareto", "independent",
                 "hankel_dmd", "diffusion_maps"],
        help="Legacy subspace selection strategy (default: hankel_dmd). "
             "Mapped internally onto the new 3-stage API. Ignored if "
             "any --stageX argument is given.",
    )
    parser.add_argument(
        "--fc-metric", type=str, default="variance_sum",
        choices=["variance_sum", "first_pc_var", "total_variance"],
    )
    parser.add_argument(
        "--n-bins", type=int, default=10,
        help="Quantile bins for Markov discretisation (default: 10)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel processes for subspace search",
    )
    parser.add_argument(
        "--search-strategy", type=str, default="exhaustive",
        choices=["exhaustive", "greedy"],
    )
    # ---- Which latent dimension(s) to keep for diagnostics ----
    parser.add_argument(
        "--analysis-dim", type=int, default=None,
        help="Number of latent dimensions used for the CK test and the "
             "latent-time-series plots. If None, uses all extracted "
             "dimensions. (In the full pipeline this fed the KM analysis; "
             "here it only scopes the pre-KM diagnostic plots.)",
    )
    parser.add_argument(
        "--column", type=int, default=None,
        help="If analysis-dim=1, which latent column to use (0=first).",
    )
    # ---- Preprocessing params ----
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--ica-method", type=str, default="picard")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument(
        "--no-verbose", action="store_false", dest="verbose",
        help="Silence the verbose INFO output of the loader/extraction.",
    )
    # ---- Hankel (legacy convenience; also usable as stage1 param) ----
    parser.add_argument(
        "--hankel-embedding-depth", type=int, default=None,
        help="Hankel embedding depth T. None = auto "
             "(clip(sfreq*0.25, 50, 200)).",
    )
    # ---- Diffusion Maps params ----
    parser.add_argument(
        "--diffusion-sigma", type=float, default=None,
        help=("Sigma for Diffusion Maps Gaussian kernel. "
              "If None, auto-computed via bgh method (Berry-Giannakis-Harlim)."),
    )
    parser.add_argument(
        "--diffusion-k", type=int, default=100,
        help="Number of nearest neighbors for sparse affinity matrix.",
    )
    parser.add_argument(
        "--diffusion-time", type=float, default=0.0,
        help=("Diffusion time t >= 0. Higher values filter fine-scale noise "
              "and highlight macroscopic dynamics (multiscale filtering)."),
    )
    parser.add_argument(
        "--diffusion-alpha", type=float, default=0.5,
        help=("Density normalization parameter (Coifman-Lafon). "
              "0.0=Laplacian Eigenmaps, 0.5=Diffusion Maps (default), 1.0=Fokker-Planck."),
    )
    # ---- Output ----
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Directory to save plots (default: BASE_RESULTS_PATH)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline-spec resolution: new stage API or legacy scoring-method mapping
# ---------------------------------------------------------------------------

def _json_params(raw: str | None) -> dict:
    """Parse a JSON params dict from the CLI (empty dict when omitted)."""
    if raw is None:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Stage params must be a JSON object, got: {raw!r}")
    return parsed


def _json_int_list(raw: str | None) -> list[int] | None:
    """Parse a JSON list of integers from the CLI (None when omitted)."""
    if raw is None:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed):
        raise ValueError(f"subject-ids must be a JSON list of ints, got: {raw!r}")
    return parsed


def _resolve_pipeline_spec(args: argparse.Namespace) -> dict:
    """
    Resolve the effective 3-stage specification.

    * If any ``--stageX`` argument is given -> new API (missing stages take
      their defaults; CLI convenience flags --diffusion-* /
      --hankel-embedding-depth / --n-bins are injected into the JSON
      params when not already present).
    * Otherwise -> legacy ``--scoring-method`` mapping via
      :func:`map_legacy_scoring_method`.

    Returns a dict with keys ``stage1_embedding``, ``stage1_params``,
    ``stage2_dynamics``, ``stage2_params``, ``stage3_selection``,
    ``stage3_params``.
    """
    use_new_api = any([
        args.stage1_embedding is not None,
        args.stage2_dynamics is not None,
        args.stage3_selection is not None,
    ])

    if use_new_api:
        s1 = None if args.stage1_embedding in (None, "none") else args.stage1_embedding
        s2 = args.stage2_dynamics or "pca_ica"
        s3 = args.stage3_selection or "top_n"

        p1 = _json_params(args.stage1_params)
        p2 = _json_params(args.stage2_params)
        p3 = _json_params(args.stage3_params)

        # Inject CLI convenience flags when not overridden in the JSON
        if s1 == "hankel":
            p1.setdefault("depth", args.hankel_embedding_depth)
        if s2 == "diffusion_maps":
            p2.setdefault("sigma", args.diffusion_sigma)
            p2.setdefault("k", args.diffusion_k)
            p2.setdefault("diffusion_time", args.diffusion_time)
            p2.setdefault("alpha", args.diffusion_alpha)
        if s2 == "pca_ica":
            p2.setdefault("ica_method", args.ica_method)
        if s2 == "cdhsa_specific_modes":
            # Auto-inject --task as condition so the mode_map resolves
            # the correct condition index without the user specifying it.
            p2.setdefault("condition", args.task)
        if s3 in ("markov_fastest", "markov_slowest"):
            p3.setdefault("n_bins", args.n_bins)
            p3.setdefault("search_strategy", args.search_strategy)

        return {
            "stage1_embedding": s1,
            "stage1_params": p1,
            "stage2_dynamics": s2,
            "stage2_params": p2,
            "stage3_selection": s3,
            "stage3_params": p3,
        }

    # Legacy mapping
    return map_legacy_scoring_method(
        args.scoring_method,
        n_dim=args.latent_dim,
        fc_metric=args.fc_metric,
        n_bins=args.n_bins,
        hankel_embedding_depth=args.hankel_embedding_depth,
        diffusion_sigma=args.diffusion_sigma,
        diffusion_k=args.diffusion_k,
        diffusion_time=args.diffusion_time,
        diffusion_alpha=args.diffusion_alpha,
        ica_method=args.ica_method,
        search_strategy=args.search_strategy,
    )


def _spec_label(spec: dict) -> str:
    """Short human-readable label for the stage chain (used in paths)."""
    s1 = spec["stage1_embedding"] or "none"
    return f"{s1}+{spec['stage2_dynamics']}+{spec['stage3_selection']}"


def _spec_hash(spec: dict) -> str:
    """
    Short hash of the full stage spec (stages + params).

    Included in the cache key so that different parameter combinations
    never collide (requirement: the cache key must contain the hashes of
    the 3-stage parameters).  Byte-identical to the full IgA pipeline so
    caches are shared between both pipelines.
    """
    payload = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Super-subject label helpers
# ---------------------------------------------------------------------------

def _super_subject_label(super_subject_id: int) -> str:
    """BIDS-style label for the super-subject, e.g. ``super_subject-01``."""
    return f"super_subject-{super_subject_id:02d}"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    if args.out_dir is None:
        args.out_dir = BASE_RESULTS_PATH
    verbose = "INFO" if args.verbose else None

    # Defaults
    if args.t_start is None:
        args.t_start = 0.0
    if args.t_end is None:
        args.t_end = 60.0

    db_path = args.db_path or DB_TEST_RETEST_GEDAI_PATH

    # Resolve the effective 3-stage spec (new API or legacy mapping)
    spec = _resolve_pipeline_spec(args)
    spec_label = _spec_label(spec)
    spec_hash = _spec_hash(spec)
    print(f"\n  Pipeline spec : {spec_label}  (hash {spec_hash})")
    print(f"  Stage params  : {json.dumps(spec, default=str)}")

    # Resolve subject pool for the super-subject
    subject_ids = _json_int_list(args.subject_ids)
    ss_label = _super_subject_label(args.super_subject)
    print(f"  Super-subject : {ss_label}")
    if subject_ids is not None:
        print(f"  Subject pool  : explicit list ({len(subject_ids)} subjects)")
    else:
        print(
            f"  Subject pool  : auto-resolved "
            f"({args.subjects_per_super_subject} subjects starting at "
            f"{args.subject_start_offset})"
        )

    # -----------------------------------------------------------------
    # Output directory:
    # test_retest_gedai_super_subject/super_subject-{id}/{session}/
    #   {latent_dim}_latent_dim_{spec_label}_{spec_hash}/
    #   from{t_start}s_to_{t_end}s_{task}
    # -----------------------------------------------------------------
    out_dir = Path(
        str(args.out_dir)
        + f"/test_retest_gedai_super_subject/{ss_label}/{args.session}"
        + f"/{args.latent_dim}_latent_dim_{spec_label}_{spec_hash}"
        + f"/from{args.t_start}s_to{args.t_end}s_{args.task}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()

    # =====================================================================
    # 1. LOAD SUPER-SUBJECT EEG SEGMENT (concatenated raw)
    # =====================================================================
    print("=" * 70)
    print("  STAGE 0: LOAD SUPER-SUBJECT EEG SEGMENT (GEDAI, concatenated)")
    print("=" * 70)

    try:
        print(f"  Loading super-subject EEG ({len(subject_ids) if subject_ids else args.subjects_per_super_subject} subjects)...")
        sys.stdout.flush()
        raw = load_super_subject_eeg(
            super_subject_id=args.super_subject,
            session=args.session,
            task=args.task,
            subject_ids=subject_ids,
            subjects_per_super_subject=args.subjects_per_super_subject,
            subject_start_offset=args.subject_start_offset,
            db_path=db_path,
            t_start=args.t_start,
            t_stop=args.t_end,
            preload=False,
            verbose=verbose,
        )
        sys.stdout.flush()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    sfreq = raw.info["sfreq"]
    print(f"  Super-subject ID   : {args.super_subject} ({ss_label})")
    print(f"  Channels (common)  : {len(raw.ch_names)}")
    print(f"  Channel names      : {raw.ch_names}")
    print(f"  Sampling freq      : {sfreq:.2f} Hz")
    print(f"  Cropped window     : {raw.times[0]:.2f} s -> {raw.times[-1]:.2f} s "
          f"(duration: {raw.times[-1] - raw.times[0]:.2f} s)")

    # -----------------------------------------------------------------
    # Apply global channel intersection (CD-HSA consistency)
    # -----------------------------------------------------------------
    channel_intersection = None
    if args.channel_intersection is not None:
        channel_intersection = json.loads(args.channel_intersection)
        if not isinstance(channel_intersection, list):
            raise ValueError("--channel-intersection must be a JSON list of channel names")
        # Validate that all requested channels exist in the raw
        missing = [ch for ch in channel_intersection if ch not in raw.ch_names]
        if missing:
            raise ValueError(
                f"--channel-intersection references {len(missing)} channels "
                f"not found in raw: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        n_before = len(raw.ch_names)
        raw.pick(channel_intersection)
        print(f"  [ChannelIntersection] Applied global intersection: "
              f"{n_before} -> {len(raw.ch_names)} channels")
        sys.stdout.flush()

    # -----------------------------------------------------------------
    # PSD of the original (concatenated) EEG channels
    # -----------------------------------------------------------------
    psds_raw, freqs_raw, ch_names, mean_psd, std_psd, raw_psd_path = compute_and_plot_raw_psd(
        raw, out_dir=out_dir, fmin=args.l_freq, fmax=args.h_freq, bandwidth=2.5,
    )

    # --- STAGE 0: EXPLORATORY PLOTS ---
    print("  Generando plots exploratorios (Stage 0)...")
    try:
        setup_plotting_style()
        plot_channel_topographies(
            raw, out_dir=out_dir,
            fmin=args.l_freq, fmax=args.h_freq,
            subject=ss_label, session=args.session, task=args.task,
        )
        plot_channel_correlation_matrix(
            raw, out_dir=out_dir,
            subject=ss_label, session=args.session, task=args.task,
        )
        print("  [OK] Plots exploratorios guardados.")
    except Exception as e:
        print(f"  [WARN] Error en plots exploratorios: {e}")

    # =====================================================================
    # 2. EXTRACT (or LOAD CACHED) LATENT SUBSPACE FROM CONCATENATED EEG
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STAGE 1: EXTRACT LATENT SUBSPACE FROM SUPER-SUBJECT EEG")
    print("=" * 70)

    if args.cache_file is None:
        args.cache_file = Path(
            str(BASE_CACHE_PATH)
            + f"/cache_eeg_test_retest_gedai_super_subject/{ss_label}/{args.session}"
            + f"/task_{args.task}_latent_dim_{args.latent_dim}_{spec_label}_{spec_hash}"
            + f"/from{args.t_start}s_to{args.t_end}s.npz"
        )
    cache_path = Path(args.cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to load from cache
    if cache_path.exists() and not args.ignore_cache:
        print(f"\n  [CACHE] Found existing cache: {cache_path}")
        print("  [CACHE] Loading latent space and metadata...")
        loaded = np.load(cache_path, allow_pickle=True)
        latent = loaded["latent"]
        meta = loaded["meta"].item()
        print("  [CACHE] Loaded successfully.")
        if "preprocessing" not in meta or "elapsed_time" not in meta:
            print("  [WARN] Cache file seems corrupted or outdated. Recomputing...")
            args.ignore_cache = True

    # Compute if no cache or --ignore-cache
    if not cache_path.exists() or args.ignore_cache:
        if args.ignore_cache and cache_path.exists():
            print("\n  [CACHE] --ignore-cache set. Recomputing latent space...")

        print("\n  [INFO] Starting latent space extraction (this may take several minutes)...")
        sys.stdout.flush()

        latent, meta = extract_latent_space(
            raw,
            n_dim=args.latent_dim,
            stage1_embedding=spec["stage1_embedding"],
            stage1_params=spec["stage1_params"],
            stage2_dynamics=spec["stage2_dynamics"],
            stage2_params=spec["stage2_params"],
            stage3_selection=spec["stage3_selection"],
            stage3_params=spec["stage3_params"],
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            n_workers=args.workers,
            verbose=verbose,
            channel_intersection=channel_intersection,
        )
        sys.stdout.flush()

        print(f"\n  [CACHE] Saving latent space to: {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, latent=latent, meta=np.array(meta, dtype=object))
        print("  [CACHE] Saved successfully.")

    # latent shape: (n_samples, latent_dim)
    n_samples, latent_dim = latent.shape
    dt = 1.0 / sfreq
    print(f"\n  dt = {dt:.6f} s")
    print(f"  Latent space shape : {latent.shape}")
    print(f"  Selected ICs       : {meta['selected_indices']}")
    print(f"  Scores             : {meta['latent_scores']}")
    print(f"  Extraction time    : {meta['elapsed_time']:.1f} s")

    # -----------------------------------------------------------------
    # STAGE 1: HANKEL EMBEDDING PLOTS
    # -----------------------------------------------------------------
    if spec["stage1_embedding"] == "hankel":
        print("  Generando plots Stage 1 (Hankel)...")
        try:
            from src.latent_space_extraction.hankel_dmd_extractor import (
                _build_multivariate_hankel,
            )
            depth = meta["stage1"].get("depth")
            X_input = meta["stage1"].get("input_data")
            if X_input is not None and depth is not None:
                H_plot = _build_multivariate_hankel(X_input, depth)
                plot_hankel_singular_values(
                    H_plot, out_dir=out_dir, embedding_depth=depth,
                )
                plot_hankel_variance_explained(
                    H_plot, out_dir=out_dir, embedding_depth=depth,
                )
                print("  [OK] Plots Stage 1 guardados.")
        except Exception as e:
            print(f"  [WARN] Error en plots Stage 1: {e}")

    # -----------------------------------------------------------------
    # STAGE 2: DYNAMICS PLOTS
    # -----------------------------------------------------------------
    print("  Generando plots de dinamica (Stage 2)...")
    try:
        stage2_meta = meta["stage2"]
        # PCA variance (para pca_ica rama MNE)
        plot_pca_variance_explained(stage2_meta, out_dir=out_dir)
        # ICLabel summary
        plot_icalabel_summary(stage2_meta, out_dir=out_dir)
        # Pre/post ICA PSDs (solo para rama MNE)
        plot_pre_ica_component_psds(
            raw, stage2_meta, out_dir=out_dir,
            fmin=args.l_freq, fmax=args.h_freq,
        )
        plot_post_ica_component_psds(
            stage2_meta, sfreq=sfreq, out_dir=out_dir,
            fmin=args.l_freq, fmax=args.h_freq,
        )
        # Hankel PCA singular values (solo para pca_ica rama Hankel)
        plot_hankel_pca_singular_values(stage2_meta, out_dir=out_dir)
        # FastICA convergence
        plot_fastica_convergence(stage2_meta, out_dir=out_dir)
        # DMD
        plot_dmd_eigenvalue_unit_circle(stage2_meta, out_dir=out_dir)
        plot_dmd_frequency_damping(stage2_meta, out_dir=out_dir)
        # Diffusion Maps
        plot_diffusion_eigenvalue_spectrum(stage2_meta, out_dir=out_dir)
        # Kernel diagnostics (puede ser costoso; best-effort)
        try:
            dm_input = stage2_meta.get("dm_input_data")
            if dm_input is None and spec["stage1_embedding"] == "hankel":
                dm_input = meta["stage1"].get("input_data")
            plot_diffusion_kernel_diagnostics(stage2_meta, dm_input, out_dir=out_dir)
        except Exception as e:
            print(f"  [WARN] Kernel diagnostics omitido: {e}")
        plot_diffusion_2d_components(meta, out_dir=out_dir)
        # CD-HSA Specific Modes
        if stage2_meta.get("dynamics") == "cdhsa_specific_modes":
            try:
                from src.plotters.cdhsa_plots import (
                    plot_cdhsa_eigenvalue_spectrum,
                    plot_cdhsa_mode_structure,
                    plot_cdhsa_projection_power,
                )
                plot_cdhsa_eigenvalue_spectrum(stage2_meta, out_dir=out_dir)
                plot_cdhsa_mode_structure(stage2_meta, out_dir=out_dir)
                plot_cdhsa_projection_power(stage2_meta, Y2=meta["Y"], out_dir=out_dir)
            except Exception as e:
                print(f"  [WARN] CD-HSA plots omitidos: {e}")
        print("  [OK] Plots Stage 2 guardados.")
    except Exception as e:
        print(f"  [WARN] Error en plots Stage 2: {e}")

    # -----------------------------------------------------------------
    # STAGE 3: MARKOV SELECTION PLOTS
    # -----------------------------------------------------------------
    if "all_markov_taus" in meta.get("stage3", {}):
        all_taus = meta["stage3"]["all_markov_taus"]
        selected = tuple(meta["selected_indices"])
        try:
            plot_markov_tau_heatmap(all_taus, out_dir=out_dir,
                                    selected_combination=selected)
            plot_markov_tau_ranked(all_taus, out_dir=out_dir,
                                   selected_combination=selected)
            tau_sel = meta["stage3"]["scores"].get("tau")
            if tau_sel is not None and selected:
                maximize = meta["stage3"]["scores"].get("maximize", False)
                plot_markov_selection_vs_distribution(
                    all_taus, tau_sel, selected,
                    maximize=maximize, out_dir=out_dir,
                )
            print("  [OK] Plots Stage 3 (Markov) guardados.")
        except Exception as e:
            print(f"  [WARN] Error en plots Stage 3: {e}")

    # Transition matrix of the selected combination (always available when markov was used)
    if "markov" in spec["stage3_selection"]:
        try:
            Y2 = meta["Y"]   # (D, T)
            n_bins_s3 = meta["stage3"]["scores"].get("n_bins", 10)
            selected = tuple(meta["selected_indices"])
            plot_markov_transition_matrix(Y2, selected, n_bins_s3, out_dir=out_dir)
            print("  [OK] Matriz de transicion guardada.")
        except Exception as e:
            print(f"  [WARN] Error en plot de matriz de transicion: {e}")

    # -----------------------------------------------------------------
    # PSD of the latent space + channel influence per latent dimension
    # -----------------------------------------------------------------
    psds_latent, freqs_latent, mean_latent, std_latent, latent_psd_path = compute_and_plot_latent_psd(
        latent, sfreq=sfreq, out_dir=out_dir, fmin=args.l_freq, fmax=args.h_freq,
    )
    influence_weights, ch_names, fig_path, data_path = compute_channel_influence_on_latent(
        raw, latent, meta, out_dir=out_dir, raw_psd_path=raw_psd_path,
    )

    # Plot latent trajectory
    plot_latent_trajectory(
        latent,
        out_dir=out_dir,
        method_name=spec_label,
    )

    # =====================================================================
    # 3. SELECT DIMENSION(S) FOR THE PRE-KM DIAGNOSTICS
    # =====================================================================
    analysis_dim = args.analysis_dim if args.analysis_dim is not None else latent_dim

    if analysis_dim > latent_dim:
        raise ValueError(
            f"analysis-dim ({analysis_dim}) cannot exceed latent-dim ({latent_dim})"
        )

    if analysis_dim == 1:
        col = args.column if args.column is not None else 0
        if col >= latent_dim:
            raise ValueError(f"column ({col}) must be < latent-dim ({latent_dim})")
        data = latent[:, col:col + 1]
        print(f"\n  Using latent column {col} for the 1D diagnostics")
    else:
        data = latent[:, :analysis_dim]
        print(f"\n  Using first {analysis_dim} latent columns for diagnostics")

    # --- Chapman-Kolmogorov test on the FULL data (global, per user choice) ---
    ck_result = chapman_kolmogorov_test(
        data,
        dt=dt,
        n_bins=20,
        threshold=0.15,
        plot=True,
        out_dir=out_dir,
        verbose=True,
    )

    if not ck_result["is_markovian"]:
        print("\n  [WARN] Latent space is NOT Markovian. KM results may be invalid.")
        print("  Consider: increasing latent_dim, increasing embedding_depth,")
        print("  or switching to a different stage combination.")
    else:
        print(f"\n  [OK] Markovian at tau* = {ck_result['tau_star']:.4f} s "
              f"({ck_result['tau_star_idx']} steps)")

    D = analysis_dim
    print(f"  Data shape for diagnostics : {data.shape}")

    # Plot latent time series (all D dims in a single figure, unchanged)
    fig_ts, axes = plt.subplots(D, 1, figsize=(14, 2.5 * D), squeeze=False)
    for d in range(D):
        ax = axes[d, 0]
        ax.plot(data[:, d], lw=0.5)
        ax.set_title(f"Latent dimension {d}")
        ax.set_xlabel("sample")
        ax.set_ylabel("amplitude")
    plt.tight_layout()
    fig_ts.savefig(out_dir / "latent_timeseries.png", dpi=150)
    plt.close(fig_ts)
    print(f"  Saved latent_timeseries.png")

    # --- LATENT DETAIL: ZOOM ---
    try:
        plot_latent_timeseries_zoom(latent, sfreq=sfreq, out_dir=out_dir)
        print("  Saved latent_timeseries_zoom.png")
    except Exception as e:
        print(f"  [WARN] Error en plot de zoom latente: {e}")

    data = outliers_cleaning(data, method="iqr", threshold=5)

    # Plot cleaned
    fig_ts, axes = plt.subplots(D, 1, figsize=(14, 2.5 * D), squeeze=False)
    for d in range(D):
        ax = axes[d, 0]
        ax.plot(data[:, d], lw=0.5)
        ax.set_title(f"Latent dimension {d} (after outlier cleaning)")
        ax.set_xlabel("sample")
        ax.set_ylabel("amplitude")
    plt.tight_layout()
    fig_ts.savefig(out_dir / "latent_timeseries_cleaned.png", dpi=150)
    plt.close(fig_ts)
    print(f"  Saved latent_timeseries_cleaned.png")

    # =====================================================================
    # 4. SUMMARY (pipeline overview plots; no KM / potential stage)
    # =====================================================================
    try:
        plot_pipeline_energy_budget(
            X_filtered=meta.get("preprocessing", {}).get("X_filtered"),
            meta=meta, latent=latent, out_dir=out_dir,
        )
        print("  Saved pipeline_energy_budget.png")
        plot_pipeline_flowchart(
            meta=meta, out_dir=out_dir,
            l_freq=args.l_freq, h_freq=args.h_freq,
            n_channels=len(raw.ch_names), sfreq=sfreq,
        )
        print("  Saved pipeline_flowchart.png")
    except Exception as e:
        print(f"  [WARN] Error en plots overview: {e}")

    total_time = time.time() - overall_t0
    print("\n" + "=" * 70)
    print("  SUPER-SUBJECT LATENT+PLOTS PIPELINE COMPLETED SUCCESSFULLY")
    print("  (Kramers-Moyal / potential reconstruction: SKIPPED by design)")
    print("=" * 70)
    print(f"  Super-subject       : {args.super_subject} ({ss_label})")
    print(f"  Session             : {args.session}")
    print(f"  Task                : {args.task}")
    print(f"  Pipeline            : {spec_label} (hash {spec_hash})")
    print(f"  Per-subject window  : {args.t_start:.1f}s -> {args.t_end:.1f}s")
    print(f"  Concatenated length : {raw.times[-1] - raw.times[0]:.1f} s")
    print(f"  Latent dim          : {latent_dim}")
    print(f"  Total wall-clock    : {total_time:.1f} s")
    print(f"  Output saved to     : {out_dir.absolute()}")

    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())
