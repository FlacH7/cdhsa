#!/usr/bin/env python3
"""
run_batch_cdhsa.py
====================
Batch executor for the **CD-HSA** pipeline.

Supports two execution modes controlled by the JSON configuration:

**Multi-SS mode** (recommended, new)::

  A single invocation of ``run_cdhsa.py`` with all super-subjects
  together (S=N, C=tasks).  CD-HSA finds common directions across
  subjects AND condition-specific modes with full cross-subject
  statistical support (permutation tests, prevalence, A6 rank).

  Triggered when ``super_subjects.n_super_subjects`` is present.

**Per-SS mode** (legacy)::

  Iterates over super-subjects and dispatches each as a subprocess
  call to ``run_cdhsa.py`` with ``--super-subject-id`` (S=1).

  Triggered when ``super_subjects.selected`` is present.

Configuration
--------------
Everything is controlled by the JSON file (default:
``./cdhsa_batch_params.json``).

Multi-SS JSON example::

    {
      "experiment_label": "multiss_cdhsa_5ss_5tasks",
      "super_subjects": {
          "n_super_subjects": 5,
          "subjects_per_super_subject": 12,
          "subject_start_offset": 1,
          "total_subjects": 60
      },
      "sessions": ["session1"],
      "tasks": ["eyesclosed", "eyesopen", "music", "memory", "mathematic"],
      "time_window": { "t_start": 0.0, "t_end": 300.0 },
      "cdhsa_params": {
          "L": 27, "hankel_depth": 10,
          "l_freq": 1.0, "h_freq": 40.0,
          "fixed_rank": 25, "rank_method": "fixed",
          "a6_n_null": 500, "bc_n_perm": 5000,
          "skip_bc": false, "skip_tangent": false, "skip_d": false
      },
      "execution": {
          "max_workers": 1, "delay": 0.0,
          "run_comparison": false
      }
    }

Usage
-----
From the directory containing this script::

    # Defaults (reads cdhsa_batch_params.json in the same directory)
    python run_batch_cdhsa.py

    # Custom JSON
    python run_batch_cdhsa.py --params-json /path/to/params.json

    # Override via environment variable
    BATCH_CDHSA_PARAMS_JSON=/path/to/params.json python run_batch_cdhsa.py

Latent phase (latent_plots)
---------------------------
After the CD-HSA jobs, the mode-index extraction and the (optional)
cross-subject comparison, the batch can run a **latent-space + plots
phase**: for every (super-subject x task x method) it dispatches a
subprocess call to the TRIMMED pipeline
``src.pipelines.latent_space_plots_from_eeg_gedai_super_subject`` (a
sibling of ``test_iga_from_eeg_latent_test_retest_gedai_super_subject``
WITHOUT the Kramers-Moyal / potential part; see its module docstring).

Key points:

* Controlled by the optional ``latent_plots`` block of the JSON
  (``"enabled": true`` activates it).  Jobs are generated for every
  super-subject of the CD-HSA config x session x task x method.
* ``mode_map_path`` and ``npz_path`` are **auto-injected** with the
  absolute paths produced by THIS batch run (params_dir /
  ``mode_map_{top_n}_modes_{session}_{tasks}_{tw}.json`` and the job's
  ``cdhsa_arrays.npz``), so no ``${VAR}`` placeholders or cross-repo
  ``.env`` entries are needed.
* When any method uses ``cdhsa_specific_modes`` the batch pre-computes
  the **global channel intersection** per (session, task) (headers only,
  same helper the latent repo uses) and passes it via
  ``--channel-intersection`` so the Hankel row dimensionality matches
  the CD-HSA ``W_specific``.
* ``stage1_params.depth`` of each method defaults to
  ``cdhsa_params.hankel_depth`` when omitted in the JSON.
* Own checkpoint file (``batch_checkpoint_cdhsa_latent.json``), own CSV
  log, per-job ntfy failure notices and per-job peak-RSS tracking,
  mirroring the CD-HSA phase.
* Requires **multi-SS mode** (``n_super_subjects``); in per-SS legacy
  mode the phase is skipped with a warning (the per-SS mode_maps share
  one filename and cannot be disambiguated per super-subject).

Notifications
------------
If ``NTFY_CHANNEL`` is defined in the ``.env`` (see ``src/ntfy/README.md``),
the batch sends push notifications through https://ntfy.sh:

* an **info** when the batch starts (doubles as a canary: if it does not
  reach the phone, the channel is misconfigured),
* an **urgent error** for every job that dies (the batch keeps going),
* a final **success** (``Fallos == 0``) or **warning** with the summary,
* an **urgent error with traceback** if an uncaught exception (or a
  ``sys.exit`` with non-zero code, e.g. a broken params JSON) kills the
  whole batch.

Without ``NTFY_CHANNEL`` all of this is a silent no-op and the batch
behaves exactly as before.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import threading

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    from src.utils.memory_tracker import MemoryMonitor, get_global_monitor
    _HAS_MEM_TRACKER = True
except ImportError:
    _HAS_MEM_TRACKER = False

# ---------------------------------------------------------------------------
# [NTFY] Push notifications for long runs (optional; see src/ntfy/README.md)
# ---------------------------------------------------------------------------
# The channel is imported from src.utils.config exactly like the rest of
# the environment variables, but in a separate try-block so that:
#   * an outdated config.py without NTFY_CHANNEL only disables notifications,
#   * main()'s decorator knows the channel even if the params JSON fails
#     before the runner is created.
try:
    from src.utils.config import NTFY_CHANNEL as _NTFY_CHANNEL
except Exception:  # ImportError if src.* is not in the path
    _NTFY_CHANNEL = None

try:
    from src.ntfy import (
        notify_error,
        notify_info,
        notify_success,
        notify_warning,
        notify_on_critical_error,
    )
    _HAS_NTFY = True
except ImportError:
    # Without src/ntfy the batch works exactly the same (notifications off).
    _HAS_NTFY = False

    def notify_on_critical_error(_func=None, **_kwargs):
        """No-op shim: src.ntfy not available (notifications disabled)."""
        def _deco(func):
            return func
        return _deco(_func) if _func is not None else _deco

# ---------------------------------------------------------------------------
# Ensure the directory containing run_cdhsa.py is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_cdhsa")

# [NTFY] Host included in the push notifications
_HOST = socket.gethostname()

# ---------------------------------------------------------------------------
# Default JSON path
# ---------------------------------------------------------------------------
DEFAULT_PARAMS_JSON = _SCRIPT_DIR / "cdhsa_batch_params.json"

# ---------------------------------------------------------------------------
# JSON loading + validation
# ---------------------------------------------------------------------------


def _load_params(json_path: Path) -> dict:
    """Load and validate the batch JSON parameters."""
    if not json_path.exists():
        logger.error("Archivo JSON de parametros no encontrado: %s", json_path)
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as fh:
        params = json.load(fh)

    # Minimal validation
    required_top_keys = ["super_subjects", "sessions", "tasks", "cdhsa_params"]
    for key in required_top_keys:
        if key not in params:
            logger.error("Falta la clave requerida '%s' en el JSON", key)
            sys.exit(1)

    ss_cfg = params["super_subjects"]

    # Detect mode: multi-SS (n_super_subjects) vs per-SS (selected)
    is_multi_ss = "n_super_subjects" in ss_cfg
    is_per_ss = "selected" in ss_cfg

    if not is_multi_ss and not is_per_ss:
        logger.error(
            "El bloque 'super_subjects' debe contener 'n_super_subjects' (modo multi-SS) "
            "o 'selected' (modo per-SS legacy)."
        )
        sys.exit(1)

    if is_per_ss:
        if not isinstance(ss_cfg["selected"], list) or not ss_cfg["selected"]:
            logger.error("'super_subjects.selected' debe ser una lista no vacia.")
            sys.exit(1)
        # If 'groups' is not provided, 'subjects_per_super_subject' must be set
        if "groups" not in ss_cfg:
            if "subjects_per_super_subject" not in ss_cfg:
                logger.error(
                    "Se requiere 'subjects_per_super_subject' cuando 'groups' "
                    "no esta definido en 'super_subjects'."
                )
                sys.exit(1)
    else:
        # multi-SS mode
        if "total_subjects" not in ss_cfg:
            logger.error(
                "En modo multi-SS, 'super_subjects.total_subjects' es requerido."
            )
            sys.exit(1)

    # Validate cdhsa_params
    cdhsa = params["cdhsa_params"]
    if "L" not in cdhsa:
        logger.error("'cdhsa_params' debe contener 'L' (subspace dimension).")
        sys.exit(1)

    # [LATENT] Validate the optional latent_plots block when enabled
    latent_cfg = params.get("latent_plots") or {}
    if latent_cfg.get("enabled", False):
        methods = latent_cfg.get("methods")
        if not isinstance(methods, list) or not methods:
            logger.error(
                "latent_plots.enabled=true requiere una lista no vacia "
                "de metodos en latent_plots.methods."
            )
            sys.exit(1)
        for method in methods:
            for req_key in ("label", "stage2_dynamics", "stage3_selection"):
                if req_key not in method:
                    logger.error(
                        "El metodo '%s' de latent_plots falta la clave "
                        "requerida '%s'",
                        method.get("label", "<sin label>"), req_key,
                    )
                    sys.exit(1)

    return params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _super_subject_label(super_subject_id: int) -> str:
    """BIDS-style label, e.g. ``super_subject-01``."""
    return f"super_subject-{super_subject_id:02d}"


def _resolve_subject_ids_for_job(
    super_subject_id: int, ss_cfg: dict,
) -> list[int] | None:
    """Resolve the explicit subject-ids for a super-subject, or None
    if the auto-resolution should be delegated to the pipeline.

    Returns
    -------
    list[int] | None
        * ``None`` when the JSON does not declare a ``groups`` block;
          the pipeline will auto-resolve from
          ``subjects_per_super_subject`` + ``subject_start_offset``.
        * A list of subject indices when ``groups`` is declared.
    """
    groups = ss_cfg.get("groups")
    if groups is None:
        return None

    # JSON object keys are strings -> coerce to int when possible
    key = super_subject_id if super_subject_id in groups else str(super_subject_id)
    if key not in groups:
        raise KeyError(
            f"super_subject_id={super_subject_id} not in groups "
            f"(keys: {list(groups.keys())})"
        )
    return [int(x) for x in groups[key]]


# ===========================================================================
# JOB GENERATION
# ===========================================================================


def _generate_jobs(params: dict) -> list[dict]:
    """Generate the list of jobs from the JSON parameters.

    **Multi-SS mode** (``n_super_subjects`` in JSON):
        One job per session.  All super-subjects are passed together
        to a single invocation of ``run_cdhsa.py`` (S=N, C=tasks).

    **Per-SS mode** (``selected`` in JSON, legacy):
        One job per (super-subject, session) pair (S=1 each).
    """
    ss_cfg = params["super_subjects"]
    sessions = params["sessions"]
    tasks = params["tasks"]
    tw = params["time_window"]
    cdhsa = params["cdhsa_params"]
    t_start = str(tw["t_start"])
    t_end = str(tw["t_end"])

    jobs: list[dict] = []

    # --- Multi-SS mode ---
    if "n_super_subjects" in ss_cfg:
        n_ss = ss_cfg["n_super_subjects"]
        for session in sessions:
            jobs.append({
                "mode": "multi_ss",
                "n_super_subjects": n_ss,
                "subjects_per_super_subject": ss_cfg.get("subjects_per_super_subject"),
                "subject_start_offset": ss_cfg.get("subject_start_offset", 1),
                "total_subjects": ss_cfg["total_subjects"],
                "session": session,
                "tasks": tasks,
                "cdhsa_params": cdhsa,
                "ss_cfg": ss_cfg,
                "t_start": t_start,
                "t_end": t_end,
            })
        return jobs

    # --- Per-SS mode (legacy) ---
    selected_ids: list[int] = list(ss_cfg["selected"])
    for sid in selected_ids:
        ss_label = _super_subject_label(sid)
        try:
            subject_ids = _resolve_subject_ids_for_job(sid, ss_cfg)
        except KeyError as exc:
            logger.error("%s -- este super-sujeto se omitira.", exc)
            continue

        for session in sessions:
            jobs.append({
                "mode": "per_ss",
                "super_subject_id": sid,
                "super_subject_label": ss_label,
                "subject_ids": subject_ids,
                "session": session,
                "tasks": tasks,
                "cdhsa_params": cdhsa,
                "ss_cfg": ss_cfg,
                "t_start": t_start,
                "t_end": t_end,
            })
    return jobs


# ===========================================================================
# BATCH RUNNER
# ===========================================================================


class CDHSABatchRunner:
    """Orchestrates batch execution of the CD-HSA pipeline."""

    DEFAULT_PIPELINE_MODULE = "src.pipelines.run_cdhsa"

    CSV_FIELDS = [
        "timestamp", "mode", "super_subject", "session", "tasks",
        "L", "fixed_rank", "rank_method", "hankel_depth",
        "t_start", "t_end", "success", "returncode",
        "elapsed_s", "command",
    ]

    LATENT_CSV_FIELDS = [
        "timestamp", "super_subject", "session", "task", "method_label",
        "spec_label", "spec_hash", "t_start", "t_end",
        "success", "returncode", "elapsed_s", "peak_rss_mb", "command",
    ]

    def __init__(self, params: dict, *, pipeline_script: Path | None = None,
                 pipeline_module: str | None = None) -> None:
        self.params = params
        self.pipeline_script = pipeline_script or (_SCRIPT_DIR.parent / "pipelines" / "run_cdhsa.py")
        self.pipeline_module = pipeline_module or self.DEFAULT_PIPELINE_MODULE
        self.ss_cfg = params["super_subjects"]
        self.exec_cfg = params.get("execution", {})

        # Execution settings
        self.delay: float = self.exec_cfg.get("delay", 2.0)
        self.max_workers: int = self.exec_cfg.get("max_workers", 1)
        self.run_comparison: bool = self.exec_cfg.get("run_comparison", True)

        # Optional paths (may be None if not configured in the project)
        self.db_path: str | None = None
        self.output_dir: Path | None = None
        self.cache_dir: Path | None = None
        self._resolve_project_paths()

        # Checkpoint
        self.checkpoint: set[str] = self._load_checkpoint()

        # CSV log
        self.log_file = self._resolve_log_path()
        self._init_csv_log()

        # Generate jobs
        self.all_jobs = _generate_jobs(params)

        # ------------------------------------------------------------------
        # [LATENT] Latent-space + plots phase (runs AFTER mode extraction)
        # ------------------------------------------------------------------
        self.latent_cfg: dict = params.get("latent_plots") or {}
        self.run_latent_phase: bool = bool(self.latent_cfg.get("enabled", False))
        self.latent_pipeline_module: str = self.latent_cfg.get(
            "pipeline_module",
            "src.pipelines.latent_space_plots_from_eeg_gedai_super_subject",
        )
        self.latent_shared: dict = self.latent_cfg.get("shared_params", {}) or {}
        _latent_exec = self.latent_cfg.get("execution", {}) or {}
        self.latent_delay: float = _latent_exec.get("delay", 2.0)
        self.latent_max_workers: int = _latent_exec.get("max_workers", 1)
        self.latent_ignore_cache: bool = bool(
            self.latent_shared.get("ignore_cache", False)
        )

        self.latent_jobs: list[dict] = []
        self._latent_ch_intersections: dict[tuple[str, str], list[str]] = {}
        self._latent_job_memory_stats: list[dict] = []
        if self.run_latent_phase:
            self.latent_checkpoint: set[str] = self._load_latent_checkpoint()
            self.latent_log_file = self._init_latent_csv_log()
            self.latent_jobs = self._generate_latent_jobs()

        self._print_banner()

        # [MEM TRACKING] Per-job memory stats
        self._job_memory_stats: list[dict] = []

    # ------------------------------------------------------------------
    # Project paths
    # ------------------------------------------------------------------

    def _resolve_project_paths(self) -> None:
        """Try to resolve project paths from src.utils.config if available."""
        try:
            from src.utils.config import (
                BASE_CACHE_PATH,
                BASE_PARAMS_FILE,
                BASE_RESULTS_PATH,
                DB_TEST_RETEST_GEDAI_PATH,
            )
            self.db_path = str(DB_TEST_RETEST_GEDAI_PATH)
            self.output_dir = Path(BASE_RESULTS_PATH)
            self.cache_dir = Path(BASE_CACHE_PATH)
            self.params_dir = Path(BASE_PARAMS_FILE)
        except ImportError:
            logger.info(
                "src.utils.config no disponible; usando rutas por defecto. "
                "Usa --db-path y --out-dir si es necesario."
            )
            self.output_dir = Path("./results")
            self.cache_dir = Path("./cache")
            self.params_dir = Path("./params")

        # [NTFY] Notification channel: imported from src.utils.config like
        # every other environment variable.  Separate try-block so an
        # outdated config.py without NTFY_CHANNEL only disables notifications.
        try:
            from src.utils.config import NTFY_CHANNEL
            self.ntfy_channel: str | None = (NTFY_CHANNEL or "").strip() or None
        except ImportError:
            self.ntfy_channel = None

        if self.ntfy_channel:
            logger.info(
                "[NTFY] Notificaciones push ACTIVAS (canal: %s)",
                self.ntfy_channel,
            )
        else:
            logger.info(
                "[NTFY] Notificaciones push desactivadas (define NTFY_CHANNEL "
                "en el .env; ver src/ntfy/README.md)",
            )

    def _resolve_log_path(self) -> Path:
        """Resolve the path for the batch CSV log file."""
        if self.output_dir:
            log_dir = self.output_dir / "batch_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
        else:
            log_dir = Path("./batch_logs")
            log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = self.params.get("experiment_label", "batch_cdhsa")
        return log_dir / f"batch_cdhsa_{label}_{ts}.csv"

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        is_multi_ss = any(j.get("mode") == "multi_ss" for j in self.all_jobs)
        n_sessions = len({j["session"] for j in self.all_jobs})
        n_tasks = len(self.params["tasks"])
        cdhsa = self.params["cdhsa_params"]
        tw = self.params["time_window"]

        logger.info("=" * 70)
        if is_multi_ss:
            n_ss = self.ss_cfg["n_super_subjects"]
            spp = self.ss_cfg.get("subjects_per_super_subject")
            logger.info("  BATCH CD-HSA -- MODO MULTI-SS (S=%d, C=%d)", n_ss, n_tasks)
        else:
            n_ss = len({j["super_subject_id"] for j in self.all_jobs})
            logger.info("  BATCH CD-HSA -- MODO PER-SS (S=1 por job, %d SS)", n_ss)
        logger.info("=" * 70)
        logger.info("  Pipeline script : %s", self.pipeline_script)
        logger.info("  DB path         : %s", self.db_path or "(default)")
        logger.info("  Output dir      : %s", self.output_dir)
        logger.info("  Cache dir       : %s", self.cache_dir)
        logger.info("  Checkpoint      : %s", self._checkpoint_path())
        logger.info("  ---")
        if is_multi_ss:
            logger.info("  Super-subjects  : %d", n_ss)
            logger.info("  Subjects/SS     : %d (offset %d)",
                        spp, self.ss_cfg.get("subject_start_offset", 1))
            logger.info("  Total subjects  : %d", self.ss_cfg["total_subjects"])
        else:
            logger.info(
                "  Super-subjects  : %d  -> %s",
                n_ss, sorted({j["super_subject_id"] for j in self.all_jobs}),
            )
            if "groups" in self.ss_cfg:
                logger.info("  Groups mode     : explicit (groups block in JSON)")
                for sid in self.ss_cfg["selected"]:
                    ids = _resolve_subject_ids_for_job(sid, self.ss_cfg)
                    logger.info(
                        "    %s -> %d subjects [%d..%d]",
                        _super_subject_label(sid), len(ids or []),
                        (ids or [0])[0], (ids or [0])[-1],
                    )
            else:
                logger.info(
                    "  Auto-resolve    : %d subjects per super-subject (offset %d)",
                    self.ss_cfg.get("subjects_per_super_subject", 20),
                    self.ss_cfg.get("subject_start_offset", 1),
                )
        logger.info("  Sessions        : %s", self.params["sessions"])
        logger.info("  Tasks (C)       : %s", self.params["tasks"])
        logger.info("  Time window     : %.1f s -> %.1f s (per subject)",
                    tw["t_start"], tw["t_end"])
        logger.info("  ---")
        logger.info("  CDHSA params    :")
        logger.info("    L              : %d", cdhsa["L"])
        logger.info("    hankel_depth   : %s", cdhsa.get("hankel_depth", "auto"))
        logger.info("    l_freq-h_freq  : %.1f-%.1f Hz",
                    cdhsa.get("l_freq", 1.0), cdhsa.get("h_freq", 40.0))
        logger.info("    fixed_rank     : %d", cdhsa.get("fixed_rank", 10))
        logger.info("    rank_method    : %s", cdhsa.get("rank_method", "fixed"))
        if cdhsa.get("rank_method", "fixed") == "variance":
            logger.info("    var_explained  : %.2f (cap=%d, piso=%d)",
                        cdhsa.get("var_explained", 0.99),
                        cdhsa.get("var_max_rank", 50),
                        cdhsa.get("var_min_rank", 1))
        logger.info("    a6_max_common  : %s",
                    cdhsa.get("a6_max_common", 0))
        logger.info("    max_common     : %s (tope del pool de A1-A5)",
                    cdhsa.get("max_common", 30))
        logger.info("    a6_null_type   : %s",
                    cdhsa.get("a6_null_type", "haar"))
        logger.info("    a6_n_null      : %d", cdhsa.get("a6_n_null", 100))
        logger.info("    bc_n_perm      : %d", cdhsa.get("bc_n_perm", 5000))
        logger.info("    tangent_blocks : %s",
                    cdhsa.get("tangent_blocks_mode", "omnibus"))
        logger.info("    d_max_specific : %s (adaptive=%s)",
                    cdhsa.get("d_max_specific", 10),
                    cdhsa.get("d_rank_adaptive", False))
        logger.info("    d_loso         : %s (n_perm=%s)",
                    cdhsa.get("d_loso", False),
                    cdhsa.get("d_loso_n_perm", 1000))
        logger.info("    replica_consist: %s",
                    cdhsa.get("run_replica_consistency", False))
        logger.info("    skip_bc        : %s", cdhsa.get("skip_bc", False))
        logger.info("    skip_tangent   : %s", cdhsa.get("skip_tangent", False))
        logger.info("    skip_d         : %s", cdhsa.get("skip_d", False))
        logger.info("  ---")
        logger.info("  Total jobs      : %d", len(self.all_jobs))
        logger.info("  Delay           : %.1f s", self.delay)
        logger.info("  Max workers     : %d (%s)",
                    self.max_workers,
                    "paralelo" if self.max_workers > 1 else "secuencial")
        logger.info("  Comparison      : %s", self.run_comparison)
        logger.info("  ---")
        if self.run_latent_phase:
            n_latent_ss = len({j["super_subject_id"] for j in self.latent_jobs})
            logger.info(
                "  [LATENT] Fase espacio latente + plots : ACTIVADA"
            )
            logger.info("  [LATENT] Pipeline  : %s", self.latent_pipeline_module)
            logger.info(
                "  [LATENT] Jobs      : %d (%d SS x %d sess x %d task x %d meth)",
                len(self.latent_jobs), n_latent_ss,
                len(self.params["sessions"]), len(self.params["tasks"]),
                len(self.latent_cfg.get("methods", [])),
            )
            logger.info("  [LATENT] Checkpoint: %s", self._latent_checkpoint_path())
            logger.info(
                "  [LATENT] Workers   : %d | delay %.1f s | ignore_cache %s",
                self.latent_max_workers, self.latent_delay,
                self.latent_ignore_cache,
            )
            logger.info(
                "  [LATENT] (KM / potenciales: RECORTADOS por diseno)"
            )
        else:
            logger.info(
                "  [LATENT] Fase espacio latente + plots : desactivada "
                "(latent_plots.enabled=false o ausente)"
            )
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _checkpoint_path(self) -> Path:
        if self.cache_dir:
            return self.cache_dir / "batch_checkpoint_cdhsa.json"
        return Path("./cache") / "batch_checkpoint_cdhsa.json"

    def _load_checkpoint(self) -> set[str]:
        cp = self._checkpoint_path()
        if cp.exists():
            try:
                with open(cp, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                ck = set(data.get("completed", []))
                logger.info(
                    "Checkpoint cargado: %d jobs previos completados", len(ck)
                )
                return ck
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Checkpoint corrupto, empezando de cero: %s", exc)
        return set()

    def _save_checkpoint(self) -> None:
        cp = self._checkpoint_path()
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            with open(cp, "w", encoding="utf-8") as fh:
                json.dump({"completed": sorted(self.checkpoint)}, fh, indent=2)
        except OSError as exc:
            logger.warning("No se pudo guardar checkpoint: %s", exc)

    @staticmethod
    def _checkpoint_key(job: dict) -> str:
        """Build a unique checkpoint key from a job dict.

        Incluye L, la etiqueta de rango (frXX / varXXcYY), el modo de
        bloques tangentes y el estado del enmascaramiento de fronteras
        para que un cambio de configuracion (p.ej. fixed -> variance,
        omnibus -> cumulative, fronteras on/off) re-ejecute el job en
        vez de saltarselo por un checkpoint viejo.
        """
        tasks_str = "+".join(job["tasks"])
        cdhsa = job["cdhsa_params"]
        rtag = CDHSABatchRunner._rank_tag(cdhsa)
        tb = cdhsa.get("tangent_blocks_mode", "cumulative")[:4]
        bm = "bm0" if cdhsa.get("no_boundary_mask", False) else "bm1"
        mc = f"mc{cdhsa.get('max_common', 30)}"
        if job.get("mode") == "multi_ss":
            return (f"multiss_{job['n_super_subjects']}|{job['session']}|"
                    f"{tasks_str}|{job['t_start']}|{job['t_end']}|"
                    f"L{cdhsa.get('L', '?')}|{rtag}|{mc}|{tb}|{bm}")
        else:
            sid = job["super_subject_id"]
            return (f"ss{sid:02d}|{job['session']}|{tasks_str}|"
                    f"{job['t_start']}|{job['t_end']}|"
                    f"L{cdhsa.get('L', '?')}|{rtag}|{mc}|{tb}|{bm}")

    # ------------------------------------------------------------------
    # CSV log
    # ------------------------------------------------------------------

    def _init_csv_log(self) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.log_file, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.CSV_FIELDS)
                writer.writeheader()
        except OSError as exc:
            logger.warning("No se pudo inicializar log CSV: %s", exc)

    def _write_csv_log(self, job: dict, success: bool,
                       returncode: int, elapsed: float,
                       cmd: list[str]) -> None:
        try:
            with open(self.log_file, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.CSV_FIELDS)
                writer.writerow({
                    "timestamp": datetime.now().isoformat(),
                    "mode": job.get("mode", "per_ss"),
                    "super_subject": job.get(
                        "super_subject_label",
                        f"multi-SS({job.get('n_super_subjects', '?')})"
                    ),
                    "session": job["session"],
                    "tasks": "+".join(job["tasks"]),
                    "L": job["cdhsa_params"].get("L", ""),
                    "fixed_rank": job["cdhsa_params"].get("fixed_rank", ""),
                    "rank_method": job["cdhsa_params"].get("rank_method", "fixed"),
                    "hankel_depth": job["cdhsa_params"].get("hankel_depth", ""),
                    "t_start": job["t_start"],
                    "t_end": job["t_end"],
                    "success": success,
                    "returncode": returncode,
                    "elapsed_s": round(elapsed, 2),
                    "command": " ".join(cmd),
                })
        except OSError as exc:
            logger.warning("Fallo al escribir log CSV: %s", exc)

    # ------------------------------------------------------------------
    # Filter out already-completed jobs
    # ------------------------------------------------------------------

    def _filter_todo(self, jobs: list[dict]) -> list[dict]:
        todo: list[dict] = []
        for job in jobs:
            key = self._checkpoint_key(job)
            if key in self.checkpoint:
                label = job.get("super_subject_label", "multi-SS")
                logger.debug(
                    "SKIP (checkpoint): %s/%s",
                    label, job["session"],
                )
                continue
            todo.append(job)

        if skipped := len(jobs) - len(todo):
            logger.info("Jobs ya completados (skip): %d / %d", skipped, len(jobs))
        return todo

    # ------------------------------------------------------------------
    # Build subprocess command
    # ------------------------------------------------------------------

    def _build_command(self, job: dict) -> list[str]:
        """Build the subprocess command for a CD-HSA job.

        Multi-SS mode: ``--n-super-subjects`` (S=N, all SS together).
        Per-SS mode:  ``--super-subject-id`` (S=1, legacy).
        """
        cdhsa = job["cdhsa_params"]
        ss_cfg = job["ss_cfg"]

        cmd = [
            sys.executable, "-m", self.pipeline_module,
            "--session", job["session"],
        ]

        # --- Mode selection (mutually exclusive in run_cdhsa.py CLI) ---
        if job.get("mode") == "multi_ss":
            cmd.extend([
                "--n-super-subjects", str(job["n_super_subjects"]),
                "--total-subjects", str(job["total_subjects"]),
            ])
            if job.get("subjects_per_super_subject") is not None:
                cmd.extend([
                    "--subjects-per-super-subject",
                    str(job["subjects_per_super_subject"]),
                ])
            if job.get("subject_start_offset", 1) != 1:
                cmd.extend([
                    "--subject-start-offset",
                    str(job["subject_start_offset"]),
                ])
        else:
            # Per-SS legacy
            cmd.extend([
                "--super-subject-id", str(job["super_subject_id"]),
            ])
            if job.get("subject_ids") is not None:
                pass  # groups handled by pipeline
            cmd.extend([
                "--subjects-per-super-subject",
                str(ss_cfg.get("subjects_per_super_subject", 20)),
                "--subject-start-offset",
                str(ss_cfg.get("subject_start_offset", 1)),
            ])

        # Tasks (all in a single --tasks invocation, since run_cdhsa.py uses nargs='+')
        cmd.extend(["--tasks"] + job["tasks"])

        # Time window
        cmd.extend([
            "--t-start", job["t_start"],
            "--t-end", job["t_end"],
        ])

        # CD-HSA parameters
        cmd.extend([
            "--L", str(cdhsa["L"]),
            "--l-freq", str(cdhsa.get("l_freq", 1.0)),
            "--h-freq", str(cdhsa.get("h_freq", 40.0)),
            "--fixed-rank", str(cdhsa.get("fixed_rank", 10)),
            "--rank-method", str(cdhsa.get("rank_method", "fixed")),
            "--max-common", str(cdhsa.get("max_common", 30)),
            "--a6-n-null", str(cdhsa.get("a6_n_null", 100)),
            "--bc-n-perm", str(cdhsa.get("bc_n_perm", 5000)),
            "--d-max-specific", str(cdhsa.get("d_max_specific", 10)),
        ])

        # --- Mejoras v3 (solo se pasan si estan presentes/no-default) ---
        if cdhsa.get("rank_method") == "variance":
            cmd.extend([
                "--var-explained", str(cdhsa.get("var_explained", 0.99)),
                "--var-max-rank", str(cdhsa.get("var_max_rank", 50)),
                "--var-min-rank", str(cdhsa.get("var_min_rank", 1)),
            ])
        if cdhsa.get("a6_max_common"):
            cmd.extend(["--a6-max-common", str(cdhsa["a6_max_common"])])
        if cdhsa.get("a6_null_type", "haar") != "haar":
            cmd.extend(["--a6-null-type", str(cdhsa["a6_null_type"])])
            if cdhsa.get("a6_hankel_row_block"):
                cmd.extend([
                    "--a6-hankel-row-block",
                    str(cdhsa["a6_hankel_row_block"]),
                ])
        # (v4) Bloques tangentes: el default del pipeline ahora es
        # 'cumulative' (paper 3.4); se pasa SIEMPRE que difiera de el.
        if cdhsa.get("tangent_blocks_mode", "cumulative") != "cumulative":
            cmd.extend([
                "--tangent-blocks-mode",
                str(cdhsa["tangent_blocks_mode"]),
            ])
        # (v4) Enmascaramiento de fronteras: default ON; solo se pasa el
        # opt-out para reproducir ejecuciones pre-v4.
        if cdhsa.get("no_boundary_mask", False):
            cmd.append("--no-boundary-mask")
        # (v4) Rango efectivo X% (Remark 3.6) y outcome del test de rango.
        if cdhsa.get("effective_rank", False):
            cmd.append("--effective-rank")
        if cdhsa.get("rank_outcome", "selected") != "selected":
            cmd.extend(["--rank-outcome", str(cdhsa["rank_outcome"])])
        if cdhsa.get("d_rank_adaptive", False):
            cmd.append("--d-rank-adaptive")
        if "d_n_null_specific" in cdhsa:
            cmd.extend([
                "--d-n-null-specific", str(cdhsa["d_n_null_specific"]),
            ])
        if "d_alpha_specific" in cdhsa:
            cmd.extend([
                "--d-alpha-specific", str(cdhsa["d_alpha_specific"]),
            ])
        if cdhsa.get("d_loso", False):
            cmd.append("--d-loso")
            if "d_loso_n_perm" in cdhsa:
                cmd.extend([
                    "--d-loso-n-perm", str(cdhsa["d_loso_n_perm"]),
                ])
        if cdhsa.get("run_replica_consistency", False):
            cmd.append("--run-replica-consistency")

        # Optional hankel depth
        if cdhsa.get("hankel_depth") is not None:
            cmd.extend(["--hankel-depth", str(cdhsa["hankel_depth"])])

        # Boolean flags
        if cdhsa.get("skip_bc", False):
            cmd.append("--skip-bc")
        if cdhsa.get("skip_tangent", False):
            cmd.append("--skip-tangent")
        if cdhsa.get("skip_d", False):
            cmd.append("--skip-d")

        # Optional paths
        if self.db_path:
            cmd.extend(["--db-path", self.db_path])
        if self.output_dir:
            cmd.extend(["--out-dir", str(self.output_dir)])

        return cmd

    # ------------------------------------------------------------------
    # Output dir for a job (mirrors run_cdhsa.py _resolve_out_dir)
    # ------------------------------------------------------------------

    @staticmethod
    def _rank_tag(cdhsa: dict) -> str:
        """Etiqueta de rango para rutas y checkpoints.

        Debe mantenerse sincronizada con
        ``src.pipelines.run_cdhsa._rank_tag``.
        """
        rm = cdhsa.get("rank_method", "fixed")
        if rm == "variance":
            ve = float(cdhsa.get("var_explained", 0.99))
            cap = cdhsa.get("var_max_rank", 50)
            return f"var{round(ve * 100)}c{cap}"
        if rm == "reproducibility":
            return "repro"
        return f"fr{cdhsa.get('fixed_rank', 10)}"

    def _get_output_dir(self, job: dict) -> Path:
        """Return the directory where the pipeline saves results.

        Mirrors the path scheme in ``run_cdhsa.py``'s ``_resolve_out_dir()``.
        Multi-SS uses ``nSS{N}``, per-SS uses ``SS{id}``.
        """
        if not self.output_dir:
            return Path(".")

        cdhsa = job["cdhsa_params"]
        tw = {"t_start": job["t_start"], "t_end": job["t_end"]}
        tasks = job["tasks"]

        t_start_tag = tw["t_start"] if tw["t_start"] != "None" else "any"
        t_end_tag = tw["t_end"] if tw["t_end"] != "None" else "any"

        L = cdhsa["L"]
        a6n = cdhsa.get("a6_n_null", 100)
        bcn = cdhsa.get("bc_n_perm", 5000)
        l_freq = cdhsa.get("l_freq", 1.0)
        h_freq = cdhsa.get("h_freq", 40.0)
        depth = cdhsa.get("hankel_depth", "auto")

        if job.get("mode") == "multi_ss":
            n_ss = job["n_super_subjects"]
            ss_label = f"nSS{n_ss}"
        else:
            ss_label = f"SS{job['super_subject_id']}"

        rtag = self._rank_tag(cdhsa)

        out_dir = Path(
            f"{self.output_dir}/cdhsa/{job['session']}"
            f"/{ss_label}_L{L}"
            f"_{rtag}_a6n{a6n}_bcn{bcn}"
            f"/{l_freq}-{h_freq}Hz"
            f"_depth{depth}"
            f"/from{t_start_tag}s_to{t_end_tag}s"
            f"_{'_'.join(tasks)}"
        )
        return out_dir

    # ------------------------------------------------------------------
    # Child process memory polling
    # ------------------------------------------------------------------

    def _poll_child_rss(
        self, proc: subprocess.Popen, interval: float = 0.5,
    ) -> tuple[float, list]:
        """Poll *proc* RSS from the parent process.

        Returns ``(peak_rss_mb, samples_list)``.
        Stops as soon as the child terminates.
        """
        peak_rss = 0.0
        samples: list[tuple[float, float]] = []   # (elapsed, rss_mb)
        if not _HAS_PSUTIL:
            return peak_rss, samples
        t0 = time.time()
        try:
            p = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    rss = p.memory_info().rss / (1024 * 1024)
                    peak_rss = max(peak_rss, rss)
                    samples.append((time.time() - t0, rss))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                time.sleep(interval)
        except psutil.NoSuchProcess:
            pass
        return peak_rss, samples

    # ------------------------------------------------------------------
    # Per-job memory report
    # ------------------------------------------------------------------

    def _print_memory_report(self) -> None:
        """Log + save a per-job peak-RSS summary table."""
        stats = self._job_memory_stats
        if not stats:
            logger.info("[MEM] No hay estadisticas de memoria por job.")
            return

        sorted_stats = sorted(
            stats, key=lambda s: s["peak_rss_mb"], reverse=True,
        )

        logger.info("")
        logger.info("=" * 70)
        logger.info("  MEMORY REPORT POR JOB  (ordenado por peak RSS)")
        logger.info("=" * 70)

        hdr = "  %-50s %10s %8s %6s" % (
            "Job", "Peak RSS", "Elapsed", "Status")
        logger.info(hdr)
        logger.info("  " + "-" * 82)

        for s in sorted_stats:
            status = "OK" if s["success"] else "FAIL"
            logger.info(
                "  %-50s %8.1f MB %6.1f s   %s",
                '%s/%s' % (s["label"], s["session"]),
                s["peak_rss_mb"],
                s["elapsed_s"],
                status,
            )

        max_s = sorted_stats[0] if sorted_stats else {}
        logger.info("  " + "-" * 82)
        logger.info(
            "  Peak maximo global: %.1f MB (%s)",
            max_s.get("peak_rss_mb", 0),
            max_s.get("label", ""),
        )
        n_measured = len([s for s in stats if s["peak_rss_mb"] > 0])
        logger.info(
            "  Jobs con memoria medida: %d / %d",
            n_measured, len(stats),
        )
        logger.info("=" * 70)

        # --- save CSV ---
        if self.output_dir:
            log_dir = self.output_dir / "batch_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = log_dir / ("memory_per_job_%s.csv" % ts)
            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=[
                        "label", "session", "success", "elapsed_s",
                        "peak_rss_mb", "n_rss_samples",
                    ])
                    writer.writeheader()
                    for s in stats:
                        writer.writerow({
                            k: s[k] for k in
                            ["label", "session", "success",
                             "elapsed_s", "peak_rss_mb", "n_rss_samples"]
                        })
                logger.info("[MEM] Per-job CSV guardado: %s", csv_path)
            except OSError as exc:
                logger.warning("[MEM] Error guardando per-job CSV: %s", exc)

    # ------------------------------------------------------------------
    # [NTFY] Push notifications
    # ------------------------------------------------------------------

    def _notify(self, level: str, message: str, title: str | None = None) -> None:
        """Send a push notification through ntfy (best-effort, never raises).

        ``level`` is one of ``info | success | warning | error``.  Without a
        configured channel (or without the ``src.ntfy`` module) this is a
        silent no-op, so it is always safe to call.
        """
        if not (_HAS_NTFY and self.ntfy_channel):
            return
        senders = {
            "info": notify_info,
            "success": notify_success,
            "warning": notify_warning,
            "error": notify_error,
        }
        fn = senders.get(level)
        if fn is None:
            logger.warning("[NTFY] Nivel de notificacion desconocido: %r", level)
            return
        try:
            if title is not None:
                fn(message, channel=self.ntfy_channel, title=title)
            else:
                fn(message, channel=self.ntfy_channel)
        except Exception as exc:
            # Paranoia: a notification must never kill the batch.
            logger.warning("[NTFY] Fallo enviando notificacion: %s", exc)

    # ------------------------------------------------------------------
    # Run a single job
    # ------------------------------------------------------------------

    def _run_single_job(self, job: dict) -> tuple[str, bool]:
        key = self._checkpoint_key(job)

        try:
            cmd = self._build_command(job)
        except Exception as exc:
            label = job.get("super_subject_label", "multi-SS")
            logger.error(
                "Error construyendo comando para %s/%s: %s",
                label, job["session"], exc,
            )
            self._notify(
                "error",
                "Error construyendo el comando para %s/%s:\n%s\n"
                "El batch continua con los demas jobs." % (
                    label, job["session"], exc,
                ),
                title="[CD-HSA] Fallo de job",
            )
            return key, False

        label = job.get("super_subject_label", f"multi-SS(S={job.get('n_super_subjects', '?')})")
        logger.info(
            "RUN | %s/%s | tasks=%s | L=%d | [%s-%s] s",
            label, job["session"],
            "+".join(job["tasks"]),
            job["cdhsa_params"]["L"],
            job["t_start"], job["t_end"],
        )
        logger.debug("CMD: %s", " ".join(cmd))

        # [MEM TRACKING] Checkpoint before job
        if _HAS_MEM_TRACKER:
            _mon = get_global_monitor()
            if _mon.is_running:
                _mon.checkpoint(
                    'JOB START: %s/%s' % (label, job["session"]))

        t0 = time.time()
        cmd_for_log = cmd  # keep reference in case of exception
        try:
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=child_env,
            )

            # [MEM TRACKING] Poll child process RSS in a daemon thread
            _child_peak = [0.0]
            _child_samples: list[tuple[float, float]] = []

            def _poll_thread():
                pk, samps = self._poll_child_rss(proc)
                _child_peak[0] = pk
                _child_samples.extend(samps)

            _thr = threading.Thread(target=_poll_thread, daemon=True)
            _thr.start()

            for line in proc.stdout:
                logger.info("  [PIPE] %s", line.rstrip())

            proc.wait()
            _thr.join(timeout=3.0)

            elapsed = time.time() - t0
            success = proc.returncode == 0
            peak_mb = _child_peak[0]

            # [MEM TRACKING] Record per-job stats
            self._job_memory_stats.append({
                "label": label,
                "session": job["session"],
                "success": success,
                "elapsed_s": round(elapsed, 2),
                "peak_rss_mb": round(peak_mb, 1),
                "n_rss_samples": len(_child_samples),
            })

            self._write_csv_log(job, success, proc.returncode, elapsed, cmd)

            if peak_mb > 0:
                logger.info(
                    "  [MEM] Peak child RSS: %.1f MB (%d samples)",
                    peak_mb, len(_child_samples),
                )

            if success:
                logger.info(
                    "OK | %s/%s (%.1f s, peak %.1f MB)",
                    label, job["session"], elapsed, peak_mb,
                )
            else:
                logger.error(
                    "ERROR | %s/%s -- codigo %d",
                    label, job["session"],
                    proc.returncode,
                )
                self._notify(
                    "error",
                    "Job %s/%s termino con codigo %d (fallo).\n"
                    "El batch continua con los demas jobs." % (
                        label, job["session"], proc.returncode,
                    ),
                    title="[CD-HSA] Fallo de job",
                )

            # [MEM TRACKING] Checkpoint after job
            if _HAS_MEM_TRACKER:
                _mon = get_global_monitor()
                if _mon.is_running:
                    _mon.checkpoint(
                        'JOB END: %s/%s' % (label, job["session"]))

            return key, success

        except Exception as exc:
            elapsed = time.time() - t0
            logger.error(
                "EXCEPTION | %s/%s: %s",
                label, job["session"], exc,
            )
            self._notify(
                "error",
                "Excepcion ejecutando %s/%s:\n%s: %s\n"
                "El batch continua con los demas jobs." % (
                    label, job["session"], type(exc).__name__, exc,
                ),
                title="[CD-HSA] Fallo de job",
            )
            self._write_csv_log(
                job, False, -1, elapsed, cmd_for_log,
            )
            return key, False

    # ------------------------------------------------------------------
    # Post-processing: cross-super-subject comparison
    # ------------------------------------------------------------------

    def _run_comparison(self) -> None:
        """Compile a cross-super-subject comparison of CD-HSA results.

        For each completed job, reads ``cdhsa_summary.txt`` and
        ``config.json`` from the output directory and produces a
        consolidated comparison report.
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("  INICIANDO POST-PROCESAMIENTO: COMPARACION CRUZADA")
        logger.info("=" * 70)

        if not self.output_dir:
            logger.warning(
                "No se definio output_dir; no se puede ejecutar la comparacion."
            )
            return

        # Group jobs by session
        by_session: dict[str, list[dict]] = {}
        for job in self.all_jobs:
            by_session.setdefault(job["session"], []).append(job)

        total_groups = len(by_session)
        reports_generated = 0
        reports_missing = 0

        for session, jobs in sorted(by_session.items()):
            logger.info("")
            logger.info("  --- Session: %s ---", session)

            comparison_lines: list[str] = []
            comparison_lines.append("")
            comparison_lines.append("=" * 70)
            comparison_lines.append(
                f"  CD-HSA CROSS-SUPER-SUBJECT COMPARISON -- {session}"
            )
            comparison_lines.append(
                f"  Experiment: {self.params.get('experiment_label', 'N/A')}"
            )
            comparison_lines.append(
                f"  Tasks: {', '.join(self.params['tasks'])}"
            )
            comparison_lines.append(
                f"  Generated: {datetime.now().isoformat()}"
            )
            comparison_lines.append("=" * 70)

            n_found = 0
            for job in jobs:
                ss_label = job.get(
                    "super_subject_label",
                    f"multi-SS(S={job.get('n_super_subjects', '?')})",
                )
                out_dir = self._get_output_dir(job)
                summary_path = out_dir / "cdhsa_summary.txt"
                config_path = out_dir / "config.json"

                comparison_lines.append("")
                comparison_lines.append(f"  {'─' * 60}")
                comparison_lines.append(f"  {ss_label}  |  {session}")
                comparison_lines.append(f"  Output: {out_dir}")
                comparison_lines.append(f"  {'─' * 60}")

                if config_path.exists():
                    try:
                        with open(config_path, "r") as fh:
                            cfg = json.load(fh)
                        comparison_lines.append(
                            f"  L={cfg.get('L', '?')}  "
                            f"fixed_rank={cfg.get('fixed_rank', '?')}  "
                            f"a6_n_null={cfg.get('a6_n_null', '?')}  "
                            f"bc_n_perm={cfg.get('bc_n_perm', '?')}"
                        )
                    except Exception as exc:
                        comparison_lines.append(
                            f"  [WARN] Error leyendo config.json: {exc}"
                        )

                if summary_path.exists():
                    try:
                        with open(summary_path, "r") as fh:
                            summary_text = fh.read().strip()
                        # Indent summary content
                        for line in summary_text.split("\n"):
                            comparison_lines.append(f"    {line}")
                        n_found += 1
                    except Exception as exc:
                        comparison_lines.append(
                            f"  [ERROR] Leyendo cdhsa_summary.txt: {exc}"
                        )
                else:
                    comparison_lines.append(
                        "  [MISSING] cdhsa_summary.txt -- job no completado o error"
                    )
                    reports_missing += 1

            comparison_lines.append("")
            comparison_lines.append(f"  Super-subjects con resultados: {n_found} / {len(jobs)}")

            # Save comparison report
            if self.output_dir:
                comp_dir = self.output_dir / "cdhsa" / session / "batch_comparison"
                comp_dir.mkdir(parents=True, exist_ok=True)
                tasks_tag = "_".join(self.params["tasks"])
                comp_path = comp_dir / f"comparison_{tasks_tag}.txt"
                try:
                    with open(comp_path, "w", encoding="utf-8") as fh:
                        fh.write("\n".join(comparison_lines))
                    logger.info("  Guardado: %s", comp_path)
                    reports_generated += 1
                except OSError as exc:
                    logger.error("  Error guardando comparacion: %s", exc)

        logger.info("")
        logger.info("=" * 70)
        logger.info("  COMPARACION CRUZADA COMPLETADA")
        logger.info("=" * 70)
        logger.info("  Sesiones procesadas   : %d", total_groups)
        logger.info("  Reportes generados    : %d", reports_generated)
        logger.info("  Con datos faltantes   : %d sesiones", reports_missing)

    # ------------------------------------------------------------------
    # Post-processing: extract mode indices JSON
    # ------------------------------------------------------------------

    def _run_mode_extraction(self) -> None:
        """Run src.cdhsa.extract_mode_indices for every job with results.

        Executes the mode-extraction script even when all jobs were
        skipped due to the checkpoint cache, so that mode_map.json
        is always generated/updated from the latest results on disk.

        The output JSON is saved under ``self.params_dir`` (i.e.
        ``BASE_PARAMS_FILE`` from ``src.utils.config``).
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("  EXTRAYENDO INDICES DE MODOS ESPECIFICOS (mode_map.json)")
        logger.info("=" * 70)

        if not self.params_dir:
            logger.warning(
                "No se definio params_dir; no se puede guardar mode_map.json"
            )
            return

        self.params_dir.mkdir(parents=True, exist_ok=True)

        top_n = self.params.get("execution", {}).get("mode_extract_top_n", 2)

        n_ok = 0
        n_skip = 0
        n_fail = 0

        for job in self.all_jobs:
            out_dir = self._get_output_dir(job)

            # Verify the results directory has the required files
            if not (out_dir / "cdhsa_arrays.npz").exists():
                logger.debug(
                    "SKIP (sin cdhsa_arrays.npz): %s", out_dir
                )
                n_skip += 1
                continue

            # Build a descriptive output filename
            tasks_tag = "_".join(job["tasks"])
            tw_tag = f"{job['t_start']}s-{job['t_end']}s"
            json_name = (
                f"mode_map_{top_n}_modes_{job['session']}_{tasks_tag}_{tw_tag}.json"
            )
            json_out = self.params_dir / json_name

            # Build command to run the extraction script as a module
            cmd = [
                sys.executable, "-m", "src.cdhsa.extract_mode_indices",
                "--results-dir", str(out_dir),
                "--top-n", str(top_n),
                "-o", str(json_out),
            ]

            label = job.get(
                "super_subject_label",
                f"multi-SS(S={job.get('n_super_subjects', '?')})",
            )
            logger.info(
                "  EXTRACT | %s/%s -> %s",
                label, job["session"], json_name,
            )
            logger.debug("CMD: %s", " ".join(cmd))

            try:
                child_env = os.environ.copy()
                child_env["PYTHONUNBUFFERED"] = "1"

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=child_env,
                )

                for line in proc.stdout:
                    logger.info("  [EXTRACT] %s", line.rstrip())

                proc.wait()

                if proc.returncode == 0:
                    logger.info("  [OK] %s", json_out)
                    n_ok += 1
                else:
                    logger.error(
                        "  [FAIL] extract_mode_indices returncode=%d",
                        proc.returncode,
                    )
                    n_fail += 1

            except Exception as exc:
                logger.error(
                    "  [FAIL] %s/%s: %s",
                    label, job["session"], exc,
                )
                n_fail += 1

        logger.info("")
        logger.info(
            "  Extraccion de modos completada: "
            "OK=%d, Skip=%d, Fail=%d",
            n_ok, n_skip, n_fail,
        )
        if n_ok > 0:
            logger.info("  JSONs guardados en: %s", self.params_dir)

        # [NTFY] v5.1: un fallo de extraccion ya no puede esconderse detras
        # de un "Batch terminado OK" (le paso a la corrida del 2026-09-30:
        # push verde mientras el mode_map.json nunca se genero).
        if n_fail > 0:
            self._notify(
                "error",
                "La extraccion de modos fallo para %d de %d job(s) con "
                "resultados.\n"
                "El pipeline termino bien (no hace falta re-correrlo); "
                "revisa el traceback en la seccion 'EXTRAYENDO INDICES "
                "DE MODOS ESPECIFICOS' del log." % (n_fail, n_ok + n_fail),
                title="[CD-HSA] Extraccion de modos FALLO",
            )

        return n_ok, n_skip, n_fail

    # ==================================================================
    # [LATENT] Latent-space + plots phase
    # ==================================================================

    def _latent_checkpoint_path(self) -> Path:
        if self.cache_dir:
            return self.cache_dir / "batch_checkpoint_cdhsa_latent.json"
        return Path("./cache") / "batch_checkpoint_cdhsa_latent.json"

    def _load_latent_checkpoint(self) -> set[str]:
        cp = self._latent_checkpoint_path()
        if cp.exists():
            try:
                with open(cp, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                ck = set(data.get("completed", []))
                logger.info(
                    "[LATENT] Checkpoint cargado: %d jobs previos completados",
                    len(ck),
                )
                return ck
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "[LATENT] Checkpoint corrupto, empezando de cero: %s", exc
                )
        return set()

    def _save_latent_checkpoint(self) -> None:
        cp = self._latent_checkpoint_path()
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            with open(cp, "w", encoding="utf-8") as fh:
                json.dump(
                    {"completed": sorted(self.latent_checkpoint)}, fh, indent=2
                )
        except OSError as exc:
            logger.warning("[LATENT] No se pudo guardar checkpoint: %s", exc)

    @staticmethod
    def _latent_checkpoint_key(job: dict) -> str:
        """Checkpoint key for one latent job.

        Incluye la etiqueta del metodo, la ventana, latent_dim y el hash
        de la spec (sin las rutas auto-inyectadas) para que un cambio de
        parametros re-ejecute el job en vez de saltarselo.
        """
        return (
            f"ss{job['super_subject_id']:02d}|{job['session']}|{job['task']}"
            f"|{job['method_label']}|{job['t_start']}|{job['t_end']}"
            f"|ld{job['shared'].get('latent_dim', 4)}|{job['spec_hash']}"
        )

    def _init_latent_csv_log(self) -> Path:
        """Create the CSV log for the latent phase. Returns its path."""
        if self.output_dir:
            log_dir = self.output_dir / "batch_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
        else:
            log_dir = Path("./batch_logs")
            log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = self.params.get("experiment_label", "batch_cdhsa")
        log_file = log_dir / f"batch_cdhsa_latent_{label}_{ts}.csv"
        try:
            with open(log_file, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=self.LATENT_CSV_FIELDS,
                )
                writer.writeheader()
        except OSError as exc:
            logger.warning("[LATENT] No se pudo inicializar log CSV: %s", exc)
        return log_file

    def _write_latent_csv_log(self, job: dict, success: bool,
                              returncode: int, elapsed: float,
                              peak_mb: float, cmd: list[str]) -> None:
        try:
            with open(self.latent_log_file, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=self.LATENT_CSV_FIELDS,
                )
                writer.writerow({
                    "timestamp": datetime.now().isoformat(),
                    "super_subject": job["super_subject_label"],
                    "session": job["session"],
                    "task": job["task"],
                    "method_label": job["method_label"],
                    "spec_label": job["spec_label"],
                    "spec_hash": job["spec_hash"],
                    "t_start": job["t_start"],
                    "t_end": job["t_end"],
                    "success": success,
                    "returncode": returncode,
                    "elapsed_s": round(elapsed, 2),
                    "peak_rss_mb": round(peak_mb, 1),
                    "command": " ".join(cmd),
                })
        except OSError as exc:
            logger.warning("[LATENT] Fallo al escribir log CSV: %s", exc)

    @staticmethod
    def _latent_spec_label(method: dict) -> str:
        """Stage-chain label, e.g. ``hankel+cdhsa_specific_modes+top_n``."""
        s1 = method.get("stage1_embedding") or "none"
        return f"{s1}+{method['stage2_dynamics']}+{method['stage3_selection']}"

    @staticmethod
    def _latent_spec_hash(method: dict) -> str:
        """Short SHA-1 of the method spec, WITHOUT the auto-injected paths.

        ``mode_map_path`` / ``npz_path`` are injected per job from this
        batch run's own outputs; hashing them would tie the checkpoint to
        a particular results directory.  Stripping them keeps the key
        stable across re-runs of the same configuration.
        """
        s2_params = dict(method.get("stage2_params", {}) or {})
        s2_params.pop("mode_map_path", None)
        s2_params.pop("npz_path", None)
        spec = {
            "stage1_embedding": method.get("stage1_embedding"),
            "stage1_params": method.get("stage1_params", {}),
            "stage2_dynamics": method["stage2_dynamics"],
            "stage2_params": s2_params,
            "stage3_selection": method["stage3_selection"],
            "stage3_params": method.get("stage3_params", {}),
        }
        payload = json.dumps(spec, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]

    def _resolve_latent_sources(self) -> dict[str, dict]:
        """Resolve, per session, the mode_map and NPZ produced by THIS run.

        Mirrors the filenames computed by ``_run_mode_extraction``::

            mode_map_{top_n}_modes_{session}_{tasks_tag}_{tw_tag}.json
                -> {params_dir}
            cdhsa_arrays.npz -> {output dir of the CD-HSA job}

        (multi-SS only: in per-SS mode every job shares the same
        mode_map filename, so sources cannot be disambiguated per
        super-subject.)
        """
        sources: dict[str, dict] = {}
        for job in self.all_jobs:
            out_dir = self._get_output_dir(job)
            tasks_tag = "_".join(job["tasks"])
            tw_tag = f"{job['t_start']}s-{job['t_end']}s"
            top_n = self.params.get("execution", {}).get("mode_extract_top_n", 2)
            json_name = (
                f"mode_map_{top_n}_modes_{job['session']}_{tasks_tag}_{tw_tag}.json"
            )
            sources[job["session"]] = {
                "mode_map_path": (
                    self.params_dir / json_name if self.params_dir
                    else Path(json_name)
                ),
                "npz_path": out_dir / "cdhsa_arrays.npz",
                "out_dir": out_dir,
            }
        return sources

    def _generate_latent_jobs(self) -> list[dict]:
        """Generate the latent-space + plots jobs.

        One job per (super-subject, session, task, method).  The
        ``mode_map_path`` / ``npz_path`` of each job point at the CD-HSA
        outputs of THIS batch run (see ``_resolve_latent_sources``).
        """
        ss_cfg = self.ss_cfg
        methods = self.latent_cfg.get("methods", [])
        tw = self.params["time_window"]
        t_start = str(tw["t_start"])
        t_end = str(tw["t_end"])

        if "n_super_subjects" in ss_cfg:
            ss_ids = list(range(1, int(ss_cfg["n_super_subjects"]) + 1))
        else:
            ss_ids = list(ss_cfg.get("selected", []))

        sources = self._resolve_latent_sources()

        jobs: list[dict] = []
        for sid in ss_ids:
            ss_label = _super_subject_label(sid)
            for session in self.params["sessions"]:
                src = sources.get(session)
                if src is None:
                    logger.error(
                        "[LATENT] No hay resultados CD-HSA para session=%s; "
                        "los jobs de %s en esa sesion se omitiran.",
                        session, ss_label,
                    )
                    continue
                for task in self.params["tasks"]:
                    for method in methods:
                        jobs.append({
                            "super_subject_id": sid,
                            "super_subject_label": ss_label,
                            "session": session,
                            "task": task,
                            "method_label": method["label"],
                            "method": method,
                            "shared": self.latent_shared,
                            "spec_label": self._latent_spec_label(method),
                            "spec_hash": self._latent_spec_hash(method),
                            "t_start": t_start,
                            "t_end": t_end,
                            "mode_map_path": src["mode_map_path"],
                            "npz_path": src["npz_path"],
                        })
        return jobs

    def _precompute_latent_channel_intersections(self) -> None:
        """Compute the global channel intersection per (session, task).

        Only needed when some method uses ``cdhsa_specific_modes``: the
        CD-HSA modes were trained on the global intersection (e.g. 53
        channels); each super-subject's own intersection may be larger
        (54, 55, ...) and would break the Hankel/W_specific dimension
        match.  Same helper the latent repo's batch uses (header-only
        loads, minimal I/O).
        """
        methods = self.latent_cfg.get("methods", [])
        if not any(
            m.get("stage2_dynamics") == "cdhsa_specific_modes" for m in methods
        ):
            return

        # resolve_all_subject_ids expects the per-SS 'selected' key; build
        # the equivalent cfg for multi-SS (same subject pools).
        ss_cfg = dict(self.ss_cfg)
        if "selected" not in ss_cfg and "n_super_subjects" in ss_cfg:
            ss_cfg["selected"] = list(
                range(1, int(self.ss_cfg["n_super_subjects"]) + 1)
            )

        try:
            from src.latent_space_extraction.super_subject_eeg import (
                compute_channel_intersection,
            )
        except ImportError as exc:
            logger.error(
                "[LATENT] No se pudo importar compute_channel_intersection "
                "(%s); los jobs cdhsa_specific_modes pueden fallar por "
                "desajuste de canales.", exc,
            )
            return

        logger.info("")
        logger.info("[LATENT] PRE-COMPUTANDO INTERSECCIONES GLOBALES DE CANALES")
        for session in self.params["sessions"]:
            for task in self.params["tasks"]:
                key = (session, task)
                logger.info(
                    "[LATENT]   Interseccion para %s/%s ...", session, task,
                )
                try:
                    intersection = compute_channel_intersection(
                        session,
                        task,
                        ss_cfg=ss_cfg,
                        db_path=self.db_path,
                        verbose=False,
                    )
                    self._latent_ch_intersections[key] = intersection
                    logger.info(
                        "[LATENT]   Interseccion global %s/%s: %d canales",
                        session, task, len(intersection),
                    )
                except Exception as exc:
                    logger.error(
                        "[LATENT]   Fallo computando la interseccion para "
                        "%s/%s: %s. Los jobs cdhsa_specific_modes de esta "
                        "combinacion pueden fallar.",
                        session, task, exc,
                    )
        logger.info(
            "[LATENT] Intersecciones computadas para %d combos (session, task).",
            len(self._latent_ch_intersections),
        )

    def _build_latent_command(self, job: dict) -> list[str]:
        """Build the subprocess command for one latent-space job."""
        method = job["method"]
        shared = job["shared"]
        cdhsa = self.params["cdhsa_params"]

        cmd = [
            sys.executable, "-m", self.latent_pipeline_module,
            "--super-subject", str(job["super_subject_id"]),
            "--session", job["session"],
            "--task", job["task"],
            "--t-start", job["t_start"],
            "--t-end", job["t_end"],
            "--latent-dim", str(shared.get("latent_dim", 4)),
            "--l-freq", str(shared.get("l_freq", cdhsa.get("l_freq", 1.0))),
            "--h-freq", str(shared.get("h_freq", cdhsa.get("h_freq", 40.0))),
            "--ica-method", shared.get("ica_method", "picard"),
        ]

        # --- Super-subject pool (mirror of the CD-HSA config) ---
        if "n_super_subjects" in self.ss_cfg:
            cmd.extend([
                "--subjects-per-super-subject",
                str(self.ss_cfg.get("subjects_per_super_subject", 20)),
                "--subject-start-offset",
                str(self.ss_cfg.get("subject_start_offset", 1)),
            ])
        else:
            cmd.extend([
                "--subjects-per-super-subject",
                str(self.ss_cfg.get("subjects_per_super_subject", 20)),
                "--subject-start-offset",
                str(self.ss_cfg.get("subject_start_offset", 1)),
            ])

        # --- Stage 1: Embedding ---
        s1 = method.get("stage1_embedding")
        if s1:
            cmd.extend(["--stage1-embedding", s1])
            s1_params = dict(method.get("stage1_params", {}) or {})
            if s1 == "hankel" and "depth" not in s1_params:
                # Default: the SAME embedding depth the CD-HSA modes were
                # trained with (otherwise the Hankel row dim mismatches).
                hdepth = cdhsa.get("hankel_depth")
                if hdepth is not None:
                    s1_params["depth"] = hdepth
            if s1_params:
                cmd.extend(["--stage1-params", json.dumps(s1_params)])
        else:
            cmd.extend(["--stage1-embedding", "none"])

        # --- Stage 2: Dynamics (+ auto-injected mode_map / npz) ---
        s2 = method["stage2_dynamics"]
        cmd.extend(["--stage2-dynamics", s2])
        s2_params = dict(method.get("stage2_params", {}) or {})
        if s2 == "cdhsa_specific_modes":
            s2_params.setdefault("mode_map_path", str(job["mode_map_path"]))
            s2_params.setdefault("npz_path", str(job["npz_path"]))
        if s2_params:
            cmd.extend(["--stage2-params", json.dumps(s2_params)])

        # --- Stage 3: Selection ---
        cmd.extend(["--stage3-selection", method["stage3_selection"]])
        s3_params = method.get("stage3_params", {})
        if s3_params:
            cmd.extend(["--stage3-params", json.dumps(s3_params)])

        # --- Shared optional params ---
        if shared.get("analysis_dim") is not None:
            cmd.extend(["--analysis-dim", str(shared["analysis_dim"])])
        if shared.get("column") is not None:
            cmd.extend(["--column", str(shared["column"])])
        if shared.get("workers") is not None:
            cmd.extend(["--workers", str(shared["workers"])])
        if shared.get("hankel_embedding_depth") is not None:
            cmd.extend([
                "--hankel-embedding-depth",
                str(shared["hankel_embedding_depth"]),
            ])

        # Diffusion maps params (only when stage2 != diffusion_maps; with
        # diffusion_maps they live inside stage2_params already)
        if s2 != "diffusion_maps":
            for dflag, dkey in [
                ("--diffusion-sigma", "diffusion_sigma"),
                ("--diffusion-k", "diffusion_k"),
                ("--diffusion-time", "diffusion_time"),
                ("--diffusion-alpha", "diffusion_alpha"),
            ]:
                val = shared.get(dkey)
                if val is not None:
                    cmd.extend([dflag, str(val)])

        # --- Boolean flags ---
        if shared.get("ignore_cache", False) or self.latent_ignore_cache:
            cmd.append("--ignore-cache")
        if not shared.get("verbose", True):
            cmd.append("--no-verbose")

        # --- Global channel intersection (CD-HSA consistency) ---
        ch_inter = self._latent_ch_intersections.get(
            (job["session"], job["task"])
        )
        if ch_inter is not None:
            cmd.extend(["--channel-intersection", json.dumps(ch_inter)])

        # --- Optional db path (same style as the CD-HSA jobs) ---
        if self.db_path:
            cmd.extend(["--db-path", self.db_path])

        return cmd

    def _run_latent_single_job(self, job: dict) -> tuple[str, bool]:
        key = self._latent_checkpoint_key(job)

        try:
            cmd = self._build_latent_command(job)
        except Exception as exc:
            logger.error(
                "[LATENT] Error construyendo comando para %s/%s/%s [%s]: %s",
                job["super_subject_label"], job["session"], job["task"],
                job["method_label"], exc,
            )
            self._notify(
                "error",
                "Error construyendo el comando latente para %s/%s/%s:\n%s\n"
                "El batch continua con los demas jobs." % (
                    job["super_subject_label"], job["session"], job["task"],
                    exc,
                ),
                title="[CD-HSA] Fallo de job latente",
            )
            return key, False

        logger.info(
            "RUN-LATENT | %s/%s/%s | method=%s (%s) | [%s-%s] s",
            job["super_subject_label"], job["session"], job["task"],
            job["method_label"], job["spec_label"],
            job["t_start"], job["t_end"],
        )
        logger.debug("CMD: %s", " ".join(cmd))

        t0 = time.time()
        cmd_for_log = cmd
        try:
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=child_env,
            )

            # [MEM TRACKING] Poll child process RSS in a daemon thread
            _child_peak = [0.0]
            _child_samples: list[tuple[float, float]] = []

            def _poll_thread():
                pk, samps = self._poll_child_rss(proc)
                _child_peak[0] = pk
                _child_samples.extend(samps)

            _thr = threading.Thread(target=_poll_thread, daemon=True)
            _thr.start()

            for line in proc.stdout:
                logger.info("  [LATENT] %s", line.rstrip())

            proc.wait()
            _thr.join(timeout=3.0)

            elapsed = time.time() - t0
            success = proc.returncode == 0
            peak_mb = _child_peak[0]

            self._latent_job_memory_stats.append({
                "label": "%s/%s [%s]" % (
                    job["super_subject_label"], job["task"],
                    job["method_label"],
                ),
                "session": job["session"],
                "success": success,
                "elapsed_s": round(elapsed, 2),
                "peak_rss_mb": round(peak_mb, 1),
                "n_rss_samples": len(_child_samples),
            })

            self._write_latent_csv_log(
                job, success, proc.returncode, elapsed, peak_mb, cmd,
            )

            if peak_mb > 0:
                logger.info(
                    "  [LATENT][MEM] Peak child RSS: %.1f MB (%d samples)",
                    peak_mb, len(_child_samples),
                )

            if success:
                logger.info(
                    "OK-LATENT | %s/%s/%s [%s] (%.1f s, peak %.1f MB)",
                    job["super_subject_label"], job["session"], job["task"],
                    job["method_label"], elapsed, peak_mb,
                )
            else:
                logger.error(
                    "ERROR-LATENT | %s/%s/%s [%s] -- codigo %d",
                    job["super_subject_label"], job["session"], job["task"],
                    job["method_label"], proc.returncode,
                )
                self._notify(
                    "error",
                    "Job latente %s/%s/%s [%s] termino con codigo %d "
                    "(fallo).\nEl batch continua con los demas jobs." % (
                        job["super_subject_label"], job["session"],
                        job["task"], job["method_label"], proc.returncode,
                    ),
                    title="[CD-HSA] Fallo de job latente",
                )

            return key, success

        except Exception as exc:
            elapsed = time.time() - t0
            logger.error(
                "EXCEPTION-LATENT | %s/%s/%s [%s]: %s",
                job["super_subject_label"], job["session"], job["task"],
                job["method_label"], exc,
            )
            self._notify(
                "error",
                "Excepcion ejecutando el job latente %s/%s/%s:\n%s: %s\n"
                "El batch continua con los demas jobs." % (
                    job["super_subject_label"], job["session"], job["task"],
                    type(exc).__name__, exc,
                ),
                title="[CD-HSA] Fallo de job latente",
            )
            self._write_latent_csv_log(
                job, False, -1, elapsed, 0.0, cmd_for_log,
            )
            return key, False

    def _run_latent_sequential(self, todo: list[dict]) -> tuple[int, int]:
        completed = 0
        failed = 0
        total = len(todo)

        for idx, job in enumerate(todo, start=1):
            logger.info("")
            logger.info("-" * 70)
            logger.info("[LATENT] Progreso: %d / %d", idx, total)
            logger.info("-" * 70)

            key, success = self._run_latent_single_job(job)
            if success:
                self.latent_checkpoint.add(key)
                completed += 1
            else:
                failed += 1

            self._save_latent_checkpoint()

            if idx < total and self.latent_delay > 0:
                logger.debug("[LATENT] Pausa %.1f s...", self.latent_delay)
                time.sleep(self.latent_delay)

        return completed, failed

    def _run_latent_parallel(self, todo: list[dict]) -> tuple[int, int]:
        completed = 0
        failed = 0
        total = len(todo)

        logger.info(
            "[LATENT] Modo PARALELO con %d workers", self.latent_max_workers
        )

        with ProcessPoolExecutor(max_workers=self.latent_max_workers) as executor:
            future_to_job = {
                executor.submit(self._run_latent_single_job, job): job
                for job in todo
            }

            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    key, success = future.result()
                    if success:
                        self.latent_checkpoint.add(key)
                        completed += 1
                    else:
                        failed += 1
                except Exception as exc:
                    logger.error(
                        "[LATENT] FUTURE EXCEPTION | %s/%s/%s: %s",
                        job["super_subject_label"], job["session"],
                        job["task"], exc,
                    )
                    failed += 1

                self._save_latent_checkpoint()
                logger.info(
                    "[LATENT] Progreso: %d / %d completados",
                    completed + failed, total,
                )

        return completed, failed

    def _print_latent_memory_report(self) -> None:
        """Log a per-job peak-RSS table for the latent phase."""
        stats = self._latent_job_memory_stats
        if not stats:
            return

        sorted_stats = sorted(
            stats, key=lambda s: s["peak_rss_mb"], reverse=True,
        )

        logger.info("")
        logger.info("=" * 70)
        logger.info("  [LATENT] MEMORY REPORT POR JOB  (ordenado por peak RSS)")
        logger.info("=" * 70)

        hdr = "  %-55s %10s %8s %6s" % (
            "Job", "Peak RSS", "Elapsed", "Status")
        logger.info(hdr)
        logger.info("  " + "-" * 87)

        for s in sorted_stats:
            status = "OK" if s["success"] else "FAIL"
            logger.info(
                "  %-55s %8.1f MB %6.1f s   %s",
                '%s/%s' % (s["label"], s["session"]),
                s["peak_rss_mb"],
                s["elapsed_s"],
                status,
            )

        max_s = sorted_stats[0]
        logger.info("  " + "-" * 87)
        logger.info(
            "  [LATENT] Peak maximo global: %.1f MB (%s)",
            max_s.get("peak_rss_mb", 0),
            max_s.get("label", ""),
        )
        logger.info("=" * 70)

        # --- save CSV ---
        if self.output_dir:
            log_dir = self.output_dir / "batch_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = log_dir / ("latent_memory_per_job_%s.csv" % ts)
            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=[
                        "label", "session", "success", "elapsed_s",
                        "peak_rss_mb", "n_rss_samples",
                    ])
                    writer.writeheader()
                    for s in stats:
                        writer.writerow({
                            k: s[k] for k in
                            ["label", "session", "success",
                             "elapsed_s", "peak_rss_mb", "n_rss_samples"]
                        })
                logger.info("[LATENT][MEM] Per-job CSV guardado: %s", csv_path)
            except OSError as exc:
                logger.warning(
                    "[LATENT][MEM] Error guardando per-job CSV: %s", exc
                )

    def _run_latent_phase(self) -> dict:
        """Run the latent-space + plots phase (after mode extraction).

        Returns a summary dict with keys ``ran`` (bool), ``reason``
        (when not ran), ``ok`` / ``fail`` / ``skip`` counts and
        ``todo``.

        Failure semantics: a job failure does not stop the phase (same
        as the CD-HSA jobs); a *phase-level* blocker (per-SS mode,
        missing mode_map / npz) skips the phase with an urgent notice.
        """
        summary = {
            "ran": False, "reason": "", "ok": 0, "fail": 0, "skip": 0,
            "todo": 0,
        }

        logger.info("")
        logger.info("=" * 70)
        logger.info("  FASE LATENTE: ESPACIO LATENTE + PLOTS (sin KM)")
        logger.info("=" * 70)

        # --- Guard 1: multi-SS mode required -----------------------------
        is_multi_ss = "n_super_subjects" in self.ss_cfg
        if not is_multi_ss:
            reason = (
                "modo per-SS legacy: los mode_map de los distintos "
                "super-sujetos comparten nombre de fichero y no pueden "
                "desambiguarse; la fase latente solo esta soportada en "
                "modo multi-SS (n_super_subjects)."
            )
            logger.warning("[LATENT] Fase omitida: %s", reason)
            summary["reason"] = reason
            self._notify(
                "warning",
                "La fase latente fue omitida: %s" % reason,
                title="[CD-HSA] Fase latente omitida",
            )
            return summary

        # --- Guard 2: mode_map / npz must exist ---------------------------
        sources = self._resolve_latent_sources()
        missing = []
        for session, src in sources.items():
            if not src["mode_map_path"].exists():
                missing.append(str(src["mode_map_path"]))
            if not src["npz_path"].exists():
                missing.append(str(src["npz_path"]))
        if missing:
            reason = (
                "faltan las entradas del CD-HSA para la fase latente "
                "(%d): %s%s. La extraccion de modos / el pipeline CD-HSA "
                "no produjeron salidas para esta configuracion." % (
                    len(missing),
                    "; ".join(Path(m).name for m in missing[:3]),
                    ", ..." if len(missing) > 3 else "",
                )
            )
            logger.error("[LATENT] Fase omitida: %s", reason)
            summary["reason"] = reason
            self._notify(
                "error",
                "La fase latente fue omitida: %s" % reason,
                title="[CD-HSA] Fase latente bloqueada",
            )
            return summary

        # --- Pre-compute channel intersections (if needed) ----------------
        self._precompute_latent_channel_intersections()

        # --- Filter by checkpoint -----------------------------------------
        todo = []
        for job in self.latent_jobs:
            key = self._latent_checkpoint_key(job)
            if key in self.latent_checkpoint:
                logger.debug(
                    "[LATENT] SKIP (checkpoint): %s/%s/%s [%s]",
                    job["super_subject_label"], job["session"], job["task"],
                    job["method_label"],
                )
                continue
            todo.append(job)

        summary["todo"] = len(todo)
        if skipped := len(self.latent_jobs) - len(todo):
            logger.info(
                "[LATENT] Jobs ya completados (skip): %d / %d",
                skipped, len(self.latent_jobs),
            )

        if not todo:
            logger.info(
                "[LATENT] Todos los jobs latentes ya estan completados."
            )
            summary["ran"] = True
            self._print_latent_memory_report()
            return summary

        logger.info("[LATENT] Total a ejecutar: %d / %d",
                    len(todo), len(self.latent_jobs))

        self._notify(
            "info",
            "Fase latente iniciada: %d job(s) por ejecutar (%d en total).\n"
            "Pipeline: %s\nHost: %s | workers: %d" % (
                len(todo), len(self.latent_jobs),
                self.latent_pipeline_module,
                _HOST, self.latent_max_workers,
            ),
            title="[CD-HSA] Fase latente iniciada",
        )

        if self.latent_max_workers > 1:
            ok, fail = self._run_latent_parallel(todo)
        else:
            ok, fail = self._run_latent_sequential(todo)

        self._save_latent_checkpoint()

        summary["ran"] = True
        summary["ok"] = ok
        summary["fail"] = fail

        logger.info("=" * 70)
        logger.info(
            "[LATENT] FASE LATENTE COMPLETADA -- OK: %d | Fallos: %d | "
            "Total: %d", ok, fail, len(todo),
        )
        logger.info("[LATENT] Log CSV: %s", self.latent_log_file)

        self._print_latent_memory_report()

        return summary

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------

    def run(self) -> int:
        # [MEM TRACKING] Start global memory monitor for the orchestrator
        _monitor = None
        if _HAS_MEM_TRACKER:
            _monitor = get_global_monitor(
                logger=logger, interval_sec=0.5,
                spike_threshold_mb=100,
            )
            _monitor.start()

        try:
            return self._run_inner()
        finally:
            # [MEM TRACKING] Stop monitor, generate final reports
            if _monitor is not None and _monitor.is_running:
                _monitor.stop()
                _monitor.report()
                if _monitor._checkpoints:
                    _monitor.summary()

                # Save timeline CSV
                if self.output_dir:
                    log_dir = self.output_dir / "batch_logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    tl_path = log_dir / ("memory_timeline_%s.csv" % ts)
                    _monitor.save_timeline_csv(str(tl_path))
                    logger.info("[MEM] Timeline CSV: %s", tl_path)

                # Top Python allocations (tracemalloc)
                try:
                    _monitor.top_allocations(10)
                except Exception:
                    pass

            # Per-job peak-RSS summary table + CSV
            self._print_memory_report()

    def _run_inner(self) -> int:
        """Original run() logic, extracted so run() can wrap with monitoring."""
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.all_jobs:
            logger.error("No se generaron jobs. Revisa el JSON de parametros.")
            return 1

        todo = self._filter_todo(self.all_jobs)
        total = len(todo)

        # [NTFY] Startup notice: doubles as a canary (if it does not reach
        # the phone, the channel is misconfigured and the run is pointless).
        if total > 0:
            tw = self.params["time_window"]
            self._notify(
                "info",
                "Batch iniciado: %d job(s) por ejecutar (%d en total).\n"
                "Label: %s\nVentana: %s - %s s | tasks: %d\n"
                "Host: %s | workers: %d\nSalidas: %s" % (
                    total, len(self.all_jobs),
                    self.params.get("experiment_label", "batch_cdhsa"),
                    tw["t_start"], tw["t_end"], len(self.params["tasks"]),
                    _HOST, self.max_workers, self.output_dir,
                ),
                title="[CD-HSA] Batch iniciado",
            )

        _batch_t0 = time.time()

        if total == 0:
            logger.info("Todos los jobs ya estan completados.")
            n_ok_x, n_skip_x, n_fail_x = self._run_mode_extraction()
            if self.run_comparison:
                self._run_comparison()

            # [LATENT] Phase runs even when the CD-HSA jobs were skipped
            # by the checkpoint: the mode extraction above regenerated
            # mode_map.json from the results on disk, so the latent jobs
            # can proceed (e.g. re-running the batch just for the latent
            # phase after a CD-HSA run completed earlier).
            latent_res = None
            if self.run_latent_phase:
                latent_res = self._run_latent_phase()

            # [LATENT] rc: la rama "ya completo" ahora hace trabajo real
            # (la fase latente); su fallo debe reflejarse en el exit code.
            _latent_fail_0 = (
                latent_res["fail"]
                if latent_res is not None and latent_res["ran"]
                else (1 if latent_res is not None else 0)
            )
            self._notify(
                "info" if (n_fail_x == 0 and _latent_fail_0 == 0) else "warning",
                "No habia jobs pendientes (checkpoint): solo se ejecuto el "
                "post-procesamiento (extraccion de modos + comparacion%s).\n"
                "Extraccion de modos: OK=%d, Skip=%d, Fail=%d%s" % (
                    " + fase latente" if latent_res is not None else "",
                    n_ok_x, n_skip_x, n_fail_x,
                    ("\nFase latente: OK=%d | Fallos=%d" % (
                        latent_res["ok"], latent_res["fail"],
                    )) if latent_res is not None and latent_res["ran"] else
                    ("\nFase latente: no ejecutada (%s)" % latent_res["reason"])
                    if latent_res is not None else "",
                ),
                title="[CD-HSA] Batch ya completo",
            )
            return 0 if _latent_fail_0 == 0 else 1

        logger.info("Total a ejecutar: %d / %d", total, len(self.all_jobs))

        completed = 0
        failed = 0

        if self.max_workers > 1:
            completed, failed = self._run_parallel(todo, total)
        else:
            completed, failed = self._run_sequential(todo, total)

        self._save_checkpoint()

        logger.info("=" * 70)
        logger.info(
            "BATCH COMPLETADO -- OK: %d | Fallos: %d | Total: %d",
            completed, failed, total,
        )
        logger.info("Log CSV: %s", self.log_file)

        n_ok_x, n_skip_x, n_fail_x = self._run_mode_extraction()

        if self.run_comparison:
            self._run_comparison()

        # [LATENT] Latent-space + plots phase (after the CD-HSA outputs
        # and the mode_map are on disk; requires multi-SS mode).
        latent_res = None
        if self.run_latent_phase:
            latent_res = self._run_latent_phase()

        # [NTFY] Final notice (after post-processing): success only when
        # not a single job failed AND the mode extraction AND the latent
        # phase succeeded.
        _elapsed = time.strftime(
            "%Hh %Mm %Ss", time.gmtime(time.time() - _batch_t0),
        )
        _latent_fail = (
            latent_res["fail"] if latent_res is not None and latent_res["ran"]
            else (1 if latent_res is not None else 0)
        )
        _latent_summary = ""
        if latent_res is not None:
            if latent_res["ran"]:
                _latent_summary = (
                    "\nFase latente: OK=%d | Fallos=%d | Total=%d" % (
                        latent_res["ok"], latent_res["fail"],
                        latent_res["todo"],
                    )
                )
            else:
                _latent_summary = (
                    "\nFase latente: NO EJECUTADA (%s)" % latent_res["reason"]
                )
        _resumen = (
            "OK: %d | Fallos: %d | Total: %d\n"
            "Extraccion de modos: OK=%d, Skip=%d, Fail=%d%s\n"
            "Duracion: %s\nLabel: %s\n"
            "Host: %s\nSalidas: %s" % (
                completed, failed, total,
                n_ok_x, n_skip_x, n_fail_x,
                _latent_summary,
                _elapsed,
                self.params.get("experiment_label", "batch_cdhsa"),
                _HOST, self.output_dir,
            )
        )
        if failed == 0 and n_fail_x == 0 and _latent_fail == 0:
            self._notify("success", _resumen, title="[CD-HSA] Batch terminado OK")
        else:
            self._notify(
                "warning",
                _resumen + "\nLog CSV: %s" % self.log_file,
                title="[CD-HSA] Batch terminado CON FALLOS",
            )

        return 0 if (failed == 0 and n_fail_x == 0 and _latent_fail == 0) else 1

    def _run_sequential(self, todo: list[dict], total: int) -> tuple[int, int]:
        completed = 0
        failed = 0

        for idx, job in enumerate(todo, start=1):
            logger.info("")
            logger.info("-" * 70)
            logger.info("Progreso: %d / %d", idx, total)
            logger.info("-" * 70)

            key, success = self._run_single_job(job)
            if success:
                self.checkpoint.add(key)
                completed += 1
            else:
                failed += 1

            self._save_checkpoint()

            if idx < total and self.delay > 0:
                logger.debug("Pausa %.1f s...", self.delay)
                time.sleep(self.delay)

        return completed, failed

    def _run_parallel(self, todo: list[dict], total: int) -> tuple[int, int]:
        completed = 0
        failed = 0

        logger.info("Modo PARALELO con %d workers", self.max_workers)

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_job = {
                executor.submit(self._run_single_job, job): job for job in todo
            }

            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    key, success = future.result()
                    if success:
                        self.checkpoint.add(key)
                        completed += 1
                    else:
                        failed += 1
                except Exception as exc:
                    logger.error(
                        "FUTURE EXCEPTION | %s/%s: %s",
                        job["super_subject_label"], job["session"], exc,
                    )
                    failed += 1

                self._save_checkpoint()
                logger.info(
                    "Progreso: %d / %d completados", completed + failed, total
                )

        return completed, failed


# ===========================================================================
# ENTRY POINT
# ===========================================================================


@notify_on_critical_error(
    channel=_NTFY_CHANNEL or None,
    title="[CD-HSA] ERROR critico del batch",
    catch_system_exit=True,
)
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch runner para el pipeline CD-HSA. "
            "Lee la configuracion desde un JSON."
        ),
    )
    parser.add_argument(
        "--params-json", type=str, default=None,
        help=(
            "Ruta al archivo JSON de parametros. "
            f"Default: {DEFAULT_PARAMS_JSON}"
        ),
    )
    parser.add_argument(
        "--pipeline-script", type=str, default=None,
        help=(
            "Ruta al script run_cdhsa.py. "
            "Default: run_cdhsa.py en el mismo directorio que este script."
        ),
    )
    args = parser.parse_args()

    json_path = Path(args.params_json) if args.params_json else DEFAULT_PARAMS_JSON
    if os.environ.get("BATCH_CDHSA_PARAMS_JSON"):
        json_path = Path(os.environ["BATCH_CDHSA_PARAMS_JSON"])

    pipeline_script = (
        Path(args.pipeline_script) if args.pipeline_script else None
    )

    params = _load_params(json_path)
    logger.info("Parametros cargados desde: %s", json_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner = CDHSABatchRunner(params, pipeline_script=pipeline_script)
        return runner.run()


if __name__ == "__main__":
    sys.exit(main())
