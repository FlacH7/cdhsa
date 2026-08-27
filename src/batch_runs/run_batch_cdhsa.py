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
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
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

    DEFAULT_PIPELINE_MODULE = "src.pipelines.run_cdhsa_v2"

    CSV_FIELDS = [
        "timestamp", "mode", "super_subject", "session", "tasks",
        "L", "fixed_rank", "hankel_depth",
        "t_start", "t_end", "success", "returncode",
        "elapsed_s", "command",
    ]

    def __init__(self, params: dict, *, pipeline_script: Path | None = None,
                 pipeline_module: str | None = None) -> None:
        self.params = params
        self.pipeline_script = pipeline_script or (_SCRIPT_DIR.parent / "pipelines" / "run_cdhsa_v2.py")
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
        logger.info("    a6_n_null      : %d", cdhsa.get("a6_n_null", 100))
        logger.info("    bc_n_perm      : %d", cdhsa.get("bc_n_perm", 5000))
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
        """Build a unique checkpoint key from a job dict."""
        tasks_str = "+".join(job["tasks"])
        if job.get("mode") == "multi_ss":
            return (f"multiss_{job['n_super_subjects']}|{job['session']}|"
                    f"{tasks_str}|{job['t_start']}|{job['t_end']}")
        else:
            sid = job["super_subject_id"]
            return (f"ss{sid:02d}|{job['session']}|{tasks_str}|"
                    f"{job['t_start']}|{job['t_end']}")

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
            "--a6-n-null", str(cdhsa.get("a6_n_null", 100)),
            "--bc-n-perm", str(cdhsa.get("bc_n_perm", 5000)),
            "--d-max-specific", str(cdhsa.get("d_max_specific", 10)),
        ])

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
        fr = cdhsa.get("fixed_rank", 10)
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

        out_dir = Path(
            f"{self.output_dir}/cdhsa/{job['session']}"
            f"/{ss_label}_L{L}"
            f"_fr{fr}_a6n{a6n}_bcn{bcn}"
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
                ss_label = job["super_subject_label"]
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

        if total == 0:
            logger.info("Todos los jobs ya estan completados.")
            self._run_mode_extraction()
            if self.run_comparison:
                self._run_comparison()
            return 0

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

        self._run_mode_extraction()

        if self.run_comparison:
            self._run_comparison()

        return 0 if failed == 0 else 1

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
