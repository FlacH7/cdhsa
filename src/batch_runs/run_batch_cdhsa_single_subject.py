#!/usr/bin/env python3
"""
run_batch_cdhsa_single_subject.py
==================================
Batch executor for the **CD-HSA single-subject** pipeline.

Single execution mode::

  One job per session.  ALL individual subjects are passed together to
  a single invocation of ``run_cdhsa_single_subject.py`` (S=n_subjects,
  C=tasks).  CD-HSA finds common directions across subjects AND
  condition-specific modes with full cross-subject statistical support
  (permutation tests, prevalence, A6 rank).

  There is no per-SS mode here (each subject IS its own sample).

Configuration
--------------
Everything is controlled by the JSON file (default:
``./cdhsa_batch_params_single_subject.json``).

JSON example::

    {
      "experiment_label": "singlesubject_cdhsa_60sub_2tasks",
      "subjects": {
          "n_subjects": 60,
          "subject_start_offset": 1
      },
      "sessions": ["session1"],
      "tasks": ["eyesclosed", "music"],
      "time_window": { "t_start": 100.0, "t_end": 200.0 },
      "cdhsa_params": {
          "L": 10, "hankel_depth": 10,
          "l_freq": 1.0, "h_freq": 40.0,
          "fixed_rank": 20, "rank_method": "fixed",
          "a6_n_null": 500, "bc_n_perm": 5000,
          "skip_bc": false, "skip_tangent": false, "skip_d": false,
          "d_max_specific": 4
      },
      "execution": {
          "max_workers": 1, "delay": 0.0,
          "run_comparison": false, "mode_extract_top_n": 4
      }
    }

  Alternative ``subjects`` block: ``{"subject_ids": [1, 2, ..., 60]}``
  for an explicit subject list.

Usage
-----
From the repository root (with ``.env`` configured)::

    # Defaults (reads cdhsa_batch_params_single_subject.json next to this script)
    python src/batch_runs/run_batch_cdhsa_single_subject.py

    # Custom JSON
    python src/batch_runs/run_batch_cdhsa_single_subject.py --params-json /path/to/params.json

    # Override via environment variable
    BATCH_CDHSA_SINGLE_SUBJECT_PARAMS_JSON=/path/to/params.json \\
        python src/batch_runs/run_batch_cdhsa_single_subject.py
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
# Ensure the directory containing the pipeline scripts is importable
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
logger = logging.getLogger("batch_cdhsa_single_subject")

# ---------------------------------------------------------------------------
# Default JSON path
# ---------------------------------------------------------------------------
DEFAULT_PARAMS_JSON = _SCRIPT_DIR / "cdhsa_batch_params_single_subject.json"

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
    required_top_keys = ["subjects", "sessions", "tasks", "cdhsa_params"]
    for key in required_top_keys:
        if key not in params:
            logger.error("Falta la clave requerida '%s' en el JSON", key)
            sys.exit(1)

    subj_cfg = params["subjects"]

    # The subjects block must declare either an explicit list or a count
    has_ids = "subject_ids" in subj_cfg
    has_n = "n_subjects" in subj_cfg

    if not has_ids and not has_n:
        logger.error(
            "El bloque 'subjects' debe contener 'n_subjects' "
            "(rango automatico) o 'subject_ids' (lista explicita)."
        )
        sys.exit(1)

    if has_ids:
        if not isinstance(subj_cfg["subject_ids"], list) or not subj_cfg["subject_ids"]:
            logger.error("'subjects.subject_ids' debe ser una lista no vacia.")
            sys.exit(1)
    else:
        if not isinstance(subj_cfg["n_subjects"], int) or subj_cfg["n_subjects"] <= 0:
            logger.error("'subjects.n_subjects' debe ser un entero positivo.")
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


def _resolve_subject_ids(subj_cfg: dict) -> list[int]:
    """Resolve the explicit subject-id list from the ``subjects`` block.

    Returns
    -------
    list[int]
        * ``subject_ids`` as-is when the JSON declares it;
        * otherwise ``range(offset, offset + n_subjects)``.
    """
    if "subject_ids" in subj_cfg:
        return [int(x) for x in subj_cfg["subject_ids"]]
    offset = int(subj_cfg.get("subject_start_offset", 1))
    n = int(subj_cfg["n_subjects"])
    return list(range(offset, offset + n))


def _subjects_label(job: dict) -> str:
    """Short label for the job's subject pool, e.g. ``nSub60`` or
    ``subj_1-60``."""
    ids = job["subject_ids"]
    n = job["n_subjects"]
    if ids == list(range(ids[0], ids[0] + n)):
        return f"nSub{n}"
    return f"subj_{ids[0]}-{ids[-1]}"


# ===========================================================================
# JOB GENERATION
# ===========================================================================


def _generate_jobs(params: dict) -> list[dict]:
    """Generate the list of jobs from the JSON parameters.

    Single mode (``single_subject``): one job per session.  All
    individual subjects are passed together to a single invocation of
    ``run_cdhsa_single_subject.py`` (S=n_subjects, C=tasks).
    """
    subj_cfg = params["subjects"]
    sessions = params["sessions"]
    tasks = params["tasks"]
    tw = params.get("time_window", {})
    cdhsa = params["cdhsa_params"]
    t_start = str(tw.get("t_start"))
    t_end = str(tw.get("t_end"))

    subject_ids = _resolve_subject_ids(subj_cfg)

    jobs: list[dict] = []
    for session in sessions:
        jobs.append({
            "mode": "single_subject",
            "subject_ids": subject_ids,
            "n_subjects": len(subject_ids),
            "session": session,
            "tasks": tasks,
            "cdhsa_params": cdhsa,
            "t_start": t_start,
            "t_end": t_end,
        })
    return jobs


# ===========================================================================
# BATCH RUNNER
# ===========================================================================


class SingleSubjectCDHSABatchRunner:
    """Orchestrates batch execution of the single-subject CD-HSA pipeline."""

    DEFAULT_PIPELINE_MODULE = "src.pipelines.run_cdhsa_single_subject"

    CSV_FIELDS = [
        "timestamp", "mode", "subjects", "session", "tasks",
        "L", "fixed_rank", "hankel_depth",
        "t_start", "t_end", "success", "returncode",
        "elapsed_s", "command",
    ]

    def __init__(self, params: dict, *, pipeline_script: Path | None = None,
                 pipeline_module: str | None = None) -> None:
        self.params = params
        self.pipeline_script = pipeline_script or (
            _SCRIPT_DIR.parent / "pipelines" / "run_cdhsa_single_subject.py")
        self.pipeline_module = pipeline_module or self.DEFAULT_PIPELINE_MODULE
        self.subj_cfg = params["subjects"]
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
        label = self.params.get("experiment_label", "batch_cdhsa_single_subject")
        return log_dir / f"batch_cdhsa_single_subject_{label}_{ts}.csv"

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        n_sessions = len({j["session"] for j in self.all_jobs})
        n_tasks = len(self.params["tasks"])
        cdhsa = self.params["cdhsa_params"]
        tw = self.params.get("time_window", {})

        first_job = self.all_jobs[0] if self.all_jobs else None
        ids = first_job["subject_ids"] if first_job else []
        n_subjects = first_job["n_subjects"] if first_job else 0

        logger.info("=" * 70)
        logger.info("  BATCH CD-HSA -- MODO SINGLE-SUBJECT (S=%d, C=%d)",
                    n_subjects, n_tasks)
        logger.info("=" * 70)
        logger.info("  Pipeline script : %s", self.pipeline_script)
        logger.info("  DB path         : %s", self.db_path or "(default)")
        logger.info("  Output dir      : %s", self.output_dir)
        logger.info("  Cache dir       : %s", self.cache_dir)
        logger.info("  Checkpoint      : %s", self._checkpoint_path())
        logger.info("  ---")
        logger.info("  Subjects (S)    : %d", n_subjects)
        if ids:
            logger.info("  Subject range   : %d..%d", ids[0], ids[-1])
        logger.info("  Sessions        : %s", self.params["sessions"])
        logger.info("  Tasks (C)       : %s", self.params["tasks"])
        if tw:
            logger.info("  Time window     : %.1f s -> %.1f s (per subject)",
                        tw.get("t_start", 0.0), tw.get("t_end", 0.0))
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
        logger.info("  Total jobs      : %d (%d sessions)", len(self.all_jobs), n_sessions)
        logger.info("  Delay           : %.1f s", self.delay)
        logger.info("  Max workers     : %d (%s)",
                    self.max_workers,
                    "paralelo" if self.max_workers > 1 else "secuencial")
        logger.info("  Comparison      : %s (NO APLICA en modo single-subject)",
                    self.run_comparison)
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _checkpoint_path(self) -> Path:
        if self.cache_dir:
            return self.cache_dir / "batch_checkpoint_cdhsa_single_subject.json"
        return Path("./cache") / "batch_checkpoint_cdhsa_single_subject.json"

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
        return (f"singlesub_n{job['n_subjects']}|{job['session']}|"
                f"{tasks_str}|{job['t_start']}|{job['t_end']}")

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
                    "mode": job.get("mode", "single_subject"),
                    "subjects": _subjects_label(job),
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
                logger.debug(
                    "SKIP (checkpoint): %s/%s",
                    _subjects_label(job), job["session"],
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
        """Build the subprocess command for a single-subject CD-HSA job.

        Always uses ``--subject-ids i1 i2 ...`` (S=n_subjects, all
        individual subjects together).
        """
        cdhsa = job["cdhsa_params"]

        cmd = [
            sys.executable, "-m", self.pipeline_module,
            "--session", job["session"],
        ]

        # Subjects (explicit list, resolved from the JSON)
        cmd.extend(["--subject-ids"] + [str(i) for i in job["subject_ids"]])

        # Tasks (all in a single --tasks invocation, since the pipeline
        # uses nargs='+')
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
    # Output dir for a job (mirrors run_cdhsa_single_subject.py _resolve_out_dir)
    # ------------------------------------------------------------------

    def _get_output_dir(self, job: dict) -> Path:
        """Return the directory where the pipeline saves results.

        Mirrors the path scheme in ``run_cdhsa_single_subject.py``'s
        ``_resolve_out_dir()``, using the ``nSub{N}`` label.
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

        subj_label = f"nSub{job['n_subjects']}"

        out_dir = Path(
            f"{self.output_dir}/cdhsa/{job['session']}"
            f"/{subj_label}_L{L}"
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
            logger.error(
                "Error construyendo comando para %s/%s: %s",
                _subjects_label(job), job["session"], exc,
            )
            return key, False

        label = _subjects_label(job)
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
                f"mode_map_{top_n}_modes_singlesub_n{job['n_subjects']}"
                f"_{job['session']}_{tasks_tag}_{tw_tag}.json"
            )
            json_out = self.params_dir / json_name

            # Build command to run the extraction script as a module
            cmd = [
                sys.executable, "-m", "src.cdhsa.extract_mode_indices",
                "--results-dir", str(out_dir),
                "--top-n", str(top_n),
                "-o", str(json_out),
            ]

            label = _subjects_label(job)
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
                logger.info(
                    "run_comparison=true: NO APLICA en modo single-subject "
                    "(un solo pool de sujetos por sesion; nada que comparar)."
                )
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
            logger.info(
                "run_comparison=true: NO APLICA en modo single-subject "
                "(un solo pool de sujetos por sesion; nada que comparar)."
            )

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
                        _subjects_label(job), job["session"], exc,
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
            "Batch runner para el pipeline CD-HSA single-subject. "
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
            "Ruta al script run_cdhsa_single_subject.py. "
            "Default: run_cdhsa_single_subject.py en src/pipelines."
        ),
    )
    args = parser.parse_args()

    json_path = Path(args.params_json) if args.params_json else DEFAULT_PARAMS_JSON
    if os.environ.get("BATCH_CDHSA_SINGLE_SUBJECT_PARAMS_JSON"):
        json_path = Path(os.environ["BATCH_CDHSA_SINGLE_SUBJECT_PARAMS_JSON"])

    pipeline_script = (
        Path(args.pipeline_script) if args.pipeline_script else None
    )

    params = _load_params(json_path)
    logger.info("Parametros cargados desde: %s", json_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner = SingleSubjectCDHSABatchRunner(params, pipeline_script=pipeline_script)
        return runner.run()


if __name__ == "__main__":
    sys.exit(main())
