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

Classification experiment (v6)
------------------------------

  If the JSON declares a **train/test split** (``subjects.train`` +
  ``subjects.test``) and a ``classification`` block, the batch runs TWO
  phases per session:

    1. **Train**: CD-HSA on the TRAIN subjects only (e.g. 1..40) with
       the v5.1 parameter set (variance rank, adaptive Step D, LOSO).
    2. **Classification (hold-out test)**: every recording of the TEST
       subjects (e.g. 41..60, both tasks) is classified independently
       by projecting its level-2 Hankel onto the condition-specific
       subspaces learned from the train subjects.  Produces accuracy
       (total and per task), confusion matrix, exact binomial p-value,
       paired permutation p-value, AUC and ~10 paper-ready figures.

  See ``src/cdhsa/classification.py`` for the mathematical framework
  and EXPERIMENTO_CLASIFICACION.md (paquete del experimento) for the
  full derivation.

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

    # --- Esquema v6: experimento de clasificacion (train/test) ---
    if ("train" in subj_cfg) or ("test" in subj_cfg):
        for sub in ("train", "test"):
            blk = subj_cfg.get(sub)
            if not isinstance(blk, dict):
                logger.error(
                    "subjects.%s debe ser un objeto con 'n_subjects' "
                    "(rango automatico) o 'subject_ids' (lista).", sub)
                sys.exit(1)
            if ("subject_ids" not in blk) and ("n_subjects" not in blk):
                logger.error(
                    "subjects.%s debe contener 'n_subjects' o "
                    "'subject_ids'.", sub)
                sys.exit(1)
            if "subject_ids" in blk and (
                    not isinstance(blk["subject_ids"], list)
                    or not blk["subject_ids"]):
                logger.error(
                    "subjects.%s.subject_ids debe ser una lista no vacia.",
                    sub)
                sys.exit(1)
            if "subject_ids" not in blk and (
                    not isinstance(blk.get("n_subjects"), int)
                    or blk["n_subjects"] <= 0):
                logger.error(
                    "subjects.%s.n_subjects debe ser un entero positivo.",
                    sub)
                sys.exit(1)

        clf = params.get("classification", {})
        if not isinstance(clf, dict):
            logger.error("'classification' debe ser un objeto.")
            sys.exit(1)
        if clf.get("enabled", False):
            if "test" not in subj_cfg:
                logger.error(
                    "classification.enabled=true requiere el bloque "
                    "subjects.test.")
                sys.exit(1)
            method = clf.get("method", "energy")
            if method not in ("energy", "ml", "both"):
                logger.error(
                    "classification.method debe ser 'energy' | 'ml' | "
                    "'both'; se recibio %r.", method)
                sys.exit(1)
            if len(params["tasks"]) != 2:
                logger.error(
                    "El experimento de clasificacion es binario: "
                    "'tasks' debe tener exactamente 2 tareas (se "
                    "recibieron %d).", len(params["tasks"]))
                sys.exit(1)
        return _validate_cdhsa_params(params)

    # --- Esquema legado: pool unico de sujetos ---
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

    # [v6] Guarda: 'classification' solo tiene sentido con el esquema
    # train/test. Con el pool unico (legado), enabled=true seria un error
    # de configuracion silenciosamente ignorado -> abortar en su lugar.
    clf = params.get("classification", {})
    if not isinstance(clf, dict):
        logger.error("'classification' debe ser un objeto.")
        sys.exit(1)
    if clf.get("enabled", False):
        logger.error(
            "classification.enabled=true requiere el esquema v6 con "
            "subjects.train y subjects.test (el bloque 'subjects' actual "
            "declara un unico pool).")
        sys.exit(1)

    return _validate_cdhsa_params(params)


def _validate_cdhsa_params(params: dict) -> dict:
    """Validacion comun del bloque cdhsa_params (esquemas legado y v6)."""
    cdhsa = params["cdhsa_params"]
    if "L" not in cdhsa:
        logger.error("'cdhsa_params' debe contener 'L' (subspace dimension).")
        sys.exit(1)
    rank_method = cdhsa.get("rank_method", "fixed")
    if rank_method not in ("fixed", "reproducibility", "variance"):
        logger.error(
            "cdhsa_params.rank_method invalido: %r (fixed | "
            "reproducibility | variance).", rank_method)
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


def _resolve_split_subjects(params: dict) -> tuple[list[int], list[int] | None]:
    """Resolver el split train/test del esquema v6 (con guarda de fuga).

    Returns
    -------
    (train_ids, test_ids | None)
    """
    subj_cfg = params["subjects"]
    if "train" not in subj_cfg and "test" not in subj_cfg:
        return _resolve_subject_ids(subj_cfg), None
    if "train" not in subj_cfg:
        logger.error("Esquema v6: falta subjects.train.")
        sys.exit(1)
    train_ids = _resolve_subject_ids(subj_cfg["train"])
    test_ids = (_resolve_subject_ids(subj_cfg["test"])
                if "test" in subj_cfg else None)
    if test_ids is not None:
        overlap = sorted(set(train_ids) & set(test_ids))
        if overlap:
            logger.error(
                "FUGA DE DATOS: los sujetos %s aparecen en train Y test.",
                overlap)
            sys.exit(1)
    return train_ids, test_ids


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

    Con el esquema v6 (train/test), los jobs usan el pool TRAIN; la
    fase de clasificacion sobre el pool TEST se orquesta aparte
    (``_run_classification_phase``).
    """
    sessions = params["sessions"]
    tasks = params["tasks"]
    tw = params.get("time_window", {})
    cdhsa = params["cdhsa_params"]
    t_start = str(tw.get("t_start"))
    t_end = str(tw.get("t_end"))

    subject_ids, _test_ids = _resolve_split_subjects(params)

    jobs: list[dict] = []
    for session in sessions:
        jobs.append({
            "mode": "single_subject",
            "phase": "train",
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

    CSV_FIELDS = [
        "timestamp", "mode", "phase", "subjects", "session", "tasks",
        "L", "fixed_rank", "hankel_depth",
        "t_start", "t_end", "success", "returncode",
        "elapsed_s", "command",
    ]

    DEFAULT_PIPELINE_MODULE = "src.pipelines.run_cdhsa_single_subject"
    DEFAULT_CLASSIFICATION_MODULE = "src.pipelines.run_cdhsa_task_classification"

    def __init__(self, params: dict, *, pipeline_script: Path | None = None,
                 pipeline_module: str | None = None,
                 classification_module: str | None = None) -> None:
        self.params = params
        self.pipeline_script = pipeline_script or (
            _SCRIPT_DIR.parent / "pipelines" / "run_cdhsa_single_subject.py")
        self.pipeline_module = pipeline_module or self.DEFAULT_PIPELINE_MODULE
        self.classification_module = (
            classification_module or self.DEFAULT_CLASSIFICATION_MODULE)
        self.subj_cfg = params["subjects"]
        self.exec_cfg = params.get("execution", {})

        # Execution settings
        self.delay: float = self.exec_cfg.get("delay", 2.0)
        self.max_workers: int = self.exec_cfg.get("max_workers", 1)
        self.run_comparison: bool = self.exec_cfg.get("run_comparison", True)

        # [v6] Experimento de clasificacion (hold-out)
        self.classification_cfg: dict = params.get("classification", {})
        _train_ids, self.test_subject_ids = _resolve_split_subjects(params)
        self.classification_enabled = (
            bool(self.classification_cfg.get("enabled", False))
            and self.test_subject_ids is not None)

        # Optional paths (may be None if not configured in the project)
        self.db_path: str | None = None
        self.output_dir: Path | None = None
        self.cache_dir: Path | None = None
        self.ntfy_channel: str | None = None
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

        # [v6] Resumenes de la fase de clasificacion (para log/ntfy)
        self._classification_summaries: list[str] = []

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

        # [NTFY] Canal de notificaciones (misma cadena que run_batch_cdhsa)
        try:
            from src.utils.config import NTFY_CHANNEL
            self.ntfy_channel = (NTFY_CHANNEL or "").strip() or None
        except Exception:
            self.ntfy_channel = None
        if self.ntfy_channel:
            logger.info(
                "  NTFY channel    : %s (notificaciones push activas)",
                self.ntfy_channel,
            )

    # ------------------------------------------------------------------
    # [NTFY] Notificaciones
    # ------------------------------------------------------------------

    def _notify(self, level: str, message: str, title: str | None = None) -> None:
        """Send a push notification through ntfy (best-effort, never raises).

        ``level`` is one of ``info | success | warning | error``. Without a
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
        if self.test_subject_ids is not None:
            logger.info("  [v6] SPLIT TRAIN/TEST:")
            logger.info("    Train         : %d sujetos (%d..%d)",
                        n_subjects, ids[0], ids[-1])
            logger.info("    Test          : %d sujetos (%d..%d)",
                        len(self.test_subject_ids),
                        self.test_subject_ids[0], self.test_subject_ids[-1])
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
        logger.info("    rank_method    : %s", cdhsa.get("rank_method", "fixed"))
        if cdhsa.get("rank_method") == "variance":
            logger.info("    var_explained  : %.2f (cap=%d, piso=%d)",
                        cdhsa.get("var_explained", 0.99),
                        cdhsa.get("var_max_rank", 50),
                        cdhsa.get("var_min_rank", 1))
        else:
            logger.info("    fixed_rank     : %d", cdhsa.get("fixed_rank", 10))
        logger.info("    max_common     : %d", cdhsa.get("max_common", 30))
        logger.info("    a6_n_null      : %d (tipo %s)",
                    cdhsa.get("a6_n_null", 100),
                    cdhsa.get("a6_null_type", "haar"))
        logger.info("    bc_n_perm      : %d", cdhsa.get("bc_n_perm", 5000))
        logger.info("    d_max_specific : %d (adaptive=%s, loso=%s)",
                    cdhsa.get("d_max_specific", 10),
                    cdhsa.get("d_rank_adaptive", False),
                    cdhsa.get("d_loso", False))
        logger.info("    replica        : %s",
                    cdhsa.get("run_replica_consistency", False))
        logger.info("  ---")
        if self.classification_enabled:
            clf = self.classification_cfg
            logger.info("  [v6] CLASIFICACION (hold-out): ACTIVA")
            logger.info("    metodo         : %s", clf.get("method", "energy"))
            logger.info("    rank_truncation : %s",
                        clf.get("rank_truncation", "min"))
            logger.info("    n_perm         : %d | seed %d",
                        clf.get("n_perm", 10000), clf.get("seed", 20261006))
            logger.info("    plots          : %s",
                        clf.get("make_plots", True))
        else:
            logger.info("  [v6] Clasificacion: inactiva")
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
    def _rank_tag(cdhsa: dict) -> str:
        """Etiqueta de rango para rutas/checkpoints (sincronizada con
        ``run_cdhsa._rank_tag`` y con ``run_cdhsa_single_subject``)."""
        if cdhsa.get("rank_method") == "variance":
            return (f"var{round(float(cdhsa.get('var_explained', 0.99)) * 100)}"
                    f"c{int(cdhsa.get('var_max_rank', 50))}")
        if cdhsa.get("rank_method") == "reproducibility":
            return "repro"
        return f"fr{int(cdhsa.get('fixed_rank', 10))}"

    @staticmethod
    def _checkpoint_key(job: dict) -> str:
        """Build a unique checkpoint key from a job dict.

        v6: la clave incluye los tokens que cambian las salidas (L,
        esquema de rango, max_common, a6/bc), de modo que cambiar el
        JSON re-entrena en lugar de colisionar con corridas viejas.
        """
        tasks_str = "+".join(job["tasks"])
        cdhsa = job["cdhsa_params"]
        ranktag = SingleSubjectCDHSABatchRunner._rank_tag(cdhsa)
        return (f"singlesub_n{job['n_subjects']}|{job['session']}|"
                f"{tasks_str}|{job['t_start']}|{job['t_end']}"
                f"|L{cdhsa['L']}|{ranktag}"
                f"|mc{int(cdhsa.get('max_common', 30))}"
                f"|a6n{int(cdhsa.get('a6_n_null', 100))}"
                f"|bcn{int(cdhsa.get('bc_n_perm', 5000))}")

    @staticmethod
    def _classification_checkpoint_key(params: dict, session: str) -> str:
        """Clave de checkpoint de la fase de clasificacion."""
        clf = params.get("classification", {})
        tw = params.get("time_window", {})
        tasks_str = "+".join(params["tasks"])
        _train_ids, test_ids = _resolve_split_subjects(params)
        n_test = len(test_ids) if test_ids else 0
        suffix = "" if clf.get("rank_truncation", "min") == "min" \
            else f"_d{clf.get('rank_truncation')}"
        return (f"classify|{session}|{tasks_str}"
                f"|{clf.get('method', 'energy')}{suffix}"
                f"|{tw.get('t_start')}|{tw.get('t_end')}"
                f"|nTest{n_test}")

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
                    "phase": job.get("phase", "train"),
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
        individual subjects together).  v6: pasa TODOS los parametros
        v5.1 del bloque cdhsa_params.
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

        # Time window (only when defined in the JSON)
        if job["t_start"] != "None":
            cmd.extend(["--t-start", job["t_start"]])
        if job["t_end"] != "None":
            cmd.extend(["--t-end", job["t_end"]])

        # CD-HSA parameters (v5.1 parity)
        cmd.extend([
            "--L", str(cdhsa["L"]),
            "--l-freq", str(cdhsa.get("l_freq", 1.0)),
            "--h-freq", str(cdhsa.get("h_freq", 40.0)),
            "--rank-method", str(cdhsa.get("rank_method", "fixed")),
            "--fixed-rank", str(cdhsa.get("fixed_rank", 10)),
            "--max-common", str(cdhsa.get("max_common", 30)),
            "--a6-n-null", str(cdhsa.get("a6_n_null", 100)),
            "--bc-n-perm", str(cdhsa.get("bc_n_perm", 5000)),
            "--d-max-specific", str(cdhsa.get("d_max_specific", 10)),
            "--seed", str(cdhsa.get("seed", 42)),
        ])
        if cdhsa.get("rank_method") == "variance":
            cmd.extend([
                "--var-explained", str(cdhsa.get("var_explained", 0.99)),
                "--var-max-rank", str(cdhsa.get("var_max_rank", 50)),
                "--var-min-rank", str(cdhsa.get("var_min_rank", 1)),
            ])
        if cdhsa.get("a6_max_common") is not None:
            cmd.extend(["--a6-max-common", str(cdhsa["a6_max_common"])])
        if cdhsa.get("a6_null_type"):
            cmd.extend(["--a6-null-type", str(cdhsa["a6_null_type"])])
        if cdhsa.get("a6_hankel_row_block"):
            cmd.extend(["--a6-hankel-row-block",
                        str(cdhsa["a6_hankel_row_block"])])
        if cdhsa.get("rank_outcome"):
            cmd.extend(["--rank-outcome", str(cdhsa["rank_outcome"])])
        if cdhsa.get("tangent_blocks_mode"):
            cmd.extend(["--tangent-blocks-mode",
                        str(cdhsa["tangent_blocks_mode"])])
        if cdhsa.get("d_rank_adaptive", False):
            cmd.append("--d-rank-adaptive")
        if cdhsa.get("d_n_null_specific") is not None:
            cmd.extend(["--d-n-null-specific",
                        str(cdhsa["d_n_null_specific"])])
        if cdhsa.get("d_loso", False):
            cmd.append("--d-loso")
        if cdhsa.get("d_loso_n_perm") is not None:
            cmd.extend(["--d-loso-n-perm", str(cdhsa["d_loso_n_perm"])])
        if cdhsa.get("run_replica_consistency", False):
            cmd.append("--run-replica-consistency")
        if cdhsa.get("effective_rank", False):
            cmd.append("--effective-rank")

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
        ``_resolve_out_dir()`` (v5.1: etiqueta de rango ``var{X}c{cap}``
        / ``repro`` / ``fr{n}``).
        """
        if not self.output_dir:
            return Path(".")

        cdhsa = job["cdhsa_params"]
        tw = {"t_start": job["t_start"], "t_end": job["t_end"]}
        tasks = job["tasks"]

        t_start_tag = (f"{tw['t_start']}s" if tw["t_start"] != "None" else "any")
        t_end_tag = (f"{tw['t_end']}s" if tw["t_end"] != "None" else "any")

        L = cdhsa["L"]
        a6n = cdhsa.get("a6_n_null", 100)
        bcn = cdhsa.get("bc_n_perm", 5000)
        l_freq = cdhsa.get("l_freq", 1.0)
        h_freq = cdhsa.get("h_freq", 40.0)
        depth = cdhsa.get("hankel_depth") or "auto"

        rtag = self._rank_tag(cdhsa)
        subj_label = f"nSub{job['n_subjects']}"

        out_dir = Path(
            f"{self.output_dir}/cdhsa/{job['session']}"
            f"/{subj_label}_L{L}_{rtag}_a6n{a6n}_bcn{bcn}"
            f"/{l_freq}-{h_freq}Hz"
            f"_depth{depth}"
            f"/from{t_start_tag}_to{t_end_tag}"
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
            self._notify(
                "error",
                f"Fallo construyendo comando de {job['session']}: "
                f"{exc}",
                title="[CD-HSA] Error de configuracion")
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
                self._notify(
                    "error",
                    f"Job CD-HSA {label}/{job['session']} FALLO "
                    f"(codigo {proc.returncode}); el batch continua.",
                    title="[CD-HSA] Fallo de job")

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
                f"Excepcion en job {label}/{job['session']}: {exc}",
                title="[CD-HSA] Excepcion en job")
            self._write_csv_log(
                job, False, -1, elapsed, cmd_for_log,
            )
            return key, False

    # ------------------------------------------------------------------
    # [v6] Classification phase (hold-out)
    # ------------------------------------------------------------------

    def _classification_out_dir(self, job: dict) -> Path | None:
        """Directorio de salida de la fase de clasificacion."""
        if not self.output_dir:
            return None
        clf = self.classification_cfg
        method_tag = str(clf.get("method", "energy"))
        rank_tr = clf.get("rank_truncation", "min")
        suffix = "" if rank_tr == "min" else f"_d{rank_tr}"
        t0 = job["t_start"] if job["t_start"] != "None" else "any"
        t1 = job["t_end"] if job["t_end"] != "None" else "any"
        return Path(
            f"{self.output_dir}/cdhsa_classification/{job['session']}"
            f"/{method_tag}{suffix}"
            f"/from{t0}s_to{t1}s_{'_'.join(job['tasks'])}")

    def _build_classification_command(self, job: dict,
                                      train_out: Path) -> list[str]:
        """Comando del pipeline de clasificacion para un job de train."""
        clf = self.classification_cfg
        cmd = [
            sys.executable, "-m", self.classification_module,
            "--train-results-dir", str(train_out),
            "--session", job["session"],
            "--subject-ids",
        ] + [str(i) for i in self.test_subject_ids] + [
            "--tasks",
        ] + list(job["tasks"])

        if job["t_start"] != "None":
            cmd.extend(["--t-start", job["t_start"]])
        if job["t_end"] != "None":
            cmd.extend(["--t-end", job["t_end"]])

        cmd.extend([
            "--method", str(clf.get("method", "energy")),
            "--rank-truncation", str(clf.get("rank_truncation", "min")),
            "--n-perm", str(clf.get("n_perm", 10000)),
            "--seed", str(clf.get("seed", 20261006)),
        ])
        if not clf.get("make_plots", True):
            cmd.append("--skip-plots")

        if self.db_path:
            cmd.extend(["--db-path", self.db_path])
        out_dir = self._classification_out_dir(job)
        if out_dir is not None:
            cmd.extend(["--out-dir", str(out_dir)])
        return cmd

    def _run_classification_job(self, job: dict, train_out: Path) -> bool:
        """Ejecutar (con polling RSS y log) el pipeline de clasificacion."""
        try:
            cmd = self._build_classification_command(job, train_out)
        except Exception as exc:
            logger.error("Error construyendo comando de clasificacion: %s",
                         exc)
            self._notify(
                "error", f"Fallo construyendo comando de clasificacion: "
                         f"{exc}", title="[CD-HSA] Clasificacion FALLO")
            return False

        out_dir = self._classification_out_dir(job)
        label = "classify/%s" % job["session"]
        logger.info(
            "RUN | %s | test=%d sujetos | train=%s",
            label, len(self.test_subject_ids), train_out,
        )
        logger.debug("CMD: %s", " ".join(cmd))

        clf_job = {
            "mode": "classification",
            "phase": "classify",
            "subject_ids": self.test_subject_ids,
            "n_subjects": len(self.test_subject_ids),
            "session": job["session"],
            "tasks": job["tasks"],
            "cdhsa_params": job["cdhsa_params"],
            "t_start": job["t_start"],
            "t_end": job["t_end"],
        }

        t0 = time.time()
        cmd_for_log = cmd
        try:
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=child_env,
            )

            _child_peak = [0.0]
            _child_samples: list[tuple[float, float]] = []

            def _poll_thread():
                pk, samps = self._poll_child_rss(proc)
                _child_peak[0] = pk
                _child_samples.extend(samps)

            _thr = threading.Thread(target=_poll_thread, daemon=True)
            _thr.start()

            for line in proc.stdout:
                logger.info("  [CLF] %s", line.rstrip())

            proc.wait()
            _thr.join(timeout=3.0)

            elapsed = time.time() - t0
            success = proc.returncode == 0
            peak_mb = _child_peak[0]

            self._job_memory_stats.append({
                "label": label,
                "session": job["session"],
                "success": success,
                "elapsed_s": round(elapsed, 2),
                "peak_rss_mb": round(peak_mb, 1),
                "n_rss_samples": len(_child_samples),
            })
            self._write_csv_log(clf_job, success, proc.returncode, elapsed,
                                cmd)

            if success:
                logger.info(
                    "OK | %s (%.1f s, peak %.1f MB)",
                    label, elapsed, peak_mb,
                )
                # Extraer el resumen de accuracy del results.json
                if out_dir is not None:
                    self._read_classification_summary(out_dir, job)
            else:
                logger.error("ERROR | %s -- codigo %d", label,
                             proc.returncode)
                self._notify(
                    "error",
                    f"Clasificacion {job['session']} FALLO "
                    f"(codigo {proc.returncode}).",
                    title="[CD-HSA] Clasificacion FALLO")
            return success

        except Exception as exc:
            elapsed = time.time() - t0
            logger.error("EXCEPTION | %s: %s", label, exc)
            self._notify(
                "error", f"Excepcion en clasificacion: {exc}",
                title="[CD-HSA] Clasificacion FALLO")
            self._write_csv_log(clf_job, False, -1, elapsed, cmd_for_log)
            return False

    def _read_classification_summary(self, out_dir: Path, job: dict) -> None:
        """Leer accuracy del results.json para log + notificacion final."""
        try:
            results_path = Path(out_dir) / "classification_results.json"
            with open(results_path, "r", encoding="utf-8") as fh:
                res = json.load(fh)
            parts = []
            for meth, m in res.get("metrics", {}).items():
                if not m:
                    continue
                parts.append(
                    f"{meth}: {m['n_correct']}/{m['n_test']} "
                    f"({m['accuracy'] * 100:.1f}%)")
            if parts:
                summary = (f"{job['session']} ({'+'.join(job['tasks'])}): "
                           + " | ".join(parts))
                self._classification_summaries.append(summary)
                self._notify(
                    "success",
                    "Clasificacion hold-out terminada: " + summary,
                    title="[CD-HSA] Resultado de clasificacion")
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("No se pudo leer el resumen de clasificacion: %s",
                           exc)

    def _run_classification_phase(self) -> bool:
        """Fase v6: clasificacion hold-out de los sujetos TEST."""
        if not self.classification_enabled:
            return True

        logger.info("")
        logger.info("=" * 70)
        logger.info("  FASE DE CLASIFICACION (hold-out, %d sujetos test)",
                    len(self.test_subject_ids))
        logger.info("=" * 70)

        all_ok = True
        for job in self.all_jobs:
            session = job["session"]
            key = self._classification_checkpoint_key(self.params, session)
            if key in self.checkpoint:
                logger.info(
                    "SKIP (checkpoint): clasificacion %s ya completada",
                    session)
                # Recuperar el resumen si existe en disco
                out_dir = self._classification_out_dir(job)
                if out_dir is not None and not self._classification_summaries:
                    self._read_classification_summary(out_dir, job)
                continue

            train_out = self._get_output_dir(job)
            if not (train_out / "cdhsa_arrays.npz").exists():
                logger.error(
                    "Sin resultados de entrenamiento en %s: fase de "
                    "clasificacion BLOQUEADA para %s", train_out, session)
                self._notify(
                    "error",
                    f"Clasificacion {session} bloqueada: no hay "
                    f"resultados de train en {train_out}.",
                    title="[CD-HSA] Clasificacion bloqueada")
                all_ok = False
                continue

            ok = self._run_classification_job(job, train_out)
            if ok:
                self.checkpoint.add(key)
                self._save_checkpoint()
            else:
                all_ok = False
        return all_ok

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

        # [NTFY] Canary de arranque
        t_batch_start = time.time()
        label = self.params.get("experiment_label", "batch_cdhsa_ss")
        try:
            import socket
            host = socket.gethostname()
        except Exception:
            host = "(host desconocido)"
        plan = (f"Experimento {label} | S={self.all_jobs[0]['n_subjects']} "
                f"train")
        if self.classification_enabled:
            plan += (f" + {len(self.test_subject_ids)} test (clasificacion)"
                     f" | tareas {' + '.join(self.params['tasks'])}")
        self._notify("info", plan + f" | host {host}",
                     title="[CD-HSA] Batch single-subject iniciado")

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
            clf_ok = self._run_classification_phase()
            return self._final_summary(0, 0, 0, clf_ok, t_batch_start)

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

        clf_ok = self._run_classification_phase()
        return self._final_summary(completed, failed, total, clf_ok,
                                   t_batch_start)

    def _final_summary(self, completed: int, failed: int, total: int,
                       clf_ok: bool, t_batch_start: float) -> int:
        """Log + notificacion final del batch (incluye accuracy si hubo)."""
        rc = 0 if (failed == 0 and clf_ok) else 1
        duration = time.time() - t_batch_start
        try:
            import socket
            host = socket.gethostname()
        except Exception:
            host = "(host desconocido)"

        if self._classification_summaries:
            logger.info("")
            logger.info("=" * 70)
            logger.info("  RESUMEN DE CLASIFICACION (hold-out)")
            for s in self._classification_summaries:
                logger.info("  %s", s)
            logger.info("=" * 70)

        msg_lines = [
            f"Duracion total: {duration / 3600.0:.2f} h ({duration:.0f} s)",
            f"Host: {host}",
            f"Train: OK {completed} / Fallos {failed} (total {total})",
        ]
        if self.classification_enabled:
            msg_lines.append(
                "Clasificacion: " + ("OK" if clf_ok else "FALLO/BLOQUEADA"))
            for s in self._classification_summaries:
                msg_lines.append(f"  {s}")
        msg_lines.append(f"Salidas: {self.output_dir}")
        msg = "\n".join(msg_lines)

        if rc == 0:
            self._notify("success", msg,
                         title="[CD-HSA] Batch single-subject terminado OK")
        else:
            self._notify("warning", msg,
                         title="[CD-HSA] Batch terminado CON PROBLEMAS")
        return rc

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


@notify_on_critical_error(
    channel=_NTFY_CHANNEL or None,
    title="[CD-HSA] ERROR critico del batch single-subject",
    catch_system_exit=True,
)
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch runner para el pipeline CD-HSA single-subject. "
            "Lee la configuracion desde un JSON (esquema legado o v6 "
            "train/test + clasificacion)."
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
    parser.add_argument(
        "--pipeline-module", type=str, default=None,
        help="Modulo del pipeline de entrenamiento "
             "(default: src.pipelines.run_cdhsa_single_subject).",
    )
    parser.add_argument(
        "--classification-module", type=str, default=None,
        help="Modulo del pipeline de clasificacion "
             "(default: src.pipelines.run_cdhsa_task_classification).",
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
        runner = SingleSubjectCDHSABatchRunner(
            params,
            pipeline_script=pipeline_script,
            pipeline_module=args.pipeline_module,
            classification_module=args.classification_module,
        )
        return runner.run()


if __name__ == "__main__":
    sys.exit(main())
