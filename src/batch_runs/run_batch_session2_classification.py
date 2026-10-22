"""
run_batch_session2_classification.py
====================================
Batch del experimento de clasificacion con hold-out de SESION:
modelos CD-HSA entrenados con SUPER-SUJETOS de la sesion 1 (corridas
``ec_mu`` / ``ec_ma`` de ``run_batch_cdhsa``) probados contra los
sujetos individuales de la sesion 2.

Fases
-----
FASE 0 -- Auditoria (ANTES de cualquier procesamiento)
    Verifica, grabacion a grabacion (solo cabeceras ``.set``, sin
    leer ``.fdt``), que cada sujeto de la sesion 2 puede alinearse a
    los canales del entrenamiento (p. ej. 53): canales presentes,
    ausentes (a interpolar o no) y duracion.  Salidas:
    ``channel_audit.json`` / ``.csv`` / ``_summary.txt`` y mapas de
    ausencias por modelo.

FASE 1 -- Experimentos de clasificacion
    Un job por entrada del bloque ``experiments`` del JSON: pipeline
    ``src.pipelines.run_cdhsa_session2_classification`` (clasificacion
    de energias de los modos especificos + estadistica + plots).
    Checkpoint por experimento; re-ejecutar el batch salta los ya
    hechos (si su ``classification_results.json`` existe).

FASE 2 -- Cross-experimento
    Agrega todos los experimentos: matriz sujetos x experimentos,
    sujetos PERFECTOS en todos, sujetos que FALLAN en todos, accuracy
    con IC por experimento y tasas de error por tarea.

Uso
---
    # Desde la raiz del repo (con .env configurado):
    python src/batch_runs/run_batch_session2_classification.py

    # JSON custom:
    python src/batch_runs/run_batch_session2_classification.py \\
        --params-json src/batch_runs/cdhsa_batch_params_session2_classification.json

    # Esperar a que terminen los entrenamientos SS antes de empezar:
    python src/batch_runs/run_batch_session2_classification.py \\
        --wait-for-training --poll-interval 600

El JSON (``cdhsa_batch_params_session2_classification.json``) define
modelos de entrenamiento (auto-descubrimiento bajo
``BASE_RESULTS_PATH/cdhsa/{session}``), sesion/sujetos/ventana de
test, politica de canales, experimentos (2+ tareas cada uno) y el
bloque de clasificacion.  Ver la cabecera del JSON para el esquema.

Notificaciones ntfy: inicio, resumen de auditoria, resultado por
experimento (accuracy) y resumen final.  Fallos urgentes por job.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# [NTFY] Push notifications for long runs (optional; see src/ntfy/README.md)
# ---------------------------------------------------------------------------
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
    _HAS_NTFY = False

    def notify_on_critical_error(_func=None, **_kwargs):
        """No-op shim: src.ntfy not available (notifications disabled)."""
        def _deco(func):
            return func
        return _deco(_func) if _func is not None else _deco

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-10s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_session2_classification")

DEFAULT_PARAMS_JSON = (_SCRIPT_DIR / "cdhsa_batch_params_session2_classification.json")

_VALID_TASKS_HINT = ("eyesclosed", "eyesopen", "mathematic", "memory",
                     "music")


# ===========================================================================
# Carga y validacion del JSON de parametros
# ===========================================================================

def _err(msg: str) -> None:
    logger.error("[PARAMS] %s", msg)


def _validate_params(params: dict) -> list[str]:
    """Validacion completa del esquema. Devuelve la lista de errores."""
    errors: list[str] = []

    # --- bloque test ---
    test = params.get("test")
    if not isinstance(test, dict):
        errors.append("'test' debe ser un objeto.")
        return errors

    session = test.get("session")
    if not isinstance(session, str) or not session:
        errors.append("test.session debe ser un string no vacio.")

    subj = test.get("subjects")
    if not isinstance(subj, dict):
        errors.append("test.subjects debe ser un objeto.")
    else:
        ids = subj.get("subject_ids")
        n = subj.get("n_subjects")
        if ids is not None:
            if (not isinstance(ids, list) or not ids
                    or not all(isinstance(x, int) and x > 0 for x in ids)):
                errors.append("test.subjects.subject_ids debe ser una "
                              "lista de enteros positivos no vacia.")
        elif not isinstance(n, int) or n < 1:
            errors.append("test.subjects.n_subjects debe ser un entero "
                          "positivo (o usar subject_ids).")
        if ids is None and subj.get("subject_start_offset") is not None \
                and not isinstance(subj.get("subject_start_offset"), int):
            errors.append("test.subjects.subject_start_offset debe ser "
                          "entero.")

    tw = test.get("time_window")
    if not isinstance(tw, dict):
        errors.append("test.time_window debe ser un objeto "
                      "{t_start, t_end}.")
    else:
        t0, t1 = tw.get("t_start"), tw.get("t_end")
        for k, v in (("t_start", t0), ("t_end", t1)):
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                errors.append(f"test.time_window.{k} debe ser numerico.")
        if (isinstance(t0, (int, float)) and isinstance(t1, (int, float))
                and not isinstance(t0, bool) and not isinstance(t1, bool)
                and t0 >= t1):
            errors.append(f"test.time_window: t_start ({t0}) >= t_end "
                         f"({t1}).")

    min_dur = test.get("min_duration_s", 130.0)
    if not isinstance(min_dur, (int, float)) or isinstance(min_dur, bool) \
            or min_dur <= 0:
        errors.append("test.min_duration_s debe ser un numero positivo.")

    ch = test.get("channels")
    if not isinstance(ch, dict):
        errors.append("test.channels debe ser un objeto.")
    else:
        policy = ch.get("missing_policy", "interpolate")
        if policy not in ("interpolate", "skip"):
            errors.append("test.channels.missing_policy debe ser "
                          "'interpolate' o 'skip'.")
        mx = ch.get("max_missing_to_interpolate", 4)
        if not isinstance(mx, int) or mx < 0:
            errors.append("test.channels.max_missing_to_interpolate "
                          "debe ser un entero >= 0.")

    # --- modelos de entrenamiento ---
    models = params.get("training_models")
    if not isinstance(models, dict) or not models:
        errors.append("training_models debe ser un objeto no vacio "
                      "{model_key: {tasks: [...], results_dir: null|str}}.")
        models = {}
    for key, m in models.items():
        if key.startswith("_"):
            continue
        if not isinstance(m, dict):
            errors.append(f"training_models.{key} debe ser un objeto.")
            continue
        tasks = m.get("tasks")
        if (not isinstance(tasks, list) or len(tasks) < 2
                or not all(isinstance(t, str) and t for t in tasks)
                or len(set(tasks)) != len(tasks)):
            errors.append(f"training_models.{key}.tasks debe ser una "
                          f"lista de >=2 strings unicos.")
        rd = m.get("results_dir")
        if rd is not None and not isinstance(rd, str):
            errors.append(f"training_models.{key}.results_dir debe ser "
                          f"null o un string.")
    if "_comment" in models and len(models) == 1:
        errors.append("training_models no define ningun modelo real "
                      "(solo _comment).")

    tsession = params.get("training_session", "session1")
    if not isinstance(tsession, str) or not tsession:
        errors.append("training_session debe ser un string no vacio.")

    # --- experimentos ---
    exps = params.get("experiments")
    if not isinstance(exps, list) or not exps:
        errors.append("experiments debe ser una lista no vacia.")
        exps = []
    names = set()
    for i, e in enumerate(exps):
        if not isinstance(e, dict):
            errors.append(f"experiments[{i}] debe ser un objeto.")
            continue
        name = e.get("name")
        if (not isinstance(name, str)
                or not name.replace("_", "").isalnum()
                or name != name.lower() or not name):
            errors.append(f"experiments[{i}].name debe ser un slug "
                          f"[a-z0-9_] (se recibio {name!r}).")
        elif name in names:
            errors.append(f"experiments[{i}].name duplicado: {name}.")
        else:
            names.add(name)
        model = e.get("model")
        if not isinstance(model, str) or (
                isinstance(models, dict) and model not in models):
            errors.append(f"experiments[{i}].model ({model!r}) no esta "
                          f"en training_models.")
        tasks = e.get("tasks")
        if not isinstance(tasks, list) or len(tasks) < 2 \
                or not all(isinstance(t, str) and t for t in tasks) \
                or len(set(tasks)) != len(tasks):
            errors.append(f"experiments[{i}].tasks debe ser una lista "
                          f"de >=2 strings unicos.")
        elif isinstance(models, dict) and isinstance(model, str) \
                and model in models and isinstance(models[model], dict):
            model_tasks = models[model].get("tasks")
            if isinstance(model_tasks, list):
                missing = [t for t in tasks if t not in model_tasks]
                if missing:
                    errors.append(
                        f"experiments[{i}].tasks contiene tareas que no "
                        f"estan en el modelo '{model}': {missing}.")

    # --- clasificacion ---
    clf = params.get("classification")
    if not isinstance(clf, dict):
        errors.append("classification debe ser un objeto.")
    else:
        if clf.get("method", "energy") not in ("energy", "ml", "both"):
            errors.append("classification.method debe ser 'energy' | "
                          "'ml' | 'both'.")
        rt = clf.get("rank_truncation", "min")
        if rt != "min":
            if not isinstance(rt, int) or isinstance(rt, bool) or rt < 1:
                errors.append("classification.rank_truncation debe ser "
                              "'min' o un entero >= 1.")
        nperm = clf.get("n_perm", 10000)
        if not isinstance(nperm, int) or isinstance(nperm, bool) \
                or nperm < 1:
            errors.append("classification.n_perm debe ser entero >= 1.")
        if not isinstance(clf.get("seed", 0), int):
            errors.append("classification.seed debe ser entero.")
        if not isinstance(clf.get("make_plots", True), bool):
            errors.append("classification.make_plots debe ser booleano.")

    # --- ejecucion ---
    exe = params.get("execution", {})
    if not isinstance(exe, dict):
        errors.append("execution debe ser un objeto.")
    else:
        mw = exe.get("max_workers", 1)
        if not isinstance(mw, int) or isinstance(mw, bool) or mw < 1:
            errors.append("execution.max_workers debe ser entero >= 1.")
        elif mw != 1:
            errors.append("execution.max_workers debe ser 1 (secuencial): "
                          "cada job de clasificacion ya itera sobre todas "
                          "las grabaciones de test.")
        if not isinstance(exe.get("delay", 0.0), (int, float)):
            errors.append("execution.delay debe ser numerico.")

    return errors


def _load_params(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        params = json.load(fh)
    if not isinstance(params, dict):
        raise ValueError("El JSON de parametros debe ser un objeto.")
    errors = _validate_params(params)
    if errors:
        for e in errors:
            _err(e)
        raise ValueError(f"JSON de parametros invalido ({len(errors)} "
                         f"errores).")
    return params


def _resolve_test_subject_ids(test: dict) -> list[int]:
    subj = test["subjects"]
    if subj.get("subject_ids"):
        return [int(x) for x in subj["subject_ids"]]
    return list(range(int(subj.get("subject_start_offset", 1)),
                      int(subj.get("subject_start_offset", 1))
                      + int(subj.get("n_subjects", 60))))


def _rank_suffix(clf: dict) -> str:
    rt = clf.get("rank_truncation", "min")
    return "" if rt == "min" else f"_d{rt}"


# ===========================================================================
# Runner
# ===========================================================================

class Session2ClassificationBatchRunner:
    """Orquesta auditoria + experimentos + cross-experimento."""

    CSV_FIELDS = [
        "timestamp", "phase", "experiment", "model", "session", "tasks",
        "method", "success", "returncode", "elapsed_s",
        "accuracy", "n_ok", "n_errors", "peak_rss_mb", "command",
    ]

    DEFAULT_PIPELINE_MODULE = "src.pipelines.run_cdhsa_session2_classification"

    # Artefactos minimos de un train completado (cdhsa_arrays.npz es el
    # ULTIMO que escribe run_cdhsa.save_results).
    TRAIN_ARTIFACTS = ("hankel_info.json", "config.json",
                       "cdhsa_arrays.npz")

    def __init__(self, params: dict, *,
                 pipeline_module: str | None = None,
                 force_audit: bool = False) -> None:
        self.params = params
        self.pipeline_module = (
            pipeline_module or self.DEFAULT_PIPELINE_MODULE)
        self.force_audit = force_audit

        self.test_cfg = params["test"]
        self.clf_cfg = params.get("classification", {})
        self.exec_cfg = params.get("execution", {})
        self.cross_cfg = params.get("cross_experiment", {})
        self.test_session = self.test_cfg["session"]
        self.test_subject_ids = _resolve_test_subject_ids(self.test_cfg)
        self.test_tasks = sorted({t for e in params["experiments"]
                                  for t in e["tasks"]})

        self.delay: float = float(self.exec_cfg.get("delay", 0.0))

        # Rutas de proyecto
        self.db_path: str | None = None
        self.output_dir: Path | None = None
        self.cache_dir: Path | None = None
        self.training_root: Path | None = None
        self.ntfy_channel: str | None = None
        self._resolve_project_paths()

        # Checkpoint + CSV
        self.checkpoint: set[str] = self._load_checkpoint()
        self.log_file = self._resolve_log_path()
        self._init_csv_log()
        self._job_memory_stats: list[dict] = []

        # Info de modelos descubiertos (fase de discovery)
        self._model_info: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Project paths
    # ------------------------------------------------------------------

    def _resolve_project_paths(self) -> None:
        try:
            from src.utils.config import (
                BASE_CACHE_PATH,
                BASE_RESULTS_PATH,
                DB_TEST_RETEST_GEDAI_PATH,
            )
            self.db_path = str(DB_TEST_RETEST_GEDAI_PATH)
            self.output_dir = Path(BASE_RESULTS_PATH)
            self.cache_dir = Path(BASE_CACHE_PATH)
        except ImportError:
            logger.info("src.utils.config no disponible; usando rutas "
                        "por defecto. Usa --db-path y --out-dir.")
            self.output_dir = Path("./results")
            self.cache_dir = Path("./cache")

        root = self.params.get("training_root")
        if root:
            self.training_root = Path(root)
        elif self.output_dir is not None:
            self.training_root = Path(self.output_dir) / "cdhsa"

        try:
            from src.utils.config import NTFY_CHANNEL
            self.ntfy_channel = (NTFY_CHANNEL or "").strip() or None
        except Exception:
            self.ntfy_channel = None
        if self.ntfy_channel:
            logger.info("  NTFY channel    : %s (notificaciones push "
                        "activas)", self.ntfy_channel)

    # ------------------------------------------------------------------
    # [NTFY]
    # ------------------------------------------------------------------

    def _notify(self, level: str, message: str,
                title: str | None = None) -> None:
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
            return
        try:
            if title is not None:
                fn(message, channel=self.ntfy_channel, title=title)
            else:
                fn(message, channel=self.ntfy_channel)
        except Exception as exc:
            logger.warning("[NTFY] Fallo enviando notificacion: %s", exc)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _checkpoint_path(self) -> Path:
        base = self.cache_dir or Path("./cache")
        return base / "batch_checkpoint_session2_classification.json"

    def _load_checkpoint(self) -> set[str]:
        cp = self._checkpoint_path()
        if cp.exists():
            try:
                with open(cp, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                ck = set(data.get("completed", []))
                logger.info("Checkpoint cargado: %s fases previas "
                            "completadas", len(ck))
                return ck
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Checkpoint corrupto, empezando de cero: %s",
                               exc)
        return set()

    def _save_checkpoint(self) -> None:
        cp = self._checkpoint_path()
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            with open(cp, "w", encoding="utf-8") as fh:
                json.dump({"completed": sorted(self.checkpoint)}, fh,
                          indent=2)
        except OSError as exc:
            logger.warning("No se pudo guardar checkpoint: %s", exc)

    # ------------------------------------------------------------------
    # CSV log
    # ------------------------------------------------------------------

    def _resolve_log_path(self) -> Path:
        log_dir = (self.output_dir or Path("./")) / "batch_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = self.params.get("experiment_label",
                                "session2_classification")
        return log_dir / f"batch_session2_classification_{label}_{ts}.csv"

    def _init_csv_log(self) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.log_file, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.CSV_FIELDS)
                writer.writeheader()
        except OSError as exc:
            logger.warning("No se pudo inicializar log CSV: %s", exc)

    def _write_csv_log(self, row: dict) -> None:
        try:
            with open(self.log_file, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.CSV_FIELDS)
                writer.writerow(row)
        except OSError as exc:
            logger.warning("Fallo al escribir log CSV: %s", exc)

    # ------------------------------------------------------------------
    # Discovery de los modelos de entrenamiento (SS)
    # ------------------------------------------------------------------

    def _read_train_info(self, train_dir: Path) -> dict | None:
        """Lee hankel_info.json + config.json de un train dir."""
        info_path = train_dir / "hankel_info.json"
        cfg_path = train_dir / "config.json"
        if not info_path.exists() or not cfg_path.exists():
            return None
        try:
            with open(info_path, "r", encoding="utf-8") as fh:
                hinfo = json.load(fh)
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None
        return {"dir": train_dir, "hankel_info": hinfo, "config": cfg}

    def _find_model_candidates(self, model_key: str,
                               model_cfg: dict) -> list[Path]:
        """Candidatos (dirs con hankel_info.json) para un modelo.

        Los trains con task_shuffle (control negativo: tareas
        permutadas por sujeto al armar los SS) se EXCLUYEN
        automaticamente: no son modelos validos para clasificar (las
        etiquetas de condicion de sus celdas son nominales).
        """
        tasks = set(model_cfg["tasks"])
        session = self.params.get("training_session", "session1")
        root = self.training_root / session
        if not root.exists():
            return []
        candidates: list[Path] = []
        for info_path in sorted(root.glob("*/*/*/hankel_info.json")):
            try:
                with open(info_path, "r", encoding="utf-8") as fh:
                    hinfo = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            if hinfo.get("task_shuffle_seed") is not None:
                logger.debug(
                    "Candidato excluido (task_shuffle seed=%s): %s",
                    hinfo.get("task_shuffle_seed"), info_path.parent,
                )
                continue
            if set(hinfo.get("tasks", [])) == tasks:
                candidates.append(info_path.parent)
        return candidates

    def discover_training_models(self, wait: bool = False,
                                 poll_interval: float = 300.0,
                                 wait_timeout: float = 0.0) -> bool:
        """Resuelve el dir de cada modelo; True si TODOS disponibles.

        Con ``wait=True`` hace polling hasta que los artefactos de los
        modelos requeridos existan (util para lanzar el batch mientras
        los entrenamientos SS siguen corriendo: cdhsa_arrays.npz es el
        ULTIMO artefacto que escribe run_cdhsa).
        """
        models = self.params["training_models"]
        t_start = time.time()
        while True:
            pending: list[str] = []
            for key, cfg in models.items():
                if key.startswith("_"):
                    continue
                if self._resolve_single_model(key, cfg):
                    continue
                pending.append(key)

            if not pending:
                for key in sorted(self._model_info):
                    info = self._model_info[key]
                    logger.info("  Modelo %-8s : %s", key,
                                info["dir"])
                return True

            if not wait:
                for key in pending:
                    logger.error(
                        "Modelo '%s' no disponible (ni results_dir "
                        "explicito ni auto-descubrimiento bajo %s).",
                        key, self.training_root)
                return False

            elapsed = time.time() - t_start
            if wait_timeout > 0 and elapsed > wait_timeout:
                logger.error("Timeout (%.0f s) esperando los modelos: %s",
                             wait_timeout, pending)
                self._notify("error",
                             f"Timeout esperando entrenamientos: "
                             f"{', '.join(pending)}",
                             title="[CD-HSA] Espera de trainings "
                                   "FALLO")
                return False

            if elapsed < poll_interval:
                logger.warning(
                    "Esperando entrenamientos SS pendientes: %s "
                    "(siguiente chequeo en %.0f s; Ctrl-C para "
                    "abortar)", pending, poll_interval)
            else:
                logger.warning(
                    "[%.0f s] Esperando entrenamientos SS: %s ...",
                    elapsed, pending)
            time.sleep(poll_interval)

    def _resolve_single_model(self, key: str, model_cfg: dict) -> bool:
        """Intenta resolver UN modelo; llena self._model_info[key]."""
        explicit = model_cfg.get("results_dir")
        if explicit:
            train_dir = Path(explicit)
            info = self._read_train_info(train_dir)
            if info is None:
                return False
            if set(info["hankel_info"].get("tasks", [])) != \
                    set(model_cfg["tasks"]):
                logger.error("Modelo '%s': results_dir tiene tareas %s "
                             "!= declaradas %s", key,
                             info["hankel_info"].get("tasks"),
                             model_cfg["tasks"])
                return False
            if info["hankel_info"].get("task_shuffle_seed") is not None:
                logger.error(
                    "Modelo '%s': results_dir=%s corresponde a un train "
                    "con task_shuffle (seed=%s, control negativo): NO "
                    "es un modelo valido para clasificar.",
                    key, train_dir,
                    info["hankel_info"].get("task_shuffle_seed"))
                return False
            if not self._train_artifacts_complete(train_dir):
                return False
            self._model_info[key] = info
            return True

        for cand in self._find_model_candidates(key, model_cfg):
            if self._train_artifacts_complete(cand):
                info = self._read_train_info(cand)
                if info is None:
                    continue
                self._model_info[key] = info
                return True
        return False

    def _train_artifacts_complete(self, train_dir: Path) -> bool:
        """Artefactos minimos presentes (y hankel_matrices si ML)."""
        needed = list(self.TRAIN_ARTIFACTS)
        if self.clf_cfg.get("method", "energy") in ("ml", "both"):
            needed.append("hankel_matrices.npz")
        return all((train_dir / f).exists() for f in needed)

    def _validate_window_against_train(self) -> None:
        """La ventana de test DEBE coincidir con la del train."""
        from src.cdhsa.classification import parse_time_window_from_dir
        tw = self.test_cfg["time_window"]
        for key, info in self._model_info.items():
            t0, t1 = parse_time_window_from_dir(info["dir"])
            if t0 is None:
                raise ValueError(
                    f"No se pudo deducir la ventana del train '{key}' "
                    f"desde {info['dir']}")
            if (abs(t0 - float(tw["t_start"])) > 1e-9
                    or abs(t1 - float(tw["t_end"])) > 1e-9):
                raise ValueError(
                    f"La ventana de test ({tw['t_start']}-"
                    f"{tw['t_end']}s) difiere de la del train '{key}' "
                    f"({t0}-{t1}s): el preprocesado debe ser IDENTICO.")

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        p = self.params
        logger.info("=" * 70)
        logger.info("  BATCH CLASIFICACION SESION 2 (train SS sesion %s)",
                    p.get("training_session", "session1"))
        logger.info("=" * 70)
        logger.info("  Pipeline module : %s", self.pipeline_module)
        logger.info("  DB path         : %s", self.db_path or "(default)")
        logger.info("  Test session    : %s (%d sujetos)",
                    self.test_session, len(self.test_subject_ids))
        logger.info("  Tareas auditadas: %s", "+".join(self.test_tasks))
        tw = self.test_cfg["time_window"]
        logger.info("  Ventana test    : %s s -> %s s",
                    tw["t_start"], tw["t_end"])
        ch = self.test_cfg["channels"]
        logger.info("  Canales         : policy=%s max_missing=%s "
                    "min_duration=%ss", ch["missing_policy"],
                    ch["max_missing_to_interpolate"],
                    self.test_cfg.get("min_duration_s", 130.0))
        logger.info("  ---")
        for e in p["experiments"]:
            logger.info("  Experimento %-12s : modelo %-8s tareas %s",
                        e["name"], e["model"], "+".join(e["tasks"]))
        logger.info("  Metodo          : %s (rank %s, %d perms, seed %s)",
                    self.clf_cfg.get("method", "energy"),
                    self.clf_cfg.get("rank_truncation", "min"),
                    self.clf_cfg.get("n_perm", 10000),
                    self.clf_cfg.get("seed", 20261006))
        logger.info("  Output root     : %s",
                    self.out_root())
        logger.info("  Checkpoint      : %s", self._checkpoint_path())
        logger.info("=" * 70)

    def out_root(self) -> Path:
        return (self.output_dir or Path("./results")) / \
            "cdhsa_classification" / self.test_session

    # ------------------------------------------------------------------
    # FASE 0: auditoria de canales
    # ------------------------------------------------------------------

    def _audit_checkpoint_key(self) -> str:
        ch = self.test_cfg["channels"]
        return ("audit|%s|%s|n%d|%s|mm%d|md%s"
                % (self.test_session, "+".join(self.test_tasks),
                   len(self.test_subject_ids),
                   ch["missing_policy"],
                   int(ch["max_missing_to_interpolate"]),
                   self.test_cfg.get("min_duration_s", 130.0)))

    def _run_audit_phase(self) -> bool:
        from src.latent_space_extraction.channel_alignment import (
            audit_session_recordings,
            plot_channel_missingness,
        )

        key = self._audit_checkpoint_key()
        audit_dir = self.out_root() / "channel_audit"
        audit_json = audit_dir / "channel_audit.json"
        if key in self.checkpoint and audit_json.exists() \
                and not self.force_audit:
            logger.info("[AUDIT] Ya completada (checkpoint): %s",
                        audit_json)
            return True

        logger.info("[AUDIT] FASE 0: verificacion de canales y duraciones")
        logger.info("[AUDIT] %d sujetos x %d tareas, sesion %s "
                    "(solo cabeceras .set)",
                    len(self.test_subject_ids), len(self.test_tasks),
                    self.test_session)

        # Conjuntos objetivo: global_channels de cada modelo disponible
        model_targets: dict[str, list[str]] = {}
        for key_m, info in self._model_info.items():
            model_targets[key_m] = list(
                info["hankel_info"].get("global_channels", []))
        if not model_targets:
            logger.error("[AUDIT] Sin modelos disponibles: no hay contra "
                         "que auditar.")
            return False

        ch = self.test_cfg["channels"]
        audit = audit_session_recordings(
            session=self.test_session,
            tasks=self.test_tasks,
            subject_ids=self.test_subject_ids,
            db_path=self.db_path,
            model_targets=model_targets,
            missing_policy=ch["missing_policy"],
            max_missing=int(ch["max_missing_to_interpolate"]),
            min_duration_s=float(
                self.test_cfg.get("min_duration_s", 130.0)),
        )

        audit_dir.mkdir(parents=True, exist_ok=True)
        with open(audit_json, "w", encoding="utf-8") as fh:
            json.dump(audit, fh, indent=2)
        self._write_audit_csv(audit, audit_dir / "channel_audit.csv")
        self._write_audit_summary(audit,
                                  audit_dir / "channel_audit_summary.txt")
        for key_m in model_targets:
            try:
                plot_channel_missingness(
                    audit, key_m,
                    audit_dir / f"channel_missingness_{key_m}.png")
            except Exception as exc:
                logger.warning("[AUDIT] Plot de ausencias '%s' fallo: %s",
                               key_m, exc)

        s = audit["summary"]
        logger.info("[AUDIT] Listo: %d/%d grabaciones existen",
                    s["n_exists"], s["n_recordings"])
        for key_m, ms in s["per_model"].items():
            logger.info("[AUDIT]   modelo %-8s: usables %d "
                        "(interp %d, skip %d)",
                        key_m, ms["n_usable"], ms["n_interpolate"],
                        ms["n_skip"])

        self._write_csv_log({
            "timestamp": datetime.now().isoformat(),
            "phase": "audit", "experiment": "-",
            "model": "+".join(sorted(model_targets)),
            "session": self.test_session,
            "tasks": "+".join(self.test_tasks),
            "method": "-", "success": True, "returncode": 0,
            "elapsed_s": s.get("elapsed_s"),
            "accuracy": "", "n_ok": s["n_exists"],
            "n_errors": s["n_missing_files"],
            "peak_rss_mb": "", "command": "",
        })

        self.checkpoint.add(key)
        self._save_checkpoint()

        self._notify(
            "info",
            f"Auditoria sesion {self.test_session}: "
            f"{s['n_exists']}/{s['n_recordings']} grabaciones; "
            + "; ".join(f"{k}: {v['n_usable']} usables "
                        f"({v['n_interpolate']} interp)"
                        for k, v in s["per_model"].items()),
            title="[CD-HSA] Auditoria canales sesion 2")
        return True

    @staticmethod
    def _write_audit_csv(audit: dict, path: Path) -> None:
        models = list(audit["models"].keys())
        fields = (["subject_id", "task", "exists", "error",
                   "n_raw_channels", "n_valid_channels", "sfreq",
                   "duration_s", "usable_any"]
                  + [f"n_missing_{m}" for m in models]
                  + [f"missing_{m}" for m in models]
                  + [f"action_{m}" for m in models])
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for r in audit["recordings"]:
                row = {
                    "subject_id": r["subject_id"],
                    "task": r["task"],
                    "exists": int(r["exists"]),
                    "error": r["error"] or "",
                    "n_raw_channels": r["n_raw_channels"],
                    "n_valid_channels": r["n_valid_channels"],
                    "sfreq": r["sfreq"],
                    "duration_s": (round(r["duration_s"], 2)
                                   if r["duration_s"] is not None else ""),
                    "usable_any": int(r["usable_any"]),
                }
                for m in models:
                    pm = r["per_model"][m]
                    row[f"n_missing_{m}"] = pm["n_missing"]
                    row[f"missing_{m}"] = ";".join(pm["missing"])
                    row[f"action_{m}"] = pm["action"]
                writer.writerow(row)

    @staticmethod
    def _write_audit_summary(audit: dict, path: Path) -> None:
        s = audit["summary"]
        lines = [
            "",
            "=" * 70,
            "  AUDITORIA DE CANALES (pase previo al procesamiento)",
            "=" * 70,
            f"  Sesion            : {audit['session']}",
            f"  Grabaciones       : {s['n_recordings']} "
            f"({s['n_exists']} existen, {s['n_missing_files']} faltantes)",
            f"  Politica          : {audit['policy']['missing_policy']}"
            f" (max {audit['policy']['max_missing']} ausentes, "
            f"min dur {audit['policy']['min_duration_s']}s)",
        ]
        for m, ms in s["per_model"].items():
            lines.append("-" * 70)
            lines.append(f"  Modelo '{m}' ({audit['models'][m]['n_channels']}"
                         f" canales de train):")
            lines.append(f"    Usables          : {ms['n_usable']}"
                         f" / {s['n_recordings']} "
                         f"({ms['usable_frac'] * 100:.1f}%)")
            lines.append(f"    Con interpolacion: {ms['n_interpolate']}")
            lines.append(f"    Saltadas         : {ms['n_skip']}")
        lines.append("-" * 70)
        all_ok = s["subjects_all_recordings_usable"]
        lines.append(f"  Sujetos con TODAS sus grabaciones usables: "
                     f"{len(all_ok)}")
        lines.append("=" * 70)
        text = "\n".join(lines)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        logger.info("[AUDIT] %s", path)

    # ------------------------------------------------------------------
    # FASE 1: experimentos de clasificacion
    # ------------------------------------------------------------------

    def _experiment_out_dir(self, exp: dict) -> Path:
        clf = self.clf_cfg
        method_tag = str(clf.get("method", "energy"))
        tw = self.test_cfg["time_window"]
        t0, t1 = tw["t_start"], tw["t_end"]
        return Path(
            f"{self.out_root()}/{exp['name']}"
            f"/{method_tag}{_rank_suffix(clf)}"
            f"/from{t0}s_to{t1}s_{'_'.join(exp['tasks'])}")

    def _experiment_checkpoint_key(self, exp: dict) -> str:
        clf = self.clf_cfg
        tw = self.test_cfg["time_window"]
        ch = self.test_cfg["channels"]
        return ("clf|%s|%s|%s|%s%s|%s|%s|n%d|%smm%d|md%s|%s|%s"
                % (exp["name"], exp["model"],
                   "+".join(exp["tasks"]),
                   clf.get("method", "energy"), _rank_suffix(clf),
                   tw["t_start"], tw["t_end"],
                   len(self.test_subject_ids),
                   ch["missing_policy"],
                   int(ch["max_missing_to_interpolate"]),
                   self.test_cfg.get("min_duration_s", 130.0),
                   clf.get("seed", 20261006),
                   clf.get("n_perm", 10000)))

    def _build_classification_command(self, exp: dict) -> list[str]:
        clf = self.clf_cfg
        tw = self.test_cfg["time_window"]
        ch = self.test_cfg["channels"]
        min_dur = float(self.test_cfg.get("min_duration_s", 130.0))
        train_dir = self._model_info[exp["model"]]["dir"]

        cmd = [
            sys.executable, "-m", self.pipeline_module,
            "--train-results-dir", str(train_dir),
            "--test-session", self.test_session,
            "--subject-ids",
        ] + [str(i) for i in self.test_subject_ids] + [
            "--tasks",
        ] + list(exp["tasks"]) + [
            "--t-start", str(tw["t_start"]),
            "--t-end", str(tw["t_end"]),
            "--method", str(clf.get("method", "energy")),
            "--rank-truncation", str(clf.get("rank_truncation", "min")),
            "--n-perm", str(clf.get("n_perm", 10000)),
            "--seed", str(clf.get("seed", 20261006)),
            "--missing-policy", str(ch["missing_policy"]),
            "--max-missing", str(int(ch["max_missing_to_interpolate"])),
            "--min-window", str(max(10.0, min_dur - float(tw["t_start"]))),
        ]
        if not clf.get("make_plots", True):
            cmd.append("--skip-plots")
        if self.db_path:
            cmd.extend(["--db-path", self.db_path])
        # Cache ML compartida entre experimentos del mismo modelo
        ml_cache = self.out_root() / "_ml_stats" / exp["model"]
        cmd.extend(["--ml-stats-cache", str(ml_cache)])
        out_dir = self._experiment_out_dir(exp)
        cmd.extend(["--out-dir", str(out_dir)])
        return cmd

    def _poll_child_rss(self, proc: subprocess.Popen,
                        interval: float = 0.5) -> tuple[float, list]:
        peak_rss = 0.0
        samples: list[tuple[float, float]] = []
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

    def _run_experiment(self, exp: dict) -> bool:
        key = self._experiment_checkpoint_key(exp)
        out_dir = self._experiment_out_dir(exp)
        results_path = out_dir / "classification_results.json"

        if key in self.checkpoint and results_path.exists():
            logger.info("[CLF] %s: ya completado (checkpoint + "
                        "results). SKIP.", exp["name"])
            return self._collect_experiment_result(exp) is not None

        if exp["model"] not in self._model_info:
            logger.error("[CLF] %s: modelo '%s' no disponible -> "
                         "BLOQUEADO", exp["name"], exp["model"])
            self._notify("error",
                         f"Experimento {exp['name']} BLOQUEADO: modelo "
                         f"{exp['model']} sin artefactos de train.",
                         title="[CD-HSA] Clasificacion BLOQUEADA")
            self._write_csv_log({
                "timestamp": datetime.now().isoformat(),
                "phase": "classify", "experiment": exp["name"],
                "model": exp["model"], "session": self.test_session,
                "tasks": "+".join(exp["tasks"]),
                "method": self.clf_cfg.get("method", "energy"),
                "success": False, "returncode": -1,
                "elapsed_s": 0.0, "accuracy": "", "n_ok": 0,
                "n_errors": 0, "peak_rss_mb": "",
                "command": "(blocked: missing training model)",
            })
            return False

        try:
            cmd = self._build_classification_command(exp)
        except Exception as exc:
            logger.error("[CLF] Error construyendo comando: %s", exc)
            self._notify("error",
                         f"Fallo construyendo comando de {exp['name']}: "
                         f"{exc}",
                         title="[CD-HSA] Clasificacion FALLO")
            return False

        logger.info("[CLF] RUN | %s | modelo %s | tareas %s | "
                    "test=%d sujetos",
                    exp["name"], exp["model"], "+".join(exp["tasks"]),
                    len(self.test_subject_ids))
        logger.debug("CMD: %s", " ".join(cmd))

        t0 = time.time()
        try:
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=child_env,
            )
        except Exception as exc:
            logger.error("[CLF] No se pudo lanzar el pipeline: %s", exc)
            self._notify("error",
                         f"No se pudo lanzar {exp['name']}: {exc}",
                         title="[CD-HSA] Clasificacion FALLO")
            return False

        _child_peak = [0.0]
        _child_samples: list[tuple[float, float]] = []

        def _poll_thread():
            pk, samps = self._poll_child_rss(proc)
            _child_peak[0] = pk
            _child_samples.extend(samps)

        _thr = threading.Thread(target=_poll_thread, daemon=True)
        _thr.start()

        for line in proc.stdout:
            logger.info("  [%s] %s", exp["name"], line.rstrip())
        proc.wait()
        _thr.join(timeout=3.0)

        elapsed = time.time() - t0
        success = proc.returncode == 0 and results_path.exists()
        peak_mb = _child_peak[0]

        self._job_memory_stats.append({
            "label": f"clf/{exp['name']}",
            "success": success,
            "elapsed_s": round(elapsed, 2),
            "peak_rss_mb": round(peak_mb, 1),
        })

        result = self._collect_experiment_result(exp) if success else None
        acc = result["accuracy"] if result else ""
        n_ok = result["n_ok"] if result else 0
        n_err = result["n_errors"] if result else 0

        self._write_csv_log({
            "timestamp": datetime.now().isoformat(),
            "phase": "classify", "experiment": exp["name"],
            "model": exp["model"], "session": self.test_session,
            "tasks": "+".join(exp["tasks"]),
            "method": self.clf_cfg.get("method", "energy"),
            "success": success, "returncode": proc.returncode,
            "elapsed_s": round(elapsed, 2),
            "accuracy": acc, "n_ok": n_ok, "n_errors": n_err,
            "peak_rss_mb": round(peak_mb, 1),
            "command": " ".join(cmd),
        })

        if success:
            self.checkpoint.add(key)
            self._save_checkpoint()
            logger.info("[CLF] OK | %s (%.1f s, peak %.1f MB, "
                        "accuracy=%s)", exp["name"], elapsed, peak_mb,
                        acc)
            self._notify("success",
                         f"Experimento {exp['name']}: accuracy "
                         f"{acc} ({n_ok} grabaciones)",
                         title=f"[CD-HSA] {exp['name']} OK")
        else:
            logger.error("[CLF] FALLO | %s (rc=%d, %.1f s)",
                         exp["name"], proc.returncode, elapsed)
            self._notify("error",
                         f"Experimento {exp['name']} FALLO "
                         f"(rc={proc.returncode}). Revisa el log.",
                         title=f"[CD-HSA] {exp['name']} FALLO")

        return success

    def _collect_experiment_result(self, exp: dict) -> dict | None:
        """Lee classification_results.json de un experimento."""
        out_dir = self._experiment_out_dir(exp)
        results_path = out_dir / "classification_results.json"
        if not results_path.exists():
            return None
        try:
            with open(results_path, "r", encoding="utf-8") as fh:
                results = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[CLF] %s: results ilegible: %s",
                           exp["name"], exc)
            return None
        methods = results.get("meta", {}).get("methods_computed", [])
        method = methods[0] if methods else "energy"
        m = results.get("metrics", {}).get(method, {})
        if not m:
            return None
        return {
            "name": exp["name"],
            "tasks": results["meta"]["tasks"],
            "method": method,
            "accuracy": round(float(m.get("accuracy", 0.0)) * 100, 1),
            "n_test": m.get("n_test", 0),
            "n_ok": results["meta"].get("n_ok", 0),
            "n_errors": results["meta"].get("n_errors", 0),
            "results": results,
        }

    # ------------------------------------------------------------------
    # FASE 2: cross-experimento
    # ------------------------------------------------------------------

    def _cross_checkpoint_key(self, completed: list[str]) -> str:
        clf = self.clf_cfg
        return ("cross|%s|%s|%s%s"
                % ("+".join(sorted(completed)),
                   self.test_session,
                   clf.get("method", "energy"),
                   _rank_suffix(clf)))

    def _run_cross_phase(self, completed: list[dict]) -> bool:
        out_dir = self.out_root() / "cross_experiment"
        out_dir.mkdir(parents=True, exist_ok=True)
        key = self._cross_checkpoint_key([c["name"] for c in completed])
        if key in self.checkpoint and (out_dir /
                                        "cross_experiment_summary.json"
                                        ).exists():
            logger.info("[CROSS] Ya completado (checkpoint).")
            return True

        logger.info("[CROSS] FASE 2: analisis cross-experimento "
                    "(%d experimentos)", len(completed))

        method = completed[0]["method"]
        cross = _aggregate_cross(completed, method)
        with open(out_dir / "cross_experiment_summary.json", "w",
                  encoding="utf-8") as fh:
            json.dump(cross, fh, indent=2)
        _write_cross_csv(cross, out_dir / "cross_subject_matrix.csv")
        text = _cross_summary_text(cross)
        with open(out_dir / "cross_summary.txt", "w",
                  encoding="utf-8") as fh:
            fh.write(text)
        logger.info("\n%s", text)
        try:
            _make_cross_plots(cross, out_dir)
            logger.info("[CROSS] Graficos en %s", out_dir)
        except Exception as exc:
            logger.warning("[CROSS] Fallo generando graficos: %s", exc)

        self._write_csv_log({
            "timestamp": datetime.now().isoformat(),
            "phase": "cross", "experiment": "+".join(
                c["name"] for c in completed),
            "model": "-", "session": self.test_session,
            "tasks": "-", "method": method, "success": True,
            "returncode": 0, "elapsed_s": 0.0,
            "accuracy": cross["overall"]["accuracy_percent"],
            "n_ok": cross["overall"]["n_test"], "n_errors": 0,
            "peak_rss_mb": "", "command": "",
        })

        self.checkpoint.add(key)
        self._save_checkpoint()
        self._notify("success",
                     f"Cross-experimento listo: "
                     f"{cross['overall']['accuracy_percent']}% overall "
                     f"({len(cross['subject_categories']['perfect'])} "
                     f"sujetos perfectos, "
                     f"{len(cross['subject_categories']['all_wrong'])} "
                     f"todos-mal)",
                     title="[CD-HSA] Cross-experimento OK")
        return True

    # ------------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------------

    def run(self, wait: bool = False, poll_interval: float = 300.0,
            wait_timeout: float = 0.0) -> int:
        self._print_banner()

        # FASE 1a: modelos de entrenamiento
        if not self.discover_training_models(
                wait=wait, poll_interval=poll_interval,
                wait_timeout=wait_timeout):
            self._notify("error",
                         "Batch abortado: faltan modelos de "
                         "entrenamiento SS. Verifica que ec_mu/ec_ma "
                         "terminaron.",
                         title="[CD-HSA] Batch ABORTADO")
            return 1
        try:
            self._validate_window_against_train()
        except ValueError as exc:
            logger.error("%s", exc)
            self._notify("error", str(exc),
                         title="[CD-HSA] Ventana train/test incompatible")
            return 1

        # FASE 0: auditoria de canales
        if not self._run_audit_phase():
            logger.error("La auditoria de canales fallo (sin datos "
                         "del dataset?). Abortando.")
            self._notify("error",
                         "Auditoria de canales fallo: revisa DB path y "
                         "estructura de session2.",
                         title="[CD-HSA] Auditoria FALLO")
            return 1

        # FASE 1: experimentos
        completed: list[dict] = []
        n_fail = 0
        for exp in self.params["experiments"]:
            if self.delay > 0 and completed:
                time.sleep(self.delay)
            ok = self._run_experiment(exp)
            if ok:
                r = self._collect_experiment_result(exp)
                if r is not None:
                    completed.append(r)
            else:
                n_fail += 1

        if not completed:
            logger.error("Ningun experimento completo; no hay "
                         "cross-analisis.")
            self._notify("error",
                         "Clasificacion sesion 2: NINGUN experimento "
                         "completo.",
                         title="[CD-HSA] Clasificacion FALLO")
            return 1

        # FASE 2: cross-experimento
        cross_ok = True
        if self.cross_cfg.get("enabled", True):
            cross_ok = self._run_cross_phase(completed)

        # Resumen final
        self._print_memory_report()
        logger.info("=" * 70)
        logger.info("  RESUMEN FINAL")
        logger.info("=" * 70)
        for r in completed:
            logger.info("  %-12s : accuracy %s%% (%d grabaciones, "
                        "metodo %s)", r["name"], r["accuracy"],
                        r["n_ok"], r["method"])
        if n_fail:
            logger.info("  Experimentos fallidos/bloqueados: %d", n_fail)
        logger.info("  Salidas         : %s", self.out_root())
        logger.info("=" * 70)

        if n_fail or not cross_ok:
            self._notify("warning",
                         f"Batch sesion 2 terminado CON problemas: "
                         f"{len(completed)} OK, {n_fail} fallidos. "
                         f"Ver {self.out_root()}",
                         title="[CD-HSA] Batch terminado con avisos")
            return 1

        accs = ", ".join(f"{r['name']}={r['accuracy']}%" for r in completed)
        self._notify("success",
                     f"Clasificacion sesion 2 completa: {accs}. "
                     f"Salidas: {self.out_root()}",
                     title="[CD-HSA] Clasificacion sesion 2 OK")
        return 0

    def _print_memory_report(self) -> None:
        if not self._job_memory_stats:
            return
        stats = sorted(self._job_memory_stats,
                       key=lambda s: s["peak_rss_mb"], reverse=True)
        logger.info("")
        logger.info("=" * 70)
        logger.info("  MEMORY REPORT POR JOB (peak RSS)")
        logger.info("=" * 70)
        for s in stats:
            logger.info("  %-24s %8.1f MB %6.1f s   %s",
                        s["label"], s["peak_rss_mb"], s["elapsed_s"],
                        "OK" if s["success"] else "FAIL")
        logger.info("=" * 70)


# ===========================================================================
# Cross-experimento: agregacion, textos y plots
# (funciones modulo: testeables de forma independiente)
# ===========================================================================

# Paleta segura para daltonicos (Okabe-Ito) + serie por clase
_C_OK = "#0072B2"
_C_BAD = "#D55E00"
_C_GREY = "#7F7F7F"
_CLASS_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def _aggregate_cross(completed: list[dict], method: str) -> dict:
    """Agrega los resultados de todos los experimentos completados.

    ``completed``: lista de dicts tal como los devuelve
    ``_collect_experiment_result`` (clave ``results`` = JSON completo).
    """
    exp_names = [c["name"] for c in completed]
    per_experiment: dict[str, dict] = {}
    for c in completed:
        m = c["results"]["metrics"].get(method, {})
        per_experiment[c["name"]] = {
            "tasks": c["tasks"],
            "accuracy": m.get("accuracy"),
            "accuracy_percent": (round(m["accuracy"] * 100, 1)
                                 if m else None),
            "chance_percent": (round(m["chance_level"] * 100, 1)
                               if m else None),
            "ci95": m.get("ci95_clopper_pearson"),
            "n_test": m.get("n_test"),
            "auc": m.get("auc"),
            "p_binomial": m.get("binomial_p_exact_onesided"),
            "p_perm": m.get("perm_p_paired_within_subject"),
        }

    # Matriz por sujeto: {(subject, exp): {n_rec, n_correct}}.
    # Solo cuentan las grabaciones PROCESADADAS (status == 'ok'): las
    # saltadas (canales/fichero/duracion) no son 'errores de
    # clasificacion' y no deben mandar al sujeto a 'all_wrong'.
    # Las filas cubren TODOS los sujetos testeados (los que quedaron
    # sin grabaciones usables caen en la categoria 'no_data').
    all_subjects = sorted({
        int(sid)
        for c in completed
        for sid in c["results"]["meta"].get("test_subject_ids", [])
    })
    matrix: dict[int, dict[str, dict]] = {
        s: {e: {"n_recordings": 0, "n_correct": 0} for e in exp_names}
        for s in all_subjects
    }
    for c in completed:
        for rec in c["results"]["recordings"]:
            if rec.get("status") != "ok":
                continue
            s = int(rec["subject_id"])
            if s not in matrix:
                matrix[s] = {e: {"n_recordings": 0, "n_correct": 0}
                             for e in exp_names}
            cell = matrix[s][c["name"]]
            cell["n_recordings"] += 1
            sc = rec.get("scores", {}).get(method)
            if sc is None:
                continue
            pred = int(np.argmax(np.asarray(sc["raw"],
                                            dtype=np.float64)))
            if pred == rec["true"]:
                cell["n_correct"] += 1

    # Categorias por sujeto
    perfect: list[int] = []
    all_wrong: list[int] = []
    mixed: list[int] = []
    no_data: list[int] = []
    for s in sorted(set(all_subjects) | set(matrix.keys())):
        row = matrix.get(s)
        if row is None:
            no_data.append(s)
            continue
        n_rec = sum(cell["n_recordings"] for cell in row.values())
        n_ok = sum(cell["n_correct"] for cell in row.values())
        if n_rec == 0:
            no_data.append(s)
        elif n_ok == n_rec:
            perfect.append(s)
        elif n_ok == 0:
            all_wrong.append(s)
        else:
            mixed.append(s)

    # Errores por tarea y experimento
    per_task_error: dict[str, dict[str, dict]] = {
        e: {} for e in exp_names
    }
    for c in completed:
        tasks = c["tasks"]
        for rec in c["results"]["recordings"]:
            if rec.get("status") != "ok":
                continue
            sc = rec.get("scores", {}).get(method)
            if sc is None:
                continue
            task = rec["task"]
            slot = per_task_error[c["name"]].setdefault(
                task, {"n": 0, "n_error": 0})
            slot["n"] += 1
            pred = int(np.argmax(np.asarray(sc["raw"],
                                            dtype=np.float64)))
            if pred != rec["true"]:
                slot["n_error"] += 1
        for task in tasks:
            per_task_error[c["name"]].setdefault(
                task, {"n": 0, "n_error": 0})

    n_test_total = sum(
        per_experiment[e]["n_test"] or 0 for e in exp_names)
    n_ok_total = 0
    for c in completed:
        m = c["results"]["metrics"].get(method, {})
        n_ok_total += int(m.get("n_correct", 0))

    # Contingencia 2x2 (solo con exactamente 2 experimentos):
    # (correcto en todos sus datos del exp A) x (idem B)
    contingency: dict | None = None
    if len(exp_names) == 2:
        eA, eB = exp_names
        cont = np.zeros((2, 2), dtype=int)  # [A correct?][B correct?]
        for s, row in matrix.items():
            cells = [row[eA], row[eB]]
            if any(cell["n_recordings"] == 0 for cell in cells):
                continue
            ia = int(cells[0]["n_correct"] == cells[0]["n_recordings"])
            ib = int(cells[1]["n_correct"] == cells[1]["n_recordings"])
            cont[ia, ib] += 1
        contingency = {
            "experiments": [eA, eB],
            "labels": [f"all-correct in {eA}", f"all-correct in {eB}"],
            "matrix": cont.tolist(),
        }

    subjects_list = sorted(matrix.keys())
    return {
        "method": method,
        "test_session": completed[0]["results"]["meta"].get("session"),
        "experiments": exp_names,
        "per_experiment": per_experiment,
        "subjects": subjects_list,
        "matrix": {str(s): {e: matrix[s][e] for e in exp_names}
                   for s in subjects_list},
        "subject_categories": {
            "perfect": perfect,
            "all_wrong": all_wrong,
            "mixed": mixed,
            "no_data": no_data,
        },
        "per_task_error": {
            e: {t: {"n": v["n"], "error_rate":
                    (v["n_error"] / v["n"] if v["n"] else None)}
                for t, v in per_task_error[e].items()}
            for e in exp_names
        },
        "contingency_2x2": contingency,
        "overall": {
            "n_test": n_test_total,
            "n_correct": n_ok_total,
            "accuracy": (n_ok_total / n_test_total
                         if n_test_total else None),
            "accuracy_percent": (round(100.0 * n_ok_total / n_test_total,
                                       1)
                                 if n_test_total else None),
        },
    }


def _write_cross_csv(cross: dict, path: Path) -> None:
    exps = cross["experiments"]
    fields = ["subject_id"] + [
        f"{e}_n_recordings" for e in exps
    ] + [f"{e}_n_correct" for e in exps] + ["category"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        cat_by_subj = {}
        for cat in ("perfect", "all_wrong", "mixed", "no_data"):
            for s in cross["subject_categories"][cat]:
                cat_by_subj[s] = cat
        for s in cross["subjects"]:
            row = {"subject_id": s}
            for e in exps:
                cell = cross["matrix"][str(s)][e]
                row[f"{e}_n_recordings"] = cell["n_recordings"]
                row[f"{e}_n_correct"] = cell["n_correct"]
            row["category"] = cat_by_subj.get(s, "mixed")
            writer.writerow(row)


def _cross_summary_text(cross: dict) -> str:
    lines = [
        "",
        "=" * 70,
        "  RESUMEN CROSS-EXPERIMENTO (sesion "
        f"{cross.get('test_session')})",
        "=" * 70,
        f"  Metodo      : {cross['method']}",
        f"  Overall     : {cross['overall']['n_correct']}/"
        f"{cross['overall']['n_test']} = "
        f"{cross['overall']['accuracy_percent']}%",
        "-" * 70,
        "  Por experimento:",
    ]
    for e, pe in cross["per_experiment"].items():
        ci = pe["ci95"] or (0.0, 1.0)
        auc = pe["auc"] if pe["auc"] is not None else float("nan")
        lines.append(
            f"    {e:<12}: {pe['accuracy_percent']}% "
            f"(azar {pe['chance_percent']}%, "
            f"IC95 [{ci[0] * 100:.1f}, "
            f"{ci[1] * 100:.1f}]%, "
            f"AUC {auc:.3f}, n={pe['n_test']})")
    lines.append("-" * 70)
    cats = cross["subject_categories"]
    lines.append(f"  Sujetos PERFECTOS (correctos en TODOS los "
                 f"experimentos): {len(cats['perfect'])}")
    lines.append(f"    {cats['perfect']}")
    lines.append(f"  Sujetos TODO-MAL (ninguno correcto): "
                 f"{len(cats['all_wrong'])}")
    lines.append(f"    {cats['all_wrong']}")
    lines.append(f"  Sujetos MIXTOS: {len(cats['mixed'])}")
    lines.append(f"  Sin datos    : {len(cats['no_data'])}")
    lines.append("-" * 70)
    lines.append("  Tasa de error por tarea:")
    for e in cross["experiments"]:
        row = ", ".join(
            f"{t}={v['error_rate'] * 100:.1f}%"
            if v["error_rate"] is not None else f"{t}=n/a"
            for t, v in cross["per_task_error"][e].items())
        lines.append(f"    {e:<12}: {row}")
    if cross.get("contingency_2x2"):
        c = cross["contingency_2x2"]
        lines.append("-" * 70)
        lines.append(f"  Contingencia {c['experiments'][0]} x "
                     f"{c['experiments'][1]} (filas/cols = "
                     f"all-correct?): {c['matrix']}")
    lines.append("=" * 70)
    return "\n".join(lines)


def _make_cross_plots(cross: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exps = cross["experiments"]
    written: list[Path] = []
    out_dir = Path(out_dir)

    # 1. Accuracy por experimento con IC + azar
    fig, ax = plt.subplots(figsize=(1.6 * len(exps) + 3.2, 4.4),
                           constrained_layout=True)
    xs = np.arange(len(exps))
    accs, los, his, chances = [], [], [], []
    for e in exps:
        pe = cross["per_experiment"][e]
        accs.append((pe["accuracy"] or 0.0) * 100)
        lo, hi = pe["ci95"] or (0.0, 1.0)
        los.append(accs[-1] - lo * 100)
        his.append(hi * 100 - accs[-1])
        chances.append((pe["chance_percent"] or 50.0))
    bars = ax.bar(xs, accs, yerr=[los, his], capsize=5, color=_C_OK,
                  error_kw={"ecolor": "#333333", "lw": 1.5})
    for x, a, ch in zip(xs, accs, chances):
        ax.text(x, a + 4, f"{a:.1f}%", ha="center", fontsize=11)
        ax.plot([x - 0.35, x + 0.35], [ch, ch], color=_C_BAD, lw=1.8,
                ls="--")
    ax.set_xticks(xs, labels=exps)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification accuracy per experiment\n"
                 "(dashed = chance level)")
    p = out_dir / "cross_accuracy_summary.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 2. Grid sujetos x experimentos
    subjects = cross["subjects"]
    M = np.full((len(subjects), len(exps)), np.nan)
    for i, s in enumerate(subjects):
        for j, e in enumerate(exps):
            cell = cross["matrix"][str(s)][e]
            if cell["n_recordings"] > 0:
                M[i, j] = cell["n_correct"] / cell["n_recordings"]
    fig, ax = plt.subplots(
        figsize=(2.2 + 1.1 * len(exps),
                 max(3.0, 0.24 * len(subjects) + 1.6)),
        constrained_layout=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "wr", [_C_BAD, "#FFFFFF", _C_OK])
    Mc = np.ma.masked_invalid(M)
    im = ax.imshow(Mc, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(exps)), labels=exps)
    step = max(1, len(subjects) // 30)
    ax.set_yticks(range(0, len(subjects), step),
                  labels=[f"{s:02d}" for s in subjects[::step]],
                  fontsize=6)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Subject")
    ax.set_title("Correct-recording fraction per subject and experiment\n"
                 "(blue = all correct, orange = all wrong, "
                 "white = mixed, blank = no data)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="correct fraction")
    p = out_dir / "cross_subject_grid.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 3. Consistencia: perfectos / mixtos / todos-mal + contingencia
    cats = cross["subject_categories"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2),
                             constrained_layout=True)
    ax = axes[0]
    vals = [len(cats["perfect"]), len(cats["mixed"]),
            len(cats["all_wrong"]), len(cats["no_data"])]
    labels = [f"Perfect\n(all correct)\nn={vals[0]}",
              f"Mixed\nn={vals[1]}",
              f"All wrong\nn={vals[2]}", f"No data\nn={vals[3]}"]
    ax.bar(labels, vals, color=[_C_OK, _C_GREY, _C_BAD, "#DDDDDD"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.15, str(v), ha="center", fontsize=11)
    ax.set_ylabel("Subjects")
    ax.set_title("Subject consistency across experiments")
    ax = axes[1]
    if cross.get("contingency_2x2"):
        c = cross["contingency_2x2"]
        cont = np.asarray(c["matrix"], dtype=float)
        ax.imshow(cont, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                color = "white" if cont[i, j] > cont.max() * 0.6 \
                    else "black"
                ax.text(j, i, int(cont[i, j]), ha="center", va="center",
                        color=color, fontsize=14)
        ax.set_xticks([0, 1], labels=["no", "yes"])
        ax.set_yticks([0, 1], labels=["no", "yes"])
        ax.set_xlabel(f"All correct in {c['experiments'][1]}")
        ax.set_ylabel(f"All correct in {c['experiments'][0]}")
        ax.set_title("2x2 subject contingency")
    else:
        ax.text(0.5, 0.5, "(2x2 disponible solo con 2 experimentos)",
                ha="center", va="center", transform=ax.transAxes,
                color=_C_GREY)
        ax.set_xticks([])
        ax.set_yticks([])
    p = out_dir / "cross_subject_consistency.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 4. Errores por tarea
    fig, ax = plt.subplots(figsize=(1.5 * len(exps) + 4.0, 4.2),
                           constrained_layout=True)
    all_tasks: list[str] = []
    for e in exps:
        for t in cross["per_task_error"][e]:
            if t not in all_tasks:
                all_tasks.append(t)
    width = 0.8 / len(exps)
    for j, e in enumerate(exps):
        xs_t = np.arange(len(all_tasks)) + (j - len(exps) / 2 + 0.5) * width
        errs = []
        for t in all_tasks:
            v = cross["per_task_error"][e].get(t)
            errs.append((v["error_rate"] * 100
                         if v and v["error_rate"] is not None
                         else np.nan))
        ax.bar(xs_t, errs, width=width * 0.92,
               color=_CLASS_COLORS[j % len(_CLASS_COLORS)], label=e)
    ax.set_xticks(range(len(all_tasks)), labels=all_tasks)
    ax.set_ylabel("Error rate (%)")
    ax.set_title("Error rate per task and experiment")
    ax.legend(frameon=False)
    ax.axhline(0, color="black", lw=0.8)
    p = out_dir / "cross_error_by_task.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    return written


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Batch de clasificacion sesion 2: modelos CD-HSA "
                     "entrenados con SS (sesion 1) contra sujetos "
                     "individuales de la sesion 2. Fases: auditoria de "
                     "canales -> experimentos -> cross-experimento."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--params-json", type=str, default=str(DEFAULT_PARAMS_JSON),
        help=("JSON de parametros (default: "
              "cdhsa_batch_params_session2_classification.json junto a "
              "este script)."),
    )
    parser.add_argument("--db-path", type=str, default=None,
                        help="Override de la raiz del dataset Gedai.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override de BASE_RESULTS_PATH.")
    parser.add_argument("--training-root", type=str, default=None,
                        help=("Override de la raiz de resultados de los "
                              "entrenamientos SS (default: "
                              "BASE_RESULTS_PATH/cdhsa)."))
    parser.add_argument("--pipeline-module", type=str, default=None,
                        help=("Modulo del pipeline de clasificacion "
                              "(default: "
                              "src.pipelines.run_cdhsa_session2_classification)."))
    parser.add_argument("--force-audit", action="store_true",
                        help="Re-ejecutar la fase de auditoria.")
    parser.add_argument("--wait-for-training", action="store_true",
                        help=("Esperar (polling) a que los entrenamientos "
                              "SS requeridos terminen (aparece "
                              "cdhsa_arrays.npz) antes de empezar."))
    parser.add_argument("--poll-interval", type=float, default=300.0,
                        help="Segundos entre chequeos de --wait-for-training.")
    parser.add_argument("--wait-timeout", type=float, default=0.0,
                        help=("Timeout (s) de la espera; 0 = infinito."))
    return parser.parse_args(argv)


@notify_on_critical_error(channel=_NTFY_CHANNEL or None,
                          title="[CD-HSA] ERROR critico del batch "
                                "sesion 2",
                          catch_system_exit=True)
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    params_path = Path(args.params_json)
    if not params_path.exists():
        logger.error("JSON de parametros no encontrado: %s", params_path)
        return 1
    try:
        params = _load_params(params_path)
    except ValueError:
        return 1

    runner = Session2ClassificationBatchRunner(
        params, pipeline_module=args.pipeline_module,
        force_audit=args.force_audit)

    if args.db_path:
        runner.db_path = args.db_path
    if args.out_dir:
        runner.output_dir = Path(args.out_dir)
        runner.training_root = (Path(args.training_root)
                               if args.training_root
                               else Path(args.out_dir) / "cdhsa")
    elif args.training_root:
        runner.training_root = Path(args.training_root)

    logger.info("Batch clasificacion sesion 2 -- params: %s", params_path)

    return runner.run(wait=args.wait_for_training,
                      poll_interval=args.poll_interval,
                      wait_timeout=args.wait_timeout)


if __name__ == "__main__":
    sys.exit(main())
