#!/usr/bin/env python3
"""
run_discriminant_analysis.py
=============================
Extrae direcciones de maxima discriminancia por tarea para cada
super-sujeto y las compara entre super-sujetos para evaluar
robustez.

Para cada super-sujeto:
  1. Construye matrices de Hankel para todas las tareas
  2. Calcula la covarianza por tarea en el espacio de Hankel
  3. Para cada tarea c, calcula  Delta_c = Sigma_c - Sigma_pool
     donde Sigma_pool es la covarianza promediada entre tareas
  4. SVD de Delta_c -> las k direcciones principales son las de
     maxima discriminancia (las que mas cambian entre la tarea c
     y el promedio de todas las tareas)
  5. Guarda las direcciones, valores singulares y metadatos

Despues de todos los super-sujetos:
  6. Para cada tarea, compara los subespacios de discriminancia
     entre pares de super-sujetos usando angulos principales
  7. Genera un reporte de consistencia

Consumo de memoria: solo las Hankel (sin block-Hankel de CD-HSA).
Con p=53, depth=10, 12 sujetos, t_end=60: ~1.5 GB por tarea,
~6 GB total por SS. Procesamiento secuencial por SS.

Configuracion: JSON (default: discriminant_analysis_params.json)
  Uso: python run_discriminant_analysis.py --params-json mi_config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray

# --- Script directory (para importar build_hankel_single_ss) ---
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

DEFAULT_PARAMS_JSON = _SCRIPT_DIR / "discriminant_analysis_params.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =====================================================================
# CORE: DIRECCIONES DE MAXIMA DISCRIMINANCIA
# =====================================================================


def compute_discriminant_directions(
    H_list: list[NDArray],
    task_names: list[str],
    k: int = 2,
) -> dict[str, dict[str, Any]]:
    """Direcciones de maxima discriminancia por tarea.

    Para cada tarea c:
      1. Sigma_c = H_c @ H_c^T / K_c   (covarianza en espacio Hankel)
      2. Sigma_pool = promedio(Sigma_1, ..., Sigma_C)
      3. Delta_c = Sigma_c - Sigma_pool
      4. SVD(Delta_c) = U S V^T
      5. Direcciones = U[:, :k]  (las k de mayor singular value)

    Parameters
    ----------
    H_list : list[NDArray]
        Lista de C matrices de Hankel, cada una (d, K_c).
    task_names : list[str]
        Nombres de las C tareas.
    k : int
        Cantidad de direcciones de discriminancia por tarea.

    Returns
    -------
    dict[str, dict]
        ``{task_name: {"U": (d, k), "S": (k,),
        "explained_var_ratio": (k,), "Delta_norm": float}}``
    """
    C = len(H_list)
    d = H_list[0].shape[0]

    # Covarianzas (segundo momento: H @ H^T / K)
    # Nota: los datos EEG estan en Voltios (~10 uV), por lo que
    # los elementos de la covarianza son ~1e-12 V^2.  Las
    # diferencias entre tareas son una fraccion de eso.
    covs: list[NDArray] = []
    Ks: list[int] = []
    for H in H_list:
        K = H.shape[1]
        Ks.append(K)
        covs.append(H @ H.T / K)  # (d, d)

    # Covarianza pooled (promedio simple)
    cov_pool = np.mean(covs, axis=0)  # (d, d)
    pool_norm = float(np.linalg.norm(cov_pool, "fro"))

    # Diagnostico de escala
    cov_norms = [float(np.linalg.norm(c, "fro")) for c in covs]
    logger.info(
        "    Escala: ||Sigma_pool||_F = %.3e  |  "
        "||Sigma_c||_F por tarea: %s",
        pool_norm,
        [f"{n:.3e}" for n in cov_norms],
    )

    results: dict[str, dict[str, Any]] = {}
    for c, (cov_c, name) in enumerate(zip(covs, task_names)):
        Delta = cov_c - cov_pool  # (d, d)
        Delta_norm = float(np.linalg.norm(Delta, "fro"))

        # Fraccion relativa de la covarianza pooled
        rel_delta = Delta_norm / pool_norm if pool_norm > 0 else 0.0

        # SVD completo (d es ~530, esto es instantaneo)
        U, S, _Vt = np.linalg.svd(Delta, full_matrices=False)

        # Varianza explicada por las direcciones
        total_var = float(np.sum(S ** 2))
        if total_var > 0:
            explained_ratio = (S ** 2) / total_var
        else:
            explained_ratio = np.zeros_like(S)

        # Seleccionar k direcciones
        kk = min(k, len(S))
        results[name] = {
            "U": U[:, :kk].copy(),           # (d, k) direcciones
            "S": S[:kk].copy(),               # (k,) singular values
            "explained_var_ratio": explained_ratio[:kk].copy(),
            "Delta_norm": Delta_norm,
            "Delta_fro_all": float(total_var ** 0.5),
            "rel_delta": rel_delta,
            "pool_norm": pool_norm,
        }

        logger.debug(
            "  Tarea %-12s  ||Delta||_F=%.3e  rel=%.4f  "
            "top-%d SVD: %s  expl_var: %s",
            name, Delta_norm, rel_delta, kk,
            [f"{s:.3e}" for s in S[:kk]],
            [f"{r:.4f}" for r in explained_ratio[:kk]],
        )

    return results


# =====================================================================
# COMPARACION CRUZADA: ANGULOS PRINCIPALES
# =====================================================================


def _principal_angles_between(
    U_a: NDArray, U_b: NDArray,
) -> tuple[NDArray, NDArray]:
    """Angulos principales entre dos subespacios de igual dimension.

    Parameters
    ----------
    U_a, U_b : NDArray, shape (d, k)
        Bases ortonormales de los subespacios.

    Returns
    -------
    angles_deg : NDArray, shape (k,)
        Angulos principales en grados.
    singular_values : NDArray, shape (k,)
        Valores singulares de U_a^T @ U_b (cosas de los angulos).
    """
    C_mat = U_a.T @ U_b  # (k, k)
    sv = np.linalg.svd(C_mat, compute_uv=False)
    sv_clipped = np.clip(sv, -1.0, 1.0)
    angles_rad = np.arccos(sv_clipped)
    return np.degrees(angles_rad), sv


def compare_discriminant_subspaces(
    all_results: dict[int, dict[str, dict]],
    task_names: list[str],
) -> dict[str, dict[tuple[int, int], dict[str, float]]]:
    """Compara subespacios de discriminancia entre pares de super-sujetos.

    Para cada tarea y cada par (i, j) de SS, calcula los angulos
    principales entre sus k-direcciones de discriminancia.

    Returns
    -------
    dict  ``{task: {(i,j): {"angles_deg": [...], "mean_angle": ...,
                        "max_angle": ..., "subspace_corr": ...}}}``
    """
    ss_ids = sorted(all_results.keys())
    comparison: dict[str, dict[tuple[int, int], dict[str, float]]] = {}

    for task in task_names:
        comparison[task] = {}
        for idx_i in range(len(ss_ids)):
            for idx_j in range(idx_i + 1, len(ss_ids)):
                ss_i = ss_ids[idx_i]
                ss_j = ss_ids[idx_j]

                U_i = all_results[ss_i][task]["U"]  # (d, k)
                U_j = all_results[ss_j][task]["U"]  # (d, k)

                # Verificar dimensiones compatibles
                if U_i.shape[0] != U_j.shape[0]:
                    comparison[task][(ss_i, ss_j)] = {
                        "error": f"incompatible d: {U_i.shape[0]} vs {U_j.shape[0]}",
                        "mean_angle_deg": 999.0,
                        "max_angle_deg": 999.0,
                        "subspace_correlation": 0.0,
                    }
                    continue

                angles_deg, sv = _principal_angles_between(U_i, U_j)

                comparison[task][(ss_i, ss_j)] = {
                    "singular_values": [float(x) for x in sv],
                    "angles_deg": [float(x) for x in angles_deg],
                    "mean_angle_deg": float(np.mean(angles_deg)),
                    "max_angle_deg": float(np.max(angles_deg)),
                    "subspace_correlation": float(np.mean(sv)),
                }

    return comparison


# =====================================================================
# REPORTES
# =====================================================================


def _consistency_label(mean_angle: float) -> str:
    """Clasificacion de consistencia segun angulo medio."""
    if mean_angle < 10:
        return "ALTA"
    elif mean_angle < 20:
        return "MODERADA"
    elif mean_angle < 35:
        return "BAJA"
    else:
        return "MUY BAJA"


def _format_ss_report(
    ss_id: int,
    directions: dict[str, dict],
    task_names: list[str],
    k: int,
    hankel_info: dict,
) -> str:
    """Reporte textual por super-sujeto."""
    lines: list[str] = []
    lines.append("")
    lines.append(f"{'=' * 70}")
    lines.append(f"  SUPER-SUJETO {ss_id}  --  Direcciones de Maxima Discriminancia")
    lines.append(f"{'=' * 70}")

    d = list(directions.values())[0]["U"].shape[0]
    lines.append(f"  d = {d}  (p x depth),  k = {k}")
    if hankel_info.get("global_channels"):
        lines.append(
            f"  canales ({len(hankel_info['global_channels'])}): "
            f"{hankel_info['global_channels'][:5]}..."
        )
    lines.append(f"  sujetos: {hankel_info.get('member_ids', '?')}")
    lines.append("")

    for task in task_names:
        if task not in directions:
            lines.append(f"  {task:<15}  [NO DATA]")
            continue
        res = directions[task]
        U = res["U"]
        S = res["S"]
        evr = res["explained_var_ratio"]
        lines.append(f"  {task}:")
        lines.append(f"    ||Delta||_F     = {res['Delta_norm']:.3e}")
        rel = res.get("rel_delta", 0.0)
        pool_n = res.get("pool_norm", 0.0)
        lines.append(
            f"    ||Sigma_pool||  = {pool_n:.3e}"
            f"    (rel: {rel*100:.4f}% del pool)"
        )
        for i in range(min(k, len(S))):
            lines.append(
                f"    dir {i+1}: sigma={S[i]:.3e}  "
                f"expl_var={evr[i]*100:.1f}%  "
                f"U[:,{i}] norm={np.linalg.norm(U[:, i]):.6f}"
            )
        lines.append("")

    # Cross-task angles (discriminancia entre tareas)
    lines.append(f"  --- Angulos entre subespacios de tareas ---")
    for i_t, t1 in enumerate(task_names):
        for t2 in task_names[i_t + 1:]:
            if t1 not in directions or t2 not in directions:
                continue
            angles, sv = _principal_angles_between(
                directions[t1]["U"], directions[t2]["U"],
            )
            lines.append(
                f"    {t1:<12} vs {t2:<12}: "
                f"angulos=[{', '.join(f'{a:.1f}' for a in angles)}]  "
                f"mean={np.mean(angles):.1f} deg  "
                f"corr={np.mean(sv):.4f}"
            )

    return "\n".join(lines)


def _format_comparison_report(
    comparison: dict,
    task_names: list[str],
    ss_ids: list[int],
    k: int,
    experiment_label: str,
) -> str:
    """Reporte de comparacion cruzada entre super-sujetos."""
    lines: list[str] = []
    lines.append("")
    lines.append(f"{'=' * 70}")
    lines.append("  COMPARACION CRUZADA DE SUBESPACIOS DE DISCRIMINANCIA")
    lines.append(f"{'=' * 70}")
    lines.append(f"  Experimento : {experiment_label}")
    lines.append(f"  Tareas      : {', '.join(task_names)}")
    lines.append(f"  k (dirs/tarea) : {k}")
    lines.append(f"  Super-sujetos  : {ss_ids}")
    lines.append(f"  Generado   : {datetime.now().isoformat()}")
    lines.append("")

    # Por tarea
    summary_rows: list[str] = []
    for task in task_names:
        lines.append(f"  --- {task} ---")
        pairs = comparison.get(task, {})
        if not pairs:
            lines.append("    (sin datos)")
            summary_rows.append(
                f"  {task:<15}  SIN DATOS"
            )
            continue

        mean_angles_all: list[float] = []
        for (ss_i, ss_j), info in sorted(pairs.items()):
            if "error" in info:
                lines.append(
                    f"    SS{ss_i} vs SS{ss_j}: ERROR - {info['error']}"
                )
                continue
            angles = info["angles_deg"]
            mean_a = info["mean_angle_deg"]
            max_a = info["max_angle_deg"]
            corr = info["subspace_correlation"]
            mean_angles_all.append(mean_a)
            lines.append(
                f"    SS{ss_i} vs SS{ss_j}: "
                f"angulos=[{', '.join(f'{a:.1f}' for a in angles)}]  "
                f"mean={mean_a:.1f} deg  max={max_a:.1f} deg  "
                f"corr={corr:.4f}"
            )

        if mean_angles_all:
            overall_mean = np.mean(mean_angles_all)
            overall_std = np.std(mean_angles_all)
            label = _consistency_label(overall_mean)
            lines.append(
                f"    -> Consistencia {label} "
                f"(mean={overall_mean:.1f} deg +/- {overall_std:.1f})"
            )
            summary_rows.append(
                f"  {task:<15}  {label:<12} "
                f"mean={overall_mean:.1f} +/- {overall_std:.1f} deg"
            )
        else:
            summary_rows.append(f"  {task:<15}  SIN PARES VALIDOS")
        lines.append("")

    # Tabla resumen
    lines.append(f"  {'=' * 60}")
    lines.append("  RESUMEN DE CONSISTENCIA")
    lines.append(f"  {'=' * 60}")
    for row in summary_rows:
        lines.append(row)
    lines.append("")

    # Interpretacion global
    lines.append("  INTERPRETACION:")
    lines.append(
        "    - Angulo medio < 10 deg  : direcciones casi identicas"
        " entre SS -> robustez ALTA"
    )
    lines.append(
        "    - Angulo medio 10-20 deg : direcciones similares con"
        " variacion moderada -> robustez MODERADA"
    )
    lines.append(
        "    - Angulo medio 20-35 deg : diferencias apreciables"
        " -> robustez BAJA, posible sobreajuste"
    )
    lines.append(
        "    - Angulo medio > 35 deg  : direcciones muy diferentes"
        " -> robustez MUY BAJA, las direcciones no son reproducibles"
    )

    return "\n".join(lines)


# =====================================================================
# GUARDADO DE RESULTADOS
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


def save_ss_results(
    out_dir: Path,
    ss_id: int,
    directions: dict[str, dict],
    hankel_info: dict,
    params: dict,
    k: int,
    task_names: list[str],
) -> None:
    """Guardar resultados de un super-sujeto.

    Archivos generados::

        directions.npz           U_{task} (d,k), S_{task} (k,),
                                explained_var_ratio_{task} (k,)
        discriminant_meta.json   Metadatos para el pipeline downstream
        hankel_info.json         Info de construccion de Hankel
        ss_report.txt            Reporte textual por SS
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  Guardando en: %s", out_dir)

    # --- 1. directions.npz ---
    npz_dict: dict[str, NDArray] = {}
    for task in task_names:
        if task in directions:
            res = directions[task]
            # Sanitizar nombre de tarea para clave
            safe = task.replace("-", "_")
            npz_dict[f"U_{safe}"] = res["U"]                # (d, k)
            npz_dict[f"S_{safe}"] = res["S"]                # (k,)
            npz_dict[f"evr_{safe}"] = res["explained_var_ratio"]  # (k,)
    np.savez_compressed(out_dir / "directions.npz", **npz_dict)
    logger.info("    [OK] directions.npz  (%d arrays)", len(npz_dict))

    # --- 2. discriminant_meta.json ---
    d = list(directions.values())[0]["U"].shape[0] if directions else 0
    meta = {
        "analysis_type": "discriminant_directions",
        "super_subject_id": ss_id,
        "k": k,
        "d": d,
        "task_names": task_names,
        "global_channels": hankel_info.get("global_channels", []),
        "hankel_depth": hankel_info.get("depths_used", [None])[0],
        "sfreq": hankel_info.get("sfreq_common"),
        "params_used": {
            "l_freq": params["discriminant_params"].get("l_freq"),
            "h_freq": params["discriminant_params"].get("h_freq"),
            "t_start": params["time_window"].get("t_start"),
            "t_end": params["time_window"].get("t_end"),
        },
        "per_task": {},
    }
    for task in task_names:
        if task in directions:
            res = directions[task]
            meta["per_task"][task] = {
                "Delta_norm": res["Delta_norm"],
                "singular_values": [float(x) for x in res["S"]],
                "explained_var_ratio": [
                    float(x) for x in res["explained_var_ratio"]
                ],
            }
    with open(out_dir / "discriminant_meta.json", "w") as fh:
        json.dump(_json_safe(meta), fh, indent=2)
    logger.info("    [OK] discriminant_meta.json")

    # --- 3. hankel_info.json ---
    with open(out_dir / "hankel_info.json", "w") as fh:
        json.dump(_json_safe(hankel_info), fh, indent=2, default=str)
    logger.info("    [OK] hankel_info.json")

    # --- 4. ss_report.txt ---
    report = _format_ss_report(ss_id, directions, task_names, k, hankel_info)
    with open(out_dir / "ss_report.txt", "w") as fh:
        fh.write(report)
    logger.info("    [OK] ss_report.txt")


def save_comparison(
    out_dir: Path,
    comparison: dict,
    task_names: list[str],
    ss_ids: list[int],
    k: int,
    experiment_label: str,
) -> Path:
    """Guardar reporte de comparacion cruzada.

    Genera::

        cross_ss_comparison.txt  Reporte textual completo
        cross_ss_comparison.json  Datos numericos para otro pipeline
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    report = _format_comparison_report(
        comparison, task_names, ss_ids, k, experiment_label,
    )
    report_path = out_dir / "cross_ss_comparison.txt"
    with open(report_path, "w") as fh:
        fh.write(report)
    logger.info("  Comparacion guardada: %s", report_path)

    # JSON para downstream
    json_path = out_dir / "cross_ss_comparison.json"
    with open(json_path, "w") as fh:
        json.dump(_json_safe(comparison), fh, indent=2)
    logger.info("  Comparacion JSON: %s", json_path)

    return report_path


# =====================================================================
# JSON LOADING
# =====================================================================


def _load_params(json_path: Path) -> dict:
    """Cargar y validar el JSON de parametros."""
    if not json_path.exists():
        logger.error("JSON no encontrado: %s", json_path)
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as fh:
        params = json.load(fh)

    # Validacion minima
    for key in ["super_subjects", "sessions", "tasks", "discriminant_params"]:
        if key not in params:
            logger.error("Falta la clave requerida '%s' en el JSON", key)
            sys.exit(1)

    ss_cfg = params["super_subjects"]
    if "selected" not in ss_cfg or not ss_cfg["selected"]:
        logger.error("'super_subjects.selected' debe ser una lista no vacia.")
        sys.exit(1)

    if "k" not in params["discriminant_params"]:
        logger.error("'discriminant_params.k' es requerido (dirs por tarea).")
        sys.exit(1)

    # Defaults opcionales
    params.setdefault("time_window", {})
    params.setdefault("execution", {})
    params["time_window"].setdefault("t_start", None)
    params["time_window"].setdefault("t_end", None)
    params["execution"].setdefault("max_workers", 1)
    params["execution"].setdefault("delay", 1.0)
    params["execution"].setdefault("run_comparison", True)

    return params


# =====================================================================
# BATCH RUNNER
# =====================================================================


class DiscriminantAnalysisRunner:
    """Orquesta el analisis de discriminancia por super-sujeto."""

    def __init__(self, params: dict) -> None:
        self.params = params
        self.ss_cfg = params["super_subjects"]
        self.task_names: list[str] = list(params["tasks"])
        self.sessions: list[str] = list(params["sessions"])
        self.k: int = params["discriminant_params"]["k"]
        self.exec_cfg = params["execution"]
        self.delay: float = self.exec_cfg.get("delay", 1.0)
        self.run_comparison: bool = self.exec_cfg.get("run_comparison", True)

        # Project paths
        self.output_dir: Path | None = None
        self.db_path: str | None = None
        self._resolve_project_paths()

        # Global channel intersection (two-pass)
        self.global_channels: list[str] | None = None
        if len(self.ss_cfg["selected"]) > 1:
            self.global_channels = self._discover_global_channels()

        # Checkpoint
        self.checkpoint: set[str] = self._load_checkpoint()

        # Results storage (para la comparacion final)
        # Estructura: {ss_id: {task: {U, S, ...}}}
        self.all_directions: dict[int, dict[str, dict]] = {}

        self._print_banner()

    # ------------------------------------------------------------------
    # Global channel intersection (two-pass)
    # ------------------------------------------------------------------

    def _discover_global_channels(self) -> list[str]:
        """Cargar headers de cada SS y calcular la interseccion global de canales.

        Usa preload=False para no cargar datos en memoria — solo lee
        los nombres de canales de cada super-sujeto.
        """
        from src.latent_space_extraction.super_subject_eeg import (
            load_super_subject_eeg,
        )

        selected = list(self.ss_cfg["selected"])
        session = self.sessions[0]
        first_task = self.task_names[0]
        sppss = self.ss_cfg.get("subjects_per_super_subject", 20)
        offset = self.ss_cfg.get("subject_start_offset", 1)
        tw = self.params["time_window"]

        logger.info("")
        logger.info("  PASS 0: Descubriendo canales globales...")
        logger.info("  " + "-" * 50)

        all_channel_sets: list[set[str]] = []

        for ss_id in selected:
            try:
                raw = load_super_subject_eeg(
                    super_subject_id=ss_id,
                    session=session,
                    task=first_task,
                    subjects_per_super_subject=sppss,
                    subject_start_offset=offset,
                    db_path=self.db_path,
                    t_start=tw.get("t_start"),
                    t_stop=tw.get("t_end"),
                    preload=False,
                    verbose=False,
                )
                ch_set = set(raw.ch_names)
                all_channel_sets.append(ch_set)
                logger.info(
                    "    SS%d: %d canales", ss_id, len(ch_set)
                )
                del raw
            except Exception as exc:
                logger.error(
                    "    SS%d: ERROR al cargar canales: %s", ss_id, exc
                )
                # Agregar set vacio para no romper la interseccion
                all_channel_sets.append(set())

        # Interseccion global
        global_ch = sorted(set.intersection(*all_channel_sets))

        logger.info("  " + "-" * 50)
        logger.info("    Interseccion global: %d canales", len(global_ch))

        # Reportar canales excluidos por SS
        for i, ss_id in enumerate(selected):
            n_total = len(all_channel_sets[i])
            n_kept = len(all_channel_sets[i] & set(global_ch))
            n_excl = n_total - n_kept
            if n_excl > 0:
                logger.info(
                    "    SS%d: %d canales excluidos (%d -> %d)",
                    ss_id, n_excl, n_total, n_kept,
                )

        return global_ch

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
        except ImportError:
            logger.info(
                "src.utils.config no disponible; usando rutas por defecto."
            )
            self.output_dir = Path("./results")

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        ss_cfg = self.ss_cfg
        dp = self.params["discriminant_params"]
        tw = self.params["time_window"]

        logger.info("=" * 70)
        logger.info("  ANALISIS DE DIRECCIONES DE MAXIMA DISCRIMINANCIA")
        logger.info("=" * 70)
        logger.info("  Super-sujetos  : %s", ss_cfg["selected"])
        logger.info("  Sujetos/SS    : %s",
                    ss_cfg.get("subjects_per_super_subject", "?"))
        logger.info("  Sesiones       : %s", self.sessions)
        logger.info("  Tareas         : %s", self.task_names)
        logger.info("  k (dirs/tarea) : %d", self.k)
        logger.info("  Ventana temporal: %s - %s s",
                    tw.get("t_start"), tw.get("t_end"))
        logger.info("  Filtro         : %.1f - %.1f Hz",
                    dp.get("l_freq", 1.0), dp.get("h_freq", 40.0))
        logger.info("  Hankel depth   : %s",
                    dp.get("hankel_depth", "auto"))
        logger.info("  Output dir     : %s", self.output_dir)
        if self.global_channels is not None:
            logger.info(
                "  Canales globales: %d (two-pass)",
                len(self.global_channels),
            )
        else:
            logger.info("  Canales globales: N/A (1 solo SS)")
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _checkpoint_path(self) -> Path:
        try:
            from src.utils.config import BASE_CACHE_PATH
            cp_dir = Path(BASE_CACHE_PATH)
        except ImportError:
            cp_dir = Path("./cache")
        return cp_dir / "batch_checkpoint_discriminant.json"

    def _load_checkpoint(self) -> set[str]:
        cp = self._checkpoint_path()
        if cp.exists():
            try:
                with open(cp, "r") as fh:
                    data = json.load(fh)
                ck = set(data.get("completed", []))
                logger.info(
                    "Checkpoint cargado: %d SS previos", len(ck)
                )
                return ck
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Checkpoint corrupto: %s", exc)
        return set()

    def _save_checkpoint(self) -> None:
        cp = self._checkpoint_path()
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            with open(cp, "w") as fh:
                json.dump(
                    {"completed": sorted(self.checkpoint)},
                    fh, indent=2,
                )
        except OSError as exc:
            logger.warning("No se pudo guardar checkpoint: %s", exc)

    def _ckpt_key(self, ss_id: int, session: str) -> str:
        return f"ss{ss_id:02d}|{session}"

    # ------------------------------------------------------------------
    # Output dir per SS
    # ------------------------------------------------------------------

    def _get_output_dir(self, ss_id: int, session: str) -> Path:
        dp = self.params["discriminant_params"]
        tw = self.params["time_window"]
        t0 = tw.get("t_start")
        t1 = tw.get("t_end")
        t0_tag = f"{t0}s" if t0 is not None else "any"
        t1_tag = f"{t1}s" if t1 is not None else "any"
        depth = dp.get("hankel_depth", "auto")
        l_freq = dp.get("l_freq", 1.0)
        h_freq = dp.get("h_freq", 40.0)

        return Path(
            f"{self.output_dir}/discriminant/{session}"
            f"/SS{ss_id}_k{self.k}"
            f"/{l_freq}-{h_freq}Hz"
            f"_depth{depth}"
            f"/from{t0_tag}_to{t1_tag}"
            f"_{'_'.join(self.task_names)}"
        )

    # ------------------------------------------------------------------
    # Run one SS
    # ------------------------------------------------------------------

    def _run_single_ss(self, ss_id: int, session: str) -> bool:
        """Construir Hankel + calcular direcciones de discriminancia + guardar."""
        from src.pipelines.run_cdhsa import build_hankel_single_ss, characterize_hankel_matrices

        dp = self.params["discriminant_params"]
        tw = self.params["time_window"]
        sppss = self.ss_cfg.get("subjects_per_super_subject", 20)
        offset = self.ss_cfg.get("subject_start_offset", 1)

        logger.info("")
        logger.info("-" * 70)
        logger.info(
            "SUPER-SUJETO %d  |  session=%s  |  %d tareas  |  k=%d",
            ss_id, session, len(self.task_names), self.k,
        )
        logger.info("-" * 70)

        t0 = time.time()

        # 1. Construir Hankel
        logger.info("  [1/3] Construyendo matrices de Hankel...")
        if self.global_channels is not None:
            logger.info(
                "         usando %d canales globales",
                len(self.global_channels),
            )
        try:
            X, hankel_info = build_hankel_single_ss(
                super_subject_id=ss_id,
                session=session,
                tasks=self.task_names,
                subjects_per_super_subject=sppss,
                subject_start_offset=offset,
                db_path=self.db_path,
                t_start=tw.get("t_start"),
                t_stop=tw.get("t_end"),
                l_freq=dp.get("l_freq", 1.0),
                h_freq=dp.get("h_freq", 40.0),
                hankel_depth=dp.get("hankel_depth"),
                verbose=False,
                global_channels=self.global_channels,
            )
        except Exception as exc:
            logger.error("  ERROR construyendo Hankel: %s", exc)
            return False

        # X es list[list[NDArray]] con S=1: X[0] = [H_c1, H_c2, ...]
        H_list = X[0]
        valid_tasks = [
            name for name, H in zip(self.task_names, H_list)
            if H is not None and H.size > 0
        ]
        H_valid = [H for H in H_list if H is not None and H.size > 0]

        if len(H_valid) < 2:
            logger.error(
                "  ERROR: solo %d tareas validas (necesitas >= 2)",
                len(H_valid),
            )
            return False

        logger.info(
            "  %d/%d tareas con Hankel valida", len(H_valid), len(self.task_names)
        )

        # 2. Calcular direcciones de discriminancia
        logger.info("  [2/3] Calculando direcciones de discriminancia...")
        directions = compute_discriminant_directions(
            H_valid, valid_tasks, k=self.k,
        )

        build_time = time.time() - t0
        logger.info(
            "  Analisis completado en %.1f s", build_time
        )

        # 3. Guardar
        if self.output_dir:
            out_dir = self._get_output_dir(ss_id, session)
            logger.info("  [3/3] Guardando resultados...")
            save_ss_results(
                out_dir=out_dir,
                ss_id=ss_id,
                directions=directions,
                hankel_info=hankel_info,
                params=self.params,
                k=self.k,
                task_names=valid_tasks,
            )

        # Guardar en memoria para comparacion final
        self.all_directions[ss_id] = directions

        # Log resumen
        pool_norm = list(directions.values())[0].get("pool_norm", 0.0)
        if pool_norm > 0:
            logger.info(
                "    %-12s  ||Sigma_pool||=%.3e",
                "(pooled)", pool_norm,
            )
        for task in valid_tasks:
            res = directions[task]
            rel = res.get("rel_delta", 0.0)
            logger.info(
                "    %-12s  ||Delta||=%.3e  (%.4f%% del pool)  "
                "top-S=[%s]",
                task,
                res["Delta_norm"],
                rel * 100,
                ", ".join(f"{s:.3e}" for s in res["S"]),
            )

        # Liberar memoria de las Hankel
        del X, H_list, H_valid

        return True

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------

    def run(self) -> int:
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Si hay checkpoint pero no global_channels, no podemos
        # comparar con resultados viejos (d inconsistente).
        # Invalidar checkpoint si no tenemos canales globales
        # y hay mas de 1 SS seleccionado.
        if (
            self.global_channels is None
            and len(self.ss_cfg["selected"]) > 1
            and self.checkpoint
        ):
            logger.warning(
                "  Invalidando checkpoint previo: se detecto"
                " posible inconsistencia de canales."
            )
            self.checkpoint = set()
            self._save_checkpoint()

        selected = list(self.ss_cfg["selected"])
        jobs = [
            (sid, session)
            for sid in selected
            for session in self.sessions
        ]

        # Filtrar por checkpoint
        todo = [
            (sid, sess) for sid, sess in jobs
            if self._ckpt_key(sid, sess) not in self.checkpoint
        ]
        if len(todo) < len(jobs):
            logger.info(
                "Checkpoint: %d/%d jobs ya completados",
                len(jobs) - len(todo), len(jobs),
            )

        if not todo:
            logger.info("Todos los jobs ya completados.")
        else:
            logger.info("Jobs a ejecutar: %d / %d", len(todo), len(jobs))

        completed = 0
        failed = 0

        for idx, (ss_id, session) in enumerate(todo, start=1):
            logger.info("")
            logger.info("Progreso: %d / %d", idx, len(todo))

            success = self._run_single_ss(ss_id, session)
            if success:
                self.checkpoint.add(self._ckpt_key(ss_id, session))
                completed += 1
            else:
                failed += 1

            self._save_checkpoint()

            if idx < len(todo) and self.delay > 0:
                time.sleep(self.delay)

        logger.info("")
        logger.info("=" * 70)
        logger.info(
            "ANALISIS COMPLETADO  --  OK: %d  |  Fallos: %d  |  Total: %d",
            completed, failed, len(todo),
        )
        logger.info("=" * 70)

        # Comparacion cruzada
        if self.run_comparison and len(self.all_directions) >= 2:
            self._run_comparison()

        return 0 if failed == 0 else 1

    # ------------------------------------------------------------------
    # Cross-SS comparison
    # ------------------------------------------------------------------

    def _run_comparison(self) -> None:
        """Comparar subespacios de discriminancia entre super-sujetos."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("  INICIANDO COMPARACION CRUZADA")
        logger.info("=" * 70)

        ss_ids = sorted(self.all_directions.keys())

        # Las tareas que tienen datos en TODOS los SS
        common_tasks = [
            t for t in self.task_names
            if all(t in self.all_directions[ss] for ss in ss_ids)
        ]

        if not common_tasks:
            logger.warning(
                "  No hay tareas con datos en todos los SS. "
                "No se puede comparar."
            )
            return

        logger.info(
            "  Tareas a comparar: %s", common_tasks
        )

        comparison = compare_discriminant_subspaces(
            self.all_directions, common_tasks,
        )

        # Guardar
        if self.output_dir:
            comp_dir = (
                self.output_dir
                / "discriminant"
                / self.sessions[0]
                / "cross_ss_comparison"
            )
            save_comparison(
                out_dir=comp_dir,
                comparison=comparison,
                task_names=common_tasks,
                ss_ids=ss_ids,
                k=self.k,
                experiment_label=self.params.get(
                    "experiment_label", "discriminant"
                ),
            )

        # Imprimir resumen corto
        for task in common_tasks:
            pairs = comparison.get(task, {})
            if not pairs:
                continue
            angles_mean = [
                info["mean_angle_deg"]
                for info in pairs.values()
                if "error" not in info
            ]
            if angles_mean:
                overall = np.mean(angles_mean)
                label = _consistency_label(overall)
                logger.info(
                    "  %-12s  ->  Consistencia %s  (mean=%.1f deg)",
                    task, label, overall,
                )

        logger.info("")
        logger.info("  COMPARACION CRUZADA COMPLETADA")
        logger.info("=" * 70)


# =====================================================================
# ENTRY POINT
# =====================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analisis de direcciones de maxima discriminancia por tarea, "
            "por super-sujeto, con comparacion cruzada de robustez."
        ),
    )
    parser.add_argument(
        "--params-json", type=str, default=None,
        help=(
            "Ruta al JSON de parametros. "
            f"Default: {DEFAULT_PARAMS_JSON}"
        ),
    )
    args = parser.parse_args()

    json_path = (
        Path(args.params_json) if args.params_json
        else DEFAULT_PARAMS_JSON
    )
    if os.environ.get("DISCRIMINANT_PARAMS_JSON"):
        json_path = Path(os.environ["DISCRIMINANT_PARAMS_JSON"])

    params = _load_params(json_path)
    logger.info("Parametros cargados desde: %s", json_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner = DiscriminantAnalysisRunner(params)
        return runner.run()


if __name__ == "__main__":
    sys.exit(main())
