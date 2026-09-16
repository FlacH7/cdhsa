#!/usr/bin/env python3
"""
run_specific_modes_robustness.py
=================================
Pipeline completo de robustez de modos especificos por tarea.

Para cada super-sujeto:
  1. Construye matrices de Hankel para todas las tareas
  2. Corre el pipeline CD-HSA completo (Steps A, A6, B/C, Tangent, D)
  3. Calcula la especificidad por modo por tarea a partir de la
     descomposicion de la covarianza pool (los L modos mas
     energeticos en promedio)
  4. Selecciona los top-k modos especificos por tarea
  5. Guarda resultados CD-HSA + especificidad

Despues de todos los super-sujetos:
  6. Compara subespacios de modos especificos entre pares de SS
     usando angulos principales + correlacion modo-a-modo
  7. Genera reporte de consistencia

Metodo de especificidad:
  Para cada modo l del pool (eigenvectores de Sigma_pool):
    E(c,l) = u_l^T @ Sigma_c @ u_l    (energia del modo l en tarea c)
    S(l)   = u_l^T @ Sigma_pool @ u_l (energia promedio del modo l)
    spec(c,l) = E(c,l) / S(l)
  Los modos con mayor spec(c,l) son los mas especificos de tarea c.

Uso: python run_specific_modes_robustness.py --params-json mi_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.pipelines.run_cdhsa import (
            build_hankel_single_ss,
            characterize_hankel_matrices,
            run_cdhsa,
            _save_result_arrays,
            CDHSAConfig
        )

NDArray = npt.NDArray

# --- Script directory (para importar de run_cdhsa.py) ---
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

DEFAULT_PARAMS_JSON = _SCRIPT_DIR / "specific_modes_params.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =====================================================================
# CORE: SPECIFIC MODES FROM POOL COVARIANCE
# =====================================================================


def compute_specific_modes(
    H_list: list[NDArray],
    task_names: list[str],
    L: int,
    k: int = 5,
) -> dict[str, Any]:
    """Modos de maxima especificidad por tarea.

    1. Sigma_c = H_c H_c^T / K_c   (covarianza por tarea)
    2. Sigma_pool = mean(Sigma_c)   (covarianza promediada)
    3. Eigendescomposicion de Sigma_pool -> U, S
    4. E(c,l) = u_l^T Sigma_c u_l  (energia por modo por tarea)
    5. spec(c,l) = E(c,l) / S(l)   (especificidad)
    6. Top-k por especificidad -> subespacio especifico de cada tarea

    Parameters
    ----------
    H_list : list[NDArray]
        Lista de C matrices de Hankel, cada una (d, K_c).
    task_names : list[str]
        Nombres de las C tareas.
    L : int
        Cantidad de modos del pool a considerar.
    k : int
        Cantidad de modos especificos a seleccionar por tarea.

    Returns
    -------
    dict con claves: U_common, S_pool, energies, specificity, selected,
    cov_pool_norm, d.
    """
    C = len(H_list)
    d = H_list[0].shape[0]

    # --- 1. Covarianzas por tarea ---
    logger.info("    [spec] Computando covarianzas por tarea...")
    covs: list[NDArray] = []
    for H in H_list:
        K = H.shape[1]
        covs.append(H @ H.T / K)  # (d, d)

    # --- 2. Covarianza pooled ---
    cov_pool = np.mean(covs, axis=0)  # (d, d)
    pool_norm = float(np.linalg.norm(cov_pool, "fro"))
    logger.info(
        "    [spec] ||Sigma_pool||_F = %.3e  (d=%d, C=%d)",
        pool_norm, d, C,
    )

    # --- 3. Eigendescomposicion de Sigma_pool ---
    logger.info("    [spec] Eigendescomposicion de Sigma_pool...")
    eigenvalues, eigenvectors = np.linalg.eigh(cov_pool)
    # eigh retorna orden ascendente; revertir
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    L_eff = min(L, d)
    U_common = eigenvectors[:, :L_eff]  # (d, L)
    S_pool = np.maximum(eigenvalues[:L_eff], 0.0)  # (L,)

    logger.info(
        "    [spec] Top-5 eigenvalues: %s",
        [f"{v:.3e}" for v in S_pool[:5]],
    )

    # --- 4. Energia por modo por tarea ---
    logger.info("    [spec] Computando energia por modo por tarea...")
    energies: dict[str, NDArray] = {}
    for cov_c, name in zip(covs, task_names):
        # E_matrix = U^T @ Sigma_c @ U  ->  (L, L)
        # E(c,l) = diag(E_matrix)
        E_matrix = U_common.T @ cov_c @ U_common
        energies[name] = np.diag(E_matrix).copy()  # (L,)

    # --- 5. Especificidad ---
    specificity: dict[str, NDArray] = {}
    for name, E_c in energies.items():
        specificity[name] = E_c / (S_pool + 1e-30)

    # --- 6. Seleccionar top-k por tarea ---
    selected: dict[str, dict[str, Any]] = {}
    for name, spec in specificity.items():
        top_k_idx = np.argsort(spec)[::-1][:k]
        selected[name] = {
            "indices": top_k_idx.copy(),
            "scores": spec[top_k_idx].copy(),
            "U": U_common[:, top_k_idx].copy(),  # (d, k)
        }

    # Log resumen
    for name in task_names:
        sel = selected[name]
        top_scores = sel["scores"]
        top_idx = sel["indices"]
        logger.info(
            "    [spec] %-12s  top-%d spec=[%s]  modos=[%s]",
            name, k,
            ", ".join(f"{s:.3f}" for s in top_scores),
            ", ".join(f"{i}" for i in top_idx),
        )

    return {
        "U_common": U_common,        # (d, L)
        "S_pool": S_pool,            # (L,)
        "energies": energies,        # {task: (L,)}
        "specificity": specificity,  # {task: (L,)}
        "selected": selected,        # {task: {indices, scores, U}}
        "cov_pool_norm": pool_norm,
        "d": d,
        "L_eff": L_eff,
    }


# =====================================================================
# CROSS-SS COMPARISON: PRINCIPAL ANGLES + MODE CORRELATION
# =====================================================================


def _principal_angles_between(
    U_a: NDArray, U_b: NDArray,
) -> tuple[NDArray, NDArray]:
    """Angulos principales entre dos subespacios de igual dimension.

    Returns (angles_deg, singular_values).
    """
    C_mat = U_a.T @ U_b  # (k, k)
    sv = np.linalg.svd(C_mat, compute_uv=False)
    sv_clipped = np.clip(sv, -1.0, 1.0)
    angles_rad = np.arccos(sv_clipped)
    return np.degrees(angles_rad), sv


def _mode_correlation_matrix(
    U_a: NDArray, U_b: NDArray,
) -> NDArray:
    """Matriz de correlacion modo-a-modo entre dos SS.

    C[i,j] = |u_a_i^T u_b_j|  (correlacion absoluta).
    """
    return np.abs(U_a.T @ U_b)  # (k, k)


def compare_specific_subspaces(
    all_results: dict[int, dict[str, dict]],
    task_names: list[str],
) -> dict[str, dict[tuple[int, int], dict[str, Any]]]:
    """Compara subespacios de modos especificos entre pares de SS.

    Para cada tarea y cada par (i,j):
      - Angulos principales entre los k modos especificos
      - Matriz de correlacion modo-a-modo
      - Mejor matching (greedy) y correlacion promedio
    """
    ss_ids = sorted(all_results.keys())
    comparison: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}

    for task in task_names:
        comparison[task] = {}
        for idx_i in range(len(ss_ids)):
            for idx_j in range(idx_i + 1, len(ss_ids)):
                ss_i = ss_ids[idx_i]
                ss_j = ss_ids[idx_j]

                sel_i = all_results[ss_i][task]
                sel_j = all_results[ss_j][task]

                U_i = sel_i["U"]  # (d, k)
                U_j = sel_j["U"]  # (d, k)

                if U_i.shape[0] != U_j.shape[0]:
                    comparison[task][(ss_i, ss_j)] = {
                        "error": (
                            f"incompatible d: {U_i.shape[0]} vs "
                            f"{U_j.shape[0]}"
                        ),
                        "mean_angle_deg": 999.0,
                        "mean_mode_corr": 0.0,
                    }
                    continue

                # Angulos principales
                angles_deg, sv = _principal_angles_between(U_i, U_j)

                # Correlacion modo-a-modo
                corr_mat = _mode_correlation_matrix(U_i, U_j)

                # Greedy matching: para cada modo de SS_i, encontrar
                # el mejor match en SS_j (sin repetir)
                used_j: set[int] = set()
                match_corrs: list[float] = []
                for i_row in range(corr_mat.shape[0]):
                    best_j = -1
                    best_val = -1.0
                    for j_col in range(corr_mat.shape[1]):
                        if j_col in used_j:
                            continue
                        if corr_mat[i_row, j_col] > best_val:
                            best_val = corr_mat[i_row, j_col]
                            best_j = j_col
                    if best_j >= 0:
                        used_j.add(best_j)
                        match_corrs.append(best_val)

                mean_match_corr = (
                    float(np.mean(match_corrs)) if match_corrs else 0.0
                )

                comparison[task][(ss_i, ss_j)] = {
                    "singular_values": [float(x) for x in sv],
                    "angles_deg": [float(x) for x in angles_deg],
                    "mean_angle_deg": float(np.mean(angles_deg)),
                    "max_angle_deg": float(np.max(angles_deg)),
                    "subspace_correlation": float(np.mean(sv)),
                    "corr_matrix": corr_mat.tolist(),
                    "mean_mode_corr": mean_match_corr,
                    "match_corrs": match_corrs,
                }

    return comparison


# =====================================================================
# REPORTS
# =====================================================================


def _consistency_label(mean_angle: float) -> str:
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
    spec_result: dict,
    task_names: list[str],
    k: int,
    hankel_info: dict,
    cdhsa_summary: str,
) -> str:
    """Reporte textual por super-sujeto."""
    lines: list[str] = []
    lines.append("")
    lines.append(f"{'=' * 70}")
    lines.append(
        f"  SUPER-SUJETO {ss_id}  --  Modos Especificos por Tarea"
    )
    lines.append(f"{'=' * 70}")

    d = spec_result["d"]
    L = spec_result["L_eff"]
    lines.append(f"  d = {d}  |  L (pool modes) = {L}  |  k = {k}")
    if hankel_info.get("global_channels"):
        lines.append(
            f"  canales ({len(hankel_info['global_channels'])}): "
            f"{hankel_info['global_channels'][:5]}..."
        )
    lines.append(
        f"  ||Sigma_pool||_F = {spec_result['cov_pool_norm']:.3e}"
    )
    lines.append("")

    # Resumen CD-HSA
    if cdhsa_summary:
        lines.append("  --- CD-HSA Summary ---")
        for line in cdhsa_summary.split("\n"):
            lines.append(f"    {line}")
        lines.append("")

    # Especificidad por tarea
    lines.append("  --- Modos Especificos por Tarea ---")
    for task in task_names:
        if task not in spec_result["selected"]:
            lines.append(f"  {task:<15}  [NO DATA]")
            continue
        sel = spec_result["selected"][task]
        scores = sel["scores"]
        indices = sel["indices"]
        lines.append(f"  {task}:")
        for i in range(min(k, len(scores))):
            lines.append(
                f"    modo {indices[i]:>2d}  "
                f"spec={scores[i]:.3f}  "
                f"(rank {i+1} de {k})"
            )
        lines.append("")

    # Tabla de especificidad completa (todos los modos, top-10)
    lines.append("  --- Especificidad Completa (top-10 modos por tarea) ---")
    header = f"  {'Modo':>5}"
    for t in task_names:
        header += f"  {t[:10]:>10}"
    lines.append(header)
    lines.append("  " + " " * (len(header) - 2))

    for l in range(min(10, spec_result["L_eff"])):
        row = f"  {l:>5d}"
        for t in task_names:
            if t in spec_result["specificity"]:
                row += f"  {spec_result['specificity'][t][l]:>10.3f}"
            else:
                row += f"{'N/A':>10}"
        lines.append(row)
    lines.append("")

    # Angulos entre subespacios de tareas (dentro del SS)
    lines.append("  --- Angulos entre subespacios de tareas (intra-SS) ---")
    for i_t, t1 in enumerate(task_names):
        for t2 in task_names[i_t + 1:]:
            if (t1 not in spec_result["selected"]
                    or t2 not in spec_result["selected"]):
                continue
            angles, sv = _principal_angles_between(
                spec_result["selected"][t1]["U"],
                spec_result["selected"][t2]["U"],
            )
            lines.append(
                f"    {t1:<12} vs {t2:<12}: "
                f"ang=[{', '.join(f'{a:.1f}' for a in angles)}]  "
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
    lines.append(
        "  COMPARACION CRUZADA DE MODOS ESPECIFICOS"
    )
    lines.append(f"{'=' * 70}")
    lines.append(f"  Experimento : {experiment_label}")
    lines.append(f"  Tareas      : {', '.join(task_names)}")
    lines.append(f"  k (mods/tarea): {k}")
    lines.append(f"  Super-sujetos: {ss_ids}")
    lines.append(f"  Generado   : {datetime.now().isoformat()}")
    lines.append("")

    summary_rows: list[str] = []
    for task in task_names:
        lines.append(f"  --- {task} ---")
        pairs = comparison.get(task, {})
        if not pairs:
            lines.append("    (sin datos)")
            summary_rows.append(f"  {task:<15}  SIN DATOS")
            continue

        mean_angles: list[float] = []
        mean_corrs: list[float] = []
        for (ss_i, ss_j), info in sorted(pairs.items()):
            if "error" in info:
                lines.append(
                    f"    SS{ss_i} vs SS{ss_j}: ERROR - {info['error']}"
                )
                continue
            angles = info["angles_deg"]
            m_corr = info["mean_mode_corr"]
            mean_a = info["mean_angle_deg"]
            max_a = info["max_angle_deg"]
            mean_angles.append(mean_a)
            mean_corrs.append(m_corr)
            lines.append(
                f"    SS{ss_i} vs SS{ss_j}: "
                f"ang=[{', '.join(f'{a:.1f}' for a in angles)}]  "
                f"mean_ang={mean_a:.1f}  max={max_a:.1f}  "
                f"mode_corr={m_corr:.3f}"
            )

        if mean_angles:
            overall_ang = np.mean(mean_angles)
            overall_corr = np.mean(mean_corrs)
            ang_label = _consistency_label(overall_ang)
            lines.append(
                f"    -> Consistencia {ang_label} "
                f"(ang={overall_ang:.1f} deg, "
                f"mode_corr={overall_corr:.3f})"
            )
            summary_rows.append(
                f"  {task:<15}  {ang_label:<12} "
                f"ang={overall_ang:.1f} deg  "
                f"mode_corr={overall_corr:.3f}"
            )
        else:
            summary_rows.append(f"  {task:<15}  SIN PARES VALIDOS")
        lines.append("")

    # Tabla resumen
    lines.append(f"  {'=' * 65}")
    lines.append("  RESUMEN DE CONSISTENCIA")
    lines.append(f"  {'=' * 65}")
    for row in summary_rows:
        lines.append(row)
    lines.append("")

    lines.append("  INTERPRETACION:")
    lines.append(
        "    angulo < 10 deg  +  mode_corr > 0.90  :"
        " robustez ALTA"
    )
    lines.append(
        "    angulo 10-20 deg +  mode_corr > 0.80  :"
        " robustez MODERADA"
    )
    lines.append(
        "    angulo 20-35 deg +  mode_corr > 0.60  :"
        " robustez BAJA"
    )
    lines.append(
        "    angulo > 35 deg  o  mode_corr < 0.60  :"
        " robustez MUY BAJA"
    )

    return "\n".join(lines)


# =====================================================================
# GUARDADO DE RESULTADOS
# =====================================================================


def _json_safe(obj: Any) -> Any:
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
    spec_result: dict,
    hankel_info: dict,
    params: dict,
    k: int,
    L: int,
    task_names: list[str],
    cdhsa_result=None,
    cdhsa_config=None,
    X=None,
    characterization: str = "",
) -> None:
    """Guardar resultados de un super-sujeto.

    Archivos:
      hankel_info.json        Metadata de Hankel
    config.json             Configuracion usada
    characterization.txt     Tabla de Hankel (si disponible)
    cdhsa_summary.txt        Resumen CD-HSA
    cdhsa_arrays.npz         Arrays CD-HSA
    specificity_arrays.npz   U, S, energies, specificity
    specificity_meta.json    Metadatos de especificidad
    specificity_report.txt   Reporte textual
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  Guardando en: %s", out_dir)

    # --- 1. hankel_info.json ---
    with open(out_dir / "hankel_info.json", "w") as fh:
        json.dump(_json_safe(hankel_info), fh, indent=2, default=str)
    logger.info("    [OK] hankel_info.json")

    # --- 2. config.json ---
    cfg_out = {
        "experiment_label": params.get("experiment_label", ""),
        "cdhsa_params": params.get("cdhsa_params", {}),
        "specificity_params": params.get("specificity_params", {}),
        "L": L,
        "k": k,
        "ss_id": ss_id,
    }
    with open(out_dir / "config.json", "w") as fh:
        json.dump(cfg_out, fh, indent=2)
    logger.info("    [OK] config.json")

    # --- 3. characterization.txt ---
    if characterization:
        with open(out_dir / "characterization.txt", "w") as fh:
            fh.write(characterization)
        logger.info("    [OK] characterization.txt")

    # --- 4. CD-HSA results ---
    if cdhsa_result is not None:
        summary_text = cdhsa_result.summary()
        with open(out_dir / "cdhsa_summary.txt", "w") as fh:
            fh.write(summary_text)
        logger.info("    [OK] cdhsa_summary.txt")

        # cdhsa_arrays.npz
        cdhsa_dict: dict[str, NDArray] = {}
        _save_result_arrays(cdhsa_result.R, "R", cdhsa_dict)
        if cdhsa_result.A6 is not None:
            _save_result_arrays(cdhsa_result.A6, "A6", cdhsa_dict)
        if cdhsa_result.BC is not None:
            _save_result_arrays(cdhsa_result.BC, "BC", cdhsa_dict)
        if cdhsa_result.G is not None:
            _save_result_arrays(cdhsa_result.G, "G", cdhsa_dict)
        if cdhsa_result.D is not None:
            _save_result_arrays(cdhsa_result.D, "D", cdhsa_dict)
        np.savez_compressed(
            out_dir / "cdhsa_arrays.npz", **cdhsa_dict
        )
        logger.info(
            "    [OK] cdhsa_arrays.npz  (%d arrays)",
            len(cdhsa_dict),
        )

    # --- 5. hankel_matrices.npz (opcional) ---
    save_hankel = params.get("execution", {}).get(
        "save_hankel_matrices", False
    )
    if save_hankel and X is not None:
        h_dict: dict[str, NDArray] = {}
        for c_idx, task in enumerate(task_names):
            H = X[0][c_idx]
            if H is not None and H.size > 0:
                h_dict[f"H_c{c_idx+1}_{task}"] = H
        np.savez_compressed(
            out_dir / "hankel_matrices.npz", **h_dict
        )
        logger.info(
            "    [OK] hankel_matrices.npz  (%d matrices)",
            len(h_dict),
        )

    # --- 6. specificity_arrays.npz ---
    npz_dict: dict[str, NDArray] = {}
    npz_dict["U_common"] = spec_result["U_common"]  # (d, L)
    npz_dict["S_pool"] = spec_result["S_pool"]  # (L,)
    for task in task_names:
        if task in spec_result["energies"]:
            safe = task.replace("-", "_")
            npz_dict[f"energy_{safe}"] = spec_result["energies"][task]
            npz_dict[f"spec_{safe}"] = spec_result["specificity"][task]
            npz_dict[f"U_sel_{safe}"] = spec_result["selected"][task]["U"]
            npz_dict[f"idx_sel_{safe}"] = spec_result["selected"][task]["indices"]
            npz_dict[f"score_sel_{safe}"] = spec_result["selected"][task]["scores"]
    np.savez_compressed(
        out_dir / "specificity_arrays.npz", **npz_dict
    )
    logger.info(
        "    [OK] specificity_arrays.npz  (%d arrays)",
        len(npz_dict),
    )

    # --- 7. specificity_meta.json ---
    meta = {
        "analysis_type": "specific_modes_robustness",
        "super_subject_id": ss_id,
        "d": spec_result["d"],
        "L": spec_result["L_eff"],
        "k": k,
        "cov_pool_norm": spec_result["cov_pool_norm"],
        "task_names": task_names,
        "per_task": {},
    }
    for task in task_names:
        if task in spec_result["selected"]:
            sel = spec_result["selected"][task]
            meta["per_task"][task] = {
                "selected_indices": sel["indices"].tolist(),
                "specificity_scores": sel["scores"].tolist(),
            }
    with open(out_dir / "specificity_meta.json", "w") as fh:
        json.dump(_json_safe(meta), fh, indent=2)
    logger.info("    [OK] specificity_meta.json")

    # --- 8. specificity_report.txt ---
    report = _format_ss_report(
        ss_id, spec_result, task_names, k, hankel_info,
        cdhsa_result.summary() if cdhsa_result else "",
    )
    with open(out_dir / "specificity_report.txt", "w") as fh:
        fh.write(report)
    logger.info("    [OK] specificity_report.txt")


def save_comparison(
    out_dir: Path,
    comparison: dict,
    task_names: list[str],
    ss_ids: list[int],
    k: int,
    experiment_label: str,
) -> Path:
    """Guardar reporte de comparacion cruzada."""
    out_dir.mkdir(parents=True, exist_ok=True)

    report = _format_comparison_report(
        comparison, task_names, ss_ids, k, experiment_label,
    )
    report_path = out_dir / "comparison.txt"
    with open(report_path, "w") as fh:
        fh.write(report)
    logger.info("  Comparacion guardada: %s", report_path)

    json_path = out_dir / "comparison.json"
    with open(json_path, "w") as fh:
        json.dump(_json_safe(comparison), fh, indent=2)
    logger.info("  Comparacion JSON: %s", json_path)

    return report_path


# =====================================================================
# JSON LOADING
# =====================================================================


def _load_params(json_path: Path) -> dict:
    if not json_path.exists():
        logger.error("JSON no encontrado: %s", json_path)
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as fh:
        params = json.load(fh)

    for key in ["super_subjects", "sessions", "tasks",
                "cdhsa_params", "specificity_params"]:
        if key not in params:
            logger.error("Falta la clave requerida '%s'", key)
            sys.exit(1)

    ss_cfg = params["super_subjects"]
    if "selected" not in ss_cfg or not ss_cfg["selected"]:
        logger.error("'super_subjects.selected' debe ser non vacia.")
        sys.exit(1)

    params.setdefault("time_window", {})
    params.setdefault("execution", {})
    params["time_window"].setdefault("t_start", None)
    params["time_window"].setdefault("t_end", None)
    params["execution"].setdefault("max_workers", 1)
    params["execution"].setdefault("delay", 1.0)
    params["execution"].setdefault("run_comparison", True)
    params["execution"].setdefault("save_hankel_matrices", False)

    return params


# =====================================================================
# BATCH RUNNER
# =====================================================================


class SpecificModesRunner:
    """Orquesta el pipeline CD-HSA + modos especificos por SS."""

    def __init__(self, params: dict) -> None:
        self.params = params
        self.ss_cfg = params["super_subjects"]
        self.task_names: list[str] = list(params["tasks"])
        self.sessions: list[str] = list(params["sessions"])
        self.k: int = params["specificity_params"]["k"]
        self.L: int = params["cdhsa_params"]["L"]
        self.exec_cfg = params["execution"]
        self.delay: float = self.exec_cfg.get("delay", 1.0)
        self.run_comparison: bool = self.exec_cfg.get(
            "run_comparison", True
        )

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

        # Results storage (para comparacion final)
        # {ss_id: {task: {U, indices, scores}}}
        self.all_selected: dict[int, dict[str, dict]] = {}

        self._print_banner()

    # ------------------------------------------------------------------
    # Global channel intersection
    # ------------------------------------------------------------------

    def _discover_global_channels(self) -> list[str]:
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

        all_ch_sets: list[set[str]] = []
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
                all_ch_sets.append(ch_set)
                logger.info("    SS%d: %d canales", ss_id, len(ch_set))
                del raw
            except Exception as exc:
                logger.error(
                    "    SS%d: ERROR: %s", ss_id, exc
                )
                all_ch_sets.append(set())

        global_ch = sorted(set.intersection(*all_ch_sets))
        logger.info("  " + "-" * 50)
        logger.info("    Interseccion global: %d canales", len(global_ch))
        for i, ss_id in enumerate(selected):
            n_total = len(all_ch_sets[i])
            n_kept = len(all_ch_sets[i] & set(global_ch))
            n_excl = n_total - n_kept
            if n_excl > 0:
                logger.info(
                    "    SS%d: %d excluidos (%d -> %d)",
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
                "src.utils.config no disponible;"
                " usando rutas por defecto."
            )
            self.output_dir = Path("./results")

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        ss_cfg = self.ss_cfg
        dp = self.params["cdhsa_params"]
        sp = self.params["specificity_params"]
        tw = self.params["time_window"]

        logger.info("=" * 70)
        logger.info(
            "  PIPELINE DE ROBUSTEZ DE MODOS ESPECIFICOS"
        )
        logger.info("=" * 70)
        logger.info(
            "  Super-sujetos  : %s", ss_cfg["selected"]
        )
        logger.info(
            "  Sujetos/SS    : %s",
            ss_cfg.get("subjects_per_super_subject", "?"),
        )
        logger.info("  Sesiones       : %s", self.sessions)
        logger.info("  Tareas         : %s", self.task_names)
        logger.info("  L (pool modes) : %d", self.L)
        logger.info("  k (spec/tarea) : %d", self.k)
        logger.info(
            "  Ventana temporal: %s - %s s",
            tw.get("t_start"), tw.get("t_end"),
        )
        logger.info(
            "  Filtro         : %.1f - %.1f Hz",
            dp.get("l_freq", 1.0), dp.get("h_freq", 40.0),
        )
        logger.info(
            "  Hankel depth   : %s",
            dp.get("hankel_depth", "auto"),
        )
        logger.info(
            "  CD-HSA rank    : %s (%s)",
            dp.get("fixed_rank"), dp.get("rank_method"),
        )
        logger.info("  Output dir     : %s", self.output_dir)
        if self.global_channels is not None:
            logger.info(
                "  Canales globales: %d (two-pass)",
                len(self.global_channels),
            )
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
        return cp_dir / "batch_checkpoint_specific_modes.json"

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
        dp = self.params["cdhsa_params"]
        tw = self.params["time_window"]
        t0 = tw.get("t_start")
        t1 = tw.get("t_end")
        t0_tag = f"{t0}s" if t0 is not None else "any"
        t1_tag = f"{t1}s" if t1 is not None else "any"
        depth = dp.get("hankel_depth", "auto")
        l_freq = dp.get("l_freq", 1.0)
        h_freq = dp.get("h_freq", 40.0)
        label = self.params.get("experiment_label", "specific_modes")

        return Path(
            f"{self.output_dir}/specific_modes/{session}"
            f"/{label}"
            f"/SS{ss_id}_L{self.L}_k{self.k}"
            f"/{l_freq}-{h_freq}Hz"
            f"_depth{depth}"
            f"/from{t0_tag}_to{t1_tag}"
            f"_{'_'.join(self.task_names)}"
        )

    # ------------------------------------------------------------------
    # Build CDHSAConfig from params
    # ------------------------------------------------------------------

    def _build_cdhsa_config(self) -> "CDHSAConfig":
        dp = self.params["cdhsa_params"]
        return CDHSAConfig(
            fixed_rank=dp.get("fixed_rank", 25),
            rank_method=dp.get("rank_method", "fixed"),
            rmax=dp.get("rmax", 20),
            n_blocks=dp.get("n_blocks", 4),
            repro_threshold=dp.get("repro_threshold", 0.80),
            repro_strategy=dp.get("repro_strategy", "consecutive"),
            max_common=dp.get("max_common", 30),
            prevalence_quantile=dp.get("prevalence_quantile", 0.10),
            a6_max_common=dp.get("a6_max_common", 0),
            a6_n_folds=dp.get("a6_n_folds", 5),
            a6_n_null=dp.get("a6_n_null", 200),
            a6_alpha=dp.get("a6_alpha", 0.05),
            a6_seed=dp.get("a6_seed", 1234),
            bc_blocks=dp.get("bc_blocks"),
            bc_energy_metric=dp.get("bc_energy_metric", "log_absolute"),
            bc_geometry_metric=dp.get(
                "bc_geometry_metric", "adjusted"
            ),
            bc_n_perm=dp.get("bc_n_perm", 2000),
            bc_seed=dp.get("bc_seed", 20260812),
            bc_alpha=dp.get("bc_alpha", 0.05),
            bc_condition_names=list(self.task_names),
            tangent_blocks=dp.get("tangent_blocks"),
            tangent_n_perm=dp.get("tangent_n_perm", 2000),
            tangent_seed=dp.get("tangent_seed", 9999),
            tangent_alpha=dp.get("tangent_alpha", 0.05),
            d_max_specific=dp.get("d_max_specific", 10),
            d_residual_rank_method=dp.get(
                "d_residual_rank_method", "local_gap"
            ),
            d_residual_rank_threshold=dp.get(
                "d_residual_rank_threshold", 0.1
            ),
            d_fixed_residual_rank=dp.get(
                "d_fixed_residual_rank", 5
            ),
            skip_bc=dp.get("skip_bc", False),
            skip_tangent=dp.get("skip_tangent", False),
            skip_d=dp.get("skip_d", False),
            seed=dp.get("seed", 42),
        )

    # ------------------------------------------------------------------
    # Run one SS
    # ------------------------------------------------------------------

    def _run_single_ss(self, ss_id: int, session: str) -> bool:

        dp = self.params["cdhsa_params"]
        tw = self.params["time_window"]
        sppss = self.ss_cfg.get("subjects_per_super_subject", 20)
        offset = self.ss_cfg.get("subject_start_offset", 1)

        logger.info("")
        logger.info("-" * 70)
        logger.info(
            "SUPER-SUJETO %d  |  session=%s  |  %d tareas  |"
            "  L=%d  k=%d",
            ss_id, session, len(self.task_names),
            self.L, self.k,
        )
        logger.info("-" * 70)

        t0_total = time.time()

        # === 1. Construir Hankel ===
        logger.info("  [1/4] Construyendo matrices de Hankel...")
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
            logger.error("  ERROR Hankel: %s", exc)
            return False

        H_list = X[0]
        valid_tasks = [
            name for name, H in zip(self.task_names, H_list)
            if H is not None and H.size > 0
        ]
        H_valid = [H for H in H_list if H is not None and H.size > 0]
        if len(H_valid) < 2:
            logger.error(
                "  ERROR: solo %d tareas validas (>= 2)",
                len(H_valid),
            )
            return False
        logger.info(
            "  %d/%d tareas con Hankel valida",
            len(H_valid), len(self.task_names),
        )
        t_hankel = time.time() - t0_total

        # === 2. Correr CD-HSA ===
        logger.info("  [2/4] Ejecutando CD-HSA...")
        cdhsa_config = self._build_cdhsa_config()
        t_cdhsa = time.time()
        try:
            cdhsa_result = run_cdhsa(X, self.L, cdhsa_config)
        except Exception as exc:
            logger.error("  ERROR CD-HSA: %s", exc)
            import traceback
            traceback.print_exc()
            cdhsa_result = None
        t_cdhsa = time.time() - t_cdhsa
        logger.info(
            "  CD-HSA completado en %.1f s", t_cdhsa
        )
        if cdhsa_result is not None:
            r0 = cdhsa_result.A6.get("r0", "?")
            logger.info(
                "  CD-HSA Step A6: r0 = %s", r0
            )

        # === 3. Calcular modos especificos ===
        logger.info("  [3/4] Calculando modos especificos...")
        spec_result = compute_specific_modes(
            H_valid, valid_tasks, L=self.L, k=self.k,
        )
        t_spec = time.time() - (t0_total + t_hankel + t_cdhsa)

        # === 4. Guardar ===
        logger.info("  [4/4] Guardando resultados...")
        if self.output_dir:
            out_dir = self._get_output_dir(ss_id, session)

            # Caracterizacion de Hankel
            try:
                characterization = characterize_hankel_matrices(
                    X, hankel_info
                )
            except Exception:
                characterization = ""

            save_ss_results(
                out_dir=out_dir,
                ss_id=ss_id,
                spec_result=spec_result,
                hankel_info=hankel_info,
                params=self.params,
                k=self.k,
                L=self.L,
                task_names=valid_tasks,
                cdhsa_result=cdhsa_result,
                cdhsa_config=cdhsa_config,
                X=X,
                characterization=characterization,
            )

        # Almacenar para comparacion final
        self.all_selected[ss_id] = spec_result["selected"]

        t_total = time.time() - t0_total
        logger.info(
            "  SS%d completado en %.1f s "
            "(Hankel=%.1f, CD-HSA=%.1f, spec=%.1f)",
            ss_id, t_total, t_hankel, t_cdhsa, t_spec,
        )

        # Liberar memoria
        del X, H_list, H_valid
        if cdhsa_result is not None:
            del cdhsa_result

        return True

    # ------------------------------------------------------------------
    # Cross-SS comparison
    # ------------------------------------------------------------------

    def _run_comparison(self) -> None:
        logger.info("")
        logger.info("=" * 70)
        logger.info("  INICIANDO COMPARACION CRUZADA")
        logger.info("=" * 70)

        ss_ids = sorted(self.all_selected.keys())
        common_tasks = [
            t for t in self.task_names
            if all(t in self.all_selected[ss] for ss in ss_ids)
        ]
        if not common_tasks:
            logger.warning(
                "  No hay tareas con datos en todos los SS."
            )
            return
        logger.info("  Tareas a comparar: %s", common_tasks)

        comparison = compare_specific_subspaces(
            self.all_selected, common_tasks,
        )

        if self.output_dir:
            comp_dir = (
                self.output_dir
                / "specific_modes"
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
                    "experiment_label", "specific_modes"
                ),
            )

        # Resumen corto
        for task in common_tasks:
            pairs = comparison.get(task, {})
            if not pairs:
                continue
            ang_means = [
                info["mean_angle_deg"]
                for info in pairs.values()
                if "error" not in info
            ]
            corr_means = [
                info["mean_mode_corr"]
                for info in pairs.values()
                if "error" not in info
            ]
            if ang_means:
                overall_ang = np.mean(ang_means)
                overall_corr = np.mean(corr_means)
                label = _consistency_label(overall_ang)
                logger.info(
                    "  %-12s  ->  Consistencia %s  "
                    "(ang=%.1f deg, mode_corr=%.3f)",
                    task, label, overall_ang, overall_corr,
                )

        logger.info("")
        logger.info("  COMPARACION CRUZADA COMPLETADA")
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------

    def run(self) -> int:
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Invalidar checkpoint si no hay canales globales
        # y hay mas de 1 SS
        if (
            self.global_channels is None
            and len(self.ss_cfg["selected"]) > 1
            and self.checkpoint
        ):
            logger.warning(
                "  Invalidando checkpoint: posible"
                " inconsistencia de canales."
            )
            self.checkpoint = set()
            self._save_checkpoint()

        selected = list(self.ss_cfg["selected"])
        jobs = [
            (sid, session)
            for sid in selected
            for session in self.sessions
        ]
        todo = [
            (sid, sess)
            for sid, sess in jobs
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
            logger.info(
                "Jobs a ejecutar: %d / %d", len(todo), len(jobs)
            )

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
            "PIPELINE COMPLETADO  --  OK: %d  |  Fallos: %d"
            "  |  Total: %d",
            completed, failed, len(todo),
        )
        logger.info("=" * 70)

        if self.run_comparison and len(self.all_selected) >= 2:
            self._run_comparison()

        return 0 if failed == 0 else 1


# =====================================================================
# ENTRY POINT
# =====================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline de robustez de modos especificos por tarea:"
            " CD-HSA + especificidad + comparacion cruzada."
        ),
    )
    parser.add_argument(
        "--params-json", type=str, default=None,
        help=(
            f"Ruta al JSON de parametros. "
            f"Default: {DEFAULT_PARAMS_JSON}"
        ),
    )
    args = parser.parse_args()

    json_path = (
        Path(args.params_json) if args.params_json
        else DEFAULT_PARAMS_JSON
    )
    if os.environ.get("SPECIFIC_MODES_PARAMS_JSON"):
        json_path = Path(os.environ["SPECIFIC_MODES_PARAMS_JSON"])

    params = _load_params(json_path)
    logger.info("Parametros cargados desde: %s", json_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner = SpecificModesRunner(params)
        return runner.run()


if __name__ == "__main__":
    sys.exit(main())
