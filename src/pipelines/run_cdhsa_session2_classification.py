"""
src.pipelines.run_cdhsa_session2_classification
================================================

Clasificacion de grabaciones de SUJETOS SIMPLES usando un modelo CD-HSA
entrenado con SUPER-SUJETOS de OTRA sesion (experimento *hold-out de
sesion*: train = SS de la sesion 1, test = sujetos individuales de la
sesion 2).

Diseno
------
1. Se carga el modelo CD-HSA entrenado con super-sujetos
   (``run_batch_cdhsa`` -> ``run_cdhsa``): subespacio comun ``W0``
   (A6) y modos especificos por condicion ``W_c`` (Step D).  Las
   matrices viven en ``R^D`` con ``D = n_ch * depth * L`` fijado por la
   interseccion GLOBAL de canales del entrenamiento.
2. Cada grabacion de test (sujeto simple, sesion 2, una tarea) se
   ALINEA primero a los canales del entrenamiento (``align_raw_to_channels``:
   pick del orden objetivo + interpolacion esferica de splines de los
   ausentes, si la politica lo permite) y despues se preprocesa de forma
   IDENTICA al entrenamiento (filtro, depth, ventana temporal).
3. La grabacion se clasifica de forma INDEPENDIENTE comparando la
   energia de nivel 2 capturada por cada subespacio especifico
   (metodo 'energy', rango comun d) y/o la regla ML de covarianza
   'spiked' ('ml').
4. Estadistica: binomial exacto con p=1/C, IC Clopper-Pearson,
   permutacion PAREADA por sujeto, AUC one-vs-rest por clase.

Diferencias clave con ``run_cdhsa_task_classification`` (hold-out de
sujetos)
---------------------------------------------------------------------
* Guarda ANTI-FUGA INVERTIDA: aqui los mismos sujetos aparecen en
  train (como miembros de los SS, sesion 1) y en test (sesion 2); la
  separacion es por SESION.  El pipeline exige
  ``model.session != --test-session`` salvo ``--allow-same-session``.
* Alineacion de canales previa (los sujetos de otra sesion pueden
  tener canales ausentes/extras).
* El experimento puede usar un SUBCONJUNTO de las condiciones del
  modelo (``--tasks`` subconjunto de las del train); las clases del
  test son las tareas pedidas, en el orden pedido.
* Multi-clase nativo (C >= 2): AUC one-vs-rest, confusion C x C.

Uso (CLI)
---------
    python -m src.pipelines.run_cdhsa_session2_classification \\
        --train-results-dir results/cdhsa/session1/nSS5_L10_var99c50_a6n200_bcn5000/1.0-40.0Hz_depth10/from100.0s_to200.0s_eyesclosed_music \
        --test-session session2 \
        --subject-ids 1 2 ... 60 \
        --tasks eyesclosed music \
        --method energy --rank-truncation min \
        --missing-policy interpolate --max-missing 4

La justificacion matematica completa (clasificador de energia con la
Prop. 1 de equivalencia con el error de reconstruccion, regla ML
spiked, estadistica) esta en ``src/cdhsa/classification.py`` y en
``EXPERIMENTO_CLASIFICACION.md`` del paquete del experimento.

El pipeline es determinista (unica aleatoriedad: el test de
permutacion, con ``--seed``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.cdhsa.classification import (  # noqa: E402
    auc_mann_whitney,
    classify_from_scores,
    clopper_pearson_interval,
    compute_recording_energies,
    compute_training_mode_stats,
    confusion_matrix,
    energy_scores_at_rank,
    exact_binomial_pvalue,
    load_cdhsa_model,
    ml_discriminants,
    paired_label_swap_pvalue,
    parse_time_window_from_dir,
)
from src.latent_space_extraction.channel_alignment import (  # noqa: E402
    MISSING_POLICIES,
    align_raw_to_channels,
)

# Paleta segura para daltonicos (Okabe-Ito)
_C_OK = "#0072B2"      # azul
_C_BAD = "#D55E00"     # bermellon
_C_OTHER = "#009E73"   # verde
_C_GREY = "#7F7F7F"
_METHOD_LABEL = {
    "energy": "Residual-energy classifier",
    "ml": "ML spiked-covariance classifier",
}
_CLASS_COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#000000", "#F0E442",
]


# =====================================================================
# Carga de una grabacion de test (SEAM para tests E2E sinteticos:
# patchear esta funcion permite inyectar Hankels sinteticos)
# =====================================================================

def load_and_build_test_hankel(
    subject_id: int,
    session: str,
    task: str,
    *,
    model: dict,
    db_path: str | Path | None,
    t_start: float | None,
    t_stop: float | None,
    missing_policy: str = "interpolate",
    max_missing: int = 4,
    min_window_s: float = 30.0,
) -> tuple[NDArray[np.floating], float, dict]:
    """Cargar EEG de test, ALINEAR canales y construir el Hankel nivel 1.

    Replica EXACTAMENTE el preprocesado del entrenamiento (filtro
    pasa-banda, profundidad de Hankel, ventana temporal), con la
    alineacion de canales previa (ver
    :func:`src.latent_space_extraction.channel_alignment.align_raw_to_channels`).

    Returns
    -------
    (H, sfreq, channel_report) : tuple

    Raises
    ------
    ValueError
        Si tras la alineacion no quedan grabables los canales del
        entrenamiento, la ventana efectiva es menor que
        ``min_window_s``, o las dimensiones no son comparables con W0.
    """
    from src.latent_space_extraction.single_subject_eeg import (
        load_single_subject_eeg,
    )
    from src.latent_space_extraction.eeg_preprocessing import (
        extract_filtered_data_matrix,
    )
    from src.latent_space_extraction.hankel_dmd_extractor import (
        _build_multivariate_hankel,
    )

    raw = load_single_subject_eeg(
        subject_id=subject_id,
        session=session,
        task=task,
        db_path=db_path,
        t_start=t_start,
        t_stop=t_stop,
        preload=True,
        verbose=False,
    )

    # --- Alineacion de canales al conjunto del entrenamiento ---
    raw_aligned, chan_report = align_raw_to_channels(
        raw,
        model["global_channels"],
        missing_policy=missing_policy,
        max_missing=max_missing,
    )
    del raw
    if raw_aligned is None:
        raise ValueError(
            f"sub-{subject_id:02d}/{session}/{task}: canal alignment "
            f"failed -> {chan_report.get('skip_reason')}"
        )

    # --- Ventana efectiva >= min_window_s ---
    eff_window = raw_aligned.times[-1] + 1.0 / raw_aligned.info["sfreq"]
    if eff_window < float(min_window_s):
        raise ValueError(
            f"sub-{subject_id:02d}/{task}: effective window "
            f"{eff_window:.1f}s < {min_window_s}s"
        )

    X_filtered, _raw_filt, sfreq = extract_filtered_data_matrix(
        raw_aligned,
        l_freq=model["l_freq"],
        h_freq=model["h_freq"],
        verbose=False,
    )
    del raw_aligned, _raw_filt

    H = _build_multivariate_hankel(X_filtered, model["hankel_depth"])
    del X_filtered

    p_expected = model["n_channels"] * model["hankel_depth"]
    if H.shape[0] != p_expected:
        raise ValueError(
            f"sub-{subject_id:02d}/{task}: H tiene {H.shape[0]} filas, "
            f"se esperaban {p_expected} (canales*depth del train)."
        )
    if H.shape[0] * model["L"] != model["W0"].shape[0]:
        raise ValueError(
            f"sub-{subject_id:02d}/{task}: p*L != filas de W0."
        )
    return H, float(sfreq), chan_report


# =====================================================================
# Seleccion del subconjunto de condiciones del experimento
# =====================================================================

def select_condition_subset(model: dict, tasks: list[str]) -> dict:
    """Subconjunto de W_specific/lambda/r_specific para las tareas pedidas.

    Las tareas del experimento deben ser un subconjunto de las del
    modelo; el indice de clase del test es la POSICION en ``tasks``
    (el orden lo fija el experimento, no el entrenamiento).
    """
    model_tasks = list(model["tasks"])
    idx = []
    for t in tasks:
        if t not in model_tasks:
            raise ValueError(
                f"La tarea '{t}' no pertenece al modelo entrenado "
                f"({model_tasks}). El experimento debe usar un "
                f"subconjunto de las condiciones del train."
            )
        idx.append(model_tasks.index(t))
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"Tareas duplicadas en el experimento: {tasks}")
    return {
        "model_indices": idx,
        "W_specific": [model["W_specific"][i] for i in idx],
        "lambda_specific": [model["lambda_specific"][i] for i in idx],
        "r_specific": [model["r_specific"][i] for i in idx],
    }


# =====================================================================
# Estadisticas ML del entrenamiento (cache compartible entre
# experimentos que usan el MISMO modelo)
# =====================================================================

def _load_or_compute_ml_stats(model: dict, cache_dir: Path | None,
                               out_dir: Path) -> list[NDArray]:
    """Varianzas por modo nu_cj del ENTRENAMIENTO (todas las condiciones
    del modelo), con cache en ``cache_dir`` (o ``out_dir``).

    La cache es COMPARTIBLE entre experimentos: se valida contra las
    ``r_specific`` COMPLETAS del modelo (una entrada por condicion del
    train), y cada experimento selecciona su subconjunto despues.
    """
    r_specific = model["r_specific"]
    C_model = len(r_specific)
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(out_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats_path = cache_dir / "train_mode_stats.json"

    if stats_path.exists():
        try:
            with open(stats_path, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            nu = [np.asarray(x, dtype=np.float64) for x in
                  cached["nu_specific"]]
            if (len(nu) == C_model
                    and all(len(nu[c]) == r_specific[c]
                            for c in range(C_model))):
                print(f"  [ML] Estadisticas de entrenamiento (cache): "
                      f"{stats_path}")
                return nu
            print("  [ML] Cache de estadisticas incompatible: "
                  "recomputando.")
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            print(f"  [ML] Cache ilegible ({exc}): recomputando.")

    print("  [ML] Estimando varianzas por modo con las Hankel de "
          "entrenamiento (SS; puede tardar)...")
    t0 = time.time()
    stats = compute_training_mode_stats(
        Path(model["train_dir"]) / "hankel_matrices.npz",
        model["W0"], model["W_specific"], model["L"],
    )
    nu = stats["nu_specific"]
    try:
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump({
                "nu_specific": [np.asarray(x).tolist() for x in nu],
                "n_used_per_condition": stats["n_used_per_condition"],
                "n_train_recordings": stats["n_train_recordings"],
                "T2_mean": stats["T2_mean"],
                "train_dir": str(model["train_dir"]),
            }, fh, indent=2)
        print(f"  [ML] Guardado: {stats_path}")
    except OSError as exc:
        print(f"  [ML] WARN: no se pudo guardar cache: {exc}")
    print(f"  [ML] Hecho en {time.time() - t0:.1f} s "
          f"({stats['n_train_recordings']} grabaciones SS)")
    return nu


# =====================================================================
# Ejecucion del test
# =====================================================================

def run_classification(args: argparse.Namespace, model: dict,
                        subject_ids: list[int]) -> dict:
    """Ejecutar el test completo sobre los sujetos indicados."""
    tasks = list(args.tasks)
    C = len(tasks)
    methods = ["energy", "ml"] if args.method == "both" else [args.method]

    subset = select_condition_subset(model, tasks)
    W_specific = subset["W_specific"]
    r_specific = subset["r_specific"]
    if any(r < 1 for r in r_specific):
        raise ValueError(
            f"El modelo tiene condiciones sin modos especificos "
            f"(r_specific={r_specific}); no se puede clasificar."
        )
    d_min = int(min(r_specific))
    if args.rank_truncation == "min":
        d = d_min
    else:
        d = int(args.rank_truncation)
        if d < 1 or d > d_min:
            raise ValueError(
                f"--rank-truncation debe estar en [1, {d_min}] = "
                f"[1, min(r_specific)] para que la comparacion entre "
                f"clases sea justa (ranks actuales: {r_specific})."
            )

    nu_specific: list[NDArray] | None = None
    if "ml" in methods:
        nu_full = _load_or_compute_ml_stats(
            model, args.ml_stats_cache, Path(args.out_dir))
        nu_specific = [nu_full[i] for i in subset["model_indices"]]

    print("=" * 70)
    print("  TEST DE CLASIFICACION (CD-HSA hold-out de SESION)")
    print("=" * 70)
    print(f"  Modelo (train SS) : {model['train_dir']}")
    print(f"    sesion train    : {model['session']}")
    print(f"    r0 = {model['r0']}, r_specific (modelo) = "
          f"{model['r_specific']}")
    print(f"    r_specific (exp) = {r_specific}, d usado = {d}")
    print(f"  Test              : sesion {args.test_session}, sujetos "
          f"{subject_ids[0]}..{subject_ids[-1]} (N={len(subject_ids)})")
    print(f"  Tareas (clases)   : {tasks} (C={C})")
    print(f"  Metodos           : {methods}")
    print(f"  Ventana           : {args.t_start} s -> {args.t_end} s")
    print(f"  Canales           : {model['n_channels']} del train, "
          f"politica faltantes = {args.missing_policy} "
          f"(max {args.max_missing})")
    print(f"  Filtro            : {model['l_freq']}-{model['h_freq']} Hz, "
          f"depth {model['hankel_depth']}, L {model['L']}")
    print("")
    sys.stdout.flush()

    recordings: list[dict] = []
    n_ok = 0
    n_err = 0
    t0 = time.time()

    for subject_id in subject_ids:
        for c, task in enumerate(tasks):
            tag = (f"[{len(recordings) + 1}/{len(subject_ids) * C}] "
                   f"sub-{subject_id:02d}/{task}")
            print(f"  {tag} ...", end=" ")
            sys.stdout.flush()
            try:
                H, _sfreq, chan = load_and_build_test_hankel(
                    subject_id=subject_id,
                    session=args.test_session,
                    task=task,
                    model=model,
                    db_path=args.db_path,
                    t_start=args.t_start,
                    t_stop=args.t_end,
                    missing_policy=args.missing_policy,
                    max_missing=args.max_missing,
                    min_window_s=args.min_window,
                )
            except Exception as exc:
                print(f"SKIP ({str(exc)[:160]})")
                recordings.append({
                    "subject_id": int(subject_id), "task": task,
                    "true": c, "status": "load_error",
                    "error": str(exc)[:300],
                })
                n_err += 1
                continue

            r = compute_recording_energies(
                H, model["L"], model["W0"], W_specific,
                with_sigma=True,
            )
            del H
            E = r["E_specific"]
            resid2 = r["residual2"] if r["residual2"] > 0 else 1.0
            total2 = r["total2"] if r["total2"] > 0 else 1.0

            rec = {
                "subject_id": int(subject_id),
                "task": task,
                "true": c,
                "status": "ok",
                "T1": r["T1"],
                "T2": r["T2"],
                "total2": r["total2"],
                "common2": r["common2"],
                "residual2": r["residual2"],
                "common_frac": r["common2"] / total2,
                "sigma2": r.get("sigma2"),
                "budget_frac": [b / total2 for b in r["budget"]],
                "channels": {
                    "n_missing": len(chan.get("missing") or []),
                    "missing": chan.get("missing") or [],
                    "interpolated": chan.get("interpolated") or [],
                    "extra_dropped": len(chan.get("extra_dropped") or []),
                },
                "mode_energy_frac": {
                    tasks[c2]: (np.asarray(E[c2]) / resid2).tolist()
                    for c2 in range(C)
                },
                "scores": {},
            }

            for meth in methods:
                if meth == "energy":
                    s = energy_scores_at_rank(E, d)
                    denom = float(np.abs(s).sum()) or 1.0
                    rec["scores"]["energy"] = {
                        "raw": s.tolist(),
                        "phi": (s / resid2).tolist(),
                        "norm": (s / denom).tolist(),
                    }
                else:  # ml
                    g = ml_discriminants(
                        E, nu_specific, r["sigma2"], r["T2"])
                    denom = float(np.abs(g).sum()) or 1.0
                    rec["scores"]["ml"] = {
                        "raw": g.tolist(),
                        "norm": (g / denom).tolist(),
                    }

            pred, _margin = classify_from_scores(
                np.array(rec["scores"][methods[0]]["raw"]))
            ok_flag = "OK " if pred == c else "MAL"
            n_interp = len(rec["channels"]["interpolated"])
            interp_str = f" interp={n_interp}" if n_interp else ""
            print(f"{ok_flag}  pred={tasks[pred]}{interp_str}")
            sys.stdout.flush()

            recordings.append(rec)
            n_ok += 1

    elapsed = time.time() - t0
    print(f"\n  Grabaciones procesadas: OK={n_ok}, errores/skips={n_err} "
          f"({elapsed:.1f} s)")

    recs_ok = [r for r in recordings if r["status"] == "ok"]
    metrics: dict = {}
    for meth in methods:
        if recs_ok:
            metrics[meth] = _compute_metrics(
                recs_ok, meth, tasks, d,
                n_perm=args.n_perm, seed=args.seed)
        else:
            metrics[meth] = {}

    results = {
        "meta": {
            "pipeline": "run_cdhsa_session2_classification",
            "timestamp": datetime.now().isoformat(),
            "design": "session_hold_out",
            "train_results_dir": str(model["train_dir"]),
            "train_session": model["session"],
            "train_super_subject_ids": model["hankel_info"].get(
                "super_subject_ids", []),
            "session": args.test_session,
            "tasks": tasks,
            "model_tasks": list(model["tasks"]),
            "test_subject_ids": [int(x) for x in subject_ids],
            "method": args.method,
            "methods_computed": methods,
            "rank_truncation_d": d,
            "rank_truncation_mode": args.rank_truncation,
            "r_specific": r_specific,
            "r_specific_model": model["r_specific"],
            "r0": model["r0"],
            "L": model["L"],
            "hankel_depth": model["hankel_depth"],
            "l_freq": model["l_freq"],
            "h_freq": model["h_freq"],
            "n_channels": model["n_channels"],
            "missing_policy": args.missing_policy,
            "max_missing": args.max_missing,
            "t_start": args.t_start,
            "t_end": args.t_end,
            "n_perm": args.n_perm,
            "seed": args.seed,
            "n_ok": n_ok,
            "n_errors": n_err,
            "elapsed_s": elapsed,
        },
        "recordings": recordings,
        "metrics": metrics,
    }
    return results


# =====================================================================
# Metricas (multi-clase: C >= 2)
# =====================================================================

def _compute_metrics(recs: list[dict], method: str, tasks: list[str],
                     d: int, *, n_perm: int, seed: int) -> dict:
    """Metricas de un metodo sobre las grabaciones OK (C clases)."""
    C = len(tasks)
    y_true = np.array([r["true"] for r in recs], dtype=int)
    subj = np.array([r["subject_id"] for r in recs], dtype=int)
    raw = np.array([r["scores"][method]["raw"] for r in recs],
                   dtype=np.float64)
    y_pred = np.array([classify_from_scores(s)[0] for s in raw], dtype=int)
    correct = y_pred == y_true
    n = len(recs)
    k = int(correct.sum())
    acc = k / n if n else float("nan")

    conf = confusion_matrix(y_true, y_pred, C)
    per_task: dict[str, float] = {}
    for c, task in enumerate(tasks):
        m = y_true == c
        per_task[task] = (float(np.mean(y_pred[m] == c))
                          if m.sum() else float("nan"))

    chance = 1.0 / C
    p_binom = exact_binomial_pvalue(k, n, chance)
    lo, hi = clopper_pearson_interval(k, n)
    p_perm = paired_label_swap_pvalue(y_true, y_pred, subj,
                                      n_perm=n_perm, seed=seed)

    norm = raw / (np.abs(raw).sum(axis=1, keepdims=True) + 1e-300)

    # AUC one-vs-rest por clase: delta_c = s_c - max_{c' != c} s_c'
    auc_per_class: dict[str, float] = {}
    for c, task in enumerate(tasks):
        others = [norm[:, c2] for c2 in range(C) if c2 != c]
        delta = norm[:, c] - np.max(others, axis=0)
        auc_per_class[task] = auc_mann_whitney(delta, y_true == c)
    auc_macro = float(np.nanmean(list(auc_per_class.values()))) \
        if n else float("nan")

    margins = np.array([
        norm[i, y_true[i]] - np.max(
            [norm[i, c2] for c2 in range(C) if c2 != y_true[i]])
        for i in range(n)])

    out = {
        "n_test": n,
        "n_correct": k,
        "accuracy": acc,
        "chance_level": chance,
        "accuracy_per_task": per_task,
        "confusion_matrix": conf.tolist(),
        "confusion_labels": {"rows": "true", "cols": "predicted",
                             "order": tasks},
        "binomial_p_exact_onesided": p_binom,
        "ci95_clopper_pearson": [lo, hi],
        "perm_p_paired_within_subject": p_perm,
        "auc_ovr_per_class": auc_per_class,
        "auc": auc_macro,
        "margin_mean": float(np.mean(margins)) if n else float("nan"),
        "margin_min": float(np.min(margins)) if n else float("nan"),
        "margin_max": float(np.max(margins)) if n else float("nan"),
    }

    if method == "energy":
        # Sensibilidad al rango de truncamiento (1..d): la fraccion de
        # energia por modo ya esta normalizada por el residual (mismo
        # denominador para todas las clases -> argmax invariante).
        acc_by_rank: dict[int, float] = {}
        for dd in range(1, d + 1):
            preds = []
            for r in recs:
                s = np.array([
                    np.sum(r["mode_energy_frac"][t][:dd])
                    for t in tasks
                ])
                preds.append(classify_from_scores(s)[0])
            acc_by_rank[dd] = float(
                np.mean(np.array(preds) == y_true)) if n else float("nan")
        out["accuracy_by_rank"] = acc_by_rank

    return out


# =====================================================================
# Textos de resumen
# =====================================================================

def _summary_text(results: dict) -> str:
    meta = results["meta"]
    tasks = meta["tasks"]
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  RESUMEN DEL TEST DE CLASIFICACION (hold-out de sesion)")
    lines.append("=" * 70)
    lines.append(f"  Modelo (train SS): {meta['train_results_dir']}")
    lines.append(f"    sesion train={meta['train_session']}, "
                 f"r0={meta['r0']}, "
                 f"r_specific={meta['r_specific']}, "
                 f"d={meta['rank_truncation_d']}")
    lines.append(f"  Test            : sesion {meta['session']}, sujetos "
                 f"{meta['test_subject_ids'][0]}.."
                 f"{meta['test_subject_ids'][-1]} "
                 f"({len(meta['test_subject_ids'])}), "
                 f"grabaciones OK={meta['n_ok']}, "
                 f"errores={meta['n_errors']}")
    n_interp = sum(len(r["channels"]["interpolated"])
                   for r in results["recordings"]
                   if r["status"] == "ok")
    n_chanskip = sum(1 for r in results["recordings"]
                     if r["status"] == "load_error"
                     and "alignment" in str(r.get("error", "")))
    lines.append(f"  Canales         : {meta['n_channels']} del train; "
                 f"{n_interp} canales interpolados en total; "
                 f"{n_chanskip} grabaciones saltadas por canales")
    for meth, m in results["metrics"].items():
        if not m:
            continue
        lines.append("-" * 70)
        lines.append(f"  Metodo: {meth} ({_METHOD_LABEL.get(meth, '')})")
        lines.append(f"    Accuracy total : {m['n_correct']}/"
                     f"{m['n_test']} = {m['accuracy'] * 100:.1f}% "
                     f"(azar = {m['chance_level'] * 100:.1f}%)")
        for t in tasks:
            lines.append(f"    Accuracy {t:<12}: "
                         f"{m['accuracy_per_task'][t] * 100:.1f}%")
        cm = m["confusion_matrix"]
        lines.append(f"    Confusion (filas=verdad, cols=pred): {cm}")
        lines.append(f"    p binomial exacto      : "
                     f"{m['binomial_p_exact_onesided']:.3e}")
        lines.append(f"    IC 95% (Clopper-Pearson): "
                     f"[{m['ci95_clopper_pearson'][0] * 100:.1f}%, "
                     f"{m['ci95_clopper_pearson'][1] * 100:.1f}%]")
        lines.append(f"    p permutacion pareada  : "
                     f"{m['perm_p_paired_within_subject']:.4f}")
        lines.append(f"    AUC (macro, one-vs-rest): "
                     f"{m['auc']:.3f}")
        for t in tasks:
            lines.append(f"      AUC {t:<12}: "
                         f"{m['auc_ovr_per_class'][t]:.3f}")
        lines.append(f"    Margen medio           : "
                     f"{m['margin_mean']:+.4f}")
        if "accuracy_by_rank" in m:
            curve = m["accuracy_by_rank"]
            s = ", ".join(f"{dd}:{v * 100:.0f}%"
                          for dd, v in curve.items())
            lines.append(f"    Accuracy por rango d   : {s}")
    lines.append("=" * 70)
    return "\n".join(lines)


# =====================================================================
# Salidas: CSV
# =====================================================================

def _write_csv(results: dict, path: Path) -> None:
    tasks = results["meta"]["tasks"]
    methods = results["meta"]["methods_computed"]
    fields = (["subject_id", "task", "true", "status"]
              + [f"pred_{m}" for m in methods]
              + [f"correct_{m}" for m in methods]
              + [f"score_raw_{m}_{t}" for m in methods for t in tasks]
              + [f"score_norm_{m}_{t}" for m in methods for t in tasks]
              + (["phi_" + t for t in tasks] if "energy" in methods else [])
              + ["n_channels_missing", "channels_interpolated",
                 "common_frac", "sigma2", "T2", "error"])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results["recordings"]:
            ok = r["status"] == "ok"
            row = {
                "subject_id": r["subject_id"],
                "task": r["task"],
                "true": r["true"],
                "status": r["status"],
                "n_channels_missing": (
                    r["channels"]["n_missing"] if ok else ""),
                "channels_interpolated": (
                    ";".join(r["channels"]["interpolated"]) if ok else ""),
                "common_frac": (round(r["common_frac"], 6) if ok else ""),
                "sigma2": (r.get("sigma2") if ok else ""),
                "T2": r.get("T2", ""),
                "error": r.get("error", ""),
            }
            for m in methods:
                sc = r.get("scores", {}).get(m)
                if sc is None:
                    row[f"pred_{m}"] = ""
                    row[f"correct_{m}"] = ""
                    continue
                pred, _mg = classify_from_scores(np.array(sc["raw"]))
                row[f"pred_{m}"] = tasks[pred]
                row[f"correct_{m}"] = int(pred == r["true"])
                for ci, t in enumerate(tasks):
                    row[f"score_raw_{m}_{t}"] = round(sc["raw"][ci], 8)
                    row[f"score_norm_{m}_{t}"] = round(sc["norm"][ci], 8)
                if m == "energy":
                    for ci, t in enumerate(tasks):
                        row[f"phi_{t}"] = round(sc["phi"][ci], 8)
            writer.writerow(row)


# =====================================================================
# Graficos (etiquetas en ingles, listas para el paper)
# =====================================================================

def _ok_recs(results: dict) -> list[dict]:
    return [r for r in results["recordings"] if r["status"] == "ok"]


def _plot_confusion(m: dict, tasks: list[str], out: Path,
                    suffix: str) -> Path:
    conf = np.asarray(m["confusion_matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    im = ax.imshow(conf, cmap="Blues", vmin=0)
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            row_n = conf[i].sum()
            pct = 100.0 * conf[i, j] / row_n if row_n else 0.0
            color = ("white" if conf[i, j] > conf.max() * 0.6
                     else "black")
            ax.text(j, i, f"{int(conf[i, j])}\n({pct:.0f}%)",
                    ha="center", va="center", color=color, fontsize=12)
    ax.set_xticks(range(len(tasks)), labels=tasks)
    ax.set_yticks(range(len(tasks)), labels=tasks)
    ax.set_xlabel("Predicted condition")
    ax.set_ylabel("True condition")
    ax.set_title(f"Confusion matrix "
                 f"(accuracy {m['accuracy'] * 100:.1f}%)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="count")
    path = out / f"confusion_matrix{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_accuracy_summary(m: dict, tasks: list[str], out: Path,
                           suffix: str) -> Path:
    lo, hi = m["ci95_clopper_pearson"]
    acc = m["accuracy"]
    chance = m["chance_level"]
    labels = ["Overall"] + [f"True {t}" for t in tasks]
    values = [acc] + [m["accuracy_per_task"][t] for t in tasks]
    errs = [[acc - lo] + [0.0] * len(tasks),
             [hi - acc] + [0.0] * len(tasks)]
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    x = np.arange(len(labels))
    colors = [_C_OK] + [_CLASS_COLORS[(i + 1) % len(_CLASS_COLORS)]
                        for i in range(len(tasks))]
    bars = ax.bar(x, np.array(values) * 100,
                  yerr=np.array(errs) * 100, capsize=5,
                  color=colors,
                  error_kw={"ecolor": "#333333", "lw": 1.5})
    for xi, v in zip(x, values):
        ax.text(xi, v * 100 + 3, f"{v * 100:.1f}%",
                ha="center", fontsize=11)
    ax.axhline(chance * 100, color=_C_GREY, ls="--", lw=1.2,
               label=f"Chance ({chance * 100:.1f}%)")
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification accuracy "
                 f"(n = {m['n_test']} recordings)")
    ax.legend(frameon=False)
    path = out / f"accuracy_summary{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_score_scatter(recs: list[dict], m: dict, tasks: list[str],
                        out: Path, suffix: str) -> Path | None:
    """Scatter 2D de puntuaciones normalizadas (solo C=2)."""
    if len(tasks) != 2:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 5.0), constrained_layout=True)
    marks = ["o", "s"]
    for c, t in enumerate(tasks):
        xs = np.array([r["scores_norm"][0] for r in recs
                       if r["true"] == c])
        ys = np.array([r["scores_norm"][1] for r in recs
                       if r["true"] == c])
        ax.scatter(xs, ys, marker=marks[c], s=48, alpha=0.8,
                   color=(_C_OK if c == 0 else _C_BAD),
                   edgecolor="white", linewidth=0.6,
                   label=f"True {t} (n={len(xs)})")
    lim = 1.05
    ax.plot([-lim, lim], [-lim, lim], color=_C_GREY, ls="--", lw=1.0,
            label="Decision boundary")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"Normalized score — {tasks[0]}")
    ax.set_ylabel(f"Normalized score — {tasks[1]}")
    ax.set_title("Per-recording scores "
                 f"(AUC = {m['auc']:.3f})")
    ax.axhline(0, color="lightgrey", lw=0.6)
    ax.axvline(0, color="lightgrey", lw=0.6)
    ax.legend(frameon=False, loc="best")
    ax.set_aspect("equal")
    path = out / f"score_scatter{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_pairwise_scatter(recs: list[dict], m: dict, tasks: list[str],
                           out: Path, suffix: str) -> list[Path]:
    """Grid de scatters por pares de clases (3 <= C <= 5)."""
    C = len(tasks)
    written: list[Path] = []
    if C < 3 or C > 5:
        return written
    pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]
    ncols = min(3, len(pairs))
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols,
                                                    4.4 * nrows),
                             squeeze=False, constrained_layout=True)
    for ax in axes.flat:
        ax.set_visible(False)
    for k, (i, j) in enumerate(pairs):
        ax = axes[k // ncols][k % ncols]
        ax.set_visible(True)
        for c in (i, j):
            xs = np.array([r["scores_norm"][i] for r in recs
                           if r["true"] == c])
            ys = np.array([r["scores_norm"][j] for r in recs
                           if r["true"] == c])
            ax.scatter(xs, ys, s=36, alpha=0.8,
                       color=_CLASS_COLORS[c % len(_CLASS_COLORS)],
                       edgecolor="white", linewidth=0.5,
                       label=f"True {tasks[c]} (n={len(xs)})")
        lim = 1.05
        ax.plot([-lim, lim], [-lim, lim], color=_C_GREY, ls="--", lw=1.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"Score — {tasks[i]}")
        ax.set_ylabel(f"Score — {tasks[j]}")
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Pairwise score scatter "
                 f"(macro AUC = {m['auc']:.3f})", fontsize=13)
    path = out / f"pairwise_scatter{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


def _plot_margin_histogram(recs: list[dict], m: dict,
                           out: Path, suffix: str) -> Path:
    margins = np.array([r["margin_true"] for r in recs])
    correct = margins > 0
    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    bins = np.linspace(-1, 1, 21)
    ax.hist(margins[correct], bins=bins, color=_C_OK, alpha=0.85,
            label=f"Correct (n={int(correct.sum())})")
    ax.hist(margins[~correct], bins=bins, color=_C_BAD, alpha=0.85,
            label=f"Incorrect (n={int((~correct).sum())})")
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("Margin  (score of true class − best other class)")
    ax.set_ylabel("Number of recordings")
    ax.set_title("Classification margin "
                 f"(mean {m['margin_mean']:+.3f})")
    ax.legend(frameon=False)
    path = out / f"margin_histogram{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_per_subject_grid(recs: list[dict], tasks: list[str],
                           out: Path, suffix: str) -> Path:
    subjects = sorted({r["subject_id"] for r in recs})
    grid = np.full((len(subjects), len(tasks)), -1, dtype=int)
    for r in recs:
        i = subjects.index(r["subject_id"])
        grid[i, r["true"]] = int(r["correct"])
    fig, ax = plt.subplots(
        figsize=(4.0, max(3.0, 0.28 * len(subjects) + 1.4)),
        constrained_layout=True)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([_C_BAD, _C_OK])
    masked = np.ma.masked_where(grid < 0, grid)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tasks)), labels=tasks)
    ax.set_yticks(range(len(subjects)),
                  labels=[f"sub-{s:02d}" for s in subjects])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] < 0:
                ax.text(j, i, "—", ha="center", va="center",
                        color=_C_GREY)
    ax.set_xlabel("Task")
    ax.set_title("Outcome per test subject\n"
                 "(blue = correct, orange = misclassified)")
    path = out / f"per_subject_grid{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_score_distributions(recs: list[dict], tasks: list[str],
                              out: Path, suffix: str) -> Path:
    own = np.array([r["scores_norm"][r["true"]] for r in recs])
    other = np.array([
        np.max([r["scores_norm"][c] for c in range(len(tasks))
                if c != r["true"]]) for r in recs])
    fig, ax = plt.subplots(figsize=(5.4, 4.2), constrained_layout=True)
    bp = ax.boxplot([own, other], patch_artist=True, widths=0.5,
                    showfliers=False)
    for patch, color in zip(bp["boxes"], [_C_OK, _C_BAD]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for vals, color, pos in ((own, _C_OK, 1), (other, _C_BAD, 2)):
        xj = rng.normal(pos, 0.05, len(vals))
        ax.scatter(xj, vals, s=22, alpha=0.75, color=color,
                   edgecolor="white", linewidth=0.4)
    ax.set_xticks([1, 2], labels=["Own-class score",
                                  "Best other-class score"])
    ax.set_ylabel("Normalized score")
    ax.set_title("Score separation (own vs other conditions)")
    path = out / f"score_distributions{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_rank_sensitivity(m: dict, out: Path, suffix: str) -> Path:
    curve = m.get("accuracy_by_rank")
    if not curve:
        raise ValueError("accuracy_by_rank no disponible")
    ds = sorted(int(k) for k in curve)
    accs = [curve[str(k)] if str(k) in curve else curve[k]
            for k in ds]
    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    ax.plot(ds, np.array(accs) * 100, "o-", color=_C_OK, lw=1.8,
            ms=5)
    d_used = m.get("d_used", ds[-1])
    ax.axvline(d_used, color=_C_GREY, ls=":", lw=1.2,
               label=f"d used = {d_used}")
    ax.axhline(m["chance_level"] * 100, color=_C_GREY, ls="--", lw=1.0,
               label=f"Chance ({m['chance_level'] * 100:.1f}%)")
    ax.set_xlabel("Truncation rank d (modes per class)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. subspace rank")
    ax.legend(frameon=False)
    path = out / f"rank_sensitivity{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_roc(m: dict, recs: list[dict], out: Path,
              suffix: str) -> Path | None:
    """ROC de la primera clase contra el resto (solo C=2)."""
    if len(m.get("tasks", [])) != 2:
        return None
    delta = np.array([r["delta_score"] for r in recs])
    y = np.array([r["true"] == 0 for r in recs])
    order = np.argsort(-delta)
    ys = y[order]
    n_pos = int(ys.sum())
    n_neg = int(len(ys) - ys.sum())
    tpr = np.concatenate([[0.0], np.cumsum(ys) / max(n_pos, 1)])
    fpr = np.concatenate([[0.0], np.cumsum(~ys) / max(n_neg, 1)])
    fig, ax = plt.subplots(figsize=(4.8, 4.6), constrained_layout=True)
    ax.step(fpr, tpr, where="post", color=_C_OK, lw=2,
            label=f"ROC (AUC = {m['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=_C_GREY, ls="--", lw=1.0,
            label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC — positive class: {m.get('tasks', ['', ''])[0]}")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    path = out / f"roc_curve{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_mode_energy_curves(recs: list[dict], tasks: list[str],
                             out: Path) -> Path:
    C = len(tasks)
    fig, axes = plt.subplots(1, C, figsize=(4.6 * C, 4.0), sharey=True,
                             squeeze=False, constrained_layout=True)
    for c in range(C):
        ax = axes[0][c]
        recs_c = [r for r in recs if r["true"] == c]
        for c2, t in enumerate(tasks):
            lens = [len(r["mode_energy"][t]) for r in recs_c
                    if t in r["mode_energy"]]
            if not lens:
                continue
            jmax = max(lens)
            curves = np.full((len(recs_c), jmax), np.nan)
            for i, r in enumerate(recs_c):
                v = r["mode_energy"][t]
                curves[i, :len(v)] = v
            mean = np.nanmean(curves, axis=0)
            style = ("o-" if c2 == c else "s--")
            color = (_C_OK if c2 == c else _C_BAD)
            ax.plot(np.arange(1, jmax + 1), mean, style, color=color,
                    ms=3.5, lw=1.4,
                    label=(f"{t} modes" if c2 == c
                           else f"{t} modes (cross)"))
        ax.set_title(f"True condition: {tasks[c]} "
                     f"(n={len(recs_c)})")
        ax.set_xlabel("Mode index j")
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel("Fraction of residual energy per mode")
    path = out / "mode_energy_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_energy_budget(recs: list[dict], tasks: list[str],
                        out: Path) -> Path:
    order = sorted(range(len(recs)), key=lambda i: recs[i]["true"])
    budget = np.array([recs[i]["budget"] for i in order])
    labels = ([f"Common (W0, r0={recs[0]['r0']})"]
              + [f"Specific {t}" for t in tasks] + ["Unexplained"])
    colors = ["#56B4E9"] + [_C_OK, _C_BAD, "#CC79A7", "#E69F00",
                            _C_GREY]
    fig, ax = plt.subplots(
        figsize=(max(7.0, 0.16 * len(recs) + 2.0), 4.2),
        constrained_layout=True)
    x = np.arange(len(recs))
    bottom = np.zeros(len(recs))
    for k in range(budget.shape[1]):
        vals = budget[:, k] * 100
        ax.bar(x, vals, bottom=bottom, width=0.85,
               color=colors[k % len(colors)], label=labels[k])
        bottom += vals
    trues = np.array([recs[i]["true"] for i in order])
    bounds = np.where(np.diff(trues) != 0)[0]
    for b in bounds:
        ax.axvline(b + 0.5, color="black", lw=1.0, ls=":")
    ax.set_xticks(x[::max(1, len(recs) // 20)])
    ax.set_xticklabels(
        [f"{recs[i]['subject_id']:02d}" for i in order
         ][::max(1, len(recs) // 20)], fontsize=7)
    ax.set_xlabel("Test recording (sorted by true condition)")
    ax.set_ylabel("Share of level-2 energy (%)")
    ax.set_title("Energy budget of each held-out recording")
    ax.legend(frameon=False, fontsize=8, ncol=min(4, len(labels)))
    path = out / "energy_budget.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_channel_alignment(results: dict, out: Path) -> Path:
    """Resumen de la alineacion de canales del experimento."""
    recs = results["recordings"]
    n_ok = sum(1 for r in recs if r["status"] == "ok")
    n_skip_align = sum(1 for r in recs
                       if r["status"] == "load_error"
                       and "alignment" in str(r.get("error", "")))
    n_skip_other = len(recs) - n_ok - n_skip_align
    n_missing = [r["channels"]["n_missing"] for r in recs
                 if r["status"] == "ok"]
    n_interp = [len(r["channels"]["interpolated"]) for r in recs
                if r["status"] == "ok"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0),
                             constrained_layout=True)
    ax = axes[0]
    cats = ["Aligned (as-is)", "Aligned (interpolated)",
            "Skipped (channels)", "Skipped (other)"]
    vals = [sum(1 for k in n_missing if k == 0),
            sum(1 for k in n_interp if k > 0),
            n_skip_align, n_skip_other]
    bars = ax.bar(cats, vals, color=[_C_OK, _C_OTHER, _C_BAD, _C_GREY])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, str(v),
                ha="center", fontsize=11)
    ax.set_ylabel("Recordings")
    ax.set_title("Channel alignment outcome")
    ax.tick_params(axis="x", rotation=15)
    ax = axes[1]
    if n_missing:
        xmax = max(2, max(n_missing))
        bins = np.arange(-0.5, xmax + 1.5, 1.0)
        ax.hist([n_missing, n_interp], bins=bins,
                color=[_C_GREY, _C_OTHER], label=["Missing",
                                                  "Interpolated"])
        ax.set_xticks(range(0, xmax + 2))
    ax.set_xlabel("Missing training channels per recording")
    ax.set_ylabel("Recordings")
    ax.set_title("Missing-channel counts (usable recordings)")
    ax.legend(frameon=False)
    path = out / "channel_alignment_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_all_plots(results: dict, out_dir: Path, tasks: list[str],
                   methods: list[str]) -> list[Path]:
    """Generar todos los graficos del experimento."""
    recs_raw = _ok_recs(results)
    written: list[Path] = []
    out_dir = Path(out_dir)

    # Registros de plot con los alias que usan las funciones
    # (mode_energy/budget/r0 comunes a todos los metodos).
    recs_plot: list[dict] = []
    for r in recs_raw:
        rec = dict(r)
        rec["mode_energy"] = r["mode_energy_frac"]
        rec["budget"] = r["budget_frac"]
        rec["r0"] = results["meta"]["r0"]
        recs_plot.append(rec)

    for meth in methods:
        m = results["metrics"].get(meth)
        if not m:
            continue
        suffix = "" if len(methods) == 1 else f"_{meth}"

        # Campos derivados por grabacion para este metodo
        recs = []
        for rp in recs_plot:
            sc = rp["scores"][meth]
            norm = np.array(sc["norm"])
            rec = dict(rp)
            rec["scores_norm"] = norm
            rec["delta_score"] = float(norm[0] - norm[1])
            rec["margin_true"] = float(
                norm[rp["true"]] - np.max(
                    [norm[c] for c in range(len(tasks))
                     if c != rp["true"]]))
            rec["correct"] = int(np.argmax(sc["raw"]) == rp["true"])
            recs.append(rec)
        m = dict(m)
        m["d_used"] = results["meta"]["rank_truncation_d"]
        m["tasks"] = tasks

        written.append(_plot_confusion(m, tasks, out_dir, suffix))
        written.append(_plot_accuracy_summary(m, tasks, out_dir, suffix))
        p = _plot_score_scatter(recs, m, tasks, out_dir, suffix)
        if p is not None:
            written.append(p)
        written.extend(_plot_pairwise_scatter(recs, m, tasks, out_dir,
                                              suffix))
        written.append(_plot_margin_histogram(recs, m, out_dir,
                                              suffix))
        written.append(_plot_per_subject_grid(recs, tasks, out_dir,
                                              suffix))
        written.append(_plot_score_distributions(recs, tasks, out_dir,
                                                suffix))
        if meth == "energy":
            written.append(_plot_rank_sensitivity(m, out_dir, suffix))
        p = _plot_roc(m, recs, out_dir, suffix)
        if p is not None:
            written.append(p)

    # Graficos agnosticos del metodo (energias puras)
    if "energy" in methods:
        written.append(_plot_mode_energy_curves(recs_plot, tasks,
                                                out_dir))
        written.append(_plot_energy_budget(recs_plot, tasks, out_dir))
    # Resumen de alineacion de canales (siempre)
    written.append(_plot_channel_alignment(results, out_dir))
    return written


# =====================================================================
# CLI
# =====================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clasificacion de grabaciones de sujetos simples (sesion de "
            "test, p. ej. session2) con un modelo CD-HSA entrenado con "
            "SUPER-SUJETOS de otra sesion (hold-out de sesion)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--train-results-dir", type=str, required=True,
        help=("Directorio de resultados del entrenamiento CD-HSA con "
              "super-sujetos (contiene cdhsa_arrays.npz, "
              "hankel_info.json, config.json)."),
    )
    parser.add_argument(
        "--test-session", type=str, required=True,
        help="Sesion de test (p. ej. session2). Debe difereir de la del train.",
    )
    parser.add_argument(
        "--session", type=str, default=None,
        help=("[deprecated] alias de --test-session (compatibilidad)."),
    )

    mode_subj = parser.add_mutually_exclusive_group()
    mode_subj.add_argument(
        "--subject-ids", type=int, nargs="+", default=None,
        help="Lista explicita de sujetos de test.",
    )
    mode_subj.add_argument(
        "--n-subjects", type=int, default=None,
        help="Cantidad de sujetos de test (rango automatico).",
    )
    parser.add_argument("--subject-start-offset", type=int, default=1)

    parser.add_argument(
        "--tasks", type=str, nargs="+", required=True,
        metavar="TASK",
        help=("2 o mas tareas (subconjunto de las del modelo); el orden "
              "fija el indice de clase del test."),
    )
    parser.add_argument(
        "--t-start", type=float, default=None,
        help=("Inicio de la ventana temporal (s). Default: la del "
              "directorio de entrenamiento (from{t0}s_to{t1}s); debe "
              "COINCIDIR con el entrenamiento."),
    )
    parser.add_argument(
        "--t-end", type=float, default=None,
        help="Fin de la ventana temporal (s). Default: idem --t-start.",
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Raiz del dataset Gedai. Default: config.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help=("Directorio de salida. Default: "
              "BASE_RESULTS_PATH/cdhsa_classification/{session}/..."),
    )

    g = parser.add_argument_group("Clasificacion")
    g.add_argument(
        "--method", type=str, default="energy",
        choices=["energy", "ml", "both"],
        help=("Regla de decision: 'energy' (energia residual capturada "
              "por W_c a rango comun d) o 'ml' (Bayes bajo covarianza "
              "spiked; requiere hankel_matrices.npz del train). "
              "Default: energy."),
    )
    g.add_argument(
        "--rank-truncation", type=str, default="min",
        help=("'min' (default: d = min r_c, comparacion justa) o un "
              "entero en [1, min r_c]."),
    )
    g.add_argument(
        "--n-perm", type=int, default=10000,
        help="Permutaciones del test pareado por sujeto. Default: 10000",
    )
    g.add_argument("--seed", type=int, default=20261006,
                   help="Semilla del test de permutacion. Default: 20261006")
    g.add_argument("--skip-plots", action="store_true",
                   help="No generar los graficos PNG.")

    g = parser.add_argument_group("Alineacion de canales")
    g.add_argument(
        "--missing-policy", type=str, default="interpolate",
        choices=list(MISSING_POLICIES),
        help=("Politica para canales del entrenamiento ausentes en la "
              "grabacion de test. Default: interpolate."),
    )
    g.add_argument(
        "--max-missing", type=int, default=4,
        help=("Maximo de canales ausentes a interpolar. Default: 4."),
    )
    g.add_argument(
        "--min-window", type=float, default=30.0,
        help=("Ventana efectiva minima (s) tras el recorte. Default: 30."),
    )

    g = parser.add_argument_group("Avanzado")
    g.add_argument(
        "--allow-same-session", action="store_true",
        help=("Permitir test en la MISMA sesion del train (rompe el "
              "diseno hold-out de sesion; solo para sanity checks "
              "in-sample)."),
    )
    g.add_argument(
        "--ml-stats-cache", type=str, default=None,
        help=("Directorio de cache compartida de las estadisticas ML "
              "del modelo (train_mode_stats.json); util cuando varios "
              "experimentos comparten modelo. Default: out-dir."),
    )

    args = parser.parse_args(argv)
    # Compatibilidad: --session como alias de --test-session
    if args.session is not None:
        args.test_session = args.session
    return args


def _resolve_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        try:
            from src.utils.config import BASE_RESULTS_PATH
            base = str(BASE_RESULTS_PATH)
        except ImportError:
            raise ValueError(
                "--out-dir es obligatorio si src.utils.config no "
                "define BASE_RESULTS_PATH."
            )
        method_tag = args.method
        suffix = "" if args.rank_truncation == "min" \
            else f"_d{args.rank_truncation}"
        t0 = args.t_start if args.t_start is not None else "any"
        t1 = args.t_end if args.t_end is not None else "any"
        out_dir = Path(
            f"{base}/cdhsa_classification/{args.test_session}"
            f"/standalone_{method_tag}{suffix}"
            f"/from{t0}s_to{t1}s_{'_'.join(args.tasks)}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    return out_dir


# =====================================================================
# MAIN
# =====================================================================

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Test multi-clase: al menos 2 tareas, sin duplicados.
    if len(args.tasks) < 2:
        print(f"[ERROR] --tasks requiere al menos 2 tareas "
              f"(se recibieron {len(args.tasks)}: {list(args.tasks)}).")
        return 1
    if len(set(args.tasks)) != len(args.tasks):
        print(f"[ERROR] --tasks con duplicados: {list(args.tasks)}.")
        return 1

    if args.rank_truncation != "min":
        try:
            int(args.rank_truncation)
        except ValueError:
            print("[ERROR] --rank-truncation debe ser 'min' o un entero.")
            return 1

    # 1. Modelo entrenado (SS)
    print("Cargando modelo entrenado (super-sujetos)...")
    model = load_cdhsa_model(args.train_results_dir)

    try:
        select_condition_subset(model, list(args.tasks))
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    # 2. Guarda ANTI-FUGA de sesion (diseno hold-out de sesion)
    train_session = model.get("session")
    if train_session == args.test_session and not args.allow_same_session:
        print(
            f"[ERROR] FUGA DE DATOS POR SESION: el modelo fue "
            f"entrenado en '{train_session}' y el test usa la MISMA "
            f"sesion. Este experimento es un hold-out de SESION "
            f"(mismos sujetos, sesion distinta); usa --allow-same-session "
            f"unicamente para sanity checks in-sample."
        )
        return 1
    if train_session is not None:
        print(f"  Diseno: hold-out de SESION (train '{train_session}' -> "
              f"test '{args.test_session}'). Los mismos sujetos "
              f"participan en train (como miembros de SS) y test; la "
              f"separacion es la sesion.")

    # 3. Ventana temporal: default = la del entrenamiento (obligatoria
    #    la coincidencia con el preprocesado del train)
    if args.t_start is None or args.t_end is None:
        t0p, t1p = parse_time_window_from_dir(args.train_results_dir)
        if args.t_start is None:
            args.t_start = t0p
        if args.t_end is None:
            args.t_end = t1p
        if args.t_start is None or args.t_end is None:
            print("[ERROR] No se pudo deducir la ventana temporal del "
                  "directorio de entrenamiento; pasa --t-start/--t-end "
                  "(deben coincidir con el entrenamiento).")
            return 1
        print(f"  Ventana tomada del nombre del directorio de train: "
              f"{args.t_start} s -> {args.t_end} s")
    if args.t_start is not None and args.t_end is not None \
            and args.t_start >= args.t_end:
        print(f"[ERROR] Ventana temporal invalida: "
              f"{args.t_start} >= {args.t_end}.")
        return 1

    # 4. Sujetos de test
    from src.latent_space_extraction.single_subject_eeg import (
        resolve_single_subject_ids,
    )
    subject_ids = resolve_single_subject_ids(
        n_subjects=args.n_subjects if args.n_subjects is not None else 60,
        subject_start_offset=args.subject_start_offset,
        subject_ids=args.subject_ids,
    )

    # 5. Directorio de salida
    args.out_dir = _resolve_out_dir(args)

    # 6. Ejecutar
    try:
        results = run_classification(args, model, subject_ids)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    # 7. Guardar
    out_dir = Path(args.out_dir)
    with open(out_dir / "classification_results.json", "w",
              encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\n  [OK] {out_dir / 'classification_results.json'}")

    _write_csv(results, out_dir / "classification_per_recording.csv")
    print(f"  [OK] {out_dir / 'classification_per_recording.csv'}")

    summary = _summary_text(results)
    print(summary)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as fh:
        fh.write(summary)

    # 8. Graficos
    if not args.skip_plots and results["meta"]["n_ok"] > 0:
        try:
            plot_files = make_all_plots(
                results, out_dir / "plots", results["meta"]["tasks"],
                results["meta"]["methods_computed"])
            print(f"  [OK] {len(plot_files)} graficos en "
                  f"{out_dir / 'plots'}")
        except Exception as exc:
            print(f"  [WARN] Fallo generando graficos: {exc}")

    if results["meta"]["n_ok"] == 0:
        print("[ERROR] Ninguna grabacion de test se pudo procesar.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
