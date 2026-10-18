"""
src.cdhsa.classification - Clasificacion de tareas desde subespacios CD-HSA
===========================================================================

Modulo nucleo del experimento de clasificacion *hold-out*: dado un modelo
CD-HSA entrenado con N sujetos (subespacio comun ``W0`` + modos
especificos por condicion ``W_c``), clasificar grabaciones de sujetos
que NO participaron en el entrenamiento.

Marco matematico
----------------
Sea ``H*`` la matriz de Hankel de nivel 1 (``p x T1``, ``p = n_ch * d1``)
de una grabacion de test y

    ``B* = block_hankel(H*, L)`` (``D x T2``),  ``D = p * L``,
    ``T2 = T1 - L + 1``

su embedding de nivel 2. El modelo CD-HSA postula la descomposicion
(ortogonal, por construccion de los Pasos A6 y D):

    ``R^D = span(W0) (+) span(W_ec) (+) span(W_mem) (+) resto``

con ``W0^T W_c = 0``. Para una grabacion de condicion ``c`` la energia
de ``B*`` fuera de ``span(W0)`` se concentra en ``span(W_c)``.

**(1) Clasificador de energia residual (metodo primario, 'energy').**

    ``E_cj = || w_{c,j}^T B* ||_F^2``               (energia por modo)

    ``s_c(d) = sum_{j<=d} E_cj``                    (rango comun d)

    ``phi_c = s_c / (||B*||_F^2 - ||W0^T B*||_F^2)`` (fraccion residual)

    ``y_hat = argmax_c s_c(d)``

Con ``d`` comun a ambas condiciones. Equivalencias exactas:

* ``argmax_c s_c  <=>  argmin_c ||(I - P_c) B*||_F^2`` con
  ``P_c`` el proyector sobre ``span([W0 | W_c[:, :d]])`` (error de
  reconstruccion del modelo de clase), porque ``||B*||^2`` y la parte
  comun son constantes entre clases.
* ``argmax_c phi_c = argmax_c s_c`` (el denominador residual no depende
  de ``c``).

**(2) Clasificador ML bajo covarianza 'spiked' (metodo 'ml').**

Modelo: columnas de ``B*`` i.i.d. ``N(0, Sigma_c)`` con

    ``Sigma_c = W0 Lambda_0 W0^T + W_c Lambda_c W_c^T + sigma^2 I_D``.

El bloque comun y el ruido son identicos entre clases, asi que se
cancelan en el log-likelihood ratio y la regla de Bayes se reduce a

    ``g_c = sum_j [ nu_cj / (sigma^2 (sigma^2 + nu_cj)) ] E_cj
            - T2 * sum_j log(1 + nu_cj / sigma^2)``

    ``y_hat = argmax_c g_c``

con ``nu_cj`` la varianza por modo estimada del ENTRENAMIENTO
(media sobre grabaciones de ``||w_cj^T B_sc||_F^2 / T2``) y
``sigma^2`` estimada de la propia grabacion de test (empirical Bayes,
simetrica entre clases: energia sobrante tras quitar ``W0`` y TODOS los
``W_c``). Con espectro plano (``nu_cj`` iguales) la regla ML se reduce
al clasificador de energia.

Estadistica
-----------
* Binomial exacto unilateral (H0: azar p=1/C).
* IC de Clopper-Pearson al 95%.
* Test de permutacion PAREADO por sujeto (intercambio de etiquetas
  dentro de cada sujeto): el diseno aporta 2 grabaciones (una por
  tarea) por sujeto, y las etiquetas verdaderas son intercambiables
  dentro del sujeto bajo H0 de ausencia de informacion de condicion.
* AUC de Mann-Whitney sobre el margen de puntuacion.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.cdhsa.a_common_subspace import (
    compute_block_hankel_fro,
    make_block_hankel_linop,
)

__all__ = [
    "load_cdhsa_model",
    "parse_time_window_from_dir",
    "compute_recording_energies",
    "energy_scores_at_rank",
    "ml_discriminants",
    "classify_from_scores",
    "compute_training_mode_stats",
    "confusion_matrix",
    "exact_binomial_pvalue",
    "clopper_pearson_interval",
    "paired_label_swap_pvalue",
    "auc_mann_whitney",
]


# =====================================================================
# 1. Carga del modelo entrenado
# =====================================================================

def load_cdhsa_model(train_results_dir: str | Path) -> dict:
    """Cargar el modelo CD-HSA (W0, modos especificos) de una corrida.

    Parameters
    ----------
    train_results_dir : str | Path
        Directorio de salida de ``run_cdhsa_single_subject`` (o
        ``run_cdhsa``): debe contener ``cdhsa_arrays.npz``,
        ``hankel_info.json`` y ``config.json``.

    Returns
    -------
    model : dict
        Claves::

            W0             (D, r0)  base comun (A6), D = p * L
            r0             int
            W_specific     list[ (D, r_c) ] por condicion (orden de tasks)
            lambda_specific list[ (qmax,) | None ]
            r_specific     list[int]
            L              int  (profundidad nivel 2)
            hankel_depth   int  (profundidad nivel 1, d1)
            n_channels     int  (canales globales del entrenamiento)
            global_channels list[str]
            tasks          list[str] (orden de las condiciones)
            l_freq, h_freq float
            session        str
            train_subject_ids list[int]
            train_dir      Path
    """
    from src.cdhsa.extract_mode_indices import _extract_per_condition

    train_dir = Path(train_results_dir)
    npz_path = train_dir / "cdhsa_arrays.npz"
    info_path = train_dir / "hankel_info.json"
    cfg_path = train_dir / "config.json"

    for path, name in ((npz_path, "cdhsa_arrays.npz"),
                       (info_path, "hankel_info.json"),
                       (cfg_path, "config.json")):
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontro {name} en el directorio de "
                f"entrenamiento: {train_dir}"
            )

    with open(info_path, "r", encoding="utf-8") as fh:
        hankel_info = json.load(fh)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    npz = np.load(npz_path)

    if "A6__W0" not in npz:
        raise KeyError(
            "cdhsa_arrays.npz no contiene 'A6__W0' (paso A6 sin "
            "ejecutar o r0=0). Claves A6__ disponibles: "
            + ", ".join(k for k in npz.files if k.startswith("A6__"))
        )
    W0 = np.asarray(npz["A6__W0"], dtype=np.float64)
    r0 = int(npz["A6__r0"]) if "A6__r0" in npz.files else int(W0.shape[1])
    if r0 < 1 or W0.shape[1] < 1:
        raise ValueError(
            f"Modelo degenerado: r0={r0}, W0 con {W0.shape[1]} columnas."
        )

    if "D__r_specific" not in npz:
        raise KeyError(
            "cdhsa_arrays.npz no contiene 'D__r_specific' (Step D sin "
            "ejecutar: hace falta --sin skip_d y A6 con r0 >= 1)."
        )
    r_specific = [int(x) for x in np.asarray(npz["D__r_specific"])]
    C = len(r_specific)

    lam_all = (np.asarray(npz["D__lambda_specific"], dtype=np.float64)
               if "D__lambda_specific" in npz.files else None)

    if "D__W_specific" in npz.files:
        per_cond = _extract_per_condition(
            np.asarray(npz["D__W_specific"], dtype=np.float64),
            lam_all, np.asarray(r_specific), C,
        )
        W_specific = [np.asarray(w, dtype=np.float64)
                      for (w, _lam) in per_cond]
        lambda_specific = [lam for (_w, lam) in per_cond]
    elif "D__W_specific__0" in npz.files:
        # Ragged (v5.1): una clave por condicion D__W_specific__{c}
        W_specific = []
        lambda_specific = []
        for c in range(C):
            W_specific.append(
                np.asarray(npz[f"D__W_specific__{c}"], dtype=np.float64))
            rc = r_specific[c]
            if f"D__lambda_specific__{c}" in npz.files:
                lambda_specific.append(
                    np.asarray(npz[f"D__lambda_specific__{c}"],
                               dtype=np.float64)[:rc])
            elif lam_all is not None and lam_all.ndim == 2 \
                    and lam_all.shape[0] == C:
                lambda_specific.append(np.asarray(lam_all[c])[:rc])
            elif lam_all is not None and lam_all.ndim == 1:
                off = int(np.sum(r_specific[:c]))
                lambda_specific.append(
                    np.asarray(lam_all[off:off + rc], dtype=np.float64))
            else:
                lambda_specific.append(None)
    else:
        raise KeyError(
            "No se encontro D__W_specific (ni D__W_specific__0) en "
            "cdhsa_arrays.npz."
        )

    for c in range(C):
        if W_specific[c].shape[1] != r_specific[c]:
            raise ValueError(
                f"Inconsistencia: r_specific[{c}]={r_specific[c]} pero "
                f"W_specific[{c}] tiene {W_specific[c].shape[1]} columnas."
            )
        # W_c debe vivir en el complemento de W0 (Def. Step D)
        cross = float(np.max(np.abs(W0.T @ W_specific[c]))) \
            if W_specific[c].shape[1] > 0 else 0.0
        if cross > 1e-8:
            raise ValueError(
                f"W_specific[{c}] no es ortogonal a W0 "
                f"(max |W0^T W_c| = {cross:.2e}); el clasificador "
                f"requiere la descomposicion ortogonal del CD-HSA."
            )

    # --- Parametros de construccion (niveles 1 y 2) ---
    L = int(config["L"])
    global_channels = list(hankel_info.get("global_channels", []))
    if not global_channels:
        raise ValueError(
            "hankel_info.json no contiene 'global_channels'; no se "
            "puede reproducir el preprocesado del entrenamiento."
        )
    n_ch = len(global_channels)

    depth_requested = hankel_info.get("hankel_depth_requested")
    depth_common = hankel_info.get("depth_common")
    # Inferencia robusta: D = n_ch * d1 * L
    D = W0.shape[0]
    depth_inferred = D / (n_ch * L)
    candidates = [d for d in (depth_requested, depth_common, depth_inferred)
                  if d is not None]
    depth = int(round(float(candidates[0])))
    for d_alt in candidates[1:]:
        if int(round(float(d_alt))) != depth:
            raise ValueError(
                f"Profundidad de Hankel inconsistente en el "
                f"entrenamiento: {depth_requested} (pedida), "
                f"{depth_common} (comun), {depth_inferred} (inferida)."
            )
    if n_ch * depth * L != D:
        raise ValueError(
            f"Dimensiones inconsistentes: n_ch*depth*L = "
            f"{n_ch}*{depth}*{L} = {n_ch * depth * L} != D = {D} "
            f"(filas de W0)."
        )

    model = {
        "W0": W0,
        "r0": r0,
        "W_specific": W_specific,
        "lambda_specific": lambda_specific,
        "r_specific": r_specific,
        "L": L,
        "hankel_depth": depth,
        "n_channels": n_ch,
        "global_channels": global_channels,
        "tasks": list(hankel_info["tasks"]),
        "l_freq": float(hankel_info["l_freq"]),
        "h_freq": float(hankel_info["h_freq"]),
        "session": hankel_info.get("session"),
        "train_subject_ids": [int(x) for x in
                              hankel_info.get("subject_ids", [])],
        "train_dir": train_dir,
        "hankel_info": hankel_info,
        "config": config,
    }
    return model


def parse_time_window_from_dir(train_results_dir: str | Path
                               ) -> tuple[float | None, float | None]:
    """Extraer (t_start, t_end) del nombre del directorio de entrenamiento.

    ``_resolve_out_dir`` codifica la ventana como
    ``from{t0}s_to{t1}s``; se usa como default del clasificador para
    garantizar que el recorte temporal coincida con el entrenamiento.
    """
    m = re.search(r"from([0-9.]+)s_to([0-9.]+)s",
                  Path(train_results_dir).name)
    if m is None:
        return None, None
    return float(m.group(1)), float(m.group(2))


# =====================================================================
# 2. Energias por modo de una grabacion de test
# =====================================================================

def compute_recording_energies(
    H: NDArray[np.floating],
    L: int,
    W0: NDArray[np.floating],
    W_specific: list[NDArray[np.floating]],
    with_sigma: bool = True,
) -> dict:
    """Energias de nivel 2 de una grabacion contra el modelo entrenado.

    Computa (sin materializar ``B*``):

    * ``E0``  : ``(|| w0_j^T B* ||^2)`` por modo comun, ``r0``
    * ``E_c`` : ``(|| w_{c,j}^T B* ||^2)`` por modo especifico, por clase
    * ``total2``    : ``||B*||_F^2``
    * ``common2``   : ``||W0^T B*||_F^2``
    * ``residual2`` : ``total2 - common2``
    * (``with_sigma``) ``sigma2`` y el desglose anidado de energia
      [comun, +especifica c1, +especifica c2, ...] via QR de
      ``[W0 | W_1 | ... ]`` (particion exacta del espacio).

    Parameters
    ----------
    H : (p, T1) array
        Hankel de nivel 1 de la grabacion (misma construccion que el
        entrenamiento).
    L : int
        Profundidad de nivel 2 (la del entrenamiento).
    W0 : (D, r0) array
    W_specific : list of (D, r_c) arrays, una por condicion
    with_sigma : bool
        Incluir la estimacion de sigma^2 y el desglose por bloques
        (coste: un rmatmat adicional con ~m columnas).

    Returns
    -------
    dict con las claves descritas arriba.
    """
    H = np.asarray(H, dtype=np.float64)
    p, T1 = H.shape
    D = p * L
    W0 = np.asarray(W0, dtype=np.float64)
    if W0.shape[0] != D:
        raise ValueError(
            f"W0 tiene {W0.shape[0]} filas pero p*L = {p}*{L} = {D}. "
            f"La grabacion no es comparable con el modelo "
            f"(canales o depth distintos)."
        )

    B = make_block_hankel_linop(H, L)
    T2 = int(B.shape[1])

    # OJO: compute_block_hankel_fro devuelve la NORMA (no su cuadrado).
    total2 = float(compute_block_hankel_fro(H, L)) ** 2

    # --- Energias puras por modo (proyecciones con las W originales) ---
    G0 = np.asarray(B.rmatmat(W0))           # (T2, r0)
    E0 = np.sum(G0 * G0, axis=0)             # (r0,)
    common2 = float(np.sum(E0))
    del G0

    E_specific: list[NDArray[np.floating]] = []
    for W_c in W_specific:
        W_c = np.asarray(W_c, dtype=np.float64)
        if W_c.shape[0] != D:
            raise ValueError(
                f"W_c tiene {W_c.shape[0]} filas, se esperaban {D}."
            )
        if W_c.shape[1] == 0:
            E_specific.append(np.zeros(0))
            continue
        Gc = np.asarray(B.rmatmat(W_c))      # (T2, r_c)
        E_specific.append(np.sum(Gc * Gc, axis=0))
        del Gc

    out = {
        "E_common": E0,
        "E_specific": E_specific,
        "total2": total2,
        "common2": common2,
        "residual2": max(total2 - common2, 0.0),
        "T1": T1,
        "T2": T2,
        "p": p,
        "D": D,
    }

    if with_sigma:
        from scipy.linalg import qr as _qr

        W_all = np.concatenate(
            [W0] + [np.asarray(Wc, dtype=np.float64)
                     for Wc in W_specific if np.asarray(Wc).shape[1] > 0],
            axis=1,
        )
        Q, R = _qr(W_all, mode="economic")
        # Rango numerico del bloque concatenado (por si W_ec y W_mem
        # se solaparan: el QR anida la atribucion de energia).
        diagR = np.abs(np.diag(R))
        rank = int(np.sum(diagR > 1e-10 * (diagR[0] if diagR.size else 1.0)))
        Gq = np.asarray(B.rmatmat(Q[:, :rank]))   # (T2, rank)
        eq = np.sum(Gq * Gq, axis=0)              # (rank,)
        del Gq

        # Desglose anidado: [comun, +c1, +c2, ...] + inexplicada
        widths = [W0.shape[1]] + [
            int(np.asarray(Wc).shape[1]) for Wc in W_specific
        ]
        bounds = np.cumsum([0] + widths[:])
        budget: list[float] = []
        for i in range(len(widths)):
            lo, hi = int(bounds[i]), int(min(bounds[i + 1], rank))
            budget.append(float(np.sum(eq[lo:hi])) if hi > lo else 0.0)
        budget.append(max(total2 - float(np.sum(eq[:rank])), 0.0))

        dof = max(D - rank, 1)
        sigma2 = budget[-1] / (T2 * dof)

        out.update({
            "sigma2": float(sigma2),
            "budget": budget,           # suma == total2 (salvo redondeo)
            "model_rank": rank,
        })

    return out


# =====================================================================
# 3. Reglas de decision
# =====================================================================

def energy_scores_at_rank(E_specific: list[NDArray[np.floating]],
                          d: int) -> NDArray[np.floating]:
    """``s_c = sum_{j<=d} E_cj`` para cada condicion (rango comun d)."""
    d = int(d)
    s = np.zeros(len(E_specific))
    for c, E_c in enumerate(E_specific):
        d_eff = min(d, len(E_c))
        s[c] = float(np.sum(E_c[:d_eff])) if d_eff > 0 else 0.0
    return s


def ml_discriminants(E_specific: list[NDArray[np.floating]],
                     nu_specific: list[NDArray[np.floating]],
                     sigma2: float,
                     T2: int) -> NDArray[np.floating]:
    """Regla de Bayes bajo el modelo de covarianza 'spiked'.

    ``g_c = sum_j [ nu_cj / (sigma^2 (sigma^2 + nu_cj)) ] E_cj
           - T2 sum_j log(1 + nu_cj / sigma^2)``

    Los bloques comun y de ruido se cancelan entre clases (identicos en
    el modelo), por lo que solo entran los terminos especificos.
    """
    sigma2 = max(float(sigma2), 1e-300)
    g = np.zeros(len(E_specific))
    for c, (E_c, nu_c) in enumerate(zip(E_specific, nu_specific)):
        if len(E_c) == 0 or len(nu_c) == 0:
            continue
        d_eff = min(len(E_c), len(nu_c))
        E_c = np.asarray(E_c[:d_eff], dtype=np.float64)
        nu_c = np.asarray(nu_c[:d_eff], dtype=np.float64)
        nu_c = np.maximum(nu_c, 0.0)
        w = nu_c / (sigma2 * (sigma2 + nu_c))
        g[c] = float(np.dot(w, E_c) - T2 * np.sum(np.log1p(nu_c / sigma2)))
    return g


def classify_from_scores(scores: NDArray[np.floating]) -> tuple[int, float]:
    """argmax + margen normalizado ``(s1 - s2) / max(|s1| + |s2|, eps)``."""
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores)
    pred = int(order[0])
    if len(scores) < 2:
        return pred, 1.0
    s1, s2 = float(scores[order[0]]), float(scores[order[1]])
    denom = abs(s1) + abs(s2)
    margin = (s1 - s2) / denom if denom > 0 else 0.0
    return pred, float(margin)


# =====================================================================
# 4. Estadisticas de entrenamiento para el metodo ML
# =====================================================================

def compute_training_mode_stats(
    hankel_npz_path: str | Path,
    W0: NDArray[np.floating],
    W_specific: list[NDArray[np.floating]],
    L: int,
) -> dict:
    """Varianza por modo ``nu_cj`` estimada de las grabaciones TRAIN.

    Para cada grabacion de entrenamiento (s, c) computa
    ``E_cj(s) / T2_sc`` (segundo momento por columna de la proyeccion)
    y promedia sobre las grabaciones de la condicion ``c``. ``nu_cj``
    son las 'intensidades' del bloque especifico del modelo de
    covarianza en unidades de energia SIN normalizar.

    Carga ``hankel_matrices.npz`` clave a clave (``H_ss{s+1}_c{c+1}``)
    para mantener el pico de memoria en una sola matriz.

    Returns
    -------
    dict con ``nu_specific`` (list de arrays), ``nu_common_mean``,
    ``n_used`` (conteos por condicion) y ``T2`` por grabacion.
    """
    W0 = np.asarray(W0, dtype=np.float64)
    path = Path(hankel_npz_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}: el metodo 'ml' necesita las "
            f"matrices de Hankel del entrenamiento para estimar las "
            f"varianzas por modo."
        )

    npz = np.load(path)
    h_keys = sorted(
        (k for k in npz.files if k.startswith("H_ss")),
        key=lambda k: tuple(int(x) for x in k.replace("H_ss", "")
                            .split("_c")),
    )
    if not h_keys:
        raise ValueError(f"{path} no contiene matrices H_ss*.")

    C = len(W_specific)
    acc: list[list[NDArray]] = [[] for _ in range(C)]
    acc0: list[NDArray] = []
    t2_list: list[int] = []
    s_idx_max = 0

    for key in h_keys:
        H = np.asarray(npz[key], dtype=np.float64)
        if H.size == 0:
            continue
        # H_ss{s}_c{c} -> indices 0-based
        parts = key.replace("H_ss", "").split("_c")
        s_idx, c_idx = int(parts[0]) - 1, int(parts[1]) - 1
        s_idx_max = max(s_idx_max, s_idx)
        if not (0 <= c_idx < C):
            continue
        r = compute_recording_energies(H, L, W0, [W_specific[c_idx]],
                                        with_sigma=False)
        acc[c_idx].append(r["E_specific"][0] / max(r["T2"], 1))
        acc0.append(r["E_common"] / max(r["T2"], 1))
        t2_list.append(r["T2"])
        del H

    nu_specific = []
    n_used = []
    for c in range(C):
        if acc[c]:
            nu_specific.append(
                np.mean(np.vstack(acc[c]), axis=0).astype(np.float64))
            n_used.append(len(acc[c]))
        else:
            nu_specific.append(np.zeros(len(W_specific[c])))
            n_used.append(0)

    return {
        "nu_specific": nu_specific,
        "nu_common_mean": (np.mean(np.vstack(acc0), axis=0)
                           if acc0 else np.zeros(W0.shape[1])),
        "n_used_per_condition": n_used,
        "n_train_recordings": len(t2_list),
        "T2_mean": float(np.mean(t2_list)) if t2_list else 0.0,
    }


# =====================================================================
# 5. Metricas
# =====================================================================

def confusion_matrix(y_true: NDArray[np.integer],
                     y_pred: NDArray[np.integer],
                     n_classes: int) -> NDArray[np.integer]:
    """Matriz de confusion ``M[verdadero, predicho]`` (conteos)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    M = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            M[t, p] += 1
    return M


def exact_binomial_pvalue(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) exacto, X ~ Binomial(n, p) (unilateral, H0 = azar)."""
    from scipy.stats import binom
    if n <= 0:
        return 1.0
    k = int(max(k, 0))
    return float(binom.sf(k - 1, n, p))


def clopper_pearson_interval(k: int, n: int,
                              alpha: float = 0.05) -> tuple[float, float]:
    """IC exacto de Clopper-Pearson para una proporcion (k/n)."""
    from scipy.stats import beta
    if n <= 0:
        return (0.0, 1.0)
    k = int(min(max(k, 0), n))
    lo = 0.0 if k == 0 else float(
        beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(
        beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


def paired_label_swap_pvalue(
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
    subject_ids: NDArray[np.integer],
    n_perm: int = 10000,
    seed: int = 20261006,
) -> float:
    """Test de permutacion PAREADO (intercambio dentro de sujeto).

    El diseno aporta hasta dos grabaciones por sujeto (una por tarea).
    Bajo H0 (las puntuaciones no llevan informacion de condicion), las
    etiquetas verdaderas son intercambiables DENTRO de cada sujeto. La
    distribucion nula de la accuracy se obtiene permutando las
    etiquetas dentro de cada sujeto manteniendo las predicciones fijas.

    p = (1 + #{acc_perm >= acc_obs}) / (1 + n_perm)   [add-one]
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    subject_ids = np.asarray(subject_ids, dtype=int)
    if len(y_true) != len(y_pred) or len(y_true) != len(subject_ids):
        raise ValueError("y_true, y_pred y subject_ids deben alinearse.")
    if len(y_true) == 0:
        return 1.0

    groups: dict[int, NDArray[np.integer]] = {}
    for i, subj in enumerate(subject_ids):
        groups.setdefault(int(subj), []).append(i)
    groups = {s: np.asarray(idx) for s, idx in groups.items()}

    acc_obs = float(np.mean(y_true == y_pred))
    rng = np.random.default_rng(seed)

    count_ge = 0
    n_perm = int(n_perm)
    for _ in range(n_perm):
        y_perm = y_true.copy()
        for _s, idx in groups.items():
            if len(idx) > 1:
                y_perm[idx] = y_true[rng.permutation(idx)]
        acc_perm = float(np.mean(y_perm == y_pred))
        if acc_perm >= acc_obs - 1e-12:
            count_ge += 1

    return float((1 + count_ge) / (1 + n_perm))


def auc_mann_whitney(scores: NDArray[np.floating],
                     is_positive: NDArray[np.bool_]) -> float:
    """AUC via U de Mann-Whitney (empates a 0.5 via rangos promedio)."""
    from scipy.stats import rankdata
    scores = np.asarray(scores, dtype=np.float64)
    is_positive = np.asarray(is_positive, dtype=bool)
    n1 = int(np.sum(is_positive))
    n0 = int(len(scores) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(scores)
    u1 = float(np.sum(ranks[is_positive])) - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n0))
