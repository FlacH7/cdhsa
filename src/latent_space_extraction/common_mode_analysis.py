"""
common_mode_analysis.py — Análisis de los modos comunes (Step A6 / W0) de CD-HSA
================================================================================

Bloque de análisis que el pipeline latente
(``latent_space_plots_from_eeg_gedai_super_subject.py``) ejecuta para cada
grabación super-subject × tarea, ANTES del análisis de los modos
específicos de condición (Step D).  Para la grabación en cuestión:

1. **Carga del subespacio común** ``W0`` (``r0`` direcciones Hankel
   compartidas validadas por Step A6 con el nulo Haar) desde el
   ``cdhsa_arrays.npz`` de la corrida CD-HSA que entrenó los modos
   (claves ``A6__W0`` / ``A6__lambda0`` / ``A6__r0``), con fallbacks al
   ``mode_map.json`` (``common_subspace``) y al ``stage2_meta`` legacy.

2. **Proyección del subespacio común**: series temporales
   ``Y0 = W0^T H_block`` donde ``H_block`` es la block-Hankel de nivel 2
   de la grabación (misma convención de bloques-lag que usa
   ``CdhsaSpecificModesDynamics`` — se reutiliza ``_build_block_hankel``).
   ``H_block`` NO se materializa completa (para 53 canales × 100 s × 20
   sujetos sería ~25 GB): se proyecta por bloques de columnas.

3. **Métricas de energía / varianza explicada** por modo común:
   - energía capturada ``e_j = ||w_j^T H_block||^2`` y su fracción
     ``e_j / ||H_block||_F^2`` (misma definición que Step B del CD-HSA);
   - varianza explicada sobre la Hankel centrada por columna
     (estilo PCA): con ``mu`` = media de columnas de ``H_block``,
     ``Y0c = Y0 - (W0^T mu) 1^T`` y
     ``v_j = ||Y0c_j||^2 / (||H_block||_F^2 - n_cols ||mu||^2)``.
     Como ``W0`` es ortonormal, ``sum_j v_j`` es exactamente la fracción
     de varianza capturada por el subespacio común completo.
   - las mismas métricas para los modos específicos seleccionados
     (``W_sel`` del stage2) para el contraste común vs específico.

4. **Influencia de los canales originales sobre TODOS los modos
   comunes**: correlación canal–serie común (el mismo fallback que usa
   ``compute_channel_influence_on_latent`` para cdhsa_specific_modes,
   con la misma normalización max-abs por dimensión), presentada como
   heatmap canales × modos + topoplots de los K modos dominantes (si el
   raw tiene montaje) + métrica espectral por bandas.

Artefactos generados en ``out_dir``
-----------------------------------
- ``common_mode_projection.png``     — series temporales de los K modos
                                        comunes dominantes (por energía).
- ``common_variance_explained.png``  — fracción por modo + acumulada,
                                        comunes vs específicos.
- ``common_channel_influence.png``   — heatmap (canales × modos comunes).
- ``common_channel_influence_topo.png`` — topoplots del top-K (si montaje).
- ``common_channel_influence_data.npz`` — pesos + métrica espectral.
- ``common_projection_data.npz``     — proyección completa ``Y0`` (opt-in).
- ``common_mode_analysis.json``      — todas las métricas numéricas.

Notas de diseño
--------------
- La proyección NO enmascara las columnas de frontera de concatenación
  (level-2 boundary mask) del super-subject: consistente con
  ``CdhsaSpecificModesDynamics``, que proyecta el ``H_block`` completo.
- El orden de los modos es el del modelo (λ0 descendente, índices
  ``CM0..CM{r0-1}``); los PLOTS reordenan por energía capturada en esta
  grabación manteniendo la etiqueta del índice del modelo.
- Ninguna métrica requiere materializar ``H_block`` completo.

Dependencies: numpy, matplotlib, scipy (indirecta), src.spectral_analysis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.latent_space_extraction.pipeline.stage2_dynamics import (
    _build_block_hankel,
)
from src.latent_space_extraction.hankel_dmd_extractor import (
    _build_multivariate_hankel,
)
from src.spectral_analysis.influence_metrics import (
    compute_channel_latent_correlation,
    compute_spectral_influence,
)
from src.spectral_analysis.psd_analysis import (
    _compute_latent_psds,
    load_psd_data,
)
from src.spectral_analysis.plotting_utils import plot_topomap_grid

# Colour palette — consistent with src.plotters.cdhsa_plots
COLOR_COMMON = "#2980b9"       # blue  — common (A6)
COLOR_COMMON_DARK = "#1f618d"
COLOR_SPECIFIC = "#e74c3c"     # red   — specific (D)
COLOR_SPECIFIC_DARK = "#c0392b"


# ============================================================================
# 1. Carga del subespacio común W0
# ============================================================================

def load_common_modes(meta: dict) -> dict | None:
    """Cargar ``W0`` / ``lambda0`` / ``r0`` desde las fuentes disponibles.

    Orden de precedencia (el primero que funciona gana):

    1. **NPZ** de la corrida CD-HSA (``stage2_meta['npz_path']``):
       claves ``A6__W0`` ``(d, r0)``, ``A6__lambda0`` ``(r0,)``,
       ``A6__r0``.  Es la fuente canónica (arrays exactos en float64).
    2. **mode_map.json** (``stage2_meta['mode_map_path']``):
       ``common_subspace.W0`` (lista anidada, serializada por
       ``extract_mode_indices._json_safe``).
    3. **stage2_meta legacy**: ``W_common`` / ``common_eigenvalues`` /
       ``r0_common`` (p. ej. caches escritos por versiones futuras).

    Returns
    -------
    dict con ``W0`` (d, r0) float64, ``lambda0`` (r0,) float64 | None,
    ``r0`` int y ``source`` str; o ``None`` si ninguna fuente funciona.
    ``W0`` se devuelve con columnas re-ortonormalizadas solo si su
    ortogonalidad numérica se ha degradado (tol 1e-6).
    """
    stage2 = meta.get("stage2") or {}
    if stage2.get("dynamics") != "cdhsa_specific_modes":
        return None

    W0 = None
    lambda0 = None
    r0 = None
    source = None

    # --- (1) NPZ -----------------------------------------------------------
    npz_path = stage2.get("npz_path")
    if npz_path and Path(npz_path).exists():
        try:
            with np.load(npz_path, allow_pickle=True) as npz:
                if "A6__W0" in npz.files:
                    W0 = np.asarray(npz["A6__W0"], dtype=np.float64)
                    r0 = int(W0.shape[1])
                    if "A6__lambda0" in npz.files:
                        lambda0 = np.ravel(
                            np.asarray(npz["A6__lambda0"], dtype=np.float64)
                        )
                    elif "A6__lambda" in npz.files:
                        lambda0 = np.ravel(
                            np.asarray(npz["A6__lambda"], dtype=np.float64)
                        )[:r0]
                    source = f"npz:{Path(npz_path).name}"
        except Exception as exc:
            # (OSError, ValueError, KeyError) no cubren npz corruptos: numpy
            # lanza pickle.UnpicklingError (bytes basura) o zipfile.BadZipFile
            # (zip truncado), que NO son subclases de ValueError.  El fallback
            # a mode_map debe ser total (best-effort).
            print(f"  [CommonModes] NPZ ilegible ({exc}); probando mode_map.")
            W0 = None

    # --- (2) mode_map.json --------------------------------------------------
    if W0 is None:
        mm_path = stage2.get("mode_map_path")
        if mm_path and Path(mm_path).exists():
            try:
                with open(mm_path, encoding="utf-8") as fh:
                    mode_map = json.load(fh)
                cs = mode_map.get("common_subspace") or {}
                w0_raw = cs.get("W0")
                if isinstance(w0_raw, list) and w0_raw:
                    W0 = np.asarray(w0_raw, dtype=np.float64)
                    r0 = int(W0.shape[1])
                    lam_raw = cs.get("lambda0")
                    if isinstance(lam_raw, list):
                        lambda0 = np.asarray(lam_raw, dtype=np.float64)
                    source = f"mode_map:{Path(mm_path).name}"
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"  [CommonModes] mode_map ilegible ({exc}).")

    # --- (3) legacy stage2_meta ----------------------------------------------
    if W0 is None:
        W0 = stage2.get("W_common")
        if W0 is not None and getattr(W0, "ndim", 0) == 2 and W0.shape[1] > 0:
            W0 = np.asarray(W0, dtype=np.float64)
            r0 = int(W0.shape[1])
            lambda0 = stage2.get("common_eigenvalues")
            if lambda0 is not None:
                lambda0 = np.asarray(lambda0, dtype=np.float64)
            source = "stage2_meta:W_common"
        else:
            W0 = None

    if W0 is None or r0 is None or r0 == 0:
        return None

    # --- Sanity: re-ortogonalizar SOLO si hace falta ------------------------
    gram = W0.T @ W0
    if not np.allclose(gram, np.eye(r0), atol=1e-6):
        # e.g. round-trip JSON (mode_map) puede perder ~1e-8 por número;
        # una desviación mayor indica corrupción — QR restaura ortonormalidad
        # sin cambiar el span (los signos de Q pueden diferir de W0, lo que
        # solo invierte series/heatmaps individualmente, no las métricas).
        print("  [CommonModes] W0 no ortonormal (tol 1e-6); re-ortogonalizando por QR.")
        W0, _ = np.linalg.qr(W0)

    return {
        "W0": W0,
        "lambda0": lambda0 if lambda0 is not None else np.full(r0, np.nan),
        "r0": r0,
        "source": source,
    }


# ============================================================================
# 2. Proyección chunked sobre el subespacio común
# ============================================================================

def _reconstruct_level1_hankel(meta: dict) -> tuple[np.ndarray, int] | None:
    """Reconstruir la Hankel de nivel 1 ``H`` (p × T) del stage 1.

    Usa ``meta['stage1']['input_data']`` (la señal filtrada, canales ×
    tiempo) y ``meta['stage1']['depth']`` con la MISMA función que el
    pipeline interno (``_build_multivariate_hankel``), garantizando la
    misma convención de bloques con la que se proyectaron los modos
    específicos.

    Returns
    -------
    (H, depth) o ``None`` si el meta no contiene lo necesario.
    """
    s1 = meta.get("stage1") or {}
    X_input = s1.get("input_data")
    depth = s1.get("depth")
    if X_input is None or depth is None:
        return None
    X_input = np.asarray(X_input, dtype=np.float64)
    if X_input.ndim != 2 or X_input.shape[1] <= int(depth):
        return None
    H = _build_multivariate_hankel(X_input, int(depth))
    if H is None or H.ndim != 2:
        return None
    return np.asarray(H, dtype=np.float64), int(depth)


def project_common_modes(
    H: np.ndarray,
    bases: dict[str, np.ndarray],
    L: int,
    chunk_cols: int = 50_000,
) -> dict:
    """Proyectar ``H_block`` (por bloques de columnas) sobre cada base.

    Parameters
    ----------
    H : ndarray (p, T)
        Hankel de nivel 1 (la que produce Stage 1).
    bases : dict label -> ndarray (p*L, k) con columnas ortonormales
        Bases a proyectar.  Debe incluir la clave ``"common"`` (W0);
        cualquier otra (p. ej. ``"specific"`` = W_sel) se proyecta en el
        mismo bucle (coste marginal mínimo) para una comparación con
        EXACTAMENTE la misma numerología.
    L : int
        Retardo de nivel 2 (block-Hankel).
    chunk_cols : int
        Columnas de ``H_block`` por bloque.  Con p*L = 5300 y
        chunk = 50 000 el pico temporal es ~2.1 GB (float64).

    Returns
    -------
    dict con:
      ``projections``: label -> (k, T-L+1)  (sin centrar)
      ``mu``         : (p*L,)  media de columnas de H_block
      ``fro_total``  : ||H_block||_F^2
      ``n_cols``      : T-L+1
    """
    p, T = H.shape
    n_cols = T - L + 1
    if n_cols < 2:
        raise ValueError(
            f"[CommonModes] H_block tendría {n_cols} columnas (T={T}, L={L}); "
            "se necesitan al menos 2."
        )
    chunk_cols = max(int(chunk_cols), 1)

    labels = list(bases.keys())
    out = {lab: np.empty((bases[lab].shape[1], n_cols), dtype=np.float64)
           for lab in labels}
    mu = np.zeros(p * L, dtype=np.float64)
    fro_total = 0.0

    for a in range(0, n_cols, chunk_cols):
        b = min(a + chunk_cols, n_cols)
        # Columnas [a, b) de H_block solo necesitan H[:, a : b+L-1]
        # (misma convención que _build_block_hankel: bloque lag ℓ de la
        # columna t = H[:, t+ℓ]).
        Hc = H[:, a: b + L - 1]
        Cb = _build_block_hankel(Hc, L)          # (p*L, b-a)
        mu += Cb.sum(axis=1)
        fro_total += float(np.vdot(Cb, Cb))
        for lab in labels:
            out[lab][:, a:b] = bases[lab].T @ Cb
        del Cb, Hc

    mu /= n_cols
    return {
        "projections": out,
        "mu": mu,
        "fro_total": fro_total,
        "n_cols": n_cols,
    }


# ============================================================================
# 3. Métricas de energía / varianza explicada
# ============================================================================

def energy_variance_metrics(
    Y: np.ndarray,
    W: np.ndarray,
    mu: np.ndarray,
    fro_total: float,
    n_cols: int,
) -> dict:
    """Métricas por modo para una base proyectada.

    Con ``Y = W^T H_block`` (sin centrar), ``mu`` la media de columnas
    de ``H_block`` y ``W`` ortonormal:

    - ``energy``       : ``e_j = ||Y_j||^2`` (energía capturada, Step B)
    - ``energy_frac``  : ``e_j / ||H_block||_F^2``
    - ``var_frac``     : fracción de la varianza TOTAL centrada de
                         ``H_block`` explicada por el modo ``j``:
                         ``||Y_j - (w_j^T mu)||^2 / (fro - n_cols·||mu||^2)``
    - ``cum_var_frac`` : acumulada de ``var_frac`` (exacta para el
                         subespacio por ortonormalidad de W).
    """
    k = Y.shape[0]
    Yc = Y - (W.T @ mu)[:, None]        # proyección del H_block centrado
    total_var = fro_total - n_cols * float(mu @ mu)
    if total_var <= 0.0:
        total_var = max(fro_total, np.finfo(np.float64).tiny)

    energy = np.einsum("ij,ij->i", Y, Y)
    energy_c = np.einsum("ij,ij->i", Yc, Yc)

    return {
        "energy": energy,
        "energy_frac": energy / fro_total if fro_total > 0 else np.zeros(k),
        "var_frac": energy_c / total_var,
        "cum_var_frac": np.cumsum(energy_c) / total_var,
        "total_var_centered": total_var,
    }


# ============================================================================
# 4. Plots
# ============================================================================

def plot_common_mode_timeseries(
    Y: np.ndarray,
    order_by_energy: np.ndarray,
    energy_frac: np.ndarray,
    lambda0: np.ndarray,
    out_dir: Path,
    top_k: int = 12,
    sfreq: float | None = None,
) -> Path | None:
    """Series temporales de los K modos comunes dominantes (z-scored)."""
    r0 = Y.shape[0]
    if r0 == 0:
        return None
    k = min(int(top_k), r0)

    fig, axes = plt.subplots(
        k, 1, figsize=(14, 1.9 * k), squeeze=False, constrained_layout=True,
    )
    t = None
    if sfreq:
        t = np.arange(Y.shape[1]) / float(sfreq)

    for row, mode_idx in enumerate(order_by_energy[:k]):
        ax = axes[row, 0]
        y = Y[mode_idx] - Y[mode_idx].mean()
        std = y.std()
        y = y / std if std > 0 else y
        x = t if t is not None else np.arange(len(y))
        ax.plot(x, y, lw=0.5, color=COLOR_COMMON)
        lam_txt = (f", \u03bb\u2080={lambda0[mode_idx]:.3f}"
                   if np.isfinite(lambda0[mode_idx]) else "")
        ax.set_title(
            f"CM{mode_idx}{lam_txt} \u2014 captured energy "
            f"{100 * energy_frac[mode_idx]:.2f}%",
            fontsize=9, loc="left",
        )
        ax.set_ylabel("z-score", fontsize=8)
        ax.set_xticks([])
        if row == k - 1:
            ax.set_xlabel("time (s)" if t is not None else "sample",
                          fontsize=9)

    fig.suptitle(
        f"Common-mode time series \u2014 top {k} of {r0} by captured energy"
        f" (projection of the common subspace W\u2080)",
        fontsize=12, fontweight="bold",
    )
    path = out_dir / "common_mode_projection.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_common_variance_explained(
    common_m: dict,
    specific_m: dict | None,
    lambda0: np.ndarray,
    out_dir: Path,
) -> Path:
    """Fracción de varianza por modo + acumulada, comunes vs específicos."""
    r0 = len(common_m["var_frac"])
    x_c = np.arange(r0)

    n_spec = len(specific_m["var_frac"]) if specific_m else 0
    total = r0 + (n_spec + 1 if n_spec else 0)

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5.5), constrained_layout=True,
    )

    # --- Panel 1: fracción por modo ---
    ax = axes[0]
    lam_txt = [f" (λ0={lambda0[j]:.3f})" if np.isfinite(lambda0[j]) else ""
               for j in range(r0)]
    ax.bar(x_c, 100 * common_m["var_frac"], color=COLOR_COMMON,
           edgecolor="black", linewidth=0.4,
           label=f"Common modes (A6, r0={r0})")
    if n_spec:
        x_s = np.arange(r0 + 1, r0 + 1 + n_spec)
        ax.bar(x_s, 100 * specific_m["var_frac"], color=COLOR_SPECIFIC,
               edgecolor=COLOR_SPECIFIC_DARK, linewidth=0.8,
               label=f"Specific modes (D, top-{n_spec})")
    ax.set_xlabel("Mode index (model order)", fontsize=10)
    ax.set_ylabel("Variance explained (% of centered block-Hankel)",
                  fontsize=10)
    ax.set_title("Per-mode variance explained", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    if r0 <= 30:
        ax.set_xticks(x_c)
        ax.set_xticklabels([f"CM{j}" for j in range(r0)], fontsize=7,
                           rotation=90)
    if n_spec:
        ax.set_xticks(list(x_c) + list(x_s))
        ax.set_xticklabels(
            [f"CM{j}" for j in range(r0)]
            + [f"SM{j}" for j in range(n_spec)], fontsize=7, rotation=90,
        )
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)

    # --- Panel 2: acumulada ---
    ax = axes[1]
    ax.plot(x_c, 100 * common_m["cum_var_frac"], "-o", ms=3,
            color=COLOR_COMMON, label="Common (cumulative)")
    if n_spec:
        ax.axhline(
            100 * float(np.sum(specific_m["var_frac"])),
            color=COLOR_SPECIFIC_DARK, linestyle="--", linewidth=1.2,
            label=f"Specific total (top-{n_spec})",
        )
    tot_c = 100 * float(common_m["cum_var_frac"][-1])
    ax.axhline(tot_c, color=COLOR_COMMON_DARK, linestyle=":", linewidth=1.0,
               label=f"Common total = {tot_c:.1f}%")
    ax.set_xlabel("Number of common modes (model order)", fontsize=10)
    ax.set_ylabel("Cumulative variance explained (%)", fontsize=10)
    ax.set_title("Cumulative variance explained by the common subspace",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.legend(fontsize=8, loc="lower right")
    if r0 > 1:
        ax.set_xticks(np.arange(0, r0, max(1, r0 // 12)))

    fig.suptitle(
        "Common (A6) vs specific (D) modes \u2014 variance explained on "
        "this recording",
        fontsize=12, fontweight="bold",
    )
    path = out_dir / "common_variance_explained.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_common_channel_influence_heatmap(
    weights: np.ndarray,
    ch_names: list[str],
    mode_labels: list[str],
    out_dir: Path,
    order_by_energy: np.ndarray,
) -> Path:
    """Heatmap canales × modos comunes (todos los r0)."""
    W = weights[:, order_by_energy]     # reordenar columnas por energía
    labels = [mode_labels[j] for j in order_by_energy]
    n_ch, r0 = W.shape

    fig_h = max(4.0, 0.22 * n_ch + 1.5)
    fig_w = max(6.0, 0.55 * r0 + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(W, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_xticks(np.arange(r0))
    step = max(1, r0 // 40)
    ax.set_xticks(np.arange(0, r0, step))
    ax.set_xticklabels(
        [labels[j] for j in range(0, r0, step)], fontsize=7, rotation=90,
    )
    ax.set_yticks(np.arange(n_ch))
    ax.set_yticklabels(ch_names, fontsize=7)
    ax.set_xlabel("Common mode (ordered by captured energy on this recording)",
                  fontsize=10)
    ax.set_ylabel("Original EEG channel", fontsize=10)
    ax.set_title(
        "Channel influence on ALL common modes (A6)\n"
        "(channel\u2013latent Pearson correlation, max-abs normalised per mode)",
        fontsize=11, fontweight="bold",
    )
    fig.colorbar(im, ax=ax, shrink=0.85, label="Influence (norm., signed)")
    path = out_dir / "common_channel_influence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_common_topoplots(
    raw,
    weights: np.ndarray,
    ch_names: list[str],
    mode_labels: list[str],
    order_by_energy: np.ndarray,
    out_dir: Path,
    top_k: int = 8,
) -> Path | None:
    """Topoplots de los K modos comunes dominantes (si hay montaje)."""
    import mne

    montage = raw.get_montage()
    if montage is None:
        return None
    available = [c for c in ch_names if c in raw.ch_names]
    if len(available) < 3 or len(available) != len(ch_names):
        return None

    k = min(int(top_k), weights.shape[1])
    idx = order_by_energy[:k]
    W_sel = weights[:, idx]

    picks = mne.pick_channels(raw.ch_names, include=available)
    info_sel = mne.pick_info(raw.info, picks, copy=True)
    dim_labels = [mode_labels[j] for j in idx]

    fig = plot_topomap_grid(
        info_sel, W_sel, dim_labels,
        title="Channel influence \u2014 top common modes (A6)",
    )
    path = out_dir / "common_channel_influence_topo.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ============================================================================
# 5. Influencia de canales sobre los modos comunes
# ============================================================================

def compute_common_channel_influence(
    raw,
    Y_series: np.ndarray,
    meta: dict,
    out_dir: Path,
    sfreq: float,
    fmin: float,
    fmax: float,
    raw_psd_path: Path | None = None,
) -> dict:
    """Influencia de los canales originales sobre cada modo común.

    Réplica del flujo de ``compute_channel_influence_on_latent`` para las
    series comunes, con nombres de fichero propios (el ``.npz`` de datos
    del específico no debe compartirse ni colisionar):

    1. correlación canal–serie común (fallback estándar para métodos
       cuyos modos viven en el espacio block-Hankel), usando
       ``X_filtered`` del meta si es consistente con los canales del raw
       (idéntica lógica que la función original);
    2. normalización max-abs por columna (signo preservado);
    3. métrica espectral por bandas (opcional, con ``raw_psd_path``).

    Returns
    -------
    dict con ``weights`` (n_ch, r0), ``ch_names``, ``method`` y
    ``spectral`` (o None).
    """
    latent = np.asarray(Y_series, dtype=float).T   # (n_samples, r0)
    n_samples, r0 = latent.shape

    data = raw.get_data()
    method = "correlation_fallback(cdhsa_common_modes)"
    if isinstance(meta, dict):
        pre = meta.get("preprocessing")
        X = pre.get("X_filtered") if isinstance(pre, dict) else None
        if (isinstance(X, np.ndarray) and X.ndim == 2
                and X.shape[0] <= data.shape[0]
                and X.shape[1] >= n_samples):
            data = X
            method = f"{method}|X_filtered"

    weights = compute_channel_latent_correlation(data, latent)

    ch_names = list(raw.ch_names)
    if weights.shape[0] != len(ch_names):
        # mantener consistencia con _resolve_channel_names del original
        from src.spectral_analysis.psd_analysis import _resolve_channel_names
        ch_names = _resolve_channel_names(raw, meta, weights.shape[0])

    weights = np.asarray(weights, dtype=float)
    col_max = np.abs(weights).max(axis=0, keepdims=True)
    col_max[col_max == 0.0] = 1.0
    weights = weights / col_max
    method += "|maxabs_norm"

    # --- métrica espectral (opcional, best-effort) ---
    spectral = None
    if raw_psd_path is not None and Path(raw_psd_path).exists():
        try:
            raw_psd = load_psd_data(raw_psd_path)
            meta_raw = raw_psd.get("metadata", {})
            latent_psds, latent_freqs = _compute_latent_psds(
                latent,
                sfreq,
                fmin=float(meta_raw.get("fmin", fmin)),
                fmax=float(meta_raw.get("fmax", fmax)),
                bandwidth=float(meta_raw.get("bandwidth", 2.5)),
            )
            spectral = compute_spectral_influence(
                raw_psd["psds"], latent_psds,
                raw_psd["freqs"], latent_freqs, weights,
            )
        except Exception as exc:  # opcional: no abortar
            print(f"  [CommonModes] Métrica espectral omitida: {exc}")

    return {
        "weights": weights,
        "ch_names": ch_names,
        "method": method,
        "spectral": spectral,
    }


# ============================================================================
# 6. Orquestador
# ============================================================================

def run_common_mode_analysis(
    raw,
    meta: dict,
    out_dir: str | Path,
    sfreq: float,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    raw_psd_path: str | Path | None = None,
    top_k_series: int = 12,
    top_k_topo: int = 8,
    chunk_cols: int = 50_000,
    save_projection: bool = True,
) -> dict | None:
    """Ejecutar el análisis completo de modos comunes para una grabación.

    Ver docstring del módulo.  Devuelve el dict de métricas (también
    escrito a ``common_mode_analysis.json``) o ``None`` si no hay W0.
    Todas las excepciones las captura el llamador (el pipeline imprime
    un warning y continúa).
    """
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. W0 ---------------------------------------------------------------
    common = load_common_modes(meta)
    if common is None:
        print("  [CommonModes] Sin subespacio común disponible "
              "(no A6__W0/W0); análisis omitido.")
        return None
    W0 = common["W0"]
    r0 = common["r0"]
    lambda0 = common["lambda0"]

    # --- 2. H (nivel 1) --------------------------------------------------------
    recon = _reconstruct_level1_hankel(meta)
    if recon is None:
        print("  [CommonModes] meta['stage1'] sin input_data/depth; "
              "análisis omitido.")
        return None
    H, depth = recon
    p = H.shape[0]

    stage2 = meta.get("stage2") or {}
    L = stage2.get("L")
    if L is None:
        print("  [CommonModes] stage2 sin 'L'; análisis omitido.")
        return None
    L = int(L)

    if W0.shape[0] != p * L:
        print(
            f"  [CommonModes] Dimensión de W0 ({W0.shape[0]}) != p·L "
            f"({p}·{L} = {p * L}); análisis omitido (¿depth o canales "
            "distintos a la corrida CD-HSA?)."
        )
        return None

    print(f"  [CommonModes] W0: r0={r0} ({common['source']}); "
          f"H {H.shape} + L={L} → H_block ({p * L} × {H.shape[1] - L + 1})")

    # --- 3. Proyección chunked (comunes + específicos para contraste) ---------
    bases = {"common": W0}
    W_sel = stage2.get("W_sel")
    has_specific = (isinstance(W_sel, np.ndarray)
                    and W_sel.ndim == 2 and W_sel.shape[0] == p * L
                    and W_sel.shape[1] > 0)
    if has_specific:
        bases["specific"] = np.asarray(W_sel, dtype=np.float64)

    proj = project_common_modes(H, bases, L, chunk_cols=chunk_cols)
    Y0 = proj["projections"]["common"]
    mu = proj["mu"]
    fro_total = proj["fro_total"]
    n_cols = proj["n_cols"]

    # --- 4. Métricas ------------------------------------------------------------
    common_m = energy_variance_metrics(Y0, W0, mu, fro_total, n_cols)
    specific_m = None
    if has_specific:
        Ws = bases["specific"]
        Ys = proj["projections"]["specific"]
        specific_m = energy_variance_metrics(Ys, Ws, mu, fro_total, n_cols)

    # Series centradas por fila (misma convención que el stage2 específico)
    Y_series = Y0 - Y0.mean(axis=1, keepdims=True)

    # Orden de "importancia" en ESTA grabación: energía capturada
    order_by_energy = np.argsort(-common_m["energy"])

    total_common_var = 100 * float(common_m["cum_var_frac"][-1])
    print(f"  [CommonModes] Subespacio común completo: "
          f"{total_common_var:.2f}% de la varianza (centrada) de la Hankel; "
          f"{100 * float(np.sum(common_m['energy_frac'])):.2f}% de la energía.")

    # --- 5. JSON de métricas -----------------------------------------------------
    metrics_json = {
        "r0": r0,
        "L": L,
        "depth": depth,
        "n_channels": int(p // depth) if depth else None,
        "hankel_block_shape": [p * L, int(n_cols)],
        "source": common["source"],
        "lambda0": [float(v) for v in lambda0],
        "common": {
            "energy": [float(v) for v in common_m["energy"]],
            "energy_frac": [float(v) for v in common_m["energy_frac"]],
            "var_frac": [float(v) for v in common_m["var_frac"]],
            "cum_var_frac": [float(v) for v in common_m["cum_var_frac"]],
            "total_var_frac": float(common_m["cum_var_frac"][-1]),
            "total_energy_frac": float(np.sum(common_m["energy_frac"])),
        },
        "specific": (
            {
                "n_modes": int(len(specific_m["var_frac"])),
                "var_frac": [float(v) for v in specific_m["var_frac"]],
                "total_var_frac": float(np.sum(specific_m["var_frac"])),
                "total_energy_frac": float(np.sum(specific_m["energy_frac"])),
            }
            if specific_m else None
        ),
        "hankel_total_fro_sq": float(fro_total),
        "hankel_total_var_centered": float(common_m["total_var_centered"]),
        "order_by_energy": [int(j) for j in order_by_energy],
        "plot_order": ("by captured energy on this recording; labels CM{j} "
                       "are model indices (λ0 order)"),
        "elapsed_s": None,   # rellenado al final
    }

    # --- 6. Plots ------------------------------------------------------------------
    plot_common_mode_timeseries(
        Y_series, order_by_energy, common_m["energy_frac"], lambda0,
        out_dir, top_k=top_k_series, sfreq=sfreq,
    )
    plot_common_variance_explained(common_m, specific_m, lambda0, out_dir)

    # --- 7. Influencia de canales ----------------------------------------------------
    infl = compute_common_channel_influence(
        raw, Y_series, meta, out_dir, sfreq=sfreq,
        fmin=l_freq, fmax=h_freq, raw_psd_path=raw_psd_path,
    )
    mode_labels = [f"CM{j}" for j in range(r0)]
    plot_common_channel_influence_heatmap(
        infl["weights"], infl["ch_names"], mode_labels, out_dir,
        order_by_energy,
    )
    topo_path = plot_common_topoplots(
        raw, infl["weights"], infl["ch_names"], mode_labels,
        order_by_energy, out_dir, top_k=top_k_topo,
    )

    payload = {
        "influence_weights": infl["weights"],
        "ch_names": np.asarray(infl["ch_names"]),
        "common_mode_labels": np.asarray(mode_labels),
        "method": np.array(infl["method"], dtype=object),
        "order_by_energy": order_by_energy,
        "lambda0": lambda0,
    }
    if infl["spectral"] is not None:
        payload.update(
            spectral_contribution=infl["spectral"]["spectral_contribution"],
            spectral_correlation=infl["spectral"]["spectral_correlation"],
            band_names=np.asarray(infl["spectral"]["band_names"]),
            band_ranges=infl["spectral"]["band_ranges"],
        )
    np.savez(out_dir / "common_channel_influence_data.npz", **payload)

    # --- 8. Proyección completa (opt-in) ---------------------------------------------
    if save_projection:
        np.savez_compressed(
            out_dir / "common_projection_data.npz",
            Y_common=Y_series,
            mu=mu,
            energy=common_m["energy"],
            energy_frac=common_m["energy_frac"],
            lambda0=lambda0,
            order_by_energy=order_by_energy,
        )

    # --- 9. JSON final -----------------------------------------------------------------
    metrics_json["elapsed_s"] = round(time.time() - t0, 2)
    metrics_json["artifacts"] = [
        "common_mode_projection.png",
        "common_variance_explained.png",
        "common_channel_influence.png",
        "common_channel_influence_data.npz",
        "common_mode_analysis.json",
    ] + ([
        "common_channel_influence_topo.png",
    ] if topo_path else []) + ([
        "common_projection_data.npz",
    ] if save_projection else [])

    with open(out_dir / "common_mode_analysis.json", "w",
              encoding="utf-8") as fh:
        json.dump(metrics_json, fh, indent=2)

    print(f"  [CommonModes] Análisis completado en "
          f"{metrics_json['elapsed_s']} s ({len(metrics_json['artifacts'])} "
          "artefactos).")
    return metrics_json
