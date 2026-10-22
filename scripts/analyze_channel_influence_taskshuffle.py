#!/usr/bin/env python3
"""
analyze_channel_influence_taskshuffle.py
========================================
Analisis POST-HOC de la influencia de los canales originales sobre los
modos COMUNES (A6 / W0) y sobre los modos ESPECIFICOS (Step D) para cada
(SS, celda) del CONTROL NEGATIVO task-shuffle, sobre resultados YA
guardados en disco.

NO re-corre nada del pipeline y NO toca nada del experimento real: todas
las figuras y metricas se escriben en una carpeta NUEVA dentro del
propio results-dir del control (que ya lleva el token ``_taskshuf{seed}``
en su nombre), por lo que la colision con el run real es imposible.

Que genera (por cada par (super-sujeto s, celda c))::

    {out_dir}/ss{s}_c{c}_{tarea_nominal}/
        common_mode_projection.png        series temporales de los modos
                                          comunes dominantes (por energia)
        common_variance_explained.png     varianza explicada por modo +
                                          acumulada, comunes vs especificos
        common_channel_influence.png      heatmap canales x r0 (TODOS los
                                          modos comunes)
        common_channel_influence_topo.png  topoplots del top-K (mejor
                                          esfuerzo, montaje 10-20 estandar)
        common_channel_influence_data.npz pesos + metadatos
        channel_influence.png             heatmap canales x top-n modos
                                          ESPECIFICOS de la condicion
                                          nominal de la celda (analogo de
                                          la "figura estrella" del pipeline
                                          latente)
        channel_influence_data.npz        pesos + metadatos
        analysis.json                     metricas + mezcla REAL de tareas
    {out_dir}/summary_channel_influence.json
    {out_dir}/README.txt

Metodo (identico al bloque common_mode_analysis de la fase latente v5.2
y al fallback por correlacion de compute_channel_influence_on_latente):

1. H = H_ss{s}_c{c} de ``hankel_matrices.npz`` — la Hankel de NIVEL 1 de
   la celda PERMUTADA (exactamente los datos con los que entreno el
   control; no se re-carga nada de la BD).
2. Proyeccion por bloques (``project_common_modes``): Y0 = W0^T H_block,
   Ys = W_sel^T H_block — misma convencion de bloques-lag que estimo los
   modos (``a_common_subspace.build_block_hankel``). H_block NO se
   materializa.
3. Metricas de energia / varianza explicada por modo
   (``energy_variance_metrics``), comunes y especificas.
4. Recuperacion EXACTA de las series de canales originales X (p x T)
   a partir de la Hankel de nivel 1 (los datos filtrados con los que se
   construyo — mismos que meta['preprocessing']['X_filtered'] del
   pipeline latente).
5. Influencia = correlacion de Pearson canal-serie de modo
   (``compute_channel_latent_correlation``, segmento centrado),
   normalizada max-abs por modo, signo preservado — IDENTICA metrica de
   common_channel_influence.png / channel_influence.png del run real.

AVISO (igual que el mode_map del control)
-----------------------------------------
Las etiquetas de condicion del run control son NOMINALES: cada celda es
una mezcla aleatoria de tareas. La mezcla REAL de cada celda (segun
``hankel_info.json`` -> ``task_shuffle.assignment``) se incluye en cada
``analysis.json`` y en el summary. Las figuras de modos especificos de
la celda c usan los modos de la condicion NOMINAL c — sirven para
INSPECCIONAR el control, no como mapa de modos "por tarea".

Uso (desde la raiz del repo, con el venv activo)::

    python scripts/analyze_channel_influence_taskshuffle.py \
        --results-dir results/cdhsa/session1/nSS5_L10_var99c50_a6n200_bcn5000/1.0-40.0Hz_depth10/from100.0s_to200.0s_eyesclosed_mathematic_taskshuf20261022

    # top-n distinto de 4 y/o carpeta de salida explicita:
    python scripts/analyze_channel_influence_taskshuffle.py \
        --results-dir <DIR> --top-n 4 \
        --out-dir results/otra_carpeta

    # mode_map ya extraido (si corrio extract_mode_map_taskshuffle.py);
    # si no se pasa, se busca el nombre con token taskshuf en params/ y
    # en ultimo termino se selecciona por lambda (misma regla que
    # extract_mode_indices):
    python scripts/analyze_channel_influence_taskshuffle.py \
        --results-dir <DIR> \
        --mode-map params/mode_map_4_modes_session1_eyesclosed_mathematic_100.0s-200.0s_taskshuf20261022.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Raiz del repo = padre del directorio scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.latent_space_extraction.common_mode_analysis import (  # noqa: E402
    energy_variance_metrics,
    plot_common_channel_influence_heatmap,
    plot_common_mode_timeseries,
    plot_common_variance_explained,
    project_common_modes,
)
from src.spectral_analysis.influence_metrics import (  # noqa: E402
    compute_channel_latent_correlation,
)

# Token del control en el nombre del results-dir: ..._taskshuf20261022
_TS_RE = re.compile(r"_taskshuf(\d+)$")

# Componente de ventana/tareas: from100.0s_to200.0s_eyesclosed_mathematic
_DIR_RE = re.compile(
    r"^from(?P<t0>[\d.]+)s_to(?P<t1>[\d.]+)s_(?P<tasks>[A-Za-z0-9_]+)$"
)

METHOD_CORR = "correlation_fallback(cdhsa_posthoc)|maxabs_norm"


def _fail(msg: str) -> int:
    print(f"[ERROR] {msg}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Recuperacion EXACTA de los canales originales desde la Hankel nivel 1
# ---------------------------------------------------------------------------

def recover_channels_from_level1_hankel(H: np.ndarray, depth: int) -> np.ndarray:
    """Recuperar X (p x T) filtrada a partir de la Hankel de nivel 1.

    La Hankel se construyo con ``_build_multivariate_hankel``::

        H[ch * depth + i, t] = X[ch, t + i]

    (bloque por canal, retardo dentro del canal; ``T = n_cols + depth - 1``).
    Por tanto::

        X[:, :n]          = H[0::depth, :]           (retardo 0)
        X[:, n + m]       = H[(m+1)::depth, -1]      (m = 0..depth-2)

    Es la MISMA matriz filtrada (``X_filtered``) con la que el pipeline
    latente calcula la influencia de canales del run real.
    """
    H = np.asarray(H, dtype=np.float64)
    p_hat, n = H.shape
    if p_hat % depth:
        raise ValueError(
            f"filas de H ({p_hat}) no divisibles por depth ({depth})"
        )
    p = p_hat // depth
    X = np.empty((p, n + depth - 1), dtype=np.float64)
    X[:, :n] = H[0::depth, :]
    for m in range(depth - 1):
        X[:, n + m] = H[(m + 1)::depth, -1]
    return X


# ---------------------------------------------------------------------------
# Carga de modos especificos (mismos 3 formatos que stage2_dynamics)
# ---------------------------------------------------------------------------

def load_specific_modes(
    npz, c_idx: int, r_specific: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """W(c) y lambda(c) de la condicion c_idx desde cdhsa_arrays.npz.

    Soporta los tres formatos de guardado (ver
    ``CDHSASpecificModesDynamics.fit_transform``):
      (a) ``D__W_specific`` 3-D apilado   (C, d, qmax)
      (b) ``D__W_specific`` 2-D concatenado (d, sum(r_c))
      (c) ``D__W_specific__{c}`` ragged   (d, r_c)   [cdhsa >= v5.1]
    """
    rc = int(r_specific[c_idx])
    if "D__W_specific" in npz.files:
        W_all = np.asarray(npz["D__W_specific"], dtype=np.float64)
        if W_all.ndim == 3:
            W_c = W_all[c_idx, :, :rc]
        else:
            offset = int(np.sum(r_specific[:c_idx]))
            W_c = W_all[:, offset:offset + rc]
    elif f"D__W_specific__{c_idx}" in npz.files:
        W_c = np.asarray(npz[f"D__W_specific__{c_idx}"], dtype=np.float64)
        if W_c.ndim != 2 or W_c.shape[1] < rc:
            raise ValueError(
                f"D__W_specific__{c_idx} tiene shape {W_c.shape}, "
                f"esperado (d, >= {rc}) segun D__r_specific"
            )
        W_c = W_c[:, :rc]
    else:
        raise KeyError(
            "cdhsa_arrays.npz no tiene 'D__W_specific' ni "
            f"'D__W_specific__{c_idx}' (paso D no guardado?)"
        )

    if f"D__lambda_specific__{c_idx}" in npz.files:
        lam_c = np.ravel(
            np.asarray(npz[f"D__lambda_specific__{c_idx}"], dtype=np.float64)
        )[:rc]
    elif "D__lambda_specific" in npz.files:
        lam_all = np.asarray(npz["D__lambda_specific"], dtype=np.float64)
        if lam_all.ndim == 2:
            lam_c = lam_all[c_idx, :rc]
        else:
            offset = int(np.sum(r_specific[:c_idx]))
            lam_c = np.ravel(lam_all)[offset:offset + rc]
    else:
        lam_c = np.full(rc, np.nan, dtype=np.float64)
    return W_c, lam_c


def select_mode_indices(
    lam_c: np.ndarray, rc: int, top_n: int,
    mode_map_cond: dict | None,
) -> list[int]:
    """Indices de los top-n modos especificos de la condicion.

    1. Si hay ``mode_map`` (condicion del JSON extraido por
       ``extract_mode_indices``): usa sus ``mode_indices_in_W`` (la
       seleccion canonica del run), recortados a r_c.
    2. Si no: ordena por lambda descendente y toma top-n (la MISMA regla
       de ``extract_mode_indices``).
    """
    if mode_map_cond:
        idx = [int(i) for i in mode_map_cond.get("mode_indices_in_W", [])]
        idx = [i for i in idx if i < rc][:top_n]
        if idx:
            return idx
    if rc == 0:
        return []
    order = np.argsort(-np.asarray(lam_c, dtype=np.float64), kind="stable")
    return [int(j) for j in order[:top_n]]


# ---------------------------------------------------------------------------
# Influencia de canales (misma metrica que el pipeline real)
# ---------------------------------------------------------------------------

def channel_influence_from_series(
    X: np.ndarray, Y: np.ndarray
) -> np.ndarray:
    """Correlacion canal-serie de modo + normalizacion max-abs por modo.

    ``X``: (p, T) canales originales (recuperados de la Hankel).
    ``Y``: (k, n_samples) series de los modos, centradas por fila.
    Devuelve (p, k) en [-1, 1] con signo preservado — identico a
    ``compute_common_channel_influence`` /
    ``compute_channel_influence_on_latent`` (fallback por correlacion,
    que es el que aplica a los modos CD-HSA del espacio block-Hankel).
    """
    latent = np.asarray(Y, dtype=float).T          # (n_samples, k)
    weights = compute_channel_latent_correlation(X, latent)
    weights = np.asarray(weights, dtype=float)
    col_max = np.abs(weights).max(axis=0, keepdims=True)
    col_max[col_max == 0.0] = 1.0
    return weights / col_max


def _maybe_topo_grid(
    ch_names: list[str],
    weights: np.ndarray,
    dim_labels: list[str],
    sfreq: float,
    title: str,
):
    """Topoplots mejor-esfuerzo con montaje 10-20 estandar (o None)."""
    try:
        import mne
        from src.spectral_analysis.plotting_utils import plot_topomap_grid
    except Exception:
        return None
    try:
        info = mne.create_info(ch_names, float(sfreq), ch_types="eeg")
        montage = mne.channels.make_standard_montage("standard_1020")
        ch_pos = montage.get_positions()["ch_pos"]
        matched = {c for c in ch_names if c in set(ch_pos)}
        if len(matched) < 3 or len(matched) != len(ch_names):
            return None
        info.set_montage(montage, on_missing="raise")
        return plot_topomap_grid(info, weights, dim_labels, title=title)
    except Exception:
        return None


def _plot_specific_heatmap(
    weights: np.ndarray,
    ch_names: list[str],
    mode_labels: list[str],
    lam_sel,
    out_dir: Path,
) -> Path:
    """Heatmap canales x modos especificos (analogo de la figura estrella)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_ch, k = weights.shape
    fig_h = max(4.0, 0.22 * n_ch + 1.5)
    fig_w = max(6.0, 0.9 * k + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(weights, aspect="auto", cmap="RdBu_r", vmin=-1.0,
                   vmax=1.0, interpolation="nearest")
    ticks = np.arange(k)
    ax.set_xticks(ticks)
    ax.set_xticklabels(mode_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_ch))
    ax.set_yticklabels(ch_names, fontsize=7)
    ax.set_xlabel("Condition-specific mode (Step D, nominal condition)",
                  fontsize=10)
    ax.set_ylabel("Original EEG channel", fontsize=10)
    lam_txt = ""
    if lam_sel is not None and len(lam_sel):
        lam_txt = "  |  " + ", ".join(
            f"\u03bb={v:.3f}" if np.isfinite(v) else "n/a"
            for v in lam_sel
        )
    ax.set_title(
        "Channel influence on condition-specific modes (D)\n"
        "(channel\u2013latent Pearson correlation, max-abs normalised "
        "per mode)" + lam_txt,
        fontsize=11, fontweight="bold",
    )
    fig.colorbar(im, ax=ax, shrink=0.85, label="Influence (norm., signed)")
    path = out_dir / "channel_influence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Mezcla real de tareas de una celda (desde hankel_info task_shuffle)
# ---------------------------------------------------------------------------

def cell_task_mix(hankel_info: dict, ss_id: int) -> list[dict]:
    """Mezcla REAL de tareas por celda del SS (etiquetas nominales!).

    ``hankel_info['task_shuffle']['assignment'][str(ss)][str(subj)]`` =
    lista de C tareas (la tarea REAL de la grabacion de ese sujeto en
    cada celda). Devuelve, por celda: conteo por tarea + total.
    """
    ts = (hankel_info.get("task_shuffle") or {})
    assignment = ts.get("assignment") or {}
    per_ss = assignment.get(str(ss_id), {})
    if not per_ss:
        return []
    C = int(hankel_info.get("C", 0)) or max(
        (len(v) for v in per_ss.values()), default=0
    )
    out: list[dict] = []
    for c in range(C):
        counts = Counter()
        for _subj, per_cell in per_ss.items():
            if c < len(per_cell):
                counts[per_cell[c]] += 1
        out.append({
            "counts": dict(sorted(counts.items())),
            "total": int(sum(counts.values())),
        })
    return out


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Analisis post-hoc de la influencia de los canales originales "
            "sobre los modos comunes (A6) y especificos (D) de cada "
            "(SS, celda) del control negativo task-shuffle. No re-corre "
            "nada; escribe en una carpeta propia dentro del results-dir "
            "del control (cero colision con el run real)."
        )
    )
    p.add_argument(
        "--results-dir", required=True,
        help="Results-dir del run con shuffle (debe terminar en "
             "_taskshuf{seed})",
    )
    p.add_argument(
        "--top-n", type=int, default=4,
        help="Modos especificos por condicion a analizar (default 4, "
             "igual que los runs reales)",
    )
    p.add_argument(
        "--top-k-series", type=int, default=12,
        help="Modos comunes en common_mode_projection.png (default 12)",
    )
    p.add_argument(
        "--top-k-topo", type=int, default=8,
        help="Modos comunes en common_channel_influence_topo.png "
             "(default 8)",
    )
    p.add_argument(
        "--chunk-cols", type=int, default=50_000,
        help="Columnas de H_block por bloque en la proyeccion "
             "(default 50000; pico ~2.1 GB con p*L=5300)",
    )
    p.add_argument(
        "--mode-map", default=None,
        help="mode_map.json del control (el de extract_mode_map_"
             "taskshuffle.py). Default: buscar en params/ el nombre con "
             "token taskshuf; en su defecto seleccion por lambda.",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Carpeta de salida. Default: <results-dir>/channel_influence"
             "_taskshuf{seed} (dentro del control: imposible colisionar "
             "con el run real).",
    )
    p.add_argument("--no-topo", action="store_true",
                   help="No generar topoplots (aunque haya montaje)")
    args = p.parse_args(argv)

    t0 = time.time()

    # ------------------------------------------------------------------
    # Validaciones del results-dir
    # ------------------------------------------------------------------
    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.is_dir():
        return _fail(f"no existe el directorio: {results_dir}")

    m = _TS_RE.search(results_dir.name)
    if not m:
        return _fail(
            f"el nombre del results-dir no termina en _taskshuf{{seed}}: "
            f"'{results_dir.name}'. Este script es solo para runs del "
            "control negativo (task-shuffle)."
        )
    seed = int(m.group(1))

    missing = [
        f for f in ("hankel_info.json", "config.json", "cdhsa_arrays.npz",
                    "hankel_matrices.npz")
        if not (results_dir / f).exists()
    ]
    if missing:
        return _fail(
            f"el results-dir no contiene: {', '.join(missing)} "
            "(analisis imposible sin esos archivos)"
        )

    # ------------------------------------------------------------------
    # Metadatos del run
    # ------------------------------------------------------------------
    hankel_info = json.loads(
        (results_dir / "hankel_info.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (results_dir / "config.json").read_text(encoding="utf-8")
    )

    S = int(hankel_info["S"])
    C = int(hankel_info["C"])
    tasks = list(hankel_info["tasks"])
    ch_names = list(hankel_info.get("global_channels") or [])
    sfreq = hankel_info.get("sfreq_common")
    depth = hankel_info.get("depth_common")
    L = int(config.get("L") or hankel_info.get("L") or 0)

    if not ch_names:
        return _fail("hankel_info.json sin 'global_channels': imposible "
                     "etiquetar los canales de la figura.")
    if sfreq is None or depth is None:
        return _fail("hankel_info.json sin 'sfreq_common'/'depth_common'.")
    if L <= 0:
        return _fail("config.json sin 'L' (retardo de nivel 2).")

    ts_seed = (hankel_info.get("task_shuffle") or {}).get("seed", seed)
    mix_per_ss = {ss: cell_task_mix(hankel_info, ss)
                  for ss in range(1, S + 1)}

    print("=" * 70)
    print("  INFLUENCIA DE CANALES -- MODOS COMUNES + ESPECIFICOS")
    print("  (CONTROL NEGATIVO task-shuffle, post-hoc)")
    print("=" * 70)
    print(f"  results-dir : {results_dir}")
    print(f"  seed shuffle: {ts_seed}")
    print(f"  SS x celdas : {S} x {C} (tareas nominales: {', '.join(tasks)})")
    print(f"  canales     : {len(ch_names)}   depth={depth}   L={L}"
          + (f"   sfreq={sfreq:.1f} Hz" if sfreq else ""))
    print()
    print("  [AVISO] Etiquetas de condicion NOMINALES (celdas = mezcla")
    print("  aleatoria de tareas). La mezcla REAL de cada celda esta en")
    print("  cada analysis.json y en summary_channel_influence.json.")
    print()

    # ------------------------------------------------------------------
    # Carpeta de salida (propia, dentro del control)
    # ------------------------------------------------------------------
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = results_dir / f"channel_influence_taskshuf{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Salida      : {out_dir}  (cero colision con el run real)")
    print()

    # ------------------------------------------------------------------
    # mode_map (opcional): seleccion canonica de modos especificos
    # ------------------------------------------------------------------
    mode_map = None
    if args.mode_map:
        mm_path = Path(args.mode_map).expanduser().resolve()
        if not mm_path.exists():
            return _fail(f"--mode-map no existe: {mm_path}")
        mode_map = json.loads(mm_path.read_text(encoding="utf-8"))
        print(f"  mode_map    : {mm_path}")
    else:
        stem = _TS_RE.sub("", results_dir.name)
        dm = _DIR_RE.match(stem)
        if dm:
            tw_tag = f"{dm.group('t0')}s-{dm.group('t1')}s"
            session = results_dir.parents[2].name
            guess = (REPO_ROOT / "params" /
                     f"mode_map_{args.top_n}_modes_{session}_"
                     f"{dm.group('tasks')}_{tw_tag}_taskshuf{seed}.json")
            if guess.exists():
                mode_map = json.loads(guess.read_text(encoding="utf-8"))
                print(f"  mode_map    : {guess} (encontrado)")
    if mode_map is None:
        print("  mode_map    : (no encontrado; seleccion de modos "
              "especificos por lambda — misma regla de "
              "extract_mode_indices)")
    print()

    # ------------------------------------------------------------------
    # Arrays del modelo: W0 (comunes) + W(c) (especificos)
    # ------------------------------------------------------------------
    with np.load(results_dir / "cdhsa_arrays.npz", allow_pickle=True) as npz:
        if "A6__W0" not in npz.files:
            return _fail(
                "cdhsa_arrays.npz sin 'A6__W0' (sin subespacio comun no "
                "hay analisis de modos comunes)."
            )
        W0 = np.asarray(npz["A6__W0"], dtype=np.float64)
        lambda0 = (np.ravel(np.asarray(npz["A6__lambda0"],
                                       dtype=np.float64))
                   if "A6__lambda0" in npz.files else None)
        r0 = int(W0.shape[1])
        if "D__r_specific" in npz.files:
            r_specific = np.asarray(
                np.ravel(npz["D__r_specific"]), dtype=int
            )
            specific_available = True
        else:
            r_specific = np.zeros(C, dtype=int)
            specific_available = False
            print("  [WARN] sin D__r_specific: se omiten los modos "
                  "especificos (solo comunes).")
        W_spec: list = [None] * C
        lam_spec: list = [None] * C
        if specific_available:
            for c in range(C):
                if int(r_specific[c]) > 0:
                    W_spec[c], lam_spec[c] = load_specific_modes(
                        npz, c, r_specific
                    )
        d_expected = W0.shape[0]

    print(f"  W0          : ({W0.shape[0]}, {r0})"
          + (f", lambda0[{lambda0[0]:.3f}..{lambda0[-1]:.3f}]"
             if lambda0 is not None and len(lambda0) else "")
          + f"  (d = {d_expected})")
    for c in range(C):
        if W_spec[c] is not None:
            print(f"  W({tasks[c]})  : ({W_spec[c].shape[0]}, "
                  f"{W_spec[c].shape[1]})  r_c={int(r_specific[c])}")
    print()

    # ------------------------------------------------------------------
    # Bucle principal: un (SS, celda) a la vez (memoria acotada)
    # ------------------------------------------------------------------
    summary_cells = []
    with np.load(results_dir / "hankel_matrices.npz", allow_pickle=True) \
            as hankel_npz:
        for s in range(1, S + 1):
            for c in range(1, C + 1):
                key = f"H_ss{s}_c{c}"
                if key not in hankel_npz.files:
                    print(f"  [SKIP] {key} no esta en hankel_matrices.npz")
                    continue
                H = np.asarray(hankel_npz[key], dtype=np.float64)
                if H.size == 0 or H.ndim != 2 or H.shape[1] <= L:
                    print(f"  [SKIP] {key} vacia o demasiado corta "
                          f"({H.shape})")
                    continue

                cell_t0 = time.time()
                nominal = tasks[c - 1] if c - 1 < len(tasks) else f"c{c}"
                cell_dir = out_dir / f"ss{s}_c{c}_{nominal}"
                cell_dir.mkdir(parents=True, exist_ok=True)

                mix = (mix_per_ss.get(s, [{}] * C)[c - 1]
                       if c - 1 < len(mix_per_ss.get(s, [])) else {})
                mix_str = ", ".join(
                    f"{t} x{n}" for t, n in mix.get("counts", {}).items()
                ) or "(sin info)"

                print(f"  --- SS{s} celda {c} (nominal '{nominal}'; "
                      f"mezcla real: {mix_str}) ---")
                sys.stdout.flush()

                # --- dimensiones coherentes -----------------------------
                p_hat = H.shape[0]
                if p_hat % depth:
                    print(f"  [SKIP] {key}: filas {p_hat} no divisibles "
                          f"por depth {depth}")
                    continue
                if p_hat * L != d_expected:
                    print(f"  [SKIP] {key}: p*L = {p_hat * L} != d = "
                          f"{d_expected} (canales/depth distintos al "
                          "modelo?)")
                    continue

                # --- proyeccion (comunes + especificos juntos) ----------
                bases = {"common": W0}
                sel_idx: list = []
                if W_spec[c - 1] is not None:
                    mm_cond = None
                    if mode_map is not None:
                        mm_cond = (mode_map.get("conditions", {})
                                   .get(nominal))
                    sel_idx = select_mode_indices(
                        lam_spec[c - 1], int(r_specific[c - 1]),
                        max(1, int(args.top_n)), mm_cond,
                    )
                    if sel_idx:
                        bases["specific"] = W_spec[c - 1][:, sel_idx]

                proj = project_common_modes(
                    H, bases, L, chunk_cols=int(args.chunk_cols),
                )
                Y0 = proj["projections"]["common"]
                Ys = proj["projections"].get("specific")
                mu, fro_total, n_cols = (proj["mu"], proj["fro_total"],
                                         proj["n_cols"])

                common_m = energy_variance_metrics(
                    Y0, W0, mu, fro_total, n_cols,
                )
                specific_m = (energy_variance_metrics(
                    Ys, bases["specific"], mu, fro_total, n_cols,
                ) if Ys is not None else None)

                # --- series centradas (convencion del pipeline) ---------
                Y0_series = Y0 - Y0.mean(axis=1, keepdims=True)
                Ys_series = (Ys - Ys.mean(axis=1, keepdims=True)
                             if Ys is not None else None)

                # --- canales originales (recuperacion exacta) -----------
                X = recover_channels_from_level1_hankel(H, depth)
                del H

                # --- influencia de canales -------------------------------
                w_common = channel_influence_from_series(X, Y0_series)
                w_specific = (channel_influence_from_series(X, Ys_series)
                              if Ys_series is not None else None)

                # --- orden por energia capturada (para los plots) --------
                order_by_energy = np.argsort(-common_m["energy"])

                # --- figuras comunes -------------------------------------
                lam0_plot = (lambda0 if lambda0 is not None
                             and len(lambda0) == r0
                             else np.full(r0, np.nan))
                plot_common_mode_timeseries(
                    Y0_series, order_by_energy,
                    common_m["energy_frac"], lam0_plot, cell_dir,
                    top_k=int(args.top_k_series), sfreq=sfreq,
                )
                plot_common_variance_explained(
                    common_m, specific_m, lam0_plot, cell_dir,
                )
                plot_common_channel_influence_heatmap(
                    w_common, ch_names,
                    [f"CM{j}" for j in range(r0)], cell_dir, order_by_energy,
                )
                if not args.no_topo:
                    k_topo = min(int(args.top_k_topo), r0)
                    idx_topo = order_by_energy[:k_topo]
                    fig_topo = _maybe_topo_grid(
                        ch_names, w_common[:, idx_topo],
                        [f"CM{j}" for j in idx_topo], sfreq,
                        "Channel influence \u2014 top common modes (A6)",
                    )
                    if fig_topo is not None:
                        fig_topo.savefig(
                            cell_dir / "common_channel_influence_topo.png",
                            dpi=150,
                        )
                        import matplotlib.pyplot as plt
                        plt.close(fig_topo)

                # --- figura especifica (analogo figura estrella) --------
                if w_specific is not None and sel_idx:
                    lam_sel = (lam_spec[c - 1][sel_idx]
                               if lam_spec[c - 1] is not None else None)
                    labels_sel = [f"SM{j}" for j in sel_idx]
                    _plot_specific_heatmap(
                        w_specific, ch_names, labels_sel, lam_sel, cell_dir,
                    )
                    np.savez_compressed(
                        cell_dir / "channel_influence_data.npz",
                        influence_weights=w_specific,
                        ch_names=np.array(ch_names, dtype=object),
                        mode_indices_in_W=np.array(sel_idx, dtype=int),
                        eigenvalues=(lam_sel if lam_sel is not None
                                     else np.full(len(sel_idx), np.nan)),
                        latent_dim_labels=np.array(labels_sel, dtype=object),
                        method=METHOD_CORR + "|specific_posthoc",
                    )

                np.savez_compressed(
                    cell_dir / "common_channel_influence_data.npz",
                    influence_weights=w_common,
                    ch_names=np.array(ch_names, dtype=object),
                    lambda0=(lambda0 if lambda0 is not None
                             else np.full(r0, np.nan)),
                    energy=common_m["energy"],
                    energy_frac=common_m["energy_frac"],
                    var_frac=common_m["var_frac"],
                    cum_var_frac=common_m["cum_var_frac"],
                    order_by_energy=order_by_energy,
                    method=METHOD_CORR + "|common_posthoc",
                )

                # --- top canales de los modos dominantes -----------------
                top_modes = order_by_energy[:3]
                top_channels = {
                    f"CM{j}": [
                        ch_names[i] for i in np.argsort(
                            -np.abs(w_common[:, j])
                        )[:5].tolist()
                    ]
                    for j in top_modes
                }

                # --- analysis.json de la celda ---------------------------
                analysis = {
                    "cell": {"ss": s, "c": c,
                             "nominal_task": nominal,
                             "real_task_mix": mix,
                             "warning": (
                                 "Etiqueta NOMINAL: la celda es una mezcla "
                                 "aleatoria de tareas (control negativo). "
                                 "Ver hankel_info.json -> task_shuffle."
                             )},
                    "dimensions": {
                        "channels": len(ch_names), "depth": depth,
                        "L": L, "d": d_expected, "r0": r0,
                        "r_specific": (int(r_specific[c - 1])
                                        if specific_available else 0),
                        "specific_selected": sel_idx,
                        "n_cols_level2": int(n_cols),
                    },
                    "common": {
                        "lambda0": (lambda0.tolist()
                                    if lambda0 is not None else None),
                        "energy_frac": common_m["energy_frac"].tolist(),
                        "var_frac": common_m["var_frac"].tolist(),
                        "cum_var_frac": common_m["cum_var_frac"].tolist(),
                        "total_var_frac_common": float(
                            common_m["cum_var_frac"][-1]),
                        "total_energy_frac_common": float(
                            np.sum(common_m["energy_frac"])),
                        "order_by_energy": order_by_energy.tolist(),
                        "top_channels_by_mode": top_channels,
                        "method": METHOD_CORR + "|common_posthoc",
                    },
                    "specific": (
                        {"selected_indices_in_W": sel_idx,
                         "eigenvalues": (lam_spec[c - 1][sel_idx].tolist()
                                         if lam_spec[c - 1] is not None
                                         else None),
                         "var_frac": specific_m["var_frac"].tolist(),
                         "total_var_frac_specific": float(
                             np.sum(specific_m["var_frac"])),
                         "method": METHOD_CORR + "|specific_posthoc"}
                        if specific_m is not None else None
                    ),
                    "files": sorted(f.name for f in cell_dir.iterdir()),
                }
                (cell_dir / "analysis.json").write_text(
                    json.dumps(analysis, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                summary_cells.append({
                    "ss": s, "c": c, "nominal_task": nominal,
                    "real_task_mix": mix,
                    "r0": r0, "r_specific": int(r_specific[c - 1]),
                    "specific_selected": sel_idx,
                    "total_var_frac_common":
                        float(common_m["cum_var_frac"][-1]),
                    "total_var_frac_specific": (
                        float(np.sum(specific_m["var_frac"]))
                        if specific_m is not None else None),
                    "dir": str(cell_dir.relative_to(out_dir)),
                })
                del X, Y0, Ys, Y0_series, Ys_series, proj
                print(f"      listo en {time.time() - cell_t0:.1f}s -> "
                      f"{cell_dir.name}/")
                sys.stdout.flush()

    if not summary_cells:
        return _fail("ninguna celda analizable (vea los [SKIP] arriba).")

    # ------------------------------------------------------------------
    # Summary + README
    # ------------------------------------------------------------------
    summary = {
        "control_negativo": {
            "seed": ts_seed,
            "results_dir": str(results_dir),
            "warning_labels": (
                "Etiquetas de condicion NOMINALES (mezcla aleatoria de "
                "tareas por celda). NO usar como mapa de modos por tarea."
            ),
        },
        "config": {"L": L, "depth": depth, "channels": len(ch_names),
                   "sfreq": sfreq, "r0": r0,
                   "r_specific": (r_specific.tolist()
                                  if specific_available else None),
                   "top_n": int(args.top_n),
                   "chunk_cols": int(args.chunk_cols),
                   "mode_map": (str(args.mode_map) if args.mode_map
                                else ("auto" if mode_map else "lambda"))},
        "cells": summary_cells,
        "no_collision_note": (
            "Todas las figuras de este analisis viven dentro del "
            "results-dir del control (token _taskshuf{seed}); el "
            "experimento real (sin token) no se toca en ningun momento."
        ),
    }
    (out_dir / "summary_channel_influence.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    readme = f"""INFLUENCIA DE CANALES — CONTROL NEGATIVO task-shuffle (seed {ts_seed})
=====================================================================

Que es esto
-----------
Analisis post-hoc de la influencia de los canales originales sobre los
modos COMUNES (A6/W0, {r0} modos) y sobre los modos ESPECIFICOS (Step D)
de cada par (super-sujeto, celda) del control negativo task-shuffle.
Se computa directamente desde los artefactos guardados del run
(hankel_matrices.npz + cdhsa_arrays.npz): no se re-corre nada y no se
re-carga nada de la base de datos.

IMPORTANTE — etiquetas nominales
--------------------------------
En el control negativo cada celda es una MEZCLA ALEATORIA de tareas.
La etiqueta del directorio (ss{{s}}_c{{c}}_{{tarea}}) es la condicion
NOMINAL; la mezcla REAL de cada celda esta en su analysis.json y en
summary_channel_influence.json (fuente: hankel_info.json ->
task_shuffle.assignment). Las figuras de modos especificos de la celda c
usan los modos de la condicion nominal c: sirven para inspeccionar el
control, no como mapa de modos "por tarea".

Contenido por celda
-------------------
  common_mode_projection.png        series de los modos comunes
                                    dominantes (por energia capturada)
  common_variance_explained.png     varianza explicada por modo +
                                    acumulada, comunes vs especificos
  common_channel_influence.png      heatmap canales x {r0} modos comunes
  common_channel_influence_topo.png topoplots del top-{args.top_k_topo}
                                    (si el montaje 10-20 coincide)
  common_channel_influence_data.npz pesos + metricas
  channel_influence.png            heatmap canales x modos especificos
                                    (top-{args.top_n} de la condicion
                                    nominal)
  channel_influence_data.npz        pesos + indices + autovalores
  analysis.json                     metricas + mezcla real de tareas

Metrica
-------
Influencia = correlacion de Pearson entre la serie del canal original
(recuperada EXACTAMENTE de la Hankel de nivel 1: los mismos datos
filtrados con los que se entrenaron los modos) y la serie del modo
(proyeccion del block-Hankel de nivel 2). Normalizacion max-abs por
modo, signo preservado — identica a la de common_channel_influence.png /
channel_influence.png de los runs reales (fallback por correlacion:
los modos CD-HSA viven en el espacio block-Hankel, no en el de canales).

Notas de fidelidad
------------------
- La proyeccion no enmascara las columnas de frontera de concatenacion
  (misma convencion que el common_mode_analysis del pipeline latente y
  que el stage 2 especifico; efecto <= L+depth-2 columnas por frontera).
- Los modos especificos usados son los del run control (W(c) del propio
  npz), con la seleccion canonica del mode_map si se proporciono o
  encontro, y en su defecto top-n por lambda (misma regla de
  extract_mode_indices).

Colision
--------
NINGUNA: todo vive en {out_dir}
(dentro del results-dir del control, que ya lleva _taskshuf{ts_seed}).
El experimento real (ec_ma/ec_mu sin token) no se lee ni se escribe.
"""
    (out_dir / "README.txt").write_text(readme, encoding="utf-8")

    print()
    print("=" * 70)
    print(f"[OK] {len(summary_cells)} celdas analizadas en "
          f"{(time.time() - t0) / 60:.1f} min")
    print(f"[OK] Salida: {out_dir}")
    print(f"[OK] Summary: {out_dir / 'summary_channel_influence.json'}")
    print()
    print("  Siguiente paso sugerido (comparacion real vs shuffle):")
    print("    python scripts/compare_task_shuffle.py \\")
    print("        --baseline <results-dir del run REAL> \\")
    print(f"        --shuffled {results_dir} \\")
    print("        -o comparacion_shuffle.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
