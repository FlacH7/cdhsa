#!/usr/bin/env python3
"""Regenera SOLO los topoplots de influencia de canales sobre los modos
especificos (Step D) a partir de los resultados ya guardados.  (v2)

Problema que resuelve
---------------------
Tanto el pipeline de latentes (runs reales) como el analizador del control
negativo task-shuffle (``analyze_channel_influence_taskshuffle.py``) guardan
en cada carpeta de job/celda:

* ``channel_influence_data.npz``  — pesos de influencia (n_ch, n_spec),
  ``ch_names`` y ``latent_dim_labels``.  ESTE ARCHIVO ES CORRECTO.
* ``channel_influence.png``       — la figura.  Es un HEATMAP (canales x
  modos, estilo "Channel influence on condition-specific modes (D)") en
  lugar de los TOPOPLOTS ("cerebritos") que ya usa la imagen de modos
  comunes (``common_channel_influence_topo.png``).

Este script NO re-corre nada: lee los datos existentes y genera la figura que
falta, con el MISMO estilo que la de modos comunes (``plot_topomap_grid``:
1 fila, RdBu_r simetrico, colorbar por panel):

* salida por defecto:  ``channel_influence_topo.png``  (no toca el heatmap)
* con ``--overwrite``:  tambien sobreescribe ``channel_influence.png``

Layouts soportados (v2)
-----------------------
1. Runs reales (fase latente del batch): raiz
   ``.../test_retest_gedai_super_subject*/super_subject-XX/sessionN/`` con
   carpetas ``fromXs_toYs_<task>`` (la condicion sale del nombre).

2. Control negativo task-shuffle (``analyze_channel_influence_taskshuffle``):
   raiz ``.../fromXs_toYs_<t1>_<t2>_taskshuf<seed>/channel_influence_taskshuf<seed>/``
   con carpetas de celda ``ss{n}_c{c}_{task}`` (5 SS x 2 celdas = 10
   imagenes).  La condicion (etiqueta NOMINAL de la celda) sale del nombre;
   los lambda vienen del mode_map del control en ``params/ss_shufle/``.

Si el ``.npz`` no se llama exactamente ``channel_influence_data.npz``, el
script hace introspeccion de los ``.npz`` chicos (excluyendo los
``*common*``, que son los modos comunes) y de los ``analysis.json``,
buscando la matriz (n_ch, n_modes) + los nombres de canales; si no
encuentra nada, imprime el contenido de las carpetas para diagnosticar.

Uso
---
    # control negativo (10 imagenes):
    python scripts/regen_specific_influence_topo.py \\
        results/ss_shufle/cdhsa/session1/nSS5_.../from100.0s_to200.0s_eyesclosed_mathematic_taskshuf20261022/channel_influence_taskshuf20261022 \\
        --mode-map params/ss_shufle/mode_map_4_modes_session1_eyesclosed_mathematic_100.0s-200.0s_taskshuf20261022.json

    # runs reales (fase latente):
    python scripts/regen_specific_influence_topo.py <root_latente> [--mode-map ...]

    # sin lambda (etiquetas SM0..SM{m-1}):  omitir --mode-map / --lambdas
    # reemplazar ademas el heatmap:           --overwrite
    # ver que haria sin escribir nada:       --dry-run

Dependencias: numpy, matplotlib, mne.  No importa nada del repo: se puede
correr desde cualquier lado.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np


def _fmt_lambda(v: float) -> str:
    """Format lambda evitando el '1.000' enganoso cuando lambda~0.9999."""
    for prec in (3, 4, 5, 6):
        s = f"{v:.{prec}f}"
        if not s.startswith("1.00"):
            return s
    return s


# ---------------------------------------------------------------------------
# Core: figura de topoplots (mismo estilo que plot_topomap_grid del repo)
# ---------------------------------------------------------------------------

def build_positions(ch_names: list[str], montage_name: str = "standard_1020"):
    """Posiciones 2D (n, 2) desde un montaje estandar de MNE.

    Devuelve (pos, kept_idx, dropped) — ``dropped`` son los canales sin
    posicion en el montaje (se excluyen del topoplot con warning).
    """
    import mne

    mont = mne.channels.make_standard_montage(montage_name)
    available = mont.ch_names
    kept, dropped = [], []
    for i, c in enumerate(ch_names):
        if c in available:
            kept.append(i)
        else:
            dropped.append(c)
    if len(kept) < 3:
        raise RuntimeError(
            f"Menos de 3 canales con posicion en el montaje '{montage_name}' "
            f"({len(kept)} de {len(ch_names)}); no se puede dibujar el topoplot."
        )
    pos = np.array(
        [mont.get_positions()["ch_pos"][ch_names[i]][:2] for i in kept]
    )
    return pos, kept, dropped


def plot_specific_topoplots(
    weights: np.ndarray,
    ch_names: list[str],
    out_path,
    lambdas=None,
    montage_name: str = "standard_1020",
    title: str = "Channel influence \u2014 condition-specific modes (D)",
    top_k: int | None = None,
    dpi: int = 150,
) -> tuple:
    """Topoplots de la influencia canal->modo especifico (estilo A6 topo).

    ``weights``: (n_ch, n_modes) en [-1, 1] (el ``influence_weights`` del
    ``channel_influence_data.npz``).  Un panel por modo, etiqueta ``SM{j}``
    (+ lambda si se pasa), colormap RdBu_r simetrico GLOBAL (identico al
    ``plot_topomap_grid`` que usa la figura de modos comunes).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError(f"weights debe ser (n_ch, n_modes); vino {weights.shape}")
    n_ch, n_modes = weights.shape

    if top_k is not None and top_k > 0 and top_k < n_modes:
        weights = weights[:, :top_k]
        n_modes = top_k
        if lambdas is not None:
            lambdas = list(lambdas)[:n_modes]

    lambdas = list(lambdas) if lambdas is not None else None
    if lambdas is not None and len(lambdas) != n_modes:
        print(f"  [WARN] {len(lambdas)} lambdas != {n_modes} modos; se ignoran.")
        lambdas = None

    pos, kept, dropped = build_positions(ch_names, montage_name)
    if dropped:
        print(f"  [WARN] Canales sin posicion en {montage_name} "
              f"(excluidos del topoplot): {dropped}")
    W = weights[kept, :]

    labels = []
    for j in range(n_modes):
        if lambdas is not None:
            labels.append(f"SM{j} (\u03bb={_fmt_lambda(lambdas[j])})")
        else:
            labels.append(f"SM{j}")

    fig, axes = plt.subplots(1, n_modes, figsize=(3.4 * n_modes, 3.8),
                             squeeze=False)
    vmax = float(np.max(np.abs(W)))
    if vmax <= 0:
        vmax = 1.0
    for j in range(n_modes):
        ax = axes[0, j]
        im, _ = mne.viz.plot_topomap(
            W[:, j], pos, axes=ax, show=False, cmap="RdBu_r",
            vlim=(-vmax, vmax), contours=0, sensors=True,
        )
        ax.set_title(labels[j], fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path, n_modes, dropped


# ---------------------------------------------------------------------------
# Lambdas desde el mode_map (o manual)
# ---------------------------------------------------------------------------

def lambdas_from_mode_map(mode_map_path: str, condition: str):
    """Lee mode_map['conditions'][cond]['modes'][j]['eigenvalue'].

    ``modes`` se construye iterando ``mode_indices_in_W`` (orden por
    autovalor) => el modo j de esa lista es la columna j de W_sel, que es
    la columna j de influence_weights.  Devuelve None si no hay match.
    """
    with open(mode_map_path, encoding="utf-8") as f:
        mm = json.load(f)
    conds = mm.get("conditions") or {}
    cinfo = conds.get(condition)
    if cinfo is None:
        # tolerancia: match por sufijo (p.ej. carpeta "..._eyesclosed" vs
        # clave "eyesclosed")
        for k, v in conds.items():
            if condition.endswith(k) or k.endswith(condition):
                cinfo = v
                break
    if cinfo is None:
        return None
    modes = cinfo.get("modes") or []
    lams = [m.get("eigenvalue") for m in modes]
    if not lams or any(l is None for l in lams):
        # fallback: eigenvalues_sorted (todo el set, orden por autovalor)
        # + mode_indices_in_W para seleccionar
        ev_sorted = cinfo.get("eigenvalues_sorted")
        idx = cinfo.get("mode_indices_in_W")
        if ev_sorted and idx:
            lams = [ev_sorted[i] for i in idx]
        else:
            return None
    return [float(l) for l in lams]


# ---------------------------------------------------------------------------
# Nombres de carpeta -> (condicion, super-sujeto) en ambos layouts
# ---------------------------------------------------------------------------

def condition_from_dirname(folder: str) -> str | None:
    """Extrae la condicion del nombre de la carpeta.

    Layout latente:  ``from100.0s_to200.0s_eyesclosed`` -> ``eyesclosed``
    Layout control:  ``ss1_c1_eyesclosed``              -> ``eyesclosed``
    (en el control task-shuffle la etiqueta es NOMINAL: la celda es una
    mezcla aleatoria de tareas; la mezcla real esta en cada analysis.json).
    """
    base = os.path.basename(os.path.normpath(folder))
    m = re.match(r"^from[\d.]+s_to[\d.]+s_(.+)$", base)
    if m:
        return m.group(1)
    m = re.match(r"^ss\d+_c\d+_(.+)$", base)
    if m:
        return m.group(1)
    return None


def _ss_label_from_dirname(folder: str) -> str | None:
    """``.../ss1_c1_eyesclosed`` -> ``ss1`` (layout control task-shuffle)."""
    base = os.path.basename(os.path.normpath(folder))
    m = re.match(r"^(ss\d+)_c\d+_", base)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Descubrimiento de los datos de influencia (canonico + fallbacks)
# ---------------------------------------------------------------------------

_WEIGHT_KEY_PREFS = ("influence_weights", "weights", "W_influence",
                     "channel_influence", "specific_weights", "influence",
                     "W")
_CHNAMES_KEY_PREFS = ("ch_names", "channel_names", "channels", "chan_names")
_LABELS_KEY_PREFS = ("latent_dim_labels", "labels", "mode_labels",
                    "dim_labels")


def _looks_like_weights(arr) -> bool:
    return (
        hasattr(arr, "ndim") and arr.ndim == 2
        and 3 <= arr.shape[0] <= 128
        and 1 <= arr.shape[1] <= 64
        and getattr(arr.dtype, "kind", "") in "fiu"
    )


def _introspect_npz(path: str, max_bytes: int = 20 * 1024 * 1024):
    """Busca en un .npz una matriz (n_ch, n_modes) + nombres de canales.

    Devuelve dict con claves 'w', 'c', 'l' (o None si no parece de
    influencia).  Llamar solo con .npz SIN 'common' en el nombre (los
    common_channel_influence_data.npz son los modos comunes, no los
    especificos) y con tama~o razonable (para no cargar cdhsa_arrays.npz).
    """
    try:
        if os.path.getsize(path) > max_bytes:
            return None
        data = np.load(path, allow_pickle=True)
        keys = list(data.keys())
    except Exception:
        return None

    w_key = next((k for k in _WEIGHT_KEY_PREFS if k in keys), None)
    if w_key is None:
        for k in keys:
            try:
                if _looks_like_weights(data[k]):
                    w_key = k
                    break
            except Exception:
                continue
    if w_key is None or not _looks_like_weights(data[w_key]):
        return None
    n_ch = int(data[w_key].shape[0])

    c_key = next((k for k in _CHNAMES_KEY_PREFS if k in keys), None)
    if c_key is None:
        for k in keys:
            try:
                v = data[k]
            except Exception:
                continue
            if (getattr(v, "dtype", None) is not None
                    and v.dtype.kind in "OU"
                    and v.ndim == 1 and v.shape[0] == n_ch):
                c_key = k
                break
    if c_key is None:
        return None

    l_key = next((k for k in _LABELS_KEY_PREFS if k in keys), None)
    return {"w": w_key, "c": c_key, "l": l_key}


def _introspect_analysis_json(path: str):
    """Fallback: pesos guardados dentro de un analysis.json (listas)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    w_key = next((k for k in _WEIGHT_KEY_PREFS
                  if isinstance(data.get(k), list)
                  and len(data[k]) > 0
                  and isinstance(data[k][0], list)), None)
    if w_key is None:
        return None
    try:
        W = np.asarray(data[w_key], dtype=float)
    except Exception:
        return None
    if W.ndim != 2 or not (3 <= W.shape[0] <= 128
                           and 1 <= W.shape[1] <= 64):
        return None
    c_key = next((k for k in _CHNAMES_KEY_PREFS
                  if isinstance(data.get(k), list)
                  and len(data[k]) == W.shape[0]
                  and all(isinstance(s, str) for s in data[k])), None)
    if c_key is None:
        return None
    l_key = next((k for k in _LABELS_KEY_PREFS
                  if isinstance(data.get(k), list)), None)
    return {"w": w_key, "c": c_key, "l": l_key}


def _discover_targets(root: str):
    """Encuentra los archivos con datos de influencia bajo la raiz.

    Devuelve (targets, fuente); cada target es (path, kind, keys) con
    kind in {'npz', 'json'}.  Orden: nombre canonico del pipeline ->
    introspeccion de .npz -> analysis.json.
    """
    strict = sorted(glob.glob(
        os.path.join(root, "**", "channel_influence_data.npz"),
        recursive=True))
    if strict:
        keys = {"w": "influence_weights", "c": "ch_names",
                "l": "latent_dim_labels"}
        return [(p, "npz", keys) for p in strict], "channel_influence_data.npz"

    found = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.npz"),
                              recursive=True)):
        if "common" in os.path.basename(p).lower():
            continue
        keys = _introspect_npz(p)
        if keys:
            found.append((p, "npz", keys))
    if found:
        return found, "introspeccion .npz"

    found = []
    for p in sorted(glob.glob(os.path.join(root, "**", "analysis.json"),
                              recursive=True)):
        keys = _introspect_analysis_json(p)
        if keys:
            found.append((p, "json", keys))
    if found:
        return found, "analysis.json"

    return [], None


def _load_influence(path: str, kind: str, keys: dict):
    """Carga (W, ch_names, old_labels) desde .npz o analysis.json."""
    if kind == "npz":
        data = np.load(path, allow_pickle=True)
        W = np.asarray(data[keys["w"]], dtype=float)
        ch_names = [str(c) for c in data[keys["c"]]]
        l_key = keys.get("l")
        try:
            old_labels = [str(c) for c in data[l_key]] if l_key else None
        except KeyError:
            old_labels = None
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        W = np.asarray(data[keys["w"]], dtype=float)
        ch_names = [str(c) for c in data[keys["c"]]]
        l_key = keys.get("l")
        old_labels = ([str(c) for c in data[l_key]]
                      if l_key and isinstance(data.get(l_key), list)
                      else None)
    if old_labels is None:
        old_labels = ["?"] * (W.shape[1] if W.ndim == 2 else 0)
    return W, ch_names, old_labels


def _print_tree_hint(root: str) -> None:
    """Diagnostico: mostrar que hay en la raiz (para poder reportarlo)."""
    try:
        sub = sorted(os.listdir(root))
    except OSError:
        return
    print("  [i] Contenido de la raiz (para diagnosticar):")
    for s in sub[:12]:
        p = os.path.join(root, s)
        print(f"      - {s}{'/' if os.path.isdir(p) else ''}")
    first_dir = next((os.path.join(root, s) for s in sub
                      if os.path.isdir(os.path.join(root, s))), None)
    if first_dir:
        try:
            inner = sorted(os.listdir(first_dir))
        except OSError:
            return
        print(f"      contenido de {os.path.basename(first_dir)}/:")
        for s in inner[:20]:
            print(f"        - {s}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_root(root: str, args) -> int:
    print("=" * 78)
    print(f"ROOT: {root}")
    print("=" * 78)
    targets, source = _discover_targets(root)
    if not targets:
        print("  [!] No se encontro ningun dato de influencia "
              "(channel_influence_data.npz ni equivalente) debajo de esta "
              "raiz.")
        _print_tree_hint(root)
        return 0
    if source != "channel_influence_data.npz":
        print(f"  [i] Datos encontrados via: {source}")

    n_done = 0
    for data_path, kind, keys in targets:
        job_dir = os.path.dirname(data_path)
        cond = condition_from_dirname(job_dir)
        ss = None
        for part in job_dir.split(os.sep):
            if part.startswith("super_subject"):
                ss = part
                break
        if ss is None:
            ss = _ss_label_from_dirname(job_dir)

        W, ch_names, old_labels = _load_influence(data_path, kind, keys)

        # --- lambdas: mode_map > manual > ninguna -------------------------
        lams = None
        lam_src = "n/d"
        if args.mode_map:
            lams = lambdas_from_mode_map(args.mode_map, cond or "")
            lam_src = (f"mode_map:{os.path.basename(args.mode_map)}"
                       if lams else
                       f"mode_map sin condicion '{cond}' (sin lambda)")
        if lams is None and args.lambdas:
            lams = [float(x) for x in args.lambdas.split(",")]
            lam_src = "manual"

        rel = os.path.relpath(job_dir, root)
        print(f"\n[{ss or '?'}] {rel}  ({os.path.basename(data_path)})")
        print(f"  condicion={cond}  modos={W.shape[1]}  "
              f"canales={W.shape[0]}  etiquetas_previas={old_labels}")
        print(f"  lambda={lams if lams else 'n/d'} (fuente: {lam_src})")

        if args.dry_run:
            msg = f"  [DRY-RUN] generaria: {args.out_name}"
            if args.overwrite:
                msg += "  +  channel_influence.png (heatmap reemplazado)"
            print(msg)
            n_done += 1
            continue

        out_path = os.path.join(job_dir, args.out_name)
        plot_specific_topoplots(
            W, ch_names, out_path,
            lambdas=lams, montage_name=args.montage,
            top_k=args.top_k, dpi=args.dpi,
        )
        msg = f"  [OK] {args.out_name}"
        if args.overwrite:
            hw = os.path.join(job_dir, "channel_influence.png")
            plot_specific_topoplots(
                W, ch_names, hw, lambdas=lams, montage_name=args.montage,
                top_k=args.top_k, dpi=args.dpi,
            )
            msg += "  +  channel_influence.png (heatmap reemplazado)"
        print(msg)
        n_done += 1

    print(f"\n--- RESUMEN ({os.path.basename(os.path.normpath(root))}): "
          f"{n_done} carpeta(s) procesadas ---")
    return n_done


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Regenera los topoplots de influencia de canales sobre "
                    "los modos especificos (D) desde los datos ya guardados. "
                    "Soporta los runs reales (fase latente) y el control "
                    "task-shuffle (channel_influence_taskshuf<seed>).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("roots", nargs="+",
                   help="raiz(es) de resultados; se busca "
                        "**/channel_influence_data.npz. Control: "
                        ".../from..s_.._taskshuf<seed>/"
                        "channel_influence_taskshuf<seed>")
    p.add_argument("--mode-map", default=None,
                   help="mode_map.json (para poner lambda=... en cada panel; "
                        "las claves son las condiciones/celdas nominales)")
    p.add_argument("--lambdas", default=None,
                   help="lambdas manuales separadas por coma (fallback si no "
                        "esta el mode_map), p.ej. 0.998,0.997,0.997,0.997")
    p.add_argument("--out-name", default="channel_influence_topo.png",
                   help="nombre del archivo de salida (en cada carpeta)")
    p.add_argument("--overwrite", action="store_true",
                   help="ademas reemplazar channel_influence.png (el heatmap)")
    p.add_argument("--top-k", type=int, default=None,
                   help="limitar a los primeros K modos (por defecto: todos)")
    p.add_argument("--montage", default="standard_1020",
                   help="montaje para las posiciones 2D")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--dry-run", action="store_true",
                   help="mostrar que se haria sin escribir nada")
    args = p.parse_args(argv)

    total = 0
    for root in args.roots:
        if not os.path.isdir(root):
            print(f"[!] No existe el directorio: {root}")
            return 2
        total += process_root(root, args)

    print(f"\nTOTAL: {total} imagen(es) "
          f"{'(dry-run)' if args.dry_run else 'generadas'}.")
    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
