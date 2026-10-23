#!/usr/bin/env python3
"""Regenera SOLO los topoplots de influencia de canales sobre los modos
especificos (Step D) a partir de los resultados ya guardados.

Problema que resuelve
---------------------
El pipeline guarda en cada carpeta de job (SS x condicion):

* ``channel_influence_data.npz``  — pesos de influencia (n_ch, n_spec),
  ``ch_names`` y ``latent_dim_labels``.  ESTE ARCHIVO ES CORRECTO.
* ``channel_influence.png``       — la figura.  En la version actual del
  servidor es un HEATMAP (canales x modos, estilo
  "Channel influence on condition-specific modes (D)") en lugar de los
  TOPOPLOTS ("cerebritos") que ya usa la imagen de modos comunes
  (``common_channel_influence_topo.png``).

Este script NO re-corre el pipeline: lee el ``.npz`` existente y genera la
figura que falta, con el MISMO estilo que la de modos comunes
(``plot_topomap_grid``: 1 fila, RdBu_r simetrico, colorbar por panel):

* salida por defecto:  ``channel_influence_topo.png``  (no toca el heatmap)
* con ``--overwrite``:  tambien sobreescribe ``channel_influence.png``

Etiquetas: ``SM{j}`` (mas ``(lambda=...)`` si hay lambda disponible via
``--mode-map`` o ``--lambdas``).

Uso
---
    # un experimento (10 carpetas -> 10 imagenes):
    python scripts/regen_specific_influence_topo.py <root_resultados_ec_mu> \
        --mode-map params/mode_map_4_modes_session1_eyesclosed_music_100.0s-200.0s.json

    # los dos experimentos:
    python scripts/regen_specific_influence_topo.py <root_ec_mu> <root_ec_ma> ...

    # sin lambda (si el mode_map ya no existe):
    python scripts/regen_specific_influence_topo.py <root> --lambdas 0.998,0.997,0.997,0.997

    # reemplazar ademas el heatmap:
    python scripts/regen_specific_influence_topo.py <root> --overwrite

    # ver que haria sin escribir nada:
    python scripts/regen_specific_influence_topo.py <root> --dry-run

Dependencias: numpy, matplotlib, mne  (las del repo).  No importa nada del
repo: se puede correr desde cualquier lado.
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


def condition_from_dirname(folder: str) -> str | None:
    """Extrae la condicion del nombre de la carpeta de ventana.

    ``from100.0s_to200.0s_eyesclosed`` -> ``eyesclosed``.
    El pipeline de latentes corre UNA condicion por job => el token
    final es la condicion de ESTE job.
    """
    base = os.path.basename(os.path.normpath(folder))
    m = re.match(r"^from[\d.]+s_to[\d.]+s_(.+)$", base)
    if not m:
        return None
    return m.group(1)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_root(root: str, args) -> int:
    print("=" * 78)
    print(f"ROOT: {root}")
    print("=" * 78)
    pattern = os.path.join(root, "**", "channel_influence_data.npz")
    npz_files = sorted(glob.glob(pattern, recursive=True))
    if not npz_files:
        print("  [!] No se encontro ningun channel_influence_data.npz "
              "debajo de esta raiz.")
        return 0

    n_done = 0
    for npz_path in npz_files:
        job_dir = os.path.dirname(npz_path)
        cond = condition_from_dirname(job_dir)
        ss = None
        for part in job_dir.split(os.sep):
            if part.startswith("super_subject"):
                ss = part
                break

        data = np.load(npz_path, allow_pickle=True)
        W = data["influence_weights"]
        ch_names = [str(c) for c in data["ch_names"]]
        try:
            old_labels = [str(c) for c in data["latent_dim_labels"]]
        except KeyError:
            old_labels = ["?"] * W.shape[1]

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
        print(f"\n[{ss or '?'}] {rel}")
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
                    "los modos especificos (D) desde los .npz ya guardados.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("roots", nargs="+",
                   help="raiz(es) de resultados (p.ej. la carpeta "
                        "test_retest_gedai_super_subject_ec_mu descargada); "
                        "se busca **/channel_influence_data.npz")
    p.add_argument("--mode-map", default=None,
                   help="mode_map.json del experimento (para poner lambda=... "
                        "en cada panel; las claves son las tareas)")
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
