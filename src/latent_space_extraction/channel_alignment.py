"""
channel_alignment.py
====================
Alineacion de canales para grabaciones de test contra el conjunto de
canales GLOBALES del entrenamiento CD-HSA.

Problema
--------
Los modos CD-HSA (``W0``, ``W_c``) viven en un espacio de dimension
``D = n_channels * depth * L`` fijado por la interseccion GLOBAL de
canales usada al entrenar (p. ej. 53 canales).  Una grabacion de test
de otra sesion puede tener:

* canales extra  -> se descartan (pick del conjunto objetivo),
* canales ausentes -> hay que RECONSTRUIRLOS (interpolacion esferica
  de splines, Perrin et al. 1989) o descartar la grabacion,
* nombres con prefijos distintos ('EEG FP1' vs 'FP1') -> se normalizan
  con las MISMAS reglas del cargador del dataset.

API principal
--------------
* :func:`align_raw_to_channels` -- pasa un ``mne.io.Raw`` (ya cargado y
  recortado) al conjunto objetivo de canales, interpolando los ausentes
  cuando la politica lo permite.
* :func:`read_recording_header` -- lectura SOLO-CABECERA de un ``.set``
  (sin tocar el ``.fdt``): canales, sfreq y duracion.  Es la base de la
  pasada de verificacion PREVIA al procesamiento.
* :func:`audit_session_recordings` -- auditoria completa de una sesion
  (sujetos x tareas) contra uno o mas conjuntos objetivo (uno por modelo
  de entrenamiento), con decision por grabacion: ``ok`` /
  ``interpolate`` / ``skip``.

Notas de diseno
---------------
* La interpolacion solo aplica a canales EEG (los EOG no tienen
  representacion topografica en el montaje y ``interpolate_bads`` los
  rechaza); si falta un EOG la grabacion se salta (motivo reportado).
* La reconstruccion usa la posicion del montaje ``standard_1020`` (la
  misma que asigna el cargador del dataset), NO digitizacion
  individual del sujeto: es la unica informacion posicional disponible
  de forma uniforme.
* ``extra_policy`` es siempre ``drop`` (pick del objetivo): los canales
  que no estaban en el entrenamiento no pueden usarse porque las
  filas de ``W0``/``W_c`` estan fijadas al orden objetivo.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("channel_alignment")

# Canales que el cargador del dataset tipa como EOG (no interpolables).
_EOG_NAMES = {"VEOG", "HEOG", "EOG"}

# Politicas soportadas para canales ausentes.
MISSING_POLICIES = ("interpolate", "skip")


# ---------------------------------------------------------------------------
# Normalizacion de nombres (espejo exacto del cargador Gedai)
# ---------------------------------------------------------------------------

def _clean_channel_name(ch_name: str) -> str:
    """Igual que ``test_retest_gedai_eeg._clean_channel_name``."""
    stripped = str(ch_name).strip()
    for prefix in ("EEG ", "EOG ", "EKG ", "EMG ", "ECG "):
        if stripped.upper().startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    return stripped.strip().upper()


def _channel_whitelist() -> frozenset:
    """Whitelist de canales validos del cargador Gedai (import perezoso)."""
    from src.latent_space_extraction.test_retest_gedai_eeg import (
        _TEST_RETEST_GEDAI_VALID_EEG_CHANNELS,
    )
    return _TEST_RETEST_GEDAI_VALID_EEG_CHANNELS


def cleaned_valid_channels(raw) -> list[str]:
    """Nombres de canales del raw, normalizados y filtrados por la
    whitelist del cargador (en el orden del raw)."""
    whitelist = _channel_whitelist()
    out: list[str] = []
    for ch in raw.ch_names:
        clean = _clean_channel_name(ch)
        if clean in whitelist:
            out.append(clean)
    return out


# ---------------------------------------------------------------------------
# Alineacion de un Raw ya cargado
# ---------------------------------------------------------------------------

def align_raw_to_channels(
    raw,
    target_channels: list[str],
    *,
    missing_policy: str = "interpolate",
    max_missing: int = 4,
) -> tuple:
    """Alinear un ``mne.io.Raw`` al conjunto objetivo del entrenamiento.

    Parameters
    ----------
    raw : mne.io.Raw
        Grabacion ya cargada (y recortada) por el cargador del dataset
        (por tanto con la whitelist aplicada y nombres normalizados).
    target_channels : list[str]
        Canales globales del entrenamiento, EN ORDEN (``global_channels``
        del ``hankel_info.json`` del train).  Define el orden de filas
        de las matrices de nivel 1/2.
    missing_policy : {'interpolate', 'skip'}
        Que hacer con los canales objetivo ausentes: reconstruirlos por
        splines esfericos o descartar la grabacion.
    max_missing : int
        Maximo numero de canales ausentes a interpolar; por encima, la
        grabacion se salta (una reconstruccion tan grande no es
        fiable).

    Returns
    -------
    (raw_aligned, report) : tuple[mne.io.Raw | None, dict]
        ``raw_aligned`` es ``None`` cuando la grabacion debe saltarse.
        ``report`` documenta la accion tomada::

            {
              "action": "aligned" | "skipped",
              "skip_reason": str | None,
              "n_target": int,
              "n_raw_valid": int,        # canales validos ANTES de alinear
              "missing": [...],           # objetivo ausentes
              "interpolated": [...],      # efectivamente reconstruidos
              "extra_dropped": [...],     # presentes pero fuera del objetivo
              "duplicates": [...],        # objetivo duplicado en el raw
            }
    """
    import mne  # noqa: F401  (garantiza el import explicito)

    target = [str(c).strip().upper() for c in target_channels]
    if len(set(target)) != len(target):
        raise ValueError(
            f"El conjunto objetivo tiene duplicados: "
            f"{[c for c in target if target.count(c) > 1]}"
        )

    # Mapa nombre-normalizado -> nombre original del raw (el cargador
    # del dataset ya renombra 'EEG FP1'->'FP1', pero esta funcion
    # tambien acepta raws sin renombrar).
    orig_by_clean: dict[str, str] = {}
    for ch in raw.ch_names:
        clean = _clean_channel_name(ch)
        if clean in _channel_whitelist() and clean not in orig_by_clean:
            orig_by_clean[clean] = ch

    present = list(orig_by_clean.keys())
    report = {
        "action": "skipped",
        "skip_reason": None,
        "n_target": len(target),
        "n_raw_valid": len(present),
        "missing": [],
        "interpolated": [],
        "extra_dropped": [],
        "duplicates": [],
    }

    # Duplicados del objetivo dentro del raw (no se puede desambiguar).
    counts: dict[str, int] = {}
    for ch in raw.ch_names:
        clean = _clean_channel_name(ch)
        if clean in _channel_whitelist():
            counts[clean] = counts.get(clean, 0) + 1
    duplicates = sorted(c for c, n in counts.items() if n > 1)
    report["duplicates"] = duplicates
    if duplicates:
        report["skip_reason"] = f"duplicate_channels:{','.join(duplicates)}"
        return None, report

    missing = [c for c in target if c not in orig_by_clean]
    extras = [c for c in present if c not in set(target)]
    report["missing"] = missing
    report["extra_dropped"] = extras

    if not missing:
        raw = _pick_and_rename(raw, target, orig_by_clean)
        report["action"] = "aligned"
        return raw, report

    if missing_policy not in MISSING_POLICIES:
        report["skip_reason"] = f"unknown_policy:{missing_policy}"
        return None, report

    if missing_policy == "skip":
        report["skip_reason"] = f"missing_channels:{','.join(missing)}"
        return None, report

    # --- politica 'interpolate' ---
    if len(missing) > int(max_missing):
        report["skip_reason"] = (
            f"too_many_missing:{len(missing)}>{max_missing} "
            f"({','.join(missing)})"
        )
        return None, report

    non_eeg = [c for c in missing if c in _EOG_NAMES]
    if non_eeg:
        report["skip_reason"] = (
            f"missing_non_interpolable:{','.join(non_eeg)}"
        )
        return None, report

    try:
        # 1) quedarse solo con los canales objetivo presentes, con sus
        #    nombres normalizados
        raw = _pick_and_rename(raw, [c for c in target
                                      if c in orig_by_clean],
                               orig_by_clean)
        # 2) anadir los ausentes como canales zero (mismo first_samp)
        zero_info = mne.create_info(
            missing, raw.info["sfreq"], ch_types=["eeg"] * len(missing))
        zero = mne.io.RawArray(
            np.zeros((len(missing), raw.n_times), dtype=np.float64),
            zero_info, first_samp=raw.first_samp,
        )
        raw = raw.add_channels([zero], force_update_info=True)
        # 3) posiciones para la interpolacion: el mismo montaje del
        #    cargador (solo quedan canales objetivo en el raw)
        raw.set_montage("standard_1020", match_case=False,
                        on_missing="raise")
        raw.info["bads"] = list(missing)
        raw = raw.interpolate_bads(reset_bads=True)
        # 4) reordenar exactamente al orden objetivo
        raw.pick(target)
    except Exception as exc:  # montage/interpolacion fallan -> skip
        report["skip_reason"] = f"interpolation_failed:{type(exc).__name__}:{exc}"
        return None, report

    report["interpolated"] = list(missing)
    report["action"] = "aligned"
    return raw, report


def _pick_and_rename(raw, target_present: list[str],
                     orig_by_clean: dict[str, str]):
    """Pick de los canales objetivo presentes + nombres normalizados."""
    picks = [orig_by_clean[c] for c in target_present]
    raw = raw.copy().pick(picks)
    renames = {orig_by_clean[c]: c for c in target_present
               if orig_by_clean[c] != c}
    if renames:
        raw.rename_channels(renames)
    return raw


# ---------------------------------------------------------------------------
# Lectura de cabecera SOLO-METADATOS (sin leer el .fdt)
# ---------------------------------------------------------------------------

def resolve_set_path(db_path, subject_id: int, session: str,
                      task: str) -> Path:
    """Ruta ``.set`` esperada (misma convencion del cargador Gedai)."""
    from src.latent_space_extraction.test_retest_gedai_eeg import (
        _resolve_set_path as _repo_resolve,
    )
    subject = f"sub-{int(subject_id):02d}"
    return _repo_resolve(subject, session, task, Path(db_path))


def read_recording_header(
    db_path,
    subject_id: int,
    session: str,
    task: str,
) -> dict:
    """Leer SOLO la cabecera de un ``.set`` (sin cargar el ``.fdt``).

    Returns
    -------
    dict ::
        {
          "exists": bool,
          "error": str | None,
          "channels": [str],      # validos, normalizados (orden del raw)
          "n_raw_channels": int,  # TODOS los canales del fichero
          "sfreq": float | None,
          "duration_s": float | None,
        }

    Notes
    -----
    ``mne.io.read_raw_eeglab(preload=False)`` lee solo el ``.set``
    (MATLAB header); ``n_times`` y ``srate`` vienen de ahi, el ``.fdt``
    no se toca.
    """
    import mne

    set_file = resolve_set_path(db_path, subject_id, session, task)
    out = {
        "exists": False, "error": None, "channels": [],
        "n_raw_channels": 0, "sfreq": None, "duration_s": None,
    }
    if not set_file.exists():
        out["error"] = "file_not_found"
        return out
    fdt = set_file.with_suffix(".fdt")
    if not fdt.exists():
        out["error"] = "fdt_not_found"
        return out

    try:
        mne.set_log_level("ERROR")
        raw = mne.io.read_raw_eeglab(str(set_file), preload=False,
                                     verbose="ERROR")
    except Exception as exc:
        out["error"] = f"read_failed:{type(exc).__name__}:{exc}"
        return out

    out["exists"] = True
    out["n_raw_channels"] = len(raw.ch_names)
    out["channels"] = cleaned_valid_channels(raw)
    out["sfreq"] = float(raw.info["sfreq"])
    out["duration_s"] = float(raw.n_times / raw.info["sfreq"])
    return out


# ---------------------------------------------------------------------------
# Auditoria de una sesion completa (pase previo al procesamiento)
# ---------------------------------------------------------------------------

def _decide(header: dict, target: list[str], *, policy: str,
            max_missing: int, min_duration_s: float) -> dict:
    """Decision de uso para UNA grabacion contra UN conjunto objetivo."""
    d: dict = {
        "usable": False,
        "action": "skip",
        "reason": header.get("error") or "not_read",
        "n_missing": 0,
        "missing": [],
        "duration_s": header.get("duration_s"),
    }
    if not header.get("exists"):
        return d
    if header.get("error"):
        return d

    dur = header.get("duration_s") or 0.0
    if dur < float(min_duration_s):
        d["reason"] = f"too_short:{dur:.1f}s<{min_duration_s}s"
        return d

    channels = header.get("channels") or []
    # Duplicados tras la normalizacion: el cargador no puede desambiguar.
    if len(channels) != len(set(channels)):
        dup = sorted({c for c in channels if channels.count(c) > 1})
        d["reason"] = f"duplicate_channels:{','.join(dup)}"
        return d

    have = set(channels)
    missing = [c for c in target if c not in have]
    d["n_missing"] = len(missing)
    d["missing"] = missing
    if not missing:
        d.update({"usable": True, "action": "ok", "reason": None})
        return d
    if policy == "interpolate" and len(missing) <= int(max_missing) \
            and not any(c in _EOG_NAMES for c in missing):
        d.update({"usable": True, "action": "interpolate",
                  "reason": None})
        return d
    d["reason"] = f"missing:{len(missing)}ch:{','.join(missing[:8])}"
    return d


def audit_session_recordings(
    session: str,
    tasks: list[str],
    subject_ids: list[int],
    *,
    db_path,
    model_targets: dict,
    missing_policy: str = "interpolate",
    max_missing: int = 4,
    min_duration_s: float = 130.0,
    verbose: bool = True,
) -> dict:
    """Auditar todas las grabaciones de una sesion ANTES de procesar.

    Parameters
    ----------
    session : str
        Sesion a auditar (p. ej. ``session2``).
    tasks : list[str]
        Tareas a auditar (union de las tareas de todos los experimentos).
    subject_ids : list[int]
        Sujetos individuales.
    db_path : str | Path
        Raiz del dataset Gedai.
    model_targets : dict[str, list[str]]
        ``{model_key: global_channels}`` de cada modelo de entrenamiento
        (leidos de sus ``hankel_info.json``).
    missing_policy, max_missing, min_duration_s
        Politica de decision (ver :func:`_decide`).

    Returns
    -------
    audit : dict ::
        {
          "session": ..., "tasks": [...], "subjects": [...],
          "policy": {...},
          "models": {model_key: {"n_channels": int}},
          "recordings": [
              {"subject_id": int, "task": str,
               "exists": bool, "error": str|None,
               "n_raw_channels": int, "n_valid_channels": int,
               "sfreq": float|None, "duration_s": float|None,
               "per_model": {model_key: {"usable", "action",
                                          "reason", "n_missing",
                                          "missing": [...]}},
               "usable_any": bool},
              ...
          ],
          "summary": {...}
        }
    """
    t0 = time.time()
    recordings: list[dict] = []
    for subj in subject_ids:
        for task in tasks:
            hdr = read_recording_header(db_path, subj, session, task)
            rec = {
                "subject_id": int(subj),
                "task": task,
                "exists": hdr["exists"],
                "error": hdr["error"],
                "n_raw_channels": hdr["n_raw_channels"],
                "n_valid_channels": len(hdr["channels"]),
                "sfreq": hdr["sfreq"],
                "duration_s": hdr["duration_s"],
                "per_model": {
                    key: _decide(hdr, tgt, policy=missing_policy,
                                 max_missing=max_missing,
                                 min_duration_s=min_duration_s)
                    for key, tgt in model_targets.items()
                },
            }
            rec["usable_any"] = any(
                v["usable"] for v in rec["per_model"].values())
            recordings.append(rec)
            if verbose:
                flag = "OK " if rec["usable_any"] else "BAD"
                print(f"  [AUDIT] sub-{subj:02d}/{task:<12} {flag} "
                      f"ch={rec['n_valid_channels']:>3} "
                      f"t={rec['duration_s'] or 0:>7.1f}s "
                      f"{rec['error'] or ''}", flush=True)

    # --- Resumen ---
    n_total = len(recordings)
    n_exists = sum(1 for r in recordings if r["exists"])
    per_model_summary = {}
    for key in model_targets:
        usable = [r for r in recordings if r["per_model"][key]["usable"]]
        interp = [r for r in recordings
                  if r["per_model"][key]["action"] == "interpolate"]
        per_model_summary[key] = {
            "n_usable": len(usable),
            "n_interpolate": len(interp),
            "n_skip": n_total - len(usable),
            "usable_frac": (len(usable) / n_total) if n_total else 0.0,
        }
    # Sujetos con TODAS sus grabaciones usables (por modelo) y en todos
    subjects_all_ok: list[int] = []
    for subj in subject_ids:
        recs = [r for r in recordings if r["subject_id"] == subj]
        if recs and all(r["usable_any"] for r in recs):
            subjects_all_ok.append(int(subj))

    audit = {
        "session": session,
        "tasks": list(tasks),
        "subjects": [int(s) for s in subject_ids],
        "policy": {
            "missing_policy": missing_policy,
            "max_missing": int(max_missing),
            "min_duration_s": float(min_duration_s),
        },
        "models": {k: {"n_channels": len(v), "channels": list(v)}
                   for k, v in model_targets.items()},
        "recordings": recordings,
        "summary": {
            "n_recordings": n_total,
            "n_exists": n_exists,
            "n_missing_files": n_total - n_exists,
            "per_model": per_model_summary,
            "subjects_all_recordings_usable": subjects_all_ok,
            "elapsed_s": round(time.time() - t0, 1),
        },
    }
    return audit


# ---------------------------------------------------------------------------
# Plot de ausencias (mapa de calor sujetos x canales, por modelo)
# ---------------------------------------------------------------------------

def plot_channel_missingness(audit: dict, model_key: str,
                             out_path) -> Path:
    """Heatmap sujetos x canales del objetivo con las ausencias.

    Una celda vale 1 si al menos una tarea del sujeto carece del canal
    (para el modelo ``model_key``).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = audit["models"][model_key]["channels"]
    subjects = audit["subjects"]
    M = np.zeros((len(subjects), len(channels)))
    ch_index = {c: j for j, c in enumerate(channels)}
    for rec in audit["recordings"]:
        i = subjects.index(rec["subject_id"])
        for c in rec["per_model"][model_key].get("missing", []):
            if c in ch_index:
                M[i, ch_index[c]] = 1.0

    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.22 * len(channels)),
                 max(4.0, 0.22 * len(subjects))),
        constrained_layout=True)
    ax.imshow(M, aspect="auto", cmap="Reds", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_xticks(range(len(channels)), labels=channels,
                  rotation=90, fontsize=6)
    step = max(1, len(subjects) // 30)
    ax.set_yticks(range(0, len(subjects), step),
                  labels=[f"{s:02d}" for s in subjects[::step]],
                  fontsize=6)
    ax.set_xlabel(f"Training channels of model '{model_key}'")
    ax.set_ylabel("Subject")
    n_missing_cells = int(M.sum())
    ax.set_title(
        f"Missing channels per subject (session {audit['session']})\n"
        f"{n_missing_cells} subject-channel gaps (red); "
        f"policy={audit['policy']['missing_policy']}")
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
