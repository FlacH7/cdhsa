"""
single_subject_eeg.py
=====================
Loader for **single-subject** EEG recordings from the test-retest Gedai
dataset (EEGLAB ``.set``/``.fdt``).

In contrast to :mod:`super_subject_eeg` (which concatenates many subjects
along the time axis into a *super-subject*), this module treats **each
individual subject as its own sample** (S = number of subjects).  The
downstream CD-HSA pipeline is agnostic: it only receives ``X[s][c]``, so
here ``s`` indexes individual subjects directly.

Three helpers are provided:

* :func:`resolve_single_subject_ids` -- resolves the list of individual
  subject indices, either from an explicit ``subject_ids`` list or from
  a contiguous ``range(subject_start_offset, subject_start_offset + n_subjects)``.

* :func:`load_single_subject_eeg` -- thin wrapper around
  :func:`load_test_retest_gedai_eeg_from_ids` that builds the BIDS-style
  subject label (``sub-{subject_id:02d}``) and forwards the crop window.

* :func:`compute_single_subject_channel_intersection` -- loads each
  subject's header (metadata only, no data) and returns the channel
  intersection across the whole subject pool, preserving the order of
  the first subject.  Missing/corrupt subjects are skipped.

Typical workflow (per-subject crop window of 100 s)::

    raw = load_single_subject_eeg(
        subject_id=1,
        session="session1",
        task="eyesclosed",
        t_start=100.0,
        t_stop=200.0,
        preload=True,
        verbose=True,
    )

Known dataset issue
-------------------
``sub-41/ses-session3/{memory,music}`` is corrupt in the GEDAI dataset;
callers should tolerate per-subject load failures (skip with warning).
"""

from __future__ import annotations

import logging
from pathlib import Path

import mne


# ----- REAL-TIME LOG FLUSH (Fix 1) -----
# Ensure print() output appears immediately even when stdout is redirected
# (nohup, pipe, subprocess).  Without this, logs accumulate in Python's
# internal buffer and all flush at once, making the pipeline appear frozen.
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)
logger = logging.getLogger("single_subject_eeg")


# ---------------------------------------------------------------------------
# Subject-ids resolution
# ---------------------------------------------------------------------------

def resolve_single_subject_ids(
    *,
    n_subjects: int = 60,
    subject_start_offset: int = 1,
    subject_ids: list[int] | None = None,
) -> list[int]:
    """
    Resolve the list of individual subject indices to process.

    Parameters
    ----------
    n_subjects : int, default 60
        Number of subjects in the auto-generated contiguous range
        (used only when ``subject_ids`` is not provided).
    subject_start_offset : int, default 1
        Index of the first subject in the dataset (typically 1 for
        ``sub-01``).
    subject_ids : list[int] | None
        Explicit list of subject indices.  When provided, it is
        returned as-is (a copy) and takes precedence over the
        auto-generated contiguous range.

    Returns
    -------
    subject_ids : list[int]
        Ordered list of individual subject indices.

    Raises
    ------
    ValueError
        If ``n_subjects`` is non-positive (auto-resolution only).
    """
    if subject_ids is not None:
        return list(subject_ids)

    if n_subjects <= 0:
        raise ValueError("n_subjects must be > 0")

    return list(range(subject_start_offset,
                      subject_start_offset + n_subjects))


# ---------------------------------------------------------------------------
# Single-subject loader
# ---------------------------------------------------------------------------

def load_single_subject_eeg(
    subject_id: int,
    session: str,
    task: str,
    *,
    db_path: str | Path | None = None,
    t_start: float | None = None,
    t_stop: float | None = None,
    preload: bool = False,
    verbose: bool | str | None = None,
) -> mne.io.Raw:
    """
    Load the EEG recording of a **single** subject for a given
    (session, task) pair.

    Thin wrapper around
    :func:`src.latent_space_extraction.test_retest_gedai_eeg.load_test_retest_gedai_eeg_from_ids`:
    builds the BIDS subject label ``sub-{subject_id:02d}`` and forwards
    every other argument unchanged.

    Parameters
    ----------
    subject_id : int
        1-indexed subject identifier (1 -> ``sub-01``).
    session, task : str
        BIDS-style session and task labels.
    db_path : str | Path | None
        Root of the Gedai-preprocessed dataset.  If ``None``, the
        loader falls back to ``DB_TEST_RETEST_GEDAI_PATH``.
    t_start, t_stop : float | None
        Crop window (seconds).  When ``None``, the full recording is
        used.
    preload : bool, default False
        Forwarded to the underlying loader.
    verbose : bool | str | None
        MNE verbosity level.

    Returns
    -------
    raw : mne.io.Raw
        Raw object for the requested subject.

    Raises
    ------
    FileNotFoundError
        If the resolved ``.set``/``.fdt`` files do not exist.
    ValueError
        If no valid EEG channels are found or the crop window is
        invalid.
    """
    from src.latent_space_extraction.test_retest_gedai_eeg import (
        load_test_retest_gedai_eeg_from_ids,
    )

    subject = f"sub-{subject_id:02d}"

    if verbose:
        print(f"  [SingleSubject] Loading {subject}/{session}/{task} "
              f"(t=[{t_start}, {t_stop}])...")

    raw = load_test_retest_gedai_eeg_from_ids(
        subject=subject,
        session=session,
        task=task,
        db_path=db_path,
        t_start=t_start,
        t_stop=t_stop,
        preload=preload,
        verbose=verbose,
    )

    if verbose:
        print(f"  [SingleSubject] {subject}: {len(raw.ch_names)} channels, "
              f"{raw.times[-1]:.1f} s")

    return raw


# ---------------------------------------------------------------------------
# Channel intersection across the subject pool
# ---------------------------------------------------------------------------

def compute_single_subject_channel_intersection(
    session: str,
    task: str,
    *,
    subject_ids: list[int],
    db_path: str | Path | None = None,
    verbose: bool = False,
) -> list[str]:
    """Compute the **global** channel intersection across ALL individual
    subjects for a given (session, task).

    Analogous to ``compute_channel_intersection`` from
    :mod:`super_subject_eeg`, but without super-subject pools: the
    intersection is computed directly over the flat list of individual
    ``subject_ids``.

    The function loads each subject's ``.set`` header (metadata only,
    no data) to read ``ch_names``, keeping memory usage minimal.
    Missing or corrupt subjects are skipped with a warning.

    Parameters
    ----------
    session, task : str
        BIDS-style labels.
    subject_ids : list[int]
        Individual subject indices to include in the intersection.
    db_path : str | Path | None
        Dataset root.  Falls back to ``DB_TEST_RETEST_GEDAI_PATH``.
    verbose : bool
        Print progress.

    Returns
    -------
    list[str]
        Channel names in the global intersection (order taken from the
        first subject that loads successfully).
    """
    from src.latent_space_extraction.test_retest_gedai_eeg import (
        load_test_retest_gedai_eeg_from_ids,
    )

    if not subject_ids:
        raise ValueError("subject_ids must be a non-empty list.")

    if verbose:
        print(f"  [ChannelIntersection] Computing intersection over "
              f"{len(subject_ids)} subjects ({session}/{task})...")

    # Load each subject's header to read channel names
    all_ch_sets: list[list[str]] = []
    n_loaded = 0
    for subj_idx in subject_ids:
        subject = f"sub-{subj_idx:02d}"
        try:
            # preload=False and no crop -> reads header only (minimal I/O)
            raw_i = load_test_retest_gedai_eeg_from_ids(
                subject=subject,
                session=session,
                task=task,
                db_path=db_path,
                t_start=None,
                t_stop=None,
                preload=False,
                verbose=False,
            )
            all_ch_sets.append(list(raw_i.ch_names))
            n_loaded += 1
            # Free memory immediately
            del raw_i
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(
                "  [ChannelIntersection] Skipping %s: %s", subject, exc,
            )
            continue

    if not all_ch_sets:
        raise FileNotFoundError(
            f"Could not load any subject headers for session={session}, "
            f"task={task}. Cannot compute channel intersection."
        )

    if verbose:
        print(f"  [ChannelIntersection] Loaded headers from {n_loaded}/"
              f"{len(subject_ids)} subjects")
        per_subject_counts = [len(s) for s in all_ch_sets]
        print(f"  [ChannelIntersection] Channels per subject: "
              f"min={min(per_subject_counts)}, max={max(per_subject_counts)}, "
              f"unique sets={len(set(tuple(sorted(s)) for s in all_ch_sets))}")

    # Compute intersection preserving the first subject's order
    intersection = list(all_ch_sets[0])
    for ch_set in all_ch_sets[1:]:
        intersection = [ch for ch in intersection if ch in ch_set]

    if verbose:
        print(f"  [ChannelIntersection] Global intersection: "
              f"{len(intersection)} channels")

    return intersection
