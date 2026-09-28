"""
cdhsa/b_energy.py — Steps B/C of CD-HSA (memory-optimized): Energy and geometry condition tests
================================================================================

Memory-Optimized version of b_energy.py.  Instead of materializing the
block-Hankel matrix H inside ``compute_common_mode_metrics``, we compute
‖H‖_F² and H^T @ W_k block-by-block directly from X, so the large
(p·L × T−L+1) matrix is never allocated.

For each fixed common Hankel mode/block (from A6.W0):

  **Step B** — Energy in the ORIGINAL (unnormalized) Hankel matrix.
    E_{sc,k} = ||H_sc^T W_k||_F²  measures how much variance each common
    direction captures in each recording, using the original scale.

  **Step C** — Geometric alignment with the local reliable Hankel subspace.
    a_{sc,k} = ||U_sc^T W_k||_F² / r_sc  (rank-adjusted) measures how much
    of each common direction is geometrically present, SEPARATE from amplitude.

Within-subject permutation tests with max-F correction determine which
blocks show significant condition effects.

CRITICAL DESIGN PRINCIPLE
--------------------------
A6.W0 is held FIXED. Since W0 was estimated from the pooled subject×condition
projectors WITHOUT using condition labels, relabeling conditions within a
subject does NOT change the pooled W0. This justifies the permutation test.

Dependencies: numpy, cdhsa.permutation_tests
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from src.cdhsa.permutation_tests import within_subject_permutation_rm
except ImportError:
    from permutation_tests import within_subject_permutation_rm


# ============================================================================
# Block-Hankel helpers — compute key quantities WITHOUT materializing H
# ============================================================================

def _block_hankel_fro_sq(
    X: NDArray[np.floating],
    L: int,
    col_mask: NDArray[np.bool_] | None = None,
) -> float:
    """Compute ||build_block_hankel(X, L)||_F^2 without materializing H.

    Uses: ||H||_F^2 = sum_ell ||X[:, L-1-ell : T-ell]||_F^2
    Each block is a VIEW into X.  Uses np.dot for memory efficiency.

    With ``col_mask`` the norm is that of the matrix with the masked
    columns REMOVED (ver ``level2_boundary_mask``): se resta la energía
    de las pocas columnas enmascaradas.

    Parameters
    ----------
    X : array, shape (p, T)
    L : int
    col_mask : ndarray of bool, shape (T - L + 1,), optional

    Returns
    -------
    fro_sq : float
    """
    X = np.asarray(X, dtype=np.float64)
    p, T = X.shape
    K = T - L + 1
    fro_sq = 0.0
    for ell in range(L):
        block = X[:, L-1-ell : T-ell]
        fro_sq += np.dot(block.ravel(), block.ravel())

    if col_mask is not None:
        m = np.asarray(col_mask).astype(bool).ravel()
        if m.shape[0] != K:
            raise ValueError(
                f"col_mask tiene longitud {m.shape[0]} pero K={K}."
            )
        bad = np.flatnonzero(~m)
        if bad.size > 0:
            lo = max(int(bad.min()) - 1, 0)
            hi = min(int(bad.max()) + L, T)
            seg = X[:, lo:hi]
            cn2 = np.einsum("ij,ij->j", seg, seg)
            idx = (bad[:, None] + (L - 1) - np.arange(L)[None, :]) - lo
            fro_sq -= float(cn2[idx].sum())

    return max(fro_sq, 0.0)


def _block_hankel_T_dot_W(
    X: NDArray[np.floating],
    L: int,
    Wk: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute H^T @ Wk where H = build_block_hankel(X, L), WITHOUT materializing H.

    H^T @ Wk = sum_ell block_ell^T @ Wk[ell*p:(ell+1)*p, :]

    Parameters
    ----------
    X : array, shape (p, T)
    L : int
    Wk : array, shape (p*L, dim_k)

    Returns
    -------
    proj : array, shape (K, dim_k)  where K = T - L + 1
    """
    X = np.asarray(X, dtype=np.float64)
    p, T = X.shape
    K = T - L + 1
    dim_k = Wk.shape[1]

    proj = np.zeros((K, dim_k), dtype=np.float64)
    for ell in range(L):
        block = X[:, L-1-ell : T-ell]  # (p, K) — VIEW
        proj += block.T @ Wk[ell*p:(ell+1)*p, :]
    return proj


# ============================================================================
# Helper: compute all energy and alignment metrics per recording and block
# ============================================================================

def compute_common_mode_metrics(
    X: list[list[NDArray[np.floating]]],
    L: int,
    R: dict,
    A6: dict,
    blocks: list[NDArray[np.integer]],
    col_masks: list | None = None,
) -> dict:
    """
    Compute energy and alignment metrics for each recording and common-mode block.

    Parameters
    ----------
    X : list of list of arrays
        X[s][c] shape (p, T) — original (unnormalized) EEG data.
    L : int
        Hankel embedding depth.
    R : dict
        Output of ``cdhsa_A1_A5``. Must contain 'U', 'rank', 'S', 'C', 'p', 'L_used'.
    A6 : dict
        Output of ``cdhsa_A6_common_rank``. Must contain 'W0', 'r0'.
    blocks : list of arrays
        Each element is an array of 1-based indices into A6.W0.
        Example: [np.array([1,2]), np.array([3]), np.array([4,5])].
    col_masks : list of list of arrays or None, optional
        Máscaras de columnas de nivel 2 (ver ``level2_boundary_mask``).
        Default: ``R.get('col_masks')`` — las MISMAS usadas en A1-A5,
        para que la energía del Step B se calcule sobre las mismas
        columnas que definieron los subespacios.

    Returns
    -------
    M : dict with keys:
        blocks : list of arrays (Kb elements)
        energy_abs : ndarray (S, C, Kb)
            ||H_sc^T W_k||_F² for each recording and block (solo columnas
            válidas si hay máscara de fronteras).
        energy_rel : ndarray (S, C, Kb)
            energy_abs / ||H_sc||_F² (norma también enmascarada).
        align_raw : ndarray (S, C, Kb)
            ||U_sc^T W_k||_F² (unadjusted).
        align_adj : ndarray (S, C, Kb)
            ||U_sc^T W_k||_F² / j — normalizada por el TAMAÑO DEL
            BLOQUE COMÚN j (= número de columnas de W_k), la
            definición del paper (a_{sc,k} es la fracción del bloque
            común W_k presente en el subespacio local). Con bloques
            singleton equivale a la alineación de A4.
        local_rank : ndarray (S, C)
        align_norm : str — 'block_size' (documenta la normalización).
        S, C, Kb, d, p : int

    Notes
    -----
    - energy_abs uses the ORIGINAL (unnormalized) Hankel, so it captures
      both geometry AND amplitude. This is the Step B metric.
    - align_adj divides by the common-block size j (paper). v3 dividía
      por el rango local r_sc, lo que reportaba la fracción del
      SUBESPACIO LOCAL ocupada por el bloque — una cantidad distinta
      (difiere en un factor j/r) de la del paper.
    - For a singleton block {j}, energy_abs = ||H^T w_j||² = w_j^T (H H^T) w_j.
    """
    S = R["S"]
    C = R["C"]
    p = R["p"]
    L_used = R["L_used"]
    Kb = len(blocks)
    d = p * L_used

    if col_masks is None:
        col_masks = R.get("col_masks")

    energy_abs = np.zeros((S, C, Kb), dtype=np.float64)
    energy_rel = np.zeros((S, C, Kb), dtype=np.float64)
    align_raw = np.zeros((S, C, Kb), dtype=np.float64)
    align_adj = np.zeros((S, C, Kb), dtype=np.float64)

    for s in range(S):
        for c in range(C):
            Xi = np.asarray(X[s][c], dtype=np.float64)

            mask_sc = None
            if col_masks is not None and col_masks[s][c] is not None:
                mask_sc = np.asarray(col_masks[s][c]).astype(bool).ravel()
                if mask_sc.shape[0] != Xi.shape[1] - L + 1:
                    raise ValueError(
                        f"col_masks[{s}][{c}] incoherente con X "
                        f"({mask_sc.shape[0]} vs {Xi.shape[1] - L + 1})."
                    )

            # ||H||_F^2 without materializing H (masked si procede)
            total_energy = _block_hankel_fro_sq(Xi, L, col_mask=mask_sc)

            Ui = R["U"][s][c]  # (d, r_sc)

            for k, idx in enumerate(blocks):
                idx_0 = np.atleast_1d(np.asarray(idx, dtype=int)) - 1  # 0-based
                Wk = A6["W0"][:, idx_0]  # (d, dim_k)
                j_k = Wk.shape[1]

                # Energy: ||H^T W_k||_F²  (computed block-wise, no H)
                proj = _block_hankel_T_dot_W(Xi, L, Wk)
                if mask_sc is None:
                    energy_abs[s, c, k] = np.dot(proj.ravel(), proj.ravel())
                else:
                    # Solo las columnas válidas contribuyen.
                    row_e = np.einsum("ij,ij->i", proj, proj)
                    energy_abs[s, c, k] = float(row_e[mask_sc].sum())
                energy_rel[s, c, k] = energy_abs[s, c, k] / max(
                    total_energy, 1e-300
                )

                # Alignment: ||U_sc^T W_k||_F²  (uses small U, no change)
                G = Ui.T @ Wk  # (r_sc, dim_k)
                raw = np.dot(G.ravel(), G.ravel())
                align_raw[s, c, k] = raw
                # v4: normalización del paper — por el tamaño del bloque
                # común j, no por el rango local r_sc.
                align_adj[s, c, k] = raw / max(j_k, 1)

    return {
        "blocks": blocks,
        "energy_abs": energy_abs,
        "energy_rel": energy_rel,
        "align_raw": align_raw,
        "align_adj": align_adj,
        "align_norm": "block_size",
        "local_rank": R["rank"].copy(),
        "S": S,
        "C": C,
        "Kb": Kb,
        "d": d,
        "p": p,
    }


def _partial_eta_sq(Y: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Partial eta-squared per block for a repeated-measures design.

    eta_p^2 = SS_condition / (SS_condition + SS_error)

    Parameters
    ----------
    Y : array, shape (S, C, K)

    Returns
    -------
    eta : array, shape (K,)
        Effect size in [0, 1] per block. With max-F permutation
    p-values all below alpha (common with super-subjects, where energy
    estimates are extremely stable), the effect size tells whether the
    significant effect is large or tiny.
    """
    Y = np.asarray(Y, dtype=np.float64)
    S, C, K = Y.shape
    eta = np.zeros(K, dtype=np.float64)
    for k in range(K):
        Yk = Y[:, :, k]
        grand = Yk.mean()
        cond_means = Yk.mean(axis=0)
        subj_means = Yk.mean(axis=1)
        SS_cond = S * np.sum((cond_means - grand) ** 2)
        resid = Yk - subj_means[:, None] - cond_means[None, :] + grand
        SS_err = np.sum(resid ** 2)
        denom = SS_cond + SS_err
        eta[k] = SS_cond / denom if denom > 0 else 0.0
    return eta


def _degenerate_permutation_test(
    K: int,
    reason: str,
) -> dict:
    """Estructura de test degenerado (S < 2: no hay intercambio válida).

    Devuelve el mismo contrato que ``within_subject_permutation_rm``
    pero con p-values NaN y flags documentando por qué no hay inferencia.
    """
    return {
        "F_obs": np.full(K, np.nan),
        "p_uncorrected": np.full(K, np.nan),
        "p_maxF": np.full(K, np.nan),
        "null_F": np.zeros((0, K)),
        "null_maxF": np.zeros(0),
        "n_perm": 0,
        "seed": None,
        "degenerate": True,
        "reason": reason,
    }


# ============================================================================
# Main: Steps B/C
# ============================================================================

def cdhsa_BC_condition_tests(
    X: list[list[NDArray[np.floating]]],
    L: int,
    R: dict,
    A6: dict,
    opts: dict | None = None,
) -> dict:
    """
    Steps B/C of CD-HSA: condition effects on energy and geometry.

    For each fixed common Hankel mode/block:
      B) energy in the ORIGINAL (unnormalized) Hankel matrix.
      C) geometrical alignment with the local reliable Hankel subspace.
    Within-subject permutation tests with max-F correction across blocks.

    Parameters
    ----------
    X : list of list of arrays
        X[s][c] shape (p, T).
    L : int
        Hankel embedding depth.
    R : dict
        Output of ``cdhsa_A1_A5``.
    A6 : dict
        Output of ``cdhsa_A6_common_rank``. Must have r0 >= 1.
    opts : dict, optional
        blocks : list of arrays — indices into A6.W0 (1-based).
            Default: singleton modes [1], [2], ..., [r0].
        energy_metric : str
            'log_absolute' (default), 'absolute', 'relative', 'log_relative'.
        geometry_metric : str
            'adjusted' (default; normalizada por el tamaño de bloque j,
            Def. del paper) or 'raw'.
        n_perm : int (default 5000)
        seed : int (default 1234)
        alpha : float (default 0.05)
        condition_names : list of str — optional names for conditions.
        col_masks : list of list of arrays or None
            Override de las máscaras de columnas de nivel 2. Default:
            las de ``R['col_masks']`` (las usadas en A1-A5), para que
            la energía del Step B se mida sobre las mismas columnas que
            definieron los subespacios.
        rank_outcome : {'selected', 'effective'} (default 'selected')
            Resultado a usar en el test de rango: el rango primario
            ('selected') o el rango efectivo X% ('effective', requiere
            ``cdhsa_A1_A5(..., effective_rank=True)``; Remark 3.6: con
            rangos fijos el test del rango seleccionado es degenerado,
            el del efectivo no).

    Returns
    -------
    BC : dict with keys:
        metrics : dict — output from compute_common_mode_metrics.
        energy_data : ndarray (S, C, Kb) — transformed energy matrix.
        geometry_data : ndarray (S, C, Kb) — transformed geometry matrix.
        energy_test : dict — permutation test results for energy
            (degenerado si S < 2, ver 'skipped_reason').
        geometry_test : dict — permutation test results for geometry.
        rank_test : dict — permutation test results for the chosen rank
            outcome ('selected' | 'effective').
        rank_outcome : str — outcome usado en el test de rango.
        sig_energy_maxF : ndarray (Kb,) bool
        sig_geometry_maxF : ndarray (Kb,) bool
        block_names : list of str
        condition_names : list of str
        skipped_reason : str or None — 'S < 2: ...' cuando los tests
            permutation se omiten (análisis descriptivo por sujeto único).
        summary : dict — compact numerical summary.

    Notes
    -----
    Single subject (S = 1): la permutación within-subject no tiene
    grados de libertad entre sujetos, así que los tests se marcan como
    degenerados (p = NaN, sig = False) y el resultado es DESCRIPTIVO.
    Esto hace ejecutable el pipeline por réplica (Framework 2 vía
    ``run_specific_modes_robustness.py``) y los pipelines single-subject.

    CRITICAL ANALYSIS vs MATLAB
    ---------------------------
    Faithful translation of ``cdhsa_BC_condition_tests.m``.

    Key design decisions:
    1. **A6.W0 is fixed** — it was estimated from label-blind pooled data,
       so condition permutation within subjects doesn't change it.
    2. **Energy uses unnormalized H** — the A1-A5 pipeline normalizes H/||H||
       before SVD, but energy must use the original scale.
    3. **Rank-adjusted alignment** — divides by r_sc to remove the mechanical
       dependence on local rank.
    4. **Rank is also tested** — treated as a single (K=1) repeated-measures
       outcome, since raw alignment depends on rank.
    5. **Block indices are 1-based** in the API (matching MATLAB convention)
       and converted to 0-based internally.

    The MATLAB ``compute_common_mode_metrics`` function body was not provided
    but its interface and output fields are fully determined from the calling
    code in ``cdhsa_BC_condition_tests.m``.
    """
    if opts is None:
        opts = {}
    if A6["r0"] < 1:
        raise ValueError(
            "A6.r0 = 0. Do not run B/C without a supported common subspace."
        )

    # Default blocks: singleton modes
    if "blocks" not in opts or not opts["blocks"]:
        blocks = [np.array([j], dtype=int) for j in range(1, A6["r0"] + 1)]
    else:
        blocks = [
            np.atleast_1d(np.asarray(b, dtype=int)) for b in opts["blocks"]
        ]

    energy_metric = opts.get("energy_metric", "log_absolute")
    geometry_metric = opts.get("geometry_metric", "adjusted")
    n_perm = opts.get("n_perm", 5000)
    seed = opts.get("seed", 1234)
    alpha = opts.get("alpha", 0.05)
    condition_names = opts.get("condition_names", [])
    rank_outcome = opts.get("rank_outcome", "selected")
    if rank_outcome not in ("selected", "effective"):
        raise ValueError(
            f"rank_outcome debe ser 'selected' o 'effective', "
            f"got '{rank_outcome}'"
        )

    S_data = R["S"]

    # ---- Compute raw metrics ----
    M = compute_common_mode_metrics(
        X, L, R, A6, blocks, col_masks=opts.get("col_masks")
    )
    Kb = M["Kb"]

    # ---- Transform energy metric ----
    tiny = np.finfo(np.float64).tiny
    if energy_metric == "log_absolute":
        Yenergy = np.log(np.maximum(M["energy_abs"], tiny))
        energy_label = "log absolute Hankel energy"
    elif energy_metric == "absolute":
        Yenergy = M["energy_abs"].copy()
        energy_label = "absolute Hankel energy"
    elif energy_metric == "relative":
        Yenergy = M["energy_rel"].copy()
        energy_label = "relative Hankel energy"
    elif energy_metric == "log_relative":
        Yenergy = np.log(np.maximum(M["energy_rel"], tiny))
        energy_label = "log relative Hankel energy"
    else:
        raise ValueError(f"Unknown energy_metric: '{energy_metric}'")

    # ---- Transform geometry metric ----
    if geometry_metric == "adjusted":
        Ygeom = M["align_adj"].copy()
        geometry_label = "block-normalized subspace alignment (||U^T W_k||²/j)"
    elif geometry_metric == "raw":
        Ygeom = M["align_raw"].copy()
        geometry_label = "raw subspace alignment"
    else:
        raise ValueError(f"Unknown geometry_metric: '{geometry_metric}'")

    # ---- Permutation tests (S >= 2) o degradación descriptiva (S = 1) ----
    if S_data >= 2:
        skipped_reason = None
        energy_test = within_subject_permutation_rm(
            Yenergy, {"n_perm": n_perm, "seed": seed}
        )
        geometry_test = within_subject_permutation_rm(
            Ygeom, {"n_perm": n_perm, "seed": seed + 1}
        )
    else:
        skipped_reason = (
            "S < 2: no hay variabilidad entre sujetos; los tests "
            "within-subject se omiten (resultado descriptivo)."
        )
        energy_test = _degenerate_permutation_test(Kb, skipped_reason)
        geometry_test = _degenerate_permutation_test(Kb, skipped_reason)

    # Rank as a repeated-measures outcome (single variable, K=1)
    if rank_outcome == "effective":
        if "rank_effective" not in R:
            raise ValueError(
                "rank_outcome='effective' requiere "
                "cdhsa_A1_A5(..., effective_rank=True)."
            )
        rank_matrix = np.asarray(R["rank_effective"], dtype=float)
    else:
        rank_matrix = M["local_rank"].astype(np.float64)

    Yrank = rank_matrix[:, :, np.newaxis]
    if S_data >= 2:
        rank_test = within_subject_permutation_rm(
            Yrank, {"n_perm": n_perm, "seed": seed + 2}
        )
    else:
        rank_test = _degenerate_permutation_test(1, skipped_reason)

    # Remark 3.6: with a fixed-rank configuration the SELECTED ranks are
    # constant and their test is degenerate (contributes no evidence).
    rank_test_degenerate = bool(
        np.all(rank_matrix == rank_matrix.flat[0])
    )

    # Effect sizes (partial eta squared)
    energy_eta = _partial_eta_sq(Yenergy)
    geometry_eta = _partial_eta_sq(Ygeom)

    # ---- Names ----
    if not condition_names:
        condition_names = [f"Condition {c + 1}" for c in range(M["C"])]
    elif len(condition_names) != M["C"]:
        raise ValueError(
            f"condition_names must have {M['C']} entries, got {len(condition_names)}"
        )

    block_names = []
    for k in range(Kb):
        idx = blocks[k]
        if len(idx) == 1:
            block_names.append(f"W{int(idx[0])}")
        else:
            block_names.append(f"W[{','.join(str(int(i)) for i in idx)}]")

    # ---- Significance masks ----
    if S_data >= 2:
        sig_energy = energy_test["p_maxF"] < alpha
        sig_geom = geometry_test["p_maxF"] < alpha
    else:
        sig_energy = np.zeros(Kb, dtype=bool)
        sig_geom = np.zeros(Kb, dtype=bool)

    # ---- Assemble output ----
    BC = {
        "metrics": M,
        "energy_metric": energy_metric,
        "geometry_metric": geometry_metric,
        "energy_label": energy_label,
        "geometry_label": geometry_label,
        "energy_data": Yenergy,
        "geometry_data": Ygeom,
        "energy_test": energy_test,
        "geometry_test": geometry_test,
        "rank_test": rank_test,
        "rank_outcome": rank_outcome,
        "rank_test_degenerate": rank_test_degenerate,
        "skipped_reason": skipped_reason,
        "alpha": alpha,
        "sig_energy_maxF": sig_energy,
        "sig_geometry_maxF": sig_geom,
        "block_names": block_names,
        "condition_names": condition_names,
        "summary": {
            "block": block_names,
            "energy_F": energy_test["F_obs"],
            "energy_p_unc": energy_test["p_uncorrected"],
            "energy_p_maxF": energy_test["p_maxF"],
            "energy_partial_eta_sq": energy_eta,
            "geometry_F": geometry_test["F_obs"],
            "geometry_p_unc": geometry_test["p_uncorrected"],
            "geometry_p_maxF": geometry_test["p_maxF"],
            "geometry_partial_eta_sq": geometry_eta,
            "rank_F": rank_test["F_obs"],
            "rank_p_maxF": rank_test["p_maxF"],
            "rank_degenerate": rank_test_degenerate,
            "rank_outcome": rank_outcome,
        },
    }

    # C=2 effect sizes (requieren S >= 2 para el dz pareado)
    if M["C"] == 2 and S_data >= 2:
        BC["summary"]["energy_difference_C2_minus_C1"] = energy_test[
            "mean_difference_C2_minus_C1"
        ]
        BC["summary"]["energy_cohen_dz"] = energy_test["cohen_dz"]
        BC["summary"]["geometry_difference_C2_minus_C1"] = geometry_test[
            "mean_difference_C2_minus_C1"
        ]
        BC["summary"]["geometry_cohen_dz"] = geometry_test["cohen_dz"]

    return BC
