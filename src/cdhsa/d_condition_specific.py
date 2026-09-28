"""
cdhsa/d_condition_specific.py - Step D: Condition-specific residual modes
==========================================================================

Step D extracts condition-specific residual Hankel modes after removing
the common subspace W0 (from A6), following a JIVE-inspired logic on
orthonormal bases.

v3 (paper-alignment update). Changes vs v2:

  1. **Cross-condition alignment for C > 2 (bug fix, paper Def. 3.14)**:
     v2 only computed the cross alignment for C=2
     (``c_other = 1 - c``), so for C>2 the "prevalence contrast" was
     just the own alignment (cross = 0). Now, for any C::

         a_cross[s, c] = max_{c' != c} ||U_res[s,c]^T W_{c'}||_F^2 / r_{c'}

     and the prevalence contrast is the true
     Delta(c) = mean_s(a_own - a_cross).

  2. **Adaptive specific rank (paper Def. 3.13)**: with
     ``rank_adaptive=True`` the number of retained modes r_c is selected
     by the same commonality logic as Step A criterion (i): a Haar null
     on the pooled residual matrix B_c (random orthonormal bases of
     matching residual ranks), retaining consecutive modes while
     lambda^(c)_l exceeds the (1-alpha) null quantile, up to the cap
     ``max_specific``. This removes the hard censoring at the cap that
     the fixed rule produces (e.g. all conditions hitting dmax=4).

  3. **LOSO calibrated prevalence contrast (paper Def. 3.15)**: the new
     function ``cdhsa_D_prevalence_loso`` evaluates the contrast with
     leave-one-super-subject-out consensus bases and calibrates it with
     a fully recomputed within-subject label-permutation null, with
     max-statistic correction across conditions.

v4 (audit fix): the Haar null of ``_haar_null_lambda_specific`` now uses
``haar_random_basis`` (QR with Mezzadri 2007 sign correction), and
``lambda_specific`` (singular values of B_c) is now consistent with
Step A's ``lambda_`` (also singular values since v4) — the paper's
λ_j convention.

Dependencies: numpy
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ============================================================================
# Helpers
# ============================================================================

def _orthogonalize_residual(
    U: NDArray[np.floating],
    W0: NDArray[np.floating],
    tol: float = 1e-10,
) -> NDArray[np.floating]:
    """
    Remove the projection of U onto W0 and re-orthogonalize.

    Parameters
    ----------
    U : array, shape (d, r_sc)
    W0 : array, shape (d, r0)

    Returns
    -------
    U_res : array, shape (d, r_res)
        Orthonormal basis for the component of U orthogonal to W0.
        r_res <= r_sc - rank(U^T W0) in exact arithmetic.
    """
    # Remove common component: U_res = (I - W0 W0^T) U
    proj = W0 @ (W0.T @ U)  # (d, r_sc)
    U_res = U - proj

    # Re-orthogonalize via thin SVD (numerical stability)
    U_res, s_res, _ = np.linalg.svd(U_res, full_matrices=False)

    # Keep columns with non-negligible singular values
    r_res = int(np.sum(s_res > tol))
    if r_res == 0:
        return np.zeros((U.shape[0], 0), dtype=np.float64)
    return U_res[:, :r_res]


def _select_residual_rank(
    U_res: NDArray[np.floating],
    method: str,
    threshold: float,
    fixed_rank: int,
) -> int:
    """
    Determine the residual rank of a projected basis (D1 rule).

    Parameters
    ----------
    U_res : array, shape (d, k) — already orthogonalized vs W0.
    method : {'local_gap', 'local_threshold', 'fixed'}
    threshold : float — threshold for 'local_threshold'.
    fixed_rank : int — rank for 'fixed'.
    """
    if method == "fixed":
        return int(min(fixed_rank, U_res.shape[1]))
    if method == "local_threshold":
        if U_res.shape[1] == 0:
            return 0
        _, sv, _ = np.linalg.svd(U_res, full_matrices=False)
        return int(np.sum(sv > threshold * sv[0]))
    if method == "local_gap":
        if U_res.shape[1] <= 1:
            return int(U_res.shape[1])
        _, sv, _ = np.linalg.svd(U_res, full_matrices=False)
        ratios = sv[:-1] / np.maximum(sv[1:], 1e-15)
        gaps = np.where(ratios > 2.0)[0]
        if len(gaps) > 0 and gaps[0] > 0:
            return int(gaps[0] + 1)
        return int(np.sum(sv > 0.05 * sv[0]))
    raise ValueError(f"Unknown residual_rank_method: '{method}'")


def _pool_specific_bases(
    bases: list[NDArray[np.floating]],
    max_specific: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Pool orthonormal residual bases and SVD -> condition modes.

    B_c = [U_1 ... U_n] / sqrt(n);  W_c, lambda_c = left SVD of B_c.

    Returns (W, lambda) with q = min(max_specific, d, total columns)
    columns. If no basis has positive rank, returns empty arrays.
    """
    valid = [B for B in bases if B.shape[1] > 0]
    if not valid:
        return (
            np.zeros((bases[0].shape[0] if bases else 0, 0)),
            np.array([]),
        )
    Ucat = np.concatenate(valid, axis=1)
    n_sub = len(valid)
    B_c = Ucat / np.sqrt(n_sub)
    q = int(min(max_specific, B_c.shape[0], B_c.shape[1]))
    W, sv, _ = np.linalg.svd(B_c, full_matrices=False)
    return W[:, :q], sv[:q]


def _haar_null_lambda_specific(
    rank_list: list[int],
    d: int,
    n_null: int,
    qmax: int,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """
    Haar null for the pooled residual singular values (Def. 3.13).

    Each draw replaces every residual basis by a random orthonormal
    basis of the SAME rank, pools them exactly like B_c, and computes the
    top-qmax singular values. (v4: muestreo Haar corregido con
    ``haar_random_basis`` — signos QR, Mezzadri 2007.)

    Returns
    -------
    lambda_null : array, shape (n_null, qmax)
    """
    from src.cdhsa.a6_common_rank import haar_random_basis

    lambda_null = np.full((n_null, qmax), np.nan)
    for b in range(n_null):
        cols = []
        for r in rank_list:
            if r <= 0:
                continue
            cols.append(haar_random_basis(rng, d, r))
        if not cols:
            continue
        B = np.concatenate(cols, axis=1) / np.sqrt(len(cols))
        _, sv, _ = np.linalg.svd(B, full_matrices=False)
        q = min(qmax, sv.shape[0])
        lambda_null[b, :q] = sv[:q]
    return lambda_null


def _select_specific_rank_adaptive(
    lam_obs: NDArray[np.floating],
    null_q: NDArray[np.floating],
    cap: int,
) -> int:
    """
    Consecutive Haar-pass rank selection for condition-specific modes.

    Retains mode l while lam_obs[l-1] > null_q[l-1], stopping at the
    first failure, capped at ``cap``.
    """
    r = 0
    for l in range(min(len(lam_obs), len(null_q), cap)):
        if np.isfinite(null_q[l]) and lam_obs[l] > null_q[l]:
            r = l + 1
        else:
            break
    return int(r)


# ============================================================================
# Main: Step D
# ============================================================================

def cdhsa_D_condition_specific_modes(
    X: list[list[NDArray[np.floating]]],
    L: int,
    R: dict,
    A6: dict,
    opts: dict | None = None,
) -> dict:
    """
    Step D: Extract condition-specific residual Hankel modes.

    After the common subspace W0 has been removed, this function:

      1. (D1) Computes residual Hankel bases for each recording.
      2. (D2) Pools residuals within each condition to find
         condition-specific directions (JIVE-like individual structure).
      3. (D3) Quantifies own- and cross-condition alignment of each
         recording with the condition-specific modes (any C).
      4. (D4) Reports the prevalence contrast
         Delta(c) = mean_s(a_own - a_cross).

    Parameters
    ----------
    X : list of list of arrays
        X[s][c] shape (p, T). NOT used internally (kept for API
        compatibility); the step operates on R["U"] and A6["W0"].
    L : int
        Hankel embedding depth (accepted for API compatibility, not used).
    R : dict
        Output of cdhsa_A1_A5. Must contain 'U', 'S', 'C', 'p', 'rank'.
    A6 : dict
        Output of cdhsa_A6_common_rank. Must contain 'W0', 'r0'.
    opts : dict, optional
        max_specific : int (default 10)
            Maximum number of condition-specific modes per condition.
        residual_rank_method : str (default 'local_gap')
            'local_gap' | 'local_threshold' | 'fixed'.
        residual_rank_threshold : float (default 0.1)
        fixed_residual_rank : int (default 5)
        prevalence_quantile : float (default 0.10)
        rank_adaptive : bool (default False)
            If True, r_c is selected by the Haar commonality test of
            Def. 3.13 (consecutive pass, capped at max_specific) instead
            of the legacy fixed cap.
        n_null_specific : int (default 100)
            Haar null draws for the adaptive rule.
        alpha_specific : float (default 0.05)
            Significance level of the Haar test for r_c.
        seed : int (default 1234)

    Returns
    -------
    D : dict with keys:
        W_specific : list of arrays — W_specific[c] shape (d, r_c).
        lambda_specific : list of arrays — singular values per condition.
        r_specific : ndarray (C,)
        U_residual : list of list of arrays
        residual_rank : ndarray (S, C)
        alignment_specific : ndarray (S, C) — own-condition alignment (23).
        alignment_cross : ndarray (S, C) — max over c'!=c (24).
        alignment_cross_argmax : ndarray (S, C) int — winning condition.
        prevalence_contrast : ndarray (C,) — Delta(c) of (25).
        mean_own_alignment, mean_cross_alignment : ndarray (C,)
        prevalence_own : ndarray (C,)
        rank_adaptive : bool
        S, C, p, d, r0
    """
    if opts is None:
        opts = {}
    max_specific = opts.get("max_specific", 10)
    res_rank_method = opts.get("residual_rank_method", "local_gap")
    res_rank_threshold = opts.get("residual_rank_threshold", 0.1)
    fixed_res_rank = opts.get("fixed_residual_rank", 5)
    prev_q = opts.get("prevalence_quantile", 0.10)
    rank_adaptive = bool(opts.get("rank_adaptive", False))
    n_null_specific = int(opts.get("n_null_specific", 100))
    alpha_specific = float(opts.get("alpha_specific", 0.05))
    seed = int(opts.get("seed", 1234))

    S = R["S"]
    C = R["C"]
    p = R["p"]
    d = R["d"]
    r0 = A6["r0"]

    if r0 < 1:
        raise ValueError(
            "Cannot compute condition-specific modes when A6.r0 = 0."
        )

    W0 = A6["W0"]  # (d, r0)

    # ---- Step D1: Compute residual bases ----
    U_residual: list[list[NDArray]] = [[None] * C for _ in range(S)]
    residual_ranks = np.zeros((S, C), dtype=int)

    for s in range(S):
        for c in range(C):
            Ui = R["U"][s][c]  # (d, r_sc)
            U_res = _orthogonalize_residual(Ui, W0)
            r_res = _select_residual_rank(
                U_res, res_rank_method, res_rank_threshold, fixed_res_rank
            )
            r_res = min(r_res, U_res.shape[1])
            U_residual[s][c] = U_res[:, :r_res]
            residual_ranks[s, c] = r_res

    # ---- Step D2: Pool residuals within condition, SVD -> W_c ----
    rng = np.random.default_rng(seed)
    W_specific: list[NDArray] = []
    lambda_specific: list[NDArray] = []
    null_lambda_q: list[NDArray] = []
    r_specific = np.zeros(C, dtype=int)

    for c in range(C):
        bases_c = [U_residual[s][c] for s in range(S)]
        Wc, sv_c = _pool_specific_bases(bases_c, max_specific)
        lambda_specific.append(sv_c)

        if rank_adaptive and sv_c.shape[0] > 0:
            rank_list = [int(U_residual[s][c].shape[1]) for s in range(S)]
            lam_null = _haar_null_lambda_specific(
                rank_list, d, n_null_specific, int(min(max_specific, sv_c.shape[0])), rng
            )
            null_q = np.nanquantile(lam_null, 1.0 - alpha_specific, axis=0)
            null_lambda_q.append(null_q)
            rc = _select_specific_rank_adaptive(sv_c, null_q, max_specific)
        else:
            null_lambda_q.append(np.array([]))
            rc = int(min(max_specific, sv_c.shape[0]))

        r_specific[c] = rc
        W_specific.append(Wc[:, :rc] if rc > 0 else Wc[:, :0])

    # ---- Step D3: own and cross alignment (any C, Def. 3.14) ----
    alignment_specific = np.zeros((S, C), dtype=np.float64)
    alignment_cross = np.zeros((S, C), dtype=np.float64)
    alignment_cross_argmax = np.full((S, C), -1, dtype=int)

    for s in range(S):
        for c in range(C):
            Ures = U_residual[s][c]
            if Ures.shape[1] == 0 or W_specific[c].shape[1] == 0:
                alignment_specific[s, c] = 0.0
            else:
                G = Ures.T @ W_specific[c]
                alignment_specific[s, c] = (
                    np.linalg.norm(G, "fro") ** 2
                    / max(W_specific[c].shape[1], 1)
                )

            # Cross-condition: strongest competing condition (24)
            best_val = 0.0
            best_c = -1
            for cp in range(C):
                if cp == c or W_specific[cp].shape[1] == 0:
                    continue
                if Ures.shape[1] == 0:
                    continue
                G_cross = Ures.T @ W_specific[cp]
                val = (
                    np.linalg.norm(G_cross, "fro") ** 2
                    / max(W_specific[cp].shape[1], 1)
                )
                if val > best_val:
                    best_val = val
                    best_c = cp
            alignment_cross[s, c] = best_val
            alignment_cross_argmax[s, c] = best_c

    # ---- Step D4: Prevalence contrast (25) ----
    prevalence_own = np.array([
        np.quantile(alignment_specific[:, c], prev_q) for c in range(C)
    ])
    mean_own = np.mean(alignment_specific, axis=0)
    mean_cross = np.mean(alignment_cross, axis=0)
    prevalence_contrast = mean_own - mean_cross

    return {
        "W_specific": W_specific,
        "lambda_specific": lambda_specific,
        "r_specific": r_specific,
        "U_residual": U_residual,
        "residual_rank": residual_ranks,
        "alignment_specific": alignment_specific,
        "alignment_cross": alignment_cross,
        "alignment_cross_argmax": alignment_cross_argmax,
        "mean_own_alignment": mean_own,
        "mean_cross_alignment": mean_cross,
        "prevalence_own": prevalence_own,
        "prevalence_contrast": prevalence_contrast,
        "rank_adaptive": rank_adaptive,
        "alpha_specific": alpha_specific,
        "n_null_specific": n_null_specific,
        "max_specific": max_specific,
        "residual_rank_method": res_rank_method,
        "S": S,
        "C": C,
        "p": p,
        "d": d,
        "r0": r0,
    }


# ============================================================================
# Step D extension: LOSO-calibrated prevalence contrast (paper Def. 3.15)
# ============================================================================

def _loso_consensus_bases(
    U_residual: list[list[NDArray]],
    s_hold: int,
    *,
    rank_adaptive: bool,
    max_specific: int,
    alpha_specific: float,
    n_null_specific: int,
    d: int,
    rng: np.random.Generator,
    fixed_ranks: NDArray[np.integer] | None = None,
) -> tuple[list[NDArray], list[int]]:
    """
    Leave-one-subject-out consensus specific bases (Def. 3.15, first half).

    For every condition c, recompute Steps D2-D3 without subject
    ``s_hold``. If ``fixed_ranks`` is provided (shape (C,)), the rank of
    each condition is forced to that value (used inside the permutation
    null for speed); otherwise the rank follows the same rule as the
    observed Step D (adaptive Haar test or legacy cap).
    """
    S = len(U_residual)
    C = len(U_residual[0])
    W_list: list[NDArray] = []
    r_list: list[int] = []

    for c in range(C):
        bases_c = [
            U_residual[sp][c] for sp in range(S) if sp != s_hold
        ]
        Wc, sv_c = _pool_specific_bases(bases_c, max_specific)
        lambda_specific = sv_c

        if fixed_ranks is not None:
            rc = int(min(fixed_ranks[c], sv_c.shape[0]))
        elif rank_adaptive and sv_c.shape[0] > 0:
            rank_list = [int(B.shape[1]) for B in bases_c]
            lam_null = _haar_null_lambda_specific(
                rank_list, d, n_null_specific,
                int(min(max_specific, sv_c.shape[0])), rng,
            )
            null_q = np.nanquantile(lam_null, 1.0 - alpha_specific, axis=0)
            rc = _select_specific_rank_adaptive(sv_c, null_q, max_specific)
        else:
            rc = int(min(max_specific, sv_c.shape[0]))

        r_list.append(rc)
        W_list.append(Wc[:, :rc] if rc > 0 else Wc[:, :0])

    return W_list, r_list


def _loso_scores(
    U_residual: list[list[NDArray]],
    W_list_by_subject: list[list[NDArray]],
    r_list_by_subject: list[list[int]],
) -> tuple[NDArray, NDArray]:
    """
    Own/cross LOSO alignment scores (26) for every (s, c).
    """
    S = len(U_residual)
    C = len(U_residual[0])
    own = np.full((S, C), np.nan)
    cross = np.full((S, C), np.nan)

    for s in range(S):
        W_s = W_list_by_subject[s]
        r_s = r_list_by_subject[s]
        for c in range(C):
            Ures = U_residual[s][c]
            if Ures.shape[1] == 0 or W_s[c].shape[1] == 0:
                own[s, c] = 0.0
            else:
                G = Ures.T @ W_s[c]
                own[s, c] = (
                    np.linalg.norm(G, "fro") ** 2 / max(W_s[c].shape[1], 1)
                )
            best = 0.0
            for cp in range(C):
                if cp == c or W_s[cp].shape[1] == 0:
                    continue
                if Ures.shape[1] == 0:
                    continue
                Gc = Ures.T @ W_s[cp]
                val = (
                    np.linalg.norm(Gc, "fro") ** 2
                    / max(W_s[cp].shape[1], 1)
                )
                best = max(best, val)
            cross[s, c] = best

    return own, cross


def cdhsa_D_prevalence_loso(
    D: dict,
    opts: dict | None = None,
) -> dict:
    """
    LOSO-calibrated prevalence contrast (paper Definition 3.15).

    Evaluates the condition-specific structure with leave-one-subject-out
    consensus bases (removing the self-inclusion asymmetry at the root)
    and calibrates the contrast against a within-subject label-permutation
    null in which Steps D2-D3 are fully recomputed, with max-statistic
    correction across conditions.

    Parameters
    ----------
    D : dict
        Output of ``cdhsa_D_condition_specific_modes``. Must contain
        'U_residual', 'S', 'C', 'd', 'rank_adaptive', 'max_specific'.
    opts : dict, optional
        n_perm : int (default 1000)
        seed : int (default 777)
        n_null_specific : int (default None -> reuse D's value)
            Haar draws when the observed D used the adaptive rank rule.
        fixed_ranks_in_null : bool (default True)
            If True, the permutation null re-estimates the LOSO consensus
            bases but keeps each (s, c) fold's rank fixed at its observed
            value (much faster; documented approximation). If False, the
            rank rule is re-applied inside every permutation (expensive).

    Returns
    -------
    LOSO : dict with keys:
        alignment_own_loso : ndarray (S, C) — a~own of (26).
        alignment_cross_loso : ndarray (S, C) — a~cross of (26).
        delta_loso : ndarray (C,) — observed contrast (26).
        p_loso : ndarray (C,) — max-statistic corrected permutation p.
        null_delta : ndarray (n_perm, C)
        null_max_delta : ndarray (n_perm,)
        sig_loso : ndarray (C,) bool — p < alpha
        n_perm, seed, alpha
    """
    if opts is None:
        opts = {}
    n_perm = int(opts.get("n_perm", 1000))
    seed = int(opts.get("seed", 777))
    alpha = float(opts.get("alpha", 0.05))
    n_null_specific = opts.get("n_null_specific", None)
    fixed_ranks_in_null = bool(opts.get("fixed_ranks_in_null", True))

    U_residual = D["U_residual"]
    S = D["S"]
    C = D["C"]
    d = D["d"]
    rank_adaptive = bool(D.get("rank_adaptive", False))
    max_specific = int(D.get("max_specific", 10))
    alpha_specific = float(D.get("alpha_specific", 0.05))
    if n_null_specific is None:
        n_null_specific = int(D.get("n_null_specific", 100))

    rng = np.random.default_rng(seed)

    # ---- Observed LOSO scores ----
    W_by_subj: list[list[NDArray]] = []
    r_by_subj: list[list[int]] = []
    for s in range(S):
        W_s, r_s = _loso_consensus_bases(
            U_residual, s,
            rank_adaptive=rank_adaptive,
            max_specific=max_specific,
            alpha_specific=alpha_specific,
            n_null_specific=n_null_specific,
            d=d,
            rng=rng,
        )
        W_by_subj.append(W_s)
        r_by_subj.append(r_s)

    own_obs, cross_obs = _loso_scores(U_residual, W_by_subj, r_by_subj)
    with np.errstate(invalid="ignore"):
        delta_obs = np.nanmean(own_obs - cross_obs, axis=0)

    # Fixed per-fold ranks for the permutation null (documented approx.)
    fixed_ranks = (
        np.array(r_by_subj, dtype=int) if fixed_ranks_in_null else None
    )

    # ---- Permutation null: recompute D2-D3 + LOSO per permutation ----
    null_delta = np.full((n_perm, C), np.nan)
    null_max = np.full(n_perm, np.nan)

    for b in range(n_perm):
        # Within-subject permutation of condition labels
        U_perm: list[list[NDArray]] = [[None] * C for _ in range(S)]
        for s in range(S):
            g = rng.permutation(C)
            for c in range(C):
                U_perm[s][c] = U_residual[s][g[c]]

        W_p: list[list[NDArray]] = []
        r_p: list[list[int]] = []
        for s in range(S):
            if fixed_ranks is not None:
                W_s, r_s = _loso_consensus_bases(
                    U_perm, s,
                    rank_adaptive=False,
                    max_specific=max_specific,
                    alpha_specific=alpha_specific,
                    n_null_specific=n_null_specific,
                    d=d,
                    rng=rng,
                    fixed_ranks=fixed_ranks[s],
                )
            else:
                W_s, r_s = _loso_consensus_bases(
                    U_perm, s,
                    rank_adaptive=rank_adaptive,
                    max_specific=max_specific,
                    alpha_specific=alpha_specific,
                    n_null_specific=n_null_specific,
                    d=d,
                    rng=rng,
                )
            W_p.append(W_s)
            r_p.append(r_s)

        own_b, cross_b = _loso_scores(U_perm, W_p, r_p)
        with np.errstate(invalid="ignore"):
            delta_b = np.nanmean(own_b - cross_b, axis=0)
        null_delta[b, :] = delta_b
        null_max[b] = np.nanmax(delta_b)

    # ---- p-values with max-statistic correction across conditions ----
    p_loso = np.array([
        (1.0 + np.nansum(null_max >= delta_obs[c])) / (n_perm + 1.0)
        for c in range(C)
    ])

    return {
        "alignment_own_loso": own_obs,
        "alignment_cross_loso": cross_obs,
        "delta_loso": delta_obs,
        "p_loso": p_loso,
        "sig_loso": p_loso < alpha,
        "null_delta": null_delta,
        "null_max_delta": null_max,
        "n_perm": n_perm,
        "seed": seed,
        "alpha": alpha,
        "fixed_ranks_in_null": fixed_ranks_in_null,
    }
