"""
null_distributions.py - Null distributions for CD-HSA common rank
================================================================

Two null models for calibrating the common rank r0:

1. **Random subspace null** (from MATLAB): replaces each U_sc with a
   Haar-random orthonormal basis of the same ambient dimension d and
   local rank r_sc. Preserves the rank structure but NOT the Hankel
   temporal correlation.

2. **Hankel-preserving null** (recommended by tutor, NOT in MATLAB code):
   applies a random channel rotation Q_sc ∈ O(p), lifted to I_L ⊗ Q_sc,
   preserving temporal autocorrelation. Strictly more conservative.

Dependencies: numpy, cdhsa.a6_common_rank
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator

from src.cdhsa.a6_common_rank import (
    common_basis_from_U,
    crossvalidate_common_rank,
    haar_random_basis,
)


# ============================================================================
# 1. Random subspace null (faithful MATLAB translation)
# ============================================================================

def random_subspace_null(
    U_cell: list[list[NDArray[np.floating]]],
    r_values: NDArray[np.integer],
    opts: dict | None = None,
) -> dict:
    """
    Haar-random null preserving every local rank r_sc.

    For every null replicate, each observed local basis U_sc is replaced
    by an independent random orthonormal basis having the SAME ambient
    dimension d and the SAME local rank r_sc.

    Parameters
    ----------
    U_cell : list of list of arrays
        U_cell[s][c] shape (d, r_sc) — observed local bases.
    r_values : array of int
        Candidate rank values.
    opts : dict, optional
        n_null (int, default 100): number of null replicates.
        alpha (float, default 0.05): quantile level for threshold.
        n_folds (int, default 5): folds for CV within null.
        seed (int, default 11): RNG seed.

    Notes
    -----
    With a single subject (S < 2) the CV part is skipped and
    ``cv_min``/``cv_min_q`` are filled with NaN (subject-level CV is
    undefined); only the commonality spectrum is simulated.

    Returns
    -------
    null : dict with keys:
        lambda : ndarray (n_null, qmax) — null commonality values λ_j.
        cv_min : ndarray (n_null, len(r_values)) — null CV_min values
            (NaN if S < 2).
        lambda_q : ndarray (qmax,) — (1-alpha) quantile per position.
        cv_min_q : ndarray (len(r_values),) — (1-alpha) quantile
            (NaN if S < 2).
        alpha, n_null

    CRITICAL ANALYSIS vs MATLAB
    ---------------------------
    Faithful translation of ``random_subspace_null.m`` with two fixes:

    1. **Haar sampling (v4)**: ``np.linalg.qr`` without sign correction
       does NOT sample uniformly from the Stiefel manifold (Mezzadri
       2007). We now use ``haar_random_basis`` (Q * sign(diag(R))).

    **Known limitation (tutor's criticism):**
    This null generates completely random orthonormal bases. In a Hankel
    matrix of rank r_sc, the columns of U_sc have intrinsic temporal
    smoothness (they are Hankel singular vectors). A random basis of the
    same rank has NO temporal structure, so the resulting pooled B matrix
    has systematically lower singular values than expected under any
    plausible null. This makes the null commonality spectrum too low,
    leading to ANTI-CONSERVATIVE inference (too many components declared
    significant).

    Computational note: runtime scales ~linearly with n_null. For
    development use 50-100; for final inference use 1000+.
    """
    if opts is None:
        opts = {}
    n_null = opts.get("n_null", 100)
    alpha = opts.get("alpha", 0.05)
    n_folds = opts.get("n_folds", 5)
    seed = opts.get("seed", 11)

    S = len(U_cell)
    C = len(U_cell[0])
    d = U_cell[0][0].shape[0]
    r_values = np.atleast_1d(np.asarray(r_values, dtype=int))
    qmax = int(np.max(r_values))

    lambda_null = np.full((n_null, qmax), np.nan)
    cvmin_null = np.full((n_null, len(r_values)), np.nan)

    # Fixed fold assignment across null replicates (reduces MC variance)
    cv_opts = {"n_folds": n_folds, "seed": seed + 100000}

    rng = np.random.default_rng(seed)

    for b in range(n_null):
        # Generate random orthonormal bases with same rank
        # (v4: Haar sampling con corrección de signos, Mezzadri 2007)
        U0: list[list[NDArray]] = [[None] * C for _ in range(S)]
        for s in range(S):
            for c in range(C):
                rsc = U_cell[s][c].shape[1]
                U0[s][c] = haar_random_basis(rng, d, rsc)

        # Null commonality spectrum
        _, lam = common_basis_from_U(U0, qmax)
        lambda_null[b, : len(lam)] = lam

        # Null CV (indefinido con S < 2)
        if S >= 2:
            cvb = crossvalidate_common_rank(U0, r_values, cv_opts)
            cvmin_null[b, :] = cvb["min_condition"]

    return {
        "lambda": lambda_null,
        "cv_min": cvmin_null,
        "lambda_q": np.nanquantile(lambda_null, 1 - alpha, axis=0),
        "cv_min_q": np.nanquantile(cvmin_null, 1 - alpha, axis=0),
        "alpha": alpha,
        "n_null": n_null,
    }


# ============================================================================
# 2. Hankel-preserving null (tutor's recommended improvement)
# ============================================================================

def _make_rotated_block_hankel_linop(
    X: NDArray[np.floating],
    L: int,
    Q: NDArray[np.floating],
    row_block: int,
    col_mask: NDArray[np.bool_] | None = None,
) -> ScipyLinearOperator:
    """
    LinearOperator for build_block_hankel(R(X), L) without materializing.

    R is a row rotation of X applied BEFORE the second-level embedding:

      - row_block > 0 (paper-faithful, I_{L1} ⊗ Q in channel-major
        layout): X rows are grouped per channel in blocks of
        ``row_block`` consecutive rows; Q ∈ O(p / row_block) mixes
        CHANNELS while leaving the lag structure untouched.
      - row_block == 0 (legacy): Q ∈ O(p) densely mixes all X rows.

    ``col_mask`` (ver ``level2_boundary_mask``) enmascara las columnas de
    nivel 2 igual que en ``make_block_hankel_linop``: las columnas que
    cruzan fronteras de concatenación se tratan como cero, de modo que
    el nulo usa exactamente la misma estructura de columnas que los
    datos observados.

    The key identity (with H2_X = build_block_hankel(X, L))::

        build_block_hankel(R X, L) = (I_L ⊗ R) H2_X

    so matvec applies R to each of the L row-blocks of length p, and
    rmatvec applies R^T first and then delegates to H2_X^T.
    """
    from src.cdhsa.a_common_subspace import make_block_hankel_linop

    X = np.asarray(X, dtype=np.float64)
    p, T = X.shape
    if T < L:
        raise ValueError(f"T ({T}) debe ser >= L ({L}).")
    K = T - L + 1
    d = p * L

    base = make_block_hankel_linop(X, L, col_mask=col_mask)

    if row_block > 0:
        if p % row_block != 0:
            raise ValueError(
                f"row_block={row_block} no divide p={p} (filas de X)."
            )
        n_ch = p // row_block
        if Q.shape != (n_ch, n_ch):
            raise ValueError(
                f"Q debe ser ({n_ch}, {n_ch}) para row_block={row_block}, "
                f"got {Q.shape}."
            )

        def _apply_R(w):
            # w: (p,) o (p, nvec); rows agrupadas por canal (row_block filas)
            if w.ndim == 1:
                return (Q @ w.reshape(n_ch, row_block)).ravel()
            W3 = w.reshape(n_ch, row_block, w.shape[1])
            return np.tensordot(Q, W3, axes=([1], [0])).reshape(p, -1)

        def _apply_Rt(w):
            if w.ndim == 1:
                return (Q.T @ w.reshape(n_ch, row_block)).ravel()
            W3 = w.reshape(n_ch, row_block, w.shape[1])
            return np.tensordot(Q.T, W3, axes=([1], [0])).reshape(p, -1)

    else:
        if Q.shape != (p, p):
            raise ValueError(
                f"Q debe ser ({p}, {p}) para row_block=0, got {Q.shape}."
            )

        def _apply_R(w):
            return Q @ w

        def _apply_Rt(w):
            return Q.T @ w

    def matvec(v):
        v = np.asarray(v).ravel()
        out = base.matvec(v).reshape(L, p)
        for ell in range(L):
            out[ell] = _apply_R(out[ell])
        return out.ravel()

    def rmatvec(w):
        w = np.asarray(w).ravel()
        win = w.reshape(L, p).copy()
        for ell in range(L):
            win[ell] = _apply_Rt(win[ell])
        return base.rmatvec(win.ravel())

    def _matmat(V):
        V = np.asarray(V)
        if V.ndim == 1:
            return matvec(V)
        # base returns (d, nvec) with rows grouped in L blocks of p
        out = base.matmat(V).reshape(L, p, -1)
        for ell in range(L):
            out[ell] = _apply_R(out[ell])
        return out.reshape(d, -1)

    def _rmatmat(W):
        W = np.asarray(W)
        if W.ndim == 1:
            return rmatvec(W)
        nvec = W.shape[1]
        win = W.reshape(L, p, nvec).copy()
        for ell in range(L):
            win[ell] = _apply_Rt(win[ell])
        return base.rmatmat(win.reshape(d, nvec))

    linop = ScipyLinearOperator((d, K), matvec=matvec, rmatvec=rmatvec,
                                dtype=np.float64)
    linop._matmat = _matmat
    linop._rmatmat = _rmatmat
    return linop


def hankel_preserving_null(
    X: list[list[NDArray[np.floating]]],
    L: int,
    U_cell: list[list[NDArray[np.floating]]],
    r_values: NDArray[np.integer],
    opts: dict | None = None,
) -> dict:
    """
    Hankel-preserving null via channel rotation (Section 7 of the paper).

    For each null replicate and each recording (s, c):

      1. Sample a random rotation Q_sc:
         - row_block > 0: Q_sc ∈ O(p_channels), Haar, applied as the
           block-diagonal I_{L1} ⊗ Q_sc on the first-level Hankel rows
           (channel-major layout, ``row_block`` = first-level depth).
           This destroys spatial alignment while PRESERVING the delay
           (lag) structure — the null described in the paper.
         - row_block == 0: Q_sc ∈ O(p) dense on all rows (legacy,
           mixes channels AND lags).
      2. Rebuild the second-level embedding of the rotated rows and
         recompute the truncated SVD at the same local rank r_sc — all
         via LinearOperators, so the (d × N) second-level matrix is
         NEVER materialized (v2 of this function materialized it,
         which was infeasible at pipeline scale).

    Parameters
    ----------
    X : list of list of arrays
        X[s][c] shape (p, T) — first-level Hankel matrices (the same
        cell passed to ``cdhsa_A1_A5``).
    L : int
        Second-level embedding depth (the same L passed to
        ``cdhsa_A1_A5``).
    U_cell : list of list of arrays
        Observed local bases (used only to extract ranks r_sc).
    r_values : array of int
        Candidate rank values.
    opts : dict, optional
        n_null (int, default 100)
        alpha (float, default 0.05)
        n_folds (int, default 5)
        seed (int, default 11)
        row_block (int, default 0)
        col_masks : list of list of arrays or None
            Máscaras de columnas de nivel 2 por grabación (ver
            ``level2_boundary_mask``). Deben ser LAS MISMAS que se usaron
            en ``cdhsa_A1_A5`` para que el nulo y los datos observados
            compartan estructura de columnas.

    Returns
    -------
    null : dict
        Same structure as ``random_subspace_null``.

    Notes
    -----
    COST WARNING: each replicate re-runs one truncated SVD per recording
    on a rotated second-level operator (~ the cost of one A1-A5 pass).
    For the full 5×5 protocol this is ~10-15 min per replicate, so use
    n_null ~ 20-100 for exploration and plan n_null = 500+ for final
    inference (the paper's planned comparison).

    The Frobenius norm of the second-level Hankel matrix is invariant
    under the rotation (orthogonal invariance per column-block), so the
    normalization uses the observed norm directly.
    """
    from src.cdhsa.a_common_subspace import (
        _make_scaled_linop,
        compute_block_hankel_fro,
        truncated_left_svd,
    )

    if opts is None:
        opts = {}
    n_null = opts.get("n_null", 100)
    alpha = opts.get("alpha", 0.05)
    n_folds = opts.get("n_folds", 5)
    seed = opts.get("seed", 11)
    row_block = int(opts.get("row_block", 0))
    col_masks = opts.get("col_masks")

    S = len(X)
    C = len(X[0])
    p = X[0][0].shape[0]
    if row_block > 0 and p % row_block != 0:
        raise ValueError(
            f"row_block={row_block} no divide p={p} (filas de X)."
        )
    n_ch = p // row_block if row_block > 0 else p
    if col_masks is not None and (
        len(col_masks) != S or any(len(row) != C for row in col_masks)
    ):
        raise ValueError(
            f"col_masks debe tener la forma (S={S}, C={C}) o ser None."
        )

    r_values = np.atleast_1d(np.asarray(r_values, dtype=int))
    qmax = int(np.max(r_values))

    lambda_null = np.full((n_null, qmax), np.nan)
    cvmin_null = np.full((n_null, len(r_values)), np.nan)

    cv_opts = {"n_folds": n_folds, "seed": seed + 100000}

    rng = np.random.default_rng(seed)

    # Pre-compute the (rotation-invariant) second-level Frobenius norms
    # (v4: sobre las columnas válidas si hay máscara de fronteras)
    hnorms = [
        [
            compute_block_hankel_fro(
                np.asarray(X[s][c]), L,
                col_mask=(
                    col_masks[s][c] if col_masks is not None
                    and col_masks[s][c] is not None else None
                ),
            )
            for c in range(C)
        ]
        for s in range(S)
    ]

    for b in range(n_null):
        U0: list[list[NDArray]] = [[None] * C for _ in range(S)]
        for s in range(S):
            for c in range(C):
                rsc = U_cell[s][c].shape[1]

                # Random rotation (Haar; v4: signos QR corregidos)
                Q = haar_random_basis(rng, n_ch, n_ch)

                mask_sc = (
                    col_masks[s][c] if col_masks is not None
                    and col_masks[s][c] is not None else None
                )

                # Rotated + normalized second-level operator
                linop_rot = _make_rotated_block_hankel_linop(
                    np.asarray(X[s][c]), L, Q, row_block,
                    col_mask=mask_sc,
                )
                linop_norm = _make_scaled_linop(linop_rot, hnorms[s][c])
                U0[s][c] = truncated_left_svd(linop_norm, rsc)

        # Null commonality spectrum
        _, lam = common_basis_from_U(U0, qmax)
        lambda_null[b, : len(lam)] = lam

        # Null CV (indefinido con S < 2)
        if S >= 2:
            cvb = crossvalidate_common_rank(U0, r_values, cv_opts)
            cvmin_null[b, :] = cvb["min_condition"]

    return {
        "lambda": lambda_null,
        "cv_min": cvmin_null,
        "lambda_q": np.nanquantile(lambda_null, 1 - alpha, axis=0),
        "cv_min_q": np.nanquantile(cvmin_null, 1 - alpha, axis=0),
        "alpha": alpha,
        "n_null": n_null,
        "row_block": row_block,
    }
