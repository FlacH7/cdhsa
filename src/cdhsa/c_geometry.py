"""
cdhsa/c_geometry.py - Direction-sensitive subspace geometry test (Step C extended)
=============================================================================

For each accumulated common block W_k = W0[:, :k] (k = 1..k_max) and each
local basis U^{(s,c)}, the paper defines the TANGENT COMPONENT as the
residual of the first k local directions after removing the common block::

    L_{s,c,k} = (I - W_k W_k^T) U^{(s,c)}_{1:k}

This is the quantity that connects with the Remark on the Grassmann
log-map: L is the vertical (off-block) part of the local block relative
to the common block. Unlike the scalar alignment a = ||U^T w||^2, the
tangent component retains DIRECTION/SIGN information, making balanced
rotations detectable.

(v4 change) The previous implementation computed
``L = (I - W_k W_k^T) P_sc W_k`` — the local projector applied to the
COMMON block. That is a related but DIFFERENT quantity (it measures how
much of W_k falls outside span(U_sc), with a projector normalization
that depends on the full local rank). The paper's definition above is
what we implement now: it is the residual of the LOCAL block, and it is
what the paper's Section on the log-map remark uses.

The omnibus statistic for each accumulated block is::

    T_k = S * sum_c ||mean_s L_sc,k - grand_mean||_F^2

with the family k = 1..k_max, k_max = min(r0, min_{s,c} r_sc) (blocks
cannot be wider than any local rank — otherwise U_{1:k} would not exist
for every recording). Condition labels are permuted within subject,
with max-T correction across the family (paper 3.4). A6.W0 is fixed
(estimated from label-blind pooled data).

Dependencies: numpy
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ============================================================================
# Helper: omnibus tangent statistic (UNCHANGED — paper-faithful)
# ============================================================================

def _tangent_stat(Lsc: list[list[NDArray]]) -> float:
    """
    Omnibus statistic for a set of tangent components.

    Parameters
    ----------
    Lsc : list of list of arrays, shape (S, C)
        Lsc[s][c] is a matrix of shape (d, dim_block).

    Returns
    -------
    T : float
        S * Σ_c ||mean_s L_sc - grand_mean||_F²
    """
    S = len(Lsc)
    C = len(Lsc[0])
    template_shape = Lsc[0][0].shape

    cond_means = []
    grand = np.zeros(template_shape, dtype=np.float64)

    for c in range(C):
        M = np.zeros(template_shape, dtype=np.float64)
        for s in range(S):
            M += Lsc[s][c]
        M /= S
        cond_means.append(M)
        grand += M / C

    T = 0.0
    for c in range(C):
        D = cond_means[c] - grand
        T += S * np.sum(D ** 2)

    return float(T)


# ============================================================================
# Helper: accumulated tangent components (paper definition)
# ============================================================================

def cumulative_tangent_blocks(
    r0: int,
    min_local_rank: int,
) -> list[NDArray[np.integer]]:
    """
    Familia acumulada de bloques B_k = {1..k}, k = 1..k_max (paper 3.4).

    k_max = min(r0, min_{s,c} r_sc): un bloque no puede ser más ancho
    que el rango local más pequeño, porque U_{1:k} debe existir en TODAS
    las grabaciones para que las matrices L tengan la misma forma y el
    estadístico omnibus esté bien definido.
    """
    k_max = int(min(r0, min_local_rank))
    return [np.arange(1, k + 1, dtype=int) for k in range(1, k_max + 1)]


def tangent_components(
    R: dict,
    A6: dict,
    blocks: list[NDArray[np.integer]],
) -> list[list[list[NDArray]]]:
    """
    Tangent components L_{s,c,k} = (I - W_k W_k^T) U^{(s,c)}_{1:k}.

    For each block ``blocks[k]`` (1-based indices into A6.W0),
    W_k = A6.W0[:, idx] and U^{(s,c)}_{1:k} = R.U[s][c][:, idx-1] —
    i.e. the SAME index set applied to the local basis, per the paper.

    Parameters
    ----------
    R : dict
        Output of ``cdhsa_A1_A5``. Must contain 'U', 'rank', 'S', 'C'.
    A6 : dict
        Output of ``cdhsa_A6_common_rank``. Must contain 'W0', 'r0'.
    blocks : list of arrays
        1-based indices into A6.W0. Every block must satisfy
        ``max(block) <= min_{s,c} r_sc`` (validated).

    Returns
    -------
    L : list (S) of list (C) of list (Kb) of arrays (d, dim_k)
    """
    S = R["S"]
    C = R["C"]
    r0 = int(A6["r0"])
    ranks = np.asarray(R["rank"])
    min_rank = int(ranks.min()) if ranks.size else 0

    # Validación: los bloques no pueden exceder el rango local mínimo.
    for k, idx in enumerate(blocks):
        idx = np.atleast_1d(np.asarray(idx, dtype=int))
        if np.any(idx < 1) or np.any(idx > r0):
            raise ValueError(
                f"Block {k} contiene índices fuera de [1, r0={r0}]."
            )
        if int(idx.max()) > min_rank:
            raise ValueError(
                f"Block {k} alcanza el índice {int(idx.max())} pero el "
                f"rango local mínimo es {min_rank}: U_{{1:{int(idx.max())}}} "
                f"no existe en todas las grabaciones. Use bloques con "
                f"k <= min(r0, min r_sc) (ver cumulative_tangent_blocks)."
            )

    L: list[list[list[NDArray]]] = [
        [[None] * len(blocks) for _ in range(C)] for _ in range(S)
    ]

    for s in range(S):
        for c in range(C):
            Ui = R["U"][s][c]  # (d, r_sc)
            for k, idx in enumerate(blocks):
                idx0 = np.atleast_1d(np.asarray(blocks[k], dtype=int)) - 1
                Wk = A6["W0"][:, idx0]  # (d, dim_k)
                U_block = Ui[:, idx0]  # (d, dim_k) — U_{1:k} del paper
                # L = U_{1:k} - W_k (W_k^T U_{1:k})
                L[s][c][k] = U_block - Wk @ (Wk.T @ U_block)

    return L


# ============================================================================
# Main: tangent geometry test
# ============================================================================

def cdhsa_tangent_geometry_test(
    R: dict,
    A6: dict,
    blocks: list[NDArray[np.integer]] | None = None,
    opts: dict | None = None,
) -> dict:
    """
    Direction-sensitive subspace geometry test (paper Def. of L_{s,c,k}).

    For the accumulated common block W_k = W0[:, :k]::

        L_sc,k = (I - W_k W_k^T) U^{(s,c)}_{1:k}

    L_sc,k is the off-block residual of the first k LOCAL directions.
    Its Frobenius norm is the "vertical" displacement of the local block
    relative to the common block (log-map remark of the paper). Unlike
    scalar alignment w'Pw, its SIGN/DIRECTION is retained, so a balanced
    rotation around a pooled midpoint is detectable.

    The omnibus statistic for each block is::

        T_k = S * Σ_c ||mean_s L_sc,k - grand_mean L||_F²

    Condition labels are permuted within subject, with max-T correction
    across blocks. A6.W0 is fixed because it was estimated from
    label-blind pooled data.

    Parameters
    ----------
    R : dict
        Output of ``cdhsa_A1_A5``. Must contain 'U', 'rank', 'S', 'C'.
    A6 : dict
        Output of ``cdhsa_A6_common_rank``. Must contain 'W0', 'r0'.
    blocks : list of arrays, optional
        Each array contains 1-based indices into A6.W0.
        Default: the ACCUMULATED family B_k = {1..k},
        k = 1..k_max with k_max = min(r0, min_{s,c} r_sc) (paper 3.4).
        Pass e.g. [np.arange(1, r0+1)] for the single omnibus block.
    opts : dict, optional
        n_perm (int, default 5000)
        seed (int, default 1234)
        alpha (float, default 0.05)

    Returns
    -------
    G : dict with keys:
        blocks : list of arrays
        T_obs : ndarray (Kb,) — observed tangent statistic.
        p_uncorrected : ndarray (Kb,)
        p_maxT : ndarray (Kb,) — max-T corrected p-value.
        sig_maxT : ndarray (Kb,) bool
        null_T : ndarray (n_perm, Kb)
        null_maxT : ndarray (n_perm,)
        alpha, n_perm, seed, S, C, d
        k_max : int — max block width (= min(r0, min r_sc) for the
            accumulated default).
        degenerate : bool — True si S < 2 (la permutación dentro del
            único sujeto no cambia el estadístico; p = 1).
        mean_difference_norm, effect_ratio : ndarray (Kb,)  [only if C=2]

    Notes
    -----
    (v4) Cambios respecto a la versión anterior:

    1. **Estadístico del paper**: antes se computaba
       ``L = (I - W_k W_k^T) P_sc W_k`` (proyector local completo
       aplicado al bloque común); ahora ``L = (I - W_k W_k^T) U_{1:k}``
       (residuo de las primeras k columnas de la base local), que es la
       definición del paper y la que conecta con el Remark del log-map.
    2. **Bloques por defecto**: familia acumulada k=1..k_max con
       k_max = min(r0, min r_sc) y corrección max-T sobre la familia
       (antes: un único bloque omnibus {1..r0}).
    3. Lo que NO cambia: el estadístico T_k, la permutación
       within-subject y el max-T con (+1)/(n+1).
    """
    if opts is None:
        opts = {}
    n_perm = opts.get("n_perm", 5000)
    seed = opts.get("seed", 1234)
    alpha_val = opts.get("alpha", 0.05)

    if A6["r0"] < 1:
        raise ValueError("No supported common subspace (A6.r0 = 0).")

    S = R["S"]
    C = R["C"]
    ranks = np.asarray(R["rank"])
    min_rank = int(ranks.min()) if ranks.size else 0
    k_max = int(min(A6["r0"], min_rank))

    # Default: familia acumulada del paper (3.4)
    if blocks is None:
        blocks = cumulative_tangent_blocks(A6["r0"], min_rank)

    if not isinstance(blocks, list) or not all(
        isinstance(b, np.ndarray) for b in blocks
    ):
        raise ValueError("blocks must be a list of numpy arrays.")
    if len(blocks) == 0:
        raise ValueError("blocks no puede estar vacío.")

    Kb = len(blocks)
    d = A6["W0"].shape[0]

    # ---- Compute tangent components L{s,c,k} (definición del paper) ----
    L = tangent_components(R, A6, blocks)

    # ---- Observed statistic ----
    T_obs = np.zeros(Kb, dtype=np.float64)
    for k in range(Kb):
        L_k = [[L[s][c][k] for c in range(C)] for s in range(S)]
        T_obs[k] = _tangent_stat(L_k)

    # ---- Null distribution ----
    if S < 2:
        # Con un único sujeto, permutar las condiciones reordena el
        # conjunto {L_{1,c}} y T_k es INVARIANTE: p = 1 (degenerado).
        degenerate = True
        null_T = np.zeros((0, Kb), dtype=np.float64)
        null_maxT = np.zeros(0, dtype=np.float64)
        p_unc = np.ones(Kb, dtype=np.float64)
        p_max = np.ones(Kb, dtype=np.float64)
    else:
        degenerate = False
        rng = np.random.default_rng(seed)
        null_T = np.zeros((n_perm, Kb), dtype=np.float64)
        null_maxT = np.zeros(n_perm, dtype=np.float64)

        for b in range(n_perm):
            for k in range(Kb):
                # Permute condition labels within each subject
                L_k = [
                    [L[s][rng.permutation(C)[c]][k] for c in range(C)]
                    for s in range(S)
                ]
                null_T[b, k] = _tangent_stat(L_k)
            null_maxT[b] = np.max(null_T[b, :])

        p_unc = np.array([
            (1 + np.sum(null_T[:, k] >= T_obs[k])) / (n_perm + 1)
            for k in range(Kb)
        ])
        p_max = np.array([
            (1 + np.sum(null_maxT >= T_obs[k])) / (n_perm + 1)
            for k in range(Kb)
        ])

    result = {
        "blocks": blocks,
        "T_obs": T_obs,
        "p_uncorrected": p_unc,
        "p_maxT": p_max,
        "sig_maxT": p_max < alpha_val,
        "null_T": null_T,
        "null_maxT": null_maxT,
        "alpha": alpha_val,
        "n_perm": n_perm if not degenerate else 0,
        "seed": seed,
        "S": S,
        "C": C,
        "d": d,
        "k_max": k_max,
        "degenerate": degenerate,
        "definition": "(I - W_k W_k^T) U_{1:k} (paper)",
    }

    # ---- Effect sizes for C=2 (descriptive only) ----
    if C == 2:
        mean_diff_norm = np.zeros(Kb, dtype=np.float64)
        effect_ratio = np.zeros(Kb, dtype=np.float64)

        for k in range(Kb):
            diffs = [L[s][1][k] - L[s][0][k] for s in range(S)]
            M = np.mean(diffs, axis=0)
            mean_diff_norm[k] = np.linalg.norm(M, "fro")

            dn_sq = [
                np.linalg.norm(diffs[s] - M, "fro") ** 2 for s in range(S)
            ]
            denom = np.sqrt(np.mean(dn_sq))
            effect_ratio[k] = mean_diff_norm[k] / max(denom, np.finfo(np.float64).eps)

        result["mean_difference_norm"] = mean_diff_norm
        result["effect_ratio"] = effect_ratio

    return result
