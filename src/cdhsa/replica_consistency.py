"""
cdhsa/replica_consistency.py — Per-replica (Framework 2) consistency
====================================================================

Implements the complementary evidentiary standard of Section 4.5 of the
paper: each super-subject is treated as an independent replica of the
same material, and every pooled (Framework 1) effect is re-examined
per replica with the "consistent in >= 4 of 5 replicas" criterion.

Computed quantities (Table 2 of the paper, general C):

  1. Backbone stability: overlap between the replica-only backbone
     W0^{(s)} (pooling only the conditions of replica s) and the pooled
     W0. Reference value for independent subspaces: r0^2 / d.

  2. Energy replication: for C = 2, the per-replica energy shift
     dE^{(s)}_j = E^{(s)}_{c2,j} - E^{(s)}_{c1,j} and its sign
     consistency with the pooled shift. For C > 2, the dominant
     condition of each direction is computed per replica and compared
     with the pooled dominant condition.

  3. Deformation replication: per-replica between-condition tangent
     dispersion T^{(s)}_k (for C = 2 this is the paper's
     ||L_{s,c',k} - L_{s,c,k}||_F^2), plus the mean pairwise cosine
     similarity between per-replica difference directions as a
     directional-consistency summary.

  4. Discriminability: per-replica residual-manifold separation
     Delta^{(s)} = 1 - a^{(s)}_cross with a^{(s)}_cross the mean
     pairwise cross-condition overlap of the residual bases.

Dependencies: numpy, cdhsa.a6_common_rank (common_basis_from_U)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.cdhsa.a6_common_rank import common_basis_from_U
from src.cdhsa.c_geometry import cumulative_tangent_blocks, tangent_components


def _backbone_stability(
    R: dict,
    A6: dict,
) -> dict:
    """
    Overlap between each replica-only backbone and the pooled backbone.
    """
    S, C = R["S"], R["C"]
    r0 = int(A6["r0"])
    W0 = A6["W0"]  # (d, r0)

    overlap = np.zeros(S, dtype=np.float64)
    for s in range(S):
        U_replica = [[R["U"][s][c] for c in range(C)]]
        W_s, _ = common_basis_from_U(U_replica, r0)
        G = W_s.T @ W0  # (r0, r0)
        overlap[s] = np.linalg.norm(G, "fro") ** 2 / max(r0, 1)

    # Reference for independent random subspaces: E||Q1^T Q2||_F^2 ~ r0^2/d
    d = W0.shape[0]
    ref_random = (r0 ** 2) / d

    return {
        "overlap": overlap,
        "ref_random": ref_random,
        "n_above_ref": int(np.sum(overlap > ref_random)),
        "consistent_4of5": bool(np.sum(overlap > ref_random) >= max(1, int(np.ceil(0.8 * S)))),
    }


def _energy_replication(
    BC: dict,
) -> dict | None:
    """
    Sign consistency of per-replica condition effects on energy.

    Requires BC['metrics']['energy_abs'] with singleton blocks
    (the default of ``cdhsa_BC_condition_tests``).
    """
    M = BC.get("metrics")
    if M is None:
        return None
    blocks = M["blocks"]
    if not all(np.atleast_1d(b).shape[0] == 1 for b in blocks):
        # Non-singleton blocks: per-direction analysis not defined.
        return None

    E = M["energy_abs"]  # (S, C, Kb), Kb == r0
    S, C, Kb = E.shape

    grand = E.mean(axis=(0, 1))  # (Kb,)
    cond_mean = E.mean(axis=0)  # (C, Kb)
    delta_pooled = cond_mean - grand[None, :]  # (C, Kb)

    out: dict = {
        "Kb": Kb,
        "delta_pooled": delta_pooled,
    }

    if C == 2:
        # Paper (Framework 2): dE^{(s)}_j = E^{(s)}_{c2,j} - E^{(s)}_{c1,j}
        dE = E[:, 1, :] - E[:, 0, :]  # (S, Kb)
        dE_pooled = delta_pooled[1, :]  # == mean(dE, axis=0)
        sign_ok = np.sign(dE) == np.sign(dE_pooled)[None, :]
        n_sign = sign_ok.sum(axis=0)  # (Kb,)
        out.update({
            "dE_replica": dE,
            "dE_pooled": dE_pooled,
            "n_sign_consistent": n_sign,
            "consistent_4of5": n_sign >= max(1, int(np.ceil(0.8 * S))),
        })
    else:
        # General C: dominant condition per direction, replica vs pooled
        dom_pooled = np.argmax(E.mean(axis=0), axis=0)  # (Kb,)
        dom_replica = np.argmax(E, axis=1)  # (S, Kb)
        n_match = (dom_replica == dom_pooled[None, :]).sum(axis=0)
        out.update({
            "dominant_condition_pooled": dom_pooled,
            "dominant_condition_replica": dom_replica,
            "n_dominant_match": n_match,
            "consistent_4of5": n_match >= max(1, int(np.ceil(0.8 * S))),
        })

    return out


def _deformation_replication(
    R: dict,
    A6: dict,
    blocks: list[NDArray] | None = None,
) -> dict | None:
    """
    Per-replica between-condition tangent dispersion T^{(s)}_k.

    Usa la MISMA definición de componente tangente que el Step C
    (``c_geometry.tangent_components``): L_{s,c,k} = (I − W_k W_kᵀ)
    U^{(s,c)}_{1:k} (v4; antes usaba el proyector local completo
    P_sc W_k, inconsistente con el paper).

    For C = 2 this reduces to the paper's (31):
    T^{(s)}_k = ||L_{s,c',k} - L_{s,c,k}||_F^2.
    Also reports the mean pairwise cosine similarity of the per-replica
    difference directions (C=2) as a directional consistency summary.

    Default blocks: la familia acumulada k=1..k_max del paper
    (k_max = min(r0, min r_sc)).
    """
    S, C = R["S"], R["C"]
    r0 = int(A6["r0"])
    ranks = np.asarray(R["rank"])
    min_rank = int(ranks.min()) if ranks.size else 0
    if r0 < 1 or min_rank < 1:
        return None
    if blocks is None:
        blocks = cumulative_tangent_blocks(r0, min_rank)
    Kb = len(blocks)

    L = tangent_components(R, A6, blocks)

    T_rep = np.zeros((S, Kb), dtype=np.float64)
    cos_sim = np.zeros(Kb, dtype=np.float64)

    for k in range(Kb):
        for s in range(S):
            if C == 2:
                D = L[s][1][k] - L[s][0][k]
                T_rep[s, k] = float(np.sum(D ** 2))
            else:
                Lbar = np.mean([L[s][c][k] for c in range(C)], axis=0)
                T_rep[s, k] = float(
                    sum(np.sum((L[s][c][k] - Lbar) ** 2) for c in range(C))
                )

        # Directional consistency (C=2): pairwise cosines of differences
        if C == 2:
            diffs = [(L[s][1][k] - L[s][0][k]).ravel() for s in range(S)]
            norms = [np.linalg.norm(dv) for dv in diffs]
            pairs = [
                float(np.dot(diffs[a], diffs[b]) /
                      max(norms[a] * norms[b], 1e-300))
                for a in range(S) for b in range(a + 1, S)
            ]
            cos_sim[k] = float(np.mean(pairs))

    return {
        "blocks": blocks,
        "T_replica": T_rep,
        "cosine_similarity": cos_sim if C == 2 else None,
        "definition": "(I - W_k W_k^T) U_{1:k} (paper)",
    }


def _discriminability_replication(
    D: dict,
) -> dict | None:
    """
    Per-replica residual-manifold separation Delta^{(s)} = 1 - a^{(s)}_cross.

    a^{(s)}_cross is the mean over ordered condition pairs (c != c') of
    ||Ures_{s,c}^T Ures_{s,c'}||_F^2 / min(r_c, r_c'), which reduces to
    the paper's (1/r)||Ures_c1^T Ures_c2||_F^2 for C = 2.
    """
    if D is None or "U_residual" not in D:
        return None
    U_res = D["U_residual"]
    S, C = D["S"], D["C"]

    delta = np.zeros(S, dtype=np.float64)
    for s in range(S):
        vals = []
        for c in range(C):
            for cp in range(C):
                if cp == c:
                    continue
                Ua, Ub = U_res[s][c], U_res[s][cp]
                if Ua.shape[1] == 0 or Ub.shape[1] == 0:
                    continue
                G = Ua.T @ Ub
                denom = max(min(Ua.shape[1], Ub.shape[1]), 1)
                vals.append(np.linalg.norm(G, "fro") ** 2 / denom)
        a_cross = float(np.mean(vals)) if vals else 0.0
        delta[s] = 1.0 - a_cross

    return {
        "delta_replica": delta,
        "n_positive": int(np.sum(delta > 0)),
        "consistent_4of5": bool(
            np.sum(delta > 0) >= max(1, int(np.ceil(0.8 * S)))
        ),
    }


def cdhsa_replica_consistency(
    R: dict,
    A6: dict,
    BC: dict | None = None,
    G: dict | None = None,
    D: dict | None = None,
    opts: dict | None = None,
) -> dict:
    """
    Framework 2 (Section 4.5 of the paper): per-replica consistency.

    Parameters
    ----------
    R : dict — output of ``cdhsa_A1_A5``.
    A6 : dict — output of ``cdhsa_A6_common_rank``.
    BC : dict, optional — output of ``cdhsa_BC_condition_tests``
        (used for energy replication; requires singleton blocks).
    G : dict, optional — output of ``cdhsa_tangent_geometry_test``
        (its blocks are reused for the deformation analysis).
    D : dict, optional — output of ``cdhsa_D_condition_specific_modes``.
    opts : dict, optional
        blocks : list of arrays — override the tangent blocks.

    Returns
    -------
    REPR : dict with keys:
        backbone : dict — overlap per replica + 4/5 criterion.
        energy : dict | None — per-direction sign/dominance consistency.
        deformation : dict | None — T^{(s)}_k per replica + cosines.
        discriminability : dict | None — Delta^{(s)} per replica.
        S, C, r0
    """
    if opts is None:
        opts = {}

    if A6.get("r0", 0) < 1:
        raise ValueError("A6.r0 = 0: no common backbone to replicate.")
    if R["S"] < 2:
        raise ValueError(
            "Framework 2 (consistencia por réplica) requiere S >= 2 "
            "réplicas; got S=1. El análisis por réplica única es "
            "descriptivo (ver run_specific_modes_robustness.py)."
        )

    result = {
        "backbone": _backbone_stability(R, A6),
        "energy": _energy_replication(BC) if BC is not None else None,
        "deformation": _deformation_replication(
            R, A6,
            blocks=(G.get("blocks") if G is not None else None) or opts.get("blocks"),
        ),
        "discriminability": _discriminability_replication(D),
        "S": R["S"],
        "C": R["C"],
        "r0": int(A6["r0"]),
    }
    return result
