"""Tests for the v3 improvements of CD-HSA
==========================================

Covers:
  1. select_rank_variance (criterio X% de varianza explicada)
  2. cdhsa_A1_A5 with rank_method='variance'
  3. Step D cross-condition alignment for C > 2 (bug fix, Def. 3.14)
  4. Adaptive specific rank (Def. 3.13)
  5. LOSO-calibrated prevalence contrast (Def. 3.15)
  6. Hankel-preserving null: rotated LinearOperator identity + end-to-end
  7. A6 with null_type='hankel'
  8. Replica consistency (Framework 2)
  9. Full pipeline run_cdhsa with the new options
  10. Rank tag consistency (pipeline vs batch runner)

Run with:  python -m pytest tests/test_improvements.py -v
"""

from __future__ import annotations
from pathlib import Path
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pytest

from src.cdhsa.a_common_subspace import (
    build_block_hankel,
    make_block_hankel_linop,
    _make_scaled_linop,
    compute_block_hankel_fro,
    level2_boundary_mask,
    select_rank_variance,
    cdhsa_A1_A5,
)
from src.cdhsa.a6_common_rank import (
    cdhsa_A6_common_rank,
    common_basis_from_U,
    crossvalidate_common_rank,
    haar_random_basis,
)
from src.cdhsa.null_distributions import (
    _make_rotated_block_hankel_linop,
    hankel_preserving_null,
    random_subspace_null,
)
from src.cdhsa.b_energy import compute_common_mode_metrics
from src.cdhsa.c_geometry import (
    cdhsa_tangent_geometry_test,
    cumulative_tangent_blocks,
    tangent_components,
)
from src.cdhsa.extract_mode_indices import project_level2
from src.cdhsa.d_condition_specific import (
    cdhsa_D_condition_specific_modes,
    cdhsa_D_prevalence_loso,
)
from src.cdhsa.replica_consistency import cdhsa_replica_consistency
from src.pipelines.run_cdhsa import CDHSAConfig, run_cdhsa, _rank_tag
from src.batch_runs.run_batch_cdhsa import CDHSABatchRunner


# ============================================================================
# Fixtures
# ============================================================================

def _make_lowrank_matrix(rng, d=40, K=300, r_true=3, noise=0.05):
    """Matriz (d, K) de rango efectivo ~r_true + ruido."""
    A = rng.standard_normal((d, r_true)) / np.sqrt(d)
    Z = rng.standard_normal((r_true, K))
    M = A @ Z + noise * rng.standard_normal((d, K)) / np.sqrt(d)
    return M / np.linalg.norm(M, "fro")


@pytest.fixture
def variance_eeg():
    """Synthetic EEG with heterogeneous effective dimensionality.

    S=3, C=3, p=4, T=400. All conditions share a 10 Hz component;
    condition 2 adds an extra 18 Hz component (more effective dims).
    """
    rng = np.random.default_rng(7)
    S, C, p, T = 3, 3, 4, 400
    fs = 200.0
    L = 10
    t = np.arange(T) / fs

    a10 = rng.normal(size=p); a10 /= np.linalg.norm(a10)
    a18 = rng.normal(size=p)
    a18 -= a10 * np.dot(a10, a18); a18 /= np.linalg.norm(a18)

    X = [[None] * C for _ in range(S)]
    for s in range(S):
        phi10 = rng.uniform(0, 2 * np.pi)
        z10 = np.sin(2 * np.pi * 10 * t + phi10)
        for c in range(C):
            Xi = np.outer(a10, z10) + 0.3 * rng.normal(size=(p, T))
            if c == 1:  # condicion 2: componente extra
                z18 = np.sin(2 * np.pi * 18 * t + rng.uniform(0, 2 * np.pi))
                Xi += 1.5 * np.outer(a18, z18)
            X[s][c] = Xi
    return X, L, S, C, p, T


@pytest.fixture
def specific_eeg():
    """S=4, C=3 con estructura especifica fuerte en la condicion 0.

    Compartido: 10 Hz. Especifico de c=0: 18 Hz (mismo patron espacial
    en todos los sujetos -> debe pasar el test Haar y el contraste LOSO).
    """
    rng = np.random.default_rng(11)
    S, C, p, T = 4, 3, 5, 500
    fs = 200.0
    L = 10
    t = np.arange(T) / fs

    a10 = rng.normal(size=p); a10 /= np.linalg.norm(a10)
    a18 = rng.normal(size=p)
    a18 -= a10 * np.dot(a10, a18); a18 /= np.linalg.norm(a18)

    X = [[None] * C for _ in range(S)]
    for s in range(S):
        z10 = np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
        for c in range(C):
            Xi = np.outer(a10, z10) + 0.25 * rng.normal(size=(p, T))
            if c == 0:
                z18 = np.sin(2 * np.pi * 18 * t + rng.uniform(0, 2 * np.pi))
                Xi += 1.2 * np.outer(a18, z18)
            X[s][c] = Xi
    return X, L, S, C, p, T


# ============================================================================
# 1. select_rank_variance
# ============================================================================

class TestSelectRankVariance:

    def test_lowrank_matrix(self):
        rng = np.random.default_rng(0)
        M = _make_lowrank_matrix(rng, d=40, K=300, r_true=3, noise=1e-6)
        U, r, s, cum, censored = select_rank_variance(
            M, var_explained=0.99, var_max_rank=20
        )
        assert r <= 5
        assert not censored
        assert cum[r - 1] >= 0.99 - 1e-9
        # Ortonormalidad
        G = U.T @ U
        np.testing.assert_allclose(G, np.eye(r), atol=1e-10)

    def test_censored_when_cap_too_small(self):
        rng = np.random.default_rng(1)
        # Matriz de rango lleno: 99% necesita casi todo el espectro
        M = rng.standard_normal((30, 60))
        M /= np.linalg.norm(M, "fro")
        U, r, s, cum, censored = select_rank_variance(
            M, var_explained=0.99, var_max_rank=5
        )
        assert censored
        assert r == 5
        assert cum[-1] < 0.99

    def test_linop_matches_dense(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((6, 120))
        L = 5
        hnorm = compute_block_hankel_fro(X, L)
        linop = _make_scaled_linop(make_block_hankel_linop(X, L), hnorm)
        U, r, s, cum, cens = select_rank_variance(
            linop, var_explained=0.95, var_max_rank=15
        )
        # El operador esta normalizado: cum es fraccion de energia
        assert np.all(cum <= 1.0 + 1e-9)
        assert 1 <= r <= 15

    def test_var_explained_bounds(self):
        rng = np.random.default_rng(3)
        M = rng.standard_normal((10, 20))
        with pytest.raises(ValueError):
            select_rank_variance(M, var_explained=0.0)
        with pytest.raises(ValueError):
            select_rank_variance(M, var_explained=1.5)


# ============================================================================
# 2. cdhsa_A1_A5 with rank_method='variance'
# ============================================================================

class TestA1A5Variance:

    def test_output_keys_and_ranks(self, variance_eeg):
        X, L, S, C, p, T = variance_eeg
        R = cdhsa_A1_A5(
            X, L,
            rank_method="variance",
            var_explained=0.90,
            var_max_rank=10,
            max_common=6,
        )
        for key in ("rank", "var_explained_achieved", "var_censored",
                    "var_curve", "rank_method"):
            assert key in R
        assert R["rank_method"] == "variance"
        assert R["var_explained"] == 0.90
        assert R["rank"].shape == (S, C)
        assert np.all(R["rank"] >= 1)
        assert np.all(R["rank"] <= 10)
        # La varianza lograda debe superar el objetivo donde no censura
        for s in range(S):
            for c in range(C):
                if not R["var_censored"][s, c]:
                    assert (R["var_explained_achieved"][s, c]
                            >= 0.90 - 1e-9)

    def test_heterogeneous_ranks_flow_through(self, variance_eeg):
        """Rangos distintos por grabacion deben fluir por A3-A5."""
        X, L, S, C, p, T = variance_eeg
        R = cdhsa_A1_A5(
            X, L, rank_method="variance",
            var_explained=0.85, var_max_rank=8, max_common=4,
        )
        for s in range(S):
            for c in range(C):
                assert R["U"][s][c].shape == (p * L, R["rank"][s, c])
                G = R["U"][s][c].T @ R["U"][s][c]
                np.testing.assert_allclose(
                    G, np.eye(R["rank"][s, c]), atol=1e-8
                )
        # lambda dentro de [0, 1]
        assert np.all(R["lambda_"] >= 0)
        assert np.all(R["lambda_"] <= 1.0 + 1e-9)


# ============================================================================
# 3. Step D: cross-alignment for C > 2 (bug fix)
# ============================================================================

class TestDCrossAlignment:

    def _run_D(self, fixture, **opts):
        X, L, S, C, p, T = fixture
        R = cdhsa_A1_A5(X, L, fixed_rank=5, max_common=5)
        A6 = cdhsa_A6_common_rank(R, opts={
            "max_common": 5, "n_folds": 2, "n_null": 5, "seed": 1,
        })
        assert A6["r0"] >= 1
        D = cdhsa_D_condition_specific_modes(
            X, L, R, A6, opts={"max_specific": 3, **opts}
        )
        return D, S, C

    def test_cross_alignment_nonzero_for_C3(self, variance_eeg):
        D, S, C = self._run_D(variance_eeg)
        assert C == 3
        # Con C=3 la v2 dejaba alignment_cross en 0; ahora debe computarse
        assert np.any(D["alignment_cross"] > 0)
        # El argmax debe apuntar a una condicion valida distinta
        for s in range(S):
            for c in range(C):
                am = D["alignment_cross_argmax"][s, c]
                assert am == -1 or am != c

    def test_contrast_is_own_minus_cross(self, variance_eeg):
        D, S, C = self._run_D(variance_eeg)
        expected = D["mean_own_alignment"] - D["mean_cross_alignment"]
        np.testing.assert_allclose(
            D["prevalence_contrast"], expected, atol=1e-12
        )
        # En la v2 (bug) el contraste igualaba a mean_own; verificar
        # que ya NO es asi cuando hay cross-alineacion
        if np.any(D["alignment_cross"] > 0):
            assert not np.allclose(
                D["prevalence_contrast"], D["mean_own_alignment"]
            )

    def test_cross_manual_C2(self, variance_eeg):
        """Para C=2 el cross debe ser exactamente la otra condicion."""
        X, L, S, C, p, T = variance_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=5, max_common=5)
        A6 = cdhsa_A6_common_rank(R, opts={
            "max_common": 5, "n_folds": 2, "n_null": 5, "seed": 1,
        })
        D = cdhsa_D_condition_specific_modes(
            X, L, R, A6, opts={"max_specific": 2}
        )
        # Reducir a C=2 manualmente: usar solo condiciones 0 y 1
        C2 = 2
        for s in range(S):
            for c in range(C2):
                Ures = D["U_residual"][s][c]
                cp = 1 - c
                Wp = D["W_specific"][cp]
                if Ures.shape[1] == 0 or Wp.shape[1] == 0:
                    manual = 0.0
                else:
                    G = Ures.T @ Wp
                    manual = np.linalg.norm(G, "fro") ** 2 / Wp.shape[1]
                # Solo valida si el maximo cross es la otra condicion
                if D["alignment_cross_argmax"][s, c] == cp:
                    assert D["alignment_cross"][s, c] == pytest.approx(
                        manual, abs=1e-10
                    )


# ============================================================================
# 4. Adaptive specific rank (Def. 3.13)
# ============================================================================

class TestAdaptiveSpecificRank:

    def test_adaptive_leq_cap_and_noise_kills_modes(self, variance_eeg):
        X, L, S, C, p, T = variance_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=5, max_common=5)
        A6 = cdhsa_A6_common_rank(R, opts={
            "max_common": 5, "n_folds": 2, "n_null": 5, "seed": 1,
        })
        D_legacy = cdhsa_D_condition_specific_modes(
            X, L, R, A6,
            opts={"max_specific": 3, "rank_adaptive": False},
        )
        D_adapt = cdhsa_D_condition_specific_modes(
            X, L, R, A6,
            opts={"max_specific": 3, "rank_adaptive": True,
                  "n_null_specific": 20},
        )
        # El rank adaptativo nunca supera el tope
        assert np.all(D_adapt["r_specific"] <= D_legacy["r_specific"])
        assert D_adapt["rank_adaptive"] is True

    def test_adaptive_detects_structure(self, specific_eeg):
        X, L, S, C, p, T = specific_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=6, max_common=6)
        A6 = cdhsa_A6_common_rank(R, opts={
            "max_common": 6, "n_folds": 2, "n_null": 5, "seed": 1,
        })
        D = cdhsa_D_condition_specific_modes(
            X, L, R, A6,
            opts={"max_specific": 4, "rank_adaptive": True,
                  "n_null_specific": 30},
        )
        # La condicion 0 tiene estructura especifica fuerte
        assert D["r_specific"][0] >= 1
        assert D["prevalence_contrast"][0] > 0


# ============================================================================
# 5. LOSO-calibrated prevalence contrast (Def. 3.15)
# ============================================================================

class TestLoso:

    def test_detects_specific_condition(self, specific_eeg):
        X, L, S, C, p, T = specific_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=6, max_common=6)
        A6 = cdhsa_A6_common_rank(R, opts={
            "max_common": 6, "n_folds": 2, "n_null": 5, "seed": 1,
        })
        D = cdhsa_D_condition_specific_modes(
            X, L, R, A6,
            opts={"max_specific": 3, "rank_adaptive": True,
                  "n_null_specific": 15},
        )
        LOSO = cdhsa_D_prevalence_loso(D, opts={"n_perm": 60, "seed": 5})
        assert LOSO["delta_loso"].shape == (C,)
        assert LOSO["p_loso"].shape == (C,)
        assert np.all(LOSO["p_loso"] >= 1.0 / 61)
        assert np.all(LOSO["p_loso"] <= 1.0)
        # La condicion especifica (c=0) debe tener contraste positivo
        assert LOSO["delta_loso"][0] > 0
        # p-valor en rango plausible (no exigir significancia extrema
        # con n_perm=60, pero si que no sea el peor)
        assert LOSO["p_loso"][0] < 0.5

    def test_non_specific_conditions_small_contrast(self, variance_eeg):
        """Solo la condicion con estructura especifica (c=1, 18 Hz extra)
        debe mostrar un contraste LOSO grande; las demas, pequeno."""
        X, L, S, C, p, T = variance_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=5, max_common=5)
        A6 = cdhsa_A6_common_rank(R, opts={
            "max_common": 5, "n_folds": 2, "n_null": 5, "seed": 1,
        })
        D = cdhsa_D_condition_specific_modes(
            X, L, R, A6, opts={"max_specific": 2}
        )
        LOSO = cdhsa_D_prevalence_loso(D, opts={"n_perm": 30, "seed": 5})
        # Condiciones 0 y 2: sin componente especifica -> contraste pequeno
        # (mucho menor que el ~0.87 que producia el bug v2 de cross=0)
        assert abs(LOSO["delta_loso"][0]) < 0.35
        assert abs(LOSO["delta_loso"][2]) < 0.35
        # La condicion 1 (18 Hz extra): contraste positivo y marcado
        assert LOSO["delta_loso"][1] > 0.3
        # Y el in-sample (D) sigue siendo mayor que el LOSO (sesgo self)
        assert D["prevalence_contrast"][1] > LOSO["delta_loso"][1]


# ============================================================================
# 6. Hankel-preserving null: rotated LinearOperator
# ============================================================================

class TestRotatedLinop:

    def test_blockwise_identity(self):
        """(I_L2 (x) R) H2_X == build_block_hankel(R X, L2)."""
        rng = np.random.default_rng(21)
        p, T, L2, rb = 6, 120, 4, 2
        n_ch = p // rb
        X = rng.standard_normal((p, T))

        Q, _ = np.linalg.qr(rng.standard_normal((n_ch, n_ch)))

        # Rotacion manual canal-major: R = Q (x) I_rb aplicada a las
        # filas de X: X_rot[ch*rb + tau] = sum_ch' Q[ch,ch'] X[ch'*rb + tau]
        X3 = X.reshape(n_ch, rb, T)
        X_rot = np.tensordot(Q, X3, axes=([1], [0])).reshape(p, T)

        H_ref = build_block_hankel(X_rot, L2)
        linop = _make_rotated_block_hankel_linop(X, L2, Q, rb)

        assert linop.shape == H_ref.shape
        v = rng.standard_normal(H_ref.shape[1])
        np.testing.assert_allclose(linop.matvec(v), H_ref @ v, atol=1e-10)
        w = rng.standard_normal(H_ref.shape[0])
        np.testing.assert_allclose(linop.rmatvec(w), H_ref.T @ w, atol=1e-10)
        V = rng.standard_normal((H_ref.shape[1], 3))
        np.testing.assert_allclose(linop.matmat(V), H_ref @ V, atol=1e-10)

    def test_dense_identity(self):
        rng = np.random.default_rng(22)
        p, T, L2 = 6, 120, 4
        X = rng.standard_normal((p, T))
        Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
        H_ref = build_block_hankel(Q @ X, L2)
        linop = _make_rotated_block_hankel_linop(X, L2, Q, 0)
        v = rng.standard_normal(H_ref.shape[1])
        np.testing.assert_allclose(linop.matvec(v), H_ref @ v, atol=1e-10)
        w = rng.standard_normal(H_ref.shape[0])
        np.testing.assert_allclose(linop.rmatvec(w), H_ref.T @ w, atol=1e-10)

    def test_row_block_validation(self):
        rng = np.random.default_rng(23)
        X = rng.standard_normal((5, 50))
        Q, _ = np.linalg.qr(rng.standard_normal((2, 2)))
        with pytest.raises(ValueError):
            _make_rotated_block_hankel_linop(X, 3, Q, 2)  # 5 % 2 != 0

    def test_hankel_null_end_to_end(self, variance_eeg):
        X, L, S, C, p, T = variance_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=4, max_common=4)
        null = hankel_preserving_null(
            X, L, R["U"], np.arange(1, 4),
            opts={"n_null": 2, "n_folds": 2, "seed": 3, "row_block": 0},
        )
        for key in ("lambda", "cv_min", "lambda_q", "cv_min_q"):
            assert key in null
        assert null["lambda"].shape[0] == 2
        assert np.all(null["lambda_q"] >= 0)


# ============================================================================
# 7. A6 with null_type='hankel'
# ============================================================================

class TestA6HankelNull:

    def test_a6_hankel_runs(self, variance_eeg):
        X, L, S, C, p, T = variance_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=4, max_common=4)
        A6 = cdhsa_A6_common_rank(R, opts={
            "max_common": 4, "n_folds": 2, "n_null": 2, "seed": 1,
            "null_type": "hankel",
            "X": X, "L_hankel": L, "hankel_row_block": 0,
        })
        assert "r0" in A6
        assert A6["r0"] >= 0

    def test_a6_hankel_requires_X(self, variance_eeg):
        X, L, S, C, p, T = variance_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=4, max_common=4)
        with pytest.raises(ValueError):
            cdhsa_A6_common_rank(R, opts={
                "max_common": 4, "n_folds": 2, "n_null": 2,
                "null_type": "hankel",  # faltan X y L_hankel
            })


# ============================================================================
# 8. Replica consistency (Framework 2)
# ============================================================================

class TestReplicaConsistency:

    def test_runs_and_shapes(self, specific_eeg):
        X, L, S, C, p, T = specific_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=6, max_common=6)
        A6 = cdhsa_A6_common_rank(R, opts={
            "max_common": 6, "n_folds": 2, "n_null": 5, "seed": 1,
        })
        D = cdhsa_D_condition_specific_modes(
            X, L, R, A6, opts={"max_specific": 3}
        )
        REPR = cdhsa_replica_consistency(R, A6, D=D)
        assert REPR["backbone"]["overlap"].shape == (S,)
        assert REPR["backbone"]["ref_random"] > 0
        # Estructura compartida fuerte -> overlap alto en >= 1 replica
        assert np.any(REPR["backbone"]["overlap"] > 0.5)
        assert REPR["deformation"]["T_replica"].shape[0] == S
        assert REPR["discriminability"]["delta_replica"].shape == (S,)


# ============================================================================
# 9. Full pipeline with new options
# ============================================================================

class TestPipelineNewOptions:

    def test_run_cdhsa_variance_full(self, variance_eeg):
        X, L, S, C, p, T = variance_eeg
        cfg = CDHSAConfig(
            rank_method="variance",
            var_explained=0.85,
            var_max_rank=6,
            var_min_rank=2,
            max_common=5,
            a6_n_null=3,
            a6_n_folds=2,
            bc_n_perm=40,
            tangent_blocks_mode="cumulative",
            tangent_n_perm=40,
            d_max_specific=3,
            d_rank_adaptive=True,
            d_n_null_specific=5,
            d_loso=True,
            d_loso_n_perm=15,
            run_replica_consistency=True,
            bc_condition_names=["c0", "c1", "c2"],
        )
        result = run_cdhsa(X, L, cfg)

        assert result.D_loso is not None
        assert result.REPR is not None
        assert result.R["rank_method"] == "variance"

        summary = result.summary()
        assert "Local rank (variance X=" in summary
        assert "Rank test:" in summary
        assert "Delta_loso=" in summary
        assert "Replica consistency" in summary
        # Bloques acumulativos: block k [1..k]
        assert "block 1 [1.." in summary or "block 1:" in summary

    def test_run_cdhsa_fixed_backward_compat(self, variance_eeg):
        """Defaults = defaults del paper (fixed, cumulative, sin loso).

        v4: tangent_blocks_mode por defecto es 'cumulative' (paper 3.4)
        — antes era el bloque omnibus único. El resto de defaults
        (fixed rank, sin LOSO, sin replica) se mantiene.
        """
        X, L, S, C, p, T = variance_eeg
        cfg = CDHSAConfig(
            fixed_rank=4, max_common=4,
            a6_n_null=3, a6_n_folds=2, bc_n_perm=20, tangent_n_perm=20,
        )
        result = run_cdhsa(X, L, cfg)
        assert result.D_loso is None
        assert result.REPR is None
        assert np.all(result.R["rank"] == 4)
        # Familia acumulada por defecto: B_k = {1..k}, k = 1..min(r0, r)
        r0 = result.A6["r0"]
        assert len(result.G["blocks"]) == r0
        for k, blk in enumerate(result.G["blocks"], start=1):
            assert list(blk) == list(range(1, k + 1))
        # Estadístico del paper (v4)
        assert result.G["definition"] == "(I - W_k W_k^T) U_{1:k} (paper)"


# ============================================================================
# 10. Rank tag consistency (pipeline vs batch runner)
# ============================================================================

class TestRankTag:

    def test_tags_match(self):
        cases = [
            {"rank_method": "fixed", "fixed_rank": 20},
            {"rank_method": "variance", "var_explained": 0.99,
             "var_max_rank": 50},
            {"rank_method": "reproducibility"},
        ]
        for cdhsa in cases:
            batch_tag = CDHSABatchRunner._rank_tag(cdhsa)
            pipe_tag = _rank_tag(
                cdhsa.get("rank_method", "fixed"),
                cdhsa.get("fixed_rank", 10),
                cdhsa.get("var_explained", 0.99),
                cdhsa.get("var_max_rank", 50),
            )
            assert batch_tag == pipe_tag

    def test_variance_tag_format(self):
        tag = CDHSABatchRunner._rank_tag(
            {"rank_method": "variance", "var_explained": 0.99,
             "var_max_rank": 50}
        )
        assert tag == "var99c50"

    def test_checkpoint_key_includes_rank(self):
        job = {
            "mode": "multi_ss", "n_super_subjects": 5, "session": "session1",
            "tasks": ["a", "b"], "t_start": "0.0", "t_end": "100.0",
            "cdhsa_params": {"L": 10, "rank_method": "variance",
                             "var_explained": 0.99, "var_max_rank": 50},
        }
        key = CDHSABatchRunner._checkpoint_key(job)
        assert "var99c50" in key
        assert "L10" in key


# ============================================================================
# 11. v4 audit fixes: tangente del paper, CV_min, lambda=sigma, /j,
#     Haar QR, mascara de fronteras, S=1, proyeccion nivel 2, rango efectivo
# ============================================================================

def _haar_basis(rng, d, r):
    Z = rng.standard_normal((d, r))
    Q, _ = np.linalg.qr(Z)
    return Q


class TestTangentPaperStatistic:
    """MAJOR 1: L = (I - W_k W_k^T) U_{1:k} (definicion del paper)."""

    def _make_R_A6(self, rng, aligned=True, d=12, r=4):
        W0 = _haar_basis(rng, d, r)
        V = _haar_basis(rng, d, r)
        U = [[W0, W0 if aligned else V],
             [W0, W0 if aligned else V]]
        R = {'U': U, 'rank': np.full((2, 2), r),
             'S': 2, 'C': 2}
        A6 = {'W0': W0, 'r0': r}
        return R, A6

    def test_aligned_local_gives_zero_tangent(self):
        rng = np.random.default_rng(0)
        R, A6 = self._make_R_A6(rng, aligned=True)
        G = cdhsa_tangent_geometry_test(
            R, A6, blocks=[np.arange(1, 5)], opts={'n_perm': 20, 'seed': 0}
        )
        # U_{1:k} == W_k => L = 0 => T_k = 0
        np.testing.assert_allclose(G['T_obs'], 0.0, atol=1e-12)
        assert G['definition'] == "(I - W_k W_k^T) U_{1:k} (paper)"

    def test_rotated_local_gives_nonzero_tangent(self):
        rng = np.random.default_rng(1)
        R, A6 = self._make_R_A6(rng, aligned=False)
        G = cdhsa_tangent_geometry_test(
            R, A6, blocks=[np.arange(1, 5)], opts={'n_perm': 20, 'seed': 0}
        )
        assert np.all(G['T_obs'] > 1e-6)

    def test_L_is_residual_of_local_block(self):
        """Verificacion algebraica directa de la definicion."""
        rng = np.random.default_rng(2)
        d, r = 15, 3
        W0 = _haar_basis(rng, d, r)
        U = _haar_basis(rng, d, r)
        R = {'U': [[U]], 'rank': np.array([[r]]), 'S': 1, 'C': 1}
        A6 = {'W0': W0, 'r0': r}
        L = tangent_components(R, A6, [np.arange(1, r + 1)])
        Wk = W0[:, :r]
        expected = U[:, :r] - Wk @ (Wk.T @ U[:, :r])
        np.testing.assert_allclose(L[0][0][0], expected, atol=1e-12)

    def test_block_exceeding_min_rank_raises(self):
        rng = np.random.default_rng(3)
        d, r = 10, 2
        W0 = _haar_basis(rng, d, 4)
        R = {'U': [[_haar_basis(rng, d, r)]],
             'rank': np.array([[r]]), 'S': 1, 'C': 1}
        A6 = {'W0': W0, 'r0': 4}
        with pytest.raises(ValueError):
            cdhsa_tangent_geometry_test(
                R, A6, blocks=[np.arange(1, 4)], opts={'n_perm': 5}
            )

    def test_default_blocks_are_cumulative(self):
        rng = np.random.default_rng(4)
        R, A6 = self._make_R_A6(rng, aligned=True)
        G = cdhsa_tangent_geometry_test(R, A6, opts={'n_perm': 5, 'seed': 0})
        expected = cumulative_tangent_blocks(4, 4)
        assert len(G['blocks']) == 4
        for got, exp in zip(G['blocks'], expected):
            np.testing.assert_array_equal(got, exp)

    def test_single_subject_degenerate(self):
        rng = np.random.default_rng(5)
        W0 = _haar_basis(rng, 12, 3)
        U = [[W0[:, :1], _haar_basis(rng, 12, 1)]]
        R = {'U': U, 'rank': np.array([[1, 1]]), 'S': 1, 'C': 2}
        A6 = {'W0': W0, 'r0': 3}
        G = cdhsa_tangent_geometry_test(
            R, A6, blocks=[np.array([1])], opts={'n_perm': 10, 'seed': 0}
        )
        assert G['degenerate'] is True
        assert np.all(G['p_maxT'] == 1.0)
        assert G['null_T'].shape[0] == 0


class TestCVMinOrder:
    """MINOR 6: CV_min = mean_folds(min_c), no min_c(mean_folds)."""

    def test_cv_min_is_mean_of_fold_minima(self):
        rng = np.random.default_rng(6)
        d, S, C = 10, 4, 3
        shared = _haar_basis(rng, d, 1)
        U_cell = []
        for s in range(S):
            row = []
            for c in range(C):
                extra = _haar_basis(rng, d, 2)
                row.append(np.concatenate([shared, extra], axis=1))
            U_cell.append(row)
        cv = crossvalidate_common_rank(U_cell, np.array([1, 2, 3]),
                                       {'n_folds': 2, 'seed': 1})
        fold_min = np.nanmin(cv['condition_score'], axis=1)
        np.testing.assert_allclose(
            cv['min_condition'], np.nanmean(fold_min, axis=0), atol=1e-12
        )
        # Orden importa: mean(min) <= min(mean)
        min_of_means = np.nanmin(
            np.nanmean(cv['condition_score'], axis=0), axis=0
        )
        assert np.all(cv['min_condition'] <= min_of_means + 1e-12)
        assert 'fold_min_condition' in cv


class TestLambdaSingularValues:
    """MINOR 12: lambda_j = sigma_j(B) (valor singular), no su cuadrado."""

    def test_lambda_is_sigma_of_B(self):
        rng = np.random.default_rng(7)
        d = 20
        u1 = _haar_basis(rng, d, 1)
        e2 = _haar_basis(rng, d, 1)
        e2 = e2 - u1 * (u1.T @ e2)
        e2 /= np.linalg.norm(e2)
        rho = 0.5
        u2 = rho * u1 + np.sqrt(1 - rho ** 2) * e2
        # B = [u1 u2]/sqrt(2): sigma_1 = sqrt((1+rho)/2)
        W, lam = common_basis_from_U([[u1, u2]], 2)
        expected_sigma = np.sqrt((1 + rho) / 2)
        old_squared = (1 + rho) / 2
        assert np.isclose(lam[0], expected_sigma, atol=1e-10)
        assert not np.isclose(lam[0], old_squared)  # distingue sigma de sigma^2

    def test_lambda_consistent_between_A3_and_a6_helper(self):
        rng = np.random.default_rng(8)
        d, S, C = 12, 3, 2
        U_cell = [[_haar_basis(rng, d, 3) for _ in range(C)]
                  for _ in range(S)]
        W1, lam1 = common_basis_from_U(U_cell, 3)
        # B^T B eigenvalues = sigma^2; lambda debe ser su raiz
        B = np.concatenate([U_cell[s][c] for s in range(S)
                            for c in range(C)], axis=1) / np.sqrt(S * C)
        ev = np.linalg.eigvalsh(B.T @ B)[::-1][:3]
        np.testing.assert_allclose(lam1, np.sqrt(ev), atol=1e-10)


class TestAlignBlockNorm:
    """MINOR 7: align_adj = ||U^T W_k||^2 / j (tamano del bloque comun)."""

    def test_divides_by_block_size_not_rank(self):
        rng = np.random.default_rng(9)
        p, T, L = 3, 120, 4
        X = [[rng.standard_normal((p, T)) for _ in range(2)]
             for _ in range(2)]
        R = cdhsa_A1_A5(X, L, fixed_rank=2, max_common=2)
        A6 = {'W0': R['W'][:, :2], 'r0': 2}
        blocks = [np.array([1]), np.array([1, 2])]
        M = compute_common_mode_metrics(X, L, R, A6, blocks)
        np.testing.assert_allclose(
            M['align_adj'][:, :, 0], M['align_raw'][:, :, 0], atol=1e-14
        )
        np.testing.assert_allclose(
            M['align_adj'][:, :, 1], M['align_raw'][:, :, 1] / 2.0, atol=1e-14
        )
        assert M['align_norm'] == 'block_size'


class TestHaarQR:
    """MINOR 10: correccion de signos QR (Mezzadri 2007)."""

    def test_matches_sign_corrected_qr(self):
        rng = np.random.default_rng(10)
        Z = rng.standard_normal((8, 3))
        Q, Rqr = np.linalg.qr(Z)
        sgns = np.sign(np.diag(Rqr))
        sgns[sgns == 0.0] = 1.0
        rng2 = np.random.default_rng(10)
        got = haar_random_basis(rng2, 8, 3)
        np.testing.assert_allclose(got, Q * sgns, atol=1e-14)

    def test_orthonormal_and_deterministic(self):
        rng = np.random.default_rng(11)
        Q = haar_random_basis(rng, 20, 5)
        np.testing.assert_allclose(Q.T @ Q, np.eye(5), atol=1e-12)
        rng2 = np.random.default_rng(11)
        np.testing.assert_allclose(haar_random_basis(rng2, 20, 5), Q)


def _concat_level1_hankel(rng, members=(40, 35, 45), depth=4, n_ch=2):
    """Hankel de nivel 1 (p, T1) de un stream concatenado + miembros."""
    T_total = int(sum(members))
    stream = rng.standard_normal((n_ch, T_total))
    T1 = T_total - depth + 1
    X1 = np.empty((n_ch * depth, T1))
    for i in range(T1):
        X1[:, i] = stream[:, i:i + depth].ravel()
    return X1, list(members)


class TestBoundaryMasking:
    """MAJOR 3: descartar columnas de nivel 2 que cruzan fronteras."""

    def test_drops_at_most_L_plus_depth_minus_2_per_boundary(self):
        mask = level2_boundary_mask([40, 35, 45], depth=4, L=3)
        K2 = (40 + 35 + 45) - 3 - 2
        assert mask.shape[0] == K2
        # 2 fronteras internas -> a lo sumo 2*(3+4-2) = 10
        assert int((~mask).sum()) <= 2 * (3 + 4 - 2)

    def test_brute_force_agreement(self):
        members = [30, 20, 25, 15]
        depth, L = 4, 3
        T_total = sum(members)
        boundaries = np.cumsum(members)[:-1]
        K2 = T_total - (depth - 1) - (L - 1)
        brute = np.ones(K2, dtype=bool)
        for t in range(K2):
            for b in boundaries:
                if t < b and t + L + depth - 2 >= b:
                    brute[t] = False
        np.testing.assert_array_equal(
            level2_boundary_mask(members, depth, L), brute
        )

    def test_single_member_no_drop(self):
        mask = level2_boundary_mask([100], depth=4, L=3)
        assert bool(mask.all())

    def test_A1A5_masked_equals_manual_column_drop(self):
        rng = np.random.default_rng(12)
        X1, members = _concat_level1_hankel(rng, members=(40, 35, 45),
                                            depth=4, n_ch=2)
        L = 3
        mask = level2_boundary_mask(members, 4, L)
        R = cdhsa_A1_A5([[X1]], L, fixed_rank=3, max_common=3,
                        col_masks=[[mask]])
        # Referencia densa: columnas enmascaradas eliminadas
        H2 = build_block_hankel(X1, L)
        Hm = H2[:, mask]
        np.testing.assert_allclose(
            R['hankel_norm'][0, 0], np.linalg.norm(Hm, 'fro'), rtol=1e-10
        )
        assert R['boundary_dropped'][0, 0] == int((~mask).sum())
        U_ref = np.linalg.svd(Hm / np.linalg.norm(Hm, 'fro'),
                              full_matrices=False)[0][:, :3]
        ov = np.linalg.norm(R['U'][0][0].T @ U_ref) ** 2 / 3
        assert ov > 0.999

    def test_energy_masked_equals_manual(self):
        rng = np.random.default_rng(13)
        X1, members = _concat_level1_hankel(rng, members=(50, 40),
                                            depth=4, n_ch=2)
        L = 3
        mask = level2_boundary_mask(members, 4, L)
        R = cdhsa_A1_A5([[X1, X1]], L, fixed_rank=3, max_common=3,
                        col_masks=[[mask, mask]])
        A6 = {'W0': R['W'][:, :2], 'r0': 2}
        M = compute_common_mode_metrics(
            [[X1, X1]], L, R, A6, [np.array([1, 2])]
        )
        H2 = build_block_hankel(X1, L)
        Hm = H2[:, mask]
        Wk = A6['W0'][:, :2]
        ref = np.linalg.norm(Hm.T @ Wk, 'fro') ** 2
        np.testing.assert_allclose(M['energy_abs'][0, 0, 0], ref, rtol=1e-10)

    def test_reproducibility_conflict_raises(self):
        rng = np.random.default_rng(14)
        X1, members = _concat_level1_hankel(rng, members=(40, 35),
                                            depth=4, n_ch=2)
        mask = level2_boundary_mask(members, 4, 3)
        with pytest.raises(NotImplementedError):
            cdhsa_A1_A5([[X1]], 3, rank_method='reproducibility',
                        col_masks=[[mask]])


class TestSingleSubjectPipeline:
    """MAJOR 4: pipeline ejecutable con S=1 (modo descriptivo)."""

    def test_run_cdhsa_S1(self):
        rng = np.random.default_rng(15)
        S, C, p, T, L = 1, 3, 3, 300, 8
        fs = 200.0
        t = np.arange(T) / fs
        a10 = rng.normal(size=p); a10 /= np.linalg.norm(a10)
        X = [[None] * C for _ in range(S)]
        for c in range(C):
            z = np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
            X[0][c] = np.outer(a10, z) + 0.05 * rng.normal(size=(p, T))
        cfg = CDHSAConfig(fixed_rank=3, max_common=3, a6_n_null=5,
                          bc_n_perm=10, tangent_n_perm=10,
                          d_max_specific=2)
        result = run_cdhsa(X, L, cfg)
        assert result.A6['single_subject'] is True
        assert result.A6['cv'] is None
        assert result.A6['r0'] >= 1
        assert result.BC is not None
        assert result.BC['skipped_reason'] is not None
        assert not result.BC['sig_energy_maxF'].any()
        assert result.G['degenerate'] is True
        assert result.D is not None
        assert result.D_loso is None
        assert result.REPR is None

    def test_replica_consistency_S1_raises(self):
        rng = np.random.default_rng(16)
        d, r = 10, 2
        U = _haar_basis(rng, d, r)
        R = {'U': [[U, U]], 'rank': np.array([[r, r]]), 'S': 1, 'C': 2}
        A6 = {'W0': U, 'r0': r}
        with pytest.raises(ValueError):
            cdhsa_replica_consistency(R, A6)


class TestProjectLevel2:
    """MAJOR 5: proyeccion correcta al espacio de nivel 2."""

    def test_matches_dense_projection(self):
        rng = np.random.default_rng(17)
        p, T, L = 3, 50, 4
        H1 = rng.standard_normal((p, T))
        W = _haar_basis(rng, p * L, 2)
        alpha = project_level2(H1, W, L)
        H2 = build_block_hankel(H1, L)
        np.testing.assert_allclose(alpha, W.T @ H2, atol=1e-12)

    def test_mask_zeroes_crossing_columns(self):
        rng = np.random.default_rng(18)
        p, T, L = 2, 60, 3
        H1 = rng.standard_normal((p, T))
        W = _haar_basis(rng, p * L, 1)
        mask = np.ones(T - L + 1, dtype=bool)
        mask[[5, 6, 20]] = False
        alpha = project_level2(H1, W, L, col_mask=mask)
        assert np.all(alpha[:, ~mask] == 0.0)
        np.testing.assert_allclose(
            alpha[:, mask], (W.T @ build_block_hankel(H1, L))[:, mask],
            atol=1e-12,
        )

    def test_wrong_dims_raise(self):
        rng = np.random.default_rng(19)
        H1 = rng.standard_normal((3, 50))
        W = _haar_basis(rng, 3 * 5 + 1, 2)  # p*L + 1 filas
        with pytest.raises(ValueError):
            project_level2(H1, W, 5)


class TestEffectiveRank:
    """MINOR 8: estimador de rango efectivo X% (Remark 3.6)."""

    def test_fixed_plus_effective_rank_varies(self):
        rng = np.random.default_rng(20)

        def _lowrank(d, K, r):
            A = rng.standard_normal((d, r)) / np.sqrt(d)
            Z = rng.standard_normal((r, K))
            return A @ Z + 1e-4 * rng.standard_normal((d, K)) / np.sqrt(d)

        X = [[_lowrank(30, 200, 2), _lowrank(30, 200, 6)],
             [_lowrank(30, 200, 2), _lowrank(30, 200, 6)]]
        R = cdhsa_A1_A5(X, 3, rank_method='fixed', fixed_rank=4,
                        max_common=2, effective_rank=True,
                        var_explained=0.99, var_max_rank=30)
        assert 'rank_effective' in R
        assert np.all(R['rank'] == 4)
        # La Hankel de nivel 2 de una matriz de rango r tiene rango
        # ~ r*L: r_eff(2, L=3) ~ 6 << r_eff(6, L=3) ~ 18.
        assert 4 <= R['rank_effective'][0, 0] <= 9
        assert R['rank_effective'][0, 1] >= 12
        assert not R['rank_effective_censored'].any()

    def test_variance_effective_equals_selected(self):
        rng = np.random.default_rng(21)

        def _lowrank(d, K, r):
            A = rng.standard_normal((d, r)) / np.sqrt(d)
            Z = rng.standard_normal((r, K))
            return A @ Z + 1e-4 * rng.standard_normal((d, K)) / np.sqrt(d)

        X = [[_lowrank(30, 200, 2), _lowrank(30, 200, 6)]]
        R = cdhsa_A1_A5(X, 3, rank_method='variance', var_explained=0.99,
                        var_max_rank=15, max_common=2, effective_rank=True)
        np.testing.assert_array_equal(R['rank_effective'], R['rank'])

    def test_bc_rank_outcome_effective(self):
        from src.cdhsa.b_energy import cdhsa_BC_condition_tests
        rng = np.random.default_rng(22)

        def _lowrank(d, K, r):
            A = rng.standard_normal((d, r)) / np.sqrt(d)
            Z = rng.standard_normal((r, K))
            return A @ Z + 1e-4 * rng.standard_normal((d, K)) / np.sqrt(d)

        X = [[_lowrank(30, 200, 2), _lowrank(30, 200, 6)],
             [_lowrank(30, 200, 2), _lowrank(30, 200, 6)]]
        R = cdhsa_A1_A5(X, 3, rank_method='fixed', fixed_rank=4,
                        max_common=2, effective_rank=True,
                        var_explained=0.99, var_max_rank=15)
        A6 = {'W0': R['W'][:, :2], 'r0': 2}
        BC = cdhsa_BC_condition_tests(
            X, 3, R, A6, opts={'n_perm': 20, 'rank_outcome': 'effective'}
        )
        assert BC['rank_outcome'] == 'effective'
        assert not BC['rank_test_degenerate']

    def test_bc_rank_outcome_effective_requires_flag(self):
        from src.cdhsa.b_energy import cdhsa_BC_condition_tests
        rng = np.random.default_rng(23)
        X = [[rng.standard_normal((10, 100)) for _ in range(2)]
             for _ in range(2)]
        R = cdhsa_A1_A5(X, 3, fixed_rank=3, max_common=2)
        A6 = {'W0': R['W'][:, :2], 'r0': 2}
        with pytest.raises(ValueError):
            cdhsa_BC_condition_tests(
                X, 3, R, A6, opts={'n_perm': 10, 'rank_outcome': 'effective'}
            )


class TestReplicaDeformationV4:
    """replica_consistency usa el estadistico tangente del paper (v4)."""

    def test_deformation_paper_definition(self):
        rng = np.random.default_rng(24)
        d, r = 12, 3
        W0 = _haar_basis(rng, d, r)
        U = [[W0, _haar_basis(rng, d, r)],
             [W0, _haar_basis(rng, d, r)],
             [W0, _haar_basis(rng, d, r)]]
        R = {'U': U, 'rank': np.full((3, 2), r), 'S': 3, 'C': 2}
        A6 = {'W0': W0, 'r0': r}
        REPR = cdhsa_replica_consistency(R, A6)
        de = REPR['deformation']
        assert de['definition'] == "(I - W_k W_k^T) U_{1:k} (paper)"
        # Default: familia acumulada k = 1..min(r0, min r_sc)
        assert len(de['blocks']) == r
        assert de['T_replica'].shape == (3, r)
        # Sujetos con U_{1:k} == W_k en la condicion 1 y distinto en la
        # 2: T^(s) > 0 para todo s
        assert np.all(de['T_replica'] > 1e-12)


class TestExtractModeMapRagged:
    """MAJOR 5: build_mode_map con W_specific ragged + mascaras guardadas.

    Regresion de la rama ragged (antes: NameError por W_all indefinido
    al sobrescribir per_condition).
    """

    def test_build_mode_map_ragged_with_masks(self, tmp_path):
        import json
        rng = np.random.default_rng(99)
        S, C, p, depth, L = 2, 2, 3, 4, 5
        members = [80, 70]
        T1 = sum(members) - depth + 1
        stream = rng.standard_normal((p, sum(members)))
        H1 = np.empty((p * depth, T1))
        for i in range(T1):
            H1[:, i] = stream[:, i:i + depth].ravel()
        mask = level2_boundary_mask(members, depth, L)

        d = p * depth * L
        W0 = np.linalg.qr(rng.standard_normal((d, 20)))[0]
        W_spec = [W0[:, :3], W0[:, :5]]
        lam_spec = [np.array([0.9, 0.5, 0.3]),
                    np.array([0.8, 0.6, 0.4, 0.2, 0.1])]

        json.dump({"tasks": ["tareaA", "tareaB"], "n_super_subjects": S,
                   "n_channels_common": p, "depth_common": depth},
                  open(tmp_path / "hankel_info.json", "w"))
        json.dump({"L": L, "fixed_rank": 4, "rank_method": "fixed"},
                  open(tmp_path / "config.json", "w"))
        arrays = {"D__r_specific": np.array([3, 5]),
                 "D__prevalence_contrast": np.array([0.42, 0.17])}
        for c in range(C):
            arrays[f"D__W_specific__{c}"] = W_spec[c]
            arrays[f"D__lambda_specific__{c}"] = lam_spec[c]
        for s in range(S):
            for c in range(C):
                arrays[f"colmask_ss{s + 1}_c{c + 1}"] = mask
        np.savez(tmp_path / "cdhsa_arrays.npz", **arrays)
        np.savez(tmp_path / "hankel_matrices.npz",
                 **{f"H_ss{s + 1}_c{c + 1}": H1
                    for s in range(S) for c in range(C)})

        from src.cdhsa.extract_mode_indices import build_mode_map
        mm = build_mode_map(tmp_path, top_n=2)
        assert mm["npz_structure"]["W_specific_ragged"] is True
        assert mm["metadata"]["level2_depth_L"] == L
        assert mm["conditions"]["tareaA"]["total_specific_modes"] == 3
        assert mm["conditions"]["tareaB"]["total_specific_modes"] == 5
        entry = mm["per_super_subject_task"][0]
        assert entry["boundary_masked_columns"] == int((~mask).sum())
        assert "project_level2" in entry["projection_code"]
        # La proyeccion documentada es consistente
        W_sel = W_spec[0][:, [0, 1]]
        alpha = project_level2(H1, W_sel, L, col_mask=mask)
        assert np.all(alpha[:, ~mask] == 0.0)
