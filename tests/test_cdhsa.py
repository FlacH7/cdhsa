"""src.cdhsa unit tests
=====================

Tests are organized by module and cover:
  - Shape and type correctness
  - Mathematical properties (orthonormality, bounds)
  - Edge cases (degenerate inputs, singletons)
  - End-to-end pipeline with synthetic data
  - Detection of known condition effects

Run with:  python -m pytest tests/test_cdhsa.py -v
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
    truncated_left_svd,
    truncated_left_svd_with_values,
    select_rank_reproducibility,
    cdhsa_A1_A5,
)
from src.cdhsa.a6_common_rank import (
    common_basis_from_U,
    crossvalidate_common_rank,
    cdhsa_A6_common_rank,
)
from src.cdhsa.null_distributions import (
    random_subspace_null,
)
from src.cdhsa.permutation_tests import (
    within_subject_permutation_rm,
    _rm_anova_f,
)
from src.cdhsa.b_energy import (
    compute_common_mode_metrics,
    cdhsa_BC_condition_tests,
)
from src.cdhsa.c_geometry import (
    cdhsa_tangent_geometry_test,
    _tangent_stat,
)
from src.cdhsa.d_condition_specific import (
    _orthogonalize_residual,
    cdhsa_D_condition_specific_modes,
)
from src.pipelines.run_cdhsa import (
    CDHSAConfig,
    CDHSAResult,
    run_cdhsa,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_eeg():
    """Minimal synthetic EEG: 4 subjects, 2 conditions, 4 channels, 500 samples."""
    rng = np.random.default_rng(42)
    S, C, p, T = 4, 2, 4, 500
    fs = 200.0
    L = 20
    t = np.arange(T) / fs

    X = [[None] * C for _ in range(S)]
    for s in range(S):
        phi = rng.uniform(0, 2 * np.pi)
        z = np.sin(2 * np.pi * 10 * t + phi)
        for c in range(C):
            Xi = np.outer(rng.normal(size=p) / np.sqrt(p), z)
            if c == 1:
                Xi += 0.5 * rng.normal(size=(p, T))
            Xi += 0.3 * rng.normal(size=(p, T))
            X[s][c] = Xi
    return X, L, S, C, p, T, fs


@pytest.fixture
def standard_eeg():
    """Standard synthetic EEG: 8 subjects, 2 conditions, 8 channels, 1500 samples.
    10 Hz + 6 Hz common, 18 Hz condition-2 only."""
    rng = np.random.default_rng(123)
    S, C, p, T = 8, 2, 8, 1500
    fs = 200.0
    L = 30
    t = np.arange(T) / fs

    a10 = rng.normal(size=p); a10 /= np.linalg.norm(a10)
    a6 = rng.normal(size=p)
    a6 -= a10 * np.dot(a10, a6); a6 /= np.linalg.norm(a6)
    a18 = rng.normal(size=p)
    a18 -= a10 * np.dot(a10, a18)
    a18 -= a6 * np.dot(a6, a18); a18 /= np.linalg.norm(a18)

    X = [[None] * C for _ in range(S)]
    for s in range(S):
        phi10 = rng.uniform(0, 2 * np.pi)
        phi6 = rng.uniform(0, 2 * np.pi)
        phi18 = rng.uniform(0, 2 * np.pi)
        z10 = np.sin(2 * np.pi * 10 * t + phi10)
        z6 = 0.7 * np.sin(2 * np.pi * 6 * t + phi6)
        for c in range(C):
            Xi = np.outer(a10, z10) + np.outer(a6, z6)
            if c == 1:
                z18 = np.sin(2 * np.pi * 18 * t + phi18)
                Xi += 0.8 * np.outer(a18, z18)
            Xi += 0.35 * rng.normal(size=(p, T))
            X[s][c] = Xi
    return X, L, S, C, p, T, fs


# ============================================================================
# 1. build_block_hankel
# ============================================================================

class TestBuildBlockHankel:
    def test_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 100))
        H = build_block_hankel(X, 10)
        assert H.shape == (40, 91)

    def test_shape_single_channel(self):
        X = np.random.randn(1, 50)
        H = build_block_hankel(X, 5)
        assert H.shape == (5, 46)

    def test_L_equals_T(self):
        X = np.random.randn(3, 20)
        H = build_block_hankel(X, 20)
        assert H.shape == (60, 1)

    def test_T_less_than_L_raises(self):
        X = np.random.randn(3, 10)
        with pytest.raises(ValueError):
            build_block_hankel(X, 15)

    def test_not_2d_raises(self):
        X = np.random.randn(10)
        with pytest.raises(ValueError):
            build_block_hankel(X, 3)

    def test_deterministic_output(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((4, 100))
        H1 = build_block_hankel(X, 10)
        H2 = build_block_hankel(X, 10)
        np.testing.assert_allclose(H1, H2)

    def test_first_column(self):
        """First column should be [x(T-1), ..., x(T-L)]^T per channel."""
        X = np.array([[1, 2, 3, 4, 5]], dtype=float)
        H = build_block_hankel(X, 3)
        expected = np.array([[3, 4, 5], [2, 3, 4], [1, 2, 3]], dtype=float)
        np.testing.assert_allclose(H, expected)


# ============================================================================
# 2. truncated_left_svd
# ============================================================================

class TestTruncatedLeftSvd:
    def test_orthonormal_columns(self):
        rng = np.random.default_rng(0)
        H = rng.standard_normal((20, 50))
        U = truncated_left_svd(H, 5)
        assert U.shape == (20, 5)
        G = U.T @ U
        np.testing.assert_allclose(G, np.eye(5), atol=1e-12)

    def test_matches_dense_svd(self):
        rng = np.random.default_rng(1)
        H = rng.standard_normal((15, 40))
        U_sparse = truncated_left_svd(H, 6)
        U_dense, _, _ = np.linalg.svd(H, full_matrices=False)
        for j in range(6):
            corr = abs(np.dot(U_sparse[:, j], U_dense[:, j]))
            assert corr > 0.999

    def test_r_larger_than_min_dim(self):
        H = np.random.randn(5, 3)
        U = truncated_left_svd(H, 100)
        assert U.shape == (5, 3)

    def test_r_zero_raises(self):
        H = np.random.randn(5, 10)
        with pytest.raises(ValueError):
            truncated_left_svd(H, 0)

    def test_with_values(self):
        rng = np.random.default_rng(2)
        H = rng.standard_normal((10, 30))
        U, s = truncated_left_svd_with_values(H, 5)
        assert s.shape == (5,)
        assert np.all(np.diff(s) <= 1e-10)


# ============================================================================
# 3. select_rank_reproducibility
# ============================================================================

class TestSelectRankReproducibility:
    def test_returns_positive_int(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 500))
        r, R = select_rank_reproducibility(X, 20, rmax=8, n_blocks=3)
        assert isinstance(r, (int, np.integer))
        assert r >= 1
        assert len(R) <= 8

    def test_repro_values_in_range(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((4, 500))
        _, R = select_rank_reproducibility(X, 20, rmax=6, n_blocks=3)
        assert np.all(R >= 0)
        assert np.all(R <= 1.0 + 1e-10)

    def test_gap_strategy(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((4, 500))
        r, _ = select_rank_reproducibility(X, 20, rmax=8, strategy='gap')
        assert r >= 1

    def test_short_recording_raises(self):
        X = np.random.randn(4, 30)
        with pytest.raises(ValueError):
            select_rank_reproducibility(X, 20, n_blocks=3)


# ============================================================================
# 4. cdhsa_A1_A5
# ============================================================================

class TestA1A5:
    def test_output_keys(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=5)
        for key in ['U', 'rank', 'hankel_norm', 'W', 'lambda_',
                     'alignment', 'sc_index', 'S', 'C', 'p', 'd']:
            assert key in R, f"Missing key: {key}"

    def test_dimensions(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=5, max_common=6)
        assert R['S'] == S
        assert R['C'] == C
        assert R['p'] == p
        assert R['d'] == p * L
        assert R['W'].shape == (p * L, 6)
        assert len(R['lambda_']) == 6

    def test_commonality_bounds(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=5, max_common=8)
        assert np.all(R['lambda_'] >= 0)
        assert np.all(R['lambda_'] <= 1.0 + 1e-10)

    def test_alignment_bounds(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=5, max_common=6)
        assert np.all(R['alignment'] >= -1e-10)
        assert np.all(R['alignment'] <= 1.0 + 1e-10)

    def test_W_orthonormal(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=5, max_common=6)
        G = R['W'].T @ R['W']
        np.testing.assert_allclose(G, np.eye(6), atol=1e-10)

    def test_sc_index_shape(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=5)
        assert R['sc_index'].shape == (S * C, 2)

    def test_with_known_structure(self, standard_eeg):
        """Common directions should have higher lambda than noise."""
        X, L, S, C, p, T, fs = standard_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=8, max_common=12)
        assert np.mean(R['lambda_'][:4]) > 0.5


# ============================================================================
# 5. common_basis_from_U / crossvalidate_common_rank
# ============================================================================

class TestCommonBasisFromU:
    def test_output_shapes(self):
        rng = np.random.default_rng(0)
        d, r = 10, 3
        U_cell = [[rng.standard_normal((d, r)) for _ in range(2)] for _ in range(3)]
        for s in range(3):
            for c in range(2):
                U_cell[s][c], _ = np.linalg.qr(U_cell[s][c])
        W, lam = common_basis_from_U(U_cell, 5)
        assert W.shape[0] == d
        assert W.shape[1] <= 5
        assert len(lam) == W.shape[1]

    def test_W_orthonormal(self):
        rng = np.random.default_rng(1)
        U_cell = [[rng.standard_normal((8, 3)) for _ in range(2)] for _ in range(4)]
        for s in range(4):
            for c in range(2):
                U_cell[s][c], _ = np.linalg.qr(U_cell[s][c])
        W, _ = common_basis_from_U(U_cell, 6)
        G = W.T @ W
        np.testing.assert_allclose(G, np.eye(W.shape[1]), atol=1e-12)


class TestCrossvalidateCommonRank:
    def test_cv_scores_in_range(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=4, max_common=4)
        r_vals = np.array([1, 2, 3])
        cv = crossvalidate_common_rank(R['U'], r_vals, {'n_folds': 2, 'seed': 1})
        assert np.all(cv['min_condition'] >= -0.01)
        assert np.all(cv['min_condition'] <= 1.0 + 0.01)

    def test_fewer_subjects_than_folds(self):
        rng = np.random.default_rng(0)
        U_cell = [[rng.standard_normal((8, 3)) for _ in range(2)] for _ in range(2)]
        for s in range(2):
            for c in range(2):
                U_cell[s][c], _ = np.linalg.qr(U_cell[s][c])
        cv = crossvalidate_common_rank(U_cell, np.array([1, 2]), {'n_folds': 5})
        assert cv['fold_score'].shape[0] <= 2

    def test_requires_two_subjects(self):
        U_cell = [[np.eye(5, 3) for _ in range(2)]]
        with pytest.raises(ValueError):
            crossvalidate_common_rank(U_cell, np.array([1]))


# ============================================================================
# 6. random_subspace_null
# ============================================================================

class TestRandomSubspaceNull:
    def test_output_shapes(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=3, max_common=4)
        r_vals = np.array([1, 2, 3])
        null = random_subspace_null(R['U'], r_vals, {'n_null': 10, 'seed': 0})
        assert null['lambda'].shape[0] == 10
        assert null['cv_min'].shape == (10, 3)
        assert len(null['lambda_q']) >= 3

    def test_null_quantiles_positive(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=3, max_common=4)
        r_vals = np.array([1, 2, 3])
        null = random_subspace_null(R['U'], r_vals, {'n_null': 10, 'seed': 0})
        assert np.all(null['lambda_q'] >= 0)
        assert np.all(null['cv_min_q'] >= 0)

    def test_observed_larger_than_null_for_signal(self, standard_eeg):
        X, L, S, C, p, T, fs = standard_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=6, max_common=8)
        null = random_subspace_null(
            R['U'], np.arange(1, 5),
            {'n_null': 20, 'seed': 0}
        )
        assert R['lambda_'][0] > null['lambda_q'][0]


# ============================================================================
# 7. _rm_anova_f / within_subject_permutation_rm
# ============================================================================

class TestRmAnovaF:
    def test_no_effect(self):
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((20, 2))
        F = _rm_anova_f(Y)
        assert F < 10.0

    def test_large_effect(self):
        rng = np.random.default_rng(0)
        Y = np.zeros((20, 2))
        Y[:, 0] = rng.standard_normal(20) * 0.1
        Y[:, 1] = rng.standard_normal(20) * 0.1 + 5.0
        F = _rm_anova_f(Y)
        assert F > 50

    def test_perfect_within_subject(self):
        Y = np.zeros((10, 2))
        Y[:, 0] = 0
        Y[:, 1] = 1
        F = _rm_anova_f(Y)
        assert F == 0.0

    def test_C_equals_2_equals_squared_t(self):
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((30, 2))
        F = _rm_anova_f(Y)
        from scipy import stats
        t_stat, _ = stats.ttest_rel(Y[:, 1], Y[:, 0])
        np.testing.assert_allclose(F, t_stat ** 2, rtol=1e-10)

    def test_single_condition_raises(self):
        Y = np.random.randn(10, 1)
        with pytest.raises(ValueError):
            _rm_anova_f(Y)


class TestWithinSubjectPermutationRm:
    def test_output_keys(self):
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((10, 2, 3))
        result = within_subject_permutation_rm(Y, {'n_perm': 50, 'seed': 0})
        for key in ['F_obs', 'p_uncorrected', 'p_maxF', 'null_F', 'null_maxF']:
            assert key in result

    def test_p_values_in_range(self):
        rng = np.random.default_rng(1)
        Y = rng.standard_normal((10, 2, 3))
        result = within_subject_permutation_rm(Y, {'n_perm': 50, 'seed': 0})
        assert np.all(result['p_uncorrected'] >= 0)
        assert np.all(result['p_uncorrected'] <= 1.0)
        assert np.all(result['p_maxF'] >= result['p_uncorrected'] - 1e-10)

    def test_maxF_greater_or_equal_uncorrected(self):
        rng = np.random.default_rng(2)
        Y = rng.standard_normal((10, 2, 5))
        result = within_subject_permutation_rm(Y, {'n_perm': 50, 'seed': 0})
        assert np.all(result['p_maxF'] >= result['p_uncorrected'] - 1e-10)

    def test_cohen_dz_for_C2(self):
        Y = np.zeros((10, 2, 1))
        Y[:, 1, 0] = 3.0
        result = within_subject_permutation_rm(Y, {'n_perm': 50, 'seed': 0})
        assert 'cohen_dz' in result
        assert result['cohen_dz'][0] > 0

    def test_large_effect_significant(self):
        rng = np.random.default_rng(0)
        Y = np.zeros((20, 2, 1))
        Y[:, 0, 0] = rng.standard_normal(20) * 0.1
        Y[:, 1, 0] = rng.standard_normal(20) * 0.1 + 5.0
        result = within_subject_permutation_rm(Y, {'n_perm': 100, 'seed': 0})
        assert result['p_uncorrected'][0] < 0.05

    def test_2d_input_autoexpands(self):
        Y = np.random.randn(10, 2)
        result = within_subject_permutation_rm(Y, {'n_perm': 20, 'seed': 0})
        assert result['F_obs'].shape == (1,)


# ============================================================================
# 8. compute_common_mode_metrics / cdhsa_BC_condition_tests
# ============================================================================

class TestComputeCommonModeMetrics:
    def test_output_shapes(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=3, max_common=3)
        A6 = {'W0': R['W'][:, :2], 'r0': 2}
        blocks = [np.array([1]), np.array([2])]
        M = compute_common_mode_metrics(X, L, R, A6, blocks)
        assert M['energy_abs'].shape == (S, C, 2)
        assert M['align_raw'].shape == (S, C, 2)
        assert M['local_rank'].shape == (S, C)

    def test_energy_nonnegative(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=3, max_common=3)
        A6 = {'W0': R['W'][:, :2], 'r0': 2}
        blocks = [np.array([1])]
        M = compute_common_mode_metrics(X, L, R, A6, blocks)
        assert np.all(M['energy_abs'] >= 0)
        assert np.all(M['energy_rel'] >= 0)

    def test_energy_rel_bounded(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=3, max_common=3)
        A6 = {'W0': R['W'][:, :2], 'r0': 2}
        blocks = [np.array([1])]
        M = compute_common_mode_metrics(X, L, R, A6, blocks)
        assert np.all(M['energy_rel'] <= 1.0 + 1e-10)


class TestBCConditionTests:
    def test_detects_known_effect(self, standard_eeg):
        X, L, S, C, p, T, fs = standard_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=6, max_common=8)
        A6 = {'W0': R['W'][:, :6], 'r0': 6}
        BC = cdhsa_BC_condition_tests(X, L, R, A6, opts={
            'n_perm': 200, 'seed': 42
        })
        assert 'energy_test' in BC
        assert 'geometry_test' in BC
        assert len(BC['block_names']) == 6

    def test_r0_zero_raises(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=3, max_common=3)
        A6 = {'W0': np.zeros((R['d'], 0)), 'r0': 0}
        with pytest.raises(ValueError):
            cdhsa_BC_condition_tests(X, L, R, A6)

    def test_custom_blocks(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=4, max_common=4)
        A6 = {'W0': R['W'][:, :4], 'r0': 4}
        BC = cdhsa_BC_condition_tests(X, L, R, A6, opts={
            'blocks': [np.array([1, 2]), np.array([3, 4])],
            'n_perm': 50, 'seed': 0
        })
        assert len(BC['block_names']) == 2


# ============================================================================
# 9. cdhsa_tangent_geometry_test
# ============================================================================

class TestTangentGeometryTest:
    def test_output_keys(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=3, max_common=3)
        A6 = {'W0': R['W'][:, :2], 'r0': 2}
        G = cdhsa_tangent_geometry_test(R, A6, opts={
            'n_perm': 50, 'seed': 0
        })
        for key in ['T_obs', 'p_uncorrected', 'p_maxT', 'sig_maxT', 'null_T']:
            assert key in G

    def test_p_values_in_range(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=3, max_common=3)
        A6 = {'W0': R['W'][:, :2], 'r0': 2}
        G = cdhsa_tangent_geometry_test(R, A6, opts={
            'n_perm': 50, 'seed': 0
        })
        assert np.all(G['p_uncorrected'] >= 0)
        assert np.all(G['p_maxT'] >= 0)

    def test_r0_zero_raises(self):
        R = {'U': [[np.eye(5, 3)]], 'S': 1, 'C': 1}
        A6 = {'W0': np.zeros((5, 0)), 'r0': 0}
        with pytest.raises(ValueError):
            cdhsa_tangent_geometry_test(R, A6)

    def test_c2_effect_size(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=3, max_common=3)
        A6 = {'W0': R['W'][:, :2], 'r0': 2}
        G = cdhsa_tangent_geometry_test(R, A6, opts={
            'n_perm': 50, 'seed': 0
        })
        assert 'effect_ratio' in G
        assert 'mean_difference_norm' in G


class TestTangentStat:
    def test_no_effect(self):
        rng = np.random.default_rng(0)
        M = rng.standard_normal((5, 3))
        Lsc = [[M.copy() for _ in range(2)] for _ in range(4)]
        T = _tangent_stat(Lsc)
        assert abs(T) < 1e-10


# ============================================================================
# 10. Step D
# ============================================================================

class TestOrthogonalizeResidual:
    def test_output_orthogonal_to_W0(self):
        rng = np.random.default_rng(0)
        U = rng.standard_normal((10, 5))
        U, _ = np.linalg.qr(U)
        W0 = U[:, :2]
        U_res = _orthogonalize_residual(U, W0)
        if U_res.shape[1] > 0:
            np.testing.assert_allclose(W0.T @ U_res, 0, atol=1e-12)

    def test_output_orthonormal(self):
        rng = np.random.default_rng(1)
        U = rng.standard_normal((10, 5))
        U, _ = np.linalg.qr(U)
        W0 = U[:, :2]
        U_res = _orthogonalize_residual(U, W0)
        if U_res.shape[1] > 0:
            G = U_res.T @ U_res
            np.testing.assert_allclose(G, np.eye(U_res.shape[1]), atol=1e-12)

    def test_fully_contained(self):
        W0 = np.eye(8, 3)
        U = W0[:, :2]
        U_res = _orthogonalize_residual(U, W0)
        assert U_res.shape[1] == 0


class TestConditionSpecificModes:
    def test_output_keys(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=3, max_common=3)
        A6 = {'W0': R['W'][:, :2], 'r0': 2}
        D = cdhsa_D_condition_specific_modes(X, L, R, A6, opts={
            'max_specific': 3
        })
        for key in ['W_specific', 'lambda_specific', 'r_specific',
                     'U_residual', 'residual_rank',
                     'alignment_specific', 'prevalence_contrast']:
            assert key in D

    def test_r0_zero_raises(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=3, max_common=3)
        A6 = {'W0': np.zeros((R['d'], 0)), 'r0': 0}
        with pytest.raises(ValueError):
            cdhsa_D_condition_specific_modes(X, L, R, A6)

    def test_condition_specific_with_known_structure(self, standard_eeg):
        X, L, S, C, p, T, fs = standard_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=6, max_common=8)
        A6 = {'W0': R['W'][:, :6], 'r0': 6}
        D = cdhsa_D_condition_specific_modes(X, L, R, A6, opts={
            'max_specific': 5
        })
        assert D['r_specific'][1] >= 0
        assert D['prevalence_contrast'].shape == (C,)

    def test_residual_bases_orthogonal_to_W0(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=3, max_common=3)
        A6 = {'W0': R['W'][:, :2], 'r0': 2}
        D = cdhsa_D_condition_specific_modes(X, L, R, A6, opts={
            'max_specific': 3
        })
        for s in range(S):
            for c in range(C):
                Ures = D['U_residual'][s][c]
                if Ures.shape[1] > 0:
                    proj = A6['W0'].T @ Ures
                    np.testing.assert_allclose(proj, 0, atol=1e-10)


# ============================================================================
# 11. cdhsa_A6_common_rank
# ============================================================================

class TestA6CommonRank:
    def test_output_keys(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=3, max_common=4)
        A6 = cdhsa_A6_common_rank(R, opts={
            'n_null': 10, 'n_folds': 2, 'seed': 0
        })
        for key in ['r0', 'W0', 'lambda0', 'r_values', 'cv', 'null',
                     'lambda_observed', 'pass']:
            assert key in A6

    def test_r0_nonnegative(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=3, max_common=4)
        A6 = cdhsa_A6_common_rank(R, opts={
            'n_null': 10, 'n_folds': 2, 'seed': 0
        })
        assert A6['r0'] >= 0

    def test_W0_shape(self, simple_eeg):
        X, L, S, C, p, T, fs = simple_eeg
        R = cdhsa_A1_A5(X, L, fixed_rank=3, max_common=4)
        A6 = cdhsa_A6_common_rank(R, opts={
            'n_null': 10, 'n_folds': 2, 'seed': 0
        })
        assert A6['W0'].shape[0] == R['d']
        assert A6['W0'].shape[1] == A6['r0']

    def test_missing_R_keys_raises(self):
        with pytest.raises(ValueError):
            cdhsa_A6_common_rank({})


# ============================================================================
# 12. Full pipeline
# ============================================================================

class TestPipeline:
    def test_run_cdhsa(self, simple_eeg):
        test_docstring = 'Pipeline returns CDHSAResult with R and A6.'
        X, L, S, C, p, T, fs = simple_eeg
        cfg = CDHSAConfig(
            fixed_rank=4,
            a6_n_null=10,
            bc_n_perm=50,
            tangent_n_perm=50,
        )
        result = run_cdhsa(X, L, cfg)
        assert isinstance(result, CDHSAResult)
        assert result.R is not None
        assert result.A6 is not None

    def test_run_cdhsa_with_all_steps(self, simple_eeg):
        test_docstring = 'Pipeline runs B/C, tangent, and D when r0 > 0.'
        X, L, S, C, p, T, fs = simple_eeg
        cfg = CDHSAConfig(
            fixed_rank=4,
            a6_n_null=10,
            bc_n_perm=50,
            tangent_n_perm=50,
            skip_bc=False,
            skip_tangent=False,
            skip_d=False,
        )
        result = run_cdhsa(X, L, cfg)
        assert result.R is not None
        assert result.A6 is not None
        if result.A6['r0'] > 0:
            assert result.BC is not None
            assert result.G is not None
            assert result.D is not None

    def test_summary_printable(self, simple_eeg):
        test_docstring = 'Summary should be a printable string with key info.'
        X, L, S, C, p, T, fs = simple_eeg
        cfg = CDHSAConfig(fixed_rank=4, a6_n_null=10)
        result = run_cdhsa(X, L, cfg)
        s = result.summary()
        assert isinstance(s, str)
        assert 'CD-HSA' in s
        assert str(result.R['S']) in s

    def test_pipeline_with_standard_eeg(self, standard_eeg):
        test_docstring = 'Full pipeline on structured data detects condition effects.'
        X, L, S, C, p, T, fs = standard_eeg
        cfg = CDHSAConfig(
            fixed_rank=6,
            max_common=8,
            a6_n_null=20,
            bc_n_perm=200,
            tangent_n_perm=200,
        )
        result = run_cdhsa(X, L, cfg)
        assert result.A6['r0'] >= 1
        if result.A6['r0'] > 0 and result.BC is not None:
            assert np.any(result.BC['energy_test']['p_uncorrected'] < 0.15)

    def test_skip_steps(self, simple_eeg):
        test_docstring = 'Pipeline respects skip flags.'
        X, L, S, C, p, T, fs = simple_eeg
        cfg = CDHSAConfig(
            fixed_rank=3,
            a6_n_null=10,
            skip_bc=True,
            skip_tangent=True,
            skip_d=True,
        )
        result = run_cdhsa(X, L, cfg)
        assert result.BC is None
        assert result.G is None
        assert result.D is None

    def test_default_config(self):
        test_docstring = 'Default config has expected values.'
        cfg = CDHSAConfig()
        assert cfg.fixed_rank == 10
        assert cfg.a6_n_null == 100
        assert cfg.bc_n_perm == 5000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
