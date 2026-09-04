import sys, os
import warnings
from math import comb

import numpy as np
from scipy.spatial.distance import cdist

from pted.utils import (
    PermutationResolutionWarning,
    allocate_columns,
    two_tailed_p,
    simulation_based_calibration_histogram,
    pit_plot,
    hdp_coverage_test,
    permutation_energy_test,
    _cdist,
    _draw_labels,
    _evaluate_statistic,
    _index_like,
    _prepare_statistic,
    _random_permutation,
    _resolve_n_columns,
    _to_scalar,
)

try:
    import torch
except ImportError:
    torch = None

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

import pytest


def _require_backend(backend):
    if backend == "torch" and torch is None:
        pytest.skip("torch not installed")
    if backend == "jax" and jax is None:
        pytest.skip("jax not installed")


def test_two_tailed_p():

    # assert np.isclose(two_tailed_p(4, 6), 1.0), "p-value at mode should be 1.0"

    assert two_tailed_p(0.01, 10) < 0.01, "p-value should be less than 0.01 for small chi2"
    assert two_tailed_p(100, 10) < 0.01, "p-value should be less than 0.01 for large chi2"
    assert two_tailed_p(10, 10) > 0.01, "p-value should be close to 0.5 for chi2 near mode"

    assert two_tailed_p(0, 10) < 0.01
    assert two_tailed_p(1e-25, 1000) < 0.01


def test_sbc_histogram(monkeypatch):

    ranks = np.random.uniform(size=1000)
    simulation_based_calibration_histogram(ranks, "sbc_hist.pdf", bins=10)
    os.remove("sbc_hist.pdf")

    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    with pytest.warns():
        simulation_based_calibration_histogram(ranks, "sbc_hist.pdf", bins=10)


def test_pit_plot_no_matplotlib(monkeypatch):

    pvals = np.random.uniform(size=50)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    with pytest.warns(UserWarning, match="matplotlib"):
        pit_plot(pvals, "pit_no_mpl.pdf")


def test_hdp_coverage_test():
    np.random.seed(42)
    # Null is true
    ground_truth = np.random.normal(loc=0, scale=1, size=128)
    posterior_samples = np.random.normal(loc=0, scale=1, size=(1024, 128))
    pvalue = hdp_coverage_test(ground_truth, posterior_samples)
    assert 1e-3 <= pvalue <= 0.999, "p-value should be between 0 and 1"

    # Posterior is biased
    posterior_samples = np.random.normal(loc=5, scale=1, size=(1024, 128))
    pvalue = hdp_coverage_test(ground_truth, posterior_samples)
    assert pvalue < 0.01, "p-value should be small for poorly calibrated posterior samples"

    # Posterior is underconfident
    posterior_samples = np.random.normal(loc=0, scale=2, size=(1024, 128))
    pvalue = hdp_coverage_test(ground_truth, posterior_samples)
    assert pvalue < 0.01, "p-value should be small for poorly calibrated posterior samples"
    pvalue = hdp_coverage_test(ground_truth, posterior_samples, two_tailed=False)
    assert pvalue > 0.01, "p-value should not be small for underconfident and one_tailed test"


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_to_scalar(backend):
    if backend == "torch" and torch is None:
        pytest.skip("torch not installed")
    if backend == "jax" and jax is None:
        pytest.skip("jax not installed")

    if backend == "numpy":
        x = np.array(5)

    elif backend == "torch":
        x = torch.tensor(5)

    elif backend == "jax":
        x = jnp.array(5)

    # Test with scalar
    assert _to_scalar(x, backend) == 5


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_random_permutation(backend):
    if backend == "torch" and torch is None:
        pytest.skip("torch not installed")
    if backend == "jax" and jax is None:
        pytest.skip("jax not installed")

    D = np.arange(16).reshape(4, 4)
    if backend == "torch":
        D = torch.tensor(D)
    elif backend == "jax":
        D = jnp.array(D)

    permuted_D = _random_permutation(D, backend=backend)

    assert set(np.array(permuted_D.flatten()).tolist()) == set(
        np.array(D.flatten()).tolist()
    ), "Permutation should contain all original elements"


# ---------------------------------------------------------------------------
# Rectangular / batched permutation backend
# ---------------------------------------------------------------------------


def _unbiased_energy(x, y):
    """Energy statistic from first principles, within-group means over
    distinct pairs only."""
    n1, n2 = len(x), len(y)
    dxx, dyy = cdist(x, x), cdist(y, y)
    exx = (dxx.sum() - np.trace(dxx)) / (n1 * (n1 - 1)) if n1 > 1 else 0.0
    eyy = (dyy.sum() - np.trace(dyy)) / (n2 * (n2 - 1)) if n2 > 1 else 0.0
    return (n1 * n2 / (n1 + n2)) * (2 * cdist(x, y).mean() - exx - eyy)


def _brute_block_means(z, small_indicator, cols, n):
    """Energy statistic by explicit block sums over the rectangular matrix."""
    D = cdist(z, z[cols])
    is_s = small_indicator.astype(bool)
    rows_s, rows_l = np.flatnonzero(is_s), np.flatnonzero(~is_s)
    ks, kl = np.flatnonzero(is_s[cols]), np.flatnonzero(~is_s[cols])
    n_s, n_l = len(rows_s), len(rows_l)

    def block(rows, kk, drop_self):
        tot = cnt = 0
        for i in rows:
            for k in kk:
                if drop_self and cols[k] == i:
                    continue
                tot += D[i, k]
                cnt += 1
        return tot, cnt

    s_ss, c_ss = block(rows_s, ks, True)
    s_ll, c_ll = block(rows_l, kl, True)
    s_sl, c_sl = block(rows_s, kl, False)
    s_ls, c_ls = block(rows_l, ks, False)
    mu_ss = 0.0 if c_ss == 0 else s_ss / c_ss
    mu_ll = 0.0 if c_ll == 0 else s_ll / c_ll
    parts = [s_sl / c_sl] + ([s_ls / c_ls] if c_ls else [])
    return (n_s * n_l / n) * (2 * sum(parts) / len(parts) - mu_ss - mu_ll)


ALLOCATIONS = [
    # n1, n2, n_columns, expected regime
    (100, 100, 200, "full"),
    (3, 40, 43, "full"),
    (1, 60, 12, "singleton"),  # c < n/2 -> the lone point stays out of C
    (1, 20, 15, "singleton"),  # c > n/2 -> it moves into C instead
    (6, 80, 40, "small_group_in_C"),
    (40, 50, 30, "proportional"),
    (40, 50, 3, "proportional"),
]


@pytest.mark.parametrize("n1,n2,c,regime", ALLOCATIONS)
def test_allocate_columns_regimes(n1, n2, c, regime):
    """Each size combination lands in the intended regime with a consistent
    set of columns."""
    alloc = allocate_columns(n1, n2, c, rng=0)
    cols = alloc["cols"]
    n = n1 + n2

    assert alloc["regime"] == regime
    assert cols.shape == (c,)
    assert len(np.unique(cols)) == c, "columns must be distinct points"
    assert cols.min() >= 0 and cols.max() < n
    assert np.all(np.diff(cols) > 0), "columns are returned sorted"

    # c_small must be the number of columns actually belonging to the small group
    assert np.sum(np.isin(cols, alloc["small_idx"])) == alloc["c_small"]
    assert alloc["c_small"] + alloc["c_large"] == c
    assert alloc["c_large"] >= 1, "the large group must keep at least one column"
    assert alloc["exact_within"] == (alloc["c_small"] == alloc["n_small"] or alloc["n_small"] == 1)


def test_allocate_columns_reference_size():
    """reference_size counts the label assignments the subgroup can reach."""
    for n1, n2, c, _ in ALLOCATIONS:
        alloc = allocate_columns(n1, n2, c, rng=1)
        n, n_s, c_s = n1 + n2, alloc["n_small"], alloc["c_small"]
        expected = comb(c, c_s) * comb(n - c, n_s - c_s)
        assert alloc["reference_size"] == min(expected, 10**15)


def test_allocate_columns_errors():
    """allocate_columns is the single gate on sample and column counts."""
    with pytest.raises(ValueError, match="at least 2 columns"):
        allocate_columns(10, 10, 1)
    with pytest.raises(ValueError, match="exceeds the pooled sample size"):
        allocate_columns(10, 10, 21)
    with pytest.raises(ValueError, match="both samples need at least one point"):
        allocate_columns(0, 10, 5)
    # and it is reached through the public API, not bypassed by the mapping
    with pytest.raises(ValueError, match="at least 2 columns"):
        permutation_energy_test(np.zeros((5, 2)), np.zeros((5, 2)), permutations=1, n_columns=1)
    with pytest.raises(ValueError, match="both samples need at least one point"):
        permutation_energy_test(np.zeros((0, 2)), np.zeros((5, 2)), permutations=1)


@pytest.mark.parametrize("n1,n2,c,regime", ALLOCATIONS)
def test_draw_labels_stays_in_subgroup(n1, n2, c, regime):
    """Permutations never move a label across the C / C^c boundary, so the
    per-group column counts -- and hence every normalising constant -- are
    fixed. This is what makes the observed labelling exchangeable with the
    permuted ones."""
    rng = np.random.default_rng(3)
    alloc = allocate_columns(n1, n2, c, rng)
    n = n1 + n2
    base = np.zeros(n)
    base[alloc["small_idx"]] = 1.0
    in_c = np.zeros(n, bool)
    in_c[alloc["cols"]] = True

    U = _draw_labels(base, np.flatnonzero(in_c), np.flatnonzero(~in_c), 200, rng.spawn(2))

    assert np.all(np.isin(U, [0.0, 1.0]))
    assert np.all(U.sum(1) == alloc["n_small"]), "small group size is preserved"
    assert np.all(U[:, alloc["cols"]].sum(1) == alloc["c_small"]), "column counts are preserved"
    # and the labels really do move around within each part
    if alloc["reference_size"] > 100:
        assert len(np.unique(U, axis=0)) > 1


@pytest.mark.parametrize("n1,n2,c,regime", ALLOCATIONS)
def test_statistic_matches_brute_force_block_means(n1, n2, c, regime):
    """The bilinear-form shortcut agrees with explicit block sums, for the
    observed labelling and for permuted ones."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((n1, 4))
    y = rng.standard_normal((n2, 4)) + 0.4
    z = np.vstack([x, y])

    alloc = allocate_columns(n1, n2, c, rng)
    D = cdist(z, z[alloc["cols"]])
    prep = _prepare_statistic(D, alloc, "numpy")

    U = np.concatenate(
        [
            prep["base_small"][None, :],
            _draw_labels(prep["base_small"], prep["idx_c"], prep["idx_o"], 5, rng.spawn(2)),
        ]
    )
    got = _evaluate_statistic(prep, U)
    expect = [_brute_block_means(z, U[b], alloc["cols"], n1 + n2) for b in range(len(U))]
    assert np.allclose(got, expect, atol=1e-9)


@pytest.mark.parametrize("n1,n2", [(30, 30), (7, 23), (1, 40), (2, 2)])
def test_full_matrix_is_the_energy_distance(n1, n2):
    """With every point a column, the statistic is exactly the energy
    distance with within-group means over distinct pairs."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal((n1, 3))
    y = rng.standard_normal((n2, 3)) + 0.6
    test_stat, _ = permutation_energy_test(x, y, permutations=0)
    assert np.isclose(test_stat, _unbiased_energy(x, y), atol=1e-9)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_statistic_backend_agreement(backend):
    """numpy, torch and jax evaluate the same statistic for the same labels."""
    _require_backend(backend)
    rng = np.random.default_rng(5)
    n1, n2, c = 40, 60, 30
    x = rng.standard_normal((n1, 5))
    y = rng.standard_normal((n2, 5)) + 0.3
    z = np.vstack([x, y])

    alloc = allocate_columns(n1, n2, c, rng)
    U = None
    results = {}
    for be in ("numpy", backend):
        zb = {"numpy": lambda: z, "torch": lambda: torch.tensor(z), "jax": lambda: jnp.array(z)}[
            be
        ]()
        D = _cdist(zb, zb[_index_like(alloc["cols"], zb, be)], be)
        prep = _prepare_statistic(D, alloc, be)
        if U is None:
            U = np.concatenate(
                [
                    prep["base_small"][None, :],
                    _draw_labels(prep["base_small"], prep["idx_c"], prep["idx_o"], 8, rng.spawn(2)),
                ]
            )
        results[be] = _evaluate_statistic(prep, U)
    # jax defaults to float32, so compare with a float32-appropriate tolerance
    assert np.allclose(results["numpy"], results[backend], rtol=1e-4, atol=1e-4)


def test_draw_labels_independent_of_batch_size():
    """The two label streams are consumed row by row, so splitting a run into
    batches yields bit-identical permutations."""
    base = np.r_[np.ones(5), np.zeros(15)]
    idx_c, idx_o = np.arange(12), np.arange(12, 20)
    whole = _draw_labels(base, idx_c, idx_o, 10, np.random.default_rng(9).spawn(2))
    rngs = np.random.default_rng(9).spawn(2)
    split = np.concatenate([_draw_labels(base, idx_c, idx_o, k, rngs) for k in (3, 3, 4)])
    assert np.array_equal(whole, split)


def test_statistic_independent_of_batch_size():
    """Batching is purely a memory/throughput knob: with the same seed it draws
    the same permutations, so the statistics agree to within the reordering
    that a different matmul shape costs in floating point."""
    rng_kwargs = dict(permutations=64, n_columns=30, rng=12345)
    x = np.random.default_rng(0).standard_normal((50, 4))
    y = np.random.default_rng(1).standard_normal((70, 4))
    ref_stat, ref_perm = permutation_energy_test(x, y, batch_size=1, **rng_kwargs)
    for batch in (7, 64, 1000):
        stat, perm = permutation_energy_test(x, y, batch_size=batch, **rng_kwargs)
        assert stat == ref_stat
        assert np.allclose(
            perm, ref_perm, rtol=1e-10, atol=1e-12
        ), f"batch_size={batch} changed the null draws"


def test_resolve_n_columns():
    # chunk_size reproduces the number of distance columns it used to request
    assert _resolve_n_columns(1000, 1000, chunk_size=100) == 200
    assert _resolve_n_columns(200, 30, chunk_size=50) == 80
    # covering both datasets falls back to the full matrix
    assert _resolve_n_columns(20, 30, chunk_size=40) == 50
    # n_columns is taken literally, capped at the pooled size
    assert _resolve_n_columns(20, 30, n_columns=17) == 17
    assert _resolve_n_columns(20, 30, n_columns=500) == 50
    # neither means the full matrix
    assert _resolve_n_columns(20, 30) == 50

    with pytest.raises(ValueError, match="not both"):
        _resolve_n_columns(20, 30, chunk_size=5, n_columns=5)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        _resolve_n_columns(20, 30, chunk_size=0)
    # a nonsense n_columns passes through here and is rejected by
    # allocate_columns, which owns every column-count check
    assert _resolve_n_columns(20, 30, n_columns=1) == 1


def test_singleton_picks_the_larger_side_of_C():
    """A lone point sits inside or outside C, whichever leaves more label
    assignments reachable. Without this the reference set collapses to 1 as c
    approaches n, and the test can only ever return p = 1."""
    n2 = 200
    n = 1 + n2
    for c in range(2, n):
        alloc = allocate_columns(1, n2, c, rng=0)
        assert alloc["regime"] == "singleton"
        assert alloc["c_small"] == (1 if c > n - c else 0)
        assert alloc["reference_size"] == max(c, n - c)
    # so the reference set never drops below half the pooled sample
    worst = min(allocate_columns(1, n2, c, rng=0)["reference_size"] for c in range(2, n))
    assert worst >= n // 2


def test_permutation_resolution_warning():
    """Column counts that starve the permutation group are flagged."""
    rng = np.random.default_rng(2)

    # singleton: only a genuinely tiny pooled sample is short of assignments now
    with pytest.warns(PermutationResolutionWarning, match="single point"):
        permutation_energy_test(
            rng.standard_normal((1, 3)), rng.standard_normal((10, 3)), permutations=100, n_columns=5
        )

    # small_group_in_C: too FEW columns is the failure mode here
    with pytest.warns(PermutationResolutionWarning, match="use more columns"):
        permutation_energy_test(
            rng.standard_normal((2, 3)), rng.standard_normal((30, 3)), permutations=100, n_columns=5
        )

    # sensible column counts are quiet, at either end of the singleton range
    x, y = rng.standard_normal((1, 3)), rng.standard_normal((60, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error", PermutationResolutionWarning)
        permutation_energy_test(x, y, permutations=100, n_columns=8)
        permutation_energy_test(x, y, permutations=100, n_columns=55)
