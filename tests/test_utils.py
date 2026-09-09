import sys, os
import warnings
from math import comb

import numpy as np
from scipy.spatial.distance import cdist

from scipy.stats import kstwo

from pted.utils import (
    _band_accept_probability,
    _lattice_band,
    _sampled_label_blocks,
    _lattice_counts,
    _enumerate_label_blocks,
    PermutationResolutionWarning,
    allocate_landmarks,
    two_tailed_p,
    simulation_based_calibration_histogram,
    pit_plot,
    hdp_coverage_test,
    permutation_energy_test,
    _cdist,
    _label_drawer,
    _evaluate_statistic,
    _index_like,
    _prepare_statistic,
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

BACKENDS = ["numpy", "torch", "jax"]


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


def _brute_block_means(z, small_indicator, landmarks, n):
    """Energy statistic by explicit block sums over the rectangular matrix."""
    D = cdist(z, z[landmarks])
    is_s = small_indicator.astype(bool)
    rows_s, rows_l = np.flatnonzero(is_s), np.flatnonzero(~is_s)
    ks, kl = np.flatnonzero(is_s[landmarks]), np.flatnonzero(~is_s[landmarks])
    n_s, n_l = len(rows_s), len(rows_l)

    def block(rows, kk, drop_self):
        tot = cnt = 0
        for i in rows:
            for k in kk:
                if drop_self and landmarks[k] == i:
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
    # n1, n2, n_landmarks, expected regime
    (100, 100, 200, "full"),
    (3, 40, 43, "full"),
    (1, 60, 12, "singleton"),  # c < n/2 -> the lone point stays out of C
    (1, 20, 15, "singleton"),  # c > n/2 -> it moves into C instead
    (6, 80, 40, "small_group_in_L"),
    (40, 50, 30, "proportional"),
    (40, 50, 3, "proportional"),
]


@pytest.mark.parametrize("n1,n2,c,regime", ALLOCATIONS)
def test_allocate_landmarks_regimes(n1, n2, c, regime):
    """Each size combination lands in the intended regime with a consistent
    set of columns."""
    alloc = allocate_landmarks(n1, n2, c, rng=0)
    landmarks = alloc["landmarks"]
    n = n1 + n2

    assert alloc["regime"] == regime
    assert landmarks.shape == (c,)
    assert len(np.unique(landmarks)) == c, "columns must be distinct points"
    assert landmarks.min() >= 0 and landmarks.max() < n
    assert np.all(np.diff(landmarks) > 0), "columns are returned sorted"

    # n_small_landmarks must be the number of columns actually belonging to the small group
    assert np.sum(np.isin(landmarks, alloc["small_idx"])) == alloc["n_small_landmarks"]
    assert alloc["n_small_landmarks"] + alloc["n_large_landmarks"] == c
    assert alloc["n_large_landmarks"] >= 1, "the large group must keep at least one column"
    assert alloc["exact_within"] == (
        alloc["n_small_landmarks"] == alloc["n_small"] or alloc["n_small"] == 1
    )


def test_allocate_landmarks_reference_size():
    """reference_size counts the label assignments the subgroup can reach."""
    for n1, n2, c, _ in ALLOCATIONS:
        alloc = allocate_landmarks(n1, n2, c, rng=1)
        n, n_s, c_s = n1 + n2, alloc["n_small"], alloc["n_small_landmarks"]
        expected = comb(c, c_s) * comb(n - c, n_s - c_s)
        assert alloc["reference_size"] == min(expected, 10**15)


def test_allocate_landmarks_errors():
    """allocate_landmarks is the single gate on sample and column counts."""
    with pytest.raises(ValueError, match="at least 2 landmarks"):
        allocate_landmarks(10, 10, 1)
    with pytest.raises(ValueError, match="exceeds the pooled sample size"):
        allocate_landmarks(10, 10, 21)
    with pytest.raises(ValueError, match="both samples need at least one point"):
        allocate_landmarks(0, 10, 5)
    # and it is reached through the public API, not bypassed by the mapping
    with pytest.raises(ValueError, match="at least 2 landmarks"):
        permutation_energy_test(np.zeros((5, 2)), np.zeros((5, 2)), permutations=1, n_landmarks=1)
    with pytest.raises(ValueError, match="both samples need at least one point"):
        permutation_energy_test(np.zeros((0, 2)), np.zeros((5, 2)), permutations=1)


def _prep_for(n1, n2, c, seed=3, backend="numpy"):
    """An allocation plus its prepared statistic, on the given backend."""
    rng = np.random.default_rng(seed)
    alloc = allocate_landmarks(n1, n2, c, rng)
    z = rng.standard_normal((n1 + n2, 4))
    if backend == "torch":
        z = torch.tensor(z)
    elif backend == "jax":
        z = jnp.array(z)
    zc = z if alloc["regime"] == "full" else z[_index_like(alloc["landmarks"], z, backend)]
    return alloc, _prepare_statistic(_cdist(z, zc, backend), alloc, backend)


@pytest.mark.parametrize("n1,n2,c,regime", ALLOCATIONS)
def test_draw_labels_stays_in_subgroup(n1, n2, c, regime):
    """Permutations never move a label across the L / L^c boundary, so the
    per-group landmark counts -- and hence every normalising constant -- are
    fixed. This is what makes the observed labelling exchangeable with the
    permuted ones."""
    alloc, prep = _prep_for(n1, n2, c)
    U = np.asarray(_label_drawer(prep, np.random.default_rng(17))(200))

    assert np.all(np.isin(U, [0.0, 1.0]))
    assert np.all(U.sum(1) == alloc["n_small"]), "small group size is preserved"
    assert np.all(
        U[:, alloc["landmarks"]].sum(1) == alloc["n_small_landmarks"]
    ), "column counts are preserved"
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

    alloc = allocate_landmarks(n1, n2, c, rng)
    D = cdist(z, z[alloc["landmarks"]])
    prep = _prepare_statistic(D, alloc, "numpy")

    U = np.concatenate(
        [
            prep["base_small"][None, :],
            _label_drawer(prep, np.random.default_rng(11))(5),
        ]
    )
    got = _evaluate_statistic(prep, U)
    expect = [_brute_block_means(z, U[b], alloc["landmarks"], n1 + n2) for b in range(len(U))]
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

    alloc = allocate_landmarks(n1, n2, c, rng)
    U = None
    results = {}
    for be in ("numpy", backend):
        zb = {"numpy": lambda: z, "torch": lambda: torch.tensor(z), "jax": lambda: jnp.array(z)}[
            be
        ]()
        D = _cdist(zb, zb[_index_like(alloc["landmarks"], zb, be)], be)
        prep = _prepare_statistic(D, alloc, be)
        if U is None:
            U = np.concatenate(
                [
                    prep["base_small"][None, :],
                    _label_drawer(prep, np.random.default_rng(13))(8),
                ]
            )
        results[be] = _evaluate_statistic(prep, U)
    # jax defaults to float32, so compare with a float32-appropriate tolerance
    assert np.allclose(results["numpy"], results[backend], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("n1,n2,c", [(40, 50, 30), (6, 80, 40), (1, 60, 12)])
def test_device_draw_labels_stays_in_subgroup(backend, n1, n2, c):
    """The on-device draw obeys the same subgroup constraints as the numpy
    one -- the labels moved to the accelerator, the guarantees did not."""
    _require_backend(backend)
    alloc, prep = _prep_for(n1, n2, c, backend=backend)
    U = np.asarray(_label_drawer(prep, np.random.default_rng(23))(128))

    assert np.all(np.isin(U, [0.0, 1.0]))
    assert np.all(U.sum(1) == alloc["n_small"])
    assert np.all(U[:, alloc["landmarks"]].sum(1) == alloc["n_small_landmarks"])
    if alloc["reference_size"] > 100:
        assert len(np.unique(U, axis=0)) > 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_label_draws_are_reproducible(backend):
    """A fixed rng pins the draw, and a run yields exactly `permutations` valid
    rows in blocks no larger than batch_size."""
    _require_backend(backend)
    _, prep = _prep_for(20, 30, 50, backend=backend)

    a = np.asarray(_label_drawer(prep, np.random.default_rng(5))(40))
    b = np.asarray(_label_drawer(prep, np.random.default_rng(5))(40))
    assert np.array_equal(a, b), "the same rng must give the same draw"

    for batch in (7, 40, 1000):
        rng = np.random.default_rng(5)
        blocks = [np.asarray(x) for x in _sampled_label_blocks(prep, 40, batch, rng)]
        assert sum(len(x) for x in blocks) == 40
        assert max(len(x) for x in blocks) <= batch
        rows = np.concatenate(blocks)
        assert np.all(rows.sum(1) == prep["base_small"].sum())


def test_evaluate_is_independent_of_batch_size():
    """Splitting one set of labellings across batches must not change their
    statistics, beyond the reordering a different matmul shape costs in
    floating point. (Which labellings get drawn does depend on batch_size --
    the draw is a running stream -- but every batching is an exact test.)"""
    x = np.random.default_rng(0).standard_normal((50, 4))
    y = np.random.default_rng(1).standard_normal((70, 4))
    alloc, prep = _prep_for(50, 70, 30)
    U = _label_drawer(prep, np.random.default_rng(2))(64)

    whole = _evaluate_statistic(prep, U)
    for batch in (1, 7, 64):
        pieces = np.concatenate(
            [_evaluate_statistic(prep, U[s : s + batch]) for s in range(0, len(U), batch)]
        )
        assert np.allclose(pieces, whole, rtol=1e-10, atol=1e-12), f"batch_size={batch}"


def test_every_batch_size_gives_a_valid_test():
    """batch_size stays a free knob: any value returns the requested number of
    null statistics and a p-value in range."""
    x = np.random.default_rng(0).standard_normal((50, 4))
    y = np.random.default_rng(1).standard_normal((70, 4))
    for batch in (1, 7, 64, 1000):
        stat, perm = permutation_energy_test(
            x, y, permutations=64, n_landmarks=30, rng=12345, batch_size=batch
        )
        assert len(perm) == 64 and np.all(np.isfinite(perm))


def test_n_landmarks_covering_the_sample_is_the_full_test():
    """None, or any count at or above the pooled size, is the exact test."""
    x = np.random.default_rng(0).standard_normal((20, 3))
    y = np.random.default_rng(1).standard_normal((30, 3))
    ref = permutation_energy_test(x, y, permutations=32, rng=5)
    for m in (50, 500):
        stat, perm = permutation_energy_test(x, y, permutations=32, n_landmarks=m, rng=5)
        assert stat == ref[0] and np.array_equal(perm, ref[1])


def test_singleton_picks_the_larger_side_of_C():
    """A lone point sits inside or outside C, whichever leaves more label
    assignments reachable. Without this the reference set collapses to 1 as c
    approaches n, and the test can only ever return p = 1."""
    n2 = 200
    n = 1 + n2
    for c in range(2, n):
        alloc = allocate_landmarks(1, n2, c, rng=0)
        assert alloc["regime"] == "singleton"
        assert alloc["n_small_landmarks"] == (1 if c > n - c else 0)
        assert alloc["reference_size"] == max(c, n - c)
    # so the reference set never drops below half the pooled sample
    worst = min(allocate_landmarks(1, n2, c, rng=0)["reference_size"] for c in range(2, n))
    assert worst >= n // 2


def test_permutation_resolution_warning():
    """Column counts that starve the permutation group are flagged."""
    rng = np.random.default_rng(2)

    # singleton: only a genuinely tiny pooled sample is short of assignments now
    with pytest.warns(PermutationResolutionWarning, match="single point"):
        permutation_energy_test(
            rng.standard_normal((1, 3)),
            rng.standard_normal((10, 3)),
            permutations=100,
            n_landmarks=5,
        )

    # small_group_in_L: too FEW columns is the failure mode here
    with pytest.warns(PermutationResolutionWarning, match="use more landmarks"):
        permutation_energy_test(
            rng.standard_normal((2, 3)),
            rng.standard_normal((30, 3)),
            permutations=100,
            n_landmarks=5,
        )

    # sensible column counts are quiet, at either end of the singleton range
    x, y = rng.standard_normal((1, 3)), rng.standard_normal((60, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error", PermutationResolutionWarning)
        permutation_energy_test(x, y, permutations=100, n_landmarks=8)
        permutation_energy_test(x, y, permutations=100, n_landmarks=55)


# ---------------------------------------------------------------------------
# PIT plot confidence bands
# ---------------------------------------------------------------------------


def _band_rejects(pvals, t, lower, upper):
    counts = _lattice_counts(np.asarray(pvals), t)[0] / len(pvals)
    return bool(np.any((counts < lower - 1e-12) | (counts > upper + 1e-12)))


def _ks_rejects(pvals, d_crit):
    n = len(pvals)
    s = np.sort(pvals)
    i = np.arange(1, n + 1)
    return max(np.max(i / n - s), np.max(s - (i - 1) / n)) > d_crit


def test_lattice_band_is_deterministic_and_pinches_at_the_corners():
    """The band tracks the binomial variance of the ECDF count, so it is far
    tighter at the ends than a constant-half-width KS band."""
    n, confidence = 100, 0.95
    t, lower, upper, _, _ = _lattice_band(n, 151, confidence)
    again, lower2, _, _, _ = _lattice_band(n, 151, confidence)
    assert np.array_equal(t, again) and np.array_equal(lower, lower2)

    assert np.all(lower <= upper)
    half = (upper - lower) / 2
    assert half[0] < half[len(t) // 2] and half[-1] < half[len(t) // 2], "must pinch at the ends"
    assert half[np.searchsorted(t, 0.05)] < kstwo.ppf(confidence, n)


@pytest.mark.parametrize("n,lattice", [(100, 151), (100, 1000), (60, 41), (100, None)])
def test_lattice_band_level_is_exact(n, lattice):
    """The escape probability is computed by a forward recursion over the
    counting process, not estimated, so the band's level is known exactly --
    and it lands on the target rather than being conservative."""
    confidence = 0.95
    _, _, _, _, achieved = _lattice_band(n, lattice, confidence)
    assert abs((1 - achieved) - (1 - confidence)) < 0.002, f"achieved {1 - achieved}"


def test_band_recursion_matches_brute_force():
    """The forward recursion agrees with simulating the null directly."""
    n, L, trials = 40, 61, 40000
    t, lo, hi, _, _ = _lattice_band(n, L, 0.95)
    lower, upper = np.round(lo * n).astype(int), np.round(hi * n).astype(int)
    exact = _band_accept_probability(t, n, lower, upper)

    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(0, trials, 4000):
        p = rng.integers(1, L + 1, size=(4000, n)) / L
        counts = _lattice_counts(p, t)
        hits += int(np.sum(np.all((counts >= lower) & (counts <= upper), axis=1)))
    mc = hits / trials
    assert abs(exact - mc) < 4 * np.sqrt(mc * (1 - mc) / trials), f"{exact} vs {mc}"


def test_lattice_band_holds_level_against_simulation():
    """Sanity check the level end to end, on the coarse lattice a real
    coverage test produces -- where an order-statistic band rejects ~60%."""
    n, lattice, confidence, trials = 100, 151, 0.95, 4000
    t, lower, upper, _, _ = _lattice_band(n, lattice, confidence)
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(trials):
        p = rng.integers(1, lattice + 1, size=n) / lattice
        hits += _band_rejects(p, t, lower, upper)
    rate = hits / trials  # se = 0.0034
    assert abs(rate - (1 - confidence)) < 0.015, f"rejected {rate}"


def test_lattice_band_beats_ks_on_tail_deviations():
    """More power against the p-value shapes a miscalibrated posterior makes,
    on the coarse lattice a real coverage test produces."""
    n, L, trials = 100, 151, 1500
    t, lower, upper, _, _ = _lattice_band(n, L, 0.95)
    d_crit = kstwo.ppf(0.95, n)
    rng = np.random.default_rng(4)
    for a, b in ((0.7, 1.0), (0.85, 0.85)):  # skewed low, and U-shaped
        draws = [np.ceil(rng.beta(a, b, n) * L) / L for _ in range(trials)]
        ks = np.mean([_ks_rejects(p, d_crit) for p in draws])
        lat = np.mean([_band_rejects(p, t, lower, upper) for p in draws])
        assert lat > ks, f"Beta({a},{b}): lattice {lat:.3f} did not beat KS {ks:.3f}"


def test_pit_plot_writes_a_file(tmp_path):
    rng = np.random.default_rng(0)
    for label, pvals, lattice in (
        ("continuous", rng.random(100), None),
        ("lattice", rng.integers(1, 152, size=100) / 151, 151),
    ):
        out = tmp_path / f"pit_{label}.pdf"
        pit_plot(pvals, str(out), lattice=lattice)
        assert out.exists()


def test_pit_plot_warns_when_discrete_pvalues_have_no_lattice(tmp_path):
    """Discrete p-values need their lattice declared; without it the band is
    built from the wrong null. Saying so is better than silently over-rejecting."""
    rng = np.random.default_rng(1)
    coarse = rng.integers(1, 201, size=100) / 200  # 199 permutations

    with pytest.warns(UserWarning, match="no lattice was given"):
        pit_plot(coarse, str(tmp_path / "coarse.pdf"))

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pit_plot(coarse, str(tmp_path / "told.pdf"), lattice=200)
        pit_plot(rng.random(100), str(tmp_path / "cont.pdf"))


# ---------------------------------------------------------------------------
# Exhaustive enumeration of small permutation groups
# ---------------------------------------------------------------------------


def test_small_groups_are_enumerated_not_sampled():
    """Asking for more permutations than the subgroup has members walks the
    whole group instead, returning reference_size - 1 null statistics."""
    x = np.random.default_rng(0).standard_normal((1, 3))
    y = np.random.default_rng(1).standard_normal((60, 3))
    reference = allocate_landmarks(1, 60, 61, rng=0)["reference_size"]
    assert reference == 61

    _, permute = permutation_energy_test(x, y, permutations=1000, rng=0)
    assert len(permute) == reference - 1, "should have enumerated the group"

    # ask for fewer than the group holds and it samples, as before
    _, permute = permutation_energy_test(x, y, permutations=30, rng=0)
    assert len(permute) == 30


def test_enumeration_covers_the_group_exactly_once():
    """Every assignment the subgroup reaches appears once, bar the observed."""
    n1, n2, m = 1, 12, 5
    alloc = allocate_landmarks(n1, n2, m, rng=0)
    z = np.random.default_rng(0).standard_normal((n1 + n2, 2))
    prep = _prepare_statistic(cdist(z, z[alloc["landmarks"]]), alloc, "numpy")

    rows = np.concatenate(list(_enumerate_label_blocks(prep, 4)))
    assert len(rows) == alloc["reference_size"] - 1
    assert len(np.unique(rows, axis=0)) == len(rows), "no assignment repeats"
    assert np.all(rows.sum(1) == alloc["n_small"]), "group size preserved"
    assert np.all(rows[:, alloc["landmarks"]].sum(1) == alloc["n_small_landmarks"])
    assert not np.any(np.all(rows == prep["base_small"], axis=1)), "observed excluded"


def test_enumerated_pvalues_are_exactly_lattice_uniform():
    """Enumeration removes the tie inflation that sampling a small group
    causes, so P(p <= t) sits on the lattice instead of below it."""
    from pted.pted import pted as pted_api

    nsamp, trials = 40, 6000  # group of 41, so p lands on multiples of 1/41
    rng = np.random.default_rng(5)
    pvals = np.empty(trials)
    for t in range(trials):
        loc, sd = rng.standard_normal(2) * 3, rng.uniform(1, 3, size=2)
        pvals[t] = pted_api(
            rng.normal(loc, sd, size=(1, 2)),
            rng.normal(loc, sd, size=(nsamp, 2)),
            permutations=500,
            two_tailed=False,
            rng=rng,
        )
    R = nsamp + 1
    assert len(np.unique(pvals)) <= R
    # a valid p-value on an R-point lattice has P(p <= t) = floor(tR)/R
    for t in (0.1, 0.2, 0.5):
        expected = np.floor(t * R) / R
        observed = np.mean(pvals <= t)
        assert abs(observed - expected) < 4 * np.sqrt(
            expected * (1 - expected) / trials
        ), f"P(p<={t}) was {observed:.4f}, lattice value is {expected:.4f}"
        assert observed <= t + 1e-9, "must stay valid"


def test_lattice_band_one_sided_lowers_the_ceiling():
    """Dropping the floor spends the whole error budget upward, so the ceiling
    comes down -- which is what makes it worth using where only an excursion
    above is evidence."""
    _, _, two_sided, _, _ = _lattice_band(100, 201, 0.95)
    _, lower, one_sided, _, achieved = _lattice_band(100, 201, 0.95, one_sided=True)
    assert np.all(lower == 0.0), "one-sided means no floor"
    assert np.all(one_sided <= two_sided)
    assert np.any(one_sided < two_sided), "and strictly lower somewhere"
    assert abs((1 - achieved) - 0.05) < 0.01, f"level was {1 - achieved}"
