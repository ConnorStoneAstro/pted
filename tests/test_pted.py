import os
import types

import pted

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

import numpy as np

import pytest

BACKENDS = ["numpy", "torch", "jax"]


def _require_backend(backend):
    if backend == "torch" and torch is None:
        pytest.skip("torch not installed")
    if backend == "jax" and jax is None:
        pytest.skip("jax not installed")


def _to_backend(arr, backend):
    if backend == "torch":
        return torch.tensor(arr)
    if backend == "jax":
        return jnp.array(arr)
    return arr


@pytest.mark.parametrize("backend", BACKENDS)
def test_inputs_extra_dims_two_sample(backend):
    _require_backend(backend)
    np.random.seed(42)
    x = _to_backend(np.random.normal(size=(100, 3, 3)), backend)
    y = _to_backend(np.random.normal(size=(100, 3, 3)), backend)
    p = pted.pted(x, y)
    assert p > 1e-2 and p < 0.99, f"p-value {p} is not in the expected range (U(0,1))"


@pytest.mark.parametrize("backend", BACKENDS)
def test_inputs_extra_dims_coverage(backend):
    _require_backend(backend)
    np.random.seed(43)
    g = _to_backend(np.random.normal(size=(100, 3, 3)), backend)
    s = _to_backend(np.random.normal(size=(50, 100, 3, 3)), backend)
    p = pted.pted_coverage_test(g, s)
    assert p > 1e-2 and p < 0.99, f"p-value {p} is not in the expected range (U(0,1))"


def test_pted_main():
    pted.test()


def test_pted_progress_bar(capsys):
    pted.pted(np.array([[1, 2], [3, 4]]), np.array([[3, 2], [1, 4]]), permutations=42)
    captured = capsys.readouterr().err
    assert (
        "42/42" not in captured
    ), "progress bar showed up when prog_bar is set to False by default"

    pted.pted(
        np.array([[1, 2], [3, 4]]), np.array([[3, 2], [1, 4]]), permutations=42, prog_bar=True
    )
    captured = capsys.readouterr().err
    assert "42/42" in captured, "progress bar did not show when prog_bar is set to True"


@pytest.mark.parametrize("backend", BACKENDS)
def test_pted_two_sample(backend):
    _require_backend(backend)
    np.random.seed(42)

    # example 2 sample test
    D = 10
    x = _to_backend(np.random.normal(size=(100, D)), backend)
    y = _to_backend(np.random.normal(size=(100, D)), backend)
    p = pted.pted(x, y, two_tailed=True)  # regular two tailed, in null
    assert p > 1e-2 and p < 0.99, f"p-value {p} is not in the expected range (U(0,1))"
    p = pted.pted(x, x, two_tailed=True)  # exact replication, two tailed
    assert p < 1e-2, f"p-value {p} is not in the expected range (~0)"
    p = pted.pted(x, x, two_tailed=False)  # exact replication, one tailed
    assert p > 1e-2, f"p-value {p} is not in the expected range (~1)"

    x = _to_backend(np.random.normal(size=(100, D)), backend)
    y = _to_backend(np.random.uniform(size=(100, D)), backend)
    p = pted.pted(x, y, two_tailed=False)  # one tailed, different distributions
    assert p < 1e-2, f"p-value {p} is not in the expected range (~0)"

    x = _to_backend(np.random.normal(size=(100, D)), backend)
    t, p, _ = pted.pted(x, x, return_all=True)
    q = 2 * min(np.sum(p > t), np.sum(p < t))
    p = (1 + q) / (len(p) + 1)  # add one to numerator and denominator to avoid p=0
    assert p < 1e-2, f"p-value {p} is not in the expected range (~0)"


@pytest.mark.parametrize("backend", BACKENDS)
def test_pted_coverage_full(backend):
    _require_backend(backend)
    np.random.seed(42)
    g = _to_backend(
        np.random.normal(size=(32, 4)), backend
    )  # ground truth (n_simulations, n_dimensions)
    s = _to_backend(
        np.random.normal(size=(128, 32, 4)), backend
    )  # posterior samples (n_samples, n_simulations, n_dimensions)

    test, permute, pval = pted.pted_coverage_test(g, s, permutations=100, return_all=True)
    assert test.shape == (32,)
    assert permute.shape == (32, 100)
    assert pval > 1e-2 and pval < 0.99, f"p-value {pval} is not in the expected range (U(0,1))"


@pytest.mark.parametrize("backend", BACKENDS)
def test_pted_chunk(backend):
    _require_backend(backend)
    np.random.seed(42)

    # example 2 sample test
    D = 10
    x = _to_backend(np.random.normal(size=(1000, D)), backend)
    y = _to_backend(np.random.normal(size=(1000, D)), backend)
    p = pted.pted(x, y, chunk_size=100)
    assert p > 1e-2 and p < 0.99, f"p-value {p} is not in the expected range (U(0,1))"

    y = _to_backend(np.random.uniform(size=(1000, D)), backend)
    p = pted.pted(x, y, chunk_size=100)
    assert p < 1e-2, f"p-value {p} is not in the expected range (~0)"


@pytest.mark.parametrize("backend", BACKENDS)
def test_pted_chunk_mismatched_sizes(backend):
    """Chunked PTED works correctly when x and y have different sizes."""
    _require_backend(backend)
    np.random.seed(0)
    D = 5
    # x has 200 samples, y has 30 samples; chunk_size=50 → nxc=50, nyc=30 landmarks
    x = _to_backend(np.random.normal(size=(200, D)), backend)
    y = _to_backend(np.random.normal(size=(30, D)), backend)
    p = pted.pted(x, y, chunk_size=50)
    assert p > 1e-2 and p < 0.99, f"p-value {p} is not in the expected range (U(0,1))"

    # Different distributions should give small p-value even with mismatched sizes
    y_diff = _to_backend(np.random.uniform(size=(30, D)), backend)
    p = pted.pted(x, y_diff, chunk_size=50)
    assert p < 1e-2, f"p-value {p} is not in the expected range (~0)"


@pytest.mark.parametrize("backend", BACKENDS)
def test_pted_coverage_edgecase(backend):
    # Test with single simulation
    _require_backend(backend)
    np.random.seed(42)
    g = _to_backend(np.random.normal(size=(1, 10)), backend)
    s = _to_backend(np.random.normal(size=(100, 1, 10)), backend)
    p = pted.pted_coverage_test(g, s)
    assert p > 1e-2 and p < 0.99, f"p-value {p} is not in the expected range (U(0,1))"


def test_pted_coverage_progress_bar(capsys):
    np.random.seed(42)
    g = np.random.normal(size=(42, 10))
    s = np.random.normal(size=(100, 42, 10))
    pted.pted_coverage_test(g, s)
    captured = capsys.readouterr().err
    assert (
        "42/42" not in captured
    ), "progress bar showed up when prog_bar is set to False by default"

    pted.pted_coverage_test(g, s, prog_bar=True)
    captured = capsys.readouterr().err
    assert "42/42" in captured, "progress bar did not show when prog_bar is set to True"


@pytest.mark.parametrize("backend", BACKENDS)
def test_pted_coverage_overunder(backend):
    _require_backend(backend)
    np.random.seed(42)
    g_np = np.random.normal(size=(100, 3))
    s_np = np.random.normal(size=(50, 100, 3))
    g = _to_backend(g_np, backend)
    with pytest.warns(pted.utils.OverconfidenceWarning):
        pted.pted_coverage_test(g, _to_backend(s_np * 0.5, backend))
    with pytest.warns(pted.utils.UnderconfidenceWarning):
        pted.pted_coverage_test(g, _to_backend(s_np * 2, backend))


def test_sbc_histogram():
    np.random.seed(42)
    g = np.random.normal(size=(100, 10))  # ground truth (nsim, ndim)
    s = np.random.normal(size=(150, 100, 10))  # posterior samples (nsamp, nsim, ndim)

    pted.pted_coverage_test(g, s, permutations=100, sbc_histogram="sbc_hist.pdf")
    assert os.path.exists("sbc_hist.pdf"), "SBC histogram file was not created"
    os.remove("sbc_hist.pdf")


def test_pit_plot_coverage_test():
    np.random.seed(42)
    g = np.random.normal(size=(100, 10))  # ground truth (nsim, ndim)
    s = np.random.normal(size=(150, 100, 10))  # posterior samples (nsamp, nsim, ndim)

    pted.pted_coverage_test(g, s, permutations=100, pit_plot="pit_coverage.pdf")
    assert os.path.exists("pit_coverage.pdf"), "PIT plot file was not created"
    os.remove("pit_coverage.pdf")


def test_pit_plot_utility_direct():
    """pit_plot utility function creates a file and handles edge cases."""
    np.random.seed(42)
    pvals = np.random.uniform(size=50)
    pted.utils.pit_plot(pvals, "pit_direct.pdf")
    assert os.path.exists("pit_direct.pdf"), "PIT plot file was not created"
    os.remove("pit_direct.pdf")

    # Edge case: fewer than 2 p-values should warn and not create a file
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pted.utils.pit_plot(np.array([0.5]), "pit_single.pdf")
        assert len(w) == 1
        assert "at least 2" in str(w[0].message)
    assert not os.path.exists("pit_single.pdf")


def test_is_jax_array_with_jax():
    """is_jax_array returns True for a real JAX array and False for other types."""
    if jax is None:
        pytest.skip("jax not installed")
    assert pted.utils.is_jax_array(jnp.zeros(3)) is True
    assert pted.utils.is_jax_array(np.zeros(3)) is False
    assert pted.utils.is_jax_array(42) is False


def test_is_jax_array_no_jax(monkeypatch):
    """is_jax_array returns False when JAX is not installed."""
    monkeypatch.setattr("pted.utils.jax", None)
    assert pted.utils.is_jax_array(42) is False


def test_pted_jax_no_jax(monkeypatch):
    """permutation_energy_test raises AssertionError when a jax array is passed but JAX is not installed."""
    monkeypatch.setattr("pted.utils.jax", None)
    monkeypatch.setattr("pted.utils.is_jax_array", lambda o: True)
    with pytest.raises(AssertionError, match="JAX is not installed"):
        pted.utils.permutation_energy_test(np.zeros((5, 2)), np.zeros((5, 2)), permutations=10)


def test_pted_chunk_jax_no_jax(monkeypatch):
    """permutation_energy_test raises AssertionError when a jax array is passed but JAX is not installed."""
    monkeypatch.setattr("pted.utils.jax", None)
    monkeypatch.setattr("pted.utils.is_jax_array", lambda o: True)
    with pytest.raises(AssertionError, match="JAX is not installed"):
        pted.utils.permutation_energy_test(
            np.zeros((5, 2)), np.zeros((5, 2)), permutations=10, chunk_size=2
        )


def test_pted_torch_no_torch(monkeypatch):
    """permutation_energy_test raises AssertionError when a torch tensor is passed but torch is not installed."""
    fake_torch = types.SimpleNamespace(__version__="null")
    monkeypatch.setattr("pted.utils.torch", fake_torch)
    monkeypatch.setattr("pted.utils.is_torch_tensor", lambda o: True)
    with pytest.raises(AssertionError, match="PyTorch is not installed"):
        pted.utils.permutation_energy_test(np.zeros((5, 2)), np.zeros((5, 2)), permutations=10)


def test_pted_chunk_torch_no_torch(monkeypatch):
    """permutation_energy_test raises AssertionError when a torch tensor is passed but torch is not installed."""
    fake_torch = types.SimpleNamespace(__version__="null")
    monkeypatch.setattr("pted.utils.torch", fake_torch)
    monkeypatch.setattr("pted.utils.is_torch_tensor", lambda o: True)
    with pytest.raises(AssertionError, match="PyTorch is not installed"):
        pted.utils.permutation_energy_test(
            np.zeros((5, 2)), np.zeros((5, 2)), permutations=10, chunk_size=2
        )


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_cdist_matches_scipy(backend):
    """_jax_cdist (L2) and scipy cdist produce the same pairwise distances."""
    _require_backend(backend)
    from scipy.spatial.distance import cdist as scipy_cdist

    np.random.seed(7)
    x_np = np.random.normal(size=(10, 4)).astype(np.float32)
    y_np = np.random.normal(size=(8, 4)).astype(np.float32)

    expected = scipy_cdist(x_np, y_np, metric="euclidean")
    got = np.asarray(
        pted.utils._cdist(_to_backend(x_np, backend), _to_backend(y_np, backend), backend=backend)
    )
    assert np.allclose(got, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# The new column/batching controls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_pted_n_columns(backend):
    """n_columns names the column count directly, and agrees with the
    chunk_size that maps onto the same value."""
    _require_backend(backend)
    np.random.seed(11)
    x = _to_backend(np.random.normal(size=(300, 6)), backend)
    y = _to_backend(np.random.normal(size=(300, 6)), backend)

    # chunk_size=40 on two equal groups requests 40 + 40 columns
    p_chunk = pted.pted(x, y, chunk_size=40, rng=0)
    p_cols = pted.pted(x, y, n_columns=80, rng=0)
    assert p_chunk == p_cols

    y_diff = _to_backend(np.random.uniform(size=(300, 6)), backend)
    assert pted.pted(x, y_diff, n_columns=80) < 1e-2

    # n_columns covering the pooled sample is just the full test
    assert pted.pted(x, y, n_columns=10_000, rng=3) == pted.pted(x, y, rng=3)


def test_pted_column_args_are_exclusive():
    x = np.random.normal(size=(20, 3))
    y = np.random.normal(size=(20, 3))
    with pytest.raises(ValueError, match="not both"):
        pted.pted(x, y, chunk_size=5, n_columns=10)


def test_pted_rng_is_reproducible():
    """An explicit rng pins the permutations; the global seed still works when
    none is given."""
    x = np.random.normal(size=(60, 4))
    y = np.random.normal(size=(60, 4))

    a = pted.pted(x, y, permutations=200, n_columns=30, rng=7, return_all=True)
    b = pted.pted(x, y, permutations=200, n_columns=30, rng=7, return_all=True)
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2] == b[2]

    np.random.seed(4)
    c = pted.pted(x, y, permutations=200, n_columns=30, return_all=True)
    np.random.seed(4)
    d = pted.pted(x, y, permutations=200, n_columns=30, return_all=True)
    assert np.array_equal(c[1], d[1]), "np.random.seed should still pin the permutations"


@pytest.mark.parametrize(
    "n1,n2,n_columns,regime",
    [
        (60, 60, 24, "proportional"),
        (1, 100, 11, "singleton"),
    ],
)
def test_chunked_pvalues_are_calibrated(n1, n2, n_columns, regime):
    """Type-I error under H0 must match the nominal rate.

    This is the property the chunked test exists to preserve, and the one the
    earlier landmark-reshuffling scheme lost: because it re-split the columns
    by group after each permutation while the observed statistic always used a
    perfectly balanced split, the observed value was not exchangeable with the
    permuted ones. It rejected at 29% (proportional) and 84% (singleton) for a
    nominal 5%. Fixed seed, so this is deterministic rather than flaky.
    """
    from pted.utils import allocate_columns

    assert allocate_columns(n1, n2, n_columns, rng=0)["regime"] == regime

    trials, permutations = 600, 49
    rng = np.random.default_rng(20240904)
    pvals = np.empty(trials)
    for t in range(trials):
        # H0 is true: both groups come from the same distribution
        x = rng.standard_normal((n1, 4))
        y = rng.standard_normal((n2, 4))
        pvals[t] = pted.pted(
            x, y, permutations=permutations, n_columns=n_columns, two_tailed=False, rng=rng
        )

    for nominal, tol in ((0.10, 0.05), (0.20, 0.06)):
        rate = float(np.mean(pvals <= nominal))
        assert (
            abs(rate - nominal) < tol
        ), f"{regime}: rejected {rate:.3f} of {trials} null trials at nominal {nominal}"


def test_coverage_test_rng_is_not_shared_across_simulations():
    """A bare seed must not hand every simulation the same permutation draws,
    while the run as a whole stays reproducible from that seed."""
    g = np.random.default_rng(0).standard_normal((6, 3))
    s = np.random.default_rng(1).standard_normal((40, 6, 3))

    _, permute, _ = pted.pted_coverage_test(g, s, permutations=50, rng=7, return_all=True)
    assert not any(np.array_equal(permute[0], permute[i]) for i in range(1, len(permute)))

    _, again, _ = pted.pted_coverage_test(g, s, permutations=50, rng=7, return_all=True)
    assert np.array_equal(permute, again)


@pytest.mark.parametrize("backend", BACKENDS)
def test_coverage_test_with_chunking(backend):
    """Chunked coverage runs end to end and still separates a calibrated
    posterior from an over/under-confident one."""
    _require_backend(backend)
    rng = np.random.default_rng(3)
    nsim, nsamp, d = 40, 300, 3
    g, draws = [], []
    for _ in range(nsim):
        loc = rng.standard_normal(d) * 5
        scale = rng.uniform(1, 4, size=d)
        g.append(rng.normal(loc, scale, size=d))
        draws.append(rng.normal(loc, scale, size=(nsamp, d)))
    g = _to_backend(np.array(g), backend)
    ok = _to_backend(np.stack(draws, axis=1), backend)
    over = _to_backend(
        np.stack([v.mean(0) + (v - v.mean(0)) * 0.4 for v in draws], axis=1), backend
    )

    p_ok = pted.pted_coverage_test(g, ok, permutations=199, chunk_size=30)
    assert 1e-2 < p_ok < 0.99, f"calibrated posterior gave p={p_ok}"

    p_over = pted.pted_coverage_test(g, over, permutations=199, chunk_size=30, warn_confidence=None)
    assert p_over < 1e-2, f"overconfident posterior gave p={p_over}"
