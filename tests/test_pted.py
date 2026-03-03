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


def test_inputs_extra_dims():
    np.random.seed(42)
    # Test with numpy arrays
    x = np.random.normal(size=(100, 30, 30))
    y = np.random.normal(size=(100, 30, 30))
    p = pted.pted(x, y)
    assert p > 1e-2 and p < 0.99, f"p-value {p} is not in the expected range (U(0,1))"

    if torch is None:
        pytest.skip("torch not installed")
    # Test with torch tensors
    g = torch.randn(100, 30, 30)
    s = torch.randn(50, 100, 30, 30)
    p = pted.pted_coverage_test(g, s)
    assert p > 1e-2 and p < 0.99, f"p-value {p} is not in the expected range (U(0,1))"


def test_pted_main():
    pted.test()


def test_pted_progress_bar(capsys):
    pted.pted(np.array([[1,2],[3,4]]), np.array([[3,2],[1,4]]), permutations=42)
    captured = capsys.readouterr().err
    assert "42/42" not in captured, "progress bar showed up when prog_bar is set to False by default"

    pted.pted(np.array([[1,2],[3,4]]), np.array([[3,2],[1,4]]), permutations=42, prog_bar=True)
    captured = capsys.readouterr().err
    assert "42/42" in captured, "progress bar did not show when prog_bar is set to True"


def test_pted_torch():
    if torch is None:
        pytest.skip("torch not installed")

    # Set the random seed for reproducibility
    torch.manual_seed(41)
    np.random.seed(42)

    # example 2 sample test
    D = 300
    for _ in range(20):
        x = torch.randn(100, D)
        y = torch.randn(100, D)
        p = pted.pted(x, y)
        assert p > 1e-2 and p < 0.99, f"p-value {p} is not in the expected range (U(0,1))"

    x = torch.randn(100, D)
    y = torch.rand(100, D)
    p = pted.pted(x, y, two_tailed=False)
    assert p < 1e-2, f"p-value {p} is not in the expected range (~0)"

    x = torch.randn(100, D)
    t, p, _ = pted.pted(x, x, return_all=True)
    q = 2 * min(np.sum(p > t), np.sum(p < t))
    p = (1 + q) / (len(p) + 1)  # add one to numerator and denominator to avoid p=0
    assert p < 1e-2, f"p-value {p} is not in the expected range (~0)"


def test_pted_coverage_full():
    g = np.random.normal(size=(100, 10))  # ground truth (n_simulations, n_dimensions)
    s = np.random.normal(
        size=(200, 100, 10)
    )  # posterior samples (n_samples, n_simulations, n_dimensions)

    test, permute, _ = pted.pted_coverage_test(g, s, permutations=100, return_all=True)
    assert test.shape == (100,)
    assert permute.shape == (100, 100)


def test_pted_chunk_torch():
    if torch is None:
        pytest.skip("torch not installed")
    np.random.seed(42)
    torch.manual_seed(42)

    # example 2 sample test
    D = 10
    x = torch.randn(1000, D)
    y = torch.randn(1000, D)
    p = pted.pted(x, y, chunk_size=100, chunk_iter=10)
    assert p > 1e-2 and p < 0.99, f"p-value {p} is not in the expected range (U(0,1))"

    y = torch.rand(1000, D)
    p = pted.pted(x, y, chunk_size=100, chunk_iter=10)
    assert p < 1e-2, f"p-value {p} is not in the expected range (~0)"


def test_pted_chunk_numpy():
    np.random.seed(42)

    # example 2 sample test
    D = 10
    x = np.random.normal(size=(1000, D))
    y = np.random.normal(size=(1000, D))
    p = pted.pted(x, y, chunk_size=100, chunk_iter=10)
    assert p > 1e-2 and p < 0.99, f"p-value {p} is not in the expected range (U(0,1))"

    y = np.random.uniform(size=(1000, D))
    p = pted.pted(x, y, chunk_size=100, chunk_iter=10)
    assert p < 1e-2, f"p-value {p} is not in the expected range (~0)"


def test_pted_coverage_edgecase():
    # Test with single simulation
    g = np.random.normal(size=(1, 10))
    s = np.random.normal(size=(100, 1, 10))
    p = pted.pted_coverage_test(g, s)
    assert p > 1e-2 and p < 0.99, f"p-value {p} is not in the expected range (U(0,1))"


def test_pted_coverage_progress_bar(capsys):
    g = np.random.normal(size=(42, 10))
    s = np.random.normal(size=(100, 42, 10))
    pted.pted_coverage_test(g, s)
    captured = capsys.readouterr().err
    assert "42/42" not in captured, "progress bar showed up when prog_bar is set to False by default"

    pted.pted_coverage_test(g, s, prog_bar=True)
    captured = capsys.readouterr().err
    assert "42/42" in captured, "progress bar did not show when prog_bar is set to True"


def test_pted_coverage_overunder():
    if torch is None:
        pytest.skip("torch not installed")
    g = torch.randn(100, 3)
    s = torch.randn(50, 100, 3)
    with pytest.warns(pted.utils.OverconfidenceWarning):
        pted.pted_coverage_test(g, s * 0.5)
    with pytest.warns(pted.utils.UnderconfidenceWarning):
        pted.pted_coverage_test(g, s * 2)


def test_sbc_histogram():
    g = np.random.normal(size=(100, 10))  # ground truth (nsim, ndim)
    s = np.random.normal(size=(150, 100, 10))  # posterior samples (nsamp, nsim, ndim)

    pted.pted_coverage_test(g, s, permutations=100, sbc_histogram="sbc_hist.pdf")
    os.remove("sbc_hist.pdf")


def test_pted_jax():
    if jax is None:
        pytest.skip("jax not installed")

    # Set the random seed for reproducibility
    np.random.seed(42)

    # example 2 sample test
    D = 300
    for _ in range(20):
        x = jnp.array(np.random.normal(size=(100, D)))
        y = jnp.array(np.random.normal(size=(100, D)))
        p = pted.pted(x, y)
        assert p > 1e-2 and p < 0.99, f"p-value {p} is not in the expected range (U(0,1))"

    x = jnp.array(np.random.normal(size=(100, D)))
    y = jnp.array(np.random.uniform(size=(100, D)))
    p = pted.pted(x, y, two_tailed=False)
    assert p < 1e-2, f"p-value {p} is not in the expected range (~0)"

    x = jnp.array(np.random.normal(size=(100, D)))
    t, p, _ = pted.pted(x, x, return_all=True)
    q = 2 * min(np.sum(p > t), np.sum(p < t))
    p = (1 + q) / (len(p) + 1)
    assert p < 1e-2, f"p-value {p} is not in the expected range (~0)"


def test_pted_chunk_jax():
    if jax is None:
        pytest.skip("jax not installed")
    np.random.seed(42)

    # example 2 sample test
    D = 10
    x = jnp.array(np.random.normal(size=(1000, D)))
    y = jnp.array(np.random.normal(size=(1000, D)))
    p = pted.pted(x, y, chunk_size=100, chunk_iter=10)
    assert p > 1e-2 and p < 0.99, f"p-value {p} is not in the expected range (U(0,1))"

    y = jnp.array(np.random.uniform(size=(1000, D)))
    p = pted.pted(x, y, chunk_size=100, chunk_iter=10)
    assert p < 1e-2, f"p-value {p} is not in the expected range (~0)"


def test_pted_coverage_jax():
    if jax is None:
        pytest.skip("jax not installed")

    g = jnp.array(np.random.normal(size=(100, 10)))
    s = jnp.array(np.random.normal(size=(50, 100, 10)))
    p = pted.pted_coverage_test(g, s)
    assert p > 1e-2 and p < 0.99, f"p-value {p} is not in the expected range (U(0,1))"


# ---------------------------------------------------------------------------
# Unit tests for newly-added utils functions
# ---------------------------------------------------------------------------


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


def test_jax_cdist():
    """_jax_cdist produces correct pairwise Euclidean distances."""
    if jax is None:
        pytest.skip("jax not installed")
    x = jnp.array([[0.0, 0.0], [3.0, 4.0]])
    y = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    D = pted.utils._jax_cdist(x, y)
    assert D.shape == (2, 2)
    assert float(D[0, 1]) == pytest.approx(1.0)
    assert float(D[1, 0]) == pytest.approx(5.0)


def test_jax_cdist_non_euclidean():
    """_jax_cdist produces correct pairwise distances for p != 2 (vmap path)."""
    if jax is None:
        pytest.skip("jax not installed")
    x = jnp.array([[0.0, 0.0], [3.0, 4.0]])
    y = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    # L1: d(x[0], y[1]) = |0-1| + |0-0| = 1; d(x[1], y[0]) = |3-0| + |4-0| = 7
    D = pted.utils._jax_cdist(x, y, p=1.0)
    assert D.shape == (2, 2)
    assert float(D[0, 1]) == pytest.approx(1.0)
    assert float(D[1, 0]) == pytest.approx(7.0)


def test_energy_distance_jax():
    """_energy_distance_jax returns 0 when x and y are identical."""
    if jax is None:
        pytest.skip("jax not installed")
    x = jnp.array(np.random.normal(size=(50, 5)))
    # Identical samples → energy distance should be ~0
    ed = pted.utils._energy_distance_jax(x, x)
    assert abs(ed) < 1e-6


def test_energy_distance_estimate_jax():
    """_energy_distance_estimate_jax returns a finite scalar."""
    if jax is None:
        pytest.skip("jax not installed")
    np.random.seed(0)
    x = jnp.array(np.random.normal(size=(100, 4)))
    y = jnp.array(np.random.normal(size=(100, 4)))
    ed = pted.utils._energy_distance_estimate_jax(x, y, chunk_size=20, chunk_iter=5)
    assert np.isfinite(ed)


def test_pted_jax_no_jax(monkeypatch):
    """pted_jax raises AssertionError when JAX is not installed."""
    monkeypatch.setattr("pted.utils.jax", None)
    with pytest.raises(AssertionError, match="JAX is not installed"):
        pted.utils.pted_jax(np.zeros((5, 2)), np.zeros((5, 2)))


def test_pted_chunk_jax_no_jax(monkeypatch):
    """pted_chunk_jax raises AssertionError when JAX is not installed."""
    monkeypatch.setattr("pted.utils.jax", None)
    with pytest.raises(AssertionError, match="JAX is not installed"):
        pted.utils.pted_chunk_jax(np.zeros((5, 2)), np.zeros((5, 2)))


def test_pted_torch_no_torch(monkeypatch):
    """pted_torch raises AssertionError when torch is not installed."""
    fake_torch = types.SimpleNamespace(__version__="null")
    monkeypatch.setattr("pted.utils.torch", fake_torch)
    with pytest.raises(AssertionError, match="PyTorch is not installed"):
        pted.utils.pted_torch(np.zeros((5, 2)), np.zeros((5, 2)))


def test_pted_chunk_torch_no_torch(monkeypatch):
    """pted_chunk_torch raises AssertionError when torch is not installed."""
    fake_torch = types.SimpleNamespace(__version__="null")
    monkeypatch.setattr("pted.utils.torch", fake_torch)
    with pytest.raises(AssertionError, match="PyTorch is not installed"):
        pted.utils.pted_chunk_torch(np.zeros((5, 2)), np.zeros((5, 2)))


# ---------------------------------------------------------------------------
# Cross-backend consistency tests
# ---------------------------------------------------------------------------


def test_jax_cdist_matches_scipy():
    """_jax_cdist (L2) and scipy cdist produce the same pairwise distances."""
    if jax is None:
        pytest.skip("jax not installed")
    from scipy.spatial.distance import cdist as scipy_cdist

    np.random.seed(7)
    x_np = np.random.normal(size=(10, 4)).astype(np.float32)
    y_np = np.random.normal(size=(8, 4)).astype(np.float32)

    expected = scipy_cdist(x_np, y_np, metric="euclidean")
    got = np.array(pted.utils._jax_cdist(jnp.array(x_np), jnp.array(y_np)))
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_energy_distance_numpy_torch_jax_agree():
    """_energy_distance_{numpy,torch,jax} return the same value for identical inputs."""
    if torch is None:
        pytest.skip("torch not installed")
    if jax is None:
        pytest.skip("jax not installed")

    np.random.seed(99)
    # Use float32 so all backends operate at the same precision
    # (JAX uses float32 by default)
    x_np = np.random.normal(size=(30, 5)).astype(np.float32)
    y_np = np.random.normal(size=(30, 5)).astype(np.float32)

    ed_numpy = pted.utils._energy_distance_numpy(x_np, y_np)
    ed_torch = pted.utils._energy_distance_torch(
        torch.tensor(x_np), torch.tensor(y_np)
    )
    ed_jax = pted.utils._energy_distance_jax(jnp.array(x_np), jnp.array(y_np))

    assert ed_numpy == pytest.approx(ed_torch, rel=1e-4), (
        f"numpy ({ed_numpy}) and torch ({ed_torch}) energy distances differ"
    )
    assert ed_numpy == pytest.approx(ed_jax, rel=1e-4), (
        f"numpy ({ed_numpy}) and jax ({ed_jax}) energy distances differ"
    )


def test_energy_distance_estimate_numpy_torch_jax_agree():
    """_energy_distance_estimate_{numpy,torch,jax} return close values for the same seed/data."""
    if torch is None:
        pytest.skip("torch not installed")
    if jax is None:
        pytest.skip("jax not installed")

    np.random.seed(123)
    # Use float32 so all backends operate at the same precision
    x_np = np.random.normal(size=(200, 5)).astype(np.float32)
    y_np = np.random.normal(size=(200, 5)).astype(np.float32)

    # Run with the same seed so the same chunks are sampled
    np.random.seed(0)
    ed_numpy = pted.utils._energy_distance_estimate_numpy(x_np, y_np, chunk_size=50, chunk_iter=5)
    np.random.seed(0)
    ed_torch = pted.utils._energy_distance_estimate_torch(
        torch.tensor(x_np), torch.tensor(y_np), chunk_size=50, chunk_iter=5
    )
    np.random.seed(0)
    ed_jax = pted.utils._energy_distance_estimate_jax(
        jnp.array(x_np), jnp.array(y_np), chunk_size=50, chunk_iter=5
    )

    assert ed_numpy == pytest.approx(ed_torch, rel=1e-4), (
        f"numpy ({ed_numpy}) and torch ({ed_torch}) energy distance estimates differ"
    )
    assert ed_numpy == pytest.approx(ed_jax, rel=1e-4), (
        f"numpy ({ed_numpy}) and jax ({ed_jax}) energy distance estimates differ"
    )
