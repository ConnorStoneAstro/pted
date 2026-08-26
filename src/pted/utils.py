from typing import Optional, Union
from warnings import warn

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import chi2 as chi2_dist, binom, kstwo, kstest
from tqdm.auto import trange

try:
    import torch
except ImportError:

    class torch:
        __version__ = "null"
        Tensor = np.ndarray


try:
    import jax
    import jax.numpy as jnp
    from jax import jit
except ImportError:
    jax = None
    jnp = None
    jit = lambda *a, **k: lambda f: f  # type: ignore


__all__ = (
    "is_torch_tensor",
    "is_jax_array",
    "pted",
    "pted_chunk",
    "two_tailed_p",
    "confidence_alert",
    "simulation_based_calibration_histogram",
    "pit_plot",
)


def is_torch_tensor(o):
    t = type(o)
    return (
        hasattr(t, "__module__")
        and t.__module__.startswith("torch")
        and hasattr(o, "device")
        and hasattr(o, "dtype")
        and hasattr(o, "shape")
    )


def is_jax_array(o):
    if jax is None:
        return False
    return isinstance(o, jax.Array)


def _backend(x) -> str:
    if is_torch_tensor(x):
        return "torch"
    if is_jax_array(x):
        return "jax"
    return "numpy"


def _concatenate(arrays, backend: str):
    if backend == "torch":
        return torch.cat(arrays, dim=0)
    if backend == "jax":
        return jnp.concatenate(arrays, axis=0)
    return np.concatenate(arrays, axis=0)


def _all_finite(z, backend: str) -> bool:
    if backend == "torch":
        return bool(torch.all(torch.isfinite(z)))
    if backend == "jax":
        return bool(jnp.all(jnp.isfinite(z)))
    return bool(np.all(np.isfinite(z)))


def _to_scalar(x, backend: str) -> float:
    if backend == "torch":
        return float(x.item())
    if backend == "jax":
        return float(x.item())
    return float(x)


def _random_permutation(D, backend: str = "numpy"):
    if backend == "torch":
        I = torch.randperm(D.shape[0], device=D.device)
    elif backend == "jax":
        I = jax.random.permutation(jax.random.PRNGKey(np.random.randint(0, 1e6)), D.shape[0])
    else:
        I = np.random.permutation(D.shape[0])
    if len(D.shape) == 1:
        return D[I]
    return D[I][:, I]


@jit
def _jax_cdist(x, y):
    return jax.vmap(lambda xi: jnp.linalg.norm(xi - y, ord=2.0, axis=-1))(x)


def _cdist(a, b, backend: str):
    if backend == "torch":
        return torch.cdist(a, b, p=2.0)
    if backend == "jax":
        return _jax_cdist(a, b)
    return cdist(a, b, metric="euclidean")


def _energy_distance_precompute(
    D: Union[np.ndarray, torch.Tensor],
    nx: int,
    ny: int,
    nxc: Optional[int] = None,
    nyc: Optional[int] = None,
) -> Union[float, torch.Tensor]:
    """Energy distance from a rectangular (nx+ny, nxc+nyc) distance matrix.

    If ``D`` is a full (nx+ny, nx+ny) distance matrix, then ``nxc=nx`` and
    ``nyc=ny`` and this is the standard energy distance formula. Otherwise, if
    ``D`` is a rectangular (nx+ny, nxc+nyc) distance matrix, then ``D`` holds
    distances from every sample (rows: ``nx`` of x followed by ``ny`` of y) to a
    set of landmark samples (columns: ``nxc`` from x followed by ``nyc`` from
    y). Exx/Eyy are estimated from each group against its own landmarks, and Exy
    is estimated by averaging the two cross blocks (x-rows vs y-landmarks, and
    y-rows vs x-landmarks).
    """
    nxc = nx if nxc is None else nxc
    nyc = ny if nyc is None else nyc
    Exx = D[:nx, :nxc].sum() / (nx * nxc)
    Eyy = D[nx:, nxc:].sum() / (ny * nyc)
    Exy = (D[:nx, nxc:].sum() + D[nx:, :nxc].sum()) / (nx * nyc + ny * nxc)
    return 2 * Exy - Exx - Eyy


def pted(
    x,
    y,
    permutations: int,
    prog_bar: bool = False,
) -> tuple[float, list[float]]:
    """Permutation-based energy distance test statistic and its permutation distribution.

    Works transparently with numpy arrays, torch tensors, and jax arrays.
    """
    if is_torch_tensor(x):
        assert torch.__version__ != "null", "PyTorch is not installed! try: `pip install torch`"
    if is_jax_array(x):
        assert jax is not None, "JAX is not installed! try: `pip install jax`"

    backend = _backend(x)
    z = _concatenate((x, y), backend)
    assert _all_finite(z, backend), "Input contains NaN or Inf!"
    dmatrix = _cdist(z, z, backend)
    assert _all_finite(
        dmatrix, backend
    ), "Distance matrix contains NaN or Inf! Consider normalizing values to be more stable (i.e. z-score norm)."
    if backend == "jax":  # numpy is faster for eager stuff
        dmatrix = np.asarray(dmatrix)
        backend = "numpy"
    nx = len(x)
    ny = len(y)

    test_stat = _to_scalar(_energy_distance_precompute(dmatrix, nx, ny), backend)
    permute_stats = []
    for _ in trange(permutations, disable=not prog_bar):
        dmatrix = _random_permutation(dmatrix, backend)
        permute_stats.append(_to_scalar(_energy_distance_precompute(dmatrix, nx, ny), backend))

    return test_stat, permute_stats


def pted_chunk(
    x,
    y,
    permutations: int,
    chunk_size: int,
    prog_bar: bool = False,
) -> tuple[float, list[float]]:
    """Chunked variant of `pted` for large datasets.

    Rather than the full ``(nx+ny, nx+ny)`` pairwise distance matrix, this
    builds a rectangular ``(nx+ny, nxc+nyc)`` matrix of distances from every
    sample to a set of randomly chosen "landmark" samples, where ``nxc =
    min(chunk_size, nx)`` landmarks are drawn from ``x`` and ``nyc =
    min(chunk_size, ny)`` from ``y``. This matrix is computed only once; the
    permutation distribution is then obtained by re-indexing (permuting) its
    rows, which reassigns samples to the x/y groups without recomputing any
    distances. Each landmark's row position is tracked and carried through the
    same row permutation, so after every shuffle the landmarks are re-split into
    "x-side"/"y-side" columns according to where they now fall, keeping rows and
    columns consistent. In pathological cases where too few landmarks are
    permuted onto either side (zero or one tenth of the original number of
    landmarks), a new permutation is drawn.

    Works transparently with numpy arrays, torch tensors, and jax arrays.
    """
    if is_torch_tensor(x):
        assert torch.__version__ != "null", "PyTorch is not installed! try: `pip install torch`"
    if is_jax_array(x):
        assert jax is not None, "JAX is not installed! try: `pip install jax`"

    backend = _backend(x)
    z = _concatenate((x, y), backend)
    assert _all_finite(z, backend), "Input contains NaN or Inf!"
    nx = len(x)
    ny = len(y)
    nxc = min(chunk_size, nx)
    nyc = min(chunk_size, ny)

    landmark_pos = np.sort(
        np.concatenate(
            [
                np.random.choice(nx, size=nxc, replace=False),
                nx + np.random.choice(ny, size=nyc, replace=False),
            ]
        )
    )
    dmatrix = _cdist(z, z[landmark_pos], backend)
    assert _all_finite(
        dmatrix, backend
    ), "Distance matrix contains NaN or Inf! Consider normalizing values to be more stable (i.e. z-score norm)."
    if backend == "jax":  # numpy is faster for eager stuff
        dmatrix = np.asarray(dmatrix)
        backend = "numpy"

    test_stat = _to_scalar(_energy_distance_precompute(dmatrix, nx, ny, nxc, nyc), backend)
    permute_stats = []
    for _ in trange(permutations, disable=not prog_bar):
        while True:
            I = np.random.permutation(len(z))
            # Track where each landmark moved to, then re-split columns by their new x/y side.
            landmark_pos = np.argsort(I)[landmark_pos]
            order = np.argsort(landmark_pos >= nx, kind="stable")
            dmatrix_i = dmatrix[I][:, order]
            landmark_pos = landmark_pos[order]
            nxc_i = int(np.sum(landmark_pos < nx))
            nyc_i = nxc + nyc - nxc_i
            if nxc_i < max(1, (nxc * 0.1)) or nyc_i < max(1, (nyc * 0.1)):
                continue  # Skip this permutation if too few landmarks remain on either side, to avoid unstable estimates.
            permute_stats.append(
                _to_scalar(_energy_distance_precompute(dmatrix_i, nx, ny, nxc_i, nyc_i), backend)
            )
            break
    return test_stat, permute_stats


def two_tailed_p(chi2, df):
    p_left = chi2_dist.cdf(chi2, df)
    p_right = chi2_dist.sf(chi2, df)
    return 2 * min(p_left, p_right)


class OverconfidenceWarning(UserWarning):
    """Warning for overconfidence in chi-squared test results."""


class UnderconfidenceWarning(UserWarning):
    """Warning for underconfidence in chi-squared test results."""


def confidence_alert(chi2, df, level):

    left_tail = chi2_dist.cdf(chi2, df)
    right_tail = chi2_dist.sf(chi2, df)

    if left_tail < level:
        warn(
            UnderconfidenceWarning(
                f"Chi^2 of {chi2:.2e} for degrees of freedom {df} indicates underconfidence (left tail p-value {left_tail:.2e} < {level:.2e})."
            )
        )
    elif right_tail < level:
        warn(
            OverconfidenceWarning(
                f"Chi^2 of {chi2:.2e} for degrees of freedom {df} indicates overconfidence (right tail p-value {right_tail:.2e} < {level:.2e})."
            )
        )


def simulation_based_calibration_histogram(ranks, saveto, bins=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warn("No SBC histogram generated! Please install matplotlib.")
        return

    if bins is None:
        bins = max(5, int(np.sqrt(len(ranks))))

    hist, bins = np.histogram(ranks, range=(0, 1), bins=bins)
    plt.bar(
        bins[:-1],
        hist,
        width=np.diff(bins),
        align="edge",
        facecolor="#A34F4F",
        edgecolor="#7F0606",
    )
    q = binom.ppf([0.16, 0.5, 0.84], len(ranks), 1 / len(bins))
    plt.axhline(q[1], color="k", alpha=0.5)
    plt.fill_between(
        [bins[0], bins[-1]], [q[0], q[0]], [q[2], q[2]], color="grey", linewidth=0, alpha=0.5
    )
    plt.xlabel("Rank")
    plt.ylabel("Count")
    plt.xlim([bins[0], bins[-1]])
    plt.title("Simulation-Based-Calibration Histogram")
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()


def pit_plot(pvals, saveto, confidence=0.95):
    """Create a Probability Integral Transform (PIT) plot.

    Plots the empirical CDF of the provided p-values against the expected
    CDF for a uniform distribution (the 1:1 diagonal). A shaded confidence
    region is drawn showing the range within which the empirical CDF should
    fall with probability ``confidence`` if the p-values are truly uniform.
    The confidence band is derived from the two-sided Kolmogorov-Smirnov
    statistic. Any portion of the empirical CDF that lies outside this band
    constitutes evidence that the p-values are not uniformly distributed.

    The KS statistic and its p-value are annotated on the plot to quantify
    the maximum deviation from the diagonal.

    Parameters
    ----------
        pvals (array-like): Array of p-values in [0, 1].
        saveto (str): File path where the plot will be saved. The format is
            inferred from the file extension (e.g. ".pdf", ".png").
        confidence (float): Confidence level for the KS confidence band.
            Default is 0.95 (95%).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warn("No PIT plot generated! Please install matplotlib.")
        return

    pvals = np.asarray(pvals, dtype=float).ravel()
    n = len(pvals)
    if n < 2:
        warn("PIT plot requires at least 2 p-values. Skipping.")
        return

    sorted_pvals = np.sort(pvals)
    ecdf = np.arange(1, n + 1) / n

    # Critical value for the two-sided KS statistic at the given confidence level.
    d_crit = kstwo.ppf(confidence, n)

    # One-sample KS test against U(0,1) for annotation
    ks_stat, ks_pval = kstest(pvals, "uniform")

    x = np.linspace(0, 1, 500)

    fig, ax = plt.subplots()
    ax.fill_between(
        x,
        np.maximum(x - d_crit, 0),
        np.minimum(x + d_crit, 1),
        color="grey",
        alpha=0.3,
        linewidth=0,
        label=f"{int(confidence * 100)}% KS confidence band",
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.8, label="Expected (Uniform)")
    ax.step(
        np.concatenate([[0], sorted_pvals, [1]]),
        np.concatenate([[0], ecdf, [1]]),
        where="post",
        color="#A34F4F",
        label=f"Empirical CDF (KS={ks_stat:.3f}, p={ks_pval:.3f})",
    )
    ax.set_xlabel("p-value")
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title("Probability Integral Transform (PIT) Plot")
    ax.legend()
    fig.savefig(saveto, bbox_inches="tight")
    plt.close(fig)


def hdp_coverage_test(
    ground_truth: np.ndarray, posterior_samples: np.ndarray, two_tailed: bool = True
) -> float:
    """
    Perform a Highest Density Posterior (HDP) coverage test. Essentially this
    rank orders the posterior samples by their posterior density and also places
    the ground truth in that ranking. The fraction of posterior samples with
    higher density than the ground truth forms a p-value under the null
    hypothesis. For many repeated experiments, we check that the p-values are
    uniformly distributed.

    Args:
        ground_truth: Posterior density (or log density) at the ground-truth parameters, shape (Nsim,)
        posterior_samples: Posterior density (or log density) for each posterior draw, shape (Nsamp, Nsim)
        two_tailed: Whether to compute a two-tailed p-value (default: True)

    Returns:
        pvalue: The p-value for the coverage test
    """
    from scipy.stats import chi2 as chi2_dist

    Nsamp, Nsim = posterior_samples.shape
    q = np.sum(posterior_samples >= ground_truth[None], axis=0)
    chi2_hdp = -2 * np.sum(np.log((q + 1) / (Nsamp + 1)))
    pvalue_right = chi2_dist.sf(chi2_hdp, 2 * Nsim)
    pvalue_left = chi2_dist.cdf(chi2_hdp, 2 * Nsim)
    if two_tailed:
        return 2 * min(pvalue_left, pvalue_right)
    return pvalue_right
