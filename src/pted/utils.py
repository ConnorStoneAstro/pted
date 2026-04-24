from typing import Union
from warnings import warn

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import chi2 as chi2_dist, binom, kstwo, kstest
from scipy.optimize import root_scalar
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
except ImportError:
    jax = None
    jnp = None


__all__ = (
    "is_torch_tensor",
    "is_jax_array",
    "pted_numpy",
    "pted_chunk_numpy",
    "pted_torch",
    "pted_chunk_torch",
    "pted_jax",
    "pted_chunk_jax",
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


def _energy_distance_precompute(
    D: Union[np.ndarray, torch.Tensor], nx: int, ny: int
) -> Union[float, torch.Tensor]:
    Exx = D[:nx, :nx].sum() / nx**2
    Eyy = D[nx:, nx:].sum() / ny**2
    Exy = D[:nx, nx:].sum() / (nx * ny)
    return 2 * Exy - Exx - Eyy


def _energy_distance_numpy(x: np.ndarray, y: np.ndarray, metric: str = "euclidean") -> float:
    nx = len(x)
    ny = len(y)
    z = np.concatenate((x, y), axis=0)
    D = cdist(z, z, metric=metric)
    return _energy_distance_precompute(D, nx, ny)


def _energy_distance_torch(
    x: torch.Tensor, y: torch.Tensor, metric: Union[str, float] = "euclidean"
) -> float:
    nx = len(x)
    ny = len(y)
    z = torch.cat((x, y), dim=0)
    if metric == "euclidean":
        metric = 2.0
    D = torch.cdist(z, z, p=metric)
    return _energy_distance_precompute(D, nx, ny).item()


def _energy_distance_estimate_numpy(
    x: np.ndarray,
    y: np.ndarray,
    chunk_size: int,
    chunk_iter: int,
    metric: Union[str, float] = "euclidean",
) -> float:

    E_est = []
    for _ in range(chunk_iter):
        # Randomly sample a chunk of data
        idx = np.random.choice(len(x), size=min(len(x), chunk_size), replace=False)
        x_chunk = x[idx]
        idy = np.random.choice(len(y), size=min(len(y), chunk_size), replace=False)
        y_chunk = y[idy]

        # Compute the energy distance
        E_est.append(_energy_distance_numpy(x_chunk, y_chunk, metric=metric))
    return np.mean(E_est)


def _energy_distance_estimate_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    chunk_size: int,
    chunk_iter: int,
    metric: Union[str, float] = "euclidean",
) -> float:

    E_est = []
    for _ in range(chunk_iter):
        # Randomly sample a chunk of data
        idx = np.random.choice(len(x), size=min(len(x), chunk_size), replace=False)
        x_chunk = x[torch.tensor(idx)]
        idy = np.random.choice(len(y), size=min(len(y), chunk_size), replace=False)
        y_chunk = y[torch.tensor(idy)]

        # Compute the energy distance
        E_est.append(_energy_distance_torch(x_chunk, y_chunk, metric=metric))
    return np.mean(E_est)


def _jax_cdist(x, y, p: float = 2.0):
    if p == 2.0:
        # Squared-norm identity avoids materializing the (nx, ny, d) diff tensor.
        # ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 * x_i . y_j
        x_sq = jnp.sum(x**2, axis=-1)  # (nx,)
        y_sq = jnp.sum(y**2, axis=-1)  # (ny,)
        sq_dist = x_sq[:, None] + y_sq[None, :] - 2.0 * (x @ y.T)
        return jnp.sqrt(jnp.maximum(sq_dist, 0.0))
    # For general p-norms use vmap to avoid the (nx, ny, d) intermediate.
    return jax.vmap(lambda xi: jnp.linalg.norm(xi - y, ord=p, axis=-1))(x)


def _energy_distance_jax(x, y, metric: Union[str, float] = "euclidean") -> float:
    nx = len(x)
    ny = len(y)
    z = jnp.concatenate([x, y], axis=0)
    if metric == "euclidean":
        metric = 2.0
    D = _jax_cdist(z, z, p=metric)
    return float(_energy_distance_precompute(D, nx, ny))


def _energy_distance_estimate_jax(
    x,
    y,
    chunk_size: int,
    chunk_iter: int,
    metric: Union[str, float] = "euclidean",
) -> float:

    E_est = []
    for _ in range(chunk_iter):
        # Randomly sample a chunk of data
        idx = np.random.choice(len(x), size=min(len(x), chunk_size), replace=False)
        x_chunk = x[idx]
        idy = np.random.choice(len(y), size=min(len(y), chunk_size), replace=False)
        y_chunk = y[idy]

        # Compute the energy distance
        E_est.append(_energy_distance_jax(x_chunk, y_chunk, metric=metric))
    return np.mean(E_est)


def pted_chunk_numpy(
    x: np.ndarray,
    y: np.ndarray,
    permutations: int = 100,
    metric: str = "euclidean",
    chunk_size: int = 100,
    chunk_iter: int = 10,
    prog_bar: bool = False,
) -> tuple[float, list[float]]:
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)), "Input contains NaN or Inf!"
    nx = len(x)

    test_stat = _energy_distance_estimate_numpy(x, y, chunk_size, chunk_iter, metric=metric)
    permute_stats = []
    for _ in trange(permutations, disable=not prog_bar):
        z = np.concatenate((x, y), axis=0)
        z = z[np.random.permutation(len(z))]
        x, y = z[:nx], z[nx:]
        permute_stats.append(
            _energy_distance_estimate_numpy(x, y, chunk_size, chunk_iter, metric=metric)
        )
    return test_stat, permute_stats


def pted_chunk_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    permutations: int = 100,
    metric: Union[str, float] = "euclidean",
    chunk_size: int = 100,
    chunk_iter: int = 10,
    prog_bar: bool = False,
) -> tuple[float, list[float]]:
    assert torch.__version__ != "null", "PyTorch is not installed! try: `pip install torch`"
    assert torch.all(torch.isfinite(x)) and torch.all(
        torch.isfinite(y)
    ), "Input contains NaN or Inf!"
    nx = len(x)

    test_stat = _energy_distance_estimate_torch(x, y, chunk_size, chunk_iter, metric=metric)
    permute_stats = []
    for _ in trange(permutations, disable=not prog_bar):
        z = torch.cat((x, y), dim=0)
        z = z[torch.randperm(len(z))]
        x, y = z[:nx], z[nx:]
        permute_stats.append(
            _energy_distance_estimate_torch(x, y, chunk_size, chunk_iter, metric=metric)
        )
    return test_stat, permute_stats


def pted_numpy(
    x: np.ndarray,
    y: np.ndarray,
    permutations: int = 100,
    metric: str = "euclidean",
    prog_bar: bool = False,
) -> tuple[float, list[float]]:
    z = np.concatenate((x, y), axis=0)
    assert np.all(np.isfinite(z)), "Input contains NaN or Inf!"
    dmatrix = cdist(z, z, metric=metric)
    assert np.all(
        np.isfinite(dmatrix)
    ), "Distance matrix contains NaN or Inf! Consider using a different metric or normalizing values to be more stable (i.e. z-score norm)."
    nx = len(x)
    ny = len(y)

    test_stat = _energy_distance_precompute(dmatrix, nx, ny)
    permute_stats = []
    for _ in trange(permutations, disable=not prog_bar):
        I = np.random.permutation(len(z))
        dmatrix = dmatrix[I][:, I]
        permute_stats.append(_energy_distance_precompute(dmatrix, nx, ny))
    return test_stat, permute_stats


def pted_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    permutations: int = 100,
    metric: Union[str, float] = "euclidean",
    prog_bar: bool = False,
) -> tuple[float, list[float]]:
    assert torch.__version__ != "null", "PyTorch is not installed! try: `pip install torch`"
    z = torch.cat((x, y), dim=0)
    assert torch.all(torch.isfinite(z)), "Input contains NaN or Inf!"
    if metric == "euclidean":
        metric = 2.0
    dmatrix = torch.cdist(z, z, p=metric)
    assert torch.all(
        torch.isfinite(dmatrix)
    ), "Distance matrix contains NaN or Inf! Consider using a different metric or normalizing values to be more stable (i.e. z-score norm)."
    nx = len(x)
    ny = len(y)

    test_stat = _energy_distance_precompute(dmatrix, nx, ny).item()
    permute_stats = []
    for _ in trange(permutations, disable=not prog_bar):
        I = torch.randperm(len(z))
        dmatrix = dmatrix[I][:, I]
        permute_stats.append(_energy_distance_precompute(dmatrix, nx, ny).item())
    return test_stat, permute_stats


def pted_jax(
    x,
    y,
    permutations: int = 100,
    metric: Union[str, float] = "euclidean",
    prog_bar: bool = False,
) -> tuple[float, list[float]]:
    assert jax is not None, "JAX is not installed! try: `pip install jax`"
    z = jnp.concatenate([x, y], axis=0)
    assert jnp.all(jnp.isfinite(z)), "Input contains NaN or Inf!"
    if metric == "euclidean":
        metric = 2.0
    dmatrix = _jax_cdist(z, z, p=metric)
    assert jnp.all(
        jnp.isfinite(dmatrix)
    ), "Distance matrix contains NaN or Inf! Consider using a different metric or normalizing values to be more stable (i.e. z-score norm)."
    nx = len(x)
    ny = len(y)

    test_stat = float(_energy_distance_precompute(dmatrix, nx, ny))
    permute_stats = []
    for _ in trange(permutations, disable=not prog_bar):
        I = np.random.permutation(len(z))
        dmatrix = dmatrix[I][:, I]
        permute_stats.append(float(_energy_distance_precompute(dmatrix, nx, ny)))
    return test_stat, permute_stats


def pted_chunk_jax(
    x,
    y,
    permutations: int = 100,
    metric: Union[str, float] = "euclidean",
    chunk_size: int = 100,
    chunk_iter: int = 10,
    prog_bar: bool = False,
) -> tuple[float, list[float]]:
    assert jax is not None, "JAX is not installed! try: `pip install jax`"
    assert jnp.all(jnp.isfinite(x)) and jnp.all(jnp.isfinite(y)), "Input contains NaN or Inf!"
    nx = len(x)

    test_stat = _energy_distance_estimate_jax(x, y, chunk_size, chunk_iter, metric=metric)
    permute_stats = []
    for _ in trange(permutations, disable=not prog_bar):
        z = jnp.concatenate([x, y], axis=0)
        z = z[np.random.permutation(len(z))]
        x, y = z[:nx], z[nx:]
        permute_stats.append(
            _energy_distance_estimate_jax(x, y, chunk_size, chunk_iter, metric=metric)
        )
    return test_stat, permute_stats


def two_tailed_p(chi2, df):
    assert df > 2, "Degrees of freedom must be greater than 2 for two-tailed p-value calculation."
    alpha = chi2_dist.pdf(chi2, df)
    mode = df - 2

    if np.isclose(chi2, mode):
        return 1.0

    def root_eq(x):
        return chi2_dist.pdf(x, df) - alpha

    # Find left root
    if chi2 < mode:
        left = chi2_dist.cdf(chi2, df)
    else:
        res_left = root_scalar(root_eq, bracket=[0, mode], method="brentq")
        left = chi2_dist.cdf(res_left.root, df)

    # Find right root
    if chi2 > mode:
        right = chi2_dist.sf(chi2, df)
    else:
        res_right = root_scalar(root_eq, bracket=[mode, 10000 * df], method="brentq")
        right = chi2_dist.sf(res_right.root, df)

    return left + right


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
