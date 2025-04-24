from typing import Union
import numpy as np
from scipy.stats import chi2 as chi2_dist
from torch import Tensor

from .utils import _pted_torch, _pted_numpy

__all__ = ["pted", "pted_coverage_test"]


def pted(
    x: Union[np.ndarray, Tensor],
    y: Union[np.ndarray, Tensor],
    permutations: int = 1000,
    metric: str = "euclidean",
    return_all: bool = False,
):
    """
    Two sample test using a permutation test on the energy distance.

    Parameters
    ----------
        x (Union[np.ndarray, Tensor]): first set of samples. Shape (N, *D)
        y (Union[np.ndarray, Tensor]): second set of samples. Shape (M, *D)
        permutations (int): number of permutations to run. This determines how
            accurately the p-value is computed.
        metric (str): distance metric to use. See scipy.spatial.distance.cdist
            for the list of available metrics with numpy. See torch.cdist when using
            PyTorch, note that the metric is passed as the "p" for torch.cdist and
            therefore is a float from 0 to inf.
        return_all (bool): if True, return the test statistic and the permuted
            statistics. If False, just return the p-value. bool (False by
            default)
    """
    assert type(x) == type(y), f"x and y must be of the same type, not {type(x)} and {type(y)}"
    assert len(x.shape) >= 2, f"x must be at least 2D, not {x.shape}"
    assert len(y.shape) >= 2, f"y must be at least 2D, not {y.shape}"
    assert (
        x.shape[1:] == y.shape[1:]
    ), f"x and y samples must have the same shape (past first dim), not {x.shape} and {y.shape}"
    if len(x.shape) > 2:
        x = x.reshape(x.shape[0], -1)
    if len(y.shape) > 2:
        y = y.reshape(y.shape[0], -1)

    if isinstance(x, Tensor) and isinstance(y, Tensor):
        return _pted_torch(x, y, permutations=permutations, metric=metric, return_all=return_all)
    return _pted_numpy(x, y, permutations=permutations, metric=metric, return_all=return_all)


def pted_coverage_test(
    g: Union[np.ndarray, Tensor],
    s: Union[np.ndarray, Tensor],
    permutations: int = 1000,
    metric: str = "euclidean",
    return_all: bool = False,
):
    """
    Coverage test using a permutation test on the energy distance.

    Parameters
    ----------
        g (Union[np.ndarray, Tensor]): Ground truth samples. Shape (n_sims, *D)
        s (Union[np.ndarray, Tensor]): Posterior samples. Shape (n_samples, n_sims, *D)
        permutations (int): number of permutations to run. This determines how
            accurately the p-value is computed.
        metric (str): distance metric to use. See scipy.spatial.distance.cdist
            for the list of available metrics with numpy. See torch.cdist when using
            PyTorch, note that the metric is passed as the "p" for torch.cdist and
            therefore is a float from 0 to inf.
        return_all (bool): if True, return the test statistic and the permuted
            statistics. If False, just return the p-value. bool (False by
            default)
    """
    nsamp, nsim, *D = s.shape
    assert (
        g.shape == s.shape[1:]
    ), f"g and s must have the same shape (past first dim of s), not {g.shape} and {s.shape}"
    if len(s.shape) > 3:
        s = s.reshape(nsamp, nsim, -1)
    g = g.reshape(1, nsim, -1)
    test_stats = []
    permute_stats = []
    for i in range(nsim):
        test, permute = pted(
            g[:, i], s[:, i], permutations=permutations, metric=metric, return_all=True
        )
        test_stats.append(test)
        permute_stats.append(permute)
    test_stats = np.array(test_stats)
    permute_stats = np.array(permute_stats)
    if return_all:
        return test_stats, permute_stats
    # Compute p-values
    pvals = np.mean(permute_stats > test_stats[:, None], axis=1)
    pvals[pvals == 0] = 1.0 / permutations  # handle pvals == 0
    chi2 = -2 * np.log(pvals)
    return 1 - chi2_dist.cdf(np.sum(chi2), 2 * nsim)
