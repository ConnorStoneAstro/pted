import numpy as np
from scipy.stats import chi2 as chi2_dist
from torch import Tensor

from .utils import _pted_torch, _pted_numpy

__all__ = ["pted", "pted_coverage_test"]


def pted(x, y, permutations=100, metric="euclidean", return_all=False):
    """
    Two sample test using a permutation test on the energy distance.

    Parameters
    ----------
        x: first set of samples. Shape (N, D)
        y: second set of samples. Shape (M, D)
        permutations: number of permutations to run. int
        metric: distance metric to use. str
        return_all: if True, return the test statistic and the permuted
        statistics. If False, just return the p-value. bool (False by default)
    """
    assert type(x) == type(y), f"x and y must be of the same type, not {type(x)} and {type(y)}"
    assert len(x.shape) >= 2, f"x must be at least 2D, not {x.shape}"
    assert len(y.shape) >= 2, f"y must be at least 2D, not {y.shape}"

    if isinstance(x, Tensor) and isinstance(y, Tensor):
        return _pted_torch(x, y, permutations=permutations, metric=metric, return_all=return_all)
    return _pted_numpy(x, y, permutations=permutations, metric=metric, return_all=return_all)


def pted_coverage_test(g, s, permutations=100, metric="euclidean", return_all=False):
    """
    Coverage test using a permutation test on the energy distance.

    Parameters
    ----------
        g: Ground truth samples. Shape (n_sims, D)
        s: Posterior samples. Shape (n_samples, n_sims, D)
        permutations: number of permutations to run. int
        metric: distance metric to use. str
    """
    _, nsim, *D = s.shape
    test_stats = []
    permute_stats = []
    for i in range(nsim):
        test, permute = pted(
            g[i].reshape(1, *D), s[:, i], permutations=permutations, metric=metric, return_all=True
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
