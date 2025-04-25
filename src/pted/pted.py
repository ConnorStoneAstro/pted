from typing import Union, Optional
import numpy as np
from scipy.stats import chi2 as chi2_dist
from torch import Tensor

from .utils import _pted_torch, _pted_numpy, _pted_chunk_torch, _pted_chunk_numpy

__all__ = ["pted", "pted_coverage_test"]


def pted(
    x: Union[np.ndarray, Tensor],
    y: Union[np.ndarray, Tensor],
    permutations: int = 1000,
    metric: str = "euclidean",
    return_all: bool = False,
    chunk_size: Optional[int] = None,
    chunk_iter: Optional[int] = None,
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
            for the list of available metrics with numpy. See torch.cdist when
            using PyTorch, note that the metric is passed as the "p" for
            torch.cdist and therefore is a float from 0 to inf.
        return_all (bool): if True, return the test statistic and the permuted
            statistics. If False, just return the p-value. bool (False by default)
        chunk_size (Optional[int]): if not None, use chunked energy distance
            estimation. This is useful for large datasets. The chunk size is the
            number of samples to use for each chunk. If None, use the full
            dataset.
        chunk_iter (Optional[int]): The chunk iter is the number of iterations
            to use with the given chunk size.

    Note
    ----
        PTED has O(n^2 * D * P) time complexity, where n is the number of
        samples in x and y, D is the number of dimensions, and P is the number
        of permutations. For large datasets this can get unwieldy, so chunking
        is recommended. For chunking, the energy distance will be estimated at
        each iteration rather than fully computed. To estimate the energy
        distance, we take `chunk_size` sub-samples from x and y, and compute the
        energy distance on those sub-samples. This is repeated `chunk_iter`
        times, and the average is taken. This is a trade-off between speed and
        accuracy. The larger the chunk size and larger chunk_iter, the more
        accurate the estimate, but the slower the computation. PTED remains an
        exact p-value test even when chunking, it simply becomes less sensitive
        to the difference between x and y.
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

    if isinstance(x, Tensor) and chunk_size is not None:
        test, permute = _pted_chunk_torch(
            x,
            y,
            permutations=permutations,
            metric=metric,
            chunk_size=chunk_size,
            chunk_iter=chunk_iter,
        )
    elif isinstance(x, Tensor):
        test, permute = _pted_torch(x, y, permutations=permutations, metric=metric)
    elif chunk_size is not None:
        test, permute = _pted_chunk_numpy(
            x,
            y,
            permutations=permutations,
            metric=metric,
            chunk_size=chunk_size,
            chunk_iter=chunk_iter,
        )
    else:
        test, permute = _pted_numpy(x, y, permutations=permutations, metric=metric)

    if return_all:
        return test, permute

    # Compute p-value
    return np.mean(np.array(permute) > test)


def pted_coverage_test(
    g: Union[np.ndarray, Tensor],
    s: Union[np.ndarray, Tensor],
    permutations: int = 1000,
    metric: str = "euclidean",
    return_all: bool = False,
    chunk_size: Optional[int] = None,
    chunk_iter: Optional[int] = None,
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
        chunk_size (Optional[int]): if not None, use chunked energy distance
            estimation. This is useful for large datasets. The chunk size is the
            number of samples to use for each chunk. If None, use the full
            dataset.
        chunk_iter (Optional[int]): The chunk iter is the number of iterations
            to use with the given chunk size.

    Note
    ----
        PTED has O(n^2 * D * P) time complexity, where n is the number of
        samples in x and y, D is the number of dimensions, and P is the number
        of permutations. For large datasets this can get unwieldy, so chunking
        is recommended. For chunking, the energy distance will be estimated at
        each iteration rather than fully computed. To estimate the energy
        distance, we take `chunk_size` sub-samples from x and y, and compute the
        energy distance on those sub-samples. This is repeated `chunk_iter`
        times, and the average is taken. This is a trade-off between speed and
        accuracy. The larger the chunk size and larger chunk_iter, the more
        accurate the estimate, but the slower the computation. PTED remains an
        exact p-value test even when chunking, it simply becomes less sensitive
        to the difference between x and y.
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
            g[:, i],
            s[:, i],
            permutations=permutations,
            metric=metric,
            return_all=True,
            chunk_size=chunk_size,
            chunk_iter=chunk_iter,
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
