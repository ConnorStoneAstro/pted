from typing import Union, Optional
import numpy as np

from .utils import (
    is_torch_tensor,
    pted_torch,
    pted_numpy,
    pted_chunk_torch,
    pted_chunk_numpy,
    two_tailed_p,
    confidence_alert,
    simulation_based_calibration_histogram,
)

__all__ = ["pted", "pted_coverage_test"]


def pted(
    x: Union[np.ndarray, "Tensor"],
    y: Union[np.ndarray, "Tensor"],
    permutations: int = 1000,
    metric: Union[str, float] = "euclidean",
    return_all: bool = False,
    chunk_size: Optional[int] = None,
    chunk_iter: Optional[int] = None,
    two_tailed: bool = True,
    prog_bar: bool = False,
) -> Union[float, tuple[float, np.ndarray, float]]:
    """
    Two sample null hypothesis test using a permutation test on the energy
    distance.

    A "two sample test" is a statistical test that compares two samples to
    determine if they come from the same distribution. The null hypothesis is
    that the two samples come from the same distribution. A permutation test is
    a non-parametric test that compares a test statistic (in this case, the
    energy distance) to that same statistic computed on random re-shuffling
    (permutations) of the data. Under the null hypothesis, x and y were drawn
    from the same distribution so test statistic should be randomly distributed
    among the permutation statistics. If the test statistic is significantly
    larger than the permuted statistics, the p-value will be very small. Before
    running pted, you should choose a threshold at which you will reject the
    null. for example, if you choose a threshold of 0.01, you will reject the
    null hypothesis if the p-value is less than 0.01. However, note that in this
    case you will reject the null 1% of the time even if the null is true. This
    is a trade-off between false positives and false negatives.

    Here is a pseudo-code description of the algorithm:
        test_stat = energy_distance(x, y)
        permute_stats = []
        for i in range(permutations):
            z = concatenate(x, y)
            z = shuffle(z)
            x, y = z[:nx], z[nx:]
            permute_stats.append(energy_distance(x, y))
        p = sum(permute_stats > test_stat)
        return (1 + p) / (1 + permutations)

    Example
    -------
        import numpy as np
        from pted import pted

        # Generate two samples from the same distribution
        x = np.random.normal(size=(100, 10))
        y = np.random.normal(size=(100, 10))

        p = pted(x, y)

        print(f"p-value: {p}") # expect p in U(0,1)

    Parameters
    ----------
        x (Union[np.ndarray, Tensor]): first set of samples. Shape (N, *D)
        y (Union[np.ndarray, Tensor]): second set of samples. Shape (M, *D)
        permutations (int): number of permutations to run. This determines how
            accurately the p-value is computed.
        metric (Union[str, float]): distance metric to use. See scipy.spatial.distance.cdist
            for the list of available metrics with numpy. See torch.cdist when
            using PyTorch, note that the metric is passed as the "p" for
            torch.cdist and therefore is a float from 0 to inf.
        return_all (bool): if True, return the test statistic and the permuted
            statistics with the p-value. If False, just return the p-value.
            bool (default: False)
        chunk_size (Optional[int]): if not None, use chunked energy distance
            estimation. This is useful for large datasets. The chunk size is the
            number of samples to use for each chunk. If None, use the full
            dataset.
        chunk_iter (Optional[int]): The chunk iter is the number of iterations
            to use with the given chunk size.
        two_tailed (bool): if True, compute a two-tailed p-value. This is useful
            if you want to reject the null hypothesis when x and y are either
            too similar or too different. Default is True.
        prog_bar (bool): if True, show a progress bar to track the progress
            of permutation tests. Default is False.


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
        to the difference between x and y. The chunked pted test has time
        complexity O(c^2 * I * D * P), where c is the chunk size, I is the
        number of iterations, D is the number of dimensions, and P is the number
        of permutations. For chunking to be worth it you should have c^2 * I << n^2.
    """
    assert type(x) == type(y), f"x and y must be of the same type, not {type(x)} and {type(y)}"
    assert len(x.shape) >= 2, f"x must be at least 2D, not {x.shape}"
    assert len(y.shape) >= 2, f"y must be at least 2D, not {y.shape}"
    assert (chunk_size is not None) is (
        chunk_iter is not None
    ), "chunk_size and chunk_iter must both be provided for chunked PTED test"
    assert (
        x.shape[1:] == y.shape[1:]
    ), f"x and y samples must have the same shape (past first dim), not {x.shape} and {y.shape}"
    if len(x.shape) > 2:
        x = x.reshape(x.shape[0], -1)
    if len(y.shape) > 2:
        y = y.reshape(y.shape[0], -1)

    if is_torch_tensor(x) and chunk_size is not None:
        test, permute = pted_chunk_torch(
            x,
            y,
            permutations=permutations,
            metric=metric,
            chunk_size=int(chunk_size),
            chunk_iter=int(chunk_iter),
            prog_bar=prog_bar,
        )
    elif is_torch_tensor(x):
        test, permute = pted_torch(x, y, permutations=permutations, metric=metric, prog_bar=prog_bar)
    elif chunk_size is not None:
        test, permute = pted_chunk_numpy(
            x,
            y,
            permutations=permutations,
            metric=metric,
            chunk_size=int(chunk_size),
            chunk_iter=int(chunk_iter),
            prog_bar=prog_bar,
        )
    else:
        test, permute = pted_numpy(x, y, permutations=permutations, metric=metric, prog_bar=prog_bar)

    permute = np.array(permute)

    # Compute p-value
    if two_tailed:
        q = 2 * min(np.sum(permute >= test), np.sum(permute <= test))
        q = min(q, permutations)
    else:
        q = np.sum(permute >= test)

    p = (1.0 + q) / (1.0 + permutations)

    if return_all:
        return test, permute, p
    return p


def pted_coverage_test(
    g: Union[np.ndarray, "Tensor"],
    s: Union[np.ndarray, "Tensor"],
    permutations: int = 1000,
    metric: Union[str, float] = "euclidean",
    warn_confidence: Optional[float] = 1e-3,
    return_all: bool = False,
    chunk_size: Optional[int] = None,
    chunk_iter: Optional[int] = None,
    sbc_histogram: Optional[str] = None,
    sbc_bins: Optional[int] = None,
    prog_bar: bool = False,
) -> Union[float, tuple[np.ndarray, np.ndarray, float]]:
    """
    Coverage test using a permutation test on the energy distance.

    A "coverage test" is a statistical test that determines if the posterior
    samples (s) cover the ground truth samples (g) with the correct uncertainty.
    By "correct uncertainty" we mean that any region R which contains a% of the
    posterior samples should contain the ground truth g with a% probability. A
    posterior s which is "overconfident" will have very little variability in
    it's samples, therefore the ground truth g will tend to be far off from the
    samples in relative terms. This will lead to a low p-value. A posterior s
    which is "underconfident" will have too much variability in it's samples,
    therefore the ground truth g will be enclosed too well within the
    distribution of samples. This will lead to a high p-value. The null
    hypothesis is that the posterior samples cover the ground truth samples with
    the correct uncertainty. If the null hypothesis is true, the p-value will be
    distributed as U(0,1).

    To perform this test, we compute the pted p-value for each simulation
    independently. We then compute the p-value under the null hypothesis that
    the pted p-values are distributed as U(0,1). This is done by computing the
    chi-squared statistic of the p-values (which for U(0,1) means chi2 = -2 *
    log(p)). The total p-value is then computed as 1 - chi2_cdf(sum(chi2), 2 *
    n_sims). Note, that because p is computed with a finite number of
    iterations, it is possible that p=0 in which case log(p) = -inf. To handle
    this, we set p=1/n_permutations. This is essentially the smallest p-value
    estimate reasonable for n_permutations.


    Example Usage
    ----------------
        import numpy as np
        from pted import pted_coverage_test

        # Generate mock ground truth samples (n_simulations, n_dimensions)
        g = np.random.normal(size=(100, 10))

        # Generate mock posterior samples (n_samples, n_simulations, n_dimensions)
        s = np.random.normal(size=(200, 100, 10))

        p = pted_coverage_test(g, s)

        print(f"p-value: {p}") # expect p in U(0,1)


    Parameters
    ----------
        g (Union[np.ndarray, Tensor]): Ground truth samples. Shape (n_sims, *D)
        s (Union[np.ndarray, Tensor]): Posterior samples. Shape (n_samples, n_sims, *D)
        permutations (int): number of permutations to run. This determines how
            accurately the p-value is computed.
        metric (Union[str, float]): distance metric to use. See scipy.spatial.distance.cdist
            for the list of available metrics with numpy. See torch.cdist when using
            PyTorch, note that the metric is passed as the "p" for torch.cdist and
            therefore is a float from 0 to inf.
        return_all (bool): if True, return the test statistic and the permuted
            statistics with the p-value. If False, just return the p-value. bool
            (default: False)
        chunk_size (Optional[int]): If not None, use chunked energy distance
            estimation. This is useful for large datasets. The chunk size is the
            number of samples to use for each chunk. If None, use the full
            dataset.
        chunk_iter (Optional[int]): The chunk iter is the number of iterations
            to use with the given chunk size.
        sbc_histogram (Optional[str]): If given, the path/filename to save a
            Simulation-Based-Calibration histogram.
        sbc_bins (Optional[int]): If given, force the histogram to have the provided
            number of bins. Otherwise, select an appropriate size: ~sqrt(N).
        prog_bar (bool): If True, show a progress bar to track the progress
            of permutation tests. Default is False.

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
        to the difference between x and y. The chunked pted test has time
        complexity O(c^2 * I * D * P), where c is the chunk size, I is the
        number of iterations, D is the number of dimensions, and P is the number
        of permutations. For chunking to be worth it you should have c^2 * I << n^2.
    """
    nsamp, nsim, *_ = s.shape
    assert nsim > 0, "need some simulations to run test, got 0 simulations"
    assert (
        g.shape == s.shape[1:]
    ), f"g and s must have the same shape (past first dim of s), not {g.shape} and {s.shape}"
    if len(s.shape) > 3:
        s = s.reshape(nsamp, nsim, -1)
    g = g.reshape(1, nsim, -1)

    test_stats = []
    permute_stats = []
    pvals = []
    for i in range(nsim):
        test, permute, p = pted(
            g[:, i],
            s[:, i],
            permutations=permutations,
            metric=metric,
            return_all=True,
            two_tailed=False,
            chunk_size=chunk_size,
            chunk_iter=chunk_iter,
            prog_bar=prog_bar,
        )
        test_stats.append(test)
        permute_stats.append(permute)
        pvals.append(p)
    test_stats = np.array(test_stats)  # (nsim,)
    permute_stats = np.stack(permute_stats)  # (nsim, npermute)
    pvals = np.array(pvals)

    # Simulation-Based-Calibration histogram
    if sbc_histogram is not None:
        ranks = np.sum(test_stats[:, None] >= permute_stats, axis=1) / permutations
        simulation_based_calibration_histogram(ranks, sbc_histogram, bins=sbc_bins)

    # Compute p-value
    if nsim == 1:
        return pvals[0]
    chi2 = np.sum(-2 * np.log(pvals))
    if warn_confidence is not None and warn_confidence is not False:
        confidence_alert(chi2, 2 * nsim, warn_confidence)

    p = two_tailed_p(chi2, 2 * nsim)

    if return_all:
        return test_stats, permute_stats, p
    return p
