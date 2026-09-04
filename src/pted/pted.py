from tqdm.auto import trange
from typing import Union, Optional
import numpy as np

from .utils import (
    permutation_energy_test as _energy_test,
    _as_rng,
    two_tailed_p,
    confidence_alert,
    simulation_based_calibration_histogram,
    pit_plot as _pit_plot,
)

__all__ = ["pted", "pted_coverage_test"]


def pted(
    x: Union[np.ndarray, "Tensor", "jax.Array"],
    y: Union[np.ndarray, "Tensor", "jax.Array"],
    permutations: int = 1000,
    return_all: bool = False,
    chunk_size: Optional[int] = None,
    two_tailed: bool = True,
    prog_bar: bool = False,
    n_columns: Optional[int] = None,
    batch_size: Optional[int] = None,
    rng=None,
) -> Union[float, tuple[float, np.ndarray, float]]:
    """
    Two sample null hypothesis test using a permutation test on the energy
    distance (Euclidean).

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

    The energy distance is computed with the within-group means taken over
    distinct pairs, excluding the zero self-pairs. This makes the statistic an
    unbiased estimator of the population energy distance, so it may come out
    slightly negative when x and y are drawn from the same distribution (and
    strongly negative when they are more alike than chance allows, e.g. shared
    samples). Only its rank among the permuted statistics matters.

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
        x (Union[np.ndarray, Tensor, jax.Array]): first set of samples. Shape (N, *D)
        y (Union[np.ndarray, Tensor, jax.Array]): second set of samples. Shape (M, *D)
        permutations (int): number of permutations to run. This determines how
            accurately the p-value is computed.
        return_all (bool): if True, return the test statistic and the permuted
            statistics with the p-value. If False, just return the p-value.
            bool (default: False)
        chunk_size (Optional[int]): if not None, estimate the energy distance
            from a rectangular distance matrix instead of the full pairwise
            matrix. Only distances from every sample to ``min(chunk_size,
            len(x)) + min(chunk_size, len(y))`` "column" points drawn from the
            pooled sample are computed, so the cost drops from ``O(n^2 d)`` to
            ``O(n c d)``. If ``chunk_size`` covers both full datasets, this
            falls back to the exact full-matrix computation. If None, use the
            full dataset. Mutually exclusive with ``n_columns``.
        two_tailed (bool): if True, compute a two-tailed p-value. This is useful
            if you want to reject the null hypothesis when x and y are either
            too similar or too different. Default is True.
        prog_bar (bool): if True, show a progress bar to track the progress
            of permutation tests. Default is False.
        n_columns (Optional[int]): the number of column points ``c`` directly.
            This is the preferred arg; ``chunk_size`` is the older one and
            simply maps onto it. Mutually exclusive with ``chunk_size``.
        batch_size (Optional[int]): number of permutations evaluated per matrix
            product. Larger values are faster (especially on GPU) at
            ``O(batch_size * n)`` extra memory. None picks a size that keeps a
            batch to a few million elements.
        rng: seed, ``np.random.Generator``, or None to draw from the global
            numpy state, so ``np.random.seed`` still controls reproducibility.

    Note
    ----
        The full test builds the ``(n, n)`` distance matrix in ``O(n^2 d)``
        time, where n is the pooled sample size and d the dimension. Each
        permutation then costs ``O(n^2)``, though permutations are evaluated in
        batches as a single matrix product rather than one at a time. For large
        datasets this gets unwieldy, so chunking is recommended: only a
        rectangular ``(n, c)`` matrix of distances to ``c`` column points is
        built, in ``O(n c d)`` time, and each permutation costs ``O(n c)``.

        Chunking keeps the p-value exact. Permutations are drawn from the
        subgroup that holds the column set fixed, shuffling labels within the
        columns and within their complement but never between, so the observed
        labelling is exchangeable with the permuted ones. The test simply
        becomes less sensitive as ``c`` shrinks: the null spread grows like
        ``sqrt(n / c)``, so the detectable energy distance scales as
        ``(n c)^-0.5`` rather than ``n^-1``.

        The one thing chunking does cost is p-value resolution, and it bites
        hardest when one group holds a single point (the per-simulation test
        inside ``pted_coverage_test``). There the subgroup reaches only
        ``n - c`` distinct label assignments, so the smallest attainable
        p-value is about ``1 / (n - c)`` however many permutations are drawn.
        Keep ``chunk_size`` well below ``len(s)`` in that case; a
        ``PermutationResolutionWarning`` is raised when it is too large.
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

    # The column controls are resolved once, inside the test itself; a count
    # covering the whole pooled sample lands in the exact full-matrix regime.
    test, permute = _energy_test(
        x,
        y,
        permutations=permutations,
        chunk_size=chunk_size,
        n_columns=n_columns,
        prog_bar=prog_bar,
        batch_size=batch_size,
        rng=rng,
    )

    # Compute p-value
    if two_tailed:
        q = 2 * min(np.sum(permute >= test), np.sum(permute <= test))
        q = min(q, permutations)
    else:
        q = np.sum(permute >= test)

    pval = (1.0 + q) / (1.0 + permutations)

    if return_all:
        return test, permute, pval
    return pval


def pted_coverage_test(
    g: Union[np.ndarray, "Tensor", "jax.Array"],
    s: Union[np.ndarray, "Tensor", "jax.Array"],
    permutations: int = 1000,
    warn_confidence: Optional[float] = 1e-3,
    return_all: bool = False,
    chunk_size: Optional[int] = None,
    sbc_histogram: Optional[str] = None,
    sbc_bins: Optional[int] = None,
    pit_plot: Optional[str] = None,
    pit_confidence: float = 0.95,
    prog_bar: bool = False,
    n_columns: Optional[int] = None,
    batch_size: Optional[int] = None,
    rng=None,
) -> Union[float, tuple[np.ndarray, np.ndarray, float]]:
    """
    Coverage test using a permutation test on the energy distance (Euclidean).

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
        g (Union[np.ndarray, Tensor, jax.Array]): Ground truth samples. Shape (n_sims, *D)
        s (Union[np.ndarray, Tensor, jax.Array]): Posterior samples. Shape (n_samples, n_sims, *D)
        permutations (int): number of permutations to run. This determines how
            accurately the p-value is computed.
        return_all (bool): if True, return the test statistic and the permuted
            statistics with the p-value. If False, just return the p-value. bool
            (default: False)
        chunk_size (Optional[int]): if not None, estimate the energy distance
            from a rectangular distance matrix instead of the full pairwise
            matrix. Only distances from every sample to ``min(chunk_size,
            len(x)) + min(chunk_size, len(y))`` "column" points drawn from the
            pooled sample are computed, so the cost drops from ``O(n^2 d)`` to
            ``O(n c d)``. If ``chunk_size`` covers both full datasets, this
            falls back to the exact full-matrix computation. If None, use the
            full dataset. Mutually exclusive with ``n_columns``.
        sbc_histogram (Optional[str]): If given, the path/filename to save a
            Simulation-Based-Calibration histogram.
        sbc_bins (Optional[int]): If given, force the histogram to have the provided
            number of bins. Otherwise, select an appropriate size: ~sqrt(N).
        pit_plot (Optional[str]): If given, the path/filename to save a
            Probability Integral Transform (PIT) plot. The plot shows the
            empirical CDF of the per-simulation p-values against the expected
            uniform CDF (1:1 diagonal), together with a shaded KS confidence
            band. Deviations outside the band indicate that the p-values are
            not uniformly distributed. Default is None (no plot saved).
        pit_confidence (float): Confidence level for the KS confidence band
            in the PIT plot. Default is 0.95 (95%). Only used when
            ``pit_plot`` is not None.
        prog_bar (bool): If True, show a progress bar to track the progress
            of simulations. Default is False.
        n_columns (Optional[int]): the number of column points ``c`` directly.
            This is the preferred arg; ``chunk_size`` is the older one and
            simply maps onto it. Mutually exclusive with ``chunk_size``.
        batch_size (Optional[int]): number of permutations evaluated per matrix
            product. Larger values are faster (especially on GPU) at
            ``O(batch_size * n)`` extra memory. None picks a size that keeps a
            batch to a few million elements.
        rng: seed, ``np.random.Generator``, or None to draw from the global
            numpy state, so ``np.random.seed`` still controls reproducibility.

    Note
    ----
        The full test builds the ``(n, n)`` distance matrix in ``O(n^2 d)``
        time, where n is the pooled sample size and d the dimension. Each
        permutation then costs ``O(n^2)``, though permutations are evaluated in
        batches as a single matrix product rather than one at a time. For large
        datasets this gets unwieldy, so chunking is recommended: only a
        rectangular ``(n, c)`` matrix of distances to ``c`` column points is
        built, in ``O(n c d)`` time, and each permutation costs ``O(n c)``.

        Chunking keeps the p-value exact. Permutations are drawn from the
        subgroup that holds the column set fixed, shuffling labels within the
        columns and within their complement but never between, so every
        normalising constant in the statistic is a constant and the observed
        labelling is exchangeable with the permuted ones. The test simply
        becomes less sensitive as ``c`` shrinks: the null spread grows like
        ``sqrt(n / c)``, so the detectable energy distance scales as
        ``(n c)^-0.5`` rather than ``n^-1``.

        The one thing chunking does cost is p-value resolution, and it bites
        hardest when one group holds a single point (the per-simulation test
        inside ``pted_coverage_test``). There the subgroup reaches only
        ``n - c`` distinct label assignments, so the smallest attainable
        p-value is about ``1 / (n - c)`` however many permutations are drawn.
        Keep ``chunk_size`` well below ``len(s)`` in that case; a
        ``PermutationResolutionWarning`` fires when it is too large.
    """
    nsamp, nsim, *_ = s.shape
    assert nsim > 0, "need some simulations to run test, got 0 simulations"
    assert (
        g.shape == s.shape[1:]
    ), f"g and s must have the same shape (past first dim of s), not {g.shape} and {s.shape}"
    if len(s.shape) > 3:
        s = s.reshape(nsamp, nsim, -1)
    g = g.reshape(1, nsim, -1)

    # Coerce once, so the generator's state advances across simulations. Passing
    # a bare seed straight through would hand every simulation the same
    # permutation draws.
    rng = _as_rng(rng)

    test_stats = []
    permute_stats = []
    pvals = []
    for i in trange(nsim, disable=not prog_bar):
        test, permute, p = pted(
            g[:, i],
            s[:, i],
            permutations=permutations,
            return_all=True,
            two_tailed=False,
            chunk_size=chunk_size,
            n_columns=n_columns,
            batch_size=batch_size,
            rng=rng,
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

    # Probability Integral Transform (PIT) plot
    if pit_plot is not None:
        _pit_plot(pvals, pit_plot, confidence=pit_confidence)

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
