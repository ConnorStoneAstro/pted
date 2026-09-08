"""Backend-agnostic machinery for the permutation test on the energy distance.

The pooled sample is ``z = [x; y]`` of size ``n = n1 + n2``. Rather than the
full ``n x n`` distance matrix of Szekely & Rizzo (2004), the test may be run
on a rectangular ``n x m`` matrix ``D = cdist(z, z[landmarks])``, where
``landmarks`` indexes ``m`` points drawn from the pooled sample. Every sample
is measured against those ``m`` landmarks and no other distance is computed:
``O(n * m)`` distances and ``O(n * m)`` work per permutation rather than
``O(n^2)``. Setting ``m = n`` recovers the exact full-matrix test, so both
live in one code path. This is the Nystrom-style subsampling familiar from
kernel methods, applied to the energy distance.

Validity
--------
Permutations are always confined to the subgroup ``S_L x S_{L^c}``, where ``L``
is the landmark set: labels are shuffled within the landmark positions and
within their complement, never across. Since ``sigma(L) = L`` identically, the
landmark set is invariant and the test is exact conditional on ``L``. This is
what licenses choosing ``L`` by design (balanced, or aligned with the group
structure). The one requirement is that ``L`` must not be chosen using the data
values -- sample positions and group labels are fine, but maximin, k-means or
leverage selection would break the argument. "Landmark" here means only "a
point everything is measured against", never "a point picked for importance".

Landmark allocation
-------------------
Dispatched on ``n_s = min(n1, n2)``, the smaller group:

``full``
    ``m >= n``. Every point is a landmark, ``S_L x S_{L^c}`` is the full
    symmetric group, and the statistic is the exact energy distance.

``singleton``
    ``n_s == 1``. Its within-group mean is identically zero (a single point
    has no within-group pairs), so nothing is unestimable and the lone point
    may sit on either side of ``L``. It goes on whichever side is larger:
    outside, its label roams over ``L^c`` for ``n - m`` assignments; inside,
    over the landmarks for ``m``. The reference set is therefore never smaller
    than ``ceil(n / 2)``.

``small_group_in_L``
    ``n_s <= m // 2``. ``L`` holds ALL of the small group plus ``m - n_s`` from
    the large group. Every permuted small group is then a subset of ``L``, so
    every within-small-group distance appears in ``D`` and that term is
    computed exactly rather than subsampled.

``proportional``
    Otherwise. ``m_s ~ m * n_s / n``, and both within-group terms are
    subsampled.

Cost
----
Null spread grows like ``sqrt(n / m)``, so the detectable energy distance
scales as ``(n * m)^{-1/2}`` versus ``n^{-1}`` for the full test: the detection
threshold degrades as one over the square root of the compute. See Janson
(1984) on incomplete U-statistics.

In the ``singleton`` regime a single row is nearly sufficient for the
statistic, so keep ``m`` small -- roughly 5-15% of ``n``. Landmarks there come
straight out of p-value resolution until ``m`` passes ``n / 2``, beyond which
you may as well pay for the full matrix.

Block means exclude the zero self-pairs, making the statistic unbiased for the
population energy distance. It can therefore be negative under H0, like
unbiased distance covariance. Only its rank among the permutations matters.
"""

from itertools import combinations, islice, product
from math import comb
from typing import Optional
from warnings import warn

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import chi2 as chi2_dist, binom
from tqdm.auto import tqdm

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
    "permutation_energy_test",
    "allocate_landmarks",
    "PermutationResolutionWarning",
    "two_tailed_p",
    "confidence_alert",
    "simulation_based_calibration_histogram",
    "pit_plot",
    "hdp_coverage_test",
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


def _to_numpy(x, backend: str) -> np.ndarray:
    if backend == "torch":
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _asarray_like(x, ref, backend: str):
    """Move ``x`` onto the device and dtype of ``ref``."""
    if backend == "torch":
        return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
    if backend == "jax":
        return jnp.asarray(x, dtype=ref.dtype)
    return np.asarray(x, dtype=ref.dtype)


def _index_like(idx, ref, backend: str):
    """Move an integer index array onto the device of ``ref``."""
    if backend == "torch":
        return torch.as_tensor(idx, dtype=torch.long, device=ref.device)
    if backend == "jax":
        return jnp.asarray(idx)
    return np.asarray(idx)


@jit
def _jax_cdist(x, y):
    return jax.vmap(lambda xi: jnp.linalg.norm(xi - y, ord=2.0, axis=-1))(x)


def _cdist(a, b, backend: str):
    if backend == "torch":
        return torch.cdist(a, b, p=2.0)
    if backend == "jax":
        return _jax_cdist(a, b)
    return cdist(a, b, metric="euclidean")


def _zero_self_pairs(D, landmarks, backend: str):
    """Set ``D[landmarks[k], k] = 0``, the entries pairing a point with itself.

    Called after ``D`` has been shifted by a constant, to restore the property
    that a point is at distance zero from itself. Self-pairs are excluded from
    every pair count, so leaving them shifted would bias the block means.

    Only positional self-pairs are touched. A genuine duplicate point elsewhere
    in the sample is a real pair at distance zero and stays counted. This also
    scrubs the ~1e-7 that ``torch.cdist`` leaves on the diagonal once it
    switches to its matmul-based algorithm.
    """
    k = np.arange(len(landmarks))
    if backend == "jax":
        return D.at[jnp.asarray(landmarks), jnp.asarray(k)].set(0.0)
    D[_index_like(landmarks, D, backend), _index_like(k, D, backend)] = 0.0
    return D


def _as_rng(rng) -> np.random.Generator:
    """Coerce ``rng`` to a numpy ``Generator``.

    ``None`` seeds a fresh generator from the legacy global state, so that
    ``np.random.seed(...)`` still controls reproducibility for callers who
    rely on it. An existing ``Generator`` is returned untouched, so a caller
    that has already coerced its own argument can pass the result down without
    reseeding anything.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng(np.random.randint(0, 2**32))
    return np.random.default_rng(rng)


class PermutationResolutionWarning(UserWarning):
    """The permutation group is too small to resolve the requested p-value."""


# comb(n, k) is astronomically large for balanced samples; we only ever need to
# know whether the reference set is small, so cap it.
_REFERENCE_CAP = 10**15

# Target element count for a batch of permutations, bounding peak memory.
_BATCH_ELEMENTS = 2**30


def _capped_comb(n: int, k: int) -> int:
    if k > 50 and n - k > 50:
        return _REFERENCE_CAP
    return min(comb(n, k), _REFERENCE_CAP)


# --------------------------------------------------------------- allocation


def allocate_landmarks(n1: int, n2: int, n_landmarks: int, rng=None) -> dict:
    """Choose which pooled-sample points serve as landmarks.

    The landmarks are the ``m`` points every sample is measured against: they
    supply the columns of the rectangular distance matrix ``D``.

    Group 0 is x (pooled indices ``0..n1-1``), group 1 is y (``n1..n-1``).
    Selection uses only group sizes and positions, never the data values, which
    together with the restricted permutation group is what keeps the test exact.

    Parameters
    ----------
        n1 (int): number of samples in x.
        n2 (int): number of samples in y.
        n_landmarks (int): number of landmarks ``m``. ``m >= n1 + n2``
            requests the exact full-matrix test.
        rng: seed, ``np.random.Generator``, or None (draw from the global state).

    Returns
    -------
        dict with keys ``landmarks``, ``regime``, ``small_idx`` (pooled indices
        of the smaller group), ``n_small``, ``n_large``, ``n_small_landmarks``,
        ``n_large_landmarks``, ``n_landmarks``, ``reference_size``
        (distinct label assignments the subgroup reaches)
        and ``exact_within`` (whether the small group's within-group term is
        computed exactly rather than subsampled).
    """
    rng = _as_rng(rng)
    n = n1 + n2
    n_landmarks = int(n_landmarks)
    if min(n1, n2) < 1:
        raise ValueError(f"both samples need at least one point, got n1={n1}, n2={n2}")
    if n_landmarks < 2:
        raise ValueError(f"need at least 2 landmarks, got n_landmarks={n_landmarks}")
    if n_landmarks > n:
        raise ValueError(f"n_landmarks={n_landmarks} exceeds the pooled sample size {n}")

    n_small, n_large = min(n1, n2), max(n1, n2)
    x_is_small = n1 <= n2
    small_idx = np.arange(n1) if x_is_small else n1 + np.arange(n2)
    large_idx = n1 + np.arange(n2) if x_is_small else np.arange(n1)

    if n_landmarks == n:
        # L is everything, so S_L x S_{L^c} is the full symmetric group and
        # this is the classical exact permutation test.
        landmarks = np.arange(n)
        regime, n_small_landmarks = "full", n_small
    elif n_small == 1:
        # A lone point has no within-group pairs, so it is free to sit on
        # either side of L. Inside, its label roams over the m landmark
        # positions; outside, over the n - m others. Take whichever side is
        # larger, which holds the reference set at ceil(n / 2) or more instead
        # of letting it collapse to 1 as m approaches n.
        regime = "singleton"
        if n_landmarks > n - n_landmarks:
            landmarks = np.concatenate(
                [small_idx, rng.choice(large_idx, size=n_landmarks - 1, replace=False)]
            )
            n_small_landmarks = 1
        else:
            landmarks = rng.choice(large_idx, size=n_landmarks, replace=False)
            n_small_landmarks = 0
    elif n_small <= n_landmarks // 2:
        # Put the whole small group in L so that every permuted small group is
        # still a subset of L and its within-group distances are all present.
        landmarks = np.concatenate(
            [small_idx, rng.choice(large_idx, size=n_landmarks - n_small, replace=False)]
        )
        regime, n_small_landmarks = "small_group_in_L", n_small
    else:
        lo = max(1, n_landmarks - n_large)
        hi = min(n_small, n_landmarks - 1)
        n_small_landmarks = int(np.clip(round(n_landmarks * n_small / n), lo, hi))
        landmarks = np.concatenate(
            [
                rng.choice(small_idx, size=n_small_landmarks, replace=False),
                rng.choice(large_idx, size=n_landmarks - n_small_landmarks, replace=False),
            ]
        )
        regime = "proportional"

    # The subgroup places n_small_landmarks of the small labels among the m
    # landmarks and the rest among the n - m other positions, independently.
    reference = min(
        _capped_comb(n_landmarks, n_small_landmarks)
        * _capped_comb(n - n_landmarks, n_small - n_small_landmarks),
        _REFERENCE_CAP,
    )

    return {
        "landmarks": np.sort(landmarks),
        "regime": regime,
        "small_idx": small_idx,
        "n_small": n_small,
        "n_large": n_large,
        "n_small_landmarks": n_small_landmarks,
        "n_large_landmarks": n_landmarks - n_small_landmarks,
        "n_landmarks": n_landmarks,
        "reference_size": reference,
        "exact_within": n_small_landmarks == n_small or n_small == 1,
    }


def _warn_reference_size(alloc: dict, permutations: int) -> None:
    """Warn when the permutation group cannot resolve the requested p-value.

    The observed labelling is itself a member of the group, so with only ``R``
    reachable assignments roughly one draw in ``R`` reproduces it exactly and
    the p-value is floored near ``1/R`` however many permutations are drawn.
    """
    reference = alloc["reference_size"]
    # In the full regime the reference set is fixed by the sample sizes alone,
    # so only flag it when it is degenerate. Everywhere else it is a
    # consequence of the landmark count, which the caller can act on.
    floor = 20 if alloc["regime"] == "full" else max(20, (permutations + 1) // 10)
    if reference >= floor:
        return

    hint = {
        "singleton": (
            f"the small group holds a single point, so the reference set is "
            f"max(n_landmarks, n - n_landmarks); move n_landmarks (currently "
            f"{alloc['n_landmarks']}) further from half the pooled sample size"
        ),
        "full": "the samples are too small for a permutation test at this resolution",
    }.get(
        alloc["regime"],
        f"use more landmarks (currently {alloc['n_landmarks']}) or more samples",
    )
    warn(
        PermutationResolutionWarning(
            f"the {alloc['regime']!r} landmark allocation reaches only {reference:,} distinct "
            f"label assignments, so the smallest attainable p-value is about "
            f"{1.0 / reference:.3g} no matter how many permutations are drawn "
            f"({permutations} requested): {hint}."
        )
    )


def _label_drawer(prep: dict, rng):
    """Return ``draw(rows) -> (rows, n)`` small-group indicator matrices.

    Each part -- the landmark positions and their complement -- gets a
    uniformly random subset of the size the subgroup fixes, found by ranking
    uniform keys: the ``k`` smallest of ``|part|`` iid uniforms are a uniform
    random ``k``-subset. Ranking is what makes this a device operation, so the
    labels are built wherever the distance matrix lives rather than shuffled on
    the host and shipped over -- for large ``n`` that shuffle costs more than
    the matrix product it feeds, and no accelerator can help with it.

    The returned function carries the generator state, so successive calls
    continue the same stream. Which permutations that yields depends on how the
    run is batched; the test is exact either way, and a fixed ``rng`` with a
    fixed ``batch_size`` reproduces a run exactly.
    """
    backend, ref = prep["backend"], prep["ref"]
    n = prep["base_small"].size
    parts = [(_index_like(idx, ref, backend), k) for idx, k in prep["parts"] if k]

    if backend == "torch":
        gen = torch.Generator(device=ref.device)
        gen.manual_seed(int(rng.integers(0, 2**62)))

        def draw(rows):
            U = torch.zeros((rows, n), dtype=ref.dtype, device=ref.device)
            for idx, k in parts:
                keys = torch.rand((rows, idx.numel()), generator=gen, device=ref.device)
                pick = torch.topk(keys, k, dim=1, largest=False, sorted=False).indices
                U.scatter_(1, idx[pick], 1.0)
            return U

    elif backend == "jax":
        key = jax.random.PRNGKey(int(rng.integers(0, 2**62)))

        def draw(rows):
            nonlocal key
            U = jnp.zeros((rows, n), dtype=ref.dtype)
            index = jnp.arange(rows)[:, None]
            for idx, k in parts:
                key, subkey = jax.random.split(key)
                keys = jax.random.uniform(subkey, (rows, int(idx.size)))
                U = U.at[index, idx[jax.lax.top_k(-keys, k)[1]]].set(1.0)
            return U

    else:

        def draw(rows):
            U = np.zeros((rows, n), dtype=ref.dtype)
            for idx, k in parts:
                keys = rng.random((rows, idx.size))
                pick = np.argpartition(keys, k - 1, axis=1)[:, :k]
                np.put_along_axis(U, idx[pick], U.dtype.type(1.0), axis=1)
            return U

    return draw


def _sampled_label_blocks(prep: dict, permutations: int, batch_size: int, rng):
    """Yield blocks of randomly drawn labels, ``permutations`` rows in total."""
    draw = _label_drawer(prep, rng)
    for start in range(0, permutations, batch_size):
        yield draw(min(batch_size, permutations - start))


def _enumerate_label_blocks(prep: dict, batch_size: int):
    """Yield blocks covering the whole subgroup, minus the observed labelling.

    The subgroup fixes how many small labels sit among the landmarks, so it is
    exactly: which of the landmark positions carry a small label, crossed with
    which of the remaining positions carry the rest. Dropping the observed
    assignment leaves ``reference_size - 1`` rows, so the caller's
    ``(1 + q) / (1 + len)`` becomes the exact rank of the observed statistic
    within the group -- no sampling, and no ties from redrawing the observed
    labelling.

    Only ever called when ``reference_size`` is small, so materialising the
    group a block at a time is cheap.
    """
    base = prep["base_small"]
    assignments = product(*(combinations(idx, k) for idx, k in prep["parts"]))
    while chunk := list(islice(assignments, batch_size)):
        U = np.zeros((len(chunk), base.size), dtype=base.dtype)
        for j, picks in enumerate(chunk):
            U[j, [i for pick in picks for i in pick]] = 1.0
        keep = np.any(U != base, axis=1)
        if keep.any():
            yield U[keep]


# ---------------------------------------------------------------- statistic


def _prepare_statistic(D, alloc: dict, backend: str) -> dict:
    """Precompute everything that is constant across permutations.

    Every block sum is a bilinear form in the row indicator ``u`` (length n)
    and the landmark indicator ``v = u[landmarks]`` (length m):

        u @ D @ v = sum_{i in A} sum_{k: landmarks[k] in A} D[i, k]

    All the denominators are constants, because the subgroup ``S_L x S_{L^c}``
    preserves the per-group landmark counts. That lets the whole statistic
    collapse into a single affine function of at most three per-permutation
    scalars, whose coefficients are folded down to plain Python floats here:

        E = a_ss * S_ss + a_row * R + a_col * C + const

    where ``R = u @ D @ 1`` and ``C = v @ (1 @ D)``. Everything else -- the
    other three block sums, the pair counts, the ``n_s n_l / n`` scaling -- is
    absorbed into those four numbers, so a permutation costs one matrix
    product and a handful of elementwise ops rather than a dozen small
    kernels on device.
    """
    n_s, n_l = alloc["n_small"], alloc["n_large"]
    n = n_s + n_l
    m_s, m_l = alloc["n_small_landmarks"], alloc["n_large_landmarks"]
    landmarks = alloc["landmarks"]
    full = alloc["regime"] == "full"

    # allocate_landmarks guarantees m_l >= 1 in every regime, and m_s >= 1
    # unless n_s == 1 (the singleton regime, where the small group has no
    # within-group pairs to estimate), so both block means below are defined.

    # The statistic is invariant to a constant offset on every non-self pair,
    # so centre D before any reduction. In high dimension the pairwise
    # distances concentrate far from zero, and the uncentred sums spend most of
    # their significant digits on an offset that cancels anyway. Any constant
    # cancels exactly, so the offset itself needs no precision -- a float32
    # mean is fine. That is not true of `total` below, whose error survives
    # into the reported statistic, so that one accumulates in float64.
    offset = float(D.mean())
    D = D - offset
    D = _zero_self_pairs(D, landmarks, backend)

    row_sums = D.sum(1)
    total = float(_to_numpy(row_sums, backend).sum(dtype=np.float64))

    # --- fold 2*mu_sl - mu_ss - mu_ll into A*S_ss + B*S_sl + C*S_ls + const --
    a, b, c, const = 0.0, 0.0, 0.0, 0.0
    if m_s > 0:  # both cross orientations are available
        b += 1.0 / (n_s * m_l)
        c += 1.0 / (n_l * m_s)
    else:  # no small landmarks, so only one orientation exists
        b += 2.0 / (n_s * m_l)
    if n_s < 2:  # a lone point has no within-group pairs; its mean is -offset
        const += offset
    else:
        a -= 1.0 / (m_s * (n_s - 1))
    if n_l < 2:
        const += offset
    else:  # S_ll = total - S_ss - S_sl - S_ls
        pairs_ll = m_l * (n_l - 1)
        a += 1.0 / pairs_ll
        b += 1.0 / pairs_ll
        c += 1.0 / pairs_ll
        const -= total / pairs_ll

    # S_sl = R - S_ss and S_ls = C - S_ss, so rewrite in terms of what a
    # permutation actually computes, then absorb the n_s n_l / n scaling.
    scale = n_s * n_l / n
    a_ss, a_row, a_col = scale * (a - b - c), scale * b, scale * c
    const *= scale

    # With no small landmarks both S_ss and C are identically zero, so the
    # matrix product is not needed at all -- a matrix-vector product against
    # the row sums is the whole statistic.
    needs_matrix = m_s > 0
    if full:
        # D is symmetric, so the column sums are the row sums and C == R.
        a_row, a_col = a_row + a_col, 0.0

    base_small = np.zeros(n)
    base_small[alloc["small_idx"]] = 1.0
    in_l = np.zeros(n, dtype=bool)
    in_l[landmarks] = True
    idx_l, idx_o = np.flatnonzero(in_l), np.flatnonzero(~in_l)
    # Whatever the statistic reads sets the device and dtype the labels are
    # built in, so they never have to cross a bus.
    ref = D if needs_matrix else row_sums

    return {
        "D": D if needs_matrix else None,
        "ref": ref,
        # (positions, how many of them carry the small label) for each half of
        # the subgroup; those counts are what the subgroup holds fixed.
        "parts": ((idx_l, m_s), (idx_o, n_s - m_s)),
        # The landmarks cover everything in the full regime, so skip the gather.
        "landmarks": None if full else _index_like(landmarks, D, backend),
        "col_sums": D.sum(0) if (needs_matrix and a_col) else None,
        "a_ss": a_ss,
        "a_row": a_row,
        "a_col": a_col,
        "const": const,
        "base_small": base_small,
        "backend": backend,
    }


def _evaluate_statistic(prep: dict, U) -> np.ndarray:
    """Energy statistic for a batch of label assignments.

    Parameters
    ----------
        prep (dict): output of :func:`_prepare_statistic`.
        U (np.ndarray): ``(B, n)`` indicator of small-group membership.

    Returns
    -------
        ``(B,)`` numpy array of statistics.
    """
    backend, D = prep["backend"], prep["D"]
    Us = _asarray_like(U, prep["ref"], backend)

    if D is None:  # no small landmarks, so ref is the row sums and they are the whole statistic
        stat = prep["a_row"] * (Us @ prep["ref"]) + prep["const"]
        return _to_numpy(stat, backend).astype(np.float64, copy=False)

    Vs = Us if prep["landmarks"] is None else Us[:, prep["landmarks"]]

    # One matrix product carries both scalars: W @ 1 is the row-sum term and
    # (W * Vs) summed is the small-by-small block.
    W = Us @ D
    stat = prep["a_ss"] * (W * Vs).sum(-1) + prep["a_row"] * W.sum(-1) + prep["const"]
    if prep["col_sums"] is not None:
        stat = stat + prep["a_col"] * (Vs @ prep["col_sums"])
    return _to_numpy(stat, backend).astype(np.float64, copy=False)


# --------------------------------------------------------------- public API


def permutation_energy_test(
    x,
    y,
    permutations: int,
    n_landmarks: Optional[int] = None,
    prog_bar: bool = False,
    batch_size: Optional[int] = None,
    rng=None,
) -> tuple[float, np.ndarray]:
    """Observed energy statistic and its permutation distribution.

    Without ``n_landmarks`` this is the exact full-matrix test: the ``(n, n)``
    distance matrix is built once and the statistic is evaluated for batches of
    permutations as a bilinear form in the group indicator, so no distance is
    recomputed and no matrix is re-indexed. Asking for fewer landmarks switches to the rectangular
    ``(n, m)`` matrix, cutting the cost from ``O(n^2 d)`` to ``O(n m d)``;
    permutations are then confined to the subgroup that fixes the landmark
    set, which keeps the p-value exact. See the module docstring for the
    landmark allocation regimes and what they cost in p-value resolution.

    Works transparently with numpy arrays, torch tensors, and jax arrays.
    Distances and all matrix algebra stay on the backend that ``x`` and ``y``
    live on; only the label bookkeeping and the resulting statistics touch
    numpy.

    Parameters
    ----------
        x, y: samples, shape ``(n1, d)`` and ``(n2, d)``.
        permutations (int): number of permutations to draw.
        n_landmarks (Optional[int]): number of landmarks ``m``, the points
            every sample is measured against. None, or any value covering the
            whole pooled sample, runs the exact full-matrix test.
        prog_bar (bool): show a progress bar over permutations.
        batch_size (Optional[int]): permutations evaluated per matrix product.
            Larger is faster on GPU, at ``O(batch_size * n)`` extra memory.
            None picks a size that bounds the batch at a few million elements.
        rng: seed, ``np.random.Generator``, or None to draw from the global
            numpy state (so ``np.random.seed`` still applies).

    Returns
    -------
        ``(test_stat, permute_stats)``, the observed statistic and an array of
        permuted statistics. That array holds ``permutations`` random draws --
        unless the subgroup has no more than ``permutations`` members, in which
        case it holds every one of them except the observed labelling
        (``reference_size - 1`` entries) and the resulting p-value is exact
        rather than sampled.
    """
    if is_torch_tensor(x):
        assert torch.__version__ != "null", "PyTorch is not installed! try: `pip install torch`"
    if is_jax_array(x):
        assert jax is not None, "JAX is not installed! try: `pip install jax`"

    backend = _backend(x)
    z = _concatenate((x, y), backend)
    assert _all_finite(z, backend), "Input contains NaN or Inf!"

    n1, n2 = len(x), len(y)
    n = n1 + n2
    # More landmarks than points is just the full matrix; anything else
    # nonsensical is rejected by allocate_landmarks, which owns that check.
    m = n if n_landmarks is None else min(int(n_landmarks), n)

    rng = _as_rng(rng)
    alloc = allocate_landmarks(n1, n2, m, rng)
    _warn_reference_size(alloc, permutations)

    landmarks = alloc["landmarks"]
    zc = z if alloc["regime"] == "full" else z[_index_like(landmarks, z, backend)]
    dmatrix = _cdist(z, zc, backend)
    assert _all_finite(dmatrix, backend), (
        "Distance matrix contains NaN or Inf! Consider normalizing values to be "
        "more stable (i.e. z-score norm)."
    )

    prep = _prepare_statistic(dmatrix, alloc, backend)

    test_stat = float(_evaluate_statistic(prep, prep["base_small"][None, :])[0])
    assert np.isfinite(test_stat), "Observed statistic is not finite!"

    if batch_size is None:
        batch_size = max(1, _BATCH_ELEMENTS // (n * m))
    batch_size = max(1, int(batch_size))

    # Sampling more permutations than the subgroup has members just redraws the
    # observed labelling over and over, and every such tie inflates q under the
    # >= convention. Below that point, walking the whole group is both exact
    # and cheaper.
    if alloc["reference_size"] <= permutations:
        n_null = int(alloc["reference_size"]) - 1
        blocks = _enumerate_label_blocks(prep, batch_size)
    else:
        n_null = permutations
        blocks = _sampled_label_blocks(prep, permutations, batch_size, rng)

    permute_stats = np.empty(n_null, dtype=np.float64)
    done = 0
    with tqdm(total=n_null, disable=not prog_bar) as bar:
        for U in blocks:
            size = len(U)
            permute_stats[done : done + size] = _evaluate_statistic(prep, U)
            done += size
            bar.update(size)
    assert done == n_null, f"produced {done} null statistics, expected {n_null}"

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


def _lattice_counts(pvals, t):
    """``N_k = #{p <= t_k}`` for each evaluation point, for one or many rows."""
    p = np.atleast_2d(pvals)
    rows, k = p.shape[0], len(t)
    flat = np.searchsorted(t, p, side="left") + np.arange(rows)[:, None] * k
    return np.cumsum(np.bincount(flat.ravel(), minlength=rows * k).reshape(rows, k), axis=1)


def _band_bounds(local, eta):
    """Counts whose local level clears ``eta``, per evaluation point.

    ``local[k]`` is a lower tail min'd with an upper tail, so it rises then
    falls and the set is a contiguous interval of counts.
    """
    inside = local > eta
    lower = np.argmax(inside, axis=1)
    return lower, lower + inside.sum(axis=1) - 1


def _band_accept_probability(t, n, lower, upper):
    """``P(lower_k <= N_k <= upper_k for every k)`` under the null, exactly.

    The counting process is Markov: given ``N_k = j`` the other ``n - j``
    p-values are iid on ``(t_k, 1]``, so the increment is
    ``Binomial(n - j, rho_k)`` with ``rho_k = (F0(t_k+1) - F0(t_k)) / (1 -
    F0(t_k))``. Pushing the distribution of ``N_k`` forward through those
    transitions, zeroing everything outside the interval at each step, leaves
    exactly the probability of never having escaped. No simulation needed.
    """
    f = binom.pmf(np.arange(n + 1), n, t[0])
    f[: lower[0]] = 0.0
    f[upper[0] + 1 :] = 0.0
    for k in range(len(t) - 1):
        rho = 1.0 if t[k] >= 1.0 else (t[k + 1] - t[k]) / (1.0 - t[k])
        j = np.arange(lower[k], upper[k] + 1)
        j_next = np.arange(lower[k + 1], upper[k + 1] + 1)
        step = binom.pmf(j_next[None, :] - j[:, None], (n - j)[:, None], rho)
        nxt = np.zeros(n + 1)
        nxt[lower[k + 1] : upper[k + 1] + 1] = f[lower[k] : upper[k] + 1] @ step
        f = nxt
    return float(f.sum())


def _lattice_band(n, lattice, confidence, max_points=512):
    """Simultaneous ECDF band built from the known null of the p-values.

    The ECDF only moves where the p-values can actually land, so this works
    with the counts there rather than with order statistics:

        N_k = #{i : p_i <= t_k}  ~  Binomial(n, F0(t_k))    exactly

    Every evaluation point gets the same two-sided *local* level ``eta``,
    chosen as large as it can be while the probability of *any* count escaping
    its interval stays within ``1 - confidence``. The ECDF then leaves the band
    with at most that probability under H0, so "pokes out of the band" reads as
    "reject", simultaneously over the whole curve. The boundary follows the
    real ``n F0(t) (1 - F0(t))`` variance of the count and pinches to nothing
    at both corners, rather than holding one constant half-width across the
    plot.

    Working with counts rather than order statistics is what lets this handle
    *discrete* p-values: ties are exactly what a binomial count expects,
    whereas the Beta distribution of an order statistic assumes they never
    happen.

    Everything here is exact -- both the escape probability, via
    :func:`_band_accept_probability`, and the search for ``eta``, which is a
    bisection over the finitely many local levels the table can take. There is
    no simulation and no seed.

    Because the null is discrete the attainable levels are discrete too, so the
    band is generally a little conservative; the level it actually achieves is
    returned rather than assumed.

    Parameters
    ----------
        n (int): number of p-values.
        lattice (Optional[int]): ``L`` if the p-values are uniform on
            ``{1/L, 2/L, ..., 1}`` under H0, as permutation p-values are. None
            for a continuous null.
        confidence (float): target simultaneous confidence level.
        max_points (int): cap on evaluation points. A finer lattice is checked
            on an evenly spaced subset, which stays valid and costs a little
            power.

    Returns
    -------
        ``(t, lower, upper, local, achieved)``: evaluation points, the allowed
        ECDF range at each as a fraction of ``n``, the table of local levels
        ``local[k, j]`` for count ``j`` at point ``k``, and the exact
        simultaneous level the band achieves.
    """
    if lattice is None:
        points = min(max_points, max(200, 20 * n))
        t = np.arange(1, points + 1) / points
    else:
        lattice = int(lattice)
        k = (
            np.arange(1, lattice + 1)
            if lattice <= max_points
            else np.unique(np.round(np.linspace(1, lattice, max_points)).astype(int))
        )
        t = k / lattice

    # F0(t_k) = t_k for both nulls; the lattice only changes where we may look.
    j = np.arange(n + 1)
    cdf = binom.cdf(j[None, :], n, t[:, None])  # P(N_k <= j)
    tail = np.empty_like(cdf)
    tail[:, 0] = 1.0
    tail[:, 1:] = 1.0 - cdf[:, :-1]  # P(N_k >= j)
    local = np.minimum(cdf, tail)

    # Raising eta tightens every interval, so the acceptance probability falls
    # monotonically: bisect for the largest eta that still clears the target.
    candidates = np.unique(local)
    lo, hi = -1, len(candidates)  # lo = -1 means "eta below every level"
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _band_accept_probability(t, n, *_band_bounds(local, candidates[mid])) >= confidence:
            lo = mid
        else:
            hi = mid
    eta = float(candidates[lo]) if lo >= 0 else -np.inf

    lower, upper = _band_bounds(local, eta)
    return t, lower / n, upper / n, local, _band_accept_probability(t, n, lower, upper)


def pit_plot(pvals, saveto, confidence=0.95, lattice=None):
    """Create a Probability Integral Transform (PIT) plot.

    Plots the empirical CDF of the provided p-values against the expected CDF
    for a uniform distribution (the 1:1 diagonal), inside a shaded band that
    the ECDF stays wholly within with probability ``confidence`` if the
    p-values really are uniform. Any excursion outside the band rejects that
    null at significance ``1 - confidence`` -- and the band is *simultaneous*,
    so it is the whole curve being read at once, not each point separately.

    The band is built from the exact null distribution of the p-values. The
    ECDF only moves where the p-values can land, so rather than constraining
    order statistics it constrains the counts there,
    ``N_k = #{p_i <= t_k} ~ Binomial(n, F0(t_k))``, giving every point the same
    local level and taking that level as large as the target simultaneous level
    allows. The boundary follows the real ``n F0(t) (1 - F0(t))`` variance of
    the count and pinches to nothing at both corners, so it is far tighter
    where miscalibrated p-values pile up -- at n = 100 it is half the width of
    a Kolmogorov-Smirnov band at t = 0.05 -- while giving up about 12% near the
    median. Worth roughly 1.2x the power of KS against skewed p-values and 2x
    against U-shaped or over-central ones.

    Pass ``lattice=L`` when the p-values are uniform on ``{1/L, ..., 1}`` under
    H0, as permutation p-values are; ``pted_coverage_test`` supplies it
    automatically. Because the band constrains counts rather than order
    statistics, ties are exactly what it expects, and it holds its level on
    lattices as coarse as ``L = 151`` where an order-statistic band would
    reject 60% of valid p-value sets.

    Both the escape probability and the search for the local level are exact,
    so the band is deterministic and its achieved level is known rather than
    estimated -- it is annotated on the plot alongside the worst local level
    the data reached and its simultaneous p-value.

    Parameters
    ----------
        pvals (array-like): Array of p-values in [0, 1].
        saveto (str): File path where the plot will be saved. The format is
            inferred from the file extension (e.g. ".pdf", ".png").
        confidence (float): Confidence level for the band. Default is 0.95
            (95%), i.e. the ECDF escapes it 5% of the time under the null.
        lattice (Optional[int]): ``L`` if the p-values are uniform on
            ``{1/L, ..., 1}`` under H0. None treats them as continuous, which
            over-rejects if they are in fact discrete -- a warning fires if
            they look it.
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

    if lattice is None and n - len(np.unique(sorted_pvals)) >= 3:
        warn(
            f"{n - len(np.unique(sorted_pvals))} of the {n} p-values are duplicates, so they "
            f"are discrete, but no lattice was given. Pass lattice=L (permutation p-values "
            f"from B permutations are uniform on L = B + 1 points) so the band is built from "
            f"the right null; treating them as continuous over-rejects."
        )

    t, lower, upper, local, achieved = _lattice_band(n, lattice, confidence)
    counts = _lattice_counts(sorted_pvals, t)[0]
    worst = float(local[np.arange(len(t)), counts].min())
    # The simultaneous p-value is the probability of any point reaching a local
    # level this small, which is one minus the acceptance probability at that
    # threshold -- the same exact recursion that calibrated the band.
    band_pval = 1.0 - _band_accept_probability(t, n, *_band_bounds(local, worst))

    fig, ax = plt.subplots()
    ax.fill_between(
        t,
        lower,
        upper,
        step="post",
        color="grey",
        alpha=0.3,
        linewidth=0,
        label=f"{achieved:.1%} simultaneous band"
        + ("" if lattice is None else f", lattice of {int(lattice)}"),
    )
    curve_label = f"Empirical CDF (worst local level={worst:.2g}, p={band_pval:.3f})"

    ax.plot([0, 1], [0, 1], "k--", alpha=0.8, label="Expected (Uniform)")
    ax.step(
        np.concatenate([[0], sorted_pvals, [1]]),
        np.concatenate([[0], ecdf, [1]]),
        where="post",
        color="#A34F4F",
        label=curve_label,
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
