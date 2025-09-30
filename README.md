# PTED: Permutation Test using the Energy Distance

![PyPI - Version](https://img.shields.io/pypi/v/pted?style=flat-square)
[![CI](https://github.com/ConnorStoneAstro/pted/actions/workflows/ci.yml/badge.svg)](https://github.com/ConnorStoneAstro/pted/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pted)
[![codecov](https://codecov.io/gh/ConnorStoneAstro/pted/graph/badge.svg?token=5LISJ5BN17)](https://codecov.io/gh/ConnorStoneAstro/pted)
[![DOI](https://zenodo.org/badge/966938845.svg)](https://doi.org/10.5281/zenodo.15353928)

Think of it like a multi-dimensional KS-test! It is used for two sample testing
and posterior coverage tests. In some cases it is even more sensitive than the
KS-test, but likely not all cases.

![pted logo](media/pted_logo.png)

## Install

To install PTED, run the following:

```bash
pip install pted
```

## Usage

PTED (pronounced "ted") takes in `x` and `y` two datasets and determines if they
come from the same underlying distribution. For information about each argument,
just use ``help(pted.pted)`` or ``help(pted.pted_coverage_test)``.

The returned value is a p-value, an estimate of the probability of a more
extreme instance occurring. Under the null hypothesis, a p-value is drawn from a
random uniform distribution (range 0 to 1). If the null hypothesis is false, one
would expect to see very low p-values and so one can set a limit such as
`p=0.01` below which we reject the null hypothesis. In this case `1/100`th of
the time even when the null hypothesis is true, we will reject the null. 

## Example: Two-Sample-Test

```python
from pted import pted
import numpy as np

x = np.random.normal(size = (500, 10)) # (n_samples_x, n_dimensions)
y = np.random.normal(size = (400, 10)) # (n_samples_y, n_dimensions)

p_value = pted(x, y)
print(f"p-value: {p_value:.3f}") # expect uniform random from 0-1
```

## Example: Coverage Test

```python
from pted import pted_coverage_test
import numpy as np

g = np.random.normal(size = (100, 10)) # ground truth (n_simulations, n_dimensions)
s = np.random.normal(size = (200, 100, 10)) # posterior samples (n_samples, n_simulations, n_dimensions)

p_value = pted_coverage_test(g, s)
print(f"p-value: {p_value:.3f}") # expect uniform random from 0-1
```

## How it works

### Two sample test

PTED uses the energy distance of the two samples `x` and `y`, this is computed as:

$$d = \frac{2}{n_xn_y}\sum_{i,j}||x_i - y_j|| - \frac{1}{n_x^2}\sum_{i,j}||x_i - x_j|| - \frac{1}{n_y^2}\sum_{i,j}||y_i - y_j||$$

The energy distance measures distances between pairs of points. It becomes more
positive if the `x` and `y` samples tend to be further from each other than from
themselves. We demonstrate this in the figure below, where the `x` samples are
drawn from a (thick) circle, while the `y` samples are drawn from a (thick)
line.

![pted demo test](media/test_PTED.png)

In the left figure, we show the two distributions, which by eye are clearly not
drawn from the same distribution (circle and line). In the center figure we show
the individual distance measurements as histograms. To compute the energy
distance, we would sum all the elements in these histograms rather than binning
them. You can also see a schematic of the distance matrix, which represents
every pair of samples and is colour coded the same as the histograms. In the
right figure we show the energy distance as a vertical line, the grey
distribution is explained below.

The next element of PTED is the permutation test. For this we combine the `x`
and `y` samples into a single collection `z`. We then randomly shuffle (permute)
the `z` collection and break it back into `x` and `y`, now with samples randomly
swapped between the two distributions (though they are the same size as before).
If we compute the energy distance again, we will get very different results.
This time we are sure that the null hypothesis is true, `x` and `y` have been
drawn from the same distribution (`z`), and so the energy distance will be quite
low. If we do this many times and track the permuted energy distances we get a
distribution, this is the grey distribution in the right figure. Below we show
an example of what this looks like.

![pted demo permute](media/permute_PTED.png)

Here we see the `x` and `y` samples have been scrambled in the left figure. In
the center figure we see the components of the energy distance matrix are much
more consistent because `x` and `y` now follow the same distribution (a mixture
of the original circle and line distribution). In the right figure we now see
that the vertical line is situated well within the grey distribution. Indeed the
grey distribution is just a histogram of many re-runs of this procedure. We
compute a p-value by taking the fraction of the energy distances that are
greater than the current one.

### Coverage test

In the coverage test we have some number of simulations `nsim` where there is a
true value `g` and some posterior samples `s`. For each simulation separately we
use PTED to compute a p-value, essentially asking the question "was `g` drawn
from the distribution that generated `s`?". Individually, these tests are not
especially informative, however their p-values must have been drawn from
`U(0,1)` under the null-hypothesis. Thus we just need a way to combine their
statistical power. It turns out that for some `p ~ U(0,1)` value, we have that
`- 2 ln(p)` is chi2 distributed with `dof = 2`. This means that we can sum the
chi2 values for the PTED test on each simulation and compare with a chi2
distribution with `dof = 2 * nsim`. We use a density based two tailed p-value
test on this chi2 distribution meaning that if your posterior is underconfident
or overconfident, you will get a small p-value that can be used to reject the
null.

## Example: Sensitivity comparison with KS-test

There is no single universally optimal two sample test, but a widely used method
in 1D is called the Kolmogorov-Smirnov (KS)-test. The KS-test operates
fundamentally differently from PTED and can only really work in 1D. Here I do a
super basic comparison of the two methods. Draw two samples of 100 Gaussian
distributed points, thus the null hypothesis is true for these points. Then
slowly add a bias to one of the samples to shift everything by up to 0.5 sigma.
This shows the point when each method would spot the issue by tracking how the
p-value drops. If you run this test hundreds of times you will find that PTED is
more sensitive to this kind of bias than the KS-test, I plotted a representative
single example. Note it is just by chance that they start at the same p-value.

```python
from pted import pted
import numpy as np
from scipy.stats import kstest
import matplotlib.pyplot as plt

np.random.seed(10101)
x = np.random.normal(size = (100, 1))
y = np.random.normal(size = (100, 1))

bias = np.linspace(0, 0.5, 25)
pted_p = np.zeros(25)
ks_p = np.zeros(25)
for i, b in enumerate(bias):
  pted_p[i] = pted(x, y + b)
  ks_p[i] = kstest(x[:,0], y[:,0] + b).pvalue

plt.plot(bias, pted_p, linewidth=3, c="b", label="PTED")
plt.plot(bias, ks_p, linewidth=3, c="r", label="KS")
plt.legend()
plt.ylim(0,1)
plt.xlim(0,0.5)
plt.xlabel("Out of distribution bias [sigma]")
plt.ylabel("p-value")
plt.show()
```

![pted demo KS comparison](media/pted_ks.png)

## Interpreting the results

### Two sample test

This is a null hypothesis test, thus we are specifically asking the question:
"if `x` and `y` were drawn from the same distribution, how likely am I to have
observed an energy distance as extreme as this?" This is fundamentally different
from the question "how likely is it that `x` and `y` were drawn from the same
distribution?" Which is really what we would like to ask, but I am unaware of
how we would do that in a meaningful way. It is also important to note that we
are specifically looking at extreme energy distances, so we are not even really
talking about the probability densities directly. If there was a transformation
between `x` and `y` that the energy distance was insensitive to, then the two
distributions could potentially be arbitrarily different without PTED
identifying it. For example, since the default energy distance is computed with
the Euclidean distance, a single dimension in which the values are orders of
magnitude larger than the others could make it so that all other dimensions are
ignored and could be very different. For this reason we suggest using the metric
`mahalanobis` if this is a potential issue in your data.

### Coverage Test

For the coverage test we apply the PTED two sample test to each simulation
separately. We then combine the resulting p-values using chi squared where the
resulting degrees of freedom is 2 times the number of simulations. Because of
this, we can detect underconfidence or overconfidence. Specifically we detect
the average over/under confidence, it is possible to be overconfident in some
parts of the posterior and underconfident in others. Underconfidence is when the
posterior distribution is too large, it covers the ground truth by spreading too
thin and not fully exploiting the information in the prior/likelihood of the
posterior sampling process. Sometimes this is acceptable/expected, for example
when using Approximate Bayesian Computation one expects the posterior to be at
least slightly underconfident. Overconfidence is when the posterior is too
narrow and so the ground truth appears as an outlier from its perspective. This
can occur in two main ways, one is by having a too narrow posterior. This could
occur if the measurement uncertainty estimates were too low or there were
sources of error not accounted for in the model. Another way is if your
posterior is biased, you may have an appropriately broad posterior, but it is in
the wrong part of your parameter space. PTED has no way to distinguish these
modes of overconfidence, however just knowing under/over-confidence can be
powerful. As such, by default the PTED coverage test will warn users as to which
kind of failure mode they are in if the `warn_confidence` parameter is not
`None` (default is 1e-3).

## GPU Compatibility

PTED works on both CPU and GPU. All that is needed is to pass the `x` and `y` as
PyTorch Tensors on the appropriate device.

## Citation

If you use PTED in your work, please include a citation to the [zenodo
record](https://doi.org/10.5281/zenodo.15353928) and also see below for
references of the underlying method.

## Reference

I didn't invent this test, I just think its neat. Here is a paper on the subject:

```
@article{szekely2004testing,
  title={Testing for equal distributions in high dimension},
  author={Sz{\'e}kely, G{\'a}bor J and Rizzo, Maria L and others},
  journal={InterStat},
  volume={5},
  number={16.10},
  pages={1249--1272},
  year={2004},
  publisher={Citeseer}
}
```

Permutation tests are a whole class of tests, with much literature. Here are
some starting points:

```
@book{good2013permutation,
  title={Permutation tests: a practical guide to resampling methods for testing hypotheses},
  author={Good, Phillip},
  year={2013},
  publisher={Springer Science \& Business Media}
}
```

```
@book{rizzo2019statistical,
  title={Statistical computing with R},
  author={Rizzo, Maria L},
  year={2019},
  publisher={Chapman and Hall/CRC}
}
```

There is also [the wikipedia
page](https://en.wikipedia.org/wiki/Permutation_test), and the more general
[scipy
implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html),
and other [python implementations](https://github.com/qbarthelemy/PyPermut)