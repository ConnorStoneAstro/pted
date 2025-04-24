# PTED: Permutation Test using the Energy Distance

![PyPI - Version](https://img.shields.io/pypi/v/pted?style=flat-square)
[![CI](https://github.com/ConnorStoneAstro/pted/actions/workflows/ci.yml/badge.svg)](https://github.com/ConnorStoneAstro/pted/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pted)
[![codecov](https://codecov.io/gh/ConnorStoneAstro/pted/graph/badge.svg?token=5LISJ5BN17)](https://codecov.io/gh/ConnorStoneAstro/pted)

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

p = np.random.normal(size = (500, 10)) # (n_samples_x, n_dimensions)
q = np.random.normal(size = (400, 10)) # (n_samples_y, n_dimensions)

p_value = pted(p, q)
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

## GPU Compatibility

PTED works on both CPU and GPU. All that is needed is to pass the `x` and `y` as
PyTorch Tensors on the appropriate device.

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

Permutation tests are a whole class of tests, with much literature. Here are some starting points:

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