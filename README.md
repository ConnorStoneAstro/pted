# PTED: Permutation Test using the Energy Distance

![PyPI - Version](https://img.shields.io/pypi/v/pted?style=flat-square)
[![CI](https://github.com/Ciela-Institute/pted/actions/workflows/ci.yml/badge.svg)](https://github.com/Ciela-Institute/pted/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pted)
[![codecov](https://codecov.io/gh/Ciela-Institute/pted/graph/badge.svg?token=wbkUiRkYtg)](https://codecov.io/gh/Ciela-Institute/pted)

Think of it like a multi-dimensional KS-test! It is used for two sample testing and posterior coverage tests.

## Install

To install PTED, run the following:

```bash
pip install pted
```

## Usage

PTED (pronounced "ted") takes in `x` and `y` two datasets and determines if they
come from the same underlying distribution. 

## Example: Two-Sample-Test

```python
from pted import pted
import numpy as np

p = np.random.normal(size = (500, 10)) # (n_samples_x, n_dimensions)
q = np.random.normal(size = (400, 10)) # (n_samples_y, n_dimensions)

p_value = pted(p, q, permutations = 1000)
print(f"p-value: {p_value:.3f}") # expect uniform random from 0-1
```

## Example: Coverage Test

```python
from pted import pted_coverage_test
import numpy as np

g = np.random.normal(size = (100, 10)) # ground truth (n_simulations, n_dimensions)
s = np.random.normal(size = (200, 100, 10)) # posterior samples (n_samples, n_simulations, n_dimensions)

p_value = pted_coverage_test(g, s, permutations = 100)
print(f"p-value: {p_value:.3f}") # expect uniform random from 0-1
```

## GPU Compatibility

PTED works on both CPU and GPU. All that is needed is to pass the `x` and `y` as
PyTorch Tensors on the appropriate device.
