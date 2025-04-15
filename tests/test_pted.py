import pted
import torch
import numpy as np


def test_pted_main():
    pted.test()


def test_pted_torch():

    # Set the random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # example 2 sample test
    D = 300
    for _ in range(20):
        x = torch.randn(100, D)
        y = torch.randn(100, D)
        p = pted.pted(x, y)
        assert p > 1e-4 and p < 0.9999, f"p-value {p} is not in the expected range (U(0,1))"

    x = torch.randn(100, D)
    y = torch.rand(100, D)
    p = pted.pted(x, y)
    assert p < 1e-4, f"p-value {p} is not in the expected range (~0)"

    x = torch.randn(100, D)
    t, p = pted.pted(x, x, return_all=True)
    p = np.mean(np.array(p) > t)
    assert p > 0.9999, f"p-value {p} is not in the expected range (~1)"
