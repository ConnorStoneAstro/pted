import sys, os

import numpy as np

from pted.utils import two_tailed_p, simulation_based_calibration_histogram

import pytest


def test_two_tailed_p():

    assert np.isclose(two_tailed_p(4, 6), 1.0), "p-value at mode should be 1.0"

    assert two_tailed_p(0.01, 10) < 0.01, "p-value should be less than 0.01 for small chi2"
    assert two_tailed_p(100, 10) < 0.01, "p-value should be less than 0.01 for large chi2"
    assert two_tailed_p(10, 10) > 0.01, "p-value should be close to 0.5 for chi2 near mode"

    assert two_tailed_p(0, 10) < 0.01
    assert two_tailed_p(1e-25, 1000) < 0.01

    with pytest.raises(AssertionError):
        two_tailed_p(4, 2)


def test_sbc_histogram(monkeypatch):

    ranks = np.random.uniform(size=1000)
    simulation_based_calibration_histogram(ranks, "sbc_hist.pdf", bins=10)
    os.remove("sbc_hist.pdf")

    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    with pytest.warns():
        simulation_based_calibration_histogram(ranks, "sbc_hist.pdf", bins=10)
