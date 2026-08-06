import sys, os

import numpy as np

from pted.utils import (
    two_tailed_p,
    simulation_based_calibration_histogram,
    pit_plot,
    hdp_coverage_test,
)

import pytest


def test_two_tailed_p():

    # assert np.isclose(two_tailed_p(4, 6), 1.0), "p-value at mode should be 1.0"

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


def test_pit_plot_no_matplotlib(monkeypatch):

    pvals = np.random.uniform(size=50)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    with pytest.warns(UserWarning, match="matplotlib"):
        pit_plot(pvals, "pit_no_mpl.pdf")


def test_hdp_coverage_test():
    np.random.seed(42)
    # Null is true
    ground_truth = np.random.normal(loc=0, scale=1, size=128)
    posterior_samples = np.random.normal(loc=0, scale=1, size=(1024, 128))
    pvalue = hdp_coverage_test(ground_truth, posterior_samples)
    assert 1e-3 <= pvalue <= 0.999, "p-value should be between 0 and 1"

    # Posterior is biased
    posterior_samples = np.random.normal(loc=5, scale=1, size=(1024, 128))
    pvalue = hdp_coverage_test(ground_truth, posterior_samples)
    assert pvalue < 0.01, "p-value should be small for poorly calibrated posterior samples"

    # Posterior is underconfident
    posterior_samples = np.random.normal(loc=0, scale=2, size=(1024, 128))
    pvalue = hdp_coverage_test(ground_truth, posterior_samples)
    assert pvalue < 0.01, "p-value should be small for poorly calibrated posterior samples"
    pvalue = hdp_coverage_test(ground_truth, posterior_samples, two_tailed=False)
    assert pvalue > 0.01, "p-value should not be small for underconfident and one_tailed test"
