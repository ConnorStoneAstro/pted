import numpy as np
from .pted import pted, pted_coverage_test


def test():
    np.random.seed(42)
    # example 2 sample test
    D = 300
    for _ in range(20):
        x = np.random.normal(size=(100, D))
        y = np.random.normal(size=(100, D))
        p = pted(x, y)
        assert p > 1e-4 and p < 0.9999, f"p-value {p} is not in the expected range (U(0,1))"

    x = np.random.normal(size=(100, D))
    y = np.random.uniform(size=(100, D))
    p = pted(x, y)
    assert p < 1e-4, f"p-value {p} is not in the expected range (~0)"

    x = np.random.normal(size=(100, D))
    p = pted(x, x)
    assert p > 0.9999, f"p-value {p} is not in the expected range (~1)"

    # example coverage
    n_sims = 100
    n_samples = 100
    D = 2
    g = []
    s_corr = []
    s_over = []
    s_under = []
    for _ in range(n_sims):
        loc = np.random.normal(size=(D)) * 10
        scale = np.random.uniform(size=(D)) * 10 + 1
        g.append(np.random.normal(loc=loc, scale=scale, size=(D)))
        s_corr.append(np.random.normal(loc=loc, scale=scale, size=(n_samples, D)))
        s_over.append(np.random.normal(loc=loc, scale=scale / 2, size=(n_samples, D)))
        s_under.append(np.random.normal(loc=loc, scale=scale * 2, size=(n_samples, D)))
    g = np.array(g)
    s_corr = np.stack(s_corr, axis=1)
    s_over = np.stack(s_over, axis=1)
    s_under = np.stack(s_under, axis=1)

    # correct
    p = pted_coverage_test(g, s_corr, permutations=200)
    assert p > 1e-4 and p < 0.9999, f"p-value {p} is not in the expected range (U(0,1))"
    # overconfident
    p = pted_coverage_test(g, s_over, permutations=200)
    assert p < 1e-4, f"p-value {p} is not in the expected range (~0)"
    # underconfident
    p = pted_coverage_test(g, s_under, permutations=200)
    assert p > 0.9999, f"p-value {p} is not in the expected range (~1)"

    print("Tests passed!")
