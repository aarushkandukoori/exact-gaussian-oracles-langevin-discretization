'''Unit tests on SYNTHETIC Gaussian targets. No benchmark claims.'''
import numpy as np
import pytest

from src import exact_gaussian as eg
from src import samplers


def test_ula_stability_threshold():
    lams = np.array([1.0, 4.0])
    assert eg.ula_is_stable(lams, 0.49)
    assert not eg.ula_is_stable(lams, 0.5)
    with pytest.raises(ValueError):
        eg.ula_stationary_variance(lams, 0.5)


def test_w2_matches_leading_order_for_small_h():
    lams = np.geomspace(1.0, 5.0, 8)
    h = 1e-3
    ratio = eg.ula_w2_squared(lams, h) / eg.w2_leading_order(lams, h)
    assert abs(ratio - 1.0) < 0.05


def test_fixed_point_matches_ula_closed_form():
    lam, h = 2.0, 0.1
    A = np.array([[1 - h * lam]])
    Q = np.array([[2 * h]])
    S = eg.stationary_second_moment([A], [Q], [1.0])
    assert S[0, 0] == pytest.approx(eg.ula_stationary_variance(np.array([lam]), h)[0])


def test_exact_vs_simulated_ula():
    lams = np.array([1.0, 2.0, 3.0, 4.0])
    h = 0.1
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((2000, 4)) / np.sqrt(lams)
    step = lambda x: samplers.ula_step(x, h, lambda z: lams * z, rng)
    sim = samplers.run_chains(step, x0, n_steps=300, n_burn=100)
    exact = eg.ula_stationary_variance(lams, h)
    assert np.mean(sim / exact) == pytest.approx(1.0, abs=0.08)
