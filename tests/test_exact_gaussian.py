'''Unit tests on SYNTHETIC Gaussian targets. They check algebra, not research claims.'''
import numpy as np
import pytest

from src import exact_gaussian as eg
from src import samplers


# ---------------------------------------------------------------- ULA (known baseline)

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


# ---------------------------------------------------------------- RMP: second moments

def _random_spd(d, lam_min, lam_max, seed):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    lams = np.geomspace(lam_min, lam_max, d)
    return (U / lams) @ U.T  # Sigma = U diag(1/lam) U^T


def test_rmp_alpha_averages_match_quadrature():
    nodes, weights = eg.alpha_quadrature(6)
    for lam, h in [(1.0, 0.3), (3.0, 0.2), (10.0, 0.05)]:
        c, s2 = eg.rmp_coefficients(lam, h, nodes)
        e_c2, e_s2 = eg.rmp_alpha_averages(h * lam, h)
        assert np.sum(weights * c ** 2) == pytest.approx(float(e_c2), rel=1e-13)
        assert np.sum(weights * s2) == pytest.approx(float(e_s2), rel=1e-13)


def test_rmp_closed_form_matches_matrix_fixed_point_nondiagonal():
    Sigma = _random_spd(4, 0.5, 5.0, seed=3)
    L = 1.0 / np.linalg.eigvalsh(Sigma).min()
    for hL in (0.1, 0.5, 0.9, 1.4):
        h = hL / L
        closed = eg.rmp_stationary_covariance(Sigma, h)
        fixed = eg.rmp_matrix_fixed_point(Sigma, h)
        iterated = eg.rmp_iterate_covariance(Sigma, h, n_steps=4000)
        assert np.allclose(closed, fixed, rtol=1e-10, atol=1e-12)
        assert np.allclose(closed, iterated, rtol=1e-8, atol=1e-10)


def test_rmp_plugging_in_mean_alpha_is_not_the_average():
    a = np.array([0.1, 0.5, 0.9])
    p = 2 - 2 * a + a ** 2
    assert np.allclose(eg.rmp_rho(a), p / (p - a ** 3 / 3))
    assert np.allclose(eg.rmp_rho_fixed_alpha(a, 0.5), p / (p - a ** 3 / 4))
    assert np.all(eg.rmp_rho(a) > eg.rmp_rho_fixed_alpha(a, 0.5))
    e_c2, _ = eg.rmp_alpha_averages(a, 1.0)
    c_half = 1 - a + 0.5 * a ** 2
    assert np.allclose(e_c2 - c_half ** 2, a ** 4 / 12)  # = a^4 Var(alpha)


def test_rmp_stability_threshold():
    assert eg.A_STAR ** 3 - 3 * eg.A_STAR ** 2 + 6 * eg.A_STAR - 6 == pytest.approx(0.0, abs=1e-12)
    assert eg.A_STAR == pytest.approx(1.5960716, abs=1e-6)
    below, _ = eg.rmp_alpha_averages(eg.A_STAR - 1e-6, 1.0)
    above, _ = eg.rmp_alpha_averages(eg.A_STAR + 1e-6, 1.0)
    assert below < 1.0 < above
    lams = np.array([1.0, 2.0])
    assert eg.rmp_is_stable(lams, 0.79)
    with pytest.raises(ValueError):
        eg.rmp_stationary_variance(lams, 0.80)


def test_rmp_leading_orders():
    for a in (1e-2, 3e-2):
        assert (eg.rmp_rho(a) - 1) / (a ** 3 / 6) == pytest.approx(1.0, rel=0.05)
    lams = np.array([1.0, 2.0, 5.0])
    h = 2e-3
    R = eg.rmp_excess(lams, h)
    lead = eg.rmp_excess_leading_order(lams, h)
    assert np.allclose(R / lead, 1.0, rtol=0.05)
    assert R[0, 0] == pytest.approx((h * lams[0]) ** 3 / 12, rel=0.05)


def test_rmp_finite_k_matches_iterated_recursion():
    lams = np.array([0.5, 2.0, 4.0])
    h = 0.15
    Sigma = np.diag(1.0 / lams)
    for k in (1, 5, 40):
        it = eg.rmp_iterate_covariance(Sigma, h, n_steps=k)
        assert np.allclose(np.diag(it), eg.rmp_finite_k_variance(lams, h, k), rtol=1e-12)


def test_exact_vs_simulated_rmp_second_moments_nondiagonal():
    Sigma = _random_spd(3, 1.0, 3.0, seed=5)
    P = np.linalg.inv(Sigma)
    h = 0.25
    rng = np.random.default_rng(2)
    x0 = rng.multivariate_normal(np.zeros(3), Sigma, size=4000)
    x = x0
    acc = np.zeros((3, 3))
    n_keep = 0
    for k in range(1200):
        x = samplers.rmp_step(x, h, lambda z: z @ P, rng)  # P symmetric
        if k >= 200:
            acc += x.T @ x / x.shape[0]
            n_keep += 1
    sim = acc / n_keep
    exact = eg.rmp_stationary_covariance(Sigma, h)
    assert np.allclose(sim, exact, atol=0.01)


# ---------------------------------------------------------------- RMP: fourth moments

def test_rmp_excess_matches_covariance_formula():
    lams = np.array([1.0, 2.0, 3.0])
    h = 0.3
    v = eg.rmp_stationary_variance(lams, h)
    nodes, weights = eg.alpha_quadrature(6)
    c, s2 = eg.rmp_coefficients(lams[:, None], h, nodes[None, :])   # shape (d, nodes)
    G = c ** 2 * v[:, None] + s2                                      # G_i(alpha)
    EG = G @ weights
    covG = (G * weights) @ G.T - np.outer(EG, EG)
    ECC = (c ** 2 * weights) @ (c ** 2).T
    R_alt = covG / (1.0 - ECC) / np.outer(v, v)
    assert np.allclose(eg.rmp_excess(lams, h), R_alt, rtol=1e-10, atol=1e-14)
    assert np.all(R_alt > 0)


def test_exact_vs_simulated_rmp_fourth_moments():
    lams = np.array([1.0, 2.0, 3.0])
    h = 0.3
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((4000, 3)) / np.sqrt(lams)
    step = lambda x: samplers.rmp_step(x, h, lambda z: lams * z, rng)
    m2, m4, se2, se4 = samplers.run_chains_moments(step, x0, n_steps=3000, n_burn=200, thin=10)
    v = eg.rmp_stationary_variance(lams, h)
    exact4 = eg.rmp_fourth_moments(lams, h)
    gauss4 = np.outer(v, v) * (1 + 2 * np.eye(3))   # Gaussian with the same covariance
    assert np.all(np.abs(m2 - v) < 4 * se2)
    assert np.all(np.abs(m4 - exact4) < 4 * se4)
    # the shared midpoint makes the law non-Gaussian: the Gaussian value is rejected
    for i, j in [(1, 2), (2, 2)]:
        assert abs(m4[i, j] - gauss4[i, j]) > 4 * se4[i, j]


# ---------------------------------------------------------------- KL / W2 identities used

def test_kl_pythagorean_split_and_gelbrich_bound_1d():
    # mu = scale mixture of centered Gaussians, pi = N(0, s_pi): checks the split
    # KL(mu||pi) = KL(mu||gamma) + KL(gamma||pi), gamma = N(0, E[V]), and W2(mu,pi) >= W2(gamma,pi)
    from scipy import stats
    V = np.array([0.7, 1.0, 1.6])
    w = np.array([0.3, 0.4, 0.3])
    s_pi = 1.2
    vbar = float(w @ V)
    x = np.linspace(-25, 25, 400001)
    dx = x[1] - x[0]
    mu = sum(wk * stats.norm.pdf(x, scale=np.sqrt(Vk)) for wk, Vk in zip(w, V))
    gam = stats.norm.pdf(x, scale=np.sqrt(vbar))
    pi = stats.norm.pdf(x, scale=np.sqrt(s_pi))
    kl = lambda p, q: float(np.sum(p * np.log(p / q)) * dx)
    assert kl(mu, pi) == pytest.approx(kl(mu, gam) + kl(gam, pi), rel=1e-6)
    assert kl(gam, pi) == pytest.approx(eg.gaussian_kl_diag(np.array([vbar]), np.array([s_pi])), rel=1e-6)
    u = (np.arange(20000) + 0.5) / 20000
    F_mu = np.cumsum(mu) * dx
    q_mu = np.interp(u, F_mu, x)
    q_pi = stats.norm.ppf(u, scale=np.sqrt(s_pi))
    w2_mu = np.sqrt(np.mean((q_mu - q_pi) ** 2))
    w2_gauss = abs(np.sqrt(vbar) - np.sqrt(s_pi))
    assert w2_mu >= w2_gauss


def test_rmp_bias_lower_bounds_leading_order():
    lams = np.array([1.0, 2.0, 4.0])
    h = 5e-3
    kl_cov = eg.rmp_kl_covariance_part(lams, h)
    w2_lb = eg.rmp_w2_gelbrich_lower_bound(lams, h)
    assert kl_cov / (h ** 6 / 144 * np.sum(lams ** 6)) == pytest.approx(1.0, rel=0.05)
    assert w2_lb / (h ** 3 / 12 * np.sqrt(np.sum(lams ** 5))) == pytest.approx(1.0, rel=0.05)
    assert eg.rmp_kl_mixture_second_order(lams, h) > 0
