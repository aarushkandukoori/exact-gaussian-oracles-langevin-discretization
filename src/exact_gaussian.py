'''Exact moments of linear samplers on SYNTHETIC Gaussian targets.

Reference implementation of a PROPOSAL. No benchmark numbers are produced or
claimed here. Target: pi = N(0, Sigma). In the eigenbasis of Sigma the
precisions are lam_i = 1 / sigma_i^2, and a_i = h * lam_i.

ULA (known baseline): the stationary law is Gaussian with variance
sigma^2 / (1 - a/2) per eigen-direction, so W2 and KL are exact.

Overdamped randomized midpoint (RMP): one alpha ~ Unif[0, 1] per step, shared
by all coordinates, and one Brownian path for the midpoint and the full step:
    y  = x - alpha h grad f(x) + sqrt(2) (B_{alpha h} - B_0)
    x' = x - h grad f(y)       + sqrt(2) (B_h - B_0)
Per coordinate, x' = c(alpha) x + eta with c(alpha) = 1 - a + alpha a^2 and
Var(eta | alpha) = s2(alpha) = 2 h (1 - alpha a (2 - a)). Averaging the
second-moment recursion over alpha gives the stationary variance
sigma^2 rho(a), rho(a) = p / (p - a^3/3), p = 2 - 2a + a^2.

Because alpha is shared, the RMP stationary law is a Gaussian scale mixture,
not a Gaussian. Gaussian W2/KL formulas evaluated at its covariance are
therefore LOWER BOUNDS for RMP (Gelbrich bound; KL Pythagorean identity), and
the exact fourth moments below measure the remaining non-Gaussian part.
Every expectation over alpha is a polynomial moment E[alpha^n] = 1 / (n + 1).
'''
from __future__ import annotations

import numpy as np

# Unique real root of a^3 - 3a^2 + 6a - 6 = 0: RMP second moments converge iff h*L < A_STAR.
A_STAR = 1.0 + np.cbrt(1.0 + np.sqrt(2.0)) - np.cbrt(np.sqrt(2.0) - 1.0)


# ----------------------------------------------------------------------------
# ULA: known baseline / sanity check
# ----------------------------------------------------------------------------

def ula_is_stable(lams: np.ndarray, h: float) -> bool:
    '''ULA on a Gaussian is stable iff h < 2/L.'''
    return bool(h * np.max(lams) < 2.0)


def ula_stationary_variance(lams: np.ndarray, h: float) -> np.ndarray:
    '''Per-coordinate stationary variance v_i = 2h / (a_i (2 - a_i)), a_i = h lam_i.'''
    lams = np.asarray(lams, dtype=float)
    if not ula_is_stable(lams, h):
        raise ValueError('ULA unstable: need h < 2/L')
    a = h * lams
    return 2.0 * h / (a * (2.0 - a))


def ula_w2_squared(lams: np.ndarray, h: float) -> float:
    '''Squared 2-Wasserstein distance between the ULA stationary law and the target.

    Exact because both laws are centered Gaussians with commuting covariances.
    '''
    v = ula_stationary_variance(lams, h)
    sigma = 1.0 / np.sqrt(np.asarray(lams, dtype=float))
    return float(np.sum((np.sqrt(v) - sigma) ** 2))


def ula_kl_to_target(lams: np.ndarray, h: float) -> float:
    '''KL(stationary || target) for diagonal Gaussians (exact for ULA).'''
    v = ula_stationary_variance(lams, h)
    return gaussian_kl_diag(v, 1.0 / np.asarray(lams, dtype=float))


def w2_leading_order(lams: np.ndarray, h: float) -> float:
    '''ULA small-h approximation W2^2 ~ (h^2 / 16) * sum(lam_i).'''
    return float(h * h / 16.0 * np.sum(lams))


# ----------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------

def gaussian_kl_diag(v: np.ndarray, sigma2: np.ndarray) -> float:
    '''KL(N(0, diag v) || N(0, diag sigma2)) = 0.5 * sum(r - 1 - log r), r = v / sigma2.'''
    r = np.asarray(v, dtype=float) / np.asarray(sigma2, dtype=float)
    return float(0.5 * np.sum(r - 1.0 - np.log(r)))


def covariance_error(v: np.ndarray, sigma2: np.ndarray) -> float:
    '''Relative Frobenius error between two diagonal covariances (given as vectors).'''
    return float(np.linalg.norm(v - sigma2) / np.linalg.norm(sigma2))


def stationary_second_moment(As, Qs, weights) -> np.ndarray:
    '''Solve S = E[A S A^T] + E[Q] for a finite mixture of linear Gaussian maps.

    As, Qs: lists of (n, n) arrays (map and noise covariance per mixture atom);
    weights: probabilities summing to 1 (e.g. Gauss-Legendre nodes for
    alpha ~ U[0, 1]). The average is taken INSIDE the Kronecker product,
    vec(S) = (I - E[A kron A])^{-1} vec(E[Q]). Gives only the stationary second
    moment, not the (generally non-Gaussian) law.
    '''
    n = As[0].shape[0]
    M = sum(w * np.kron(A, A) for w, A in zip(weights, As))
    rhs = sum(w * Q for w, Q in zip(weights, Qs))
    vec = np.linalg.solve(np.eye(n * n) - M, rhs.reshape(-1))
    return vec.reshape(n, n)


def alpha_quadrature(n_nodes: int = 3):
    '''Gauss-Legendre nodes and weights for alpha ~ U[0, 1] (exact for degree <= 2n - 1).'''
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    return 0.5 * (x + 1.0), 0.5 * w


def _alpha_mean(coefs) -> float:
    '''E[sum_n coefs[n] alpha^n] for alpha ~ U[0, 1].'''
    return float(sum(c / (n + 1.0) for n, c in enumerate(coefs)))


def _hilbert(m: int, n: int) -> np.ndarray:
    '''H[r, s] = E[alpha^(r + s)] = 1 / (r + s + 1).'''
    r = np.arange(m)[:, None]
    s = np.arange(n)[None, :]
    return 1.0 / (r + s + 1.0)


# ----------------------------------------------------------------------------
# Overdamped randomized midpoint (RMP): exact second moments
# ----------------------------------------------------------------------------

def rmp_coefficients(lam: float, h: float, alpha: float):
    '''(c(alpha), s2(alpha)) for one coordinate: x' = c x + eta, Var(eta | alpha) = s2.'''
    a = h * lam
    c = 1.0 - a + alpha * a * a
    s2 = 2.0 * h * (1.0 - alpha * a * (2.0 - a))
    return c, s2


def rmp_alpha_averages(a, h: float):
    '''(E[c(alpha)^2], E[s2(alpha)]) with the average taken over alpha ~ U[0, 1].

    E[c^2] = (1 - a)^2 + (1 - a) a^2 + a^4 / 3 = 1 - 2a + 2a^2 - a^3 + a^4 / 3,
    E[s2]  = h (2 - 2a + a^2).
    '''
    a = np.asarray(a, dtype=float)
    e_c2 = (1.0 - a) ** 2 + (1.0 - a) * a ** 2 + a ** 4 / 3.0
    e_s2 = h * (2.0 - 2.0 * a + a ** 2)
    return e_c2, e_s2


def rmp_is_stable(lams: np.ndarray, h: float) -> bool:
    '''Second moments converge iff E[c^2] < 1 in every direction, i.e. h L < A_STAR.'''
    return bool(h * np.max(lams) < A_STAR)


def rmp_rho(a):
    '''Exact variance inflation rho(a) = p / (p - a^3/3), p = 2 - 2a + a^2 (= 1 + a^3/6 + ...).'''
    a = np.asarray(a, dtype=float)
    p = 2.0 - 2.0 * a + a ** 2
    return p / (p - a ** 3 / 3.0)


def rmp_rho_fixed_alpha(a, alpha: float = 0.5):
    '''Variance inflation if alpha is FIXED (a deterministic midpoint rule, Gaussian law).

    For alpha = 1/2 this equals p / (p - a^3/4) = 1 + a^3/8 + ..., which differs from
    rmp_rho: plugging alpha = E[alpha] into the recursion is not the same as averaging.
    '''
    a = np.asarray(a, dtype=float)
    c = 1.0 - a + alpha * a * a
    s2_over_h = 2.0 * (1.0 - alpha * a * (2.0 - a))
    return s2_over_h * a / (1.0 - c * c)  # (s2 / (1 - c^2)) / sigma^2 with sigma^2 = h / a


def rmp_stationary_variance(lams: np.ndarray, h: float) -> np.ndarray:
    '''Per-coordinate stationary variance sigma_i^2 rho(h lam_i).'''
    lams = np.asarray(lams, dtype=float)
    if not rmp_is_stable(lams, h):
        raise ValueError('RMP second moments diverge: need h L < A_STAR (about 1.596)')
    return rmp_rho(h * lams) / lams


def rmp_stationary_covariance(Sigma: np.ndarray, h: float) -> np.ndarray:
    '''Exact stationary covariance S_h = Sigma rho(h Sigma^{-1}) for a full SPD Sigma.'''
    Sigma = np.asarray(Sigma, dtype=float)
    sig2, U = np.linalg.eigh(Sigma)
    v = rmp_stationary_variance(1.0 / sig2, h)
    return (U * v) @ U.T


def rmp_finite_k_variance(lams: np.ndarray, h: float, k: int) -> np.ndarray:
    '''Exact variance after k steps from x_0 ~ pi: sigma^2 [rho + (1 - rho) E[c^2]^k].'''
    lams = np.asarray(lams, dtype=float)
    a = h * lams
    e_c2, _ = rmp_alpha_averages(a, h)
    rho = rmp_rho(a)
    return (rho + (1.0 - rho) * e_c2 ** k) / lams


def rmp_matrix_coefficients(Sigma: np.ndarray, h: float):
    '''(A0, A1, Q0, Q1) with C(alpha) = A0 + alpha A1 and Q(alpha) = Q0 + alpha Q1.

    For P = Sigma^{-1}: x' = (I - hP + alpha h^2 P^2) x + sqrt(2) (W - h P W'), and
    Cov(sqrt(2)(W - h P W') | alpha) = 2h (I - 2 alpha h P + alpha h^2 P^2).
    Used for the matrix fixed-point check in the original (non-diagonal) basis.
    '''
    P = np.linalg.inv(np.asarray(Sigma, dtype=float))
    eye = np.eye(P.shape[0])
    P2 = P @ P
    return eye - h * P, h * h * P2, 2.0 * h * eye, 2.0 * h * (h * h * P2 - 2.0 * h * P)


def rmp_matrix_fixed_point(Sigma: np.ndarray, h: float, n_nodes: int = 3) -> np.ndarray:
    '''Stationary covariance from vec(S) = (I - E[C kron C])^{-1} vec(E[Q]) (no closed form used).'''
    A0, A1, Q0, Q1 = rmp_matrix_coefficients(Sigma, h)
    nodes, weights = alpha_quadrature(n_nodes)
    As = [A0 + t * A1 for t in nodes]
    Qs = [Q0 + t * Q1 for t in nodes]
    return stationary_second_moment(As, Qs, weights)


def rmp_iterate_covariance(Sigma: np.ndarray, h: float, n_steps: int, S0=None) -> np.ndarray:
    '''Iterate S <- E[C S C^T] + E[Q] directly (exact alpha average via quadrature).'''
    A0, A1, Q0, Q1 = rmp_matrix_coefficients(Sigma, h)
    nodes, weights = alpha_quadrature(3)
    S = np.array(Sigma if S0 is None else S0, dtype=float)
    EQ = Q0 + 0.5 * Q1
    for _ in range(n_steps):
        S = sum(w * (A0 + t * A1) @ S @ (A0 + t * A1).T for t, w in zip(nodes, weights)) + EQ
    return S


# ----------------------------------------------------------------------------
# RMP: exact fourth moments (shared alpha makes the law a Gaussian scale mixture)
# ----------------------------------------------------------------------------

def _rmp_poly_coefs(lams: np.ndarray, h: float):
    '''Polynomial coefficients in alpha of C_i = c_i(alpha)^2 (degree 2) and s2_i (degree 1).'''
    a = h * np.asarray(lams, dtype=float)
    b, e = 1.0 - a, a * a                     # c_i = b + e alpha
    Cc = np.stack([b * b, 2.0 * b * e, e * e], axis=1)
    Sc = np.stack([np.full_like(a, 2.0 * h), -2.0 * h * a * (2.0 - a)], axis=1)
    return Cc, Sc


def rmp_conditional_variance_moments(lams: np.ndarray, h: float):
    '''Return (v, M) with v_i = E[V_i] and M_ij = E[V_i V_j] at stationarity.

    V_i is the conditional variance of coordinate i given the alpha sequence;
    V_i' = C_i V_i + s2_i with C_i = c_i(alpha)^2 and one alpha shared by all i. Then
    E[V_i V_j] = (E[C_i s2_j] v_i + E[s2_i C_j] v_j + E[s2_i s2_j]) / (1 - E[C_i C_j]).
    Requires h L < 1, where 0 < c_i(alpha) < 1 for all alpha (all moments finite).
    '''
    lams = np.asarray(lams, dtype=float)
    if h * np.max(lams) >= 1.0:
        raise ValueError('fourth-moment formulas are stated for h L < 1')
    v = rmp_stationary_variance(lams, h)
    Cc, Sc = _rmp_poly_coefs(lams, h)
    E_CC = Cc @ _hilbert(3, 3) @ Cc.T
    E_CS = Cc @ _hilbert(3, 2) @ Sc.T          # E_CS[i, j] = E[C_i s2_j]
    E_SS = Sc @ _hilbert(2, 2) @ Sc.T
    num = E_CS * v[:, None] + E_CS.T * v[None, :] + E_SS
    M = num / (1.0 - E_CC)
    return v, M


def rmp_fourth_moments(lams: np.ndarray, h: float) -> np.ndarray:
    '''Exact E[x_i^2 x_j^2] under the RMP stationary law (eigenbasis).

    Off-diagonal: E[V_i V_j]; diagonal: E[x_i^4] = 3 E[V_i^2]. A Gaussian with the same
    covariance would give v_i v_j and 3 v_i^2 instead.
    '''
    _, M = rmp_conditional_variance_moments(lams, h)
    out = M.copy()
    out[np.diag_indices_from(out)] *= 3.0
    return out


def rmp_excess(lams: np.ndarray, h: float) -> np.ndarray:
    '''R_ij = Cov(V_i, V_j) / (v_i v_j): zero for a Gaussian law, positive for RMP.'''
    v, M = rmp_conditional_variance_moments(lams, h)
    return M / np.outer(v, v) - 1.0


def rmp_excess_leading_order(lams: np.ndarray, h: float) -> np.ndarray:
    '''Leading order of R_ij: (a_i a_j)^2 / (6 (a_i + a_j)); R_ii ~ a_i^3 / 12.'''
    a = h * np.asarray(lams, dtype=float)
    return np.outer(a, a) ** 2 / (6.0 * (a[:, None] + a[None, :]))


# ----------------------------------------------------------------------------
# RMP: what the exact moments give for KL and W2
# ----------------------------------------------------------------------------

def rmp_kl_covariance_part(lams: np.ndarray, h: float) -> float:
    '''KL(N(0, S_h) || pi) = 0.5 sum(rho - 1 - log rho): an exact LOWER bound on KL(mu_h || pi).

    Exact split: KL(mu_h || pi) = KL(N(0, S_h) || pi) + KL(mu_h || N(0, S_h)).
    '''
    lams = np.asarray(lams, dtype=float)
    return gaussian_kl_diag(rmp_stationary_variance(lams, h), 1.0 / lams)


def rmp_w2_gelbrich_lower_bound(lams: np.ndarray, h: float) -> float:
    '''Gelbrich bound: W2(mu_h, pi) >= W2(N(0, S_h), pi) = sqrt(sum sigma^2 (sqrt(rho) - 1)^2).'''
    lams = np.asarray(lams, dtype=float)
    v = rmp_stationary_variance(lams, h)
    return float(np.sqrt(np.sum((np.sqrt(v) - 1.0 / np.sqrt(lams)) ** 2)))


def rmp_kl_mixture_second_order(lams: np.ndarray, h: float) -> float:
    '''SECOND-ORDER APPROXIMATION (not exact) of KL(mu_h || N(0, S_h)).

    (1/16) (2 sum_i R_ii^2 + sum_ij R_ij^2), from expanding
    1 + chi^2 = E prod_i (1 - u_i u_i')^{-1/2} with V_i = v_i (1 + u_i). Valid when
    sum_ij R_ij^2 is small; checking it is part of the planned evaluation.
    '''
    R = rmp_excess(lams, h)
    return float((2.0 * np.sum(np.diag(R) ** 2) + np.sum(R ** 2)) / 16.0)


# ----------------------------------------------------------------------------
# Planned extension
# ----------------------------------------------------------------------------

def underdamped_rmp_coefficients(lam: float, h: float, alpha: float, gamma: float = 2.0):
    '''PLANNED, NOT YET DERIVED: (A_alpha, Q_alpha) for the underdamped randomized midpoint step.

    Once derived, stationary_second_moment([A_alpha], [Q_alpha], weights) over alpha
    nodes gives the exact 2x2 stationary second moment per eigen-direction.
    '''
    raise NotImplementedError('underdamped midpoint coefficients are planned and not yet derived')
