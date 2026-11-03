'''Exact stationary quantities for linear Gaussian samplers (SYNTHETIC Gaussian targets).

Reference implementation of a PROPOSAL. No benchmark numbers are produced or
claimed here. Target: pi = N(0, diag(1/lam_i)).

ULA has a closed form. Randomized-midpoint schemes have u-dependent coefficients,
so their stationary law is generally a non-Gaussian mixture; only stationary
second moments follow from a fixed-point equation. The midpoint coefficients
are NOT derived here (planned, see `randomized_midpoint_coefficients`).
'''
from __future__ import annotations

import numpy as np


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

    Valid because both laws are centered Gaussians with commuting (diagonal) covariances.
    '''
    v = ula_stationary_variance(lams, h)
    sigma = 1.0 / np.sqrt(np.asarray(lams, dtype=float))
    return float(np.sum((np.sqrt(v) - sigma) ** 2))


def ula_kl_to_target(lams: np.ndarray, h: float) -> float:
    '''KL(stationary || target) for diagonal Gaussians.'''
    v = ula_stationary_variance(lams, h)
    ratio = v * np.asarray(lams, dtype=float)
    return float(0.5 * np.sum(ratio - 1.0 - np.log(ratio)))


def w2_leading_order(lams: np.ndarray, h: float) -> float:
    '''Small-h approximation W2^2 ~ (h^2 / 16) * sum(lam_i).'''
    return float(h * h / 16.0 * np.sum(lams))


def covariance_error(v: np.ndarray, sigma2: np.ndarray) -> float:
    '''Relative Frobenius error between two diagonal covariances (given as vectors).'''
    return float(np.linalg.norm(v - sigma2) / np.linalg.norm(sigma2))


def stationary_second_moment(As, Qs, weights) -> np.ndarray:
    '''Solve S = E[A S A^T] + E[Q] for a finite mixture of linear Gaussian maps.

    As, Qs: lists of (n, n) arrays (map and noise covariance per mixture atom);
    weights: probabilities summing to 1 (e.g. a quadrature of u ~ U[0,1]).
    Gives only the stationary second moment, not the (generally non-Gaussian) law.
    '''
    n = As[0].shape[0]
    M = sum(w * np.kron(A, A) for w, A in zip(weights, As))
    rhs = sum(w * Q for w, Q in zip(weights, Qs))
    vec = np.linalg.solve(np.eye(n * n) - M, rhs.reshape(-1))
    return vec.reshape(n, n)


def randomized_midpoint_coefficients(lam: float, h: float, u: float):
    '''PLANNED, NOT YET DERIVED: (A_u, Q_u) for the underdamped randomized midpoint step.

    The proposal plans to derive these symbolically and cross-check them against
    simulation before use as a reference.
    '''
    raise NotImplementedError('midpoint coefficients are planned and not yet derived')
