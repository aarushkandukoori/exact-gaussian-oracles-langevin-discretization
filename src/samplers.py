'''Simulation implementations of samplers for Monte Carlo cross-checks.

Reference implementation of a PROPOSAL. Targets are SYNTHETIC analytic potentials.
Chains are vectorized: x has shape (n_chains, d).
'''
from __future__ import annotations

from typing import Callable

import numpy as np

Grad = Callable[[np.ndarray], np.ndarray]
Pot = Callable[[np.ndarray], np.ndarray]


def ula_step(x: np.ndarray, h: float, grad: Grad, rng: np.random.Generator) -> np.ndarray:
    '''One unadjusted Langevin step.'''
    return x - h * grad(x) + np.sqrt(2.0 * h) * rng.standard_normal(x.shape)


def mala_step(x: np.ndarray, h: float, f: Pot, grad: Grad, rng: np.random.Generator) -> np.ndarray:
    '''One Metropolis-adjusted Langevin step (simulation-only reference).'''
    g = grad(x)
    y = x - h * g + np.sqrt(2.0 * h) * rng.standard_normal(x.shape)
    gy = grad(y)
    log_fwd = -np.sum((y - x + h * g) ** 2, axis=1) / (4.0 * h)
    log_rev = -np.sum((x - y + h * gy) ** 2, axis=1) / (4.0 * h)
    log_alpha = f(x) - f(y) + log_rev - log_fwd
    accept = np.log(rng.uniform(size=x.shape[0])) < log_alpha
    return np.where(accept[:, None], y, x)


def rmp_step(x: np.ndarray, h: float, grad: Grad, rng: np.random.Generator) -> np.ndarray:
    '''One overdamped randomized midpoint step.

    One alpha ~ U[0, 1] per chain and step, SHARED by all coordinates; the midpoint
    and the full step use the same Brownian path (W' = B_{alpha h}, W = B_h):
        y  = x - alpha h grad(x) + sqrt(2) W'
        x' = x - h grad(y)       + sqrt(2) W
    '''
    alpha = rng.uniform(size=(x.shape[0], 1))
    w_mid = np.sqrt(alpha * h) * rng.standard_normal(x.shape)
    w_full = w_mid + np.sqrt((1.0 - alpha) * h) * rng.standard_normal(x.shape)
    y = x - alpha * h * grad(x) + np.sqrt(2.0) * w_mid
    return x - h * grad(y) + np.sqrt(2.0) * w_full


def underdamped_rmp_step(x, v, h, grad, rng):
    '''STUB: randomized midpoint step on underdamped Langevin (Shen and Lee). Planned.'''
    raise NotImplementedError('underdamped randomized midpoint is planned, not implemented')


def run_chains(step: Callable[[np.ndarray], np.ndarray], x0: np.ndarray,
               n_steps: int, n_burn: int) -> np.ndarray:
    '''Run chains and return the per-coordinate empirical second moment after burn-in.

    Assumes a centered target, so the second moment estimates the variance.
    '''
    if n_burn >= n_steps:
        raise ValueError('n_burn must be < n_steps')
    x = x0.copy()
    acc = np.zeros(x.shape[1])
    for k in range(n_steps):
        x = step(x)
        if k >= n_burn:
            acc += np.mean(x ** 2, axis=0)
    return acc / (n_steps - n_burn)


def run_chains_moments(step: Callable[[np.ndarray], np.ndarray], x0: np.ndarray,
                       n_steps: int, n_burn: int, thin: int = 10):
    '''Snapshots of second moments E[x_i^2] and fourth moments E[x_i^2 x_j^2] after burn-in.

    Returns (m2, m4, se2, se4): means over thinned snapshots and their standard errors
    (snapshots treated as roughly independent; each averages over all chains).
    '''
    if n_burn >= n_steps:
        raise ValueError('n_burn must be < n_steps')
    x = x0.copy()
    rec2, rec4 = [], []
    for k in range(n_steps):
        x = step(x)
        if k >= n_burn and (k - n_burn) % thin == 0:
            x2 = x ** 2
            rec2.append(x2.mean(axis=0))
            rec4.append(x2.T @ x2 / x.shape[0])
    rec2, rec4 = np.array(rec2), np.array(rec4)
    n = len(rec2)
    return rec2.mean(0), rec4.mean(0), rec2.std(0) / np.sqrt(n), rec4.std(0) / np.sqrt(n)
