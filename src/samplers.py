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


def randomized_midpoint_step(x, v, h, grad, rng):
    '''STUB: randomized midpoint step on underdamped Langevin (Shen and Lee).

    Not implemented in this skeleton.
    '''
    raise NotImplementedError('randomized midpoint is planned, not implemented')


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
