'''Controlled non-Gaussian perturbations of SYNTHETIC Gaussian targets.

Reference implementation of a PROPOSAL. f_eps(x) = 0.5 sum(lam_i x_i^2) + eps * sum(log cosh x_i).
No prediction about outcomes is made.
'''
from __future__ import annotations

import numpy as np


def logcosh_potential(lams: np.ndarray, eps: float):
    '''Return (f, grad) for the SYNTHETIC perturbed potential, batched over rows of x.'''
    lams = np.asarray(lams, dtype=float)

    def f(x: np.ndarray) -> np.ndarray:
        quad = 0.5 * np.sum(lams * x ** 2, axis=1)
        pert = np.sum(np.logaddexp(x, -x) - np.log(2.0), axis=1)
        return quad + eps * pert

    def grad(x: np.ndarray) -> np.ndarray:
        return lams * x + eps * np.tanh(x)

    return f, grad
