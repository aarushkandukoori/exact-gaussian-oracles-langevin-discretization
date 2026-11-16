'''Baseline: overdamped randomized midpoint Langevin (simulation). SYNTHETIC targets only.

Reference code for a PROPOSAL. The exact second and fourth moments of this chain on
Gaussian targets are in src/exact_gaussian.py (rmp_*); this module only simulates it.
The underdamped variant (following Shen and Lee) is planned, not implemented.
'''
import numpy as np

from src import perturbations, samplers


def run(lams, h, eps=0.0, n_chains=500, n_steps=400, n_burn=200, seed=0):
    '''Return the empirical per-coordinate second moment of overdamped RMP on a SYNTHETIC target.'''
    rng = np.random.default_rng(seed)
    lams = np.asarray(lams, dtype=float)
    _, grad = perturbations.logcosh_potential(lams, eps)
    x0 = rng.standard_normal((n_chains, len(lams))) / np.sqrt(lams)
    step = lambda x: samplers.rmp_step(x, h, grad, rng)
    return samplers.run_chains(step, x0, n_steps, n_burn)


def run_underdamped(*args, **kwargs):
    raise NotImplementedError('underdamped randomized midpoint baseline is planned, not implemented')
