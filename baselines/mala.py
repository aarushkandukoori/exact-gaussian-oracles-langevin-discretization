'''Baseline stub: MALA, simulation-only reference. SYNTHETIC targets only.'''
import numpy as np

from src import perturbations, samplers


def run(lams, h, eps=0.0, n_chains=500, n_steps=400, n_burn=200, seed=0):
    '''Return the empirical per-coordinate second moment of MALA on a SYNTHETIC target.'''
    rng = np.random.default_rng(seed)
    lams = np.asarray(lams, dtype=float)
    f, grad = perturbations.logcosh_potential(lams, eps)
    x0 = rng.standard_normal((n_chains, len(lams))) / np.sqrt(lams)
    step = lambda x: samplers.mala_step(x, h, f, grad, rng)
    return samplers.run_chains(step, x0, n_steps, n_burn)
