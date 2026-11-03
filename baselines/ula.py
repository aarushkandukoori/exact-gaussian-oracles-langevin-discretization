'''Baseline stub: ULA (simulation). Reference code for a PROPOSAL; SYNTHETIC targets only.'''
import numpy as np

from src import samplers


def run(lams, h, n_chains=500, n_steps=400, n_burn=200, seed=0):
    '''Return the empirical per-coordinate variance of ULA on a SYNTHETIC Gaussian.'''
    rng = np.random.default_rng(seed)
    lams = np.asarray(lams, dtype=float)
    x0 = rng.standard_normal((n_chains, len(lams))) / np.sqrt(lams)
    step = lambda x: samplers.ula_step(x, h, lambda z: lams * z, rng)
    return samplers.run_chains(step, x0, n_steps, n_burn)
