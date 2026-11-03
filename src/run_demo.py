'''End-to-end demo on SYNTHETIC data. Reference implementation of a PROPOSAL.

Compares the exact ULA stationary variance with a Monte Carlo estimate on a
synthetic Gaussian target. Prints measured values for this run only; they are
not benchmarks and carry no claim about the literature.

Run from the repo root: python -m src.run_demo
'''
from __future__ import annotations

import json
import pathlib

import numpy as np

from src import exact_gaussian as eg
from src import report, samplers, spectra

SPEC_PATH = pathlib.Path(__file__).resolve().parent.parent / 'data' / 'synthetic' / 'spectra.json'


def load_precisions(name: str = 'geometric') -> np.ndarray:
    '''Load a SYNTHETIC spectrum from data/synthetic if present, else generate one.'''
    if SPEC_PATH.exists():
        data = json.loads(SPEC_PATH.read_text())
        if data.get('synthetic') is not True:
            raise ValueError('refusing to load data not labeled synthetic')
        return np.array(data['spectra'][name], dtype=float)
    return spectra.geometric(20, 1.0, 10.0)


def main() -> None:
    lams = load_precisions()
    d, L = len(lams), lams.max()
    sigma2 = 1.0 / lams
    rng = np.random.default_rng(0)
    hs = [h for h in (0.02, 0.05, 0.1, 0.2) if h < 2.0 / L]
    print(f'SYNTHETIC Gaussian target, d={d}, L={L:.3g}')
    print('h, exact_W2, leading_order_W2, cov_rel_err_exact, mean_rel_dev_sim_vs_exact')
    w2s = []
    for h in hs:
        v = eg.ula_stationary_variance(lams, h)
        w2 = np.sqrt(eg.ula_w2_squared(lams, h))
        w2s.append(w2)
        grad = lambda x: lams * x
        step = lambda x: samplers.ula_step(x, h, grad, rng)
        x0 = rng.standard_normal((500, d)) * np.sqrt(sigma2)
        sim = samplers.run_chains(step, x0, n_steps=400, n_burn=200)
        dev = float(np.mean(np.abs(sim / v - 1.0)))
        print(f'{h}, {w2:.4g}, {np.sqrt(eg.w2_leading_order(lams, h)):.4g}, '
              f'{eg.covariance_error(v, sigma2):.4g}, {dev:.3g}')
    out = SPEC_PATH.parent / 'bias_vs_step.png'
    if SPEC_PATH.parent.exists() and report.plot_bias_vs_step(hs, {'ULA exact (SYNTHETIC)': w2s}, str(out)):
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
