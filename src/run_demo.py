'''End-to-end demo on SYNTHETIC data. Reference implementation of a PROPOSAL.

For a synthetic Gaussian target it prints the exact stationary quantities of ULA
and of the overdamped randomized midpoint method (RMP), plus a small Monte Carlo
cross-check of the RMP variance. Values are for this run only; they are not
benchmarks and carry no claim about the literature.

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
    rng = np.random.default_rng(0)
    hs = [h for h in (0.01, 0.02, 0.05, 0.08) if h * L < 1.0]
    print(f'SYNTHETIC Gaussian target, d={d}, L={L:.3g}, kappa={L / lams.min():.3g}')
    print('h, ULA_W2_exact, RMP_W2_lower_bound, RMP_KL_cov_part, RMP_KL_mix_2nd_order, '
          'RMP_var_mean_rel_dev_sim_vs_exact')
    ula_w2, rmp_w2 = [], []
    for h in hs:
        w2_ula = np.sqrt(eg.ula_w2_squared(lams, h))
        w2_rmp = eg.rmp_w2_gelbrich_lower_bound(lams, h)
        ula_w2.append(w2_ula)
        rmp_w2.append(w2_rmp)
        v = eg.rmp_stationary_variance(lams, h)
        grad = lambda x: lams * x
        step = lambda x: samplers.rmp_step(x, h, grad, rng)
        x0 = rng.standard_normal((500, d)) / np.sqrt(lams)
        sim = samplers.run_chains(step, x0, n_steps=400, n_burn=200)
        dev = float(np.mean(np.abs(sim / v - 1.0)))
        print(f'{h}, {w2_ula:.4g}, {w2_rmp:.4g}, {eg.rmp_kl_covariance_part(lams, h):.4g}, '
              f'{eg.rmp_kl_mixture_second_order(lams, h):.4g}, {dev:.3g}')
    out = SPEC_PATH.parent / 'bias_vs_step.png'
    curves = {'ULA exact W2 (SYNTHETIC)': ula_w2, 'RMP covariance-part W2 lower bound (SYNTHETIC)': rmp_w2}
    if SPEC_PATH.parent.exists() and report.plot_bias_vs_step(hs, curves, str(out)):
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
