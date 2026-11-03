'''Plotting for the proposed testbed. All figures are labeled SYNTHETIC.'''
from __future__ import annotations

import numpy as np


def plot_bias_vs_step(hs, curves: dict, path: str) -> bool:
    '''Log-log plot of W2 bias vs step size. Returns False if matplotlib is missing.'''
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots()
    for name, ys in curves.items():
        ax.loglog(hs, np.asarray(ys), marker='o', label=name)
    ax.set_xlabel('step size h')
    ax.set_ylabel('W2 bias to target')
    ax.set_title('SYNTHETIC Gaussian target (proposal reference code)')
    ax.legend()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True
