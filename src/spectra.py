'''SYNTHETIC precision-spectrum generators for the proposed testbed.

Reference implementation of a PROPOSAL (no results). Each function returns the
precisions lambda_i = sigma_i^-2 of a synthetic centered Gaussian target.
'''
from __future__ import annotations

import numpy as np


def isotropic(d: int, value: float) -> np.ndarray:
    '''SYNTHETIC: all precisions equal to `value`.'''
    return np.full(d, float(value))


def geometric(d: int, m: float, L: float) -> np.ndarray:
    '''SYNTHETIC: precisions geometrically spaced between m and L (kappa = L/m).'''
    return np.geomspace(m, L, d)


def few_large(d: int, m: float, L: float, k: int = 2) -> np.ndarray:
    '''SYNTHETIC: k precisions equal to L, the rest equal to m.

    Separates dependence on d from dependence on tr(Sigma^-1).
    '''
    if not 0 <= k <= d:
        raise ValueError('k must be in [0, d]')
    lams = np.full(d, float(m))
    lams[:k] = L
    return lams
