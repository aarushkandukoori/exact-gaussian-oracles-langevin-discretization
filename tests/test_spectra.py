'''Tests for SYNTHETIC spectrum generators.'''
import numpy as np

from src import spectra


def test_geometric_endpoints():
    lams = spectra.geometric(10, 1.0, 100.0)
    assert lams[0] == 1.0 and abs(lams[-1] - 100.0) < 1e-9


def test_few_large_counts():
    lams = spectra.few_large(10, 1.0, 50.0, k=3)
    assert np.sum(lams == 50.0) == 3 and len(lams) == 10
