'''Write SYNTHETIC data into data/synthetic/. Reference code for a PROPOSAL.

Everything written here is mock/synthetic: precision spectra of analytic
Gaussian targets and a few mock samples drawn from one of them. No real data.

Run: python data/generate_synthetic.py
'''
import json
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).resolve().parent / 'synthetic'


def main(d: int = 20, m: float = 1.0, L: float = 10.0, k: int = 2) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    few = np.full(d, m)
    few[:k] = L
    spec = {
        'isotropic': np.full(d, L).tolist(),
        'geometric': np.geomspace(m, L, d).tolist(),
        'few_large': few.tolist(),
    }
    payload = {'synthetic': True, 'note': 'SYNTHETIC mock spectra, no real data', 'spectra': spec}
    (OUT / 'spectra.json').write_text(json.dumps(payload, indent=2))

    rng = np.random.default_rng(0)
    lams = np.array(spec['geometric'])
    samples = rng.standard_normal((100, d)) / np.sqrt(lams)
    header = '# SYNTHETIC mock samples from N(0, diag(1/lam)), geometric spectrum; not real data'
    np.savetxt(OUT / 'mock_samples.csv', samples, delimiter=',', header=header, comments='')
    print(f'wrote SYNTHETIC data to {OUT}')


if __name__ == '__main__':
    main()
