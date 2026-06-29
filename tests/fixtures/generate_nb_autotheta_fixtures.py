#!/usr/bin/env python3
"""Generate input data for negative-binomial auto-θ validation against glm.nb.

Writes ``nb_autotheta_cases.json`` (inputs only). The R reference outputs come
from ``run_r_nb_autotheta_validation.R`` (uses ``MASS::glm.nb``, which estimates
θ by profile likelihood) into ``nb_autotheta_r_results.json``;
``tests/regression/test_nb_autotheta_r_validation.py`` validates PyStatistics'
``fit(family='negative.binomial', ...)`` (which also estimates θ) against them.

Each case supplies prior ``weights`` and/or an ``offset`` so the reference
exercises ``glm.nb(y ~ . + offset(off), weights=w)``.

Run from /path/to/pystatistics:
    python tests/fixtures/generate_nb_autotheta_fixtures.py
"""

import json
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(20260629)
FIXTURES_DIR = Path(__file__).parent


def _nb_sample(mu, theta, rng):
    return rng.negative_binomial(theta, theta / (theta + mu)).astype(float)


def build_cases():
    cases = {}

    n = 180
    X = np.column_stack([np.ones(n), RNG.standard_normal(n), RNG.standard_normal(n)])
    mu = np.exp(X @ np.array([0.7, 0.4, -0.3]))
    y = _nb_sample(mu, 3.0, RNG)
    w = RNG.uniform(0.5, 2.0, n)
    off = RNG.standard_normal(n) * 0.2
    cases['nb_auto_weights_offset'] = dict(X=X, y=y, weights=w, offset=off)
    cases['nb_auto_weights'] = dict(X=X, y=y, weights=w, offset=None)

    # A plain (unweighted, no-offset) case pins the auto-θ baseline to glm.nb.
    n = 150
    X = np.column_stack([np.ones(n), RNG.standard_normal(n)])
    mu = np.exp(X @ np.array([1.0, 0.5]))
    y = _nb_sample(mu, 2.0, RNG)
    cases['nb_auto_plain'] = dict(X=X, y=y, weights=None, offset=None)

    return cases


def main():
    cases = build_cases()
    out = {}
    for name, c in cases.items():
        out[name] = dict(
            X=c['X'].tolist(),
            y=c['y'].tolist(),
            weights=None if c['weights'] is None else np.asarray(c['weights']).tolist(),
            offset=None if c['offset'] is None else np.asarray(c['offset']).tolist(),
            n=int(c['X'].shape[0]),
            p=int(c['X'].shape[1]),
        )
    path = FIXTURES_DIR / 'nb_autotheta_cases.json'
    with open(path, 'w') as f:
        json.dump(out, f)
    print(f"Wrote {len(out)} cases to {path}")
    for name, rec in out.items():
        print(f"  {name}: n={rec['n']} p={rec['p']} "
              f"weights={'yes' if rec['weights'] else 'no'} "
              f"offset={'yes' if rec['offset'] else 'no'}")


if __name__ == '__main__':
    main()
