#!/usr/bin/env python3
"""Generate input data for the prior-weights / offset GLM+OLS validation.

Writes ``weights_offset_cases.json`` (inputs only). The R reference outputs are
produced by ``run_r_weights_offset_validation.R`` into
``weights_offset_r_results.json``; ``tests/regression/test_weights_offset_r_validation.py``
validates PyStatistics against them.

Each case fixes a family + link and supplies prior ``weights`` and/or an
``offset`` so the reference exercises R's ``glm(..., weights=, offset=)`` /
``lm(..., weights=)`` semantics. Binomial weights are integers (case weights),
where R's AIC is well defined; the others use positive real weights.

Run from /path/to/pystatistics:
    python tests/fixtures/generate_weights_offset_fixtures.py
"""

import json
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(20260629)
FIXTURES_DIR = Path(__file__).parent


def _design(n, p, rng):
    return np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])


def build_cases():
    cases = {}

    # --- Gaussian: WLS + offset, weights-only, offset-only ----------------
    n = 90
    X = _design(n, 3, RNG)
    y = X @ np.array([1.0, 2.0, -0.5]) + RNG.standard_normal(n) * 0.5
    w = RNG.uniform(0.2, 3.0, n)
    off = RNG.standard_normal(n) * 0.3
    cases['gaussian_wls_offset'] = dict(
        family='gaussian', link='identity', X=X, y=y, weights=w, offset=off)
    cases['gaussian_wls'] = dict(
        family='gaussian', link='identity', X=X, y=y, weights=w, offset=None)
    cases['gaussian_offset'] = dict(
        family='gaussian', link='identity', X=X, y=y, weights=None, offset=off)

    # --- Binomial logit: integer case weights + offset --------------------
    n = 160
    X = _design(n, 3, RNG)
    prob = 1.0 / (1.0 + np.exp(-(X @ np.array([0.2, 1.0, -0.7]))))
    y = RNG.binomial(1, prob).astype(float)
    w = RNG.integers(1, 5, n).astype(float)
    off = RNG.standard_normal(n) * 0.2
    cases['binomial_weights_offset'] = dict(
        family='binomial', link='logit', X=X, y=y, weights=w, offset=off)

    # --- Poisson log: weights + log-exposure offset (rate model) ----------
    n = 130
    X = _design(n, 3, RNG)
    mu = np.exp(X @ np.array([0.5, 0.3, -0.2]) + 0.4 * RNG.standard_normal(n))
    y = RNG.poisson(mu).astype(float)
    w = RNG.uniform(0.5, 2.0, n)
    off = np.log(RNG.uniform(0.5, 2.0, n))
    cases['poisson_weights_offset'] = dict(
        family='poisson', link='log', X=X, y=y, weights=w, offset=off)

    # --- Gamma: log link and default inverse link, weights + offset -------
    n = 110
    X = _design(n, 2, RNG)
    mu = np.exp(X @ np.array([0.7, 0.4]))
    y = RNG.gamma(shape=5.0, scale=mu / 5.0)
    w = RNG.uniform(0.5, 2.0, n)
    off = RNG.standard_normal(n) * 0.15
    cases['gamma_log_weights_offset'] = dict(
        family='Gamma', link='log', X=X, y=y, weights=w, offset=off)

    n = 110
    X = _design(n, 2, RNG)
    mu = 1.0 / (0.5 + 0.2 * np.abs(X[:, 1]))   # keep eta = 1/mu positive
    y = RNG.gamma(shape=8.0, scale=mu / 8.0)
    w = RNG.uniform(0.5, 2.0, n)
    cases['gamma_inverse_weights'] = dict(
        family='Gamma', link='inverse', X=X, y=y, weights=w, offset=None)

    # --- Negative binomial (fixed theta) log: weights + offset ------------
    n = 140
    X = _design(n, 3, RNG)
    mu = np.exp(X @ np.array([0.6, 0.3, -0.25]))
    theta = 2.5
    y = RNG.negative_binomial(theta, theta / (theta + mu)).astype(float)
    w = RNG.uniform(0.5, 2.0, n)
    off = RNG.standard_normal(n) * 0.2
    cases['negbin_weights_offset'] = dict(
        family='negative.binomial', link='log', theta=theta,
        X=X, y=y, weights=w, offset=off)

    return cases


def main():
    cases = build_cases()
    out = {}
    for name, c in cases.items():
        rec = {k: v for k, v in c.items() if not isinstance(v, np.ndarray)}
        rec['X'] = c['X'].tolist()
        rec['y'] = c['y'].tolist()
        rec['weights'] = None if c['weights'] is None else np.asarray(c['weights']).tolist()
        rec['offset'] = None if c['offset'] is None else np.asarray(c['offset']).tolist()
        rec['n'], rec['p'] = c['X'].shape
        out[name] = rec
    path = FIXTURES_DIR / 'weights_offset_cases.json'
    with open(path, 'w') as f:
        json.dump(out, f)
    print(f"Wrote {len(out)} cases to {path}")
    for name, rec in out.items():
        print(f"  {name}: {rec['family']}/{rec['link']} n={rec['n']} p={rec['p']} "
              f"weights={'yes' if rec['weights'] else 'no'} "
              f"offset={'yes' if rec['offset'] else 'no'}")


if __name__ == '__main__':
    main()
