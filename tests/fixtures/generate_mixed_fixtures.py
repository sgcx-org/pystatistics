"""
Generate test fixtures for mixed model R validation.

Creates JSON metadata and CSV data files that the R script reads
to produce reference values.
"""

import json
import os
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'mixed')
SEED = 2024


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    scenarios = {}

    # --- LMM: Random intercept ---
    scenarios['lmm_intercept'] = _generate_random_intercept(rng)

    # --- LMM: Random intercept + slope (sleepstudy-like) ---
    scenarios['lmm_slope'] = _generate_random_slope(rng)

    # --- LMM: Crossed random effects ---
    scenarios['lmm_crossed'] = _generate_crossed(rng)

    # --- LMM: ML estimation ---
    scenarios['lmm_ml'] = _generate_random_intercept(rng, name='lmm_ml')

    # --- LMM: No effect (null model) ---
    scenarios['lmm_no_effect'] = _generate_no_effect(rng)

    # --- GLMM: Binomial ---
    scenarios['glmm_binomial'] = _generate_glmm_binomial(rng)

    # --- GLMM: Poisson ---
    scenarios['glmm_poisson'] = _generate_glmm_poisson(rng)

    # Write master metadata
    meta = {name: s['meta'] for name, s in scenarios.items()}
    meta_path = os.path.join(OUTPUT_DIR, 'mixed_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")

    # Write CSV files
    for name, s in scenarios.items():
        csv_path = os.path.join(OUTPUT_DIR, f'{name}.csv')
        _write_csv(csv_path, s['data'])
        print(f"Wrote {csv_path}")

    print(f"\nGenerated {len(scenarios)} fixture scenarios.")


def _generate_random_intercept(rng, name='lmm_intercept'):
    n_groups = 15
    n_per = 8
    n = n_groups * n_per

    beta0, beta1 = 5.0, 2.0
    sigma_group = 3.0
    sigma_resid = 1.0

    group_effects = rng.normal(0, sigma_group, n_groups)
    group = np.repeat(np.arange(n_groups), n_per)
    x = rng.normal(0, 1, n)
    y = beta0 + beta1 * x + group_effects[group] + rng.normal(0, sigma_resid, n)

    data = {'y': y, 'x': x, 'group': group}
    meta = {
        'type': 'lmm', 'name': name,
        'n': n, 'n_groups': n_groups,
        'formula': 'y ~ x + (1|group)',
        'true_beta': [beta0, beta1],
        'true_sigma_group': sigma_group,
        'true_sigma_resid': sigma_resid,
    }
    return {'data': data, 'meta': meta}


def _generate_random_slope(rng):
    n_subjects = 18
    n_days = 10
    n = n_subjects * n_days

    beta0, beta1 = 250.0, 10.0
    sigma_int = 25.0
    sigma_slope = 6.0
    rho = 0.07
    sigma_resid = 25.0

    cov = np.array([
        [sigma_int**2, rho * sigma_int * sigma_slope],
        [rho * sigma_int * sigma_slope, sigma_slope**2],
    ])
    re = rng.multivariate_normal([0, 0], cov, n_subjects)

    subject = np.repeat(np.arange(n_subjects), n_days)
    days = np.tile(np.arange(n_days, dtype=float), n_subjects)

    y = np.zeros(n)
    for i in range(n):
        s = subject[i]
        y[i] = beta0 + re[s, 0] + (beta1 + re[s, 1]) * days[i] + rng.normal(0, sigma_resid)

    data = {'y': y, 'days': days, 'subject': subject}
    meta = {
        'type': 'lmm', 'name': 'lmm_slope',
        'n': n, 'n_subjects': n_subjects,
        'formula': 'y ~ days + (1 + days|subject)',
    }
    return {'data': data, 'meta': meta}


def _generate_crossed(rng):
    n_subjects = 20
    n_items = 8
    n = n_subjects * n_items

    beta0, beta1 = 3.0, 1.5
    sigma_subj = 2.0
    sigma_item = 1.5
    sigma_resid = 1.0

    subj_eff = rng.normal(0, sigma_subj, n_subjects)
    item_eff = rng.normal(0, sigma_item, n_items)

    subject = np.repeat(np.arange(n_subjects), n_items)
    item = np.tile(np.arange(n_items), n_subjects)
    x = rng.normal(0, 1, n)

    y = beta0 + beta1 * x + subj_eff[subject] + item_eff[item] + rng.normal(0, sigma_resid, n)

    data = {'y': y, 'x': x, 'subject': subject, 'item': item}
    meta = {
        'type': 'lmm', 'name': 'lmm_crossed',
        'n': n, 'n_subjects': n_subjects, 'n_items': n_items,
        'formula': 'y ~ x + (1|subject) + (1|item)',
    }
    return {'data': data, 'meta': meta}


def _generate_no_effect(rng):
    n_groups = 10
    n_per = 10
    n = n_groups * n_per

    group = np.repeat(np.arange(n_groups), n_per)
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)  # pure noise

    data = {'y': y, 'x': x, 'group': group}
    meta = {
        'type': 'lmm', 'name': 'lmm_no_effect',
        'n': n, 'n_groups': n_groups,
        'formula': 'y ~ x + (1|group)',
    }
    return {'data': data, 'meta': meta}


def _generate_glmm_binomial(rng):
    n_groups = 20
    n_per = 20
    n = n_groups * n_per

    beta0, beta1 = -0.5, 1.0
    sigma_group = 1.0

    group_eff = rng.normal(0, sigma_group, n_groups)
    group = np.repeat(np.arange(n_groups), n_per)
    x = rng.normal(0, 1, n)

    eta = beta0 + beta1 * x + group_eff[group]
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(float)

    data = {'y': y, 'x': x, 'group': group}
    meta = {
        'type': 'glmm', 'name': 'glmm_binomial',
        'family': 'binomial', 'n': n, 'n_groups': n_groups,
        'formula': 'y ~ x + (1|group)',
    }
    return {'data': data, 'meta': meta}


def _generate_glmm_poisson(rng):
    n_groups = 15
    n_per = 20
    n = n_groups * n_per

    beta0, beta1 = 1.0, 0.5
    sigma_group = 0.5

    group_eff = rng.normal(0, sigma_group, n_groups)
    group = np.repeat(np.arange(n_groups), n_per)
    x = rng.normal(0, 1, n)

    eta = beta0 + beta1 * x + group_eff[group]
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)

    data = {'y': y, 'x': x, 'group': group}
    meta = {
        'type': 'glmm', 'name': 'glmm_poisson',
        'family': 'poisson', 'n': n, 'n_groups': n_groups,
        'formula': 'y ~ x + (1|group)',
    }
    return {'data': data, 'meta': meta}


def _write_csv(path, data):
    """Write data dict to CSV."""
    import csv
    keys = list(data.keys())
    n = len(data[keys[0]])
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([data[k][i] for k in keys])


if __name__ == '__main__':
    main()
