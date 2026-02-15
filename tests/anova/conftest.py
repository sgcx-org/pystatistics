"""
Shared fixtures for ANOVA tests.

Provides reusable datasets for one-way, factorial, repeated-measures,
and ANCOVA test scenarios.
"""

import numpy as np
import pytest


# =====================================================================
# One-way fixtures
# =====================================================================


@pytest.fixture
def oneway_balanced():
    """3-group balanced design (n=10 each), clear group differences."""
    rng = np.random.default_rng(42)
    n_per_group = 10
    y = np.concatenate([
        rng.normal(10.0, 2.0, n_per_group),
        rng.normal(15.0, 2.0, n_per_group),
        rng.normal(20.0, 2.0, n_per_group),
    ])
    group = np.array(['A'] * n_per_group + ['B'] * n_per_group + ['C'] * n_per_group)
    return y, group


@pytest.fixture
def oneway_unbalanced():
    """3-group unbalanced design (n=5, 10, 15)."""
    rng = np.random.default_rng(123)
    y = np.concatenate([
        rng.normal(10.0, 2.0, 5),
        rng.normal(15.0, 2.0, 10),
        rng.normal(20.0, 2.0, 15),
    ])
    group = np.array(['A'] * 5 + ['B'] * 10 + ['C'] * 15)
    return y, group


@pytest.fixture
def oneway_no_effect():
    """3-group design where all groups have same mean."""
    rng = np.random.default_rng(99)
    n_per = 15
    y = rng.normal(10.0, 2.0, n_per * 3)
    group = np.array(['A'] * n_per + ['B'] * n_per + ['C'] * n_per)
    return y, group


@pytest.fixture
def oneway_two_groups():
    """2-group design (should match independent t-test)."""
    rng = np.random.default_rng(77)
    n = 20
    y = np.concatenate([
        rng.normal(10.0, 3.0, n),
        rng.normal(14.0, 3.0, n),
    ])
    group = np.array(['control'] * n + ['treatment'] * n)
    return y, group


# =====================================================================
# Factorial fixtures
# =====================================================================


@pytest.fixture
def twoway_balanced():
    """2x3 balanced factorial design."""
    rng = np.random.default_rng(42)
    n_per_cell = 10

    y_list = []
    a_list = []
    b_list = []
    for a_level in ['low', 'high']:
        for b_level in ['X', 'Y', 'Z']:
            mean = 10.0
            if a_level == 'high':
                mean += 5.0
            if b_level == 'Y':
                mean += 3.0
            elif b_level == 'Z':
                mean += 6.0
            y_list.append(rng.normal(mean, 2.0, n_per_cell))
            a_list.extend([a_level] * n_per_cell)
            b_list.extend([b_level] * n_per_cell)

    return np.concatenate(y_list), np.array(a_list), np.array(b_list)


@pytest.fixture
def twoway_unbalanced():
    """2x2 unbalanced factorial design."""
    rng = np.random.default_rng(55)

    cells = {
        ('A', 'X'): (10.0, 8),
        ('A', 'Y'): (15.0, 12),
        ('B', 'X'): (12.0, 6),
        ('B', 'Y'): (18.0, 10),
    }

    y_list = []
    f1_list = []
    f2_list = []
    for (a, b), (mean, n) in cells.items():
        y_list.append(rng.normal(mean, 2.0, n))
        f1_list.extend([a] * n)
        f2_list.extend([b] * n)

    return np.concatenate(y_list), np.array(f1_list), np.array(f2_list)


@pytest.fixture
def ancova_data():
    """One factor + one covariate (ANCOVA)."""
    rng = np.random.default_rng(42)
    n_per = 15

    y_list = []
    group_list = []
    cov_list = []
    for level, base_mean in [('control', 10.0), ('drug_A', 15.0), ('drug_B', 18.0)]:
        x = rng.uniform(20, 60, n_per)  # age covariate
        y_val = base_mean + 0.1 * x + rng.normal(0, 2, n_per)
        y_list.append(y_val)
        group_list.extend([level] * n_per)
        cov_list.append(x)

    return (
        np.concatenate(y_list),
        np.array(group_list),
        np.concatenate(cov_list),
    )


# =====================================================================
# Repeated measures fixtures
# =====================================================================


@pytest.fixture
def rm_within_3():
    """10 subjects, 3 conditions (within), long format."""
    rng = np.random.default_rng(42)
    n_subjects = 10
    n_conditions = 3

    # Subject effects + condition effects + noise
    subject_effects = rng.normal(0, 3, n_subjects)
    condition_means = [10.0, 15.0, 20.0]

    y_list = []
    subj_list = []
    cond_list = []

    for i in range(n_subjects):
        for j, cond_mean in enumerate(condition_means):
            y_val = cond_mean + subject_effects[i] + rng.normal(0, 1.0)
            y_list.append(y_val)
            subj_list.append(f"S{i:02d}")
            cond_list.append(f"cond_{j}")

    return np.array(y_list), np.array(subj_list), np.array(cond_list)


@pytest.fixture
def rm_within_2():
    """15 subjects, 2 conditions (epsilon = 1, sphericity trivially satisfied)."""
    rng = np.random.default_rng(88)
    n_subjects = 15
    subject_effects = rng.normal(0, 4, n_subjects)

    y_list = []
    subj_list = []
    cond_list = []

    for i in range(n_subjects):
        for cond_mean, cond_name in [(10.0, 'pre'), (14.0, 'post')]:
            y_list.append(cond_mean + subject_effects[i] + rng.normal(0, 1.5))
            subj_list.append(f"S{i:02d}")
            cond_list.append(cond_name)

    return np.array(y_list), np.array(subj_list), np.array(cond_list)


@pytest.fixture
def rm_mixed():
    """Mixed design: 2 between-groups x 3 within-conditions, 8 subjects/group."""
    rng = np.random.default_rng(42)
    n_per_group = 8

    y_list = []
    subj_list = []
    within_list = []
    between_list = []

    subject_id = 0
    for group in ['control', 'treatment']:
        group_effect = 5.0 if group == 'treatment' else 0.0
        for _ in range(n_per_group):
            subj_effect = rng.normal(0, 3)
            for j, cond_mean in enumerate([10.0, 15.0, 20.0]):
                y_val = cond_mean + group_effect + subj_effect + rng.normal(0, 1.5)
                y_list.append(y_val)
                subj_list.append(f"S{subject_id:02d}")
                within_list.append(f"time_{j}")
                between_list.append(group)
            subject_id += 1

    return (
        np.array(y_list),
        np.array(subj_list),
        np.array(within_list),
        np.array(between_list),
    )
