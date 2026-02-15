"""
Shared fixtures for mixed model tests.

Provides realistic test datasets with known structure for LMM and GLMM.
"""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(2024)


@pytest.fixture
def sleepstudy_like(rng):
    """Sleepstudy-like dataset: reaction time ~ days + (1 + days | subject).

    18 subjects, 10 days each = 180 observations.
    Random intercept SD ≈ 25, random slope SD ≈ 6, correlation ≈ 0.07.
    Residual SD ≈ 25.
    """
    n_subjects = 18
    n_days = 10
    n = n_subjects * n_days

    # True parameters
    beta_intercept = 250.0
    beta_days = 10.0
    sigma_intercept = 25.0
    sigma_slope = 6.0
    rho = 0.07
    sigma_resid = 25.0

    # Random effects covariance
    cov_matrix = np.array([
        [sigma_intercept**2, rho * sigma_intercept * sigma_slope],
        [rho * sigma_intercept * sigma_slope, sigma_slope**2],
    ])

    # Generate random effects
    re = rng.multivariate_normal([0, 0], cov_matrix, size=n_subjects)

    # Generate data
    subject = np.repeat(np.arange(n_subjects), n_days)
    days = np.tile(np.arange(n_days, dtype=float), n_subjects)

    y = np.zeros(n)
    for i in range(n):
        s = subject[i]
        y[i] = (beta_intercept + re[s, 0]
                + (beta_days + re[s, 1]) * days[i]
                + rng.normal(0, sigma_resid))

    # X with intercept
    X = np.column_stack([np.ones(n), days])

    return {
        'y': y, 'X': X, 'subject': subject, 'days': days,
        'n_subjects': n_subjects, 'n_days': n_days,
        'beta_intercept': beta_intercept, 'beta_days': beta_days,
        'sigma_intercept': sigma_intercept, 'sigma_slope': sigma_slope,
        'sigma_resid': sigma_resid,
    }


@pytest.fixture
def random_intercept_simple(rng):
    """Simple random intercept dataset: y ~ x + (1 | group).

    20 groups, 10 observations each = 200 observations.
    """
    n_groups = 20
    n_per_group = 10
    n = n_groups * n_per_group

    beta0 = 5.0
    beta1 = 2.0
    sigma_group = 3.0
    sigma_resid = 1.0

    group_effects = rng.normal(0, sigma_group, size=n_groups)
    group = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.normal(0, 1, size=n)

    y = beta0 + beta1 * x + group_effects[group] + rng.normal(0, sigma_resid, size=n)

    X = np.column_stack([np.ones(n), x])

    return {
        'y': y, 'X': X, 'group': group, 'x': x,
        'n_groups': n_groups, 'n_per_group': n_per_group,
        'beta0': beta0, 'beta1': beta1,
        'sigma_group': sigma_group, 'sigma_resid': sigma_resid,
    }


@pytest.fixture
def crossed_effects(rng):
    """Crossed random effects: y ~ x + (1 | subject) + (1 | item).

    30 subjects × 10 items = 300 observations.
    """
    n_subjects = 30
    n_items = 10
    n = n_subjects * n_items

    beta0 = 3.0
    beta1 = 1.5
    sigma_subject = 2.0
    sigma_item = 1.5
    sigma_resid = 1.0

    subject_effects = rng.normal(0, sigma_subject, size=n_subjects)
    item_effects = rng.normal(0, sigma_item, size=n_items)

    subject = np.repeat(np.arange(n_subjects), n_items)
    item = np.tile(np.arange(n_items), n_subjects)
    x = rng.normal(0, 1, size=n)

    y = (beta0 + beta1 * x
         + subject_effects[subject]
         + item_effects[item]
         + rng.normal(0, sigma_resid, size=n))

    X = np.column_stack([np.ones(n), x])

    return {
        'y': y, 'X': X, 'subject': subject, 'item': item, 'x': x,
        'n_subjects': n_subjects, 'n_items': n_items,
        'beta0': beta0, 'beta1': beta1,
        'sigma_subject': sigma_subject, 'sigma_item': sigma_item,
        'sigma_resid': sigma_resid,
    }


@pytest.fixture
def nested_effects(rng):
    """Nested random effects: y ~ x + (1 | classroom) + (1 | classroom:student).

    5 classrooms × 6 students × 4 observations = 120 observations.
    """
    n_classrooms = 5
    n_students_per = 6
    n_obs_per = 4
    n_students = n_classrooms * n_students_per
    n = n_students * n_obs_per

    beta0 = 10.0
    beta1 = 0.5
    sigma_classroom = 3.0
    sigma_student = 1.5
    sigma_resid = 1.0

    classroom_effects = rng.normal(0, sigma_classroom, size=n_classrooms)
    student_effects = rng.normal(0, sigma_student, size=n_students)

    classroom = np.repeat(
        np.repeat(np.arange(n_classrooms), n_students_per),
        n_obs_per
    )
    student = np.repeat(np.arange(n_students), n_obs_per)
    x = rng.normal(0, 1, size=n)

    y = (beta0 + beta1 * x
         + classroom_effects[classroom]
         + student_effects[student]
         + rng.normal(0, sigma_resid, size=n))

    X = np.column_stack([np.ones(n), x])

    return {
        'y': y, 'X': X, 'classroom': classroom, 'student': student, 'x': x,
        'n_classrooms': n_classrooms, 'n_students': n_students,
        'beta0': beta0, 'beta1': beta1,
        'sigma_classroom': sigma_classroom, 'sigma_student': sigma_student,
        'sigma_resid': sigma_resid,
    }


@pytest.fixture
def glmm_binomial(rng):
    """Binomial GLMM dataset: binary y ~ x + (1 | group).

    20 groups, 20 observations each = 400 observations.
    """
    n_groups = 20
    n_per_group = 20
    n = n_groups * n_per_group

    beta0 = -0.5
    beta1 = 1.0
    sigma_group = 1.0

    group_effects = rng.normal(0, sigma_group, size=n_groups)
    group = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.normal(0, 1, size=n)

    eta = beta0 + beta1 * x + group_effects[group]
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(float)

    X = np.column_stack([np.ones(n), x])

    return {
        'y': y, 'X': X, 'group': group, 'x': x,
        'n_groups': n_groups,
        'beta0': beta0, 'beta1': beta1,
        'sigma_group': sigma_group,
    }


@pytest.fixture
def glmm_poisson(rng):
    """Poisson GLMM dataset: count y ~ x + (1 | group).

    15 groups, 20 observations each = 300 observations.
    """
    n_groups = 15
    n_per_group = 20
    n = n_groups * n_per_group

    beta0 = 1.0
    beta1 = 0.5
    sigma_group = 0.5

    group_effects = rng.normal(0, sigma_group, size=n_groups)
    group = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.normal(0, 1, size=n)

    eta = beta0 + beta1 * x + group_effects[group]
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)

    X = np.column_stack([np.ones(n), x])

    return {
        'y': y, 'X': X, 'group': group, 'x': x,
        'n_groups': n_groups,
        'beta0': beta0, 'beta1': beta1,
        'sigma_group': sigma_group,
    }
