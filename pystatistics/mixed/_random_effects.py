"""
Random effects specification, Z matrix construction, and Λ_θ parameterization.

This module handles:
1. Parsing user-provided grouping variables and random effect terms
2. Building the random effects design matrix Z
3. Constructing the relative covariance factor Λ_θ from the θ parameter vector
4. Computing θ bounds for the optimizer

The θ parameterization follows Bates et al. (2015): θ contains the elements
of the lower-triangular Cholesky factor of the *relative* covariance matrix
(i.e., the covariance divided by σ²).
"""

from __future__ import annotations

from pystatistics.core.exceptions import ValidationError

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RandomEffectSpec:
    """Specification for one grouping factor's random effects.

    Attributes:
        group_name: Name of the grouping factor (e.g. 'subject').
        group_ids: Integer group labels for each observation, shape (n,).
            Values are 0-indexed consecutive integers.
        terms: Names of the random effect terms (e.g. ('1',) or ('1', 'time')).
        Z_block: Dense design matrix block for this grouping factor, shape
            (n, J*q) where J = n_groups and q = n_terms. ``None`` when the
            spec was parsed with ``build_dense=False`` (the structure-exploiting
            LMM path, which never materializes the dense block).
        n_groups: Number of unique groups (J).
        n_terms: Number of random effect terms per group (q).
        theta_size: Number of θ parameters for this block = q*(q+1)/2.
        value_cols: The per-observation term values, shape (n, q): column t is
            the value of term t at each observation (1.0 for an intercept term,
            the slope variable for a slope term). This is the compact (n×q)
            representation the structured solver uses instead of the dense
            (n×J*q) ``Z_block``.
    """
    group_name: str
    group_ids: NDArray
    terms: tuple[str, ...]
    Z_block: NDArray | None
    n_groups: int
    n_terms: int
    theta_size: int
    value_cols: NDArray | None = None


def parse_random_effects(
    groups: dict[str, NDArray],
    random_effects: dict[str, list[str]] | None,
    random_data: dict[str, NDArray] | None,
    n: int,
    build_dense: bool = True,
) -> list[RandomEffectSpec]:
    """Parse user input into structured RandomEffectSpec objects.

    Args:
        groups: Mapping of grouping factor name → group labels array (n,).
        random_effects: Mapping of group name → list of term names.
            If None, defaults to random intercept ('1') for each group.
            Example: {'subject': ['1', 'time']} for (1 + time | subject).
        random_data: Mapping of variable name → data array (n,) for
            random slope variables. Required if any term in random_effects
            is not '1' (intercept).
        n: Number of observations.
        build_dense: If True (default), build the dense (n, J*q) ``Z_block``
            for each factor — the GLMM path needs it. If False, skip the dense
            block (``Z_block=None``) and only populate ``value_cols``; the
            structure-exploiting LMM solver works from ``value_cols`` +
            ``group_ids`` and never materializes the dense design, so this is
            what avoids the O(n·J·q) memory blow-up at large group counts.

    Returns:
        List of RandomEffectSpec, one per grouping factor.
    """
    if random_effects is None:
        random_effects = {name: ['1'] for name in groups}

    if random_data is None:
        random_data = {}

    specs = []
    for group_name in groups:
        group_raw = np.asarray(groups[group_name])
        if group_raw.shape[0] != n:
            raise ValidationError(
                f"Group '{group_name}' has {group_raw.shape[0]} elements, "
                f"expected {n}"
            )

        # Map to consecutive 0-indexed integers
        unique_levels, group_ids = np.unique(group_raw, return_inverse=True)
        n_groups = len(unique_levels)

        # Get terms for this group
        terms = random_effects.get(group_name, ['1'])
        terms = tuple(terms)
        n_terms = len(terms)

        # Compact (n, q) term-value matrix — always cheap to build.
        value_cols = _build_value_cols(terms, random_data, n)

        # Dense (n, J*q) block — only for the GLMM/dense path.
        Z_block = (
            _build_z_block(group_ids, n_groups, terms, random_data, n)
            if build_dense else None
        )

        theta_size = n_terms * (n_terms + 1) // 2

        specs.append(RandomEffectSpec(
            group_name=group_name,
            group_ids=group_ids,
            terms=terms,
            Z_block=Z_block,
            n_groups=n_groups,
            n_terms=n_terms,
            theta_size=theta_size,
            value_cols=value_cols,
        ))

    return specs


def _build_value_cols(
    terms: tuple[str, ...],
    random_data: dict[str, NDArray],
    n: int,
) -> NDArray:
    """Build the compact (n, q) term-value matrix for one grouping factor.

    Column t holds the value of term t at each observation: 1.0 for an
    intercept term ('1'), the slope variable otherwise. This is the
    structured solver's analogue of the dense Z block, without the per-group
    expansion.
    """
    q = len(terms)
    V = np.zeros((n, q), dtype=np.float64)
    for t_idx, term in enumerate(terms):
        if term == '1':
            V[:, t_idx] = 1.0
        else:
            if term not in random_data:
                raise ValidationError(
                    f"Random slope term '{term}' requires data in "
                    f"random_data dict, but '{term}' was not found. "
                    f"Available: {list(random_data.keys())}"
                )
            var_data = np.asarray(random_data[term], dtype=np.float64)
            if var_data.shape[0] != n:
                raise ValidationError(
                    f"Random data '{term}' has {var_data.shape[0]} elements, "
                    f"expected {n}"
                )
            V[:, t_idx] = var_data
    return V


def _build_z_block(
    group_ids: NDArray,
    n_groups: int,
    terms: tuple[str, ...],
    random_data: dict[str, NDArray],
    n: int,
) -> NDArray:
    """Build the Z matrix block for one grouping factor.

    For each group j and each term t, there's a column in Z.
    Layout: columns are ordered as [term0_group0, term0_group1, ...,
    term1_group0, term1_group1, ...] — i.e., term-major ordering.

    For intercept ('1'): Z[i, j] = 1 if observation i belongs to group j.
    For slope (e.g. 'time'): Z[i, j] = time[i] if observation i belongs to group j.

    Args:
        group_ids: (n,) integer group indices.
        n_groups: Number of groups (J).
        terms: Term names.
        random_data: Variable data for slope terms.
        n: Number of observations.

    Returns:
        Z block of shape (n, J * q) where q = len(terms).
    """
    q = len(terms)
    Z = np.zeros((n, n_groups * q), dtype=np.float64)

    for t_idx, term in enumerate(terms):
        base_col = t_idx * n_groups
        if term == '1':
            # Intercept: indicator columns
            for i in range(n):
                Z[i, base_col + group_ids[i]] = 1.0
        else:
            # Slope: indicator × variable value
            if term not in random_data:
                raise ValidationError(
                    f"Random slope term '{term}' requires data in "
                    f"random_data dict, but '{term}' was not found. "
                    f"Available: {list(random_data.keys())}"
                )
            var_data = np.asarray(random_data[term], dtype=np.float64)
            if var_data.shape[0] != n:
                raise ValidationError(
                    f"Random data '{term}' has {var_data.shape[0]} elements, "
                    f"expected {n}"
                )
            for i in range(n):
                Z[i, base_col + group_ids[i]] = var_data[i]

    return Z


def build_z_matrix(specs: list[RandomEffectSpec]) -> NDArray:
    """Concatenate Z blocks from all grouping factors.

    Z = [Z_1 | Z_2 | ...], shape (n, total_q) where
    total_q = sum(J_k * q_k) across all grouping factors.

    Args:
        specs: List of RandomEffectSpec objects.

    Returns:
        Full Z matrix of shape (n, total_q).
    """
    if not specs:
        raise ValidationError("At least one random effect specification required")
    blocks = [spec.Z_block for spec in specs]
    return np.hstack(blocks)


def build_lambda(theta: NDArray, specs: list[RandomEffectSpec]) -> NDArray:
    """Build block-diagonal Λ_θ from the theta parameter vector.

    Λ is block-diagonal, with one block per grouping factor.
    For grouping factor k with q_k terms and J_k groups:
        - Extract the q_k*(q_k+1)/2 elements of θ for this factor
        - Form the q_k × q_k lower-triangular Cholesky factor T_k
        - The block for this factor is T_k ⊗ I_J_k

    The Z matrix uses term-major ordering: columns are
    [term0_grp0, term0_grp1, ..., term1_grp0, term1_grp1, ...]
    so the corresponding b vector is also term-major.
    To match this layout, Λ must be T_k ⊗ I_J_k (not I_J_k ⊗ T_k).

    T ⊗ I_J produces blocks of T[r,c] × I_J:
        [[T[0,0]*I  0       ...]
         [T[1,0]*I  T[1,1]*I...]
         [...                   ]]

    The full Λ has shape (total_q, total_q) where total_q = Σ(J_k * q_k).

    Args:
        theta: Parameter vector of length Σ(q_k*(q_k+1)/2).
        specs: List of RandomEffectSpec objects.

    Returns:
        Block-diagonal Λ matrix.
    """
    total_q = sum(s.n_groups * s.n_terms for s in specs)
    Lambda = np.zeros((total_q, total_q), dtype=np.float64)

    theta_start = 0
    base_col = 0

    for spec in specs:
        q = spec.n_terms
        J = spec.n_groups
        n_theta = spec.theta_size  # q*(q+1)/2

        # Extract theta elements for this grouping factor
        theta_k = theta[theta_start:theta_start + n_theta]
        theta_start += n_theta

        # Form q × q lower-triangular Cholesky factor
        T = np.zeros((q, q), dtype=np.float64)
        idx = 0
        for row in range(q):
            for col in range(row + 1):
                T[row, col] = theta_k[idx]
                idx += 1

        # T ⊗ I_J: for each pair of terms (r, c), place T[r,c] × I_J
        # Term r's columns start at base_col + r*J
        # Term c's columns start at base_col + c*J
        for r in range(q):
            for c in range(r + 1):  # T is lower triangular
                if T[r, c] != 0.0:
                    row_start = base_col + r * J
                    col_start = base_col + c * J
                    for j in range(J):
                        Lambda[row_start + j, col_start + j] = T[r, c]

        base_col += J * q

    return Lambda


def theta_lower_bounds(specs: list[RandomEffectSpec]) -> NDArray:
    """Compute lower bounds for θ for the L-BFGS-B optimizer.

    Diagonal elements of the Cholesky factor must be ≥ 0 (variance is non-negative).
    Off-diagonal elements are unbounded (correlations can be negative).

    Args:
        specs: List of RandomEffectSpec objects.

    Returns:
        Array of lower bounds, same length as θ.
    """
    bounds = []
    for spec in specs:
        q = spec.n_terms
        for row in range(q):
            for col in range(row + 1):
                if row == col:
                    bounds.append(0.0)      # diagonal: variance ≥ 0
                else:
                    bounds.append(-np.inf)   # off-diagonal: unbounded
    return np.array(bounds, dtype=np.float64)


def is_singular_fit(
    theta: NDArray, specs: list[RandomEffectSpec], tol: float = 1e-4
) -> bool:
    """Detect a boundary (singular) random-effects fit, matching lme4.

    A mixed-model fit is *singular* when the estimated random-effects
    covariance is on the boundary of the feasible region: a variance has
    collapsed to (near) zero, or an implied correlation has reached ±1. In
    the θ (relative-covariance Cholesky) parameterisation both show up the
    same way — as a **diagonal** element of a Cholesky factor at (near) its
    lower bound of 0. A variance→0 collapses its own diagonal directly; a
    correlation→±1 collapses a *trailing* diagonal of the same block. So a
    single test over the diagonal θ elements detects both.

    This mirrors lme4's ``isSingular()`` exactly: flag if any θ element whose
    lower bound is 0 (i.e. a Cholesky-factor diagonal) is below ``tol``.

    Args:
        theta: The (converged) θ parameter vector.
        specs: Random effect specifications.
        tol: Boundary tolerance on the relative-covariance scale.
            Default 1e-4, matching lme4's ``isSingular`` default.

    Returns:
        True if the fit is singular (on the boundary), else False.
    """
    idx = 0
    for spec in specs:
        q = spec.n_terms
        for row in range(q):
            for col in range(row + 1):
                if row == col and theta[idx] < tol:
                    return True
                idx += 1
    return False


def theta_start(specs: list[RandomEffectSpec]) -> NDArray:
    """Generate starting values for θ.

    Diagonal elements start at 1.0 (σ_b/σ = 1, equal variance partition).
    Off-diagonal elements start at 0.0 (no initial correlation).

    Args:
        specs: List of RandomEffectSpec objects.

    Returns:
        Starting θ vector.
    """
    theta0 = []
    for spec in specs:
        q = spec.n_terms
        for row in range(q):
            for col in range(row + 1):
                if row == col:
                    theta0.append(1.0)
                else:
                    theta0.append(0.0)
    return np.array(theta0, dtype=np.float64)
