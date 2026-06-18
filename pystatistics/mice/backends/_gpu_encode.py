"""
Batched predictor-matrix construction for the GPU MICE sweep.

GPU counterpart of ``mice/_encode.py``: builds the predictor matrix for imputing
column ``j`` from every other column, treatment-dummy-encoding categorical
predictors (drop-first contrasts, matching ``_encode._treatment_dummies``), with
the chain index as the leading batch dimension.

Numeric-only problems keep the fast path (a plain column slice). When any
predictor is categorical, columns are concatenated block by block: a numeric
predictor contributes one column; a K-level categorical predictor contributes
``K-1`` 0/1 dummy columns (the first level is the reference). No intercept column
(the regression methods add one).

A categorical predictor's dummy encoding is exact because its values are always
valid category codes — either fully observed, or (for an incomplete categorical
column imputed earlier in the sweep) imputed back as one of its level codes. So
equality against the level set is exact regardless of which chain or sweep step.

This module also owns the batched code<->index mapping for categorical *targets*:
``codes_to_indices`` / ``indices_to_codes`` translate between a column's stored
category codes and the consecutive ``0..K-1`` indices the categorical GPU methods
speak in (the tensor counterpart of ``_encode.codes_to_indices``).
"""

from __future__ import annotations


def build_predictor_tensor(data, j, p, cat_levels):
    """Predictor tensor for target column ``j`` (chain index leading).

    Parameters
    ----------
    data : (m, n, p) tensor
        The working data, categorical columns held as float category codes.
    j : int
        Target column being imputed (excluded from predictors).
    p : int
        Number of columns.
    cat_levels : dict[int, tensor]
        ``col -> sorted unique level codes (K,)`` for each categorical column;
        numeric columns are absent. Empty dict ⇒ numeric-only fast path.

    Returns
    -------
    (m, n, q') tensor
        Predictors, with categorical columns treatment-dummy-encoded. ``q'`` is
        ``#numeric_predictors + sum(K_c - 1)`` over categorical predictors.
    """
    import torch

    # Fast path: no categorical columns anywhere — a plain slice (current behaviour).
    if not cat_levels:
        cols = [c for c in range(p) if c != j]
        return data[:, :, cols]

    blocks = []
    for c in range(p):
        if c == j:
            continue
        levels = cat_levels.get(c)
        if levels is None:
            blocks.append(data[:, :, c:c + 1])                       # numeric: (m, n, 1)
        else:
            # treatment dummies: 0/1 columns for levels[1:], reference dropped.
            block = (data[:, :, c:c + 1] == levels[1:]).to(data.dtype)  # (m, n, K-1)
            blocks.append(block)
    return torch.cat(blocks, dim=2)


def codes_to_indices(values, levels):
    """Map category codes to consecutive ``0..K-1`` indices (tensor counterpart of
    ``_encode.codes_to_indices``).

    ``values`` holds exact category codes (drawn from observed values or imputed
    back as codes), so equality against the sorted ``levels`` (K,) is exact. The
    one-hot ``argmax`` gives the index; returned in ``values`` dtype so it flows
    straight into the methods' float arithmetic.
    """
    import torch

    onehot = values.unsqueeze(-1) == levels                  # (..., K)
    return onehot.to(torch.long).argmax(dim=-1).to(values.dtype)


def indices_to_codes(indices, levels):
    """Map ``0..K-1`` class indices back to stored category codes (gather)."""
    return levels[indices.long()]
