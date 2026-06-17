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

Categorical *predictors* must be fully observed — the sweep never imputes them
here (an incomplete categorical column needs a categorical method, refused
upstream), so their codes, and hence their dummy encoding, are constant across
chains and sweep steps.
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
