"""Shared result dataclass for nonparametric MCAR tests."""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class NonparametricMCARResult:
    """Result of a distribution-free MCAR test.

    Attributes
    ----------
    statistic : float
        Test statistic. Interpretation depends on ``method``:
          - propensity: mean per-column out-of-fold AUC minus 0.5,
            clipped to [0, 0.5]. 0 = indistinguishable from chance
            (consistent with MCAR); 0.5 = perfect separation.
          - HSIC: Hilbert-Schmidt independence criterion estimate.
          - MissMech: Hawkins test statistic.
    p_value : float
        P-value against the null hypothesis of MCAR.
    rejected : bool
        True iff ``p_value < alpha``.
    alpha : float
        Significance level at which ``rejected`` was evaluated.
    method : str
        Human-readable method identifier. Must contain enough
        information to disambiguate among nonparametric tests and
        (for propensity / MissMech) their key hyperparameters, so
        downstream code or a log reader can tell them apart.
    n_observations : int
        Rows in the input matrix.
    n_variables : int
        Columns in the input matrix.
    n_missing_cells : int
        Cells with ``np.nan`` in the input matrix.
    extra : dict
        Per-method auxiliary output (e.g. propensity's per-column AUCs,
        HSIC's estimated bandwidth, MissMech's number of patterns used).
        Field names inside ``extra`` are method-specific and documented
        by each test function's docstring.
    """

    statistic: float
    p_value: float
    rejected: bool
    alpha: float
    method: str
    n_observations: int
    n_variables: int
    n_missing_cells: int
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary for logs and notebooks."""
        decision = "Reject MCAR" if self.rejected else "Fail to reject MCAR"
        lines = [
            "Nonparametric MCAR Test",
            "=" * 40,
            f"Method: {self.method}",
            f"Statistic: {self.statistic:.4f}",
            f"P-value: {self.p_value:.4f}",
            f"Decision at alpha={self.alpha}: {decision}",
            f"n_observations: {self.n_observations}, n_variables: {self.n_variables}",
            f"n_missing_cells: {self.n_missing_cells}",
        ]
        return "\n".join(lines)
