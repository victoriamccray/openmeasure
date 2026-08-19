"""
Portfolio context - compares one grantee's indicator value against the
rest of the portfolio, and flags cross-grantee comparability problems.

All functions are pure: they accept a pandas DataFrame and return frozen
dataclasses. No I/O, no UI logic.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.validation import validate_is_dataframe

POSITION_WITHIN_RANGE = "within_range"
POSITION_ABOVE_RANGE = "above_range"
POSITION_BELOW_RANGE = "below_range"


@dataclass(frozen=True)
class PortfolioIndicatorSummary:
    """
    Where one value falls relative to the rest of the portfolio for the
    same indicator.

    position is never "good" or "bad": whether higher or lower is
    preferable depends on the indicator, which this function does not
    know. Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.
    Fences are Q1 - 1.5*IQR and Q3 + 1.5*IQR.
    """

    indicator_id: str
    n_grantees_reporting: int
    median_value: float
    q1: float
    q3: float
    lower_fence: float
    upper_fence: float
    this_value: float
    position: str


@dataclass(frozen=True)
class PortfolioComparabilityFlag:
    """One reason an indicator's values may not be comparable across grantees."""

    indicator_id: str
    message: str
    severity: str


@dataclass(frozen=True)
class PortfolioContextResult:
    """Portfolio-level context for one claim's indicator, if any was loaded."""

    indicator_summary: PortfolioIndicatorSummary | None
    comparability_flags: tuple[PortfolioComparabilityFlag, ...]


def summarize_portfolio_indicator(
    portfolio: pd.DataFrame, indicator_id: str, this_value: float
) -> PortfolioIndicatorSummary:
    """Summarize where this_value falls among the portfolio's other reports of indicator_id."""

    validate_is_dataframe(portfolio)

    subset = portfolio.loc[portfolio["indicator_id"] == indicator_id, "result_value"]

    if subset.empty:
        raise ValueError(
            f"No rows for indicator_id '{indicator_id}' in the portfolio."
        )

    values = subset.to_numpy(dtype=float)
    q1, median_value, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    if this_value > upper_fence:
        position = POSITION_ABOVE_RANGE
    elif this_value < lower_fence:
        position = POSITION_BELOW_RANGE
    else:
        position = POSITION_WITHIN_RANGE

    return PortfolioIndicatorSummary(
        indicator_id=indicator_id,
        n_grantees_reporting=int(subset.shape[0]),
        median_value=float(median_value),
        q1=float(q1),
        q3=float(q3),
        lower_fence=float(lower_fence),
        upper_fence=float(upper_fence),
        this_value=float(this_value),
        position=position,
    )


def check_comparability(portfolio: pd.DataFrame) -> tuple[PortfolioComparabilityFlag, ...]:
    """Flag any indicator reported under more than one distinct unit across grantees."""

    validate_is_dataframe(portfolio)

    flags = []
    for indicator_id, group in portfolio.groupby("indicator_id"):
        units = sorted({str(u) for u in group["unit"].dropna().unique()})
        if len(units) > 1:
            flags.append(
                PortfolioComparabilityFlag(
                    indicator_id=str(indicator_id),
                    message=(
                        f"Indicator '{indicator_id}' is reported in "
                        f"{len(units)} different units across grantees "
                        f"({', '.join(units)}), so values are not directly "
                        "comparable."
                    ),
                    severity="advisory",
                )
            )

    return tuple(flags)
