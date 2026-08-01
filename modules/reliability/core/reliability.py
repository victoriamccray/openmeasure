"""
Core reliability statistics.

All functions are pure: they accept a pandas DataFrame of item scores
(rows = participants, columns = items) and return numbers or small
dataclasses. There is no I/O, UI logic, or external state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


ITEM_TOTAL_FLAG_THRESHOLD = 0.30


@dataclass(frozen=True)
class ItemDiagnostic:
    """Diagnostic statistics for one scale item."""

    item: str
    item_total_corr: float
    alpha_if_dropped: float
    flagged: bool


@dataclass(frozen=True)
class ReliabilityResult:
    """Complete output from the reliability analysis pipeline."""

    n_participants: int
    n_items: int
    n_complete_cases: int
    n_excluded_cases: int
    pct_excluded_cases: float
    pct_missing_cells: float
    cronbach_alpha: float
    split_half_correlation: float | None
    spearman_brown: float | None
    item_diagnostics: list[ItemDiagnostic] = field(default_factory=list)


def _validate_dataframe(
    data: pd.DataFrame,
    *,
    minimum_items: int = 2,
    minimum_rows: int = 2,
) -> None:
    """Validate the structure and types of an item-level DataFrame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be provided as a pandas DataFrame.")

    if data.shape[1] < minimum_items:
        raise ValueError(
            f"At least {minimum_items} item columns are required."
        )

    if data.shape[0] < minimum_rows:
        raise ValueError(
            f"At least {minimum_rows} participant rows are required."
        )

    # Duplicate column names must be checked BEFORE the numeric-dtype check.
    # Selecting a duplicate-named column (data[column]) returns a DataFrame
    # instead of a Series, which pd.api.types.is_numeric_dtype treats as
    # non-numeric. That would surface a confusing TypeError about "invalid
    # columns" instead of the real, more specific problem: duplicate names.
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        duplicate_names = ", ".join(map(str, duplicates))
        raise ValueError(
            f"Duplicate item column names are not allowed: {duplicate_names}"
        )

    non_numeric = [
        str(column)
        for column in data.columns
        if not pd.api.types.is_numeric_dtype(data[column])
    ]

    if non_numeric:
        invalid = ", ".join(non_numeric)
        raise TypeError(
            "All selected item columns must be numeric. "
            f"Invalid columns: {invalid}"
        )

    values = data.to_numpy(dtype=float, na_value=np.nan)

    if np.isinf(values).any():
        raise ValueError(
            "Item scores cannot contain positive or negative infinity."
        )


def _listwise_delete(data: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with a missing value in any selected item."""
    return data.dropna(axis=0, how="any").copy()


def cronbach_alpha(data: pd.DataFrame) -> float:
    """
    Compute Cronbach's alpha from complete item-level data.

    Formula:
        alpha = (k / (k - 1)) *
                (1 - sum(item variances) / variance(total score))
    """
    _validate_dataframe(data)

    item_variances = data.var(axis=0, ddof=1)
    total_scores = data.sum(axis=1)
    total_variance = float(total_scores.var(ddof=1))

    if np.isnan(total_variance) or total_variance == 0:
        raise ValueError(
            "Total-score variance is zero or undefined. "
            "Cronbach's alpha cannot be calculated."
        )

    number_of_items = data.shape[1]

    alpha = (
        number_of_items
        / (number_of_items - 1)
        * (1 - float(item_variances.sum()) / total_variance)
    )

    return float(alpha)


def item_total_correlations(data: pd.DataFrame) -> pd.Series:
    """
    Compute corrected item-total correlations.

    Each item is correlated with the sum of all other items, excluding
    itself from the total score.
    """
    _validate_dataframe(data)

    correlations: dict[str, float] = {}

    for column in data.columns:
        item = data[column]
        remaining_score = data.drop(columns=[column]).sum(axis=1)

        item_variance = float(item.var(ddof=1))
        remaining_variance = float(remaining_score.var(ddof=1))

        if (
            np.isnan(item_variance)
            or np.isnan(remaining_variance)
            or item_variance == 0
            or remaining_variance == 0
        ):
            correlations[str(column)] = np.nan
            continue

        correlations[str(column)] = float(
            item.corr(remaining_score)
        )

    return pd.Series(
        correlations,
        name="corrected_item_total_correlation",
        dtype=float,
    )


def alpha_if_item_dropped(data: pd.DataFrame) -> pd.Series:
    """Recalculate Cronbach's alpha after removing each item."""
    _validate_dataframe(data)

    results: dict[str, float] = {}

    for column in data.columns:
        remaining = data.drop(columns=[column])

        if remaining.shape[1] < 2:
            results[str(column)] = np.nan
            continue

        try:
            results[str(column)] = cronbach_alpha(remaining)
        except ValueError:
            results[str(column)] = np.nan

    return pd.Series(
        results,
        name="alpha_if_item_dropped",
        dtype=float,
    )


def split_half_reliability(
    data: pd.DataFrame,
) -> tuple[float, float]:
    """
    Compute odd-even split-half reliability.

    Items in positions 1, 3, 5, ... form the odd half, and items in
    positions 2, 4, 6, ... form the even half.

    The function requires at least four items and an even number of items
    so that both halves contain the same number of items.

    Returns:
        A tuple containing:
        - Raw correlation between the two half scores
        - Spearman-Brown corrected reliability
    """
    _validate_dataframe(data, minimum_items=4)

    number_of_items = data.shape[1]

    if number_of_items % 2 != 0:
        raise ValueError(
            "Odd-even split-half reliability currently requires an "
            "even number of items."
        )

    odd_items = data.columns[::2]
    even_items = data.columns[1::2]

    odd_scores = data.loc[:, odd_items].sum(axis=1)
    even_scores = data.loc[:, even_items].sum(axis=1)

    odd_variance = float(odd_scores.var(ddof=1))
    even_variance = float(even_scores.var(ddof=1))

    if (
        np.isnan(odd_variance)
        or np.isnan(even_variance)
        or odd_variance == 0
        or even_variance == 0
    ):
        raise ValueError(
            "One or both split-half scores have zero or undefined variance."
        )

    correlation = float(odd_scores.corr(even_scores))

    if np.isnan(correlation):
        raise ValueError(
            "The split-half correlation could not be calculated."
        )

    if np.isclose(correlation, -1.0):
        raise ValueError(
            "The Spearman-Brown correction is undefined when "
            "the split-half correlation is -1."
        )

    corrected = (2 * correlation) / (1 + correlation)

    return correlation, float(corrected)


def analyze(data: pd.DataFrame) -> ReliabilityResult:
    """
    Run the complete reliability analysis.

    Rows containing a missing value in any selected item are removed using
    listwise deletion. The number and percentage of excluded participants
    and missing cells are included in the result.
    """
    _validate_dataframe(
        data,
        minimum_items=2,
        minimum_rows=1,
    )

    number_of_participants = data.shape[0]
    number_of_items = data.shape[1]

    complete = _listwise_delete(data)

    number_of_complete_cases = complete.shape[0]
    number_of_excluded_cases = (
        number_of_participants - number_of_complete_cases
    )

    percent_excluded_cases = (
        100 * number_of_excluded_cases / number_of_participants
        if number_of_participants > 0
        else 0.0
    )

    total_cells = number_of_participants * number_of_items
    missing_cells = int(data.isna().sum().sum())

    percent_missing_cells = (
        100 * missing_cells / total_cells
        if total_cells > 0
        else 0.0
    )

    if number_of_complete_cases < 2:
        raise ValueError(
            f"Only {number_of_complete_cases} complete case(s) remain "
            "after listwise deletion. At least 2 are required."
        )

    alpha = cronbach_alpha(complete)

    try:
        split_correlation, spearman_brown = split_half_reliability(
            complete
        )
    except ValueError:
        split_correlation = None
        spearman_brown = None

    item_correlations = item_total_correlations(complete)
    dropped_alphas = alpha_if_item_dropped(complete)

    diagnostics: list[ItemDiagnostic] = []

    for column in complete.columns:
        item_name = str(column)
        correlation = float(item_correlations[item_name])
        dropped_alpha = float(dropped_alphas[item_name])

        flagged = (
            np.isnan(correlation)
            or correlation < ITEM_TOTAL_FLAG_THRESHOLD
        )

        diagnostics.append(
            ItemDiagnostic(
                item=item_name,
                item_total_corr=correlation,
                alpha_if_dropped=dropped_alpha,
                flagged=bool(flagged),
            )
        )

    return ReliabilityResult(
        n_participants=number_of_participants,
        n_items=number_of_items,
        n_complete_cases=number_of_complete_cases,
        n_excluded_cases=number_of_excluded_cases,
        pct_excluded_cases=float(percent_excluded_cases),
        pct_missing_cells=float(percent_missing_cells),
        cronbach_alpha=alpha,
        split_half_correlation=split_correlation,
        spearman_brown=spearman_brown,
        item_diagnostics=diagnostics,
    )
