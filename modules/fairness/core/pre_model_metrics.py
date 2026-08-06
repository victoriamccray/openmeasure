"""
Pre-model fairness metrics.

These functions examine whether favorable observed labels occur at
different rates across groups before a model is trained.

The module currently supports:

- favorable-label rates by group;
- disparate impact;
- statistical parity difference; and
- small-group warnings.

These metrics describe differences in observed labels. They do not, by
themselves, establish discrimination, unfairness, or the cause of a
disparity.
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


MIN_GROUP_SIZE = 5

# Why rows were dropped, recorded on the result per
# docs/design-standards.md section 2.
MISSING_LABEL_OR_GROUP = "missing label or group value"


@dataclass(frozen=True)
class GroupRate:
    """Observed favorable-label rate for one group."""

    group: str
    n: int
    favorable_count: int
    unfavorable_count: int
    favorable_rate: float
    small_sample: bool


@dataclass(frozen=True)
class PreModelBiasResult:
    """
    Pairwise comparison of favorable-label rates.

    Disparate impact is calculated as:

        unprivileged favorable rate / privileged favorable rate

    Statistical parity difference is calculated as:

        unprivileged favorable rate - privileged favorable rate
    """

    privileged_group: str
    unprivileged_group: str
    privileged_n: int
    unprivileged_n: int
    privileged_favorable_count: int
    unprivileged_favorable_count: int
    privileged_rate: float
    unprivileged_rate: float
    disparate_impact: float
    statistical_parity_difference: float

    # Exclusion accounting, per docs/design-standards.md section 2. Rows
    # missing either the label or the group cannot be placed in a group, so
    # they are dropped before any rate is computed. Note that the two
    # per-group counts above cover only the selected pair, while these
    # describe the whole upload.
    n_input_rows: int
    n_rows_used: int
    n_excluded_rows: int
    exclusion_reason: str


def _validate_dataframe(data: pd.DataFrame) -> None:
    """Validate the input dataset."""

    validate_is_dataframe(data)

    if data.empty:
        raise ValueError("The dataset contains no rows.")


def _validate_columns(
    data: pd.DataFrame,
    label_col: str,
    group_col: str,
) -> None:
    """Validate the required label and group columns."""

    _validate_dataframe(data)

    if label_col not in data.columns:
        raise ValueError(
            f"Label column '{label_col}' was not found in the data."
        )

    if group_col not in data.columns:
        raise ValueError(
            f"Group column '{group_col}' was not found in the data."
        )

    if label_col == group_col:
        raise ValueError(
            "The label and group columns must be different columns."
        )


def _prepare_data(
    data: pd.DataFrame,
    label_col: str,
    group_col: str,
) -> pd.DataFrame:
    """
    Select required columns and remove incomplete observations.

    Rows missing either the label or group value are excluded.
    """

    _validate_columns(
        data,
        label_col,
        group_col,
    )

    clean = data[
        [label_col, group_col]
    ].dropna()

    if clean.empty:
        raise ValueError(
            "No complete observations remain after removing missing "
            "label or group values."
        )

    return clean


def _validate_binary_label(
    clean: pd.DataFrame,
    label_col: str,
    favorable_label: object,
) -> list[object]:
    """Validate that the observed outcome is binary."""

    label_values = clean[
        label_col
    ].unique().tolist()

    if len(label_values) != 2:
        raise ValueError(
            f"Label column '{label_col}' must contain exactly two "
            f"nonmissing values; found {len(label_values)}: "
            f"{label_values}."
        )

    if favorable_label not in label_values:
        raise ValueError(
            f"Favorable label '{favorable_label}' was not found in "
            f"column '{label_col}'. Available values: {label_values}."
        )

    return label_values


def _validate_group_exists(
    clean: pd.DataFrame,
    group_col: str,
    group_value: object,
    *,
    role: str,
) -> None:
    """Validate that a requested group exists."""

    available_groups = clean[
        group_col
    ].unique().tolist()

    if group_value not in available_groups:
        raise ValueError(
            f"{role.capitalize()} group '{group_value}' was not found "
            f"in column '{group_col}'. Available groups: "
            f"{available_groups}."
        )


def _favorable_rate(
    group_data: pd.DataFrame,
    label_col: str,
    favorable_label: object,
) -> tuple[int, int, float]:
    """Return sample size, favorable count, and favorable rate."""

    n = int(len(group_data))

    if n == 0:
        raise ValueError(
            "A group contains no complete observations."
        )

    favorable_count = int(
        (group_data[label_col] == favorable_label).sum()
    )

    favorable_rate = favorable_count / n

    return n, favorable_count, float(favorable_rate)


def compute_pre_model_bias(
    data: pd.DataFrame,
    label_col: str,
    group_col: str,
    *,
    favorable_label: object,
    privileged_group: object,
    unprivileged_group: object,
) -> PreModelBiasResult:
    """
    Compare favorable-label rates between two selected groups.

    Parameters
    ----------
    data:
        Input dataset.

    label_col:
        Binary observed outcome column.

    group_col:
        Group membership column.

    favorable_label:
        Label value interpreted as favorable.

    privileged_group:
        Reference group used in the denominator of disparate impact.

    unprivileged_group:
        Comparison group.

    Returns
    -------
    PreModelBiasResult
        Favorable rates, disparate impact, and statistical parity
        difference.

    Notes
    -----
    Disparate impact:

        P(favorable | unprivileged)
        --------------------------------
        P(favorable | privileged)

    Statistical parity difference:

        P(favorable | unprivileged)
        - P(favorable | privileged)
    """

    clean = _prepare_data(
        data,
        label_col,
        group_col,
    )

    _validate_binary_label(
        clean,
        label_col,
        favorable_label,
    )

    if privileged_group == unprivileged_group:
        raise ValueError(
            "The privileged and unprivileged groups must be different."
        )

    _validate_group_exists(
        clean,
        group_col,
        privileged_group,
        role="privileged",
    )

    _validate_group_exists(
        clean,
        group_col,
        unprivileged_group,
        role="unprivileged",
    )

    privileged_data = clean.loc[
        clean[group_col] == privileged_group
    ]

    unprivileged_data = clean.loc[
        clean[group_col] == unprivileged_group
    ]

    (
        privileged_n,
        privileged_favorable_count,
        privileged_rate,
    ) = _favorable_rate(
        privileged_data,
        label_col,
        favorable_label,
    )

    (
        unprivileged_n,
        unprivileged_favorable_count,
        unprivileged_rate,
    ) = _favorable_rate(
        unprivileged_data,
        label_col,
        favorable_label,
    )

    if privileged_rate == 0:
        raise ValueError(
            "The privileged-group favorable rate is zero, so disparate "
            "impact is undefined."
        )

    disparate_impact = (
        unprivileged_rate
        / privileged_rate
    )

    statistical_parity_difference = (
        unprivileged_rate
        - privileged_rate
    )

    return PreModelBiasResult(
        privileged_group=str(privileged_group),
        unprivileged_group=str(unprivileged_group),
        privileged_n=privileged_n,
        unprivileged_n=unprivileged_n,
        privileged_favorable_count=(
            privileged_favorable_count
        ),
        unprivileged_favorable_count=(
            unprivileged_favorable_count
        ),
        privileged_rate=float(
            privileged_rate
        ),
        unprivileged_rate=float(
            unprivileged_rate
        ),
        disparate_impact=float(
            disparate_impact
        ),
        statistical_parity_difference=float(
            statistical_parity_difference
        ),
        n_input_rows=int(len(data)),
        n_rows_used=int(len(clean)),
        n_excluded_rows=int(len(data) - len(clean)),
        exclusion_reason=MISSING_LABEL_OR_GROUP,
    )


def compute_group_rates(
    data: pd.DataFrame,
    label_col: str,
    group_col: str,
    *,
    favorable_label: object,
    minimum_group_size: int = MIN_GROUP_SIZE,
) -> list[GroupRate]:
    """
    Compute favorable-label rates for every observed group.

    Groups are returned in alphabetical order for stable reporting.

    Parameters
    ----------
    minimum_group_size:
        Groups with fewer observations than this threshold are flagged
        as small samples. They are still included in the results.
    """

    if minimum_group_size < 1:
        raise ValueError(
            "minimum_group_size must be at least 1."
        )

    clean = _prepare_data(
        data,
        label_col,
        group_col,
    )

    _validate_binary_label(
        clean,
        label_col,
        favorable_label,
    )

    group_values = sorted(
        clean[group_col].unique().tolist(),
        key=str,
    )

    results: list[GroupRate] = []

    for group_value in group_values:
        group_data = clean.loc[
            clean[group_col] == group_value
        ]

        (
            n,
            favorable_count,
            favorable_rate,
        ) = _favorable_rate(
            group_data,
            label_col,
            favorable_label,
        )

        unfavorable_count = (
            n
            - favorable_count
        )

        results.append(
            GroupRate(
                group=str(group_value),
                n=n,
                favorable_count=(
                    favorable_count
                ),
                unfavorable_count=(
                    unfavorable_count
                ),
                favorable_rate=float(
                    favorable_rate
                ),
                small_sample=(
                    n < minimum_group_size
                ),
            )
        )

    return results
