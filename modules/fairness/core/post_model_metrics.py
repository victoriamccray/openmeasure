"""
Post-model fairness metrics.

These functions examine whether a model's predictions behave differently
across groups, given the observed (ground-truth) outcome. Unlike
pre_model_metrics, they require a predicted label in addition to the
observed outcome and the group column.

The module currently supports:

- per-group true-positive rate, false-positive rate, and positive
  predictive value;
- equal opportunity (true-positive-rate parity);
- predictive equality (false-positive-rate parity);
- equalized odds (both of the above); and
- calibration within groups, approximated here by positive-predictive-value
  parity, a binary proxy rather than a full calibration curve.

Hardt, Price, & Srebro (2016) define equal opportunity and equalized odds
in terms of true-positive and false-positive rates. Chouldechova (2017)
and Kleinberg, Mullainathan, & Raghavan (2017) show that when a model is
imperfect and outcome base rates differ across groups, equalized odds and
calibration cannot both hold exactly. This module reports each metric's
gap separately rather than combining them, so that tension is visible
rather than hidden behind a single score.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.validation import format_value_sample, validate_is_dataframe


MIN_GROUP_SIZE = 5

# Why rows were dropped, recorded on the result per
# docs/design-standards.md section 2.
MISSING_LABEL_PREDICTION_OR_GROUP = (
    "missing true label, predicted label, or group value"
)


@dataclass(frozen=True)
class GroupConfusion:
    """
    Confusion-matrix counts and rates for one group.

    true_positive_rate is undefined (None) when the group has no actual
    positives; false_positive_rate is undefined when it has no actual
    negatives; positive_predictive_value is undefined when the model made
    no positive predictions in the group. All three are real, not rare,
    outcomes in a small or imbalanced group, so they are reported as
    "not applicable" rather than a fabricated 0.0.
    """

    group: str
    n: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    true_positive_rate: float | None
    false_positive_rate: float | None
    positive_predictive_value: float | None
    small_sample: bool


@dataclass(frozen=True)
class PostModelBiasResult:
    """
    Pairwise comparison of a privileged and unprivileged group's
    prediction behavior, given the observed outcome.

    Equal opportunity difference:

        unprivileged true-positive rate - privileged true-positive rate

    Predictive equality difference:

        unprivileged false-positive rate - privileged false-positive rate

    Equalized odds holds only when both differences above are close to
    zero; this module does not combine them into one number, because
    two gaps of opposite sign are not the same as no gap at all.

    Calibration-within-groups difference (a binary proxy, using positive
    predictive value rather than a full calibration curve over predicted
    probabilities):

        unprivileged positive predictive value
        - privileged positive predictive value
    """

    privileged_group: str
    unprivileged_group: str
    privileged_n: int
    unprivileged_n: int

    privileged_true_positive_rate: float
    unprivileged_true_positive_rate: float
    equal_opportunity_difference: float

    privileged_false_positive_rate: float
    unprivileged_false_positive_rate: float
    predictive_equality_difference: float

    privileged_positive_predictive_value: float
    unprivileged_positive_predictive_value: float
    calibration_within_groups_difference: float

    # Exclusion accounting, per docs/design-standards.md section 2.
    n_input_rows: int
    n_rows_used: int
    n_excluded_rows: int
    exclusion_reason: str


def _validate_dataframe(data: pd.DataFrame) -> None:
    validate_is_dataframe(data)

    if data.empty:
        raise ValueError("The dataset contains no rows.")


def _validate_columns(
    data: pd.DataFrame,
    true_label_col: str,
    predicted_label_col: str,
    group_col: str,
) -> None:
    _validate_dataframe(data)

    columns = {"true label": true_label_col, "predicted label": predicted_label_col, "group": group_col}

    for role, column in columns.items():
        if column not in data.columns:
            raise ValueError(f"The {role} column '{column}' was not found in the data.")

    if len({true_label_col, predicted_label_col, group_col}) != 3:
        raise ValueError(
            "The true label, predicted label, and group columns must be "
            "three different columns."
        )


def _prepare_data(
    data: pd.DataFrame,
    true_label_col: str,
    predicted_label_col: str,
    group_col: str,
) -> pd.DataFrame:
    _validate_columns(data, true_label_col, predicted_label_col, group_col)

    clean = data[[true_label_col, predicted_label_col, group_col]].dropna()

    if clean.empty:
        raise ValueError(
            "No complete observations remain after removing rows missing "
            "a true label, predicted label, or group value."
        )

    return clean


def _validate_binary_column(clean: pd.DataFrame, column: str, positive_label: object, *, role: str) -> None:
    values = clean[column].unique().tolist()

    if len(values) != 2:
        raise ValueError(
            f"The {role} column '{column}' must contain exactly two "
            f"nonmissing values; found {len(values)}: "
            f"{format_value_sample(values)}."
        )

    if positive_label not in values:
        raise ValueError(
            f"Positive label '{positive_label}' was not found in the "
            f"{role} column '{column}'. Available values: "
            f"{format_value_sample(values)}."
        )


def _validate_group_exists(clean: pd.DataFrame, group_col: str, group_value: object, *, role: str) -> None:
    available_groups = clean[group_col].unique().tolist()

    if group_value not in available_groups:
        raise ValueError(
            f"{role.capitalize()} group '{group_value}' was not found in "
            f"column '{group_col}'. Available groups: "
            f"{format_value_sample(available_groups)}."
        )


def _confusion_rates(
    group_data: pd.DataFrame,
    true_label_col: str,
    predicted_label_col: str,
    positive_label: object,
) -> tuple[int, int, int, int, float | None, float | None, float | None]:
    n = int(len(group_data))

    if n == 0:
        raise ValueError("A group contains no complete observations.")

    actual_positive = group_data[true_label_col] == positive_label
    predicted_positive = group_data[predicted_label_col] == positive_label

    tp = int((actual_positive & predicted_positive).sum())
    fp = int((~actual_positive & predicted_positive).sum())
    tn = int((~actual_positive & ~predicted_positive).sum())
    fn = int((actual_positive & ~predicted_positive).sum())

    n_actual_positive = tp + fn
    n_actual_negative = tn + fp
    n_predicted_positive = tp + fp

    tpr = (tp / n_actual_positive) if n_actual_positive > 0 else None
    fpr = (fp / n_actual_negative) if n_actual_negative > 0 else None
    ppv = (tp / n_predicted_positive) if n_predicted_positive > 0 else None

    return tp, fp, tn, fn, tpr, fpr, ppv


def compute_group_confusion_rates(
    data: pd.DataFrame,
    true_label_col: str,
    predicted_label_col: str,
    group_col: str,
    *,
    positive_label: object,
    minimum_group_size: int = MIN_GROUP_SIZE,
) -> list[GroupConfusion]:
    """
    Compute true-positive rate, false-positive rate, and positive
    predictive value for every observed group.

    Groups are returned in alphabetical order for stable reporting.
    """

    if minimum_group_size < 1:
        raise ValueError("minimum_group_size must be at least 1.")

    clean = _prepare_data(data, true_label_col, predicted_label_col, group_col)
    _validate_binary_column(clean, true_label_col, positive_label, role="true label")
    _validate_binary_column(clean, predicted_label_col, positive_label, role="predicted label")

    group_values = sorted(clean[group_col].unique().tolist(), key=str)

    results: list[GroupConfusion] = []

    for group_value in group_values:
        group_data = clean.loc[clean[group_col] == group_value]
        n = int(len(group_data))

        tp, fp, tn, fn, tpr, fpr, ppv = _confusion_rates(
            group_data, true_label_col, predicted_label_col, positive_label
        )

        results.append(
            GroupConfusion(
                group=str(group_value),
                n=n,
                true_positive=tp,
                false_positive=fp,
                true_negative=tn,
                false_negative=fn,
                true_positive_rate=tpr,
                false_positive_rate=fpr,
                positive_predictive_value=ppv,
                small_sample=n < minimum_group_size,
            )
        )

    return results


def compare_post_model_bias(
    data: pd.DataFrame,
    true_label_col: str,
    predicted_label_col: str,
    group_col: str,
    *,
    positive_label: object,
    privileged_group: object,
    unprivileged_group: object,
) -> PostModelBiasResult:
    """
    Compare true-positive rate, false-positive rate, and positive
    predictive value between two selected groups.

    Raises a ValueError if any of the three rates is undefined for
    either group (for example, a group with no actual positives has an
    undefined true-positive rate), rather than silently substituting a
    placeholder value.
    """

    clean = _prepare_data(data, true_label_col, predicted_label_col, group_col)
    _validate_binary_column(clean, true_label_col, positive_label, role="true label")
    _validate_binary_column(clean, predicted_label_col, positive_label, role="predicted label")

    if privileged_group == unprivileged_group:
        raise ValueError("The privileged and unprivileged groups must be different.")

    _validate_group_exists(clean, group_col, privileged_group, role="privileged")
    _validate_group_exists(clean, group_col, unprivileged_group, role="unprivileged")

    privileged_data = clean.loc[clean[group_col] == privileged_group]
    unprivileged_data = clean.loc[clean[group_col] == unprivileged_group]

    privileged_n = int(len(privileged_data))
    unprivileged_n = int(len(unprivileged_data))

    _, _, _, _, priv_tpr, priv_fpr, priv_ppv = _confusion_rates(
        privileged_data, true_label_col, predicted_label_col, positive_label
    )
    _, _, _, _, unpriv_tpr, unpriv_fpr, unpriv_ppv = _confusion_rates(
        unprivileged_data, true_label_col, predicted_label_col, positive_label
    )

    for label, value in (
        (f"'{privileged_group}' true-positive rate", priv_tpr),
        (f"'{unprivileged_group}' true-positive rate", unpriv_tpr),
        (f"'{privileged_group}' false-positive rate", priv_fpr),
        (f"'{unprivileged_group}' false-positive rate", unpriv_fpr),
        (f"'{privileged_group}' positive predictive value", priv_ppv),
        (f"'{unprivileged_group}' positive predictive value", unpriv_ppv),
    ):
        if value is None:
            raise ValueError(
                f"The {label} is undefined for the selected groups and "
                "columns (no observations in the required category), so "
                "a comparison cannot be computed."
            )

    return PostModelBiasResult(
        privileged_group=str(privileged_group),
        unprivileged_group=str(unprivileged_group),
        privileged_n=privileged_n,
        unprivileged_n=unprivileged_n,
        privileged_true_positive_rate=float(priv_tpr),
        unprivileged_true_positive_rate=float(unpriv_tpr),
        equal_opportunity_difference=float(unpriv_tpr - priv_tpr),
        privileged_false_positive_rate=float(priv_fpr),
        unprivileged_false_positive_rate=float(unpriv_fpr),
        predictive_equality_difference=float(unpriv_fpr - priv_fpr),
        privileged_positive_predictive_value=float(priv_ppv),
        unprivileged_positive_predictive_value=float(unpriv_ppv),
        calibration_within_groups_difference=float(unpriv_ppv - priv_ppv),
        n_input_rows=int(len(data)),
        n_rows_used=int(len(clean)),
        n_excluded_rows=int(len(data) - len(clean)),
        exclusion_reason=MISSING_LABEL_PREDICTION_OR_GROUP,
    )
