"""
Acquisition-robustness core for the HealthRing worked example.

These functions take an already-loaded, subject-tagged DataFrame of ring
heart-rate windows (one row per measurement window) and compute the
statistics behind the Explore -> Baseline -> Split -> Model -> Evaluate ->
Inspect retention -> Stress-test steps of the worked example. Reading the
HealthRing archive itself is page-level I/O, done in
pages/HealthRing_Worked_Example.py; nothing here opens a file.

Expected columns on the input DataFrame:

- subject_id: a subject identifier, assigned by the page from the source
  filename (the raw HealthRing pickle has no such column).
- Label: the recorded activity/acquisition condition.
- hr: reference heart rate (bpm).
- bvp_hr: ring-derived heart rate estimate (bpm).
- ir-quality, red-quality: the ring's own per-channel signal-quality
  estimate for the window.

These functions describe measurement agreement and its sensitivity to
acquisition condition and signal quality. They do not determine whether
that level of agreement is acceptable for any particular use.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.validation import validate_is_dataframe

REQUIRED_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "Label",
    "hr",
    "bvp_hr",
    "ir-quality",
    "red-quality",
)

# Why rows were dropped, recorded on the result per
# docs/design-standards.md section 2.
MISSING_HR_BVP_OR_LABEL = "missing hr, bvp_hr, or Label value"

# Widely used agreement convention (Bland & Altman, 1986): the limits of
# agreement span the mean difference plus or minus 1.96 sample standard
# deviations, covering ~95% of differences under a normal-difference
# assumption.
LOA_MULTIPLIER = 1.96


def _validate_columns(data: pd.DataFrame) -> None:
    """Validate that every required column is present."""

    validate_is_dataframe(data)

    if data.empty:
        raise ValueError("The dataset contains no rows.")

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]

    if missing:
        raise ValueError(
            f"Expected columns missing from the data: {missing}. Required "
            f"columns: {list(REQUIRED_COLUMNS)}."
        )


@dataclass(frozen=True)
class WindowFrame:
    """
    Usable measurement windows, with error and quality already computed.

    data carries every REQUIRED_COLUMNS column plus abs_error, signed_error,
    mean_hr, and quality (mean of the two per-channel quality estimates).
    """

    data: pd.DataFrame
    n_input_windows: int
    n_usable_windows: int
    condition_order: tuple[str, ...]

    @property
    def n_excluded_windows(self) -> int:
        return self.n_input_windows - self.n_usable_windows


def prepare_windows(data: pd.DataFrame) -> WindowFrame:
    """
    Drop windows missing hr, bvp_hr, or Label, and derive the error and
    quality columns every later step relies on.
    """

    _validate_columns(data)

    usable = data.dropna(subset=["hr", "bvp_hr", "Label"]).copy()

    if usable.empty:
        raise ValueError(
            "No usable windows remain after dropping rows missing hr, "
            "bvp_hr, or Label."
        )

    usable["abs_error"] = (usable["bvp_hr"] - usable["hr"]).abs()
    usable["signed_error"] = usable["bvp_hr"] - usable["hr"]
    usable["mean_hr"] = (usable["bvp_hr"] + usable["hr"]) / 2
    usable["quality"] = usable[["ir-quality", "red-quality"]].mean(axis=1)

    condition_order = tuple(sorted(usable["Label"].unique().tolist(), key=str))

    return WindowFrame(
        data=usable,
        n_input_windows=int(len(data)),
        n_usable_windows=int(len(usable)),
        condition_order=condition_order,
    )


@dataclass(frozen=True)
class AgreementResult:
    """
    Measurement agreement between a predicted and a reference series.

    Bias and the limits of agreement follow Bland & Altman (1986): bias is
    the mean signed difference (predicted - reference), and the limits of
    agreement are bias +/- 1.96 sample standard deviations of that
    difference.
    """

    n: int
    mae: float
    bias: float
    sd: float
    lower_loa: float
    upper_loa: float


def agreement_summary(predicted: pd.Series, reference: pd.Series) -> AgreementResult:
    """Summarize agreement between a predicted and a reference series."""

    if len(predicted) != len(reference):
        raise ValueError(
            f"predicted has {len(predicted)} values but reference has "
            f"{len(reference)}; they must describe the same windows."
        )

    n = int(len(predicted))

    if n == 0:
        raise ValueError("Cannot summarize agreement over zero windows.")

    predicted_values = np.asarray(predicted, dtype=float)
    reference_values = np.asarray(reference, dtype=float)

    diffs = predicted_values - reference_values
    bias = float(np.mean(diffs))

    if n < 2:
        raise ValueError(
            "At least two windows are required to estimate a standard "
            "deviation of agreement."
        )

    sd = float(np.std(diffs, ddof=1))
    mae = float(np.mean(np.abs(diffs)))

    return AgreementResult(
        n=n,
        mae=mae,
        bias=bias,
        sd=sd,
        lower_loa=bias - LOA_MULTIPLIER * sd,
        upper_loa=bias + LOA_MULTIPLIER * sd,
    )


@dataclass(frozen=True)
class SplitResult:
    """A subject-level train/test split: no subject appears in both sides."""

    train_data: pd.DataFrame
    test_data: pd.DataFrame
    train_subjects: tuple[object, ...]
    test_subjects: tuple[object, ...]

    @property
    def n_train_windows(self) -> int:
        return int(len(self.train_data))

    @property
    def n_test_windows(self) -> int:
        return int(len(self.test_data))


def split_by_subject(
    windows: WindowFrame,
    *,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> SplitResult:
    """
    Split windows into train/test by subject, not by window.

    A window-level split would let the same subject's windows land on both
    sides, so the model could learn that subject's individual calibration
    and appear to generalize when it has only memorized who it was
    evaluated on. Holding out whole subjects is what actually tests
    generalization to a person the model has not seen.
    """

    if not 0 < test_fraction < 1:
        raise ValueError(
            f"test_fraction must be between 0 and 1 (exclusive); got "
            f"{test_fraction}."
        )

    subjects = sorted(windows.data["subject_id"].unique().tolist(), key=str)
    n_subjects = len(subjects)

    if n_subjects < 2:
        raise ValueError(
            f"A subject-level split requires at least 2 distinct subjects; "
            f"found {n_subjects}."
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(subjects).tolist()

    n_test = max(1, round(test_fraction * n_subjects))
    n_test = min(n_test, n_subjects - 1)

    test_subjects = tuple(sorted(shuffled[:n_test], key=str))
    train_subjects = tuple(sorted(shuffled[n_test:], key=str))

    train_data = windows.data.loc[windows.data["subject_id"].isin(train_subjects)].copy()
    test_data = windows.data.loc[windows.data["subject_id"].isin(test_subjects)].copy()

    return SplitResult(
        train_data=train_data,
        test_data=test_data,
        train_subjects=train_subjects,
        test_subjects=test_subjects,
    )


def split_by_window(
    windows: WindowFrame,
    *,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> SplitResult:
    """
    Split windows into train/test by window, ignoring subject identity.

    Exists to make the leakage risk in split_by_subject's docstring
    concrete rather than asserted: a subject can land on both sides here,
    so a model can fit that subject's individual calibration during
    training and then score well on "their" held-out windows for that
    reason alone. Comparing this split's test agreement against
    split_by_subject's is what turns the leakage warning into a number.
    """

    if not 0 < test_fraction < 1:
        raise ValueError(
            f"test_fraction must be between 0 and 1 (exclusive); got "
            f"{test_fraction}."
        )

    n_windows = len(windows.data)

    if n_windows < 2:
        raise ValueError(
            f"A window-level split requires at least 2 windows; found "
            f"{n_windows}."
        )

    rng = np.random.default_rng(seed)
    shuffled_index = rng.permutation(windows.data.index.to_numpy())

    n_test = max(1, round(test_fraction * n_windows))
    n_test = min(n_test, n_windows - 1)

    test_index = shuffled_index[:n_test]
    train_index = shuffled_index[n_test:]

    train_data = windows.data.loc[train_index].copy()
    test_data = windows.data.loc[test_index].copy()

    train_subjects = tuple(sorted(train_data["subject_id"].unique().tolist(), key=str))
    test_subjects = tuple(sorted(test_data["subject_id"].unique().tolist(), key=str))

    return SplitResult(
        train_data=train_data,
        test_data=test_data,
        train_subjects=train_subjects,
        test_subjects=test_subjects,
    )


@dataclass(frozen=True)
class RecalibrationModel:
    """
    A one-feature linear recalibration of reference hr from ring bvp_hr.

    hr ~ intercept + slope * bvp_hr, fit by ordinary least squares on the
    training windows only. This corrects a systematic linear bias; it
    cannot correct condition-specific or nonlinear error, which the
    Stress-test step examines separately.
    """

    intercept: float
    slope: float
    r_squared: float
    n_train: int


def fit_recalibration(
    train_data: pd.DataFrame,
    *,
    feature_col: str = "bvp_hr",
    target_col: str = "hr",
) -> RecalibrationModel:
    """Fit hr ~ intercept + slope * feature_col by OLS on train_data."""

    validate_is_dataframe(train_data)

    n_train = int(len(train_data))

    if n_train < 2:
        raise ValueError(
            f"At least 2 training windows are required to fit a "
            f"recalibration model; found {n_train}."
        )

    feature = train_data[feature_col].to_numpy(dtype=float)
    target = train_data[target_col].to_numpy(dtype=float)

    design = sm.add_constant(feature)
    fit = sm.OLS(target, design).fit()

    intercept, slope = fit.params

    return RecalibrationModel(
        intercept=float(intercept),
        slope=float(slope),
        r_squared=float(fit.rsquared),
        n_train=n_train,
    )


def apply_recalibration(
    model: RecalibrationModel,
    feature: pd.Series,
) -> pd.Series:
    """Apply a fitted recalibration model to a feature series."""

    return model.intercept + model.slope * feature


def evaluate_on_test_data(
    model: RecalibrationModel,
    test_data: pd.DataFrame,
    *,
    feature_col: str = "bvp_hr",
    target_col: str = "hr",
) -> tuple[pd.DataFrame, AgreementResult]:
    """
    Apply a fitted model to test_data, derive per-window prediction and
    error columns (predicted_hr, pred_abs_error, pred_diff, pred_mean_hr),
    and summarize agreement over them in one call.

    Reused for the chosen split, the "what if" comparison split, and (with
    a different split) nowhere else -- exists once here rather than as a
    page-level helper recomputing the same three derived columns.
    """

    validate_is_dataframe(test_data)

    enriched = test_data.copy()
    enriched["predicted_hr"] = apply_recalibration(model, enriched[feature_col])
    enriched["pred_abs_error"] = (enriched["predicted_hr"] - enriched[target_col]).abs()
    enriched["pred_diff"] = enriched["predicted_hr"] - enriched[target_col]
    enriched["pred_mean_hr"] = (enriched["predicted_hr"] + enriched[target_col]) / 2

    evaluation = agreement_summary(enriched["predicted_hr"], enriched[target_col])

    return enriched, evaluation


@dataclass(frozen=True)
class RetentionResult:
    """
    What signal-quality filtering kept, and what agreement looks like on
    the retained windows only.

    Always carries both retention and agreement together, so improved
    agreement after filtering can never be read without knowing how much
    data produced it.
    """

    quality_threshold: float
    n_input_windows: int
    n_retained_windows: int
    agreement: AgreementResult

    @property
    def n_excluded_windows(self) -> int:
        return self.n_input_windows - self.n_retained_windows

    @property
    def retention_rate(self) -> float:
        return self.n_retained_windows / self.n_input_windows


def filter_by_quality(
    data: pd.DataFrame,
    *,
    predicted_col: str,
    target_col: str = "hr",
    quality_col: str = "quality",
    threshold: float,
) -> RetentionResult:
    """Keep only windows with quality >= threshold, and re-summarize agreement."""

    validate_is_dataframe(data)

    n_input = int(len(data))

    if n_input == 0:
        raise ValueError("Cannot filter zero windows.")

    retained = data.loc[data[quality_col] >= threshold]

    if retained.empty:
        raise ValueError(
            f"No windows remain at quality threshold {threshold}. Choose a "
            "lower threshold."
        )

    agreement = agreement_summary(retained[predicted_col], retained[target_col])

    return RetentionResult(
        quality_threshold=float(threshold),
        n_input_windows=n_input,
        n_retained_windows=int(len(retained)),
        agreement=agreement,
    )


@dataclass(frozen=True)
class ConditionBreakdown:
    """Agreement for one group (an activity condition or a quality bin)."""

    group: str
    n: int
    mae: float
    bias: float


def breakdown_by_condition(
    data: pd.DataFrame,
    *,
    predicted_col: str,
    target_col: str = "hr",
    group_col: str = "Label",
) -> tuple[ConditionBreakdown, ...]:
    """
    Per-group MAE and bias, so a per-condition failure is visible even when
    the pooled/aggregate MAE looks acceptable.

    Groups are returned in alphabetical order for stable reporting, the same
    convention used by modules/fairness/core/pre_model_metrics.py.
    """

    validate_is_dataframe(data)

    if group_col not in data.columns:
        raise ValueError(f"Group column '{group_col}' was not found in the data.")

    if data.empty:
        raise ValueError("Cannot break down agreement over zero windows.")

    groups = sorted(data[group_col].dropna().unique().tolist(), key=str)

    results: list[ConditionBreakdown] = []

    for group_value in groups:
        subset = data.loc[data[group_col] == group_value]

        # MAE and bias are defined for n=1, unlike AgreementResult's
        # standard deviation of agreement, so this computes them directly
        # rather than through agreement_summary: a thin condition should
        # still show up in a stress-test breakdown, not disappear because
        # one group was too small for a limits-of-agreement estimate.
        diffs = (
            subset[predicted_col].to_numpy(dtype=float)
            - subset[target_col].to_numpy(dtype=float)
        )

        results.append(
            ConditionBreakdown(
                group=str(group_value),
                n=int(len(subset)),
                mae=float(np.mean(np.abs(diffs))),
                bias=float(np.mean(diffs)),
            )
        )

    return tuple(results)
