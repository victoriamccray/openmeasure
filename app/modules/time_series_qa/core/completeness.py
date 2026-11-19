"""
Completeness and coverage checks for a single time series.

Distinguishes two different ways an observation can be unavailable, which
are easy to conflate and mean different things:

- An *absent* observation: no row exists for an expected timestamp.
- A *missing* value: a row exists but its value is empty.

Coverage per period is therefore computed against the number of *expected*
observations in that period, not against the rows that happen to be
present. A month where 28 of 30 days never arrived, and the two days that
did arrive have values, is 7% covered, not 100% complete.

Nothing here imputes, fills, or drops anything. Sentinel values such as
-999 or "NA" are counted as present, because deciding that a particular
value stands for "missing" is a value-anomaly judgment outside this
version's scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .frequency import FrequencyEstimate
from .grid import build_grid, match_to_grid, resolve_jitter_tolerance
from .prepare import PreparedSeries


# Share of expected observations a period must contain before it counts as
# adequately covered. This is an OpenMeasure project convention, not a
# published standard: no canonical cross-domain completeness threshold
# exists. It is user-editable, and the report states its provenance.
DEFAULT_PERIOD_COVERAGE_THRESHOLD = 0.90

# Defaults for the consecutive-run and total-missing flags applied to daily
# data grouped into months. These are OpenMeasure project conventions, not a
# published standard: they are a starting point for reviewing whether a
# monthly summary can be computed from daily observations, and are meant to
# be adjusted to the requirements of the analysis at hand.
#
# The flag reports that a threshold was crossed. It does not assert that the
# month is unusable, because whether it is depends on what will be computed
# from it.
DEFAULT_MAX_MISSING_PER_MONTH = 10
DEFAULT_MAX_CONSECUTIVE_MISSING = 5

# What the longest-missing-run figure was measured against.
BASIS_EXPECTED_GRID = "expected_grid"
BASIS_PRESENT_ROWS = "present_rows_only"


@dataclass(frozen=True)
class PeriodCoverage:
    """
    Coverage for one calendar period.

    n_expected:
        Expected observations in this period, from the expected grid.
    n_rows_present:
        Rows actually present in this period.
    n_values_nonmissing:
        Present rows whose value is not missing.
    effective_coverage_ratio:
        n_values_nonmissing / n_expected. Deliberately measured against
        expected rather than present rows, so absent observations cannot be
        hidden.
    is_partial_period:
        Whether the observed span covers only part of this period, which is
        normally true of the first and last period. Partial periods are
        excluded from the headline share by default, because an incomplete
        first month is not a data-quality problem.
    """

    period_label: str
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    n_expected: int
    n_rows_present: int
    n_values_nonmissing: int
    effective_coverage_ratio: float
    is_partial_period: bool
    meets_threshold: bool


@dataclass(frozen=True)
class CompletenessResult:
    """
    Completeness findings.

    Indeterminate measures are None with an explanation in
    reason_not_assessable, never 0.0 or NaN, because a NaN passed to
    shared.report.classify() silently renders the worst verdict.
    """

    n_rows_used: int
    n_missing_values: int
    value_completeness_ratio: float

    longest_missing_run: int
    longest_missing_run_start: pd.Timestamp | None
    longest_missing_run_end: pd.Timestamp | None
    missing_run_basis: str

    per_period: tuple[PeriodCoverage, ...]
    period_label_freq: str | None
    n_periods: int | None
    n_periods_assessed: int | None
    share_of_periods_meeting_coverage: float | None
    coverage_threshold: float

    exceeds_max_missing_per_month: bool | None
    exceeds_max_consecutive_missing: bool | None
    monthly_run_rule_applicable: bool

    reason_not_assessable: str


def _period_freq_for(frequency: FrequencyEstimate) -> str | None:
    """
    Choose a reporting period granularity coarser than the sampling rate.

    Sub-daily data is summarized by day, daily and weekly data by month,
    and monthly or coarser data by year.
    """

    modal = frequency.modal_interval

    if modal is None:
        return None

    if modal < pd.Timedelta(days=1):
        return "D"

    if modal < pd.Timedelta(days=28):
        return "M"

    return "Y"


def _longest_run(flags: list[bool]) -> tuple[int, int | None]:
    """Return the longest run of True values and where it starts."""

    best = 0
    best_start: int | None = None
    current = 0
    current_start = 0

    for position, flag in enumerate(flags):
        if flag:
            if current == 0:
                current_start = position
            current += 1
            if current > best:
                best = current
                best_start = current_start
        else:
            current = 0

    return best, best_start


def _drop_tz(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Remove timezone information for calendar-period grouping.

    Periods are wall-clock calendar concepts. Converting explicitly avoids
    pandas emitting a warning about dropping the timezone implicitly.
    """

    return index.tz_localize(None) if index.tz is not None else index


def check_completeness(
    prepared: PreparedSeries,
    frequency: FrequencyEstimate,
    *,
    period_freq: str | None = None,
    coverage_threshold: float = DEFAULT_PERIOD_COVERAGE_THRESHOLD,
    max_missing_per_month: int = DEFAULT_MAX_MISSING_PER_MONTH,
    max_consecutive_missing: int = DEFAULT_MAX_CONSECUTIVE_MISSING,
) -> CompletenessResult:
    """
    Check missing values, missing runs, and coverage per calendar period.

    Parameters
    ----------
    prepared:
        Output of core.prepare.prepare_series.
    frequency:
        Output of core.frequency.infer_frequency.
    period_freq:
        pandas period alias for grouping ("D", "M", "Y"). Derived from the
        sampling rate when omitted.
    coverage_threshold:
        Share of expected observations a period must contain to count as
        adequately covered.

    Returns
    -------
    CompletenessResult

    Raises
    ------
    ValueError
        If coverage_threshold is outside [0, 1].
    """

    if not 0.0 <= coverage_threshold <= 1.0:
        raise ValueError(
            f"coverage_threshold must be between 0 and 1; got "
            f"{coverage_threshold}."
        )

    values = prepared.values
    timestamps = prepared.timestamps

    n_rows_used = int(prepared.n_rows_used)
    missing_mask = values.isna()
    n_missing_values = int(missing_mask.sum())

    value_completeness_ratio = float(
        (n_rows_used - n_missing_values) / n_rows_used
    )

    reason = ""
    resolved_period_freq = period_freq or _period_freq_for(frequency)

    # Without an expected grid, coverage per period cannot be computed and
    # missing runs can only be measured across the rows that are present.
    if frequency.offset is None or resolved_period_freq is None:
        reason = frequency.reason or (
            "No expected frequency is available, so coverage per period "
            "cannot be computed."
        )

        run_length, run_start = _longest_run(missing_mask.tolist())

        return CompletenessResult(
            n_rows_used=n_rows_used,
            n_missing_values=n_missing_values,
            value_completeness_ratio=value_completeness_ratio,
            longest_missing_run=run_length,
            longest_missing_run_start=(
                timestamps[run_start] if run_start is not None else None
            ),
            longest_missing_run_end=(
                timestamps[run_start + run_length - 1]
                if run_start is not None
                else None
            ),
            missing_run_basis=BASIS_PRESENT_ROWS,
            per_period=(),
            period_label_freq=resolved_period_freq,
            n_periods=None,
            n_periods_assessed=None,
            share_of_periods_meeting_coverage=None,
            coverage_threshold=coverage_threshold,
            exceeds_max_missing_per_month=None,
            exceeds_max_consecutive_missing=None,
            monthly_run_rule_applicable=False,
            reason_not_assessable=reason,
        )

    distinct = timestamps.unique().sort_values()
    first = distinct[0]
    last = distinct[-1]

    grid = build_grid(first, last, frequency.offset, prepared.timezone)

    # Value availability per distinct timestamp: a timestamp counts as
    # having a value if any row at that timestamp is non-missing.
    present = pd.DataFrame(
        {
            "timestamp": timestamps,
            "nonmissing": (~missing_mask).to_numpy(),
        }
    )
    per_timestamp = present.groupby("timestamp", sort=True).agg(
        n_rows=("nonmissing", "size"),
        n_nonmissing=("nonmissing", "sum"),
    )

    # Match observations to grid points with the same tolerance the temporal
    # check uses. Exact alignment would count an observation logged a few
    # seconds off its expected time as absent here while the temporal check
    # counted it as present, so the two reports would disagree.
    tolerance = resolve_jitter_tolerance(frequency)
    matched = match_to_grid(distinct, grid, tolerance)

    matched_rows = []
    matched_nonmissing = []

    for index in matched:
        if index == -1:
            matched_rows.append(0)
            matched_nonmissing.append(0)
        else:
            timestamp = distinct[index]
            matched_rows.append(int(per_timestamp.at[timestamp, "n_rows"]))
            matched_nonmissing.append(
                int(per_timestamp.at[timestamp, "n_nonmissing"])
            )

    grid_frame = pd.DataFrame(index=grid)
    grid_frame["n_rows"] = matched_rows
    grid_frame["n_nonmissing"] = matched_nonmissing
    # Availability is per expected observation, not per row. Counting rows
    # would let two rows sharing a timestamp report more values present
    # than the grid expects, producing a coverage ratio above 1.0.
    grid_frame["has_value"] = grid_frame["n_nonmissing"] > 0
    grid_frame["unavailable"] = ~grid_frame["has_value"]

    run_length, run_start = _longest_run(
        grid_frame["unavailable"].tolist()
    )

    periods = pd.PeriodIndex(
        _drop_tz(pd.DatetimeIndex(grid_frame.index)),
        freq=resolved_period_freq,
    )
    grid_frame["period"] = periods

    coverages: list[PeriodCoverage] = []

    # A period is truncated only if the grid could have extended further
    # into it. Comparing the observed span to wall-clock period boundaries
    # instead would mark the final period partial for every daily series,
    # since a day's period runs to midnight.
    grid_start = grid[0]
    grid_end = grid[-1]
    first_period = periods[0]
    last_period = periods[-1]

    step_before = pd.PeriodIndex(
        _drop_tz(pd.DatetimeIndex([grid_start - frequency.offset])),
        freq=resolved_period_freq,
    )[0]
    step_after = pd.PeriodIndex(
        _drop_tz(pd.DatetimeIndex([grid_end + frequency.offset])),
        freq=resolved_period_freq,
    )[0]

    first_is_partial = step_before == first_period
    last_is_partial = step_after == last_period

    for period, chunk in grid_frame.groupby("period", sort=True):
        n_expected = int(len(chunk))
        n_values_nonmissing = int(chunk["has_value"].sum())
        n_rows_present = int(chunk["n_rows"].sum())

        ratio = float(n_values_nonmissing / n_expected)

        is_partial = bool(
            (period == first_period and first_is_partial)
            or (period == last_period and last_is_partial)
        )

        coverages.append(
            PeriodCoverage(
                period_label=str(period),
                period_start=period.start_time,
                period_end=period.end_time,
                n_expected=n_expected,
                n_rows_present=n_rows_present,
                n_values_nonmissing=n_values_nonmissing,
                effective_coverage_ratio=ratio,
                is_partial_period=is_partial,
                meets_threshold=bool(ratio >= coverage_threshold),
            )
        )

    assessed = [item for item in coverages if not item.is_partial_period]

    share = (
        float(
            sum(1 for item in assessed if item.meets_threshold) / len(assessed)
        )
        if assessed
        else None
    )

    if share is None:
        reason = (
            "Every period in this series is partially covered at its "
            "start or end, so no complete period is available to assess."
        )

    # The consecutive-run and total-missing flags are defined for daily
    # observations grouped into months, and are not extrapolated elsewhere.
    monthly_rule_applies = bool(
        resolved_period_freq == "M"
        and frequency.modal_interval is not None
        and pd.Timedelta(days=1)
        <= frequency.modal_interval
        < pd.Timedelta(days=2)
    )

    exceeds_total: bool | None = None
    exceeds_run: bool | None = None

    if monthly_rule_applies:
        worst_monthly_missing = max(
            (item.n_expected - item.n_values_nonmissing for item in assessed),
            default=0,
        )
        exceeds_total = bool(worst_monthly_missing > max_missing_per_month)
        exceeds_run = bool(run_length >= max_consecutive_missing)

    return CompletenessResult(
        n_rows_used=n_rows_used,
        n_missing_values=n_missing_values,
        value_completeness_ratio=value_completeness_ratio,
        longest_missing_run=run_length,
        longest_missing_run_start=(
            grid_frame.index[run_start] if run_start is not None else None
        ),
        longest_missing_run_end=(
            grid_frame.index[run_start + run_length - 1]
            if run_start is not None
            else None
        ),
        missing_run_basis=BASIS_EXPECTED_GRID,
        per_period=tuple(coverages),
        period_label_freq=resolved_period_freq,
        n_periods=int(len(coverages)),
        n_periods_assessed=int(len(assessed)),
        share_of_periods_meeting_coverage=share,
        coverage_threshold=coverage_threshold,
        exceeds_max_missing_per_month=exceeds_total,
        exceeds_max_consecutive_missing=exceeds_run,
        monthly_run_rule_applicable=monthly_rule_applies,
        reason_not_assessable=reason,
    )
