"""
Estimate the expected sampling frequency of a time series.

This is the single most consequential judgment the module makes: gap
detection and coverage-per-period are only meaningful relative to an
expected frequency, so this module reports not just an estimate but how it
was reached and how well it describes the data.

The frequency is represented as a pandas DateOffset, never a Timedelta.
A Timedelta cannot express "one month" (28-31 days) or absorb a daylight
saving transition, so downstream gap detection builds a calendar-aware
date_range grid from this offset instead of comparing raw differences.

Inference is two-tier because pandas' own pd.infer_freq returns None in
exactly the cases that matter most: it needs a gapless series to succeed,
and it does not recognize regular mid-month monthly data. The fallback is
the modal strictly-positive difference, promoted to a calendar offset where
the spacing implies one.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BaseOffset


# Day counts that imply a calendar offset rather than a fixed interval.
# A monthly series has 28-31 day spacing depending on the month, so the
# modal difference alone must be promoted to a real calendar offset.
_CALENDAR_CANDIDATES: tuple[tuple[int, int, str, pd.DateOffset], ...] = (
    (28, 31, "monthly", pd.DateOffset(months=1)),
    (89, 92, "quarterly", pd.DateOffset(months=3)),
    (365, 366, "annual", pd.DateOffset(years=1)),
)

# Minimum share of intervals the modal difference must describe before it is
# used to build a grid. This is an operational guard, not a quality
# judgment: generating a grid from an interval that describes a minority of
# the data produces a meaningless number of expected points. Whether a
# series is "regular enough" to report on is a separate decision made by
# core/recommend.py.
MIN_MODAL_SHARE_FOR_GRID = 0.5

# Intervals are snapped to a clean time unit before the mode is taken.
# Without snapping, a daily series logged at 09:00:01 and then 09:00:03
# produces a different difference for every pair, so no modal interval
# emerges and a perfectly regular series looks event-driven. The same 5%
# notion is used as the grid-matching tolerance in core/grid.py.
INTERVAL_SNAP_FRACTION = 0.05

# Snapping rounds to the largest of these units that is no coarser than
# INTERVAL_SNAP_FRACTION of the interval. Rounding to a fraction of the
# median instead would inherit the median's own fractional offset and
# produce an interval like "1 day and 1.5 seconds".
#
# The ladder includes sub-multiples, not just the base units, because gaps
# between units leave jitter unabsorbed. One-second data with a couple of
# milliseconds of jitter has a snap budget of 50 ms, which sits between
# "s" and "ms"; without a 10 ms rung it would round to the millisecond,
# keep every difference distinct, and be misreported as event-driven.
# Every rung divides its coarser neighbours evenly so that rounding cannot
# shift a true interval off its own value.
_SNAP_UNITS: tuple[tuple[str, pd.Timedelta], ...] = (
    ("D", pd.Timedelta(days=1)),
    ("12h", pd.Timedelta(hours=12)),
    ("6h", pd.Timedelta(hours=6)),
    ("h", pd.Timedelta(hours=1)),
    ("30min", pd.Timedelta(minutes=30)),
    ("10min", pd.Timedelta(minutes=10)),
    ("5min", pd.Timedelta(minutes=5)),
    ("min", pd.Timedelta(minutes=1)),
    ("30s", pd.Timedelta(seconds=30)),
    ("10s", pd.Timedelta(seconds=10)),
    ("5s", pd.Timedelta(seconds=5)),
    ("s", pd.Timedelta(seconds=1)),
    ("100ms", pd.Timedelta(milliseconds=100)),
    ("10ms", pd.Timedelta(milliseconds=10)),
    ("ms", pd.Timedelta(milliseconds=1)),
    ("100us", pd.Timedelta(microseconds=100)),
    ("10us", pd.Timedelta(microseconds=10)),
    ("us", pd.Timedelta(microseconds=1)),
)

# Inference outcomes. Recorded so core/recommend.py can explain which
# checks are defensible and why.
SOURCE_PANDAS = "pandas_infer_freq"
SOURCE_MODAL = "modal_interval"
SOURCE_CALENDAR = "calendar_candidate"
SOURCE_SINGLE_INTERVAL = "single_interval"
SOURCE_INSUFFICIENT = "insufficient_data"
SOURCE_ALL_IDENTICAL = "all_timestamps_identical"
SOURCE_NOT_DEFENSIBLE = "not_defensible"


@dataclass(frozen=True)
class FrequencyEstimate:
    """
    An estimated sampling frequency, with its provenance.

    offset:
        Calendar-aware expected interval, or None when no defensible
        frequency could be established.
    source:
        How the estimate was reached; one of the SOURCE_* constants.
    alias:
        pandas frequency alias for display only. Never key logic off this
        string: the aliases for monthly, hourly, and quarterly were renamed
        across pandas versions.
    modal_interval:
        Most common strictly-positive difference between consecutive
        distinct timestamps, or None when it could not be computed.
    modal_interval_share:
        Share of positive intervals equal to modal_interval, in [0, 1].
        None when there are too few intervals to be meaningful (fewer than
        two). This is the module's measure of interval regularity.
    is_calendar_anchored:
        Whether the offset has variable real duration (monthly, quarterly,
        annual, weekly, business-day) rather than a fixed one.
    is_business_day:
        Whether a business-day frequency was detected. Gap detection over
        business days is out of scope for v0.1, so this is reported and
        declined rather than silently flagging every weekend as a gap.
    n_distinct_timestamps:
        Distinct timestamps used for inference.
    reason:
        Plain-language explanation of the outcome, shown to the user.
    """

    offset: BaseOffset | None
    source: str
    alias: str | None
    modal_interval: pd.Timedelta | None
    modal_interval_share: float | None
    is_calendar_anchored: bool
    is_business_day: bool
    n_distinct_timestamps: int
    reason: str


def _is_calendar_anchored(offset: BaseOffset) -> bool:
    """
    Return whether an offset has variable real duration.

    Fixed-duration offsets (Day, Hour, Minute) expose .nanos; calendar
    offsets (MonthEnd, QuarterEnd, Week, BusinessDay, DateOffset(months=...))
    raise instead, because their length depends on where in the calendar
    they are applied.
    """

    try:
        offset.nanos
        return False
    except (ValueError, AttributeError, TypeError):
        return True


def _snap_unit_for(median_interval: pd.Timedelta) -> str | None:
    """
    Choose the rounding unit for interval snapping.

    Returns the largest clean unit no coarser than INTERVAL_SNAP_FRACTION of
    the typical interval, or None when even the finest unit is too coarse to
    round safely.
    """

    budget = median_interval * INTERVAL_SNAP_FRACTION

    for alias, duration in _SNAP_UNITS:
        if duration <= budget:
            return alias

    return None


def _modal_positive_interval(
    distinct: pd.DatetimeIndex,
) -> tuple[pd.Timedelta | None, float | None]:
    """
    Return the modal strictly-positive interval and the share it describes.

    Zero-length differences are excluded before taking the mode. With more
    than half the timestamps duplicated the raw mode is zero, and a
    zero-length grid step makes date_range unbounded.

    For tz-aware input the differences are taken on local wall-clock time.
    In absolute terms a daily series spans 23 or 25 hours across a daylight
    saving transition, which would otherwise score correctly-spaced data as
    irregular. On the wall clock, midnight to midnight is one day
    year-round.
    """

    if distinct.tz is not None:
        distinct = distinct.tz_localize(None)

    differences = pd.Series(distinct).diff().dropna()
    positive = differences[differences > pd.Timedelta(0)]

    if len(positive) == 0:
        return None, None

    # Snap intervals to a clean unit before taking the mode, so
    # sub-interval logging jitter does not make every difference unique.
    snap_unit = _snap_unit_for(positive.median())

    if snap_unit is not None:
        positive = positive.dt.round(snap_unit)
        positive = positive[positive > pd.Timedelta(0)]

        if len(positive) == 0:
            return None, None

    modes = positive.mode()

    if len(modes) == 0:
        return None, None

    # mode() returns ties sorted ascending; taking the first makes the
    # choice deterministic across runs.
    modal = pd.Timedelta(modes.iloc[0])

    # A single interval cannot distinguish the sampling frequency from a
    # gap, so a share of 1.0 computed from one interval would be
    # misleadingly reassuring.
    share = (
        float((positive == modal).sum() / len(positive))
        if len(positive) >= 2
        else None
    )

    return modal, share


def _promote_to_calendar_offset(
    modal: pd.Timedelta,
) -> tuple[pd.DateOffset, str] | None:
    """Return a calendar offset if the modal spacing implies one."""

    days = modal / pd.Timedelta(days=1)

    for low, high, label, offset in _CALENDAR_CANDIDATES:
        if low <= days <= high:
            return offset, label

    return None


def infer_frequency(timestamps: pd.DatetimeIndex) -> FrequencyEstimate:
    """
    Estimate the expected sampling frequency of a series of timestamps.

    Duplicates are collapsed for inference only. This does not modify the
    caller's data; duplicate timestamps are reported as a finding by
    core/temporal.py.

    Parameters
    ----------
    timestamps:
        Sorted timestamps from a PreparedSeries.

    Returns
    -------
    FrequencyEstimate
        The estimate, its provenance, and its regularity measure. Never
        raises for degenerate input; indeterminate cases are reported via
        source and reason with offset=None.
    """

    distinct = pd.DatetimeIndex(timestamps).unique().sort_values()
    n_distinct = int(len(distinct))

    def estimate(**overrides) -> FrequencyEstimate:
        defaults = dict(
            offset=None,
            source=SOURCE_NOT_DEFENSIBLE,
            alias=None,
            modal_interval=None,
            modal_interval_share=None,
            is_calendar_anchored=False,
            is_business_day=False,
            n_distinct_timestamps=n_distinct,
            reason="",
        )
        defaults.update(overrides)
        return FrequencyEstimate(**defaults)

    if n_distinct == 0:
        return estimate(
            source=SOURCE_INSUFFICIENT,
            reason="There are no usable timestamps.",
        )

    if n_distinct == 1:
        return estimate(
            source=SOURCE_ALL_IDENTICAL,
            reason=(
                "Every observation shares the same timestamp, so no "
                "sampling interval exists. Duplicate timestamps are "
                "reported separately."
            ),
        )

    modal, share = _modal_positive_interval(distinct)

    if n_distinct == 2:
        return estimate(
            offset=None,
            source=SOURCE_SINGLE_INTERVAL,
            modal_interval=modal,
            modal_interval_share=None,
            reason=(
                "Only two distinct timestamps are present. A single "
                "interval cannot distinguish the sampling frequency from "
                "a gap, so no expected frequency is proposed."
            ),
        )

    # Tier 1: pandas' own inference. Succeeds only on a gapless, regularly
    # spaced series, but when it succeeds it is authoritative.
    try:
        alias = pd.infer_freq(distinct)
    except ValueError:
        # Raised when there are fewer than three dates.
        alias = None

    if alias is not None:
        offset = to_offset(alias)
        is_business_day = alias.startswith("B")
        return estimate(
            offset=offset,
            source=SOURCE_PANDAS,
            alias=alias,
            modal_interval=modal,
            modal_interval_share=share,
            is_calendar_anchored=_is_calendar_anchored(offset),
            is_business_day=is_business_day,
            reason=(
                f"pandas recognized a regular '{alias}' frequency across "
                "all observations."
            ),
        )

    # Tier 2: the modal positive interval.
    if modal is None:
        return estimate(
            source=SOURCE_ALL_IDENTICAL,
            reason=(
                "No positive interval exists between timestamps, so no "
                "sampling frequency could be estimated."
            ),
        )

    # Calendar promotion is attempted before the regularity gate below.
    # Monthly spacing is genuinely 28-31 days, so a perfectly regular
    # monthly series has a low modal share by construction. Applying the
    # gate first would reject valid calendar data as irregular.
    promoted = _promote_to_calendar_offset(modal)

    if promoted is not None:
        offset, label = promoted
        return estimate(
            offset=offset,
            source=SOURCE_CALENDAR,
            alias=None,
            modal_interval=modal,
            modal_interval_share=share,
            is_calendar_anchored=True,
            reason=(
                f"Spacing of about {modal.days} days implies a {label} "
                "series, so a calendar offset is used rather than a fixed "
                "interval. This keeps month lengths and leap years correct. "
                "Interval regularity is not reported for calendar "
                "frequencies, because the spacing varies by design."
            ),
        )

    if share is not None and share < MIN_MODAL_SHARE_FOR_GRID:
        return estimate(
            source=SOURCE_NOT_DEFENSIBLE,
            modal_interval=modal,
            modal_interval_share=share,
            reason=(
                f"The most common interval ({modal}) describes only "
                f"{share:.0%} of the intervals in this series, so treating "
                "it as an expected frequency is not defensible. This "
                "pattern is typical of event-driven data, where "
                "observations occur when something happens rather than on "
                "a schedule."
            ),
        )

    offset = to_offset(modal)

    return estimate(
        offset=offset,
        source=SOURCE_MODAL,
        alias=None,
        modal_interval=modal,
        modal_interval_share=share,
        is_calendar_anchored=_is_calendar_anchored(offset),
        reason=(
            f"pandas did not recognize a single regular frequency, so the "
            f"most common interval ({modal}) is used as the expected "
            f"frequency. It describes "
            + (
                f"{share:.0%} of the intervals present."
                if share is not None
                else "the intervals present."
            )
        ),
    )
