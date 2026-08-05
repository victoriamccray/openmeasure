"""
Temporal integrity checks for a single time series.

Answers one question: is the time axis itself trustworthy? Specifically it
reports gaps against an expected frequency, duplicate timestamps,
chronological ordering, and interval regularity.

Gaps are found by generating a calendar-aware expected grid with
pd.date_range and set-differencing the observed timestamps against it,
rather than by comparing consecutive differences to a fixed Timedelta.
Difference comparison reports a false gap at every daylight saving
transition and cannot express a monthly interval at all.

Nothing here modifies, deduplicates, or fills data. A reported gap is a
description of the time axis, not a claim that the missing observations
should exist: a clinic may have been closed, a sensor may have been
intentionally offline, or no event may have occurred.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .frequency import FrequencyEstimate
from .grid import (
    build_grid,
    grid_is_aligned,
    match_to_grid,
    resolve_jitter_tolerance,
)
from .prepare import PreparedSeries


@dataclass(frozen=True)
class GapFinding:
    """
    One run of consecutive expected observations that are absent.

    gap_start / gap_end:
        The observed timestamps bracketing the gap.
    n_expected_missing:
        How many expected grid points fall inside the gap. This is the
        meaningful size of a gap; duration is reported alongside it because
        one missing month is 28 days in February and 31 in March.
    duration:
        Elapsed time between the bracketing observations.
    """

    gap_start: pd.Timestamp
    gap_end: pd.Timestamp
    n_expected_missing: int
    duration: pd.Timedelta


@dataclass(frozen=True)
class DuplicateFinding:
    """
    One timestamp carrying more than one observation.

    has_conflicting_values is True when the duplicated rows disagree, which
    is a different problem from a harmless exact repeat. Neither is
    resolved here; resolving would mean choosing which value is correct.
    """

    timestamp: pd.Timestamp
    n_rows: int
    n_distinct_values: int
    has_conflicting_values: bool


@dataclass(frozen=True)
class TemporalIntegrityResult:
    """
    Temporal integrity findings.

    Ingest accounting is carried through from PreparedSeries rather than
    recomputed. Recomputing was_out_of_order on the already-sorted
    timestamps would always return False.

    Indeterminate measures are None with an explanation in
    reason_not_assessable, never 0.0 or NaN. A NaN passed to
    shared.report.classify() would silently render the worst possible
    verdict.
    """

    # Carried through from PreparedSeries.
    n_input_rows: int
    n_rows_used: int
    n_null_timestamps: int
    n_unparseable_timestamps: int
    input_was_out_of_order: bool
    n_out_of_order_steps: int

    # Time span covered.
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp
    span: pd.Timedelta

    # Duplicate timestamps.
    n_distinct_timestamps: int
    distinct_timestamp_ratio: float
    n_duplicate_timestamp_rows: int
    n_distinct_duplicated_timestamps: int
    n_conflicting_duplicate_timestamps: int
    duplicates: tuple[DuplicateFinding, ...]

    # Gaps against the expected grid.
    gaps_assessable: bool
    gaps: tuple[GapFinding, ...]
    n_expected_observations: int | None
    n_missing_observations: int | None
    grid_occupancy_ratio: float | None

    # Jitter absorbed during grid matching.
    jitter_tolerance: pd.Timedelta | None
    n_jittered_observations: int | None
    max_jitter: pd.Timedelta | None

    # Interval regularity. Not assessable for calendar frequencies, whose
    # spacing varies by design, nor when there are too few intervals.
    regularity_assessable: bool
    modal_interval_share: float | None

    reason_not_assessable: str


def _duplicate_findings(
    timestamps: pd.DatetimeIndex,
    values: pd.Series,
) -> tuple[DuplicateFinding, ...]:
    """Describe every timestamp carrying more than one observation."""

    frame = pd.DataFrame(
        {"timestamp": timestamps, "value": values.to_numpy()}
    )

    findings: list[DuplicateFinding] = []

    counts = frame["timestamp"].value_counts()
    duplicated = counts[counts > 1].index.sort_values()

    for timestamp in duplicated:
        rows = frame.loc[frame["timestamp"] == timestamp, "value"]
        n_distinct_values = int(rows.dropna().nunique())

        findings.append(
            DuplicateFinding(
                timestamp=timestamp,
                n_rows=int(len(rows)),
                n_distinct_values=n_distinct_values,
                has_conflicting_values=bool(n_distinct_values > 1),
            )
        )

    return tuple(findings)


def _group_consecutive(positions: list[int]) -> list[list[int]]:
    """Group ascending integers into runs of consecutive values."""

    runs: list[list[int]] = []

    for position in positions:
        if runs and position == runs[-1][-1] + 1:
            runs[-1].append(position)
        else:
            runs.append([position])

    return runs


def check_temporal_integrity(
    prepared: PreparedSeries,
    frequency: FrequencyEstimate,
    *,
    jitter_tolerance: pd.Timedelta | None = None,
) -> TemporalIntegrityResult:
    """
    Check gaps, duplicates, ordering, and interval regularity.

    Parameters
    ----------
    prepared:
        Output of core.prepare.prepare_series.
    frequency:
        Output of core.frequency.infer_frequency.
    jitter_tolerance:
        How far an observation may sit from its expected grid point and
        still count as present. Defaults to
        DEFAULT_JITTER_TOLERANCE_FRACTION of the modal interval.

    Returns
    -------
    TemporalIntegrityResult

    Raises
    ------
    ValueError
        If fewer than two rows are available, which is too few to assess a
        time axis.
    """

    if prepared.n_rows_used < 2:
        raise ValueError(
            f"At least 2 usable observations are required to check temporal "
            f"integrity; {prepared.n_rows_used} available."
        )

    timestamps = prepared.timestamps

    # Grid matching requires a unique, monotonic index. Duplicates are
    # counted as a finding, then collapsed for matching only; pandas raises
    # InvalidIndexError if a duplicated index reaches get_indexer.
    distinct = timestamps.unique().sort_values()

    n_distinct = int(len(distinct))
    n_rows_used = int(prepared.n_rows_used)

    duplicates = _duplicate_findings(timestamps, prepared.values)

    first = distinct[0]
    last = distinct[-1]

    tolerance = resolve_jitter_tolerance(frequency, jitter_tolerance)

    gaps: tuple[GapFinding, ...] = ()
    n_expected: int | None = None
    n_missing: int | None = None
    occupancy: float | None = None
    n_jittered: int | None = None
    max_jitter: pd.Timedelta | None = None
    gaps_assessable = False
    reason = ""

    if frequency.offset is None:
        reason = frequency.reason
    elif frequency.is_business_day:
        reason = (
            "A business-day frequency was detected. Gap detection over "
            "business-day calendars is outside the scope of this version, "
            "because weekends and holidays would otherwise be reported as "
            "gaps."
        )
    else:
        grid = build_grid(first, last, frequency.offset, prepared.timezone)

        if not grid_is_aligned(grid, first, tolerance):
            reason = (
                f"The expected frequency ({frequency.offset}) does not "
                "align with where the observations actually fall, so a "
                "meaningful expected grid could not be built."
            )
        else:
            matched = match_to_grid(distinct, grid, tolerance)

            unmatched_positions = [
                position
                for position, index in enumerate(matched)
                if index == -1
            ]

            n_expected = int(len(grid))
            n_missing = int(len(unmatched_positions))
            occupancy = float((n_expected - n_missing) / n_expected)

            found: list[GapFinding] = []

            for run in _group_consecutive(unmatched_positions):
                before = run[0] - 1
                after = run[-1] + 1

                # grid[0] and grid[-1] are the observed endpoints, so a run
                # of absent points is always bracketed by matched points.
                if before < 0 or after >= len(grid):
                    continue

                gap_start = distinct[matched[before]]
                gap_end = distinct[matched[after]]

                found.append(
                    GapFinding(
                        gap_start=gap_start,
                        gap_end=gap_end,
                        n_expected_missing=int(len(run)),
                        duration=gap_end - gap_start,
                    )
                )

            gaps = tuple(found)
            gaps_assessable = True

            offsets = [
                abs(distinct[index] - grid[position])
                for position, index in enumerate(matched)
                if index != -1
            ]
            nonzero = [value for value in offsets if value > pd.Timedelta(0)]
            n_jittered = int(len(nonzero))
            max_jitter = max(nonzero) if nonzero else pd.Timedelta(0)

    # Interval regularity is meaningless for calendar frequencies, whose
    # real spacing varies by design, and for a single interval.
    regularity_assessable = bool(
        frequency.modal_interval_share is not None
        and not frequency.is_calendar_anchored
    )

    return TemporalIntegrityResult(
        n_input_rows=prepared.n_input_rows,
        n_rows_used=n_rows_used,
        n_null_timestamps=prepared.n_null_timestamps,
        n_unparseable_timestamps=prepared.n_unparseable_timestamps,
        input_was_out_of_order=prepared.was_out_of_order,
        n_out_of_order_steps=prepared.n_out_of_order_steps,
        first_timestamp=first,
        last_timestamp=last,
        span=last - first,
        n_distinct_timestamps=n_distinct,
        distinct_timestamp_ratio=float(n_distinct / n_rows_used),
        n_duplicate_timestamp_rows=int(n_rows_used - n_distinct),
        n_distinct_duplicated_timestamps=int(len(duplicates)),
        n_conflicting_duplicate_timestamps=int(
            sum(1 for item in duplicates if item.has_conflicting_values)
        ),
        duplicates=duplicates,
        gaps_assessable=gaps_assessable,
        gaps=gaps,
        n_expected_observations=n_expected,
        n_missing_observations=n_missing,
        grid_occupancy_ratio=occupancy,
        jitter_tolerance=tolerance,
        n_jittered_observations=n_jittered,
        max_jitter=max_jitter,
        regularity_assessable=regularity_assessable,
        modal_interval_share=frequency.modal_interval_share,
        reason_not_assessable=reason,
    )
