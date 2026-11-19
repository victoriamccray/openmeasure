"""
Time-Series QA pipeline.

Single entry point for the module. Its purpose is to own the order of
operations and the decisions about what can be assessed, so the Streamlit
page contains no analytical logic of its own.

Nothing in this module modifies the input data. Every result describes the
series as supplied.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .completeness import (
    DEFAULT_PERIOD_COVERAGE_THRESHOLD,
    CompletenessResult,
    check_completeness,
)
from .frequency import FrequencyEstimate, infer_frequency
from .prepare import PreparedSeries, prepare_series
from .recommend import QARecommendation, recommend_checks
from .temporal import TemporalIntegrityResult, check_temporal_integrity


MINIMUM_OBSERVATIONS = 2


@dataclass(frozen=True)
class TimeSeriesQAResult:
    """
    Complete Time-Series QA findings.

    The ingest accounting is repeated at the top level so a report can state
    what was loaded and what was excluded without reaching into the
    component results.
    """

    prepared: PreparedSeries
    frequency: FrequencyEstimate
    temporal: TemporalIntegrityResult
    completeness: CompletenessResult
    recommendation: QARecommendation

    timestamp_col: str
    value_col: str

    n_input_rows: int
    n_rows_used: int
    n_null_timestamps: int
    n_unparseable_timestamps: int
    example_unparseable: tuple[str, ...]


def run_time_series_qa(
    data: pd.DataFrame,
    timestamp_col: str,
    value_col: str,
    *,
    jitter_tolerance: pd.Timedelta | None = None,
    period_freq: str | None = None,
    coverage_threshold: float = DEFAULT_PERIOD_COVERAGE_THRESHOLD,
) -> TimeSeriesQAResult:
    """
    Run every v0.1 quality check on a single long-format series.

    Parameters
    ----------
    data:
        Input dataset, one row per observation.
    timestamp_col:
        Column holding the observation timestamp.
    value_col:
        Column holding the observed value.
    jitter_tolerance:
        How far an observation may sit from its expected time and still
        count as present. Derived from the sampling interval when omitted.
    period_freq:
        Period grouping for coverage ("D", "M", "Y"). Derived from the
        sampling interval when omitted.
    coverage_threshold:
        Share of expected observations a period must contain to count as
        adequately covered.

    Returns
    -------
    TimeSeriesQAResult

    Raises
    ------
    TypeError
        If data is not a pandas DataFrame.
    ValueError
        If a column is missing or unusable, or if fewer than
        MINIMUM_OBSERVATIONS rows have a usable timestamp.
    """

    prepared = prepare_series(data, timestamp_col, value_col)

    if prepared.n_rows_used < MINIMUM_OBSERVATIONS:
        raise ValueError(
            f"At least {MINIMUM_OBSERVATIONS} observations with a usable "
            f"timestamp are required; {prepared.n_rows_used} of "
            f"{prepared.n_input_rows} rows are usable."
        )

    frequency = infer_frequency(prepared.timestamps)

    temporal = check_temporal_integrity(
        prepared,
        frequency,
        jitter_tolerance=jitter_tolerance,
    )

    completeness = check_completeness(
        prepared,
        frequency,
        period_freq=period_freq,
        coverage_threshold=coverage_threshold,
    )

    recommendation = recommend_checks(
        n_rows_used=prepared.n_rows_used,
        n_distinct_timestamps=temporal.n_distinct_timestamps,
        frequency=frequency,
    )

    return TimeSeriesQAResult(
        prepared=prepared,
        frequency=frequency,
        temporal=temporal,
        completeness=completeness,
        recommendation=recommendation,
        timestamp_col=timestamp_col,
        value_col=value_col,
        n_input_rows=prepared.n_input_rows,
        n_rows_used=prepared.n_rows_used,
        n_null_timestamps=prepared.n_null_timestamps,
        n_unparseable_timestamps=prepared.n_unparseable_timestamps,
        example_unparseable=prepared.example_unparseable,
    )
