"""
Ingest and normalize a single long-format time series.

This module does one job: turn a raw DataFrame plus a timestamp column and a
value column into a validated, chronologically sorted series, while recording
exactly which rows were excluded and why.

It performs no frequency inference (see core/frequency.py) and no quality
checks (see core/temporal.py and core/completeness.py). It also never
imputes, fills, drops-on-your-behalf, or deduplicates values: rows are
excluded only when their timestamp cannot be used as an index at all.

Two facts recorded here are simultaneously ingest accounting and check
findings: was_out_of_order and the unparseable/null timestamp counts. They
are computed once here and carried through to the check results, following
the same convention as ReliabilityResult's n_excluded_cases.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.validation import validate_is_dataframe


# How many example unparseable values to retain for the report. Purely a
# display cap so a badly formatted column does not produce an unbounded
# error message.
MAX_UNPARSEABLE_EXAMPLES = 5


@dataclass(frozen=True)
class PreparedSeries:
    """
    A validated, chronologically sorted single time series.

    Normalized data
    ---------------
    timestamps:
        Sorted DatetimeIndex of usable timestamps. May contain duplicates;
        deduplicating would be a cleaning decision, which this module does
        not make.
    values:
        Observation values, aligned positionally to timestamps. May contain
        missing values; those are counted by core/completeness.py, not
        removed here.
    timezone:
        String name of the timezone if the timestamps are tz-aware, else
        None. Naive timestamps are never localized.

    Ingest accounting (which rows were excluded, and why)
    -----------------------------------------------------
    n_input_rows:
        Rows in the input DataFrame.
    n_null_timestamps:
        Rows whose timestamp was already missing (None/NaN) on input.
    n_unparseable_timestamps:
        Rows whose timestamp was present but could not be parsed as a
        datetime. An empty string counts here rather than as a null,
        because it is a present-but-unusable value.
    example_unparseable:
        Up to MAX_UNPARSEABLE_EXAMPLES raw values that failed to parse,
        so the user can see what went wrong.
    n_rows_used:
        Rows remaining after excluding unusable timestamps.
    was_out_of_order:
        Whether the input rows were in chronological order before sorting.
    n_out_of_order_steps:
        How many consecutive input pairs went backwards in time.
    """

    timestamps: pd.DatetimeIndex
    values: pd.Series
    timezone: str | None

    n_input_rows: int
    n_null_timestamps: int
    n_unparseable_timestamps: int
    example_unparseable: tuple[str, ...]
    n_rows_used: int
    was_out_of_order: bool
    n_out_of_order_steps: int


def _validate_columns(
    data: pd.DataFrame,
    timestamp_col: str,
    value_col: str,
) -> None:
    """Validate that both required columns exist and are distinct."""

    validate_is_dataframe(data)

    if timestamp_col not in data.columns:
        raise ValueError(
            f"Timestamp column '{timestamp_col}' was not found in the data."
        )

    if value_col not in data.columns:
        raise ValueError(
            f"Value column '{value_col}' was not found in the data."
        )

    if timestamp_col == value_col:
        raise ValueError(
            "The timestamp and value columns must be different columns."
        )


def _reject_numeric_timestamps(column: pd.Series, timestamp_col: str) -> None:
    """
    Refuse to guess the unit of a numeric timestamp column.

    pandas will happily read 1577836800 as 1970-01-01 00:00:01.577836800
    rather than 2020-01-01, silently corrupting every downstream result.
    Requiring the user to convert explicitly is the only safe behavior.
    """

    if pd.api.types.is_numeric_dtype(column):
        raise ValueError(
            f"Timestamp column '{timestamp_col}' is numeric. OpenMeasure "
            "does not guess whether numeric timestamps are seconds, "
            "milliseconds, or microseconds since the epoch, because "
            "guessing wrong silently shifts every result. Convert the "
            "column to datetime strings before uploading."
        )


def _parse_timestamps(column: pd.Series, timestamp_col: str) -> pd.Series:
    """
    Parse a timestamp column, rejecting mixed UTC offsets.

    Mixed offsets are refused rather than normalized. pandas 3 raises on
    them; pandas 2 returns an object-dtype Index that silently breaks
    diffing and grid matching. Converting to UTC on the user's behalf can
    also collapse two distinct input strings into the same instant,
    manufacturing a duplicate timestamp that was not in the data.
    """

    mixed_offset_message = (
        f"Timestamp column '{timestamp_col}' contains mixed UTC offsets. "
        "OpenMeasure does not convert these automatically, because "
        "converting can merge two different input values into the same "
        "instant and create a duplicate that was not in your data. "
        "Normalize the column to a single offset or timezone before "
        "uploading."
    )

    try:
        parsed = pd.to_datetime(column, errors="coerce")

        # pandas infers one format from the first value, so a column mixing
        # "09:00:00" with "09:00:00.4" rejects every value whose
        # fractional-second precision differs from the first. Those are
        # valid timestamps, so retry per-element before giving up on them.
        # The retry is only accepted if it recovers strictly more rows,
        # which keeps the stricter single-format result when it is better.
        unresolved = parsed.isna() & column.notna()

        if unresolved.any():
            retried = pd.to_datetime(column, errors="coerce", format="mixed")

            if int(retried.notna().sum()) > int(parsed.notna().sum()):
                parsed = retried
    except ValueError as error:
        # pandas 3 raises for mixed offsets even with errors="coerce".
        raise ValueError(mixed_offset_message) from error

    # pandas 2 instead returns object dtype holding per-element timezones.
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        raise ValueError(mixed_offset_message)

    return parsed


def prepare_series(
    data: pd.DataFrame,
    timestamp_col: str,
    value_col: str,
) -> PreparedSeries:
    """
    Validate, parse, and chronologically sort a single long-format series.

    Parameters
    ----------
    data:
        Input dataset, one row per observation.
    timestamp_col:
        Column holding the observation timestamp.
    value_col:
        Column holding the observed value.

    Returns
    -------
    PreparedSeries
        Sorted timestamps and values, plus full accounting of excluded rows.

    Raises
    ------
    TypeError
        If data is not a pandas DataFrame.
    ValueError
        If a column is missing, the two columns are the same, the timestamp
        column is numeric, the timestamp column has mixed UTC offsets, or no
        rows have a usable timestamp.
    """

    _validate_columns(data, timestamp_col, value_col)

    raw_timestamps = data[timestamp_col]

    _reject_numeric_timestamps(raw_timestamps, timestamp_col)

    n_input_rows = int(len(data))
    n_null_timestamps = int(raw_timestamps.isna().sum())

    parsed = _parse_timestamps(raw_timestamps, timestamp_col)

    # A value that was present on input but did not parse is unparseable;
    # a value that was already missing on input is null. The two have
    # different causes and different fixes, so they are counted separately.
    unparseable_mask = parsed.isna() & ~raw_timestamps.isna()
    n_unparseable_timestamps = int(unparseable_mask.sum())

    example_unparseable = tuple(
        str(value)
        for value in raw_timestamps[unparseable_mask]
        .head(MAX_UNPARSEABLE_EXAMPLES)
        .tolist()
    )

    usable = pd.DataFrame(
        {
            "timestamp": parsed,
            "value": data[value_col].to_numpy(),
        }
    ).loc[parsed.notna().to_numpy()]

    n_rows_used = int(len(usable))

    if n_rows_used == 0:
        raise ValueError(
            "No rows remain after excluding missing and unparseable "
            f"timestamps ({n_null_timestamps} missing, "
            f"{n_unparseable_timestamps} unparseable, out of "
            f"{n_input_rows} rows)."
        )

    # Order must be assessed before sorting; recomputing it afterwards
    # would always report False.
    steps = usable["timestamp"].diff().dropna()
    n_out_of_order_steps = int((steps < pd.Timedelta(0)).sum())
    was_out_of_order = bool(n_out_of_order_steps > 0)

    # Stable sort so rows sharing a timestamp keep their input order,
    # which keeps duplicate reporting deterministic between runs.
    ordered = usable.sort_values("timestamp", kind="stable")

    timestamps = pd.DatetimeIndex(ordered["timestamp"])

    timezone = str(timestamps.tz) if timestamps.tz is not None else None

    return PreparedSeries(
        timestamps=timestamps,
        values=ordered["value"].reset_index(drop=True),
        timezone=timezone,
        n_input_rows=n_input_rows,
        n_null_timestamps=n_null_timestamps,
        n_unparseable_timestamps=n_unparseable_timestamps,
        example_unparseable=example_unparseable,
        n_rows_used=n_rows_used,
        was_out_of_order=was_out_of_order,
        n_out_of_order_steps=n_out_of_order_steps,
    )
