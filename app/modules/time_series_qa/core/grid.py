"""
Expected-observation grid construction and matching.

Both the temporal-integrity and completeness checks need to decide the same
thing: which expected observations are present. They must decide it
identically. When completeness used exact matching while temporal matching
allowed jitter, jittered observations counted as present in one check and
absent in the other, so the two reports contradicted each other.

Matching is deliberately tolerant. An observation logged a few seconds off
its expected time is that observation, not a missing one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BaseOffset

from .frequency import FrequencyEstimate

# How far an observation may sit from its expected grid point and still
# count as present, as a fraction of the sampling interval. An OpenMeasure
# operational convention, not a published standard; it is surfaced in the
# report so results remain reproducible.
DEFAULT_JITTER_TOLERANCE_FRACTION = 0.05


def resolve_jitter_tolerance(
    frequency: FrequencyEstimate,
    override: pd.Timedelta | None = None,
) -> pd.Timedelta:
    """Resolve the grid-matching tolerance for a frequency estimate."""

    if override is not None:
        return override

    if frequency.modal_interval is None:
        return pd.Timedelta(0)

    return pd.Timedelta(
        frequency.modal_interval * DEFAULT_JITTER_TOLERANCE_FRACTION
    )


def build_grid(
    first: pd.Timestamp,
    last: pd.Timestamp,
    offset: BaseOffset,
    tz: str | None,
) -> pd.DatetimeIndex:
    """
    Generate the expected observation grid across the observed span.

    The grid is calendar-aware because it is generated from a DateOffset,
    so a monthly grid steps by real months and a daily grid absorbs
    daylight saving transitions.
    """

    return pd.date_range(start=first, end=last, freq=offset, tz=tz)


def grid_is_aligned(
    grid: pd.DatetimeIndex,
    first: pd.Timestamp,
    tolerance: pd.Timedelta,
) -> bool:
    """
    Return whether the generated grid actually lines up with the data.

    An anchored offset such as month-end snaps the grid to its own anchor.
    Asking date_range to start at the 15th with a month-end frequency
    yields month-ends, so every observation would be reported absent. The
    grid starts from the first observation, so if that observation is not on
    the grid the offset does not describe where the data actually falls.
    """

    return len(grid) > 0 and bool(abs(grid[0] - first) <= tolerance)


def match_to_grid(
    distinct: pd.DatetimeIndex,
    grid: pd.DatetimeIndex,
    tolerance: pd.Timedelta,
) -> np.ndarray:
    """
    Map each grid point to the observation that satisfies it.

    Returns an array parallel to grid holding, for each expected point, the
    position of the matching observation in distinct, or -1 when no
    observation falls within tolerance.

    distinct must be unique and sorted: pandas raises InvalidIndexError if a
    duplicated index reaches get_indexer.
    """

    return distinct.get_indexer(
        grid,
        method="nearest",
        tolerance=tolerance,
    )
