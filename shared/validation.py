"""Domain-neutral validation helper duplicated across module core packages."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def validate_is_dataframe(data: object) -> None:
    """Raise TypeError if data is not a pandas DataFrame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be provided as a pandas DataFrame.")


def format_value_sample(values: Sequence[object], limit: int = 8) -> str:
    """
    Render a column's distinct values for an error message, truncated.

    Selecting a continuous column (a predicted probability, an ID, a raw
    measurement) where a binary or categorical one is expected is a
    common mistake, and the resulting "found N distinct values" error is
    the main way a user finds out. Interpolating the full value list
    directly, as modules/fairness/core did before this helper existed,
    produces an unreadable wall of numbers once N is more than a
    handful; this shows only the first `limit` and states how many more
    there are, so the message stays a message rather than a data dump.
    """

    shown = list(values[:limit])
    remainder = len(values) - len(shown)

    if remainder <= 0:
        return str(shown)

    return f"{shown} and {remainder} more"
