"""Domain-neutral validation helper duplicated across module core packages."""

from __future__ import annotations

import pandas as pd


def validate_is_dataframe(data: object) -> None:
    """Raise TypeError if data is not a pandas DataFrame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be provided as a pandas DataFrame.")
