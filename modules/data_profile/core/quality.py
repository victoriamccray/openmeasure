"""
Quality flags for an uploaded dataset: structural issues worth a second
look before choosing which columns feed a workflow.

Every flag names a mechanism (what was found, and where), never a
verdict: nothing here decides that a dataset is unusable, only that a
specific column or set of rows is worth a reader's attention before they
build on it. Column-role judgments (is this an outcome, a group) still
belong to whichever module the reader ends up in; this only looks at
shape.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .profile import DataProfile

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.validation import validate_is_dataframe  # noqa: E402

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"

SEVERITIES: tuple[str, ...] = (SEVERITY_INFO, SEVERITY_WARNING)

# A column missing more than this share of its values is flagged, not
# blocked -- a defensible column can still legitimately be this sparse.
HIGH_MISSINGNESS_THRESHOLD_PCT = 30.0


@dataclass(frozen=True)
class QualityFlag:
    """One structural issue. column is None for a dataset-level flag."""

    column: str | None
    message: str
    severity: str

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"'{self.severity}' is not a known severity. Valid "
                f"severities: {', '.join(SEVERITIES)}."
            )
        if not self.message:
            raise ValueError("A QualityFlag must have a message.")


def _shape_matches(data: pd.DataFrame, profile: DataProfile) -> bool:
    return profile.n_rows == len(data) and profile.n_columns == len(data.columns)


def _column_flags(data: pd.DataFrame, profile: DataProfile) -> list[QualityFlag]:
    flags: list[QualityFlag] = []

    for column in profile.columns:
        all_missing = column.n_missing == profile.n_rows

        if all_missing:
            flags.append(
                QualityFlag(
                    column.name,
                    "Every value is missing.",
                    SEVERITY_WARNING,
                )
            )
            continue  # A fully missing column is not also "constant."

        if column.n_unique <= 1:
            flags.append(
                QualityFlag(
                    column.name,
                    "Every non-missing value is the same.",
                    SEVERITY_INFO,
                )
            )

        if column.pct_missing >= HIGH_MISSINGNESS_THRESHOLD_PCT:
            flags.append(
                QualityFlag(
                    column.name,
                    f"{column.pct_missing:.0f}% of values are missing.",
                    SEVERITY_WARNING,
                )
            )

    return flags


def _duplicate_row_flag(data: pd.DataFrame) -> QualityFlag | None:
    n_duplicates = int(data.duplicated().sum())

    if n_duplicates == 0:
        return None

    return QualityFlag(
        None,
        f"{n_duplicates} duplicate row(s) found (identical across every column).",
        SEVERITY_WARNING,
    )


def quality_flags(data: pd.DataFrame, profile: DataProfile) -> tuple[QualityFlag, ...]:
    """
    Every structural quality flag for data, given its already-computed
    profile. Raises if profile does not describe data's actual shape, so
    a stale profile from a previous upload can never be paired silently
    with a new file.
    """

    validate_is_dataframe(data)

    if not _shape_matches(data, profile):
        raise ValueError(
            f"profile describes {profile.n_rows} rows x {profile.n_columns} "
            f"columns, but data has {len(data)} rows x {len(data.columns)} "
            "columns."
        )

    flags = _column_flags(data, profile)

    duplicate_flag = _duplicate_row_flag(data)
    if duplicate_flag is not None:
        flags.append(duplicate_flag)

    return tuple(flags)
