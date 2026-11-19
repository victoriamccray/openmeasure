"""
Column and dataset structure profiling.

Runs immediately after a file loads, before any workflow-specific setup
(picking an outcome column, a timestamp column, ...) happens. Gives a
reader a summary of what their upload actually contains, and gives each
page's own column-picking UI a role guess to default to.

A role guess is a hint, never a decision: it is a plain string label
attached to a column, surfaced to a reader as "this looks like a
datetime column," and it never blocks, filters, or silently drops a
column from anything a page still lets the reader choose. The identifier
heuristic mirrors modules/program_evaluation/core/recommend.py's
_looks_like_identifier (unique values, id-suggesting name or sequential
integers); it is reimplemented here rather than imported, since data_profile
runs before any module-specific context (an outcome, a group) exists to
apply program_evaluation's heuristic to.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.validation import validate_is_dataframe  # noqa: E402

ROLE_IDENTIFIER = "identifier-like"
ROLE_DATETIME = "datetime-like"
ROLE_CATEGORICAL = "categorical-like"
ROLE_CONTINUOUS = "continuous-like"
ROLE_TEXT = "free text"

ROLES: tuple[str, ...] = (
    ROLE_IDENTIFIER,
    ROLE_DATETIME,
    ROLE_CATEGORICAL,
    ROLE_CONTINUOUS,
    ROLE_TEXT,
)

_ID_NAME_PATTERN = re.compile(r"(^|_)(id|index|key|uuid)($|_)", re.IGNORECASE)

# A numeric column with this few distinct values reads as categorical
# (e.g. a 1-5 Likert item, a 0/1 flag) rather than continuous. Matches
# program_evaluation's own categorical/binary heuristics in spirit, but
# generalized past exactly two values since this runs with no outcome
# column singled out yet.
CATEGORICAL_MAX_UNIQUE = 10

# Said wherever the role guess is shown as a grouping rather than as one
# cell in a table, because a grouping looks like a ruling on where a
# column may go and this one is not.
#
# The threshold is interpolated rather than written out, so the sentence
# cannot come to disagree with the number it describes. A test holds it
# to that.
LOW_CARDINALITY_NOTE = (
    f"A numeric column with {CATEGORICAL_MAX_UNIQUE} or fewer distinct "
    "values is grouped as categorical-like. That is read off the count of "
    "distinct values, not a restriction on the column: a rating scale is "
    "often analysed as a continuous outcome, and every column stays "
    "selectable whichever group it appears in."
)

# A nonnumeric column with more distinct values than this, relative to
# row count, reads as free text (e.g. an open-ended response) rather
# than a category to group by.
CATEGORICAL_MAX_UNIQUE_RATIO = 0.5

# How many nonmissing values to try parsing as a date before deciding a
# nonnumeric column is datetime-like. Small and fixed, so profiling one
# column never becomes a full-column parse of a large upload.
_DATETIME_PROBE_SIZE = 20
_DATETIME_SUCCESS_THRESHOLD = 0.8


@dataclass(frozen=True)
class ColumnProfile:
    """One column's shape: how much data it has, and a role guess."""

    name: str
    dtype: str
    n_missing: int
    pct_missing: float
    n_unique: int
    role: str

    @property
    def is_numeric(self) -> bool:
        """
        Whether the column holds numbers, read off the dtype recorded.

        A property rather than a stored field so it cannot disagree with
        the dtype beside it, and so callers that build a ColumnProfile by
        hand do not have to remember to keep the two in step.

        Distinct from the role guess: role puts a numeric column with few
        distinct values in categorical-like, which is a reading of the
        column, while this is a fact about how it is stored.
        """
        return pd.api.types.is_numeric_dtype(pd.api.types.pandas_dtype(self.dtype))

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(
                f"'{self.role}' is not a known role. Valid roles: "
                f"{', '.join(ROLES)}."
            )
        if self.n_missing < 0:
            raise ValueError(f"n_missing cannot be negative: {self.n_missing}.")
        if not 0.0 <= self.pct_missing <= 100.0:
            raise ValueError(f"pct_missing must be within 0-100: {self.pct_missing}.")


@dataclass(frozen=True)
class DataProfile:
    """A dataset's shape: row/column counts and every column's profile."""

    n_rows: int
    n_columns: int
    columns: tuple[ColumnProfile, ...]

    def column(self, name: str) -> ColumnProfile:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"No column named '{name}' in this profile.")

    def columns_with_role(self, role: str) -> tuple[str, ...]:
        if role not in ROLES:
            raise ValueError(
                f"'{role}' is not a known role. Valid roles: {', '.join(ROLES)}."
            )
        return tuple(c.name for c in self.columns if c.role == role)

    # Counts of what is in the dataset, as facts rather than as readings
    # of it. A summary built out of role guesses ("no continuous
    # columns") states a heuristic more confidently than the heuristic
    # supports; these four say how the data is stored and how much of it
    # is there, which is true whatever anyone decides the roles are.

    @property
    def n_numeric(self) -> int:
        """How many columns hold numbers."""
        return sum(1 for column in self.columns if column.is_numeric)

    @property
    def n_low_cardinality_numeric(self) -> int:
        """
        Numeric columns with few enough distinct values to read either
        way, as a Likert item or a count does.

        A subset of n_numeric, not a category beside it. These are the
        columns where the role guess is a judgement call rather than a
        formality, which is exactly what a reader should be told before
        being asked to pick an outcome.
        """
        return sum(
            1
            for column in self.columns
            if column.is_numeric and column.n_unique <= CATEGORICAL_MAX_UNIQUE
        )

    @property
    def n_non_numeric(self) -> int:
        """How many columns hold text, categories, dates, or anything else."""
        return self.n_columns - self.n_numeric

    @property
    def n_missing_cells(self) -> int:
        """How many values are absent across the whole dataset."""
        return sum(column.n_missing for column in self.columns)


def _looks_like_identifier(series: pd.Series, column_name: str) -> bool:
    clean = series.dropna()

    if clean.empty or clean.nunique() != len(clean):
        return False

    name_suggests_id = bool(_ID_NAME_PATTERN.search(str(column_name)))

    is_sequential = False
    if pd.api.types.is_numeric_dtype(clean):
        try:
            as_int = clean.astype(int)
            if (as_int == clean).all():
                sorted_vals = sorted(as_int.tolist())
                is_sequential = sorted_vals == list(
                    range(sorted_vals[0], sorted_vals[0] + len(sorted_vals))
                )
        except (ValueError, OverflowError):
            is_sequential = False

    return name_suggests_id or is_sequential


def _looks_like_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    clean = series.dropna()

    if clean.empty or pd.api.types.is_numeric_dtype(clean):
        return False

    probe = clean.head(_DATETIME_PROBE_SIZE)
    parsed = pd.to_datetime(probe, errors="coerce", format="mixed")

    return (parsed.notna().mean() if len(parsed) else 0.0) >= _DATETIME_SUCCESS_THRESHOLD


def _guess_role(series: pd.Series, column_name: str) -> str:
    clean = series.dropna()

    if clean.empty:
        return ROLE_TEXT

    if _looks_like_identifier(series, column_name):
        return ROLE_IDENTIFIER

    if _looks_like_datetime(series):
        return ROLE_DATETIME

    if pd.api.types.is_numeric_dtype(clean):
        return ROLE_CATEGORICAL if clean.nunique() <= CATEGORICAL_MAX_UNIQUE else ROLE_CONTINUOUS

    unique_ratio = clean.nunique() / len(clean)
    return ROLE_CATEGORICAL if unique_ratio <= CATEGORICAL_MAX_UNIQUE_RATIO else ROLE_TEXT


def profile_column(series: pd.Series, name: str | None = None) -> ColumnProfile:
    """Profile one column. name overrides series.name (e.g. for an unnamed Series)."""

    column_name = name if name is not None else str(series.name)
    n_rows = len(series)
    n_missing = int(series.isna().sum())

    return ColumnProfile(
        name=column_name,
        dtype=str(series.dtype),
        n_missing=n_missing,
        pct_missing=(100.0 * n_missing / n_rows) if n_rows else 0.0,
        n_unique=int(series.nunique(dropna=True)),
        role=_guess_role(series, column_name),
    )


def profile_dataframe(data: pd.DataFrame) -> DataProfile:
    """Profile every column of an uploaded DataFrame."""

    validate_is_dataframe(data)

    if data.empty:
        raise ValueError("The dataset contains no rows.")

    columns = tuple(profile_column(data[col], name=col) for col in data.columns)

    return DataProfile(n_rows=len(data), n_columns=len(data.columns), columns=columns)
