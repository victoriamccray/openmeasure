"""
Recording analysis results so they can be compared across modules.

Each module page records what its analysis used and excluded. The
Cross-Analysis Implications page reads those records and reports them
together, so a user can see that one analysis kept 180 of 200 rows while
another kept 165.

This module deliberately does not import streamlit. It stores into any
MutableMapping, which st.session_state satisfies in the app and a plain
dict satisfies in tests. The staleness and grouping rules are the riskiest
logic in the feature, and they would otherwise be the only part that cannot
be unit tested.

Records hold primitives, never module result objects. A module's result
dataclass is a different class depending on how it was imported: the tests
import "from core import reliability" while a page imports
"modules.reliability.core.reliability", producing two distinct classes for
the same dataclass, so an isinstance check against one would fail against
the other. Pages translate their own results into primitives at record
time.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, MutableMapping

import pandas as pd


# Bumped when the shape of a stored record changes. Records written by an
# older version are discarded on read rather than misinterpreted.
HANDOFF_SCHEMA_VERSION = 1

# Where records live inside the mapping.
STORE_KEY = "openmeasure_handoff"

# What kind of thing was unavailable. These are not interchangeable and are
# never summed: a dropped row, an empty cell inside a row that was kept, and
# an observation that never arrived have no common denominator.
KIND_ROWS_DROPPED = "ROWS_DROPPED"
KIND_CELLS_EMPTY = "CELLS_EMPTY"
KIND_OBSERVATIONS_ABSENT = "OBSERVATIONS_ABSENT"

ALL_KINDS = (
    KIND_ROWS_DROPPED,
    KIND_CELLS_EMPTY,
    KIND_OBSERVATIONS_ABSENT,
)

_DIGEST_LENGTH = 12


@dataclass(frozen=True)
class DatasetFingerprint:
    """
    Identifies an uploaded dataset, not an analysis of it.

    Deliberately excludes column selections and post-exclusion row counts,
    so two analyses of the same upload share a fingerprint even when they
    use different columns.

    Used for grouping and display only. Nothing is gated on it: reporting
    what each analysis excluded is meaningful across separate uploads as
    long as the report says which upload each figure came from.
    """

    digest: str
    filename: str
    n_rows: int
    n_columns: int
    column_names: tuple[str, ...]

    @property
    def short_digest(self) -> str:
        """First few characters of the digest, for display."""

        return self.digest[:8]

    @property
    def label(self) -> str:
        """Human-readable identifier for grouping in a report."""

        return f"{self.filename} ({self.short_digest})"


@dataclass(frozen=True)
class RetentionItem:
    """One reason some part of the data was unavailable."""

    label: str
    count: int
    kind: str
    mechanism: str


@dataclass(frozen=True)
class ExclusionAccount:
    """
    What one analysis used, and what it left out.

    Most analyses report n_retained_rows: participants kept out of those
    received. A row-expanding analysis cannot. The multi-select sensitivity
    analysis emits one row per selection, so a participant who selected two
    categories becomes two observations, and there is no single number of
    "retained participants" to report.

    Such an analysis therefore sets rows_can_exceed_participants, leaves
    n_retained_rows as None, and reports n_expanded_observations instead.
    Reporting a retained count for it would either invent a figure or
    silently claim that nothing was excluded.
    """

    module: str
    analysis_label: str
    columns_considered: tuple[str, ...]
    n_input_rows: int
    n_retained_rows: int | None = None
    n_expanded_observations: int | None = None
    items: tuple[RetentionItem, ...] = ()
    rows_can_exceed_participants: bool = False

    def __post_init__(self) -> None:
        if self.n_input_rows < 0:
            raise ValueError(
                f"n_input_rows cannot be negative; got {self.n_input_rows}."
            )

        if self.rows_can_exceed_participants:
            if self.n_expanded_observations is None:
                raise ValueError(
                    f"{self.analysis_label} expands rows, so it must report "
                    "n_expanded_observations."
                )
            if self.n_expanded_observations < 0:
                raise ValueError(
                    f"n_expanded_observations cannot be negative; got "
                    f"{self.n_expanded_observations}."
                )
            if self.n_retained_rows is not None:
                raise ValueError(
                    f"{self.analysis_label} expands rows, so it has no "
                    "count of retained participants and must leave "
                    "n_retained_rows as None."
                )
        else:
            if self.n_retained_rows is None:
                raise ValueError(
                    f"{self.analysis_label} must report n_retained_rows, or "
                    "declare rows_can_exceed_participants and report "
                    "n_expanded_observations instead."
                )
            if self.n_retained_rows < 0:
                raise ValueError(
                    f"n_retained_rows cannot be negative; got "
                    f"{self.n_retained_rows}."
                )
            if self.n_retained_rows > self.n_input_rows:
                raise ValueError(
                    f"{self.analysis_label} reports keeping "
                    f"{self.n_retained_rows} rows out of "
                    f"{self.n_input_rows}. An analysis cannot retain more "
                    "rows than it received unless it expands rows, which "
                    "must be declared with rows_can_exceed_participants."
                )

        for item in self.items:
            if item.count < 0:
                raise ValueError(
                    f"Retention item '{item.label}' has a negative count "
                    f"({item.count})."
                )
            if item.kind not in ALL_KINDS:
                raise ValueError(
                    f"Retention item '{item.label}' has unknown kind "
                    f"'{item.kind}'. Valid kinds: {', '.join(ALL_KINDS)}."
                )

    @property
    def n_excluded_rows(self) -> int | None:
        """
        Participants received but not used.

        None for a row-expanding analysis, which has no retained-participant
        count to subtract from.
        """

        if self.n_retained_rows is None:
            return None

        return self.n_input_rows - self.n_retained_rows


@dataclass(frozen=True)
class HandoffEntry:
    """One module's recorded analysis."""

    module: str
    fingerprint: DatasetFingerprint
    exclusion: ExclusionAccount
    primary_statistics: Mapping[str, float] = field(default_factory=dict)
    schema_version: int = HANDOFF_SCHEMA_VERSION
    recorded_at: str = ""
    sequence: int = 0


def fingerprint_dataframe(
    data: pd.DataFrame,
    filename: str = "uploaded data",
) -> DatasetFingerprint:
    """
    Fingerprint an uploaded dataset.

    The digest covers cell values, column names, and dtypes. It is stable
    across re-exports that change only formatting, which raw byte hashing is
    not, and it changes when any value changes.
    """

    row_hashes = pd.util.hash_pandas_object(data, index=False).to_numpy()

    hasher = hashlib.sha256()
    hasher.update(row_hashes.tobytes())
    hasher.update("|".join(map(str, data.columns)).encode("utf-8"))
    hasher.update("|".join(str(dtype) for dtype in data.dtypes).encode("utf-8"))

    return DatasetFingerprint(
        digest=hasher.hexdigest()[:_DIGEST_LENGTH],
        filename=filename,
        n_rows=int(data.shape[0]),
        n_columns=int(data.shape[1]),
        column_names=tuple(str(column) for column in data.columns),
    )


class HandoffStore:
    """
    Records module results in a caller-supplied mapping.

    One record per module: re-running a module replaces its previous record,
    because a stale record from an earlier run of the same module is never
    what the user wants to see.
    """

    def __init__(self, mapping: MutableMapping) -> None:
        self._mapping = mapping

    def _entries(self) -> dict[str, HandoffEntry]:
        stored = self._mapping.get(STORE_KEY)

        if not isinstance(stored, dict):
            return {}

        # Discard anything written by a different schema rather than risk
        # reading fields that have since changed meaning.
        return {
            module: entry
            for module, entry in stored.items()
            if isinstance(entry, HandoffEntry)
            and entry.schema_version == HANDOFF_SCHEMA_VERSION
        }

    def record(
        self,
        module: str,
        fingerprint: DatasetFingerprint,
        exclusion: ExclusionAccount,
        primary_statistics: Mapping[str, float] | None = None,
    ) -> HandoffEntry:
        """Record one module's analysis, replacing any earlier record."""

        entries = self._entries()

        entry = HandoffEntry(
            module=module,
            fingerprint=fingerprint,
            exclusion=exclusion,
            primary_statistics=dict(primary_statistics or {}),
            schema_version=HANDOFF_SCHEMA_VERSION,
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            sequence=self._next_sequence(entries),
        )

        entries[module] = entry
        self._mapping[STORE_KEY] = entries

        return entry

    @staticmethod
    def _next_sequence(entries: Mapping[str, HandoffEntry]) -> int:
        """Recording order, so a report can show which analysis ran first."""

        if not entries:
            return 1

        return max(entry.sequence for entry in entries.values()) + 1

    def entries(self) -> tuple[HandoffEntry, ...]:
        """Every valid record, in the order it was recorded."""

        return tuple(
            sorted(self._entries().values(), key=lambda entry: entry.sequence)
        )

    def has(self, module: str) -> bool:
        """Whether a module has a valid record."""

        return module in self._entries()

    def clear(self) -> None:
        """Discard every record."""

        self._mapping.pop(STORE_KEY, None)


def group_by_dataset(
    entries: tuple[HandoffEntry, ...],
) -> dict[str, tuple[HandoffEntry, ...]]:
    """
    Group records by the dataset they came from.

    Keyed by fingerprint digest. Analyses of different uploads must be
    reported separately, because their row counts describe different data
    and comparing them directly would be misleading.
    """

    grouped: dict[str, list[HandoffEntry]] = {}

    for entry in entries:
        grouped.setdefault(entry.fingerprint.digest, []).append(entry)

    return {
        digest: tuple(sorted(group, key=lambda entry: entry.sequence))
        for digest, group in grouped.items()
    }
