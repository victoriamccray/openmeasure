"""
Every expected observation, and what became of it.

The module's checks were each correct and each reported separately: a
gaps table, a duplicate plot, a coverage table, a coverage chart, and the
series itself several pages later. A reader had to hold five views in
their head to answer the first question anyone asks of a series, which is
where it is healthy and where it breaks.

This is that question as one object. One slot per expected observation,
each carrying what happened at it, so the shape of the series' health is
something to look at rather than something to reconstruct.

The distinction the module is built on survives here. An observation that
is ABSENT has no row: nothing was recorded for that moment. An
observation that is EMPTY has a row whose value is blank: something was
recorded, and what it recorded was nothing. Those have different causes
and different fixes, and collapsing them into "missing" would throw away
the module's strongest idea to make a prettier picture.

Nothing here re-runs a check. It reads what temporal.py and
completeness.py already found and places each finding on the axis it
happened on.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# What became of one expected observation. Ordered by how much they
# matter when several land in the same bin: a conflicting duplicate is
# worse news than a plain duplicate, which is worse than jitter.
SLOT_PRESENT = "Present"
SLOT_EMPTY = "Row present, value blank"
SLOT_DUPLICATE = "Duplicate timestamp"
SLOT_CONFLICTING = "Duplicate timestamp, values disagree"
SLOT_ABSENT = "No row for this expected time"

# Worst last, so binning a stretch reports the most serious thing in it.
SLOT_SEVERITY: tuple[str, ...] = (
    SLOT_PRESENT,
    SLOT_DUPLICATE,
    SLOT_EMPTY,
    SLOT_CONFLICTING,
    SLOT_ABSENT,
)

# Jitter is counted and not drawn per slot.
#
# The module absorbs offsets inside its tolerance during grid matching:
# an observation arriving 58 minutes after the last one on an hourly
# series is tolerated, not faulted. Drawing each as its own state made 87
# of this module's own sample's 118 observations look defective, on a
# series the same checks call 98% regular. The count belongs in the
# summary, where it says what it is.

# Past this many slots a mark per observation is thinner than a hairline,
# so slots are binned and each bin reports the worst state inside it. The
# count of bins is what a reader sees; the counts underneath stay exact.
MAX_DRAWN_SLOTS = 180


@dataclass(frozen=True)
class Slot:
    """One expected observation, and what became of it."""

    start: pd.Timestamp
    end: pd.Timestamp
    state: str
    n_observations: int

    @property
    def is_healthy(self) -> bool:
        """Whether anything at all was wrong here."""
        return self.state == SLOT_PRESENT


@dataclass(frozen=True)
class TimelineHealth:
    """The whole series' health, as slots and as counts."""

    slots: tuple[Slot, ...]
    binned: bool
    observations_per_slot: int

    n_absent: int
    n_empty: int
    n_duplicate_timestamps: int
    n_conflicting_duplicates: int
    n_out_of_order_steps: int
    n_jittered: int
    regularity: float | None

    reason_not_assessable: str

    @property
    def is_assessable(self) -> bool:
        """Whether an expected schedule could be established at all."""
        return bool(self.slots)

    def summary_parts(self) -> tuple[str, ...]:
        """
        The compact line under the timeline, in a fixed order.

        Absent and empty are named separately and always both, even at
        zero, because a reader comparing two series needs to see which of
        the two a number refers to without counting positions.
        """
        # Conflicting duplicates are a subset of duplicates, so they are
        # reported inside that count rather than beside it. Listing both
        # separately read as two problems where there is one.
        duplicates = f"{self.n_duplicate_timestamps} duplicate"
        if self.n_conflicting_duplicates:
            duplicates += f" ({self.n_conflicting_duplicates} conflicting)"

        parts = [
            f"{self.n_absent} absent",
            f"{self.n_empty} empty",
            duplicates,
        ]

        if self.n_out_of_order_steps:
            parts.append(f"{self.n_out_of_order_steps} out of order")
        if self.n_jittered:
            parts.append(f"{self.n_jittered} off-schedule")
        if self.regularity is not None:
            parts.append(f"{self.regularity:.0%} regular")

        return tuple(parts)


def _worst(states) -> str:
    """The most serious state among several, for a binned stretch."""
    return max(states, key=SLOT_SEVERITY.index)


def build_timeline(result) -> TimelineHealth:
    """
    Place every finding on the axis it happened on.

    Returns an empty slot tuple when no expected schedule could be
    established, which is a real situation rather than a failure: a series
    whose frequency is not defensible has no grid to compare against, and
    drawing one would invent the schedule the check declined to infer.
    """
    temporal = result.temporal
    completeness = result.completeness
    prepared = result.prepared

    counts = {
        "n_absent": temporal.n_missing_observations or 0,
        "n_empty": completeness.n_missing_values,
        "n_duplicate_timestamps": temporal.n_distinct_duplicated_timestamps,
        "n_conflicting_duplicates": temporal.n_conflicting_duplicate_timestamps,
        "n_out_of_order_steps": temporal.n_out_of_order_steps,
        "n_jittered": temporal.n_jittered_observations or 0,
        "regularity": temporal.modal_interval_share
        if temporal.regularity_assessable
        else None,
    }

    if not temporal.gaps_assessable or not temporal.n_expected_observations:
        return TimelineHealth(
            slots=(),
            binned=False,
            observations_per_slot=0,
            reason_not_assessable=(
                temporal.reason_not_assessable
                or "No expected schedule could be established for this series."
            ),
            **counts,
        )

    timestamps = prepared.timestamps
    values = prepared.values

    # Which observed timestamps carry a duplicate, a conflict, or a blank.
    # Read from what temporal.py and completeness.py already found rather
    # than recomputed: a second matching pass here would be a second
    # opinion, and an earlier version of it invented jitter by comparing
    # observations against an evenly spaced grid this module never uses.
    conflicting = {
        finding.timestamp
        for finding in temporal.duplicates
        if finding.has_conflicting_values
    }
    duplicated = {
        finding.timestamp
        for finding in temporal.duplicates
        if not finding.has_conflicting_values
    }
    blank = {
        timestamps[position]
        for position in range(len(timestamps))
        if pd.isna(values.iloc[position])
    }

    def _state(moment) -> str:
        if moment in conflicting:
            return SLOT_CONFLICTING
        if moment in blank:
            return SLOT_EMPTY
        if moment in duplicated:
            return SLOT_DUPLICATE
        return SLOT_PRESENT

    # Absent observations are placed where the gaps say they fall, so the
    # timeline shows where a series stops rather than only how much of it
    # is missing.
    absent_after = {
        finding.gap_start: finding.n_expected_missing
        for finding in temporal.gaps
    }

    raw: list[Slot] = []
    seen: set = set()

    for moment in timestamps:
        if moment in seen:
            continue
        seen.add(moment)

        raw.append(
            Slot(
                start=moment,
                end=moment,
                state=_state(moment),
                n_observations=1,
            )
        )

        for _ in range(absent_after.get(moment, 0)):
            raw.append(
                Slot(
                    start=moment,
                    end=moment,
                    state=SLOT_ABSENT,
                    n_observations=1,
                )
            )

    if len(raw) <= MAX_DRAWN_SLOTS:
        return TimelineHealth(
            slots=tuple(raw),
            binned=False,
            observations_per_slot=1,
            reason_not_assessable="",
            **counts,
        )

    per_bin = -(-len(raw) // MAX_DRAWN_SLOTS)
    binned: list[Slot] = []

    for start in range(0, len(raw), per_bin):
        chunk = raw[start : start + per_bin]
        binned.append(
            Slot(
                start=chunk[0].start,
                end=chunk[-1].end,
                state=_worst(slot.state for slot in chunk),
                n_observations=len(chunk),
            )
        )

    return TimelineHealth(
        slots=tuple(binned),
        binned=True,
        observations_per_slot=per_bin,
        reason_not_assessable="",
        **counts,
    )
