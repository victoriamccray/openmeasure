"""
Suggest which OpenMeasure workflow might fit an uploaded file's shape.

A suggestion is a hint drawn only from column roles (see
modules/data_profile/core/profile.py), never a decision: it names the
column-shape pattern it matched and the workflow that shape fits, and it
never ranks or scores matches against each other. More than one workflow
can match the same shape (a categorical column next to a continuous one
fits both Impact Evaluation's group-comparison shape and, if there is
also a binary outcome, Fairness's group-disparity shape); when that
happens, every match is returned together, since picking one would
collapse a real, contested judgment (what the reader is actually asking)
into a single answer the column shapes alone cannot make.

Cross-Analysis Implications and Portfolio Impact Analysis are
deliberately not suggested here: the first reads already-recorded
analyses rather than a fresh upload, and the second expects a specific,
named-column evidence/claim shape this generic profiler cannot recognize
from dtypes alone.

This module can infer data structure; it cannot infer research intent or
methodological validity from structure alone. A categorical column next
to a continuous one does not establish that a group difference is
causally meaningful, and three or more low-cardinality numeric columns
does not establish that they form a coherent scale. Every function here
names the shape it matched, never the judgment that shape would still
require.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .profile import (
    ROLE_CATEGORICAL,
    ROLE_CONTINUOUS,
    ROLE_DATETIME,
    ROLE_IDENTIFIER,
    ROLES,
    DataProfile,
)

WORKFLOW_RELIABILITY = "Reliability"
WORKFLOW_TIME_SERIES_QA = "Time-Series QA"
WORKFLOW_IMPACT_EVALUATION = "Impact Evaluation"
WORKFLOW_FAIRNESS = "Fairness"

# Fewer numeric, low-cardinality columns than this is too weak a signal
# to distinguish "a multi-item scale" from a handful of unrelated
# categorical variables (e.g. age bracket, region, plan tier).
MIN_SCALE_ITEMS = 3
SCALE_ITEM_MAX_UNIQUE = 10

# A categorical column with this few levels reads as a binary outcome
# (e.g. approved/denied), the shape Fairness's favorable-rate comparison
# expects, rather than a multi-level group variable.
BINARY_OUTCOME_MAX_UNIQUE = 2


@dataclass(frozen=True)
class WorkflowSuggestion:
    """One workflow this file's shape fits, and why."""

    workflow: str
    reasoning: str
    matched_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.workflow:
            raise ValueError("A WorkflowSuggestion must name a workflow.")
        if not self.reasoning:
            raise ValueError(f"{self.workflow}'s suggestion is missing its reasoning.")
        if not self.matched_columns:
            raise ValueError(f"{self.workflow}'s suggestion names no matched columns.")


def _suggest_time_series_qa(profile: DataProfile) -> WorkflowSuggestion | None:
    datetime_columns = profile.columns_with_role(ROLE_DATETIME)

    if not datetime_columns:
        return None

    return WorkflowSuggestion(
        workflow=WORKFLOW_TIME_SERIES_QA,
        reasoning=(
            f"'{datetime_columns[0]}' looks like a timestamp column, the "
            "shape Time-Series QA checks for gaps, duplicates, and "
            "coverage."
        ),
        matched_columns=datetime_columns,
    )


def _suggest_reliability(profile: DataProfile) -> WorkflowSuggestion | None:
    scale_like = tuple(
        column.name
        for column in profile.columns
        if column.role == ROLE_CATEGORICAL
        and column.n_unique > BINARY_OUTCOME_MAX_UNIQUE
        and column.n_unique <= SCALE_ITEM_MAX_UNIQUE
        and any(token in column.dtype for token in ("int", "float"))
    )

    if len(scale_like) < MIN_SCALE_ITEMS:
        return None

    return WorkflowSuggestion(
        workflow=WORKFLOW_RELIABILITY,
        reasoning=(
            f"{len(scale_like)} numeric columns with few distinct values "
            "each look like scale items (e.g. a Likert survey), the "
            "shape Reliability checks for internal consistency."
        ),
        matched_columns=scale_like,
    )


def _suggest_impact_evaluation(profile: DataProfile) -> WorkflowSuggestion | None:
    categorical_columns = profile.columns_with_role(ROLE_CATEGORICAL)
    continuous_columns = profile.columns_with_role(ROLE_CONTINUOUS)

    if not categorical_columns or not continuous_columns:
        return None

    return WorkflowSuggestion(
        workflow=WORKFLOW_IMPACT_EVALUATION,
        reasoning=(
            f"'{categorical_columns[0]}' (a small set of categories) next "
            f"to '{continuous_columns[0]}' (a continuous measure) fits a "
            "group-comparison shape. Impact Evaluation tests whether the "
            "continuous column differs by group."
        ),
        matched_columns=(categorical_columns[0], continuous_columns[0]),
    )


def _suggest_fairness(profile: DataProfile) -> WorkflowSuggestion | None:
    categorical_columns = profile.columns_with_role(ROLE_CATEGORICAL)
    binary_columns = tuple(
        column.name
        for column in profile.columns
        if column.role == ROLE_CATEGORICAL and column.n_unique <= BINARY_OUTCOME_MAX_UNIQUE
    )

    # A single column that is itself the only categorical column can't
    # supply both a group and a binary outcome; at least two distinct
    # categorical columns are required.
    if len(categorical_columns) < 2 or not binary_columns:
        return None

    group_column = next(
        (name for name in categorical_columns if name not in binary_columns),
        categorical_columns[0],
    )
    outcome_column = next(name for name in binary_columns if name != group_column)

    return WorkflowSuggestion(
        workflow=WORKFLOW_FAIRNESS,
        reasoning=(
            f"'{group_column}' (a group) alongside '{outcome_column}' (a "
            "two-valued outcome) fits a group-and-outcome shape. "
            "Fairness compares a favorable-outcome rate across groups."
        ),
        matched_columns=(group_column, outcome_column),
    )


def suggest_workflows(profile: DataProfile) -> tuple[WorkflowSuggestion, ...]:
    """
    Every workflow this file's column shape fits, in a fixed order.

    Returns an empty tuple when nothing recognizable matched -- a caller
    must treat that as "state plainly that nothing matched," never as a
    reason to guess.
    """

    candidates = (
        _suggest_time_series_qa(profile),
        _suggest_reliability(profile),
        _suggest_impact_evaluation(profile),
        _suggest_fairness(profile),
    )

    return tuple(suggestion for suggestion in candidates if suggestion is not None)


# A column with only one or two distinct values is not something to
# compare as a quantity, and one with a great many is not a grouping.
MIN_OUTCOME_LEVELS = 3
MAX_GROUP_LEVELS = 12

_NUMERIC_DTYPE_MARKERS = ("int", "float")


def _is_numeric(column) -> bool:
    """Whether a column's recorded dtype is a numeric one."""
    return any(marker in column.dtype.lower() for marker in _NUMERIC_DTYPE_MARKERS)


def _pick(profile: DataProfile, options: Sequence[str], *tests) -> str | None:
    """
    First option satisfying the earliest test that any option satisfies.

    Tests are tried in order, each against every option, so a later test
    is a fallback for the whole list rather than for one column. Returns
    the first non-identifier option when no test matches, and the first
    option when every column looks like an identifier, since at that
    point there is no better answer available.
    """
    if not options:
        return None

    by_name = {column.name: column for column in profile.columns}
    identifiers = set(profile.columns_with_role(ROLE_IDENTIFIER))
    candidates = [name for name in options if name not in identifiers]

    for test in tests:
        for name in candidates:
            column = by_name.get(name)
            if column is not None and test(column):
                return name

    return candidates[0] if candidates else options[0]


def default_outcome_column(
    profile: DataProfile, options: Sequence[str]
) -> str | None:
    """
    Which column an outcome picker should open on.

    Prefers a numeric column with enough distinct values to be worth
    comparing as a quantity, which is also what
    modules/program_evaluation/core/recommend.py treats as continuous-like
    when it chooses a test. Note that this deliberately does not defer to
    the profile's own role guess: profile_dataframe marks a 1-to-5 Likert
    column categorical-like, while the recommender treats any numeric
    column with three or more distinct values as continuous-like. For
    choosing an outcome the recommender is the right authority, since it
    is what decides which test runs.

    Falls back to any numeric column, then to any non-identifier column,
    then to the first option. An identifier is never preferred and always
    still selectable.
    """
    return _pick(
        profile,
        options,
        lambda column: _is_numeric(column) and column.n_unique >= MIN_OUTCOME_LEVELS,
        _is_numeric,
    )


def default_group_column(
    profile: DataProfile, options: Sequence[str]
) -> str | None:
    """
    Which column a group picker should open on.

    Prefers a non-numeric column with at least two but not many distinct
    values, which is the shape of a real grouping variable. A numeric
    column with the same shape is the next choice, then any
    non-identifier column, then the first option.
    """
    def is_grouping(column) -> bool:
        return 2 <= column.n_unique <= MAX_GROUP_LEVELS

    return _pick(
        profile,
        options,
        lambda column: is_grouping(column) and not _is_numeric(column),
        is_grouping,
    )


# Prefixes that mark the two halves of a repeated measurement. Matched on
# a shared remainder ("pre_confidence" with "post_confidence"), never on
# the prefix alone, so "pretest_score" is not paired with "post_weight".
_BASELINE_PREFIXES = ("pre_", "pre", "baseline_", "baseline")
_FOLLOW_UP_PREFIXES = ("post_", "post", "followup_", "follow_up_")


def _strip_prefix(name: str, prefixes: Sequence[str]) -> str | None:
    """The remainder after the first matching prefix, or None."""
    lowered = name.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix) and len(lowered) > len(prefix):
            return lowered[len(prefix):].strip("_")
    return None


def default_prepost_columns(
    profile: DataProfile, options: Sequence[str]
) -> tuple[str | None, str | None]:
    """
    Which two columns a baseline/follow-up pair of pickers should open on.

    Prefers a genuine pair: two columns whose names share a remainder
    after a baseline prefix and a follow-up prefix respectively, so
    pre_confidence and post_confidence are matched to each other rather
    than each being picked independently. Without this the two pickers
    default to the first two numeric columns, which are usually not a
    repeated measure of the same thing at all.

    Falls back to two different outcome-like columns when no pair is
    found, and to (something, None) when only one column is available.
    """
    remainders: dict[str, str] = {}
    for name in options:
        remainder = _strip_prefix(name, _BASELINE_PREFIXES)
        if remainder:
            remainders.setdefault(remainder, name)

    for name in options:
        remainder = _strip_prefix(name, _FOLLOW_UP_PREFIXES)
        if remainder and remainder in remainders:
            baseline = remainders[remainder]
            if baseline != name:
                return baseline, name

    baseline = default_outcome_column(profile, options)
    remaining = [name for name in options if name != baseline]
    return baseline, default_outcome_column(profile, remaining)
