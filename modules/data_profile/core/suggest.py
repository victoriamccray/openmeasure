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

from .profile import (
    ROLE_CATEGORICAL,
    ROLE_CONTINUOUS,
    ROLE_DATETIME,
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
