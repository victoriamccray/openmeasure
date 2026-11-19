"""
Which fairness analysis a set of columns can actually support.

The two analyses this module offers are not two configurations of one
thing. A pre-model comparison asks whether a favorable outcome occurred
at different rates across groups, using only what was observed. A
post-model comparison asks whether a model's errors fall differently
across groups, and needs the model's decision recorded alongside the
outcome it was trying to predict. Presented as two configuration forms
one after the other, they looked like the same analysis with more
fields.

Nothing here inspects intent, and nothing could: a two-valued column
might be a model's decision or another thing that was observed, and no
property of the file distinguishes them. So every reading below is about
shape. It says which analyses these columns have the shape for, and
leaves what the columns mean to the reader, who is the only one who
knows.

The point of saying a path is unavailable is that it stays unavailable.
A dataset recording what happened to real people and no model's
predictions supports a pre-model comparison and not a post-model one.
The useful thing to show is that boundary, rather than a way to
manufacture predictions so every feature lights up.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.data_profile.core.profile import (  # noqa: E402
    ROLE_IDENTIFIER,
    DataProfile,
)

# A favorable/unfavorable outcome, or a model's decision, has exactly two
# values. Not "few": three values is a different thing, and treating it
# as binary would mean silently dropping or merging one of them.
BINARY_VALUES = 2

# Past this a column labels each row rather than grouping rows to compare
# across. Matches modules/data_profile/core/suggest.py's own ceiling, so
# what a portrait shows and what a path check accepts agree.
MAX_GROUP_LEVELS = 12

PATH_PRE_MODEL = "pre_model"
PATH_POST_MODEL = "post_model"


@dataclass(frozen=True)
class ColumnShapes:
    """Which columns have the shape for each role, by their values alone."""

    binary: tuple[str, ...]
    grouping: tuple[str, ...]
    scores: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisPath:
    """One fairness analysis, and whether these columns can support it."""

    id: str
    label: str
    question: str
    needs: tuple[str, ...]
    available: bool
    # Short enough to sit beside the path on a diagram. The two paths
    # word an absence differently on purpose: a pre-model comparison the
    # columns cannot support is a fact about the columns, while a
    # post-model one is a claim the data do not carry, and flattening
    # both into "unavailable" loses that.
    status_label: str
    explanation: str

    def __post_init__(self) -> None:
        for field_name in ("id", "label", "question", "status_label", "explanation"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"{self.id or 'A path'} is missing a value for "
                    f"'{field_name}'."
                )

        if not self.needs:
            raise ValueError(f"{self.id} names nothing it needs.")


@dataclass(frozen=True)
class PathReading:
    """Both paths, and the column shapes they were read from."""

    shapes: ColumnShapes
    pre_model: AnalysisPath
    post_model: AnalysisPath

    @property
    def paths(self) -> tuple[AnalysisPath, ...]:
        """Both paths in reading order, pre-model first."""
        return (self.pre_model, self.post_model)


# What a reader is told when a two-valued column is available to hold the
# model's decision. Kept separate from the availability check because it
# is the part the check cannot settle: shape says a column could hold a
# decision, and only the reader knows whether it does.
DECISION_PROVENANCE_NOTE = (
    "A two-valued column has the shape of a model's decision and the "
    "shape of anything else recorded with two values. Which one you have "
    "is yours to confirm, and the comparison means something different if "
    "it is the latter."
)


def _score_note(scores: tuple[str, ...]) -> str:
    """What to say about columns holding many distinct numeric values."""
    if not scores:
        return ""

    named = ", ".join(scores)
    subject = "holds" if len(scores) == 1 else "hold"

    return (
        f"{named} {subject} too many distinct values to be a decision, so "
        "it is a score. Turning a score into a decision means choosing a "
        "threshold, which changes who counts as favorably treated and is "
        "a modelling choice this page leaves to you."
    )


def column_shapes(profile: DataProfile) -> ColumnShapes:
    """
    Which columns could fill which role, read from their values.

    A column can appear in more than one list. A two-valued group column
    is both binary and grouping, which is exactly the case in this
    module's own sample, where the observed label, the predicted label
    and sex all have two values.

    Identifier-like columns are excluded from every list. A column with a
    distinct value per row is not an outcome, a grouping or a score, and
    offering it as any of them is how a picker ends up defaulting to
    something meaningless.
    """
    identifiers = set(profile.columns_with_role(ROLE_IDENTIFIER))
    usable = [column for column in profile.columns if column.name not in identifiers]

    return ColumnShapes(
        binary=tuple(
            column.name for column in usable if column.n_unique == BINARY_VALUES
        ),
        grouping=tuple(
            column.name
            for column in usable
            if BINARY_VALUES <= column.n_unique <= MAX_GROUP_LEVELS
        ),
        scores=tuple(
            column.name
            for column in usable
            if column.is_numeric and column.n_unique > MAX_GROUP_LEVELS
        ),
    )


def _can_fill(shapes: ColumnShapes, *, binary_needed: int) -> bool:
    """
    Whether distinct columns exist for every role a path requires.

    Checked as an assignment rather than by counting, because the lists
    overlap. Two two-valued columns where the only grouping candidate is
    one of them is a different situation from two where it is not, and
    counting the lists cannot tell those apart.
    """
    for chosen in permutations(shapes.binary, binary_needed):
        for group in shapes.grouping:
            if group not in chosen:
                return True

    return False


def read_paths(profile: DataProfile) -> PathReading:
    """
    Which fairness analyses these columns have the shape for.

    Availability is a statement about the columns, never about the
    reader's data being good or bad. A file recording what happened to
    real people and no model's predictions supports one of these two
    analyses, and that is a complete answer rather than a shortfall.
    """
    shapes = column_shapes(profile)

    pre_available = _can_fill(shapes, binary_needed=1)
    post_available = _can_fill(shapes, binary_needed=2)

    if pre_available:
        pre_explanation = (
            "A two-valued observed outcome and a separate grouping column "
            "are both present."
        )
    else:
        pre_explanation = (
            "Not available from these columns. This comparison needs an "
            "observed outcome with exactly two values and a separate "
            "column holding a small number of groups, and no two columns "
            "here have those shapes."
        )

    if post_available:
        post_explanation = (
            "A second two-valued column is present, so one can hold the "
            "model's decision alongside the observed outcome and the "
            "group. " + DECISION_PROVENANCE_NOTE
        )
    else:
        post_explanation = (
            "Not established from these data. Comparing a model's errors "
            "needs its decision recorded as its own two-valued column, "
            "separate from the observed outcome and from the group, and "
            "there is no third column of that shape here."
        )

    score_note = _score_note(shapes.scores)
    if score_note:
        post_explanation = f"{post_explanation} {score_note}"

    return PathReading(
        shapes=shapes,
        pre_model=AnalysisPath(
            id=PATH_PRE_MODEL,
            label="Pre-model",
            question=(
                "Did the observed outcome occur at different rates across "
                "groups?"
            ),
            needs=("Observed outcome", "Group"),
            available=pre_available,
            status_label=(
                "Available with these columns"
                if pre_available
                else "Not available from these columns"
            ),
            explanation=pre_explanation,
        ),
        post_model=AnalysisPath(
            id=PATH_POST_MODEL,
            label="Post-model",
            question="Do the model's errors fall differently across groups?",
            needs=("Observed outcome", "Group", "Model decision"),
            available=post_available,
            status_label=(
                "Available with these columns"
                if post_available
                else "Not established from these data"
            ),
            explanation=post_explanation,
        ),
    )
