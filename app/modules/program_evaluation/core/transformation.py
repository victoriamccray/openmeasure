"""
How a pile of observations becomes an effect size, one step at a time.

A two-group result reports means, a difference, and Cohen's d as three
finished numbers. Where they came from is left implicit, and the step
readers most often lose is the last one: why a difference of 0.16 counts
as small, when the same 0.16 would be enormous on a different scale.

This describes that journey as five stages a reader steps through:

    observations -> groups -> means -> difference -> effect size

Pure, and it computes nothing the analysis did not already compute. The
means, the pooled standard deviation and d come from the result object;
what this adds is the per-observation values a chart needs to draw, the
stage sequence, and a headline per stage. Rendering is the page's, which
scales these values into pixels.

Jitter is deterministic. Points drawn at random offsets would jump on
every rerun, so a reader moving the stage control would see the cloud
reshuffle underneath them and read that as the data changing.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from shared.validation import validate_is_dataframe  # noqa: E402

from .comparison import TwoGroupResult  # noqa: E402
from .formulas import pooled_standard_deviation  # noqa: E402

STAGE_OBSERVATIONS = "observations"
STAGE_GROUPS = "groups"
STAGE_MEANS = "means"
STAGE_DIFFERENCE = "difference"
STAGE_EFFECT_SIZE = "effect_size"

STAGE_ORDER: tuple[str, ...] = (
    STAGE_OBSERVATIONS,
    STAGE_GROUPS,
    STAGE_MEANS,
    STAGE_DIFFERENCE,
    STAGE_EFFECT_SIZE,
)

# How far a point may be nudged off its row, as a fraction of the row's
# height, so overlapping values stay individually visible.
_JITTER_SPREAD = 0.8

# The step of the deterministic jitter sequence: the golden-ratio
# conjugate, whose successive multiples mod 1 fill the interval more
# evenly than any other constant. That is what keeps the offsets from
# falling into a visible repeating pattern at these sample sizes.
_JITTER_STEP = 0.6180339887498949


@dataclass(frozen=True)
class TransformationStage:
    """One step in the journey from observations to an effect size."""

    key: str
    label: str
    explanation: str
    headline: str

    def __post_init__(self) -> None:
        if self.key not in STAGE_ORDER:
            raise ValueError(
                f"'{self.key}' is not a known stage. Known stages: "
                f"{', '.join(STAGE_ORDER)}."
            )

        for field_name in ("label", "explanation", "headline"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"Stage '{self.key}' is missing a value for "
                    f"'{field_name}'."
                )


@dataclass(frozen=True)
class TransformationView:
    """
    Everything a chart needs to draw the five stages of one comparison.

    values_a and values_b are the observations actually used, in the
    order the data supplied them. jitter_a and jitter_b are matching
    offsets in the range -1 to 1, for spreading overlapping points off
    their row without moving them along the value axis.
    """

    label_a: str
    label_b: str
    values_a: tuple[float, ...]
    values_b: tuple[float, ...]
    jitter_a: tuple[float, ...]
    jitter_b: tuple[float, ...]
    mean_a: float
    mean_b: float
    pooled_sd: float
    mean_difference: float
    cohens_d: float
    stages: tuple[TransformationStage, ...]

    @property
    def all_values(self) -> tuple[float, ...]:
        """Every observation, for deciding one shared value axis."""
        return self.values_a + self.values_b

    def stage_index(self, key: str) -> int:
        """Where a stage falls in the sequence."""
        if key not in STAGE_ORDER:
            raise ValueError(
                f"'{key}' is not a known stage. Known stages: "
                f"{', '.join(STAGE_ORDER)}."
            )

        return STAGE_ORDER.index(key)


def _deterministic_jitter(count: int) -> tuple[float, ...]:
    """
    Offsets in -1..1 that look scattered but never change between runs.

    A low-discrepancy sequence rather than a random one: successive
    values spread evenly instead of clumping, and the same count always
    produces the same offsets, so stepping through the stages does not
    reshuffle the cloud.
    """
    offsets = []

    for index in range(count):
        position = ((index + 1) * _JITTER_STEP) % 1.0
        offsets.append((position * 2.0 - 1.0) * _JITTER_SPREAD)

    return tuple(offsets)


def _stages(
    result: TwoGroupResult,
    pooled_sd: float,
) -> tuple[TransformationStage, ...]:
    """The five stage descriptions, filled in from this comparison."""
    total = result.n_a + result.n_b

    return (
        TransformationStage(
            key=STAGE_OBSERVATIONS,
            label="Every observation",
            explanation=(
                "Each mark is one observation, placed by its outcome value. "
                "Nothing has been grouped or averaged yet."
            ),
            headline=f"{total} observations",
        ),
        TransformationStage(
            key=STAGE_GROUPS,
            label="Split into groups",
            explanation=(
                "The same marks, separated by which group each one belongs "
                "to. Any overlap between the rows is outcome values the two "
                "groups share."
            ),
            headline=(
                f"{result.n_a} {result.group_a_label}, "
                f"{result.n_b} {result.group_b_label}"
            ),
        ),
        TransformationStage(
            key=STAGE_MEANS,
            label="Average each group",
            explanation=(
                "One number now stands for each row. Everything the spread "
                "of the marks told you is set aside at this step, which is "
                "why it comes back at the last one."
            ),
            headline=f"{result.mean_a:.2f} and {result.mean_b:.2f}",
        ),
        TransformationStage(
            key=STAGE_DIFFERENCE,
            label="Take the difference",
            explanation=(
                "The gap between the two averages, in the outcome's own "
                "units. On its own this says nothing about whether the gap "
                "is large, because that depends on the scale."
            ),
            headline=f"difference of {result.mean_difference:.2f}",
        ),
        TransformationStage(
            key=STAGE_EFFECT_SIZE,
            label="Measure it against the spread",
            explanation=(
                "The same gap, measured in units of how much scores "
                "typically vary within a group. This is what makes it "
                "comparable to a difference measured on some other scale."
            ),
            headline=(
                f"{result.mean_difference:.2f} / {pooled_sd:.2f} = "
                f"{result.cohens_d:.2f}"
            ),
        ),
    )


def effect_size_transformation(
    data: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    result: TwoGroupResult,
) -> TransformationView:
    """
    Build the five-stage view for a two-group comparison.

    ``result`` must be the result of comparing these same columns: its
    means and effect size are used as computed rather than recalculated,
    so the chart cannot show a different number from the analysis. The
    observations are re-read here because a result keeps summaries, not
    the values themselves.

    Raises
    ------
    TypeError
        If ``data`` is not a DataFrame.
    ValueError
        If a column is missing, or if the rows kept for these columns do
        not match the counts the result reports, which would mean the
        chart is drawing a different set of observations from the one
        that was analyzed.
    """
    validate_is_dataframe(data)

    for column, role in ((group_col, "Group"), (outcome_col, "Outcome")):
        if column not in data.columns:
            raise ValueError(f"{role} column '{column}' not found in data.")

    clean = data[[group_col, outcome_col]].dropna()

    values_a = tuple(
        float(value)
        for value in clean.loc[clean[group_col] == result.group_a_label, outcome_col]
    )
    values_b = tuple(
        float(value)
        for value in clean.loc[clean[group_col] == result.group_b_label, outcome_col]
    )

    if len(values_a) != result.n_a or len(values_b) != result.n_b:
        raise ValueError(
            "The observations found for these columns do not match the "
            f"result: found {len(values_a)} and {len(values_b)}, the result "
            f"reports {result.n_a} and {result.n_b}. The chart would be "
            "drawing a different set of observations from the one analyzed."
        )

    pooled_sd = pooled_standard_deviation(result)

    if not math.isfinite(pooled_sd) or pooled_sd == 0:
        raise ValueError(
            "The pooled standard deviation is zero or undefined, so the "
            "final step has nothing to measure the difference against."
        )

    return TransformationView(
        label_a=result.group_a_label,
        label_b=result.group_b_label,
        values_a=values_a,
        values_b=values_b,
        jitter_a=_deterministic_jitter(len(values_a)),
        jitter_b=_deterministic_jitter(len(values_b)),
        mean_a=result.mean_a,
        mean_b=result.mean_b,
        pooled_sd=pooled_sd,
        mean_difference=result.mean_difference,
        cohens_d=result.cohens_d,
        stages=_stages(result, pooled_sd),
    )
