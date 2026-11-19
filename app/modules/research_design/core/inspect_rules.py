"""
Deterministic, non-AI inspection rules for a user-entered study structure.

This is the "Enter your own" path's implications engine: given the
structural choices a reader makes about their own study (not the
built-in chronic-pain scenario simulate.py models), name what those
choices imply is worth checking, using fixed if/then rules rather than
any inference from data or free text. Each rule fires on structure
alone, the same way modules/data_profile/core/suggest.py's workflow
suggestions fire on a DataProfile's shape alone: a structural signal,
not a determination that a listed module is the right choice, or that
no other consideration applies.

StudyStructure is independent of DesignAssumptions: DesignAssumptions
fully parameterizes the one built-in pain simulation, while
StudyStructure describes an arbitrary user-entered study that v0.1
never simulates. Keeping them separate means extending "Enter your
own" cannot accidentally change what the worked simulation does.
"""

from __future__ import annotations

from dataclasses import dataclass

DESIGN_TYPE_OBSERVATIONAL = "Observational"
DESIGN_TYPE_EXPERIMENTAL = "Experimental"

COMPARISON_WITHIN_PERSON = "Within-person"
COMPARISON_BETWEEN_PERSON = "Between-person"

TIME_STRUCTURE_CROSS_SECTIONAL = "Cross-sectional"
TIME_STRUCTURE_LONGITUDINAL = "Longitudinal, repeated-measures"

MEASUREMENT_TYPE_SURVEY_SCALE = "Survey or self-report scale"
MEASUREMENT_TYPE_WEARABLE_SENSOR = "Wearable or sensor signal"
MEASUREMENT_TYPE_BODY_MAP = "Body map or spatial marking"
MEASUREMENT_TYPE_BEHAVIORAL_TASK = "Behavioral or cognitive task"
MEASUREMENT_TYPE_CLINICAL_RECORD = "Clinical or administrative record"
MEASUREMENT_TYPE_OTHER = "Other structured observation"

MEASUREMENT_TYPE_OPTIONS = (
    MEASUREMENT_TYPE_SURVEY_SCALE,
    MEASUREMENT_TYPE_WEARABLE_SENSOR,
    MEASUREMENT_TYPE_BODY_MAP,
    MEASUREMENT_TYPE_BEHAVIORAL_TASK,
    MEASUREMENT_TYPE_CLINICAL_RECORD,
    MEASUREMENT_TYPE_OTHER,
)


@dataclass(frozen=True)
class StudyStructure:
    """
    The structural choices behind an arbitrary, user-entered study:
    everything inspect_study_structure() needs, and nothing about the
    research question's actual subject matter, which these rules never
    read.

    measurement_types: zero or more of MEASUREMENT_TYPE_OPTIONS: empty
        is allowed (a reader has not chosen yet), in which case no
        measurement-type-driven rule fires.
    compares_subgroups: whether the design compares demographic
        subgroups (e.g. age, gender, race), stated directly rather than
        inferred from population/outcomes text, since guessing that
        from free text would be exactly the kind of automation this
        toolkit avoids.
    """

    design_type: str
    comparison_structure: str
    time_structure: str
    measurement_types: tuple[str, ...]
    compares_subgroups: bool

    def __post_init__(self) -> None:
        if self.design_type not in (DESIGN_TYPE_OBSERVATIONAL, DESIGN_TYPE_EXPERIMENTAL):
            raise ValueError(
                f"design_type must be one of {DESIGN_TYPE_OBSERVATIONAL!r}, "
                f"{DESIGN_TYPE_EXPERIMENTAL!r}; got {self.design_type!r}."
            )
        if self.comparison_structure not in (COMPARISON_WITHIN_PERSON, COMPARISON_BETWEEN_PERSON):
            raise ValueError(
                f"comparison_structure must be one of {COMPARISON_WITHIN_PERSON!r}, "
                f"{COMPARISON_BETWEEN_PERSON!r}; got {self.comparison_structure!r}."
            )
        if self.time_structure not in (TIME_STRUCTURE_CROSS_SECTIONAL, TIME_STRUCTURE_LONGITUDINAL):
            raise ValueError(
                f"time_structure must be one of {TIME_STRUCTURE_CROSS_SECTIONAL!r}, "
                f"{TIME_STRUCTURE_LONGITUDINAL!r}; got {self.time_structure!r}."
            )
        unknown = set(self.measurement_types) - set(MEASUREMENT_TYPE_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown measurement_types: {sorted(unknown)}.")


@dataclass(frozen=True)
class Inspection:
    """
    One thing to inspect, not a recommendation: trigger names the
    structural fact that fired it, note states what to check, and
    suggested_module optionally names an existing OpenMeasure page
    where that check can actually be done.
    """

    trigger: str
    note: str
    suggested_module: str | None = None


def inspect_study_structure(structure: StudyStructure) -> tuple[Inspection, ...]:
    """
    Apply fixed, structure-only rules to a user-entered study.

    Every rule here is a direct if/then on StudyStructure's fields, in
    the same spirit as modules/data_profile/core/suggest.py: no
    weighting, no score, no ranking by relevance, just a list of
    structural facts worth checking against the actual research
    question, which these rules never see.
    """

    inspections: list[Inspection] = []

    if structure.design_type == DESIGN_TYPE_EXPERIMENTAL and structure.comparison_structure == COMPARISON_BETWEEN_PERSON:
        inspections.append(
            Inspection(
                trigger="Experimental, between-person comparison",
                note=(
                    "How participants are assigned to groups, and whether "
                    "the groups are comparable at baseline, becomes "
                    "central to whether a between-group difference can be "
                    "attributed to the intervention."
                ),
            )
        )

    if structure.time_structure == TIME_STRUCTURE_LONGITUDINAL:
        inspections.append(
            Inspection(
                trigger="Longitudinal, repeated-measures",
                note=(
                    "Repeated observations from the same participants are "
                    "not independent of each other; an analysis that "
                    "treats them as independent will understate "
                    "uncertainty."
                ),
            )
        )
        inspections.append(
            Inspection(
                trigger="Longitudinal, repeated-measures",
                note=(
                    "Repeated, timestamped observations raise coverage "
                    "and missingness questions before any effect can be "
                    "estimated from them."
                ),
                suggested_module="Time-Series QA",
            )
        )

    if MEASUREMENT_TYPE_SURVEY_SCALE in structure.measurement_types:
        inspections.append(
            Inspection(
                trigger="Survey or self-report scale",
                note=(
                    "If this outcome is measured with more than one item, "
                    "whether those items measure the same underlying "
                    "construct consistently is worth checking before "
                    "treating them as one score."
                ),
                suggested_module="Reliability",
            )
        )

    if structure.compares_subgroups:
        inspections.append(
            Inspection(
                trigger="Compares demographic subgroups",
                note=(
                    "Comparing outcomes, or measurement quality itself, "
                    "across demographic subgroups raises whether the "
                    "measurement performs equivalently across those "
                    "groups, separate from whether the outcome itself "
                    "differs."
                ),
                suggested_module="Fairness",
            )
        )

    return tuple(inspections)
