"""
Worked examples, expressed in the same vocabulary a researcher plans in.

The chronic-pain study used to be the planner. Its five measures were the
only measures on offer, so a question about mental models in
implementation research was answered with pain ratings and electrodermal
activity, and its design properties printed in the record of whatever
study a reader had actually described.

It is a good example and it was in the wrong place. Here it is an
AssembledStudy like any other: the same concepts, the same measure
library, the same structural properties. Loading it fills the workspace
and every part of it can then be changed, which is what makes it teach
the planner rather than replace it.

A worked example added here has to be expressible in the ontology. If one
is not, that is a gap in the ontology worth fixing rather than a reason
for the example to carry its own private vocabulary again.
"""

from __future__ import annotations

from dataclasses import dataclass

from .assembly import AssembledStudy, Concept
from .ontology import (
    KIND_BODILY_PROCESS,
    KIND_EXPERIENCE,
)


@dataclass(frozen=True)
class WorkedExample:
    """One study a reader can load, take apart, and change."""

    key: str
    title: str
    summary: str
    entered: dict[str, str]
    study: AssembledStudy

    def __post_init__(self) -> None:
        for field_name in ("key", "title", "summary"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"{self.key or 'A worked example'} is missing a value "
                    f"for '{field_name}'."
                )

        if not self.study.concepts:
            raise ValueError(
                f"{self.key} names no concepts, so loading it would fill "
                "the workspace with nothing."
            )


CHRONIC_PAIN = WorkedExample(
    key="chronic_pain",
    title="Chronic pain in daily life",
    summary=(
        "Whether the coupling between what a person reports feeling and "
        "what their body is doing differs by how their pain is spatially "
        "distributed, observed in everyday settings rather than in a lab."
    ),
    entered={
        "question": (
            "Does the coupling between subjective pain and physiological "
            "signals change when chronic pain is localized rather than "
            "spatially distributed?"
        ),
        "population": "Adults with chronic pain, observed in daily life.",
        "setting": "Naturalistic: everyday environments, not a lab visit.",
    },
    study=AssembledStudy(
        concepts=(
            Concept("Pain experience", KIND_EXPERIENCE),
            Concept("Spatial pain pattern", KIND_EXPERIENCE),
            Concept("Physiological arousal", KIND_BODILY_PROCESS),
        ),
        selected_measures=(
            "Rating scale, repeated in daily life",
            "Electrodermal activity",
            "Heart rate and heart-rate variability",
        ),
        # Four prompts a day for seven days, as the simulation this
        # example came from assumes.
        occasions=28,
    ),
)

WORKED_EXAMPLES: tuple[WorkedExample, ...] = (CHRONIC_PAIN,)


def get_example(key: str) -> WorkedExample:
    """Return one worked example by key, raising on an unknown one."""
    for example in WORKED_EXAMPLES:
        if example.key == key:
            return example

    raise ValueError(
        f"'{key}' is not a known worked example. Known: "
        f"{', '.join(item.key for item in WORKED_EXAMPLES)}."
    )
