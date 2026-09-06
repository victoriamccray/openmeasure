"""
Recognising concepts in a research question, without inventing any.

A planner that made a researcher name every concept and classify it from
scratch was ontology-driven rather than question-driven: correct
underneath, and it did not feel like their question had been read. A
planner that generated concepts from the wording would be worse, because
a construct nobody chose would enter their study looking as though they
had.

So this recognises, and only recognises. A curated list of phrases, each
tied to a concept name and to the kind of thing that concept is. A phrase
in the list is matched; a phrase not in the list is not guessed at. When
nothing matches, that is said plainly rather than covered with a
plausible suggestion.

The rule the whole module exists to keep:

    The question can suggest concepts. Only confirmed concepts enter the
    study.

Nothing here writes into an assembled study. It returns suggestions,
each carrying the phrase that produced it, so a researcher can see why
something was offered and throw it away.

The list is deliberately finite and will have gaps. A gap shows up as
"nothing recognised", which is a smaller problem than a wrong concept
that looks confident, and it is fixed by adding an entry rather than by
loosening the matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .ontology import (
    CONCEPT_KINDS,
    KIND_AGREEMENT,
    KIND_BEHAVIOR,
    KIND_BIOLOGICAL_STATE,
    KIND_BODILY_PROCESS,
    KIND_ECONOMIC_STATE,
    KIND_EXPERIENCE,
    KIND_KNOWLEDGE_STRUCTURE,
    KIND_RECORDED_EVENT,
    KIND_SETTING,
    KIND_SYSTEM_OUTPUT,
)


@dataclass(frozen=True)
class LexiconEntry:
    """One phrase that names a concept, and what sort of concept it is."""

    phrase: str
    concept: str
    kind: str

    def __post_init__(self) -> None:
        for field_name in ("phrase", "concept", "kind"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"{self.phrase or 'A lexicon entry'} is missing a value "
                    f"for '{field_name}'."
                )

        if self.kind not in CONCEPT_KINDS:
            raise ValueError(
                f"'{self.phrase}' maps to kind '{self.kind}', which is not "
                "a declared concept kind."
            )

        if self.phrase != self.phrase.lower():
            raise ValueError(
                f"'{self.phrase}' has uppercase characters. Phrases are "
                "matched case-insensitively and stored lowercase, so a "
                "mixed-case entry would never be reached."
            )


@dataclass(frozen=True)
class ConceptSuggestion:
    """One recognised concept, and the wording that produced it."""

    concept: str
    kind: str
    matched_phrase: str


LEXICON: tuple[LexiconEntry, ...] = (
    # Economic and material
    LexiconEntry("financial security", "Financial security", KIND_ECONOMIC_STATE),
    LexiconEntry("financial well-being", "Financial well-being", KIND_ECONOMIC_STATE),
    LexiconEntry("financial wellbeing", "Financial well-being", KIND_ECONOMIC_STATE),
    LexiconEntry("income", "Income", KIND_ECONOMIC_STATE),
    LexiconEntry("earnings", "Earnings", KIND_ECONOMIC_STATE),
    LexiconEntry("savings", "Savings", KIND_ECONOMIC_STATE),
    LexiconEntry("debt", "Debt burden", KIND_ECONOMIC_STATE),
    LexiconEntry("material hardship", "Material hardship", KIND_ECONOMIC_STATE),
    LexiconEntry("poverty", "Poverty", KIND_ECONOMIC_STATE),
    LexiconEntry("food insecurity", "Food insecurity", KIND_ECONOMIC_STATE),
    LexiconEntry("housing stability", "Housing stability", KIND_ECONOMIC_STATE),
    LexiconEntry("homelessness", "Housing stability", KIND_ECONOMIC_STATE),
    # Experience and belief
    LexiconEntry("pain", "Pain experience", KIND_EXPERIENCE),
    LexiconEntry("depression", "Depressive symptoms", KIND_EXPERIENCE),
    LexiconEntry("depressive symptoms", "Depressive symptoms", KIND_EXPERIENCE),
    LexiconEntry("anxiety", "Anxiety", KIND_EXPERIENCE),
    LexiconEntry("stress", "Perceived stress", KIND_EXPERIENCE),
    LexiconEntry("quality of life", "Quality of life", KIND_EXPERIENCE),
    LexiconEntry("self-efficacy", "Self-efficacy", KIND_EXPERIENCE),
    LexiconEntry("confidence", "Confidence", KIND_EXPERIENCE),
    LexiconEntry("attitudes", "Attitudes", KIND_EXPERIENCE),
    LexiconEntry("beliefs", "Beliefs", KIND_EXPERIENCE),
    LexiconEntry("satisfaction", "Satisfaction", KIND_EXPERIENCE),
    LexiconEntry("burnout", "Burnout", KIND_EXPERIENCE),
    # How someone organises what they know
    LexiconEntry("mental model", "Mental-model structure", KIND_KNOWLEDGE_STRUCTURE),
    LexiconEntry("mental models", "Mental-model structure", KIND_KNOWLEDGE_STRUCTURE),
    LexiconEntry("knowledge", "Knowledge", KIND_KNOWLEDGE_STRUCTURE),
    LexiconEntry("understanding", "Understanding", KIND_KNOWLEDGE_STRUCTURE),
    LexiconEntry("expertise", "Expertise", KIND_KNOWLEDGE_STRUCTURE),
    # Convergence across people
    LexiconEntry(
        "shared mental model", "Shared mental models", KIND_AGREEMENT
    ),
    LexiconEntry(
        "shared mental models", "Shared mental models", KIND_AGREEMENT
    ),
    LexiconEntry("shared understanding", "Shared mental models", KIND_AGREEMENT),
    LexiconEntry("consensus", "Consensus", KIND_AGREEMENT),
    LexiconEntry("interrater", "Rater agreement", KIND_AGREEMENT),
    # What people do
    LexiconEntry("adherence", "Adherence", KIND_BEHAVIOR),
    LexiconEntry("attendance", "Attendance", KIND_BEHAVIOR),
    LexiconEntry("uptake", "Uptake", KIND_BEHAVIOR),
    LexiconEntry("fidelity", "Implementation fidelity", KIND_BEHAVIOR),
    LexiconEntry("violence", "Violent behaviour", KIND_BEHAVIOR),
    LexiconEntry("bullying", "Peer victimization", KIND_BEHAVIOR),
    LexiconEntry("physical activity", "Physical activity", KIND_BEHAVIOR),
    LexiconEntry("achievement", "Academic achievement", KIND_BEHAVIOR),
    LexiconEntry("test scores", "Academic achievement", KIND_BEHAVIOR),
    # Processes in the body
    LexiconEntry("heart rate", "Cardiac activity", KIND_BODILY_PROCESS),
    LexiconEntry("arousal", "Physiological arousal", KIND_BODILY_PROCESS),
    LexiconEntry("sleep", "Sleep", KIND_BODILY_PROCESS),
    LexiconEntry("blood pressure", "Blood pressure", KIND_BODILY_PROCESS),
    LexiconEntry("brain activity", "Brain activity", KIND_BODILY_PROCESS),
    # Molecular and cellular
    LexiconEntry("gene expression", "Gene expression", KIND_BIOLOGICAL_STATE),
    LexiconEntry("biomarker", "Biomarker level", KIND_BIOLOGICAL_STATE),
    LexiconEntry("cortisol", "Cortisol", KIND_BIOLOGICAL_STATE),
    LexiconEntry("inflammation", "Inflammation", KIND_BIOLOGICAL_STATE),
    # What a computational system produces
    LexiconEntry("model performance", "Model performance", KIND_SYSTEM_OUTPUT),
    LexiconEntry("predictions", "Model predictions", KIND_SYSTEM_OUTPUT),
    LexiconEntry("algorithm", "Algorithmic output", KIND_SYSTEM_OUTPUT),
    # Conditions of the setting
    LexiconEntry("air quality", "Air quality", KIND_SETTING),
    LexiconEntry("temperature", "Temperature", KIND_SETTING),
    LexiconEntry("neighborhood", "Neighbourhood conditions", KIND_SETTING),
    LexiconEntry("neighbourhood", "Neighbourhood conditions", KIND_SETTING),
    # Already recorded by someone
    LexiconEntry("readmission", "Hospital readmission", KIND_RECORDED_EVENT),
    LexiconEntry("hospitalization", "Hospitalization", KIND_RECORDED_EVENT),
    LexiconEntry("hospitalisation", "Hospitalization", KIND_RECORDED_EVENT),
    LexiconEntry("employment", "Employment", KIND_RECORDED_EVENT),
    LexiconEntry("graduation", "Graduation", KIND_RECORDED_EVENT),
    LexiconEntry("service use", "Service utilization", KIND_RECORDED_EVENT),
    LexiconEntry("utilization", "Service utilization", KIND_RECORDED_EVENT),
)

# What a researcher is told when nothing in the list matched. Stated
# rather than covered with a plausible guess, which is the entire
# difference between recognising and inventing.
NOTHING_RECOGNISED = (
    "No concept in OpenMeasure's list was recognised in this question. "
    "That is a gap in the list rather than a problem with the question. "
    "Name the concepts yourself below."
)


def _matches(question: str) -> list[LexiconEntry]:
    """Every entry whose phrase appears in the question, as a whole word."""
    lowered = question.lower()

    return [
        entry
        for entry in LEXICON
        if re.search(rf"(?<!\w){re.escape(entry.phrase)}(?!\w)", lowered)
    ]


def suggest_concepts(question: str) -> tuple[ConceptSuggestion, ...]:
    """
    Concepts recognised in a question, for a researcher to confirm.

    A longer matched phrase wins over a shorter one contained in it, so
    "financial well-being" is offered once rather than alongside a
    separate "well-being". The same concept reached through two phrases
    is offered once, keeping the longer phrase, since that is the wording
    that shows most clearly why it was suggested.

    Returns an empty tuple when nothing matched, which the caller should
    report rather than paper over.
    """
    found = _matches(question)

    surviving = [
        entry
        for entry in found
        if not any(
            other is not entry and entry.phrase in other.phrase
            for other in found
        )
    ]

    by_concept: dict[str, LexiconEntry] = {}
    for entry in surviving:
        current = by_concept.get(entry.concept)
        if current is None or len(entry.phrase) > len(current.phrase):
            by_concept[entry.concept] = entry

    ordered = sorted(
        by_concept.values(), key=lambda entry: question.lower().find(entry.phrase)
    )

    return tuple(
        ConceptSuggestion(
            concept=entry.concept, kind=entry.kind, matched_phrase=entry.phrase
        )
        for entry in ordered
    )
