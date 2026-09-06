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
    get_measure,
    measures_for,
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


# Which measures actually apply to a named concept, where the concept
# kind is too coarse on its own.
#
# The kind finds a candidate universe and does not settle applicability.
# Cardiac activity and brain activity are both processes in the body, and
# functional neuroimaging observes one of them; offering it for the other
# because they share a kind is the kind doing work it cannot do.
#
# Curated and conservative. A concept absent from this mapping falls back
# to its kind's whole universe, which is the honest answer for a concept
# nobody has narrowed yet, and the page says which of the two a reader is
# looking at. Every list here is a subset of its kind's universe, checked
# by a test: naming a measure that its own record says does not observe
# this kind would be asserting something the library contradicts.
CONCEPT_MEASURES: dict[str, tuple[str, ...]] = {
    # Processes in the body, where the kind is at its coarsest.
    "Cardiac activity": ("Heart rate and heart-rate variability",),
    "Physiological arousal": (
        "Electrodermal activity",
        "Heart rate and heart-rate variability",
    ),
    "Brain activity": ("Functional neuroimaging",),
    "Blood pressure": ("Clinical assessment or chart review",),
    # Experiences, where a scale built for one thing should not be
    # offered for another.
    "Pain experience": (
        "Rating scale, repeated in daily life",
        "Structured survey",
        "Semi-structured interview",
    ),
    "Spatial pain pattern": ("Body map or pain drawing",),
    "Depressive symptoms": (
        "Structured survey",
        "Semi-structured interview",
        "Rating scale, repeated in daily life",
    ),
    "Anxiety": ("Structured survey", "Semi-structured interview"),
    "Perceived stress": (
        "Structured survey",
        "Rating scale, repeated in daily life",
    ),
    "Quality of life": ("Structured survey",),
    "Self-efficacy": ("Structured survey",),
    "Financial well-being": ("Financial well-being scale", "Structured survey"),
    # How someone organises what they know.
    "Mental-model structure": (
        "Card sorting",
        "Causal or cognitive mapping",
        "Think-aloud protocol",
        "Semi-structured interview",
    ),
    "Understanding": ("Think-aloud protocol", "Semi-structured interview"),
    # Convergence across people.
    "Shared mental models": ("Delphi or group elicitation", "Structured survey"),
    # Economic and material.
    "Income": ("Income or earnings record",),
    "Earnings": ("Income or earnings record",),
    "Savings": ("Assets and savings inventory",),
    "Debt burden": ("Debt burden measure",),
    "Material hardship": ("Material hardship indicators",),
    "Food insecurity": ("Material hardship indicators",),
    "Poverty": ("Income or earnings record", "Material hardship indicators"),
    "Housing stability": ("Material hardship indicators",),
    # Molecular and cellular.
    "Gene expression": ("Assay of a biological sample",),
    "Cortisol": ("Assay of a biological sample",),
    "Inflammation": ("Assay of a biological sample",),
    "Biomarker level": ("Assay of a biological sample",),
}

# Said when a concept has no narrowed list, so a reader can tell the two
# situations apart. A broad list is not a worse answer; it is a different
# one, and hiding which they are looking at is what would mislead.
BROAD_CANDIDATES_NOTE = (
    "These are every measure that can observe this kind of thing. "
    "OpenMeasure has no narrowed list for this concept, so judge "
    "applicability yourself."
)

NARROWED_CANDIDATES_NOTE = (
    "Narrowed to the measures that apply to this concept, out of "
    "everything that can observe this kind of thing."
)


def measures_for_concept(concept: str, kind: str):
    """
    The measures that apply to one named concept.

    Falls back to the kind's whole universe for a concept nobody has
    narrowed, which is honest rather than empty: a concept a researcher
    typed themselves has no curated list and should still reach
    something.

    Returns the measures and whether the list was narrowed, so a caller
    can say which of the two a reader is looking at.
    """
    universe = measures_for(kind)
    narrowed = CONCEPT_MEASURES.get(concept)

    if not narrowed:
        return universe, False

    by_name = {measure.name: measure for measure in universe}

    return tuple(by_name[name] for name in narrowed if name in by_name), True

