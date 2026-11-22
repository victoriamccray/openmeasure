"""
Three agency questions per modality, answered only where the cited
control text answers them.

The illustrative agency_cost rating is a single author-assigned number.
This module replaces it, for drawing, with three plain questions:

    stop       Can the person stop the collection themselves?
    each_time  Does each collection need the person's own action?
    undo       Can the person correct or retract what was recorded or inferred?

Each answer is "yes", "no", or "not stated". A "yes" or "no" carries the
exact phrase from that modality's agency_control_point (in
sample_data/modality_profiles.csv, which cites its sources) that supports
it. Where the text is silent the answer is "not stated", never a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

YES, NO, NOT_STATED = "yes", "no", "not stated"
ANSWERS = (YES, NO, NOT_STATED)

QUESTIONS = (
    ("stop", "Can they stop it?"),
    ("each_time", "Asked each time?"),
    ("undo", "Can they undo it?"),
)

# A plain sentence for every question and answer, so a reader never has to
# decode a symbol or a bare "not stated".
PLAIN: dict[str, dict[str, str]] = {
    "stop": {
        YES: "Yes. The person can stop the recording themselves.",
        NO: "No. The person cannot stop it themselves; someone else controls it.",
        NOT_STATED: "Unknown. The sources do not say whether the person can stop it.",
    },
    "each_time": {
        YES: "Yes. Nothing is recorded unless the person does something each time.",
        NO: "No. It can keep recording without asking the person each time.",
        NOT_STATED: "Unknown. The sources do not say whether the person is asked each time.",
    },
    "undo": {
        YES: "Yes. The person can correct or take back what was recorded.",
        NO: "No. Once recorded, the person cannot correct it or take it back.",
        NOT_STATED: "Unknown. The sources do not say whether the person can correct or take it back.",
    },
}


class UnknownCategoryError(ValueError):
    """The category has no agency answers recorded."""


@dataclass(frozen=True)
class AgencyAnswer:
    key: str
    question: str
    answer: str
    evidence: str  # the supporting phrase from agency_control_point; empty when not stated
    plain: str  # the answer as one plain sentence


@dataclass(frozen=True)
class AgencyProfile:
    category: str
    answers: tuple[AgencyAnswer, ...]


# category -> {question key: (answer, supporting phrase)}
_EVIDENCE: dict[str, dict[str, tuple[str, str]]] = {
    "Neural": {
        "stop": (YES, "can be removed at will"),
        "each_time": (NOT_STATED, ""),
        "undo": (NO, "no established way to know afterward what was inferred from a session, or to retract it"),
    },
    "Autonomic": {
        "stop": (YES, "can be removed at will"),
        "each_time": (NO, "can otherwise run without an ongoing prompt at each measurement"),
        "undo": (NOT_STATED, ""),
    },
    "Muscular": {
        "stop": (YES, "can be removed at will"),
        "each_time": (YES, "each session requires deliberate placement"),
        "undo": (NOT_STATED, ""),
    },
    "Behavioral": {
        "stop": (YES, "can be silenced or set aside"),
        "each_time": (NO, "designed to continue without a prompt at each measurement"),
        "undo": (NOT_STATED, ""),
    },
    "Subjective": {
        "stop": (YES, "nothing is recorded unless the person actively responds"),
        "each_time": (YES, "nothing is recorded unless the person actively responds"),
        "undo": (NOT_STATED, ""),
    },
    "Environmental": {
        "stop": (NOT_STATED, ""),
        "each_time": (NO, "many people are not shown each time a location point is captured"),
        "undo": (NOT_STATED, ""),
    },
    "Institutional": {
        "stop": (NO, "entry, correction, and sharing of the record are usually mediated by clinicians or administrators"),
        "each_time": (NO, "entry, correction, and sharing of the record are usually mediated by clinicians or administrators"),
        "undo": (NO, "entry, correction, and sharing of the record are usually mediated by clinicians or administrators"),
    },
}

CATEGORIES = tuple(_EVIDENCE)


def agency_profile(category: str) -> AgencyProfile:
    """The three answers for one modality category, each with its supporting phrase."""
    if category not in _EVIDENCE:
        raise UnknownCategoryError(f"No agency answers for category {category!r}; known: {', '.join(CATEGORIES)}.")
    row = _EVIDENCE[category]
    return AgencyProfile(
        category=category,
        answers=tuple(AgencyAnswer(key, label, *row[key], PLAIN[key][row[key][0]]) for key, label in QUESTIONS),
    )
