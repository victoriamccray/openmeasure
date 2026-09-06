"""
What a set of chosen measures observes, and what it leaves unobserved.

A planner that only collected selections would have nothing to say back.
The point of assembling a measurement strategy inside OpenMeasure rather
than in a document is that the assembled thing can be inspected: which of
the concepts a researcher named are actually reached by something they
chose, and which are not.

An unobserved concept is not an error. A study that names four concepts
and measures three of them is an ordinary study, and the fourth is a
stated boundary rather than a defect. What would be a defect is a Design
Record that listed four concepts and left a reader to work out which ones
the design could speak to.

Nothing here fills a gap. Where a design property has not been chosen it
stays unset, and the record says so, because the failure this module was
written after was a worked example's properties appearing in a
researcher's own record as though they had chosen them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ontology import CONCEPT_KINDS, Measure, get_measure, modalities_of

# What the assembled measures do to a concept. Two states rather than a
# score: a concept is either reached by something chosen or it is not,
# and a partial-coverage number would invite a reading the selection
# cannot support.
OBSERVED = "Observed"
NOT_OBSERVED = "Not measured by anything selected"

# How a study describes itself once measures are chosen. A headline, for
# a record and a caption.
SHAPE_SINGLE = "one measure, one occasion"
SHAPE_MULTIMODAL = "several kinds of evidence about the same concepts"
SHAPE_REPEATED = "the same measures on more than one occasion"
SHAPE_MULTIMODAL_REPEATED = "several kinds of evidence, more than once"

# Which pictures of this study are worth drawing.
#
# Selected from the assembled structure rather than fixed, because the
# useful view of a multimodal cross-section and of one measure repeated
# across eight weeks are not the same picture, and a single diagram
# trying to serve both serves neither. More than one can apply at once: a
# clustered trial measured repeatedly has a nesting and a timeline, and
# both are true of it.
VIEW_MEASUREMENT = "Measurement architecture"
VIEW_CONVERGENCE = "Cross-modal convergence"
VIEW_TIMELINE = "Timeline"
VIEW_NESTING = "Nesting"
VIEW_ARMS = "Arms"

VIEWS: tuple[str, ...] = (
    VIEW_MEASUREMENT,
    VIEW_CONVERGENCE,
    VIEW_TIMELINE,
    VIEW_NESTING,
    VIEW_ARMS,
)


@dataclass(frozen=True)
class Concept:
    """One thing a study has to observe, and what sort of thing it is."""

    name: str
    kind: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A concept needs a name.")

        if self.kind not in CONCEPT_KINDS:
            raise ValueError(
                f"'{self.name}' has kind '{self.kind}', which is not a "
                f"declared concept kind. Known kinds: "
                f"{'; '.join(CONCEPT_KINDS)}."
            )


@dataclass(frozen=True)
class ConceptCoverage:
    """One concept, and what reaches it."""

    concept: Concept
    status: str
    measures: tuple[str, ...]

    @property
    def observed(self) -> bool:
        return self.status == OBSERVED


@dataclass(frozen=True)
class AssembledStudy:
    """
    A researcher's concepts, their chosen measures, and what follows.

    occasions defaults to 1 rather than to None because a study is
    measured at least once by definition. Every other structural property
    a Design Record wants is deliberately absent from this object: this
    records what was assembled, and a property nobody set is reported as
    unset rather than defaulted into looking like a decision.
    """

    concepts: tuple[Concept, ...]
    selected_measures: tuple[str, ...]
    occasions: int = 1

    # What the units sit inside, where they sit inside anything. Named by
    # the researcher rather than detected, because no property of a plan
    # establishes that students were sampled through schools; only the
    # person designing it knows.
    nesting: str = ""

    # The arms a study compares, where it compares any. Two or more is a
    # comparison; one or none is not, and an empty tuple is the honest
    # default for a study that has not said.
    arms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.occasions < 1:
            raise ValueError(
                f"A study is measured at least once; got {self.occasions}."
            )

        if len(self.arms) == 1:
            raise ValueError(
                f"'{self.arms[0]}' is the only arm named. One arm is not a "
                "comparison; name the arm it would be compared with, or "
                "name none."
            )

        for name in self.selected_measures:
            get_measure(name)

    @property
    def measures(self) -> tuple[Measure, ...]:
        """The chosen measures, resolved."""
        return tuple(get_measure(name) for name in self.selected_measures)

    @property
    def modalities(self) -> tuple[str, ...]:
        """Which kinds of evidence this study would produce."""
        return modalities_of(self.selected_measures)

    @property
    def coverage(self) -> tuple[ConceptCoverage, ...]:
        """
        Each named concept, and which chosen measures reach it.

        In the order the researcher named them, so the list reads as
        their study rather than as a reordering of it.
        """
        rows = []

        for concept in self.concepts:
            reaching = tuple(
                measure.name
                for measure in self.measures
                if concept.kind in measure.observes
            )
            rows.append(
                ConceptCoverage(
                    concept=concept,
                    status=OBSERVED if reaching else NOT_OBSERVED,
                    measures=reaching,
                )
            )

        return tuple(rows)

    @property
    def unobserved(self) -> tuple[Concept, ...]:
        """The concepts nothing selected reaches."""
        return tuple(row.concept for row in self.coverage if not row.observed)

    @property
    def applicable_views(self) -> tuple[str, ...]:
        """
        Which pictures of this study are worth drawing, in fixed order.

        The measurement architecture always applies, because a study with
        one measure on one occasion still has a structure. The rest apply
        when the thing they draw is actually present, so a view never
        appears with nothing in it.
        """
        views = [VIEW_MEASUREMENT]

        if len(self.modalities) > 1:
            views.append(VIEW_CONVERGENCE)
        if self.occasions > 1:
            views.append(VIEW_TIMELINE)
        if self.nesting.strip():
            views.append(VIEW_NESTING)
        if len(self.arms) > 1:
            views.append(VIEW_ARMS)

        return tuple(view for view in VIEWS if view in views)

    @property
    def shape(self) -> str:
        """
        Which view of this study is worth drawing.

        Chosen from what was assembled rather than fixed, because the
        useful picture of three modalities on one occasion and of one
        measure across eight weeks are different pictures.
        """
        multimodal = len(self.modalities) > 1
        repeated = self.occasions > 1

        if multimodal and repeated:
            return SHAPE_MULTIMODAL_REPEATED
        if multimodal:
            return SHAPE_MULTIMODAL
        if repeated:
            return SHAPE_REPEATED

        return SHAPE_SINGLE


# What a Design Record prints where a property was never chosen. One
# token, so a reader scanning a record can see every open question at
# once, and so nothing is quietly defaulted into looking decided.
UNSET = "not established"


def design_record_lines(study: AssembledStudy, entered: dict[str, str]) -> tuple[str, ...]:
    """
    The record of what this researcher assembled, and only that.

    ``entered`` is whatever they typed about their own question. A blank
    field prints as UNSET rather than borrowing from anywhere, which is
    the whole reason this function exists: a worked example's population
    once printed under a researcher's own question with nothing marking
    the join.
    """
    lines = [
        f"Research question: {entered.get('question') or UNSET}",
        f"Population: {entered.get('population') or UNSET}",
        f"Setting: {entered.get('setting') or UNSET}",
        "",
        "Concepts to observe",
    ]

    if study.concepts:
        for row in study.coverage:
            reached = ", ".join(row.measures) if row.measures else UNSET
            lines.append(f"- {row.concept.name} ({row.status}): {reached}")
    else:
        lines.append(f"- {UNSET}")

    lines += [
        "",
        "Evidence this would produce",
        f"- Modalities: {', '.join(study.modalities) or UNSET}",
        f"- Occasions: {study.occasions}",
        f"- Units nested within: {study.nesting or UNSET}",
        f"- Arms compared: {', '.join(study.arms) or UNSET}",
        f"- Shape: {study.shape}",
        "",
        "What this design would not observe",
    ]

    if study.unobserved:
        lines += [f"- {concept.name}" for concept in study.unobserved]
    else:
        lines.append(
            "- Every concept named above is reached by a selected measure. "
            "That is coverage of the concepts stated, not of the ones that "
            "were never named."
        )

    return tuple(lines)
