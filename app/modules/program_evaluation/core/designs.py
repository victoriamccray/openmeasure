"""
The designs this module supports, described in one place.

Stage 4 of the Impact Evaluation workflow lays out what each design
compares, what columns it needs, and what it leaves open. That content
lived inline in the page as three near-parallel blocks of prose, which
made it long, impossible to test, and easy to let drift apart from the
recommender that actually chooses between them.

Here it is data. A page renders the tuple; it does not restate it. The
design ids match the method names recommend.py returns, so a design a
reader read about in stage 4 is the same one named in a recommendation,
and a test asserts that correspondence rather than leaving it to review.

Descriptions take a Domain so a field's own word for the comparison group
or the unit appears in them, following the same rule as everywhere else:
the domain changes the vocabulary, never which design fits.
"""

from __future__ import annotations

from dataclasses import dataclass

from .domains import CONCEPT_COMPARISON_GROUP, CONCEPT_UNIT, Domain


@dataclass(frozen=True)
class DesignOption:
    """One study design, as a reader meets it before choosing."""

    id: str
    # Rendered as a heading, so it follows OpenMeasure's title case
    # rather than sentence case.
    label: str
    # Three short cells, one short sentence each, so a reader can compare
    # the designs by scanning a row rather than by reading three
    # structurally identical paragraphs. Each is rendered with
    # .format(unit=..., comparison=...), so a field's own vocabulary
    # appears without this module importing a specific domain.
    compares_template: str
    needs_template: str
    leaves_open_template: str
    # A published example anchored beside this design, and why it is
    # placed here. Both empty for a design with no anchored study.
    case_study_key: str = ""
    case_study_connection: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "id",
            "label",
            "compares_template",
            "needs_template",
            "leaves_open_template",
        ):
            if not getattr(self, field_name):
                raise ValueError(
                    f"{self.id or 'A design'} is missing a value for "
                    f"'{field_name}'."
                )

        if bool(self.case_study_key) != bool(self.case_study_connection):
            raise ValueError(
                f"{self.id} names a case study without a connection, or a "
                "connection without a case study. A published example has "
                "to say why it sits where it does."
            )

    def cells_for(self, domain: Domain) -> dict[str, str]:
        """
        The three comparison cells, in one field's vocabulary.

        Returned together because they are only ever shown together, as
        one row of a table, and keeping them in one call is what stops a
        caller rendering two of the three.
        """
        vocabulary = {
            "unit": domain.term_for(CONCEPT_UNIT),
            "comparison": domain.term_for(CONCEPT_COMPARISON_GROUP),
        }

        return {
            "What it compares": self.compares_template.format(**vocabulary),
            "What it needs": self.needs_template.format(**vocabulary),
            "What it leaves open": self.leaves_open_template.format(**vocabulary),
        }


DESIGN_OPTIONS: tuple[DesignOption, ...] = (
    DesignOption(
        id="two_or_more_groups",
        label="Two or More Groups",
        compares_template="An outcome across groups, each measured once.",
        needs_template=(
            "An outcome column, and a column saying which group each "
            "{unit} is in."
        ),
        leaves_open_template=(
            "Whatever already differed between the groups before the "
            "program did."
        ),
        case_study_key="lalonde_1986",
        case_study_connection=(
            "This design compares the groups as they are, and has no way "
            "to see how anyone ended up in one rather than the other. If "
            "group membership was not randomly assigned, whatever "
            "distinguished the groups beforehand is carried along in the "
            "difference it reports, and the confidence interval around "
            "that difference will look no wider for it."
        ),
    ),
    DesignOption(
        id="pre_post",
        label="Pre/Post, Same Participants",
        compares_template=(
            "One group before and after, each {unit} against its own "
            "baseline."
        ),
        needs_template="A baseline column and a follow-up column.",
        leaves_open_template=(
            "Anything else that changed over the same period."
        ),
        case_study_key="scared_straight",
        case_study_connection=(
            "This design measures how much one group changed between two "
            "measurements. Nothing in it observes what would have happened "
            "without the program, so maturation, regression to the mean, "
            "and outside events stay open as explanations for the change."
        ),
    ),
    DesignOption(
        id="difference_in_differences",
        label="Two Groups, Before and After",
        compares_template=(
            "How much the treated group changed, against how much the "
            "{comparison} changed."
        ),
        needs_template=(
            "A group column, plus a baseline and a follow-up column."
        ),
        leaves_open_template=(
            "Whether the two were on the same path already, which two "
            "time points cannot show."
        ),
    ),
)

DESIGN_IDS: tuple[str, ...] = tuple(design.id for design in DESIGN_OPTIONS)

# Shown after the designs, in place of a recommendation. Stage 4 is where
# a reader reads about designs, not where one is chosen for them.
DESIGN_CHOICE_IMPLICATION = (
    "The design that fits is the one your data can support. The analysis "
    "stage recommends a test from your data's shape and lets you override "
    "it."
)


def get_design(design_id: str) -> DesignOption:
    """Return one design by id, raising on an unknown one."""
    for design in DESIGN_OPTIONS:
        if design.id == design_id:
            return design

    raise ValueError(
        f"'{design_id}' is not a known design. Known designs: "
        f"{', '.join(DESIGN_IDS)}."
    )
