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
    label: str
    # Rendered with .format(unit=..., comparison=...), so a field's own
    # vocabulary appears without this module importing a specific domain.
    summary_template: str
    needs: str
    # The trade the design makes, shown as a quieter note beneath it.
    caveat: str = ""
    # A published example anchored beside this design, and why it is
    # placed here. Both empty for a design with no anchored study.
    case_study_key: str = ""
    case_study_connection: str = ""

    def __post_init__(self) -> None:
        for field_name in ("id", "label", "summary_template", "needs"):
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

    def summary_for(self, domain: Domain) -> str:
        """The summary in one field's vocabulary."""
        return self.summary_template.format(
            unit=domain.term_for(CONCEPT_UNIT),
            comparison=domain.term_for(CONCEPT_COMPARISON_GROUP),
        )


DESIGN_OPTIONS: tuple[DesignOption, ...] = (
    DesignOption(
        id="two_or_more_groups",
        label="Two or more groups",
        summary_template=(
            "Compares an outcome across groups measured once. Needs an "
            "outcome column and a column identifying which group each "
            "{unit} belongs to."
        ),
        needs="An outcome column and a group column.",
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
        label="Pre/post, same participants",
        summary_template=(
            "Compares one group's outcome before and after, using each "
            "{unit} as its own baseline. Needs a baseline column and a "
            "follow-up column."
        ),
        needs="A baseline column and a follow-up column.",
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
        label="Two groups, each measured before and after",
        summary_template=(
            "Difference-in-differences. Subtracts the {comparison}'s "
            "change from the treated group's, which removes anything that "
            "moved both equally and any fixed gap between them at "
            "baseline. Needs a group column plus a baseline and a "
            "follow-up column."
        ),
        needs="A group column, a baseline column, and a follow-up column.",
        caveat=(
            "It buys that with an assumption instead: that the treated "
            "group would have followed the comparison group's path. Two "
            "time points give no way to check it, so this page states the "
            "assumption alongside the estimate."
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
