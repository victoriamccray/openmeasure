"""
Screening decisions for literature search results, and the counts
summarizing them.

Include/Exclude/Uncertain is a common systematic-review screening
convention (used by tools such as Covidence and Rayyan), structured here
for compatibility with PRISMA 2020's reporting requirements (Page et al.,
2021, BMJ 372:n71). PRISMA itself is a reporting guideline, not the
source of this specific three-way label set: it requires that a review
define its own eligibility criteria and report its selection process,
including reasons for excluding records that might otherwise look
eligible (see reason on ScreeningDecision below), not that it use this
exact vocabulary. It appears elsewhere in this codebase only as an
illustrative docstring example (modules/reliability/core/interrater.py);
this is its first use as a real, typed value.
"""

from __future__ import annotations

from dataclasses import dataclass

DECISION_INCLUDE = "Include"
DECISION_EXCLUDE = "Exclude"
DECISION_UNCERTAIN = "Uncertain"

DECISIONS: tuple[str, ...] = (DECISION_INCLUDE, DECISION_EXCLUDE, DECISION_UNCERTAIN)


@dataclass(frozen=True)
class ScreeningDecision:
    """
    One reviewer's Include/Exclude/Uncertain call on one search result.

    reason is optional (not every exclusion needs justifying -- an
    obviously unrelated result does not), but PRISMA 2020 specifically
    recommends recording why a record was excluded when it might
    otherwise look eligible, so pages/6_Evidence_Review.py prompts for
    one whenever a reviewer picks Exclude.
    """

    record_title: str
    decision: str
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.record_title:
            raise ValueError("A ScreeningDecision must name the record it screens.")
        if self.decision not in DECISIONS:
            raise ValueError(
                f"'{self.decision}' is not a known screening decision. "
                f"Valid decisions: {', '.join(DECISIONS)}."
            )


@dataclass(frozen=True)
class ScreeningSummary:
    """How many results were found, and how each was screened."""

    n_found: int
    n_included: int
    n_excluded: int
    n_uncertain: int

    def __post_init__(self) -> None:
        for name, value in (
            ("n_found", self.n_found),
            ("n_included", self.n_included),
            ("n_excluded", self.n_excluded),
            ("n_uncertain", self.n_uncertain),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative: {value}.")

        total = self.n_included + self.n_excluded + self.n_uncertain
        if total != self.n_found:
            raise ValueError(
                f"n_included ({self.n_included}) + n_excluded "
                f"({self.n_excluded}) + n_uncertain ({self.n_uncertain}) = "
                f"{total}, which does not equal n_found ({self.n_found})."
            )


def summarize_screening(
    n_found: int, decisions: tuple[ScreeningDecision, ...]
) -> ScreeningSummary:
    """
    Summarize screening decisions against how many results were found.

    A result with no decision yet counts as uncertain: a reviewer who
    has not screened a result has not decided to exclude it, and
    counting it as included would overstate what has actually been
    reviewed.
    """

    if n_found < 0:
        raise ValueError(f"n_found cannot be negative: {n_found}.")

    if len(decisions) > n_found:
        raise ValueError(
            f"{len(decisions)} decisions were given for only {n_found} "
            "results found."
        )

    n_included = sum(1 for d in decisions if d.decision == DECISION_INCLUDE)
    n_excluded = sum(1 for d in decisions if d.decision == DECISION_EXCLUDE)
    n_explicitly_uncertain = sum(
        1 for d in decisions if d.decision == DECISION_UNCERTAIN
    )
    n_undecided = n_found - len(decisions)

    return ScreeningSummary(
        n_found=n_found,
        n_included=n_included,
        n_excluded=n_excluded,
        n_uncertain=n_explicitly_uncertain + n_undecided,
    )
