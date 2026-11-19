"""
Eligibility criteria applied to literature search results, before
screening.

PRISMA 2020 (Page et al., 2021, BMJ 372:n71) requires a review to define
its own eligibility criteria and report how many records were excluded
by them. This module implements the parts of that concretely computable
from what OpenAlex returns: a minimum publication year, and terms
required to appear in a result's title or abstract. Population, outcome,
and study-design criteria are not encoded in OpenAlex's metadata in a
structured way, so they are not checked here; the reviewer still applies
those by judgment during screening (modules/evidence_review/core/
screening.py).

This module never removes a result from the search: it labels each one
against the stated criteria, so the reviewer sees the effect of a
criterion before deciding how to screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .record import LiteratureRecord


@dataclass(frozen=True)
class EligibilityCriteria:
    """Criteria stated before screening, applied uniformly to every result."""

    min_year: int | None = None
    required_terms: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.min_year is not None and self.min_year < 0:
            raise ValueError(f"min_year cannot be negative: {self.min_year}.")

        for term in self.required_terms:
            if not term.strip():
                raise ValueError("A required term cannot be blank.")

    @property
    def is_empty(self) -> bool:
        """Whether any criterion has actually been stated."""

        return self.min_year is None and not self.required_terms


@dataclass(frozen=True)
class EligibilityAssessment:
    """Whether one result meets every stated criterion, and why not."""

    record_title: str
    eligible: bool
    reasons_excluded: tuple[str, ...]


@dataclass(frozen=True)
class EligibilitySummary:
    """How many search results meet the stated eligibility criteria."""

    n_found: int
    n_eligible: int
    n_ineligible: int
    assessments: tuple[EligibilityAssessment, ...]


def assess_eligibility(
    results: tuple[LiteratureRecord, ...],
    criteria: EligibilityCriteria,
) -> EligibilitySummary:
    """
    Apply eligibility criteria to every search result.

    A result missing the information a criterion needs (for example, no
    recorded publication year) is treated as not meeting that criterion:
    an unknown value is never assumed to comply.
    """

    assessments: list[EligibilityAssessment] = []

    for result in results:
        reasons: list[str] = []

        if criteria.min_year is not None:
            if result.year is None:
                reasons.append(
                    f"no recorded publication year (minimum {criteria.min_year})"
                )
            elif result.year < criteria.min_year:
                reasons.append(
                    f"published {result.year}, before the minimum year "
                    f"{criteria.min_year}"
                )

        haystack = f"{result.title} {result.abstract}".lower()
        for term in criteria.required_terms:
            if term.strip().lower() not in haystack:
                reasons.append(f"does not contain required term '{term}'")

        assessments.append(
            EligibilityAssessment(
                record_title=result.title,
                eligible=not reasons,
                reasons_excluded=tuple(reasons),
            )
        )

    n_found = len(results)
    n_eligible = sum(1 for assessment in assessments if assessment.eligible)

    return EligibilitySummary(
        n_found=n_found,
        n_eligible=n_eligible,
        n_ineligible=n_found - n_eligible,
        assessments=tuple(assessments),
    )
