"""
Deterministic relevance scoring for a literature search result against a
finding's description.

Scores by shared keyword count between the finding's own words and the
record's title and abstract -- classic, inspectable information
retrieval, not a semantic or generative model. Every score names exactly
which words matched, so a reader can check the match rather than trust
it. This is the "no LLM calls" half of Evidence Review: relevance is
shown as evidence (the matched terms), never asserted as a verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .record import LiteratureRecord

# Common words excluded from keyword matching so overlap reflects
# subject-matter terms, not shared grammar.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with",
    "is", "are", "was", "were", "by", "at", "as", "this", "that", "from",
    "be", "it", "its", "into", "than", "not", "no", "we", "our", "their",
    "which", "these", "those", "can", "has", "have", "had",
})

_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def _keywords(text: str) -> frozenset[str]:
    tokens = _WORD_PATTERN.findall(text.lower())
    return frozenset(
        token for token in tokens if token not in _STOPWORDS and len(token) > 2
    )


@dataclass(frozen=True)
class RelevanceScore:
    """
    How a search result's own text overlaps a finding's description.

    matched_keywords is the entire basis for overlap_count, named
    explicitly so a reader sees exactly what matched rather than a bare
    number.
    """

    matched_keywords: tuple[str, ...]
    overlap_count: int
    citation_count: int | None

    def __post_init__(self) -> None:
        if self.overlap_count != len(self.matched_keywords):
            raise ValueError(
                f"overlap_count ({self.overlap_count}) does not match "
                f"len(matched_keywords) ({len(self.matched_keywords)})."
            )
        if self.overlap_count < 0:
            raise ValueError("overlap_count cannot be negative.")
        if self.citation_count is not None and self.citation_count < 0:
            raise ValueError(
                f"citation_count cannot be negative: {self.citation_count}."
            )


def score_relevance(finding_text: str, record: LiteratureRecord) -> RelevanceScore:
    """
    Score record's keyword overlap with finding_text.

    An empty or keyword-free finding_text is not an error: it is a
    legitimate zero-overlap result, since a reviewer typing a vague
    description is still allowed to see all the results a search
    returned and judge relevance themselves.
    """

    finding_keywords = _keywords(finding_text)
    record_keywords = _keywords(f"{record.title} {record.abstract}")

    matched = tuple(sorted(finding_keywords & record_keywords))

    return RelevanceScore(
        matched_keywords=matched,
        overlap_count=len(matched),
        citation_count=record.citation_count,
    )
