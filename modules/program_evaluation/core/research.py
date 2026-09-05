"""
Turning a literature search's raw results into what a reader is shown.

The search itself is a network call and lives in shared/literature.py.
This is the pure half: taking the works that came back, pairing each with
its keyword-overlap score against the researcher's own question, and
producing rows a page can render without doing any of that inline.

Extracted from pages/2_Impact_Evaluation.py's Find Research stage so the
transformation is unit-testable and so a second page (a real-study
journey) can present the same results the same way rather than rebuilding
the loop.

Ordering is deliberately the order OpenAlex returned, not overlap score.
Overlap counts shared words; it is not a relevance ranking, and sorting by
it would present it as one.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.evidence_review.core.record import (  # noqa: E402
    LiteratureRecord,
    from_openalex_work,
)
from modules.evidence_review.core.relevance import score_relevance  # noqa: E402

# What the overlap column is called wherever these rows are rendered.
OVERLAP_COLUMN = "Shared keywords"


@dataclass(frozen=True)
class ResearchRow:
    """One search result, with what it shares with the question."""

    record: LiteratureRecord
    overlap_count: int
    matched_keywords: tuple[str, ...]

    def as_display_row(self) -> dict[str, object]:
        """The flat mapping a table renders, one key per column."""
        return {
            "Title": self.record.title,
            "Authors": self.record.author_summary,
            "Year": self.record.year,
            "Venue": self.record.venue,
            OVERLAP_COLUMN: self.overlap_count,
        }


def research_rows(
    works: list[dict],
    question_terms: str,
) -> tuple[ResearchRow, ...]:
    """
    Pair each returned work with its overlap against the question.

    ``works`` are raw OpenAlex dicts, exactly as shared/literature.py's
    cache returns them. ``question_terms`` is the researcher's own
    wording, not the composed query: overlap should be measured against
    what they asked, not against the search terms a domain contributed.

    Returns an empty tuple for no works, which is a different situation
    from a search never having run and is left for the caller to
    distinguish.
    """
    rows = []

    for work in works:
        record = from_openalex_work(work)
        score = score_relevance(question_terms, record)
        rows.append(
            ResearchRow(
                record=record,
                overlap_count=score.overlap_count,
                matched_keywords=tuple(score.matched_keywords),
            )
        )

    return tuple(rows)


def selectable_titles(rows: tuple[ResearchRow, ...]) -> list[str]:
    """
    Titles for a "keep this study" picker, in the order shown.

    Deduplicated, because OpenAlex can return the same title twice (a
    preprint and its published version, most often) and a picker with two
    identical options cannot tell a reader which one they chose.
    """
    seen: set[str] = set()
    titles: list[str] = []

    for row in rows:
        if row.record.title not in seen:
            seen.add(row.record.title)
            titles.append(row.record.title)

    return titles
