"""
A structured record of one published work found while comparing a
finding against existing literature, and how to build one from a raw
OpenAlex API response.

Fetching from OpenAlex is I/O and lives in pages/6_Evidence_Review.py,
the same reason pages/FMRI_QC_Worked_Example.py's remote fetch lives in
its own page rather than in a core/ module. Parsing the JSON OpenAlex
already returned into this shape has no I/O of its own, so it lives here
where it can be tested against a fixed sample response rather than a
live network call.
"""

from __future__ import annotations

from dataclasses import dataclass

UNTITLED = "Untitled (no title returned)"
UNKNOWN_VENUE = "Unknown venue"


@dataclass(frozen=True)
class LiteratureRecord:
    """One work returned by a literature search, with full provenance."""

    title: str
    authors: tuple[str, ...]
    year: int | None
    venue: str
    doi: str | None
    url: str
    abstract: str
    citation_count: int | None
    source_api: str

    def __post_init__(self) -> None:
        if not self.title:
            raise ValueError("A LiteratureRecord must have a title.")
        if not self.source_api:
            raise ValueError(f"'{self.title}' is missing its source_api.")
        if not self.url:
            raise ValueError(f"'{self.title}' is missing a url.")
        if not self.venue:
            raise ValueError(f"'{self.title}' is missing a venue.")
        if self.year is not None and self.year < 0:
            raise ValueError(f"'{self.title}' has a negative year: {self.year}.")
        if self.citation_count is not None and self.citation_count < 0:
            raise ValueError(
                f"'{self.title}' has a negative citation_count: "
                f"{self.citation_count}."
            )

    @property
    def author_summary(self) -> str:
        """First author plus 'et al.' when there is more than one."""

        if not self.authors:
            return "Unknown author"
        if len(self.authors) == 1:
            return self.authors[0]
        return f"{self.authors[0]} et al."


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """
    Rebuild plain text from OpenAlex's abstract_inverted_index.

    OpenAlex does not return plain abstract text (publisher licensing),
    only a word -> [positions] map. This is the standard, documented way
    to reconstruct it: place each word at every position it occupies and
    read the result off in position order.
    """

    if not inverted_index:
        return ""

    positions: dict[int, str] = {}
    for word, indices in inverted_index.items():
        for index in indices:
            positions[index] = word

    return " ".join(positions[index] for index in sorted(positions))


def from_openalex_work(work: dict) -> LiteratureRecord:
    """Build a LiteratureRecord from one raw OpenAlex 'work' JSON object."""

    authors = tuple(
        name
        for authorship in work.get("authorships") or []
        if (name := (authorship.get("author") or {}).get("display_name"))
    )

    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    venue = source.get("display_name") or UNKNOWN_VENUE

    doi = work.get("doi")
    url = work.get("id") or doi or ""

    return LiteratureRecord(
        title=work.get("display_name") or work.get("title") or UNTITLED,
        authors=authors,
        year=work.get("publication_year"),
        venue=venue,
        doi=doi,
        url=url or "https://openalex.org",
        abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
        citation_count=work.get("cited_by_count"),
        source_api="OpenAlex",
    )
