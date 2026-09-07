"""
The one literature search OpenMeasure makes, shared by every page that
offers one.

This was page-local in pages/6_Evidence_Review.py until a second page
(Impact Evaluation's Find Research stage) needed the same call. Extracted
rather than copied, per the project's rule of extracting only what is
actually duplicated, and extracted whole: the endpoint, the user agent,
the timeout, and the cache all belong to the same decision about how
OpenMeasure talks to OpenAlex.

OpenAlex is a fully open, keyless bibliographic index, so this needs no
credentials and nothing here is sent anywhere else. Results are fetched
live at runtime and never bundled into the repository.

search_openalex returns raw dicts rather than LiteratureRecord objects on
purpose. It is wrapped in st.cache_data, which pickles what it returns,
and returning plain JSON keeps the cache independent of any dataclass
definition. Callers pass each dict through
modules/evidence_review/core/record.from_openalex_work to get a typed
record.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

import streamlit as st

OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"

# Which field the query is matched against.
#
# The bare `search` parameter matches full text as well as title and
# abstract, and ranks with citation count in the mix. A question about a
# peer support program returned the 2014 AHA/ACC/HRS atrial fibrillation
# guideline and a critical-care nutrition guideline ahead of any study of
# peer support: heavily cited documents long enough to mention almost
# anything. Restricting the match to title and abstract returns studies
# of the thing that was asked about.
OPENALEX_SEARCH_FIELD = "title_and_abstract.search"

# Characters OpenAlex reads as syntax rather than as words.
#
# `?` and `*` are wildcards, and a stemmed field rejects them outright
# with a 400. This stage asks a researcher for their evaluation
# question, so the single most natural thing to type ended in the one
# character that guaranteed the search failed.
#
# `,` separates filters and `|` separates OR values, so either one
# silently changes the query into a different one, or fails at the API
# edge. `:` separates a filter from its value.
_QUERY_WILDCARDS = "?*"
_QUERY_SEPARATORS = ",|:!"

# OpenAlex asks callers to identify themselves so it can contact a heavy
# user rather than simply blocking them.
USER_AGENT = "OpenMeasure/0.1 (+https://github.com/victoriamccray/openmeasure)"

MAX_RESULTS = 10
REQUEST_TIMEOUT_SECONDS = 15

# What a caller should catch around search_openalex. Named here so a page
# does not have to know that the call is built on urllib, and so every
# page catches the same set.
SEARCH_ERRORS = (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError)


def searchable_terms(query: str) -> str:
    """
    A researcher's wording, with the characters OpenAlex reads as syntax
    taken out of it.

    Wildcards are dropped and separators become spaces, so the words on
    either side survive as words. Nothing else is changed: no stemming,
    no stop-word removal, no reordering. What a reader typed is what is
    searched for, minus the punctuation that would stop it being a search
    at all.

    Raises when nothing is left, rather than sending an empty query and
    returning whatever OpenAlex considers the most cited works in
    existence.
    """
    cleaned = "".join(
        " " if character in _QUERY_SEPARATORS else character
        for character in query
        if character not in _QUERY_WILDCARDS
    )
    cleaned = " ".join(cleaned.split())

    if not cleaned:
        raise ValueError(
            f"'{query}' has no searchable words left once OpenAlex's own "
            "syntax characters are removed. Try wording the question "
            "without wildcards or punctuation."
        )

    return cleaned


# Words that carry no topic in a keyword index over titles and
# abstracts. Two groups, kept apart because they are dropped for
# different reasons and a reader is owed the distinction.
#
# Ordinary function words first.
_STOP_WORDS = frozenset(
    """
    a an and are as at be been being but by did do does for from had has
    have how if in into is it its of on or our that the their there
    these this those to was were what when where whether which who why
    will with would
    """.split()
)

# Then the words a research question is framed with. A researcher writes
# "does the program increase X"; a paper about X does not put "does" or
# "increase" in its title, and leaving them in dilutes the terms that
# would actually match.
_FRAMING_WORDS = frozenset(
    """
    affect affected affects associated association cause caused causes
    change changed changes decrease decreased decreases difference
    effect effects evaluate evaluated evaluation evidence find finding
    findings impact impacts improve improved improvement improves
    increase increased increases influence intervention level levels
    outcome outcomes participants program programme reduce reduced
    reduces reduction result results study trial
    """.split()
)

# Words whose research sense collides with a common sense from another
# literature entirely. Curated, deliberately short, and certainly
# incomplete: a word missing from this list produces a noisy search,
# which is a smaller problem than a confident claim that a search is
# clean. Fixed by adding an entry rather than by guessing.
AMBIGUOUS_TERMS = {
    "security": "financial securities, information security",
    "performance": "athletic and mechanical performance",
    "capital": "financial capital, capital cities",
    "mobility": "physical mobility, mobile networks",
    "engagement": "military and marketing senses",
    "retention": "employee retention, data retention",
    "attrition": "military attrition, dental attrition",
    "exposure": "photographic and radiation exposure",
    "adherence": "material adhesion",
    "arousal": "affective and physiological senses",
    "stress": "material and mechanical stress",
    "trust": "legal trusts, trusted computing",
    "growth": "economic growth, biological growth",
    "development": "child development, land development",
}

# Below this, a query is thin enough that whatever it does match is
# mostly chance.
_THIN_QUERY_TERMS = 2


@dataclass(frozen=True)
class DerivedQuery:
    """
    A search query worked out from a researcher's own wording.

    Carries what was dropped and what is worth a second look, because a
    query a page silently rewrote is worse than the noisy one it
    replaced: the reader could no longer tell why the results looked
    like that.
    """

    finding: str
    terms: str
    dropped_function_words: tuple[str, ...]
    dropped_framing_words: tuple[str, ...]
    ambiguous: tuple[tuple[str, str], ...]

    @property
    def term_count(self) -> int:
        return len(self.terms.split())

    @property
    def is_thin(self) -> bool:
        """Whether so little survived that a match is mostly chance."""
        return self.term_count <= _THIN_QUERY_TERMS

    def notes(self) -> tuple[str, ...]:
        """
        What a reader should know before pressing search, in order.

        Before rather than after: a warning under a page of results
        explains a disappointment instead of preventing one.
        """
        notes = []

        if self.dropped_framing_words:
            notes.append(
                "Dropped as question framing rather than topic: "
                + ", ".join(self.dropped_framing_words)
                + ". A paper about your outcome does not usually put "
                "these in its title."
            )

        if self.dropped_function_words:
            notes.append(
                "Dropped as function words: "
                + ", ".join(self.dropped_function_words)
                + "."
            )

        for term, senses in self.ambiguous:
            notes.append(
                f"'{term}' means different things in different "
                f"literatures ({senses}), so expect matches from those "
                "too. Adding a word that pins your sense of it helps "
                "more than removing it."
            )

        if self.is_thin:
            notes.append(
                f"Only {self.term_count} term"
                + ("" if self.term_count == 1 else "s")
                + " survived, which is thin enough that much of what "
                "comes back will be chance. Consider naming the "
                "population or the setting as well."
            )

        return tuple(notes)


def derive_query(finding: str) -> DerivedQuery:
    """
    Turn a stated finding into terms worth sending to a keyword index.

    Deterministic and reversible: words are dropped from two named
    lists, in order, and every one is reported. No stemming, no
    reordering, no synonym expansion, and nothing added that the
    researcher did not write. What comes back is a suggestion they can
    edit, not a query a page decided on their behalf.

    Raises when nothing topical survives, rather than sending a query of
    function words and returning whatever OpenAlex considers the most
    cited works in existence.
    """
    words = searchable_terms(finding).split()

    kept = []
    function_words = []
    framing_words = []

    for word in words:
        lowered = word.lower().strip("'\u2019")

        if lowered in _STOP_WORDS:
            function_words.append(word)
        elif lowered in _FRAMING_WORDS:
            framing_words.append(word)
        else:
            kept.append(word)

    if not kept:
        raise ValueError(
            f"'{finding}' is all question framing and function words, so "
            "there is nothing topical left to search for. Name the "
            "outcome or the population you mean."
        )

    return DerivedQuery(
        finding=finding,
        terms=" ".join(kept),
        dropped_function_words=tuple(dict.fromkeys(function_words)),
        dropped_framing_words=tuple(dict.fromkeys(framing_words)),
        ambiguous=tuple(
            (word, AMBIGUOUS_TERMS[word.lower()])
            for word in dict.fromkeys(kept)
            if word.lower() in AMBIGUOUS_TERMS
        ),
    )


@st.cache_data(ttl=3600, show_spinner="Searching OpenAlex...")
def search_openalex(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """
    Keyword search against OpenAlex's public Works API.

    Cached for an hour so re-rendering a page (after a screening decision
    changes, or after a later stage unlocks) does not re-fire the same
    search.
    """

    params = urllib.parse.urlencode(
        {
            "filter": f"{OPENALEX_SEARCH_FIELD}:{searchable_terms(query)}",
            "per-page": max_results,
        }
    )
    url = f"{OPENALEX_WORKS_ENDPOINT}?{params}"
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )

    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))

    return payload.get("results", [])
