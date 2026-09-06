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
