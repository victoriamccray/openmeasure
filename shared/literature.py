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

# OpenAlex asks callers to identify themselves so it can contact a heavy
# user rather than simply blocking them.
USER_AGENT = "OpenMeasure/0.1 (+https://github.com/victoriamccray/openmeasure)"

MAX_RESULTS = 10
REQUEST_TIMEOUT_SECONDS = 15

# What a caller should catch around search_openalex. Named here so a page
# does not have to know that the call is built on urllib, and so every
# page catches the same set.
SEARCH_ERRORS = (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError)


@st.cache_data(ttl=3600, show_spinner="Searching OpenAlex...")
def search_openalex(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """
    Keyword search against OpenAlex's public Works API.

    Cached for an hour so re-rendering a page (after a screening decision
    changes, or after a later stage unlocks) does not re-fire the same
    search.
    """

    params = urllib.parse.urlencode({"search": query, "per-page": max_results})
    url = f"{OPENALEX_WORKS_ENDPOINT}?{params}"
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )

    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))

    return payload.get("results", [])
