"""
OpenMeasure - Evidence Review.

A transparent literature discovery and screening aid, not a
systematic-review tool (see modules/evidence_review/README.md's
Non-goals): compares a finding against real published literature, with
no generative AI call anywhere in the page. Search terms go to
OpenAlex's public Works API exactly as shown, relevance is a
deterministic keyword-overlap score between the finding and each
result's own title/abstract, and screening uses Include/Exclude/
Uncertain decisions, structured for compatibility with PRISMA 2020
reporting (Page et al., 2021) -- PRISMA is a reporting guideline that
requires a review to define its own eligibility criteria and report its
selection process; it does not itself supply this three-way label set,
and v0.1 does not yet collect predefined eligibility criteria before
screening starts. Core logic lives in modules/evidence_review/core; this
file handles presentation and the one network call, matching
pages/FMRI_QC_Worked_Example.py's existing pattern for a public-remote
fetch.

Records to shared/handoff.py like any other workflow, but the "dataset"
fingerprinted is the search-results table OpenAlex returned (public
data), never the reviewer's own typed finding text.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from modules.evidence_review.core import record as record_core
from modules.evidence_review.core import relevance as relevance_core
from modules.evidence_review.core import screening as screening_core
from shared.catalog import MODULE_EVIDENCE_REVIEW
from shared.data_handling import disclosure_for, render_data_handling_summary
from shared.handoff import (
    KIND_ROWS_DROPPED,
    ExclusionAccount,
    HandoffStore,
    RetentionItem,
    fingerprint_dataframe,
)
from shared.report import (
    caveat,
    flagged_item_note,
    implications,
    inspect_note,
    render_lifecycle_tracker,
    section_header,
    show_case_studies,
)

OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"
USER_AGENT = "OpenMeasure/0.1 (+https://github.com/victoriamccray/openmeasure)"
MAX_RESULTS = 10
REQUEST_TIMEOUT_SECONDS = 15


@st.cache_data(ttl=3600, show_spinner="Searching OpenAlex...")
def _search_openalex(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """
    The one network call on this page: a keyword search against OpenAlex's
    public Works API. Cached for an hour so re-rendering this page (e.g.
    after a screening decision changes) does not re-fire the same search.
    """

    params = urllib.parse.urlencode({"search": query, "per-page": max_results})
    url = f"{OPENALEX_WORKS_ENDPOINT}?{params}"
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )

    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))

    return payload.get("results", [])


st.set_page_config(
    page_title="OpenMeasure · Evidence Review",
    layout="centered",
)

st.title("Evidence Review")
st.subheader("Cross-Cutting Validation")
st.caption(
    "Compare a finding against real published literature, with a "
    "transparent search and no generative AI call."
)

st.divider()

render_lifecycle_tracker(current_workflow="Evidence Review")

render_data_handling_summary(disclosure_for("pages/6_Evidence_Review.py"))

with st.expander("What this page does"):
    st.markdown(
        """
Type a finding, and this page searches [OpenAlex](https://openalex.org) -- a
fully open, keyless bibliographic index -- for published works that share its
keywords. Every result is real and dereferenceable: a title, authors, year,
venue, DOI, and abstract, exactly as OpenAlex reports them.

Relevance is a **keyword-overlap score**, not a semantic judgment: it counts
how many meaningful words the finding and a result's title/abstract share, and
always names them, so the match can be checked rather than trusted. Nothing
on this page asks a language model to search, summarize, or rank anything.

Each result is then screened **Include / Exclude / Uncertain**, a common
screening convention structured for compatibility with PRISMA 2020 reporting
(Page et al., 2021), and the exact query, every result, and every decision
(including a stated reason for an Exclude) are recorded together as one
structured, exportable record.

### What this page will not do

- It will not tell you a result is "the" relevant paper. A high keyword
  overlap can still be irrelevant, and a low one can still matter; screening
  that judgment is the reviewer's, not this page's.
- It will not grade how strong the literature it finds is (no Nesta-level
  scoring, unlike Portfolio Impact Analysis's evidence grading). It finds and
  screens candidate literature; it does not grade that literature's own
  evidentiary strength.
- It will not infer your finding from a workflow's recorded result
  automatically in this version. Describe the finding yourself.
- It will not ask you to state eligibility criteria (population, outcome,
  study design, date range, ...) before screening starts, which is what
  PRISMA 2020 actually requires a systematic review to define and report.
  This page is a transparent literature discovery and screening aid, not
  yet a systematic-review tool.
"""
    )

with st.expander("Assumptions and limitations"):
    st.markdown(
        """
### Assumptions

- OpenAlex's index is broad but not exhaustive; a result absent here is not
  evidence that no literature exists.
- The finding description is treated as a bag of keywords, the same way a
  search engine would. Rephrasing it will change which results come back.

### Limitations

- Keyword overlap does not establish topical relevance, let alone that a
  result supports or contradicts the finding, only that specific words
  co-occur.
- OpenAlex does not return plain abstract text for every work (publisher
  licensing); some results will show no abstract to match against, so their
  overlap score reflects the title alone.
- A screening decision recorded here is one reviewer's judgment, not a
  consensus. PRISMA's own convention calls for at least two independent
  screeners; this page supports one.
"""
    )

st.divider()

# ---------------------------------------------------------------------
# 1. Describe the finding
# ---------------------------------------------------------------------

section_header("1. Describe the Finding", "What are you comparing against existing evidence?")

finding_text = st.text_area(
    "Describe the finding you want to compare",
    placeholder=(
        "e.g., Reliability alpha was 0.85 for a 10-item anxiety scale in "
        "postpartum women."
    ),
    height=100,
)

query = st.text_input(
    "Search terms sent to OpenAlex (editable -- this is exactly what is searched)",
    value=finding_text,
)

search_clicked = st.button("Search OpenAlex", type="primary", disabled=not query.strip())

if not query.strip():
    st.info("Describe a finding above to search for related literature.")
    show_case_studies("program_validation")
    st.stop()

if not search_clicked and "evidence_review_last_query" not in st.session_state:
    show_case_studies("program_validation")
    st.stop()

if search_clicked:
    st.session_state["evidence_review_last_query"] = query

active_query = st.session_state.get("evidence_review_last_query", query)

# ---------------------------------------------------------------------
# 2. Search
# ---------------------------------------------------------------------

try:
    raw_results = _search_openalex(active_query)
except (URLError, HTTPError, ValueError, KeyError) as error:
    st.error(f"Could not reach OpenAlex: {error}")
    st.stop()

section_header("2. Search Results", f'OpenAlex query: "{active_query}"')

if not raw_results:
    st.info("OpenAlex returned no results for this query. Try different terms.")
    st.stop()

results = tuple(record_core.from_openalex_work(work) for work in raw_results)
scores = tuple(relevance_core.score_relevance(finding_text, item) for item in results)

inspect_note("The matched keywords listed under each result, not the overlap count alone.")

# ---------------------------------------------------------------------
# 3. Screen each result
# ---------------------------------------------------------------------

section_header(
    "3. Screen Each Result",
    "Include / Exclude / Uncertain, structured for PRISMA 2020-compatible reporting",
)

query_hash = hashlib.sha256(active_query.encode("utf-8")).hexdigest()[:10]
decisions: list[screening_core.ScreeningDecision] = []

for index, (item, score) in enumerate(zip(results, scores)):
    with st.container(border=True):
        st.markdown(f"**[{item.title}]({item.url})**")
        st.caption(
            f"{item.author_summary} · {item.year or 'n/a'} · {item.venue} · "
            f"{item.citation_count if item.citation_count is not None else 'n/a'} citations"
        )

        if item.abstract:
            st.write(item.abstract)
        else:
            st.caption("No abstract available from OpenAlex for this result.")

        if score.matched_keywords:
            st.caption(
                f"Matched keywords ({score.overlap_count}): "
                + ", ".join(score.matched_keywords)
            )
        else:
            st.caption("Matched keywords (0): none.")

        decision_key = f"evidence_review_decision_{query_hash}_{index}"
        decision_value = st.radio(
            "Screening decision",
            options=screening_core.DECISIONS,
            index=screening_core.DECISIONS.index(screening_core.DECISION_UNCERTAIN),
            key=decision_key,
            horizontal=True,
        )

        decision_reason = ""
        if decision_value == screening_core.DECISION_EXCLUDE:
            decision_reason = st.text_input(
                "Reason for excluding this result (optional, but PRISMA 2020 "
                "recommends stating why when a result might otherwise look "
                "eligible)",
                key=f"evidence_review_reason_{query_hash}_{index}",
            )

        decisions.append(
            screening_core.ScreeningDecision(
                record_title=item.title,
                decision=decision_value,
                reason=decision_reason,
            )
        )

summary = screening_core.summarize_screening(len(results), tuple(decisions))

# ---------------------------------------------------------------------
# 4. Screening summary and record
# ---------------------------------------------------------------------

section_header("4. Screening Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Found", summary.n_found)
c2.metric("Included", summary.n_included)
c3.metric("Excluded", summary.n_excluded)
c4.metric("Uncertain", summary.n_uncertain)

implications(
    "Included results are what you have judged relevant enough to read in "
    "full; overlap and citation counts above are sorting aids, not that "
    "judgment."
)

if summary.n_included == 0:
    flagged_item_note(
        "No results included",
        "Nothing has been marked Include yet. A record with zero included "
        "results is a valid outcome: it states that this search did not "
        "surface literature you judged relevant, not that none exists.",
    )

caveat(
    "OpenAlex's coverage and this query's exact wording both shape what "
    "was found. A different phrasing of the same finding can return "
    "different results."
)

records_df = pd.DataFrame(
    [
        {
            "Title": item.title,
            "Authors": item.author_summary,
            "Year": item.year,
            "Venue": item.venue,
            "DOI": item.doi,
            "URL": item.url,
            "Citation count": item.citation_count,
            "Matched keywords": ", ".join(score.matched_keywords),
            "Overlap count": score.overlap_count,
            "Screening decision": decision.decision,
            "Exclusion reason": decision.reason,
        }
        for item, score, decision in zip(results, scores, decisions)
    ]
)

st.dataframe(records_df, width="stretch", hide_index=True)

st.download_button(
    "Download structured evidence record (CSV)",
    data=records_df.to_csv(index=False).encode("utf-8"),
    file_name="evidence_review_record.csv",
    mime="text/csv",
)


def _record_evidence_review(query_text: str, df: pd.DataFrame, summary_: screening_core.ScreeningSummary) -> None:
    """
    Record this search+screen for Cross-Analysis Implications.

    Fingerprints the search-results table (public OpenAlex data), never
    the reviewer's own typed finding text, so nothing the reviewer wrote
    is retained in session state.
    """

    items = [
        RetentionItem(
            label="Marked Exclude by reviewer",
            count=summary_.n_excluded,
            kind=KIND_ROWS_DROPPED,
            mechanism="reviewer screened this result out",
        ),
        RetentionItem(
            label="Marked Uncertain or not yet screened",
            count=summary_.n_uncertain,
            kind=KIND_ROWS_DROPPED,
            mechanism="reviewer has not confirmed this result as relevant",
        ),
    ]

    HandoffStore(st.session_state).record(
        module=MODULE_EVIDENCE_REVIEW,
        fingerprint=fingerprint_dataframe(df, "OpenAlex search results"),
        exclusion=ExclusionAccount(
            module=MODULE_EVIDENCE_REVIEW,
            analysis_label="Evidence Review",
            columns_considered=tuple(df.columns),
            n_input_rows=summary_.n_found,
            n_retained_rows=summary_.n_included,
            items=tuple(items),
        ),
        primary_statistics={
            "n_found": float(summary_.n_found),
            "n_included": float(summary_.n_included),
        },
    )
    st.caption(
        f"Query: \"{query_text}\" (search terms only; the finding "
        "description above is not stored)."
    )


_record_evidence_review(active_query, records_df, summary)
