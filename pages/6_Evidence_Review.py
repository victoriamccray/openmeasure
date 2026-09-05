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
selection process; it does not itself supply this three-way label set.
Eligibility criteria are collected before screening (modules/
evidence_review/core/eligibility.py) but only for what OpenAlex's
metadata encodes in a structured way (year, title/abstract text);
population, outcome, and study-design criteria remain the reviewer's
judgment. Core logic lives in modules/evidence_review/core; this file
handles presentation and the one network call, matching
pages/FMRI_QC_Worked_Example.py's existing pattern for a public-remote
fetch.

Once a result is screened Include, this page can also preview
evidence_to_claim's Nesta grading (modules/evidence_to_claim/core) on
it, using arbitrary stand-in indicator/program values stated as such,
since this page has no portfolio taxonomy of its own.

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
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from modules.evidence_review.core import eligibility as eligibility_core
from modules.evidence_review.core import record as record_core
from modules.evidence_review.core import relevance as relevance_core
from modules.evidence_review.core import screening as screening_core
from modules.evidence_to_claim.core import claim as claim_core
from modules.evidence_to_claim.core import evidence as evidence_core
from modules.evidence_to_claim.core import strength as strength_core
from modules.evidence_to_claim.core import validate as validate_core
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
    Band,
    caveat,
    classify,
    flagged_item_note,
    implications,
    inspect_note,
    render_lifecycle_tracker,
    render_verdict,
    section_header,
    show_case_studies,
)

# Same bands as pages/Portfolio_Impact_Analysis.py's NESTA_LEVEL_BANDS,
# duplicated rather than imported: this page's use is illustrative-only
# and the two pages otherwise share no fairness/report state.
NESTA_LEVEL_BANDS = (
    Band(1, "Nesta Level 1: logic model only, no outcome data yet.", "error"),
    Band(2, "Nesta Level 2: data shows change, causality not established.", "warning"),
    Band(3, "Nesta Level 3: causality demonstrated via a comparison group.", "warning"),
    Band(4, "Nesta Level 4: causality confirmed by independent replication.", "success"),
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

Before screening, you can optionally state **eligibility criteria** -- a
minimum publication year, and terms required in the title or abstract -- and
watch how many results meet them update immediately, before any manual
judgment is applied.

Each result is then screened **Include / Exclude / Uncertain**, a common
screening convention structured for compatibility with PRISMA 2020 reporting
(Page et al., 2021), and the exact query, every result, and every decision
(including a stated reason for an Exclude) are recorded together as one
structured, exportable record.

Once at least one result is marked Include, this page can also preview
evidence_to_claim's Nesta Standards of Evidence grading on it, so a
screened-in result's evidentiary strength can be checked before treating it
as support for anything.

### What this page will not do

- It will not tell you a result is "the" relevant paper. A high keyword
  overlap can still be irrelevant, and a low one can still matter; screening
  that judgment is the reviewer's, not this page's.
- It will not infer your finding from a workflow's recorded result
  automatically in this version. Describe the finding yourself.
- Its eligibility criteria cover only what OpenAlex's metadata encodes in a
  structured way (publication year, title/abstract text). Population, outcome,
  and study-design criteria, which PRISMA 2020 also expects a review to
  define, are not checked mechanically here; the reviewer still applies those
  by judgment.
- Its Nesta-level preview uses arbitrary stand-in values for the indicator,
  program, and claim level, stated as such where they appear, because this
  page has no portfolio or indicator taxonomy of its own. It previews what
  the grading would show; it is not a substitute for assembling real evidence
  in Portfolio Impact Analysis with a real indicator and program.
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
    "Search terms sent to OpenAlex",
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
# 3. Set eligibility criteria
# ---------------------------------------------------------------------

section_header(
    "3. Set Eligibility Criteria",
    "Optional, but PRISMA 2020 expects criteria stated before screening",
)

st.caption(
    "These criteria label each result below; they do not remove any "
    "result from the list. Population, outcome, and study-design "
    "criteria are not checked here, since OpenAlex's metadata does not "
    "encode them in a structured way -- apply those by judgment during "
    "screening."
)

elig_col1, elig_col2 = st.columns([1, 2])

with elig_col1:
    use_min_year = st.checkbox("Require a minimum publication year")
    min_year = (
        st.number_input(
            "Minimum publication year",
            min_value=1900,
            max_value=2100,
            value=2015,
            step=1,
            label_visibility="collapsed",
        )
        if use_min_year
        else None
    )

with elig_col2:
    required_terms_text = st.text_input(
        "Terms that must appear in the title or abstract (comma-separated, optional)",
        placeholder="e.g., randomized, adolescent",
    )
    required_terms = tuple(
        term.strip() for term in required_terms_text.split(",") if term.strip()
    )

criteria = eligibility_core.EligibilityCriteria(
    min_year=min_year,
    required_terms=required_terms,
)

eligibility_summary = eligibility_core.assess_eligibility(results, criteria)

elig_metric_col, elig_note_col = st.columns([1, 3])
elig_metric_col.metric(
    "Eligible of found",
    f"{eligibility_summary.n_eligible} / {eligibility_summary.n_found}",
)
with elig_note_col:
    if criteria.is_empty:
        st.caption(
            "No criteria stated yet, so every result is eligible. Add a "
            "criterion above and watch this count change."
        )
    else:
        st.caption(
            "This count updates immediately as criteria change above: it "
            "shows how much a single criterion can prune the evidence "
            "base before any human judgment is applied."
        )

# ---------------------------------------------------------------------
# 4. Screen each result
# ---------------------------------------------------------------------

section_header(
    "4. Screen Each Result",
    "Include / Exclude / Uncertain, structured for PRISMA 2020-compatible reporting",
)

query_hash = hashlib.sha256(active_query.encode("utf-8")).hexdigest()[:10]
decisions: list[screening_core.ScreeningDecision] = []

for index, (item, score, assessment) in enumerate(
    zip(results, scores, eligibility_summary.assessments)
):
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

        if not criteria.is_empty:
            if assessment.eligible:
                st.caption("Meets the eligibility criteria stated above.")
            else:
                st.caption(
                    "Does not meet the eligibility criteria stated above: "
                    + "; ".join(assessment.reasons_excluded)
                    + ". Still yours to screen either way."
                )

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
# 5. Screening summary and record
# ---------------------------------------------------------------------

section_header("5. Screening Summary")

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

# ---------------------------------------------------------------------
# 6. From included results to a claim (illustrative preview)
# ---------------------------------------------------------------------

included_items = [
    item
    for item, decision in zip(results, decisions)
    if decision.decision == screening_core.DECISION_INCLUDE
]

if included_items:
    section_header(
        "6. From Included Results to a Claim",
        "Preview evidence_to_claim's Nesta grading on what you just screened in",
    )

    caveat(
        "Illustrative only. The indicator, program, and level identifiers "
        "below are arbitrary stand-in values, because this page has no "
        "portfolio or indicator taxonomy of its own. Assemble real "
        "evidence in Portfolio Impact Analysis for a claim that is not a "
        "preview."
    )

    claim_text = st.text_area(
        "Claim text (defaults to the finding you searched for)",
        value=finding_text,
        height=80,
        key=f"e2c_claim_text_{query_hash}",
    )

    claim_type = st.radio(
        "Claim type",
        options=claim_core.CLAIM_TYPES,
        horizontal=True,
        help=(
            "Output: what was delivered. Outcome: a measured change in "
            "participants. Impact: that change attributed to a program "
            "specifically, which conventionally requires a comparison "
            "group."
        ),
        key=f"e2c_claim_type_{query_hash}",
    )

    st.markdown(f"**Evidence items ({len(included_items)} included result(s))**")
    st.caption(
        "Sample size and comparison-group status are your manual read of "
        "each abstract, not values OpenAlex reports. Time lag is derived "
        "from publication year, a year-level approximation."
    )

    current_year = date.today().year
    evidence_items: list[evidence_core.EvidenceItem] = []

    for item_index, item in enumerate(included_items):
        cols = st.columns([3, 1, 1])
        cols[0].markdown(f"*{item.title}*")
        sample_size_input = cols[1].number_input(
            "Sample size (0 = unknown)",
            min_value=0,
            value=0,
            key=f"e2c_sample_{query_hash}_{item_index}",
        )
        has_comparison = cols[2].checkbox(
            "Comparison group",
            key=f"e2c_comparison_{query_hash}_{item_index}",
        )

        time_lag_days = (
            (current_year - item.year) * 365 if item.year is not None else None
        )

        evidence_items.append(
            evidence_core.EvidenceItem(
                source=item.title,
                finding_text=item.abstract or item.title,
                indicator_id=f"evidence-review-{query_hash}",
                sample_size=sample_size_input or None,
                has_comparison_group=has_comparison,
                collection_method="published_literature",
                time_lag_days=time_lag_days,
            )
        )

    claim_id = f"evidence-review-{query_hash}"

    try:
        preview_claim = claim_core.ClaimDraft(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_type=claim_type,
            level="program",
            program_id="evidence-review-illustration",
        )
        bundle = evidence_core.summarize_evidence(evidence_items, claim_id)
        validation = validate_core.validate_evidence(bundle)
        supported = strength_core.determine_supported_claim(
            preview_claim, bundle, validation
        )
    except ValueError as error:
        st.error(str(error))
    else:
        render_verdict(classify(supported.nesta_level.level, NESTA_LEVEL_BANDS))

        st.caption(f"Framework: {supported.framework_citation}")

        if supported.claim_type_alignment_warning:
            st.warning(supported.claim_type_alignment_warning)

        if supported.next_level_hint:
            st.info(supported.next_level_hint)

        implications(
            "Being found and marked Include is not the same as clearing "
            "an evidentiary bar for a claim: the level above reflects "
            "only the causal rigor of what was just assembled, not "
            "whether it is enough evidence for this specific claim."
        )
