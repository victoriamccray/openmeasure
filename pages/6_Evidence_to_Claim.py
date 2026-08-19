"""
OpenMeasure - Evidence to Claim Journey.

Given a program, grantee, or portfolio result, what can be responsibly
claimed about it, and how should that be communicated? Walks an analyst
through defining a claim, assembling evidence behind it, checking that
evidence against configurable minimum-bar thresholds, grading its causal
rigor against Nesta's Standards of Evidence, flagging limitations, placing
it in portfolio context, and drafting a leadership-ready summary. Core
logic lives in modules/evidence_to_claim/core, this file is presentation
only.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from modules.evidence_to_claim.core import claim as claim_core
from modules.evidence_to_claim.core import evidence as evidence_core
from modules.evidence_to_claim.core import limitations as limitations_core
from modules.evidence_to_claim.core import portfolio as portfolio_core
from modules.evidence_to_claim.core import record as record_core
from modules.evidence_to_claim.core import strength as strength_core
from modules.evidence_to_claim.core import validate as validate_core
from shared.catalog import MODULE_EVIDENCE_TO_CLAIM
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
    render_lifecycle_tracker,
    render_verdict,
    section_header,
    show_case_studies,
)

SAMPLE_DIR = ROOT / "modules" / "evidence_to_claim" / "sample_data"

# Tone per Nesta level, for this page's headline verdict only. The level
# number itself (1-4) is core.strength's determination; this is just how it
# is colored on screen.
NESTA_LEVEL_BANDS = (
    Band(1, "Nesta Level 1: logic model only, no outcome data yet.", "error"),
    Band(2, "Nesta Level 2: data shows change, causality not established.", "warning"),
    Band(3, "Nesta Level 3: causality demonstrated via a comparison group.", "warning"),
    Band(4, "Nesta Level 4: causality confirmed by independent replication.", "success"),
)

PORTFOLIO_POSITION_MESSAGES = {
    portfolio_core.POSITION_WITHIN_RANGE: (
        "Within the rest of the portfolio's typical range for this indicator."
    ),
    portfolio_core.POSITION_ABOVE_RANGE: (
        "Above the rest of the portfolio's typical range for this indicator "
        "(outside the Tukey fence). Worth checking why, not necessarily a "
        "problem."
    ),
    portfolio_core.POSITION_BELOW_RANGE: (
        "Below the rest of the portfolio's typical range for this indicator "
        "(outside the Tukey fence). Worth checking why, not necessarily a "
        "problem."
    ),
}

STAGE_KEY = "etc_stage"

STAGE_DEFINE_CLAIM = 0
STAGE_DESCRIBE_EVIDENCE = 1
STAGE_VALIDATE = 2
STAGE_DETERMINE_SUPPORTED_CLAIM = 3
STAGE_EXAMINE_LIMITATIONS = 4
STAGE_PORTFOLIO_CONTEXT = 5
STAGE_EVIDENCE_RECORD = 6

JOURNEY_STAGES = (
    "Define claim",
    "Describe evidence",
    "Validate",
    "Determine supported claim",
    "Examine limitations",
    "Portfolio context",
    "Evidence record",
)

SESSION_KEYS = (
    STAGE_KEY,
    "etc_claim",
    "etc_evidence_frame",
    "etc_evidence_filename",
    "etc_bundle",
    "etc_validation",
    "etc_supported",
    "etc_limitations",
    "etc_portfolio_context",
)


def _current_stage() -> int:
    return st.session_state.get(STAGE_KEY, STAGE_DEFINE_CLAIM)


def _advance_to(stage: int) -> None:
    st.session_state[STAGE_KEY] = max(_current_stage(), stage)
    st.rerun()


def _none_if_nan(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _claim_from_row(row: pd.Series) -> claim_core.ClaimDraft:
    related_raw = str(row.get("related_indicator_ids", "") or "")
    related = tuple(x.strip() for x in related_raw.split(",") if x.strip())
    return claim_core.ClaimDraft(
        claim_id=str(row["claim_id"]),
        claim_text=str(row["claim_text"]),
        claim_type=str(row["claim_type"]),
        level=str(row["level"]),
        program_id=_none_if_nan(row.get("program_id")),
        grantee_id=_none_if_nan(row.get("grantee_id")),
        portfolio_id=_none_if_nan(row.get("portfolio_id")),
        related_indicator_ids=related,
    )


def _item_from_row(row: pd.Series) -> evidence_core.EvidenceItem:
    sample_size = row.get("sample_size")
    time_lag = row.get("time_lag_days")
    return evidence_core.EvidenceItem(
        source=str(row["source"]),
        finding_text=str(row["finding_text"]),
        indicator_id=_none_if_nan(row.get("indicator_id")),
        sample_size=int(sample_size) if pd.notna(sample_size) else None,
        has_comparison_group=str(row.get("has_comparison_group", "")).strip().lower() == "yes",
        collection_method=str(row["collection_method"]),
        time_lag_days=int(time_lag) if pd.notna(time_lag) else None,
    )


def _this_value_for_claim(portfolio_df: pd.DataFrame, claim: claim_core.ClaimDraft):
    if not claim.grantee_id or not claim.related_indicator_ids:
        return None, None

    indicator_id = claim.related_indicator_ids[0]
    matches = portfolio_df[
        (portfolio_df["grantee_id"] == claim.grantee_id)
        & (portfolio_df["indicator_id"] == indicator_id)
    ]

    if matches.empty:
        return None, None

    return indicator_id, float(matches.iloc[0]["result_value"])


def record_evidence_to_claim(
    evidence_frame: pd.DataFrame,
    evidence_filename: str,
    bundle: evidence_core.EvidenceBundle,
    claim: claim_core.ClaimDraft,
    supported: strength_core.SupportedClaimResult,
    validation: validate_core.ValidationResult,
) -> None:
    """
    Record this analysis for the Cross-Analysis Implications page.

    Translates into primitives here rather than storing the result object,
    because the same dataclass is a different class depending on how it was
    imported.
    """
    items = (
        (
            RetentionItem(
                label="Evidence items excluded",
                count=bundle.n_excluded_items,
                kind=KIND_ROWS_DROPPED,
                mechanism=bundle.exclusion_reason,
            ),
        )
        if bundle.n_excluded_items
        else ()
    )

    account = ExclusionAccount(
        module=MODULE_EVIDENCE_TO_CLAIM,
        analysis_label=f"Evidence to Claim: {claim.claim_id}",
        columns_considered=(
            "finding_text",
            "indicator_id",
            "sample_size",
            "has_comparison_group",
            "collection_method",
            "time_lag_days",
        ),
        n_input_rows=bundle.n_input_items,
        n_retained_rows=bundle.n_usable_items,
        items=items,
    )

    HandoffStore(st.session_state).record(
        module=MODULE_EVIDENCE_TO_CLAIM,
        fingerprint=fingerprint_dataframe(evidence_frame, evidence_filename),
        exclusion=account,
        primary_statistics={
            "nesta_level": float(supported.nesta_level.level),
            "n_checks_passed": float(validation.n_checks_passed),
        },
    )


st.set_page_config(
    page_title="OpenMeasure · Evidence to Claim",
    page_icon=":material/fact_check:",
    layout="centered",
)

st.title("Evidence to Claim")
st.subheader("Cross-cutting validation")
st.caption(
    "What can we responsibly say about impact, based on the evidence we have?"
)

st.divider()

render_lifecycle_tracker(current_workflow="Evidence to Claim")

show_case_studies("program_validation")

st.divider()

with st.expander("About this journey"):
    st.markdown(
        """
This module never issues a go/no-go verdict on a claim. It states which
level of causal rigor the evidence assembled behind a claim reaches
(Puttick & Ludlow's Nesta Standards of Evidence, Levels 1-4 of 5), flags
limitations and comparability issues, and drafts reporting language you can
edit rather than publish unread.

**The seven stages:** Define claim -> Describe evidence -> Validate ->
Determine supported claim -> Examine limitations -> Portfolio context ->
Evidence record.

v0.1 ships with a small fictional portfolio ("Riverbend Youth Futures"),
not a real integration. The data shape is intentionally close to what a
MEL system export (e.g. GivingData, Rural Senses) would look like.
"""
    )

stage = _current_stage()

_stage_parts = [
    f"**{label}**" if index == stage else label
    for index, label in enumerate(JOURNEY_STAGES)
]

with st.container(border=True):
    st.markdown(" → ".join(_stage_parts))

if stage > STAGE_DEFINE_CLAIM:
    if st.button("Restart", icon=":material/restart_alt:"):
        for key in SESSION_KEYS:
            st.session_state.pop(key, None)
        st.rerun()

st.divider()

# ---------------------------------------------------------------------
# 1. Define claim
# ---------------------------------------------------------------------

section_header("1. Define claim", "What are you claiming, and at what level?")

sample_claims = pd.read_csv(SAMPLE_DIR / "claims.csv")

use_sample = st.radio(
    "Claim source",
    options=["Use a sample claim", "Define a custom claim"],
    index=0,
    horizontal=True,
)

if use_sample == "Use a sample claim":
    chosen_id = st.selectbox(
        "Sample claim",
        options=sample_claims["claim_id"],
        format_func=lambda cid: f"{cid}: {sample_claims.set_index('claim_id').loc[cid, 'claim_text']}",
    )
    row = sample_claims.set_index("claim_id").loc[chosen_id]
    row["claim_id"] = chosen_id
    st.caption(f"Claim type: **{row['claim_type']}**, level: **{row['level']}**")

    if st.button("Use this claim", type="primary"):
        try:
            st.session_state["etc_claim"] = _claim_from_row(row)
            for key in SESSION_KEYS[2:]:
                st.session_state.pop(key, None)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
else:
    claim_id = st.text_input("Claim ID", value="CLAIM-CUSTOM-01")
    claim_text = st.text_area("Claim text", placeholder="e.g. Our program increased participants' confidence.")
    claim_type = st.selectbox("Claim type", options=claim_core.CLAIM_TYPES, index=1)
    level = st.selectbox("Level", options=claim_core.CLAIM_LEVELS, index=1)

    program_id = grantee_id = portfolio_id = None
    if level == "program":
        program_id = st.text_input("Program ID")
    elif level == "grantee":
        grantee_id = st.text_input("Grantee ID")
    else:
        portfolio_id = st.text_input("Portfolio ID")

    related_raw = st.text_input("Related indicator ID(s), comma-separated", value="")
    related = tuple(x.strip() for x in related_raw.split(",") if x.strip())

    if st.button("Create claim", type="primary"):
        try:
            st.session_state["etc_claim"] = claim_core.ClaimDraft(
                claim_id=claim_id,
                claim_text=claim_text,
                claim_type=claim_type,
                level=level,
                program_id=program_id or None,
                grantee_id=grantee_id or None,
                portfolio_id=portfolio_id or None,
                related_indicator_ids=related,
            )
            for key in SESSION_KEYS[2:]:
                st.session_state.pop(key, None)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

if "etc_claim" in st.session_state and stage < STAGE_DESCRIBE_EVIDENCE:
    if st.button("Continue to describe evidence", type="primary"):
        _advance_to(STAGE_DESCRIBE_EVIDENCE)

# ---------------------------------------------------------------------
# 2. Describe evidence
# ---------------------------------------------------------------------

if stage >= STAGE_DESCRIBE_EVIDENCE and "etc_claim" in st.session_state:
    claim = st.session_state["etc_claim"]

    section_header("2. Describe evidence", f"Evidence assembled behind {claim.claim_id}")

    sample_evidence = pd.read_csv(SAMPLE_DIR / "evidence_items.csv")
    matching_sample = sample_evidence[sample_evidence["claim_id"] == claim.claim_id]

    uploaded_evidence = st.file_uploader(
        "Upload a custom evidence CSV (optional; columns: source, finding_text, "
        "indicator_id, sample_size, has_comparison_group, collection_method, "
        "time_lag_days)",
        type="csv",
    )

    if uploaded_evidence is not None:
        evidence_frame = pd.read_csv(uploaded_evidence)
        evidence_filename = uploaded_evidence.name
    elif not matching_sample.empty:
        evidence_frame = matching_sample
        evidence_filename = "evidence_items.csv (sample)"
    else:
        evidence_frame = sample_evidence
        evidence_filename = "evidence_items.csv (sample, unfiltered)"
        st.info(
            "No bundled evidence rows match this claim ID. Showing the full "
            "sample; upload a custom evidence CSV to describe evidence for "
            "a custom claim."
        )

    st.dataframe(evidence_frame, width="stretch", hide_index=True)

    if st.button("Summarize evidence", type="primary"):
        try:
            items = tuple(_item_from_row(r) for _, r in evidence_frame.iterrows())
            bundle = evidence_core.summarize_evidence(items, claim_id=claim.claim_id)
            st.session_state["etc_bundle"] = bundle
            st.session_state["etc_evidence_frame"] = evidence_frame
            st.session_state["etc_evidence_filename"] = evidence_filename
            for key in SESSION_KEYS[5:]:
                st.session_state.pop(key, None)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    if "etc_bundle" in st.session_state:
        bundle = st.session_state["etc_bundle"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Evidence items received", bundle.n_input_items)
        c2.metric("Usable", bundle.n_usable_items)
        c3.metric("Excluded", bundle.n_excluded_items)
        if bundle.n_excluded_items:
            caveat(f"Excluded because: {bundle.exclusion_reason}.")

        if stage < STAGE_VALIDATE:
            if st.button("Continue to validate", type="primary"):
                _advance_to(STAGE_VALIDATE)

# ---------------------------------------------------------------------
# 3. Validate
# ---------------------------------------------------------------------

if stage >= STAGE_VALIDATE and "etc_bundle" in st.session_state:
    bundle = st.session_state["etc_bundle"]

    section_header("3. Validate", "Configurable minimum-bar thresholds")

    t1, t2, t3 = st.columns(3)
    min_sample_size = t1.number_input(
        "Minimum sample size", min_value=1, value=validate_core.MIN_SAMPLE_SIZE_DEFAULT
    )
    min_corroboration = t2.number_input(
        "Minimum independent sources", min_value=1, value=validate_core.MIN_CORROBORATION_DEFAULT
    )
    max_time_lag_days = t3.number_input(
        "Maximum time lag (days)", min_value=1, value=validate_core.MAX_TIME_LAG_DAYS_DEFAULT
    )
    st.caption(
        "These defaults are OpenMeasure conventions informed by general "
        "principles in the cited literature, not fixed thresholds drawn "
        "from it. See the module README for citations."
    )

    if st.button("Run validation", type="primary"):
        try:
            validation = validate_core.validate_evidence(
                bundle,
                min_sample_size=int(min_sample_size),
                min_corroboration=int(min_corroboration),
                max_time_lag_days=int(max_time_lag_days),
            )
            st.session_state["etc_validation"] = validation
            for key in SESSION_KEYS[6:]:
                st.session_state.pop(key, None)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    if "etc_validation" in st.session_state:
        validation = st.session_state["etc_validation"]
        checks_df = pd.DataFrame(
            [
                {
                    "Criterion": c.criterion,
                    "Passed": "Yes" if c.passed else "No",
                    "Severity": c.severity,
                    "Detail": c.detail,
                }
                for c in validation.checks
            ]
        )
        st.dataframe(checks_df, width="stretch", hide_index=True)

        if stage < STAGE_DETERMINE_SUPPORTED_CLAIM:
            if st.button("Continue to determine supported claim", type="primary"):
                _advance_to(STAGE_DETERMINE_SUPPORTED_CLAIM)

# ---------------------------------------------------------------------
# 4. Determine supported claim
# ---------------------------------------------------------------------

if stage >= STAGE_DETERMINE_SUPPORTED_CLAIM and "etc_validation" in st.session_state:
    claim = st.session_state["etc_claim"]
    bundle = st.session_state["etc_bundle"]
    validation = st.session_state["etc_validation"]

    section_header("4. Determine supported claim", "Graded against Nesta's Standards of Evidence")

    supported = strength_core.determine_supported_claim(claim, bundle, validation)
    st.session_state["etc_supported"] = supported

    render_verdict(classify(supported.nesta_level.level, NESTA_LEVEL_BANDS))
    st.write(supported.suggested_language)
    st.caption(supported.framework_citation)

    if supported.claim_type_alignment_warning:
        st.warning(supported.claim_type_alignment_warning)

    caveat(supported.level_5_note)
    caveat(
        "A Nesta level describes the evidence's causal rigor, not whether "
        "the underlying claim is true."
    )

    if stage < STAGE_EXAMINE_LIMITATIONS:
        if st.button("Continue to examine limitations", type="primary"):
            _advance_to(STAGE_EXAMINE_LIMITATIONS)

# ---------------------------------------------------------------------
# 5. Examine limitations
# ---------------------------------------------------------------------

if stage >= STAGE_EXAMINE_LIMITATIONS and "etc_supported" in st.session_state:
    bundle = st.session_state["etc_bundle"]
    validation = st.session_state["etc_validation"]

    section_header("5. Examine limitations")

    limitations = limitations_core.examine_limitations(bundle, validation)
    st.session_state["etc_limitations"] = limitations

    if limitations.n_flags == 0:
        st.success("No limitations flagged against the configured thresholds.")
    else:
        for flag in limitations.flags:
            flagged_item_note(f"{flag.category} ({flag.severity})", flag.message)

    if stage < STAGE_PORTFOLIO_CONTEXT:
        if st.button("Continue to portfolio context", type="primary"):
            _advance_to(STAGE_PORTFOLIO_CONTEXT)

# ---------------------------------------------------------------------
# 6. Portfolio context
# ---------------------------------------------------------------------

if stage >= STAGE_PORTFOLIO_CONTEXT and "etc_limitations" in st.session_state:
    claim = st.session_state["etc_claim"]

    section_header("6. Portfolio context", "Optional: how this compares to the rest of a portfolio")

    uploaded_portfolio = st.file_uploader(
        "Upload a custom portfolio CSV (optional; columns include grantee_id, "
        "indicator_id, result_value, unit)",
        type="csv",
        key="etc_portfolio_uploader",
    )

    if uploaded_portfolio is not None:
        portfolio_df = pd.read_csv(uploaded_portfolio)
    else:
        portfolio_df = pd.read_csv(SAMPLE_DIR / "portfolio_indicators.csv")

    indicator_id, this_value = _this_value_for_claim(portfolio_df, claim)

    portfolio_context = None
    if indicator_id is None:
        st.info(
            "No matching portfolio row for this claim's grantee and "
            "indicator; skipping portfolio context."
        )
    else:
        indicator_summary = portfolio_core.summarize_portfolio_indicator(
            portfolio_df, indicator_id, this_value
        )
        comparability_flags = portfolio_core.check_comparability(portfolio_df)
        portfolio_context = portfolio_core.PortfolioContextResult(
            indicator_summary=indicator_summary,
            comparability_flags=comparability_flags,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("This value", indicator_summary.this_value)
        c2.metric("Portfolio median", round(indicator_summary.median_value, 1))
        c3.metric("Grantees reporting", indicator_summary.n_grantees_reporting)

        message = PORTFOLIO_POSITION_MESSAGES[indicator_summary.position]
        if indicator_summary.position == portfolio_core.POSITION_WITHIN_RANGE:
            st.info(message)
        else:
            st.warning(message)

        for flag in comparability_flags:
            flagged_item_note(f"Comparability ({flag.severity})", flag.message)

    st.session_state["etc_portfolio_context"] = portfolio_context

    if stage < STAGE_EVIDENCE_RECORD:
        if st.button("Continue to evidence record", type="primary"):
            _advance_to(STAGE_EVIDENCE_RECORD)

# ---------------------------------------------------------------------
# 7. Evidence record
# ---------------------------------------------------------------------

if stage >= STAGE_EVIDENCE_RECORD and "etc_portfolio_context" in st.session_state:
    claim = st.session_state["etc_claim"]
    bundle = st.session_state["etc_bundle"]
    validation = st.session_state["etc_validation"]
    supported = st.session_state["etc_supported"]
    limitations = st.session_state["etc_limitations"]
    portfolio_context = st.session_state["etc_portfolio_context"]

    section_header("7. Evidence record", "Leadership-ready summary, with technical detail on request")

    record = record_core.EvidenceRecord(
        claim=claim,
        evidence=bundle,
        validation=validation,
        supported_claim=supported,
        limitations=limitations,
        portfolio_context=portfolio_context,
    )

    st.success(record_core.build_leadership_summary(record))

    caveat(
        "This is a model of evidence strength, not a determination of "
        "whether the claim is true or a recommendation to publish it."
    )

    with st.expander("Technical detail"):
        st.write("**Claim**")
        st.write(record.claim)
        st.write("**Evidence bundle**")
        st.write(record.evidence)
        st.write("**Validation**")
        st.write(record.validation)
        st.write("**Supported claim**")
        st.write(record.supported_claim)
        st.write("**Limitations**")
        st.write(record.limitations)
        if record.portfolio_context is not None:
            st.write("**Portfolio context**")
            st.write(record.portfolio_context)

    record_evidence_to_claim(
        st.session_state["etc_evidence_frame"],
        st.session_state["etc_evidence_filename"],
        bundle,
        claim,
        supported,
        validation,
    )
    st.caption(
        "Recorded for the Cross-Analysis Implications page, which shows "
        "how much of your data each analysis used."
    )
