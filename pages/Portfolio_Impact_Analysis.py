"""
Portfolio Impact Analysis - a guided research journey, not a workflow.

A review layer over evidence an analyst has already assembled about a
program, grantee, or portfolio result: what does it support, why, can it
be compared to other results, and what is defensible to report. It
records nothing to shared/handoff.py, carries no module_key, and is
deliberately not a numbered page, so it needs no entry in
shared/catalog.py (see shared/tests/test_catalog.py, which only requires
a catalog entry for numbered pages). Explore Real Data, Method Selection,
and the other Research Journeys already establish this "not a workflow"
pattern.

Core logic lives in modules/evidence_to_claim/core, this file is
presentation only. (The module folder keeps its original name; see the
module README.)
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
from shared.data_handling import disclosure_for, render_data_handling_summary
from shared.report import (
    Band,
    caveat,
    classify,
    flagged_item_note,
    render_verdict,
    section_header,
)

SAMPLE_DIR = ROOT / "modules" / "evidence_to_claim" / "sample_data"

# Validated default palette (dataviz skill, references/palette.md).
# Categorical slots used in fixed order: slot 1 (blue) for portfolio
# points, slot 2 (orange) to highlight the claim currently being reviewed.
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
CATEGORICAL_1 = "#2a78d6"
CATEGORICAL_2 = "#eb6834"

_VEGA_CHART_CONFIG = {
    "background": SURFACE,
    "axis": {
        "gridColor": GRIDLINE,
        "domainColor": "#c3c2b7",
        "tickColor": "#c3c2b7",
        "labelColor": "#898781",
        "titleColor": INK_SECONDARY,
    },
    "view": {"stroke": "transparent"},
}

# Tone for the headline verdict banner. The level number (1-4) is
# core.strength's determination; this is only how it is colored on screen.
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
        "Above the rest of the portfolio's typical range for this "
        "indicator (outside the Tukey fence: more than 1.5x the "
        "middle 50% of values above the top of that middle range). "
        "Worth checking why, not necessarily a problem."
    ),
    portfolio_core.POSITION_BELOW_RANGE: (
        "Below the rest of the portfolio's typical range for this "
        "indicator (outside the Tukey fence: more than 1.5x the "
        "middle 50% of values below the bottom of that middle range). "
        "Worth checking why, not necessarily a problem."
    ),
}

CRITERION_LABELS = {
    validate_core.CRITERION_COMPARISON_GROUP: "Comparison group",
    validate_core.CRITERION_SAMPLE_SIZE: "Sample size",
    validate_core.CRITERION_CORROBORATION: "Multiple sources",
    validate_core.CRITERION_TIME_LAG: "Recency",
}

CRITERION_WHY = {
    validate_core.CRITERION_COMPARISON_GROUP: (
        "The only blocking check. Without a comparison group, a change "
        "cannot be attributed to the program rather than another cause."
    ),
    validate_core.CRITERION_SAMPLE_SIZE: (
        "A small sample makes the estimate unstable and easily overturned "
        "by a few cases."
    ),
    validate_core.CRITERION_CORROBORATION: (
        "A single source is easier to overturn than several independent "
        "ones reporting the same finding."
    ),
    validate_core.CRITERION_TIME_LAG: (
        "Older evidence may no longer describe current conditions."
    ),
}

STAGE_KEY = "pia_stage"

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
    "pia_claim",
    "pia_evidence_frame",
    "pia_evidence_filename",
    "pia_bundle",
    "pia_validation",
    "pia_supported",
    "pia_limitations",
    "pia_portfolio_context",
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


def _friendly(value, names: dict, show_ids: bool):
    """Map an ID to a human-readable label, unless the technical-ID toggle is on."""
    if value is None or show_ids:
        return value
    return names.get(value, value)


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


def _portfolio_evidence_map_points(
    sample_claims_df: pd.DataFrame,
    sample_evidence_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    current_claim_id: str | None,
) -> pd.DataFrame:
    """
    One row per sample grantee claim: reported value, unit, and evidence
    strength, for the portfolio evidence map. Skips any claim or evidence
    row that fails to construct (e.g. incomplete sample data).
    """
    rows = []
    for _, crow in sample_claims_df.iterrows():
        try:
            claim = _claim_from_row(crow)
        except ValueError:
            continue

        matching = sample_evidence_df[sample_evidence_df["claim_id"] == claim.claim_id]
        if matching.empty:
            continue

        try:
            items = tuple(_item_from_row(r) for _, r in matching.iterrows())
            bundle = evidence_core.summarize_evidence(items, claim_id=claim.claim_id)
            validation = validate_core.validate_evidence(bundle)
            supported = strength_core.determine_supported_claim(claim, bundle, validation)
        except ValueError:
            continue

        indicator_id, value = _this_value_for_claim(portfolio_df, claim)
        if indicator_id is None:
            continue

        unit_rows = portfolio_df[
            (portfolio_df["grantee_id"] == claim.grantee_id)
            & (portfolio_df["indicator_id"] == indicator_id)
        ]
        unit = str(unit_rows.iloc[0]["unit"]) if not unit_rows.empty else "unknown"

        rows.append(
            {
                "claim_id": claim.claim_id,
                "grantee_id": claim.grantee_id,
                "indicator_id": indicator_id,
                "value": value,
                "unit": unit,
                "level": supported.nesta_level.level,
                "is_current": claim.claim_id == current_claim_id,
            }
        )

    return pd.DataFrame(rows)


def _kv_table(pairs: list[tuple[str, object]]) -> None:
    """A static two-column table, for readable technical detail (no raw JSON/repr)."""
    rows = [(field, str(value)) for field, value in pairs]
    df = pd.DataFrame(rows, columns=["Field", "Value"]).set_index("Field")
    st.table(df)


def _render_claim_detail(claim: claim_core.ClaimDraft) -> None:
    _kv_table(
        [
            ("Claim ID", claim.claim_id),
            ("Claim text", claim.claim_text),
            ("Claim type", claim.claim_type),
            ("Level", claim.level),
            ("Program ID", claim.program_id or "-"),
            ("Grantee ID", claim.grantee_id or "-"),
            ("Portfolio ID", claim.portfolio_id or "-"),
            ("Related indicators", ", ".join(claim.related_indicator_ids) or "-"),
        ]
    )


def _render_evidence_detail(bundle: evidence_core.EvidenceBundle) -> None:
    _kv_table(
        [
            ("Evidence items received", bundle.n_input_items),
            ("Usable", bundle.n_usable_items),
            ("Excluded", bundle.n_excluded_items),
            ("Exclusion reason", bundle.exclusion_reason if bundle.n_excluded_items else "-"),
            ("Independent sources", bundle.n_sources),
            ("Collection methods used", bundle.n_collection_methods),
            ("Any comparison group reported", "Yes" if bundle.has_any_comparison_group else "No"),
            (
                "Smallest reported sample size",
                bundle.min_sample_size if bundle.min_sample_size is not None else "-",
            ),
            (
                "Oldest evidence (days)",
                bundle.max_time_lag_days if bundle.max_time_lag_days is not None else "-",
            ),
        ]
    )
    items_df = pd.DataFrame(
        [
            {
                "Source": item.source,
                "Finding": item.finding_text,
                "Comparison group": "Yes" if item.has_comparison_group else "No",
                "Method": item.collection_method,
                "Sample size": item.sample_size if item.sample_size is not None else "-",
                "Age (days)": item.time_lag_days if item.time_lag_days is not None else "-",
            }
            for item in bundle.items
        ]
    )
    st.dataframe(items_df, width="stretch", hide_index=True)


def _render_validation_detail(validation: validate_core.ValidationResult) -> None:
    checks_df = pd.DataFrame(
        [
            {
                "Criterion": CRITERION_LABELS.get(c.criterion, c.criterion),
                "Passed": "Yes" if c.passed else "No",
                "Severity": c.severity,
                "Detail": c.detail,
            }
            for c in validation.checks
        ]
    )
    st.dataframe(checks_df, width="stretch", hide_index=True)
    _kv_table(
        [
            ("Checks passed", f"{validation.n_checks_passed} of {validation.n_checks_total}"),
            ("Meets minimum bar", "Yes" if validation.meets_minimum_bar else "No"),
        ]
    )


def _render_supported_detail(supported: strength_core.SupportedClaimResult) -> None:
    _kv_table(
        [
            ("Nesta level", f"Level {supported.nesta_level.level}: {supported.nesta_level.label}"),
            ("What that level means", supported.nesta_level.description),
            ("Framework citation", supported.framework_citation),
            (
                "Claim-type alignment",
                supported.claim_type_alignment_warning or "No tension flagged",
            ),
            (
                "What would strengthen it further",
                supported.next_level_hint or "Already at the highest level this module assesses",
            ),
        ]
    )


def _render_limitations_detail(limitations: limitations_core.LimitationsResult) -> None:
    if limitations.n_flags == 0:
        st.write("No limitations flagged.")
        return
    flags_df = pd.DataFrame(
        [
            {"Category": f.category, "Severity": f.severity, "Message": f.message}
            for f in limitations.flags
        ]
    )
    st.dataframe(flags_df, width="stretch", hide_index=True)


def _render_portfolio_context_detail(
    pc: portfolio_core.PortfolioContextResult | None, show_ids: bool
) -> None:
    if pc is None or pc.indicator_summary is None:
        st.write("No portfolio context available for this claim.")
        return
    s = pc.indicator_summary
    _kv_table(
        [
            ("Indicator", _friendly(s.indicator_id, INDICATOR_NAMES, show_ids)),
            ("This grantee's result", s.this_value),
            ("Portfolio median", round(s.median_value, 1)),
            ("Portfolio range (Q1-Q3)", f"{round(s.q1, 1)} - {round(s.q3, 1)}"),
            ("Typical range (Tukey fences)", f"{round(s.lower_fence, 1)} - {round(s.upper_fence, 1)}"),
            ("Grantees reporting", s.n_grantees_reporting),
            ("Position", s.position.replace("_", " ")),
        ]
    )
    if pc.comparability_flags:
        flags_df = pd.DataFrame(
            [{"Indicator": f.indicator_id, "Message": f.message} for f in pc.comparability_flags]
        )
        st.dataframe(flags_df, width="stretch", hide_index=True)


st.set_page_config(
    page_title="OpenMeasure · Portfolio Impact Analysis",
    page_icon=":material/fact_check:",
    layout="centered",
)

st.title("Portfolio Impact Analysis")
st.caption(
    "A transparent review layer that helps analysts move from heterogeneous "
    "program evidence to defensible portfolio-level conclusions."
)
st.caption(
    "Bring your own data: an evidence CSV (source, finding, indicator, "
    "sample size, comparison group, method, age) and a portfolio CSV "
    "(grantee, indicator, value, unit) can replace the sample data below."
)

render_data_handling_summary(disclosure_for("pages/Portfolio_Impact_Analysis.py"))

st.divider()

with st.expander("How to use this workspace"):
    st.markdown(
        """
**Use it to:**
- Review whether reported results support a proposed claim.
- Identify limitations or evidence gaps.
- Determine which grantee results can reasonably be compared or synthesized.
- Produce concise, defensible language for leadership, board, donor, or
  public reporting.

**Seven steps**, each separating a distinct question - *finding* (what you
claim), *evidence* (what backs it), *interpretation* (how strong that
evidence is and what it does not establish), and *claim* (defensible
language to report).

**Output vs. outcome vs. impact** are escalating evidentiary expectations,
not synonyms. Output is what was delivered. Outcome is a measured change,
with no claim about cause. Impact is that change attributed to the
program, which conventionally needs a comparison group. The claim type you
pick sets the evidence level conventionally expected of it; a mismatch is
flagged, not blocked.

No step issues an automated decision. Each states what the evidence
supports; the judgment stays with you.

Sample data below is synthetic (no real grantees). It follows the same
row-per-indicator shape as a typical grant-management or MEL export, and
the workflow generalizes across domains such as health, education,
workforce development, environment, or any portfolio that reports
indicators and results.
"""
    )

show_ids = st.toggle("Show technical IDs", value=False)

_label_source = pd.read_csv(SAMPLE_DIR / "portfolio_indicators.csv")
GRANTEE_NAMES = (
    _label_source.drop_duplicates("grantee_id").set_index("grantee_id")["grantee_name"].to_dict()
)
INDICATOR_NAMES = (
    _label_source.drop_duplicates("indicator_id").set_index("indicator_id")["indicator_name"].to_dict()
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
# 1. Define claim (finding)
# ---------------------------------------------------------------------

section_header("1. Define Claim", "Finding - what you are claiming, and at what level.")

st.caption(
    "Output, outcome, and impact are escalating evidentiary expectations, "
    "not synonyms. Output is what was delivered. Outcome is a measured "
    "change, with no claim about cause. Impact is that change attributed "
    "to the program, which conventionally needs a comparison group."
)

sample_claims = pd.read_csv(SAMPLE_DIR / "claims.csv")

use_sample = st.radio(
    "Claim source",
    options=["Use a sample claim", "Define a custom claim"],
    index=0,
    horizontal=True,
)

if use_sample == "Use a sample claim":
    def _claim_option_label(cid: str) -> str:
        row = sample_claims.set_index("claim_id").loc[cid]
        if show_ids:
            return f"{cid}: {row['claim_text']}"
        grantee_label = _friendly(row.get("grantee_id"), GRANTEE_NAMES, show_ids)
        return f"{grantee_label} - {row['claim_text']}"

    chosen_id = st.selectbox(
        "Sample claim",
        options=sample_claims["claim_id"],
        format_func=_claim_option_label,
    )
    row = sample_claims.set_index("claim_id").loc[chosen_id]
    row["claim_id"] = chosen_id
    st.caption(f"Claim type **{row['claim_type']}**, at the **{row['level']}** level.")

    if st.button("Use this claim", type="primary"):
        try:
            st.session_state["pia_claim"] = _claim_from_row(row)
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
            st.session_state["pia_claim"] = claim_core.ClaimDraft(
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

if "pia_claim" in st.session_state and stage < STAGE_DESCRIBE_EVIDENCE:
    if st.button("Continue to describe evidence", type="primary"):
        _advance_to(STAGE_DESCRIBE_EVIDENCE)

# ---------------------------------------------------------------------
# 2. Describe evidence (evidence)
# ---------------------------------------------------------------------

if stage >= STAGE_DESCRIBE_EVIDENCE and "pia_claim" in st.session_state:
    claim = st.session_state["pia_claim"]

    section_header(
        "2. Describe Evidence",
        f"Evidence - what backs {claim.claim_id}. Each row is one independent "
        "look at the same finding; more of them, from different methods, "
        "makes the finding harder to overturn (checked in Step 3).",
    )

    sample_evidence = pd.read_csv(SAMPLE_DIR / "evidence_items.csv")
    matching_sample = sample_evidence[sample_evidence["claim_id"] == claim.claim_id]

    st.caption("Sample data is used until you upload your own.")
    uploaded_evidence = st.file_uploader(
        "Upload a custom evidence CSV (optional, with columns source, "
        "finding_text, indicator_id, sample_size, has_comparison_group, "
        "collection_method, time_lag_days)",
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

    display_evidence = evidence_frame.copy()
    if not show_ids and "indicator_id" in display_evidence.columns:
        display_evidence["indicator_id"] = display_evidence["indicator_id"].map(
            lambda v: _friendly(v, INDICATOR_NAMES, show_ids)
        )
        display_evidence = display_evidence.rename(columns={"indicator_id": "indicator"})
    st.dataframe(display_evidence, width="stretch", hide_index=True)

    if st.button("Summarize evidence", type="primary"):
        try:
            items = tuple(_item_from_row(r) for _, r in evidence_frame.iterrows())
            bundle = evidence_core.summarize_evidence(items, claim_id=claim.claim_id)
            st.session_state["pia_bundle"] = bundle
            st.session_state["pia_evidence_frame"] = evidence_frame
            st.session_state["pia_evidence_filename"] = evidence_filename
            for key in SESSION_KEYS[5:]:
                st.session_state.pop(key, None)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    if "pia_bundle" in st.session_state:
        bundle = st.session_state["pia_bundle"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Evidence items received", bundle.n_input_items)
        c2.metric("Usable", bundle.n_usable_items)
        c3.metric("Excluded", bundle.n_excluded_items)
        st.caption(
            "A usable row names what changed (a finding) and which "
            "indicator it supports, so it can be traced back to the claim "
            "and checked in Step 3. An excluded row is missing one of "
            "those - for example, a note with no linked indicator - so it "
            "can't be tied to anything and doesn't count as evidence for "
            "this claim."
        )
        if bundle.n_excluded_items:
            caveat(f"Excluded because {bundle.exclusion_reason}.")

        if stage < STAGE_VALIDATE:
            if st.button("Continue to validate", type="primary"):
                _advance_to(STAGE_VALIDATE)

# ---------------------------------------------------------------------
# 3. Validate (evidence -> interpretation)
# ---------------------------------------------------------------------

if stage >= STAGE_VALIDATE and "pia_bundle" in st.session_state:
    bundle = st.session_state["pia_bundle"]

    section_header(
        "3. Validate",
        "Set the bar you'd want to see before trusting this evidence "
        "enough to report on it.",
    )

    st.caption(
        "Each check below compares what the evidence reports against the "
        "number you set. Comparison group is a yes-or-no check; sample "
        "size, sources, and recency are each compared against the minimum "
        "(or maximum) you choose. What counts as 'enough' varies by field "
        "and program size - a small pilot may reasonably use lower "
        "minimums than a national program - so set these to fit the "
        "program you're assessing, not the defaults below."
    )

    t1, t2, t3 = st.columns(3)
    min_sample_size = t1.number_input(
        "Minimum sample size",
        min_value=1,
        value=validate_core.MIN_SAMPLE_SIZE_DEFAULT,
        help="How many participants must be represented before you'd trust an average from this evidence.",
    )
    min_corroboration = t2.number_input(
        "Minimum independent sources",
        min_value=1,
        value=validate_core.MIN_CORROBORATION_DEFAULT,
        help="How many separate reports of the same finding you'd want before treating it as more than a one-off.",
    )
    max_time_lag_days = t3.number_input(
        "Maximum time lag (days)",
        min_value=1,
        value=validate_core.MAX_TIME_LAG_DAYS_DEFAULT,
        help="How old evidence can be before you'd want it re-checked against current conditions.",
    )
    st.caption(
        "These starting numbers are OpenMeasure defaults, not fixed rules. "
        "The sample-size default is informed by the general discussion of "
        "statistical power in Gertler, Martinez, Premand, Rawlings, & "
        "Vermeersch (2016), *Impact Evaluation in Practice*, World Bank. "
        "The independent-sources default is informed by the triangulation "
        "principle in Patton (1999), *Health Services Research*, 34(5 Pt "
        "2), 1189-1208. Recency has no cited convention, so set it to what "
        "your field or funder treats as current. Neither source specifies "
        "these exact numbers; adjust them to fit your context."
    )

    if st.button("Run validation", type="primary"):
        try:
            validation = validate_core.validate_evidence(
                bundle,
                min_sample_size=int(min_sample_size),
                min_corroboration=int(min_corroboration),
                max_time_lag_days=int(max_time_lag_days),
            )
            st.session_state["pia_validation"] = validation
            for key in SESSION_KEYS[6:]:
                st.session_state.pop(key, None)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    if "pia_validation" in st.session_state:
        validation = st.session_state["pia_validation"]

        st.caption("Evidence profile: each check below, what it verifies, and how this evidence measures up.")
        profile_cols = st.columns(len(validation.checks))
        for col, check in zip(profile_cols, validation.checks):
            with col:
                label = CRITERION_LABELS[check.criterion]
                if check.passed:
                    st.badge(label, icon=":material/check_circle:", color="green")
                elif check.severity == validate_core.SEVERITY_BLOCKING:
                    st.badge(label, icon=":material/cancel:", color="red")
                else:
                    st.badge(label, icon=":material/warning:", color="orange")
                st.caption(CRITERION_WHY[check.criterion])
                st.caption(check.detail)

        if stage < STAGE_DETERMINE_SUPPORTED_CLAIM:
            if st.button("Continue to determine supported claim", type="primary"):
                _advance_to(STAGE_DETERMINE_SUPPORTED_CLAIM)

# ---------------------------------------------------------------------
# 4. Determine supported claim (interpretation)
# ---------------------------------------------------------------------

if stage >= STAGE_DETERMINE_SUPPORTED_CLAIM and "pia_validation" in st.session_state:
    claim = st.session_state["pia_claim"]
    bundle = st.session_state["pia_bundle"]
    validation = st.session_state["pia_validation"]

    section_header(
        "4. Determine Supported Claim",
        "Interpretation - evidentiary rigor, graded against Nesta's "
        "Standards of Evidence.",
    )

    supported = strength_core.determine_supported_claim(claim, bundle, validation)
    st.session_state["pia_supported"] = supported

    render_verdict(classify(supported.nesta_level.level, NESTA_LEVEL_BANDS))

    st.caption("Evidence-strength ladder")
    ladder_cols = st.columns(4)
    for level_number, level_col in zip(range(1, 5), ladder_cols):
        with level_col:
            level_def = strength_core.NESTA_LEVELS[level_number - 1]
            if level_number < supported.nesta_level.level:
                st.badge(f"Level {level_number}", icon=":material/check:", color="green")
            elif level_number == supported.nesta_level.level:
                st.badge(f"Level {level_number}", icon=":material/radio_button_checked:", color="blue")
            else:
                st.badge(f"Level {level_number}", icon=":material/radio_button_unchecked:", color="gray")
            st.caption(level_def.description)

    if supported.next_level_hint:
        st.info(supported.next_level_hint)

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
# 5. Examine limitations (interpretation)
# ---------------------------------------------------------------------

if stage >= STAGE_EXAMINE_LIMITATIONS and "pia_supported" in st.session_state:
    bundle = st.session_state["pia_bundle"]
    validation = st.session_state["pia_validation"]

    section_header("5. Examine Limitations", "Interpretation - what the evidence does not establish.")

    limitations = limitations_core.examine_limitations(bundle, validation)
    st.session_state["pia_limitations"] = limitations

    if limitations.n_flags == 0:
        st.success("No limitations flagged against the configured thresholds.")
    else:
        for flag in limitations.flags:
            flagged_item_note(f"{flag.category} ({flag.severity})", flag.message)

    if stage < STAGE_PORTFOLIO_CONTEXT:
        if st.button("Continue to portfolio context", type="primary"):
            _advance_to(STAGE_PORTFOLIO_CONTEXT)

# ---------------------------------------------------------------------
# 6. Portfolio context (interpretation, across grantees)
# ---------------------------------------------------------------------

if stage >= STAGE_PORTFOLIO_CONTEXT and "pia_limitations" in st.session_state:
    claim = st.session_state["pia_claim"]

    section_header(
        "6. Portfolio Context",
        "Interpretation - how this compares to the rest of a portfolio. "
        "Optional, and skipped if no matching portfolio row is found.",
    )

    st.caption("Sample data is used until you upload your own.")
    uploaded_portfolio = st.file_uploader(
        "Upload a custom portfolio CSV (optional; columns include grantee_id, "
        "indicator_id, result_value, unit)",
        type="csv",
        key="pia_portfolio_uploader",
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

        grantee_label = _friendly(claim.grantee_id, GRANTEE_NAMES, show_ids)
        indicator_label = _friendly(indicator_id, INDICATOR_NAMES, show_ids)
        st.caption(
            f"How {grantee_label}'s reported {indicator_label} compares to "
            "other grantees reporting the same indicator in this portfolio."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "This grantee's reported result",
            indicator_summary.this_value,
            help=f"What {grantee_label} reported for this indicator.",
        )
        c2.metric(
            "Portfolio median",
            round(indicator_summary.median_value, 1),
            help="The middle value across every grantee reporting this same indicator.",
        )
        c3.metric(
            "Grantees reporting",
            indicator_summary.n_grantees_reporting,
            help="How many grantees, including this one, reported this indicator.",
        )

        message = PORTFOLIO_POSITION_MESSAGES[indicator_summary.position]
        if indicator_summary.position == portfolio_core.POSITION_WITHIN_RANGE:
            st.info(message)
        else:
            st.warning(message)

        for flag in comparability_flags:
            flagged_item_note(f"Comparability ({flag.severity})", flag.message)

        st.caption("Portfolio evidence map")
        st.caption(
            "A large reported result and strong evidence are not the same "
            "thing. This places each grantee by what was reported and by "
            "how well-supported that report is."
        )

        sample_claims_for_map = pd.read_csv(SAMPLE_DIR / "claims.csv")
        sample_evidence_for_map = pd.read_csv(SAMPLE_DIR / "evidence_items.csv")
        points = _portfolio_evidence_map_points(
            sample_claims_for_map, sample_evidence_for_map, portfolio_df, claim.claim_id
        )

        if points.empty:
            st.caption("Not enough sample data to place grantees on the map.")
        else:
            common_unit = points["unit"].mode().iloc[0]
            comparable = points[points["unit"] == common_unit].copy()
            excluded = points[points["unit"] != common_unit]

            comparable["grantee"] = comparable["grantee_id"].map(
                lambda v: _friendly(v, GRANTEE_NAMES, show_ids)
            )
            comparable["current_label"] = comparable["is_current"].map(
                {True: "This claim", False: "Other reported results"}
            )

            # Grantee identity is unbounded (any number of comparable
            # grantees), so it goes in the tooltip rather than color/shape,
            # which are reserved for the one bounded distinction that
            # matters here: this claim versus every other reported result.
            # Color is validated CVD-safe (dataviz skill palette, first two
            # categorical slots) and shown with a real legend, same pattern
            # used for GAIA's frontier chart.
            map_spec = {
                "data": {"values": comparable.to_dict("records")},
                "mark": {"type": "point", "filled": True, "size": 200},
                "encoding": {
                    "x": {
                        "field": "value",
                        "type": "quantitative",
                        "title": f"Reported result ({common_unit})",
                    },
                    "y": {
                        "field": "level",
                        "type": "quantitative",
                        "title": "Evidence strength (Nesta level)",
                        "scale": {"domain": [0.5, 4.5]},
                        "axis": {"values": [1, 2, 3, 4], "tickMinStep": 1},
                    },
                    "shape": {
                        "field": "is_current",
                        "type": "nominal",
                        "scale": {
                            "domain": [False, True],
                            "range": ["circle", "diamond"],
                        },
                        "legend": None,
                    },
                    "color": {
                        "field": "current_label",
                        "type": "nominal",
                        "scale": {
                            "domain": ["Other reported results", "This claim"],
                            "range": [CATEGORICAL_1, CATEGORICAL_2],
                        },
                        "legend": {"title": None, "orient": "bottom"},
                    },
                    "tooltip": [
                        {"field": "grantee", "type": "nominal", "title": "Grantee"},
                        {
                            "field": "value",
                            "type": "quantitative",
                            "title": f"Result ({common_unit})",
                        },
                        {"field": "level", "type": "quantitative", "title": "Nesta level"},
                        {"field": "current_label", "type": "nominal", "title": "Status"},
                    ],
                },
                "width": "container",
                "height": 260,
                "config": _VEGA_CHART_CONFIG,
            }
            st.vega_lite_chart(map_spec, width="stretch")
            st.caption(
                "Diamond marks the claim you are currently reviewing. "
                "Hover a point for the grantee and exact values."
            )

            if not excluded.empty:
                with st.container(border=True):
                    st.caption(
                        "Not shown above - measured in a different unit, not "
                        "directly comparable to the rest:"
                    )
                    for _, r in excluded.iterrows():
                        grantee_label_excl = _friendly(r["grantee_id"], GRANTEE_NAMES, show_ids)
                        st.write(f"- {grantee_label_excl}: {r['value']} {r['unit']}")

    st.session_state["pia_portfolio_context"] = portfolio_context

    if stage < STAGE_EVIDENCE_RECORD:
        if st.button("Continue to evidence record", type="primary"):
            _advance_to(STAGE_EVIDENCE_RECORD)

# ---------------------------------------------------------------------
# 7. Evidence record (claim)
# ---------------------------------------------------------------------

if stage >= STAGE_EVIDENCE_RECORD and "pia_portfolio_context" in st.session_state:
    claim = st.session_state["pia_claim"]
    bundle = st.session_state["pia_bundle"]
    validation = st.session_state["pia_validation"]
    supported = st.session_state["pia_supported"]
    limitations = st.session_state["pia_limitations"]
    portfolio_context = st.session_state["pia_portfolio_context"]

    section_header(
        "7. Evidence Record",
        "Claim - defensible language for reporting, with full detail on request.",
    )

    record = record_core.EvidenceRecord(
        claim=claim,
        evidence=bundle,
        validation=validation,
        supported_claim=supported,
        limitations=limitations,
        portfolio_context=portfolio_context,
    )

    st.caption(
        "Use the summary below as a starting point; edit before it goes into "
        "a report. Open 'Full detail' only if you need to justify the "
        "methodology, e.g. to a board or funder."
    )

    st.success(record_core.build_leadership_summary(record))

    caveat(
        "This is a model of evidence strength, not a determination of "
        "whether the claim is true or a recommendation to publish it."
    )

    with st.expander("Full detail"):
        st.markdown("**Claim**")
        _render_claim_detail(record.claim)
        st.markdown("**Evidence**")
        _render_evidence_detail(record.evidence)
        st.markdown("**Validation**")
        _render_validation_detail(record.validation)
        st.markdown("**Supported claim**")
        _render_supported_detail(record.supported_claim)
        st.markdown("**Limitations**")
        _render_limitations_detail(record.limitations)
        if record.portfolio_context is not None:
            st.markdown("**Portfolio context**")
            _render_portfolio_context_detail(record.portfolio_context, show_ids)
