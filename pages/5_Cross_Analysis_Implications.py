"""
OpenMeasure - Cross-Analysis Implications.

Brings together what each module's analysis actually used, so the amount of
data behind each result is visible in one place. Core logic lives in
modules/validation_chain/core; this file handles presentation only.

Deliberately produces no overall figure. Analyses of different uploads
describe different data, and the kinds of exclusion within one analysis are
not additive, so there is nothing honest to combine.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from the repository root regardless of where Streamlit
# is launched.
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from modules.cross_analysis.core import compare
from modules.validation_chain.core.retention import summarize_retention
from shared.handoff import (
    KIND_CELLS_EMPTY,
    KIND_OBSERVATIONS_ABSENT,
    KIND_ROWS_DROPPED,
    HandoffStore,
    group_by_dataset,
)
from shared import visuals
from shared.report import (
    caveat,
    implications,
    inspect_note,
    interpretation_note,
    render_lifecycle_tracker,
    section_header,
    show_case_studies,
)


st.set_page_config(
    page_title="OpenMeasure · Cross-Analysis Implications",
    layout="centered",
)

st.title("Cross-Analysis Implications")
st.subheader("Across modules")
st.caption(
    "How much of your data each analysis actually used, and what that means "
    "for what the results describe."
)

st.divider()

render_lifecycle_tracker(current_workflow="Cross-Analysis Implications")


# The convergence diagram. Two findings and what can be said about the
# pair, which is the relationship this page is named for and had never
# drawn.
#
# Solid when the two can be set beside each other, dashed when they
# cannot, the same rule used elsewhere for established and not. A pair
# that is not comparable still gets drawn: it is a fact about the two
# analyses, and hiding it would leave a reader unable to tell a
# comparison that was attempted and could not be made from one that was
# never attempted.
_PAIR_W, _PAIR_H = 640.0, 160.0
_PAIR_BOX_W, _PAIR_BOX_H = 292.0, 54.0
_PAIR_TOP_Y, _PAIR_BOTTOM_Y = 16.0, 90.0
_PAIR_OUT_X, _PAIR_OUT_W = 412.0, 228.0

# SVG text does not wrap, so a label longer than this is shortened here
# and read in full in the table underneath.
_PAIR_MAX_LABEL = 38


def _pair_text(value: str) -> str:
    """A label short enough to sit inside a box."""
    if len(value) <= _PAIR_MAX_LABEL:
        return value

    return value[: _PAIR_MAX_LABEL - 1] + "…"


def _pair_box(y: float, module: str, reading: str, *, solid: bool) -> str:
    """One analysis and how it read, as one side of a comparison."""
    stroke = visuals.ACCENT if solid else visuals.INK_MUTED
    dash = "" if solid else f' stroke-dasharray="{visuals.DASH_UNESTABLISHED}"'

    return (
        f'<rect x="0" y="{y:.0f}" width="{_PAIR_BOX_W:.0f}" '
        f'height="{_PAIR_BOX_H:.0f}" rx="5" fill="none" stroke="{stroke}" '
        f'stroke-width="1.5"{dash}/>'
        f'<text x="14" y="{y + 22:.0f}" font-size="13" fill="{stroke}">'
        f"{_pair_text(module)}</text>"
        f'<text x="14" y="{y + 42:.0f}" font-size="13" '
        f'fill="{visuals.INK_MUTED}">{_pair_text(reading)}</text>'
    )


def _convergence_svg(pair) -> str:
    """
    Two findings, and what their pairing does and does not support.

    Nothing here scores the pair. The diagram shows two readings meeting
    at a junction and names what the junction is; whether that is worth
    anything depends on knowing what the two analyses are, which is the
    reader's to bring.
    """
    solid = pair.comparison != compare.NOT_COMPARABLE
    stroke = visuals.ACCENT if solid else visuals.INK_MUTED
    dash = "" if solid else f' stroke-dasharray="{visuals.DASH_UNESTABLISHED}"'

    top_mid = _PAIR_TOP_Y + _PAIR_BOX_H / 2
    bottom_mid = _PAIR_BOTTOM_Y + _PAIR_BOX_H / 2
    junction_y = (top_mid + bottom_mid) / 2

    joins = (
        f'<path d="M {_PAIR_BOX_W + 6:.0f} {top_mid:.0f} '
        f'H 350 V {junction_y:.0f}" fill="none" stroke="{stroke}" '
        f'stroke-width="1"{dash}/>'
        f'<path d="M {_PAIR_BOX_W + 6:.0f} {bottom_mid:.0f} '
        f'H 350 V {junction_y:.0f}" fill="none" stroke="{stroke}" '
        f'stroke-width="1"{dash}/>'
    )

    return visuals.figure(
        _pair_box(_PAIR_TOP_Y, pair.module_a, pair.reading_a, solid=solid)
        + _pair_box(_PAIR_BOTTOM_Y, pair.module_b, pair.reading_b, solid=solid)
        + joins
        + visuals.arrow(
            350, junction_y, _PAIR_OUT_X - 8, junction_y, established=solid
        )
        + f'<rect x="{_PAIR_OUT_X:.0f}" y="{junction_y - 27:.0f}" '
        f'width="{_PAIR_OUT_W:.0f}" height="{_PAIR_BOX_H:.0f}" rx="5" '
        f'fill="none" stroke="{stroke}" stroke-width="1.5"{dash}/>'
        + f'<text x="{_PAIR_OUT_X + 16:.0f}" y="{junction_y + 5:.0f}" '
        f'font-size="14" fill="{stroke}">{pair.comparison}</text>',
        width=_PAIR_W,
        height=_PAIR_H,
        label=(
            f"{pair.module_a} read as {pair.reading_a}. {pair.module_b} read "
            f"as {pair.reading_b}. Together: {pair.comparison.lower()}. "
            f"{pair.explanation}"
        ),
    )


KIND_LABELS = {
    KIND_ROWS_DROPPED: "Rows dropped",
    KIND_CELLS_EMPTY: "Empty cells in retained rows",
    KIND_OBSERVATIONS_ABSENT: "Observations that never arrived",
}

# Every primary_statistics key any module currently records (see each
# module's record_*() function: pages/1_Reliability.py, 3_Fairness.py,
# 4_Time_Series_QA.py, 6_Evidence_Review.py). A key with no label here
# falls back to itself, so a future module's new statistic still renders
# rather than raising, just without a human-readable name until this
# dict is extended.
STATISTIC_LABELS = {
    "cronbach_alpha": "Cronbach's alpha",
    "disparate_impact": "Disparate impact",
    "statistical_parity_difference": "Statistical parity difference",
    "equal_opportunity_difference": "Equal opportunity difference",
    "predictive_equality_difference": "Predictive equality difference",
    "calibration_within_groups_difference": "Calibration within groups difference",
    "value_completeness_ratio": "Value completeness ratio",
    "n_found": "Records found",
    "n_included": "Records included",
}


with st.expander("What this page does"):
    st.markdown(
        """
Each module reports how much data it received and how much it kept. Run an
analysis on any module page and it is recorded here, so you can see the
whole picture at once.

Two analyses of the same file often keep **different amounts**. Reliability
drops a participant missing any scale item; a group comparison drops anyone
missing the group or the outcome. Both results are valid, but each describes
only the observations it retained, and those subsets are not the same.

### What this page will not do

- It produces **no overall score** and no combined exclusion rate. Rows
  dropped, empty cells inside retained rows, and observations that never
  arrived are different things with no shared denominator, so adding them
  would invent a number that means nothing.
- It applies **no threshold**. There is no line above which exclusion
  becomes unacceptable, because how much is too much depends on what you
  intend to compute.
- It does **not** tell you why data is missing, which usually matters more
  than how much. Observations are often missing precisely because
  circumstances were unusual.
"""
    )


store = HandoffStore(st.session_state)
entries = store.entries()

if not entries:
    st.info(
        "No analyses recorded yet. Run an analysis on any module page and it "
        "will appear here."
    )

    st.markdown(
        """
Try the Reliability page and the Impact Evaluation page on the same file, and
this page will show how many participants each one kept.
"""
    )

    with st.expander("Examples"):
        show_case_studies("data_validation")

    st.stop()


grouped = group_by_dataset(entries)

accounts_by_dataset = {
    group[0].fingerprint: tuple(entry.exclusion for entry in group)
    for group in grouped.values()
}

summary = summarize_retention(accounts_by_dataset)


# ---------------------------------------------------------------------
# Provenance, before any figure
# ---------------------------------------------------------------------

section_header(
    "What Was Recorded",
    "Where every number below came from",
)

st.dataframe(
    pd.DataFrame(
        [
            {
                "Module": entry.module,
                "Analysis": entry.exclusion.analysis_label,
                "File": entry.fingerprint.filename,
                "Data": entry.fingerprint.short_digest,
                "Uploaded shape": (
                    f"{entry.fingerprint.n_rows} x "
                    f"{entry.fingerprint.n_columns}"
                ),
                "Columns used": ", ".join(entry.exclusion.columns_considered),
                "Recorded": entry.recorded_at,
            }
            for entry in entries
        ]
    ),
    width="stretch",
    hide_index=True,
)

st.caption(
    f"{summary.n_analyses} analysis(es) across {summary.n_datasets} "
    "dataset(s). The Data column is a short digest of the uploaded file, so "
    "analyses of the same upload share a value."
)

if summary.n_datasets > 1:
    st.warning(
        "These analyses come from more than one upload. Results are grouped "
        "by dataset below, because row counts from different files describe "
        "different data and cannot be compared directly."
    )


# ---------------------------------------------------------------------
# Retention per dataset
# ---------------------------------------------------------------------

for dataset in summary.datasets:
    section_header(
        f"1. Data Used: {dataset.fingerprint.filename}",
        f"{dataset.fingerprint.n_rows} rows uploaded, "
        f"digest {dataset.fingerprint.short_digest}",
    )

    # Row-expanding analyses have no retained-participant count, so they
    # cannot be plotted on a used-versus-excluded axis. They are reported
    # separately below rather than given a fabricated bar.
    plottable = [
        account
        for account in dataset.accounts
        if account.n_retained_rows is not None
    ]

    # A bar chart of one analysis is not a comparison. With a single
    # recorded analysis the table below is the whole record, and a chart
    # of one bar implies a comparison that is not there.
    if len(plottable) > 1:
        st.bar_chart(
            pd.DataFrame(
                [
                    {
                        "Analysis": account.analysis_label,
                        "Used": account.n_retained_rows,
                        "Not used": account.n_excluded_rows,
                    }
                    for account in plottable
                ]
            ).set_index("Analysis")
        )
        inspect_note("Whether Used differs across analyses of the same dataset.")

    # Every cell is rendered as text. Some columns legitimately hold either
    # a count or "not applicable", and a mixed-type column makes Streamlit
    # rewrite the column types to serialize it, which changes what is
    # displayed.
    def as_text(value: int | None) -> str:
        return "not applicable" if value is None else f"{value:,}"

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Analysis": account.analysis_label,
                    "Participants received": f"{account.n_input_rows:,}",
                    "Participants used": as_text(account.n_retained_rows),
                    "Not used": as_text(account.n_excluded_rows),
                    "Expanded observations": (
                        ""
                        if account.n_expanded_observations is None
                        else f"{account.n_expanded_observations:,}"
                    ),
                    "Columns": ", ".join(account.columns_considered),
                }
                for account in dataset.accounts
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    if dataset.smallest_retained_analysis is not None:
        st.caption(
            f"Fewest observations retained: "
            f"**{dataset.smallest_retained_analysis}** at "
            f"{dataset.smallest_retained_n}. This is a count, not a verdict."
        )

    if dataset.incomparable_analyses:
        caveat(
            "Reported as expanded observations rather than retained "
            "participants: "
            + ", ".join(dataset.incomparable_analyses)
            + ". These analyses turn one participant into one observation "
            "per selection, so a participant who selected two categories "
            "appears twice. There is no retained-participant count for "
            "them, so they are shown as an observation count and are left "
            "out of the comparison above and the chart."
        )

    detail_rows = [
        {
            "Analysis": account.analysis_label,
            "What": KIND_LABELS.get(item.kind, item.kind),
            "Count": item.count,
            "Why": item.mechanism,
        }
        for account in dataset.accounts
        for item in account.items
    ]

    if detail_rows:
        st.markdown("**Why data was unavailable**")
        st.dataframe(
            pd.DataFrame(detail_rows), width="stretch", hide_index=True
        )
        st.caption(
            "These rows are reported separately and never added together. A "
            "row that never arrived and a blank cell inside a row that did "
            "arrive are different problems with different causes."
        )

    dataset_entries = grouped.get(dataset.fingerprint.digest, ())
    stat_rows = [
        {
            "Module": entry.module,
            "Analysis": entry.exclusion.analysis_label,
            "Statistic": STATISTIC_LABELS.get(key, key),
            "Value": value,
        }
        for entry in dataset_entries
        for key, value in entry.primary_statistics.items()
    ]

    if stat_rows:
        with st.expander("Other recorded signals on this dataset"):
            st.dataframe(
                pd.DataFrame(stat_rows), width="stretch", hide_index=True
            )
            caveat(
                "Each analysis's own recorded number, shown as-is: not "
                "compared, combined, or flagged against a threshold. A "
                "Reliability alpha and a Fairness disparity measure "
                "different things and have no shared scale, and "
                "co-occurring with exclusion does not establish that one "
                "caused the other."
            )

    # -----------------------------------------------------------------
    # 2, 3 and 4. What each analysis found, what it rests on, and how the
    # findings stand to each other.
    #
    # This page meant one thing by cross-analysis, which was how many
    # rows each module kept from the same file. That is worth knowing and
    # it is not what the name promises. These three views are the rest of
    # it, and they are empty until a module records what it found, which
    # is why a module that has not been wired for that is named rather
    # than silently absent.
    # -----------------------------------------------------------------

    across = compare.read_across(dataset_entries)

    section_header("2. Results", "What each analysis found, side by side")

    finding_rows = [
        {
            "Module": entry.module,
            "Found": finding.label,
            "Value": f"{finding.value:.3f}",
            "Reads as": finding.reading,
            "In the module's words": finding.statement,
        }
        for entry in dataset_entries
        for finding in entry.findings
    ]

    if finding_rows:
        st.dataframe(
            pd.DataFrame(finding_rows), width="stretch", hide_index=True
        )
        caveat(
            "Each reading is the recording module's own, against the "
            "convention that module states. Nothing here re-reads another "
            "module's number."
        )
    else:
        st.info(
            "None of the analyses on this dataset recorded a finding yet."
        )

    if across.modules_without_findings:
        st.caption(
            "Recorded retention but not a finding: "
            + ", ".join(across.modules_without_findings)
            + ". Those modules are not yet wired to record what they "
            "found, so they are absent from the comparison below."
        )

    section_header(
        "3. Assumptions", "What each result depends on, and where those meet"
    )

    if across.assumptions:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Condition": item.name,
                        "Rests on it": ", ".join(item.modules),
                        "Shared": "yes" if item.shared else "no",
                    }
                    for item in across.assumptions
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        inspect_note(
            "The shared rows. A condition two results both rest on is the "
            "one whose failure would move both of them at once."
        )
    else:
        st.info(
            "None of the analyses on this dataset recorded the conditions "
            "its result rests on."
        )

    section_header(
        "4. Convergence",
        "Where findings point the same way, point differently, or cannot "
        "be set beside each other",
    )

    if across.pairs:
        for pair in across.pairs:
            st.markdown(
                f"**{pair.label_a}** and **{pair.label_b}**"
            )
            st.markdown(_convergence_svg(pair), unsafe_allow_html=True)
            st.caption(pair.explanation)
            st.caption(
                compare.RETENTION_MATCHES_NOTE
                if pair.same_retained_count
                else compare.RETENTION_DIFFERS_NOTE
            )

        if not across.has_comparable_pair:
            caveat(
                "No two of these findings report the same kind of quantity, "
                "so none of them can be read as agreeing or disagreeing. "
                "That is a property of the analyses run, not of the data."
            )
    else:
        st.info(
            "A comparison needs findings from two analyses of this dataset."
        )

    caveat(
        "This page does not score agreement. Whether two analyses pointing "
        "the same way strengthens a claim depends on what the analyses are "
        "and what they share, which is yours to judge."
    )


# ---------------------------------------------------------------------
# Implication
# ---------------------------------------------------------------------

section_header("What This Means")

interpretation_note(
    "The retention table(s) above show that analyses of the same file did "
    "not necessarily keep the same rows, or the same number of them."
)

implications(summary.shared_implication)

st.markdown("**Real-world takeaway**")
st.info(summary.real_world_takeaway)

for limitation in summary.limitations:
    caveat(limitation)


# ---------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------

# Named apart from "2. Results" above, which is where findings are
# compared. This is the housekeeping section that clears them.
section_header("Stored Records")

st.caption(
    "Records last for this browser session only. Re-running a module "
    "replaces its previous record."
)

if st.button("Clear recorded results"):
    store.clear()
    st.rerun()


with st.expander("Examples"):
    show_case_studies("data_validation")
