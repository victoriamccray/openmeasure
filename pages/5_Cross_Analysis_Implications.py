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

from modules.validation_chain.core.retention import summarize_retention
from shared.handoff import (
    KIND_CELLS_EMPTY,
    KIND_OBSERVATIONS_ABSENT,
    KIND_ROWS_DROPPED,
    HandoffStore,
    group_by_dataset,
)
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


KIND_LABELS = {
    KIND_ROWS_DROPPED: "Rows dropped",
    KIND_CELLS_EMPTY: "Empty cells in retained rows",
    KIND_OBSERVATIONS_ABSENT: "Observations that never arrived",
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
        f"Retention: {dataset.fingerprint.filename}",
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

    if plottable:
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

section_header("Recorded Results")

st.caption(
    "Records last for this browser session only. Re-running a module "
    "replaces its previous record."
)

if st.button("Clear recorded results"):
    store.clear()
    st.rerun()


show_case_studies("data_validation")
