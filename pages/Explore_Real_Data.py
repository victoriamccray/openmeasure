"""
Explore Real Data - discovery page pointing to real research datasets.

This page is a pointer, not a workflow. It records nothing to
shared/handoff.py and carries no module_key, so it cannot appear on the
overview's progress cards or the stage strip, and it is deliberately not a
numbered page: shared/tests/test_catalog.py only requires a catalog entry
for numbered pages, so this one needs none.

Each card names an open-ended validation question rather than steps to
follow. Columns, procedures, and expected results are left for the user to
find, because the point of this page is to practice validation judgment on
real data, not to reproduce a worked example.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from shared.catalog import WORKFLOWS
from shared.datasets import DATASETS

st.set_page_config(
    page_title="OpenMeasure - Explore Real Data",
    layout="centered",
)

st.title("Explore Real Data")
st.caption(
    "Real research datasets to try against OpenMeasure's existing "
    "workflows."
)

st.markdown(
    """
These measurement datasets reflect real-world complexities rather than
standardized benchmarks. Consequently, you should anticipate missing
values, condition-dependent traits, and outcomes that vary based on
data filtering and splitting. Rather than addressing questions with a
single definitive answer, they are best suited for validation
inquiries, such as evaluating whether a model or measurement
demonstrates consistency across different methods, conditions, or
groups.

Each dataset below names one open-ended question worth asking of it.
Deciding how to approach the data, and what its answer would actually
support, is the exercise. They are grouped by the workflow they suit; a
dataset that suits two appears under both.
"""
)

st.divider()

_PAGE_BY_WORKFLOW = {item.workflow: item.page for item in WORKFLOWS}


def render_dataset(dataset, shown_under: str) -> None:
    """
    One dataset card.

    Access and delivery are shown together because they answer different
    questions and are easy to conflate: access is whether a reader may
    obtain the data, delivery is how it would reach a workflow. A dataset
    can be openly downloadable and still be one OpenMeasure will not keep
    a copy of.
    """
    with st.container(border=True):
        st.markdown(f"**{dataset.name}**")
        st.badge(dataset.domain)
        st.write(dataset.description)

        # Grouping by workflow means a card no longer lists every
        # workflow it suits, only the one it is filed under. Naming the
        # others here keeps that from being something a reader can only
        # learn by scrolling to a different section.
        elsewhere = [w for w in dataset.try_with if w != shown_under]
        if elsewhere:
            st.caption(f"Also suits: {', '.join(elsewhere)}")

        st.caption("Explore:")
        st.write(dataset.explore_question)

        st.caption(
            f"Access: {dataset.access}  ·  How you get it: {dataset.delivery}"
        )

        for source in dataset.sources:
            st.markdown(f"[{source.label}]({source.url})")

        if dataset.citation:
            with st.expander("Citation"):
                st.caption(dataset.citation)


# Grouped by the workflow each dataset suits, rather than listed flat.
# The page's whole framing is "try this against a workflow", so that is
# what a reader is scanning for, and a flat list buried the newest
# entries at the bottom once the catalog grew past a handful. A dataset
# suiting two workflows appears under both, because it genuinely does.
for workflow in WORKFLOWS:
    matching = [d for d in DATASETS if workflow.workflow in d.try_with]

    if not matching:
        continue

    st.markdown(f"#### For {workflow.workflow}")
    st.page_link(
        _PAGE_BY_WORKFLOW[workflow.workflow],
        label=f"Open {workflow.workflow}",
        icon=":material/arrow_forward:",
    )

    for dataset in matching:
        render_dataset(dataset, workflow.workflow)

st.divider()

st.caption(
    "Access terms and links were checked against the official source at "
    "the time this page was written and can change; confirm current terms "
    "before relying on them."
)
