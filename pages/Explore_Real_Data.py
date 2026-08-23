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

Each dataset below names a workflow it fits and one open-ended question
worth asking of it. Deciding how to approach the data, and what its
answer would actually support, is the exercise.
"""
)

st.divider()

_PAGE_BY_WORKFLOW = {item.workflow: item.page for item in WORKFLOWS}

for dataset in DATASETS:
    with st.container(border=True):
        st.markdown(f"**{dataset.name}**")
        st.badge(dataset.domain)
        st.write(dataset.description)

        st.caption("Try with:")
        for workflow_name in dataset.try_with:
            st.page_link(
                _PAGE_BY_WORKFLOW[workflow_name],
                label=workflow_name,
                icon=":material/arrow_forward:",
            )

        st.caption("Explore:")
        st.write(dataset.explore_question)

        st.caption(f"Access: {dataset.access}")

        for source in dataset.sources:
            st.markdown(f"[{source.label}]({source.url})")

        if dataset.citation:
            with st.expander("Citation"):
                st.caption(dataset.citation)

st.divider()

st.caption(
    "Access terms and links were checked against the official source at "
    "the time this page was written and can change; confirm current terms "
    "before relying on them."
)
