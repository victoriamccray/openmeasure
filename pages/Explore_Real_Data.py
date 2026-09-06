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
from shared.dataset_loaders import can_load
from shared.datasets import DATASETS
from shared.portraits import catalog_portrait_svg

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

Each dataset below leads with one open-ended question worth asking of
it. Deciding how to approach the data, and what its answer would
actually support, is the exercise. Browse by the workflow a dataset
suits, by its field, by how it was measured, or by how it is arranged.
"""
)

st.divider()

_PAGE_BY_WORKFLOW = {item.workflow: item.page for item in WORKFLOWS}

# What a reader is browsing by. Workflow was the only lens, which assumes
# they already know which analysis they want; someone looking for what a
# wearable study or a repeated-measures file actually looks like was left
# scrolling. Each lens reads a different field of the same entries, so no
# dataset can appear under one lens and vanish under another.
LENS_WORKFLOW = "Workflow"
LENS_DOMAIN = "Domain"
LENS_MODALITY = "How it was measured"
LENS_STRUCTURE = "How it is arranged"

_LENS_FIELDS = {
    LENS_DOMAIN: lambda dataset: (dataset.domain,),
    LENS_MODALITY: lambda dataset: (dataset.modality,),
    LENS_STRUCTURE: lambda dataset: (dataset.structure,),
    LENS_WORKFLOW: lambda dataset: dataset.try_with,
}


def render_dataset(dataset, shown_under: str) -> None:
    """
    One dataset, led by the question it is worth asking.

    The question is what a reader is actually choosing between, and it
    used to sit below the description under a small "Explore:" caption.
    Everything that describes rather than invites, the prose, the access
    terms, the links and the citation, moves into Inspect. None of it is
    removed: the hierarchy changes, the content does not.
    """
    with st.container(border=True):
        st.markdown(f"**{dataset.name}**")

        strip = catalog_portrait_svg(dataset)
        if strip:
            st.markdown(strip, unsafe_allow_html=True)

        st.write(dataset.explore_question)

        # The facets, including whichever one this dataset is filed
        # under. Shown whole rather than minus the current lens, so a
        # card carries the same description wherever it is read.
        chips = st.columns(4)
        for column, value in zip(
            chips,
            (
                dataset.domain,
                dataset.modality,
                dataset.structure,
                ", ".join(dataset.try_with),
            ),
        ):
            with column:
                st.badge(value)

        with st.expander("Inspect dataset"):
            st.write(dataset.description)

            # Access and delivery answer different questions and are easy
            # to conflate: access is whether a reader may obtain the
            # data, delivery is how it would reach a workflow. A dataset
            # can be openly downloadable and still be one OpenMeasure
            # will not keep a copy of.
            st.caption(
                f"Access: {dataset.access}  ·  "
                f"How you get it: {dataset.delivery}"
            )

            if can_load(dataset.id):
                st.caption(
                    "OpenMeasure can open this one directly. The rest are "
                    "catalogued for you to obtain yourself."
                )

            for source in dataset.sources:
                st.markdown(f"[{source.label}]({source.url})")

            if dataset.citation:
                st.caption(dataset.citation)

        for workflow in dataset.try_with:
            st.page_link(
                _PAGE_BY_WORKFLOW[workflow],
                label=f"Open {workflow}",
                icon=":material/arrow_forward:",
            )


lens = st.radio(
    "Browse by",
    options=(LENS_WORKFLOW, LENS_DOMAIN, LENS_MODALITY, LENS_STRUCTURE),
    horizontal=True,
)

# Groups come from the datasets themselves under every lens except
# workflow, which uses the catalog's own order so the sections match the
# navigation. A dataset suiting two workflows appears under both, because
# it genuinely does.
if lens == LENS_WORKFLOW:
    groups = [item.workflow for item in WORKFLOWS]
else:
    groups = sorted(
        {value for dataset in DATASETS for value in _LENS_FIELDS[lens](dataset)}
    )

for group in groups:
    matching = [d for d in DATASETS if group in _LENS_FIELDS[lens](d)]

    if not matching:
        continue

    st.markdown(f"#### {group}")

    for dataset in matching:
        render_dataset(dataset, group)

st.divider()

st.caption(
    "Access terms and links were checked against the official source at "
    "the time this page was written and can change; confirm current terms "
    "before relying on them."
)
