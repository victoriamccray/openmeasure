"""
Resources - existing toolkits and dataset directories outside OpenMeasure.

Not a workflow: no catalog entry, no module_key, no lifecycle stage - same
"reference page declared directly in Home.py" pattern as
pages/Explore_Real_Data.py, pages/Method_Selection.py, and
pages/Privacy_and_Data_Access.py.

Distinct from Explore Real Data: that page's entries each pair with one
specific OpenMeasure workflow (shared/datasets.py's try_with requires it).
This page is for tools, methods guides, and dataset directories that do
not make that specific a pairing, or that are not a dataset at all.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from shared.resources import resources_by_kind

st.set_page_config(
    page_title="OpenMeasure - Resources",
    page_icon=":material/collections_bookmark:",
    layout="centered",
)

st.title("Resources")
st.caption(
    "Existing toolkits, methods guides, and dataset directories outside "
    "OpenMeasure that are worth knowing about, grouped by what they are."
)

st.markdown(
    """
The following are curated resources for existing applied research for
social impact and open science networks and platforms.
"""
)

st.divider()

for kind, resources in resources_by_kind().items():
    st.markdown(f"#### {kind}")

    for resource in resources:
        with st.container(border=True):
            st.markdown(f"**{resource.name}**")
            st.write(resource.description)

            if resource.url:
                st.markdown(f"[{resource.url}]({resource.url})")
            else:
                st.caption("No confirmed link yet.")

st.divider()

st.caption(
    "Know a resource that belongs here? Open an issue or a pull request "
    "(see CONTRIBUTING.md) with the tool, a one- or two-sentence "
    "description, and a link."
)
