"""
Research Journeys - a landing page for the guided worked examples.

Not a workflow: no catalog entry, no module_key, no lifecycle stage - same
"reference page declared directly in Home.py" pattern as
pages/Explore_Real_Data.py, pages/Method_Selection.py,
pages/Privacy_and_Data_Access.py, and pages/Resources.py.

Individual journey pages are declared in Home.py with visibility="hidden":
each one still resolves for st.page_link and a direct URL (so no existing
link breaks), but none of them get their own sidebar entry any more. This
page, plus pages/Overview.py's "Research Question" card, are what a
reader actually clicks through to pick one, so a domain is chosen here
rather than scattered across the sidebar as one section per journey.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from shared.research_journeys import journeys_by_domain

st.set_page_config(
    page_title="OpenMeasure - Research Journeys",
    page_icon=":material/route:",
    layout="centered",
)

st.title("Research Journeys")
st.caption(
    "Guided worked examples of what a real dataset supports, grouped by "
    "field. Pick a domain, then a journey."
)

st.markdown(
    """
Each journey walks a real (or, where marked, illustrative) dataset
through a validation question none of OpenMeasure's numbered workflows
cover on their own: what a dataset's evidence can and cannot support.
"""
)

st.divider()

for domain, journeys in journeys_by_domain().items():
    st.markdown(f"#### {domain}")

    for journey in journeys:
        with st.container(border=True):
            st.markdown(f"**{journey.title}**")
            st.caption(journey.subdomain)
            st.write(journey.summary)
            st.page_link(
                journey.page,
                label=f"Open {journey.title}",
                icon=":material/arrow_forward:",
            )
