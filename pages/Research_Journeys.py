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
    "Explore research questions through data, interactive experiences, "
    "and evidence."
)

st.markdown(
    """
Follow information from measurement to interpretation, investigate
methodological decisions, and discover how those choices shape what we
can conclude.
"""
)

st.divider()

# Become the Signal is a static 3D experience hosted on GitHub Pages, so
# it has no page in this app's navigation. It is shown beside its
# Streamlit companion, Multimodal Signal Convergence, as an external link.
BECOME_THE_SIGNAL_URL = "https://victoriamccray.github.io/openmeasure/become-the-signal/"
BECOME_THE_SIGNAL_AFTER = "pages/Multimodal_Signal_Convergence.py"


def render_become_the_signal() -> None:
    with st.container(border=True):
        st.markdown("**Become the Signal**")
        st.caption("Privacy & Neurosecurity, interactive 3D experience")
        st.write(
            "Travel through a neurotechnology system as one EEG recording "
            "from a public dataset, choose safeguards, and see what each "
            "protects and leaves exposed."
        )
        st.link_button(
            "Open Become the Signal",
            BECOME_THE_SIGNAL_URL,
            icon=":material/open_in_new:",
        )


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

        if journey.page == BECOME_THE_SIGNAL_AFTER:
            render_become_the_signal()

st.divider()

st.page_link(
    "pages/Method_Selection.py",
    label="None of these fit: answer one question and get routed to a workflow instead",
    icon=":material/alt_route:",
)
