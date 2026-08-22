"""
OpenMeasure entrypoint and navigation.

This file declares the navigation and nothing else. It renders no content,
because anything drawn before nav.run() would appear on top of every page.
The landing page itself is pages/Overview.py.

The sidebar is built from shared/catalog.py, grouped by validation category.
That makes the catalog the single source for the sidebar, the overview
cards, and the drift test in shared/tests/test_catalog.py, so a workflow
cannot exist in one and be missing from another.

Explore Real Data, Method Selection, Resources, and the Research
Journeys are declared separately, below, rather than through the
catalog. None of them is a validation workflow: they have no lifecycle
stage, no validation category, and record nothing to shared/handoff.py,
so folding any of them into workflows_by_category() would imply a status
it can never have.

Research Journeys does not get one sidebar entry per journey, or one
section per domain: both read as repetitive once there are six of them
across three domains. Instead there is a single visible "Research
Journeys" entry, pages/Research_Journeys.py, which is a landing page
listing every journey grouped by domain (shared/research_journeys.py's
journeys_by_domain()) for a reader to pick one from. The six individual
journey pages are still declared below, with visibility="hidden": each
still resolves for st.page_link and its existing direct URL (e.g.
/HealthRing_Worked_Example keeps working), it just is not listed in the
sidebar itself. pages/Research_Journeys.py and pages/Overview.py's
"Research Question" card are what a reader actually clicks through.

Two consequences of declaring navigation explicitly, both intended:

- Streamlit stops auto-discovering pages/. Its own documentation is blunt
  about this: once any session executes st.navigation, the app ignores the
  pages/ directory. Files stay where they are and are declared here instead.
- The numeric filename prefixes no longer control anything. They are left in
  place because renaming would touch the catalog, the drift test, and every
  page link for no user-visible gain. url_path strips them, so existing
  links such as /Reliability and /Fairness keep working.
"""

import streamlit as st

from shared.catalog import workflows_by_category
from shared.research_journeys import JOURNEYS

st.set_page_config(
    page_title="OpenMeasure Lab",
    layout="centered",
)

# The overview sits above the sections rather than inside one. An
# empty-string section header is how Streamlit renders an ungrouped entry.
sections: dict[str, list[st.Page]] = {
    "": [
        st.Page(
            "pages/Overview.py",
            title="Home",
            url_path="Overview",
            default=True,
        ),
        st.Page(
            "pages/Explore_Real_Data.py",
            title="Explore Real Data",
            url_path="Explore_Real_Data",
        ),
        st.Page(
            "pages/Method_Selection.py",
            title="Method Selection",
            url_path="Method_Selection",
        ),
        st.Page(
            "pages/Privacy_and_Data_Access.py",
            title="Privacy & Data Access",
            url_path="Privacy_and_Data_Access",
        ),
        st.Page(
            "pages/Resources.py",
            title="Resources",
            url_path="Resources",
        ),
        st.Page(
            "pages/Research_Journeys.py",
            title="Research Journeys",
            url_path="Research_Journeys",
        ),
    ],
    # Hidden from the sidebar (each journey would otherwise be its own
    # entry, or its own domain section -- both read as repetitive). Still
    # part of the navigation graph, so st.page_link and each existing
    # direct URL keep resolving; pages/Research_Journeys.py and
    # pages/Overview.py's "Research Question" card are what links to them.
    "Research Journeys (hidden)": [
        st.Page(
            journey.page,
            title=journey.title,
            url_path=journey.url_path,
            visibility="hidden",
        )
        for journey in JOURNEYS
    ],
}

# Category headings come from the catalog verbatim. Inventing a shorter
# sidebar-only label per category would mean a second name for the same
# thing, which is what drifts.
for category, workflows in workflows_by_category().items():
    sections[category] = [
        st.Page(
            workflow.page,
            title=workflow.workflow,
            url_path=workflow.url_path,
        )
        for workflow in workflows
    ]

st.navigation(sections).run()
