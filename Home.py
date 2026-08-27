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

Research Journeys is meant to have a single visible "Research Journeys"
entry, pages/Research_Journeys.py, which is a landing page listing every
journey grouped by domain (shared/research_journeys.py's
journeys_by_domain()) for a reader to pick one from, with each individual
journey page (visibility="hidden" below) reachable from there or by
direct URL rather than getting its own sidebar entry.

KNOWN ISSUE, deliberately left as-is: visibility="hidden" does not
currently suppress these pages from the sidebar's automatic widget in
this app once its real page set is registered (every journey still shows
its own sidebar entry) -- confirmed locally, cause not isolated. A first
attempt to work around it (calling st.navigation with position="hidden"
and rendering the sidebar by hand instead) caused a hard
StreamlitPageNotFoundError in production: on the deployed Streamlit
version, position="hidden" broke string-path st.page_link resolution for
every hidden page, including from pages/Research_Journeys.py and
pages/3_Fairness.py, which is worse than the cosmetic issue it was meant
to fix. That attempt is reverted. Local testing environment and the
deployed Streamlit Cloud version are not confirmed to match
(requirements.txt pins streamlit>=1.49 with no upper bound), which is
likely why a locally-verified fix broke in production -- any further
attempt at this should be verified against the exact deployed version, or
behind a change small enough to revert instantly if it isn't.

Two other consequences of declaring navigation explicitly, both intended:

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
            "pages/Quality_and_Validation.py",
            title="Quality & Validation",
            url_path="Quality_and_Validation",
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

# Hidden from the sidebar (each journey would otherwise be its own entry,
# or its own domain section -- both read as repetitive). Still part of the
# navigation graph, so st.page_link and each existing direct URL keep
# resolving; pages/Research_Journeys.py and pages/Overview.py's "Research
# Question" card are what links to them. See the module docstring's
# KNOWN ISSUE note: this visibility="hidden" does not currently suppress
# the sidebar entry either, but unlike the position="hidden" alternative,
# it does not break st.page_link, so it is the safer of the two known-bad
# options until this is properly fixed.
sections["Research Journeys (hidden)"] = [
    st.Page(
        journey.page,
        title=journey.title,
        url_path=journey.url_path,
        visibility="hidden",
    )
    for journey in JOURNEYS
]

st.navigation(sections).run()
