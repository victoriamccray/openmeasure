"""
OpenMeasure entrypoint and navigation.

This file declares the navigation and nothing else. It renders no content,
because anything drawn before nav.run() would appear on top of every page.
The landing page itself is pages/Overview.py.

The sidebar is built from shared/catalog.py, grouped by validation category.
That makes the catalog the single source for the sidebar, the overview
cards, and the drift test in shared/tests/test_catalog.py, so a workflow
cannot exist in one and be missing from another.

Explore Real Data is declared separately, below, rather than through the
catalog. It is a pointer to datasets, not a validation workflow: it has no
lifecycle stage, no validation category, and records nothing to
shared/handoff.py, so folding it into workflows_by_category() would imply
a status it can never have.

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
    ]
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
