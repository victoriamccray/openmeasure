"""
OpenMeasure overview, the default landing page.

Home.py is now the navigation entrypoint and renders nothing of its own,
because anything drawn before nav.run() would appear on every page. This
file holds what the landing page actually shows.

The module map is rendered from shared/catalog.py, which is the single place
workflow names, versions, and descriptions are recorded, and which the
sidebar is also built from. A test in shared/tests/test_catalog.py fails if
that catalog and the pages/ directory disagree, so this page cannot silently
fall out of date the way the hand-written list it replaced did.

The sidebar groups workflows by validation category. This page arranges them
by research stage. That is deliberate: the sidebar answers "where do I go",
and this page answers "when in a study would I need this".

Cards also carry a recording status once anything has been recorded, so the
page answers what has not been looked at rather than only what exists. The
status is absent on a first visit: a wall of "Not assessed" before a user has
had the chance to do anything reads as a scolding rather than as guidance.
"""

import streamlit as st

from shared.catalog import (
    LIFECYCLE_STAGES,
    STAGE_QUESTIONS,
    workflows_by_stage,
)
from shared.handoff import HandoffStore
from shared.progress import (
    SESSION_SCOPE_NOTE,
    has_any_records,
    status_caption,
    workflow_progress,
)
from shared.report import render_lifecycle_tracker

st.title("OpenMeasure Lab")
st.caption(
    "An open-source validation toolkit for research data, measures, models, and programs."
)

st.divider()

st.markdown(
    """
OpenMeasure brings together statistical methods, transparent reporting, and
plain-language interpretation to support validation throughout the research
process.
"""
)

st.divider()


# ---------------------------------------------------------------------
# How to use OpenMeasure
# ---------------------------------------------------------------------

st.subheader("How To Use OpenMeasure")

st.markdown("OpenMeasure can be used in multiple ways:")

st.markdown(
    """
**Learn with Research Journeys**
Work through real datasets step by step to see how validation decisions
arise across the research workflow.

**Apply to your research**
Use validation modules to identify appropriate checks, understand
assumptions and tradeoffs, interpret results, and document decisions
for your own analysis.
"""
)

st.caption(
    "OpenMeasure is designed to support methodological reasoning "
    "alongside statistical expertise, domain knowledge, and established "
    "analysis tools."
)

st.divider()


# ---------------------------------------------------------------------
# Module map, by research lifecycle stage
# ---------------------------------------------------------------------

st.subheader("Where Each Module Fits")

st.markdown(
    """
The modules below are arranged by the stage of a study where the question
arises, from framing a question through to interpreting a result. Each card
also names the kind of validation it performs, which is how the modules are
grouped in the sidebar and described in their own documentation.
"""
)

entries = HandoffStore(st.session_state).entries()
show_status = has_any_records(entries)
status_by_workflow = {
    item.workflow.workflow: status_caption(item)
    for item in workflow_progress(entries)
}

# The richer, status-aware variant of shared/report.py's tracker: every
# stage's Recorded / Not assessed / etc. state is shown once anything has
# been recorded, same as before this moved into a shared component. The
# tracker itself is always shown, because it is a map of the lifecycle and
# useful before anything is recorded; only the state labels are gated.
render_lifecycle_tracker(entries, show_status=show_status)

if show_status:
    st.caption(SESSION_SCOPE_NOTE)

grouped = workflows_by_stage()

for stage in LIFECYCLE_STAGES:
    workflows = grouped[stage]

    st.markdown(f"#### {stage}")
    st.caption(STAGE_QUESTIONS[stage])

    if stage == "Research Question":
        # No numbered workflow covers this stage (shared/catalog.py's
        # STAGES_WITHOUT_WORKFLOWS), because framing a research question is
        # not something a statistic validates. Research Journeys and
        # Explore Real Data occupy it instead -- see
        # shared/research_journeys.py's docstring for the fuller reasoning.
        st.caption(
            "No numbered workflow covers this stage. Research Journeys and "
            "Explore Real Data occupy it instead: each walks a real "
            "dataset through what it can and cannot support, rather than "
            "computing a statistic."
        )

        with st.container(border=True):
            st.markdown("**Research Journeys**")
            st.write(
                "Guided worked examples, grouped by field: Multi-Modal "
                "Health Imaging, Social Impact Evaluation, Responsible AI."
            )
            st.page_link(
                "pages/Research_Journeys.py",
                label="Open Research Journeys",
                icon=":material/arrow_forward:",
            )

        with st.container(border=True):
            st.markdown("**Explore Real Data**")
            st.write(
                "Real, citable datasets to practice validation judgment "
                "on, with no prescribed columns or expected results."
            )
            st.page_link(
                "pages/Explore_Real_Data.py",
                label="Open Explore Real Data",
                icon=":material/arrow_forward:",
            )
        continue

    if not workflows:
        # Stated rather than hidden. Omitting the stage would imply the
        # lifecycle begins later than it does.
        st.info(
            "Not yet covered. No OpenMeasure module currently supports this "
            "stage."
        )
        continue

    # Two across only when there are two, since the centered layout is
    # narrow and a lone half-width card reads as a mistake.
    columns = st.columns(2) if len(workflows) == 2 else [st.container()]

    for column, workflow in zip(columns, workflows):
        with column:
            with st.container(border=True):
                st.markdown(f"**{workflow.workflow}**")
                st.badge(workflow.category)
                st.caption(f"Version {workflow.version}")
                st.write(workflow.summary)

                if show_status:
                    # Plain muted text, with no color coding. A green tick
                    # would read as "passed", and a recorded analysis is not
                    # a passing verdict on the validation stage.
                    st.caption(status_by_workflow[workflow.workflow])

                st.page_link(
                    workflow.page,
                    label=f"Open {workflow.workflow}",
                    icon=":material/arrow_forward:",
                )

st.divider()

st.subheader("Design Principles")

st.markdown(
    """
Every module follows the same principles:

- Transparent statistical methods
- Reproducible analyses
- Plain-language interpretation
- Explicit assumptions and limitations
- Documented references
- Responsible use grounded in established research and professional ethics
"""
)

st.caption(
    "OpenMeasure is designed for researchers and practitioners working in "
    "community health, social services, education, public policy, and applied "
    "research. The toolkit emphasizes transparent validation methods that are "
    "accessible, reproducible, and adaptable across disciplines."
)

st.divider()


# ---------------------------------------------------------------------
# Getting started, limitations, AI use, and contact
# ---------------------------------------------------------------------

st.subheader("Getting Started")
st.markdown(
    "New to a dataset or question? Start with Research Journeys or "
    "Explore Real Data to build validation judgment on real examples. "
    "Already know which check you need? Jump straight to a module from "
    "the sidebar, or use Method Selection to get routed to one."
)

st.subheader("Best-Suited Workflows")
st.markdown(
    "OpenMeasure fits reproducibility, reliability, fairness, data-"
    "quality, and program-evaluation questions on your own research "
    "data, models, or programs. It does not replace domain expertise, "
    "and it does not produce a single composite pass/fail score."
)

st.subheader("Limitations & Future Directions")
st.markdown(
    "Current modules cover a defined set of established statistical "
    "methods and worked examples, not every validation question or "
    "dataset type. Future releases are expected to expand data-"
    "validation and fairness coverage and connect further findings "
    "across the research workflow."
)

st.subheader("Generative AI Use Statement")
st.markdown(
    "Portions of this codebase were developed with the assistance of "
    "generative AI tools for code drafting, debugging, and "
    "documentation; all statistical methods were independently "
    "verified, and all design and scope decisions were made by the "
    "project's author. See `docs/Authorship.md` for details."
)

st.subheader("Contact")
st.markdown(
    "Questions, feedback, or contributions are welcome: please open an "
    "issue on the [GitHub repository](https://github.com/victoriamccray/openmeasure) "
    "before submitting a pull request. See `CONTRIBUTING.md` for details."
)
