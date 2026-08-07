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
    stage_progress,
    status_caption,
    workflow_progress,
)

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
# Module map, by research lifecycle stage
# ---------------------------------------------------------------------

st.subheader("Where each module fits")

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

# The stage strip. Deliberately without connecting arrows: the stages are
# ordered, but the workflows are not a prerequisite chain, and an arrow would
# imply Fairness needs Reliability to have run first. It carries no arrows, no
# tick marks, and no counter, since each of those turns "an analysis was
# recorded" into "this step is done".
#
# The strip itself is always shown, because it is a map of the lifecycle and
# useful before anything is recorded. Only the state labels are gated, which
# is the same rule the cards below follow.
with st.container(border=True):
    strip = st.columns(len(LIFECYCLE_STAGES))

    for column, stage in zip(strip, stage_progress(entries)):
        with column:
            st.markdown(f"**{stage.stage}**")

            if show_status:
                st.caption(stage.state)

if show_status:
    st.caption(SESSION_SCOPE_NOTE)

grouped = workflows_by_stage()

for stage in LIFECYCLE_STAGES:
    workflows = grouped[stage]

    st.markdown(f"#### {stage}")
    st.caption(STAGE_QUESTIONS[stage])

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

st.subheader("Design principles")

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
