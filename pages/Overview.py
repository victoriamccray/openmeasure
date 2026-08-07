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
"""

import streamlit as st

from shared.catalog import (
    LIFECYCLE_STAGES,
    STAGE_QUESTIONS,
    workflows_by_stage,
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
Validation is not one step. The modules below are arranged by the stage of a
study where the question arises, from framing a question through to
interpreting a result. Each card also names the kind of validation it
performs, which is how the modules are grouped in the sidebar and described
in their own documentation.
"""
)

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
