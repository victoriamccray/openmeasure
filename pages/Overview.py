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

"How OpenMeasure Works" sits between "How To Use OpenMeasure" and the module
map, so a reader meets the ways in, then the shape of the thing, then the
workflows themselves. It describes the five systems the toolkit is built from
and leads with a diagram, because five interacting parts is structure, and
shared/visuals.py's grammar draws structure rather than listing it twice.
"""

import streamlit as st

from shared import visuals
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

# Body-text ink. shared/visuals.py's INK_MUTED is tuned for strokes beside
# a number and is too light to read as a box label, the same reason
# pages/Quality_and_Validation.py carries its own _INK.
_INK = "#52514e"
_SURFACE = "#fcfcfb"

# One line, because a multi-line attribute value carries its own newlines
# and indentation into what a screen reader announces.
_ARCHITECTURE_LABEL = (
    "Research UI reads and writes research state, which drives both the "
    "scientific logic and the evidence and data infrastructure. "
    "Validation infrastructure checks all of them."
)


def _architecture_svg() -> str:
    """
    Static diagram of the five systems, drawn as structure rather than
    as a list a second time.

    The stack is the path an analysis actually takes: the UI reads and
    writes research state, state drives both the scientific logic and
    the evidence infrastructure, and validation sits under all of them.
    Solid strokes throughout, since every one of these relationships is
    established in the codebase; shared/visuals.py reserves dashes for
    what is assumed.

    Returned as one line with no indentation, the same as
    shared/visuals.py's primitives. st.markdown renders raw HTML only
    while it still looks like raw HTML: four-space-indented lines are
    read as an indented code block instead, and the diagram silently
    loses every element after the first one or two. That is what the
    hand-indented template on pages/Quality_and_Validation.py currently
    hits, and it is why this builder concatenates rather than using a
    formatted block.
    """
    box = f'rx="6" fill="{_SURFACE}" stroke="{visuals.ACCENT}" stroke-width="1.4"'
    connector = f'stroke="{_INK}" stroke-width="1.4" opacity="0.7"'
    head = f'fill="{_INK}" opacity="0.8"'
    label = f'text-anchor="middle" font-size="12" fill="{_INK}"'

    parts = (
        # Research UI, and the two-way link down to research state.
        f'<rect x="20" y="12" width="600" height="40" {box}/>',
        f'<text x="320" y="37" {label}>Research UI</text>',
        f'<line x1="320" y1="56" x2="320" y2="76" {connector}/>',
        f'<polygon points="320,56 316,62 324,62" {head}/>',
        f'<polygon points="320,76 316,70 324,70" {head}/>',
        # Research state, then the split into the two systems it drives.
        f'<rect x="20" y="80" width="600" height="40" {box}/>',
        f'<text x="320" y="105" {label}>Research state</text>',
        f'<line x1="320" y1="120" x2="320" y2="134" {connector}/>',
        f'<line x1="165" y1="134" x2="475" y2="134" {connector}/>',
        f'<line x1="165" y1="134" x2="165" y2="146" {connector}/>',
        f'<polygon points="165,150 161,144 169,144" {head}/>',
        f'<line x1="475" y1="134" x2="475" y2="146" {connector}/>',
        f'<polygon points="475,150 471,144 479,144" {head}/>',
        f'<rect x="20" y="150" width="290" height="52" {box}/>',
        f'<text x="165" y="181" {label}>Scientific logic</text>',
        f'<rect x="330" y="150" width="290" height="52" {box}/>',
        f'<text x="475" y="174" {label}>Evidence and data</text>',
        f'<text x="475" y="190" {label}>infrastructure</text>',
        # Validation, checking the layers above it.
        f'<line x1="165" y1="220" x2="165" y2="208" {connector}/>',
        f'<polygon points="165,202 161,208 169,208" {head}/>',
        f'<line x1="475" y1="220" x2="475" y2="208" {connector}/>',
        f'<polygon points="475,202 471,208 479,208" {head}/>',
        f'<rect x="20" y="220" width="600" height="40" {box}/>',
        f'<text x="320" y="245" {label}>Validation infrastructure</text>',
    )

    return (
        '<svg viewBox="0 0 640 272" role="img" '
        'preserveAspectRatio="xMidYMid meet" '
        'style="width:100%;height:auto;display:block;'
        "font-family:system-ui,-apple-system,'Segoe UI',sans-serif\" "
        f'aria-label="{_ARCHITECTURE_LABEL}">' + "".join(parts) + "</svg>"
    )


_ARCHITECTURE_SVG = _architecture_svg()

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
# How OpenMeasure works
# ---------------------------------------------------------------------

st.subheader("How OpenMeasure Works")

st.markdown(
    """
OpenMeasure is built from five interacting systems. Each holds one part
of an analysis, and the boundaries between them are enforced by tests in
the codebase.
"""
)

st.markdown(_ARCHITECTURE_SVG, unsafe_allow_html=True)

st.caption(
    "The four layers an analysis passes through, and the validation "
    "infrastructure that checks each of them."
)

st.markdown(
    """
**Research UI**
Streamlit pages, stage navigation, visual measure explorers, diagrams,
and controls. This layer presents the results that the scientific logic
returns.

**Research state**
What you have selected, what depends on what, what has changed, and what
still needs review, carried across stages and across workflows.

**Scientific logic**
Statistical calculations, diagnostics, comparison rules, methodological
constraints, and the measurement ontology, the vocabulary that connects
a concept you want to observe to the measures able to observe it. These
are pure functions, and they run independently of the interface around
them.

**Evidence and data infrastructure**
Public datasets, uploads, provenance, literature, artifacts, and the
handoffs that let one workflow read another workflow's result.

**Validation infrastructure**
Unit tests against hand-calculable and literature-cited values, edge
cases that must raise a clear error, page-render tests, and comparison
against independent reference software for the statistics that have a
reference implementation.
"""
)

st.caption(
    "The software computes, organizes, visualizes, and tracks evidence. "
    "The researcher retains the methodological judgment."
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
            "Research Journeys and Explore Real Data occupy this stage. "
            "Framing a research question is a judgment rather than "
            "something a statistic validates, so each walks a real "
            "dataset through the questions it can and cannot support."
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
    "Explore Real Data to build validation judgment on real examples."
)
st.page_link(
    "pages/Method_Selection.py",
    label="Not sure which check fits your question? Open Method Selection",
    icon=":material/alt_route:",
)
st.caption(
    "Already know which module you need? Jump straight to it from the "
    "sidebar."
)

st.subheader("Best-Suited Workflows")
st.markdown(
    "OpenMeasure fits reproducibility, reliability, fairness, data-"
    "quality, and program-evaluation questions on your own research "
    "data, models, or programs. It reports each finding on its own "
    "terms, alongside the assumptions and tradeoffs behind it, and "
    "leaves the methodological judgment and the domain expertise with "
    "you."
)

st.subheader("Limitations & Future Directions")
st.markdown(
    "Current modules cover a defined set of established statistical "
    "methods and worked examples, which is narrower than the full "
    "range of validation questions and dataset types. Future releases "
    "are expected to expand data-validation and fairness coverage and "
    "connect further findings across the research workflow."
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
