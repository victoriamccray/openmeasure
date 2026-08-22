"""
Shared reporting utilities used across OpenMeasure modules.

Every module (reliability, fairness, program evaluation, ...) renders its
results through these helpers instead of building its own ad hoc verdict
or layout logic. This is what "standardize validation, reporting, and
visual design across modules" actually means in code: one place that
defines what a verdict banner, a section header, and a flagged-item
callout look like, reused everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import streamlit as st

from shared.case_studies import get_case_studies
from shared.catalog import LIFECYCLE_STAGES, STAGE_QUESTIONS, WORKFLOWS, workflows_by_stage
from shared.handoff import HandoffEntry
from shared.progress import stage_progress

# One heading for the research examples section, used on every page so the
# section is always named rather than left for the reader to infer.
CASE_STUDIES_HEADING = "Research Examples & Case Studies"


@dataclass(frozen=True)
class Band:
    """One labeled tier in a threshold-based verdict scale."""

    threshold: float  # inclusive lower bound for this band
    label: str
    tone: str  # "success" | "warning" | "error" | "info"


def classify(value: float, bands: Sequence[Band]) -> Band:
    """
    Return the highest band whose threshold the value meets or exceeds.

    Example:
        bands = [
            Band(0.0, "Unacceptable", "error"),
            Band(0.5, "Poor", "error"),
            Band(0.6, "Questionable", "warning"),
            Band(0.7, "Acceptable", "info"),
            Band(0.8, "Good", "success"),
            Band(0.9, "Excellent", "success"),
        ]
        classify(0.82, bands)  -> Band(0.8, "Good", "success")
    """
    sorted_bands = sorted(bands, key=lambda b: b.threshold)
    matched = sorted_bands[0]
    for band in sorted_bands:
        if value >= band.threshold:
            matched = band
    return matched


_RENDERERS = {
    "success": st.success,
    "warning": st.warning,
    "error": st.error,
    "info": st.info,
}


def render_verdict(band: Band) -> None:
    """Render a Band using the tone-appropriate Streamlit component."""
    renderer = _RENDERERS.get(band.tone, st.info)
    renderer(f"**{band.label}**")


def section_header(title: str, caption: str | None = None) -> None:
    """Consistent section divider + header used at the start of every
    report section, in every module."""
    st.divider()
    st.subheader(title)
    if caption:
        st.caption(caption)


def flagged_item_note(name: str, message: str) -> None:
    """Consistent styling for a single flagged item/case in a diagnostics
    table (used for flagged scale items, flagged subgroups, flagged
    outcome cases, etc. depending on the module)."""
    st.caption(f"**{name}**: {message}")


def caveat(text: str) -> None:
    """Consistent styling for a methodological caveat shown under a
    headline result (e.g. 'these are conventions, not laws')."""
    st.caption(text)


def show_case_studies(module: str) -> None:
    """
    Display research case studies relevant to a module, most relevant first.

    module should match one of the taxonomy keys used in
    shared/case_studies.py (e.g. "measurement_validation",
    "program_validation", "model_validation", "data_validation"), not
    necessarily the folder name of the calling module.

    Each study is collapsed behind its own title, so the reader scans a
    short list and opens what is relevant instead of reading several hundred
    words laid out in full. The research stage is shown alongside each title
    rather than used as a grouping heading, so a page can lead with its most
    relevant example while still telling the reader where in a study that
    lesson applies.

    Renders nothing when no case studies are tagged for the module, so it is
    safe to call from every page.
    """
    studies = get_case_studies(module)

    if not studies:
        return

    # The heading lives here rather than in each page. There are nine call
    # sites across five pages, including early-return branches, and only
    # three of them used to carry a heading at all, so the section was
    # unlabelled on two pages entirely. Owning it here makes the label
    # impossible to omit or to word differently from one page to the next.
    section_header(CASE_STUDIES_HEADING)

    # One bounded panel, so the examples read as a discrete section rather
    # than as loose boxes in the page flow.
    with st.container(border=True):
        st.caption(_stage_coverage(studies))

        for study in studies:
            with st.expander(f"{study.title} ({study.stage})"):
                st.caption(study.principle)
                st.write(study.summary)
                st.info(study.takeaway)
                st.caption(study.citation)


def _stage_coverage(studies: Sequence) -> str:
    """
    Describe which research stages a set of examples covers.

    Stages are listed in research order rather than in the order the
    examples happen to appear, because the point of the line is to show
    where in a study these lessons apply. Reading "Measurement, Analysis,
    Interpretation" conveys a progression; the display order does not.
    """
    present = [
        stage
        for stage in LIFECYCLE_STAGES
        if any(study.stage == stage for study in studies)
    ]

    if len(present) == 1:
        return f"All at the {present[0].lower()} stage of a study."

    return "Spanning " + ", ".join(present).lower() + "."


# Shown under the current stage's column only, on a validation page. A
# location marker, not a status: it uses a fixed, neutral color (blue, the
# same convention a map uses for "you are here") rather than the
# green/red vocabulary that would make it read as pass or fail.
CURRENT_STAGE_MARKER = "You are here"
CURRENT_STAGE_MARKER_ICON = ":material/near_me:"
CURRENT_STAGE_MARKER_COLOR = "blue"

# Shown inside a stage's popover when no workflow exists for it yet, and
# the stage has no other page to point to instead (unlike Research
# Question, which points to pages/Research_Journeys.py below).
NO_WORKFLOW_FOR_STAGE = "Not yet covered by a module."

# One icon per stage, purely for visual identity. Muted Material Symbols,
# matching the icon treatment already used for page and expander icons
# elsewhere, not colored or judgmental: a stage's icon does not change
# based on whether anything has been recorded for it.
_STAGE_ICONS: dict[str, str] = {
    "Research Question": ":material/help:",
    "Data": ":material/database:",
    "Measurement": ":material/straighten:",
    "Analysis": ":material/query_stats:",
    "Interpretation": ":material/lightbulb:",
}


def render_lifecycle_tracker(
    entries: tuple[HandoffEntry, ...] = (),
    *,
    current_workflow: str | None = None,
    show_status: bool = False,
) -> None:
    """
    One column per lifecycle stage, reused on Home and on every validation
    page so there is exactly one implementation of the tracker rather than
    a slightly different copy per page.

    Clicking a stage never navigates by itself. Each stage is a popover
    naming its workflow(s), via a page_link inside the popover, so picking
    where to go stays a deliberate second step rather than an immediate
    jump. It carries no arrows, tick marks, or counter: the stages are
    ordered, but the workflows are not a prerequisite chain, and any of
    those would turn "an analysis was recorded" into "this step is done".

    current_workflow marks "you are here" by matching a Workflow.workflow
    name from shared/catalog.py, so a rename there fails loudly instead of
    silently leaving the marker on the wrong stage. None on Home, where no
    single stage applies.

    show_status additionally shows each stage's existing Recorded / Not
    assessed / Partly recorded / Reads records / No module yet caption
    from shared/progress.py, gated exactly as on Home today (hidden until
    something has been recorded). False on every validation page, which is
    what keeps the tracker there lightweight: entries defaults to an empty
    tuple, since a page that never asks for status never needs to fetch
    the handoff store just to draw its tracker.
    """
    highlighted_stage = None

    if current_workflow is not None:
        match = next(
            (item for item in WORKFLOWS if item.workflow == current_workflow),
            None,
        )
        if match is None:
            raise ValueError(
                f"'{current_workflow}' does not match any workflow in "
                "shared/catalog.py."
            )
        highlighted_stage = match.stage

    grouped = workflows_by_stage()
    stage_state_by_name = (
        {item.stage: item.state for item in stage_progress(entries)}
        if show_status
        else {}
    )

    with st.container(border=True):
        columns = st.columns(len(LIFECYCLE_STAGES))

        for column, stage in zip(columns, LIFECYCLE_STAGES):
            with column:
                with st.popover(
                    stage,
                    icon=_STAGE_ICONS.get(stage),
                    type="tertiary",
                    width="stretch",
                ):
                    st.caption(STAGE_QUESTIONS[stage])

                    workflows = grouped[stage]

                    if not workflows and stage == "Research Question":
                        # No numbered workflow covers this stage; Research
                        # Journeys does instead (see
                        # shared/research_journeys.py's docstring).
                        st.page_link(
                            "pages/Research_Journeys.py",
                            label="Open Research Journeys",
                            icon=":material/arrow_forward:",
                        )
                    elif not workflows:
                        st.caption(NO_WORKFLOW_FOR_STAGE)

                    for workflow in workflows:
                        st.page_link(
                            workflow.page,
                            label=f"Open {workflow.workflow}",
                            icon=":material/arrow_forward:",
                        )

                if stage == highlighted_stage:
                    st.badge(
                        CURRENT_STAGE_MARKER,
                        icon=CURRENT_STAGE_MARKER_ICON,
                        color=CURRENT_STAGE_MARKER_COLOR,
                    )

                if show_status:
                    st.caption(stage_state_by_name[stage])
