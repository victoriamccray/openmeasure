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
from shared.catalog import LIFECYCLE_STAGES

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
    section_header(
        CASE_STUDIES_HEADING,
        "Published cases behind this module's assumptions and limitations",
    )

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
