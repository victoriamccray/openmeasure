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

from shared.case_studies import get_case_studies_grouped


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
    Display research case studies relevant to a module, if any exist,
    grouped into tabs by category when a module has more than one.

    module should match one of the taxonomy keys used in
    shared/case_studies.py (e.g. "measurement_validation",
    "program_validation", "model_validation", "data_validation"), not
    necessarily the folder name of the calling module.

    Renders nothing if no case studies are tagged for the given module,
    so it's safe to call from every page without an empty expander
    showing up before content exists for that module.
    """
    grouped = get_case_studies_grouped(module)

    if not grouped:
        return

    def _render_study(study) -> None:
        st.markdown(f"#### {study.title}")
        st.caption(f"Principle: {study.principle}")
        st.write(study.summary)
        st.info(study.takeaway)
        st.caption(study.citation)

    with st.expander("Why these assumptions matter"):
        categories = list(grouped.keys())

        if len(categories) == 1:
            for study in grouped[categories[0]]:
                _render_study(study)
        else:
            tabs = st.tabs(categories)
            for tab, category in zip(tabs, categories):
                with tab:
                    for study in grouped[category]:
                        _render_study(study)
