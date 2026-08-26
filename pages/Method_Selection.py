"""
OpenMeasure - Method Selection Decision Tree, and Research Design.

Two modes on one page, chosen at the top:

- "Choose an analysis" (the page's original content, unchanged): one
  guided question routing to the workflow or research journey that
  fits, with Try, Why, You'll learn, and Limitations for the chosen
  branch. This mode assumes a question or dataset already exists.
- "Plan a Study": for a question upstream of that, before any data
  exists. Measure-first, not vocabulary-first: Research question ->
  Explore & Assemble Measures (click a measure for a mini-lesson, then
  choose which to include) -> Timing & Synchronization -> Simulate the
  Design (a fixed chronic-pain worked example) -> Reveal Terminology &
  Implications, where formal design vocabulary (observational,
  within-person, repeated-measures) is named only after it has already
  been experienced, and modules/research_design/core/inspect_rules.py's
  deterministic rules run against what was actually assembled. Uses
  modules/research_design/core/ to simulate the fixed scenario, then
  hands the resulting measurement plan's shape to the same
  suggest_workflows() the upload branch below already uses, landing in
  "Choose an analysis" territory from the design side rather than the
  data side.

This is a guidance page, not an analysis: it records nothing to
shared/handoff.py, carries no module_key, and is deliberately not a
numbered page, so it needs no entry in shared/catalog.py and cannot
appear on the overview's progress cards or stage strip. Design mode's
simulated data is always synthetic (see modules/research_design/core/
for exactly what is and is not modeled), so it carries no data-handling
disclosure of its own; "Choose an analysis"'s optional CSV upload
predates this file's disclosure conventions and has none either - a
pre-existing gap, not something this change introduces or resolves.

The entry question in "Choose an analysis" is phrased as what you're
trying to determine, not as which OpenMeasure tool or category you
need, so it works for a reader who does not yet know that "several
items measuring one thing consistently" is what reliability means. Each
destination already has its own, more specific method selection logic
once you are inside it (for example, Impact Evaluation recommends a
statistical test based on your data's shape) or, for a research
journey, its own fixed sequence of stages. This page only routes
between destinations, one level up from that, and stops there.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from modules.data_profile.core.suggest import WorkflowSuggestion, suggest_workflows
from modules.research_design.core.design import DesignAssumptions
from modules.research_design.core.estimate import estimate_coupling_difference
from modules.research_design.core.inspect_rules import (
    MEASUREMENT_TYPE_BODY_MAP,
    MEASUREMENT_TYPE_SURVEY_SCALE,
    MEASUREMENT_TYPE_WEARABLE_SENSOR,
    TIME_STRUCTURE_LONGITUDINAL,
    Inspection,
    StudyStructure,
    inspect_study_structure,
)
from modules.research_design.core.schema import measurement_plan_profile
from modules.research_design.core.simulate import generate_naturalistic_pain_study
from shared.catalog import WORKFLOWS
from shared.journey_stages import StageTracker
from shared.method_guide import BRANCHES
from shared.report import caveat, implications, inspect_note, interpretation_note, section_header
from shared.research_journeys import JOURNEYS
from shared.upload import render_data_profile

st.set_page_config(
    page_title="OpenMeasure - Method Selection",
    page_icon=":material/alt_route:",
    layout="centered",
)

st.title("Method Selection")

_PAGE_BY_WORKFLOW = {item.workflow: item.page for item in WORKFLOWS}
_PAGE_BY_JOURNEY = {item.title: item.page for item in JOURNEYS}
_PAGE_BY_DESTINATION = {**_PAGE_BY_WORKFLOW, **_PAGE_BY_JOURNEY}

INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
ACCENT = "#2a78d6"
ACCENT_2 = "#c0392b"

MISSING_COLOR = "#d8d6cd"


def _synchronization_svg(misalignment_minutes: float) -> str:
    """
    Two horizontal timelines, redrawn on each slider change rather
    than animated continuously: the wearable line's tick marks shift
    right of the pain-report line's by an amount proportional to the
    slider, making "these two streams drift apart" visible directly
    rather than only as a number of minutes.
    """

    offset = (misalignment_minutes / 60.0) * 70.0
    pain_x = (70, 220)
    wearable_x = tuple(x + offset for x in pain_x)

    pain_ticks = "".join(
        f'<line x1="{x}" y1="24" x2="{x}" y2="36" stroke="{ACCENT_2}" stroke-width="2.5"/>'
        for x in pain_x
    )
    wearable_ticks = "".join(
        f'<line x1="{x:.0f}" y1="74" x2="{x:.0f}" y2="86" stroke="{ACCENT}" stroke-width="2.5"/>'
        for x in wearable_x
    )

    return f"""
    <svg width="100%" height="100" viewBox="0 0 320 100" preserveAspectRatio="xMidYMid meet">
      <text x="8" y="14" font-size="10" fill="{ACCENT_2}">Pain report</text>
      <line x1="8" y1="30" x2="312" y2="30" stroke="{GRIDLINE}" stroke-width="1.5"/>
      {pain_ticks}
      <text x="8" y="64" font-size="10" fill="{ACCENT}">Wearable signal</text>
      <line x1="8" y1="80" x2="312" y2="80" stroke="{GRIDLINE}" stroke-width="1.5"/>
      {wearable_ticks}
      <line x1="{pain_x[0]}" y1="36" x2="{wearable_x[0]:.0f}" y2="74" stroke="{INK_MUTED}" stroke-width="1" stroke-dasharray="3,2"/>
      <text x="{(pain_x[0] + wearable_x[0]) / 2:.0f}" y="55" font-size="9" fill="{INK_MUTED}" text-anchor="middle">{misalignment_minutes:.0f} min</text>
    </svg>
    """

_DEFAULT_BODY_ZONE = "Abdomen"

_BODY_ZONES = (
    {"zone": "Head", "x": 60, "y": 14},
    {"zone": "Left shoulder", "x": 38, "y": 42},
    {"zone": "Right shoulder", "x": 82, "y": 42},
    {"zone": "Chest", "x": 60, "y": 55},
    {"zone": "Abdomen", "x": 60, "y": 85},
    {"zone": "Left arm", "x": 24, "y": 80},
    {"zone": "Right arm", "x": 96, "y": 80},
    {"zone": "Left hip", "x": 48, "y": 100},
    {"zone": "Right hip", "x": 72, "y": 100},
    {"zone": "Left leg", "x": 43, "y": 150},
    {"zone": "Right leg", "x": 77, "y": 150},
    {"zone": "Left foot", "x": 38, "y": 200},
    {"zone": "Right foot", "x": 82, "y": 200},
)
_BODY_ZONE_BY_NAME = {z["zone"]: z for z in _BODY_ZONES}

# Which zones a marker "radiates" into next door, used only for the
# distributed/radiating pain pattern; not a claim about how real
# referred pain spreads, just a plausible neighbor to fan out toward.
_ZONE_ADJACENCY = {
    "Head": ("Left shoulder", "Right shoulder"),
    "Left shoulder": ("Head", "Chest", "Left arm"),
    "Right shoulder": ("Head", "Chest", "Right arm"),
    "Chest": ("Left shoulder", "Right shoulder", "Abdomen"),
    "Abdomen": ("Chest", "Left hip", "Right hip"),
    "Left arm": ("Left shoulder",),
    "Right arm": ("Right shoulder",),
    "Left hip": ("Abdomen", "Left leg"),
    "Right hip": ("Abdomen", "Right leg"),
    "Left leg": ("Left hip", "Left foot"),
    "Right leg": ("Right hip", "Right foot"),
    "Left foot": ("Left leg",),
    "Right foot": ("Right leg",),
}

_HEAD_OUTLINE = [
    (60 + 13 * math.cos(2 * math.pi * i / 24), 16 + 15 * math.sin(2 * math.pi * i / 24))
    for i in range(25)
]

# Articulated at the joints (shoulder/elbow/wrist, hip/knee/ankle) and a
# tapered torso outline, rather than a rectangle-and-straight-lines stick
# figure, so it reads as a body rather than a diagram symbol; still an
# abstract line drawing, not an anatomically precise or measured shape.
_SILHOUETTE_PARTS = (
    ("head", _HEAD_OUTLINE),
    ("neck", [(60, 30), (60, 38)]),
    ("torso", [(40, 40), (80, 40), (74, 96), (46, 96), (40, 40)]),
    ("left_arm", [(40, 44), (26, 72), (19, 100)]),
    ("right_arm", [(80, 44), (94, 72), (101, 100)]),
    ("left_hand", [(14, 98), (24, 103)]),
    ("right_hand", [(96, 98), (106, 103)]),
    ("left_leg", [(52, 96), (46, 146), (40, 196)]),
    ("right_leg", [(68, 96), (74, 146), (80, 196)]),
    ("left_foot", [(32, 200), (48, 200)]),
    ("right_foot", [(72, 200), (88, 200)]),
)


def _body_map_chart_spec(selected_zone: str, pain_state: str) -> dict:
    """
    A tap-to-place digital body map, drawn as Vega-Lite marks (not a
    static image) so Streamlit's chart click selection can report
    which zone was tapped: the silhouette and the faint zone dots are
    fixed, and only the colored pain marker layer depends on
    selected_zone and pain_state, redrawing on each click or radio
    toggle rather than animating.

    Zone placement, adjacency, and marker position are drawn for
    legibility, not measured or derived from real referred-pain
    patterns.
    """

    silhouette_rows = [
        {"part": part, "order": i, "x": x, "y": y}
        for part, points in _SILHOUETTE_PARTS
        for i, (x, y) in enumerate(points)
    ]

    marker_zones = [selected_zone] if selected_zone in _BODY_ZONE_BY_NAME else [_DEFAULT_BODY_ZONE]
    if pain_state == "distributed":
        marker_zones += list(_ZONE_ADJACENCY.get(marker_zones[0], ()))

    marker_rows = [
        {
            "zone": zone,
            "x": _BODY_ZONE_BY_NAME[zone]["x"],
            "y": _BODY_ZONE_BY_NAME[zone]["y"],
            "opacity": 0.85 if i == 0 else max(0.25, 0.6 - 0.15 * i),
            "rank": i,
        }
        for i, zone in enumerate(marker_zones)
    ]

    shared_encoding = {
        "x": {
            "field": "x",
            "type": "quantitative",
            "axis": None,
            "scale": {"domain": [0, 120]},
        },
        "y": {
            "field": "y",
            "type": "quantitative",
            "axis": None,
            "scale": {"domain": [0, 220], "reverse": True},
        },
    }

    return {
        "encoding": shared_encoding,
        "layer": [
            {
                "data": {"values": silhouette_rows},
                "mark": {"type": "line", "color": INK_MUTED, "strokeWidth": 1.5},
                "encoding": {
                    "detail": {"field": "part", "type": "nominal"},
                    "order": {"field": "order", "type": "quantitative"},
                },
            },
            {
                "data": {"values": [dict(z) for z in _BODY_ZONES]},
                "mark": {"type": "circle", "color": INK_MUTED, "opacity": 0.14, "size": 260},
                "encoding": {"tooltip": [{"field": "zone", "type": "nominal", "title": "Tap to place pain here"}]},
                "params": [
                    {"name": "zone_click", "select": {"type": "point", "fields": ["zone"], "on": "click"}}
                ],
            },
            {
                "data": {"values": marker_rows},
                "mark": {"type": "circle", "color": ACCENT_2},
                "encoding": {
                    "size": {
                        "field": "rank",
                        "type": "ordinal",
                        "legend": None,
                        "scale": {"range": [260, 90]},
                    },
                    "opacity": {"field": "opacity", "type": "quantitative", "legend": None},
                    "tooltip": [{"field": "zone", "type": "nominal"}],
                },
            },
        ],
        "width": 150,
        "height": 275,
        "config": {"view": {"stroke": None}},
    }


def _scr_bump(t: float, onset: float, rise: float = 0.35, decay: float = 1.6, amplitude: float = 1.0) -> float:
    """One phasic skin-conductance-response bump: a fast rise and a slower exponential decay, the textbook EDA shape."""

    if t < onset:
        return 0.0
    dt = t - onset
    return amplitude * (1 - math.exp(-dt / rise)) * math.exp(-dt / decay)


def _pulse(t: float, beat: float, width: float = 0.05) -> float:
    """One narrow heartbeat-like pulse centered at `beat`."""

    return math.exp(-((t - beat) ** 2) / (2 * width * width))


def _single_trace_spec(rows: list[dict], title: str) -> dict:
    """One small drawn line trace, shared shape for every measure mini-lesson's chart."""

    return {
        "data": {"values": rows},
        "mark": {"type": "line", "color": ACCENT, "strokeWidth": 2},
        "encoding": {
            "x": {"field": "t", "type": "quantitative", "axis": None},
            "y": {"field": "y", "type": "quantitative", "axis": None},
        },
        "title": {"text": title, "fontSize": 10, "color": INK_MUTED, "fontWeight": "normal"},
        "width": "container",
        "height": 90,
        "config": {"view": {"stroke": None}},
    }


_HR_REGULAR_BEATS = tuple(0.5 + 0.8 * i for i in range(10))
_HR_IRREGULAR_BEATS = (0.4, 1.1, 1.5, 2.4, 2.7, 3.7, 4.0, 5.1, 5.4, 6.6, 6.9, 7.9)


def _eda_trace_spec(pain_state: str) -> dict:
    """
    A drawn, normative EDA (skin-conductance-response) shape: a fast
    rise and slower exponential decay, the textbook electrodermal
    signature. Only the number of bumps changes with pain_state, to
    make "coupling can differ by state" visible before any statistics
    are introduced; the shape itself is not computed from data.
    """

    eda_onsets = (
        [(1.0, 1.0), (3.1, 0.75), (5.3, 0.9)] if pain_state == "distributed" else [(2.6, 1.0)]
    )
    ts = [i * 0.08 for i in range(101)]
    rows = [
        {"t": t, "y": 0.15 + sum(_scr_bump(t, onset, amplitude=amp) for onset, amp in eda_onsets)}
        for t in ts
    ]
    return _single_trace_spec(rows, "EDA, skin conductance (normative shape)")


def _hr_trace_spec(beats: tuple[float, ...], title: str) -> dict:
    """
    A drawn heartbeat pulse train at the given beat times, not computed
    from any data. Overlapping narrow pulses alone make regular and
    irregular spacing hard to tell apart at a glance, so a row of tick
    marks at the actual beat times is layered underneath as an explicit
    timing rug.
    """

    ts = [i * 0.02 for i in range(401)]
    rows = [{"t": t, "y": sum(_pulse(t, beat) for beat in beats)} for t in ts]
    tick_rows = [{"t": beat, "y": -0.2} for beat in beats]
    return {
        "layer": [
            {
                "data": {"values": rows},
                "mark": {"type": "line", "color": ACCENT, "strokeWidth": 2},
                "encoding": {
                    "x": {"field": "t", "type": "quantitative", "axis": None},
                    "y": {
                        "field": "y",
                        "type": "quantitative",
                        "axis": None,
                        "scale": {"domain": [-0.3, 1.1]},
                    },
                },
            },
            {
                "data": {"values": tick_rows},
                "mark": {"type": "tick", "color": INK_MUTED, "thickness": 2, "size": 12},
                "encoding": {
                    "x": {"field": "t", "type": "quantitative", "axis": None},
                    "y": {"field": "y", "type": "quantitative", "axis": None},
                },
            },
        ],
        "title": {"text": title, "fontSize": 10, "color": INK_MUTED, "fontWeight": "normal"},
        "width": "container",
        "height": 90,
        "config": {"view": {"stroke": None}},
    }


# The measure gallery for "Explore & Assemble Measures": each entry's
# generic_type maps a pain-case-specific measure onto one of
# inspect_rules.py's MEASUREMENT_TYPE_OPTIONS labels, so the same
# deterministic rule engine built for "Enter your own" studies still
# runs unmodified here. eda, heart_rate, and hrv all map to the same
# generic type deliberately: simulate.py's v0.1 model collapses all
# three into one combined physio_signal channel (see its module
# docstring), so the measurement-plan table built from this mapping
# should show that collapse rather than implying three separate columns.
_MEASURE_GALLERY = (
    {"key": "pain_rating", "label": "Pain rating", "generic_type": MEASUREMENT_TYPE_SURVEY_SCALE},
    {"key": "body_map", "label": "Body map", "generic_type": MEASUREMENT_TYPE_BODY_MAP},
    {"key": "eda", "label": "EDA", "generic_type": MEASUREMENT_TYPE_WEARABLE_SENSOR},
    {"key": "heart_rate", "label": "Heart rate", "generic_type": MEASUREMENT_TYPE_WEARABLE_SENSOR},
    {"key": "hrv", "label": "HRV", "generic_type": MEASUREMENT_TYPE_WEARABLE_SENSOR},
)
_MEASURE_LABEL_BY_KEY = {m["key"]: m["label"] for m in _MEASURE_GALLERY}
_ALL_MEASURE_KEYS = [m["key"] for m in _MEASURE_GALLERY]

_MEASURE_COLUMN_INFO = {
    "pain_rating": ("Pain intensity", "pain_rating"),
    "body_map": ("Spatial pain pattern", "pain_state"),
    "eda": ("Physiological arousal (combined wearable channel)", "physio_signal"),
    "heart_rate": ("Physiological arousal (combined wearable channel)", "physio_signal"),
    "hrv": ("Physiological arousal (combined wearable channel)", "physio_signal"),
}


def _acquisition_pictograph_svg(kind: str) -> str:
    """A small hand-drawn icon for one acquisition method, matching this page's static line-and-marker pictograph style."""

    if kind == "rating":
        ticks = "".join(
            f'<line x1="{10 + i * 10}" y1="45" x2="{10 + i * 10}" y2="55" '
            f'stroke="{INK_MUTED}" stroke-width="1"/>'
            for i in range(11)
        )
        body = (
            f'<line x1="10" y1="50" x2="110" y2="50" stroke="{INK_MUTED}" stroke-width="1.5"/>'
            f"{ticks}"
            f'<circle cx="67" cy="50" r="6" fill="{ACCENT_2}"/>'
        )
    elif kind == "wearable":
        body = (
            f'<line x1="35" y1="18" x2="25" y2="80" stroke="{INK_MUTED}" stroke-width="1.5" stroke-linecap="round"/>'
            f'<line x1="85" y1="18" x2="95" y2="80" stroke="{INK_MUTED}" stroke-width="1.5" stroke-linecap="round"/>'
            f'<rect x="42" y="34" width="36" height="30" rx="6" fill="none" stroke="{ACCENT}" stroke-width="1.5"/>'
            f'<path d="M47,49 L54,49 L57,41 L61,57 L65,49 L73,49" fill="none" '
            f'stroke="{ACCENT}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'
            f'<path d="M83,31 A6,6 0 0 1 89,25" fill="none" stroke="{INK_MUTED}" stroke-width="1.2"/>'
            f'<path d="M79,35 A12,12 0 0 1 91,19" fill="none" stroke="{INK_MUTED}" stroke-width="1.2"/>'
        )
    else:
        body = (
            f'<circle cx="60" cy="45" r="26" fill="none" stroke="{INK_MUTED}" stroke-width="1.5"/>'
            f'<line x1="60" y1="45" x2="60" y2="28" stroke="{INK_MUTED}" stroke-width="1.5" stroke-linecap="round"/>'
            f'<line x1="60" y1="45" x2="74" y2="52" stroke="{INK_MUTED}" stroke-width="1.5" stroke-linecap="round"/>'
        )

    return f"""
    <svg width="100%" height="70" viewBox="0 0 120 70" preserveAspectRatio="xMidYMid meet">
      {body}
    </svg>
    """


def _render_workflow_suggestions(suggestions: tuple[WorkflowSuggestion, ...]) -> None:
    """
    Shared rendering for a set of WorkflowSuggestions, used both by
    "Choose an analysis"'s upload branch and "Design a study"'s
    simulated-data-structure handoff, so the two entry points describe
    a structural match identically rather than drifting into two
    wordings for the same thing.
    """

    if not suggestions:
        st.info(
            "This shape didn't clearly match a pattern this suggester "
            "recognizes."
        )
        return

    st.caption(
        "These are structural matches, not a determination of "
        "methodological appropriateness: shape alone can't establish "
        "your actual research question or intent. If more than one "
        "workflow is shown, that reflects a real ambiguity structure "
        "can't resolve on its own, not an error."
    )

    for suggestion in suggestions:
        with st.container(border=True):
            st.markdown(f"**{suggestion.workflow}** may be relevant")
            st.write(suggestion.reasoning)
            st.page_link(
                _PAGE_BY_DESTINATION[suggestion.workflow],
                label=f"Open {suggestion.workflow}",
                icon=":material/arrow_forward:",
            )

    st.page_link(
        "pages/Explore_Real_Data.py",
        label="Browse real datasets shaped like these workflows",
        icon=":material/dataset:",
    )


def _render_inspections(inspections: tuple[Inspection, ...]) -> None:
    """
    Render deterministic structure-only inspection points for
    "Enter your own", the same "things to inspect, not a
    determination" framing as _render_workflow_suggestions above, but
    driven by fixed if/then rules on study structure
    (modules/research_design/core/inspect_rules.py) rather than a
    DataProfile's shape.
    """

    if not inspections:
        st.caption(
            "No structural triggers fire for the current selections: "
            "try Experimental + Between-person, Longitudinal, a "
            "measurement type, or the subgroup checkbox above."
        )
        return

    for inspection in inspections:
        with st.container(border=True):
            st.markdown(f"**{inspection.trigger}**")
            st.write(inspection.note)
            if inspection.suggested_module:
                st.page_link(
                    _PAGE_BY_DESTINATION[inspection.suggested_module],
                    label=f"Open {inspection.suggested_module}",
                    icon=":material/arrow_forward:",
                )


MODE_DESIGN = "design"
MODE_ANALYSIS = "analysis"

mode = st.radio(
    "Method Selection is the entry point whether you're before or after "
    "data collection:",
    # Analyze Existing Data first/default: it was this page's sole
    # purpose before Plan a Study existed, so a bare page load keeps
    # behaving the way every existing link to this page already expects.
    options=(MODE_ANALYSIS, MODE_DESIGN),
    format_func=lambda key: {
        MODE_DESIGN: "Plan a Study",
        MODE_ANALYSIS: "Analyze Existing Data",
    }[key],
    horizontal=True,
)

st.divider()

# =======================================================================
# Mode: Choose an analysis (original page content, unchanged below)
# =======================================================================

if mode == MODE_ANALYSIS:
    st.caption(
        "Answer one question about what you're trying to determine, and "
        "OpenMeasure points to the workflow or research journey that "
        "fits, and explains why."
    )

    _BRANCH_BY_ID = {branch.id: branch for branch in BRANCHES}

    selected_id = st.radio(
        "What are you trying to determine?",
        options=[branch.id for branch in BRANCHES],
        format_func=lambda branch_id: _BRANCH_BY_ID[branch_id].question,
    )

    branch = _BRANCH_BY_ID[selected_id]

    # -------------------------------------------------------------
    # Pictograph: one small, original, hand-drawn icon per destination.
    #
    # Static on purpose, not animated: a flickering/pulsing icon set
    # tried earlier on the GRAND worked example was flagged as both a
    # seizure-trigger risk (rapid, high-contrast flicker) and useless to
    # a screen-reader user, who already has the same information in
    # text -- here, the radio options above and the "Try:" line below.
    # This pictograph is a decorative supplement to that text, never
    # its only copy.
    #
    # These are original pictographs (a checklist, a gapped timeline, a
    # balance, ...), not a reproduction of any icon library's artwork.
    # -------------------------------------------------------------

    INK_SECONDARY = "#52514e"
    SURFACE = "#fcfcfb"
    # INK_MUTED, GRIDLINE, and ACCENT are module-level (shared with the
    # Plan a Study mode's body-map and synchronization diagrams).

    _DESTINATION_ICON_PATHS: dict[str, str] = {
        "reliability": (
            '<line x1="5" y1="10" x2="17" y2="10" stroke="{c}" stroke-width="2.2" stroke-linecap="round"/>'
            '<line x1="5" y1="16" x2="17" y2="16" stroke="{c}" stroke-width="2.2" stroke-linecap="round"/>'
            '<line x1="5" y1="22" x2="17" y2="22" stroke="{c}" stroke-width="2.2" stroke-linecap="round"/>'
            '<path d="M21 16L24.5 19.5L30 12" stroke="{c}" stroke-width="2.2" fill="none" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        ),
        "time_series_qa": (
            '<line x1="3" y1="22" x2="29" y2="22" stroke="{c}" stroke-width="1.3"/>'
            '<circle cx="8" cy="22" r="2.1" fill="{c}"/>'
            '<circle cx="14" cy="22" r="2.1" fill="{c}"/>'
            '<circle cx="20" cy="22" r="2.1" fill="none" stroke="{c}" stroke-width="1.4" stroke-dasharray="1.4,1.6"/>'
            '<circle cx="26" cy="22" r="2.1" fill="{c}"/>'
            '<line x1="20" y1="8" x2="20" y2="16" stroke="{c}" stroke-width="1.8" stroke-linecap="round"/>'
            '<circle cx="20" cy="5.2" r="1.2" fill="{c}"/>'
        ),
        "impact_evaluation": (
            '<line x1="3" y1="26" x2="29" y2="26" stroke="{c}" stroke-width="1.2"/>'
            '<rect x="6" y="16" width="7" height="10" rx="1" fill="{c}" opacity="0.35"/>'
            '<rect x="19" y="8" width="7" height="18" rx="1" fill="{c}"/>'
        ),
        "fairness": (
            '<line x1="16" y1="5" x2="16" y2="24" stroke="{c}" stroke-width="1.6"/>'
            '<line x1="6" y1="9" x2="26" y2="9" stroke="{c}" stroke-width="1.6"/>'
            '<path d="M3 9L6.5 17A5 5 0 0 0 9.5 17Z" fill="none" stroke="{c}" '
            'stroke-width="1.4" stroke-linejoin="round"/>'
            '<path d="M22.5 9L26 17A5 5 0 0 0 29 17Z" fill="none" stroke="{c}" '
            'stroke-width="1.4" stroke-linejoin="round"/>'
            '<line x1="11" y1="25" x2="21" y2="25" stroke="{c}" stroke-width="2" stroke-linecap="round"/>'
        ),
        "cross_analysis": (
            '<circle cx="12.5" cy="16" r="9" fill="{c}" opacity="0.16" stroke="{c}" stroke-width="1.4"/>'
            '<circle cx="19.5" cy="16" r="9" fill="{c}" opacity="0.16" stroke="{c}" stroke-width="1.4"/>'
        ),
        "portfolio_impact": (
            '<rect x="4" y="5" width="9" height="9" rx="1.3" fill="{c}" opacity="0.25" stroke="{c}" stroke-width="1.1"/>'
            '<rect x="18" y="5" width="9" height="9" rx="1.3" fill="{c}" opacity="0.25" stroke="{c}" stroke-width="1.1"/>'
            '<rect x="4" y="18" width="9" height="9" rx="1.3" fill="{c}" stroke="{c}" stroke-width="1.1"/>'
            '<rect x="18" y="18" width="9" height="9" rx="1.3" fill="{c}" opacity="0.25" stroke="{c}" stroke-width="1.1"/>'
        ),
    }

    def _icon_svg(path_template: str, color: str) -> str:
        return path_template.format(c=color)

    def _destination_pictograph_html(branches: tuple, selected: str) -> str:
        """
        One small card per destination, with the selected one in the
        accent color and the rest muted. Wording lives only in the
        radio/Try/Why text above and is repeated here as plain text
        inside each card, so a screen-reader user reaches the same
        information whether or not the icon itself renders.
        """

        cards = []
        for item in branches:
            is_selected = item.id == selected
            color = ACCENT if is_selected else INK_MUTED
            border = ACCENT if is_selected else GRIDLINE
            background = "rgba(42, 120, 214, 0.08)" if is_selected else SURFACE
            weight = "600" if is_selected else "400"

            cards.append(
                f'<div style="display:flex; flex-direction:column; align-items:center; '
                f'width:116px; min-width:0; box-sizing:border-box; gap:4px; padding:8px 6px; '
                f'border-radius:8px; border:1.4px solid {border}; background:{background};">'
                f'<svg width="32" height="32" viewBox="0 0 32 32" style="flex-shrink:0;">'
                f"{_icon_svg(_DESTINATION_ICON_PATHS[item.id], color)}"
                f"<title>{item.destination}</title>"
                f"</svg>"
                f'<span style="font-size:11px; font-weight:{weight}; color:{color}; '
                f'text-align:center; line-height:1.25; min-width:0; '
                f'word-break:break-word;">{item.destination}</span>'
                f"</div>"
            )

        return f"""
        <div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
                    display:flex; flex-wrap:wrap; justify-content:center; gap:8px;">
          {"".join(cards)}
        </div>
        """

    components.html(_destination_pictograph_html(BRANCHES, selected_id), height=190)
    st.caption("The highlighted card is where your answer points.")

    st.markdown(f"**Try: {branch.destination}**")
    st.caption(
        "This is a research journey: a guided worked example, not a "
        "versioned analysis workflow."
        if branch.destination_kind == "research journey"
        else "This is an analysis workflow: it records a result you can "
        "revisit from Cross-Analysis Implications."
    )
    st.page_link(
        _PAGE_BY_DESTINATION[branch.destination],
        label=f"Open {branch.destination}",
        icon=":material/arrow_forward:",
    )

    st.markdown("**Why**")
    st.write(branch.why)

    with st.expander("You'll learn"):
        st.write(branch.youll_learn)

    with st.expander("Limitations"):
        for limitation in branch.limitations:
            st.markdown(f"- {limitation}")

    st.divider()

    st.caption(
        "Each destination above has its own, more specific logic once "
        "you're inside it: a workflow's own method recommendation, or a "
        "journey's own sequence of stages. This page only points you to "
        "where to start."
    )

    st.divider()

    st.subheader("Or, Upload a File Instead")
    st.caption(
        "Don't know what to ask yet? Upload a file and OpenMeasure "
        "names which workflows may be relevant based on its column "
        "structure. Structure can't establish that a workflow is the "
        "methodologically appropriate choice, only that its shape is "
        "worth a look. The question above (and each destination's own "
        "Why) is what actually states a case for one."
    )

    uploaded = st.file_uploader(
        "CSV file", type="csv", label_visibility="collapsed", key="method_selection_upload"
    )

    if uploaded is not None:
        upload_df = pd.read_csv(uploaded)
        upload_profile = render_data_profile(upload_df)
        suggestions = suggest_workflows(upload_profile)
        _render_workflow_suggestions(suggestions)

    st.page_link(
        "pages/Research_Journeys.py",
        label="None of these fit yet: browse all Research Journeys",
        icon=":material/route:",
    )

# =======================================================================
# Mode: Plan a Study
# =======================================================================

else:
    st.caption(
        "Build a study and explore how design choices shape the "
        "evidence, before any data exists. v0.1's simulation always "
        "models one built-in scenario, a naturalistic pain study, "
        "regardless of what you enter below. It never scores a design "
        "as good or bad, only what it does and does not support."
    )
    st.caption(
        "This walkthrough is OpenMeasure's own structural logic, not "
        "drawn from a published design-selection framework. For a "
        "broader existing method-selection tool, see the Co-Creation "
        "Methods Navigator on the Resources page."
    )
    st.page_link(
        "pages/Resources.py",
        label="Open Resources",
        icon=":material/collections_bookmark:",
    )

    STAGE_QUESTION = 0
    STAGE_MEASURES = 1
    STAGE_TIMING = 2
    STAGE_SIMULATE = 3
    STAGE_IMPLICATIONS = 4

    DESIGN_STAGE_LABELS = (
        "Research question",
        "Explore & assemble measures",
        "Timing & synchronization",
        "Simulate the design",
        "Reveal terminology & implications",
    )

    design_tracker = StageTracker(
        session_key="research_design_stage", stage_labels=DESIGN_STAGE_LABELS
    )

    design_stage = design_tracker.render_breadcrumb()
    design_tracker.render_restart_button()

    st.divider()

    # -------------------------------------------------------------
    # 0. Research question
    # -------------------------------------------------------------

    section_header("Research Question", "Enter your own, or load the built-in example")

    PAIN_EXAMPLE = {
        "rq_hypothesis": (
            "The coupling between subjective pain and physiological "
            "signals (electrodermal activity and heart rate/heart-rate "
            "variability) changes when chronic pain is localized versus "
            "spatially distributed, referred, or radiating."
        ),
        "rq_population": "Adults with chronic pain, observed in daily life.",
        "rq_exposure": "Spatial pain state: localized vs. distributed/referred/radiating.",
        "rq_outcomes": "Within-person coupling between pain rating and a wearable physiological signal.",
        "rq_setting": "Naturalistic: participants' everyday environments, not a lab visit.",
        "assembled_measures": list(_ALL_MEASURE_KEYS),
        "study_compares_subgroups": False,
    }

    rq_button_cols = st.columns([1, 1, 4])
    with rq_button_cols[0]:
        if st.button("Load pain example"):
            for key, value in PAIN_EXAMPLE.items():
                st.session_state[key] = value
            st.rerun()
    with rq_button_cols[1]:
        if st.button("Clear"):
            for key in PAIN_EXAMPLE:
                st.session_state.pop(key, None)
            st.rerun()

    hypothesis = st.text_area(
        "Research question / hypothesis", key="rq_hypothesis", height=100
    )

    question_cols = st.columns(2)
    with question_cols[0]:
        population = st.text_input("Population", key="rq_population")
        exposure = st.text_input("Exposure / intervention", key="rq_exposure")
    with question_cols[1]:
        outcomes = st.text_input("Outcomes", key="rq_outcomes")
        setting = st.text_input("Setting", key="rq_setting")

    caveat(
        "These fields describe your question and go into the Design "
        "Record below. What comes next is a fixed, hands-on worked "
        "example (the chronic-pain scenario) regardless of what you "
        "enter here: v0.1 has no generic simulator for an arbitrary "
        "study yet."
    )

    if design_stage < STAGE_MEASURES:
        if st.button("Continue to explore measures", type="primary"):
            design_tracker.advance_to(STAGE_MEASURES)

    # -------------------------------------------------------------
    # 1. Explore & assemble measures
    # -------------------------------------------------------------

    n_participants = observations_per_day = duration_days = None

    if design_stage >= STAGE_MEASURES:
        section_header(
            "Explore & Assemble Measures",
            "How could we observe pain in everyday life? Tap a measure to see what it captures.",
        )

        lesson_choice = st.radio(
            "Measure to explore",
            options=_ALL_MEASURE_KEYS,
            format_func=lambda k: _MEASURE_LABEL_BY_KEY[k],
            horizontal=True,
            key="measure_lesson_choice",
        )

        if lesson_choice == "pain_rating":
            st.markdown(_acquisition_pictograph_svg("rating"), unsafe_allow_html=True)
            st.write(
                "A pain rating is subjective and self-reported: the "
                "participant taps a number from 0 to 10. The same "
                "person's own ratings changing over time, not one "
                "rating compared across different people, is what this "
                "study's within-person comparison relies on."
            )

        elif lesson_choice == "body_map":
            body_map_state = st.radio(
                "Pain pattern to show",
                options=("localized", "distributed"),
                format_func=lambda key: {
                    "localized": "Localized",
                    "distributed": "Distributed / radiating",
                }[key],
                horizontal=True,
                key="measurement_plan_body_map_state",
            )
            if "body_map_zone" not in st.session_state:
                st.session_state["body_map_zone"] = _DEFAULT_BODY_ZONE
            st.caption(
                f"Tap a body region to place the pain marker. Current: "
                f"**{st.session_state['body_map_zone']}**."
            )
            click_event = st.vega_lite_chart(
                _body_map_chart_spec(st.session_state["body_map_zone"], body_map_state),
                use_container_width=False,
                on_select="rerun",
                key="body_map_zone_click",
            )
            selection = getattr(click_event, "selection", None)
            clicked = selection.get("zone_click") if selection else None
            clicked_zone = None
            if isinstance(clicked, dict) and clicked.get("zone"):
                zone_values = clicked["zone"]
                clicked_zone = zone_values[0] if isinstance(zone_values, list) else zone_values
            elif isinstance(clicked, list) and clicked:
                clicked_zone = clicked[0].get("zone")
            if clicked_zone and clicked_zone != st.session_state["body_map_zone"]:
                st.session_state["body_map_zone"] = clicked_zone
                st.rerun()
            st.write(
                "Marking one point captures a **localized** pattern; "
                "marking a spreading pattern captures **distributed / "
                "referred / radiating** pain. That single tap becomes "
                "the pain_state column: a category derived from where "
                "the marker was placed, not a continuous number."
            )
            st.caption(
                "An abstract drawing, not a real body map: marker "
                "placement is user-chosen, drawn for legibility, not "
                "measured."
            )

        elif lesson_choice == "eda":
            eda_cols = st.columns([2, 1])
            with eda_cols[0]:
                st.vega_lite_chart(_eda_trace_spec("localized"), use_container_width=True)
            with eda_cols[1]:
                st.markdown(_acquisition_pictograph_svg("wearable"), unsafe_allow_html=True)
            st.write(
                "Skin conductance rises sharply when the sympathetic "
                "nervous system activates, then decays slowly over "
                "several seconds. One sharp rise-and-fall is one "
                "phasic response: this shape, not its raw amplitude, "
                "is the textbook electrodermal signature this study "
                "looks for."
            )
            st.caption("A drawn, normative shape, not a measured signal, read continuously by a worn sensor.")

        else:
            rhythm = st.radio(
                "Rhythm to show",
                options=("regular", "irregular"),
                format_func=lambda k: {
                    "regular": "Regular spacing",
                    "irregular": "Irregular spacing",
                }[k],
                horizontal=True,
                key="hr_rhythm_choice",
            )
            beats = _HR_REGULAR_BEATS if rhythm == "regular" else _HR_IRREGULAR_BEATS
            st.vega_lite_chart(_hr_trace_spec(beats, "Heartbeat pulses"), use_container_width=True)
            if lesson_choice == "heart_rate":
                st.write(
                    "Heart rate is simply how many beats happen per "
                    "unit time, regardless of whether the spacing "
                    "between them is even or uneven."
                )
            else:
                st.write(
                    "Heart rate variability is about the *spacing* "
                    "between beats, not the count: the irregular "
                    "pattern above has roughly the same number of "
                    "beats as the regular one, but the interval "
                    "between each beat differs."
                )
            st.caption(
                "A drawn, normative shape of the signal. The tick marks "
                "below the curve mark each beat's actual timing."
            )

        st.divider()
        st.markdown("**Assemble Your Measurement System**")
        st.caption("Choose which of the measures above to include in this study.")

        assembled_measures = st.multiselect(
            "Measures included in this study",
            options=_ALL_MEASURE_KEYS,
            format_func=lambda k: _MEASURE_LABEL_BY_KEY[k],
            default=st.session_state.get("assembled_measures", list(_ALL_MEASURE_KEYS)),
            key="assembled_measures",
        )

        if assembled_measures:
            measurement_rows = pd.DataFrame(
                [
                    {
                        "Measure": _MEASURE_LABEL_BY_KEY[key],
                        "Construct": _MEASURE_COLUMN_INFO[key][0],
                        "Column": _MEASURE_COLUMN_INFO[key][1],
                    }
                    for key in assembled_measures
                ]
            )
            st.dataframe(measurement_rows, width="stretch", hide_index=True)
            wearable_keys = {"eda", "heart_rate", "hrv"}
            if len(wearable_keys & set(assembled_measures)) > 1:
                st.caption(
                    "EDA, heart rate, and HRV all map to the same "
                    "physio_signal column here: v0.1's simulation "
                    "models one combined wearable channel standing in "
                    "for all three, not three separate ones."
                )
        else:
            st.info("Add at least one measure above to continue.")

        measurement_types = tuple(
            sorted({
                m["generic_type"] for m in _MEASURE_GALLERY if m["key"] in assembled_measures
            })
        )

        if design_stage < STAGE_TIMING and assembled_measures:
            if st.button("Continue to timing & synchronization", type="primary"):
                design_tracker.advance_to(STAGE_TIMING)

    # -------------------------------------------------------------
    # 2. Timing & synchronization
    # -------------------------------------------------------------

    if design_stage >= STAGE_TIMING:
        section_header(
            "Timing & Synchronization",
            "When are measures collected, and how closely aligned are they?",
        )

        sample_cols = st.columns(3)
        with sample_cols[0]:
            n_participants = st.slider("Number of participants", 5, 100, 30)
        with sample_cols[1]:
            observations_per_day = st.slider("Measurement frequency (per day)", 1, 10, 4)
        with sample_cols[2]:
            duration_days = st.slider("Study duration (days)", 3, 30, 7)

        # The living study diagram: a single reactive summary line, not a
        # static description - every value in it comes from the choices
        # directly above and from the measures assembled in the previous
        # stage, so moving one changes what this line says without
        # waiting for a later stage.
        measurement_label = (
            " + ".join(_MEASURE_LABEL_BY_KEY[k] for k in assembled_measures)
            if assembled_measures
            else "no measures chosen yet"
        )
        st.markdown(
            f":material/group: **{n_participants} participants** &rarr; "
            f":material/calendar_month: **{duration_days} days** &rarr; "
            f":material/repeat: **{observations_per_day}x/day** &rarr; "
            f":material/sensors: **{measurement_label}**"
        )
        st.caption(
            f"{n_participants * observations_per_day * duration_days:,} "
            "observations planned in total, before adherence or missingness."
        )

        st.markdown("**How Are the Two Streams Aligned?**")
        temporal_misalignment_minutes = st.slider(
            "Temporal misalignment between rating and wearable (minutes)", 0, 60, 10
        )
        st.markdown(
            _synchronization_svg(float(temporal_misalignment_minutes)),
            unsafe_allow_html=True,
        )
        st.caption(
            "The gap between when a pain rating is logged and when the "
            "wearable actually reads. Larger values make the recorded "
            "rating a noisier stand-in for what was actually happening "
            "physiologically at that moment."
        )

        inspect_note(
            "Temporal alignment is a timing decision here, and an "
            "adjustable assumption in the next stage."
        )

        if design_stage < STAGE_SIMULATE:
            if st.button("Continue to simulate the design", type="primary"):
                design_tracker.advance_to(STAGE_SIMULATE)

    # -------------------------------------------------------------
    # 3. Simulate the design
    # -------------------------------------------------------------

    assumptions: DesignAssumptions | None = None
    study = None
    estimate = None

    if design_stage >= STAGE_SIMULATE and n_participants is not None:
        section_header(
            "Explore With a Worked Simulation",
            "A fixed chronic-pain scenario, not a simulation of your own study above",
        )

        st.write(
            "This stage always models the same built-in scenario "
            "(observational, within-person, repeated-measures chronic "
            "pain) regardless of the research question or study "
            "structure you entered: v0.1 has no generic simulator for "
            "an arbitrary study yet. The sliders below are assumptions "
            "about what is *true* in that fixed scenario, not facts a "
            "real version of it would already know, and not new design "
            "choices: it would need pilot data or published estimates "
            "to set them credibly."
        )

        noise_cols = st.columns(2)
        with noise_cols[0]:
            adherence_rate = st.slider(
                "Adherence rate (fraction of planned observations actually captured)",
                0.1, 1.0, 0.8, step=0.05,
            )
            st.caption(
                ":material/grid_on: Lower values mean more gray gaps in "
                "the participant timeline below."
            )
            sensor_noise_sd = st.slider("Wearable measurement noise (SD)", 0.0, 2.0, 0.5, step=0.1)
            st.caption(
                ":material/sensors: Higher values blur the wearable "
                "signal's relationship to pain state in the retained rows."
            )
            within_person_sd = st.slider(
                "Within-person physiological variability (SD)", 0.0, 2.0, 0.3, step=0.1
            )
            st.caption(
                "How much one person's own readings bounce around, "
                "observation to observation."
            )
            between_person_sd = st.slider(
                "Between-person variability in baseline coupling (SD)", 0.0, 2.0, 0.3, step=0.1
            )
            st.caption(
                "How different people's typical coupling strength is "
                "from each other's."
            )
        with noise_cols[1]:
            effect_magnitude = st.slider(
                "True effect: coupling difference, distributed minus localized",
                -1.0, 1.0, 0.4, step=0.05,
            )
            st.caption(
                ":material/target: The gap the estimate below is trying "
                "to recover."
            )
            pain_state_prevalence = st.slider(
                "Share of observations in the distributed pain state", 0.05, 0.95, 0.35, step=0.05
            )
            st.caption(
                ":material/scatter_plot: Shifts the localized/distributed "
                "color mix in the timeline below."
            )
            seed = st.number_input("Random seed (for reproducibility)", value=42, step=1)
            st.caption(
                ":material/replay: Same seed and assumptions reproduce "
                "the same rows below, exactly."
            )

        assumptions = DesignAssumptions(
            n_participants=n_participants,
            observations_per_day=observations_per_day,
            duration_days=duration_days,
            adherence_rate=adherence_rate,
            sensor_noise_sd=sensor_noise_sd,
            within_person_sd=within_person_sd,
            between_person_sd=between_person_sd,
            effect_magnitude=effect_magnitude,
            pain_state_prevalence=pain_state_prevalence,
            temporal_misalignment_minutes=float(temporal_misalignment_minutes),
            seed=int(seed),
        )

        st.warning(
            "**Synthetic data, not study participant data.** "
            "modules/research_design/core/simulate.py documents the "
            "exact generative model and which real-world confounders "
            "(medication, activity, sleep, stress) v0.1 does not model."
        )

        study = generate_naturalistic_pain_study(assumptions)

        st.caption(
            f"{study.n_observations_retained:,} of "
            f"{study.n_observations_planned:,} planned observations "
            f"retained ({study.pct_missing:.0%} missing), "
            f"{study.n_localized_observations:,} localized and "
            f"{study.n_distributed_observations:,} distributed."
        )

        example_ids = sorted(study.data["participant_id"].unique())[:5]
        planned_grid = pd.DataFrame(
            [
                {"participant_id": p, "day": d, "observation_index": o}
                for p in example_ids
                for d in range(assumptions.duration_days)
                for o in range(assumptions.observations_per_day)
            ]
        )
        timeline = planned_grid.merge(
            study.data[study.data["participant_id"].isin(example_ids)][
                ["participant_id", "day", "observation_index", "pain_state"]
            ],
            on=["participant_id", "day", "observation_index"],
            how="left",
        )
        timeline["status"] = timeline["pain_state"].fillna("missing")
        timeline["slot"] = (
            timeline["day"] * assumptions.observations_per_day + timeline["observation_index"]
        )
        timeline["participant_label"] = "Participant " + (timeline["participant_id"] + 1).astype(str)

        st.vega_lite_chart(
            {
                "data": {"values": timeline.to_dict("records")},
                "mark": {"type": "point", "filled": True, "size": 90},
                "encoding": {
                    "x": {
                        "field": "slot",
                        "type": "ordinal",
                        "title": "Observation slot, across the study",
                        "axis": {"labels": False, "ticks": False},
                    },
                    "y": {
                        "field": "participant_label",
                        "type": "nominal",
                        "title": None,
                        "sort": None,
                        "axis": {"labelOverlap": False},
                    },
                    "color": {
                        "field": "status",
                        "type": "nominal",
                        "scale": {
                            "domain": ["localized", "distributed", "missing"],
                            "range": [ACCENT, ACCENT_2, MISSING_COLOR],
                        },
                        "legend": {"title": None, "orient": "top"},
                    },
                    "tooltip": [
                        {"field": "participant_label", "type": "nominal", "title": "Participant"},
                        {"field": "day", "type": "ordinal"},
                        {"field": "status", "type": "nominal"},
                    ],
                },
                "width": "container",
                "height": 30 * len(example_ids) + 40,
            },
            use_container_width=True,
        )
        st.caption(
            f"First {len(example_ids)} of {assumptions.n_participants} participants, one "
            "row of dots per person across the study. Gray means that "
            "observation was planned but not captured, given the "
            "adherence rate above."
        )

        st.markdown("**One participant's pain rating and physio signal, together**")
        coupling_participant = st.selectbox(
            "Participant to inspect",
            options=example_ids,
            format_func=lambda p: f"Participant {p + 1}",
            key="coupling_participant_select",
        )
        participant_rows = (
            study.data[study.data["participant_id"] == coupling_participant]
            .sort_values(["day", "observation_index"])
            .assign(slot=lambda d: d["day"] * assumptions.observations_per_day + d["observation_index"])
        )
        coupling_long = pd.concat(
            [
                participant_rows[["slot", "pain_rating"]]
                .rename(columns={"pain_rating": "value"})
                .assign(signal="Pain rating"),
                participant_rows[["slot", "physio_signal"]]
                .rename(columns={"physio_signal": "value"})
                .assign(signal="Physio signal"),
            ]
        )
        st.vega_lite_chart(
            {
                "data": {"values": coupling_long.to_dict("records")},
                "mark": {"type": "line", "point": True, "strokeWidth": 2},
                "encoding": {
                    "x": {"field": "slot", "type": "ordinal", "title": "Observation slot, across the study"},
                    "y": {"field": "value", "type": "quantitative", "title": "Rating / signal value"},
                    "color": {
                        "field": "signal",
                        "type": "nominal",
                        "scale": {"domain": ["Pain rating", "Physio signal"], "range": [ACCENT_2, ACCENT]},
                        "legend": {"title": None, "orient": "top"},
                    },
                },
                "width": "container",
                "height": 180,
            },
            use_container_width=True,
        )
        st.caption(
            "This participant's own retained observations, connected in "
            "order. Gaps from missing observations are skipped, not "
            "interpolated. When the two lines move together, coupling "
            "is strong for this participant, in this mix of states; "
            "when they diverge, it is weak, which the estimate below "
            "quantifies across everyone rather than one person's chart."
        )

        st.dataframe(study.data.head(10), width="stretch", hide_index=True)
        st.caption("First 10 simulated rows, out of the retained total above.")

        estimate = estimate_coupling_difference(study)

        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "Estimated coupling difference",
            f"{estimate.estimated_difference:+.2f}" if estimate.estimated_difference is not None else "n/a",
        )
        metric_cols[1].metric(
            "Standard error",
            f"{estimate.standard_error:.2f}" if estimate.standard_error is not None else "n/a",
        )
        metric_cols[2].metric(
            "Participants used",
            f"{estimate.n_participants_used} / {assumptions.n_participants}",
        )

        inspect_note(
            "How many participants were excluded for insufficient data "
            f"({estimate.n_participants_excluded_insufficient_data}) "
            "relative to how many were used, not just the estimate "
            "itself."
        )

        interpretation_note(
            "This is one analysis aligned to this simulated design (a "
            "within-person correlation difference), not a general "
            "measure of the design's quality and not a recommended "
            "analysis for a real version of this study."
        )

        if design_stage < STAGE_IMPLICATIONS:
            if st.button("Continue to reveal terminology & implications", type="primary"):
                design_tracker.advance_to(STAGE_IMPLICATIONS)

    # -------------------------------------------------------------
    # 4. Reveal terminology & implications
    # -------------------------------------------------------------

    if design_stage >= STAGE_IMPLICATIONS and study is not None and estimate is not None:
        section_header(
            "Reveal Terminology & Implications",
            "What you built, named, and what it does and does not support",
        )

        st.markdown(
            "**The study you've built is a naturalistic, observational, "
            "within-person, repeated-measures design with multimodal "
            "measurements.**"
        )
        st.caption(
            "These terms describe the fixed chronic-pain scenario "
            "Explore With a Worked Simulation just ran, not something "
            "picked from a list: observational (no one assigns pain "
            "state), repeated-measures and longitudinal (the same "
            "participants observed many times), within-person "
            "(each participant compared against their own localized "
            "and distributed episodes), and multimodal (more than one "
            "kind of measure assembled together)."
        )

        recap_steps = (
            (":material/person:", "Participant", "One of the enrolled adults"),
            (
                ":material/sensors:",
                "Assembled measures",
                "The measures you chose in Explore & Assemble Measures",
            ),
            (":material/sync:", "Synchronization", "Aligning the two measurement streams in time"),
            (":material/functions:", "Derived measures", "Per-observation pain state and signal values"),
            (":material/insights:", "Within-person analysis", "Coupling estimated separately per participant"),
            (":material/compare_arrows:", "Comparison", "Coupling compared across pain states"),
        )
        for row_start in (0, 3):
            recap_cols = st.columns(3)
            for column, (icon, label, note) in zip(recap_cols, recap_steps[row_start : row_start + 3]):
                with column:
                    st.markdown(f"{icon} **{label}**")
                    st.caption(note)

        caveat(
            "An observational, within-person design can describe "
            "association within a person over time. It cannot, on its "
            "own, establish that pain state causes a change in "
            "physiology, since anything else that also varies with pain "
            "state (activity, medication timing, sleep) is not "
            "controlled for here."
        )

        section_header(
            "Design Implications",
            "Deterministic, structure-only triggers: things to inspect, not recommendations",
        )
        compares_subgroups = st.checkbox(
            "This design compares demographic subgroups (e.g. age, gender, race)",
            key="study_compares_subgroups",
        )
        structure = StudyStructure(
            design_type="Observational",
            comparison_structure="Within-person",
            time_structure=TIME_STRUCTURE_LONGITUDINAL,
            measurement_types=measurement_types,
            compares_subgroups=compares_subgroups,
        )
        inspections = inspect_study_structure(structure)
        _render_inspections(inspections)

        st.write(
            "A within-person, observational, repeated-measures design "
            "like this one can describe how strongly pain and "
            "physiology move together, and whether that association "
            "differs by pain state, within the participants observed. "
            "It cannot, by itself, establish that spatial pain pattern "
            "*causes* a change in that coupling, rule out confounders "
            "this v0.1 does not model (medication, activity, sleep, "
            "stress), or guarantee that a result here would replicate "
            "in a real study run under these same nominal settings."
        )

        implications(
            "Adherence and temporal misalignment mainly affect how much "
            "usable data survives to be analyzed, between-person "
            "variability and sensor noise mainly affect how uncertain "
            "the estimate is, and pain-state imbalance affects how many "
            "participants have enough of the rarer state to contribute "
            "at all. Changing one does not just move the headline "
            "number, it changes which of these limits binds."
        )

        caveat(
            "Simulated precision and uncertainty describe this "
            "simulation under these assumptions. They are not a "
            "guarantee about how a real study, even one matching these "
            "settings closely, would actually perform."
        )

        st.write("**Relevant OpenMeasure methods**")
        st.caption(
            "The measurement plan above, on its own, implies a column "
            "shape. Matching that shape to OpenMeasure's own workflows "
            "is the same suggest_workflows() the Analyze Existing Data "
            "mode's upload branch uses, run here on a shape derived "
            "from the design instead of an uploaded file."
        )

        profile = measurement_plan_profile(assumptions)
        _render_workflow_suggestions(suggest_workflows(profile))

        with st.expander("How this connects to other OpenMeasure modules"):
            st.markdown(
                """
- **Reliability** would matter if the wearable reported more than one
  derived channel meant to represent the same construct.
- **Fairness** would matter if coupling were compared across a
  demographic or clinical subgroup, not just across a participant's own
  pain states.
- **Evidence Review** is where a real version of this study would start:
  finding what related work already exists on EDA/HR-pain coupling
  before running a new study.

None of these modules are called from here; this is a conceptual map,
not an integration.
"""
            )

        section_header("Design Record", "A summary of this design, to carry forward")

        record_text = f"""OpenMeasure Research Design Record
===================================

Research question
------------------
{hypothesis or PAIN_EXAMPLE["rq_hypothesis"]}

Population: {population or PAIN_EXAMPLE["rq_population"]}
Exposure: {exposure or PAIN_EXAMPLE["rq_exposure"]}
Outcomes: {outcomes or PAIN_EXAMPLE["rq_outcomes"]}
Setting: {setting or PAIN_EXAMPLE["rq_setting"]}

Study design (revealed from what you assembled and ran)
--------------------------------------------------------
Observational, Within-person, Longitudinal, repeated-measures.
Assembled measures: {", ".join(_MEASURE_LABEL_BY_KEY[k] for k in assembled_measures) if assembled_measures else "none chosen"}
Compares demographic subgroups: {"yes" if compares_subgroups else "no"}

Design implications (deterministic, structure-only; things to inspect, not recommendations)
---------------------------------------------------------------------------------------------
{chr(10).join(f"- {i.trigger}: {i.note}" for i in inspections) if inspections else "- No structural triggers fired for the choices above."}

Measurement plan
-----------------
{chr(10).join(f"- {_MEASURE_LABEL_BY_KEY[k]} -> {_MEASURE_COLUMN_INFO[k][1]}" for k in assembled_measures) if assembled_measures else "- None chosen."}

Simulation assumptions (chronic-pain worked example)
-----------------------------------------------------
- Participants: {assumptions.n_participants}
- Observations per day: {assumptions.observations_per_day}
- Duration: {assumptions.duration_days} days
- Adherence rate: {assumptions.adherence_rate:.0%}
- Sensor noise SD: {assumptions.sensor_noise_sd}
- Within-person SD: {assumptions.within_person_sd}
- Between-person SD: {assumptions.between_person_sd}
- True effect (distributed minus localized): {assumptions.effect_magnitude:+.2f}
- Distributed-state prevalence: {assumptions.pain_state_prevalence:.0%}
- Temporal misalignment: {assumptions.temporal_misalignment_minutes:.0f} minutes
- Random seed: {assumptions.seed}

Simulated outcome
-----------------
- Observations retained: {study.n_observations_retained} of {study.n_observations_planned} ({study.pct_missing:.0%} missing)
- Participants used in the estimate: {estimate.n_participants_used} of {assumptions.n_participants}
- Estimated coupling difference: {estimate.estimated_difference if estimate.estimated_difference is not None else "n/a"}
- Standard error: {estimate.standard_error if estimate.standard_error is not None else "n/a"}

Proposed analysis considerations
-----------------------------------
Time-Series QA (timestamp completeness), then Impact Evaluation
(comparing the physiological signal across pain states).

Limitations
-----------
Observational, not experimental: cannot establish causation. Does not
model medication, activity, sleep, or stress. Simulated precision does
not guarantee real-world performance. Synthetic data, not study
participant data.
"""

        st.text_area("Design record", record_text, height=400)
        st.download_button(
            "Download design record (.txt)",
            data=record_text,
            file_name="openmeasure_research_design_record.txt",
            mime="text/plain",
        )
