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

from modules.data_profile.core.profile import profile_dataframe
from modules.data_profile.core.suggest import WorkflowSuggestion, suggest_workflows
from modules.research_design.core import assembly, examples, lexicon, ontology
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
from shared import visuals
from shared.measure_visuals import ANIMATED_TYPES, measure_visual_svg
from shared.catalog import WORKFLOWS
from shared.stage_workspace import Gate, Stage, StageWorkspace
from shared.method_guide import BRANCHES
from shared.report import caveat, implications, inspect_note, interpretation_note, section_header
from shared.research_journeys import JOURNEYS
from shared.upload import render_data_profile, render_dataset_portrait

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


# The longest stream name the diagram can letter without colliding with
# the timeline it labels. SVG text does not wrap, so a longer name is
# shortened at the caller rather than drawn off the canvas.
# Measures the library describes as producing more than one reading per
# participant: sampled continuously, or reported on a schedule. Minute-
# scale alignment is a question about these. A survey and an
# administrative extract each arrive once, and the correspondence
# between them is about which period each covers.
#
# Derived from the visual vocabulary rather than listed again, so it
# cannot drift out of step with what the drawings say a measure
# produces.
_REPEATEDLY_SAMPLED = ANIMATED_TYPES + (
    ontology.VISUAL_DIARY_GRID,
    ontology.VISUAL_EVENT_TIMELINE,
)


_STREAM_LABEL_LIMIT = 34


def _short_stream_label(measure_name: str) -> str:
    """
    A measure name the synchronization diagram can letter.

    Cut at a word boundary rather than mid-word, and only for the two
    library names long enough to need it. Refusing outright would mean a
    reader could assemble a study whose timing diagram will not draw,
    which is a worse trade than a shortened label beside a slider that
    already spells the name out in full.
    """
    if len(measure_name) <= _STREAM_LABEL_LIMIT:
        return measure_name

    words = measure_name.split()
    shortened = words[0]

    for word in words[1:]:
        if len(f"{shortened} {word}") > _STREAM_LABEL_LIMIT - 1:
            break
        shortened = f"{shortened} {word}"

    return f"{shortened}\u2026"


def _synchronization_svg(
    misalignment_minutes: float, *, first_stream: str, second_stream: str
) -> str:
    """
    Two horizontal timelines, one shifted right of the other.

    Redrawn on each slider change rather than animated continuously: the
    second line's ticks move by an amount proportional to the slider,
    making "these two streams drift apart" visible directly rather than
    only as a number of minutes.

    The streams are named by the caller from what was actually
    assembled. They used to be captioned "Pain report" and "Wearable
    signal", which was the worked example's pair written into a diagram
    every study reached.
    """
    for label in (first_stream, second_stream):
        if len(label) > _STREAM_LABEL_LIMIT:
            raise ValueError(
                f"'{label}' is {len(label)} characters and this diagram "
                f"letters up to {_STREAM_LABEL_LIMIT}. Shorten it at the "
                "caller: SVG text does not wrap, so a longer label is "
                "drawn off the canvas rather than truncated."
            )

    offset = (misalignment_minutes / 60.0) * 70.0
    first_x = (70, 220)
    second_x = tuple(x + offset for x in first_x)

    first_ticks = "".join(
        f'<line x1="{x}" y1="24" x2="{x}" y2="36" stroke="{ACCENT_2}" stroke-width="2.5"/>'
        for x in first_x
    )
    second_ticks = "".join(
        f'<line x1="{x:.0f}" y1="74" x2="{x:.0f}" y2="86" stroke="{ACCENT}" stroke-width="2.5"/>'
        for x in second_x
    )

    inner = (
        f'<text x="8" y="14" font-size="10" fill="{ACCENT_2}">{first_stream}</text>'
        f'<line x1="8" y1="30" x2="312" y2="30" stroke="{GRIDLINE}" stroke-width="1.5"/>'
        f'{first_ticks}'
        f'<text x="8" y="64" font-size="10" fill="{ACCENT}">{second_stream}</text>'
        f'<line x1="8" y1="80" x2="312" y2="80" stroke="{GRIDLINE}" stroke-width="1.5"/>'
        f'{second_ticks}'
        f'<line x1="{first_x[0]}" y1="36" x2="{second_x[0]:.0f}" y2="74" '
        f'stroke="{INK_MUTED}" stroke-width="1" stroke-dasharray="3,2"/>'
        f'<text x="{(first_x[0] + second_x[0]) / 2:.0f}" y="55" font-size="9" '
        f'fill="{INK_MUTED}" text-anchor="middle">'
        f'{misalignment_minutes:.0f} min</text>'
    )

    return visuals.figure(
        inner,
        width=320,
        height=100,
        label=(
            f"{first_stream} and {second_stream} on two timelines, "
            f"{misalignment_minutes:.0f} minutes apart."
        ),
    )

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

# Which library measures this page's simulation can actually model.
#
# The planner is where measures are chosen; this says which of those
# choices the chronic-pain simulation has a channel for. Heart rate and
# HRV map to the same key deliberately, because simulate.py collapses
# every wearable channel into one physio_signal column, and pretending
# otherwise here would imply two columns the simulation does not produce.
#
# A selection absent from this mapping is not dropped silently. It is
# named as something the simulation cannot model, which is a limit of
# this worked example rather than of the measure.
_LIBRARY_TO_SIMULATION = {
    "Rating scale, repeated in daily life": "pain_rating",
    "Body map or pain drawing": "body_map",
    "Electrodermal activity": "eda",
    "Heart rate and heart-rate variability": "heart_rate",
}

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
        upload_profile = profile_dataframe(upload_df)

        # What is in the file, before what it might be for. The
        # suggestions below are drawn from the same column shapes the
        # portrait shows, so a reader can see why a workflow was named
        # rather than take the naming on trust.
        render_dataset_portrait(
            upload_profile, source_name=uploaded.name, is_sample=False
        )
        render_data_profile(upload_df, profile=upload_profile)

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
    # The simulation's one built-in scenario used to be disclosed here,
    # at the top of the generalized path, where it read as a statement
    # that nothing a researcher entered would be used. It is disclosed
    # where it applies now: when the worked example that owns it is
    # loaded.
    st.caption(
        "Build a study and explore how design choices shape the evidence, "
        "before any data exists. It never scores a design as good or bad, "
        "only what it does and does not support."
    )
    st.caption(
        "This walkthrough is OpenMeasure's own structural logic, not "
        "drawn from a published design-selection framework. For a "
        "broader existing method-selection tool, see the Co-Creation "
        "Methods Navigator on the Research Resources page."
    )
    st.page_link(
        "pages/Resources.py",
        label="Open Research Resources",
        icon=":material/collections_bookmark:",
    )

    STAGE_QUESTION = 0
    STAGE_MEASURES = 1
    STAGE_TIMING = 2
    STAGE_SIMULATE = 3
    STAGE_IMPLICATIONS = 4

    # What the planner has assembled, read before any stage renders.
    #
    # It has to be available to every stage rather than only to the one
    # that built it, because the timing, the simulation and the record
    # all describe it, and in a workspace those are separate screens.
    # Session state carries it, so nothing recomputes.
    # The assembled study, rebuilt from the two things a reader actually
    # edits: the concept table and one measure multiselect per concept.
    #
    # Derived here rather than inside the Measures stage, which is where
    # it used to be written. That made the study a side effect of one
    # screen having rendered: loading the worked example on the Question
    # stage and going straight to Timing left planner_study unset, the
    # timing gate unsatisfied, and three stages blank. Derived state that
    # later stages depend on has to exist whichever stage is on screen.
    #
    # The Measures stage still owns editing it. This only reconstitutes
    # what those widgets last held.
    def _assemble_from_session() -> assembly.AssembledStudy:
        concepts = tuple(
            assembly.Concept(
                str(row.get("Concept", "")).strip(),
                str(row.get("What sort of thing is it?", "")),
            )
            for row in st.session_state.get("planner_concepts", [])
            if str(row.get("Concept", "")).strip()
        )

        chosen: list[str] = []
        for concept in concepts:
            chosen.extend(
                st.session_state.get(f"planner_measures_{concept.name}", [])
            )

        return assembly.AssembledStudy(
            concepts=concepts,
            selected_measures=tuple(dict.fromkeys(chosen)),
        )

    planner_state = st.session_state.get("planner_study")

    if planner_state is None or not planner_state.concepts:
        rebuilt = _assemble_from_session()
        if rebuilt.concepts:
            st.session_state["planner_study"] = rebuilt
            planner_state = rebuilt
    planner_measures = (
        tuple(planner_state.selected_measures) if planner_state is not None else ()
    )
    planner_concepts = (
        tuple(concept.name for concept in planner_state.concepts)
        if planner_state is not None
        else ()
    )

    # Whether the worked example is loaded decides what the timing,
    # simulation and record stages show, and in a workspace each of those
    # is a separate run that never executes the question stage where the
    # button lives. Read before the gates, which now test it.
    worked_example_loaded = bool(
        st.session_state.get("worked_example_loaded", False)
    )

    # Gates. The simulation needs something it can model; the rest is
    # reading and choosing, which nothing has to unlock.
    #
    # Population and setting are optional rather than required. A page
    # that treated them the same as a measure would make a researcher
    # invent a population to reach the next screen, which is the opposite
    # of what this toolkit is for.
    design_workspace = StageWorkspace(
        session_key="design",
        stages=(
            Stage("question", "Question"),
            Stage("measures", "Measures"),
            Stage("timing", "Timing"),
            Stage("simulate", "Simulation"),
            Stage("record", "Design Record"),
        ),
        gates={
            # Timing describes whatever was assembled, so a measure is
            # all it needs.
            "timing": Gate(
                satisfied=bool(planner_measures),
                requirement="Assemble at least one measure to continue",
            ),

            "question": Gate(
                satisfied=bool(st.session_state.get("rq_population")),
                requirement="Population not stated",
                optional=True,
            ),
        },
    )


    # Which assembled measures this page's simulation has a channel for,
    # and which it does not. Derived here rather than in the stage that
    # displays it, because timing, the simulation and the record all
    # describe the same selection and each is a separate run.
    assembled_measures = list(
        dict.fromkeys(
            _LIBRARY_TO_SIMULATION[name]
            for name in planner_measures
            if name in _LIBRARY_TO_SIMULATION
        )
    )
    unsimulated = [
        name for name in planner_measures if name not in _LIBRARY_TO_SIMULATION
    ]

    # The generic types the deterministic rule engine reasons over.
    # Derived from the same selection, so the record and the inspections
    # describe what was assembled whichever stage is on screen.
    measurement_types = tuple(
        sorted(
            {
                item["generic_type"]
                for item in _MEASURE_GALLERY
                if item["key"] in assembled_measures
            }
        )
    )

    # Timing choices are made in one stage and used in the next, and a
    # widget that is not rendered has no value, so they are held by name.
    n_participants = design_workspace.kept("n_participants")
    observations_per_day = design_workspace.kept("observations_per_day")
    duration_days = design_workspace.kept("duration_days")
    temporal_misalignment_minutes = design_workspace.kept("misalignment", 10)

    # What the researcher typed about their own question, held by name so
    # the record can print it from a stage that never renders the fields.
    entered = design_workspace.kept("entered", {})
    hypothesis = entered.get("hypothesis", "")
    population = entered.get("population", "")
    exposure = entered.get("exposure", "")
    outcomes = entered.get("outcomes", "")
    setting = entered.get("setting", "")

    design_stage = design_workspace.render_rail()
    design_workspace.render_review_notice()

    st.divider()

    # -------------------------------------------------------------
    # 0. Research question
    # -------------------------------------------------------------

    if design_stage == STAGE_QUESTION:
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
            "study_compares_subgroups": False,
        }

        # Loading the example fills the same workspace a researcher plans in,
        # rather than switching the page into a different mode. Everything it
        # sets can then be changed, which is what makes it teach the planner
        # instead of replacing it.
        worked = examples.CHRONIC_PAIN

        rq_button_cols = st.columns([2, 1, 3])
        with rq_button_cols[0]:
            if st.button(f"Load worked example: {worked.title}"):
                for key, value in PAIN_EXAMPLE.items():
                    st.session_state[key] = value
                st.session_state["planner_concepts"] = [
                    {
                        "Concept": concept.name,
                        "What sort of thing is it?": concept.kind,
                    }
                    for concept in worked.study.concepts
                ]
                # The editor keeps its own copy once touched, so it has to be
                # dropped for the rows above to take.
                st.session_state.pop("planner_concept_editor", None)
                for concept in worked.study.concepts:
                    st.session_state[f"planner_measures_{concept.name}"] = [
                        name
                        for name in worked.study.selected_measures
                        if concept.kind in ontology.get_measure(name).observes
                    ]
                st.session_state["worked_example_loaded"] = True
                st.rerun()
        with rq_button_cols[1]:
            if st.button("Clear"):
                for key in PAIN_EXAMPLE:
                    st.session_state.pop(key, None)
                for key in list(st.session_state):
                    if str(key).startswith("planner_"):
                        st.session_state.pop(key, None)
                st.session_state.pop("worked_example_loaded", None)
                # And what the simulation produced. Without this the
                # Design Record kept reporting a coupling estimate for
                # thirty participants directly under "Measures assembled
                # in the example: none chosen".
                for key in list(st.session_state):
                    if str(key).startswith("design_kept_"):
                        st.session_state.pop(key, None)
                st.rerun()

        if worked_example_loaded:
            st.caption(
                f"The {worked.title} worked example is loaded. Everything below "
                "is its study, and every part of it can be changed. Its "
                "simulation models this one scenario, so the simulated results "
                "further down describe it rather than a study you describe "
                "yourself."
            )

        # Restored rather than re-typed. These widgets are gone from
        # session state while a later stage is open, so coming back would
        # otherwise find an empty form and a question that survived only
        # in the Design Record.
        for field, remembered in entered.items():
            slot = f"rq_{field}"
            if remembered and slot not in st.session_state:
                st.session_state[slot] = remembered

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
            "Record."
        )

        design_workspace.keep(
            "entered",
            {
                "hypothesis": hypothesis,
                "population": population,
                "exposure": exposure,
                "outcomes": outcomes,
                "setting": setting,
            },
        )

        design_workspace.record_input(
            "question",
            [hypothesis, population, exposure, outcomes, setting],
            affects=("record",),
            label="Research question",
        )

        # -------------------------------------------------------------
        # Concepts, and how they could be observed
        # -------------------------------------------------------------
        #
        # This stage is the answer to the cold-test failure that produced it:
        # a researcher asking about mental models in implementation research
        # was offered pain ratings, body maps and electrodermal activity,
        # because the measures on this page belonged to one worked example
        # and the example was the planner.
        #
        # The join here is the kind of thing a concept is, never the field it
        # belongs to. Naming "mental-model structure" as something a person
        # organizes in their head surfaces card sorting and causal mapping;
        # naming "autonomic arousal" as a bodily process surfaces EDA. Same
        # component, same library, different study.
        #
        # Nothing is generated from the wording of the question. The library
        # in modules/research_design/core/ontology.py is curated and finite,
        # and a measure it lacks is a gap to add deliberately.

        # -------------------------------------------------------------
    # Drawing the assembled study
    # -------------------------------------------------------------
    #
    # Which picture gets drawn follows from what was assembled, because
    # the useful view of three modalities on one occasion and of one
    # measure across eight weeks are different pictures and a single
    # diagram serving both serves neither. More than one can apply: a
    # clustered trial measured repeatedly has a nesting and a timeline,
    # and both are true of it.
    #
    # These take the study, never a domain. A study of mental models and
    # a study of chronic pain with the same structure get the same
    # drawing, which is the point of having an ontology underneath.

    def _svg_text(x, y, value, *, size=13, color=visuals.INK_MUTED, anchor_at="start"):
        return (
            f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" '
            f'fill="{color}" text-anchor="{anchor_at}">{value}</text>'
        )

    def _measurement_svg(study):
        """Concepts on the left, what reaches them on the right."""
        rows = study.coverage
        height = max(120, 46 + len(rows) * 52)
        inner = _svg_text(0, 20, "Concept", size=12) + _svg_text(
            300, 20, "Observed by", size=12
        )

        for index, row in enumerate(rows):
            y = 52 + index * 52
            established = row.observed
            color = visuals.ACCENT if established else visuals.INK_MUTED
            dash = (
                ""
                if established
                else f' stroke-dasharray="{visuals.DASH_UNESTABLISHED}"'
            )
            inner += (
                f'<circle cx="8" cy="{y - 5:.0f}" r="5" fill="none" '
                f'stroke="{color}" stroke-width="1.5"{dash}/>'
                + (
                    f'<circle cx="8" cy="{y - 5:.0f}" r="2.5" fill="{color}"/>'
                    if established
                    else ""
                )
                + _svg_text(24, y, row.concept.name, size=14)
                + visuals.arrow(
                    250, y - 5, 292, y - 5, established=established
                )
                + _svg_text(
                    300,
                    y,
                    ", ".join(row.measures) if row.measures else "nothing selected",
                    size=13,
                    color=color,
                )
            )

        return visuals.figure(
            inner,
            width=640,
            height=height,
            label=(
                "Each concept this study names, and which selected measures "
                "reach it. A hollow circle and a dashed arrow mark a concept "
                "nothing selected observes."
            ),
        )

    def _convergence_svg(study):
        """Several kinds of evidence meeting at the same unit."""
        modalities = study.modalities
        width = 640.0
        column = width / len(modalities)
        inner = ""

        for index, modality in enumerate(modalities):
            x = column * (index + 0.5)
            inner += _svg_text(x, 24, modality, size=13, anchor_at="middle")
            inner += (
                f'<path d="M {x:.0f} 34 V 60 H {width / 2:.0f} V 84" '
                f'fill="none" stroke="{visuals.ACCENT}" stroke-width="1"/>'
            )

        inner += (
            f'<rect x="{width / 2 - 90:.0f}" y="84" width="180" height="44" '
            f'rx="5" fill="none" stroke="{visuals.ACCENT}" stroke-width="1.5"/>'
            + _svg_text(
                width / 2, 111, "the same unit", size=14, anchor_at="middle"
            )
            + _svg_text(
                width / 2,
                156,
                "Agreement between them is evidence; it is not independent "
                "evidence",
                size=12,
                anchor_at="middle",
            )
        )

        return visuals.figure(
            inner,
            width=width,
            height=176,
            label=(
                "This study would produce "
                + ", ".join(modalities)
                + " evidence about the same unit."
            ),
        )

    def _timeline_svg(study):
        """The same measures, on more than one occasion."""
        shown = min(study.occasions, 8)
        spacing = 600.0 / max(shown - 1, 1)
        inner = _svg_text(0, 20, f"{study.occasions} occasions", size=13)
        inner += (
            f'<line x1="8" y1="60" x2="628" y2="60" '
            f'stroke="{visuals.GRIDLINE}" stroke-width="1"/>'
        )

        for index in range(shown):
            x = 8 + index * spacing
            inner += (
                f'<circle cx="{x:.0f}" cy="60" r="5" '
                f'fill="{visuals.ACCENT}" fill-opacity="0.8"/>'
            )

        if study.occasions > shown:
            inner += _svg_text(628, 44, "...", size=13, anchor_at="end")

        inner += _svg_text(
            0,
            92,
            "Repeated on the same units, so observations within a unit are "
            "not independent",
            size=12,
        )

        return visuals.figure(
            inner,
            width=640,
            height=110,
            label=(
                f"The selected measures repeated across {study.occasions} "
                "occasions on the same units."
            ),
        )

    def _nesting_svg(study):
        """Units inside whatever they were sampled through."""
        inner = (
            f'<rect x="0" y="14" width="200" height="46" rx="5" fill="none" '
            f'stroke="{visuals.ACCENT}" stroke-width="1.5"/>'
            + _svg_text(14, 42, study.nesting, size=14)
            + f'<path d="M 100 60 V 78 H 40 V 96" fill="none" '
            f'stroke="{visuals.INK_MUTED}" stroke-width="1"/>'
            + f'<path d="M 100 78 H 160 V 96" fill="none" '
            f'stroke="{visuals.INK_MUTED}" stroke-width="1"/>'
            + visuals.unit_cluster(40, 116, visuals.ACCENT, count=5, radius=6)
            + visuals.unit_cluster(160, 116, visuals.ACCENT, count=5, radius=6)
            + _svg_text(230, 120, "units, sampled through the level above", size=13)
            + _svg_text(
                0,
                168,
                "Units inside one group are more alike than units drawn at "
                "random",
                size=12,
            )
        )

        return visuals.figure(
            inner,
            width=640,
            height=186,
            label=(
                f"Units nested within {study.nesting}, so units inside one "
                "are more alike than units drawn at random."
            ),
        )

    def _arms_svg(study):
        """What this study compares."""
        arms = study.arms
        column = 640.0 / len(arms)
        inner = _svg_text(0, 20, "Compared arms", size=12)
        inner += (
            f'<path d="M 320 30 V 48" fill="none" '
            f'stroke="{visuals.INK_MUTED}" stroke-width="1"/>'
        )

        for index, arm in enumerate(arms):
            x = column * (index + 0.5)
            colour = visuals.ACCENT_2 if index == 0 else visuals.ACCENT
            inner += (
                f'<path d="M 320 48 H {x:.0f} V 66" fill="none" '
                f'stroke="{visuals.INK_MUTED}" stroke-width="1"/>'
                + f'<rect x="{x - 90:.0f}" y="66" width="180" height="42" '
                f'rx="5" fill="none" stroke="{colour}" stroke-width="1.5"/>'
                + _svg_text(x, 92, arm, size=14, anchor_at="middle", color=colour)
                + visuals.unit_cluster(x, 140, colour, count=5, radius=6)
            )

        return visuals.figure(
            inner,
            width=640,
            height=180,
            label="This study compares " + " and ".join(arms) + ".",
        )

    _VIEW_BUILDERS = {
        assembly.VIEW_MEASUREMENT: _measurement_svg,
        assembly.VIEW_CONVERGENCE: _convergence_svg,
        assembly.VIEW_TIMELINE: _timeline_svg,
        assembly.VIEW_NESTING: _nesting_svg,
        assembly.VIEW_ARMS: _arms_svg,
    }

    if design_stage == STAGE_MEASURES:
        section_header(
            "Concepts And How To Observe Them",
            "What has to be observed, and what could observe it",
        )

        st.caption(
            "Name each thing your study has to observe, and say what sort of "
            "thing it is. The sort is what decides which measurement "
            "approaches are even applicable; your field does not restrict "
            "them, so a study can draw on several kinds of evidence at once."
        )

        # Where the mapping comes from and how far it reaches, stated
        # before anything is offered from it. A researcher deciding how
        # much to trust this step is entitled to the number, and half of
        # the concepts OpenMeasure recognises do not have a curated list.
        curated_concepts, known_concepts = lexicon.curated_coverage()
        with st.expander("Where these suggestions come from"):
            st.caption(
                f"OpenMeasure recognises {known_concepts} concepts and has a "
                f"hand-curated measure list for {curated_concepts} of them. "
                "The lists were written by the maintainer against the "
                "measurement literature for each construct; they are not "
                "derived from a published crosswalk, and none of them is a "
                "construct-validity claim."
            )
            st.caption(
                "A concept without a curated list still reaches something: "
                "every measure whose own record says it can observe that "
                "kind of thing. That is a category join rather than a "
                "recommendation, it is labelled as one where it appears, "
                "and some of what it offers will be wrong for your "
                "construct."
            )

        # Recognised, never generated. A phrase in OpenMeasure's list is
        # matched and a phrase outside it is not guessed at, so a construct
        # nobody chose cannot enter a study looking as though they had.
        # Adding a suggestion is a click, which is the confirmation step: the
        # question can suggest concepts, and only confirmed concepts enter
        # the study.
        question_text = " ".join(
            part for part in (hypothesis, outcomes, exposure) if part
        )
        suggested = lexicon.suggest_concepts(question_text) if question_text else ()

        if question_text:
            st.markdown("**Concepts recognised in your question**")

            if suggested:
                already = {
                    str(row.get("Concept", "")).strip()
                    for row in st.session_state.get("planner_concepts", [])
                }
                suggestion_cols = st.columns(min(len(suggested), 3))

                for index, suggestion in enumerate(suggested):
                    with suggestion_cols[index % len(suggestion_cols)]:
                        st.caption(f"from \"{suggestion.matched_phrase}\"")
                        if st.button(
                            f"Add {suggestion.concept}",
                            key=f"add_concept_{suggestion.concept}",
                            disabled=suggestion.concept in already,
                        ):
                            rows = list(st.session_state.get("planner_concepts", []))
                            rows.append(
                                {
                                    "Concept": suggestion.concept,
                                    "What sort of thing is it?": suggestion.kind,
                                }
                            )
                            st.session_state["planner_concepts"] = rows
                            st.session_state.pop("planner_concept_editor", None)
                            st.rerun()
                        st.caption(suggestion.kind)
            else:
                st.caption(lexicon.NOTHING_RECOGNISED)

        concept_rows = st.data_editor(
            pd.DataFrame(
                st.session_state.get(
                    "planner_concepts",
                    [{"Concept": "", "What sort of thing is it?": ontology.CONCEPT_KINDS[0]}],
                )
            ),
            column_config={
                "What sort of thing is it?": st.column_config.SelectboxColumn(
                    options=list(ontology.CONCEPT_KINDS), required=True
                )
            },
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            key="planner_concept_editor",
        )

        named_concepts = tuple(
            assembly.Concept(str(row["Concept"]).strip(), str(row["What sort of thing is it?"]))
            for _, row in concept_rows.iterrows()
            if str(row.get("Concept", "")).strip()
        )

        selected_measure_names: list[str] = []

        if named_concepts:
            # The visual explorer, for every concept rather than for one
            # worked example. Choose a measure, see the shape of what it
            # produces, read what it captures and what it costs. The
            # chronic-pain gallery had this interaction and it was
            # pain-specific, so every other study got a list of names.
            #
            # Candidates are narrowed to the concept where OpenMeasure has a
            # narrowed list, and fall back to everything its kind can observe
            # where it does not. Which of the two a reader is looking at is
            # said, because a broad list is a different answer rather than a
            # worse one.
            def render_measure_detail(measure) -> None:
                """One measure in full, wherever it is being listed."""
                st.markdown(f"**{measure.name}**")
                st.caption(
                    f"Captures {measure.captures.lower()}. Produces "
                    f"{measure.produces.lower()}."
                )
                st.caption(f"Modality: {measure.modality}")
                st.caption(f"Burden: {measure.burden}")
                st.caption(f"Limitation: {measure.limitation}")
                st.caption(
                    f"{measure.documented_as}. Search: "
                    f"`{measure.search_terms}`"
                )

            for concept in named_concepts:
                candidates = lexicon.measures_for_concept(
                    concept.name, concept.kind
                )

                st.markdown(f"**{concept.name}**")
                st.caption(concept.kind)

                # The two cases are drawn differently rather than
                # labelled differently.
                #
                # A row of pictograph cards reads as a recommendation
                # whatever the caption underneath says, and for a concept
                # nobody has curated the list is a category join: every
                # measure whose own record says it can observe this kind
                # of thing. That offered functional neuroimaging as a way
                # to measure violent behaviour among students, because
                # violence is something a person does. It can study
                # neural processes associated with a behaviour; it does
                # not measure the behaviour.
                #
                # So the gallery is for curated matches. A category join
                # arrives as a warning first and a collapsed list second,
                # still selectable, because a researcher who knows
                # structured observation is the right answer here has to
                # be able to choose it.
                if candidates.validated_for_the_concept:
                    st.caption(
                        f"**{candidates.basis}.** {candidates.note}"
                    )

                    columns = st.columns(min(len(candidates.measures), 4) or 1)
                    for index, measure in enumerate(candidates.measures):
                        with columns[index % len(columns)]:
                            st.markdown(
                                measure_visual_svg(measure),
                                unsafe_allow_html=True,
                            )
                            st.caption(f"**{measure.name}**")
                            # The drawing above is of the artifact, so
                            # name it: a reader comparing an ECG trace
                            # against a skin-conductance trace should not
                            # have to guess which one they are looking at.
                            st.caption(
                                f"{measure.produces}, {measure.visual_type}"
                            )
                            st.caption(measure.modality)
                else:
                    st.warning(
                        f"**{candidates.basis}.** {candidates.note}"
                    )

                chosen = st.multiselect(
                    f"Ways to observe {concept.name}",
                    options=[
                        measure.name for measure in candidates.measures
                    ],
                    key=f"planner_measures_{concept.name}",
                    label_visibility="collapsed",
                    placeholder=f"Add measures for {concept.name}",
                )
                selected_measure_names.extend(chosen)

                # Counted rather than assumed plural. Several concepts
                # have exactly one candidate, and "Inspect these 1
                # measures" is the kind of seam that makes a careful
                # page look careless.
                count = len(candidates.measures)
                if candidates.validated_for_the_concept:
                    heading = (
                        f"Inspect this measure"
                        if count == 1
                        else f"Inspect these {count} measures"
                    )
                else:
                    heading = (
                        "1 category-level possibility, and what it would "
                        "actually capture"
                        if count == 1
                        else f"{count} category-level possibilities, and "
                        "what each would actually capture"
                    )

                with st.expander(heading):
                    if not candidates.validated_for_the_concept:
                        st.caption(
                            "Read what each captures against what you mean "
                            "by this concept. That comparison is the "
                            "decision, and OpenMeasure has not made it."
                        )

                    for measure in candidates.measures:
                        if not candidates.validated_for_the_concept:
                            st.markdown(
                                measure_visual_svg(measure),
                                unsafe_allow_html=True,
                            )
                        render_measure_detail(measure)
        elif suggested:
            # The case that read as broken: a recognised concept sitting
            # in a button nobody had said to press, under a message
            # telling the reader to name one.
            st.info(
                "Add one of the concepts recognised above, or type your "
                "own into the table, to see how it could be observed."
            )
        else:
            st.info(
                "Complete the Question stage to get concept suggestions, or "
                "type a concept into the table above. Either way, nothing "
                "enters the study until you add it."
            )

        planner_study = assembly.AssembledStudy(
            concepts=named_concepts,
            selected_measures=tuple(dict.fromkeys(selected_measure_names)),
        )
        st.session_state["planner_study"] = planner_study

        # The measures reach timing, the simulation and the record: all
        # three describe what was assembled. The concepts reach the
        # record alone, since naming one more thing to observe does not
        # change how often anything is sampled.
        # The gates are drawn above this, so a measure picked just now
        # was not available to them.
        design_workspace.record_gate_input(
            "measures", list(planner_study.selected_measures)
        )

        design_workspace.record_input(
            "measures",
            list(planner_study.selected_measures),
            affects=("timing", "simulate", "record"),
            label="Measures",
        )
        design_workspace.record_input(
            "concepts",
            [concept.name for concept in planner_study.concepts],
            affects=("record",),
            label="Concepts",
        )

        if planner_study.selected_measures:
            st.markdown("**Your study, as assembled**")

            for view in planner_study.applicable_views:
                st.caption(view)
                st.markdown(
                    _VIEW_BUILDERS[view](planner_study), unsafe_allow_html=True
                )

            st.caption(
                f"Kinds of evidence: {', '.join(planner_study.modalities)}. "
                f"Shape: {planner_study.shape}."
            )

            if planner_study.unobserved:
                inspect_note(
                    "The concepts nothing selected reaches. They are not "
                    "errors; they are what this design would leave to another "
                    "study, and stating them is the point of listing concepts "
                    "separately from measures."
                )


    # -------------------------------------------------------------
    # 1. Explore & assemble measures
    # -------------------------------------------------------------


    if design_stage == STAGE_MEASURES and worked_example_loaded:
        # Illustrations, not assembly. Assembly happens once, in the
        # planner above; this stage draws the example's own measures in
        # more detail than the shared primitives do, which is what made it
        # worth keeping rather than replacing.
        section_header(
            "A Closer Look At These Measures",
            "The worked example's own measures, drawn in detail. Choosing "
            "what the study includes happens in the planner above.",
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
        # Named as the example's, at the point of choosing rather than
        # only in the record. These five measures and the simulation
        # underneath them are the chronic-pain scenario; calling the
        # selection "this study" made them read as the reader's own,
        # whatever question they had typed above.
        # One selector, upstream. This stage used to hold a second
        # measure picker of its own, so loading the worked example showed
        # the generalized explorer and then a parallel list that did the
        # same job with different names. The simulation reads what was
        # assembled in the planner instead.
        st.markdown("**What The Simulation Will Model**")

        st.caption(
            "Taken from the measures you assembled above. Change them "
            "there and this changes with them."
        )

        if unsimulated:
            st.caption(
                "Selected and not modelled here: "
                + ", ".join(unsimulated)
                + ". This simulation has channels for the chronic-pain "
                "example's measures only, which is a limit of the example "
                "rather than of those measures."
            )

        if not assembled_measures:
            st.info(
                "None of the measures assembled above can be modelled by "
                "this simulation. Add one of the worked example's own "
                "measures to run it, or read the stages below as a "
                "description of what it would do."
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



    # -------------------------------------------------------------
    # 2. Timing & synchronization
    # -------------------------------------------------------------

    if design_stage == STAGE_TIMING:
        section_header(
            "Timing & Synchronization",
            "When are measures collected, and how closely aligned are they?",
        )

        st.caption(
            "Two measures can each be accurate and still appear unrelated "
            "if they capture different moments. When what you are measuring "
            "can change quickly, poor synchronization can weaken or distort "
            "the relationship you observe."
        )

        sample_cols = st.columns(3)
        with sample_cols[0]:
            n_participants = st.slider(
                "Number of participants",
                5,
                100,
                design_workspace.kept("n_participants", 30),
            )
        with sample_cols[1]:
            observations_per_day = st.slider(
                "Measurement frequency (per day)",
                1,
                10,
                design_workspace.kept("observations_per_day", 4),
            )
        with sample_cols[2]:
            duration_days = st.slider(
                "Study duration (days)",
                3,
                30,
                design_workspace.kept("duration_days", 7),
            )

        design_workspace.keep("n_participants", n_participants)
        design_workspace.keep("observations_per_day", observations_per_day)
        design_workspace.keep("duration_days", duration_days)

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

        # Named from what was assembled rather than from the worked
        # example. Which two streams are compared is the reader's
        # selection, and a diagram captioned "Pain report" and "Wearable
        # signal" was describing someone else's study.
        #
        # Continuously sampled measures are the ones this question bites
        # hardest for: their timestamps come from a device clock while an
        # episodic report comes from whenever a person got to it. The
        # animated visual types are exactly that set, which is why the
        # split is read from there rather than from a second list that
        # could drift out of step with it.
        continuous = [
            name
            for name in planner_measures
            if ontology.get_measure(name).visual_type in ANIMATED_TYPES
        ]
        episodic = [name for name in planner_measures if name not in continuous]
        repeatedly_sampled = [
            name
            for name in planner_measures
            if ontology.get_measure(name).visual_type in _REPEATEDLY_SAMPLED
        ]

        # Two conditions, not one. A single measure has nothing to align
        # against, and two measures that each arrive once have no
        # minute-scale gap between them either. A study should not
        # inherit the misalignment concept just because this stage
        # exists.
        alignment_applies = (
            len(planner_measures) >= 2 and bool(repeatedly_sampled)
        )

        st.markdown("**How Are the Streams Aligned?**")

        if len(planner_measures) < 2:
            st.info(
                "Alignment is a question about two or more streams, and this "
                "study assembles one. Add another measure to inspect how "
                "their timing affects the comparison between them."
            )
            temporal_misalignment_minutes = design_workspace.kept(
                "misalignment", 10
            )
        elif not alignment_applies:
            st.info(
                "As OpenMeasure describes them, each measure here produces "
                "one reading per participant, so there is no minute-scale "
                "gap between them to inspect. Correspondence between them "
                "is a question about which period each one covers."
            )
            temporal_misalignment_minutes = design_workspace.kept(
                "misalignment", 10
            )
        else:
            first_stream = (episodic or continuous)[0]
            second_stream = (continuous or episodic)[-1]

            # The names go in the caption rather than the slider label.
            # Two library names contain a comma, and "between rating
            # scale, repeated in daily life and electrodermal activity"
            # loses the boundary between the two streams it is naming.
            temporal_misalignment_minutes = st.slider(
                "Temporal misalignment (minutes)",
                0,
                60,
                design_workspace.kept("misalignment", 10),
            )

            st.caption(
                "**Temporal misalignment** is the gap between when "
                f"**{first_stream}** is logged and when "
                f"**{second_stream}** is recorded. In the next stage you "
                "can change this gap to see how timing affects the apparent "
                "relationship between the two streams."
            )

            if continuous and episodic:
                st.caption(
                    f"**{second_stream}** is sampled continuously and "
                    f"**{first_stream}** is not, so the two carry "
                    "timestamps from different clocks. That is where this "
                    "gap does the most damage to a comparison."
                )
            elif not continuous:
                st.caption(
                    "Both streams here are episodic, so alignment is a "
                    "question about which occasions correspond rather than "
                    "about a device clock drifting from a person's report."
                )

        design_workspace.keep("misalignment", temporal_misalignment_minutes)

        # The timing choices reach the simulation and the record, which
        # both report them.
        design_workspace.record_input(
            "timing",
            [
                n_participants,
                observations_per_day,
                duration_days,
                temporal_misalignment_minutes,
            ],
            affects=("simulate", "record"),
            label="Timing",
        )
        if alignment_applies:
            st.markdown(
                _synchronization_svg(
                    float(temporal_misalignment_minutes),
                    first_stream=_short_stream_label(first_stream),
                    second_stream=_short_stream_label(second_stream),
                ),
                unsafe_allow_html=True,
            )
            st.caption(
                "Larger values make each reading a noisier stand-in for "
                "what the other stream was doing at that moment."
            )

        inspect_note(
            "Whether the measurements are close enough in time to support "
            "the comparison you want to make."
        )


    # -------------------------------------------------------------
    # 3. Simulate the design
    # -------------------------------------------------------------

    # The simulation's outputs, held across the boundary between the
    # stage that produces them and the record that reports them. They are
    # frozen dataclasses, so session state carries them as they are.
    assumptions: DesignAssumptions | None = design_workspace.kept("assumptions")
    study = design_workspace.kept("study")
    estimate = design_workspace.kept("estimate")

    if design_stage == STAGE_SIMULATE:
        section_header(
            "Explore With a Worked Simulation",
            "How design conditions change the evidence a study ends up with",
        )

        # What a simulation is for, before any slider. A reader who meets
        # five parameters and an estimate first has to reverse-engineer
        # the lesson from the controls, and most will read them as
        # statistical settings to get right rather than as conditions to
        # vary.
        st.write(
            "The simulation shows how study-design conditions can change "
            "the evidence you end up observing, even when the underlying "
            "effect is fixed. Change assumptions such as missing "
            "observations, measurement noise, and differences between "
            "participants to see how much information is retained and how "
            "well the simulated analysis recovers the underlying "
            "relationship."
        )

    if (
        design_stage == STAGE_SIMULATE
        and not worked_example_loaded
    ):
        # The generic path stops here, and says so. OpenMeasure can
        # simulate one built-in scenario today; offering a reader's own
        # study to a model built for someone else's would be worse than
        # saying what the limit is.
        st.info(
            "OpenMeasure can simulate one built-in scenario so far, and it "
            "is not a model of the study you assembled above. Load the "
            "worked example on the Question stage to work through what "
            "design conditions do to an estimate."
        )

        inspect_note(
            "Change one assumption at a time and watch how the simulated "
            "observations and estimate respond. The point is not to find a "
            "good result, it is to see which design conditions make the "
            "relationship you are studying easier or harder to detect."
        )

    if (
        design_stage == STAGE_SIMULATE
        and worked_example_loaded
        and n_participants is not None
        # Kept alongside n_participants and used in the same arithmetic,
        # so the guard names them rather than relying on that.
        and observations_per_day is not None
        and duration_days is not None
    ):
        # Labelled as the example it is. These sliders are one scenario's
        # parameters, not OpenMeasure's general simulation model, and a
        # reader who took them for the latter would be reading a pain
        # study's assumptions as the toolkit's own.
        st.markdown("**Worked example: Chronic pain**")
        st.caption(
            "Pain states, pain ratings, physiological signals and the "
            "coupling between them below are this scenario's, and are "
            "illustrative. The sliders are assumptions about what is "
            "*true* in it, not facts a real version would already know: a "
            "real study would need pilot data or published estimates to "
            "set them credibly."
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

        with st.expander("Advanced simulation assumptions"):
            within_person_sd = st.slider(
                "Within-person physiological variability (SD)", 0.0, 2.0, 0.3, step=0.1
            )
            st.caption(
                "How much one person's own readings bounce around, "
                "observation to observation."
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
        design_workspace.keep("assumptions", assumptions)
        design_workspace.keep("study", study)

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

        with st.expander("Participant-level detail"):
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

            st.dataframe(study.data.head(10), width="stretch", hide_index=True)
            st.caption("First 10 simulated rows, out of the retained total above.")

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

        estimate = estimate_coupling_difference(study)
        design_workspace.keep("estimate", estimate)

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
            "Change one assumption at a time and watch what happens to "
            "missingness, the participant-level signals, and the estimated "
            "coupling difference. The point is not to find a good result, "
            "it is to see which design conditions make the underlying "
            "relationship easier or harder to recover."
        )
        st.caption(
            "Here, "
            f"{estimate.n_participants_excluded_insufficient_data} "
            "participants were excluded for insufficient data, against "
            f"{estimate.n_participants_used} used."
        )

        interpretation_note(
            "This is one analysis aligned to this simulated design (a "
            "within-person correlation difference), not a general "
            "measure of the design's quality and not a recommended "
            "analysis for a real version of this study."
        )


    # -------------------------------------------------------------
    # 4. Reveal terminology & implications
    # -------------------------------------------------------------

    if (
        design_stage == STAGE_IMPLICATIONS
        and study is not None
        and estimate is not None
        # measurement_plan_profile reads straight through this one.
        and assumptions is not None
    ):
        section_header("Interpretation", "What you built, named, and what it does and does not support")

        interpretation_note(
            "This is an observational, within-person, repeated-measures "
            "design with multimodal measurements. It can estimate how "
            "pain and physiology vary together within participants, and "
            "whether that association differs by pain state, but it "
            "cannot establish that pain state causes a change in "
            "physiology: confounders such as medication, activity, and "
            "sleep are not controlled for or modeled here."
        )
        caveat(
            "Simulated precision and uncertainty describe this "
            "simulation under these assumptions. They are not a "
            "guarantee about how a real study, even one matching these "
            "settings closely, would actually perform."
        )

        with st.expander("What these terms mean, and how this was built"):
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

        implications(
            "Adherence and temporal misalignment mainly affect how much "
            "usable data survives to be analyzed, between-person "
            "variability and sensor noise mainly affect how uncertain "
            "the estimate is, and pain-state imbalance affects how many "
            "participants have enough of the rarer state to contribute "
            "at all. Changing one does not just move the headline "
            "number, it changes which of these limits binds."
        )

        st.write("**Relevant OpenMeasure methods**")

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

        # Two parts, kept apart on purpose.
        #
        # This record used to open with the reader's own question and
        # then state "Study design (revealed from what you assembled and
        # ran): Observational, within-person, longitudinal" underneath
        # it, followed by the pain example's five measures. None of that
        # was derived from their question. It is the fixed chronic-pain
        # scenario this page simulates, and printing it under their
        # question made those choices read as properties of the study
        # they had designed.
        #
        # The blanks were worse. An unanswered Population printed the
        # pain example's population, so a record could carry a
        # researcher's own question above a stranger's study population
        # with nothing marking the join.
        #
        # A record that merges the two is wrong in a way the reader
        # cannot see, so it does not merge them. Blank stays blank, and
        # everything the example contributes is named as the example's.
        section_header(
            "Design Record",
            "Two parts: what you entered, and what the worked example "
            "contributed",
        )

        not_stated = "not stated"

        # The measurement strategy this researcher assembled, from their
        # own concepts. Rendered through the same function the module
        # tests hold to printing "not established" for anything unset, so
        # nothing from the worked example can reach this half of the
        # record.
        planner_study = st.session_state.get("planner_study")
        if planner_study is not None and planner_study.concepts:
            planner_section = "\n".join(
                assembly.design_record_lines(
                    planner_study,
                    entered={
                        "question": hypothesis,
                        "population": population,
                        "setting": setting,
                    },
                )[4:]
            )
        else:
            planner_section = (
                "Concepts to observe\n"
                f"- {assembly.UNSET}. No concepts were named in "
                "Concepts And How To Observe Them above."
            )

        record_text = f"""OpenMeasure Research Design Record
===================================

This record has two parts and they are different kinds of thing. The
first is what you entered. The second describes the chronic-pain worked
example, a fixed scenario this page simulates to demonstrate the
workflow. Nothing in the second part was derived from your question.

Part 1. What you entered and assembled
--------------------------------------
Research question: {hypothesis or not_stated}
Population: {population or not_stated}
Exposure: {exposure or not_stated}
Outcomes: {outcomes or not_stated}
Setting: {setting or not_stated}

{planner_section}

Part 2. The chronic-pain worked example
----------------------------------------
Everything below belongs to the worked example, not to the question
above. Its structure, measures and suggested analyses would be different
for a different study, and this page does not yet build them from what
you entered.

Structure: observational, within-person, longitudinal, repeated-measures.
Measures assembled in the example: {", ".join(_MEASURE_LABEL_BY_KEY[k] for k in assembled_measures) if assembled_measures else "none chosen"}
Compares demographic subgroups: {"yes" if compares_subgroups else "no"}

Design implications of the example (deterministic, structure-only; things to inspect, not recommendations)
-----------------------------------------------------------------------------------------------------------
{chr(10).join(f"- {i.trigger}: {i.note}" for i in inspections) if inspections else "- No structural triggers fired for the choices above."}

Measurement plan for the example
---------------------------------
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

Analyses the example's structure points to
--------------------------------------------
Time-Series QA (timestamp completeness), then Impact Evaluation
(comparing the physiological signal across pain states). These follow
from the worked example's columns, not from your question.

Limitations
-----------
Observational, not experimental: cannot establish causation. Does not
model medication, activity, sleep, or stress. Simulated precision does
not guarantee real-world performance. Synthetic data, not study
participant data. Part 2 of this record describes the worked example
throughout; a builder that assembles a design from your own question is
not yet built.
"""

        with st.expander("Show design record", expanded=False):
            st.text_area("Design record", record_text, height=400)
            st.download_button(
                "Download design record (.txt)",
                data=record_text,
                file_name="openmeasure_research_design_record.txt",
                mime="text/plain",
            )

    st.divider()

    design_workspace.render_navigation()
    design_workspace.render_optional_gaps()
