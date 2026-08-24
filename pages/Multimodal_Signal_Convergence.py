"""
Multimodal Signal Convergence - a guided research journey, not a workflow.

Teaches a validation question none of the numbered workflows cover: once
a pipeline can combine more than one signal about the same person or
process, does the added interpretive value justify the added privacy,
security, and agency cost of collecting each additional signal? Starts
with a single neurotech signal (EEG) and adds Autonomic, Muscular,
Behavioral, Subjective, Environmental, and Institutional modalities one
at a time, using the same general pipeline shape throughout: Signals ->
Sensors -> Processing -> Inference -> Decision -> Action -> Feedback.

Like the other Research Journeys, this page records nothing to
shared/handoff.py, carries no module_key, and is deliberately not a
numbered page, so it needs no entry in shared/catalog.py.

Core logic (domain-agnostic: any set of signal categories rated on one
gain dimension and three cost dimensions, not specific to neurotech or
any one study) lives in modules/signal_pipeline/core; this file is
presentation only.

Citation-integrity note - read this before trusting any number below:
unlike GAIA, fMRI QC, or HealthRing, this journey is not anchored to one
real, cited study. sample_data/modality_profiles.csv contains no value
any single cited source reports directly. Every interpretive-gain and
cost rating is an illustrative, author-assigned score informed by the
cited literature's directional findings (e.g. "location data is highly
re-identifying," per de Montjoye et al., 2013) - never a number that
literature reports. See modules/signal_pipeline/README.md for the full
discussion and modules/signal_pipeline/sample_data/modality_profiles.csv
for the citation behind every row.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from modules.signal_pipeline.core import modality as modality_core
from modules.signal_pipeline.core import pipeline as pipeline_core
from modules.signal_pipeline.core import tradeoff as tradeoff_core
from shared.data_handling import disclosure_for, render_data_handling_summary
from shared.report import caveat, flagged_item_note, section_header

SAMPLE_DIR = ROOT / "modules" / "signal_pipeline" / "sample_data"

# Validated default palette (dataviz skill, references/palette.md), same
# values used elsewhere in this app (e.g. GAIA_Worked_Example.py).
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
MUTED = "#898781"

# Categorical slots 1-7 in the palette's fixed, validated order, one per
# modality category. "Pipeline" (the shared Processing..Feedback nodes)
# deliberately does not consume an 8th categorical slot: those nodes are
# structural, not a competing identity to be told apart from the seven
# modality colors, so they use the neutral muted ink instead.
CATEGORY_COLORS = {
    "Neural": "#2a78d6",
    "Autonomic": "#eb6834",
    "Muscular": "#1baf7a",
    "Behavioral": "#eda100",
    "Subjective": "#e87ba4",
    "Environmental": "#008300",
    "Institutional": "#4a3aa7",
}

# A second, redundant identity channel alongside color: with up to seven
# categories visible on screen at once (a "scatter"-like diagram, per the
# dataviz skill), color alone does not clear the palette's all-pairs CVD
# floor past three simultaneous series. Shape plus an always-visible text
# label on every node keeps category identity legible without relying on
# hue alone.
CATEGORY_SHAPES = {
    "Neural": "circle",
    "Autonomic": "square",
    "Muscular": "triangle-up",
    "Behavioral": "diamond",
    "Subjective": "triangle-down",
    "Environmental": "cross",
    "Institutional": "triangle-right",
}

# Static pictographs, not the animated kind: this project moved away from
# flickering step animations toward still icons (see GAIA, HealthRing,
# pyfMRIqc). One per pipeline_core.STAGES entry, in that order - what
# each stage means in general, not tied to any one modality.
PIPELINE_STAGE_NOTES = (
    (":material/waves:", "Signals", "The raw physical signal a body or context produces"),
    (":material/sensors:", "Sensors", "What a device actually measures from that signal"),
    (":material/tune:", "Processing", "Cleaning and transforming the raw measurement"),
    (":material/psychology:", "Inference", "Turning the processed signal into an estimate"),
    (":material/rule:", "Decision", "Comparing that estimate against a threshold or rule"),
    (":material/bolt:", "Action", "The system responds based on that decision"),
    (":material/loop:", "Feedback", "The response changes the signal that comes next"),
)

# Anatomy scene layout: an original, simplified biological-illustration
# style (brain / heart / muscle / soundwave icons on a soft body outline),
# not a trace or copy of any commercial illustration library's assets -
# those are copyrighted stock icons this app has no license to reproduce.
# Pixel coordinates in a fixed 320x380 viewBox. Body-sensed categories get
# an icon on the figure itself; non-body-sensed categories (Subjective,
# Environmental, Institutional) are never drawn on the body, since none
# of the three is actually picked up from the body by a sensor - drawing
# one there would misrepresent what the modality is.
SCENE_VIEWBOX = (320, 380)
HEAD_CENTER_PX = (100, 55)
HEAD_RADIUS_PX = 30
ENTRY_POINT_PX = (100, 355)
BODY_ICON_PX = {
    "Neural": (100, 55),
    "Autonomic": (100, 172),
    "Muscular": (25, 210),
    "Behavioral": (100, 124),
}
EXTERNAL_ICON_PX = {
    "Subjective": (258, 68),
    "Environmental": (258, 150),
    "Institutional": (258, 232),
}
# Local (0-32 box) canonical icon paths - hand-authored here, simplified
# and stylized rather than anatomically literal, in the spirit of a clean
# biological-illustration look. fill takes the category's palette color;
# stroke_light is a near-white line for contrast on filled shapes.
_ICON_PATHS = {
    "Neural": (
        '<path d="M16 4C10 4 6 8 6 13C4 14 3 16 3 18C3 21 5 23 8 23'
        "C8 26 11 28 14 28C15 28 16 27 16 27C16 27 17 28 18 28"
        "C21 28 24 26 24 23C27 23 29 21 29 18C29 16 28 14 26 13"
        'C26 8 22 4 16 4Z" fill="{fill}"/>'
        '<path d="M16 6L16 26M11 9C13 10 13 13 11 15M21 9C19 10 19 13 21 15" '
        'stroke="{stroke_light}" stroke-width="1.2" fill="none" stroke-linecap="round"/>'
    ),
    "Autonomic": (
        '<path d="M16 29C16 29 3 20 3 11C3 6 6 3 10 3C13 3 15 5 16 7'
        "C17 5 19 3 22 3C26 3 29 6 29 11C29 20 16 29 16 29"
        'Z" fill="{fill}"/>'
    ),
    "Muscular": (
        '<path d="M7 27C4 20 5 11 12 7C17 4 24 6 27 11C30 16 28 22 24 25'
        "C23 21 20 22 19 25C17 22 15 23 13 26C11 23 9 25 7 27"
        'Z" fill="{fill}"/>'
    ),
    "Behavioral": (
        '<g class="sound-bar" style="animation-delay:0s">'
        '<rect x="9" y="14" width="4" height="9" rx="2" fill="{fill}"/></g>'
        '<g class="sound-bar" style="animation-delay:0.15s">'
        '<rect x="15" y="6" width="4" height="21" rx="2" fill="{fill}"/></g>'
        '<g class="sound-bar" style="animation-delay:0.3s">'
        '<rect x="21" y="11" width="4" height="13" rx="2" fill="{fill}"/></g>'
    ),
    "Subjective": (
        '<path d="M4 5H26C27 5 28 6 28 7V19C28 20 27 21 26 21H13L7 27V21H4'
        "C3 21 2 20 2 19V7C2 6 3 5 4 5"
        'Z" fill="{fill}"/>'
    ),
    "Environmental": (
        '<path d="M16 4C10 4 6 8 6 14C6 21 16 29 16 29C16 29 26 21 26 14'
        "C26 8 22 4 16 4"
        'Z" fill="{fill}"/>'
        '<circle cx="16" cy="14" r="4.2" fill="{stroke_light}"/>'
    ),
    "Institutional": (
        '<rect x="6" y="6" width="20" height="24" rx="3" fill="{fill}"/>'
        '<rect x="12" y="3" width="8" height="5" rx="1.5" fill="{fill}"/>'
        '<line x1="10" y1="14" x2="22" y2="14" stroke="{stroke_light}" stroke-width="1.6" stroke-linecap="round"/>'
        '<line x1="10" y1="19" x2="22" y2="19" stroke="{stroke_light}" stroke-width="1.6" stroke-linecap="round"/>'
        '<line x1="10" y1="24" x2="18" y2="24" stroke="{stroke_light}" stroke-width="1.6" stroke-linecap="round"/>'
    ),
}
# Body-sensed categories get a looping pulse (amplitude, duration) and a
# scrolling waveform ticker (a small illustrative, non-recorded pattern
# tiled twice for a seamless CSS loop); external (non-body-sensed)
# categories get a gentle float instead of a pulse, and no ticker, since
# none of them has a raw signal waveform to illustrate.
_ICON_PULSE = {
    "Neural": (1.05, "2.4s"),
    "Autonomic": (1.16, "1.1s"),
    "Muscular": (1.1, "0.8s"),
    "Behavioral": (1.0, "1.4s"),  # bars animate themselves; icon stays still
}
_TICKER_TILE = {
    "Neural": "0,0 4,9 8,-8 12,8 16,-7 20,5 24,-9 28,6 32,-5 36,3 40,0",
    "Autonomic": "0,0 14,0 16,-3 18,10 20,-13 22,6 24,-1 40,0",
    "Muscular": "0,0 3,-6 5,7 8,-9 10,5 13,-7 16,8 19,-4 22,6 25,-8 28,4 40,0",
    "Behavioral": "0,-5 5,5 10,-5 15,5 20,-5 25,5 30,-5 35,0 40,0",
}
_TICKER_DURATION = {
    "Neural": "1.0s",
    "Autonomic": "1.0s",
    "Muscular": "0.7s",
    "Behavioral": "1.5s",
}

PERSPECTIVE_KEY = "signal_pipeline_perspective"
PERSPECTIVE_BUILD = "Build"
PERSPECTIVE_PRIVACY = "Privacy"
PERSPECTIVE_AGENCY = "Agency"
PERSPECTIVES = (PERSPECTIVE_BUILD, PERSPECTIVE_PRIVACY, PERSPECTIVE_AGENCY)
PERSPECTIVE_DESCRIPTIONS = {
    PERSPECTIVE_BUILD: "Build: what the pipeline collects and combines.",
    PERSPECTIVE_PRIVACY: "Privacy: where each signal is sensitive, or leaves the person.",
    PERSPECTIVE_AGENCY: (
        "Agency: where the person can understand, authorize, override, or "
        "loses the ability to."
    ),
}
CONVERGENCE_PERSPECTIVE_TEXT = {
    PERSPECTIVE_BUILD: (
        "In the Build view: convergence is where separate signals stop "
        "being separate signals and become one combined input the model "
        "reasons over."
    ),
    PERSPECTIVE_PRIVACY: (
        "In the Privacy view: convergence is exactly where signals that "
        "were each hard to re-identify on their own can become "
        "re-identifying in combination - a modality with a low privacy "
        "cost by itself can still raise the combined risk once it "
        "converges with others."
    ),
    PERSPECTIVE_AGENCY: (
        "In the Agency view: convergence means a single override no "
        "longer suffices. Once modalities converge, withdrawing consent "
        "for one signal does not undo what the model already inferred by "
        "combining it with the others."
    ),
}

TRADEOFF_STEP_KEY = "signal_pipeline_tradeoff_step"
TRADEOFF_GAIN_ONLY = 0
TRADEOFF_ADD_PRIVACY = 1
TRADEOFF_ADD_SECURITY = 2
TRADEOFF_ADD_AGENCY = 3

_VEGA_CHART_CONFIG = {
    "background": SURFACE,
    "axis": {
        "gridColor": GRIDLINE,
        "domainColor": "#c3c2b7",
        "tickColor": "#c3c2b7",
        "labelColor": "#898781",
        "titleColor": INK_SECONDARY,
    },
    "view": {"stroke": "transparent"},
}

IENCA_ANDORNO_CITATION = (
    "Ienca, M., & Andorno, R. (2017). Towards new human rights in the age "
    "of neuroscience and neurotechnology. Life Sciences, Society and "
    "Policy, 13, 5. https://doi.org/10.1186/s40504-017-0050-1"
)
DE_MONTJOYE_CITATION = (
    "de Montjoye, Y.-A., Hidalgo, C. A., Verleysen, M., & Blondel, V. D. "
    "(2013). Unique in the crowd: The privacy bounds of human mobility. "
    "Scientific Reports, 3, 1376. https://doi.org/10.1038/srep01376"
)
KOELSTRA_CITATION = (
    "Koelstra, S. et al. (2012). DEAP: A Database for Emotion Analysis "
    "using Physiological Signals. IEEE Transactions on Affective "
    "Computing, 3(1), 18-31. https://doi.org/10.1109/T-AFFC.2011.15"
)
BAGLEY_ET_AL_CITATION = (
    "Bagley, B. A., Rose, N., Kilbourn, Q., & Canham, M. (2026). Threat "
    "vectors and the state of the art in defense methods for security "
    "in neurotechnology. arXiv:2607.10451. "
    "https://arxiv.org/abs/2607.10451"
)

BASELINE_MODALITY_NAME = "Neural (EEG)"

STAGE_KEY = "signal_pipeline_stage"
SELECTED_MODALITIES_KEY = "signal_pipeline_selected_modalities"
PRIVACY_WEIGHT_KEY = "signal_pipeline_privacy_weight"
SECURITY_WEIGHT_KEY = "signal_pipeline_security_weight"
AGENCY_WEIGHT_KEY = "signal_pipeline_agency_weight"

STAGE_RESEARCH_QUESTION = 0
STAGE_BUILD_PIPELINE = 1
STAGE_ADD_MODALITIES = 2
STAGE_CONVERGENCE = 3
STAGE_WEIGH_TRADEOFF = 4
STAGE_RESEARCH_DECISION = 5

JOURNEY_STAGES = (
    "Research question",
    "Build the pipeline",
    "Add modalities",
    "Examine convergence",
    "Weigh the tradeoff",
    "Research decision",
)


def _current_stage() -> int:
    return st.session_state.get(STAGE_KEY, STAGE_RESEARCH_QUESTION)


def _advance_to(stage: int) -> None:
    st.session_state[STAGE_KEY] = max(_current_stage(), stage)
    st.rerun()


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _to_str(value) -> str:
    return "" if pd.isna(value) else str(value)


def _load_modalities() -> tuple[modality_core.Modality, ...]:
    frame = pd.read_csv(SAMPLE_DIR / "modality_profiles.csv")
    return tuple(
        modality_core.Modality(
            name=row["name"],
            category=row["category"],
            signal_examples=row["signal_examples"],
            interpretive_gain=float(row["interpretive_gain"]),
            privacy_cost=float(row["privacy_cost"]),
            security_cost=float(row["security_cost"]),
            agency_cost=float(row["agency_cost"]),
            citation=str(row["citation"]),
            is_body_sensed=_to_bool(row["is_body_sensed"]),
            body_region=_to_str(row["body_region"]),
            privacy_exit_point=str(row["privacy_exit_point"]),
            agency_control_point=str(row["agency_control_point"]),
            notes=str(row["notes"]),
        )
        for _, row in frame.iterrows()
    )


def _pipeline_pictograph_html(pipeline: pipeline_core.SignalPipeline) -> str:
    """
    A drawn pictograph of the pipeline: each modality's own icon (the
    same ones and, where applicable, the same gentle pulse as the
    anatomy scene above, for one consistent icon language on this page)
    at its own per-modality stages, converging into one shared, unlabeled
    node from convergence_stage onward. Static shapes, animated only
    where the anatomy scene above already animates the same category.
    """

    stages = pipeline_core.STAGES
    convergence_index = stages.index(pipeline.convergence_stage)
    pre_stages = stages[:convergence_index]
    shared_stages = stages[convergence_index:]
    modalities = pipeline.modalities
    n = len(modalities)

    stage_gap, left_margin, row_gap, top_margin = 92, 46, 46, 30
    stage_x = {stage: left_margin + i * stage_gap for i, stage in enumerate(stages)}
    row_y = {m.name: top_margin + i * row_gap for i, m in enumerate(modalities)}
    shared_y = top_margin + (n - 1) * row_gap / 2

    width = left_margin + (len(stages) - 1) * stage_gap + 46
    height = top_margin + max(n - 1, 0) * row_gap + 46

    lines: list[str] = []
    icons: list[str] = []

    for m in modalities:
        y = row_y[m.name]
        color = CATEGORY_COLORS[m.category]

        if len(pre_stages) > 1:
            x1, x2 = stage_x[pre_stages[0]], stage_x[pre_stages[-1]]
            lines.append(
                f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" '
                f'stroke="{color}" stroke-width="2" opacity="0.5"/>'
            )
        conv_x = stage_x[shared_stages[0]]
        lines.append(
            f'<line x1="{stage_x[pre_stages[-1]]}" y1="{y}" x2="{conv_x}" '
            f'y2="{shared_y}" stroke="{color}" stroke-width="1.5" opacity="0.4"/>'
        )

        amt, dur = _ICON_PULSE.get(m.category, (1.08, "1.2s"))
        pulse_style = f"--pulse-amt:{amt};--pulse-dur:{dur};"
        pulse_class = "icon-pulse" if m.category in _ICON_PULSE else "icon-float"

        for stage in pre_stages:
            x = stage_x[stage]
            icons.append(f'<g style="{pulse_style}cursor:help;">')
            icons.append(f"<title>{m.name} ({stage})</title>")
            icons.append(_icon_markup(m.category, x, y, 0.75, pulse_class))
            icons.append("</g>")

        icons.append(
            f'<text x="{stage_x[pre_stages[0]]}" y="{y - 22}" text-anchor="middle" '
            f'font-size="10" fill="{INK_SECONDARY}">{m.name}</text>'
        )

    if len(shared_stages) > 1:
        x1, x2 = stage_x[shared_stages[0]], stage_x[shared_stages[-1]]
        lines.append(
            f'<line x1="{x1}" y1="{shared_y}" x2="{x2}" y2="{shared_y}" '
            f'stroke="{MUTED}" stroke-width="2.5"/>'
        )
    for stage in shared_stages:
        x = stage_x[stage]
        icons.append(
            f'<circle cx="{x}" cy="{shared_y}" r="6" fill="{MUTED}" '
            f'style="cursor:help;"><title>{stage}: shared across every '
            f"modality above from here on.</title></circle>"
        )

    stage_labels = "".join(
        f'<text x="{stage_x[stage]}" y="{height - 8}" text-anchor="middle" '
        f'font-size="10" fill="{MUTED}">{stage}</text>'
        for stage in stages
    )

    return f"""
    <div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
                background:{SURFACE}; border-radius:6px; padding:6px 0;">
      <style>
        @keyframes pulseScale {{
          0%, 100% {{ transform: scale(1); }}
          50% {{ transform: scale(var(--pulse-amt, 1.08)); }}
        }}
        .icon-pulse {{
          animation: pulseScale var(--pulse-dur, 1.2s) ease-in-out infinite;
          transform-box: fill-box;
          transform-origin: center;
        }}
        @keyframes floatBob {{
          0%, 100% {{ transform: translateY(0); }}
          50% {{ transform: translateY(-3px); }}
        }}
        .icon-float {{ animation: floatBob 2.6s ease-in-out infinite; }}
      </style>
      <div style="display:flex; justify-content:center;">
        <svg width="100%" height="{height}" viewBox="0 0 {width} {height}"
             preserveAspectRatio="xMidYMid meet">
          {"".join(lines)}
          {"".join(icons)}
          {stage_labels}
        </svg>
      </div>
    </div>
    """


ANATOMY_LEGEND_BUFFER_PX = 56


def _icon_markup(category: str, cx: float, cy: float, scale: float, pulse_class: str) -> str:
    """One category's canonical 0-32 icon, centered at (cx, cy), scaled."""

    path = _ICON_PATHS[category].format(fill=CATEGORY_COLORS[category], stroke_light=SURFACE)
    tx = cx - 16 * scale
    ty = cy - 16 * scale
    return (
        f'<g class="{pulse_class}">'
        f'<g transform="translate({tx:.1f},{ty:.1f}) scale({scale:.3f})">{path}</g>'
        f"</g>"
    )


def _ticker_markup(category: str, x: float, y: float) -> str:
    """
    A small, seamlessly-looping waveform strip beside a body-sensed
    icon. The tile is drawn twice back to back and the pair scrolled by
    exactly one tile-width via CSS, which is what makes the loop seamless
    - not a real recording, an illustrative cue that each category
    "feels" different from the others.
    """

    tile = _TICKER_TILE[category]
    points_b = " ".join(
        f"{float(part.split(',')[0]) + 40:.0f},{part.split(',')[1]}" for part in tile.split()
    )
    color = CATEGORY_COLORS[category]
    duration = _TICKER_DURATION[category]
    return (
        f'<svg x="{x:.1f}" y="{y:.1f}" width="86" height="24" viewBox="0 -12 80 24">'
        f'<g class="ticker-scroll" style="--tick-dur:{duration};">'
        f'<polyline points="{tile}" fill="none" stroke="{color}" stroke-width="1.6" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'<polyline points="{points_b}" fill="none" stroke="{color}" stroke-width="1.6" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f"</g></svg>"
    )


def _anatomy_scene_html(
    modalities: tuple[modality_core.Modality, ...],
    perspective: str,
    height: int = 340,
    show_tickers: bool = True,
) -> str:
    """
    An original, illustrative biological scene - brain / heart / muscle /
    soundwave icons on a soft body outline for body-sensed modalities,
    and speech-bubble / map-pin / clipboard icons for modalities that are
    never sensed from the body. These are hand-authored, simplified
    shapes, not a trace or copy of any commercial illustration library's
    assets - this app has no license to reproduce those.

    Icons pulse; body-sensed ones also carry a small seamlessly-looping
    illustrative waveform (see _ticker_markup) shown only in the Build
    perspective, since Privacy/Agency spend that same visual attention on
    icon size instead. perspective changes icon size (privacy_cost or
    agency_cost) and nothing about the underlying data. show_tickers is
    turned off for the combined, multi-modality figure: up to seven
    simultaneous waveforms would crowd each other's fixed body-region
    slots rather than stay legible, so the ticker is reserved for scenes
    with one modality on screen (the solo baseline, per-modality
    expanders, and the decision stage), where there is no crowding risk.
    """

    width, view_h = SCENE_VIEWBOX
    entry_x, entry_y = ENTRY_POINT_PX

    icons_svg: list[str] = []
    connectors_svg: list[str] = []
    legend_items: list[str] = []
    has_external = False

    for m in modalities:
        if m.is_body_sensed:
            cx, cy = BODY_ICON_PX.get(m.category, (100, 190))
        else:
            cx, cy = EXTERNAL_ICON_PX.get(m.category, (258, 190))
            has_external = True

        if perspective == PERSPECTIVE_PRIVACY:
            cost = m.privacy_cost
        elif perspective == PERSPECTIVE_AGENCY:
            cost = m.agency_cost
        else:
            cost = 0.5
        scale = 0.75 + 0.55 * cost  # canonical icon is a 32px box; ~24px-42px rendered

        color = CATEGORY_COLORS[m.category]
        connectors_svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{entry_x}" y2="{entry_y}" '
            f'stroke="{color}" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.55"/>'
        )

        if m.is_body_sensed:
            amt, dur = _ICON_PULSE.get(m.category, (1.08, "1.2s"))
            pulse_style = f"--pulse-amt:{amt};--pulse-dur:{dur};"
            pulse_class = "icon-pulse"
        else:
            pulse_style = ""
            pulse_class = "icon-float"

        icons_svg.append(f'<g style="{pulse_style}">')
        icons_svg.append(_icon_markup(m.category, cx, cy, scale, pulse_class))
        icons_svg.append(
            f'<text x="{cx}" y="{cy - 16 * scale - 8:.1f}" text-anchor="middle" '
            f'font-size="10" fill="{INK_SECONDARY}">{m.name}</text>'
        )
        icons_svg.append("</g>")

        if m.is_body_sensed and perspective == PERSPECTIVE_BUILD and show_tickers:
            icons_svg.append(_ticker_markup(m.category, cx + 16 * scale + 6, cy - 12))

        legend_items.append(
            f'<span style="display:inline-flex;align-items:center;gap:5px;">'
            f'<span style="width:9px;height:9px;border-radius:2px;background:{color};'
            f'display:inline-block;"></span>{m.category}</span>'
        )

    external_label = ""
    if has_external:
        ex, ey = EXTERNAL_ICON_PX["Subjective"]
        external_label = (
            f'<text x="{ex}" y="{ey - 34}" text-anchor="middle" font-size="10" '
            f'font-style="italic" fill="{MUTED}">Not sensed from the body:</text>'
        )

    body_svg = f"""
        <ellipse cx="{HEAD_CENTER_PX[0]}" cy="{HEAD_CENTER_PX[1]}" rx="{HEAD_RADIUS_PX}" ry="{HEAD_RADIUS_PX}"
            fill="#ece7df" stroke="{MUTED}" stroke-width="1.5" opacity="0.9"/>
        <rect x="72" y="118" width="56" height="110" rx="24" fill="#ece7df" stroke="{MUTED}" stroke-width="1.5" opacity="0.9"/>
        <line x1="78" y1="128" x2="30" y2="204" stroke="#ece7df" stroke-width="16" stroke-linecap="round"/>
        <line x1="122" y1="128" x2="158" y2="182" stroke="#ece7df" stroke-width="16" stroke-linecap="round"/>
        <line x1="86" y1="224" x2="80" y2="332" stroke="#ece7df" stroke-width="18" stroke-linecap="round"/>
        <line x1="114" y1="224" x2="120" y2="332" stroke="#ece7df" stroke-width="18" stroke-linecap="round"/>
    """

    entry_label = (
        f'<text x="{entry_x}" y="{entry_y + 16}" text-anchor="middle" font-size="11" '
        f'font-weight="bold" fill="{INK_SECONDARY}">&#8594; Signals</text>'
    )

    legend_row = "".join(f"<div>{item}</div>" for item in legend_items)
    scale_factor = height / view_h
    svg_w = width * scale_factor

    return f"""
    <div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
                background:{SURFACE}; border-radius:6px; padding:6px 0;">
      <style>
        @keyframes pulseScale {{
          0%, 100% {{ transform: scale(1); }}
          50% {{ transform: scale(var(--pulse-amt, 1.08)); }}
        }}
        .icon-pulse {{
          animation: pulseScale var(--pulse-dur, 1.2s) ease-in-out infinite;
          transform-box: fill-box;
          transform-origin: center;
        }}
        @keyframes floatBob {{
          0%, 100% {{ transform: translateY(0); }}
          50% {{ transform: translateY(-4px); }}
        }}
        .icon-float {{ animation: floatBob 2.6s ease-in-out infinite; }}
        @keyframes barBounce {{
          0%, 100% {{ transform: scaleY(0.55); }}
          50% {{ transform: scaleY(1); }}
        }}
        .sound-bar {{
          animation: barBounce 0.9s ease-in-out infinite;
          transform-box: fill-box;
          transform-origin: bottom;
        }}
        @keyframes tickerScroll {{
          from {{ transform: translateX(0); }}
          to {{ transform: translateX(-50%); }}
        }}
        .ticker-scroll {{ animation: tickerScroll var(--tick-dur, 1.2s) linear infinite; }}
      </style>
      <div style="display:flex; justify-content:center;">
        <svg width="{svg_w:.0f}" height="{height}" viewBox="0 0 {width} {view_h}">
          {body_svg}
          {"".join(connectors_svg)}
          {entry_label}
          {external_label}
          {"".join(icons_svg)}
        </svg>
      </div>
      <div style="display:flex; flex-wrap:wrap; justify-content:center; gap:12px; margin-top:2px;
                  font-size:11px; color:{INK_SECONDARY};">
        {legend_row}
      </div>
    </div>
    """


def _frontier_spec(
    modalities,
    weighted_cost: tradeoff_core.WeightedCostResult,
    frontier: tradeoff_core.GainCostFrontierResult,
    cost_title: str = "Combined cost under these weights (lower is better)",
) -> dict:
    rows = [
        {
            "name": m.name,
            "category": m.category,
            "cost": weighted_cost.combined_cost[m.name],
            "gain": m.interpretive_gain,
            "on_frontier": frontier.is_efficient[m.name],
            "label": m.name,
        }
        for m in modalities
    ]

    categories_present = [c for c in CATEGORY_COLORS if any(r["category"] == c for r in rows)]
    color_domain = categories_present
    color_range = [CATEGORY_COLORS[c] for c in categories_present]
    shape_domain = categories_present
    shape_range = [CATEGORY_SHAPES[c] for c in categories_present]

    return {
        "data": {"values": rows},
        # A text-label layer was tried here and dropped: with up to seven
        # modalities, points can sit close enough together that their
        # labels overlap illegibly. Color + shape + the legend already
        # carry category identity without relying on hue alone, so
        # identity for a specific point is read via hover (tooltip)
        # instead of an always-on label, unlike the DAG diagram above
        # where each node's label is structural, not redundant.
        "mark": {"type": "point", "filled": True, "size": 200},
        "encoding": {
            "x": {
                "field": "cost",
                "type": "quantitative",
                "title": cost_title,
                "scale": {"domain": [0, 1]},
            },
            "y": {
                "field": "gain",
                "type": "quantitative",
                "title": "Interpretive gain (higher is better)",
                "scale": {"domain": [0, 1]},
            },
            "color": {
                "field": "category",
                "type": "nominal",
                "scale": {"domain": color_domain, "range": color_range},
                "legend": {"title": None, "orient": "bottom", "columns": 4},
            },
            "shape": {
                "field": "category",
                "type": "nominal",
                "scale": {"domain": shape_domain, "range": shape_range},
            },
            "opacity": {
                "field": "on_frontier",
                "type": "ordinal",
                "sort": [False, True],
                "scale": {"domain": [False, True], "range": [0.35, 1.0]},
                "legend": None,
            },
            "tooltip": [
                {"field": "label", "type": "nominal", "title": "Modality"},
                {"field": "cost", "type": "quantitative", "title": "Combined cost", "format": ".2f"},
                {"field": "gain", "type": "quantitative", "title": "Interpretive gain", "format": ".2f"},
                {"field": "on_frontier", "type": "nominal", "title": "On frontier"},
            ],
        },
        "width": "container",
        "height": 280,
        "config": _VEGA_CHART_CONFIG,
    }


def _gain_bar_spec(modalities: tuple[modality_core.Modality, ...]) -> dict:
    """One horizontal bar per modality's interpretive gain alone - the
    first step of the gradual tradeoff reveal, before any cost enters."""

    rows = [
        {"name": m.name, "category": m.category, "gain": m.interpretive_gain}
        for m in modalities
    ]
    categories_present = [c for c in CATEGORY_COLORS if any(r["category"] == c for r in rows)]

    return {
        "data": {"values": rows},
        "mark": {"type": "bar"},
        "encoding": {
            "y": {"field": "name", "type": "nominal", "sort": "-x", "title": None},
            "x": {
                "field": "gain",
                "type": "quantitative",
                "title": "Interpretive gain alone (higher is better)",
                "scale": {"domain": [0, 1]},
            },
            "color": {
                "field": "category",
                "type": "nominal",
                "scale": {
                    "domain": categories_present,
                    "range": [CATEGORY_COLORS[c] for c in categories_present],
                },
                "legend": None,
            },
        },
        "width": "container",
        "height": 32 * max(len(rows), 1) + 40,
        "config": _VEGA_CHART_CONFIG,
    }


st.set_page_config(
    page_title="OpenMeasure · Multimodal Signal Convergence",
    page_icon=":material/hub:",
    layout="centered",
)

st.title("Multimodal Signal Convergence")
st.caption(
    "This journey starts with one neurotech signal (EEG) and adds "
    "modalities one at a time to a general Signals -> Sensors -> "
    "Processing -> Inference -> Decision -> Action -> Feedback pipeline."
)
st.caption(
    "A guided illustrative walkthrough: each stage unlocks after you "
    "make a decision or inspect its consequence."
)

render_data_handling_summary(disclosure_for("pages/Multimodal_Signal_Convergence.py"))

stage = _current_stage()

_stage_parts = [
    f"**{label}**" if index == stage else label
    for index, label in enumerate(JOURNEY_STAGES)
]

with st.container(border=True):
    st.markdown(" → ".join(_stage_parts))

if stage > STAGE_RESEARCH_QUESTION:
    if st.button("Restart study", icon=":material/restart_alt:"):
        for key in (
            STAGE_KEY,
            SELECTED_MODALITIES_KEY,
            PRIVACY_WEIGHT_KEY,
            SECURITY_WEIGHT_KEY,
            AGENCY_WEIGHT_KEY,
            PERSPECTIVE_KEY,
            TRADEOFF_STEP_KEY,
        ):
            st.session_state.pop(key, None)
        st.rerun()

st.divider()

# -----------------------------------------------------------------
# 1. Research question
# -----------------------------------------------------------------

section_header("1. Research Question")

st.markdown(
    "### Does Combining Signals Improve Interpretation Enough To Justify "
    "the Added Privacy, Security, and Agency Cost Of Collecting Them?"
)

st.write(
    "A neurotech pipeline built around a single signal - EEG, say - can "
    "always be made 'more informative' by adding another stream: heart "
    "rate, movement, self-report, even a clinician's notes. Each addition "
    "can sharpen what the pipeline infers, but each also adds its own "
    "cost: a new way for the data to be re-identifying (traceable back "
    "to a specific person, even without a name attached), a new point "
    "of exposure, a new way the system can act on someone without their "
    "deliberate say. This journey asks whether an added modality's gain "
    "is validated as worth its cost, rather than assuming more signal is "
    "automatically better."
)

rq_col1, rq_col2 = st.columns(2)
with rq_col1:
    st.badge("Interpretive gain", icon=":material/trending_up:", color="blue")
with rq_col2:
    st.badge("Privacy/security/agency cost", icon=":material/shield:", color="blue")

with st.expander("Validation Question"):
    st.write(
        "Mental privacy and cognitive liberty (the right to control "
        "access to one's own brain data and mental processes) have been "
        "proposed as human rights specifically because neural data can "
        "reveal more about a person than they intended to disclose. The same "
        "concern scales to any signal that is collected continuously and "
        "combined with others: the risk is not any one signal alone, but "
        "what their combination can infer."
    )
    st.caption(IENCA_ANDORNO_CITATION)
    st.write(
        "Security is a separate concern from privacy, and BCI research "
        "surveys it across the full \"BCI cycle\": acquisition hardware, "
        "on-device processing, transmission between devices, the machine "
        "learning models decoding the signal, and the applications or "
        "cloud services consuming the result. More traditional points of "
        "exposure in that cycle, such as wireless communication or cloud "
        "storage, can adopt existing methods (encryption, secure "
        "communication protocols, differential privacy) directly; the "
        "closer to the brain a surface is, the more those methods need "
        "to be modified or replaced entirely."
    )
    st.caption(BAGLEY_ET_AL_CITATION)

if stage < STAGE_BUILD_PIPELINE:
    if st.button("Begin study", type="primary"):
        _advance_to(STAGE_BUILD_PIPELINE)

# -----------------------------------------------------------------
# 2. Build the pipeline
# -----------------------------------------------------------------

if stage >= STAGE_BUILD_PIPELINE:
    section_header(
        "2. Build the Pipeline",
        "One general shape, starting with a single neurotech signal.",
    )

    st.write(
        "Every pipeline in this journey has the same seven stages, "
        "whether it carries one signal or seven: **" + " -> ".join(pipeline_core.STAGES) + "**. "
        "Adding a modality means adding a Signals/Sensors node to this "
        "shape, not rebuilding it - the diagram below is the same engine "
        "used at every later stage of this journey."
    )

    with st.expander("Using this module with your own modalities"):
        st.write(
            "The Multimodal Signal Pipeline core this journey is built "
            "on (`modules/signal_pipeline/core`) needs, for each "
            "modality, one interpretive-gain rating (higher is better) "
            "and three cost ratings, privacy, security, and agency "
            "(each higher is worse), plus a supporting citation - see "
            "`modules/signal_pipeline/README.md` for the exact "
            "`Modality` format and what `combine_costs` and "
            "`compute_gain_cost_frontier` compute from them."
        )

    all_modalities = _load_modalities()
    modalities_by_name = {m.name: m for m in all_modalities}
    baseline = modalities_by_name[BASELINE_MODALITY_NAME]

    st.caption("Where the signal actually comes from:")
    components.html(
        _anatomy_scene_html((baseline,), PERSPECTIVE_BUILD, height=340),
        height=340 + ANATOMY_LEGEND_BUFFER_PX,
    )
    st.caption(f"Signal examples: {baseline.signal_examples}.")
    st.caption(baseline.citation)

    st.caption(f"How {baseline.name}'s signal travels through the pipeline:")
    pipeline_cols = st.columns(len(PIPELINE_STAGE_NOTES))
    for col, (icon, label, note) in zip(pipeline_cols, PIPELINE_STAGE_NOTES):
        with col:
            st.badge(label, icon=icon, color="blue")
            st.caption(note)

    st.write(
        "**Interpretation**: a pipeline with one modality already "
        "has every stage a multimodal pipeline will have. What changes "
        "when a modality is added is not the shape of the pipeline, but "
        "how many signals converge into it, and what that convergence "
        "costs."
    )

    if stage < STAGE_ADD_MODALITIES:
        if st.button("Continue to add modalities", type="primary"):
            _advance_to(STAGE_ADD_MODALITIES)

# -----------------------------------------------------------------
# 3. Add modalities
# -----------------------------------------------------------------

if stage >= STAGE_ADD_MODALITIES:
    section_header(
        "3. Add Modalities",
        "Add categories one at a time; the diagram below updates as you do.",
    )

    all_modalities = _load_modalities()
    modalities_by_name = {m.name: m for m in all_modalities}
    baseline = modalities_by_name[BASELINE_MODALITY_NAME]
    addable = tuple(m for m in all_modalities if m.name != BASELINE_MODALITY_NAME)

    selected_names = st.multiselect(
        "Modalities to add to the EEG-only pipeline",
        options=[m.name for m in addable],
        default=st.session_state.get(SELECTED_MODALITIES_KEY, []),
        key=SELECTED_MODALITIES_KEY,
    )

    current_modalities = (baseline,) + tuple(
        modalities_by_name[name] for name in selected_names
    )
    current_pipeline = pipeline_core.build_pipeline(current_modalities)

    st.caption(
        "Perspective - same modalities, different question about them:"
    )
    perspective = st.segmented_control(
        "Perspective",
        options=PERSPECTIVES,
        default=PERSPECTIVE_BUILD,
        key=PERSPECTIVE_KEY,
        label_visibility="collapsed",
    ) or PERSPECTIVE_BUILD
    st.caption(PERSPECTIVE_DESCRIPTIONS[perspective])

    components.html(
        _anatomy_scene_html(current_modalities, perspective, height=340, show_tickers=False),
        height=340 + ANATOMY_LEGEND_BUFFER_PX,
    )
    if perspective == PERSPECTIVE_BUILD:
        st.caption(
            "Dashed lines show every selected signal feeding the same "
            "pipeline entry point; illustrative trace shapes are not real "
            "recordings."
        )
    elif perspective == PERSPECTIVE_PRIVACY:
        st.caption("Marker size here scales with each modality's illustrative privacy cost.")
    else:
        st.caption("Marker size here scales with each modality's illustrative agency cost.")

    st.caption("How those signals travel through the pipeline:")
    pictograph_height = 30 + max(len(current_modalities) - 1, 0) * 46 + 46
    components.html(
        _pipeline_pictograph_html(current_pipeline),
        height=pictograph_height + 10,
    )
    st.caption(
        "Hover an icon for detail. The small gray node marks where "
        "every selected modality's chain converges into one shared "
        "pipeline."
    )

    for name in selected_names:
        added = modalities_by_name[name]
        with st.expander(f"{added.name} - what this adds"):
            components.html(
                _anatomy_scene_html((added,), perspective, height=220),
                height=220 + ANATOMY_LEGEND_BUFFER_PX,
            )
            if added.is_body_sensed:
                st.caption(f"Sensed at: {added.body_region}.")
            else:
                st.caption("Not sensed from the body - shown as an external marker above.")
            st.caption(f"Signal examples: {added.signal_examples}.")
            st.write(
                f"Interpretive gain (illustrative): **{added.interpretive_gain:.2f}** "
                f"/ 1.00. Privacy cost: **{added.privacy_cost:.2f}**, "
                f"security cost: **{added.security_cost:.2f}**, agency "
                f"cost: **{added.agency_cost:.2f}** (each 0 = negligible, "
                "1 = severe)."
            )
            if perspective == PERSPECTIVE_PRIVACY:
                st.write(f"**Where it leaves the person**: {added.privacy_exit_point}")
            elif perspective == PERSPECTIVE_AGENCY:
                st.write(f"**Where the person retains or loses control**: {added.agency_control_point}")
            st.caption(added.notes)
            st.caption(added.citation)

    caveat(
        "These ratings are illustrative, author-assigned scores informed "
        "by the cited literature's directional findings, not values any "
        "cited source reports directly - see the page docstring and "
        "modules/signal_pipeline/README.md."
    )

    st.write(
        "**Limitations**: each added node is easy to draw. What "
        "the diagram does not show on its own is whether the resulting "
        "pipeline's combined interpretive value is worth its combined "
        "cost - that comparison is Step 5."
    )

    if stage < STAGE_CONVERGENCE:
        if st.button("Continue to examine convergence", type="primary"):
            _advance_to(STAGE_CONVERGENCE)

# -----------------------------------------------------------------
# 4. Examine convergence
# -----------------------------------------------------------------

if stage >= STAGE_CONVERGENCE:
    section_header(
        "4. Examine Convergence",
        "Where the added signals actually meet, and what meets there.",
    )

    all_modalities = _load_modalities()
    modalities_by_name = {m.name: m for m in all_modalities}
    baseline = modalities_by_name[BASELINE_MODALITY_NAME]
    selected_names = st.session_state.get(SELECTED_MODALITIES_KEY, [])
    current_modalities = (baseline,) + tuple(
        modalities_by_name[name] for name in selected_names
    )
    current_pipeline = pipeline_core.build_pipeline(current_modalities)

    convergence = pipeline_core.compute_convergence(current_pipeline, stage="Inference")

    st.metric("Modalities converging at Inference", convergence.n_modalities)
    st.write(
        f"Categories represented: **{', '.join(convergence.categories)}**."
    )

    st.write(
        "Every modality's Sensors node feeds the same Processing stage, "
        "so **Inference is where the pipeline's model, not any single "
        "sensor, decides what the combination means** - a claim about the "
        "person or process that no single modality's Sensors node makes "
        "on its own."
    )

    if convergence.n_modalities == 1:
        st.info(
            "Only the baseline EEG signal is in the pipeline. Go back to "
            "Step 3 to add modalities and see convergence with more than "
            "one signal."
        )

    perspective = st.session_state.get(PERSPECTIVE_KEY, PERSPECTIVE_BUILD) or PERSPECTIVE_BUILD
    st.caption(f"Viewed from Step 3's current perspective ({perspective}):")
    st.write(CONVERGENCE_PERSPECTIVE_TEXT[perspective])

    st.write(
        "**Implications**: convergence is where a multimodal "
        "pipeline's distinctive claim to power - and its distinctive "
        "risk - both live. The model can infer something from the "
        "combination that it could not infer from any signal alone, "
        "which is exactly why the combination's cost has to be assessed "
        "as a combination, not signal by signal."
    )

    if stage < STAGE_WEIGH_TRADEOFF:
        if st.button("Continue to weigh the tradeoff", type="primary"):
            _advance_to(STAGE_WEIGH_TRADEOFF)

# -----------------------------------------------------------------
# 5. Weigh the tradeoff
# -----------------------------------------------------------------

if stage >= STAGE_WEIGH_TRADEOFF:
    section_header(
        "5. Weigh the Tradeoff",
        "An illustrative synthesis built on the ratings from Step 3, "
        "grounded where a real measured result is available.",
    )

    all_modalities = _load_modalities()
    modalities_by_name = {m.name: m for m in all_modalities}
    baseline = modalities_by_name[BASELINE_MODALITY_NAME]
    selected_names = st.session_state.get(SELECTED_MODALITIES_KEY, [])
    current_modalities = (baseline,) + tuple(
        modalities_by_name[name] for name in selected_names
    )

    if len(current_modalities) < 2:
        st.info(
            "Add at least one modality in Step 3 to compare gain against "
            "cost across more than one signal."
        )
        if stage < STAGE_RESEARCH_DECISION:
            if st.button("Continue to the research decision", type="primary"):
                _advance_to(STAGE_RESEARCH_DECISION)
    else:
        tradeoff_step = st.session_state.get(TRADEOFF_STEP_KEY, TRADEOFF_GAIN_ONLY)

        st.caption(
            "Introduced gradually: gain alone, then each cost dimension "
            "one at a time, before the full weighted picture."
        )

        st.vega_lite_chart(_gain_bar_spec(current_modalities), width="stretch")
        st.write(
            "**Reading this**: further right means a modality is rated "
            "as adding more independent interpretive value on its own. "
            "Nothing here yet says anything about what it costs."
        )

        if any(m.category in ("Autonomic", "Muscular") for m in current_modalities):
            st.info(
                "A real, measured result behind these Autonomic/Muscular "
                "ratings: on DEAP's own single-trial emotion classifiers, "
                "EEG alone scored 0.563 F1 and its bundled peripheral "
                "channels (ECG, EMG, GSR, respiration, skin temperature, "
                "eye movement, not separable into this page's Autonomic "
                "and Muscular categories individually) scored 0.608 F1 on "
                "valence - not significantly different from each other "
                "(p=0.41). Fusing peripheral with a third modality this "
                "page does not model (multimedia content analysis) "
                "reached 0.652 F1, the only fusion result in the paper "
                "that reached significance (p=0.025); fusing all three "
                "available modalities together did not improve on that "
                "two-modality result. The paper's own conclusion: fusion "
                "generally helped, but only slightly, and rarely enough "
                "to call significant."
            )
            st.caption(KOELSTRA_CITATION)

        if tradeoff_step < TRADEOFF_ADD_PRIVACY:
            if st.button("Now add privacy cost", type="primary"):
                st.session_state[TRADEOFF_STEP_KEY] = TRADEOFF_ADD_PRIVACY
                st.rerun()

        if tradeoff_step >= TRADEOFF_ADD_PRIVACY:
            st.divider()
            privacy_only_cost = tradeoff_core.combine_costs(current_modalities, 1.0, 0.0, 0.0)
            privacy_only_frontier = tradeoff_core.compute_gain_cost_frontier(
                current_modalities, privacy_only_cost
            )
            st.vega_lite_chart(
                _frontier_spec(
                    current_modalities,
                    privacy_only_cost,
                    privacy_only_frontier,
                    cost_title="Privacy cost alone (lower is better)",
                ),
                width="stretch",
            )
            st.write(
                "**Reading this**: moving right now means more privacy "
                "cost specifically, not more of everything. A modality "
                "can sit high on gain and low on privacy cost at once - "
                "being high on one axis says nothing about the other."
            )

            if tradeoff_step < TRADEOFF_ADD_SECURITY:
                if st.button("Now add security cost", type="primary"):
                    st.session_state[TRADEOFF_STEP_KEY] = TRADEOFF_ADD_SECURITY
                    st.rerun()

        if tradeoff_step >= TRADEOFF_ADD_SECURITY:
            st.divider()
            privacy_security_cost = tradeoff_core.combine_costs(current_modalities, 0.5, 0.5, 0.0)
            privacy_security_frontier = tradeoff_core.compute_gain_cost_frontier(
                current_modalities, privacy_security_cost
            )
            st.vega_lite_chart(
                _frontier_spec(
                    current_modalities,
                    privacy_security_cost,
                    privacy_security_frontier,
                    cost_title="Privacy + security cost, equally weighted (lower is better)",
                ),
                width="stretch",
            )
            st.write(
                "**Reading this**: a modality can move right here even if "
                "its privacy cost alone was low, if its security cost is "
                "high - an intercepted or stolen signal is a different "
                "kind of harm than a re-identifying one, and this axis "
                "now carries both at once."
            )

            if any(m.category == "Neural" for m in current_modalities):
                st.info(
                    "A real, demonstrated security threat behind Neural's "
                    "rating: researchers showed that amplitude-modulated "
                    "radio-frequency signals from a remote antenna can be "
                    "picked up by EEG electrode wires acting as unintended "
                    "antennas and injected into the acquisition hardware "
                    "as fabricated brain activity - through walls, at "
                    "roughly three meters, with no access to the BCI "
                    "software or data. Separately, EEG's own activity "
                    "patterns are distinctive enough to identify who "
                    "produced them, so a device collecting EEG for one "
                    "purpose is, in effect, also collecting a biometric "
                    "identifier as a byproduct - the same signal serves "
                    "both purposes at once."
                )
                st.caption(BAGLEY_ET_AL_CITATION)

            if tradeoff_step < TRADEOFF_ADD_AGENCY:
                if st.button("Now add agency cost, and set your own weights", type="primary"):
                    st.session_state[TRADEOFF_STEP_KEY] = TRADEOFF_ADD_AGENCY
                    st.rerun()

        if tradeoff_step >= TRADEOFF_ADD_AGENCY:
            st.divider()
            st.caption(
                "The full picture: all three costs, weighted however you "
                "choose. Weights are normalized automatically, so they do "
                "not need to add up to anything in particular."
            )
            w1, w2, w3 = st.columns(3)
            privacy_weight = w1.slider(
                "Privacy weight", 0.0, 1.0, 0.34, 0.01, key=PRIVACY_WEIGHT_KEY
            )
            security_weight = w2.slider(
                "Security weight", 0.0, 1.0, 0.33, 0.01, key=SECURITY_WEIGHT_KEY
            )
            agency_weight = w3.slider(
                "Agency weight", 0.0, 1.0, 0.33, 0.01, key=AGENCY_WEIGHT_KEY
            )

            weighted_cost = tradeoff_core.combine_costs(
                current_modalities, privacy_weight, security_weight, agency_weight
            )
            frontier = tradeoff_core.compute_gain_cost_frontier(current_modalities, weighted_cost)

            st.caption(
                "Normalized weights used above: "
                f"privacy {weighted_cost.privacy_weight:.2f}, "
                f"security {weighted_cost.security_weight:.2f}, "
                f"agency {weighted_cost.agency_weight:.2f}."
            )

            st.vega_lite_chart(
                _frontier_spec(current_modalities, weighted_cost, frontier), width="stretch"
            )
            st.caption(
                "Faded points are dominated under these weights: some other "
                "modality here costs no more and gains no less."
            )

            for m in current_modalities:
                if not frontier.is_efficient[m.name]:
                    flagged_item_note(
                        m.name,
                        "Under these weights and these illustrative ratings, "
                        "this modality is dominated by another and would not "
                        "be favored by any weighting of these two dimensions "
                        "alone.",
                    )

            caveat(
                "A weighted combination of privacy, security, and agency cost "
                "is only one way to reduce three costs that are not directly "
                "comparable into a single number for this chart. It reflects "
                "the weights you set, not a judgment this page is making - a "
                "low combined cost does not, by itself, mean a modality is "
                "safe to collect."
            )

            st.write(
                "**What to inspect**: try setting privacy weight to 1.0 "
                "and the others to 0.0, then reverse it. Which modalities are "
                "dominated changes with the weights - the frontier is a map "
                "of what the ratings imply under a given weighting, not a "
                "single verdict on any modality."
            )

            if stage < STAGE_RESEARCH_DECISION:
                if st.button("Continue to the research decision", type="primary"):
                    _advance_to(STAGE_RESEARCH_DECISION)

# -----------------------------------------------------------------
# 6. Research decision
# -----------------------------------------------------------------

if stage >= STAGE_RESEARCH_DECISION:
    section_header("6. Research Decision")

    all_modalities = _load_modalities()
    modalities_by_name = {m.name: m for m in all_modalities}
    baseline = modalities_by_name[BASELINE_MODALITY_NAME]
    selected_names = st.session_state.get(SELECTED_MODALITIES_KEY, [])
    current_modalities = (baseline,) + tuple(
        modalities_by_name[name] for name in selected_names
    )

    if len(current_modalities) < 2:
        st.write(
            "Only EEG is in the pipeline, so there is no added modality "
            "to decide about yet. Go back to Step 3 to add at least one."
        )
    else:
        privacy_weight = st.session_state.get(PRIVACY_WEIGHT_KEY, 0.34)
        security_weight = st.session_state.get(SECURITY_WEIGHT_KEY, 0.33)
        agency_weight = st.session_state.get(AGENCY_WEIGHT_KEY, 0.33)

        weighted_cost = tradeoff_core.combine_costs(
            current_modalities, privacy_weight, security_weight, agency_weight
        )
        frontier = tradeoff_core.compute_gain_cost_frontier(current_modalities, weighted_cost)

        addable_selected = tuple(m for m in current_modalities if m.name != baseline.name)

        decision = st.radio(
            "Given the frontier above, which added modality would you keep in a real deployment?",
            options=[m.name for m in addable_selected] + ["Insufficient evidence to decide"],
            index=len(addable_selected),
        )

        if decision != "Insufficient evidence to decide":
            components.html(
                _anatomy_scene_html((modalities_by_name[decision],), PERSPECTIVE_BUILD, height=220),
                height=220 + ANATOMY_LEGEND_BUFFER_PX,
            )

        if decision == "Insufficient evidence to decide":
            st.write(
                "A reasonable position: these ratings are illustrative "
                "and informed by literature that was not collected for "
                "this specific pipeline, population, or deployment "
                "context, so generalizing to a specific real system is "
                "not directly supported by what is rated here."
            )
        elif frontier.is_efficient.get(decision, False):
            st.write(
                f"{decision} is on the gain-cost frontier under the "
                "weights set in Step 5, meaning no other modality here "
                "both costs no more and gains no less. Whether it is "
                "actually worth collecting still depends on how much "
                "privacy, security, and agency cost should weigh against "
                "interpretive gain in your specific context, and on "
                "evidence this illustrative rating doesn't provide about "
                "a real deployment."
            )
        else:
            st.write(
                f"Under the weights set in Step 5, {decision} is "
                "dominated by another modality here and would not be "
                "favored by any weighting of these two dimensions alone - "
                "though a real deployment could involve factors these "
                "four illustrative ratings do not capture."
            )

        st.write(
            "**Research considerations**: when a pipeline can act on what "
            "it infers - flagging a clinician, triggering a device, "
            "shaping what someone sees next - the cost of being wrong, or "
            "of acting on an inference the person did not consent to, is "
            "not the same question as whether the inference itself is "
            "accurate. Validating a multimodal pipeline means validating "
            "both."
        )

    st.divider()
    with st.container(border=True):
        st.write(
            "**Implications**: the Signals -> Sensors -> Processing -> "
            "Inference -> Decision -> Action -> Feedback shape used "
            "throughout this journey does not change when a modality is "
            "added - only how many signals converge into it. That is what "
            "makes 'is this combination worth its cost' an answerable, "
            "revisitable question instead of a one-time architectural "
            "decision."
        )
    st.caption(DE_MONTJOYE_CITATION)
