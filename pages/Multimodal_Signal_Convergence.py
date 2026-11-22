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

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from modules.signal_pipeline.core import agency as agency_core
from modules.signal_pipeline.core import intervention as intervention_core
from modules.signal_pipeline.core import modality as modality_core
from modules.signal_pipeline.core import neuroanatomy as neuroanatomy_core
from modules.signal_pipeline.core import pipeline as pipeline_core
from modules.signal_pipeline.core import resting_state as resting_core
from modules.signal_pipeline.core import hemodynamics as hemo_core
from modules.signal_pipeline.core import correlation_lesson as corr_core
from modules.signal_pipeline.core import filtering as filtering_core
from modules.signal_pipeline.core import tradeoff as tradeoff_core
from shared.data_handling import disclosure_for, render_data_handling_summary
from shared.journey_stages import StageTracker
from shared.stage_workspace import Stage, StageWorkspace
from shared.report import caveat, flagged_item_note, section_header
from shared import neuroanatomy_figure
from shared import resting_state_figure
from shared import pipeline_figures

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
    (":material/loop:", "Feedback", "The response can change the signal that comes next"),
)

# What is actually passed from one stage to the next. The stage names
# alone said what each stage is, not what crosses the boundary between
# them, which is the part a reader following the order needs.
PIPELINE_HANDOFFS = (
    "a physical signal a sensor can pick up",
    "a raw measured trace",
    "a cleaned, transformed signal",
    "an estimate of a state",
    "a threshold result",
    "a response the person experiences",
)
PIPELINE_RETURN_HANDOFF = "a signal that may change the next time around"


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
    "padding": 8,
    # Keep axis titles and legends inside the chart's frame on narrow screens.
    "autosize": {"type": "fit", "contains": "padding"},
    "axis": {
        "gridColor": GRIDLINE,
        "domainColor": "#c3c2b7",
        "tickColor": "#c3c2b7",
        "labelColor": "#898781",
        "titleColor": INK_SECONDARY,
    },
    # Explicit ink on the light surface, so a dark app theme cannot set
    # light legend text on it.
    "legend": {"labelColor": INK_SECONDARY, "titleColor": INK_SECONDARY},
    "title": {"color": INK_SECONDARY},
    "view": {"stroke": "transparent"},
}

# Room for the title, legend, and axes around a spec's plot height.
_CHART_CHROME_PX = 90


def _chart(spec: dict, width: str = "stretch") -> None:
    """
    Draw a Vega-Lite spec at its own height, in this page's light theme.

    Streamlit sizes a chart's container from its height argument, and
    without one a "container"-width chart could collapse to a few pixels,
    flattening a trace and stacking a strip chart's rows. theme=None keeps
    Streamlit's dark theme from recolouring text on the light surface.
    """
    height = int(spec.get("height", 200)) + _CHART_CHROME_PX
    st.vega_lite_chart(spec, width=width, height=height, theme=None)

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
ARMENGOL_URPI_ET_AL_CITATION = (
    "Armengol-Urpi, A., Kovacs, R., & Sarma, S. E. (2023). Brain-Hack: "
    "Remotely injecting false brain-waves with RF to take control of a "
    "brain-computer interface. Proceedings of the 5th Workshop on "
    "CPS&IoT Security and Privacy (CPSIoTSec). "
    "https://doi.org/10.1145/3605758.3623497 (the primary study behind "
    "the RF-injection finding below; Bagley et al., 2026 relays it as "
    "part of its survey)"
)
DENNING_ET_AL_CITATION = (
    "Denning, T., Matsuoka, Y., & Kohno, T. (2009). Neurosecurity: "
    "security and privacy for neural devices. Neurosurgical Focus, "
    "27(1), E7. https://doi.org/10.3171/2009.4.FOCUS0985"
)
SANCHEZ_GAMA_ET_AL_CITATION = (
    "Sánchez Gama, M. de J., Barradas Chacón, L. A., Chacón Gutiérrez, "
    "L., Fernández Harmony, T., Novo Olivas, C. A., & De La Roca "
    "Chiapas, J. M. (2024). Resting state EEG with closed eyes and open "
    "eyes in females from 60 to 80 years old [Data set]. OpenNeuro. "
    "https://doi.org/10.18112/openneuro.ds005420.v1.0.0 (CC0)"
)
BARRY_ET_AL_CITATION = (
    "Barry, R. J., Clarke, A. R., Johnstone, S. J., Magee, C. A., & "
    "Rushby, J. A. (2007). EEG differences between eyes-closed and "
    "eyes-open resting conditions. Clinical Neurophysiology, 118(12), "
    "2765-2773. https://doi.org/10.1016/j.clinph.2007.07.028"
)

# Citations are numbered and listed once at the end of the page rather
# than printed under the sentence they support, where they interrupted
# the point each stage is making. PAGE_REFERENCE_ORDER fixes the numbers
# for the references this page cites in its own prose; references that
# appear only in a modality's data row are appended after these, in
# modality order, by _reference_registry below.
PAGE_REFERENCE_ORDER: tuple[str, ...] = (
    IENCA_ANDORNO_CITATION,
    DE_MONTJOYE_CITATION,
    KOELSTRA_CITATION,
    BAGLEY_ET_AL_CITATION,
    ARMENGOL_URPI_ET_AL_CITATION,
    DENNING_ET_AL_CITATION,
    SANCHEZ_GAMA_ET_AL_CITATION,
    BARRY_ET_AL_CITATION,
)


def _split_references(text: str) -> tuple[str, ...]:
    """
    Split a citation string that holds more than one reference.

    A modality's citation field joins its references with "; ", but a
    single reference can itself contain a semicolon inside parentheses,
    so splitting on every semicolon would cut one reference in half.
    Only top-level semicolons separate references.
    """

    parts: list[str] = []
    current: list[str] = []
    depth = 0

    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)

        if char == ";" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)

    parts.append("".join(current))

    return tuple(part.strip() for part in parts if part.strip())


BASELINE_MODALITY_NAME = "Neural (EEG)"

STAGE_KEY = "signal_pipeline_stage"
SELECTED_MODALITIES_KEY = "signal_pipeline_selected_modalities"
PRIVACY_WEIGHT_KEY = "signal_pipeline_privacy_weight"
SECURITY_WEIGHT_KEY = "signal_pipeline_security_weight"
AGENCY_WEIGHT_KEY = "signal_pipeline_agency_weight"

# Where each of the above survives leaving the stage that draws it. A
# widget's own key is dropped the moment its widget stops being
# rendered, and every one of these is read by a later stage: the
# selection by three of them, the weights by the research decision.
KEPT = {
    SELECTED_MODALITIES_KEY: "mmsc_kept_modalities",
    PERSPECTIVE_KEY: "mmsc_kept_perspective",
    PRIVACY_WEIGHT_KEY: "mmsc_kept_privacy",
    SECURITY_WEIGHT_KEY: "mmsc_kept_security",
    AGENCY_WEIGHT_KEY: "mmsc_kept_agency",
}


def _held(widget_key, default):
    """What a later stage should read instead of the widget's own key."""
    return st.session_state.get(KEPT[widget_key], default)


def _hold(widget_key, value):
    """Mirror a control's value where a later stage can still find it."""
    st.session_state[KEPT[widget_key]] = value

    return value

STAGE_RESEARCH_QUESTION = 0
STAGE_BUILD_PIPELINE = 1
STAGE_ADD_MODALITIES = 2
STAGE_CONVERGENCE = 3
STAGE_WEIGH_TRADEOFF = 4
STAGE_REAL_SIGNAL = 5
STAGE_RESEARCH_DECISION = 6

JOURNEY_STAGES = (
    "Research question",
    "Build the pipeline",
    "Add modalities",
    "Examine convergence",
    "Weigh the tradeoff",
    "Protect a recorded signal",
    "Research decision",
)


# A second, nested StageTracker: the "Weigh the tradeoff" stage above
# unlocks its own cost dimensions one at a time (gain alone, then
# privacy, then security, then agency), which is a gated sequence in its
# own right rather than part of the top-level journey breadcrumb.
TRADEOFF_TRACKER = StageTracker(
    session_key=TRADEOFF_STEP_KEY,
    stage_labels=("Gain only", "+ Privacy", "+ Security", "+ Agency"),
)


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


ANATOMY = neuroanatomy_core.atlas()
ANATOMY_REFERENCES: dict[str, str] = {
    r.key: f"{r.text} {r.url}".strip() for r in ANATOMY.references
}

LESSON_REFERENCES: dict[str, str] = {
    key: f"{text} {url}"
    for key, (text, url) in {**hemo_core.REFERENCES, **corr_core.REFERENCES}.items()
}

RESTING_REFERENCES: dict[str, str] = {
    r.key: f"{r.text} {r.url}".strip() for r in resting_core.REFERENCES
}


def _reference_registry(
    modalities: tuple[modality_core.Modality, ...],
) -> tuple[str, ...]:
    """Every reference this page can cite, in the order it is numbered."""

    ordered = list(PAGE_REFERENCE_ORDER)

    for modality in modalities:
        for reference in _split_references(modality.citation):
            if reference not in ordered:
                ordered.append(reference)

    # The neuroanatomy figure's references, formatted the way this page
    # writes citations, so a work cited both ways is listed once.
    for reference in (*ANATOMY_REFERENCES.values(), *RESTING_REFERENCES.values(), *LESSON_REFERENCES.values()):
        if reference not in ordered:
            ordered.append(reference)

    return tuple(ordered)


def _lesson_ref(*keys: str) -> str:
    """Reference markers for a lesson's keys, wherever the work is listed."""
    pool = {**ANATOMY_REFERENCES, **RESTING_REFERENCES, **LESSON_REFERENCES}
    return _ref(*(pool[key] for key in keys))


def _resting_ref(*keys: str) -> str:
    """Reference markers for resting-state study citation keys."""
    return _ref(*(RESTING_REFERENCES[key] for key in keys))


def _anatomy_ref(*keys: str) -> str:
    """Reference markers for neuroanatomy citation keys."""
    return _ref(*(ANATOMY_REFERENCES[key] for key in keys))


REFERENCES: tuple[str, ...] = _reference_registry(_load_modalities())


def _ref(*citations: str) -> str:
    """
    The bracketed reference markers for one or more citations.

    Raises ValueError through list.index if a citation is not in
    REFERENCES, rather than silently printing a marker that points at no
    entry in the list at the end of the page.
    """

    numbers = {
        REFERENCES.index(reference) + 1
        for citation in citations
        for reference in _split_references(citation)
    }

    return "[" + ", ".join(str(number) for number in sorted(numbers)) + "]"


def _render_neuroanatomy_explorer() -> None:
    """Where each sensor gets a signal, and what kind of signal it is."""
    with st.container(border=True):
        st.markdown("**Where Each Sensor Gets a Signal**")
        st.write(
            "The EEG in this journey comes from electrode O1, where the alpha rhythm is "
            "strongest: posterior scalp sites over the back of the head. Its generators are "
            "distributed across occipital and parietal cortex and shaped by loops that link "
            "cortex and thalamus. Choose a sensor "
            "to compare where it gets a signal (shading) and what kind of signal that is "
            "(colour). Only EEG measures electrical activity directly."
        )
        by_name = {m.short_name: m for m in ANATOMY.modalities}
        chosen_name = st.segmented_control(
            "Sensor",
            options=list(by_name),
            default=next(m.short_name for m in ANATOMY.modalities if m.in_journey),
            key="mmsc_anatomy_modality",
        ) or next(m.short_name for m in ANATOMY.modalities if m.in_journey)
        chosen = by_name[chosen_name]

        systems = {s.name: s for s in ANATOMY.systems}
        system_name = st.selectbox(
            "Region to inspect",
            options=list(systems),
            index=[s.key for s in ANATOMY.systems].index("occipital"),
            key="mmsc_anatomy_system",
        )
        system = systems[system_name]

        components.html(
            neuroanatomy_figure.side_view_svg(chosen.key, selected_system=system.key)
            + neuroanatomy_figure.legend_html(chosen.key),
            height=440,
        )

        cover = chosen.coverage_for(system.key)
        st.markdown(
            f"**{system.name}**: {neuroanatomy_core.coverage_label(chosen.key, system.key).lower()} "
            f"for {chosen.short_name}."
        )
        st.write(f"{system.location}. {system.role} {_anatomy_ref(*system.refs)}")
        if cover.note:
            st.caption(cover.note)

        st.markdown(
            f"- **Setup:** {chosen.configuration}\n"
            f"- **Measures:** {chosen.measures}\n"
            f"- **Link to neurons:** {chosen.neural_link}\n"
            f"- **Timing:** {chosen.timing}\n"
            f"- **Also captures:** {chosen.also_captures} {chosen.privacy_note} "
            f"{_anatomy_ref(*chosen.refs)}"
        )

        with st.expander("Every region, for every sensor"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Region": s.name,
                            **{
                                m.short_name: neuroanatomy_core.coverage_label(m.key, s.key)
                                for m in ANATOMY.modalities
                            },
                        }
                        for s in ANATOMY.systems
                    ]
                    + [
                        {
                            "Region": "Signal type",
                            **{
                                m.short_name: neuroanatomy_figure.MEASURE_LABEL[m.measurement]
                                for m in ANATOMY.modalities
                            },
                        }
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

        caveat(f"{ANATOMY.caveat} {_anatomy_ref(*ANATOMY.caveat_refs)}")


_RANGE_LABEL = {
    resting_core.RANGE_BY_DESIGN: "Fixed by design",
    resting_core.RANGE_CALCULATED: "Calculated for this sample",
    resting_core.RANGE_DIRECTION: "Expected direction",
    resting_core.RANGE_WITHIN_PERSON: "Compare within a person",
}


def _study_card(study: resting_core.Study, title: str) -> None:
    with st.expander(title):
        st.markdown(
            f"**Who:** {study.participants}  \n"
            f"**Conditions:** {'; '.join(study.conditions)}.  \n"
            f"**Recorded:** {' '.join(study.recordings)}"
        )
        st.write(study.protocol)
        st.markdown("\n".join(f"- {finding}" for finding in study.findings))
        st.write(f"{study.interpretation} {_resting_ref(*study.refs)}")


def _render_resting_study() -> None:
    """How an eyes-closed / eyes-open recording is made, and what its numbers mean."""
    snippet = _load_eeg_snippet()
    result = intervention_core.classify_eyes_state(
        snippet["eyes_closed"], snippet["eyes_open"], intervention_core.SAMPLING_RATE_HZ
    )
    n_trials = result.n_windows_eyes_closed + result.n_windows_eyes_open
    react = resting_core.reactivity(result.alpha_power_eyes_closed, result.alpha_power_eyes_open)
    odds = resting_core.chance(round(result.accuracy * n_trials), n_trials)

    with st.container(border=True):
        st.markdown("**The Study Behind This Signal**")
        room, numbers, dichotomies = st.tabs(["The recording", "What the numbers mean", "Resting vs task"])

        with room:
            eyes = st.segmented_control(
                "Condition", ["Eyes closed", "Eyes open"], default="Eyes closed",
                key="mmsc_study_eyes", label_visibility="collapsed",
            ) or "Eyes closed"
            closed = eyes == "Eyes closed"
            components.html(resting_state_figure.participant_svg(closed), height=330)
            mean = react.mean_closed if closed else react.mean_open
            st.markdown(
                f":blue-badge[{resting_core.SOURCE_LABEL[resting_core.SOURCE_RECORDED]}] "
                f"Mean alpha power at O1: **{mean:.1f} µV²**."
            )
            st.markdown(
                f":orange-badge[{resting_core.SOURCE_LABEL[resting_core.SOURCE_PUBLISHED]}] "
                f"Skin conductance: **{'lower' if closed else 'higher'}**."
            )
            st.caption(
                "Barry et al. (2007) measured skin conductance in 28 students. This dataset has no "
                "skin conductance channel, so no value is shown for this recording. "
                f"{_resting_ref('barry')}"
            )
            steps = st.columns(4)
            for col, (title, text) in zip(steps, (
                ("1. Seated at rest", "Cap on, electrodes checked, no task."),
                ("2. Eyes closed", "Alpha strongest. Skin conductance lower (published finding)."),
                ("3. Eyes open", "Alpha drops. Skin conductance rises (published finding)."),
                ("4. Compare", "Same person, two conditions, second by second."),
            )):
                col.markdown(f"**{title}**")
                col.caption(text)
            st.caption(
                "Barry and De Blasio (2017) recorded 3 minutes of each, eyes closed first. This "
                "stage uses a 15-second excerpt of each condition from one channel. "
                f"{_resting_ref('barry_2017')}{_ref(SANCHEZ_GAMA_ET_AL_CITATION)}"
            )
            _study_card(resting_core.STUDY, "The study: Barry et al. (2007)")
            _study_card(resting_core.AGEING_NOTE, "Does it hold with age? Barry and De Blasio (2017)")

        with numbers:
            st.write(
                "Each number on this stage, in plain terms, with the range you can expect and "
                "how that range is known."
            )
            graphics = {
                "frequency": resting_state_figure.bands_svg,
                "reactivity": lambda: resting_state_figure.reactivity_svg(react),
                "accuracy": lambda: resting_state_figure.accuracy_svg(odds),
                "skin_conductance": resting_state_figure.skin_conductance_svg,
            }
            threshold_note = resting_core.explain_threshold(react).text
            badge = {
                resting_core.SOURCE_RECORDED: "blue",
                resting_core.SOURCE_PUBLISHED: "orange",
                resting_core.SOURCE_CONVENTION: "gray",
            }
            for m in resting_core.MEASURES:
                with st.expander(f"{m.name}: {m.unit}", expanded=m.key == "reactivity"):
                    st.markdown(f":{badge[m.source]}-badge[{resting_core.SOURCE_LABEL[m.source]}]")
                    if m.key in ("alpha_power", "accuracy"):
                        st.caption(threshold_note)
                    if m.key in graphics:
                        components.html(graphics[m.key](), height=120)
                    elif m.key == "alpha_power":
                        _chart(_alpha_power_strip_spec(result, "Alpha power per 1-second window, this recording"))
                    st.markdown(
                        f"- **What it is:** {m.what_it_is}\n"
                        f"- **How it is computed:** {m.how_computed}\n"
                        f"- **How to read it:** {m.how_to_read}\n"
                        f"- **{_RANGE_LABEL[m.range_kind]}:** {m.expected}\n"
                        f"- **What shifts it:** {m.what_shifts_it} {_resting_ref(*m.refs)}"
                    )
            st.caption(f"{resting_core.BAND_CAVEAT} {_resting_ref(*resting_core.BAND_REFS)}")
            st.caption(
                f"This recording: ratio {react.ratio:.1f}, accuracy {odds.n_correct} of "
                f"{odds.n_trials}, p {'< 0.001' if odds.p_value < 0.001 else f'= {odds.p_value:.3f}'} against guessing. "
                f"{_resting_ref('combrisson')}"
            )

        with dichotomies:
            st.write(
                "Two questions to ask of any measurement, whatever the sensor. Barry et al. (2007) "
                "answers both for EEG: the study is resting, and it found one domain-general change "
                f"(alpha) and several domain-specific ones. {_resting_ref('barry')}"
            )
            axis_labels = {k: f"{a} vs {b}" for k, (a, b) in resting_core.AXIS_POLES.items()}
            chosen = st.segmented_control(
                "Dichotomy", list(axis_labels.values()), default=axis_labels[resting_core.AXIS_STATE],
                key="mmsc_study_axis", label_visibility="collapsed",
            ) or axis_labels[resting_core.AXIS_STATE]
            axis = next(k for k, v in axis_labels.items() if v == chosen)
            first, second = resting_core.AXIS_POLES[axis]
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Modality": d.modality, first: d.first, second: d.second}
                        for d in resting_core.DICHOTOMIES if d.axis == axis
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            refs = sorted({k for d in resting_core.DICHOTOMIES if d.axis == axis for k in d.refs})
            st.caption(f"Sources: {_resting_ref(*refs)}")



# -----------------------------------------------------------------
# Interactive lessons: each one a question, the obstacle, an experiment,
# the formula, and its limitations. Numbers come from core.
# -----------------------------------------------------------------

_ASSUMPTION_BADGE = {"checked": "green", "violated": "red", "unverified": "orange", "assumed": "gray"}


def _assumption_rows(rows: list[tuple[str, str, str]]) -> None:
    for name, status, why in rows:
        st.markdown(f":{_ASSUMPTION_BADGE[status]}-badge[{status}] **{name}.** {why}")


_BODY_FIGURE_HEIGHT = 690


@st.cache_data(show_spinner=False)
def _pipeline_eeg() -> dict:
    """The recorded EEG at each stage, computed by core, for the pipeline drawing."""
    snippet = _load_eeg_snippet()
    rate = intervention_core.SAMPLING_RATE_HZ
    second = int(rate)
    filtered = filtering_core.band_pass(snippet["eyes_closed"], rate).filtered[:second]
    result = intervention_core.classify_eyes_state(snippet["eyes_closed"], snippet["eyes_open"], rate)
    windows = intervention_core.labelled_windows(result)
    first = windows.powers[0]
    return {
        "raw": [float(v) for v in snippet["eyes_closed"][:second:4]],
        "filtered": [float(v) for v in filtered[::4]],
        "powers": list(windows.powers),
        "labels": list(windows.labelled_closed),
        "threshold": windows.threshold,
        "decision_value": first,
        "decision_label": "eyes closed" if windows.labelled_closed[0] else "eyes open",
    }


def _pipeline_figure(modalities) -> None:
    """The pipeline as illustrated stages, showing the signal at each one."""
    notes = {stage: note for _, stage, note in PIPELINE_STAGE_NOTES}
    handoffs = dict(zip(pipeline_core.STAGES, PIPELINE_HANDOFFS))
    n = len(modalities)
    height = 40 + 2 * (50 + 18 * n) + (112 if n > 1 else 72) + 92 + 78 + 86 + 86 + 6 * 20 + 7 * 14 + 150
    components.html(
        pipeline_figures.pipeline_html(
            tuple(modalities), pipeline_core.STAGES, pipeline_core.DEFAULT_CONVERGENCE_STAGE,
            notes, handoffs, PIPELINE_RETURN_HANDOFF, CATEGORY_COLORS, eeg=_pipeline_eeg(),
        ),
        height=int(height * 1.55),
    )


def _body_figure(modalities, perspective: str) -> None:
    """The illustrated person with each modality's sensor, under one perspective."""
    components.html(
        pipeline_figures.body_html(
            tuple(modalities),
            perspective,
            CATEGORY_COLORS,
            agency={m.category: agency_core.agency_profile(m.category) for m in modalities},
        ),
        height=_BODY_FIGURE_HEIGHT
        + (
            # The Agency table adds rows below the figure, drawn at about 1.28x.
            int(pipeline_figures.agency_table_height(len(modalities)) * 1.3)
            if perspective == PERSPECTIVE_AGENCY
            else 0
        ),
    )


def _annotated_formula(mathml: str, symbols: dict[str, str], labels: dict[str, str], height: int = 112) -> None:
    """
    A formula whose symbols can be tapped to show what each one stands for.

    Streamlit's st.latex is static, so the formula is MathML in a small
    embedded panel; each symbol is an <mrow data-sym="key">. Definitions
    come from the lesson's core SYMBOLS. A plain symbol list follows for
    screen readers and in case the panel does not load.
    """
    import html as _html
    import json as _json

    defs = {k: {"label": labels.get(k, k), "text": v} for k, v in symbols.items()}
    page = f"""<div style="font-family:system-ui,sans-serif;background:{SURFACE};color:{INK_SECONDARY};border:1px solid {GRIDLINE};border-radius:8px;padding:10px 14px">
<div style="font-size:22px;text-align:center;color:#1f1e1c">{mathml}</div>
<p id="note" aria-live="polite" style="font-size:13px;margin:8px 0 0;border-left:3px solid {CATEGORY_COLORS['Autonomic']};padding-left:8px">Tap a symbol in the formula to see what it stands for.</p></div>
<style>.sym{{cursor:pointer;border-radius:4px;padding:0 2px}}.sym:hover,.sym:focus{{background:#dbe7f7;outline:none}}.sym.on{{background:#fbd9c4}}</style>
<script>
const D = {_json.dumps(defs)};
const note = document.getElementById("note");
function pick(el) {{
  document.querySelectorAll(".sym").forEach(s => s.classList.toggle("on", s === el));
  const d = D[el.dataset.sym];
  note.innerHTML = "<strong style='color:#1f1e1c'>" + d.label + "</strong> " + d.text;
}}
document.querySelectorAll(".sym").forEach(el => {{
  el.setAttribute("tabindex", "0"); el.setAttribute("role", "button");
  el.setAttribute("aria-label", D[el.dataset.sym].label + ": " + D[el.dataset.sym].text);
  el.addEventListener("click", () => pick(el));
  el.addEventListener("keydown", e => {{ if (e.key === "Enter" || e.key === " ") {{ e.preventDefault(); pick(el); }} }});
}});
</script>"""
    components.html(page, height=height)
    with st.expander("Symbols in this formula"):
        for key, info in defs.items():
            st.markdown(f"- **{_html.escape(info['label'])}**: {info['text']}")


def _sym(key: str, inner: str) -> str:
    return f'<mrow class="sym" data-sym="{key}">{inner}</mrow>'


_CONVOLUTION_MATHML = (
    "<math display='block'>"
    + _sym("h", "<mi>h</mi><mo>(</mo><mi>t</mi><mo>)</mo>")
    + "<mo>=</mo><mo>(</mo>" + _sym("n", "<mi>n</mi>") + _sym("conv", "<mo>∗</mo>") + _sym("k", "<mi>k</mi>")
    + "<mo>)</mo><mo>(</mo>" + _sym("t", "<mi>t</mi>") + "<mo>)</mo><mo>=</mo><mo>∫</mo>"
    + _sym("n", "<mi>n</mi><mo>(</mo><mi>τ</mi><mo>)</mo>") + "<mspace width='0.2em'/>"
    + _sym("k", "<mi>k</mi><mo>(</mo><mi>t</mi><mo>−</mo><mi>τ</mi><mo>)</mo>")
    + "<mspace width='0.2em'/>" + _sym("tau", "<mi>d</mi><mi>τ</mi>") + "</math>"
)
_CONVOLUTION_LABELS = {"h": "h(t)", "n": "n(τ)", "k": "k(t − τ)", "conv": "∗", "tau": "dτ", "t": "t"}

_CORRELATION_MATHML = (
    "<math display='block'>" + _sym("r", "<mi>r</mi>") + "<mo>=</mo><mfrac>"
    + _sym("cov", "<mi>Cov</mi><mo>(</mo><mi>X</mi><mo>,</mo><mi>Y</mi><mo>)</mo>")
    + "<mrow>" + _sym("sigma_x", "<msub><mi>σ</mi><mi>X</mi></msub>") + _sym("sigma_y", "<msub><mi>σ</mi><mi>Y</mi></msub>")
    + "</mrow></mfrac></math>"
)
_CORRELATION_LABELS = {"r": "r", "cov": "Cov(X, Y)", "sigma_x": "σX", "sigma_y": "σY"}

_HEMO_STEP_KEY = "mmsc_lesson_hemo_step"


def _hemo_step(delta: int) -> None:
    last = len(hemo_core.CONVOLUTION_STEPS) - 1
    st.session_state[_HEMO_STEP_KEY] = max(0, min(last, st.session_state.get(_HEMO_STEP_KEY, 0) + delta))


def _render_hemodynamics_lesson() -> None:
    lesson = hemo_core.LESSON
    with st.expander("Lesson: why does fNIRS respond more slowly than EEG?"):
        st.markdown(f"**The question.** {lesson['question']}")
        st.markdown(f"**The obstacle.** {lesson['obstacle']}")

        step = st.session_state.get(_HEMO_STEP_KEY, 0)
        title, text = hemo_core.CONVOLUTION_STEPS[step]
        events = hemo_core.STEP_EVENTS
        parts = hemo_core.response_components(events, duration=30.0)
        result = hemo_core.hemodynamic_response(events, duration=30.0)
        times = result.times

        st.markdown(f"**Step {step + 1} of {len(hemo_core.CONVOLUTION_STEPS)}: {title}.** {text}")
        shown_events = events[: 1 if step < 2 else 2]
        eeg_rows = [{"t": e} for e in shown_events]
        eeg_spec = {
            "data": {"values": eeg_rows},
            "mark": {"type": "rule", "strokeWidth": 3, "color": CATEGORY_COLORS["Neural"]},
            "encoding": {"x": {"field": "t", "type": "quantitative", "title": None, "scale": {"domain": [0, 30]}}},
            "title": {"text": "What EEG sees: each event as it happens", "fontSize": 11, "color": INK_SECONDARY, "fontWeight": "normal"},
            "width": "container", "height": 40, "config": _VEGA_CHART_CONFIG,
        }
        _chart(eeg_spec)

        series = []
        if step >= 1:
            series.append(("Copy of k(t) for event 1", parts[0], "#c4b0e3" if step == 4 else "#8a5cc9", [4, 3] if step == 4 else None))
        if step >= 3:
            series.append(("Copy of k(t) for event 2", parts[1], "#c4b0e3" if step == 4 else "#b07cd9", [4, 3] if step == 4 else None))
        if step == 4:
            series.append(("Sum h(t): what fNIRS sees", result.response, "#5b2a9e", None))
        layers = []
        for name, values, color, dash in series:
            mark = {"type": "line", "strokeWidth": 2.5 if name.startswith("Sum") else 1.8, "color": color}
            if dash:
                mark["strokeDash"] = dash
            layers.append({
                "data": {"values": [{"t": t, "v": v, "s": name} for t, v in zip(times, values)]},
                "mark": mark,
                "encoding": {
                    "x": {"field": "t", "type": "quantitative", "title": "Seconds", "scale": {"domain": [0, 30]}},
                    "y": {"field": "v", "type": "quantitative", "title": "Blood-oxygen response", "scale": {"domain": [-0.3, 2.0]}},
                    "tooltip": [{"field": "s", "title": "Curve"}, {"field": "t", "title": "Seconds", "format": ".1f"},
                                {"field": "v", "title": "Response", "format": ".2f"}],
                },
            })
        if not layers:
            layers.append({"data": {"values": [{"t": 0, "v": 0}]}, "mark": {"type": "point", "opacity": 0},
                           "encoding": {"x": {"field": "t", "type": "quantitative", "title": "Seconds", "scale": {"domain": [0, 30]}},
                                        "y": {"field": "v", "type": "quantitative", "title": "Blood-oxygen response", "scale": {"domain": [-0.3, 2.0]}}}})
        _chart({
            "layer": layers,
            "title": {"text": "What fNIRS sees: the same events, seconds later (model)", "fontSize": 11, "color": INK_SECONDARY, "fontWeight": "normal"},
            "width": "container", "height": 150, "config": _VEGA_CHART_CONFIG,
        })
        back, forward = st.columns(2)
        back.button("Previous step", on_click=_hemo_step, args=(-1,), disabled=step == 0, key="mmsc_hemo_back", width="stretch")
        forward.button("Next step", on_click=_hemo_step, args=(1,), disabled=step == len(hemo_core.CONVOLUTION_STEPS) - 1,
                       key="mmsc_hemo_next", type="primary", width="stretch")
        if step == len(hemo_core.CONVOLUTION_STEPS) - 1:
            st.caption(
                f"Peak {result.first_peak_delay:.1f} s after the first event. The two events, "
                f"{events[1] - events[0]:g} s apart, leave one bump in the blood response, while EEG "
                "kept them apart."
            )

            st.markdown("**The formula.** Tap a symbol.")
            _annotated_formula(_CONVOLUTION_MATHML, hemo_core.SYMBOLS, _CONVOLUTION_LABELS)
            st.caption(
                f"{lesson['formula_plain']} Here k(t) is a double gamma that peaks at 5 s with a small "
                f"undershoot after. {_lesson_ref('boynton', 'friston_1998')}"
            )

        st.markdown("**Limitations.**")
        for item in lesson["limitations"]:
            st.markdown(f"- {item}")
        st.markdown("**Assumptions in this experiment.**")
        _assumption_rows([
            ("Linear response", "violated",
             f"Holds approximately for events at least about 4 s apart. These events are "
             f"{events[1] - events[0]:g} s apart, inside the range where measured responses stop "
             "adding linearly (Glover, 1999), so real data would differ from this sum."),
            ("Response shape", "assumed", "A standard double gamma; real shapes vary by region and person."),
            ("fNIRS follows the BOLD shape", "assumed", "A common modeling choice for oxygenated hemoglobin."),
        ])
        st.caption(f"Sources: {_lesson_ref('glover', 'lindquist', 'yucel', 'logothetis')}")


def _scatter_spec(sim: corr_core.Simulation, title: str, color: str) -> dict:
    rows = [{"x": x, "y": y} for x, y in zip(sim.x, sim.y)]
    return {
        "data": {"values": rows},
        "mark": {"type": "point", "filled": True, "size": 18, "opacity": 0.5, "color": color},
        "encoding": {
            "x": {"field": "x", "type": "quantitative", "title": "Signal X"},
            "y": {"field": "y", "type": "quantitative", "title": "Signal Y"},
            "tooltip": [{"field": "x", "format": ".2f"}, {"field": "y", "format": ".2f"}],
        },
        "title": {"text": title, "fontSize": 11, "color": INK_SECONDARY, "fontWeight": "normal"},
        "width": "container", "height": 180, "config": _VEGA_CHART_CONFIG,
    }


def _render_correlation_lesson() -> None:
    lesson = corr_core.LESSON
    with st.expander("Lesson: how can correlation mislead?"):
        st.markdown(f"**The question.** {lesson['question']}")
        st.markdown(f"**The obstacle.** {lesson['obstacle']}")

        st.markdown(
            "**Compare.** On the left, a third source drives both signals and they share no link. "
            "On the right, X drives Y directly, with its strength set so the expected correlation "
            "matches."
        )
        # One fixed example makes the point; a slider would only move both
        # values together. Shared-input strength 2 gives an expected r of 0.8.
        a = 2.0
        common = corr_core.simulate(corr_core.WIRING_COMMON, a)
        b = corr_core.matching_direct_strength(a)
        direct = corr_core.simulate(corr_core.WIRING_DIRECT, b, seed=corr_core.DEFAULT_SEED + 1)
        left, right = st.columns(2)
        with left:
            _chart(_scatter_spec(common, "No link: a third source drives both", "#8a5cc9"))
            st.metric("Correlation r", f"{common.r:.2f}", help=f"Expected {common.expected_r:.2f}")
        with right:
            _chart(_scatter_spec(direct, "Direct link: X drives Y", CATEGORY_COLORS["Neural"]))
            st.metric("Correlation r", f"{direct.r:.2f}", help=f"Expected {direct.expected_r:.2f}")
        st.caption(
            f"Both panels: {len(common.x)} simulated samples, noise SD {common.noise_sd:g}. The two "
            "wirings give the same expected r, so r alone cannot say which one produced it. Small "
            "differences between the two values are sampling noise."
        )

        with st.container(border=True):
            st.markdown("**Two electrodes, one source (volume conduction).**")
            share = 0.75
            shared = corr_core.simulate(corr_core.WIRING_SHARED, 1.0, second_weight=share)
            st.markdown(
                f"With one source under electrode 1 and {share:.0%} of it reaching electrode 2, the two "
                f"electrodes correlate at **r = {shared.r:.2f}** (expected {shared.expected_r:.2f}), "
                "with only one source present."
            )

        st.markdown("**The formula.** Tap a symbol.")
        _annotated_formula(_CORRELATION_MATHML, corr_core.SYMBOLS, _CORRELATION_LABELS, height=130)
        st.caption(lesson["formula_plain"])
        st.latex(r"\text{common input: } r = \frac{a^2}{a^2 + s^2} \qquad \text{direct link: } r = \frac{b}{\sqrt{b^2 + s^2}}")
        st.caption("Expected values for unit-variance sources and noise of standard deviation s, which the panels above check.")

        st.markdown("**Limitations.**")
        for item in lesson["limitations"]:
            st.markdown(f"- {item}")
        st.caption(f"Sources: {_lesson_ref(*lesson['refs'])}")


def _render_electrode_map() -> None:
    """The 10-20 map, with any electrode label read out."""
    with st.container(border=True):
        st.markdown("**Where the Sensor Sits**")
        site = ANATOMY.recording_electrode
        st.write(
            "Electrode positions follow the 10-20 system, which places each electrode at 10 "
            "or 20 percent of the distance across the head. Each label tells you the region "
            "and the side. This map uses the modern names T7, T8, P7, and P8 for the classic "
            f"T3, T4, T5, and T6. {_anatomy_ref(*site.refs)}"
        )
        labels = [e.label for e in ANATOMY.electrodes]
        label = st.selectbox(
            "Electrode to read",
            options=labels,
            index=labels.index(site.label),
            key="mmsc_anatomy_electrode",
        )
        chosen = neuroanatomy_core.electrode_site(label)
        map_col, side_col = st.columns(2)
        with map_col:
            components.html(neuroanatomy_figure.electrode_map_svg(chosen.label), height=310)
        with side_col:
            components.html(neuroanatomy_figure.side_view_svg("eeg", electrode_only=True), height=310)
        region = neuroanatomy_core.system(chosen.system).name.lower()
        note = " This stage's recording comes from here." if chosen.label == site.label else ""
        st.write(f"**{chosen.label}.** {chosen.naming_note} It sits over the {region}.{note}")
        caveat(f"{ANATOMY.electrode_caveat} {_anatomy_ref(*ANATOMY.electrode_caveat_refs)}")
















def _frontier_spec(
    modalities,
    weighted_cost: tradeoff_core.WeightedCostResult,
    frontier: tradeoff_core.GainCostFrontierResult,
    cost_title: str = "Cost based on your priorities (lower is better)",
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
                "title": "What this modality adds (higher is better)",
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
                {"field": "gain", "type": "quantitative", "title": "What it adds", "format": ".2f"},
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
                "title": "What this modality adds, on its own (higher is better)",
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


_CONDITION_COLORS = {"Eyes closed": CATEGORY_COLORS["Neural"], "Eyes open": MUTED}
_CONDITION_LABEL = {"eyes_closed": "Eyes closed", "eyes_open": "Eyes open"}


def _load_eeg_snippet() -> dict[str, np.ndarray]:
    """
    Real bundled excerpt: one subject (sub-5), one channel (O1), 15
    seconds per condition from OpenNeuro ds005420 (CC0). See
    modules/signal_pipeline/README.md for the extraction details.
    """

    frame = pd.read_csv(SAMPLE_DIR / "ds005420_sub-5_O1_eyes_snippet.csv")
    return {
        condition: frame.loc[frame["condition"] == condition, "voltage_uv"].to_numpy()
        for condition in ("eyes_closed", "eyes_open")
    }


def _signal_segment_spec(
    eyes_closed: np.ndarray, eyes_open: np.ndarray, sampling_rate: float, title: str
) -> dict:
    """First 3 seconds of each condition, plotted by real elapsed time -
    the actual retained/degraded waveform, not a stylized pictograph."""

    seconds_shown = 3.0
    n = max(int(sampling_rate * seconds_shown), 1)
    rows = []
    for key, segment in (("eyes_closed", eyes_closed), ("eyes_open", eyes_open)):
        rows.extend(
            {"t": i / sampling_rate, "voltage_uv": float(v), "condition": _CONDITION_LABEL[key]}
            for i, v in enumerate(segment[:n])
        )

    return {
        "data": {"values": rows},
        "mark": {"type": "line", "strokeWidth": 1},
        "encoding": {
            "x": {"field": "t", "type": "quantitative", "title": "Seconds"},
            "y": {"field": "voltage_uv", "type": "quantitative", "title": "µV"},
            "tooltip": [
                {"field": "condition", "title": "Condition"},
                {"field": "t", "title": "Seconds", "format": ".3f"},
                {"field": "voltage_uv", "title": "Voltage (µV)", "format": ".1f"},
            ],
            "color": {
                "field": "condition",
                "type": "nominal",
                "scale": {
                    "domain": list(_CONDITION_COLORS),
                    "range": list(_CONDITION_COLORS.values()),
                },
                "legend": {"title": None, "orient": "top"},
            },
        },
        "title": {"text": title, "fontSize": 11, "color": INK_SECONDARY, "fontWeight": "normal"},
        "width": "container",
        "height": 120,
        "config": _VEGA_CHART_CONFIG,
    }


def _alpha_power_strip_spec(result: "intervention_core.ClassificationResult", title: str) -> dict:
    """One dot per 1-second window's alpha power, split by condition -
    how separated the two classes actually are, not just the accuracy
    number that separation implies."""

    rows = [
        {"condition": "Eyes closed", "power": p} for p in result.alpha_power_eyes_closed
    ] + [{"condition": "Eyes open", "power": p} for p in result.alpha_power_eyes_open]

    # The threshold the labels come from: the midpoint of the two means.
    threshold_layer = {
        "data": {"values": [{"threshold": result.threshold, "label": f"threshold {result.threshold:.1f} µV² (midpoint of means)"}]},
        "layer": [
            {"mark": {"type": "rule", "strokeDash": [4, 3], "color": INK_SECONDARY},
             "encoding": {"x": {"field": "threshold", "type": "quantitative"}}},
            {"mark": {"type": "text", "align": "left", "dx": 4, "dy": -4, "baseline": "bottom", "fontSize": 10, "color": INK_SECONDARY, "y": 0},
             "encoding": {"x": {"field": "threshold", "type": "quantitative"}, "text": {"field": "label"}}},
        ],
    }

    points = {
        "data": {"values": rows},
        "mark": {"type": "point", "filled": True, "size": 80, "opacity": 0.75},
        "encoding": {
            "y": {"field": "condition", "type": "nominal", "title": None},
            "x": {
                "field": "power",
                "type": "quantitative",
                "title": "Alpha power (µV²) per 1-second window",
            },
            "tooltip": [
                {"field": "condition", "title": "Condition"},
                {"field": "power", "title": "Alpha power (µV²)", "format": ".1f"},
            ],
            "color": {
                "field": "condition",
                "type": "nominal",
                "scale": {
                    "domain": list(_CONDITION_COLORS),
                    "range": list(_CONDITION_COLORS.values()),
                },
                "legend": None,
            },
        },
    }

    return {
        "layer": [points, threshold_layer],
        "title": {"text": title, "fontSize": 11, "color": INK_SECONDARY, "fontWeight": "normal"},
        "width": "container",
        "height": 100,
        "config": _VEGA_CHART_CONFIG,
    }


st.set_page_config(
    page_title="OpenMeasure · Multimodal Signal Convergence",
    page_icon=":material/hub:",
    layout="centered",
)

st.title("Multimodal Signal Convergence")
st.caption(
    "This journey starts with one neurotech signal, EEG "
    "(electroencephalography, a scalp-recorded brain signal), and adds "
    "modalities one at a time to a general Signals -> Sensors -> "
    "Processing -> Inference -> Decision -> Action -> Feedback pipeline."
)
st.caption(
    "A guided illustrative walkthrough: each stage unlocks after you "
    "make a decision or inspect its consequence."
)
st.caption(
    "Written for early-career researchers learning to validate a study "
    "design, so every stage states its reasoning in plain language and "
    "numbers its sources against the References list at the end."
)

render_data_handling_summary(disclosure_for("pages/Multimodal_Signal_Convergence.py"))

# One stage in the workspace at a time, and the rail is how you move.
# Reading this used to mean scrolling past every earlier stage, which
# made seven decisions read as seven sections of a report.
#
# No gates. Every stage here is something to read, build, or weigh, and
# the tradeoff stage already says for itself when it has no modality to
# compare, which is more use than a locked rail entry.
WORKSPACE = StageWorkspace(
    session_key="mmsc",
    stages=(
        Stage("question", "Question"),
        Stage("pipeline", "Pipeline"),
        Stage("modalities", "Modalities"),
        Stage("convergence", "Convergence"),
        Stage("tradeoff", "Tradeoff"),
        Stage("real_signal", "A Recorded Signal"),
        Stage("decision", "Decision"),
    ),
)

stage = WORKSPACE.render_rail()
WORKSPACE.render_review_notice()

st.divider()

# -----------------------------------------------------------------
# 1. Research question
# -----------------------------------------------------------------

if stage == STAGE_RESEARCH_QUESTION:
    # The tracker used to draw this, clearing the modality selection,
    # the three cost weights, the perspective, and the nested tradeoff
    # sequence's own position with it.
    if st.button("Restart study"):
        for key in [
            name for name in st.session_state
            if str(name).startswith("mmsc")
        ] + [
            SELECTED_MODALITIES_KEY,
            PRIVACY_WEIGHT_KEY,
            SECURITY_WEIGHT_KEY,
            AGENCY_WEIGHT_KEY,
            PERSPECTIVE_KEY,
            TRADEOFF_STEP_KEY,
        ] + list(KEPT.values()):
            st.session_state.pop(key, None)
        st.rerun()


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
        + _ref(DE_MONTJOYE_CITATION)
    )

    rq_col1, rq_col2 = st.columns(2)
    with rq_col1:
        st.badge(
            "What this modality adds",
            icon=":material/trending_up:",
            color="blue",
        )
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
            + _ref(IENCA_ANDORNO_CITATION)
        )
        st.write(
            "Security is a separate concern from privacy, and "
            "brain-computer interface (BCI) research surveys it across the "
            "full \"BCI cycle\": acquisition hardware, "
            "on-device processing, transmission between devices, the machine "
            "learning models decoding the signal, and the applications or "
            "cloud services consuming the result. More traditional points of "
            "exposure in that cycle, such as wireless communication or cloud "
            "storage, can adopt existing methods (encryption, secure "
            "communication protocols, differential privacy) directly; the "
            "closer to the brain a surface is, the more those methods need "
            "to be modified or replaced entirely."
            + _ref(BAGLEY_ET_AL_CITATION)
        )


# -----------------------------------------------------------------
# 2. Build the pipeline
# -----------------------------------------------------------------

if stage == STAGE_BUILD_PIPELINE:
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
    _body_figure((baseline,), PERSPECTIVE_BUILD)
    st.caption(
        f"Signal examples: {baseline.signal_examples}. "
        f"{_ref(baseline.citation)}"
    )

    _render_neuroanatomy_explorer()

    st.caption(f"How {baseline.name}'s signal travels through the pipeline:")
    _pipeline_figure((baseline,))

    st.write(
        "**Interpretation**: a pipeline with one modality already "
        "has every stage a multimodal pipeline will have. What changes "
        "when a modality is added is not the shape of the pipeline, but "
        "how many signals converge into it, and what that convergence "
        "costs."
    )


# -----------------------------------------------------------------
# 3. Add modalities
# -----------------------------------------------------------------

if stage == STAGE_ADD_MODALITIES:
    section_header(
        "3. Add Modalities",
        "Add categories one at a time; the diagram below updates as you do.",
    )

    all_modalities = _load_modalities()
    modalities_by_name = {m.name: m for m in all_modalities}
    baseline = modalities_by_name[BASELINE_MODALITY_NAME]
    addable = tuple(m for m in all_modalities if m.name != BASELINE_MODALITY_NAME)

    if SELECTED_MODALITIES_KEY not in st.session_state:
        st.session_state[SELECTED_MODALITIES_KEY] = list(
            _held(SELECTED_MODALITIES_KEY, [])
        )

    selected_names = _hold(
        SELECTED_MODALITIES_KEY,
        list(
            st.multiselect(
                "Modalities to add to the EEG-only pipeline",
                options=[m.name for m in addable],
                key=SELECTED_MODALITIES_KEY,
            )
        ),
    )

    current_modalities = (baseline,) + tuple(
        modalities_by_name[name] for name in selected_names
    )
    current_pipeline = pipeline_core.build_pipeline(current_modalities)

    st.caption(
        "Perspective - same modalities, different question about them:"
    )
    if PERSPECTIVE_KEY not in st.session_state:
        st.session_state[PERSPECTIVE_KEY] = _held(
            PERSPECTIVE_KEY, PERSPECTIVE_BUILD
        )

    perspective = _hold(
        PERSPECTIVE_KEY,
        st.segmented_control(
            "Perspective",
            options=PERSPECTIVES,
            key=PERSPECTIVE_KEY,
            label_visibility="collapsed",
        )
        or PERSPECTIVE_BUILD,
    )
    st.caption(PERSPECTIVE_DESCRIPTIONS[perspective])

    _body_figure(current_modalities, perspective)
    if perspective == PERSPECTIVE_BUILD:
        st.caption(
            "Dashed lines show every selected signal feeding the same "
            "pipeline entry point."
        )
    elif perspective == PERSPECTIVE_PRIVACY:
        st.caption(
            "The dashed outline is the edge of the person. Each "
            "body-sensed signal crosses it at the marked point, and a "
            "modality drawn outside it was never sensed from the body at "
            "all."
        )
    else:
        st.caption(
            "Each path carries three marks: can the person stop the "
            "collection, does each collection need their action, and can "
            "they correct or retract it. Each is answered only where the "
            "cited control text answers it; a question mark means the "
            "sources do not say. The illustrative agency cost rating is "
            "not drawn here."
        )

    st.caption("How those signals travel through the pipeline:")
    _pipeline_figure(current_modalities)
    st.caption(
        "Tap a stage for its note. With more than one modality, the "
        "combined row in Processing marks where the signals meet."
    )

    for name in selected_names:
        added = modalities_by_name[name]
        with st.expander(f"{added.name} - what this adds"):
            _body_figure((added,), perspective)
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
            st.caption(f"{added.notes} {_ref(added.citation)}")

    caveat(
        "These ratings are illustrative, author-assigned scores informed "
        "by the cited literature's directional findings, not values any "
        "cited source reports directly - see the page docstring and "
        "modules/signal_pipeline/README.md."
    )

    st.write(
        "**Interpretation**: adding a modality here is visually free - "
        "gain and cost are drawn as independent per-modality ratings, "
        "with no link enforced between them."
    )

    st.write(
        "**Limitations**: each added node is easy to draw. What "
        "the diagram does not show on its own is whether the resulting "
        "pipeline's combined interpretive value is worth its combined "
        "cost - that comparison is Step 5."
    )

    _render_hemodynamics_lesson()


# -----------------------------------------------------------------
# 4. Examine convergence
# -----------------------------------------------------------------

if stage == STAGE_CONVERGENCE:
    section_header(
        "4. Examine Convergence",
        "Where the added signals actually meet, and what meets there.",
    )

    all_modalities = _load_modalities()
    modalities_by_name = {m.name: m for m in all_modalities}
    baseline = modalities_by_name[BASELINE_MODALITY_NAME]
    selected_names = _held(SELECTED_MODALITIES_KEY, [])
    current_modalities = (baseline,) + tuple(
        modalities_by_name[name] for name in selected_names
    )
    current_pipeline = pipeline_core.build_pipeline(current_modalities)

    convergence = pipeline_core.compute_convergence(current_pipeline, stage="Inference")

    _pipeline_figure(current_modalities)
    st.caption("The same pipeline from Step 3, for reference while reading convergence below.")

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

    perspective = _held(PERSPECTIVE_KEY, PERSPECTIVE_BUILD) or PERSPECTIVE_BUILD
    st.caption(f"Viewed from Step 3's current perspective ({perspective}):")
    st.write(CONVERGENCE_PERSPECTIVE_TEXT[perspective])

    _render_correlation_lesson()

    st.write(
        "**Implications**: convergence is where a multimodal "
        "pipeline's distinctive claim to power - and its distinctive "
        "risk - both live. The model can infer something from the "
        "combination that it could not infer from any signal alone, "
        "which is exactly why the combination's cost has to be assessed "
        "as a combination, not signal by signal."
    )


# -----------------------------------------------------------------
# 5. Weigh the tradeoff
# -----------------------------------------------------------------

if stage == STAGE_WEIGH_TRADEOFF:
    section_header(
        "5. Weigh the Tradeoff",
        "More modalities are not automatically better",
    )

    # The decision first, in words that do not require the vocabulary
    # the rest of the stage uses. A reader who meets weighted costs, a
    # frontier and dominated points before meeting the question has to
    # reverse-engineer what they are deciding.
    st.write(
        "Adding a signal can provide new information, but it can also "
        "introduce new costs. Here you can explore that tradeoff. First "
        "compare how much each modality might add, then consider "
        "privacy, security, and participant agency, and finally choose "
        "how much each cost matters for your study."
    )

    st.caption(
        "A modality is useful when the information it adds justifies the "
        "costs it introduces. Which costs count is your judgment, and "
        "the chart follows it rather than settling it."
    )

    all_modalities = _load_modalities()
    modalities_by_name = {m.name: m for m in all_modalities}
    baseline = modalities_by_name[BASELINE_MODALITY_NAME]
    selected_names = _held(SELECTED_MODALITIES_KEY, [])
    current_modalities = (baseline,) + tuple(
        modalities_by_name[name] for name in selected_names
    )

    if len(current_modalities) < 2:
        st.info(
            "Add at least one modality in Step 3 to compare gain against "
            "cost across more than one signal."
        )
    else:
        tradeoff_step = TRADEOFF_TRACKER.current()

        st.caption(
            "One thing at a time: what each modality adds, then each "
            "cost on its own, then the picture under your own "
            "priorities. An illustrative synthesis built on the ratings "
            "from Step 3, grounded where a measured result is "
            "available."
        )

        _chart(_gain_bar_spec(current_modalities), width="stretch")
        st.write(
            "**Reading this**: further right means a modality is rated "
            "as adding more independent interpretive value on its own. "
            "Nothing here yet says anything about what it costs."
        )

        if any(m.category in ("Autonomic", "Muscular") for m in current_modalities):
            with st.expander("See the research behind these ratings"):
                st.info(
                    "A measured result behind these Autonomic/Muscular "
                    "ratings: on DEAP's own single-trial emotion classifiers, "
                    "EEG alone scored 0.563 F1 and its bundled peripheral "
                    "channels (ECG - heart electrical activity, EMG - muscle "
                    "activity, GSR - skin conductance, plus respiration, "
                    "skin temperature, and eye movement, not separable into "
                    "this page's Autonomic and Muscular categories "
                    "individually) scored 0.608 F1 on valence - not a "
                    "statistically reliable difference (p=0.41). Fusing "
                    "peripheral with a third modality this page does not "
                    "model (multimedia content analysis) reached 0.652 F1, "
                    "the only fusion result in the paper that was a "
                    "statistically reliable improvement (p=0.025); fusing "
                    "all three "
                    "available modalities together did not improve on that "
                    "two-modality result. The paper's own conclusion: fusion "
                    "generally helped, but only slightly, and rarely enough "
                    "to be a statistically reliable improvement."
                    + _ref(KOELSTRA_CITATION)
                )

        if tradeoff_step < TRADEOFF_ADD_PRIVACY:
            if st.button("Now add privacy cost", type="primary"):
                TRADEOFF_TRACKER.advance_to(TRADEOFF_ADD_PRIVACY)

        if tradeoff_step >= TRADEOFF_ADD_PRIVACY:
            st.divider()
            privacy_only_cost = tradeoff_core.combine_costs(current_modalities, 1.0, 0.0, 0.0)
            privacy_only_frontier = tradeoff_core.compute_gain_cost_frontier(
                current_modalities, privacy_only_cost
            )
            _chart(
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
                    TRADEOFF_TRACKER.advance_to(TRADEOFF_ADD_SECURITY)

        if tradeoff_step >= TRADEOFF_ADD_SECURITY:
            st.divider()
            privacy_security_cost = tradeoff_core.combine_costs(current_modalities, 0.5, 0.5, 0.0)
            privacy_security_frontier = tradeoff_core.compute_gain_cost_frontier(
                current_modalities, privacy_security_cost
            )
            _chart(
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
                with st.expander("See the research behind these ratings"):
                    st.info(
                        "A demonstrated security threat behind Neural's "
                        "rating: researchers showed that radio-frequency "
                        "signals from a remote antenna can be picked up by EEG "
                        "electrode wires acting as unintended antennas and "
                        "injected into the acquisition hardware as fabricated "
                        "brain activity, enough to take control of a "
                        "brain-computer interface without physical contact "
                        "with the device. Separately, EEG's own activity "
                        "patterns are distinctive enough to identify who "
                        "produced them, so a device collecting EEG for one "
                        "purpose is, in effect, also collecting a biometric "
                        "identifier as a byproduct - the same signal serves "
                        "both purposes at once."
                        + _ref(
                            ARMENGOL_URPI_ET_AL_CITATION,
                            BAGLEY_ET_AL_CITATION,
                        )
                    )

            if tradeoff_step < TRADEOFF_ADD_AGENCY:
                if st.button("Now add agency cost, and set your own weights", type="primary"):
                    TRADEOFF_TRACKER.advance_to(TRADEOFF_ADD_AGENCY)

        if tradeoff_step >= TRADEOFF_ADD_AGENCY:
            st.divider()
            st.caption(
                "The full picture: all three costs, weighted however you "
                "choose. Weights are normalized automatically, so they do "
                "not need to add up to anything in particular."
            )
            w1, w2, w3 = st.columns(3)
            if PRIVACY_WEIGHT_KEY not in st.session_state:
                st.session_state[PRIVACY_WEIGHT_KEY] = float(
                    _held(PRIVACY_WEIGHT_KEY, 0.34)
                )

            privacy_weight = _hold(
                PRIVACY_WEIGHT_KEY,
                w1.slider("Privacy weight", 0.0, 1.0, step=0.01, key=PRIVACY_WEIGHT_KEY),
            )
            if SECURITY_WEIGHT_KEY not in st.session_state:
                st.session_state[SECURITY_WEIGHT_KEY] = float(
                    _held(SECURITY_WEIGHT_KEY, 0.33)
                )

            security_weight = _hold(
                SECURITY_WEIGHT_KEY,
                w2.slider("Security weight", 0.0, 1.0, step=0.01, key=SECURITY_WEIGHT_KEY),
            )
            if AGENCY_WEIGHT_KEY not in st.session_state:
                st.session_state[AGENCY_WEIGHT_KEY] = float(
                    _held(AGENCY_WEIGHT_KEY, 0.33)
                )

            agency_weight = _hold(
                AGENCY_WEIGHT_KEY,
                w3.slider("Agency weight", 0.0, 1.0, step=0.01, key=AGENCY_WEIGHT_KEY),
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

            _chart(
                _frontier_spec(current_modalities, weighted_cost, frontier), width="stretch"
            )
            st.caption(
                "A faded modality has no advantage under your current "
                "priorities: another option adds at least as much "
                "information at the same or lower overall cost. That is "
                "what \"dominated\" means on a chart like this."
            )

            for m in current_modalities:
                if not frontier.is_efficient[m.name]:
                    flagged_item_note(
                        m.name,
                        "Another modality here adds at least as much and "
                        "costs no more under the priorities you set, so "
                        "these ratings give no reason to prefer this one. "
                        "Change the priorities and that can change.",
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
                "**Implications**: which modalities look worth adding "
                "depends on the priorities you set, not on the ratings "
                "alone. One that has no advantage under your current "
                "priorities can be the obvious choice under different "
                "ones."
            )

            st.write(
                "**What to inspect**: set the privacy weight to 1.0 and "
                "the others to 0.0, then reverse it. Which modalities "
                "fade changes as you do, which is the point: this chart "
                "maps what the ratings imply under a given set of "
                "priorities, and does not deliver a verdict on any "
                "modality."
            )


# -----------------------------------------------------------------
# 6. Protect a real signal
# -----------------------------------------------------------------

if stage == STAGE_REAL_SIGNAL:
    section_header(
        "6. Protect a Recorded Signal",
        "Apply one protective measure to an EEG recording from a public dataset, then "
        "re-run the same inference to see what that protection actually "
        "costs it.",
    )

    st.markdown(
        ":material/psychology: Neural source &rarr; "
        ":material/sensors: Acquisition &rarr; "
        "**:material/tune: Processing (intervene here)** &rarr; "
        ":material/insights: Inference (re-run) &rarr; "
        ":material/bolt: Action &rarr; "
        ":material/cloud: Storage/Sharing"
    )
    st.caption(
        "Where this stage sits in the Signals -> ... -> Feedback pipeline "
        "above: each intervention below is applied before Processing, "
        "then Inference is re-run on the changed signal."
    )

    st.write(
        "This stage uses a recorded 15-second-per-condition excerpt of one "
        "occipital channel (O1, a scalp position at the back of the "
        "head) from one subject in a public dataset, licensed CC0 (no "
        "restrictions on reuse), organized in BIDS (Brain Imaging Data "
        "Structure, the standard file layout for neuroimaging datasets) "
        "- alternating eyes-closed and eyes-open resting EEG. Unlike "
        "every rating in this journey so far, the numbers below are "
        "measured, not author-assigned."
        + _ref(SANCHEZ_GAMA_ET_AL_CITATION)
    )
    _render_electrode_map()
    _render_resting_study()

    st.write(
        "Occipital alpha power (8-13 Hz) rises when the eyes close and "
        "falls when they open, a long-established EEG finding: this is "
        "the physiological signal the baseline below detects, not something "
        "fitted to this specific excerpt."
        + _ref(BARRY_ET_AL_CITATION)
    )
    st.write(
        "Denning, Matsuoka, & Kohno (2009) coined *neurosecurity*: "
        "protecting a neural device's confidentiality, integrity, and "
        "availability to preserve a person's neural mechanisms, neural "
        "computation, and free will. This stage applies that same "
        "privacy/security/agency vocabulary to one recorded signal instead "
        "of an illustrative rating."
        + _ref(DENNING_ET_AL_CITATION)
    )
    with st.expander("Using data like this with other OpenMeasure tools"):
        st.write(
            "Single-subject resting-state EEG like this supports "
            "within-person state-classification questions (does a "
            "signal distinguish two conditions for one person), not "
            "between-group or clinical claims, which need many subjects "
            "and a different design. To reuse a channel like this "
            "elsewhere in OpenMeasure, it needs the same shape "
            "Time-Series QA expects: a timestamp or sample-index column "
            "and one value column per channel, at a known, constant "
            "sampling rate."
        )

    snippet = _load_eeg_snippet()
    eyes_closed_raw = snippet["eyes_closed"]
    eyes_open_raw = snippet["eyes_open"]

    baseline_result = intervention_core.classify_eyes_state(
        eyes_closed_raw, eyes_open_raw, intervention_core.SAMPLING_RATE_HZ
    )

    st.markdown("**Baseline: infer eyes-open/closed from alpha power alone**")
    _chart(
        _signal_segment_spec(
            eyes_closed_raw, eyes_open_raw, intervention_core.SAMPLING_RATE_HZ,
            "First 3 seconds of the original recording, by condition",
        ),
        width="stretch",
    )
    _chart(
        _alpha_power_strip_spec(
            baseline_result, "Alpha power per 1-second window, original signal"
        ),
        width="stretch",
    )
    st.metric("Baseline classification accuracy (in-sample)", f"{baseline_result.accuracy:.0%}")
    held_out = intervention_core.leave_one_out_accuracy(baseline_result)
    st.caption(
        "In-sample: the threshold was set from the same windows it labels. "
        f"Labelling each window with a threshold set from all the others "
        f"(leave-one-out) gives {held_out.n_correct} of {held_out.n_windows}. "
        "Both use consecutive seconds from one person, so neither predicts "
        "accuracy for a new session or a new person."
    )
    st.caption(
        f"{baseline_result.n_windows_eyes_closed} eyes-closed and "
        f"{baseline_result.n_windows_eyes_open} eyes-open 1-second "
        "windows, separated by a simple midpoint-of-class-means "
        "threshold - not a fitted model."
    )

    st.divider()
    st.markdown("**Choose one intervention to apply to the signal before it is retained or shared**")

    intervention_choice = st.radio(
        "Intervention",
        options=("metadata_removal", "noise", "temporal_degradation"),
        format_func=lambda k: {
            "metadata_removal": "Metadata removal",
            "noise": "Additive signal noise",
            "temporal_degradation": "Temporal degradation (coarser sampling)",
        }[k],
        key="real_signal_intervention_choice",
    )

    if intervention_choice == "metadata_removal":
        example_fields = {
            "device_serial_number": "SN-48213",
            "recording_timestamp": "2024-03-11T09:42:00Z",
            "subject_session_code": "sub-5_ses-01",
            "task": "eyes_closed",
        }
        stripped_fields = intervention_core.strip_metadata(example_fields)
        field_cols = st.columns(2)
        with field_cols[0]:
            st.caption("Before")
            st.json(example_fields)
        with field_cols[1]:
            st.caption("After")
            st.json(dict(stripped_fields))
        st.caption(
            "Illustrative of the kind of fields a recording/sharing "
            "pipeline attaches - not fields present in the bundled CSV "
            "itself, whose own BIDS sidecars are already mostly blank."
        )

        intervened_closed, intervened_open = eyes_closed_raw, eyes_open_raw
        intervened_rate = intervention_core.SAMPLING_RATE_HZ
        intervention_summary = (
            "Metadata removal never touches the signal, so the "
            "classification below is identical to the baseline by "
            "construction."
        )

    elif intervention_choice == "noise":
        noise_sd = st.slider(
            "Noise standard deviation (microvolts)", 0.0, 300.0, 80.0, step=10.0
        )
        st.caption(
            "Exploratory signal perturbation, not a differential-privacy "
            "mechanism: this carries no formal privacy guarantee, only an "
            "observed effect on this one inference. The result below is one "
            "random draw of noise; at 50 µV, 200 other draws ranged from "
            "about 60% to 83% accuracy, so treat a single draw as an example."
        )
        intervened_closed = intervention_core.apply_noise(eyes_closed_raw, noise_sd, seed=0)
        intervened_open = intervention_core.apply_noise(eyes_open_raw, noise_sd, seed=1)
        intervened_rate = intervention_core.SAMPLING_RATE_HZ
        intervention_summary = (
            f"At SD={noise_sd:.0f} µV, noise degrades this inference "
            "smoothly: rising noise moves accuracy toward chance rather "
            "than failing at one sharp threshold. This measures eyes-state "
            "decoding only; identification was not tested."
        )

    else:
        degradation_factor = st.select_slider(
            "Downsampling factor", options=(1, 2, 5, 10, 20, 40), value=10
        )
        st.caption(
            "An anti-aliased downsample (a zero-phase low-pass filter, "
            "then keeping one sample in every factor): it removes what a "
            "lower acquisition rate would never have captured. The filter "
            "edge sits at the new Nyquist frequency, so frequencies just "
            "below it are partly attenuated and a little energy just above "
            "it folds back. At factor 20 (25 Hz) that touches the top of "
            "the alpha band: 12 Hz keeps about half its power."
        )
        intervened_closed, intervened_rate = intervention_core.apply_temporal_degradation(
            eyes_closed_raw, intervention_core.SAMPLING_RATE_HZ, degradation_factor
        )
        intervened_open, _ = intervention_core.apply_temporal_degradation(
            eyes_open_raw, intervention_core.SAMPLING_RATE_HZ, degradation_factor
        )
        intervention_summary = (
            "Temporal degradation is a cliff, not a smooth cost: accuracy "
            "holds close to baseline until the effective sampling rate can "
            "no longer resolve the 8-13 Hz alpha band, then the inference "
            "fails outright. This measures one inference only. Slower "
            "rhythms survive a lower rate, and protection against "
            "identifying the person was not tested."
        )

    st.divider()
    st.markdown("**After the intervention**")
    _chart(
        _signal_segment_spec(
            intervened_closed, intervened_open, intervened_rate,
            "First 3 seconds of the intervened signal, by condition",
        ),
        width="stretch",
    )

    try:
        intervened_result = intervention_core.classify_eyes_state(
            intervened_closed, intervened_open, intervened_rate
        )
    except ValueError as error:
        st.error(
            f"Classification failed on the intervened signal: {error} "
            "Past this point, the intervention has removed the "
            "information the inference needs, not just degraded it."
        )
    else:
        _chart(
            _alpha_power_strip_spec(
                intervened_result, "Alpha power per 1-second window, intervened signal"
            ),
            width="stretch",
        )
        before_after_cols = st.columns(2)
        before_after_cols[0].metric(
            "Accuracy before (in-sample)", f"{baseline_result.accuracy:.0%}"
        )
        change = intervened_result.accuracy - baseline_result.accuracy
        unchanged = round(change * 100) == 0
        # A change that rounds to 0% gets no arrow or colour: a green
        # "up +0%" read as an improvement when nothing moved.
        before_after_cols[1].metric(
            "Accuracy after (in-sample)",
            f"{intervened_result.accuracy:.0%}",
            delta="no change" if unchanged else f"{change:+.0%}",
            delta_color="off" if unchanged else "normal",
            delta_arrow="off" if unchanged else "auto",
        )

    st.write(f"**Implications**: {intervention_summary}")
    caveat(
        "One subject, one channel, one fixed inference task: this shows "
        "a mechanism, not a population-level estimate of how costly any "
        "of these interventions would be for a specific deployment or a "
        "different downstream use of the same signal."
    )


# -----------------------------------------------------------------
# 7. Research decision
# -----------------------------------------------------------------

if stage == STAGE_RESEARCH_DECISION:
    section_header(
        "7. Research Decision",
        "Which added signal would you keep?",
    )

    st.write(
        "You have compared what each modality might add against its "
        "privacy, security, and agency costs. Based on the priorities you "
        "set in Step 5, this is where you choose."
    )

    st.caption(
        "Step 6 showed one measured cost of protecting a signal; "
        "the ratings below are the illustrative counterpart, extended to "
        "the modalities you added in Step 3."
    )

    all_modalities = _load_modalities()
    modalities_by_name = {m.name: m for m in all_modalities}
    baseline = modalities_by_name[BASELINE_MODALITY_NAME]
    selected_names = _held(SELECTED_MODALITIES_KEY, [])
    current_modalities = (baseline,) + tuple(
        modalities_by_name[name] for name in selected_names
    )

    if len(current_modalities) < 2:
        st.write(
            "Only EEG is in the pipeline, so there is no added modality "
            "to decide about yet. Go back to Step 3 to add at least one."
        )
    else:
        privacy_weight = _held(PRIVACY_WEIGHT_KEY, 0.34)
        security_weight = _held(SECURITY_WEIGHT_KEY, 0.33)
        agency_weight = _held(AGENCY_WEIGHT_KEY, 0.33)

        weighted_cost = tradeoff_core.combine_costs(
            current_modalities, privacy_weight, security_weight, agency_weight
        )
        frontier = tradeoff_core.compute_gain_cost_frontier(current_modalities, weighted_cost)

        addable_selected = tuple(m for m in current_modalities if m.name != baseline.name)

        decision = st.radio(
            "Based on the benefits and costs you explored, which added "
            "modality would you keep for a deployment?",
            options=[m.name for m in addable_selected] + ["Insufficient evidence to decide"],
            index=len(addable_selected),
        )

        if decision != "Insufficient evidence to decide":
            _body_figure((modalities_by_name[decision],), PERSPECTIVE_BUILD)

        if decision == "Insufficient evidence to decide":
            st.write(
                "A reasonable position: these ratings are illustrative "
                "and informed by literature that was not collected for "
                "this specific pipeline, population, or deployment "
                "context, so generalizing to a specific deployed system is "
                "not directly supported by what is rated here."
            )
        elif frontier.is_efficient.get(decision, False):
            st.write(
                f"**{decision} remains a viable option under your "
                "current priorities.** No other modality in this "
                "comparison provides at least as much interpretive value "
                "at the same or lower combined cost."
            )
            st.write(
                f"This does not mean {decision} is the best or safest "
                "choice for a specific deployment. The comparison uses "
                "illustrative ratings, and a deployment decision would require "
                "evidence specific to the system, the participants, and "
                "the intended use."
            )
        else:
            st.write(
                f"**Under your current priorities, {decision} has no "
                "advantage here.** Another modality in this comparison "
                "provides at least as much interpretive value at the same "
                "or lower combined cost, so these ratings give no reason "
                "to prefer it."
            )
            st.write(
                "That is a statement about four illustrative ratings and "
                "the priorities you set, not about the modality. Change "
                "the priorities and it can change, and a specific deployment "
                "could turn on factors none of these ratings capture."
            )

        st.write(
            "**One more consideration**: validating the signal is not "
            "enough if the system acts on what it infers. Flagging a "
            "clinician, triggering a device, or shaping what someone sees "
            "next each raise a separate question from whether the "
            "inference was accurate: what happens when it is wrong, and "
            "whether the person agreed to the action it produces. "
            "Validating a multimodal pipeline means validating both."
        )

        st.caption(
            "Which is the lesson of the whole journey. Choosing a "
            "modality is not "
            "\"which signal improves prediction\"; it is whether the "
            "information it adds justifies what collecting and acting on "
            "it introduces."
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

WORKSPACE.render_navigation()

st.divider()

with st.expander(f"References ({len(REFERENCES)})"):
    st.caption(
        "Numbered in the order the journey first cites them. A marker "
        "such as [4] in any stage above refers to this list."
    )
    for number, reference in enumerate(REFERENCES, start=1):
        st.caption(f"{number}. {reference}")
