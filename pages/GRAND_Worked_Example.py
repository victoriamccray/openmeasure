"""
GRAND: a real-data worked example for the multimodal signal-flow engine.

Teaches how separate imaging and behavioral modalities are acquired,
quality-checked, and processed on their own before being combined into
evidence about a research question - not a generic neuroimaging
explainer, and not an argument that combining modalities is automatically
worthwhile. Core logic (modules/signal_pipeline/core: Modality,
build_pipeline, compute_convergence, and, new for this page,
feature_selection.select_necessary_feature_set) is domain-agnostic and
unchanged in its guarantees for the illustrative EEG/ECG/EMG journey;
this page's only core addition is a convergence_stage parameter on
build_pipeline, so a worked example can show each modality processed
separately before convergence instead of converging immediately after
Sensors - a strictly backward-compatible generalization (see
modules/signal_pipeline/core/pipeline.py).

Like the other Research Journeys, this page records nothing to
shared/handoff.py, carries no module_key, and is deliberately not a
numbered page, so it needs no entry in shared/catalog.py.

Register (a note on style): this page's prose is written closer to the
GRAND manuscript's own register - short, declarative, describing what
was measured and what it supports - than the guided-discovery voice used
in Multimodal_Signal_Convergence.py's illustrative journey. Headings
throughout are Interpretation / Implications / Limitations / Research
considerations / What to inspect rather than "Why this matters," on
purpose: this page states evidence and its bounds and leaves the
research decision to the reader, rather than telling the reader what to
find important.

Citation-integrity note - read this before trusting any number below.

1. Dataset-level facts (participant count, age range, scanner, task
   name, license) were fetched directly from GRAND's own
   dataset_description.json, README, and participants.tsv, or from the
   cited papers' own metadata - not recalled from memory.
2. Sample size discrepancy: GRAND's README and dataset_description.json
   both describe 110 participants, matching the 110 rows actually in
   participants.tsv. The preprint's abstract instead states "116
   neurotypical adults," and its Methods section explains the 116 as the
   sum of two parent recruitment protocols - BUILD ("Brain-based
   Understanding of Individual Language Differences after stroke,"
   ClinicalTrials.gov NCT04991519, N=69) and ReadMap ("Reading in Stroke
   Alexia and Typical Aging," ClinicalTrials.gov NCT06700005, N=47).
   Nothing in the accessible text explains the gap between that 116 and
   the 110 rows actually released in participants.tsv; presented as-is,
   unreconciled - the same treatment GAIA_Worked_Example.py gives its own
   paper/figure discrepancy.
3. Connectome construction, previously flagged here as unverified, was
   confirmed on a later attempt (see item 4): connectomes are derived
   from the HARDI diffusion acquisition (not a separately acquired DTI
   sequence) via probabilistic, anatomically-constrained tractography,
   assigning streamlines to parcels of the Brainnetome atlas; each edge
   is the apparent fiber density between two parcels.
4. Quality-control figures in Step 3: the 0.5 mm mean framewise-
   displacement cutoff, its exact exception counts, the tSNR description,
   and the Semantic > Pseudofont-vs-NeuroSynth overlap were confirmed
   directly against the preprint's Methods and Technical Validation
   sections once bioRxiv's rate limit (which returned HTTP 429 for every
   attempt while this page was first built, including one fetch that
   returned a fluent but unusable answer under the same failed request -
   caught and discarded rather than used) stopped blocking the fetch.
   The exact split-half reliability values (Spearman-Brown corrected)
   remain unconfirmed: the accessible text states only that "results
   indicate high reliability across measures for all tasks," without the
   Tables 2-4 figures. The specific 0.92-0.99 range shown at that step is
   presented as provided for this page, not independently confirmed.
5. Step 6's feature-set comparison (Pearson correlation and coefficient
   of determination by feature set) uses illustrative, invented numbers
   to teach the best-vs-necessary distinction - GRAND's own per-feature-
   set model-comparison results were not available to this page. This is
   stated again at that step, prominently, not only here.
6. Every interpretive-gain and privacy/security/agency-cost rating in
   sample_data/grand_modalities.csv is an illustrative, author-assigned
   score informed by GRAND's own acquisition parameters and by cited
   re-identification/privacy literature - not a value GRAND or that
   literature reports directly.
7. The preprint's accessible Methods and Technical Validation text names
   no explicit hemispheric-lateralization finding. Step 8's mention of a
   left-lateralized language network is standard, independently citable
   neuroscience background (e.g. Broca's and Wernicke's areas), presented
   as context for GRAND's confirmed frontal/temporal activation - not as
   a lateralization result this page verified GRAND's own paper to
   report.

Core logic lives in modules/signal_pipeline/core; this file is
presentation plus GRAND's own facts and citations, which are specific to
this worked example and so live here rather than in the reusable core.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from modules.signal_pipeline.core import feature_selection as feature_selection_core
from modules.signal_pipeline.core import modality as modality_core
from modules.signal_pipeline.core import pipeline as pipeline_core
from shared.data_handling import disclosure_for, render_data_handling_summary
from shared.report import caveat, flagged_item_note, section_header

SAMPLE_DIR = ROOT / "modules" / "signal_pipeline" / "sample_data"

# Validated default palette (dataviz skill, references/palette.md).
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
MUTED = "#898781"

CATEGORY_COLORS = {
    "Structural": "#2a78d6",
    "Functional": "#eb6834",
    "Diffusion": "#1baf7a",
    "Network": "#4a3aa7",
    "Behavioral": "#e87ba4",
}
CATEGORY_SHAPES = {
    "Structural": "square",
    "Functional": "circle",
    "Diffusion": "triangle-up",
    "Network": "diamond",
    "Behavioral": "triangle-down",
}
PIPELINE_NODE_COLOR = MUTED
PIPELINE_NODE_SHAPE = "circle"

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

# ---------------------------------------------------------------------
# GRAND's own facts. Fetched directly from dataset_description.json,
# README, and participants.tsv (all at
# https://openneuro.org/datasets/ds007831/versions/1.0.1), and from the
# cited papers - not recalled from memory. Step 3's QC-specific figures
# were confirmed directly against the preprint's Methods and Technical
# Validation text; see citation-integrity note items 3 and 4 above for
# exactly what is confirmed and what (the exact split-half values) is
# not.
# ---------------------------------------------------------------------

GRAND_DATASET_CITATION = (
    "Anderson, E. J., Staples, R., Dyslin, S. M., Chang, E. H. T., Laks, A. B., "
    "Dickens, J. V., Mathur, D., Paul, S., Dvorak, E., & Turkeltaub, P. E. (2026). "
    "The Georgetown Reading in Aging Neuroimaging Dataset (GRAND) [Data set]. "
    "OpenNeuro. https://doi.org/10.18112/openneuro.ds007831.v1.0.1"
)
GRAND_PREPRINT_CITATION = (
    "Anderson, E. J. et al. (2026). The Georgetown Reading in Aging "
    "Neuroimaging Dataset (GRAND): Reading and multimodal MRI data in older "
    "adults. bioRxiv. https://doi.org/10.64898/2026.05.18.725986"
)
WILSON_CITATION = (
    "Wilson, S. M., Yen, M., & Eriksson, D. K. (2018). An adaptive semantic "
    "matching paradigm for reliable and valid language mapping in individuals "
    "with aphasia. Human Brain Mapping, 39, 3285-3307. "
    "https://doi.org/10.1002/hbm.24077"
)
SCHWARZ_CITATION = (
    "Schwarz, C. G., Kremers, W. K., Therneau, T. M., et al. (2019). "
    "Identification of anonymous MRI research participants with "
    "face-recognition software. New England Journal of Medicine, 381(17), "
    "1684-1686. https://doi.org/10.1056/NEJMc1908881"
)
BUILD_CITATION = (
    "Turkeltaub, P. E. (Principal Investigator). Brain-based Understanding "
    "of Individual Language Differences after Stroke (BUILD). "
    "ClinicalTrials.gov identifier NCT04991519."
)
READMAP_CITATION = (
    "Turkeltaub, P. E. (Principal Investigator). Reading in Stroke Alexia "
    "and Typical Aging (ReadMap). ClinicalTrials.gov identifier NCT06700005."
)

GRAND_N_PARTICIPANTS = 110  # participants.tsv row count, fetched directly
GRAND_N_PARTICIPANTS_PREPRINT_ABSTRACT = 116  # stated in the preprint's abstract - see docstring
GRAND_N_BUILD = 69  # of the 116, recruited via BUILD - preprint Methods
GRAND_N_READMAP = 47  # of the 116, recruited via ReadMap - preprint Methods
GRAND_FD_CUTOFF_MM = 0.5  # mean framewise-displacement cutoff, both MRI modalities
GRAND_N_OVER_FD_CUTOFF_FUNCTIONAL = 2  # preprint Technical Validation
GRAND_N_OVER_FD_CUTOFF_DIFFUSION = 7  # preprint Technical Validation
GRAND_AGE_MIN = 22.42
GRAND_AGE_MAX = 84.15
GRAND_AGE_MEAN = 59.7
GRAND_SCANNER = "3T Siemens MAGNETOM Prisma, 20-channel head coil"
GRAND_SITE = "Center for Functional and Molecular Imaging, Georgetown University Medical Center"
GRAND_TASK_NAME = "the Adaptive Language Mapping Task"
GRAND_PURPOSE = (
    "to establish a normative database of visual word recognition and oral "
    "reading, as well as language-associated fMRI activations"
)
QC_FIGURES_CAVEAT = (
    "The motion cutoffs, exception counts, tSNR description, and task-"
    "contrast overlap below are confirmed directly against the published "
    "manuscript. The exact split-half reliability values are not - see the "
    "page's citation-integrity note, item 4."
)

BASELINE_MODALITY_NAME = "Structural MRI (T1w/FLAIR)"

# ---------------------------------------------------------------------
# Brain scene: an original, illustrative graphic (not a trace or copy of
# any commercial illustration library's assets) showing which of GRAND's
# five modalities are selected. Four are scans of the brain, converging
# on one central brain icon; Behavioral is measured outside the scanner
# entirely and is drawn as a separate, external element rather than a
# fifth position around the brain, so the figure does not imply it was
# sensed from the body the way the MRI modalities were.
# ---------------------------------------------------------------------

SCENE_WIDTH_PX = 300
SCENE_HEIGHT_PX = 340
BRAIN_CENTER_PX = (150, 110)
BRAIN_RADIUS_PX = 38
SATELLITE_PX = {
    "Structural": (55, 45),
    "Functional": (245, 45),
    "Diffusion": (55, 185),
    "Network": (245, 185),
}
BEHAVIORAL_PX = (150, 300)
_BRAIN_PATH = (
    '<path d="M16 4C10 4 6 8 6 13C4 14 3 16 3 18C3 21 5 23 8 23'
    "C8 26 11 28 14 28C15 28 16 27 16 27C16 27 17 28 18 28"
    "C21 28 24 26 24 23C27 23 29 21 29 18C29 16 28 14 26 13"
    'C26 8 22 4 16 4Z" fill="{fill}"/>'
    '<path d="M16 6L16 26M11 9C13 10 13 13 11 15M21 9C19 10 19 13 21 15" '
    'stroke="{stroke_light}" stroke-width="1.2" fill="none" stroke-linecap="round"/>'
)
_MODALITY_ICON_PATHS = {
    "Structural": (
        '<rect x="4" y="4" width="24" height="24" rx="4" fill="{fill}"/>'
        '<line x1="8" y1="12" x2="24" y2="12" stroke="{stroke_light}" stroke-width="1.4"/>'
        '<line x1="8" y1="17" x2="24" y2="17" stroke="{stroke_light}" stroke-width="1.4"/>'
        '<line x1="8" y1="22" x2="24" y2="22" stroke="{stroke_light}" stroke-width="1.4"/>'
    ),
    "Functional": (
        '<path d="M16 4C10 4 6 8 6 13C4 14 3 16 3 18C3 21 5 23 8 23'
        "C8 26 11 28 14 28C15 28 16 27 16 27C16 27 17 28 18 28"
        "C21 28 24 26 24 23C27 23 29 21 29 18C29 16 28 14 26 13"
        'C26 8 22 4 16 4Z" fill="{fill}" opacity="0.3"/>'
        '<circle class="activation-blob" style="animation-delay:0s" cx="12" cy="14" r="3" fill="{fill}"/>'
        '<circle class="activation-blob" style="animation-delay:0.3s" cx="19" cy="12" r="2.4" fill="{fill}"/>'
        '<circle class="activation-blob" style="animation-delay:0.6s" cx="16" cy="20" r="2.8" fill="{fill}"/>'
    ),
    "Diffusion": (
        '<path d="M4 8C10 4 14 10 10 16C6 22 12 28 18 24" stroke="{fill}" '
        'stroke-width="2" fill="none" stroke-linecap="round"/>'
        '<path d="M8 4C14 8 10 14 16 18C22 22 18 28 24 28" stroke="{fill}" '
        'stroke-width="2" fill="none" stroke-linecap="round" opacity="0.7"/>'
        '<path d="M14 3C20 6 16 12 22 16C28 20 24 27 29 29" stroke="{fill}" '
        'stroke-width="2" fill="none" stroke-linecap="round" opacity="0.5"/>'
    ),
    "Network": (
        '<line x1="8" y1="24" x2="24" y2="24" stroke="{fill}" stroke-width="1.6"/>'
        '<line x1="8" y1="24" x2="16" y2="6" stroke="{fill}" stroke-width="1.6"/>'
        '<line x1="24" y1="24" x2="16" y2="6" stroke="{fill}" stroke-width="1.6"/>'
        '<circle cx="8" cy="24" r="3.4" fill="{fill}"/>'
        '<circle cx="24" cy="24" r="3.4" fill="{fill}"/>'
        '<circle cx="16" cy="6" r="3.4" fill="{fill}"/>'
    ),
    "Behavioral": (
        '<rect x="12" y="2" width="8" height="16" rx="4" fill="{fill}"/>'
        '<path d="M7 15C7 20 11 23 16 23C21 23 25 20 25 15" stroke="{fill}" '
        'stroke-width="2" fill="none" stroke-linecap="round"/>'
        '<line x1="16" y1="23" x2="16" y2="29" stroke="{fill}" stroke-width="2" stroke-linecap="round"/>'
        '<line x1="10" y1="29" x2="22" y2="29" stroke="{fill}" stroke-width="2" stroke-linecap="round"/>'
    ),
}
# Structural intentionally has no pulse: it is a static anatomical
# snapshot, not a dynamic process - amt=1.0 means no scale change at all.
_ICON_PULSE = {
    "Structural": (1.0, "1s"),
    "Functional": (1.15, "1.2s"),
    "Diffusion": (1.05, "1.6s"),
    "Network": (1.08, "2.2s"),
    "Behavioral": (1.1, "1.3s"),
}
ANATOMY_LEGEND_BUFFER_PX = 56


def _icon_markup(path_template: str, fill: str, cx: float, cy: float, scale: float, pulse_class: str) -> str:
    path = path_template.format(fill=fill, stroke_light=SURFACE)
    tx = cx - 16 * scale
    ty = cy - 16 * scale
    return (
        f'<g class="{pulse_class}">'
        f'<g transform="translate({tx:.1f},{ty:.1f}) scale({scale:.3f})">{path}</g>'
        f"</g>"
    )


def _brain_scene_html(modalities: tuple[modality_core.Modality, ...], height: int = 340) -> str:
    """
    One person's brain, with each selected scanned modality (Structural,
    Functional, Diffusion, Network) shown as a satellite icon converging
    on it, and Behavioral shown separately below as measured outside the
    scanner rather than sensed from the body.
    """

    bx, by = BRAIN_CENTER_PX
    icons_svg: list[str] = []
    connectors_svg: list[str] = []
    legend_items: list[str] = []
    has_behavioral = False

    for m in modalities:
        color = CATEGORY_COLORS[m.category]
        amt, dur = _ICON_PULSE.get(m.category, (1.0, "1s"))

        if m.category == "Behavioral":
            has_behavioral = True
            sx, sy = BEHAVIORAL_PX
        else:
            sx, sy = SATELLITE_PX.get(m.category, (bx, by))
            connectors_svg.append(
                f'<line x1="{sx}" y1="{sy}" x2="{bx}" y2="{by}" stroke="{color}" '
                f'stroke-width="1.5" stroke-dasharray="4 3" opacity="0.5"/>'
            )

        icons_svg.append(f'<g style="--pulse-amt:{amt};--pulse-dur:{dur};">')
        icons_svg.append(_icon_markup(_MODALITY_ICON_PATHS[m.category], color, sx, sy, 1.05, "icon-pulse"))
        label_y = sy - 24 if sy < by else sy + 34
        # The category (e.g. "Structural"), not the full m.name (e.g.
        # "Structural MRI (T1w/FLAIR)"): the satellites sit close to the
        # 300px-wide scene's edges, and a long centered label there
        # overflows past the viewBox and gets clipped. The full name is
        # still shown in the legend below and in each modality's own
        # expander.
        icons_svg.append(
            f'<text x="{sx}" y="{label_y}" text-anchor="middle" font-size="10" '
            f'fill="{INK_SECONDARY}">{m.category}</text>'
        )
        icons_svg.append("</g>")

        legend_items.append(
            f'<span style="display:inline-flex;align-items:center;gap:5px;">'
            f'<span style="width:9px;height:9px;border-radius:2px;background:{color};'
            f'display:inline-block;"></span>{m.category}</span>'
        )

    behavioral_label = ""
    if has_behavioral:
        lx, ly = BEHAVIORAL_PX
        behavioral_label = (
            f'<text x="{lx}" y="{ly - 40}" text-anchor="middle" font-size="10" '
            f'font-style="italic" fill="{MUTED}">Outside the scanner:</text>'
        )

    brain_svg = _icon_markup(_BRAIN_PATH, MUTED, bx, by, BRAIN_RADIUS_PX / 16 * 0.9, "icon-pulse")

    legend_row = "".join(f"<div>{item}</div>" for item in legend_items)
    scale_factor = height / SCENE_HEIGHT_PX
    svg_w = SCENE_WIDTH_PX * scale_factor

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
        @keyframes blobPulse {{
          0%, 100% {{ opacity: 0.4; transform: scale(0.8); }}
          50% {{ opacity: 1; transform: scale(1.15); }}
        }}
        .activation-blob {{
          animation: blobPulse 1.1s ease-in-out infinite;
          transform-box: fill-box;
          transform-origin: center;
        }}
      </style>
      <div style="display:flex; justify-content:center;">
        <svg width="{svg_w:.0f}" height="{height}" viewBox="0 0 {SCENE_WIDTH_PX} {SCENE_HEIGHT_PX}">
          {"".join(connectors_svg)}
          {brain_svg}
          {behavioral_label}
          {"".join(icons_svg)}
        </svg>
      </div>
      <div style="display:flex; flex-wrap:wrap; justify-content:center; gap:12px; margin-top:2px;
                  font-size:11px; color:{INK_SECONDARY};">
        {legend_row}
      </div>
    </div>
    """


def _node_rows(pipeline: pipeline_core.SignalPipeline):
    modality_order = {m.name: i for i, m in enumerate(pipeline.modalities)}
    n = len(pipeline.modalities)
    rows = []
    coords = {}
    for node in pipeline.nodes:
        if node.modality is not None:
            y = modality_order[node.modality.name] - (n - 1) / 2
            label = node.modality.name if node.stage == "Signals" else ""
            category = node.modality.category
            key = (node.stage, node.modality.name)
        else:
            y = 0.0
            label = node.stage
            category = "Pipeline"
            key = (node.stage, None)
        coords[key] = (node.stage, y)
        rows.append({"stage": node.stage, "y": y, "label": label, "category": category})
    return rows, coords


def _edge_rows(pipeline: pipeline_core.SignalPipeline, coords: dict) -> list[dict]:
    rows = []
    for edge in pipeline.edges:
        source_key = (edge.source.stage, edge.source.modality.name if edge.source.modality else None)
        target_key = (edge.target.stage, edge.target.modality.name if edge.target.modality else None)
        x, y = coords[source_key]
        x2, y2 = coords[target_key]
        rows.append({"stage": x, "y": y, "stage2": x2, "y2": y2})
    return rows


def _pipeline_diagram_spec(pipeline: pipeline_core.SignalPipeline) -> dict:
    node_rows, coords = _node_rows(pipeline)
    edge_rows = _edge_rows(pipeline, coords)

    categories_present = [c for c in CATEGORY_COLORS if any(r["category"] == c for r in node_rows)]
    color_domain = categories_present + ["Pipeline"]
    color_range = [CATEGORY_COLORS[c] for c in categories_present] + [PIPELINE_NODE_COLOR]
    shape_domain = categories_present + ["Pipeline"]
    shape_range = [CATEGORY_SHAPES[c] for c in categories_present] + [PIPELINE_NODE_SHAPE]

    stage_x = {
        "field": "stage",
        "type": "nominal",
        "sort": list(pipeline_core.STAGES),
        "axis": {"title": None, "labelAngle": 0},
    }
    stage_x2 = {"field": "stage2", "type": "nominal", "sort": list(pipeline_core.STAGES)}
    hidden_y = {"field": "y", "type": "quantitative", "axis": None}

    height = min(360, 90 + 40 * max(len(pipeline.modalities), 3))

    return {
        "layer": [
            {
                "data": {"values": edge_rows},
                "mark": {"type": "rule", "color": GRIDLINE, "strokeWidth": 1.5},
                "encoding": {"x": stage_x, "y": hidden_y, "x2": stage_x2, "y2": {"field": "y2"}},
            },
            {
                "data": {"values": node_rows},
                "mark": {"type": "point", "filled": True, "size": 220},
                "encoding": {
                    "x": stage_x,
                    "y": hidden_y,
                    "color": {
                        "field": "category",
                        "type": "nominal",
                        "scale": {"domain": color_domain, "range": color_range},
                        "legend": {"title": None, "orient": "bottom", "columns": 3},
                    },
                    "shape": {
                        "field": "category",
                        "type": "nominal",
                        "scale": {"domain": shape_domain, "range": shape_range},
                    },
                },
            },
            {
                "data": {"values": [row for row in node_rows if row["label"]]},
                "mark": {"type": "text", "dy": -16, "fontSize": 11, "color": INK_SECONDARY},
                "encoding": {
                    "x": stage_x,
                    "y": {"field": "y", "type": "quantitative"},
                    "text": {"field": "label", "type": "nominal"},
                },
            },
        ],
        "width": "container",
        "height": height,
        "config": _VEGA_CHART_CONFIG,
    }


def _feature_set_spec(results: tuple[feature_selection_core.FeatureSetResult, ...], selection) -> dict:
    rows = [
        {
            "name": r.name,
            "n_features": r.n_features,
            "performance": r.performance,
            "se_low": r.performance - r.standard_error,
            "se_high": r.performance + r.standard_error,
            "status": (
                "best" if r.name == selection.best
                else "necessary" if r.name == selection.necessary
                else "within 1 SE of best" if selection.within_one_se_of_best[r.name]
                else "not within 1 SE of best"
            ),
        }
        for r in results
    ]
    status_domain = ["best", "necessary", "within 1 SE of best", "not within 1 SE of best"]
    status_range = ["#2a78d6", "#1baf7a", "#c3c2b7", "#e1e0d9"]

    return {
        "data": {"values": rows},
        "layer": [
            {
                "mark": {"type": "rule", "color": INK_SECONDARY, "strokeWidth": 1.5},
                "encoding": {
                    "x": {"field": "n_features", "type": "ordinal", "title": "Number of features"},
                    "y": {"field": "se_low", "type": "quantitative", "title": "Performance (+/- 1 SE)"},
                    "y2": {"field": "se_high"},
                },
            },
            {
                "mark": {"type": "point", "filled": True, "size": 200},
                "encoding": {
                    "x": {"field": "n_features", "type": "ordinal"},
                    "y": {"field": "performance", "type": "quantitative"},
                    "color": {
                        "field": "status",
                        "type": "nominal",
                        "scale": {"domain": status_domain, "range": status_range},
                        "legend": {"title": None, "orient": "bottom", "columns": 2},
                    },
                    "tooltip": [
                        {"field": "name", "type": "nominal", "title": "Feature set"},
                        {"field": "performance", "type": "quantitative", "title": "Performance", "format": ".2f"},
                        {"field": "status", "type": "nominal", "title": "Status"},
                    ],
                },
            },
            {
                "mark": {"type": "text", "dy": -14, "fontSize": 10, "color": INK_SECONDARY},
                "encoding": {
                    "x": {"field": "n_features", "type": "ordinal"},
                    "y": {"field": "performance", "type": "quantitative"},
                    "text": {"field": "name", "type": "nominal"},
                },
            },
        ],
        "width": "container",
        "height": 280,
        "config": _VEGA_CHART_CONFIG,
    }


def _md(text: str) -> str:
    """Escape literal '*' (e.g. in 'T2*') so it renders as text, not a
    stray or mismatched markdown emphasis delimiter, when interpolated
    into an st.write/st.markdown string."""

    return text.replace("*", "\\*")


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _to_str(value) -> str:
    return "" if pd.isna(value) else str(value)


def _load_modalities() -> tuple[modality_core.Modality, ...]:
    frame = pd.read_csv(SAMPLE_DIR / "grand_modalities.csv")
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


DERIVED_FEATURE_BY_CATEGORY = {
    "Structural": "Regional cortical thickness / volume",
    "Functional": "Task contrast estimate per region (Semantic > Pseudofont)",
    "Diffusion": "Tract-wise diffusion metric (e.g. fractional anisotropy) per pathway",
    "Network": "Edge-weight matrix between regions",
    "Behavioral": "Per-participant accuracy and reaction-time summary scores",
}

QC_NOTES = {
    "Structural MRI (T1w/FLAIR)": {
        "what_to_inspect": "Visual inspection for gross artifacts, motion blurring, or incidental findings.",
        "interpretation": (
            "FLAIR (fluid-attenuated inversion recovery) is a structural "
            "sequence tuned to suppress fluid signal, which makes white-"
            "matter change and certain lesions easier to see than on a "
            "plain T1-weighted scan alone. This page does not have a "
            "modality-specific quantitative QC figure for Structural MRI; "
            "standard practice is qualitative visual review unless a "
            "specific metric is reported."
        ),
    },
    "Functional MRI (T2*, language task)": {
        "what_to_inspect": (
            "Mean framewise displacement (motion) during the task; temporal "
            "signal-to-noise ratio (tSNR) and its spatial pattern across the brain; "
            "the Semantic > Pseudofont contrast compared against a NeuroSynth "
            "\"language\" meta-analytic map."
        ),
        "interpretation": (
            f"A commonly used mean framewise-displacement cutoff for "
            f"excluding a run is {GRAND_FD_CUTOFF_MM} mm; GRAND's preprint "
            f"reports that all but {GRAND_N_OVER_FD_CUTOFF_FUNCTIONAL} "
            "participants average less than this. tSNR is the voxelwise "
            "mean signal divided by its standard deviation across time; "
            "comparable tSNR across participants supports comparable "
            "signal quality, and tSNR varying spatially across the brain is "
            "expected (MRI sensitivity is not uniform across regions), not "
            "evidence of a data-quality problem on its own. \"Semantic > "
            "Pseudofont\" contrasts real-word reading against a visually "
            "matched control condition (pseudofont strings: shapes similar "
            "in size and spacing to letters, but not readable as words), so "
            "the resulting activation reflects word-reading itself rather "
            "than just looking at something on a screen. NeuroSynth is an "
            "automated meta-analysis tool that pools thousands of published "
            "fMRI studies into a single map per search term; GRAND's "
            "preprint reports strong spatial overlap between this contrast "
            "and NeuroSynth's \"language\" map, external evidence that the "
            "task engaged expected language-associated regions."
        ),
    },
    "Diffusion MRI (HARDI/DTI)": {
        "what_to_inspect": "Mean framewise displacement during the diffusion sequence.",
        "interpretation": (
            "HARDI (high angular resolution diffusion imaging) acquires "
            "diffusion-weighted volumes along many more gradient directions "
            "than a standard DTI (diffusion tensor imaging) scan, which "
            "supports more detailed tractography; GRAND acquires HARDI-"
            "density data (128 volumes total, see Step 2) and derives its "
            "connectomes from it. "
            f"The same {GRAND_FD_CUTOFF_MM} mm mean framewise-displacement "
            f"cutoff is reported to apply here; GRAND's preprint reports "
            f"that {GRAND_N_OVER_FD_CUTOFF_DIFFUSION} participants exceed "
            "it for this specific, longer sequence."
        ),
    },
    "Connectome (derived)": {
        "what_to_inspect": "Whichever QC was applied to the modality this connectome was derived from.",
        "interpretation": (
            "GRAND's preprint reports that each connectome is built by "
            "probabilistic, anatomically-constrained tractography, "
            "tracing likely white-matter fiber pathways through the "
            "diffusion data, then assigning those pathways to parcels "
            "(anatomically defined regions) of the Brainnetome atlas, a "
            "published whole-brain parcellation scheme. Each edge in the "
            "resulting connectome is the apparent fiber density between two "
            "parcels. Any connectome's quality is still bounded by the "
            "quality of the diffusion data it was built from, the same "
            "framewise-displacement figures above."
        ),
    },
    "Behavioral reading/language measures": {
        "what_to_inspect": "Split-half reliability of accuracy and reaction time, computed separately.",
        "interpretation": (
            "Split-half reliability (Spearman-Brown corrected: a formula "
            "that adjusts a reliability estimate for the fact that each "
            "half of a split test is shorter than the full test) is "
            "reported as generally high, on the order of 0.92 to 0.99 for "
            "reaction-time and time-on-item measures specifically. The "
            "preprint's accessible text confirms reliability is high across "
            "measures, but not this exact numeric range; see the "
            "citation-integrity note, item 4. Accuracy is reported as more "
            "variable, partly attributable to ceiling effects on an easier task."
        ),
    },
}

REDUNDANCY_NOTES = {
    "Functional MRI (T2*, language task)": (
        "Measures which regions activate during reading, not their anatomy. Low "
        "overlap with Structural or Diffusion: this is the only modality that "
        "measures the task itself."
    ),
    "Diffusion MRI (HARDI/DTI)": (
        "Measures white-matter tract integrity, a different anatomical axis from "
        "Structural's gray-matter morphometry. Overlaps more with Connectome, since "
        "a connectome derived from diffusion tractography summarizes the same "
        "tracts at a coarser, network level."
    ),
    "Connectome (derived)": (
        "If derived from Diffusion, a substantial share of its evidence is already "
        "implicit in the Diffusion data it was built from. Removing Diffusion while "
        "keeping Connectome would remove the ability to check or rebuild that "
        "summary, even though the summary itself would remain available."
    ),
    "Behavioral reading/language measures": (
        "Measures reading and language performance directly, rather than a neural "
        "correlate of it. No imaging modality substitutes for this measurement; the "
        "imaging modalities are candidate explanations of variation in it."
    ),
}

STAGE_KEY = "grand_stage"
SELECTED_MODALITIES_KEY = "grand_selected_modalities"

STAGE_RESEARCH_QUESTION = 0
STAGE_ACQUIRE = 1
STAGE_QC = 2
STAGE_PROCESS = 3
STAGE_ALIGN = 4
STAGE_INTEGRATE = 5
STAGE_EVALUATE = 6
STAGE_INTERPRET = 7

JOURNEY_STAGES = (
    "Research question",
    "Acquire modalities",
    "QC each modality",
    "Process separately",
    "Align & derive features",
    "Integrate evidence",
    "Evaluate added value",
    "Interpret",
)


def _current_stage() -> int:
    return st.session_state.get(STAGE_KEY, STAGE_RESEARCH_QUESTION)


def _advance_to(stage: int) -> None:
    st.session_state[STAGE_KEY] = max(_current_stage(), stage)
    st.rerun()


def _current_modalities() -> tuple[modality_core.Modality, ...]:
    all_modalities = _load_modalities()
    modalities_by_name = {m.name: m for m in all_modalities}
    baseline = modalities_by_name[BASELINE_MODALITY_NAME]
    selected_names = st.session_state.get(SELECTED_MODALITIES_KEY, [])
    return (baseline,) + tuple(modalities_by_name[name] for name in selected_names)


st.set_page_config(
    page_title="OpenMeasure · GRAND",
    page_icon=":material/psychology:",
    layout="centered",
)

st.title("GRAND: A Real-Data Multimodal Signal-Flow Example")
st.caption(
    "This journey uses one real, cited dataset, GRAND (Anderson et al., "
    "2026), to run the same signal-flow engine built for illustrative "
    "modalities in Multimodal Signal Convergence, on real structural, "
    "functional, and diffusion MRI, a derived connectome, and behavioral "
    "reading/language measures."
)
st.caption(
    "A guided research simulation: each stage unlocks after you make a "
    "decision or inspect its consequence."
)

render_data_handling_summary(disclosure_for("pages/GRAND_Worked_Example.py"))

stage = _current_stage()
_stage_parts = [
    f"**{label}**" if index == stage else label for index, label in enumerate(JOURNEY_STAGES)
]
with st.container(border=True):
    st.markdown(" → ".join(_stage_parts))

if stage > STAGE_RESEARCH_QUESTION:
    if st.button("Restart study", icon=":material/restart_alt:"):
        st.session_state.pop(STAGE_KEY, None)
        st.session_state.pop(SELECTED_MODALITIES_KEY, None)
        st.rerun()

st.divider()

# -----------------------------------------------------------------
# 1. Research question
# -----------------------------------------------------------------

section_header("1. Research Question")

st.markdown("### What Does Each Imaging Modality Contribute To Understanding Reading and Language?")

st.write(
    f"GRAND scanned {GRAND_N_PARTICIPANTS} healthy older adults with "
    "structural, functional, and diffusion MRI, derived a connectome (a "
    "map of how strongly different brain regions are structurally "
    "connected) from that data, and separately measured reading and "
    "language performance outside the scanner. Each of these five "
    "modalities measures a "
    "different thing about the same participants. This page follows the "
    "sequence a researcher moves through to turn that raw acquisition "
    "into a defensible statement about what each modality adds: "
    "acquisition, quality control, separate processing, alignment into "
    "derived features, integration, evaluation of added value, and "
    "interpretation."
)

st.write(
    "Reading and language ability are frequently disrupted by stroke, and "
    "stroke risk rises with age, so a study of stroke-related language "
    "impairment needs a normative picture of how reading and its brain "
    "basis work in healthy older adults first, to know what a patient's "
    "performance is being compared against. GRAND's participants were "
    "recruited through two studies aimed at exactly that: BUILD "
    "(\"Brain-based Understanding of Individual Language Differences after "
    "stroke\") and ReadMap (\"Reading in Stroke Alexia and Typical Aging\"), "
    "both run by GRAND's senior author. GRAND itself scans only "
    "neurotypical adults - it is the normative half of that research "
    "program, not a stroke dataset."
)

with st.expander("About GRAND"):
    st.write(
        f"Purpose, as stated by the dataset: \"{GRAND_PURPOSE}.\" Data was "
        f"collected at {GRAND_SITE} using a {GRAND_SCANNER}. The functional "
        f"task was {GRAND_TASK_NAME}."
    )
    st.write(
        f"Of the preprint's {GRAND_N_PARTICIPANTS_PREPRINT_ABSTRACT} "
        f"participants, {GRAND_N_BUILD} were recruited via BUILD and "
        f"{GRAND_N_READMAP} via ReadMap ({GRAND_N_BUILD} + "
        f"{GRAND_N_READMAP} = {GRAND_N_BUILD + GRAND_N_READMAP}). "
        f"participants.tsv in the released dataset has "
        f"{GRAND_N_PARTICIPANTS} rows; nothing in the accessible text "
        "explains that gap - see the citation-integrity note, item 2."
    )
    st.caption(GRAND_DATASET_CITATION)
    st.caption(GRAND_PREPRINT_CITATION)
    st.caption(WILSON_CITATION)
    st.caption(BUILD_CITATION)
    st.caption(READMAP_CITATION)

if stage < STAGE_ACQUIRE:
    if st.button("Begin study", type="primary"):
        _advance_to(STAGE_ACQUIRE)

# -----------------------------------------------------------------
# 2. Acquire modalities
# -----------------------------------------------------------------

if stage >= STAGE_ACQUIRE:
    section_header(
        "2. Acquire Modalities",
        "Select modalities to add to the pipeline.",
    )

    all_modalities = _load_modalities()
    modalities_by_name = {m.name: m for m in all_modalities}
    baseline = modalities_by_name[BASELINE_MODALITY_NAME]
    addable = tuple(m for m in all_modalities if m.name != BASELINE_MODALITY_NAME)

    st.write(
        "Every GRAND participant received a structural scan, so this "
        "journey starts there. Add the other four: functional MRI, "
        "diffusion MRI, the derived connectome, and behavioral reading/"
        "language measures collected outside the scanner."
    )

    selected_names = st.multiselect(
        "Modalities to add",
        options=[m.name for m in addable],
        default=st.session_state.get(SELECTED_MODALITIES_KEY, []),
        key=SELECTED_MODALITIES_KEY,
    )
    current_modalities = (baseline,) + tuple(modalities_by_name[name] for name in selected_names)

    components.html(
        _brain_scene_html(current_modalities, height=340),
        height=340 + ANATOMY_LEGEND_BUFFER_PX,
    )
    st.caption(
        "Structural is static above: it is one anatomical snapshot, not a "
        "dynamic process. The other icons animate gently as a visual cue "
        "that they measure different kinds of things; none of this is "
        "real recorded activity."
    )

    for m in current_modalities:
        with st.expander(f"{m.name}, what this measures"):
            st.caption(f"Examples: {m.signal_examples}.")
            st.write(m.notes)
            st.caption(m.citation)

    with st.expander("Research considerations: acquisition"):
        st.write(
            "Structural MRI can be used to reconstruct a recognizable "
            "image of a participant's face; face-recognition software has "
            "been shown to re-identify de-identified research MRI scans "
            "specifically from this modality."
        )
        st.caption(SCHWARZ_CITATION)
        st.write(
            "Voice recordings from the oral-reading trials carry their own "
            "re-identification risk, independent of any MRI modality."
        )

    caveat(
        "Every interpretive-gain and cost rating on this page is an "
        "illustrative, author-assigned score informed by GRAND's own "
        "acquisition parameters and by cited literature, not a value "
        "GRAND or that literature reports directly. See the page "
        "docstring for what is flagged rather than resolved."
    )

    if stage < STAGE_QC:
        if st.button("Continue to QC each modality", type="primary"):
            _advance_to(STAGE_QC)

# -----------------------------------------------------------------
# 3. QC each modality
# -----------------------------------------------------------------

if stage >= STAGE_QC:
    section_header(
        "3. QC Each Modality",
        "What to inspect before trusting a modality's data, and what it supports.",
    )

    st.caption(QC_FIGURES_CAVEAT)

    _fd_over_cutoff = {
        "Functional MRI (T2*, language task)": GRAND_N_OVER_FD_CUTOFF_FUNCTIONAL,
        "Diffusion MRI (HARDI/DTI)": GRAND_N_OVER_FD_CUTOFF_DIFFUSION,
    }
    _fd_columns_needed = [
        m for m in _current_modalities() if m.name in _fd_over_cutoff
    ]
    if _fd_columns_needed:
        st.write(
            f"**Motion QC**: participants exceeding the "
            f"{GRAND_FD_CUTOFF_MM} mm mean framewise-displacement cutoff, "
            "per modality:"
        )
        fd_columns = st.columns(len(_fd_columns_needed))
        for column, m in zip(fd_columns, _fd_columns_needed):
            with column:
                st.metric(m.category, _fd_over_cutoff[m.name])
        st.caption(GRAND_PREPRINT_CITATION)

    for m in _current_modalities():
        qc = QC_NOTES.get(m.name)
        if qc is None:
            continue
        with st.expander(m.name):
            st.write(f"**What to inspect**: {qc['what_to_inspect']}")
            st.write(f"**Interpretation**: {qc['interpretation']}")

    st.write(
        "**Implications**: a modality that fails its own quality check "
        "should not proceed to the same processing and integration steps "
        "as one that passes; QC is a gate on the pipeline, not a summary "
        "written after the fact."
    )

    if stage < STAGE_PROCESS:
        if st.button("Continue to process separately", type="primary"):
            _advance_to(STAGE_PROCESS)

# -----------------------------------------------------------------
# 4. Process separately
# -----------------------------------------------------------------

if stage >= STAGE_PROCESS:
    section_header(
        "4. Process Separately",
        "Structural, functional, diffusion, and behavioral data require different preprocessing.",
    )

    current_modalities = _current_modalities()
    current_pipeline = pipeline_core.build_pipeline(current_modalities, convergence_stage="Inference")

    st.write(
        "Structural volumes are bias-field corrected and segmented into "
        "tissue types. Functional volumes are motion-corrected, "
        "slice-time corrected, and spatially normalized. Diffusion "
        "volumes are motion- and eddy-current-corrected, then used for "
        "tractography. Behavioral responses are scored trial-by-trial "
        "into accuracy and reaction-time summaries. None of these steps "
        "is shared across modalities."
    )

    st.vega_lite_chart(_pipeline_diagram_spec(current_pipeline), width="stretch")
    st.caption(
        "Each modality keeps its own Signals, Sensors, and Processing "
        "node; the diagram converges only at Inference, using "
        "modules/signal_pipeline/core's convergence_stage parameter "
        "(the illustrative EEG/ECG/EMG journey converges immediately "
        "after Sensors instead, both are the same underlying engine)."
    )

    if stage < STAGE_ALIGN:
        if st.button("Continue to align & derive features", type="primary"):
            _advance_to(STAGE_ALIGN)

# -----------------------------------------------------------------
# 5. Align & derive features
# -----------------------------------------------------------------

if stage >= STAGE_ALIGN:
    section_header(
        "5. Align & Derive Features",
        "Each modality's processed output is registered to a common space and reduced to one derived feature.",
    )

    current_modalities = _current_modalities()
    derived_rows = pd.DataFrame(
        [
            {"Modality": m.name, "Derived feature": DERIVED_FEATURE_BY_CATEGORY.get(m.category, "")}
            for m in current_modalities
        ]
    )
    st.dataframe(derived_rows, width="stretch", hide_index=True)

    st.write(
        "**What this shows**: alignment to a common space is what makes "
        "a derived feature from one modality comparable, region by "
        "region, to a derived feature from another. A cortical-thickness "
        "map and a tract-integrity map are only combinable once both are "
        "expressed in the same anatomical coordinates."
    )

    if stage < STAGE_INTEGRATE:
        if st.button("Continue to integrate evidence", type="primary"):
            _advance_to(STAGE_INTEGRATE)

# -----------------------------------------------------------------
# 6. Integrate evidence
# -----------------------------------------------------------------

if stage >= STAGE_INTEGRATE:
    section_header(
        "6. Integrate Evidence",
        "Modalities are combined only after acquisition, QC, processing, and alignment.",
    )

    current_modalities = _current_modalities()
    current_pipeline = pipeline_core.build_pipeline(current_modalities, convergence_stage="Inference")
    convergence = pipeline_core.compute_convergence(current_pipeline, stage="Inference")

    st.metric("Modalities integrated at Inference", convergence.n_modalities)
    st.write(f"Categories represented: **{', '.join(convergence.categories)}**.")

    st.write(
        "**Research considerations**: integration is a choice, motivated "
        "by the research question, not a default outcome of having "
        "acquired multiple modalities. A question about anatomy alone "
        "does not need Functional or Behavioral data integrated into it; "
        "a question about brain-behavior relationships does."
    )

    if stage < STAGE_EVALUATE:
        if st.button("Continue to evaluate added value", type="primary"):
            _advance_to(STAGE_EVALUATE)

# -----------------------------------------------------------------
# 7. Evaluate added value
# -----------------------------------------------------------------

if stage >= STAGE_EVALUATE:
    section_header(
        "7. Evaluate Added Value",
        "Does this modality add enough information to justify collecting and integrating it?",
    )

    st.warning(
        "The feature sets and performance figures below are illustrative "
        "numbers invented to demonstrate the method, not GRAND's own "
        "model-comparison results; this page did not have access to "
        "GRAND's raw imaging or per-subject model outputs. See the page's "
        "citation-integrity note, item 5."
    )

    with st.expander("Pearson correlation and coefficient of determination"):
        st.write(
            "Pearson correlation describes the correspondence between a "
            "model's predicted values and the observed values. The "
            "coefficient of determination (COD) instead describes "
            "predictive performance relative to predicting the sample "
            "mean for every case: a COD of 0 performs the same as that "
            "baseline, and a COD below 0 performs worse than it."
        )

    feature_sets = (
        feature_selection_core.FeatureSetResult("Structural only", 1, 0.18, 0.08),
        feature_selection_core.FeatureSetResult("+ Functional", 2, 0.34, 0.07),
        feature_selection_core.FeatureSetResult("+ Diffusion", 3, 0.37, 0.07),
        feature_selection_core.FeatureSetResult("+ Connectome", 4, 0.39, 0.09),
        feature_selection_core.FeatureSetResult("+ Behavioral", 5, 0.41, 0.10),
    )
    selection = feature_selection_core.select_necessary_feature_set(feature_sets)

    st.vega_lite_chart(_feature_set_spec(feature_sets, selection), width="stretch")
    st.caption(
        "Vertical lines show +/- 1 standard error around each illustrative "
        "point estimate."
    )

    st.write(
        f"**What this shows**: the best-performing illustrative feature "
        f"set is **{selection.best}**. The necessary feature set, the "
        f"fewest features whose performance is not distinguishable from "
        f"best by the one-standard-error rule, is **{selection.necessary}**."
    )
    with st.expander("The one-standard-error rule"):
        st.write(
            "Among feature sets whose performance is within one standard "
            "error of the best-observed performance, prefer the one with "
            "the fewest features. This is a convention for trading a "
            "small, uncertain performance gain against model simplicity, "
            "not a significance test."
        )
        st.caption(feature_selection_core.ONE_STANDARD_ERROR_RULE_CITATION)

    for name, within in selection.within_one_se_of_best.items():
        if not within:
            flagged_item_note(
                name,
                "Not within one standard error of the best illustrative "
                "feature set's performance.",
            )

    current_modalities = _current_modalities()
    if len(current_modalities) > 1:
        st.write("**Research considerations**: what each added modality contributed, per Step 2:")
        for m in current_modalities:
            if m.name in REDUNDANCY_NOTES:
                st.write(f"- **{_md(m.name)}**: {REDUNDANCY_NOTES[m.name]}")

    if stage < STAGE_INTERPRET:
        if st.button("Continue to interpret", type="primary"):
            _advance_to(STAGE_INTERPRET)

# -----------------------------------------------------------------
# 8. Interpret
# -----------------------------------------------------------------

if stage >= STAGE_INTERPRET:
    section_header("8. Interpret")

    current_modalities = _current_modalities()

    st.write("**What did each modality measure?**")
    for m in current_modalities:
        st.write(f"- {_md(m.name)}: {_md(m.signal_examples)}.")

    st.write("**Was its quality adequate for this analysis?**")
    st.write(
        f"Per Step 3: nearly all participants meet the "
        f"{GRAND_FD_CUTOFF_MM} mm motion cutoff used for the MRI "
        f"modalities: all but {GRAND_N_OVER_FD_CUTOFF_FUNCTIONAL} for "
        f"the functional task, all but {GRAND_N_OVER_FD_CUTOFF_DIFFUSION} "
        "for the longer diffusion sequence, and behavioral measures are "
        "reported as reliable, particularly the reaction-time-based ones, "
        "though the exact reliability figures remain unconfirmed (see the "
        "citation-integrity note, item 4)."
    )

    st.write("**Where did the task activate?**")
    st.write(
        "Per Step 3: the Semantic > Pseudofont contrast overlapped "
        "strongly with core language-critical regions in frontal and "
        "temporal cortex, consistent with the classic, left-lateralized "
        "language network described since Broca and Wernicke, background "
        "neuroscience offered as context, not a page-verified claim about "
        "what GRAND's own paper reports on lateralization specifically "
        "(citation-integrity note, item 7). That network's relevance here "
        "is not academic: BUILD and ReadMap, the parent studies GRAND's "
        "normative data supports, study how surviving brain regions take "
        "on new or expanded roles after a stroke damages part of this "
        "network: neuroplasticity in aphasia recovery."
    )

    st.write("**What information did it add?**")
    for m in current_modalities:
        if m.name in REDUNDANCY_NOTES:
            st.write(f"- {_md(m.name)}: {REDUNDANCY_NOTES[m.name]}")

    st.write("**Did integration materially improve the result?**")
    st.write(
        "Per Step 7's illustrative demonstration: performance increased "
        "with each added feature set, but not every increase exceeded one "
        "standard error, so not every addition would be judged necessary "
        "under that rule."
    )

    st.write("**What can the combined evidence actually support?**")
    st.write(
        "A statement about which modalities' derived features "
        "statistically improve prediction of the behavioral outcome in "
        "this illustrative demonstration, not a statement about which "
        "modalities are biologically necessary for reading and language, "
        "which would require evidence this page does not have."
    )

    st.divider()
    st.caption(GRAND_DATASET_CITATION)
    st.caption(GRAND_PREPRINT_CITATION)
