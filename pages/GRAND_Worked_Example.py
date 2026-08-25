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
8. Step 2's adaptive-staircase simulator uses a generic two-correct-in-
   a-row-to-increase, one-wrong-to-decrease rule to illustrate what
   "adaptive" means. GRAND's task is confirmed to use an adaptive
   staircase (Wilson, Yen, & Eriksson, 2018, WILSON_CITATION), but this
   page could not verify that paper's own up/down rule or step sizes, so
   the simulator does not claim to reproduce it.

Core logic lives in modules/signal_pipeline/core; this file is
presentation plus GRAND's own facts and citations, which are specific to
this worked example and so live here rather than in the reusable core.
"""

import math
import random
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
from shared.journey_stages import StageTracker
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
PIPELINE_NODE_COLOR = MUTED

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
CORNSWEET_CITATION = (
    "Cornsweet, T. N. (1962). The staircase-method in psychophysics. "
    "American Journal of Psychology, 75(3), 485-491. "
    "https://doi.org/10.2307/1419876"
)
LEVITT_CITATION = (
    "Levitt, H. (1971). Transformed up-down methods in psychoacoustics. "
    "Journal of the Acoustical Society of America, 49(2B), 467-477. "
    "https://doi.org/10.1121/1.1912375"
)
FUSION3D_CITATION = (
    "Biehler, M., Li, J., & Shi, J. (2025). FUSION3D: Multimodal data "
    "fusion for 3D shape reconstruction: A soft-sensing approach. IISE "
    "Transactions, 57(9), 1041-1055. "
    "https://doi.org/10.1080/24725854.2024.2376650"
)
# A real study, a different dataset, outcome, and modality set from
# GRAND's own five feature sets in Step 7 below -- checked directly
# against its own abstract, not recalled from memory (see that step's
# warning for why GRAND's own numbers are not substituted here).
KONOPKINA_CITATION = (
    "Konopkina, K., Buianova, I., Lal Khakpoor, F., Pornprasertmanit, S., "
    "Chan, M., & Pat, N. (2026). Multimodal MRI prediction of cognitive "
    "functioning across the lifespan: separating between-person "
    "differences from within-person changes. GeroScience. "
    "https://doi.org/10.1007/s11357-026-02441-2"
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

# One small, static, drawn glyph per modality, keyed by name rather than
# category since each modality's own real acquisition numbers -- already
# in signal_examples above -- differ even where two modalities share a
# category. The glyph is a mnemonic for the real numbers stated next to
# it in ACQUISITION_SUMMARY_BY_MODALITY below, not a substitute for them:
# a screen-reader user gets the real numbers from that text either way.
# Slice-stack line spacing/thickness is drawn coarser for a larger real
# voxel size, so Structural's fine hatching and Functional's fewer, thicker
# bars visually echo the same resolution-versus-coverage tradeoff QC_NOTES
# describes in words.
MODALITY_ACQUISITION_GLYPHS: dict[str, str] = {
    # A literal stack of thin, tightly-spaced slices, viewed edge-on --
    # the standard way "many thin slices" is drawn in radiology figures.
    # Seven slices here, deliberately more and thinner than Functional's
    # three below, using the same stack grammar so the two read as a
    # direct resolution contrast rather than two unrelated icons.
    "Structural MRI (T1w/FLAIR)": (
        '<rect x="5" y="4.0" width="22" height="2.6" rx="1" fill="{fill}"/>'
        '<rect x="5" y="7.6" width="22" height="2.6" rx="1" fill="{fill}"/>'
        '<rect x="5" y="11.2" width="22" height="2.6" rx="1" fill="{fill}"/>'
        '<rect x="5" y="14.8" width="22" height="2.6" rx="1" fill="{fill}"/>'
        '<rect x="5" y="18.4" width="22" height="2.6" rx="1" fill="{fill}"/>'
        '<rect x="5" y="22.0" width="22" height="2.6" rx="1" fill="{fill}"/>'
        '<rect x="5" y="25.6" width="22" height="2.6" rx="1" fill="{fill}"/>'
    ),
    # The same stack, now fewer and thicker: fewer, coarser slices.
    "Functional MRI (T2*, language task)": (
        '<rect x="5" y="5" width="22" height="6" rx="1.5" fill="{fill}"/>'
        '<rect x="5" y="13" width="22" height="6" rx="1.5" fill="{fill}"/>'
        '<rect x="5" y="21" width="22" height="6" rx="1.5" fill="{fill}"/>'
    ),
    # A gradient-direction sampling sphere: a center point (the voxel)
    # measured from several directions around it, each drawn as a dot at
    # the end of a spoke -- the same kind of plot a dMRI methods figure
    # would use to show its b-vector scheme, not an abstract starburst.
    "Diffusion MRI (HARDI/DTI)": (
        '<circle cx="16" cy="16" r="1.8" fill="{fill}"/>'
        '<line x1="16" y1="16" x2="27" y2="16" stroke="{fill}" stroke-width="1" opacity="0.5"/>'
        '<line x1="16" y1="16" x2="23.8" y2="8.2" stroke="{fill}" stroke-width="1" opacity="0.5"/>'
        '<line x1="16" y1="16" x2="16" y2="5" stroke="{fill}" stroke-width="1" opacity="0.5"/>'
        '<line x1="16" y1="16" x2="8.2" y2="8.2" stroke="{fill}" stroke-width="1" opacity="0.5"/>'
        '<line x1="16" y1="16" x2="5" y2="16" stroke="{fill}" stroke-width="1" opacity="0.5"/>'
        '<line x1="16" y1="16" x2="8.2" y2="23.8" stroke="{fill}" stroke-width="1" opacity="0.5"/>'
        '<line x1="16" y1="16" x2="16" y2="27" stroke="{fill}" stroke-width="1" opacity="0.5"/>'
        '<line x1="16" y1="16" x2="23.8" y2="23.8" stroke="{fill}" stroke-width="1" opacity="0.5"/>'
        '<circle cx="27" cy="16" r="1.7" fill="{fill}"/>'
        '<circle cx="23.8" cy="8.2" r="1.7" fill="{fill}"/>'
        '<circle cx="16" cy="5" r="1.7" fill="{fill}"/>'
        '<circle cx="8.2" cy="8.2" r="1.7" fill="{fill}"/>'
        '<circle cx="5" cy="16" r="1.7" fill="{fill}"/>'
        '<circle cx="8.2" cy="23.8" r="1.7" fill="{fill}"/>'
        '<circle cx="16" cy="27" r="1.7" fill="{fill}"/>'
        '<circle cx="23.8" cy="23.8" r="1.7" fill="{fill}"/>'
    ),
    "Connectome (derived)": _MODALITY_ICON_PATHS["Network"],
    "Behavioral reading/language measures": (
        '<path d="M6 16L10 20L16 10" stroke="{fill}" stroke-width="2.2" fill="none" stroke-linecap="round"/>'
        '<path d="M20 9L26 15M26 9L20 15" stroke="{fill}" stroke-width="2.2" stroke-linecap="round"/>'
        '<circle cx="16" cy="24" r="5" fill="none" stroke="{fill}" stroke-width="1.4"/>'
        '<line x1="16" y1="24" x2="16" y2="20.5" stroke="{fill}" stroke-width="1.2"/>'
        '<line x1="16" y1="24" x2="18.5" y2="24" stroke="{fill}" stroke-width="1.2"/>'
    ),
}

# The real numbers the glyphs above are mnemonics for -- already present
# in each modality's own signal_examples, repeated here as a short,
# stated summary rather than left implicit in the icon alone.
ACQUISITION_SUMMARY_BY_MODALITY: dict[str, str] = {
    "Structural MRI (T1w/FLAIR)": "T1w: 176 slices. FLAIR: 192 slices. Both at 1 mm3 voxels.",
    "Functional MRI (T2*, language task)": "48 slices at 2.9 mm3 voxels: fewer, coarser slices than structural, trading resolution for the speed a task-based scan needs.",
    "Diffusion MRI (HARDI/DTI)": "121 gradient directions (81 at b=3000, 40 at b=1200) plus 7 at b=0: 128 volumes total.",
    "Connectome (derived)": "Derived from already-acquired data: no independent scan time of its own.",
    "Behavioral reading/language measures": "Trial-wise accuracy and reaction time, scored outside the scanner.",
}


def _acquisition_glyph_html(name: str, color: str) -> str:
    """A small, static, centered glyph for one modality's real acquisition
    numbers -- see MODALITY_ACQUISITION_GLYPHS and
    ACQUISITION_SUMMARY_BY_MODALITY above for what it stands for and the
    real numbers it is a mnemonic for."""

    glyph = MODALITY_ACQUISITION_GLYPHS.get(name)
    if glyph is None:
        return ""

    icon = _icon_markup(glyph, color, 20, 20, 1.1, "")
    return f"""
    <div style="display:flex; justify-content:center;">
      <svg width="40" height="40" viewBox="0 0 40 40">{icon}</svg>
    </div>
    """


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


STAIRCASE_MIN_LEVEL = 1
STAIRCASE_MAX_LEVEL = 10
STAIRCASE_START_LEVEL = 1
STAIRCASE_N_TRIALS = 20


def _simulate_staircase(ability_level: int, n_trials: int = STAIRCASE_N_TRIALS, seed: int = 0) -> list[int]:
    """
    A generic 2-down-1-up adaptive staircase: two consecutive correct
    responses raise the difficulty one level, one incorrect response
    lowers it one level. This is illustrative of what "adaptive" means in
    an adaptive-staircase task, not a reproduction of GRAND's own
    algorithm -- see WILSON_CITATION for the actual paradigm, which this
    page cannot fully verify (see the citation-integrity note).

    ability_level is the level at which this simulated participant would
    respond correctly about half the time; responses below it are more
    often correct, responses above it less often, via a logistic curve.
    seed makes the same ability_level always draw the same run, so moving
    the slider back to a value you've already seen shows the same trace
    rather than a new random one.
    """
    rng = random.Random(seed + ability_level)
    level = STAIRCASE_START_LEVEL
    levels = [level]
    correct_streak = 0

    for _ in range(n_trials):
        p_correct = 1 / (1 + math.exp(level - ability_level))
        correct = rng.random() < p_correct

        if correct:
            correct_streak += 1
            if correct_streak >= 2:
                level = min(level + 1, STAIRCASE_MAX_LEVEL)
                correct_streak = 0
        else:
            correct_streak = 0
            level = max(level - 1, STAIRCASE_MIN_LEVEL)

        levels.append(level)

    return levels


def _staircase_reversals(levels: list[int]) -> list[int]:
    """
    Trial indices where the staircase changed direction (a reversal):
    the standard landmark psychophysics reads a staircase by (Cornsweet,
    1962; Levitt, 1971), since the level oscillating around a value,
    rather than the raw level at any one trial, is what shows the
    procedure has converged.
    """
    reversals: list[int] = []
    direction = 0

    for i in range(1, len(levels)):
        step = levels[i] - levels[i - 1]
        if step == 0:
            continue
        if direction != 0 and step != direction:
            reversals.append(i - 1)
        direction = step

    return reversals


def _staircase_threshold_estimate(levels: list[int], reversals: list[int]) -> float | None:
    """
    Mean level at the last several reversals -- the conventional way to
    read a threshold off a staircase (Cornsweet, 1962; Levitt, 1971). The
    first reversal is dropped when there are others to use, since it is
    biased by the arbitrary starting level rather than by the
    participant's responses yet. Returns None when too few reversals
    occurred in this short a run to estimate anything.
    """
    usable = reversals[1:] if len(reversals) > 1 else reversals
    if not usable:
        return None
    return sum(levels[i] for i in usable) / len(usable)


def _staircase_spec(levels: list[int], ability_level: int, reversals: list[int]) -> dict:
    rows = [{"trial": i, "level": level} for i, level in enumerate(levels)]
    reversal_rows = [{"trial": i, "level": levels[i]} for i in reversals]

    return {
        "layer": [
            {
                "data": {"values": [{"y": ability_level}]},
                "mark": {"type": "rule", "color": INK_SECONDARY, "strokeDash": [4, 4]},
                "encoding": {"y": {"field": "y", "type": "quantitative"}},
            },
            {
                "data": {"values": rows},
                "mark": {"type": "line", "color": PIPELINE_NODE_COLOR, "strokeWidth": 1.5},
                "encoding": {
                    "x": {"field": "trial", "type": "quantitative", "title": "Trial"},
                    "y": {
                        "field": "level",
                        "type": "quantitative",
                        "title": "Difficulty level",
                        "scale": {"domain": [STAIRCASE_MIN_LEVEL, STAIRCASE_MAX_LEVEL]},
                    },
                },
            },
            {
                "data": {"values": rows},
                "mark": {"type": "point", "filled": True, "size": 40, "color": "#2a78d6"},
                "encoding": {
                    "x": {"field": "trial", "type": "quantitative"},
                    "y": {"field": "level", "type": "quantitative"},
                },
            },
            {
                "data": {"values": reversal_rows},
                "mark": {"type": "point", "filled": True, "size": 100, "shape": "diamond", "color": "#eb6834"},
                "encoding": {
                    "x": {"field": "trial", "type": "quantitative"},
                    "y": {"field": "level", "type": "quantitative"},
                },
            },
        ],
        "config": _VEGA_CHART_CONFIG,
    }


FLOW_STAGE_LABELS = ("Signals", "Sensors", "Processing")
FLOW_COL_X = (60.0, 172.0, 284.0)
FLOW_INFERENCE_X = 396.0
FLOW_BOX_W = 88.0
FLOW_BOX_H = 34.0
FLOW_ROW_GAP = 56.0
FLOW_TOP_MARGIN = 44.0

# One icon per stage concept, not per modality: a reader asked for the
# "Signals" column to actually look like a signal rather than repeating
# the same per-modality icon (_MODALITY_ICON_PATHS, used elsewhere for
# the brain scene) in all three columns. Modality identity is still
# color-coded per row; shape now encodes which stage that box is.
FLOW_STAGE_ICON_PATHS = {
    "Signals": (
        '<path d="M3 16L8 16L11 8L14 24L17 12L20 20L23 10L26 16L29 16" '
        'stroke="{fill}" stroke-width="1.7" fill="none" stroke-linecap="round"/>'
    ),
    "Sensors": (
        '<rect x="10" y="10" width="12" height="12" rx="2" fill="{fill}" opacity="0.25"/>'
        '<rect x="13" y="13" width="6" height="6" rx="1" fill="{fill}"/>'
        '<line x1="16" y1="4" x2="16" y2="8" stroke="{fill}" stroke-width="1.4"/>'
        '<line x1="16" y1="24" x2="16" y2="28" stroke="{fill}" stroke-width="1.4"/>'
        '<line x1="4" y1="16" x2="8" y2="16" stroke="{fill}" stroke-width="1.4"/>'
        '<line x1="24" y1="16" x2="28" y2="16" stroke="{fill}" stroke-width="1.4"/>'
    ),
    "Processing": (
        '<circle cx="16" cy="16" r="6" fill="none" stroke="{fill}" stroke-width="2"/>'
        '<circle cx="16" cy="16" r="2" fill="{fill}"/>'
        '<line x1="16" y1="6" x2="16" y2="9" stroke="{fill}" stroke-width="2"/>'
        '<line x1="16" y1="23" x2="16" y2="26" stroke="{fill}" stroke-width="2"/>'
        '<line x1="6" y1="16" x2="9" y2="16" stroke="{fill}" stroke-width="2"/>'
        '<line x1="23" y1="16" x2="26" y2="16" stroke="{fill}" stroke-width="2"/>'
        '<line x1="9.5" y1="9.5" x2="11.5" y2="11.5" stroke="{fill}" stroke-width="2"/>'
        '<line x1="20.5" y1="20.5" x2="22.5" y2="22.5" stroke="{fill}" stroke-width="2"/>'
        '<line x1="9.5" y1="22.5" x2="11.5" y2="20.5" stroke="{fill}" stroke-width="2"/>'
        '<line x1="20.5" y1="11.5" x2="22.5" y2="9.5" stroke="{fill}" stroke-width="2"/>'
    ),
}
FLOW_INFERENCE_ICON = (
    '<circle cx="9" cy="9" r="2.4" fill="{fill}" opacity="0.6"/>'
    '<circle cx="9" cy="23" r="2.4" fill="{fill}" opacity="0.6"/>'
    '<circle cx="22" cy="9" r="2.4" fill="{fill}" opacity="0.6"/>'
    '<line x1="11" y1="10" x2="18" y2="15" stroke="{fill}" stroke-width="1.3" opacity="0.7"/>'
    '<line x1="11" y1="22" x2="18" y2="17" stroke="{fill}" stroke-width="1.3" opacity="0.7"/>'
    '<line x1="20" y1="10" x2="19" y2="15" stroke="{fill}" stroke-width="1.3" opacity="0.7"/>'
    '<circle cx="20" cy="16" r="3.5" fill="{fill}"/>'
)


def _flow_arrow(x1: float, y1: float, x2: float, y2: float, color: str) -> str:
    """One arrowed connector from (x1, y1) to (x2, y2), used for every edge
    in _pipeline_flow_html so the diagram reads as boxes-and-arrows rather
    than an axis of colored dots (see FUSION3D_CITATION for the converging-
    arrows convention this borrows, e.g. its Figure 3)."""

    angle = math.atan2(y2 - y1, x2 - x1)
    head_len = 7.0
    hx1 = x2 - head_len * math.cos(angle - math.pi / 7)
    hy1 = y2 - head_len * math.sin(angle - math.pi / 7)
    hx2 = x2 - head_len * math.cos(angle + math.pi / 7)
    hy2 = y2 - head_len * math.sin(angle + math.pi / 7)
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="1.6" opacity="0.65"/>'
        f'<polygon points="{x2:.1f},{y2:.1f} {hx1:.1f},{hy1:.1f} {hx2:.1f},{hy2:.1f}" '
        f'fill="{color}" opacity="0.85"/>'
    )


def _pipeline_flow_html(modalities: tuple[modality_core.Modality, ...], height: int = 260) -> str:
    """
    Each modality's own Signals -> Sensors -> Processing lane, converging
    with an arrow into one shared Inference box -- an original diagram,
    drawn as boxes and arrows rather than the axis-of-colored-dots this
    replaced, in the same converging-per-source-box convention used for
    multimodal-fusion methods figures (see FUSION3D_CITATION, e.g. its
    Figure 3). Decision/Action/Feedback (later pipeline_core.STAGES) are
    left out on purpose: this page's prose never discusses them, and
    drawing stages nothing here explains was part of why the previous
    version of this diagram was hard to read.
    """

    n = max(len(modalities), 1)
    scene_h = FLOW_TOP_MARGIN + n * FLOW_ROW_GAP + 16
    inference_center_y = FLOW_TOP_MARGIN + ((n - 1) * FLOW_ROW_GAP) / 2 + FLOW_BOX_H / 2

    header_svg = [
        f'<text x="{x + FLOW_BOX_W / 2:.0f}" y="{FLOW_TOP_MARGIN - 16:.0f}" text-anchor="middle" '
        f'font-size="11" fill="{INK_SECONDARY}">{label}</text>'
        for label, x in zip(FLOW_STAGE_LABELS, FLOW_COL_X)
    ]
    header_svg.append(
        f'<text x="{FLOW_INFERENCE_X + FLOW_BOX_W / 2:.0f}" y="{FLOW_TOP_MARGIN - 16:.0f}" '
        f'text-anchor="middle" font-size="11" fill="{INK_SECONDARY}">Inference</text>'
    )

    boxes_svg: list[str] = []
    arrows_svg: list[str] = []

    for row, m in enumerate(modalities):
        color = CATEGORY_COLORS[m.category]
        y = FLOW_TOP_MARGIN + row * FLOW_ROW_GAP
        row_center_y = y + FLOW_BOX_H / 2

        boxes_svg.append(
            f'<text x="0" y="{row_center_y + 3:.0f}" font-size="10" '
            f'fill="{INK_SECONDARY}">{m.category}</text>'
        )

        for col, x in enumerate(FLOW_COL_X):
            boxes_svg.append(
                f'<rect x="{x:.0f}" y="{y:.0f}" width="{FLOW_BOX_W:.0f}" '
                f'height="{FLOW_BOX_H:.0f}" rx="6" fill="{color}" opacity="0.16" '
                f'stroke="{color}" stroke-width="1.3"/>'
            )
            boxes_svg.append(
                _icon_markup(
                    FLOW_STAGE_ICON_PATHS[FLOW_STAGE_LABELS[col]], color,
                    x + FLOW_BOX_W / 2, row_center_y, (FLOW_BOX_H - 8) / 32, "",
                )
            )
            if col + 1 < len(FLOW_COL_X):
                arrows_svg.append(
                    _flow_arrow(x + FLOW_BOX_W, row_center_y, FLOW_COL_X[col + 1], row_center_y, color)
                )

        arrows_svg.append(
            _flow_arrow(
                FLOW_COL_X[-1] + FLOW_BOX_W, row_center_y,
                FLOW_INFERENCE_X, inference_center_y, color,
            )
        )

    boxes_svg.append(
        f'<rect x="{FLOW_INFERENCE_X:.0f}" y="{inference_center_y - FLOW_BOX_H / 2:.0f}" '
        f'width="{FLOW_BOX_W:.0f}" height="{FLOW_BOX_H:.0f}" rx="6" '
        f'fill="{PIPELINE_NODE_COLOR}" opacity="0.22" stroke="{PIPELINE_NODE_COLOR}" stroke-width="1.5"/>'
    )
    boxes_svg.append(
        _icon_markup(
            FLOW_INFERENCE_ICON, PIPELINE_NODE_COLOR,
            FLOW_INFERENCE_X + FLOW_BOX_W / 2, inference_center_y, (FLOW_BOX_H - 8) / 32, "",
        )
    )

    scene_w = FLOW_INFERENCE_X + FLOW_BOX_W + 12

    return f"""
    <div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
                background:{SURFACE}; border-radius:6px; padding:10px 4px;">
      <svg width="100%" height="{height}" viewBox="-4 0 {scene_w:.0f} {scene_h:.0f}"
           preserveAspectRatio="xMidYMid meet">
        {"".join(header_svg)}
        {"".join(arrows_svg)}
        {"".join(boxes_svg)}
      </svg>
    </div>
    """


PROCESSING_STEPS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "Structural": ("Raw volume", "Bias-field corrected", "Segmented into tissue types"),
    "Functional": (
        "Raw volume", "Motion-corrected", "Slice-time corrected", "Spatially normalized",
    ),
    "Diffusion": ("Raw volume", "Motion/eddy-current corrected", "Tractography"),
    "Behavioral": ("Raw trial responses", "Scored per trial", "Accuracy/RT summary"),
    # Connectome has no raw signal of its own to process here -- it is
    # derived downstream from already-processed Diffusion (and/or
    # Functional) data, per QC_NOTES above, not acquired and preprocessed
    # independently the way the other four modalities are.
    "Network": ("Derived from Diffusion/Functional", "No separate raw signal to process"),
}

# One drawn, static pictograph per step, each visually gesturing at what
# that step changes (noisy -> even shading, offset -> centered outline,
# scattered -> aligned points, and so on) rather than a generic colored
# box. No motion or flashing anywhere here on purpose: a "traveling
# highlight" was flagged as both a seizure-trigger risk (rapid, high-
# contrast flicker) and useless to a screen-reader user, and the plain-
# language step labels already carry the same information a moving
# highlight would only gesture at. Coordinates are drawn on the same
# 0-32 box _icon_markup already assumes for _MODALITY_ICON_PATHS.
PROCESSING_STEP_ICONS: dict[str, tuple[str, ...]] = {
    "Structural": (
        # Raw volume: uneven shading (hatching of mixed weight/opacity).
        '<rect x="4" y="4" width="24" height="24" rx="10" fill="{fill}" opacity="0.22"/>'
        '<line x1="8" y1="10" x2="14" y2="9" stroke="{fill}" stroke-width="2" opacity="0.9"/>'
        '<line x1="17" y1="13" x2="24" y2="11" stroke="{fill}" stroke-width="0.8" opacity="0.35"/>'
        '<line x1="9" y1="18" x2="16" y2="16" stroke="{fill}" stroke-width="1.6" opacity="0.6"/>'
        '<line x1="18" y1="22" x2="24" y2="23" stroke="{fill}" stroke-width="0.6" opacity="0.25"/>',
        # Bias-field corrected: same volume, evenly shaded.
        '<rect x="4" y="4" width="24" height="24" rx="10" fill="{fill}" opacity="0.4"/>'
        '<line x1="8" y1="12" x2="24" y2="12" stroke="{stroke_light}" stroke-width="1" opacity="0.6"/>'
        '<line x1="8" y1="20" x2="24" y2="20" stroke="{stroke_light}" stroke-width="1" opacity="0.6"/>',
        # Segmented into tissue types: nested regions of different density.
        '<rect x="4" y="4" width="24" height="24" rx="10" fill="{fill}" opacity="0.15"/>'
        '<circle cx="16" cy="16" r="9" fill="{fill}" opacity="0.35"/>'
        '<circle cx="16" cy="16" r="4.5" fill="{fill}" opacity="0.65"/>',
    ),
    "Functional": (
        # Raw volume: two offset outlines suggest motion blur/ghosting.
        '<path d="M6 9C10 6 20 6 24 9C27 12 27 19 24 23C20 27 10 27 6 23'
        'C3 19 3 12 6 9Z" stroke="{fill}" stroke-width="1.6" fill="none" opacity="0.9"/>'
        '<path d="M9 11C13 8 23 8 27 11C30 14 30 21 27 25C23 29 13 29 9 25'
        'C6 21 6 14 9 11Z" stroke="{fill}" stroke-width="1.6" fill="none" opacity="0.35"/>',
        # Motion-corrected: a single, centered outline.
        '<path d="M7 9C11 6 21 6 25 9C28 12 28 20 25 24C21 28 11 28 7 24'
        'C4 20 4 12 7 9Z" stroke="{fill}" stroke-width="1.8" fill="none"/>',
        # Slice-time corrected: evenly aligned parallel slices.
        '<rect x="6" y="6" width="20" height="20" rx="8" fill="{fill}" opacity="0.15"/>'
        '<line x1="6" y1="11" x2="26" y2="11" stroke="{fill}" stroke-width="1.4"/>'
        '<line x1="6" y1="16" x2="26" y2="16" stroke="{fill}" stroke-width="1.4"/>'
        '<line x1="6" y1="21" x2="26" y2="21" stroke="{fill}" stroke-width="1.4"/>',
        # Spatially normalized: resized to fit a standard-space template.
        '<rect x="4" y="4" width="24" height="24" rx="4" fill="none" '
        'stroke="{stroke_light}" stroke-width="1.2" stroke-dasharray="3 2"/>'
        '<rect x="8" y="8" width="16" height="16" rx="6" fill="{fill}" opacity="0.4"/>',
    ),
    "Diffusion": (
        # Raw volume: a scattered, unaligned point cloud.
        '<circle cx="8" cy="10" r="1.6" fill="{fill}"/><circle cx="15" cy="7" r="1.3" fill="{fill}"/>'
        '<circle cx="22" cy="11" r="1.7" fill="{fill}"/><circle cx="10" cy="19" r="1.4" fill="{fill}"/>'
        '<circle cx="19" cy="23" r="1.6" fill="{fill}"/><circle cx="25" cy="17" r="1.3" fill="{fill}"/>'
        '<circle cx="16" cy="15" r="1.5" fill="{fill}"/>',
        # Motion/eddy-current corrected: the same points, now on a grid.
        '<circle cx="10" cy="10" r="1.5" fill="{fill}"/><circle cx="16" cy="10" r="1.5" fill="{fill}"/>'
        '<circle cx="22" cy="10" r="1.5" fill="{fill}"/><circle cx="10" cy="16" r="1.5" fill="{fill}"/>'
        '<circle cx="16" cy="16" r="1.5" fill="{fill}"/><circle cx="22" cy="16" r="1.5" fill="{fill}"/>'
        '<circle cx="10" cy="22" r="1.5" fill="{fill}"/><circle cx="16" cy="22" r="1.5" fill="{fill}"/>'
        '<circle cx="22" cy="22" r="1.5" fill="{fill}"/>',
        # Tractography: the same flowing streamlines as the Diffusion
        # modality icon above, reused rather than redrawn.
        _MODALITY_ICON_PATHS["Diffusion"],
    ),
    "Behavioral": (
        # Raw trial responses: unscored right/wrong marks.
        '<path d="M5 15L9 19L15 9" stroke="{fill}" stroke-width="2.2" fill="none" stroke-linecap="round"/>'
        '<path d="M19 8L25 14M25 8L19 14" stroke="{fill}" stroke-width="2.2" stroke-linecap="round"/>'
        '<path d="M8 24L12 28L20 22" stroke="{fill}" stroke-width="2" fill="none" '
        'stroke-linecap="round" opacity="0.55"/>',
        # Scored per trial: each response now tallied in its own box.
        '<rect x="3" y="5" width="11" height="9" rx="2" fill="{fill}" opacity="0.18"/>'
        '<path d="M6 10L8 12L12 7" stroke="{fill}" stroke-width="1.6" fill="none" stroke-linecap="round"/>'
        '<rect x="17" y="5" width="11" height="9" rx="2" fill="{fill}" opacity="0.18"/>'
        '<path d="M19 7L26 13M26 7L19 13" stroke="{fill}" stroke-width="1.6" stroke-linecap="round"/>'
        '<rect x="10" y="18" width="11" height="9" rx="2" fill="{fill}" opacity="0.18"/>'
        '<path d="M12 22L14 25L19 20" stroke="{fill}" stroke-width="1.6" fill="none" stroke-linecap="round"/>',
        # Accuracy/RT summary: a small bar-chart summary.
        '<line x1="4" y1="28" x2="28" y2="28" stroke="{stroke_light}" stroke-width="1.2"/>'
        '<rect x="7" y="16" width="5" height="12" fill="{fill}"/>'
        '<rect x="14" y="10" width="5" height="18" fill="{fill}" opacity="0.75"/>'
        '<rect x="21" y="20" width="5" height="8" fill="{fill}" opacity="0.5"/>',
    ),
    "Network": (
        # Derived from Diffusion/Functional: two sources converging into one.
        '<circle cx="6" cy="8" r="3" fill="{fill}" opacity="0.5"/>'
        '<circle cx="6" cy="24" r="3" fill="{fill}" opacity="0.5"/>'
        '<path d="M10 9L20 15M10 23L20 17" stroke="{fill}" stroke-width="1.4" opacity="0.65"/>'
        '<circle cx="24" cy="16" r="4" fill="{fill}"/>',
        # No separate raw signal: the same network icon used elsewhere here.
        _MODALITY_ICON_PATHS["Network"],
    ),
}

STEP_SLOT_W = 118.0
STEP_SLOT_GAP = 34.0
STEP_BADGE_SIZE = 40.0
STEP_ROW_GAP = 70.0
STEP_TOP_MARGIN = 14.0


def _processing_steps_html(modalities: tuple[modality_core.Modality, ...], height: int = 280) -> str:
    """
    One static, drawn pictograph per processing step per modality (Raw
    -> ... -> ready for alignment), each icon visually showing what that
    step changes rather than an animated or generic placeholder. Static
    on purpose: an earlier flickering/pulsing version was both a
    seizure-trigger risk (rapid, high-contrast flashing) and added
    nothing for a screen-reader user, who already has the same
    information in the step labels and the surrounding prose. These are
    original, generic icons, not real preprocessed images or a
    reproduction of any specific software's output.
    """

    n = max(len(modalities), 1)
    scene_h = STEP_TOP_MARGIN + n * STEP_ROW_GAP + 10

    rows_svg: list[str] = []
    max_steps = 1

    for row, m in enumerate(modalities):
        steps = PROCESSING_STEPS_BY_CATEGORY.get(m.category, ())
        icons = PROCESSING_STEP_ICONS.get(m.category, ())
        max_steps = max(max_steps, len(steps))
        color = CATEGORY_COLORS[m.category]
        y = STEP_TOP_MARGIN + row * STEP_ROW_GAP
        badge_cy = y + STEP_BADGE_SIZE / 2

        rows_svg.append(
            f'<text x="0" y="{y - 6:.0f}" font-size="10" fill="{INK_SECONDARY}">{m.category}</text>'
        )

        badge_x_by_step = [
            i * (STEP_SLOT_W + STEP_SLOT_GAP) + (STEP_SLOT_W - STEP_BADGE_SIZE) / 2
            for i in range(len(steps))
        ]

        for i, step_label in enumerate(steps):
            slot_x = i * (STEP_SLOT_W + STEP_SLOT_GAP)
            badge_x = badge_x_by_step[i]
            badge_cx = badge_x + STEP_BADGE_SIZE / 2

            if i + 1 < len(steps):
                rows_svg.append(
                    _flow_arrow(
                        badge_x + STEP_BADGE_SIZE, badge_cy,
                        badge_x_by_step[i + 1], badge_cy,
                        GRIDLINE,
                    )
                )

            rows_svg.append(
                f'<rect x="{badge_x:.0f}" y="{y:.0f}" width="{STEP_BADGE_SIZE:.0f}" '
                f'height="{STEP_BADGE_SIZE:.0f}" rx="8" fill="{color}" opacity="0.10" '
                f'stroke="{color}" stroke-width="1.1"/>'
            )
            if i < len(icons):
                rows_svg.append(
                    _icon_markup(icons[i], color, badge_cx, badge_cy, STEP_BADGE_SIZE / 32, "")
                )
            rows_svg.append(
                f'<text x="{slot_x + STEP_SLOT_W / 2:.0f}" y="{y + STEP_BADGE_SIZE + 13:.0f}" '
                f'text-anchor="middle" font-size="9.5" fill="{INK_SECONDARY}">{step_label}</text>'
            )

    scene_w = max_steps * (STEP_SLOT_W + STEP_SLOT_GAP) - STEP_SLOT_GAP + 8

    return f"""
    <div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
                background:{SURFACE}; border-radius:6px; padding:10px 4px;">
      <svg width="100%" height="{height}" viewBox="-4 0 {scene_w:.0f} {scene_h:.0f}"
           preserveAspectRatio="xMidYMid meet">
        {"".join(rows_svg)}
      </svg>
    </div>
    """


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

# One small, static icon per QC concept named in "what to inspect" above,
# so a reader gets a quick visual sense of what that check actually
# looks like rather than only prose. Static, not animated, for the same
# accessibility reasons as the other icons on this page (see
# PROCESSING_STEP_ICONS above). Two are reused rather than redrawn:
# "derived_from" and "artifact_check" are the same icons already used
# for the Connectome and Functional processing steps.
QC_VISUAL_ICON_PATHS: dict[str, str] = {
    "motion_trace": (
        '<line x1="2" y1="10" x2="30" y2="10" stroke="{fill}" stroke-width="1" '
        'stroke-dasharray="2 2" opacity="0.55"/>'
        '<path d="M2 22L6 20L10 23L14 8L18 21L22 19L26 22L30 20" stroke="{fill}" '
        'stroke-width="1.6" fill="none" stroke-linecap="round"/>'
        '<circle cx="14" cy="8" r="2.4" fill="none" stroke="{fill}" stroke-width="1.3"/>'
    ),
    "tsnr_map": (
        '<circle cx="16" cy="16" r="13" fill="{fill}" opacity="0.12"/>'
        '<circle cx="16" cy="16" r="9" fill="{fill}" opacity="0.28"/>'
        '<circle cx="16" cy="16" r="5" fill="{fill}" opacity="0.55"/>'
    ),
    # Two semi-transparent, overlapping regions: SVG's own alpha
    # compositing makes the overlap visibly darker on its own, without a
    # separately drawn "agreement" shape.
    "activation_overlap": (
        '<circle cx="12" cy="16" r="9" fill="{fill}" opacity="0.3"/>'
        '<circle cx="20" cy="16" r="9" fill="{fill}" opacity="0.3"/>'
    ),
    "split_half": (
        '<rect x="5" y="14" width="8" height="14" rx="2" fill="{fill}" opacity="0.4"/>'
        '<rect x="19" y="14" width="8" height="14" rx="2" fill="{fill}" opacity="0.4"/>'
        '<path d="M13 10C16 4 16 4 19 10" stroke="{fill}" stroke-width="1.4" fill="none"/>'
        '<polygon points="13,10 15.5,9.3 14.6,12.6" fill="{fill}"/>'
        '<polygon points="19,10 16.5,9.3 17.4,12.6" fill="{fill}"/>'
    ),
    "derived_from": PROCESSING_STEP_ICONS["Network"][0],
    "artifact_check": PROCESSING_STEP_ICONS["Functional"][0],
}

QC_VISUAL_LABELS: dict[str, str] = {
    "motion_trace": "Motion trace vs. cutoff",
    "tsnr_map": "tSNR: high center, low edges",
    "activation_overlap": "Task map vs. reference overlap",
    "split_half": "Split-half agreement",
    "derived_from": "Inherits source's QC",
    "artifact_check": "Motion/ghosting check",
}

QC_VISUAL_KEYS_BY_MODALITY: dict[str, tuple[str, ...]] = {
    "Structural MRI (T1w/FLAIR)": ("artifact_check",),
    "Functional MRI (T2*, language task)": ("motion_trace", "tsnr_map", "activation_overlap"),
    "Diffusion MRI (HARDI/DTI)": ("motion_trace",),
    "Connectome (derived)": ("derived_from",),
    "Behavioral reading/language measures": ("split_half",),
}


def _qc_visual_row_html(name: str, color: str) -> str:
    keys = QC_VISUAL_KEYS_BY_MODALITY.get(name, ())
    if not keys:
        return ""

    items = []
    for key in keys:
        icon = _icon_markup(QC_VISUAL_ICON_PATHS[key], color, 16, 16, 1.0, "")
        items.append(
            f'<div style="display:flex; flex-direction:column; align-items:center; '
            f'width:132px; gap:2px;">'
            f'<svg width="32" height="32" viewBox="0 0 32 32">{icon}</svg>'
            f'<span style="font-size:10px; line-height:1.25; color:{INK_SECONDARY}; '
            f'text-align:center;">{QC_VISUAL_LABELS[key]}</span>'
            f"</div>"
        )

    return f"""
    <div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
                background:{SURFACE}; border-radius:6px; padding:8px 4px;
                display:flex; flex-wrap:wrap; justify-content:center; gap:6px;">
      {"".join(items)}
    </div>
    """


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

TRACKER = StageTracker(session_key=STAGE_KEY, stage_labels=JOURNEY_STAGES)


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
    "A guided case study: each stage unlocks after you make a decision "
    "or inspect its consequence."
)

render_data_handling_summary(disclosure_for("pages/GRAND_Worked_Example.py"))

stage = TRACKER.render_breadcrumb()

TRACKER.render_restart_button(extra_session_keys=(SELECTED_MODALITIES_KEY,))

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

rq_col1, rq_col2 = st.columns(2)
with rq_col1:
    st.badge("Five modalities", icon=":material/hub:", color="blue")
with rq_col2:
    st.badge("Reading & language", icon=":material/menu_book:", color="blue")

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
        f"According to the dataset documentation, the primary objective is "
        f"{GRAND_PURPOSE}. Data acquisition occurred at {GRAND_SITE}, "
        f"using a {GRAND_SCANNER}. The functional protocol was "
        f"{GRAND_TASK_NAME}."
    )
    st.write(
        f"While the preprint details {GRAND_N_PARTICIPANTS_PREPRINT_ABSTRACT} "
        f"participants, comprising {GRAND_N_BUILD} from BUILD and "
        f"{GRAND_N_READMAP} from ReadMap, the released participants.tsv "
        f"file contains only {GRAND_N_PARTICIPANTS} entries. The existing "
        "documentation does not account for this discrepancy; for further "
        "details, refer to item 2 in the citation-integrity note."
    )
    st.caption(GRAND_DATASET_CITATION)
    st.caption(GRAND_PREPRINT_CITATION)
    st.caption(WILSON_CITATION)
    st.caption(BUILD_CITATION)
    st.caption(READMAP_CITATION)

if stage < STAGE_ACQUIRE:
    if st.button("Begin study", type="primary"):
        TRACKER.advance_to(STAGE_ACQUIRE)

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
        with st.expander(f"{m.name} Measures"):
            st.caption(f"Examples: {m.signal_examples}.")

            glyph_col, summary_col = st.columns([1, 5])
            with glyph_col:
                components.html(
                    _acquisition_glyph_html(m.name, CATEGORY_COLORS[m.category]),
                    height=44,
                )
            with summary_col:
                st.caption(ACQUISITION_SUMMARY_BY_MODALITY.get(m.name, ""))

            st.write(
                f"Interpretive gain (illustrative): **{m.interpretive_gain:.2f}** "
                "/ 1.00 -- how much information this modality adds that "
                "the others don't already carry, higher is more. Cost "
                f"(illustrative): privacy **{m.privacy_cost:.2f}**, "
                f"security **{m.security_cost:.2f}**, agency "
                f"**{m.agency_cost:.2f}** -- each 0 = negligible, 1 = "
                "severe. See 'Research considerations: acquisition' "
                "below for what these scores are rated against."
            )
            st.write(m.notes)
            st.caption(m.citation)

    # Not gated on Functional MRI being selected above: the task happened
    # for every GRAND participant regardless of which modality a reader
    # is currently exploring in this journey, so hiding it behind that
    # selection just made it hard to find.
    st.write(f"**How {GRAND_TASK_NAME} adapts to each participant**")
    st.write(
        "Every participant performed this task during the functional "
        "scan. It is adaptive: a participant doing well sees harder "
        "trials, and a participant struggling sees easier ones, so most "
        "trials land near that participant's own threshold rather than "
        "being wasted on trials that are trivially easy or effectively "
        "impossible."
    )

    st.write(
        "The objective of the following demonstration is to illustrate "
        "how a staircase algorithm operates without direct access to a "
        "participant's actual aptitude, relying solely on the accuracy "
        "of individual responses. By establishing a \"true\" proficiency "
        "level, you can observe the algorithm, unaware of your preset "
        "value, identify that threshold through response patterns "
        "alone. This utility explains the selection of adaptive designs "
        f"for research encompassing ages {GRAND_AGE_MIN:.0f} to "
        f"{GRAND_AGE_MAX:.0f}; while static trials might prove either "
        "trivial or insurmountable for various individuals, a staircase "
        "methodology determines a personalized level independent of the "
        "initial starting point."
    )

    ability_level = st.slider(
        "This simulated participant's true ability level (hidden from the algorithm)",
        min_value=STAIRCASE_MIN_LEVEL,
        max_value=STAIRCASE_MAX_LEVEL,
        value=5,
        help=(
            "The difficulty level at which this simulated participant "
            "would respond correctly about half the time. Only you can "
            "see this value; the staircase below only ever sees right or "
            "wrong."
        ),
    )
    staircase_levels = _simulate_staircase(ability_level)
    reversals = _staircase_reversals(staircase_levels)
    threshold_estimate = _staircase_threshold_estimate(staircase_levels, reversals)

    st.vega_lite_chart(
        _staircase_spec(staircase_levels, ability_level, reversals), width="stretch"
    )
    st.caption(
        "Dashed line = the true level you set. Diamonds = reversals, the "
        "trials where the staircase switched from rising to falling or "
        "back -- the conventional landmark for reading a threshold off a "
        "staircase (see citations below)."
    )

    if threshold_estimate is None:
        st.write(
            "Too few reversals occurred in this short a run to estimate a "
            "threshold; a real study runs more trials or a longer "
            "staircase than this 20-trial illustration."
        )
    else:
        st.write(
            f"**Threshold estimate from this run: {threshold_estimate:.1f}.** "
            f"Computed as the mean level across the reversals above, "
            "dropping the first (biased by the arbitrary starting level, "
            f"not yet by this participant's responses). Compare it to the "
            f"true level you set, **{ability_level}**: the algorithm never "
            "saw that number, only right/wrong answers, and still landed "
            "close to it."
        )

    caveat(
        "This is a generic two-correct-in-a-row-to-increase, "
        "one-wrong-to-decrease staircase, illustrating what "
        "\"adaptive\" means and how a threshold is conventionally read "
        "off one, not a reproduction of GRAND's own algorithm. See the "
        "first citation below for the actual paradigm, which this page "
        "cannot fully verify (see the citation-integrity note)."
    )
    st.caption(WILSON_CITATION)
    st.caption(CORNSWEET_CITATION)
    st.caption(LEVITT_CITATION)

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
            TRACKER.advance_to(STAGE_QC)

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
            visual_row = _qc_visual_row_html(m.name, CATEGORY_COLORS[m.category])
            if visual_row:
                components.html(visual_row, height=90)
            st.write(f"**Interpretation**: {qc['interpretation']}")

    st.write(
        "**Implications**: a modality that fails its own quality check "
        "should not proceed to the same processing and integration steps "
        "as one that passes; QC is a gate on the pipeline, not a summary "
        "written after the fact."
    )

    if stage < STAGE_PROCESS:
        if st.button("Continue to process separately", type="primary"):
            TRACKER.advance_to(STAGE_PROCESS)

# -----------------------------------------------------------------
# 4. Process separately
# -----------------------------------------------------------------

if stage >= STAGE_PROCESS:
    section_header(
        "4. Process Separately",
        "Structural, functional, diffusion, and behavioral data require different preprocessing.",
    )

    current_modalities = _current_modalities()

    st.write(
        "Structural volumes are bias-field corrected and segmented into "
        "tissue types. Functional volumes are motion-corrected, "
        "slice-time corrected, and spatially normalized. Diffusion "
        "volumes are motion- and eddy-current-corrected, then used for "
        "tractography. Behavioral responses are scored trial-by-trial "
        "into accuracy and reaction-time summaries. None of these steps "
        "is shared across modalities."
    )

    components.html(
        _processing_steps_html(current_modalities),
        height=280,
    )
    st.caption(
        "Generic, drawn icons standing in for each modality's own "
        "processing steps above, not real preprocessed images: each one "
        "gestures at what that step changes (for example, uneven "
        "shading becoming even, or scattered points becoming aligned), "
        "in the order named just above."
    )

    components.html(
        _pipeline_flow_html(current_modalities),
        height=260,
    )
    st.caption(
        "Alt text: Each modality keeps its own Signals, Sensors, and "
        "Processing box; all four converge with an arrow only at "
        "Inference, using modules/signal_pipeline/core's "
        "convergence_stage parameter (the illustrative EEG/ECG/EMG "
        "journey converges immediately after Sensors instead, both are "
        "the same underlying engine). Decision, Action, and Feedback, "
        "later stages this same engine supports, are left off the "
        "diagram because this page's prose never discusses them."
    )
    st.caption(
        "Diagram style, not the pipeline itself, follows the same "
        "converging-per-source-box convention used in multimodal-fusion "
        "methods figures:"
    )
    st.caption(FUSION3D_CITATION)

    if stage < STAGE_ALIGN:
        if st.button("Continue to align & derive features", type="primary"):
            TRACKER.advance_to(STAGE_ALIGN)

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
            TRACKER.advance_to(STAGE_INTEGRATE)

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
        "**Research considerations**: data integration represents a "
        "deliberate methodological decision driven by specific research "
        "objectives, rather than an automatic consequence of multi-modal "
        "acquisition. While inquiries focused solely on structural "
        "anatomy do not require the incorporation of behavioral or "
        "functional datasets, investigations concerning brain-behavior "
        "correlations necessitate such a synthesized approach."
    )

    if stage < STAGE_EVALUATE:
        if st.button("Continue to evaluate added value", type="primary"):
            TRACKER.advance_to(STAGE_EVALUATE)

# -----------------------------------------------------------------
# 7. Evaluate added value
# -----------------------------------------------------------------

if stage >= STAGE_EVALUATE:
    section_header(
        "7. Evaluate Added Value",
        "Does this modality add enough information to justify collecting and integrating it?",
    )

    st.write(
        "For a real sense of whether this kind of pattern shows up in "
        "practice: a 2026 study of 450 adults (ages 21-90, the Dallas "
        "Lifespan Brain Study) found that stacking five real MRI "
        "modalities (task fMRI, functional connectivity, structural MRI, "
        "diffusion-weighted imaging, and arterial spin labeling) to "
        "predict cognitive functioning explained about half the variance "
        "(R² ≈ .51), more than any single modality alone, with "
        "diffusion-weighted imaging and functional connectivity the "
        "next-strongest individually. That is a different dataset, "
        "outcome, and modality set from GRAND's own, so it cannot "
        "substitute for GRAND's own numbers -- but it is real evidence "
        "that combining modalities can genuinely add predictive value."
    )
    st.caption(KONOPKINA_CITATION)

    with st.expander("Pearson correlation and coefficient of determination"):
        st.write(
            "Pearson correlation describes the correspondence between a "
            "model's predicted values and the observed values. The "
            "coefficient of determination (COD) instead describes "
            "predictive performance relative to predicting the sample "
            "mean for every case: a COD of 0 performs the same as that "
            "baseline, and a COD below 0 performs worse than it."
        )

    st.write(
        "**A preliminary comparative analysis derived from GRAND's "
        "publicly accessible records**"
    )
    st.write(
        "Dual feature sets are authentically calculated from GRAND's "
        "public derivatives hosted on OpenNeuro (ds007831). For 109 out "
        "of the 110 participants included in the release, excluding one "
        "individual due to an unusable connectome file, structural "
        "connectomes utilizing the Brainnetome atlas and Semantic > "
        "Pseudofont functional beta maps were acquired. These datasets "
        "were condensed into representative values: the mean weight of "
        "nonzero edges for connectomes, and the mean absolute value "
        "across nonzero voxels for beta maps. A linear regression model "
        "targets **age at the time of MRI** as the dependent variable, "
        "rather than linguistic or reading proficiency; since "
        "trial-level behavioral data remains excluded from the "
        "accessible public release, age serves as the sole tangible "
        "outcome available for pairing with authentic GRAND features. "
        "The reported performance metrics and associated standard "
        "errors reflect the mean and standard error of R² across 15 "
        "validation folds, utilizing a 5-fold cross-validation protocol "
        "repeated thrice with varied random partitions. See the page's "
        "citation-integrity note, item 5, for why GRAND's own preprint "
        "could not supply this comparison instead."
    )

    real_feature_sets = (
        feature_selection_core.FeatureSetResult("Connectome only", 1, 0.0030, 0.0093),
        feature_selection_core.FeatureSetResult("+ Functional", 2, -0.0266, 0.0172),
    )
    real_selection = feature_selection_core.select_necessary_feature_set(real_feature_sets)

    st.vega_lite_chart(_feature_set_spec(real_feature_sets, real_selection), width="stretch")
    st.caption(
        "Vertical lines show +/- 1 standard error around each real, "
        "computed point estimate. R² ≈ 0.003 (Connectome only) "
        "and R² ≈ -0.027 (+ Functional); the chart's tooltip "
        "rounds to two decimals, which is not enough to show that "
        "Connectome only's estimate is slightly positive rather than "
        "exactly zero."
    )
    st.write(
        f"**What this shows**: both real R² values are close to "
        "zero, and adding the functional feature made the "
        "cross-validated estimate worse, not better. Under the "
        "one-standard-error rule, the necessary feature set here is "
        f"**{real_selection.necessary}**; neither feature set shows "
        "clear evidence of predicting age from these two coarse, global "
        "summaries."
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

    for name, within in real_selection.within_one_se_of_best.items():
        if not within:
            flagged_item_note(
                name,
                "Not within one standard error of the best real, "
                "computed feature set's performance.",
            )

    caveat(
        "This is a deliberately blunt analysis, not a claim that "
        "GRAND's connectome or functional data carry no real signal: a "
        "single whole-brain mean throws away almost all spatial "
        "information, and a richer feature set, like the per-region "
        "features a full analysis (or Konopkina et al.'s stacked-"
        "modality model above) would use, could look very different. "
        "What this real comparison does show honestly is what a fast, "
        "coarse first pass on real GRAND data actually looks like, not "
        "necessarily the steadily-improving pattern combining modalities "
        "can produce in a larger, richer analysis like Konopkina et "
        "al.'s above."
    )

    current_modalities = _current_modalities()
    if len(current_modalities) > 1:
        st.write("**Research considerations**: what each added modality contributed, per Step 2:")
        for m in current_modalities:
            if m.name in REDUNDANCY_NOTES:
                st.write(f"- **{_md(m.name)}**: {REDUNDANCY_NOTES[m.name]}")

    if stage < STAGE_INTERPRET:
        if st.button("Continue to interpret", type="primary"):
            TRACKER.advance_to(STAGE_INTERPRET)

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
        "Per Step 7's real, computed comparison: no. Adding the "
        "functional feature made the cross-validated estimate worse, "
        "not better, and neither feature set's real R² was clearly "
        "distinguishable from zero, so this specific pairing shows no "
        "support for integration mattering, at least for two coarse "
        "global summaries predicting age."
    )

    st.write("**What can the combined evidence actually support?**")
    st.write(
        "A statement about whether two coarse, real feature sets "
        "(a connectome-strength summary and a functional-activation "
        "summary) improve prediction of age at MRI in this specific "
        "reduced analysis, not a statement about which modalities are "
        "biologically necessary for reading and language, which would "
        "require the trial-wise behavioral outcome data this page could "
        "not access."
    )

    st.write("**Could this data support a fairness check?**")
    st.write(
        "Yes: participants.tsv records sex, race, ethnicity, education, "
        "and handedness alongside age. Checking whether a predictive "
        "model built on this data performs consistently across those "
        "groups is exactly the kind of question OpenMeasure's Fairness "
        "module is for; this page does not run that check itself."
    )

    st.divider()
    st.caption(GRAND_DATASET_CITATION)
    st.caption(GRAND_PREPRINT_CITATION)
