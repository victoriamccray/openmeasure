"""
AI & Environmental Equity: GAIA - a guided research journey, not a workflow.

Teaches a validation question none of the numbered workflows cover: when
is a more efficient model appropriate to replace a larger one, once
resource use (energy, CO2, size) is weighed alongside predictive
performance? Uses GAIA (Jallais, Mancini, & Palombo, 2026) - a real study
comparing a Teacher/Light-Model/Student trio of U-Nets for diffusion MRI
signal prediction via knowledge distillation - as the worked example.

Like the other Research Journeys, this page records nothing to
shared/handoff.py, carries no module_key, and is deliberately not a
numbered page, so it needs no entry in shared/catalog.py.

Core logic (domain-agnostic: performance/resource tradeoffs for any set
of models, not specific to GAIA or medical imaging) lives in
modules/model_efficiency/core; this file is presentation plus GAIA's own
facts and citations, which are specific to this worked example and so
live here rather than in the reusable core.

Citation-integrity note: the GAIA abstract gives exact figures for
parameter counts, training setup, and four relative percentages (model
size, energy/CO2, per-deployment CO2, CO2 per million uses). It gives no
exact absolute MSE or CO2 value anywhere in its text - only in an
unlabeled scatter plot (its Figure 5). Every value read from that figure
is marked approximate wherever it appears below, kept visually secondary
to the exact quoted findings, and is pending confirmation from the
authors. The paper's stated 78% model-size reduction is also presented
as-is, without attempting to reconcile it against the exact parameter
counts, which would imply a much larger reduction - see modules/
model_efficiency/README.md for the full explanation.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from modules.model_efficiency.core import deployment as deployment_core
from modules.model_efficiency.core import frontier as frontier_core
from modules.model_efficiency.core import models as models_core
from modules.model_efficiency.core import preference as preference_core
from shared.data_handling import disclosure_for, render_data_handling_summary
from shared.report import caveat, flagged_item_note, section_header

SAMPLE_DIR = ROOT / "modules" / "model_efficiency" / "sample_data"

# Validated default palette (dataviz skill, references/palette.md), same
# values used elsewhere in this app (e.g. Portfolio_Impact_Analysis.py).
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
CATEGORICAL_1 = "#2a78d6"
CATEGORICAL_2 = "#eb6834"
# Palette slot 3 (aqua): the reference palette's first three slots are the
# ones validated for all-pairs scatter use, needed here for a 3-model chart.
CATEGORICAL_3 = "#1baf7a"

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


def _svg_figure(inner: str, caption: str, viewbox: str = "0 0 220 64") -> None:
    """
    One small static pictograph (never animated - see PIPELINE_STEPS'
    comment on this project's move away from flickering step animation)
    illustrating a glossary term, with its caption below.
    """

    st.markdown(
        f'<div style="text-align:center;">'
        f'<svg width="100%" height="64" viewBox="{viewbox}" '
        f'preserveAspectRatio="xMidYMid meet">{inner}</svg>'
        f'<div style="color:{INK_SECONDARY}; font-size:0.8rem;">{caption}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _b_value_svg() -> str:
    """Three shells of increasing diffusion weighting, left to right."""

    circles = "".join(
        f'<circle cx="{cx}" cy="32" r="{r}" fill="{CATEGORICAL_1}" opacity="{op}" />'
        for cx, r, op in ((40, 7, 0.35), (110, 11, 0.65), (180, 15, 1.0))
    )
    return (
        f"{circles}"
        f'<line x1="15" y1="55" x2="205" y2="55" stroke="{INK_SECONDARY}" '
        f'stroke-width="1.5" marker-end="url(#gaia-arrow)" />'
        f'<defs><marker id="gaia-arrow" markerWidth="8" markerHeight="8" '
        f'refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" '
        f'fill="{INK_SECONDARY}" /></marker></defs>'
    )


def _powder_average_svg() -> str:
    """Several directional signals (spokes) collapsing to one voxel value."""

    spokes = "".join(
        f'<line x1="40" y1="32" '
        f'x2="{40 + 20 * math.cos(a):.1f}" y2="{32 + 20 * math.sin(a):.1f}" '
        f'stroke="{CATEGORICAL_1}" stroke-width="2" />'
        for a in (i * math.pi / 4 for i in range(8))
    )
    return (
        f"{spokes}"
        f'<circle cx="40" cy="32" r="3" fill="{INK_SECONDARY}" />'
        f'<line x1="75" y1="32" x2="130" y2="32" stroke="{INK_SECONDARY}" '
        f'stroke-width="1.5" marker-end="url(#gaia-arrow2)" />'
        f'<defs><marker id="gaia-arrow2" markerWidth="8" markerHeight="8" '
        f'refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" '
        f'fill="{INK_SECONDARY}" /></marker></defs>'
        f'<circle cx="175" cy="32" r="13" fill="{CATEGORICAL_2}" />'
    )


def _pulse_sequence_svg(highlight: str) -> str:
    """
    PGSE's two gradient pulses (duration delta) separated by the
    diffusion time (Delta), with whichever part the caller names in
    highlight drawn in the accent color and the rest muted.
    """

    pulse_color = CATEGORICAL_1 if highlight == "pulses" else GRIDLINE
    gap_color = CATEGORICAL_2 if highlight == "gap" else INK_SECONDARY
    pulse_stroke = INK_SECONDARY if highlight == "pulses" else "#c3c2b7"
    return (
        f'<line x1="10" y1="45" x2="210" y2="45" stroke="{GRIDLINE}" stroke-width="1.5" />'
        f'<rect x="45" y="18" width="14" height="27" fill="{pulse_color}" '
        f'stroke="{pulse_stroke}" />'
        f'<rect x="145" y="18" width="14" height="27" fill="{pulse_color}" '
        f'stroke="{pulse_stroke}" />'
        f'<text x="52" y="58" font-size="10" text-anchor="middle" fill="{INK_SECONDARY}">δ</text>'
        f'<text x="152" y="58" font-size="10" text-anchor="middle" fill="{INK_SECONDARY}">δ</text>'
        f'<line x1="59" y1="10" x2="145" y2="10" stroke="{gap_color}" stroke-width="1.5" />'
        f'<text x="102" y="8" font-size="10" text-anchor="middle" fill="{gap_color}">Δ</text>'
    )


def _signal_flow_svg(b_values: tuple, n_inputs: int) -> str:
    """
    GAIA's own Figure 1, redrawn in miniature: the b-values below
    n_inputs are "Low info" feeding the 3D U-Net; the rest are "High
    info" predicted at the output, drawn with a dashed outline because
    they are not acquired directly.
    """

    inputs, outputs = b_values[:n_inputs], b_values[n_inputs:]
    parts = []
    in_y0, in_step = 4, 60 / len(inputs)
    for i, b in enumerate(inputs):
        y = in_y0 + i * in_step
        parts.append(
            f'<rect x="5" y="{y:.1f}" width="52" height="11" rx="2" '
            f'fill="{CATEGORICAL_1}" />'
            f'<text x="31" y="{y + 8:.1f}" font-size="7" text-anchor="middle" '
            f'fill="{SURFACE}">b={b}</text>'
            f'<line x1="57" y1="{y + 5.5:.1f}" x2="88" y2="35" '
            f'stroke="{GRIDLINE}" stroke-width="1" />'
        )
    parts.append(
        f'<rect x="88" y="20" width="60" height="30" rx="4" fill="{SURFACE}" '
        f'stroke="{INK_SECONDARY}" stroke-width="1.5" />'
        f'<text x="118" y="33" font-size="8" text-anchor="middle" '
        f'fill="{INK_SECONDARY}">3D</text>'
        f'<text x="118" y="43" font-size="8" text-anchor="middle" '
        f'fill="{INK_SECONDARY}">U-Net</text>'
    )
    out_y0, out_step = 12, 46 / len(outputs)
    for i, b in enumerate(outputs):
        y = out_y0 + i * out_step
        parts.append(
            f'<line x1="148" y1="35" x2="163" y2="{y + 5.5:.1f}" '
            f'stroke="{GRIDLINE}" stroke-width="1" />'
            f'<rect x="163" y="{y:.1f}" width="52" height="11" rx="2" fill="none" '
            f'stroke="{CATEGORICAL_2}" stroke-width="1.5" stroke-dasharray="3,2" />'
            f'<text x="189" y="{y + 8:.1f}" font-size="7" text-anchor="middle" '
            f'fill="{CATEGORICAL_2}">b={b}</text>'
        )
    return "".join(parts)


# ---------------------------------------------------------------------
# GAIA's own facts, exact unless noted. See modules/model_efficiency/
# README.md "The GAIA worked example" for the full citation-integrity
# discussion of what is exact versus approximate here.
# ---------------------------------------------------------------------

GAIA_CITATION = (
    "Jallais, M., Mancini, M., & Palombo, M. (2026). GAIA - Green "
    "Artificial Intelligence for Accelerated medical imaging: "
    "Sustainable and Efficient Diffusion MRI Analysis. Research output, "
    "Cardiff University Brain Research Imaging Centre (CUBRIC). "
    "https://doi.org/10.13140/RG.2.2.15336.12807 - an early-stage "
    "research output (e.g. a conference abstract), not confirmed as "
    "peer-reviewed."
)
WAND_CITATION = (
    "McNabb, C. B., Driver, I. D., Hyde, V. et al. (2025). WAND: A "
    "multi-modal dataset integrating advanced MRI, MEG, and TMS for "
    "multi-scale brain analysis. Sci Data, 12, 220. "
    "https://doi.org/10.1038/s41597-024-04154-7 (170 participants; GAIA "
    "used a 161-participant subset)."
)
HINTON_CITATION = (
    "Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the "
    "knowledge in a neural network. NeurIPS Deep Learning Workshop. "
    "https://doi.org/10.48550/arXiv.1503.02531"
)
ROMERO_CITATION = (
    "Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & "
    "Bengio, Y. (2015). FitNets: Hints for thin deep nets. "
    "International Conference on Learning Representations. "
    "https://doi.org/10.48550/arXiv.1412.6550"
)
KAACK_CITATION = (
    "Kaack, L. H., Donti, P. L., Strubell, E., Kamiya, G., Creutzig, "
    "F., & Rolnick, D. (2022). Aligning artificial intelligence with "
    "climate change mitigation. Nature Climate Change, 12, 518-527. "
    "https://doi.org/10.1038/s41558-022-01377-7"
)
DHAR_CITATION = (
    "Dhar, P. (2020). The carbon impact of artificial intelligence. "
    "Nature Machine Intelligence, 2, 423-425. "
    "https://doi.org/10.1038/s42256-020-0219-9"
)

GAIA_TEACHER_PARAMETERS = 4_118_219
GAIA_LIGHT_STUDENT_PARAMETERS = 52_848
GAIA_TEACHER_DEPTH = 5
GAIA_LIGHT_STUDENT_DEPTH = 2
GAIA_TRAINING_SUBJECTS = 161
GAIA_TEST_SUBJECTS = 17
GAIA_TRAINING_EPOCHS = 80
GAIA_DISTILLATION_LOSS_PCT = 25  # lambda_KD=0.25: this share of the student's loss mimics the teacher

GAIA_MODEL_SIZE_REDUCTION_PCT = 78          # paper's own stated figure, Student vs. Teacher
GAIA_ENERGY_CO2_REDUCTION_PCT = 35          # per subject, Student vs. Teacher
GAIA_CO2_PER_DEPLOYMENT_REDUCTION_PCT = 20  # Student vs. Teacher, scaling
GAIA_CO2_SAVED_PER_MILLION_USES_KG = 0.44   # Student vs. Teacher

# Acquisition scheme (PGSE, WAND protocol): b-values, direction counts, and
# which shells are model inputs versus the two predicted targets. Exact,
# from the paper's Methods section.
ACQUISITION_B_VALUES = (200, 500, 1200, 2400, 4000, 6000)
ACQUISITION_DIRECTIONS = (20, 20, 30, 61, 61, 61)
ACQUISITION_N_INPUTS = 4  # first four shells are model inputs; the rest are predicted

# Static pictographs, not the animated kind: this project moved away from
# flickering step animations toward still icons (see GRAND, HealthRing).
PIPELINE_STEPS = (
    (":material/scanner:", "Acquire", "PGSE scan, 6 b-value shells"),
    (":material/tune:", "Preprocess", "Per the WAND protocol"),
    (":material/hub:", "Predict", "Teacher / Light Model / Student"),
    (":material/fact_check:", "Compare", "Predicted vs. acquired ground truth"),
    (":material/eco:", "Report", "Energy and CO2 alongside accuracy"),
)

MODEL_ICON_NOTES = (
    (":material/school:", "Teacher", "Largest network; standard loss only"),
    (":material/lightbulb:", "Light Model", "Same small size as Student; standard loss only (control)"),
    (":material/auto_awesome:", "Student", "Same small size as Light Model; standard loss + distillation"),
)

# Same fixed order as MODEL_ICON_NOTES, mapped to the palette's first three
# categorical slots (references/palette.md) - the ones validated for the
# all-pairs comparisons a scatter plot needs.
MODEL_COLORS = {
    "Teacher": CATEGORICAL_1,
    "Light Model": CATEGORICAL_2,
    "Student": CATEGORICAL_3,
}

# One hover tile per symbol in the loss equations below, ordered to match
# where each symbol appears in the formula it explains (st.metric's own
# "help" tooltip icon - the interactive layer sits on the formula itself
# rather than in a separate selector).
LOSS_TERM_ITEMS = (
    (
        "λ1",
        "Whole-image L1",
        "Mean absolute difference between predicted and acquired "
        "signal, averaged over every voxel. Weight = 1 in both the "
        "standard loss and the distillation loss.",
    ),
    (
        "λ2",
        "SSIM",
        "Structural similarity index, comparing local structure rather "
        "than per-voxel values. Weight = 1 in the standard loss, but 0 "
        "in the distillation loss - GAIA does not use SSIM when "
        "matching latent representations.",
    ),
    (
        "λ3",
        "White matter L1",
        "L1 error restricted to white matter voxels. Weight = 1 in the "
        "standard loss, but 0.75 in the distillation loss - "
        "down-weighted relative to the standard loss.",
    ),
    (
        "λ4",
        "Gray matter L1",
        "L1 error restricted to gray matter voxels. Weight = 1 in the "
        "standard loss, but 1.25 in the distillation loss - "
        "up-weighted relative to the standard loss.",
    ),
)

LOSS_BLEND_ITEM = (
    "λ_KD",
    "Distillation blend",
    "How much of the Student's total loss comes from matching the "
    "Teacher's latent representation, versus the standard loss against "
    "ground truth. GAIA used λ_KD = 0.25: 25% distillation, 75% "
    "standard loss.",
)

STAGE_KEY = "gaia_stage"

STAGE_RESEARCH_QUESTION = 0
STAGE_UNDERSTAND_TASK = 1
STAGE_COMPARE_PERFORMANCE = 2
STAGE_COMPARE_EFFICIENCY = 3
STAGE_EVALUATE_TRADEOFF = 4
STAGE_GENERALIZABILITY = 5
STAGE_RESEARCH_DECISION = 6

JOURNEY_STAGES = (
    "Research question",
    "Understand the task",
    "Compare performance",
    "Compare efficiency",
    "Evaluate tradeoff",
    "Examine generalizability",
    "Research decision",
)

PREDICT_KEY = "gaia_predict_light_vs_student"
REVEAL_KEY = "gaia_reveal_light_vs_student"


def _current_stage() -> int:
    return st.session_state.get(STAGE_KEY, STAGE_RESEARCH_QUESTION)


def _advance_to(stage: int) -> None:
    st.session_state[STAGE_KEY] = max(_current_stage(), stage)
    st.rerun()


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _load_profiles() -> tuple[models_core.ModelProfile, ...]:
    frame = pd.read_csv(SAMPLE_DIR / "gaia_models.csv")
    return tuple(
        models_core.ModelProfile(
            name=row["name"],
            performance_metric_name=row["performance_metric_name"],
            performance_value=float(row["performance_value"]),
            performance_is_approximate=_to_bool(row["performance_is_approximate"]),
            n_parameters=int(row["n_parameters"]),
            resource_metric_name=row["resource_metric_name"],
            resource_value=float(row["resource_value"]),
            resource_is_approximate=_to_bool(row["resource_is_approximate"]),
            notes=str(row["notes"]),
        )
        for _, row in frame.iterrows()
    )


st.set_page_config(
    page_title="OpenMeasure · AI & Environmental Equity: GAIA",
    page_icon=":material/eco:",
    layout="centered",
)

st.title(
    "Green Artificial Intelligence for Accelerated medical imaging: "
    "Sustainable and Efficient Diffusion MRI Analysis"
)
st.caption("This journey uses the GAIA study specifically (Jallais, Mancini, & Palombo, 2026).")
st.caption(
    "A guided case study: each stage unlocks after you make a decision "
    "or inspect its consequence."
)

render_data_handling_summary(disclosure_for("pages/GAIA_Worked_Example.py"))

stage = _current_stage()

_stage_parts = [
    f"**{label}**" if index == stage else label
    for index, label in enumerate(JOURNEY_STAGES)
]

with st.container(border=True):
    st.markdown(" → ".join(_stage_parts))

if stage > STAGE_RESEARCH_QUESTION:
    if st.button("Restart study", icon=":material/restart_alt:"):
        for key in (STAGE_KEY, PREDICT_KEY, REVEAL_KEY):
            st.session_state.pop(key, None)
        st.rerun()

st.divider()

# -----------------------------------------------------------------
# 1. Research question
# -----------------------------------------------------------------

section_header("1. Research Question")

st.markdown("### When Is a More Efficient Model Appropriate To Replace a Larger One?")

st.write(
    "AI-based applications in MRI have shown substantial potential for "
    "improving image quality, reconstruction speed, and diagnostic "
    "accuracy. But developing and running these models is "
    "energy-intensive: it produces greenhouse gas emissions from the "
    "data storage and computation that both training and inference "
    "require, and larger, more demanding models can be harder to "
    "deploy where hardware or energy are constrained. This journey "
    "asks whether a smaller, more efficient model can be validated as "
    "appropriate once performance and resource use are considered "
    "together, rather than performance alone."
)

with st.expander("Context: energy use in AI-based MRI"):
    st.write(
        "AI development and deployment is energy-intensive. Reducing "
        "that footprint is treated here as part of model evaluation "
        "itself, alongside accuracy, rather than as a separate concern."
    )
    st.caption(KAACK_CITATION)
    st.caption(DHAR_CITATION)

if stage < STAGE_UNDERSTAND_TASK:
    if st.button("Begin study", type="primary"):
        _advance_to(STAGE_UNDERSTAND_TASK)

# -----------------------------------------------------------------
# 2. Understand the task
# -----------------------------------------------------------------

if stage >= STAGE_UNDERSTAND_TASK:
    section_header(
        "2. Understand the Task",
        "Three models, one task: predicting hard-to-acquire diffusion MRI signals.",
    )

    pipeline_cols = st.columns(5)
    for col, (icon, label, note) in zip(pipeline_cols, PIPELINE_STEPS):
        with col:
            st.badge(label, icon=icon, color="blue")
            st.caption(note)

    st.write(
        "GAIA predicts high b-value powder-averaged diffusion MRI "
        "signals (b=4000 and 6000 s/mm²), which are informative but "
        "slow and difficult to acquire directly, from four lower "
        "b-value inputs that are faster to acquire, using a 3D U-Net."
    )

    with st.expander("Diffusion MRI terms used on this page"):
        term_col, fig_col = st.columns([3, 2])
        with term_col:
            st.write(
                "**b-value**: a single number (s/mm²) summarizing how "
                "strongly an acquisition is sensitized to water "
                "movement. Higher b-values carry more microstructural "
                "information but take longer to acquire and have a "
                "lower signal-to-noise ratio."
            )
        with fig_col:
            _svg_figure(
                _b_value_svg(),
                "Low b-value (weak weighting) → high b-value (strong "
                "weighting, more microstructural information)",
            )

        term_col, fig_col = st.columns([3, 2])
        with term_col:
            st.write(
                "**Powder-averaged signal**: diffusion MRI measures a "
                "separate signal per gradient direction; "
                "powder-averaging combines those directional "
                "measurements into one direction-independent value per "
                "voxel, which is what lets GAIA treat signal prediction "
                "as a per-voxel problem rather than one that also has "
                "to model fiber orientation."
            )
        with fig_col:
            _svg_figure(
                _powder_average_svg(),
                "Signal from multiple gradient directions → one "
                "powder-averaged value per voxel",
            )

        term_col, fig_col = st.columns([3, 2])
        with term_col:
            st.write(
                "**PGSE (Pulsed Gradient Spin Echo)**: the diffusion-"
                "encoding sequence GAIA's data was acquired with, using "
                "two gradient pulses of duration δ separated by a "
                "diffusion time Δ."
            )
        with fig_col:
            _svg_figure(
                _pulse_sequence_svg("pulses"),
                "The two gradient pulses, each of duration δ",
            )

        term_col, fig_col = st.columns([3, 2])
        with term_col:
            st.write(
                "**Diffusion time (Δ)**: the interval between those two "
                "pulses, during which water diffusing through tissue "
                "(e.g. across cell membranes) attenuates the signal - "
                "GAIA used δ/Δ = 7/24 ms, with TE/TR = 59/3000 ms."
            )
        with fig_col:
            _svg_figure(
                _pulse_sequence_svg("gap"),
                "The diffusion time Δ, the gap between the two pulses",
            )

    st.write(
        "GAIA's Figure 1 labels the four lower b-value shells \"Low "
        "info\" (acquired directly) and the two higher b-values \"High "
        "info\" (predicted by the model, not scanned)."
    )
    _svg_figure(
        _signal_flow_svg(ACQUISITION_B_VALUES, ACQUISITION_N_INPUTS),
        "\"Low info\" (4 lower b-values, acquired) → 3D U-Net → "
        "\"High info\" (2 higher b-values, predicted)",
        viewbox="0 0 220 70",
    )
    st.caption(
        "Diffusion-encoding directions acquired per shell, in the order "
        f"shown ({', '.join(str(b) for b in ACQUISITION_B_VALUES)} "
        f"s/mm²): {', '.join(str(d) for d in ACQUISITION_DIRECTIONS)}."
    )
    st.caption(GAIA_CITATION)

    model_cols = st.columns(3)
    for col, (icon, label, note) in zip(model_cols, MODEL_ICON_NOTES):
        with col:
            st.badge(label, icon=icon, color="blue")
            st.caption(note)

    architecture_df = pd.DataFrame(
        [
            {
                "Model": "Teacher",
                "Depth": GAIA_TEACHER_DEPTH,
                "Parameters": f"{GAIA_TEACHER_PARAMETERS:,}",
                "Training": "Standard loss",
            },
            {
                "Model": "Light Model",
                "Depth": GAIA_LIGHT_STUDENT_DEPTH,
                "Parameters": f"{GAIA_LIGHT_STUDENT_PARAMETERS:,}",
                "Training": "Standard loss (no teacher guidance)",
            },
            {
                "Model": "Student",
                "Depth": GAIA_LIGHT_STUDENT_DEPTH,
                "Parameters": f"{GAIA_LIGHT_STUDENT_PARAMETERS:,}",
                "Training": (
                    f"{100 - GAIA_DISTILLATION_LOSS_PCT}% standard loss + "
                    f"{GAIA_DISTILLATION_LOSS_PCT}% knowledge distillation"
                ),
            },
        ]
    )
    st.dataframe(architecture_df, width="stretch", hide_index=True)

    st.write(
        "**Knowledge distillation** trains a small 'student' network to "
        "mimic a large 'teacher' network's internal representation, not "
        "just its final answer. The student is trained to reproduce "
        "information from the teacher's internal representation, "
        "providing an additional learning signal beyond the original "
        "target."
    )
    st.caption(HINTON_CITATION)
    st.caption(ROMERO_CITATION)

    st.write(
        "**The Light Model's role**: the exact same size and "
        "architecture as the Student, trained the exact same way except "
        "with no teacher guidance at all. Because Student and Light "
        "Model use the same small architecture, their comparison "
        "isolates the contribution of the distillation training "
        "procedure."
    )

    with st.expander("How the models were trained: the loss functions"):
        st.write(
            "The Teacher and Light Model were both trained with the "
            "same loss, comparing each model's prediction ŷ to the "
            "acquired ground truth y:"
        )
        st.latex(
            r"L(y,\hat y)=\lambda_1\frac{1}{N}\sum_{n=1}^{N}|y_n-\hat y_n|"
            r"+\lambda_2\left(1-\mathrm{SSIM}(y,\hat y)\right)"
            r"+\lambda_3\frac{1}{N_{WM}}\sum_{WM}|y_n-\hat y_n|"
            r"+\lambda_4\frac{1}{N_{GM}}\sum_{GM}|y_n-\hat y_n|"
        )
        st.caption("Hover a term below for what it weights and its value:")
        for symbol, name, note in LOSS_TERM_ITEMS:
            st.badge(f"{symbol}: {name}", help=note, color="blue")

        st.write(
            "The Student adds a second loss, computed the same way but "
            "between the Teacher's and Student's last internal (latent) "
            "representations, z and ẑ, rather than the final "
            "prediction, then blends the two:"
        )
        st.latex(
            r"L_{total}=(1-\lambda_{KD})\,L(y,\hat y)"
            r"+\lambda_{KD}\,L_{KD}(z,\hat z)"
        )
        blend_symbol, blend_name, blend_note = LOSS_BLEND_ITEM
        st.badge(f"{blend_symbol}: {blend_name}", help=blend_note, color="blue")
        st.caption(GAIA_CITATION)

    st.radio(
        "Before revealing the result: do you expect a small network "
        "trained normally (Light Model) to perform about as well as one "
        "the same size trained with distillation (Student)?",
        options=["Yes, about the same", "No, the Student should do better", "Not sure"],
        index=2,
        key=PREDICT_KEY,
    )

    if not st.session_state.get(REVEAL_KEY, False):
        if st.button("Reveal result", key="gaia_reveal_button_light_vs_student"):
            st.session_state[REVEAL_KEY] = True
            st.rerun()

    if st.session_state.get(REVEAL_KEY, False):
        prediction = st.session_state.get(PREDICT_KEY, "Not sure")
        st.write(
            "The paper reports the Student \"outperformed the light "
            "model, showing lower relative error, mean squared error "
            "(MSE) and mean absolute error (MAE)\" for both high "
            "b-values, despite being exactly the same size as the Light "
            "Model."
        )
        st.caption(f"You predicted: {prediction}.")

    with st.expander("Data sources and citations"):
        st.write(
            f"**Dataset**: GAIA used {GAIA_TRAINING_SUBJECTS} "
            "participants from WAND (which contains 170 healthy "
            "volunteers in total), an 80/10/10 train/test/validation "
            f"split (test set = {GAIA_TEST_SUBJECTS} subjects), "
            f"{GAIA_TRAINING_EPOCHS} epochs, Adam optimizer (learning "
            "rate 1e-4). Training took approximately 30 minutes per "
            "model."
        )
        st.caption(
            "**Adam optimizer**: adapts each weight's update size using "
            "running estimates of its gradient's momentum and variance, "
            "with a bias correction for early training steps - in "
            "practice, faster convergence with less manual tuning than "
            "plain gradient descent."
        )
        st.write(
            "**What research questions WAND supports**: WAND combines "
            "structural, functional, and diffusion MRI with MEG and TMS "
            "in the same 170 participants, so it supports questions "
            "that need more than one imaging modality measured in the "
            "same people. GAIA uses only its diffusion MRI arm."
        )
        st.write(
            "**Using this module with your own models**: the Model "
            "Efficiency core this journey is built on "
            "(`modules/model_efficiency/core`) needs one performance "
            "value and one resource value (energy, CO2, parameter "
            "count, or any other lower-is-better cost) per model being "
            "compared - see `modules/model_efficiency/README.md` for "
            "the exact `ModelProfile` format."
        )
        st.caption(WAND_CITATION)
        st.caption(GAIA_CITATION)

    if stage < STAGE_COMPARE_PERFORMANCE:
        if st.button("Continue to compare performance", type="primary"):
            _advance_to(STAGE_COMPARE_PERFORMANCE)

# -----------------------------------------------------------------
# 3. Compare performance (primary evidence - exact quotes lead)
# -----------------------------------------------------------------

if stage >= STAGE_COMPARE_PERFORMANCE:
    section_header(
        "3. Compare Performance",
        "The paper's own reported findings, quoted directly.",
    )

    st.info(
        "\"The distilled student network achieved a 78% reduction in "
        "model size compared to the teacher, while maintaining "
        "comparable prediction accuracy and precision.\""
    )
    st.info(
        "\"the student outperformed the light model, showing lower "
        "relative error, mean squared error (MSE) and mean absolute "
        "error (MAE) for the prediction of both high b-values.\""
    )

    st.caption(
        "Approximate values read from Figure 5, shown only as "
        "supporting detail (the full frontier chart is in Step 5):"
    )
    profiles = _load_profiles()
    perf_cols = st.columns(len(profiles))
    for col, profile in zip(perf_cols, profiles):
        with col:
            st.metric(f"{profile.name}", f"{profile.performance_value:.1e}")
    st.caption(f"Metric: {profiles[0].performance_metric_name.lower()} (lower is better).")

    caveat(
        "These MSE values are approximate, read from the published "
        "figure rather than stated exactly in the text, and are pending "
        "confirmation from the authors."
    )

    st.write(
        "**Interpretation**: because Student and Light Model share "
        "the same architecture, their comparison is evidence that "
        "knowledge distillation itself contributed to Student's "
        "advantage on this task. It does not establish that distillation "
        "improves every model or every task - only that it did here, "
        "under these training conditions."
    )

    if stage < STAGE_COMPARE_EFFICIENCY:
        if st.button("Continue to compare efficiency", type="primary"):
            _advance_to(STAGE_COMPARE_EFFICIENCY)

# -----------------------------------------------------------------
# 4. Compare efficiency (primary evidence - exact quotes lead)
# -----------------------------------------------------------------

if stage >= STAGE_COMPARE_EFFICIENCY:
    section_header(
        "4. Compare Efficiency",
        "The paper's own reported findings, quoted directly.",
    )

    e1, e2, e3 = st.columns(3)
    e1.metric("Model-size reduction (Student vs. Teacher)", f"{GAIA_MODEL_SIZE_REDUCTION_PCT}%")
    e2.metric("Energy/CO2 decrease per subject (Student vs. Teacher)", f"{GAIA_ENERGY_CO2_REDUCTION_PCT}%")
    e3.metric("Less CO2 per deployment (Student vs. Teacher)", f"~{GAIA_CO2_PER_DEPLOYMENT_REDUCTION_PCT}%")

    parameter_reduction_pct = (1 - GAIA_LIGHT_STUDENT_PARAMETERS / GAIA_TEACHER_PARAMETERS) * 100
    st.warning(
        f"The paper reports a {GAIA_MODEL_SIZE_REDUCTION_PCT}% "
        "model-size reduction, but the meaning of \"model size\" is not "
        "specified by the provided parameter counts: Teacher has "
        f"{GAIA_TEACHER_PARAMETERS:,} parameters, Student has "
        f"{GAIA_LIGHT_STUDENT_PARAMETERS:,} - by parameter count alone "
        f"that would be roughly a {parameter_reduction_pct:.1f}% "
        "reduction, not 78%. This is not reconciled here; it is stated "
        "as reported, pending clarification from the authors."
    )

    st.write(
        "The paper also reports the Student model saves "
        f"**{GAIA_CO2_SAVED_PER_MILLION_USES_KG} kg of CO2** when "
        "applied 1,000,000 times, compared to the Teacher - explored "
        "further in the next step."
    )

    st.write(
        "**Implications**: a reported reduction in energy and "
        "CO2 per subject is a reason to treat resource use as part of "
        "what a model evaluation checks, along with accuracy - "
        "particularly for a model that will be run many times over its "
        "deployed life."
    )

    if stage < STAGE_EVALUATE_TRADEOFF:
        if st.button("Continue to evaluate tradeoff", type="primary"):
            _advance_to(STAGE_EVALUATE_TRADEOFF)

# -----------------------------------------------------------------
# 5. Evaluate tradeoff (illustrative synthesis, secondary to 3-4)
# -----------------------------------------------------------------

if stage >= STAGE_EVALUATE_TRADEOFF:
    section_header(
        "5. Evaluate Tradeoff",
        "An illustrative synthesis built on approximate values, secondary "
        "to the exact findings in Steps 3-4.",
    )

    profiles = _load_profiles()

    left, mid, right = st.columns([1, 3, 1])
    with left:
        st.badge("Efficiency", icon=":material/eco:", color="blue")
    with right:
        st.badge("Performance", icon=":material/insights:", color="blue")
    with mid:
        performance_weight = st.slider(
            "How much weight should performance carry, versus resource efficiency?",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            label_visibility="collapsed",
        )

    preference_result = preference_core.rank_by_preference(profiles, performance_weight)

    st.metric("Model favored by these weights", preference_result.favored_by_weights)
    caveat(
        "This is one way to combine two quantities that can't be "
        "directly compared, and is sensitive both to which models are "
        "being compared and to the weight chosen above. It reflects the "
        "weight you set, not a judgment this page is making - lower "
        "resource use does not, by itself, make a model preferable."
    )

    st.write(
        "GAIA's own Figure 5 plots MSE against CO2 emissions for the "
        "three models, described in the paper as showing \"the "
        "trade-off between predictive accuracy and precision, and "
        "environmental efficiency.\" The chart below reproduces that "
        "comparison from the same figure's approximate values, faded "
        "out for whichever model the weight above does not favor."
    )

    frontier_result = frontier_core.compute_frontier(profiles)

    chart_rows = [
        {
            "name": profile.name,
            "performance": profile.performance_value,
            "resource": profile.resource_value,
            "favored": (
                "Favored by weights above"
                if profile.name == preference_result.favored_by_weights
                else "Not favored"
            ),
        }
        for profile in profiles
    ]

    frontier_spec = {
        "data": {"values": chart_rows},
        "mark": {"type": "point", "filled": True, "size": 220},
        "encoding": {
            "x": {
                "field": "resource",
                "type": "quantitative",
                "title": f"{profiles[0].resource_metric_name} (lower is better)",
            },
            "y": {
                "field": "performance",
                "type": "quantitative",
                "title": f"{profiles[0].performance_metric_name} (lower is better)",
            },
            "color": {
                "field": "name",
                "type": "nominal",
                "scale": {
                    "domain": list(MODEL_COLORS),
                    "range": list(MODEL_COLORS.values()),
                },
                "legend": {"title": "Model", "orient": "bottom"},
            },
            "opacity": {
                "field": "favored",
                "type": "nominal",
                "sort": ["Not favored", "Favored by weights above"],
                "scale": {
                    "domain": ["Not favored", "Favored by weights above"],
                    "range": [0.35, 1.0],
                },
                "legend": None,
            },
            "tooltip": [
                {"field": "name", "type": "nominal", "title": "Model"},
                {
                    "field": "performance",
                    "type": "quantitative",
                    "title": profiles[0].performance_metric_name,
                    "format": ".2e",
                },
                {
                    "field": "resource",
                    "type": "quantitative",
                    "title": profiles[0].resource_metric_name,
                    "format": ".2f",
                },
                {"field": "favored", "type": "nominal", "title": "Under weights above"},
            ],
        },
        "width": "container",
        "height": 260,
        "config": _VEGA_CHART_CONFIG,
    }
    st.vega_lite_chart(frontier_spec, width="stretch")
    st.caption(
        "Approximate values read from Figure 5. Hover a point for its "
        "exact reading; the faded point is not favored by the weight "
        "set above."
    )
    st.caption(GAIA_CITATION)

    for profile in profiles:
        if not frontier_result.is_efficient[profile.name]:
            flagged_item_note(
                profile.name,
                "Under the two approximate metrics shown here, this "
                "model is dominated by another and would not be favored "
                "by any weighting of these two dimensions alone.",
            )

    caveat(
        "A large reported difference in resource use is not, by itself, "
        "evidence that a model is better - performance and resource use "
        "answer different questions."
    )

    st.divider()
    st.caption("Deployment scale")

    n_deployments = st.number_input(
        "Number of deployments", min_value=1, value=1_000_000, step=100_000
    )
    per_unit_difference = GAIA_CO2_SAVED_PER_MILLION_USES_KG / 1_000_000
    projection = deployment_core.project_deployment_savings(
        per_unit_difference, int(n_deployments), "CO2 (kg)"
    )
    st.metric(
        f"Projected CO2 saved (Student vs. Teacher) at {int(n_deployments):,} deployments",
        f"{projection.projected_difference:.3f} kg",
    )
    st.caption(
        "This uses the paper's own reported rate (0.44 kg saved per "
        "1,000,000 deployments) and its own stated assumption that this "
        "scales linearly - not a new estimate."
    )

    st.write(
        "**What to inspect**: try raising the number above. A "
        "per-subject difference that looks small in Step 4 compounds "
        "once it is multiplied across a deployment's full scale, so the "
        "total depends on deployment volume as well as the per-subject "
        "figures."
    )

    if stage < STAGE_GENERALIZABILITY:
        if st.button("Continue to examine generalizability", type="primary"):
            _advance_to(STAGE_GENERALIZABILITY)

# -----------------------------------------------------------------
# 6. Examine generalizability
# -----------------------------------------------------------------

if stage >= STAGE_GENERALIZABILITY:
    section_header(
        "6. Examine Generalizability",
        "What was measured, versus what it implies.",
    )

    st.write(
        "**Measured**: on one dataset, one task, and one held-out test "
        "set, the authors report comparable prediction accuracy between "
        "Student and Teacher, with Student using substantially less "
        "energy and CO2."
    )
    st.write("**Not measured, and not something this result can establish on its own:**")

    flagged_item_note(
        "Single dataset",
        f"GAIA used {GAIA_TRAINING_SUBJECTS} participants from WAND, "
        "one acquisition protocol, one site. Whether these results hold "
        "on a different scanner, protocol, or population is untested "
        "here.",
    )
    flagged_item_note(
        "Single task",
        "Predicting two specific high b-values from four specific lower "
        "b-values. A different diffusion MRI task, or a different "
        "imaging modality, might show a different performance/efficiency "
        "balance.",
    )
    flagged_item_note(
        "Energy estimates",
        "Energy/CO2 figures are estimated from UK National Grid "
        "carbon-intensity data (per the paper), which will not match "
        "every deployment location's actual grid mix.",
    )
    flagged_item_note(
        "Approximate figures",
        "The performance and resource values used in Step 5's frontier "
        "chart are read from a figure, not the paper's own exact "
        "reported numbers.",
    )

    st.write(
        "**Implications (potential, not tested here)**: a smaller "
        "model needs less compute to run, which could lower a barrier to "
        "using it in settings with limited hardware - low-resource "
        "hospitals, mobile devices, or point-of-care tools. GAIA did not "
        "test deployment in any of these settings; this is a plausible "
        "implication of the size and efficiency result, not a finding."
    )

    if stage < STAGE_RESEARCH_DECISION:
        if st.button("Continue to the research decision", type="primary"):
            _advance_to(STAGE_RESEARCH_DECISION)

# -----------------------------------------------------------------
# 7. Research decision
# -----------------------------------------------------------------

if stage >= STAGE_RESEARCH_DECISION:
    section_header("7. Research Decision")

    decision = st.radio(
        "Given everything above, which would you deploy?",
        options=["Teacher", "Light Model", "Student", "Insufficient evidence to decide"],
        index=3,
    )

    profiles = _load_profiles()
    frontier_result = frontier_core.compute_frontier(profiles)

    if decision == "Insufficient evidence to decide":
        st.write(
            "A reasonable position: this evidence comes from a single "
            "dataset and a single task, so generalizing to a specific "
            "deployment context (a different scanner, population, or "
            "imaging task) is not directly supported by what's measured "
            "here."
        )
    elif decision == "Light Model":
        st.warning(
            "Under the two approximate metrics used in Step 5, Light "
            "Model is dominated by Student and would not be favored by "
            "any weighting of these two dimensions alone - though "
            "deployment could involve other factors not captured by "
            "those two metrics. Light Model's role in this study is as "
            "the control that shows what distillation adds beyond "
            "simply using a smaller architecture."
        )
    elif frontier_result.is_efficient.get(decision, False):
        st.write(
            f"{decision} is on the performance-efficiency frontier under "
            "the approximate values used in Step 5, meaning no other "
            "model measured here beats it on both dimensions at once. "
            "Whether it is the right choice still depends on how much "
            "resource use should weigh against performance in your "
            "specific context, and on evidence this study doesn't "
            "provide about how well it generalizes beyond WAND."
        )
    else:
        st.write(
            f"Under the two approximate metrics used in Step 5, "
            f"{decision} is dominated by another model here and would "
            "not be favored by any weighting of those two dimensions "
            "alone."
        )

    st.write(
        "**Implications**: when computational requirements "
        "materially affect whether a model can be deployed at all, "
        "evaluating performance alone is not enough - resource use "
        "becomes part of what 'appropriate' means."
    )

    st.divider()
    with st.container(border=True):
        st.write(
            "**Implications**: GAIA illustrates how model evaluation can "
            "consider predictive performance alongside computational "
            "efficiency. In settings where resource requirements affect "
            "sustainability, scale, or deployment feasibility, the most "
            "accurate model may not automatically be the most appropriate "
            "one."
        )
