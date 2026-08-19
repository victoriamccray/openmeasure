"""
AI & Environmental Equity: GAIA - a guided research journey, not a workflow.

Teaches a validation question none of the numbered workflows cover: when
is a more efficient model good enough to replace a larger one, once
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
from shared.report import caveat, flagged_item_note, section_header

SAMPLE_DIR = ROOT / "modules" / "model_efficiency" / "sample_data"

# Validated default palette (dataviz skill, references/palette.md), same
# values used elsewhere in this app (e.g. Portfolio_Impact_Analysis.py).
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
CATEGORICAL_1 = "#2a78d6"
CATEGORICAL_2 = "#eb6834"

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
# GAIA's own facts, exact unless noted. See modules/model_efficiency/
# README.md "The GAIA worked example" for the full citation-integrity
# discussion of what is exact versus approximate here.
# ---------------------------------------------------------------------

GAIA_CITATION = (
    "Jallais, M., Mancini, M., & Palombo, M. (2026). GAIA - Green "
    "Artificial Intelligence for Accelerated medical imaging: "
    "Sustainable and Efficient Diffusion MRI Analysis. Conference "
    "abstract, Cardiff University Brain Research Imaging Centre "
    "(CUBRIC). Venue and DOI pending confirmation from the authors."
)
WAND_CITATION = (
    "McNabb, C. B., Driver, I. D., Hyde, V. et al. (2025). WAND: A "
    "multi-modal dataset integrating advanced MRI, MEG, and TMS for "
    "multi-scale brain analysis. Sci Data, 12, 220. "
    "https://doi.org/10.1038/s41597-024-04154-7"
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

st.title("AI & Environmental Equity: GAIA")
st.caption("This journey uses the GAIA study specifically (Jallais, Mancini, & Palombo, 2026).")
st.caption(
    "A guided research simulation: each stage unlocks after you make a "
    "decision or inspect its consequence."
)

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

section_header("1. Research question")

st.markdown("### When is a more efficient model good enough to replace a larger one?")

st.write(
    "AI models used in medical imaging are usually judged on predictive "
    "performance alone. But training and running these models consumes "
    "energy and produces greenhouse gas emissions, and the largest, most "
    "accurate models are often the hardest to deploy on standard hospital "
    "hardware or in low-resource settings. This journey asks whether a "
    "smaller, more efficient model can be validated as good enough once "
    "performance and resource use are considered together, rather than "
    "performance alone."
)

with st.expander("Why resource use matters for AI validation"):
    st.write(
        "AI development is energy-intensive, and reducing that "
        "environmental impact is part of making advanced imaging tools "
        "sustainable and accessible, not a side concern separate from "
        "model quality."
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
        "2. Understand the task",
        "Three models, one task: predicting hard-to-acquire diffusion MRI signals.",
    )

    st.write(
        "GAIA predicts high b-value diffusion MRI signals (b=4000 and "
        "6000 s/mm²), which are informative but slow and difficult to "
        "acquire directly, from four lower b-value inputs that are "
        "faster to acquire, using a 3D U-Net."
    )

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
        "just its final answer - the idea being that the teacher's "
        "internal representation carries information a student trained "
        "from scratch would otherwise need many more parameters to "
        "discover on its own."
    )
    st.caption(HINTON_CITATION)
    st.caption(ROMERO_CITATION)

    st.write(
        "**Why the Light Model matters**: it is the exact same size and "
        "architecture as the Student, trained the exact same way except "
        "with no teacher guidance at all. If the Student outperforms the "
        "Light Model, that gap is attributable to distillation itself, "
        "not merely to having fewer parameters than the Teacher."
    )

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
            f"**Dataset**: WAND ({GAIA_TRAINING_SUBJECTS} subjects), "
            f"80/10/10 train/test/validation split (test set = "
            f"{GAIA_TEST_SUBJECTS} subjects), {GAIA_TRAINING_EPOCHS} "
            "epochs, Adam optimizer (learning rate 1e-4)."
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
        "3. Compare performance",
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

    if stage < STAGE_COMPARE_EFFICIENCY:
        if st.button("Continue to compare efficiency", type="primary"):
            _advance_to(STAGE_COMPARE_EFFICIENCY)

# -----------------------------------------------------------------
# 4. Compare efficiency (primary evidence - exact quotes lead)
# -----------------------------------------------------------------

if stage >= STAGE_COMPARE_EFFICIENCY:
    section_header(
        "4. Compare efficiency",
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

    if stage < STAGE_EVALUATE_TRADEOFF:
        if st.button("Continue to evaluate tradeoff", type="primary"):
            _advance_to(STAGE_EVALUATE_TRADEOFF)

# -----------------------------------------------------------------
# 5. Evaluate tradeoff (illustrative synthesis, secondary to 3-4)
# -----------------------------------------------------------------

if stage >= STAGE_EVALUATE_TRADEOFF:
    section_header(
        "5. Evaluate tradeoff",
        "An illustrative synthesis built on approximate values, secondary "
        "to the exact findings in Steps 3-4.",
    )

    profiles = _load_profiles()

    left, mid, right = st.columns([1, 3, 1])
    with left:
        st.caption("Efficiency")
    with right:
        st.caption("Performance")
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

    st.caption("Performance-efficiency frontier")

    frontier_result = frontier_core.compute_frontier(profiles)

    chart_rows = [
        {
            "name": profile.name,
            "performance": profile.performance_value,
            "resource": profile.resource_value,
            "is_favored": profile.name == preference_result.favored_by_weights,
            "label": (
                f"◆ {profile.name} (favored)"
                if profile.name == preference_result.favored_by_weights
                else profile.name
            ),
        }
        for profile in profiles
    ]

    frontier_spec = {
        "title": {
            "text": "Approximate values read from Figure 5",
            "color": INK_SECONDARY,
            "fontSize": 11,
            "fontWeight": "normal",
        },
        "data": {"values": chart_rows},
        "layer": [
            {
                "mark": {"type": "point", "filled": True, "size": 180},
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
                    "shape": {
                        "field": "is_favored",
                        "type": "nominal",
                        "scale": {"domain": [False, True], "range": ["circle", "diamond"]},
                        "legend": None,
                    },
                    "color": {
                        "field": "is_favored",
                        "type": "nominal",
                        "scale": {"domain": [False, True], "range": [CATEGORICAL_1, CATEGORICAL_2]},
                        "legend": None,
                    },
                },
            },
            {
                "mark": {"type": "text", "dy": -14, "fontSize": 11, "color": INK_SECONDARY},
                "encoding": {
                    "x": {"field": "resource", "type": "quantitative"},
                    "y": {"field": "performance", "type": "quantitative"},
                    "text": {"field": "label", "type": "nominal"},
                },
            },
        ],
        "width": "container",
        "height": 260,
        "config": _VEGA_CHART_CONFIG,
    }
    st.vega_lite_chart(frontier_spec, width="stretch")
    st.caption("Diamond marks the model favored by the weights set above.")

    for profile in profiles:
        if not frontier_result.is_efficient[profile.name]:
            flagged_item_note(
                profile.name,
                "Beaten on both performance and resource use by another "
                "model here - not a defensible choice at any weighting "
                "of the two.",
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

    if stage < STAGE_GENERALIZABILITY:
        if st.button("Continue to examine generalizability", type="primary"):
            _advance_to(STAGE_GENERALIZABILITY)

# -----------------------------------------------------------------
# 6. Examine generalizability
# -----------------------------------------------------------------

if stage >= STAGE_GENERALIZABILITY:
    section_header(
        "6. Examine generalizability",
        "What was measured, versus what it implies.",
    )

    st.write(
        "**Measured**: on one dataset, one task, and one held-out test "
        "set, a distilled model matched a larger model's accuracy and "
        "used substantially less energy and CO2 than it."
    )
    st.write("**Not measured, and not something this result can establish on its own:**")

    flagged_item_note(
        "Single dataset",
        f"WAND ({GAIA_TRAINING_SUBJECTS} subjects, one acquisition "
        "protocol, one site). Whether these results hold on a different "
        "scanner, protocol, or population is untested here.",
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

    if stage < STAGE_RESEARCH_DECISION:
        if st.button("Continue to the research decision", type="primary"):
            _advance_to(STAGE_RESEARCH_DECISION)

# -----------------------------------------------------------------
# 7. Research decision
# -----------------------------------------------------------------

if stage >= STAGE_RESEARCH_DECISION:
    section_header("7. Research decision")

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
            "Under the approximate values used in Step 5, Light Model is "
            "beaten on both performance and resource use by Student - it "
            "is not on the performance-efficiency frontier, so this "
            "choice is hard to defend against Student specifically. "
            "Light Model's role in this study is as the control that "
            "shows what distillation adds, not as a deployment candidate."
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
            f"{decision} is beaten on both performance and resource use "
            "by another model here, under the approximate values used in "
            "Step 5."
        )

    st.divider()
    st.success(
        "Model validation can include resource efficiency alongside "
        "predictive performance when computational requirements affect "
        "sustainability, scalability, or feasibility of deployment."
    )
