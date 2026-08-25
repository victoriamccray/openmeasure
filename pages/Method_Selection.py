"""
OpenMeasure - Method Selection Decision Tree, and Research Design.

Two modes on one page, chosen at the top:

- "Choose an analysis" (the page's original content, unchanged): one
  guided question routing to the workflow or research journey that
  fits, with Try, Why, You'll learn, and Limitations for the chosen
  branch. This mode assumes a question or dataset already exists.
- "Design a study": for a question upstream of that, before any data
  exists. Walks Research question -> Study design -> Measurement plan
  -> Design assumptions -> Simulation -> Interactive exploration ->
  Implications -> Design Record, using
  modules/research_design/core/ to simulate a naturalistic pain study
  under whatever assumptions a reader sets, then hands the resulting
  measurement plan's shape to the same suggest_workflows() the upload
  branch below already uses, landing in "Choose an analysis" territory
  from the design side rather than the data side.

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


MODE_DESIGN = "design"
MODE_ANALYSIS = "analysis"

mode = st.radio(
    "What do you want to do?",
    # "Choose an analysis" first/default: it was this page's sole
    # purpose before Design mode existed, so a bare page load keeps
    # behaving the way every existing link to this page already expects.
    options=(MODE_ANALYSIS, MODE_DESIGN),
    format_func=lambda key: {
        MODE_DESIGN: "Design a study (before data exists)",
        MODE_ANALYSIS: "Choose an analysis (I have a question or dataset)",
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
    INK_MUTED = "#898781"
    GRIDLINE = "#e1e0d9"
    SURFACE = "#fcfcfb"
    ACCENT = "#2a78d6"

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
# Mode: Design a study
# =======================================================================

else:
    st.caption(
        "Build a study and explore how design choices shape the "
        "evidence, before any data exists. v0.1 walks one built-in "
        "case: a naturalistic pain study. It does not build a design "
        "for an arbitrary question yet, and it never scores a design "
        "as good or bad - only what it does and does not support."
    )

    STAGE_QUESTION = 0
    STAGE_STRUCTURE = 1
    STAGE_MEASUREMENT = 2
    STAGE_ASSUMPTIONS = 3
    STAGE_SIMULATION = 4
    STAGE_EXPLORATION = 5
    STAGE_IMPLICATIONS = 6
    STAGE_RECORD = 7

    DESIGN_STAGE_LABELS = (
        "Research question",
        "Study design",
        "Measurement plan",
        "Design assumptions",
        "Simulation",
        "Interactive exploration",
        "Implications",
        "Design record",
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

    section_header("Research Question")

    st.markdown(
        "### Does the coupling between pain and physiology change with "
        "how pain is spatially distributed?"
    )

    st.write(
        "**Hypothesis**: the coupling between subjective pain and "
        "physiological signals (electrodermal activity and heart "
        "rate/heart-rate variability) changes when chronic pain is "
        "localized versus spatially distributed, referred, or radiating."
    )

    question_cols = st.columns(2)
    with question_cols[0]:
        st.markdown("**Population**")
        st.caption("Adults with chronic pain, naturalistically observed in daily life.")
        st.markdown("**Exposure**")
        st.caption("Spatial pain state: localized vs. distributed/referred/radiating.")
    with question_cols[1]:
        st.markdown("**Outcomes**")
        st.caption("Within-person coupling between pain rating and a wearable physiological signal.")
        st.markdown("**Setting**")
        st.caption("Naturalistic: participants' everyday environments, not a lab visit.")

    if design_stage < STAGE_STRUCTURE:
        if st.button("Continue to study design", type="primary"):
            design_tracker.advance_to(STAGE_STRUCTURE)

    # -------------------------------------------------------------
    # 1. Study design
    # -------------------------------------------------------------

    if design_stage >= STAGE_STRUCTURE:
        section_header(
            "Study Design",
            "Observational, within-person, repeated-measures",
        )

        st.write(
            "This design is **observational**, not experimental: no one "
            "assigns a participant's pain state. It is "
            "**repeated-measures**: the same participants are observed "
            "many times, over roughly a week, so the planned comparison "
            "is **within-person** - each participant compared against "
            "themselves across their own localized and distributed "
            "episodes - rather than between two separately recruited "
            "groups."
        )

        flow_cols = st.columns(6)
        flow_steps = (
            (":material/person:", "Participant", "One of the enrolled adults"),
            (
                ":material/sensors:",
                "Rating + wearable",
                "Pain rating, body map, and physiological reading",
            ),
            (":material/sync:", "Synchronization", "Aligning the two measurement streams in time"),
            (":material/functions:", "Derived measures", "Per-observation pain state and signal values"),
            (":material/insights:", "Within-person analysis", "Coupling estimated separately per participant"),
            (":material/compare_arrows:", "Comparison", "Coupling compared across pain states"),
        )
        for column, (icon, label, note) in zip(flow_cols, flow_steps):
            with column:
                st.badge(label, icon=icon, color="blue")
                st.caption(note)

        caveat(
            "An observational, within-person design can describe "
            "association within a person over time. It cannot, on its "
            "own, establish that pain state causes a change in "
            "physiology, since anything else that also varies with pain "
            "state (activity, medication timing, sleep) is not "
            "controlled for here."
        )

        if design_stage < STAGE_MEASUREMENT:
            if st.button("Continue to the measurement plan", type="primary"):
                design_tracker.advance_to(STAGE_MEASUREMENT)

    # -------------------------------------------------------------
    # 2. Measurement plan
    # -------------------------------------------------------------

    if design_stage >= STAGE_MEASUREMENT:
        section_header(
            "Measurement Plan",
            "What gets measured, how often, and how it will be aligned",
        )

        measurement_rows = pd.DataFrame(
            [
                {
                    "Construct": "Pain intensity and spatial pattern",
                    "Measure": "Numeric pain rating (0-10) + digital body map",
                    "Column": "pain_rating, pain_state",
                },
                {
                    "Construct": "Physiological arousal",
                    "Measure": "Wearable signal (illustrative single channel standing in "
                    "for EDA/HR/HRV - see Design Assumptions)",
                    "Column": "physio_signal",
                },
                {
                    "Construct": "Timing",
                    "Measure": "Timestamp per observation, several times a day",
                    "Column": "timestamp",
                },
            ]
        )
        st.dataframe(measurement_rows, width="stretch", hide_index=True)

        inspect_note(
            "The gap between when a pain rating is logged and when the "
            "wearable actually reads - temporal alignment - is a "
            "measurement-plan decision here, and an adjustable "
            "assumption in the next stage."
        )

        if design_stage < STAGE_ASSUMPTIONS:
            if st.button("Continue to design assumptions", type="primary"):
                design_tracker.advance_to(STAGE_ASSUMPTIONS)

    # -------------------------------------------------------------
    # 3. Design assumptions
    # -------------------------------------------------------------

    assumptions: DesignAssumptions | None = None

    if design_stage >= STAGE_ASSUMPTIONS:
        section_header(
            "Design Assumptions",
            "Every slider here is also live in Interactive Exploration below",
        )

        st.write(
            "These are choices, not facts about the real world - a real "
            "version of this study would need pilot data or published "
            "estimates to set them credibly. Moving any slider changes "
            "the simulated data and the estimate further down this page."
        )

        sample_cols = st.columns(2)
        with sample_cols[0]:
            n_participants = st.slider("Number of participants", 5, 100, 30)
            observations_per_day = st.slider("Observations per day", 1, 10, 4)
        with sample_cols[1]:
            duration_days = st.slider("Study duration (days)", 3, 30, 7)
            adherence_rate = st.slider(
                "Adherence rate (fraction of planned observations actually captured)",
                0.1, 1.0, 0.8, step=0.05,
            )

        noise_cols = st.columns(2)
        with noise_cols[0]:
            sensor_noise_sd = st.slider("Wearable measurement noise (SD)", 0.0, 2.0, 0.5, step=0.1)
            within_person_sd = st.slider(
                "Within-person physiological variability (SD)", 0.0, 2.0, 0.3, step=0.1
            )
        with noise_cols[1]:
            between_person_sd = st.slider(
                "Between-person variability in baseline coupling (SD)", 0.0, 2.0, 0.3, step=0.1
            )
            temporal_misalignment_minutes = st.slider(
                "Temporal misalignment between rating and wearable (minutes)", 0, 60, 10
            )

        effect_cols = st.columns(2)
        with effect_cols[0]:
            effect_magnitude = st.slider(
                "True effect: coupling difference, distributed minus localized",
                -1.0, 1.0, 0.4, step=0.05,
            )
        with effect_cols[1]:
            pain_state_prevalence = st.slider(
                "Share of observations in the distributed pain state", 0.05, 0.95, 0.35, step=0.05
            )

        seed = st.number_input("Random seed (for reproducibility)", value=42, step=1)

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

        if design_stage < STAGE_SIMULATION:
            if st.button("Continue to simulation", type="primary"):
                design_tracker.advance_to(STAGE_SIMULATION)

    # -------------------------------------------------------------
    # 4. Simulation
    # -------------------------------------------------------------

    study = None

    if design_stage >= STAGE_SIMULATION and assumptions is not None:
        section_header(
            "Simulation",
            "A synthetic dataset generated from the assumptions above",
        )

        st.warning(
            "**Illustrative simulation, not study participant data.** "
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

        st.dataframe(study.data.head(10), width="stretch", hide_index=True)
        st.caption("First 10 simulated rows, out of the retained total above.")

        if design_stage < STAGE_EXPLORATION:
            if st.button("Continue to interactive exploration", type="primary"):
                design_tracker.advance_to(STAGE_EXPLORATION)

    # -------------------------------------------------------------
    # 5. Interactive exploration
    # -------------------------------------------------------------

    estimate = None

    if design_stage >= STAGE_EXPLORATION and study is not None:
        section_header(
            "Interactive Exploration",
            "Scroll back up, change an assumption, and these numbers move",
        )

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
            "This is one illustrative analysis aligned to this "
            "simulated design (a within-person correlation difference), "
            "not a general measure of the design's quality and not a "
            "recommended analysis for a real version of this study."
        )

        st.write("**Simulated data structure -> Method Selection**")
        st.caption(
            "The measurement plan above, on its own, implies a column "
            "shape. Matching that shape to OpenMeasure's own workflows "
            "is the same suggest_workflows() the upload branch of "
            "Choose an analysis uses - here, run on a shape derived "
            "from the design instead of an uploaded file."
        )

        profile = measurement_plan_profile(assumptions)
        _render_workflow_suggestions(suggest_workflows(profile))

        if design_stage < STAGE_IMPLICATIONS:
            if st.button("Continue to implications", type="primary"):
                design_tracker.advance_to(STAGE_IMPLICATIONS)

    # -------------------------------------------------------------
    # 6. Implications
    # -------------------------------------------------------------

    if design_stage >= STAGE_IMPLICATIONS and study is not None and estimate is not None:
        section_header("Implications", "What this design does and does not support")

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
            "usable data survives to be analyzed; between-person "
            "variability and sensor noise mainly affect how uncertain "
            "the estimate is; pain-state imbalance affects how many "
            "participants have enough of the rarer state to contribute "
            "at all. Changing one does not just move the headline "
            "number - it changes which of these limits binds."
        )

        caveat(
            "Simulated precision and uncertainty describe this "
            "simulation under these assumptions. They are not a "
            "guarantee about how a real study, even one matching these "
            "settings closely, would actually perform."
        )

        with st.expander("How this connects to other OpenMeasure modules"):
            st.markdown(
                """
- **Time-Series QA** and **Impact Evaluation** are the two workflows
  this measurement plan's shape actually matches (see above) - a real
  version of this study would use Time-Series QA to check the
  timestamp column's completeness and regularity, then Impact
  Evaluation to compare the physiological signal across pain states.
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

        if design_stage < STAGE_RECORD:
            if st.button("Continue to the design record", type="primary"):
                design_tracker.advance_to(STAGE_RECORD)

    # -------------------------------------------------------------
    # 7. Design record
    # -------------------------------------------------------------

    if design_stage >= STAGE_RECORD and assumptions is not None and study is not None and estimate is not None:
        section_header("Design Record", "A summary of this design, to carry forward")

        record_text = f"""OpenMeasure Research Design Record
===================================

Research question
------------------
Does the coupling between subjective pain and a wearable physiological
signal change when chronic pain is localized versus spatially
distributed, referred, or radiating?

Study design
------------
Observational, within-person, repeated-measures, naturalistic setting.

Measurement plan
-----------------
- Pain rating (0-10) + digital body map -> pain_rating, pain_state
- Wearable physiological signal -> physio_signal
- Timestamp per observation -> timestamp

Design assumptions
--------------------
- Participants: {assumptions.n_participants}
- Observations per day: {assumptions.observations_per_day}
- Duration: {assumptions.duration_days} days
- Adherence rate: {assumptions.adherence_rate:.0%}
- Sensor noise SD: {assumptions.sensor_noise_sd}
- Within-person SD: {assumptions.within_person_sd}
- Between-person SD: {assumptions.between_person_sd}
- True effect (distributed - localized): {assumptions.effect_magnitude:+.2f}
- Distributed-state prevalence: {assumptions.pain_state_prevalence:.0%}
- Temporal misalignment: {assumptions.temporal_misalignment_minutes:.0f} minutes
- Random seed: {assumptions.seed}

Simulated outcome (illustrative)
-----------------------------------
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
not guarantee real-world performance. Illustrative simulation, not
study participant data.
"""

        st.text_area("Design record", record_text, height=400)
        st.download_button(
            "Download design record (.txt)",
            data=record_text,
            file_name="openmeasure_research_design_record.txt",
            mime="text/plain",
        )
