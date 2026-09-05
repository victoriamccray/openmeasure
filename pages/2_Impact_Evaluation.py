"""
OpenMeasure - Program Evaluation module.

Compares outcomes across groups or across time (pre/post), auto-
recommending the appropriate test based on data shape, then lets the
user confirm or override before running it. Core logic lives in
modules/program_evaluation/core, this file is presentation only.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from modules.data_profile.core.suggest import (
    default_group_column,
    default_outcome_column,
    default_prepost_columns,
)
from modules.program_evaluation.core import comparison as comp
from modules.program_evaluation.core import designs
from modules.program_evaluation.core import did as did_core
from modules.program_evaluation.core import domains
from modules.program_evaluation.core import formulas
from modules.program_evaluation.core import interpret
from modules.program_evaluation.core import recommend as rec
from modules.program_evaluation.core import research
from modules.program_evaluation.core import teaching
from shared.catalog import MODULE_PROGRAM_EVALUATION
from shared.handoff import (
    KIND_ROWS_DROPPED,
    ExclusionAccount,
    HandoffStore,
    RetentionItem,
    fingerprint_dataframe,
)
from shared.data_handling import disclosure_for, render_data_handling_summary
from shared.journey_stages import StageTracker
from shared.literature import SEARCH_ERRORS, search_openalex
from shared.report import (
    section_header,
    case_study_note,
    caveat,
    flagged_item_note,
    implications,
    inspect_note,
    interpretation_note,
    render_formula,
    render_lifecycle_tracker,
)
from shared.upload import render_data_entry, render_data_profile


def render_sensitivity_sub_result(sub_result) -> None:
    """
    Formatted display for one coding scheme's result inside the
    sensitivity-analysis expander, instead of the raw dataclass repr
    st.write() would otherwise show (every attribute name printed
    verbatim). sub_result is comp.TwoGroupResult or comp.MultiGroupResult,
    per comp.SensitivityResult.coding_results' own type hint.
    """
    if isinstance(sub_result, comp.TwoGroupResult):
        c1, c2 = st.columns(2)
        c1.metric(
            f"{sub_result.group_a_label} mean",
            f"{sub_result.mean_a:.2f}",
            f"n={sub_result.n_a}",
        )
        c2.metric(
            f"{sub_result.group_b_label} mean",
            f"{sub_result.mean_b:.2f}",
            f"n={sub_result.n_b}",
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("t-statistic", f"{sub_result.t_statistic:.2f}")
        m2.metric("p-value", f"{sub_result.p_value:.4f}")
        m3.metric("Cohen's d", f"{sub_result.cohens_d:.2f}")
    else:
        means_df = pd.DataFrame(
            {
                "Group": sub_result.group_labels,
                "n": [sub_result.group_ns[g] for g in sub_result.group_labels],
                "Mean": [
                    round(sub_result.group_means[g], 3)
                    for g in sub_result.group_labels
                ],
            }
        )
        st.dataframe(means_df, width="stretch", hide_index=True)
        m1, m2 = st.columns(2)
        m1.metric("F-statistic", f"{sub_result.f_statistic:.2f}")
        m2.metric("p-value", f"{sub_result.p_value:.4f}")


# The same page-local SVG palette used by Method Selection, fMRI QC,
# HealthRing, and GAIA. Repeated here rather than imported because no
# shared palette module exists yet; extracting one would touch six other
# pages, which is a wider change than this feature.
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
ACCENT = "#2a78d6"
ACCENT_2 = "#c0392b"

# Fixed y-window for the teaching diagram, wide enough to contain every
# value the slider can produce. Fixed rather than fitted to the current
# values on purpose: an axis that rescaled as the slider moved would keep
# the lines looking the same distance apart no matter what the comparison
# group did, which is the one thing the diagram exists to show.
_TEACH_Y_MIN, _TEACH_Y_MAX = 45.0, 85.0
_TEACH_PLOT_TOP, _TEACH_PLOT_BOTTOM = 35.0, 150.0
_TEACH_X_PRE, _TEACH_X_POST = 95.0, 265.0


def _teach_y(value: float) -> float:
    """Map an outcome percentage onto the diagram's fixed vertical axis."""
    span = (_TEACH_PLOT_BOTTOM - _TEACH_PLOT_TOP) / (_TEACH_Y_MAX - _TEACH_Y_MIN)
    return _TEACH_PLOT_BOTTOM - (value - _TEACH_Y_MIN) * span


def _did_teaching_svg(scenario, outcome) -> str:
    """
    The standard difference-in-differences figure, redrawn on each slider
    change rather than animated.

    Three lines: what the treated clinic did, what the comparison clinic
    did, and the dashed path the treated clinic is assumed to have been
    on without the program. The dashed line is the comparison clinic's
    slope pinned to the treated clinic's own starting point, so the
    parallel-trends assumption is visible as two parallel segments rather
    than described in a sentence. The bracket at the right is the
    estimate, which is the gap the dashed line leaves behind.

    Solid against dashed, and circle against square endpoints, carry the
    same distinctions the colors do, so the figure does not depend on
    color alone.
    """
    y_pre_treated = _teach_y(scenario.pre_treated)
    y_post_treated = _teach_y(scenario.post_treated)
    y_pre_comparison = _teach_y(scenario.pre_comparison)
    y_post_comparison = _teach_y(outcome.post_comparison)
    y_counterfactual = _teach_y(scenario.pre_treated + outcome.comparison_change)

    gridlines = "".join(
        f'<line x1="88" y1="{_teach_y(v):.1f}" x2="270" y2="{_teach_y(v):.1f}" '
        f'stroke="{GRIDLINE}" stroke-width="1"/>'
        f'<text x="82" y="{_teach_y(v) + 3:.1f}" font-size="8" fill="{INK_MUTED}" '
        f'text-anchor="end">{v:.0f}%</text>'
        for v in (50, 60, 70, 80)
    )

    bracket_x = 274.0
    gap_label_y = (y_post_treated + y_counterfactual) / 2 + 3

    return f"""
    <svg width="100%" height="185" viewBox="0 0 360 185"
         preserveAspectRatio="xMidYMid meet" role="img"
         aria-label="{scenario.outcome_label.capitalize()} for
         {scenario.treated_label} and {scenario.comparison_label} before and
         after the {scenario.program_label}, with the path
         {scenario.treated_label} is assumed to have been on without it.">
      <line x1="8" y1="10" x2="24" y2="10" stroke="{ACCENT_2}" stroke-width="2"/>
      <text x="28" y="13" font-size="9" fill="{INK_MUTED}">{scenario.treated_label}</text>
      <line x1="90" y1="10" x2="106" y2="10" stroke="{ACCENT}" stroke-width="2"/>
      <text x="110" y="13" font-size="9" fill="{INK_MUTED}">{scenario.comparison_label}</text>
      <line x1="180" y1="10" x2="196" y2="10" stroke="{INK_MUTED}" stroke-width="2"
            stroke-dasharray="4,3"/>
      <text x="200" y="13" font-size="9" fill="{INK_MUTED}">
        {scenario.treated_label} without the {scenario.program_label}
      </text>

      {gridlines}

      <line x1="{_TEACH_X_PRE}" y1="{_TEACH_PLOT_TOP}" x2="{_TEACH_X_PRE}"
            y2="{_TEACH_PLOT_BOTTOM}" stroke="{GRIDLINE}" stroke-width="1"/>
      <line x1="{_TEACH_X_POST}" y1="{_TEACH_PLOT_TOP}" x2="{_TEACH_X_POST}"
            y2="{_TEACH_PLOT_BOTTOM}" stroke="{GRIDLINE}" stroke-width="1"/>

      <line x1="{_TEACH_X_PRE}" y1="{y_pre_treated:.1f}" x2="{_TEACH_X_POST}"
            y2="{y_counterfactual:.1f}" stroke="{INK_MUTED}" stroke-width="1.5"
            stroke-dasharray="4,3"/>
      <line x1="{_TEACH_X_PRE}" y1="{y_pre_comparison:.1f}" x2="{_TEACH_X_POST}"
            y2="{y_post_comparison:.1f}" stroke="{ACCENT}" stroke-width="2"/>
      <line x1="{_TEACH_X_PRE}" y1="{y_pre_treated:.1f}" x2="{_TEACH_X_POST}"
            y2="{y_post_treated:.1f}" stroke="{ACCENT_2}" stroke-width="2"/>

      <circle cx="{_TEACH_X_PRE}" cy="{y_pre_treated:.1f}" r="3.5" fill="{ACCENT_2}"/>
      <circle cx="{_TEACH_X_POST}" cy="{y_post_treated:.1f}" r="3.5" fill="{ACCENT_2}"/>
      <rect x="{_TEACH_X_PRE - 3:.1f}" y="{y_pre_comparison - 3:.1f}" width="6"
            height="6" fill="{ACCENT}"/>
      <rect x="{_TEACH_X_POST - 3:.1f}" y="{y_post_comparison - 3:.1f}" width="6"
            height="6" fill="{ACCENT}"/>

      <line x1="{bracket_x}" y1="{y_post_treated:.1f}" x2="{bracket_x}"
            y2="{y_counterfactual:.1f}" stroke="{INK_MUTED}" stroke-width="1"/>
      <line x1="{bracket_x - 3}" y1="{y_post_treated:.1f}" x2="{bracket_x + 3}"
            y2="{y_post_treated:.1f}" stroke="{INK_MUTED}" stroke-width="1"/>
      <line x1="{bracket_x - 3}" y1="{y_counterfactual:.1f}" x2="{bracket_x + 3}"
            y2="{y_counterfactual:.1f}" stroke="{INK_MUTED}" stroke-width="1"/>
      <text x="{bracket_x + 7}" y="{gap_label_y:.1f}" font-size="10" fill="{INK_MUTED}">
        {outcome.did_estimate:+.1f} pts
      </text>
      <title>Difference of {outcome.did_estimate:+.1f} {scenario.unit_label}
      between what {scenario.treated_label} reached and where it would have
      ended up on {scenario.comparison_label}'s path.</title>

      <text x="{_TEACH_X_PRE}" y="165" font-size="9" fill="{INK_MUTED}"
            text-anchor="middle">{scenario.pre_period_label}</text>
      <text x="{_TEACH_X_POST}" y="165" font-size="9" fill="{INK_MUTED}"
            text-anchor="middle">{scenario.post_period_label}</text>
      <text x="8" y="178" font-size="8" fill="{INK_MUTED}">
        {scenario.outcome_label.capitalize()}
      </text>
    </svg>
    """


def render_did_teaching_example(domain_id: str) -> None:
    """
    A fixed scenario with one adjustable number, shown collapsed beside
    the difference-in-differences design.

    Collapsed by default so it stays out of the way of an analysis, and
    placed before the recommendation rather than after the result,
    because the thing it teaches (what the comparison group is doing to
    the estimate) is a design decision the reader is making right here.

    Every value and every sentence comes from
    modules/program_evaluation/core/teaching.py, so the prose and the
    arithmetic cannot drift apart.

    The field selector changes which telling of the scenario is shown and
    nothing else. It is scoped to this example rather than offered as a
    page-level setting, because that is all it currently does: the wider
    domain layer (search seeding, outcome suggestions and their caveats)
    has no stage to live in until the workflow is restructured.
    """
    domain_labels = {domain.id: domain.label for domain in domains.DOMAINS}
    scenario = teaching.did_scenario_for(domain_id)

    with st.expander("See how a comparison group changes the estimate", expanded=True):

        if not teaching.has_own_did_scenario(domain_id):
            fallback_label = domain_labels[scenario.domain_id]
            st.caption(
                f"{domain_labels[domain_id]} does not have its own telling "
                f"of this example yet, so it is shown in {fallback_label.lower()} "
                "terms. The arithmetic is identical either way, which is "
                "the point: the field changes the story, not the method."
            )

        st.markdown("**Scenario**")
        st.write(scenario.scenario)

        st.markdown("**Research question**")
        st.write(scenario.question)

        st.markdown("**Why difference-in-differences fits**")
        st.write(scenario.method_fit)

        st.markdown("**Assumptions this rests on**")
        for assumption in scenario.assumptions:
            st.write(f"- {assumption}")

        st.markdown("**Result**")
        comparison_change = st.slider(
            f"How much {scenario.comparison_label}'s "
            f"{scenario.outcome_label} changed over the same "
            f"{scenario.period_label} ({scenario.unit_label})",
            min_value=-10.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
        )
        outcome = teaching.teaching_did(comparison_change, scenario=scenario)

        st.markdown(
            _did_teaching_svg(scenario, outcome),
            unsafe_allow_html=True,
        )
        st.caption(
            f"The dashed line is {scenario.comparison_label}'s slope drawn "
            f"from {scenario.treated_label}'s own starting point. Parallel "
            "trends is the assumption that the dashed line is where "
            f"{scenario.treated_label} would have ended up, and the bracket "
            "is what is left over."
        )

        st.dataframe(
            pd.DataFrame(
                {
                    "Group": [scenario.treated_label, scenario.comparison_label],
                    scenario.pre_period_label: [
                        scenario.pre_treated,
                        scenario.pre_comparison,
                    ],
                    scenario.post_period_label: [
                        scenario.post_treated,
                        outcome.post_comparison,
                    ],
                    "Change": [outcome.change_treated, outcome.comparison_change],
                }
            ),
            width="stretch",
            hide_index=True,
        )

        t1, t2 = st.columns(2)
        t1.metric(
            "Before-and-after estimate",
            f"{outcome.before_after_estimate:+.1f}",
        )
        t2.metric(
            "Difference-in-differences estimate",
            f"{outcome.did_estimate:+.1f}",
        )
        inspect_note(
            "Which of the two estimates moves when you move the slider, and "
            "which one cannot see the comparison group at all."
        )
        st.write(outcome.reading)

        st.markdown("**What this can conclude**")
        for item in scenario.can_conclude:
            st.write(f"- {item}")

        st.markdown("**What this cannot conclude**")
        for item in scenario.cannot_conclude:
            st.write(f"- {item}")


def render_did_result(result) -> None:
    """
    The four cell means, the estimate, its assumptions, and its reading.

    The 2x2 grid is shown before the estimate because the estimate is a
    difference of two differences: a reader cannot tell a large treated
    change from a small comparison change without both rows in front of
    them.

    The assumptions and the plain-language reading are rendered
    separately by render_did_interpretation(), which the interpretation
    stage calls.
    """
    section_header("Result")

    st.dataframe(
        pd.DataFrame(
            {
                "Group": [result.treated_label, result.comparison_label],
                "n": [result.n_treated, result.n_comparison],
                "Pre": [
                    round(result.mean_pre_treated, 3),
                    round(result.mean_pre_comparison, 3),
                ],
                "Post": [
                    round(result.mean_post_treated, 3),
                    round(result.mean_post_comparison, 3),
                ],
                "Change": [
                    round(result.change_treated, 3),
                    round(result.change_comparison, 3),
                ],
            }
        ),
        width="stretch",
        hide_index=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Difference-in-differences", f"{result.did_estimate:.2f}")
    m2.metric("Standard error", f"{result.standard_error:.2f}")
    m3.metric("p-value", f"{result.p_value:.4f}")

    st.caption(
        f"95% CI: {result.ci_95_low:.2f} to {result.ci_95_high:.2f}, "
        f"t = {result.t_statistic:.2f}, df = {result.degrees_of_freedom:.1f}. "
        f"Standard error method: {result.standard_error_method}, which does "
        "not assume the two groups' changes vary by the same amount."
    )

    if result.small_groups_flagged:
        caveat(
            "Small group(s) flagged: "
            + "; ".join(result.small_groups_flagged)
            + "."
        )

    render_formula(formulas.did_explanation(result))

    inspect_note(
        "The two Change values. The estimate is the gap between them, so a "
        "comparison group that moved on its own is doing as much work here "
        "as the treated group's own change."
    )

def render_did_interpretation(result) -> None:
    """
    The assumptions a causal reading rests on, and that reading.

    Split from render_did_result() so the numbers belong to the analysis
    stage and this belongs to the interpretation stage. Assumptions render
    visibly rather than inside an expander: parallel trends is the
    condition the causal reading rests on and cannot be tested with two
    time points, so it is not something a reader should have to open
    something to find.
    """
    section_header("Assumptions Behind a Causal Reading")
    st.caption(
        "The arithmetic above is the same whether these hold or not. Each "
        "one states what would have to be true of the setting for the "
        "estimate to be the program's effect."
    )

    for assumption in interpret.did_assumptions():
        st.markdown(f"**{assumption.name}**")
        st.write(assumption.statement)
        st.caption(f"{assumption.checkable}. {assumption.citation}")

    section_header("Interpretation")

    reading = interpret.interpret_did(result)

    interpretation_note(reading.headline)

    st.markdown("**What this estimate is about**")
    st.write(reading.estimand)

    st.markdown("**What it supports**")
    for item in reading.supports:
        st.write(f"- {item}")

    st.markdown("**What it does not support**")
    for item in reading.does_not_support:
        st.write(f"- {item}")

    if reading.observations:
        st.markdown("**Worth noticing in this result**")
        for item in reading.observations:
            st.write(f"- {item}")


def record_comparison(frame, source_name, analysis_context, recommendation, result) -> None:
    """
    Record this analysis for the Cross-Analysis Implications page.

    Translates into primitives here rather than storing the result object,
    because the same dataclass is a different class depending on how it was
    imported.

    The multi-select sensitivity analysis expands one row per selection, so
    its retained count is not comparable to other analyses and is flagged
    rather than counted.
    """
    expands_rows = getattr(result, "rows_can_exceed_participants", False)

    columns = tuple(
        str(value)
        for key, value in analysis_context.items()
        if key.endswith("_col") and value
    )

    if expands_rows:
        # The expanded coding turns one participant into one observation per
        # selection, so there is no retained-participant count to report.
        account = ExclusionAccount(
            module=MODULE_PROGRAM_EVALUATION,
            analysis_label=recommendation.display_name,
            columns_considered=columns,
            n_input_rows=result.n_input_rows,
            n_expanded_observations=result.n_expanded_rows,
            rows_can_exceed_participants=True,
        )
    else:
        account = ExclusionAccount(
            module=MODULE_PROGRAM_EVALUATION,
            analysis_label=recommendation.display_name,
            columns_considered=columns,
            n_input_rows=result.n_input_rows,
            n_retained_rows=result.n_rows_used,
            items=(
                RetentionItem(
                    label="Rows excluded",
                    count=result.n_excluded_rows,
                    kind=KIND_ROWS_DROPPED,
                    mechanism=result.exclusion_reason,
                ),
            ),
        )

    statistics = {}
    for name in ("cohens_d", "did_estimate", "p_value"):
        value = getattr(result, name, None)
        if isinstance(value, (int, float)):
            statistics[name] = float(value)

    HandoffStore(st.session_state).record(
        module=MODULE_PROGRAM_EVALUATION,
        fingerprint=fingerprint_dataframe(frame, source_name),
        exclusion=account,
        primary_statistics=statistics,
    )

st.set_page_config(page_title="OpenMeasure · Program Evaluation", page_icon=":material/monitoring:", layout="centered")

st.title("Impact Evaluation")
st.subheader("Program Validation")
st.caption(
    "Work from an evaluation question to a defensible estimate, one step "
    "at a time."
)

st.divider()

render_lifecycle_tracker(current_workflow="Impact Evaluation")

render_data_handling_summary(disclosure_for("pages/2_Impact_Evaluation.py"))

# One gated sequence, in the order an evaluation is actually reasoned
# through. Before this, the page put its case studies, its method
# discussion, and a standalone teaching example above a numbered upload
# workflow, so it read as two pages stacked rather than as one path.
STAGE_QUESTION = 0
STAGE_DOMAIN = 1
STAGE_RESEARCH = 2
STAGE_DESIGNS = 3
STAGE_EXAMPLE = 4
STAGE_ANALYZE = 5
STAGE_INTERPRET = 6

STAGE_LABELS = (
    "Evaluation question",
    "Domain",
    "Find research",
    "Explore designs",
    "Interactive example",
    "Analyze your data",
    "Interpret",
)

TRACKER = StageTracker(session_key="pe_stage", stage_labels=STAGE_LABELS)

stage = TRACKER.render_breadcrumb()
TRACKER.render_restart_button(
    extra_session_keys=(
        "pe_recommendation",
        "pe_context",
        "pe_uploaded_file_id",
        "pe_search_results",
        "pe_search_provenance",
        "pe_selected_studies",
    )
)

# Stages reveal one at a time by stopping the script rather than by
# indenting each stage into a block. The analysis further down is the same
# code it was before this sequence existed, which a re-indentation would
# have quietly put at risk.

# ---------------------------------------------------------------------
# 1. Evaluation question
# ---------------------------------------------------------------------

section_header(
    "1. Evaluation Question",
    "What is being evaluated, for whom, against what, and over what period",
)

question_columns = st.columns(2)
with question_columns[0]:
    q_program = st.text_input(
        "Program or intervention",
        key="pe_q_program",
        placeholder="e.g. text-message appointment reminders",
    )
    q_population = st.text_input(
        "Population",
        key="pe_q_population",
        placeholder="e.g. adult primary-care patients",
    )
    q_timing = st.text_input(
        "Timing",
        key="pe_q_timing",
        placeholder="e.g. six months before and after launch",
    )
with question_columns[1]:
    q_outcome = st.text_input(
        "Outcome",
        key="pe_q_outcome",
        placeholder="e.g. share of appointments kept",
    )
    q_comparison = st.text_input(
        "Comparison",
        key="pe_q_comparison",
        placeholder="e.g. a similar clinic that sent no reminders",
    )

question_terms = " ".join(
    part.strip() for part in (q_program, q_outcome) if part.strip()
)

if question_terms:
    st.caption(
        "These words seed the literature search two stages on. Nothing is "
        "sent anywhere until you run that search."
    )

case_study_note(
    "head_start_impact_study",
    "A question names an outcome and the period it is measured over, and "
    "an evaluation answers it only for those. Settling both here is what "
    "keeps a later result from being read as a claim about outcomes it "
    "never measured, or about a period it never covered.",
)

if stage < STAGE_DOMAIN:
    if st.button(
        "Continue to domain",
        type="primary",
        disabled=not question_terms,
    ):
        TRACKER.advance_to(STAGE_DOMAIN)
    if not question_terms:
        st.caption("Name at least the program and the outcome to continue.")
    st.stop()

# ---------------------------------------------------------------------
# 2. Domain
# ---------------------------------------------------------------------

section_header(
    "2. Domain",
    "Your field, which sets vocabulary and search terms and nothing else",
)

domain_labels = {domain.id: domain.label for domain in domains.DOMAINS}
domain_id = st.selectbox(
    "Field of practice",
    options=list(domain_labels),
    format_func=lambda key: domain_labels[key],
    key="pe_domain",
)
selected_domain = domains.get_domain(domain_id)

st.caption(
    "The field changes the words this page uses, the terms it suggests "
    "for a literature search, and which telling of the worked example you "
    "see. It does not change which method is recommended, or what any "
    "statistic comes out as."
)

st.markdown("**How this field names each design concept**")
st.dataframe(
    pd.DataFrame(
        {
            "Concept": list(domains.CONCEPTS),
            "In this field": [
                selected_domain.term_for(concept) for concept in domains.CONCEPTS
            ],
        }
    ),
    width="stretch",
    hide_index=True,
)

if selected_domain.outcomes:
    st.markdown("**Outcomes this field commonly measures**")
    for outcome_suggestion in selected_domain.outcomes:
        st.write(f"- {outcome_suggestion.label}")
        if outcome_suggestion.caveat:
            st.caption(outcome_suggestion.caveat)
    inspect_note(
        "The notes under some outcomes. They describe what that measure "
        "responds to besides the thing being studied."
    )
else:
    st.caption(
        "No field-specific outcome suggestions, so the search below uses "
        "your own words alone."
    )

if stage < STAGE_RESEARCH:
    if st.button("Continue to find research", type="primary"):
        TRACKER.advance_to(STAGE_RESEARCH)
    st.stop()

# ---------------------------------------------------------------------
# 3. Find research
# ---------------------------------------------------------------------

section_header(
    "3. Find Research",
    "Optional: how this question has been studied before",
)

st.info(
    "Published work informs your reasoning here. It does not choose your "
    "method. Whatever design these studies used, the designs the next "
    "stage surfaces come from your own answers and your data's shape."
)

default_query = domains.build_search_query(question_terms, domain_id)
query = st.text_input(
    "Search terms sent to OpenAlex",
    value=default_query,
    key="pe_query",
)

if st.button("Search OpenAlex", disabled=not query.strip()):
    try:
        st.session_state["pe_search_results"] = search_openalex(query)
        # Provenance for a later Evaluation Record: the query actually
        # sent (which the reader may have edited), the field context it
        # was composed under, and when it ran. Captured at search time
        # because none of it can be reconstructed afterwards.
        st.session_state["pe_search_provenance"] = {
            "query": query,
            "domain_id": domain_id,
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    except SEARCH_ERRORS as error:
        st.session_state["pe_search_results"] = []
        st.error(f"The search could not be completed: {error}")

raw_results = st.session_state.get("pe_search_results")

if raw_results:
    rows = research.research_rows(raw_results, question_terms)

    st.dataframe(
        pd.DataFrame([row.as_display_row() for row in rows]),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"{research.OVERLAP_COLUMN} counts words your question and a "
        "result's title or abstract have in common. It is a text overlap, "
        "not a judgment that a result is relevant."
    )

    st.multiselect(
        "Studies worth keeping in mind",
        options=research.selectable_titles(rows),
        key="pe_selected_studies",
    )
elif raw_results is not None:
    st.caption("That search returned no results. Try broader terms.")

if stage < STAGE_DESIGNS:
    skip_column, continue_column = st.columns(2)
    with continue_column:
        if st.button("Continue to designs", type="primary"):
            TRACKER.advance_to(STAGE_DESIGNS)
    with skip_column:
        if st.button("Skip this step"):
            TRACKER.advance_to(STAGE_DESIGNS)
    st.stop()

# ---------------------------------------------------------------------
# 4. Explore designs
# ---------------------------------------------------------------------

section_header(
    "4. Explore Designs",
    "What each design compares, and what it can support",
)

selected_studies = st.session_state.get("pe_selected_studies") or []
if selected_studies:
    with st.expander(f"Studies you kept ({len(selected_studies)})"):
        for title in selected_studies:
            st.write(f"- {title}")
        st.caption(
            "Shown as context. Which design fits depends on your own "
            "answers and your data's shape, not on what these studies did."
        )

for design in designs.DESIGN_OPTIONS:
    st.markdown(f"**{design.label}**")
    st.write(design.summary_for(selected_domain))

    if design.caveat:
        st.caption(design.caveat)

    if design.case_study_key:
        case_study_note(design.case_study_key, design.case_study_connection)

implications(designs.DESIGN_CHOICE_IMPLICATION)

if stage < STAGE_EXAMPLE:
    if st.button("Continue to the worked example", type="primary"):
        TRACKER.advance_to(STAGE_EXAMPLE)
    st.stop()

# ---------------------------------------------------------------------
# 5. Interactive example
# ---------------------------------------------------------------------

section_header(
    "5. Interactive Example",
    "Difference-in-differences on numbers you can move, before your own data",
)

render_did_teaching_example(domain_id)

if stage < STAGE_ANALYZE:
    if st.button("Continue to your own data", type="primary"):
        TRACKER.advance_to(STAGE_ANALYZE)
    st.stop()

# ---------------------------------------------------------------------
# 6. Analyze your data
# ---------------------------------------------------------------------

section_header("6. Analyze Your Data", "CSV file, one row per participant")

loaded = render_data_entry(
    MODULE_PROGRAM_EVALUATION,
    empty_prompt=(
        "Load the sample dataset to try a comparison straight away, or "
        "upload your own CSV."
    ),
)

if loaded is None:
    st.stop()

df = loaded.frame
profile = render_data_profile(df)

# Discard a recommendation carried over from different data. Without this,
# loading a second dataset whose column names happen to match the first
# would analyze the new data under the previous plan, silently.
if st.session_state.get("pe_uploaded_file_id") != loaded.token:
    st.session_state["pe_uploaded_file_id"] = loaded.token
    st.session_state.pop("pe_recommendation", None)
    st.session_state.pop("pe_context", None)

st.write(f"Loaded **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
st.dataframe(df.head(), width="stretch")

# Column defaults come from the data profile's role guesses, so the
# zero-friction path opens on a meaningful analysis rather than on
# whichever column happens to be first. An identifier column stays
# selectable; it is simply not the default. Without this the bundled
# example opened with participant_id as its outcome, produced a
# technically valid but meaningless comparison, and only warned about it
# two steps later.
def index_of(options, chosen):
    """Where a picker should open, as an index into its own options."""
    return options.index(chosen) if chosen in options else 0


DESIGN_GROUPS = "Two or more groups"
DESIGN_PRE_POST = "Pre/post (same participants)"
DESIGN_DID = "Two groups, each measured before and after"

design = st.radio(
    "What are you comparing?",
    options=[DESIGN_GROUPS, DESIGN_PRE_POST, DESIGN_DID],
    index=0,
)

recommendation = None
context = {}

if design == DESIGN_GROUPS:
    outcome_options = list(df.columns)
    outcome_col = st.selectbox(
        "Outcome column",
        options=outcome_options,
        index=index_of(outcome_options, default_outcome_column(profile, outcome_options)),
    )
    group_options = [c for c in df.columns if c != outcome_col]
    group_col = st.selectbox(
        "Group column",
        options=group_options,
        index=index_of(group_options, default_group_column(profile, group_options)),
    )
    is_multiselect = st.checkbox(
        "This group column allows multiple selections per participant "
        "(e.g. a comma-separated demographic field)",
        value=False,
    )
    delimiter = ","
    if is_multiselect:
        delimiter = st.text_input("Delimiter used to separate multiple selections", value=",")

    context = {
        "outcome_col": outcome_col,
        "group_col": group_col,
        "is_multiselect": is_multiselect,
        "delimiter": delimiter,
    }

    if st.button("Get recommendation", type="primary"):
        try:
            recommendation = rec.recommend_method(
                df,
                outcome_col=outcome_col,
                group_col=group_col,
                is_multiselect_group=is_multiselect,
                multiselect_delimiter=delimiter,
            )
        except (ValueError, TypeError) as e:
            st.error(str(e))
            st.stop()

elif design == DESIGN_PRE_POST:
    pre_options = list(df.columns)
    default_pre, default_post = default_prepost_columns(profile, pre_options)
    pre_col = st.selectbox(
        "Pre (baseline) column",
        options=pre_options,
        index=index_of(pre_options, default_pre),
    )
    post_options = [c for c in df.columns if c != pre_col]
    post_col = st.selectbox(
        "Post (follow-up) column",
        options=post_options,
        index=index_of(post_options, default_post),
    )
    context = {"pre_col": pre_col, "post_col": post_col}

    if st.button("Get recommendation", type="primary"):
        try:
            recommendation = rec.recommend_method(df, pre_col=pre_col, post_col=post_col)
        except (ValueError, TypeError) as e:
            st.error(str(e))
            st.stop()

else:
    st.caption(
        "This design needs one row per unit, a column marking which two "
        "groups the units belong to, and that unit's outcome measured at "
        "the same two time points in two more columns."
    )

    group_options = list(df.columns)
    group_col = st.selectbox(
        "Group column (which units were treated)",
        options=group_options,
        index=index_of(group_options, default_group_column(profile, group_options)),
    )
    remaining = [c for c in df.columns if c != group_col]
    default_pre, default_post = default_prepost_columns(profile, remaining)
    pre_col = st.selectbox(
        "Pre (baseline) column",
        options=remaining,
        index=index_of(remaining, default_pre),
    )
    post_options = [c for c in remaining if c != pre_col]
    post_col = st.selectbox(
        "Post (follow-up) column",
        options=post_options,
        index=index_of(post_options, default_post),
    )

    group_values = sorted(df[group_col].dropna().unique().tolist(), key=str)
    treated_label = st.selectbox(
        "Which group received the program?",
        options=group_values,
    )
    st.caption(
        "Nothing in the data marks which group was treated, and this choice "
        "sets the sign of the estimate, so it is asked rather than guessed."
    )

    context = {
        "group_col": group_col,
        "pre_col": pre_col,
        "post_col": post_col,
        "treated_label": treated_label,
    }

    if st.button("Get recommendation", type="primary"):
        try:
            recommendation = rec.recommend_method(
                df,
                group_col=group_col,
                pre_col=pre_col,
                post_col=post_col,
            )
        except (ValueError, TypeError) as e:
            st.error(str(e))
            st.stop()

if recommendation is not None:
    st.session_state["pe_recommendation"] = recommendation
    st.session_state["pe_context"] = context

if "pe_recommendation" in st.session_state:
    recommendation = st.session_state["pe_recommendation"]
    context = st.session_state["pe_context"]

    section_header("Recommendation")
    st.markdown(f"**{recommendation.display_name}**")
    for r in recommendation.reasoning:
        st.write(f"- {r}")
    for w in recommendation.warnings:
        st.warning(w)

    if not recommendation.supported:
        st.error(
            "This method is not yet implemented in OpenMeasure. "
            "See the module README for current scope."
        )
        st.stop()

    run_clicked = st.button("Run analysis", type="primary")

    if run_clicked:
        method = recommendation.method
        result = None

        try:
            if method == "compare_two_groups":
                result = comp.compare_two_groups(df, context["group_col"], context["outcome_col"])

                section_header("Result")
                c1, c2 = st.columns(2)
                c1.metric(f"{result.group_a_label} mean", f"{result.mean_a:.2f}", f"n={result.n_a}")
                c2.metric(f"{result.group_b_label} mean", f"{result.mean_b:.2f}", f"n={result.n_b}")

                m1, m2, m3 = st.columns(3)
                m1.metric("t-statistic", f"{result.t_statistic:.2f}")
                m2.metric("p-value", f"{result.p_value:.4f}")
                m3.metric("Cohen's d", f"{result.cohens_d:.2f}")

                st.caption(
                    f"Mean difference: {result.mean_difference:.2f} "
                    f"(95% CI: {result.ci_95_low:.2f} to {result.ci_95_high:.2f}), "
                    f"df = {result.degrees_of_freedom:.1f}"
                )

                render_formula(formulas.cohens_d_explanation(result))

                inspect_note("The p-value against your significance threshold, and Cohen's d for effect size.")
                implications(
                    "A p-value below the threshold is evidence of a "
                    "difference between the observed groups, under Welch's "
                    "t-test assumptions. Whether that difference can be "
                    "attributed to the program depends on the study design, "
                    "not on the p-value. Above the threshold, no difference "
                    "was detected at this sample size."
                )

                nonparametric = comp.compare_two_groups_nonparametric(
                    df, context["group_col"], context["outcome_col"]
                )
                with st.expander("Compare with a rank-based test (Mann-Whitney U)"):
                    st.markdown(
                        "Welch's t-test above compares means and assumes the "
                        "outcome is roughly normally distributed within each "
                        "group. Mann-Whitney U compares ranks instead, and "
                        "makes no distributional assumption. Choosing between "
                        "them is a choice, not a formality: it can change the "
                        "conclusion."
                    )
                    nc1, nc2 = st.columns(2)
                    nc1.metric(
                        f"{nonparametric.group_a_label} median",
                        f"{nonparametric.median_a:.2f}",
                    )
                    nc2.metric(
                        f"{nonparametric.group_b_label} median",
                        f"{nonparametric.median_b:.2f}",
                    )
                    nm1, nm2, nm3 = st.columns(3)
                    nm1.metric("U statistic", f"{nonparametric.u_statistic:.1f}")
                    nm2.metric("p-value", f"{nonparametric.p_value:.4f}")
                    nm3.metric(
                        "Rank-biserial r",
                        f"{nonparametric.rank_biserial_correlation:.2f}",
                    )

                    if (result.p_value < 0.05) == (nonparametric.p_value < 0.05):
                        st.success(
                            "Welch's t-test and Mann-Whitney U fall on the "
                            "same side of α=0.05 here."
                        )
                    else:
                        st.warning(
                            "Welch's t-test and Mann-Whitney U fall on "
                            "opposite sides of α=0.05 here. The two tests "
                            "compare different quantities (means versus "
                            "ranks) and can disagree, particularly with "
                            "skewed data or outliers. Treat the choice of "
                            "test as consequential for this result, not a "
                            "formality."
                        )
                    inspect_note(
                        "Whether the two tests land on the same side of "
                        "α=0.05."
                    )

            elif method == "compare_multiple_groups_welch":
                try:
                    result = comp.compare_multiple_groups_welch(df, context["group_col"], context["outcome_col"])
                except ImportError as e:
                    st.error(str(e))
                    st.stop()

                section_header("Result")
                means_df = pd.DataFrame({
                    "Group": result.group_labels,
                    "n": [result.group_ns[g] for g in result.group_labels],
                    "Mean": [round(result.group_means[g], 3) for g in result.group_labels],
                })
                st.dataframe(means_df, width="stretch", hide_index=True)

                m1, m2 = st.columns(2)
                m1.metric("F-statistic (Welch)", f"{result.f_statistic:.2f}")
                m2.metric("p-value", f"{result.p_value:.4f}")
                st.caption(f"df between = {result.df_between:.1f}, df within = {result.df_within:.1f}")

                if result.small_groups_flagged:
                    caveat(
                        f"Small group(s) flagged (fewer than 5 observations): "
                        f"{', '.join(result.small_groups_flagged)}. Estimates for these "
                        "groups may be unstable."
                    )

                section_header("Pairwise Comparisons (Games-Howell)")
                pairwise_df = pd.DataFrame([
                    {
                        "Group A": p.group_a,
                        "Group B": p.group_b,
                        "Mean difference": round(p.mean_difference, 3),
                        "p (adjusted)": round(p.p_value, 4),
                        "Significant": "Yes" if p.significant else "No",
                    }
                    for p in result.pairwise_comparisons
                ])
                st.dataframe(pairwise_df, width="stretch", hide_index=True)
                st.caption("\"Significant\" means the adjusted p-value fell below the conventional threshold α = 0.05.")
                inspect_note("Which pairs are flagged Significant.")
                implications("Only flagged pairs support a claim of a group difference.")

                case_study_note(
                    "dead_salmon",
                    "The \"p (adjusted)\" column above is already corrected "
                    "for the number of pairs being compared. Reading those "
                    "adjusted values, rather than testing each pair "
                    "separately and reading each at 0.05, is what keeps the "
                    "number of comparisons from inflating the chance that "
                    "one of them looks significant by accident.",
                )

            elif method == "compare_multiple_groups":
                result = comp.compare_multiple_groups(df, context["group_col"], context["outcome_col"])

                section_header("Result")
                means_df = pd.DataFrame({
                    "Group": result.group_labels,
                    "n": [result.group_ns[g] for g in result.group_labels],
                    "Mean": [round(result.group_means[g], 3) for g in result.group_labels],
                })
                st.dataframe(means_df, width="stretch", hide_index=True)

                m1, m2 = st.columns(2)
                m1.metric("F-statistic", f"{result.f_statistic:.2f}")
                m2.metric("p-value", f"{result.p_value:.4f}")
                st.caption(f"df between = {result.df_between}, df within = {result.df_within}")

                if result.small_groups_flagged:
                    caveat(
                        f"Small group(s) flagged (fewer than 5 observations): "
                        f"{', '.join(result.small_groups_flagged)}. Estimates for these "
                        "groups may be unstable."
                    )

                section_header("Pairwise Comparisons (Tukey HSD)")
                pairwise_df = pd.DataFrame([
                    {
                        "Group A": p.group_a,
                        "Group B": p.group_b,
                        "Mean difference": round(p.mean_difference, 3),
                        "p (adjusted)": round(p.p_value, 4),
                        "Significant": "Yes" if p.significant else "No",
                    }
                    for p in result.pairwise_comparisons
                ])
                st.dataframe(pairwise_df, width="stretch", hide_index=True)
                st.caption("\"Significant\" means the adjusted p-value fell below the conventional threshold α = 0.05.")
                inspect_note("Which pairs are flagged Significant.")
                implications("Only flagged pairs support a claim of a group difference.")

                case_study_note(
                    "dead_salmon",
                    "The \"p (adjusted)\" column above is already corrected "
                    "for the number of pairs being compared. Reading those "
                    "adjusted values, rather than testing each pair "
                    "separately and reading each at 0.05, is what keeps the "
                    "number of comparisons from inflating the chance that "
                    "one of them looks significant by accident.",
                )

            elif method == "compare_categorical":
                result = comp.compare_categorical(df, context["group_col"], context["outcome_col"])

                section_header("Result")
                st.write("Contingency table (observed counts):")
                st.dataframe(result.contingency_table, width="stretch")

                m1, m2 = st.columns(2)
                m1.metric("Chi-square", f"{result.chi2_statistic:.2f}")
                m2.metric("p-value", f"{result.p_value:.4f}")
                st.caption(f"Degrees of freedom = {result.degrees_of_freedom}")

                if result.low_expected_frequency_warning:
                    caveat(
                        "One or more expected cell frequencies are below 5. The "
                        "chi-square approximation may be unreliable here; consider "
                        "Fisher's exact test for small samples (not yet implemented "
                        "in OpenMeasure)."
                    )

                inspect_note("The contingency table's cell counts.")
                implications("An association does not establish that the group caused the outcome.")

            elif method == "compare_pre_post":
                result = comp.compare_pre_post(df[context["pre_col"]], df[context["post_col"]])

                section_header("Result")
                c1, c2 = st.columns(2)
                c1.metric("Pre mean", f"{result.mean_pre:.2f}")
                c2.metric("Post mean", f"{result.mean_post:.2f}")

                m1, m2, m3 = st.columns(3)
                m1.metric("t-statistic", f"{result.t_statistic:.2f}")
                m2.metric("p-value", f"{result.p_value:.4f}")
                m3.metric("Cohen's d", f"{result.cohens_d:.2f}")
                st.caption(f"n = {result.n}, df = {result.degrees_of_freedom}, mean change = {result.mean_difference:.2f}")
                inspect_note("The p-value and Cohen's d, computed on the same participants' change over time.")
                implications(
                    "A significant change supports that something changed "
                    "for this group. A pre/post design alone does not rule "
                    "out other explanations, such as regression to the mean."
                )

                nonparametric = comp.compare_pre_post_nonparametric(
                    df[context["pre_col"]], df[context["post_col"]]
                )
                with st.expander("Compare with a rank-based test (Wilcoxon signed-rank)"):
                    st.markdown(
                        "The paired t-test above compares the mean change "
                        "and assumes those changes are roughly normally "
                        "distributed. The Wilcoxon signed-rank test compares "
                        "ranks of the changes instead, and makes no "
                        "distributional assumption. Choosing between them is "
                        "a choice, not a formality: it can change the "
                        "conclusion."
                    )
                    nc1, nc2 = st.columns(2)
                    nc1.metric("Pre median", f"{nonparametric.median_pre:.2f}")
                    nc2.metric("Post median", f"{nonparametric.median_post:.2f}")
                    nm1, nm2, nm3 = st.columns(3)
                    nm1.metric("W statistic", f"{nonparametric.w_statistic:.1f}")
                    nm2.metric("p-value", f"{nonparametric.p_value:.4f}")
                    nm3.metric(
                        "Matched-pairs r",
                        f"{nonparametric.matched_pairs_rank_biserial_correlation:.2f}",
                    )
                    if nonparametric.n_zero_differences_dropped:
                        st.caption(
                            f"{nonparametric.n_zero_differences_dropped} "
                            "participant(s) with no change are excluded from "
                            "ranking, per the Wilcoxon test's convention, "
                            "but are not treated as missing data."
                        )

                    if (result.p_value < 0.05) == (nonparametric.p_value < 0.05):
                        st.success(
                            "The paired t-test and Wilcoxon signed-rank test "
                            "fall on the same side of α=0.05 here."
                        )
                    else:
                        st.warning(
                            "The paired t-test and Wilcoxon signed-rank test "
                            "fall on opposite sides of α=0.05 here. The two "
                            "tests compare different quantities (mean versus "
                            "ranked change) and can disagree, particularly "
                            "with skewed data or outliers. Treat the choice "
                            "of test as consequential for this result, not a "
                            "formality."
                        )
                    inspect_note(
                        "Whether the two tests land on the same side of "
                        "α=0.05."
                    )

            elif method == "estimate_did":
                result = did_core.estimate_did(
                    df,
                    context["group_col"],
                    context["pre_col"],
                    context["post_col"],
                    treated_label=context["treated_label"],
                )
                render_did_result(result)

            elif method == "sensitivity_analysis":
                result = comp.sensitivity_analysis(
                    df,
                    context["group_col"],
                    context["outcome_col"],
                    delimiter=context.get("delimiter", ","),
                )

                section_header("Result: Sensitivity Across Coding Schemes")
                p_df = pd.DataFrame([
                    {"Coding scheme": name, "p-value": round(p, 4)}
                    for name, p in result.p_values_by_coding.items()
                ])
                st.dataframe(p_df, width="stretch", hide_index=True)
                inspect_note("Whether every row's p-value falls on the same side of α.")

                if result.consistent_conclusion:
                    st.success(
                        f"The conclusion (significant at α={result.alpha}, or not) "
                        "is consistent across all three coding schemes."
                    )
                else:
                    st.warning(
                        f"The conclusion at α={result.alpha} differs depending on "
                        "how multi-select responses are coded. Treat this result as "
                        "sensitive to an arbitrary coding choice, not as settled."
                    )

                case_study_note(
                    "narps",
                    "The rows above apply that idea to one specific choice, "
                    "how to count a participant who selected more than one "
                    "category. Agreement across the three codings says the "
                    "conclusion does not turn on that choice. Disagreement "
                    "says it does, and that reporting a single coding would "
                    "have hidden it.",
                )

                for name, sub_result in result.coding_results.items():
                    with st.expander(f"Full result: {name} coding"):
                        render_sensitivity_sub_result(sub_result)

            else:
                st.error(
                    "OpenMeasure could not run the recommended method. "
                    "This is unexpected; please report it as a bug."
                )

            if result is not None:
                record_comparison(df, loaded.name, context, recommendation, result)
                st.caption(
                    "Recorded for the Cross-Analysis Implications page, which "
                    "shows how much of your data each analysis used."
                )

                # ---------------------------------------------------------
                # 7. Interpret
                # ---------------------------------------------------------
                #
                # Marked reached rather than advanced to: this stage opens
                # because an analysis just produced a result, and that
                # result exists only in this pass. A rerun would discard
                # the very thing that unlocked it.
                TRACKER.mark_reached(STAGE_INTERPRET)

                section_header(
                    "7. Interpret",
                    "What this design and this result together support",
                )

                if method == "estimate_did":
                    render_did_interpretation(result)
                else:
                    st.markdown("**What the design leaves open**")
                    for warning in recommendation.warnings:
                        st.write(f"- {warning}")
                    st.caption(
                        "These are properties of the design, not of the "
                        "numbers. A smaller p-value does not retire any of "
                        "them."
                    )

        except (ValueError, TypeError) as e:
            st.error(str(e))
