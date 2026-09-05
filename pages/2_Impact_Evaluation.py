"""
OpenMeasure - Program Evaluation module.

Compares outcomes across groups or across time (pre/post), auto-
recommending the appropriate test based on data shape, then lets the
user confirm or override before running it. Core logic lives in
modules/program_evaluation/core, this file is presentation only.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from modules.program_evaluation.core import comparison as comp
from modules.program_evaluation.core import did as did_core
from modules.program_evaluation.core import interpret
from modules.program_evaluation.core import recommend as rec
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
from shared.report import (
    section_header,
    case_study_note,
    caveat,
    flagged_item_note,
    implications,
    inspect_note,
    interpretation_note,
    show_case_studies,
    render_lifecycle_tracker,
)
from shared.upload import render_data_profile


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
         aria-label="Kept-appointment rate at two clinics before and after
         the reminder program, with the treated clinic's assumed path
         without it.">
      <line x1="8" y1="10" x2="24" y2="10" stroke="{ACCENT_2}" stroke-width="2"/>
      <text x="28" y="13" font-size="9" fill="{INK_MUTED}">{scenario.treated_label}</text>
      <line x1="90" y1="10" x2="106" y2="10" stroke="{ACCENT}" stroke-width="2"/>
      <text x="110" y="13" font-size="9" fill="{INK_MUTED}">{scenario.comparison_label}</text>
      <line x1="180" y1="10" x2="196" y2="10" stroke="{INK_MUTED}" stroke-width="2"
            stroke-dasharray="4,3"/>
      <text x="200" y="13" font-size="9" fill="{INK_MUTED}">
        {scenario.treated_label} without the program
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

      <text x="{_TEACH_X_PRE}" y="165" font-size="9" fill="{INK_MUTED}"
            text-anchor="middle">December</text>
      <text x="{_TEACH_X_POST}" y="165" font-size="9" fill="{INK_MUTED}"
            text-anchor="middle">June</text>
      <text x="8" y="178" font-size="8" fill="{INK_MUTED}">
        Percentage of scheduled appointments kept
      </text>
    </svg>
    """


def render_did_teaching_example() -> None:
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
    """
    scenario = teaching.DID_TEACHING_SCENARIO

    with st.expander("See how a comparison group changes the estimate"):
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
            f"How much {scenario.comparison_label}'s kept-appointment rate "
            "changed over the same six months (percentage points)",
            min_value=-10.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
        )
        outcome = teaching.teaching_did(comparison_change)

        st.markdown(
            _did_teaching_svg(scenario, outcome),
            unsafe_allow_html=True,
        )
        st.caption(
            "The dashed line is the comparison clinic's slope drawn from "
            "the treated clinic's own starting point. Parallel trends is "
            "the assumption that the dashed line is where Clinic A would "
            "have ended up, and the bracket is what is left over."
        )

        st.dataframe(
            pd.DataFrame(
                {
                    "Group": [scenario.treated_label, scenario.comparison_label],
                    "December": [scenario.pre_treated, scenario.pre_comparison],
                    "June": [scenario.post_treated, outcome.post_comparison],
                    "Change": [outcome.change_treated, outcome.comparison_change],
                }
            ),
            width="stretch",
            hide_index=True,
        )

        t1, t2 = st.columns(2)
        t1.metric(
            "Before-and-after estimate",
            f"{outcome.before_after_estimate:+.1f} points",
        )
        t2.metric(
            "Difference-in-differences estimate",
            f"{outcome.did_estimate:+.1f} points",
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

    Assumptions render as a visible section rather than inside an
    expander. Parallel trends is the condition the causal reading rests
    on and cannot be tested with two time points, so it is not something
    a reader should have to open something to find.
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

    inspect_note(
        "The two Change values. The estimate is the gap between them, so a "
        "comparison group that moved on its own is doing as much work here "
        "as the treated group's own change."
    )

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


def record_comparison(frame, upload, analysis_context, recommendation, result) -> None:
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
        fingerprint=fingerprint_dataframe(frame, upload.name),
        exclusion=account,
        primary_statistics=statistics,
    )

st.set_page_config(page_title="OpenMeasure · Program Evaluation", page_icon=":material/monitoring:", layout="centered")

st.title("Impact Evaluation")
st.subheader("Program Validation")
st.caption("Compare outcomes across groups, or across time, and see which test fits your data before running it.")

st.divider()

render_lifecycle_tracker(current_workflow="Impact Evaluation")

render_data_handling_summary(disclosure_for("pages/2_Impact_Evaluation.py"))

show_case_studies("program_validation")

st.divider()

with st.expander("Method Selection"):
    st.markdown(
        """
This module looks at the shape of your data, how many groups you're
comparing, whether the outcome is continuous or categorical, whether
you're comparing groups or comparing the same people before and after,
and recommends a test. You can always override the recommendation.

**Design decisions this module makes:**

- **2 groups, continuous outcome** → Welch's t-test, which does not
  assume equal variances between groups.
- **3+ groups, continuous outcome** → Welch's one-way ANOVA with
  Games-Howell post-hoc comparisons, which also does not assume equal
  variances across groups.
- **Categorical outcome** → chi-square test of independence.
- **Multi-select group field** (e.g. participants who could select more
  than one demographic category) → a sensitivity analysis comparing
  three different ways of coding those selections, rather than picking
  one arbitrarily.
- **Pre/post, same participants** → paired t-test.
- **Two groups, each measured before and after** →
  difference-in-differences, which subtracts the comparison group's
  change from the treated group's. Inference comes from a Welch
  comparison of the two groups' change scores.

None of these establish causation on their own. If group membership
wasn't randomized, a difference may reflect selection or pre-existing
differences between groups rather than a program effect. This caveat is
shown with two-group and multi-group continuous-outcome comparisons.

Difference-in-differences removes anything that moved both groups
equally, and any fixed gap between them that was already there at
baseline. It buys that with an assumption instead: that the treated
group would have followed the comparison group's path. Two time points
give no way to check it, so the module states the assumption alongside
the estimate rather than treating a well-shaped dataset as evidence the
assumption holds.
"""
    )

section_header("1. Upload Your Data", "CSV file, one row per participant")

uploaded = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")

if uploaded is None:
    st.info("Upload a CSV to get started, or try the sample dataset below.")
    sample_path = ROOT / "modules" / "program_evaluation" / "sample_data" / "program_eval_example.csv"
    with open(sample_path, "rb") as f:
        st.download_button(
            "Download sample_data/program_eval_example.csv",
            f,
            file_name="program_eval_example.csv",
        )
    st.stop()

df = pd.read_csv(uploaded)
render_data_profile(df)

# Discard a recommendation carried over from a different upload. Without
# this, uploading a second file whose column names happen to match the first
# would analyze the new data under the previous file's plan, silently.
if st.session_state.get("pe_uploaded_file_id") != uploaded.file_id:
    st.session_state["pe_uploaded_file_id"] = uploaded.file_id
    st.session_state.pop("pe_recommendation", None)
    st.session_state.pop("pe_context", None)

st.write(f"Loaded **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
st.dataframe(df.head(), width="stretch")

section_header("2. Describe Your Comparison")

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
    outcome_col = st.selectbox("Outcome column", options=list(df.columns))
    group_col = st.selectbox(
        "Group column",
        options=[c for c in df.columns if c != outcome_col],
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

    case_study_note(
        "lalonde_1986",
        "The test this recommends compares the groups as they are, and "
        "has no way to see how anyone ended up in one rather than the "
        "other. If group membership was not randomly assigned, whatever "
        "distinguished the groups beforehand is carried along in the "
        "difference it reports, and the confidence interval around that "
        "difference will look no wider for it.",
    )

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
    pre_col = st.selectbox("Pre (baseline) column", options=list(df.columns))
    post_col = st.selectbox(
        "Post (follow-up) column",
        options=[c for c in df.columns if c != pre_col],
    )
    context = {"pre_col": pre_col, "post_col": post_col}

    case_study_note(
        "scared_straight",
        "The paired test this recommends measures how much one group "
        "changed between two measurements. Nothing in it observes what "
        "would have happened without the program, so maturation, "
        "regression to the mean, and outside events stay open as "
        "explanations for the change. The difference-in-differences "
        "option adds a comparison group to stand in for that, in exchange "
        "for an assumption about the group it adds.",
    )

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
        "the same two time points in two more columns. The sample dataset "
        "above has this shape already, in event_format with pre_confidence "
        "and post_confidence."
    )

    group_col = st.selectbox(
        "Group column (which units were treated)",
        options=list(df.columns),
    )
    remaining = [c for c in df.columns if c != group_col]
    pre_col = st.selectbox("Pre (baseline) column", options=remaining)
    post_col = st.selectbox(
        "Post (follow-up) column",
        options=[c for c in remaining if c != pre_col],
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

    render_did_teaching_example()

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
                inspect_note("The p-value against your significance threshold, and Cohen's d for effect size.")
                implications(
                    "A p-value below threshold supports attributing the "
                    "difference to what distinguishes the groups, subject "
                    "to Welch's t-test assumptions. Above threshold, no "
                    "difference was detected at this sample size."
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
                record_comparison(df, uploaded, context, recommendation, result)
                st.caption(
                    "Recorded for the Cross-Analysis Implications page, which "
                    "shows how much of your data each analysis used."
                )

        except (ValueError, TypeError) as e:
            st.error(str(e))
