"""
OpenMeasure — Program Evaluation module.

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
from modules.program_evaluation.core import recommend as rec
from shared.report import section_header, caveat, flagged_item_note, show_case_studies

st.set_page_config(page_title="OpenMeasure · Program Evaluation", page_icon=":bar_chart:", layout="centered")

st.title("Program Validation: Evaluation")
st.caption("Compare outcomes across groups, or across time, and see which test fits your data before running it.")

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
- **3+ groups, continuous outcome** → one-way ANOVA with Tukey HSD
  post-hoc comparisons.
- **Categorical outcome** → chi-square test of independence.
- **Multi-select group field** (e.g. participants who could select more
  than one demographic category) → a sensitivity analysis comparing
  three different ways of coding those selections, rather than picking
  one arbitrarily.
- **Pre/post, same participants** → paired t-test.

None of these tests establish causation on their own. If group
membership wasn't randomized, a difference may reflect selection or
pre-existing differences between groups rather than a program effect.
This caveat is shown with every group-comparison result.
"""
    )

section_header("1. Upload your data", "CSV file, one row per participant")

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
st.write(f"Loaded **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
st.dataframe(df.head(), use_container_width=True)

section_header("2. Describe your comparison")

design = st.radio(
    "What are you comparing?",
    options=["Two or more groups", "Pre/post (same participants)"],
    index=0,
)

recommendation = None
context = {}

if design == "Two or more groups":
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

else:
    pre_col = st.selectbox("Pre (baseline) column", options=list(df.columns))
    post_col = st.selectbox(
        "Post (follow-up) column",
        options=[c for c in df.columns if c != pre_col],
    )
    context = {"pre_col": pre_col, "post_col": post_col}

    if st.button("Get recommendation", type="primary"):
        try:
            recommendation = rec.recommend_method(df, pre_col=pre_col, post_col=post_col)
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
                st.dataframe(means_df, use_container_width=True, hide_index=True)

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

                section_header("Pairwise comparisons (Games-Howell)")
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
                st.dataframe(pairwise_df, use_container_width=True, hide_index=True)

            elif method == "compare_multiple_groups":
                result = comp.compare_multiple_groups(df, context["group_col"], context["outcome_col"])

                section_header("Result")
                means_df = pd.DataFrame({
                    "Group": result.group_labels,
                    "n": [result.group_ns[g] for g in result.group_labels],
                    "Mean": [round(result.group_means[g], 3) for g in result.group_labels],
                })
                st.dataframe(means_df, use_container_width=True, hide_index=True)

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

                section_header("Pairwise comparisons (Tukey HSD)")
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
                st.dataframe(pairwise_df, use_container_width=True, hide_index=True)

            elif method == "compare_categorical":
                result = comp.compare_categorical(df, context["group_col"], context["outcome_col"])

                section_header("Result")
                st.write("Contingency table (observed counts):")
                st.dataframe(result.contingency_table, use_container_width=True)

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

            elif method == "sensitivity_analysis":
                result = comp.sensitivity_analysis(
                    df,
                    context["group_col"],
                    context["outcome_col"],
                    delimiter=context.get("delimiter", ","),
                )

                section_header("Result: sensitivity across coding schemes")
                p_df = pd.DataFrame([
                    {"Coding scheme": name, "p-value": round(p, 4)}
                    for name, p in result.p_values_by_coding.items()
                ])
                st.dataframe(p_df, use_container_width=True, hide_index=True)

                if result.consistent_conclusion:
                    st.success(
                        f"The conclusion (significant at α={result.alpha}, or not) "
                        "is consistent across all three coding schemes."
                    )
                else:
                    st.warning(
                        f"The conclusion at α={result.alpha} DIFFERS depending on "
                        "how multi-select responses are coded. Treat this result as "
                        "sensitive to an arbitrary coding choice, not as settled."
                    )

                for name, sub_result in result.coding_results.items():
                    with st.expander(f"Full result: {name} coding"):
                        st.write(sub_result)

            else:
                st.error(f"Unknown method '{method}'.")

        except (ValueError, TypeError) as e:
            st.error(str(e))
