"""
OpenMeasure — Program Evaluation module.

Compares outcomes across groups or across time and recommends a method
based on the reported study design and data structure. Users can review
the reasoning before running the analysis.

Core statistical logic lives in modules/program_evaluation/core. This
file handles presentation only.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure imports work when Streamlit runs this page directly.
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from modules.program_evaluation.core import comparison as comp
from modules.program_evaluation.core import recommend as rec
from shared.report import (
    caveat,
    section_header,
    show_case_studies,
)


st.set_page_config(
    page_title="OpenMeasure · Program Evaluation",
    layout="centered",
)

st.title("Program Evaluation")
st.caption(
    "Compare outcomes across groups or across time, and review which "
    "method fits your data before running the analysis."
)

st.divider()


# ---------------------------------------------------------------------
# Method overview
# ---------------------------------------------------------------------

with st.expander("How method selection works"):
    st.markdown(
        """
OpenMeasure recommends a method based on the study design and data
structure you report. Because software cannot fully determine a
variable's meaning from its data type alone, you should review the
recommendation before running the analysis.

**Current decision rules**

- **Two independent groups with a continuous-like outcome:** Welch's
  independent-samples t-test
- **Three or more independent groups with a continuous-like outcome:**
  Welch's one-way ANOVA with Games-Howell comparisons
- **Independent groups with a categorical outcome:** chi-square test of
  independence
- **A multiselect group field:** sensitivity analysis using three coding
  approaches
- **The same participants measured before and after:** paired t-test

These methods do not establish causation on their own. Without random
assignment or a strong comparison design, an observed difference may
reflect selection, baseline imbalance, external events, or unmeasured
confounding rather than the program itself.
"""
    )


# ---------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------

section_header(
    "1. Upload your data",
    "CSV file, one row per participant",
)

uploaded = st.file_uploader(
    "CSV file",
    type="csv",
    label_visibility="collapsed",
)


# ---------------------------------------------------------------------
# Analysis workflow
# ---------------------------------------------------------------------

if uploaded is None:
    st.info("Upload a CSV to begin an analysis.")

else:
    try:
        df = pd.read_csv(uploaded)
    except Exception as error:
        st.error(f"The CSV could not be read: {error}")
        df = None

    if df is not None:
        st.write(
            f"Loaded **{df.shape[0]} rows** and "
            f"**{df.shape[1]} columns**."
        )

        st.dataframe(
            df.head(),
            use_container_width=True,
            hide_index=True,
        )

        section_header("2. Describe your comparison")

        design = st.radio(
            "Which study design best describes your data?",
            options=[
                "Two or more independent groups",
                "Pre/post measurements from the same participants",
            ],
            index=0,
        )

        recommendation = None
        context: dict[str, object] = {}

        # -------------------------------------------------------------
        # Independent-group design
        # -------------------------------------------------------------

        if design == "Two or more independent groups":
            outcome_col = st.selectbox(
                "Outcome column",
                options=list(df.columns),
            )

            available_group_columns = [
                column
                for column in df.columns
                if column != outcome_col
            ]

            group_col = st.selectbox(
                "Group column",
                options=available_group_columns,
            )

            is_multiselect = st.checkbox(
                "This group column allows multiple selections per "
                "participant, such as a comma-separated demographic field.",
                value=False,
            )

            delimiter = ","

            if is_multiselect:
                delimiter = st.text_input(
                    "Delimiter used between selections",
                    value=",",
                )

            context = {
                "outcome_col": outcome_col,
                "group_col": group_col,
                "is_multiselect": is_multiselect,
                "delimiter": delimiter,
            }

            if st.button(
                "Get recommendation",
                type="primary",
                key="group_recommendation",
            ):
                try:
                    recommendation = rec.recommend_method(
                        df,
                        outcome_col=outcome_col,
                        group_col=group_col,
                        is_multiselect_group=is_multiselect,
                    )
                except (ValueError, TypeError) as error:
                    st.error(str(error))

        # -------------------------------------------------------------
        # Paired pre/post design
        # -------------------------------------------------------------

        else:
            pre_col = st.selectbox(
                "Pre or baseline column",
                options=list(df.columns),
            )

            available_post_columns = [
                column
                for column in df.columns
                if column != pre_col
            ]

            post_col = st.selectbox(
                "Post or follow-up column",
                options=available_post_columns,
            )

            context = {
                "pre_col": pre_col,
                "post_col": post_col,
            }

            if st.button(
                "Get recommendation",
                type="primary",
                key="pre_post_recommendation",
            ):
                try:
                    recommendation = rec.recommend_method(
                        df,
                        pre_col=pre_col,
                        post_col=post_col,
                    )
                except (ValueError, TypeError) as error:
                    st.error(str(error))

        # Store a successful recommendation so it remains visible after
        # Streamlit reruns the page.
        if recommendation is not None:
            st.session_state["pe_recommendation"] = recommendation
            st.session_state["pe_context"] = context

        # -------------------------------------------------------------
        # Recommendation
        # -------------------------------------------------------------

        if "pe_recommendation" in st.session_state:
            recommendation = st.session_state["pe_recommendation"]
            context = st.session_state["pe_context"]

            section_header("Recommendation")

            st.markdown(f"**{recommendation.display_name}**")

            with st.expander(
                "Why this method?",
                expanded=True,
            ):
                for item in recommendation.reasoning:
                    st.markdown(f"- {item}")

            if recommendation.assumptions:
                with st.expander("Assumptions"):
                    for item in recommendation.assumptions:
                        st.markdown(f"- {item}")

            if recommendation.tradeoffs:
                with st.expander("Tradeoffs"):
                    for item in recommendation.tradeoffs:
                        st.markdown(f"- {item}")

            if recommendation.alternatives:
                with st.expander("Alternative methods"):
                    for item in recommendation.alternatives:
                        st.markdown(f"- {item}")

            for warning in recommendation.warnings:
                st.warning(warning)

            # ---------------------------------------------------------
            # Run supported method
            # ---------------------------------------------------------

            if not recommendation.supported:
                st.error(
                    "This method is recommended for the selected design "
                    "but is not yet implemented in OpenMeasure."
                )

            else:
                run_clicked = st.button(
                    "Run recommended analysis",
                    type="primary",
                )

                if run_clicked:
                    method = recommendation.method

                    try:
                        # ---------------------------------------------
                        # Two independent groups
                        # ---------------------------------------------

                        if method == "compare_two_groups":
                            result = comp.compare_two_groups(
                                df,
                                context["group_col"],
                                context["outcome_col"],
                            )

                            section_header("Result")

                            group_a_column, group_b_column = st.columns(2)

                            group_a_column.metric(
                                f"{result.group_a_label} mean",
                                f"{result.mean_a:.2f}",
                                f"n = {result.n_a}",
                            )

                            group_b_column.metric(
                                f"{result.group_b_label} mean",
                                f"{result.mean_b:.2f}",
                                f"n = {result.n_b}",
                            )

                            metric_1, metric_2, metric_3 = st.columns(3)

                            metric_1.metric(
                                "t-statistic",
                                f"{result.t_statistic:.2f}",
                            )

                            metric_2.metric(
                                "p-value",
                                f"{result.p_value:.4f}",
                            )

                            metric_3.metric(
                                "Cohen's d",
                                f"{result.cohens_d:.2f}",
                            )

                            st.caption(
                                f"Mean difference: "
                                f"{result.mean_difference:.2f} "
                                f"(95% CI: {result.ci_95_low:.2f} to "
                                f"{result.ci_95_high:.2f}); "
                                f"df = {result.degrees_of_freedom:.1f}"
                            )

                        # ---------------------------------------------
                        # Three or more independent groups
                        # ---------------------------------------------

                        elif method in {
                            "compare_multiple_groups",
                            "compare_multiple_groups_welch",
                        }:
                            if method == "compare_multiple_groups_welch":
                                caveat(
                                    "Welch's ANOVA with Games-Howell "
                                    "comparisons is recommended but is not "
                                    "yet implemented. The current analysis "
                                    "uses standard one-way ANOVA with Tukey "
                                    "HSD, which assumes equal variances."
                                )

                            result = comp.compare_multiple_groups(
                                df,
                                context["group_col"],
                                context["outcome_col"],
                            )

                            section_header("Result")

                            means_df = pd.DataFrame(
                                {
                                    "Group": result.group_labels,
                                    "n": [
                                        result.group_ns[group]
                                        for group in result.group_labels
                                    ],
                                    "Mean": [
                                        round(
                                            result.group_means[group],
                                            3,
                                        )
                                        for group in result.group_labels
                                    ],
                                }
                            )

                            st.dataframe(
                                means_df,
                                use_container_width=True,
                                hide_index=True,
                            )

                            metric_1, metric_2 = st.columns(2)

                            metric_1.metric(
                                "F-statistic",
                                f"{result.f_statistic:.2f}",
                            )

                            metric_2.metric(
                                "p-value",
                                f"{result.p_value:.4f}",
                            )

                            st.caption(
                                f"df between = {result.df_between}; "
                                f"df within = {result.df_within}"
                            )

                            if result.small_groups_flagged:
                                caveat(
                                    "Small groups were flagged because "
                                    "they contain fewer than five "
                                    "observations: "
                                    f"{', '.join(result.small_groups_flagged)}. "
                                    "Estimates may be unstable."
                                )

                            section_header(
                                "Pairwise comparisons",
                                "Tukey HSD adjusted comparisons",
                            )

                            pairwise_df = pd.DataFrame(
                                [
                                    {
                                        "Group A": pair.group_a,
                                        "Group B": pair.group_b,
                                        "Mean difference": round(
                                            pair.mean_difference,
                                            3,
                                        ),
                                        "Adjusted p-value": round(
                                            pair.p_value,
                                            4,
                                        ),
                                        "Significant": (
                                            "Yes"
                                            if pair.significant
                                            else "No"
                                        ),
                                    }
                                    for pair
                                    in result.pairwise_comparisons
                                ]
                            )

                            st.dataframe(
                                pairwise_df,
                                use_container_width=True,
                                hide_index=True,
                            )

                        # ---------------------------------------------
                        # Categorical outcome
                        # ---------------------------------------------

                        elif method == "compare_categorical":
                            result = comp.compare_categorical(
                                df,
                                context["group_col"],
                                context["outcome_col"],
                            )

                            section_header("Result")

                            st.write(
                                "Contingency table of observed counts"
                            )

                            st.dataframe(
                                result.contingency_table,
                                use_container_width=True,
                            )

                            metric_1, metric_2 = st.columns(2)

                            metric_1.metric(
                                "Chi-square",
                                f"{result.chi2_statistic:.2f}",
                            )

                            metric_2.metric(
                                "p-value",
                                f"{result.p_value:.4f}",
                            )

                            st.caption(
                                "Degrees of freedom = "
                                f"{result.degrees_of_freedom}"
                            )

                            if result.low_expected_frequency_warning:
                                caveat(
                                    "One or more expected cell counts are "
                                    "below five. The chi-square "
                                    "approximation may be unreliable. "
                                    "Fisher's exact test or another exact "
                                    "method may be more appropriate."
                                )

                        # ---------------------------------------------
                        # Paired pre/post
                        # ---------------------------------------------

                        elif method == "compare_pre_post":
                            result = comp.compare_pre_post(
                                df[context["pre_col"]],
                                df[context["post_col"]],
                            )

                            section_header("Result")

                            pre_column, post_column = st.columns(2)

                            pre_column.metric(
                                "Pre mean",
                                f"{result.mean_pre:.2f}",
                            )

                            post_column.metric(
                                "Post mean",
                                f"{result.mean_post:.2f}",
                            )

                            metric_1, metric_2, metric_3 = st.columns(3)

                            metric_1.metric(
                                "t-statistic",
                                f"{result.t_statistic:.2f}",
                            )

                            metric_2.metric(
                                "p-value",
                                f"{result.p_value:.4f}",
                            )

                            metric_3.metric(
                                "Cohen's d",
                                f"{result.cohens_d:.2f}",
                            )

                            st.caption(
                                f"n = {result.n}; "
                                f"df = {result.degrees_of_freedom}; "
                                f"mean change = "
                                f"{result.mean_difference:.2f}"
                            )

                        # ---------------------------------------------
                        # Multiselect sensitivity analysis
                        # ---------------------------------------------

                        elif method == "sensitivity_analysis":
                            result = comp.sensitivity_analysis(
                                df,
                                context["group_col"],
                                context["outcome_col"],
                                delimiter=context.get(
                                    "delimiter",
                                    ",",
                                ),
                            )

                            section_header(
                                "Sensitivity across coding schemes"
                            )

                            p_value_df = pd.DataFrame(
                                [
                                    {
                                        "Coding scheme": name,
                                        "p-value": round(p_value, 4),
                                    }
                                    for name, p_value
                                    in result.p_values_by_coding.items()
                                ]
                            )

                            st.dataframe(
                                p_value_df,
                                use_container_width=True,
                                hide_index=True,
                            )

                            if result.consistent_conclusion:
                                st.success(
                                    "The significance conclusion at "
                                    f"α = {result.alpha} is consistent "
                                    "across all three coding schemes."
                                )
                            else:
                                st.warning(
                                    "The conclusion at "
                                    f"α = {result.alpha} changes depending "
                                    "on how multiselect responses are "
                                    "coded. The finding is sensitive to "
                                    "this analytic choice."
                                )

                            for name, sub_result in (
                                result.coding_results.items()
                            ):
                                with st.expander(
                                    f"Full result: {name} coding"
                                ):
                                    st.write(sub_result)

                        else:
                            st.error(
                                f"Unknown method '{method}'."
                            )

                    except (ValueError, TypeError, ImportError) as error:
                        st.error(str(error))


# ---------------------------------------------------------------------
# Always-visible research examples
# ---------------------------------------------------------------------

section_header(
    "Research examples",
    "Published examples illustrating why methodological assumptions, "
    "tradeoffs, and analytic choices matter",
)

show_case_studies("program_validation")
