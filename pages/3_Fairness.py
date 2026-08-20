"""
OpenMeasure - Fairness module.

Guides users toward fairness metrics based on the evaluation goal and
examines whether favorable labels are distributed differently across
groups before a model is trained.

Core calculations live in modules/fairness/core. This file handles
presentation only.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from the repository root regardless of where Streamlit
# is launched.
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from modules.fairness.core import pre_model_metrics as pm
from modules.fairness.core.recommend import recommend_fairness_metric
from shared.catalog import MODULE_FAIRNESS
from shared.handoff import (
    KIND_ROWS_DROPPED,
    ExclusionAccount,
    HandoffStore,
    RetentionItem,
    fingerprint_dataframe,
)
from shared.data_handling import disclosure_for, render_data_handling_summary
from shared.report import (
    caveat,
    flagged_item_note,
    render_lifecycle_tracker,
    section_header,
    show_case_studies,
)


def record_fairness(frame, upload, label_column, group_column, result) -> None:
    """
    Record this analysis for the Cross-Analysis Implications page.

    Translates into primitives here rather than storing the result object,
    because the same dataclass is a different class depending on how it was
    imported.
    """
    HandoffStore(st.session_state).record(
        module=MODULE_FAIRNESS,
        fingerprint=fingerprint_dataframe(frame, upload.name),
        exclusion=ExclusionAccount(
            module=MODULE_FAIRNESS,
            analysis_label="Fairness (pre-model)",
            columns_considered=(str(label_column), str(group_column)),
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
        ),
        primary_statistics={
            "disparate_impact": float(result.disparate_impact),
            "statistical_parity_difference": float(
                result.statistical_parity_difference
            ),
        },
    )


st.set_page_config(
    page_title="OpenMeasure · Fairness",
    layout="centered",
)

st.title("Fairness Auditing")
st.subheader("Model Validation")
st.caption(
    "Evaluate group fairness, understand the assumptions behind common "
    "fairness metrics, and review tradeoffs between competing definitions."
)

st.divider()

render_lifecycle_tracker(current_workflow="Fairness")

render_data_handling_summary(disclosure_for("pages/3_Fairness.py"))


# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------

with st.expander("What does fairness evaluation mean?"):
    st.markdown(
        """
Fairness evaluation examines whether data, predictions, errors, or model
scores behave differently across groups.

There is no single fairness metric that is appropriate for every
application. Different metrics protect against different harms.

### Common fairness goals

- **Demographic parity** asks whether groups receive favorable decisions
  at similar rates.
- **Equal opportunity** asks whether people who genuinely qualify or need
  help are identified at similar rates across groups.
- **Predictive equality** asks whether people are wrongly flagged at
  similar rates across groups.
- **Equalized odds** asks whether both true-positive and false-positive
  rates are similar across groups.
- **Calibration within groups** asks whether the same predicted score has
  the same observed meaning across groups.

These goals can conflict, especially when outcome base rates differ
between groups and predictions are imperfect. Metric selection should
therefore be treated as a methodological and ethical decision rather
than a purely technical choice.
"""
    )


with st.expander("Current scope and limitations"):
    st.markdown(
        """
### Current scope

The current module performs **pre-model label-distribution analysis**.
It evaluates how frequently a favorable observed label occurs within
different groups.

Available pre-model metrics include:

- Favorable-label rate by group
- Disparate impact
- Statistical parity difference
- Small-group warnings

### Important limitation

A difference in favorable-label rates does not, by itself, establish
that the data or decision process is unfair.

Observed disparities may reflect:

- historical inequity;
- differences in access or measurement;
- sampling or selection effects;
- label bias;
- meaningful differences in the target population; or
- data-quality problems.

Equal opportunity, predictive equality, equalized odds, and calibration
require model predictions, and for most metrics they also require observed
ground-truth outcomes. Those analyses belong to a later post-model stage.
"""
    )


# ---------------------------------------------------------------------
# Fairness-goal recommendation
# ---------------------------------------------------------------------

section_header(
    "1. Identify the fairness goal",
    "Start with the harm or protection that matters in the application",
)

GOAL_LABELS = {
    "opportunity_access": (
        "No group should be excluded from favorable opportunities"
    ),
    "avoid_missed_need": (
        "People who qualify or need help should not be missed"
    ),
    "avoid_wrong_flags": (
        "People should not be wrongly flagged at different rates"
    ),
    "comparable_risk_scores": (
        "The same risk score should have the same meaning across groups"
    ),
}

selected_goal = st.radio(
    "Which concern is most important in this context?",
    options=list(GOAL_LABELS),
    format_func=lambda key: GOAL_LABELS[key],
)

recommendation = recommend_fairness_metric(selected_goal)

st.markdown(f"**Recommended metric: {recommendation.display_name}**")

with st.expander("Why this metric?", expanded=True):
    for item in recommendation.reasoning:
        st.markdown(f"- {item}")

with st.expander("Assumptions"):
    for item in recommendation.assumptions:
        st.markdown(f"- {item}")

with st.expander("Tradeoffs"):
    for item in recommendation.tradeoffs:
        st.markdown(f"- {item}")

with st.expander("Alternative metrics"):
    for item in recommendation.alternatives:
        st.markdown(f"- {item}")

with st.expander("Applicable domains or contexts", icon=":material/category:"):
    st.caption(
        "Illustrative and non-exhaustive. A domain appearing here shows "
        "where this goal can come up, not that the domain automatically "
        "requires this specific fairness metric - the right choice still "
        "depends on the decision being evaluated."
    )
    for context in recommendation.applicable_domains:
        st.markdown(f"**{context.domain}**: {context.relevance}")

if recommendation.metric != "demographic_parity":
    st.info(
        f"{recommendation.display_name} requires model predictions and is "
        "not calculated by the current pre-model analysis. The analysis "
        "below can still examine favorable-label rates already present "
        "in the data."
    )


# ---------------------------------------------------------------------
# Upload and sample data
# ---------------------------------------------------------------------

section_header(
    "2. Upload your data",
    "CSV file, one row per participant, observation, or prediction",
)

uploaded = st.file_uploader(
    "CSV file",
    type="csv",
    label_visibility="collapsed",
)

if uploaded is None:
    st.info(
        "Upload a CSV to begin an analysis, or download the sample dataset."
    )

    with st.expander("About the sample dataset"):
        st.markdown(
            """
The sample dataset represents a **simulated binary classification
problem**. Each row contains one observed outcome and one model
prediction.

| Column | Description |
|---|---|
| `true_label` | The observed or ground-truth outcome |
| `predicted_label` | The model's binary decision after applying a threshold |
| `predicted_probability` | The model's estimated probability or risk score |
| `sex` | An example group attribute used for fairness comparisons |

These columns can support several fairness analyses:

- **Demographic parity** compares favorable prediction rates across groups.
- **Equal opportunity** compares true-positive rates across groups.
- **Predictive equality** compares false-positive rates across groups.
- **Equalized odds** compares both true-positive and false-positive rates.
- **Calibration within groups** evaluates whether predicted probabilities
  have the same meaning across groups.

The current OpenMeasure analysis focuses on favorable-label rates.
Later versions can use the prediction and probability columns for
post-model fairness and calibration analyses.
"""
        )

    sample_path = (
        ROOT
        / "modules"
        / "fairness"
        / "sample_data"
        / "fairness_example.csv"
    )

    if sample_path.exists():
        with sample_path.open("rb") as sample_file:
            st.download_button(
                label="Download sample fairness dataset",
                data=sample_file,
                file_name=sample_path.name,
                mime="text/csv",
            )
    else:
        st.warning(
            "Sample fairness dataset could not be found. Add a CSV under "
            "modules/fairness/sample_data."
        )

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
            width="stretch",
            hide_index=True,
        )

        # -------------------------------------------------------------
        # Configure analysis
        # -------------------------------------------------------------

        section_header("3. Configure the pre-model analysis")

        label_col = st.selectbox(
            "Observed label or outcome column",
            options=list(df.columns),
            help=(
                "Select the observed binary outcome. Do not select a "
                "participant identifier or continuous probability column."
            ),
        )

        possible_group_columns = [
            column
            for column in df.columns
            if column != label_col
        ]

        group_col = st.selectbox(
            "Group column",
            options=possible_group_columns,
            help=(
                "Select a demographic, clinical, geographic, or other "
                "grouping variable relevant to the evaluation."
            ),
        )

        label_values = (
            df[label_col]
            .dropna()
            .unique()
            .tolist()
        )

        label_values = sorted(
            label_values,
            key=str,
        )

        if len(label_values) != 2:
            st.warning(
                f"The selected label contains {len(label_values)} distinct "
                "nonmissing values. The current module requires a binary "
                "label."
            )

        favorable_label = st.selectbox(
            "Which label represents the favorable outcome?",
            options=label_values,
            help=(
                "Examples include receiving an opportunity, being approved, "
                "recovering, or having a positive clinical outcome. In some "
                "contexts, the numerically lower value may be favorable."
            ),
        )

        group_values = (
            df[group_col]
            .dropna()
            .unique()
            .tolist()
        )

        group_values = sorted(
            group_values,
            key=str,
        )

        privileged_group = st.selectbox(
            "Reference or privileged group",
            options=group_values,
            help=(
                "This group is used as the reference in the denominator of "
                "disparate impact and as the reference for statistical "
                "parity difference."
            ),
        )

        available_comparison_groups = [
            group
            for group in group_values
            if group != privileged_group
        ]

        unprivileged_group = st.selectbox(
            "Comparison or unprivileged group",
            options=available_comparison_groups,
        )

        analyze_clicked = st.button(
            "Analyze pre-model group differences",
            type="primary",
            disabled=(
                len(label_values) != 2
                or len(group_values) < 2
            ),
        )

        # -------------------------------------------------------------
        # Results
        # -------------------------------------------------------------

        if analyze_clicked:
            try:
                bias_result = pm.compute_pre_model_bias(
                    df,
                    label_col,
                    group_col,
                    favorable_label=favorable_label,
                    privileged_group=privileged_group,
                    unprivileged_group=unprivileged_group,
                )

                group_rates = pm.compute_group_rates(
                    df,
                    label_col,
                    group_col,
                    favorable_label=favorable_label,
                )

            except (ValueError, TypeError) as error:
                st.error(str(error))

            else:
                record_fairness(df, uploaded, label_col, group_col, bias_result)
                st.caption(
                    "Recorded for the Cross-Analysis Implications page, "
                    "which shows how much of your data each analysis used."
                )

                section_header("Result")

                metric_1, metric_2 = st.columns(2)

                metric_1.metric(
                    f"{privileged_group} favorable rate",
                    f"{bias_result.privileged_rate:.1%}",
                )

                metric_2.metric(
                    f"{unprivileged_group} favorable rate",
                    f"{bias_result.unprivileged_rate:.1%}",
                )

                metric_3, metric_4 = st.columns(2)

                metric_3.metric(
                    "Disparate impact",
                    f"{bias_result.disparate_impact:.3f}",
                )

                metric_4.metric(
                    "Statistical parity difference",
                    f"{bias_result.statistical_parity_difference:+.3f}",
                )

                st.caption(
                    "Disparate impact = comparison-group favorable rate ÷ "
                    "reference-group favorable rate."
                )

                st.caption(
                    "Statistical parity difference = comparison-group "
                    "favorable rate − reference-group favorable rate."
                )

                # -----------------------------------------------------
                # Interpretation
                # -------------------------------------------------------------

                section_header("Interpretation")

                if abs(bias_result.disparate_impact - 1.0) < 1e-12:
                    st.info(
                        "The two selected groups have equal favorable-label "
                        "rates in this dataset."
                    )

                elif bias_result.disparate_impact < 1:
                    st.warning(
                        f"The favorable-label rate for {unprivileged_group} "
                        f"is {bias_result.disparate_impact:.1%} of the rate "
                        f"for {privileged_group}."
                    )

                else:
                    st.info(
                        f"The favorable-label rate for {unprivileged_group} "
                        f"is {bias_result.disparate_impact:.1%} of the rate "
                        f"for {privileged_group}. The selected comparison "
                        "group has the higher favorable rate."
                    )

                caveat(
                    "These metrics describe differences in observed label "
                    "rates. They do not identify the cause of the difference "
                    "or determine whether it is ethically acceptable."
                )

                # -----------------------------------------------------
                # All-group table
                # -------------------------------------------------------------

                section_header(
                    "Favorable-label rates by group",
                    "Review all groups rather than only one selected pair",
                )

                rate_rows = [
                    {
                        "Group": group_result.group,
                        "n": group_result.n,
                        "Favorable count": group_result.favorable_count,
                        "Favorable rate": round(
                            group_result.favorable_rate,
                            3,
                        ),
                        "Small sample": (
                            "Yes"
                            if group_result.small_sample
                            else "No"
                        ),
                    }
                    for group_result in group_rates
                ]

                rates_df = pd.DataFrame(rate_rows)

                st.dataframe(
                    rates_df,
                    width="stretch",
                    hide_index=True,
                )

                for group_result in group_rates:
                    if group_result.small_sample:
                        flagged_item_note(
                            str(group_result.group),
                            (
                                "This group has a small sample. Its observed "
                                "rate may be unstable and may also create "
                                "privacy or disclosure concerns."
                            ),
                        )

                # -----------------------------------------------------
                # Chart
                # -------------------------------------------------------------

                if not rates_df.empty:
                    section_header("Group-rate chart")

                    chart_df = (
                        rates_df[
                            ["Group", "Favorable rate"]
                        ]
                        .set_index("Group")
                    )

                    st.bar_chart(chart_df)

                # -----------------------------------------------------
                # Limitations
                # -------------------------------------------------------------

                section_header("What this result does not establish")

                st.markdown(
                    """
- It does not determine why the group rates differ.
- It does not establish discrimination or unfair treatment.
- It does not evaluate model errors because predictions are not used in
  the current calculation.
- It does not show whether labels are valid, unbiased, or measured
  consistently across groups.
- It does not determine which fairness metric should govern a real
  decision without considering the application and affected communities.
"""
                )


# ---------------------------------------------------------------------
# Research examples remain visible without an uploaded dataset
# ---------------------------------------------------------------------

show_case_studies("model_validation")
