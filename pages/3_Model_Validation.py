"""
OpenMeasure — Model Validation: Fairness module.

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
from modules.fairness.core.recommend import (
    FAIRNESS_GOALS,
    recommend_fairness_metric,
)
from shared.report import (
    caveat,
    flagged_item_note,
    section_header,
    show_case_studies,
)


st.set_page_config(
    page_title="OpenMeasure · Model Validation · Fairness",
    layout="centered",
)

st.title("Model Validation: Fairness")
st.caption(
    "Review group-level outcome distributions and select fairness metrics "
    "based on the harms and protections that matter in the application."
)

st.divider()


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
different groups before a classifier is trained.

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
require model predictions—and, for most metrics, observed ground-truth
outcomes. Those analyses belong to a later model-performance stage.
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


if recommendation.metric != "demographic_parity":
    st.info(
        f"{recommendation.display_name} requires model predictions and is "
        "not calculated by the current pre-model module. The analysis below "
        "can still examine favorable-label rates already present in the data."
    )


# ---------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------

section_header(
    "2. Upload your data",
    "CSV file, one row per participant or observation",
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

    sample_candidates = [
        (
            ROOT
            / "modules"
            / "fairness"
            / "sample_data"
            / "biomedical_fairness_example.csv"
        ),
        (
            ROOT
            / "modules"
            / "fairness"
            / "sample_data"
            / "fairness_example.csv"
        ),
    ]

    sample_path = next(
        (
            path
            for path in sample_candidates
            if path.exists()
        ),
        None,
    )

    if sample_path is not None:
        with sample_path.open("rb") as sample_file:
            st.download_button(
                "Download sample fairness dataset",
                sample_file,
                file_name=sample_path.name,
                mime="text/csv",
            )
    else:
        st.caption(
            "Add a sample CSV under modules/fairness/sample_data to enable "
            "the sample-data download."
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
            use_container_width=True,
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
                "This should be the observed binary outcome, not a model "
                "prediction or participant identifier."
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
                "This is the comparison group used in the denominator of "
                "disparate impact and as the reference for statistical "
                "parity difference."
            ),
        )

        available_unprivileged_groups = [
            group
            for group in group_values
            if group != privileged_group
        ]

        unprivileged_group = st.selectbox(
            "Comparison or unprivileged group",
            options=available_unprivileged_groups,
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
                    (
                        f"{bias_result.statistical_parity_difference:+.3f}"
                    ),
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
                # -----------------------------------------------------

                section_header("Interpretation")

                if bias_result.disparate_impact == 1:
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
                        f"for {privileged_group}, meaning the selected "
                        "comparison group has the higher favorable rate."
                    )

                caveat(
                    "These metrics describe differences in observed label "
                    "rates. They do not identify the cause of the difference "
                    "or determine whether it is ethically acceptable."
                )

                # -----------------------------------------------------
                # All-group table
                # -----------------------------------------------------

                section_header(
                    "Favorable-label rates by group",
                    "Review all groups rather than only one selected pair",
                )

                rate_rows = []

                for group_result in group_rates:
                    favorable_rate = getattr(
                        group_result,
                        "favorable_rate",
                        getattr(
                            group_result,
                            "rate",
                            getattr(
                                group_result,
                                "selection_rate",
                                None,
                            ),
                        ),
                    )

                    group_n = getattr(
                        group_result,
                        "n",
                        getattr(
                            group_result,
                            "sample_size",
                            None,
                        ),
                    )

                    rate_rows.append(
                        {
                            "Group": group_result.group,
                            "n": group_n,
                            "Favorable rate": (
                                round(favorable_rate, 3)
                                if favorable_rate is not None
                                else None
                            ),
                            "Small sample": (
                                "Yes"
                                if group_result.small_sample
                                else "No"
                            ),
                        }
                    )

                rates_df = pd.DataFrame(rate_rows)

                st.dataframe(
                    rates_df,
                    use_container_width=True,
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
                # -----------------------------------------------------

                if (
                    not rates_df.empty
                    and rates_df["Favorable rate"].notna().any()
                ):
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
                # -----------------------------------------------------

                section_header("What this result does not establish")

                st.markdown(
                    """
- It does not determine why the group rates differ.
- It does not establish discrimination or unfair treatment.
- It does not evaluate model errors because no predictions are being
  analyzed.
- It does not show whether labels are valid, unbiased, or measured
  consistently across groups.
- It does not determine which fairness metric should govern a real
  decision without considering the application and affected communities.
"""
                )


# ---------------------------------------------------------------------
# Research examples remain visible without an uploaded dataset
# ---------------------------------------------------------------------

section_header(
    "Research examples",
    "Published examples illustrating why fairness definitions, assumptions, "
    "and tradeoffs matter",
)

show_case_studies("model_validation")
