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

from modules.fairness.core import post_model_metrics as pmm
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
    inspect_note,
    render_lifecycle_tracker,
    section_header,
    show_case_studies,
)
from shared.upload import render_data_profile

FAIRNESS_ACCENT = "#2a78d6"


def _two_group_rate_chart_spec(
    privileged_group: str, privileged_rate: float, unprivileged_group: str, unprivileged_rate: float
) -> dict:
    """One bar per selected group's favorable rate, axis labeled and expressed as a percentage."""

    rows = [
        {"Group": str(privileged_group), "rate": privileged_rate},
        {"Group": str(unprivileged_group), "rate": unprivileged_rate},
    ]

    return {
        "data": {"values": rows},
        "mark": {"type": "bar", "color": FAIRNESS_ACCENT},
        "encoding": {
            "y": {"field": "Group", "type": "nominal", "title": None, "sort": None},
            "x": {
                "field": "rate",
                "type": "quantitative",
                "title": "Favorable rate (%)",
                "axis": {"format": ".0%"},
                "scale": {"domain": [0, 1]},
            },
            "tooltip": [
                {"field": "Group", "type": "nominal"},
                {"field": "rate", "type": "quantitative", "format": ".1%"},
            ],
        },
        "width": "container",
        "height": 90,
    }


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


def record_post_model_fairness(
    frame, upload, true_label_column, predicted_label_column, group_column, result
) -> None:
    """
    Record this analysis for the Cross-Analysis Implications page.

    Replaces any pre-model record for this module, per HandoffStore's one
    record per module rule: running the post-model analysis after the
    pre-model analysis means only the post-model numbers appear there
    until pre-model is run again.
    """
    HandoffStore(st.session_state).record(
        module=MODULE_FAIRNESS,
        fingerprint=fingerprint_dataframe(frame, upload.name),
        exclusion=ExclusionAccount(
            module=MODULE_FAIRNESS,
            analysis_label="Fairness (post-model)",
            columns_considered=(
                str(true_label_column),
                str(predicted_label_column),
                str(group_column),
            ),
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
            "equal_opportunity_difference": float(
                result.equal_opportunity_difference
            ),
            "predictive_equality_difference": float(
                result.predictive_equality_difference
            ),
            "calibration_within_groups_difference": float(
                result.calibration_within_groups_difference
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

The current module performs **pre-model label-distribution analysis**
and, when a predicted-label column is available, **post-model
prediction analysis**.

Available pre-model metrics include:

- Favorable-label rate by group
- Disparate impact
- Statistical parity difference
- Small-group warnings

Available post-model metrics include:

- Equal opportunity (true-positive-rate parity)
- Predictive equality (false-positive-rate parity)
- Equalized odds (both of the above, reported together rather than
  combined into one number)
- Calibration within groups, approximated with positive predictive
  value rather than a full calibration curve over predicted
  probabilities

### Important limitation

A difference in favorable-label rates, or in prediction rates, does not,
by itself, establish that the data or decision process is unfair.

Observed disparities may reflect:

- historical inequity;
- differences in access or measurement;
- sampling or selection effects;
- label bias;
- meaningful differences in the target population; or
- data-quality problems.

Equal opportunity, predictive equality, and equalized odds require both
model predictions and observed ground-truth outcomes. They can move in
different directions on the same dataset, so satisfying one does not
imply satisfying another.
"""
    )


# ---------------------------------------------------------------------
# Fairness-goal recommendation
# ---------------------------------------------------------------------

section_header(
    "1. Identify the Fairness Goal",
    "Start with the harm or protection that matters in the application",
)

# Icons purely for visual scanning of the domain cards below; the domain
# name and relevance text (from modules/fairness/core/recommend.py) carry
# the actual content, so a missing icon (the .get(..., default) fallback)
# never leaves a domain undescribed.
DOMAIN_ICONS = {
    "Employment": ":material/work:",
    "Public-resource allocation": ":material/account_balance:",
    "Credit/economics": ":material/payments:",
    "Healthcare": ":material/local_hospital:",
    "Education": ":material/school:",
    "Law enforcement": ":material/gavel:",
}
DEFAULT_DOMAIN_ICON = ":material/category:"

# A worked example already in this toolkit for a domain, shown alongside
# its card rather than left as an abstract description. Only domains with
# a genuinely on-point match are listed here; a weak or stretched match
# would overstate what the linked page actually demonstrates.
DOMAIN_JOURNEY_LINKS = {
    "Healthcare": (
        "pages/Pulse_Oximeter_Worked_Example.py",
        "Worked example in this toolkit: Pulse Oximeter Racial Bias",
    ),
}

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
        icon = DOMAIN_ICONS.get(context.domain, DEFAULT_DOMAIN_ICON)
        # Icon inline with the heading, not inside st.badge: a badge does
        # not wrap, so a longer domain name (e.g. "Public-resource
        # allocation") silently truncated there.
        st.markdown(f"{icon} **{context.domain}**")
        st.write(context.relevance)

        journey_link = DOMAIN_JOURNEY_LINKS.get(context.domain)
        if journey_link:
            page, label = journey_link
            st.page_link(page, label=label, icon=":material/arrow_forward:")

        st.write("")

if recommendation.metric != "demographic_parity":
    st.info(
        f"{recommendation.display_name} requires model predictions. "
        "Step 3 below examines favorable-label rates already present in "
        "the data; step 4 adds this and other post-model metrics if a "
        "predicted-label column is available."
    )


# ---------------------------------------------------------------------
# Upload and sample data
# ---------------------------------------------------------------------

section_header(
    "2. Upload Your Data",
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

Step 3 below uses `true_label` and `sex` for the pre-model analysis.
Step 4 adds `predicted_label` for equal opportunity, predictive
equality, and equalized odds, and approximates calibration within
groups with positive predictive value rather than using
`predicted_probability` directly; a full calibration curve over that
column is a planned feature.
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
        render_data_profile(df)

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

        section_header("3. Configure The Pre-Model Analysis")

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

                st.vega_lite_chart(
                    _two_group_rate_chart_spec(
                        privileged_group,
                        bias_result.privileged_rate,
                        unprivileged_group,
                        bias_result.unprivileged_rate,
                    ),
                    theme=None,
                    use_container_width=True,
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
                    f"Disparate impact = {unprivileged_group} favorable rate "
                    f"÷ {privileged_group} favorable rate."
                )

                st.caption(
                    f"Statistical parity difference = {unprivileged_group} "
                    f"favorable rate − {privileged_group} favorable rate."
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
                    "Favorable-Label Rates By Group",
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
                    section_header(
                        "Group-Rate Chart",
                        "Every group's favorable rate, side by side",
                    )

                    chart_rows = [
                        {"Group": str(row["Group"]), "rate": row["Favorable rate"]}
                        for row in rate_rows
                    ]

                    st.vega_lite_chart(
                        {
                            "data": {"values": chart_rows},
                            "mark": {"type": "bar", "color": FAIRNESS_ACCENT},
                            "encoding": {
                                "y": {"field": "Group", "type": "nominal", "title": None},
                                "x": {
                                    "field": "rate",
                                    "type": "quantitative",
                                    "title": "Favorable rate (%)",
                                    "axis": {"format": ".0%"},
                                    "scale": {"domain": [0, 1]},
                                },
                                "tooltip": [
                                    {"field": "Group", "type": "nominal"},
                                    {"field": "rate", "type": "quantitative", "format": ".1%"},
                                ],
                            },
                            "width": "container",
                            "height": 30 * len(chart_rows) + 40,
                        },
                        theme=None,
                        use_container_width=True,
                    )

                    inspect_note(
                        "Each bar is one group's favorable-label rate (the "
                        "same numbers as the table above, in one column). "
                        "It exists to compare every group in the data at "
                        "once, not just the two selected in Result above - "
                        "a real dataset can have more than two groups, and "
                        "a pairwise comparison alone can miss a group with "
                        "a strikingly different rate."
                    )

                # -----------------------------------------------------
                # Limitations
                # -------------------------------------------------------------

                section_header("What This Result Does Not Establish")

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

        # -------------------------------------------------------------
        # Post-model analysis
        # -------------------------------------------------------------

        section_header(
            "4. Configure The Post-Model Analysis",
            "Optional: add a predicted-label column to compare "
            "true-positive and false-positive rates across groups",
        )

        st.caption(
            "Uses the observed label, group, reference group, and "
            "comparison group selected in step 3 above, plus one "
            "additional column: the model's predicted label."
        )

        remaining_columns = [
            column
            for column in df.columns
            if column not in (label_col, group_col)
        ]

        if not remaining_columns:
            st.info(
                "No columns remain to serve as a predicted-label column. "
                "Post-model metrics require the observed label, the "
                "group, and a separate model prediction."
            )
        else:
            predicted_label_col = st.selectbox(
                "Predicted label column (the model's decision, not a "
                "continuous probability or risk score)",
                options=remaining_columns,
                help=(
                    "Should use the same two values as the observed label "
                    "column above, one of which is the favorable label "
                    "selected in step 3."
                ),
            )

            analyze_post_clicked = st.button(
                "Analyze post-model prediction differences",
                disabled=(
                    len(label_values) != 2
                    or len(group_values) < 2
                ),
            )

            if analyze_post_clicked:
                try:
                    post_result = pmm.compare_post_model_bias(
                        df,
                        label_col,
                        predicted_label_col,
                        group_col,
                        positive_label=favorable_label,
                        privileged_group=privileged_group,
                        unprivileged_group=unprivileged_group,
                    )

                    confusion_rates = pmm.compute_group_confusion_rates(
                        df,
                        label_col,
                        predicted_label_col,
                        group_col,
                        positive_label=favorable_label,
                    )

                except (ValueError, TypeError) as error:
                    st.error(str(error))

                else:
                    record_post_model_fairness(
                        df,
                        uploaded,
                        label_col,
                        predicted_label_col,
                        group_col,
                        post_result,
                    )
                    st.caption(
                        "Recorded for the Cross-Analysis Implications page, "
                        "replacing any pre-model record for this module."
                    )

                    section_header("Result: Equal Opportunity")

                    eo1, eo2, eo3 = st.columns(3)
                    eo1.metric(
                        f"{privileged_group} true-positive rate",
                        f"{post_result.privileged_true_positive_rate:.1%}",
                    )
                    eo2.metric(
                        f"{unprivileged_group} true-positive rate",
                        f"{post_result.unprivileged_true_positive_rate:.1%}",
                    )
                    eo3.metric(
                        "Equal opportunity difference",
                        f"{post_result.equal_opportunity_difference:+.3f}",
                    )
                    st.caption(
                        "True-positive rate: of those with a favorable "
                        "observed outcome, the share the model also "
                        "predicted favorable."
                    )

                    section_header("Result: Predictive Equality")

                    pe1, pe2, pe3 = st.columns(3)
                    pe1.metric(
                        f"{privileged_group} false-positive rate",
                        f"{post_result.privileged_false_positive_rate:.1%}",
                    )
                    pe2.metric(
                        f"{unprivileged_group} false-positive rate",
                        f"{post_result.unprivileged_false_positive_rate:.1%}",
                    )
                    pe3.metric(
                        "Predictive equality difference",
                        f"{post_result.predictive_equality_difference:+.3f}",
                    )
                    st.caption(
                        "False-positive rate: of those with an "
                        "unfavorable observed outcome, the share the "
                        "model predicted favorable anyway."
                    )

                    section_header("Result: Calibration Within Groups (Proxy)")

                    cw1, cw2, cw3 = st.columns(3)
                    cw1.metric(
                        f"{privileged_group} positive predictive value",
                        f"{post_result.privileged_positive_predictive_value:.1%}",
                    )
                    cw2.metric(
                        f"{unprivileged_group} positive predictive value",
                        f"{post_result.unprivileged_positive_predictive_value:.1%}",
                    )
                    cw3.metric(
                        "Calibration-within-groups difference",
                        f"{post_result.calibration_within_groups_difference:+.3f}",
                    )
                    caveat(
                        "Positive predictive value is a binary proxy for "
                        "calibration, not a calibration curve. A full "
                        "calibration check compares predicted "
                        "probabilities to observed outcome frequency "
                        "across score bins, which requires a probability "
                        "column this module does not yet collect."
                    )

                    # -------------------------------------------------
                    # Teach through consequences: metric tension
                    # -------------------------------------------------

                    section_header(
                        "What Changes If You Pick a Different Metric",
                        "The same predictions score differently depending "
                        "on which fairness definition is applied",
                    )

                    gaps = {
                        "Equal opportunity": post_result.equal_opportunity_difference,
                        "Predictive equality": post_result.predictive_equality_difference,
                        "Calibration within groups (proxy)": (
                            post_result.calibration_within_groups_difference
                        ),
                    }

                    gaps_df = pd.DataFrame(
                        [
                            {"Metric": name, "Gap (unprivileged - privileged)": round(value, 3)}
                            for name, value in gaps.items()
                        ]
                    )
                    st.dataframe(gaps_df, width="stretch", hide_index=True)

                    smallest = min(gaps, key=lambda name: abs(gaps[name]))
                    largest = max(gaps, key=lambda name: abs(gaps[name]))

                    if smallest == largest:
                        st.info(
                            "All three gaps are the same size on this "
                            "dataset, so no single metric stands out as "
                            "more favorable here."
                        )
                    else:
                        st.warning(
                            f"On this dataset, choosing **{smallest}** as "
                            f"the fairness goal shows the smallest gap "
                            f"({gaps[smallest]:+.3f}) between the two "
                            f"groups, while **{largest}** shows the "
                            f"largest ({gaps[largest]:+.3f}). The same "
                            "predictions look more or less equitable "
                            "depending on which definition is applied, "
                            "and this module does not resolve that choice."
                        )

                    caveat(
                        "Hardt, Price, & Srebro (2016) define equal "
                        "opportunity and equalized odds using these "
                        "true-positive and false-positive rates. "
                        "Chouldechova (2017) and Kleinberg, Mullainathan, "
                        "& Raghavan (2017) show that when outcome base "
                        "rates differ across groups and the model is "
                        "imperfect, equalized odds and calibration cannot "
                        "both hold exactly. A small gap in one metric "
                        "does not imply a small gap in another."
                    )

                    section_header(
                        "Post-Model Rates By Group",
                        "Review all groups rather than only one selected pair",
                    )

                    def _rate_text(value: float | None) -> str:
                        return "not applicable" if value is None else f"{value:.1%}"

                    confusion_rows = [
                        {
                            "Group": rate.group,
                            "n": rate.n,
                            "True-positive rate": _rate_text(
                                rate.true_positive_rate
                            ),
                            "False-positive rate": _rate_text(
                                rate.false_positive_rate
                            ),
                            "Positive predictive value": _rate_text(
                                rate.positive_predictive_value
                            ),
                            "Small sample": "Yes" if rate.small_sample else "No",
                        }
                        for rate in confusion_rates
                    ]

                    st.dataframe(
                        pd.DataFrame(confusion_rows),
                        width="stretch",
                        hide_index=True,
                    )

                    for rate in confusion_rates:
                        if rate.small_sample:
                            flagged_item_note(
                                str(rate.group),
                                (
                                    "This group has a small sample. Its "
                                    "rates may be unstable."
                                ),
                            )

                    section_header("What This Result Does Not Establish")

                    st.markdown(
                        """
- It does not determine why the rates differ across groups.
- It does not establish that the model is discriminatory or fair.
- It does not evaluate the underlying labels for bias.
- It does not perform a full calibration analysis over predicted
  probabilities.
- It does not rank the three metrics above by importance; that depends
  on the harm the application is most concerned with.
"""
                    )


# ---------------------------------------------------------------------
# Research examples remain visible without an uploaded dataset
# ---------------------------------------------------------------------

show_case_studies("model_validation")

st.page_link(
    "pages/Pulse_Oximeter_Worked_Example.py",
    label="Try the interactive version: Pulse Oximeter Racial Bias (Research Journey)",
    help=(
        "Adjust the device's alarm threshold and watch the detection gap "
        "between groups respond, using this module's own metrics."
    ),
    icon=":material/arrow_forward:",
)
