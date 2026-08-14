"""
OpenMeasure - Reliability module.

Computes Cronbach's alpha and item diagnostics from wide-format survey
data. Core statistics live in modules/reliability/core, this file is
presentation only, built on the shared reporting helpers in shared/.
"""

import sys
from pathlib import Path
from shared.report import section_header, flagged_item_note, caveat, show_case_studies, render_lifecycle_tracker

import pandas as pd
import streamlit as st

from modules.reliability.core import reliability as rel
from modules.reliability.core import interpret as interp
from shared.catalog import MODULE_RELIABILITY
from shared.handoff import (
    KIND_CELLS_EMPTY,
    KIND_ROWS_DROPPED,
    ExclusionAccount,
    HandoffStore,
    RetentionItem,
    fingerprint_dataframe,
)


def record_reliability(frame, upload, columns, result) -> None:
    """
    Record this analysis for the Cross-Analysis Implications page.

    Translates the result into primitives here rather than storing the
    result object, because the same dataclass is a different class
    depending on how it was imported.
    """
    missing_cells = int(
        round(result.pct_missing_cells / 100 * result.n_participants * result.n_items)
    )

    HandoffStore(st.session_state).record(
        module=MODULE_RELIABILITY,
        fingerprint=fingerprint_dataframe(frame, upload.name),
        exclusion=ExclusionAccount(
            module=MODULE_RELIABILITY,
            analysis_label="Reliability",
            columns_considered=tuple(str(column) for column in columns),
            n_input_rows=result.n_participants,
            n_retained_rows=result.n_complete_cases,
            items=(
                RetentionItem(
                    label="Participants excluded",
                    count=result.n_excluded_cases,
                    kind=KIND_ROWS_DROPPED,
                    mechanism=(
                        "listwise deletion: a missing value in any selected "
                        "item removes the participant"
                    ),
                ),
                RetentionItem(
                    label="Missing item responses",
                    count=missing_cells,
                    kind=KIND_CELLS_EMPTY,
                    mechanism="blank responses across all selected items",
                ),
            ),
        ),
        primary_statistics={"cronbach_alpha": float(result.cronbach_alpha)},
    )

st.set_page_config(page_title="OpenMeasure · Reliability", page_icon=":material/verified:", layout="centered")

st.title("Reliability")
st.subheader("Measurement Validation")
st.caption("Cronbach's alpha and item diagnostics for a scale or survey.")

st.divider()

render_lifecycle_tracker(current_workflow="Reliability")

show_case_studies("measurement_validation")
# Allow imports from the repo root regardless of where Streamlit is launched from.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

with st.expander("What is Cronbach's alpha?", icon=":material/menu_book:"):
    st.markdown(
        """
### What does reliability mean?

Researchers often use **scales** to estimate **latent variables**, which are concepts that cannot be measured directly, such as extroversion, anxiety, satisfaction, or depression.

A scale combines several questions intended to measure the same underlying construct. The items should be **meaningfully related**, but not simply repeat one another.

**Cronbach's alpha** estimates **internal consistency**, or the extent to which the selected items produce related responses as a group. In other words, it helps answer the question:

> *How consistently do these items measure the same construct?*

For example, an extroversion scale may include questions about sociability, assertiveness, and enjoyment of group activities. Cronbach's alpha assesses whether those items behave consistently as one scale.

Cronbach's alpha depends on:

- the number of items;
- variation within each item; and
- covariance among the items.

### Item diagnostics

**Corrected item-total correlations** show how strongly each item relates to the remaining scale.

**Alpha if item dropped** shows how the overall alpha would change if an item were removed. These results should support item review rather than automatic deletion.

### Conventional interpretation

| Cronbach's α | Interpretation |
|--------------|----------------|
| ≥ 0.90 | Excellent |
| ≥ 0.80 | Good |
| ≥ 0.70 | Acceptable |
| ≥ 0.60 | Questionable |
| ≥ 0.50 | Poor |
| < 0.50 | Unacceptable |

These labels follow a commonly cited scale (e.g., George & Mallery, 2003) rather than a fixed statistical standard. Interpretation should also consider the scale's purpose, length, population, and intended use.

Alpha is also a **lower-bound** estimate of reliability, not an exact one: it equals true reliability only under the assumption that all items measure the construct equally well (tau-equivalence). True reliability may be somewhat higher than the alpha value shown.
"""
    )

with st.expander("Assumptions & limitations", icon=":material/balance:"):
    st.markdown(
        """
### Assumptions

- The selected items are intended to measure the same underlying construct.
- Items should be coded in the same direction. Reverse-coded items should be recoded before analysis.
- Cronbach's alpha estimates **reliability (internal consistency)**, not **validity**.

### Tradeoffs

**Missing data**

- OpenMeasure uses **listwise deletion**.
- Participants missing one or more selected items are excluded from all reliability calculations.
- This approach is simple and reproducible but may reduce sample size if missingness is common.

**Item diagnostics**

- Corrected item-total correlations below **0.30** are flagged as items to review, following a commonly cited rule of thumb (e.g., Nunnally & Bernstein, 1994).
- A flagged item should not be removed automatically. Low values may reflect reverse coding, limited variability, multidimensionality, or poor conceptual fit.

**Split-half reliability**

- OpenMeasure computes odd-even split-half reliability.
- By default, all selected items are retained. When an odd number of items is selected, the two halves contain different numbers of items.
- Retaining all items preserves information but makes the Spearman-Brown estimate approximate.
- Users who prefer equal-sized halves may instead analyze an even number of selected items.

### Limitations

- A high alpha does **not** prove that a scale measures only one construct.
- High alpha can result from redundant or highly similar items.
- Scale dimensionality should be evaluated separately using methods such as exploratory or confirmatory factor analysis.
- Reliability should always be interpreted alongside the study design, target population, and purpose of the measure.
"""
    )

section_header("1. Upload your data", "CSV file in wide format, one row per participant")

uploaded = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")

if uploaded is None:
    st.info("Upload a CSV to get started, or try the sample dataset below.")
    sample_path = ROOT / "modules" / "reliability" / "sample_data" / "survey_example.csv"
    with open(sample_path, "rb") as f:
        st.download_button("Download sample_data/survey_example.csv", f, file_name="survey_example.csv")
    st.stop()

df = pd.read_csv(uploaded)
st.write(f"Loaded **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
st.dataframe(df.head(), width="stretch")

section_header("2. Select columns")

id_col = st.selectbox(
    "Participant ID column (optional, excluded from analysis)",
    options=["(none)"] + list(df.columns),
)

candidate_cols = [c for c in df.columns if c != id_col]
numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]

item_cols = st.multiselect(
    "Items to include in the reliability analysis",
    options=candidate_cols,
    default=numeric_cols,
)

split_half_method = st.radio(
    "Split-half method",
    options=["Use all items", "Require equal-sized halves"],
    index=0,
    help=(
        "Using all items allows odd-length scales but may produce unequal "
        "halves, making the Spearman-Brown estimate an approximation. "
        "Equal-sized halves is methodologically stricter but requires an "
        "even number of selected items."
    ),
)
require_equal_halves = split_half_method == "Require equal-sized halves"

analyze_clicked = st.button("Analyze", type="primary", disabled=len(item_cols) < 2)

if len(item_cols) < 2:
    st.warning("Select at least 2 items to compute reliability.")

if analyze_clicked:
    item_data = df[item_cols].apply(pd.to_numeric, errors="coerce")

    try:
        result = rel.analyze(item_data, require_equal_halves=require_equal_halves)
    except (ValueError, TypeError) as e:
        st.error(str(e))
        st.stop()

    section_header("Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Participants", result.n_participants)
    c2.metric("Items", result.n_items)
    c3.metric("Excluded cases", result.n_excluded_cases)
    st.caption(
        f"Complete cases used: {result.n_complete_cases} | "
        f"Excluded cases: {result.pct_excluded_cases:.1f}% | "
        f"Missing item cells: {result.pct_missing_cells:.1f}%"
    )
    caveat("Rows containing a missing value in any selected item were excluded from all "
           "reliability calculations using listwise deletion.")

    section_header("Reliability")

    # Result
    m1, m2, m3 = st.columns(3)
    m1.metric("Cronbach's α", f"{result.cronbach_alpha:.2f}")

    n_items = len(item_cols)
    odd_n = (n_items + 1) // 2
    even_n = n_items // 2
    unequal_halves = odd_n != even_n and not require_equal_halves

    if result.split_half_correlation is not None:
        m2.metric("Split-half r", f"{result.split_half_correlation:.2f}")
        m3.metric("Spearman-Brown", f"{result.spearman_brown:.2f}")
    else:
        m2.metric("Split-half r", "Not available")
        m3.metric("Spearman-Brown", "Not available")

    # Takeaway
    st.write(
        f"**Takeaway:** These {n_items} items produced Cronbach's α = "
        f"{result.cronbach_alpha:.2f} in this sample, a measure of how "
        "consistently they moved together, not of whether they measure one "
        "meaningful construct. Alpha can be pushed high by items that "
        "simply restate each other, and can be low for a short or "
        "deliberately broad scale even when every item is conceptually "
        "sound."
    )

    for w in interp.alpha_warnings(result.cronbach_alpha):
        st.warning(w)

    # Details / assumptions
    if result.split_half_correlation is not None:
        if unequal_halves:
            caveat(
                f"Halves are unequal length ({odd_n} vs {even_n} items). "
                "Spearman-Brown assumes equal-length halves, so this is "
                "an approximation, not a textbook-exact estimate."
            )
    else:
        caveat(result.split_half_unavailable_reason)

    caveat(
        "The conventional Excellent/Good/Acceptable/etc. labels for this "
        "range are shown for reference in \"What is Cronbach's alpha?\" "
        "above. Treat them as commonly cited guidelines, not a verdict on "
        "this scale."
    )

    section_header("Item diagnostics")

    diag_rows = []
    for d in result.item_diagnostics:
        diag_rows.append(
            {
                "Item": d.item,
                "Item-total corr.": round(d.item_total_corr, 3),
                "α if dropped": round(d.alpha_if_dropped, 3),
                "Flag": "Review" if d.flagged else "",
            }
        )
    st.dataframe(pd.DataFrame(diag_rows), width="stretch", hide_index=True)

    for d in result.item_diagnostics:
        msg = interp.item_warning(d.item_total_corr)
        if msg:
            flagged_item_note(d.item, msg)

    record_reliability(df, uploaded, item_cols, result)
    st.caption(
        "Recorded for the Cross-Analysis Implications page, which shows how "
        "much of your data each analysis used."
    )

    best_drop = max(result.item_diagnostics, key=lambda d: d.alpha_if_dropped)
    if best_drop.alpha_if_dropped > result.cronbach_alpha:
        st.info(
            f"Removing **{best_drop.item}** would raise alpha from "
            f"{result.cronbach_alpha:.2f} to {best_drop.alpha_if_dropped:.2f}. "
            "This item should be reviewed rather than automatically removed."
        )

    section_header("Item-total correlation chart")
    chart_df = pd.DataFrame(
        {d.item: [d.item_total_corr] for d in result.item_diagnostics}
    ).T
    chart_df.columns = ["Item-total correlation"]
    st.bar_chart(chart_df)
