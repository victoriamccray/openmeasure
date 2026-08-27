"""
Quality & Validation - how OpenMeasure checks its own calculations.

Not a workflow: no catalog entry, no module_key, no lifecycle stage -
matches pages/Privacy_and_Data_Access.py's precedent for a reference page
declared directly in Home.py rather than through shared/catalog.py.

A plain-language summary of docs/validation/reference-validation.md for a
reader who will never open that file or the test suite. States the three
validation layers OpenMeasure applies to every core/ statistic, and the
one module (Reliability) that has completed the third layer so far. Does
not restate the full dataset tables or regression-test values from that
document; links to it instead for the technical record, and keeps every
number here consistent with what that document actually reports.
"""

import pandas as pd
import streamlit as st

from shared.report import caveat, section_header

st.set_page_config(
    page_title="OpenMeasure · Quality & Validation",
    page_icon=":material/checklist:",
    layout="centered",
)

st.title("Quality & Validation")
st.write("**How does OpenMeasure check its calculations?**")
st.caption(
    "A plain-language summary. The full technical record is linked at "
    "the bottom of this page."
)

section_header("Three Layers Of Validation")

layer_columns = st.columns(3)

with layer_columns[0]:
    with st.container(border=True):
        st.markdown("**1. Calculation checks**")
        st.write(
            "Small, hand-calculable examples verify that a statistical "
            "method produces the expected result."
        )

with layer_columns[1]:
    with st.container(border=True):
        st.markdown("**2. Edge cases**")
        st.write(
            "Methods are tested under conditions such as missing data, "
            "very high or negative reliability, reverse-coded items, "
            "and small samples."
        )

with layer_columns[2]:
    with st.container(border=True):
        st.markdown("**3. Independent reference validation**")
        st.write(
            "OpenMeasure's results are compared with established "
            "statistical software, using the same data and the same "
            "methodological assumptions."
        )

st.info("Hand calculation ↔ OpenMeasure ↔ Independent reference software")

section_header("Current Validation")

with st.container(border=True):
    name_column, status_column = st.columns([3, 2])
    with name_column:
        st.markdown("**Reliability**")
    with status_column:
        st.badge(
            "Reference validated",
            icon=":material/verified:",
            color="blue",
        )

    st.caption(
        "Checked against an independent hand-derived formula and R's "
        "psych package, on 7 small fixed datasets covering typical "
        "behavior and the edge cases described below."
    )

    comparison_table = pd.DataFrame(
        [
            {
                "Method": "Cronbach's alpha",
                "Independent comparison": "R psych + hand calculation",
                "Result": "Match",
            },
            {
                "Method": "Item-total correlation",
                "Independent comparison": "R psych + hand calculation",
                "Result": "Match",
            },
            {
                "Method": "Alpha if item dropped",
                "Independent comparison": "R psych + hand calculation",
                "Result": "Match*",
            },
            {
                "Method": "Split-half reliability",
                "Independent comparison": "Independent R + hand calculation",
                "Result": "Match",
            },
            {
                "Method": "Spearman-Brown",
                "Independent comparison": "Independent R + hand calculation",
                "Result": "Match",
            },
        ]
    )
    st.dataframe(comparison_table, width="stretch", hide_index=True)
    st.caption('*See "What did we find?" below.')

    with st.expander("What did we find?"):
        st.write(
            "Beyond typical scale data, these methods were tested "
            "against conditions deliberately constructed to be unusual:"
        )
        st.markdown(
            """
- A scale with very high internal consistency
- A scale with low or negative internal consistency
- A scale with one reverse-coded item that was not recoded
- A dataset with missing responses
- A minimal scale (2 items, 5 participants)
"""
        )
        st.write(
            "One comparison needed a closer look. Dropping either item "
            "from a 2-item scale leaves a single item, and Cronbach's "
            "alpha's formula is not mathematically defined for a "
            "1-item scale. OpenMeasure reports this case as "
            "unavailable rather than showing a number. The comparison "
            "software returned a number for the same case, but that "
            "number was not internally consistent with a real "
            "reliability coefficient, and the same software errors "
            "when asked to evaluate a genuine 1-item scale directly. "
            "OpenMeasure kept its unavailable result, because a "
            "1-item scale does not provide a meaningful Cronbach's "
            "alpha to report."
        )

    caveat(
        "Matching an independent reference on these datasets confirms "
        "that OpenMeasure's arithmetic is correct. It does not "
        "establish that Cronbach's alpha, or any other method, is the "
        "right choice for a given study design; that judgment still "
        "belongs to the researcher."
    )

    st.markdown(
        "[Full technical record on GitHub]"
        "(https://github.com/victoriamccray/openmeasure/blob/main/"
        "docs/validation/reference-validation.md)"
    )

section_header("Validation Status")

status_table = pd.DataFrame(
    [
        {"Module": "Reliability", "Status": "Reference validated"},
        {"Module": "Impact Evaluation", "Status": "Next"},
        {"Module": "Fairness", "Status": "Planned"},
        {"Module": "Time-Series QA", "Status": "Planned"},
        {
            "Module": "Evidence Review",
            "Status": "Separate evaluation approach needed",
        },
    ]
)
st.dataframe(status_table, width="stretch", hide_index=True)
st.caption(
    "Evidence Review searches and screens published literature rather "
    "than computing a fixed statistic from fixed data, so this kind of "
    "numeric cross-implementation comparison does not apply to it in "
    "the same way; it will need its own evaluation approach."
)
