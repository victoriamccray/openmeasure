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

# Matches the muted-ink / blue-accent palette already used for static
# pictographs on pages/Method_Selection.py and pages/GRAND_Worked_Example.py
# (each page keeps its own copy of these constants; see those pages'
# INK_MUTED/INK_SECONDARY/SURFACE/ACCENT for the shared convention).
_INK = "#52514e"
_ACCENT = "#2a78d6"
_SURFACE = "#fcfcfb"


def _validation_triangle_svg() -> str:
    """
    Static (never animated - see feedback on decorative icons) diagram
    of the three-way check every validated statistic passes: hand
    calculation, OpenMeasure, and independent reference software are
    expected to agree, each a check on the other two.
    """
    return f"""
    <div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
      <svg width="100%" height="90" viewBox="0 0 640 90" preserveAspectRatio="xMidYMid meet">
        <rect x="10" y="20" width="180" height="50" rx="6" fill="{_SURFACE}" stroke="{_ACCENT}" stroke-width="1.4"/>
        <text x="100" y="49" text-anchor="middle" font-size="12" fill="{_INK}">Hand calculation</text>

        <line x1="196" y1="45" x2="224" y2="45" stroke="{_INK}" stroke-width="1.4" opacity="0.7"/>
        <polygon points="196,45 202,41 202,49" fill="{_INK}" opacity="0.8"/>
        <polygon points="224,45 218,41 218,49" fill="{_INK}" opacity="0.8"/>

        <rect x="230" y="20" width="180" height="50" rx="6" fill="{_SURFACE}" stroke="{_ACCENT}" stroke-width="1.4"/>
        <text x="320" y="49" text-anchor="middle" font-size="12" fill="{_INK}">OpenMeasure</text>

        <line x1="416" y1="45" x2="444" y2="45" stroke="{_INK}" stroke-width="1.4" opacity="0.7"/>
        <polygon points="416,45 422,41 422,49" fill="{_INK}" opacity="0.8"/>
        <polygon points="444,45 438,41 438,49" fill="{_INK}" opacity="0.8"/>

        <rect x="450" y="20" width="180" height="50" rx="6" fill="{_SURFACE}" stroke="{_ACCENT}" stroke-width="1.4"/>
        <text x="540" y="42" text-anchor="middle" font-size="11" fill="{_INK}">Independent reference</text>
        <text x="540" y="58" text-anchor="middle" font-size="11" fill="{_INK}">software</text>
      </svg>
    </div>
    """


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
        st.markdown(":material/calculate: **1. Calculation checks**")
        st.write(
            "Small, hand-calculable examples verify that a statistical "
            "method produces the expected result."
        )

with layer_columns[1]:
    with st.container(border=True):
        st.markdown(":material/rule: **2. Edge cases**")
        st.write(
            "Methods are tested under conditions such as missing data, "
            "very high or negative reliability, reverse-coded items, "
            "and small samples."
        )

with layer_columns[2]:
    with st.container(border=True):
        st.markdown(":material/compare_arrows: **3. Independent reference validation**")
        st.write(
            "OpenMeasure's results are compared with established "
            "statistical software, using the same data and the same "
            "methodological assumptions."
        )

st.markdown(_validation_triangle_svg(), unsafe_allow_html=True)

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
    st.page_link(
        "pages/1_Reliability.py",
        label="New to these terms? See \"What is Cronbach's alpha?\" on the Reliability page",
        icon=":material/menu_book:",
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
    st.caption(
        "Cronbach, L. J. (1951). Coefficient alpha and the internal "
        "structure of tests. Psychometrika, 16(3), 297-334. Revelle, "
        "W. (2024). psych: Procedures for Psychological, Psychometric, "
        "and Personality Research. Northwestern University, Evanston, "
        "Illinois."
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
st.caption(
    "This page is updated as each module completes the process "
    "described above."
)
