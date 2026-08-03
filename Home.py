"""
OpenMeasure — landing page.

Entry point for the Streamlit multipage application.
Individual modules live in pages/ and are automatically
discovered by Streamlit's sidebar navigation.
"""

import streamlit as st

st.set_page_config(
    page_title="OpenMeasure Lab",
    layout="centered",
)

st.title("OpenMeasure Lab")
st.caption(
    "An open-source validation toolkit for research data, measures, models, and programs."
)

st.divider()

st.markdown(
    """
OpenMeasure brings together statistical methods, transparent reporting, and
plain-language interpretation to support validation throughout the research
process.

OpenMeasure provides modular tools for evaluating different parts of a study across the research lifecycle:

- **Measurement validation** evaluates whether instruments consistently measure the intended construct and supports future assessment of additional measurement properties.
- **Data validation** evaluates the quality, completeness, consistency, and integrity of datasets before analysis.
- **Model validation** evaluates predictive performance, robustness, calibration, subgroup behavior, and fairness using transparent, documented metrics.
- **Program validation** supports evaluation of interventions using established study designs and statistical analyses appropriate to the research context.

Use the sidebar to open a module.
"""
)

st.divider()

st.subheader("Modules")

st.markdown(
    """
```
OpenMeasure
├── Measurement Validation
│   └── Reliability (v0.1)
├── Data Validation
│   └── Data quality and integrity (under development)
├── Model Validation
│   └── fairness and subgroup evaluation (v0.05)
└── Program Validation
    └── Study design and impact evaluation (v0.1)
```
"""
)

st.divider()

st.subheader("Design principles")

st.markdown(
    """
Every module follows the same principles:

- Transparent statistical methods
- Reproducible analyses
- Plain-language interpretation
- Explicit assumptions and limitations
- Documented references
- Responsible use grounded in established research and professional ethics
"""
)

st.caption(
    "OpenMeasure is designed for researchers and practitioners working in "
    "community health, social services, education, public policy, and applied "
    "research. The toolkit emphasizes transparent validation methods that are "
    "accessible, reproducible, and adaptable across disciplines."
)
