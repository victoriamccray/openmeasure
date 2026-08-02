"""
OpenMeasure — landing page.

This is the entry point for the Streamlit multipage app. Individual
modules live in pages/ and are auto-discovered by Streamlit's sidebar
navigation.
"""

import streamlit as st

st.set_page_config(page_title="OpenMeasure", page_icon="✅", layout="centered")

st.title("✅ OpenMeasure")
st.caption("An open-source validation toolkit for research data, measures, models, and programs.")

st.divider()

st.markdown(
    """
OpenMeasure answers one question, applied to four different objects:

**Transparent validation for applied research.**

- Can you trust this **measure** to score consistently? → Reliability
- Can you trust this **data** to be clean and well-understood? → Data Validation
- Can you trust this **model** to behave fairly across groups? → Model Validation
- Can you trust this **program** to have caused the effect you're seeing? → Program Validation

Use the sidebar to open a module.
"""
)

st.divider()
st.subheader("Roadmap")

st.markdown(
    """
```
OpenMeasure
├── Measurement Validation
│   └── Reliability (v0.1)         ✅ available
├── Data Validation                 planned
├── Model Validation
│   └── Fairness auditing           # model validation roadmap
└── Program Validation
    └── Pre/post & treatment/control # program validation roadmap
```
"""
)

st.caption(
    "Every module follows the same design: a plain-language verdict first, "
    "diagnostics second, and an explicit statement of what the result does "
    "NOT tell you. See docs/design-standards.md for the conventions used "
    "across modules."
)
