"""
OpenMeasure - Method Selection Decision Tree.

One guided question routing to the workflow that fits, with Try, Why,
You'll learn, and Limitations for the chosen branch. This is a guidance
page, not an analysis: it records nothing to shared/handoff.py, carries
no module_key, and is deliberately not a numbered page, so it needs no
entry in shared/catalog.py and cannot appear on the overview's progress
cards or stage strip.

Each destination workflow already has its own, more specific method
selection logic once you are inside it (for example, Impact Evaluation
recommends a statistical test based on your data's shape). This page
only routes between workflows, one level up from that, and stops there.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from shared.catalog import WORKFLOWS
from shared.method_guide import BRANCHES

st.set_page_config(
    page_title="OpenMeasure - Method Selection",
    page_icon=":material/alt_route:",
    layout="centered",
)

st.title("Method Selection")
st.caption(
    "Answer one question to find the workflow that fits what you're "
    "trying to validate."
)

st.divider()

_PAGE_BY_WORKFLOW = {item.workflow: item.page for item in WORKFLOWS}
_BRANCH_BY_ID = {branch.id: branch for branch in BRANCHES}

selected_id = st.radio(
    "What are you trying to validate?",
    options=[branch.id for branch in BRANCHES],
    format_func=lambda branch_id: _BRANCH_BY_ID[branch_id].situation,
)

branch = _BRANCH_BY_ID[selected_id]


def _decision_tree_dot(branches: tuple, selected: str) -> str:
    """A small routing diagram: one question, one destination workflow per
    branch, with the currently selected one highlighted. Situation wording
    lives only in the radio/Try/Why text above, not duplicated here, so the
    diagram cannot drift out of sync with it."""
    lines = [
        "digraph {",
        "  rankdir=LR;",
        "  bgcolor=transparent;",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
        'fontsize=11, color="#c3c2b7", fillcolor="#fcfcfb", fontcolor="#0b0b0b"];',
        '  edge [color="#c3c2b7", arrowsize=0.7];',
        '  Q [label="What are you\\ntrying to validate?", shape=diamond, '
        'fillcolor="#f9f9f7"];',
    ]
    for item in branches:
        node_id = f"w_{item.id}"
        if item.id == selected:
            lines.append(
                f'  {node_id} [label="{item.workflow}", fillcolor="#2a78d6", '
                f'fontcolor="#ffffff", color="#2a78d6"];'
            )
        else:
            lines.append(f'  {node_id} [label="{item.workflow}"];')
        lines.append(f"  Q -> {node_id};")
    lines.append("}")
    return "\n".join(lines)


st.graphviz_chart(_decision_tree_dot(BRANCHES, selected_id))
st.caption("The highlighted box is where your answer points.")

st.markdown(f"**Try: {branch.workflow}**")
st.page_link(
    _PAGE_BY_WORKFLOW[branch.workflow],
    label=f"Open {branch.workflow}",
    icon=":material/arrow_forward:",
)

st.markdown("**Why**")
st.write(branch.why)

with st.expander("You'll learn"):
    st.write(branch.youll_learn)

with st.expander("Limitations"):
    for limitation in branch.limitations:
        st.markdown(f"- {limitation}")

st.divider()

st.caption(
    "Each workflow above has its own, more specific method selection "
    "logic once you're inside it. This page only points you to the "
    "workflow to start with."
)
