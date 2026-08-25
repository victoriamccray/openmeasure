"""
OpenMeasure - Method Selection Decision Tree.

One guided question routing to the workflow or research journey that
fits, with Try, Why, You'll learn, and Limitations for the chosen branch.
This is a guidance page, not an analysis: it records nothing to
shared/handoff.py, carries no module_key, and is deliberately not a
numbered page, so it needs no entry in shared/catalog.py and cannot
appear on the overview's progress cards or stage strip.

The entry question is phrased as what you're trying to determine, not as
which OpenMeasure tool or category you need, so it works for a reader who
does not yet know that "several items measuring one thing consistently"
is what reliability means. Each destination already has its own, more
specific method selection logic once you are inside it (for example,
Impact Evaluation recommends a statistical test based on your data's
shape) or, for a research journey, its own fixed sequence of stages. This
page only routes between destinations, one level up from that, and stops
there.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from modules.data_profile.core.suggest import suggest_workflows
from shared.catalog import WORKFLOWS
from shared.method_guide import BRANCHES
from shared.research_journeys import JOURNEYS
from shared.upload import render_data_profile

st.set_page_config(
    page_title="OpenMeasure - Method Selection",
    page_icon=":material/alt_route:",
    layout="centered",
)

st.title("Method Selection")
st.caption(
    "Answer one question about what you're trying to determine, and "
    "OpenMeasure points to the workflow or research journey that fits, "
    "and explains why."
)

st.divider()

_PAGE_BY_WORKFLOW = {item.workflow: item.page for item in WORKFLOWS}
_PAGE_BY_JOURNEY = {item.title: item.page for item in JOURNEYS}
_PAGE_BY_DESTINATION = {**_PAGE_BY_WORKFLOW, **_PAGE_BY_JOURNEY}
_BRANCH_BY_ID = {branch.id: branch for branch in BRANCHES}

selected_id = st.radio(
    "What are you trying to determine?",
    options=[branch.id for branch in BRANCHES],
    format_func=lambda branch_id: _BRANCH_BY_ID[branch_id].question,
)

branch = _BRANCH_BY_ID[selected_id]


# ---------------------------------------------------------------------
# Pictograph: one small, original, hand-drawn icon per destination.
#
# Static on purpose, not animated: a flickering/pulsing icon set tried
# earlier on the GRAND worked example was flagged as both a seizure-
# trigger risk (rapid, high-contrast flicker) and useless to a screen-
# reader user, who already has the same information in text -- here,
# the radio options above and the "Try:" line below. This pictograph is
# a decorative supplement to that text, never its only copy.
#
# These are original pictographs (a checklist, a gapped timeline, a
# balance, ...), not a reproduction of any icon library's artwork.
# ---------------------------------------------------------------------

INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
ACCENT = "#2a78d6"

_DESTINATION_ICON_PATHS: dict[str, str] = {
    "reliability": (
        '<line x1="5" y1="10" x2="17" y2="10" stroke="{c}" stroke-width="2.2" stroke-linecap="round"/>'
        '<line x1="5" y1="16" x2="17" y2="16" stroke="{c}" stroke-width="2.2" stroke-linecap="round"/>'
        '<line x1="5" y1="22" x2="17" y2="22" stroke="{c}" stroke-width="2.2" stroke-linecap="round"/>'
        '<path d="M21 16L24.5 19.5L30 12" stroke="{c}" stroke-width="2.2" fill="none" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    ),
    "time_series_qa": (
        '<line x1="3" y1="22" x2="29" y2="22" stroke="{c}" stroke-width="1.3"/>'
        '<circle cx="8" cy="22" r="2.1" fill="{c}"/>'
        '<circle cx="14" cy="22" r="2.1" fill="{c}"/>'
        '<circle cx="20" cy="22" r="2.1" fill="none" stroke="{c}" stroke-width="1.4" stroke-dasharray="1.4,1.6"/>'
        '<circle cx="26" cy="22" r="2.1" fill="{c}"/>'
        '<line x1="20" y1="8" x2="20" y2="16" stroke="{c}" stroke-width="1.8" stroke-linecap="round"/>'
        '<circle cx="20" cy="5.2" r="1.2" fill="{c}"/>'
    ),
    "impact_evaluation": (
        '<line x1="3" y1="26" x2="29" y2="26" stroke="{c}" stroke-width="1.2"/>'
        '<rect x="6" y="16" width="7" height="10" rx="1" fill="{c}" opacity="0.35"/>'
        '<rect x="19" y="8" width="7" height="18" rx="1" fill="{c}"/>'
    ),
    "fairness": (
        '<line x1="16" y1="5" x2="16" y2="24" stroke="{c}" stroke-width="1.6"/>'
        '<line x1="6" y1="9" x2="26" y2="9" stroke="{c}" stroke-width="1.6"/>'
        '<path d="M3 9L6.5 17A5 5 0 0 0 9.5 17Z" fill="none" stroke="{c}" '
        'stroke-width="1.4" stroke-linejoin="round"/>'
        '<path d="M22.5 9L26 17A5 5 0 0 0 29 17Z" fill="none" stroke="{c}" '
        'stroke-width="1.4" stroke-linejoin="round"/>'
        '<line x1="11" y1="25" x2="21" y2="25" stroke="{c}" stroke-width="2" stroke-linecap="round"/>'
    ),
    "cross_analysis": (
        '<circle cx="12.5" cy="16" r="9" fill="{c}" opacity="0.16" stroke="{c}" stroke-width="1.4"/>'
        '<circle cx="19.5" cy="16" r="9" fill="{c}" opacity="0.16" stroke="{c}" stroke-width="1.4"/>'
    ),
    "portfolio_impact": (
        '<rect x="4" y="5" width="9" height="9" rx="1.3" fill="{c}" opacity="0.25" stroke="{c}" stroke-width="1.1"/>'
        '<rect x="18" y="5" width="9" height="9" rx="1.3" fill="{c}" opacity="0.25" stroke="{c}" stroke-width="1.1"/>'
        '<rect x="4" y="18" width="9" height="9" rx="1.3" fill="{c}" stroke="{c}" stroke-width="1.1"/>'
        '<rect x="18" y="18" width="9" height="9" rx="1.3" fill="{c}" opacity="0.25" stroke="{c}" stroke-width="1.1"/>'
    ),
}


def _icon_svg(path_template: str, color: str) -> str:
    return path_template.format(c=color)


def _destination_pictograph_html(branches: tuple, selected: str) -> str:
    """
    One small card per destination, with the selected one in the accent
    color and the rest muted. Wording lives only in the radio/Try/Why
    text above and is repeated here as plain text inside each card, so a
    screen-reader user reaches the same information whether or not the
    icon itself renders.
    """

    cards = []
    for item in branches:
        is_selected = item.id == selected
        color = ACCENT if is_selected else INK_MUTED
        border = ACCENT if is_selected else GRIDLINE
        background = "rgba(42, 120, 214, 0.08)" if is_selected else SURFACE
        weight = "600" if is_selected else "400"

        cards.append(
            f'<div style="display:flex; flex-direction:column; align-items:center; '
            f'width:116px; min-width:0; box-sizing:border-box; gap:4px; padding:8px 6px; '
            f'border-radius:8px; border:1.4px solid {border}; background:{background};">'
            f'<svg width="32" height="32" viewBox="0 0 32 32" style="flex-shrink:0;">'
            f"{_icon_svg(_DESTINATION_ICON_PATHS[item.id], color)}"
            f"<title>{item.destination}</title>"
            f"</svg>"
            f'<span style="font-size:11px; font-weight:{weight}; color:{color}; '
            f'text-align:center; line-height:1.25; min-width:0; '
            f'word-break:break-word;">{item.destination}</span>'
            f"</div>"
        )

    return f"""
    <div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
                display:flex; flex-wrap:wrap; justify-content:center; gap:8px;">
      {"".join(cards)}
    </div>
    """


components.html(_destination_pictograph_html(BRANCHES, selected_id), height=190)
st.caption("The highlighted card is where your answer points.")

st.markdown(f"**Try: {branch.destination}**")
st.caption(
    "This is a research journey: a guided worked example, not a "
    "versioned analysis workflow."
    if branch.destination_kind == "research journey"
    else "This is an analysis workflow: it records a result you can "
    "revisit from Cross-Analysis Implications."
)
st.page_link(
    _PAGE_BY_DESTINATION[branch.destination],
    label=f"Open {branch.destination}",
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
    "Each destination above has its own, more specific logic once you're "
    "inside it: a workflow's own method recommendation, or a journey's "
    "own sequence of stages. This page only points you to where to "
    "start."
)

st.divider()

st.subheader("Or, Upload a File Instead")
st.caption(
    "Don't know what to ask yet? Upload a file and OpenMeasure names "
    "which workflows may be relevant based on its column structure. "
    "Structure can't establish that a workflow is the methodologically "
    "appropriate choice, only that its shape is worth a look. The "
    "question above (and each destination's own Why) is what actually "
    "states a case for one."
)

uploaded = st.file_uploader(
    "CSV file", type="csv", label_visibility="collapsed", key="method_selection_upload"
)

if uploaded is not None:
    upload_df = pd.read_csv(uploaded)
    upload_profile = render_data_profile(upload_df)
    suggestions = suggest_workflows(upload_profile)

    if not suggestions:
        st.info(
            "This file's column shape didn't clearly match a pattern this "
            "suggester recognizes. Use the question above instead, or "
            "browse Research Journeys below."
        )
    else:
        st.caption(
            "These are structural matches, not a determination of "
            "methodological appropriateness: dataset structure alone "
            "can't establish your actual research question or intent. "
            "If more than one workflow is shown, that reflects a real "
            "ambiguity structure can't resolve on its own, not an error."
        )

        for suggestion in suggestions:
            with st.container(border=True):
                st.markdown(f"**{suggestion.workflow}** may be relevant")
                st.write(suggestion.reasoning)
                st.page_link(
                    _PAGE_BY_DESTINATION[suggestion.workflow],
                    label=f"Open {suggestion.workflow}",
                    icon=":material/arrow_forward:",
                )

st.page_link(
    "pages/Research_Journeys.py",
    label="None of these fit yet: browse all Research Journeys",
    icon=":material/route:",
)
