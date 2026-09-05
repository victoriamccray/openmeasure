"""
Shared reporting utilities used across OpenMeasure modules.

Every module (reliability, fairness, program evaluation, ...) renders its
results through these helpers instead of building its own ad hoc verdict
or layout logic. This is what "standardize validation, reporting, and
visual design across modules" actually means in code: one place that
defines what a verdict banner, a section header, and a flagged-item
callout look like, reused everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
import streamlit as st

from shared.case_studies import get_case_studies, get_case_study
from shared.catalog import LIFECYCLE_STAGES, STAGE_QUESTIONS, WORKFLOWS, workflows_by_stage
from shared.formula import FormulaExplanation
from shared.handoff import HandoffEntry
from shared.progress import stage_progress

# One heading for the research examples section, used on every page so the
# section is always named rather than left for the reader to infer.
CASE_STUDIES_HEADING = "Research Examples & Case Studies"


@dataclass(frozen=True)
class Band:
    """One labeled tier in a threshold-based verdict scale."""

    threshold: float  # inclusive lower bound for this band
    label: str
    tone: str  # "success" | "warning" | "error" | "info"


def classify(value: float, bands: Sequence[Band]) -> Band:
    """
    Return the highest band whose threshold the value meets or exceeds.

    Example:
        bands = [
            Band(0.0, "Unacceptable", "error"),
            Band(0.5, "Poor", "error"),
            Band(0.6, "Questionable", "warning"),
            Band(0.7, "Acceptable", "info"),
            Band(0.8, "Good", "success"),
            Band(0.9, "Excellent", "success"),
        ]
        classify(0.82, bands)  -> Band(0.8, "Good", "success")
    """
    sorted_bands = sorted(bands, key=lambda b: b.threshold)
    matched = sorted_bands[0]
    for band in sorted_bands:
        if value >= band.threshold:
            matched = band
    return matched


_RENDERERS = {
    "success": st.success,
    "warning": st.warning,
    "error": st.error,
    "info": st.info,
}


def render_verdict(band: Band) -> None:
    """Render a Band using the tone-appropriate Streamlit component."""
    renderer = _RENDERERS.get(band.tone, st.info)
    renderer(f"**{band.label}**")


def section_header(title: str, caption: str | None = None) -> None:
    """Consistent section divider + header used at the start of every
    report section, in every module."""
    st.divider()
    st.subheader(title)
    if caption:
        st.caption(caption)


def flagged_item_note(name: str, message: str) -> None:
    """Consistent styling for a single flagged item/case in a diagnostics
    table (used for flagged scale items, flagged subgroups, flagged
    outcome cases, etc. depending on the module)."""
    st.caption(f"**{name}**: {message}")


def caveat(text: str) -> None:
    """Consistent styling for a methodological caveat shown under a
    headline result (e.g. 'these are conventions, not laws')."""
    st.caption(text)


def inspect_note(text: str) -> None:
    """
    Consistent styling for "what to look at first" guidance, shown before
    or alongside a chart, table, or metric so a reader knows which part
    of it carries the meaning before they start interpreting it.

    Distinct from caveat(): a caveat qualifies a result after the fact
    (a limitation); this orients a reader toward a result before they
    read it. Wording ("**Label**: Sentence.", not "**Label:** sentence")
    matches the convention already established, before this shared
    version existed, in GRAND_Worked_Example.py's own ad hoc "What to
    inspect" / "Interpretation" / "Implications" headings (see that
    page's "Register" docstring note) -- extracted here rather than left
    as page-local prose once a second page needed the same three labels.
    """
    st.write(f"**What to inspect**: {text}")


def interpretation_note(text: str) -> None:
    """
    Consistent styling for a free-text interpretation of a result.

    Distinct from render_verdict(): that renders a value classified into
    a threshold Band; this is for a page with no banding scheme, stating
    in prose what a result shows without collapsing it into a labeled
    tier. Use whichever fits the result -- not both for the same result.
    """
    st.write(f"**Interpretation**: {text}")


def implications(text: str) -> None:
    """
    Consistent styling for what a result means for the reader's next
    step, shown after a result (and its caveat(s)/interpretation, if
    any) rather than before it.

    Deliberately not a recommendation to act ("do X"): it states what
    the result does and does not support, leaving the decision itself to
    the reader, consistent with the toolkit's validation-over-automation
    stance.
    """
    st.write(f"**Implications**: {text}")


def case_study_note(key: str, connection: str) -> None:
    """
    Anchor one published example beside the decision it speaks to.

    Distinct from show_case_studies(), which renders a page's whole
    examples section at the top. This puts a single example at the point
    a reader is actually making the choice it is about, collapsed, so it
    is available without competing with the analysis.

    The reading order is concept, example, what it demonstrates, then
    connection to this analysis. The concept is the expander's own label
    rather than a line inside it: a study's `principle` field is already
    a one-line statement of the idea, so using it as the label means a
    collapsed note tells a reader which concept is behind it, and
    repeating it inside would say the same sentence twice in one box.

    `connection` is supplied by the call site because it is the only part
    that depends on where the note appears. Everything else comes from
    shared/case_studies.py, so the summary, takeaway, and citation a
    reader sees here are the same verified text the examples section
    shows, and cannot drift into a second, looser paraphrase.

    A connection must say how the example bears on the analysis at hand
    without claiming the study showed more than it did. Empty text
    raises, so a note cannot render as an example with no stated reason
    for being where it is.
    """
    if not connection.strip():
        raise ValueError(
            f"The case-study note for '{key}' has no connection text. A "
            "note needs to state how the example bears on the analysis it "
            "is placed beside."
        )

    study = get_case_study(key)

    with st.expander(study.principle):
        st.markdown("**Example**")
        st.write(f"**{study.title}**")
        st.write(study.summary)

        st.markdown("**What it demonstrates**")
        st.write(study.takeaway)

        st.markdown("**Connection to this analysis**")
        st.write(connection)

        st.caption(study.citation)


def render_formula(explanation: FormulaExplanation) -> None:
    """
    Show a statistic as concept, then this reader's numbers, then result.

    Notation and per-symbol anatomy are real and are reachable, but both
    sit behind a collapsed expander. Leading with notation is the thing
    this ordering exists to avoid: a reader should not have to decode
    symbols before understanding what the calculation is doing.

    The model, shared/formula.py, guarantees the substituted line is
    assembled from the same terms the anatomy lists, so nothing rendered
    here can show a number that disagrees with its own definition.
    Interactivity is the caller's: a page that moves an input recomputes
    its result and rebuilds the explanation, so the figure and the
    arithmetic update together rather than through separate paths.
    """
    st.markdown(f"**{explanation.name}**")

    # The concept, as labeled parts with no notation in them at all.
    st.markdown(
        " ".join(
            block if block in {"/", "-", "+", "x", "="} else f"`{block}`"
            for block in explanation.blocks
        )
    )

    # The same arithmetic, with this reader's own values in it.
    st.markdown(f"### {explanation.substituted}")
    st.caption(explanation.reading)

    with st.expander("Show the notation and what each symbol means"):
        st.latex(explanation.formal_latex)

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Symbol": term.symbol,
                        "Is": term.plain_name,
                        "Value here": term.display_value,
                        "From": term.source,
                    }
                    for term in explanation.terms
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        for term in explanation.terms:
            st.caption(f"**{term.plain_name}**: {term.meaning}")

        if explanation.citation:
            st.caption(explanation.citation)


def show_case_studies(module: str) -> None:
    """
    Display research case studies relevant to a module, most relevant first.

    module should match one of the taxonomy keys used in
    shared/case_studies.py (e.g. "measurement_validation",
    "program_validation", "model_validation", "data_validation"), not
    necessarily the folder name of the calling module.

    Each study is collapsed behind its own title, so the reader scans a
    short list and opens what is relevant instead of reading several hundred
    words laid out in full. The research stage is shown alongside each title
    rather than used as a grouping heading, so a page can lead with its most
    relevant example while still telling the reader where in a study that
    lesson applies.

    Renders nothing when no case studies are tagged for the module, so it is
    safe to call from every page.
    """
    studies = get_case_studies(module)

    if not studies:
        return

    # The heading lives here rather than in each page. There are nine call
    # sites across five pages, including early-return branches, and only
    # three of them used to carry a heading at all, so the section was
    # unlabelled on two pages entirely. Owning it here makes the label
    # impossible to omit or to word differently from one page to the next.
    section_header(CASE_STUDIES_HEADING)

    # One bounded panel, so the examples read as a discrete section rather
    # than as loose boxes in the page flow.
    with st.container(border=True):
        st.caption(_stage_coverage(studies))

        for study in studies:
            with st.expander(f"{study.title} ({study.stage})"):
                st.caption(study.principle)
                st.write(study.summary)
                st.info(study.takeaway)
                st.caption(study.citation)


def _stage_coverage(studies: Sequence) -> str:
    """
    Describe which research stages a set of examples covers.

    Stages are listed in research order rather than in the order the
    examples happen to appear, because the point of the line is to show
    where in a study these lessons apply. Reading "Measurement, Analysis,
    Interpretation" conveys a progression; the display order does not.
    """
    present = [
        stage
        for stage in LIFECYCLE_STAGES
        if any(study.stage == stage for study in studies)
    ]

    if len(present) == 1:
        return f"All at the {present[0].lower()} stage of a study."

    return "Spanning " + ", ".join(present).lower() + "."


# Shown under the current stage's column only, on a validation page. A
# location marker, not a status: it uses a fixed, neutral color (blue, the
# same convention a map uses for "you are here") rather than the
# green/red vocabulary that would make it read as pass or fail.
CURRENT_STAGE_MARKER = "You are here"
CURRENT_STAGE_MARKER_ICON = ":material/near_me:"
CURRENT_STAGE_MARKER_COLOR = "blue"

# Shown inside a stage's popover when no workflow exists for it yet, and
# the stage has no other page to point to instead (unlike Research
# Question, which points to pages/Research_Journeys.py below).
NO_WORKFLOW_FOR_STAGE = "Not yet covered by a module."

# One icon per stage, purely for visual identity. Muted Material Symbols,
# matching the icon treatment already used for page and expander icons
# elsewhere, not colored or judgmental: a stage's icon does not change
# based on whether anything has been recorded for it.
_STAGE_ICONS: dict[str, str] = {
    "Research Question": ":material/help:",
    "Data": ":material/database:",
    "Measurement": ":material/straighten:",
    "Analysis": ":material/query_stats:",
    "Interpretation": ":material/lightbulb:",
}


def render_lifecycle_tracker(
    entries: tuple[HandoffEntry, ...] = (),
    *,
    current_workflow: str | None = None,
    show_status: bool = False,
) -> None:
    """
    One column per lifecycle stage, reused on Home and on every validation
    page so there is exactly one implementation of the tracker rather than
    a slightly different copy per page.

    Clicking a stage never navigates by itself. Each stage is a popover
    naming its workflow(s), via a page_link inside the popover, so picking
    where to go stays a deliberate second step rather than an immediate
    jump. It carries no arrows, tick marks, or counter: the stages are
    ordered, but the workflows are not a prerequisite chain, and any of
    those would turn "an analysis was recorded" into "this step is done".

    current_workflow marks "you are here" by matching a Workflow.workflow
    name from shared/catalog.py, so a rename there fails loudly instead of
    silently leaving the marker on the wrong stage. None on Home, where no
    single stage applies. Whenever it is set, a link back to Method
    Selection is also shown below the tracker, for a reader who landed on
    this page directly (a bookmark, a search result) and is not sure it is
    the workflow their question actually needs.

    show_status additionally shows each stage's existing Recorded / Not
    assessed / Partly recorded / Reads records / No module yet caption
    from shared/progress.py, gated exactly as on Home today (hidden until
    something has been recorded). False on every validation page, which is
    what keeps the tracker there lightweight: entries defaults to an empty
    tuple, since a page that never asks for status never needs to fetch
    the handoff store just to draw its tracker.
    """
    highlighted_stage = None

    if current_workflow is not None:
        match = next(
            (item for item in WORKFLOWS if item.workflow == current_workflow),
            None,
        )
        if match is None:
            raise ValueError(
                f"'{current_workflow}' does not match any workflow in "
                "shared/catalog.py."
            )
        highlighted_stage = match.stage

    grouped = workflows_by_stage()
    stage_state_by_name = (
        {item.stage: item.state for item in stage_progress(entries)}
        if show_status
        else {}
    )

    with st.container(border=True):
        columns = st.columns(len(LIFECYCLE_STAGES))

        for column, stage in zip(columns, LIFECYCLE_STAGES):
            with column:
                with st.popover(
                    stage,
                    icon=_STAGE_ICONS.get(stage),
                    type="tertiary",
                    width="stretch",
                ):
                    st.caption(STAGE_QUESTIONS[stage])

                    workflows = grouped[stage]

                    if not workflows and stage == "Research Question":
                        # No numbered workflow covers this stage; Research
                        # Journeys does instead (see
                        # shared/research_journeys.py's docstring).
                        st.page_link(
                            "pages/Research_Journeys.py",
                            label="Open Research Journeys",
                            icon=":material/arrow_forward:",
                        )
                    elif not workflows:
                        st.caption(NO_WORKFLOW_FOR_STAGE)

                    for workflow in workflows:
                        st.page_link(
                            workflow.page,
                            label=f"Open {workflow.workflow}",
                            icon=":material/arrow_forward:",
                        )

                if stage == highlighted_stage:
                    st.badge(
                        CURRENT_STAGE_MARKER,
                        icon=CURRENT_STAGE_MARKER_ICON,
                        color=CURRENT_STAGE_MARKER_COLOR,
                    )

                if show_status:
                    st.caption(stage_state_by_name[stage])

    if current_workflow is not None:
        st.page_link(
            "pages/Method_Selection.py",
            label="Not sure this is the right workflow? Open Method Selection",
            icon=":material/alt_route:",
        )
