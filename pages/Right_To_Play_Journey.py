"""
Right To Play: evaluating a school-based violence prevention program.

A Research Journey rather than an analysis page. Impact Evaluation helps
a researcher analyse their own evaluation; this walks through how a real,
published one was built, and stops precisely where its public artifacts
stop.

That stopping point is the subject, not a shortcoming. The baseline
participant microdata are published and the 24-month participant
microdata are not, so a reader can interrogate the design, the
instruments and the baseline measurements, and cannot recompute the
reported effect. A journey that glossed over that would teach the
opposite of what this study is useful for.

Deliberately not a difference-in-differences tutorial. This is a cluster
randomized trial, randomized by school, and the clustering is the thing
that makes it different from the 2x2 comparison Impact Evaluation
teaches. Where the journey approaches an estimate it says what
OpenMeasure's current tests would get wrong on data shaped like this,
and in which direction.

Every fact shown here comes from modules/right_to_play/core/study.py,
where each one carries its own provenance and citation. This page renders
them; it does not add any.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from modules.right_to_play.core import study
from shared import visuals
from shared.datasets import get_dataset
from shared.journey_stages import StageTracker
from shared.report import caveat, inspect_note, section_header

st.set_page_config(
    page_title="OpenMeasure - Right To Play",
    page_icon=":material/school:",
    layout="centered",
)

STAGE_KEY = "rtp_stage"

STAGE_QUESTION = 0
STAGE_DESIGN = 1
STAGE_MEASUREMENT = 2
STAGE_ARTIFACTS = 3
STAGE_BOUNDARY = 4

TRACKER = StageTracker(
    session_key=STAGE_KEY,
    stage_labels=(
        "Study question",
        "Design",
        "Measurement",
        "Available artifacts",
        "Replication boundary",
    ),
)

# The palette, from the shared grammar.
INK = visuals.INK_MUTED
ACCENT = visuals.ACCENT
ACCENT_2 = visuals.ACCENT_2


# ---------------------------------------------------------------------
# The cluster randomization diagram
# ---------------------------------------------------------------------
#
# What makes this trial different from the comparison Impact Evaluation
# teaches is that the school was randomized, not the student. Drawn, that
# is a row of schools splitting into two arms with students inside each
# school, and the students never being assigned individually. In prose it
# is a sentence a reader skims.
#
# The school counts per arm are deliberately not drawn as 20 and 20. The
# publications report 40 schools in two arms; how many landed in each is
# not among the facts recorded in core/study.py, and drawing an even
# split would assert one.
_TRIAL_W, _TRIAL_H = 640.0, 320.0
_SCHOOL_R = 7.0

# One school, drawn as a box with its students inside it. Containment is
# the point: a student did not get assigned to an arm, their school did,
# and they came with it. Drawn as a caption saying "students within each
# school" that reads as a footnote; drawn as enclosure it is the fact.
_BOX_W, _BOX_H = 92.0, 50.0
_BOX_GAP = 14.0
_ARM_X = 150.0
_BOXES_PER_ARM = 4


def _label(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 13,
    color: str = INK,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" fill="{color}" '
        f'text-anchor="{anchor}">{text}</text>'
    )


def _school_pool(y: float, count: int) -> str:
    """The schools before assignment, as units."""
    spacing = 26.0

    return "".join(
        f'<circle cx="{8 + index * spacing:.0f}" cy="{y:.0f}" '
        f'r="{_SCHOOL_R}" fill="{INK}" fill-opacity="0.6"/>'
        for index in range(count)
    )


def _school_box(x: float, y: float, color: str) -> str:
    """One school, with students inside it."""
    students = "".join(
        f'<circle cx="{x + 18 + column * 14:.0f}" cy="{y + 30 + row * 12:.0f}" '
        f'r="3.4" fill="{color}" fill-opacity="0.8"/>'
        for row in range(2)
        for column in range(4)
    )

    return (
        f'<rect x="{x:.0f}" y="{y:.0f}" width="{_BOX_W:.0f}" '
        f'height="{_BOX_H:.0f}" rx="4" fill="none" stroke="{color}" '
        f'stroke-width="1.2"/>'
        f'<text x="{x + 8:.0f}" y="{y + 15:.0f}" font-size="10" '
        f'fill="{color}">school</text>' + students
    )


def _arm(y: float, name: str, color: str) -> str:
    """One trial arm: its schools, each holding its own students."""
    boxes = "".join(
        _school_box(_ARM_X + index * (_BOX_W + _BOX_GAP), y, color)
        for index in range(_BOXES_PER_ARM)
    )

    return (
        _label(0, y + 28, name, size=13, color=color)
        + boxes
        + _label(
            _ARM_X + _BOXES_PER_ARM * (_BOX_W + _BOX_GAP) - 6,
            y + 30,
            "...",
            size=13,
        )
    )


def _cluster_randomization_svg() -> str:
    """
    Randomization by school, and where the students sit inside it.

    Two things this has to get across that prose keeps failing to. The
    unit in the top row is a school, so the randomization is of forty
    things rather than of 1,752. And every student sits inside one of
    those boxes, so a student's arm was decided by their school.

    The number of schools per arm is deliberately not drawn as twenty and
    twenty. The publications report forty schools in two arms; how many
    landed in each is not among the facts core/study.py records, and an
    even split would assert one. The trailing ellipsis says the row is a
    sample of the arm rather than all of it.
    """
    split_y = 74.0
    intervention_y = 100.0
    control_y = 190.0

    inner = (
        _label(0, 20, "40 schools in Hyderabad", size=14)
        + _school_pool(44, 12)
        + _label(340, 48, "randomized as whole schools", size=12)
        # Down from the pool, then out to each arm.
        + f'<path d="M 8 58 V {split_y:.0f} H {_ARM_X - 20:.0f} '
        f'V {intervention_y + 25:.0f} H {_ARM_X - 6:.0f}" fill="none" '
        f'stroke="{INK}" stroke-width="1"/>'
        + f'<path d="M {_ARM_X - 20:.0f} {split_y:.0f} '
        f'V {control_y + 25:.0f} H {_ARM_X - 6:.0f}" fill="none" '
        f'stroke="{INK}" stroke-width="1"/>'
        + _arm(intervention_y, "Intervention arm", ACCENT_2)
        + _arm(control_y, "Control arm", ACCENT)
        + f'<line x1="0" y1="270" x2="640" y2="270" '
        f'stroke="{visuals.GRIDLINE}" stroke-width="1"/>'
        + _label(
            0,
            292,
            "1,752 grade 6 students, each inside a school, none "
            "individually randomized",
            size=13,
        )
    )

    return visuals.figure(
        inner,
        width=_TRIAL_W,
        height=_TRIAL_H,
        label=(
            "Forty schools in Hyderabad were randomized as whole schools "
            "into an intervention arm and a control arm. Each school is "
            "drawn as a box containing its own students, because the 1,752 "
            "grade 6 students were never randomized individually: a "
            "student's arm was decided by their school. The number of "
            "schools drawn per arm is illustrative; how the forty split "
            "between the arms is not among the facts this journey records."
        ),
    )


def _fact_table(facts, heading: str) -> None:
    """
    One group of facts, each badged with how it is known.

    The badge rather than the sentence. Printing "Reported by the
    publication" and a full citation under every row turned six facts
    into six paragraphs, and at that point a reader's question is only
    which of the three states this is. The citations move into one
    expander, where they are still one click from the fact they belong
    to.
    """
    st.markdown(f"**{heading}**")

    for fact in facts:
        badge_column, fact_column = st.columns([1, 4])
        with badge_column:
            st.badge(
                fact.badge,
                color="blue" if fact.is_established else "gray",
            )
        with fact_column:
            st.markdown(f"**{fact.label}**  \n{fact.value}")

    with st.expander("Sources for the above"):
        for fact in facts:
            st.caption(f"**{fact.label}**: {fact.provenance}. {fact.source}")


# The measurement map. Peer violence was measured as two separate things,
# being victimized and perpetrating, on two separate scales. Listed as
# three instruments that structure disappears; drawn, it is the first
# thing a reader sees.
def _measurement_map_svg(constructs) -> str:
    """
    Constructs, the facets they were split into, and the instruments.

    Facets are laid out across the whole width rather than inside their
    construct's box. Nesting them made two columns 110 pixels wide, and
    "Peer Victimization Scale" and "Peer Perpetration Scale" ran into
    each other; an instrument's real name is not something to shorten to
    fit a diagram.
    """
    total_facets = sum(len(construct.facets) for construct in constructs)
    slot = 640.0 / total_facets
    inner = ""
    facet_index = 0

    for construct in constructs:
        first = facet_index
        last = facet_index + len(construct.facets) - 1
        centre = slot * (first + last + 1) / 2
        box_half = min(slot * len(construct.facets) / 2 - 8, 150)

        inner += (
            f'<rect x="{centre - box_half:.0f}" y="8" '
            f'width="{box_half * 2:.0f}" height="38" rx="5" fill="none" '
            f'stroke="{ACCENT}" stroke-width="1.5"/>'
            + _label(
                centre, 33, construct.name, size=14, color=ACCENT, anchor="middle"
            )
        )

        for facet in construct.facets:
            x = slot * (facet_index + 0.5)
            inner += (
                f'<path d="M {centre:.0f} 46 V 62 H {x:.0f} V 78" '
                f'fill="none" stroke="{INK}" stroke-width="1"/>'
                + _label(x, 96, facet.name, size=13, anchor="middle")
                + f'<line x1="{x:.0f}" y1="108" x2="{x:.0f}" y2="128" '
                f'stroke="{INK}" stroke-width="1" '
                f'stroke-dasharray="{visuals.DASH_GUIDE}"/>'
                + _label(x, 148, facet.instrument, size=12, anchor="middle")
            )
            facet_index += 1

    inner += _label(
        320,
        184,
        "The dashed line is where a construct becomes an instrument",
        size=12,
        anchor="middle",
    )

    return visuals.figure(
        inner,
        width=640,
        height=200,
        label=(
            "What the study set out to measure, split into the aspects it "
            "was measured as, and the instrument standing in for each: "
            + "; ".join(
                f"{construct.name} as "
                + " and ".join(
                    f"{facet.name} on the {facet.instrument}"
                    for facet in construct.facets
                )
                for construct in constructs
            )
            + "."
        ),
    )


# The replication boundary. The payoff of the journey, and previously two
# columns of text.
#
# Drawn as one study splitting into what exists and what does not, and
# the two arriving at different places: the available artifacts support
# inspecting the study, and nothing supports recomputing its effect. The
# cross is the point, and it marks a boundary rather than a fault.
def _boundary_svg(boundary) -> str:
    """One published study, and where each half of it can be taken."""
    left, right = 160.0, 480.0
    row_step = 22.0
    first_row = 134.0

    def _column(x, heading, facts, *, established):
        colour = ACCENT if established else INK
        dash = (
            "" if established else f' stroke-dasharray="{visuals.DASH_UNESTABLISHED}"'
        )
        rows = "".join(
            _label(
                x,
                first_row + index * row_step,
                fact.for_diagram,
                size=12,
                anchor="middle",
            )
            for index, fact in enumerate(facts)
        )

        return (
            f'<path d="M 320 62 H {x:.0f} V 76" fill="none" '
            f'stroke="{INK}" stroke-width="1"{dash}/>'
            + f'<rect x="{x - 140:.0f}" y="76" width="280" height="34" rx="5" '
            f'fill="none" stroke="{colour}" stroke-width="1.5"{dash}/>'
            + _label(x, 99, heading, size=14, color=colour, anchor="middle")
            + rows
        )

    rows_end = first_row + max(
        len(boundary.established), len(boundary.unresolved)
    ) * row_step
    outcome_y = rows_end + 46

    inner = (
        _label(320, 26, "The published study", size=15, anchor="middle")
        + f'<path d="M 320 34 V 62" fill="none" stroke="{INK}" '
        f'stroke-width="1"/>'
        + _column(left, "Available", boundary.established, established=True)
        + _column(right, "Not available", boundary.unresolved, established=False)
        # Where each half can be taken. Solid down to what is supported;
        # dashed to a stop at what is not.
        + visuals.arrow(
            left, rows_end + 4, left, outcome_y - 18, established=True
        )
        + _label(
            left,
            outcome_y,
            "Inspect the study",
            size=14,
            color=ACCENT,
            anchor="middle",
        )
        + _label(left, outcome_y + 20, "supported", size=12, anchor="middle")
        + f'<line x1="{right:.0f}" y1="{rows_end + 4:.0f}" x2="{right:.0f}" '
        f'y2="{outcome_y - 30:.0f}" stroke="{INK}" stroke-width="1" '
        f'stroke-dasharray="{visuals.DASH_UNESTABLISHED}"/>'
        + f'<line x1="{right - 8:.0f}" y1="{outcome_y - 30:.0f}" '
        f'x2="{right + 8:.0f}" y2="{outcome_y - 16:.0f}" stroke="{INK}" '
        f'stroke-width="1.5"/>'
        + f'<line x1="{right + 8:.0f}" y1="{outcome_y - 30:.0f}" '
        f'x2="{right - 8:.0f}" y2="{outcome_y - 16:.0f}" stroke="{INK}" '
        f'stroke-width="1.5"/>'
        + _label(
            right, outcome_y, "Recompute the effect", size=14, anchor="middle"
        )
        + _label(right, outcome_y + 20, "not supported", size=12, anchor="middle")
    )

    return visuals.figure(
        inner,
        width=640,
        height=outcome_y + 38,
        label=(
            "The published study divides into artifacts that exist and "
            "artifacts that do not. What exists supports inspecting the "
            "study methodologically: "
            + ", ".join(fact.label for fact in boundary.established)
            + ". What is missing is what recomputing the reported effect "
            "would need, so that path stops: "
            + ", ".join(fact.label for fact in boundary.unresolved)
            + "."
        ),
    )


# ---------------------------------------------------------------------
# 1. Study question
# ---------------------------------------------------------------------

st.title("Right To Play")
st.subheader("Evaluating a School-Based Violence Prevention Program")
st.caption(
    "How a real published evaluation was built, and where its public "
    "artifacts stop."
)

st.divider()

stage = TRACKER.render_breadcrumb()
TRACKER.render_restart_button()

section_header("1. Study Question", "What this trial set out to find out")

dataset = get_dataset("right_to_play_baseline")

st.write(dataset.explore_question)

if stage < STAGE_DESIGN:
    if st.button("Continue to the design", type="primary"):
        TRACKER.advance_to(STAGE_DESIGN)
    st.stop()

# ---------------------------------------------------------------------
# 2. Design
# ---------------------------------------------------------------------

section_header("2. Design", "What was randomized, and what came with it")

st.markdown(_cluster_randomization_svg(), unsafe_allow_html=True)

inspect_note(
    "That the units in the top row are schools, not students. Randomizing "
    "40 schools is a much smaller randomization than randomizing 1,752 "
    "students, and it is the whole reason this trial is analysed "
    "differently."
)

_fact_table(study.DESIGN, "What is known about the design")

caveat(study.CLUSTERING_LIMIT)

if stage < STAGE_MEASUREMENT:
    if st.button("Continue to measurement", type="primary"):
        TRACKER.advance_to(STAGE_MEASUREMENT)
    st.stop()

# ---------------------------------------------------------------------
# 3. Measurement
# ---------------------------------------------------------------------

section_header(
    "3. Measurement", "The concepts, and the instruments standing in for them"
)

st.markdown(_measurement_map_svg(study.MEASUREMENT_MAP), unsafe_allow_html=True)

inspect_note(
    "That peer violence is two things, not one. Being victimized and "
    "perpetrating were measured on separate scales, which is why the "
    "trial reports them separately."
)

_fact_table(study.MEASUREMENT, "What was measured, and with what")

st.caption(
    "Children were asked about being victimized, about perpetrating "
    "violence, and about depression. The responses are de-identified and "
    "were published by the authors for reuse, and a reader should know "
    "that before opening them."
)

if stage < STAGE_ARTIFACTS:
    if st.button("Continue to the artifacts", type="primary"):
        TRACKER.advance_to(STAGE_ARTIFACTS)
    st.stop()

# ---------------------------------------------------------------------
# 4. Available artifacts
# ---------------------------------------------------------------------

section_header(
    "4. Available Artifacts", "What the study actually published"
)

st.caption(study.BASELINE_FILE_SUPPORT)

for source in dataset.sources:
    st.markdown(f"[{source.label}]({source.url})")

_fact_table(study.ARTIFACTS, "Each artifact, and whether it exists")

if stage < STAGE_BOUNDARY:
    if st.button("Continue to the replication boundary", type="primary"):
        TRACKER.advance_to(STAGE_BOUNDARY)
    st.stop()

# ---------------------------------------------------------------------
# 5. Replication boundary
# ---------------------------------------------------------------------

section_header(
    "5. Replication Boundary",
    "What the available artifacts support, and where that stops",
)

boundary = study.replication_boundary()

st.markdown(_boundary_svg(boundary), unsafe_allow_html=True)

st.markdown(f"**{boundary.lesson}**")

with st.expander("Every artifact, and which side it falls on"):
    established_column, unresolved_column = st.columns(2)

    with established_column:
        st.markdown("**Available**")
        for fact in boundary.established:
            st.markdown(f"- {fact.label}")
            st.caption(fact.value)

    with unresolved_column:
        st.markdown("**Not available, or incompletely reported**")
        for fact in boundary.unresolved:
            st.markdown(f"- {fact.label}")
            st.caption(fact.value)

caveat(
    "None of this is a criticism of the study. Publishing a baseline wave "
    "under CC BY is more than most trials do, and the boundary above is "
    "the ordinary state of published evaluation research rather than a "
    "failure particular to this one."
)

st.caption(f"Baseline: {study.BASELINE_CITATION}")
st.caption(f"Trial results: {study.TRIAL_CITATION}")
