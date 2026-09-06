"""
What a measure's data looks like, drawn from a small shared vocabulary.

The chronic-pain worked example had a good interaction that the
generalized planner lost: pick a measure, see a small picture of what it
produces, then read what it captures. The problem was never that the pain
example had it. The problem was that it was pain-specific, so every other
study got a list of names instead.

These are the primitives that make it general. A researcher choosing
between card sorting and causal mapping is choosing between clusters and
a directed graph, and seeing those two shapes is closer to seeing what
the choice means than reading two paragraphs is.

Each builder draws a shape, not an instrument. The signal trace is not an
ECG and the image is not a brain: they are what the data looks like, so a
measure added later reuses one rather than needing its own artwork. A new
primitive is added only when something genuinely produces a shape none of
these describe, because a vocabulary with one entry per measure is not a
vocabulary.

No streamlit import. These return SVG strings, the same as the rest of
shared/visuals.py's grammar, so they are testable without a running app.
"""

from __future__ import annotations

import math

from modules.research_design.core.ontology import (
    VISUAL_BODY_MAP,
    VISUAL_CARD_CLUSTER,
    VISUAL_CAUSAL_MAP,
    VISUAL_DISTRIBUTION,
    VISUAL_EVENT_TIMELINE,
    VISUAL_IMAGE,
    VISUAL_INVENTORY,
    VISUAL_RATING_SCALE,
    VISUAL_RECORD,
    VISUAL_SIGNAL_TRACE,
    VISUAL_TEXT,
    VISUAL_TYPES,
)
from shared import visuals

# Every primitive is drawn in the same box, so a row of measures reads as
# a row of alternatives rather than as differently sized pictures.
CARD_W, CARD_H = 150.0, 84.0

_INK = visuals.INK_MUTED
_ACCENT = visuals.ACCENT
_GRID = visuals.GRIDLINE


def _rating_scale() -> str:
    """A point on a line: an item score."""
    marks = "".join(
        f'<line x1="{20 + index * 22:.0f}" y1="46" '
        f'x2="{20 + index * 22:.0f}" y2="54" stroke="{_GRID}" '
        f'stroke-width="1.5"/>'
        for index in range(6)
    )

    return (
        f'<line x1="20" y1="50" x2="130" y2="50" stroke="{_INK}" '
        f'stroke-width="1"/>' + marks
        + f'<circle cx="86" cy="50" r="6" fill="{_ACCENT}"/>'
    )


def _body_map() -> str:
    """A region marked on a figure: where, not how much."""
    return (
        f'<ellipse cx="75" cy="26" rx="9" ry="10" fill="none" '
        f'stroke="{_INK}" stroke-width="1.2"/>'
        f'<path d="M 75 36 V 60 M 62 44 H 88 M 68 60 V 74 M 82 60 V 74" '
        f'fill="none" stroke="{_INK}" stroke-width="1.2"/>'
        f'<circle cx="66" cy="50" r="7" fill="{_ACCENT}" fill-opacity="0.5"/>'
    )


def _signal_trace() -> str:
    """A continuous line over time: a sensor stream."""
    points = " ".join(
        f"{20 + index * 5:.0f},"
        f"{50 - 18 * math.sin(index / 2.6) * math.cos(index / 7.0):.0f}"
        for index in range(23)
    )

    return (
        f'<polyline points="{points}" fill="none" stroke="{_ACCENT}" '
        f'stroke-width="1.6"/>'
        f'<line x1="20" y1="70" x2="130" y2="70" stroke="{_GRID}" '
        f'stroke-width="1"/>'
    )


def _event_timeline() -> str:
    """Marks at moments: something happened, and when."""
    offsets = (0, 14, 40, 52, 78, 96)

    return (
        f'<line x1="20" y1="50" x2="130" y2="50" stroke="{_GRID}" '
        f'stroke-width="1"/>'
        + "".join(
            f'<line x1="{22 + offset:.0f}" y1="38" x2="{22 + offset:.0f}" '
            f'y2="50" stroke="{_ACCENT}" stroke-width="2"/>'
            for offset in offsets
        )
    )


def _card_cluster() -> str:
    """Items grouped: which things a person put together."""
    groups = ((44, 40, 3), (86, 34, 2), (100, 62, 3))

    return "".join(
        visuals.unit_cluster(x, y, _ACCENT, count=count, radius=5)
        for x, y, count in groups
    ) + (
        f'<ellipse cx="44" cy="40" rx="22" ry="18" fill="none" '
        f'stroke="{_GRID}" stroke-width="1"/>'
        f'<ellipse cx="93" cy="48" rx="28" ry="24" fill="none" '
        f'stroke="{_GRID}" stroke-width="1"/>'
    )


def _causal_map() -> str:
    """Arrows between things: what a person believes affects what."""
    nodes = ((40, 32), (105, 32), (40, 66), (105, 66))
    circles = "".join(
        f'<circle cx="{x}" cy="{y}" r="8" fill="none" stroke="{_ACCENT}" '
        f'stroke-width="1.4"/>'
        for x, y in nodes
    )

    # No zero-length arrows. visuals.arrow draws its head at the end
    # point, so a call whose start and end match renders a stray
    # arrowhead floating with no line under it.
    return circles + (
        visuals.arrow(50, 32, 95, 32)
        + f'<path d="M 40 42 V 58" stroke="{_INK}" stroke-width="1" '
        f'fill="none"/>'
        f'<polygon points="40,58 37,52 43,52" fill="{_INK}"/>'
        f'<path d="M 105 42 V 58" stroke="{_INK}" stroke-width="1" '
        f'fill="none"/>'
        f'<polygon points="105,58 102,52 108,52" fill="{_INK}"/>'
        + visuals.arrow(50, 66, 95, 66)
    )


def _distribution() -> str:
    """A spread of values: many measurements of one quantity."""
    heights = (8, 16, 26, 34, 30, 20, 11, 6)

    return "".join(
        f'<rect x="{22 + index * 14:.0f}" y="{68 - height:.0f}" width="10" '
        f'height="{height}" fill="{_ACCENT}" fill-opacity="0.55"/>'
        for index, height in enumerate(heights)
    ) + (
        f'<line x1="20" y1="68" x2="130" y2="68" stroke="{_INK}" '
        f'stroke-width="1"/>'
    )


def _inventory() -> str:
    """A checklist: which of a set of things applies."""
    rows = ((30, True), (44, True), (58, False), (72, True))

    return "".join(
        f'<rect x="40" y="{y - 8:.0f}" width="11" height="11" rx="2" '
        f'fill="none" stroke="{_INK}" stroke-width="1.2"/>'
        + (
            f'<path d="M 42.5 {y - 2.5:.0f} L 45 {y + 0.5:.0f} '
            f'L 48.5 {y - 5.5:.0f}" fill="none" stroke="{_ACCENT}" '
            f'stroke-width="2"/>'
            if ticked
            else ""
        )
        + f'<line x1="58" y1="{y - 2:.0f}" x2="112" y2="{y - 2:.0f}" '
        f'stroke="{_GRID}" stroke-width="2"/>'
        for y, ticked in rows
    )


def _record() -> str:
    """Fields in rows: something a system already wrote down."""
    return (
        f'<rect x="30" y="22" width="90" height="52" rx="3" fill="none" '
        f'stroke="{_INK}" stroke-width="1.2"/>'
        f'<line x1="30" y1="36" x2="120" y2="36" stroke="{_INK}" '
        f'stroke-width="1"/>'
        f'<line x1="66" y1="22" x2="66" y2="74" stroke="{_GRID}" '
        f'stroke-width="1"/>'
        + "".join(
            f'<line x1="72" y1="{y}" x2="112" y2="{y}" stroke="{_ACCENT}" '
            f'stroke-width="2" stroke-opacity="0.6"/>'
            for y in (46, 58, 68)
        )
    )


def _image() -> str:
    """A field of values in space: a scan or a slice."""
    cells = "".join(
        f'<rect x="{46 + column * 12:.0f}" y="{26 + row * 12:.0f}" '
        f'width="11" height="11" fill="{_ACCENT}" '
        f'fill-opacity="{0.15 + 0.16 * ((row * 5 + column) % 5):.2f}"/>'
        for row in range(4)
        for column in range(5)
    )

    return cells + (
        f'<rect x="46" y="26" width="60" height="48" fill="none" '
        f'stroke="{_INK}" stroke-width="1.2"/>'
    )


def _text() -> str:
    """Lines of language: what someone said or wrote."""
    widths = (72, 88, 60, 80, 44)

    return "".join(
        f'<line x1="34" y1="{26 + index * 11:.0f}" '
        f'x2="{34 + width:.0f}" y2="{26 + index * 11:.0f}" '
        f'stroke="{_INK if index else _ACCENT}" stroke-width="2.5" '
        f'stroke-opacity="{1.0 if index == 0 else 0.4}"/>'
        for index, width in enumerate(widths)
    )


_BUILDERS = {
    VISUAL_RATING_SCALE: _rating_scale,
    VISUAL_BODY_MAP: _body_map,
    VISUAL_SIGNAL_TRACE: _signal_trace,
    VISUAL_EVENT_TIMELINE: _event_timeline,
    VISUAL_CARD_CLUSTER: _card_cluster,
    VISUAL_CAUSAL_MAP: _causal_map,
    VISUAL_DISTRIBUTION: _distribution,
    VISUAL_INVENTORY: _inventory,
    VISUAL_RECORD: _record,
    VISUAL_IMAGE: _image,
    VISUAL_TEXT: _text,
}


def measure_visual_svg(measure) -> str:
    """
    A small picture of what this measure's data looks like.

    Raises on a visual type with no builder rather than drawing an empty
    box, so a primitive added to the vocabulary without a drawing fails
    where it is added instead of rendering as a gap in a row of measures.
    """
    builder = _BUILDERS.get(measure.visual_type)

    if builder is None:
        raise ValueError(
            f"'{measure.visual_type}' is in the vocabulary and has no "
            f"builder, so {measure.name} cannot be drawn. Add one in "
            "shared/measure_visuals.py."
        )

    return visuals.figure(
        builder(),
        width=CARD_W,
        height=CARD_H,
        label=(
            f"{measure.name}: {measure.produces.lower()}, shown as a "
            f"{measure.visual_type.replace('_', ' ')}."
        ),
    )


def builders_cover_the_vocabulary() -> bool:
    """Whether every declared primitive can actually be drawn."""
    return set(_BUILDERS) == set(VISUAL_TYPES)
