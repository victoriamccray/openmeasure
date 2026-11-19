"""
OpenMeasure's visual grammar: one palette, and the primitives every page
draws from.

The grammar decides which form a thing takes, not merely how it looks:

    Entities              -> pictographs
    Structure             -> diagrams
    Quantities            -> charts
    Analytical changes    -> animation, and only when a reader causes one
    Evidence and provenance states -> visual states

This module holds the parts of that grammar that are not specific to any
one statistic: the colours, the dash patterns, a unit pictograph, an
arrow, and the sized wrapper the rest are drawn into. Builders that
describe one particular thing stay with the page that owns them, because
a pulse-sequence diagram, a support boundary, or a
difference-in-differences chart is not a primitive. They move here when a
second page needs one, not before.

It exists because the same pieces were being rewritten. The palette below
was duplicated in twelve pages, and arrows and unit clusters had been
independently reinvented in at least three. A shared vocabulary is what
makes the grammar a system rather than a description of one page.

No streamlit import. These return SVG strings for a caller to pass to
st.markdown, which keeps them testable without a running app, the same
reason shared/charts.py returns a spec rather than drawing one.

Two conventions carried by every primitive here:

- **Solid means established, dashed means not.** A dashed line is an
  assumed counterfactual, an unverified claim, or a value nobody has
  confirmed. It is the same mark in each case, so a reader learns it
  once.
- **Colour is not the signal for doubt.** Something unresolved is not a
  failure, and red would assert that it was. State is carried by stroke
  and shape, leaving colour free to distinguish groups.
"""

from __future__ import annotations

# The palette, previously copied into twelve page files. Muted on
# purpose: these sit next to body text, and a saturated chart pulls the
# eye away from the number it is supposed to explain.
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
ACCENT = "#2a78d6"
ACCENT_2 = "#c0392b"

# For a value that is absent rather than low. Distinct from GRIDLINE so a
# gap does not read as structure.
MISSING = "#d8d6cd"

# Dash patterns, named so "not established" is one decision rather than a
# number retyped per call site.
DASH_UNESTABLISHED = "4,3"
DASH_GUIDE = "3,3"


def unit_cluster(
    x: float,
    y: float,
    color: str,
    *,
    count: int = 5,
    radius: float = 3.1,
    opacity: float = 0.75,
) -> str:
    """
    A group of units, drawn as a small cluster.

    Entities get pictographs: a group of people, records or observations
    should look countable rather than be reduced to a number beside a
    label. Offsets are fixed rather than random so the same group drawn
    twice looks the same, and a reader moving a control does not see the
    cluster reshuffle and read that as the data changing.
    """
    # Offsets are expressed in units of the radius, so a cluster keeps
    # its shape at any size. Tuning them for one radius made the
    # primitive usable at exactly one scale, which is the opposite of
    # what a primitive is for: at pictograph size the units overlapped.
    offsets = (
        (-2.3, -2.3), (2.3, -1.9), (0.0, 0.0), (-2.6, 2.3), (2.6, 2.6),
        (-1.0, -3.9), (3.5, 0.3), (-3.9, 0.0), (1.3, 4.2), (3.9, -3.5),
    )

    if not 1 <= count <= len(offsets):
        raise ValueError(
            f"count must be between 1 and {len(offsets)}, got {count}. A "
            "pictograph of more units than that stops being countable, "
            "which is the only reason to draw one."
        )

    return "".join(
        f'<circle cx="{x + dx * radius:.1f}" cy="{y + dy * radius:.1f}" '
        f'r="{radius}" fill="{color}" fill-opacity="{opacity}"/>'
        for dx, dy in offsets[:count]
    )


def arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = INK_MUTED,
    established: bool = True,
    head: float = 4.0,
) -> str:
    """
    A left-to-right arrow, solid when established and dashed when not.

    Only horizontal arrows carry a head here, which is every use so far;
    a diagonal one would need the head rotated and is worth adding when
    something actually needs it rather than before.
    """
    dash = "" if established else f' stroke-dasharray="{DASH_UNESTABLISHED}"'
    line = (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="1"{dash}/>'
    )
    point = (
        f'<polygon points="{x2:.1f},{y2:.1f} {x2 - head * 1.8:.1f},'
        f'{y2 - head:.1f} {x2 - head * 1.8:.1f},{y2 + head:.1f}" '
        f'fill="{color}"/>'
    )

    return line + point


def figure(inner: str, *, width: float, height: float, label: str) -> str:
    """
    Wrap primitives in a sized, labelled SVG.

    label becomes the accessible description, and is required: a figure
    nobody can read without seeing it is not finished. Callers pass a
    sentence describing what the figure shows, not a title.
    """
    if not label.strip():
        raise ValueError(
            "A figure needs a label describing what it shows, so it is "
            "readable without being seen."
        )

    # Sized by the viewBox aspect ratio rather than a pixel height. A
    # fixed height alongside width="100%" and preserveAspectRatio="meet"
    # scales the drawing by min(container/viewBox_w, height/viewBox_h),
    # which pins it to 1:1 whenever the height matches the viewBox: the
    # art renders at native size, centred in a much wider empty box, with
    # hairline strokes. Letting height follow the aspect ratio lets a
    # figure fill the column it is given, so strokes and labels grow with
    # it instead of staying small in a large space.
    return (
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" '
        f'style="width:100%;height:auto;display:block" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'aria-label="{label}">{inner}</svg>'
    )
