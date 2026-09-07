"""
What a measure's data looks like, drawn as the thing it actually is.

An earlier version of this module drew shapes rather than instruments. One
line served electrodermal activity, heart rate and an air-quality sensor;
one histogram served a Delphi round, a blood assay and a held-out test
set. It was a small tidy vocabulary and it hid the choice it was there to
support. A researcher deciding between EDA and ECG is not picking between
two instances of a line: one is a slow tonic level interrupted by phasic
responses, the other is a beat-by-beat waveform whose intervals are
themselves the measurement. Drawn honestly they look nothing alike, and
that difference is the decision.

So these are specific. An anterior body outline, an axial brain slice with
a thresholded cluster, a P-QRS-T complex with visible R-R intervals, a
rack of assay tubes at different concentrations. A type is still shared
where the artifact really is the same thing, which is why one record table
serves administrative extracts, clinical charts and earnings records, and
one transcript serves interviews, think-aloud protocols and document
analysis.

Three of them move. A signal is a thing happening over time, and a still
line asks the reader to supply that themselves; a travelling highlight
along the trace says it instead. The motion is explanatory rather than
decorative: it appears only on the three continuously sampled measures,
where "this arrives as a stream" is the property that separates them from
everything else here. It respects prefers-reduced-motion, and the static
trace underneath carries the whole drawing on its own, so nothing is lost
when the animation is off.

No streamlit import. These return SVG strings, the same as the rest of
shared/visuals.py's grammar, so they are testable without a running app.
"""

from __future__ import annotations

import math

from modules.research_design.core.ontology import (
    VISUAL_ASSAY,
    VISUAL_BODY_MAP,
    VISUAL_BRAIN_SLICE,
    VISUAL_CARD_CLUSTER,
    VISUAL_CAUSAL_MAP,
    VISUAL_DIARY_GRID,
    VISUAL_ECG,
    VISUAL_EVENT_TIMELINE,
    VISUAL_HELD_OUT_SPLIT,
    VISUAL_INVENTORY,
    VISUAL_RATING_SCALE,
    VISUAL_RATIO_BAR,
    VISUAL_RECORD,
    VISUAL_ROUND_CONVERGENCE,
    VISUAL_SIGNAL_TRACE,
    VISUAL_SKIN_CONDUCTANCE,
    VISUAL_TEXT,
    VISUAL_TYPES,
)
from shared import visuals

# Every drawing is made in the same box, so a row of measures reads as a
# row of alternatives rather than as differently sized pictures. Taller
# than it once was: an anatomical figure and a brain slice need vertical
# room, and shrinking them to fit a letterbox is what made the previous
# set look like clip art.
CARD_W, CARD_H = 150.0, 104.0

_INK = visuals.INK_MUTED
_ACCENT = visuals.ACCENT
_ACCENT_2 = visuals.ACCENT_2
_GRID = visuals.GRIDLINE

# A short bright segment travelling along a trace, on a dash cycle fixed
# at 336 units so the keyframes are identical for every figure. Path
# length varies; the cycle does not, which keeps this one stylesheet
# rather than one per drawing. Repeated verbatim in each animated SVG,
# which is safe precisely because it is identical: inline SVG styles are
# document-scoped, so a per-figure value here would leak between figures.
_SWEEP_STYLE = (
    "<style>"
    "@keyframes om-measure-sweep"
    "{from{stroke-dashoffset:0}to{stroke-dashoffset:-336}}"
    ".om-measure-sweep{animation:om-measure-sweep 3.2s linear infinite}"
    "@media (prefers-reduced-motion:reduce)"
    "{.om-measure-sweep{display:none}}"
    "</style>"
)


def _points(values) -> str:
    """An x,y sequence as a polyline attribute."""
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in values)


def _trace(values, *, stroke: str = _ACCENT) -> str:
    """
    A sampled signal: the trace, and a highlight running along it.

    The highlight is a second copy of the same geometry rather than a
    moving dot, so the motion follows the data instead of hovering over
    it.
    """
    geometry = _points(values)

    return (
        _SWEEP_STYLE
        + f'<polyline points="{geometry}" fill="none" stroke="{stroke}" '
        f'stroke-width="1.6" stroke-linejoin="round"/>'
        + f'<polyline points="{geometry}" fill="none" stroke="{stroke}" '
        f'stroke-width="3" stroke-linecap="round" '
        f'stroke-dasharray="16 320" class="om-measure-sweep"/>'
    )


def _axis(y: float, *, x0: float = 22, x1: float = 132) -> str:
    """The line a trace sits on."""
    return (
        f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{_GRID}" '
        f'stroke-width="1"/>'
    )


def _rating_scale() -> str:
    """
    One item from a scale: a stem, five options, one chosen.

    Drawn as the thing a respondent sees rather than as a point on a
    line, because what a rating scale costs and misses is a property of
    the item, not of the number it returns.
    """
    stem = "".join(
        f'<line x1="30" y1="{26 + index * 9}" x2="{30 + width}" '
        f'y2="{26 + index * 9}" stroke="{_INK}" stroke-width="2.5" '
        f'stroke-opacity="0.45"/>'
        for index, width in enumerate((88, 62))
    )

    options = "".join(
        f'<circle cx="{34 + index * 21}" cy="66" r="6.5" fill="none" '
        f'stroke="{_INK}" stroke-width="1.3"/>'
        + (
            f'<circle cx="{34 + index * 21}" cy="66" r="3.6" '
            f'fill="{_ACCENT}"/>'
            if index == 3
            else ""
        )
        for index in range(5)
    )

    anchors = "".join(
        f'<line x1="{x}" y1="80" x2="{x + 20}" y2="80" stroke="{_GRID}" '
        f'stroke-width="2"/>'
        for x in (24, 100)
    )

    return stem + options + anchors


def _diary_grid() -> str:
    """
    Prompts by day, answered or missed.

    A repeated rating is not one rating. Its shape is a grid with holes
    in it, and the holes are the measurement problem: compliance decides
    what the series can support.
    """
    answered = {
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 1), (3, 2),
        (4, 0), (4, 1), (4, 2), (4, 3),
        (5, 0), (5, 2), (5, 3),
        (6, 0), (6, 1), (6, 2),
    }

    marks = []
    for day in range(7):
        for prompt in range(4):
            x = 32 + day * 14.5
            y = 32 + prompt * 14

            if (day, prompt) in answered:
                marks.append(
                    f'<circle cx="{x}" cy="{y}" r="4.6" fill="{_ACCENT}" '
                    f'fill-opacity="0.75"/>'
                )
            else:
                marks.append(
                    f'<circle cx="{x}" cy="{y}" r="4.6" fill="none" '
                    f'stroke="{visuals.MISSING}" stroke-width="1.4"/>'
                )

    days = "".join(
        f'<line x1="{32 + day * 14.5}" y1="94" x2="{32 + day * 14.5}" '
        f'y2="98" stroke="{_INK}" stroke-width="1.2"/>'
        for day in range(7)
    )

    return (
        "".join(marks)
        + f'<line x1="26" y1="88" x2="123" y2="88" stroke="{_GRID}" '
        f'stroke-width="1"/>' + days
    )


# An anterior outline, as a right half traced neck to crotch and then
# mirrored. Written as coordinates rather than as one long path string
# because a figure that is symmetric by construction cannot drift.
_BODY_HALF = (
    (4, 27), (14, 32), (20, 36),                      # neck, shoulder
    (22, 44), (23, 55), (22, 65), (21, 73), (22, 79),  # arm, outer edge
    (17, 79), (16, 71), (16, 59), (15, 46),            # arm, inner edge
    (13, 38), (11, 50), (14, 58),                      # armpit, waist, hip
    (13, 70), (11, 84), (10, 96), (12, 99),            # leg, outer edge
    (5, 99), (4, 86), (3, 72), (0, 62),                # leg, inner edge
)


def _body_map() -> str:
    """
    A region marked on a body: where it hurts, and how much.

    Two shaded areas at different intensities, because a pain drawing
    records extent and severity together and a single blob would drop
    half of that.
    """
    right = [(75 + dx, y) for dx, y in _BODY_HALF]
    left = [(75 - dx, y) for dx, y in reversed(_BODY_HALF[:-1])]
    outline = " L ".join(f"{x:.1f} {y:.1f}" for x, y in right + left)

    regions = (
        f'<ellipse cx="87" cy="41" rx="7" ry="6" fill="{_ACCENT_2}" '
        f'fill-opacity="0.45"/>'
        f'<ellipse cx="75" cy="55" rx="9" ry="6" fill="{_ACCENT}" '
        f'fill-opacity="0.3"/>'
    )

    return (
        f'<ellipse cx="75" cy="15" rx="8.5" ry="10" fill="none" '
        f'stroke="{_INK}" stroke-width="1.3"/>'
        f'<path d="M {outline} Z" fill="none" stroke="{_INK}" '
        f'stroke-width="1.3" stroke-linejoin="round"/>' + regions
    )


def _ecg() -> str:
    """
    Beats, and the intervals between them.

    A P-QRS-T complex at four different R-R intervals, with the
    intervals bracketed above. Variability is the measurement, so the
    drawing has to show unequal spacing rather than a tidy repeat.
    """
    baseline = 66.0
    # Fractions of one beat's width, as (position, height above baseline).
    beat = (
        (0.00, 0), (0.12, 0), (0.16, 3), (0.20, 4.5), (0.24, 3), (0.28, 0),
        (0.34, 0), (0.36, -3), (0.40, 30), (0.44, -9), (0.47, 0),
        (0.55, 0), (0.62, 7), (0.68, 8), (0.74, 4), (0.80, 0), (1.00, 0),
    )
    widths = (30.0, 26.0, 33.0, 28.0)

    samples: list[tuple[float, float]] = []
    peaks: list[float] = []
    x = 16.0

    for width in widths:
        peaks.append(x + 0.40 * width)
        samples.extend(
            (x + share * width, baseline - height) for share, height in beat
        )
        x += width

    brackets = "".join(
        f'<line x1="{start}" y1="22" x2="{end}" y2="22" stroke="{_INK}" '
        f'stroke-width="1" stroke-opacity="0.7"/>'
        f'<line x1="{start}" y1="19" x2="{start}" y2="25" '
        f'stroke="{_INK}" stroke-width="1"/>'
        f'<line x1="{end}" y1="19" x2="{end}" y2="25" stroke="{_INK}" '
        f'stroke-width="1"/>'
        for start, end in zip(peaks, peaks[1:])
    )

    return brackets + _axis(88, x0=16, x1=133) + _trace(samples)


def _skin_conductance() -> str:
    """
    A tonic level, and phasic responses on top of it.

    The dashed line is the tonic baseline. Each response rises in about
    a second and recovers over many, which is why this measure needs a
    long enough window to let one finish before the next begins.
    """
    events = ((38.0, 17.0), (68.0, 23.0), (100.0, 13.0))
    rise, decay = 6.0, 15.0

    def tonic(x: float) -> float:
        return 76.0 - 0.09 * (x - 22.0)

    def level(x: float) -> float:
        height = 0.0

        for onset, amplitude in events:
            offset = x - onset

            if offset < 0:
                continue
            if offset < rise:
                height += amplitude * offset / rise
            else:
                height += amplitude * math.exp(-(offset - rise) / decay)

        return tonic(x) - height

    samples = [(x, level(float(x))) for x in range(22, 134)]
    guide = (
        f'<line x1="22" y1="{tonic(22.0):.1f}" x2="132" '
        f'y2="{tonic(132.0):.1f}" stroke="{_INK}" stroke-width="1" '
        f'stroke-dasharray="{visuals.DASH_GUIDE}"/>'
    )

    return _axis(92) + guide + _trace(samples)


def _signal_trace() -> str:
    """
    An ambient stream from a fixed sensor, over days.

    The shaded bands are night. A sensor measures the place rather than
    the person, so its structure is the cycle of the site, and a study
    that samples across only part of one is sampling a phase.
    """
    def level(x: float) -> float:
        cycle = math.sin((x - 30.0) / 17.0)
        ripple = 0.25 * math.sin((x - 30.0) / 3.3)

        return 60.0 - 20.0 * cycle - 4.0 * ripple

    # Placed on the troughs of the cycle above, so the shading explains
    # the dip rather than sitting behind it at a regular interval.
    nights = "".join(
        f'<rect x="{x0}" y="24" width="{x1 - x0}" height="64" '
        f'fill="{_GRID}" fill-opacity="0.55"/>'
        for x0, x1 in ((22, 34), (99, 121))
    )

    samples = [(x, level(float(x))) for x in range(22, 134)]

    return nights + _axis(88) + _trace(samples)


def _brain_slice() -> str:
    """
    An axial slice with one thresholded cluster.

    Wider than tall, with a gyral edge and the interhemispheric fissure
    showing at the front and back rather than straight through: this is
    the view a reader has seen before, and a smooth oval is not it.

    The cluster is drawn in graded rings because that is what a
    statistical map is, a threshold applied to a continuous field, so
    moving the threshold moves the blob.
    """
    def contour(scale: float, gyri: float) -> str:
        steps = 96
        points = []

        for step in range(steps + 1):
            angle = 2.0 * math.pi * step / steps
            wobble = gyri * math.sin(12.0 * angle)
            points.append(
                (
                    75.0 + (37.0 * scale + wobble) * math.sin(angle),
                    54.0 - (30.0 * scale + wobble) * math.cos(angle),
                )
            )

        return _points(points)

    return (
        f'<polygon points="{contour(1.0, 0.9)}" fill="{_GRID}" '
        f'fill-opacity="0.45" stroke="{_INK}" stroke-width="1.3"/>'
        f'<polygon points="{contour(0.74, 0.5)}" fill="none" '
        f'stroke="{_INK}" stroke-width="0.8" stroke-opacity="0.3"/>'
        f'<line x1="75" y1="22" x2="75" y2="34" stroke="{_INK}" '
        f'stroke-width="1.1"/>'
        f'<line x1="75" y1="74" x2="75" y2="86" stroke="{_INK}" '
        f'stroke-width="1.1"/>'
        f'<path d="M 72 45 C 68 51 68 59 72 64 C 73.5 58 73.5 51 72 45 Z" '
        f'fill="{_INK}" fill-opacity="0.35"/>'
        f'<path d="M 78 45 C 82 51 82 59 78 64 C 76.5 58 76.5 51 78 45 Z" '
        f'fill="{_INK}" fill-opacity="0.35"/>'
        + "".join(
            f'<ellipse cx="93" cy="62" rx="{rx}" ry="{ry}" '
            f'fill="{_ACCENT}" fill-opacity="{opacity}"/>'
            for rx, ry, opacity in (
                (10, 8, 0.18), (6.8, 5.4, 0.45), (3.6, 2.9, 0.8)
            )
        )
    )


def _assay() -> str:
    """
    Tubes at different concentrations: one value per sample.

    Three rather than one, because an assay's unit is the sample and its
    problem is comparability between them.
    """
    levels = (0.62, 0.34, 0.78)
    tubes = []

    for index, share in enumerate(levels):
        x = 46.0 + index * 26.0
        top, bottom = 24.0, 84.0
        surface = bottom - share * (bottom - top)

        tubes.append(
            f'<path d="M {x} {surface} V {bottom} A 8 8 0 0 0 {x + 16} '
            f'{bottom} V {surface} Z" fill="{_ACCENT}" fill-opacity="0.45"/>'
            f'<path d="M {x} 22 V {bottom} A 8 8 0 0 0 {x + 16} {bottom} '
            f'V 22" fill="none" stroke="{_INK}" stroke-width="1.3"/>'
            f'<line x1="{x - 1}" y1="22" x2="{x + 17}" y2="22" '
            f'stroke="{_INK}" stroke-width="1.6"/>'
            + "".join(
                f'<line x1="{x + 11}" y1="{34 + step * 12}" '
                f'x2="{x + 16}" y2="{34 + step * 12}" stroke="{_INK}" '
                f'stroke-width="0.9" stroke-opacity="0.6"/>'
                for step in range(4)
            )
        )

    return "".join(tubes)


def _event_timeline() -> str:
    """
    Coded behaviours in lanes, with durations.

    Bars rather than ticks: a coding scheme that records onset and
    offset supports different questions from one that records instants,
    and the drawing should not blur the two.
    """
    lanes = (
        (34, ((30, 18), (62, 9), (86, 22))),
        (56, ((40, 12), (70, 30))),
        (78, ((32, 8), (52, 14), (78, 10), (104, 18))),
    )

    drawn = []
    for y, spans in lanes:
        drawn.append(
            f'<line x1="26" y1="{y}" x2="126" y2="{y}" stroke="{_GRID}" '
            f'stroke-width="1"/>'
        )
        drawn.extend(
            f'<rect x="{start}" y="{y - 5}" width="{length}" height="10" '
            f'rx="2" fill="{_ACCENT}" fill-opacity="0.6"/>'
            for start, length in spans
        )

    ticks = "".join(
        f'<line x1="{26 + step * 25}" y1="90" x2="{26 + step * 25}" '
        f'y2="95" stroke="{_INK}" stroke-width="1.1"/>'
        for step in range(5)
    )

    return "".join(drawn) + (
        f'<line x1="26" y1="90" x2="126" y2="90" stroke="{_INK}" '
        f'stroke-width="1"/>' + ticks
    )


def _card() -> str:
    """One card, as a template positioned by transform."""
    return (
        f'<rect width="22" height="15" rx="2.5" fill="#ffffff" '
        f'stroke="{_INK}" stroke-width="1.1"/>'
        f'<line x1="4" y1="6" x2="16" y2="6" stroke="{_INK}" '
        f'stroke-width="1.5" stroke-opacity="0.5"/>'
        f'<line x1="4" y1="10" x2="12" y2="10" stroke="{_INK}" '
        f'stroke-width="1.5" stroke-opacity="0.3"/>'
    )


def _card_cluster() -> str:
    """
    Cards a person put into piles.

    Drawn as cards, because the deck is the constraint: a concept that
    is not on one cannot appear in the result, and an abstract cluster
    of dots does not say that.
    """
    piles = (
        ((30, 30, -6), (36, 40, 3), (32, 51, -2)),
        ((88, 26, 4), (94, 37, -5)),
        ((84, 66, -3), (92, 76, 5)),
    )

    lassos = (
        (44, 47, 25, 24),
        (101, 39, 23, 19),
        (97, 79, 22, 19),
    )

    cards = "".join(
        f'<g transform="translate({x} {y}) rotate({angle} 11 7.5)">'
        + _card()
        + "</g>"
        for pile in piles
        for x, y, angle in pile
    )

    return (
        "".join(
            f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" '
            f'stroke="{_ACCENT}" stroke-width="1" stroke-opacity="0.5" '
            f'stroke-dasharray="{visuals.DASH_GUIDE}"/>'
            for cx, cy, rx, ry in lassos
        )
        + cards
    )


def _causal_map() -> str:
    """Arrows between things: what a person believes affects what."""
    nodes = ((44, 34), (106, 34), (44, 76), (106, 76))
    circles = "".join(
        f'<circle cx="{x}" cy="{y}" r="9" fill="#ffffff" stroke="{_ACCENT}" '
        f'stroke-width="1.4"/>'
        for x, y in nodes
    )

    # No zero-length arrows. visuals.arrow draws its head at the end
    # point, so a call whose start and end match renders a stray
    # arrowhead floating with no line under it.
    return circles + (
        visuals.arrow(55, 34, 95, 34)
        + f'<path d="M 44 45 V 63" stroke="{_INK}" stroke-width="1" '
        f'fill="none"/>'
        f'<polygon points="44,65 41,58 47,58" fill="{_INK}"/>'
        f'<path d="M 106 45 V 63" stroke="{_INK}" stroke-width="1" '
        f'fill="none"/>'
        f'<polygon points="106,65 103,58 109,58" fill="{_INK}"/>'
        + visuals.arrow(55, 76, 95, 76)
    )


def _round_convergence() -> str:
    """
    Three rounds, each narrower than the last.

    The narrowing is the artifact a Delphi produces, and it is also its
    limitation: a group can converge on the procedure rather than on an
    answer, and the drawing cannot tell those apart either.
    """
    rounds = (
        (32, (28, 44, 58, 71, 84, 97, 112, 124)),
        (58, (46, 58, 70, 79, 90, 104)),
        (84, (64, 72, 79, 86, 96)),
    )

    drawn = []
    for y, positions in rounds:
        drawn.append(
            f'<line x1="{min(positions) - 4}" y1="{y}" '
            f'x2="{max(positions) + 4}" y2="{y}" stroke="{_GRID}" '
            f'stroke-width="1.4"/>'
        )
        drawn.extend(
            f'<circle cx="{x}" cy="{y}" r="3.4" fill="{_ACCENT}" '
            f'fill-opacity="0.6"/>'
            for x in positions
        )
        median = sorted(positions)[len(positions) // 2]
        drawn.append(
            f'<line x1="{median}" y1="{y - 8}" x2="{median}" '
            f'y2="{y + 8}" stroke="{_INK}" stroke-width="1.6"/>'
        )

    return "".join(drawn)


def _inventory() -> str:
    """A checklist: which of a set of things applies."""
    rows = ((32, True), (49, True), (66, False), (83, True))

    return "".join(
        f'<rect x="34" y="{y - 8}" width="13" height="13" rx="2.5" '
        f'fill="none" stroke="{_INK}" stroke-width="1.2"/>'
        + (
            f'<path d="M 37 {y - 1.5} L 40 {y + 2} L 44.5 {y - 5}" '
            f'fill="none" stroke="{_ACCENT}" stroke-width="2.2"/>'
            if ticked
            else ""
        )
        + f'<line x1="54" y1="{y - 1.5}" x2="{116 if ticked else 104}" '
        f'y2="{y - 1.5}" stroke="{_INK}" stroke-width="2.5" '
        f'stroke-opacity="0.35"/>'
        for y, ticked in rows
    )


def _ratio_bar() -> str:
    """
    A part against the whole it is measured against.

    A debt figure on its own is not a burden. The burden is the ratio,
    so both bars are drawn and the dashed line carries the one across
    the other.
    """
    x0, x1 = 30.0, 124.0
    share = 0.38
    edge = x0 + share * (x1 - x0)

    return (
        f'<rect x="{x0}" y="34" width="{x1 - x0}" height="14" rx="2" '
        f'fill="{_GRID}" fill-opacity="0.7" stroke="{_INK}" '
        f'stroke-width="1.1"/>'
        f'<rect x="{x0}" y="66" width="{edge - x0}" height="14" rx="2" '
        f'fill="{_ACCENT}" fill-opacity="0.55" stroke="{_ACCENT}" '
        f'stroke-width="1.1"/>'
        f'<rect x="{x0}" y="66" width="{x1 - x0}" height="14" rx="2" '
        f'fill="none" stroke="{_INK}" stroke-width="1.1"/>'
        f'<line x1="{edge}" y1="48" x2="{edge}" y2="66" stroke="{_INK}" '
        f'stroke-width="1" stroke-dasharray="{visuals.DASH_GUIDE}"/>'
    )


def _record() -> str:
    """Fields in rows: something a system already wrote down."""
    return (
        f'<rect x="26" y="24" width="98" height="62" rx="3" fill="none" '
        f'stroke="{_INK}" stroke-width="1.2"/>'
        f'<rect x="26" y="24" width="98" height="14" fill="{_GRID}" '
        f'fill-opacity="0.7"/>'
        f'<line x1="26" y1="38" x2="124" y2="38" stroke="{_INK}" '
        f'stroke-width="1"/>'
        f'<line x1="52" y1="24" x2="52" y2="86" stroke="{_GRID}" '
        f'stroke-width="1"/>'
        f'<line x1="88" y1="24" x2="88" y2="86" stroke="{_GRID}" '
        f'stroke-width="1"/>'
        + "".join(
            f'<line x1="32" y1="{y}" x2="46" y2="{y}" stroke="{_INK}" '
            f'stroke-width="2" stroke-opacity="0.35"/>'
            f'<line x1="58" y1="{y}" x2="82" y2="{y}" stroke="{_ACCENT}" '
            f'stroke-width="2" stroke-opacity="0.6"/>'
            f'<line x1="94" y1="{y}" x2="118" y2="{y}" stroke="{_ACCENT}" '
            f'stroke-width="2" stroke-opacity="0.6"/>'
            for y in (48, 60, 72, 82)
        )
    )


def _held_out_split() -> str:
    """
    Rows kept back, and what is done with them.

    The split is drawn because it is the claim: a number computed on the
    rows a model was fitted on is not the same quantity, and the dashed
    edge is where that distinction lives. Below it, predictions against
    what actually happened, on the identity line they are read against.
    """
    rows = "".join(
        f'<line x1="28" y1="{24 + index * 8}" x2="88" '
        f'y2="{24 + index * 8}" stroke="{_INK}" stroke-width="3" '
        f'stroke-opacity="0.22"/>'
        f'<line x1="94" y1="{24 + index * 8}" x2="122" '
        f'y2="{24 + index * 8}" stroke="{_ACCENT}" stroke-width="3" '
        f'stroke-opacity="0.65"/>'
        for index in range(4)
    )

    boundary = (
        f'<line x1="91" y1="18" x2="91" y2="58" stroke="{_INK}" '
        f'stroke-width="1.2" stroke-dasharray="{visuals.DASH_UNESTABLISHED}"/>'
    )

    # y on the identity line, and residuals either side of it.
    def identity(x: float) -> float:
        return 96.0 - 0.6 * (x - 58.0)

    residuals = (2.5, -3.0, 1.5, -1.5, 3.5, -2.5, 1.0, -3.5, 2.0)
    scatter = "".join(
        f'<circle cx="{58 + index * 7.5}" '
        f'cy="{identity(58 + index * 7.5) + residual:.1f}" r="2.8" '
        f'fill="{_ACCENT}" fill-opacity="0.6"/>'
        for index, residual in enumerate(residuals)
    )

    return rows + boundary + (
        f'<line x1="56" y1="{identity(56):.1f}" x2="126" '
        f'y2="{identity(126):.1f}" stroke="{_INK}" stroke-width="1" '
        f'stroke-dasharray="{visuals.DASH_GUIDE}"/>'
    ) + scatter


def _text() -> str:
    """
    A transcript: turns, and who took them.

    Speaker marks rather than plain lines, because what this produces is
    an exchange, and who prompted a statement changes what it is
    evidence of.
    """
    turns = ((26, 0, 86), (44, 1, 66), (58, 0, 94), (82, 1, 52))

    drawn = []
    for y, speaker, width in turns:
        x = 30 + speaker * 12
        drawn.append(
            f'<rect x="{x}" y="{y - 4}" width="7" height="7" rx="1.5" '
            f'fill="{_ACCENT if speaker == 0 else _INK}" '
            f'fill-opacity="{0.75 if speaker == 0 else 0.5}"/>'
        )
        drawn.append(
            f'<line x1="{x + 12}" y1="{y}" x2="{x + 12 + width * 0.6:.0f}" '
            f'y2="{y}" stroke="{_INK}" stroke-width="2.5" '
            f'stroke-opacity="0.4"/>'
        )
        drawn.append(
            f'<line x1="{x + 12}" y1="{y + 9}" '
            f'x2="{x + 12 + width * 0.45:.0f}" y2="{y + 9}" '
            f'stroke="{_INK}" stroke-width="2.5" stroke-opacity="0.25"/>'
        )

    return "".join(drawn)


_BUILDERS = {
    VISUAL_RATING_SCALE: _rating_scale,
    VISUAL_DIARY_GRID: _diary_grid,
    VISUAL_BODY_MAP: _body_map,
    VISUAL_ECG: _ecg,
    VISUAL_SKIN_CONDUCTANCE: _skin_conductance,
    VISUAL_SIGNAL_TRACE: _signal_trace,
    VISUAL_BRAIN_SLICE: _brain_slice,
    VISUAL_ASSAY: _assay,
    VISUAL_EVENT_TIMELINE: _event_timeline,
    VISUAL_CARD_CLUSTER: _card_cluster,
    VISUAL_CAUSAL_MAP: _causal_map,
    VISUAL_ROUND_CONVERGENCE: _round_convergence,
    VISUAL_INVENTORY: _inventory,
    VISUAL_RATIO_BAR: _ratio_bar,
    VISUAL_RECORD: _record,
    VISUAL_HELD_OUT_SPLIT: _held_out_split,
    VISUAL_TEXT: _text,
}

# The three continuously sampled measures, which is where a travelling
# highlight says something a still line cannot.
ANIMATED_TYPES: tuple[str, ...] = (
    VISUAL_ECG,
    VISUAL_SKIN_CONDUCTANCE,
    VISUAL_SIGNAL_TRACE,
)


def measure_visual_svg(measure) -> str:
    """
    A small drawing of what this measure produces.

    Raises on a visual type with no builder rather than drawing an empty
    box, so a type added to the vocabulary without a drawing fails where
    it is added instead of rendering as a gap in a row of measures.
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
            f"{measure.name}: {measure.produces.lower()}, drawn as a "
            f"{measure.visual_type}."
        ),
    )


def builders_cover_the_vocabulary() -> bool:
    """Whether every declared type can actually be drawn."""
    return set(_BUILDERS) == set(VISUAL_TYPES)
