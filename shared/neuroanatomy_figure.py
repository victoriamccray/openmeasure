"""
SVG drawings of the neuroanatomy atlas, for Streamlit pages.

The facts (systems, coverage, measurement type, electrode sites) come from
modules/signal_pipeline/core/neuroanatomy.py. This file only places them
on a drawing, and holds no statistical or interpretive logic. It mirrors
the drawing in experiences/become-the-signal/main.js, with the same
geometry, so the static experience and the Streamlit page show the same
figure. The two are kept in step by hand: a change to one should be made
to the other.

Pure string builders with no streamlit import, so a page can render the
result with components.html and a test can inspect it directly.
"""

from __future__ import annotations

import math
from html import escape

from modules.signal_pipeline.core import neuroanatomy as na

# Light theme, matching the page palette. Electrical uses the page's
# Neural category colour and body response its Autonomic colour, so the
# figure reads with the rest of the Multimodal Signal Convergence page.
MEASURE_COLOR = {
    na.MEASURE_ELECTRICAL: "#2a78d6",
    na.MEASURE_HEMODYNAMIC: "#8a5cc9",
    na.MEASURE_ANATOMY: "#898781",
    na.MEASURE_PERIPHERAL: "#eb6834",
}
MEASURE_LABEL = {
    na.MEASURE_ELECTRICAL: "Electrical potentials",
    na.MEASURE_HEMODYNAMIC: "Blood oxygen",
    na.MEASURE_ANATOMY: "Anatomy only",
    na.MEASURE_PERIPHERAL: "Body response",
}
LEVEL_TEXT = {
    na.COVERED: "Signal from here",
    na.PARTIAL: "Partial or weak",
    na.OUT_OF_RANGE: "Out of range",
}

INK = "#1f1e1c"
LINE = "#b9b7b0"
LINE_STRONG = "#52514e"
SURFACE = "#fcfcfb"
FACE = "#c98a00"
SIGNAL = "#e07a3f"

CEREBRUM = (
    "M92,150 C66,132 62,92 84,66 C108,38 150,26 198,28 C248,30 286,52 296,92 "
    "C304,124 296,156 270,170 C252,178 232,172 214,170 C196,176 172,182 150,180 "
    "C124,178 104,168 92,150 Z"
)
HEAD = (
    "M165,14 C245,8 322,45 326,112 C330,165 300,200 272,214 L270,262 L120,262 L118,224 "
    "C95,224 72,216 64,202 C60,194 62,186 58,180 C52,178 48,172 52,166 C46,162 46,156 52,152 "
    "C44,148 38,142 36,136 L47,123 C47,111 45,100 49,88 C59,40 105,18 165,14 Z"
)
LOBES = {
    "frontal": ("M0,0 L167,0 L167,28 L140,131 L98,140 L0,150 Z", (112, 92, "Frontal", 0)),
    "sensorimotor": ("M167,0 L195,0 L193,32 L164,125 L140,131 L167,28 Z", (166, 80, "Motor / sensory", -72)),
    "parietal": ("M195,0 L254,0 L252,44 L255,124 L210,122 L164,125 L193,32 Z", (222, 66, "Parietal", 0)),
    "occipital": ("M254,0 L360,0 L360,270 L262,175 L255,124 L252,44 Z", (276, 104, "Occipital", 0)),
    "temporal": ("M0,150 L98,140 L140,131 L164,125 L210,122 L255,124 L262,175 L0,270 Z", (128, 166, "Temporal", 0)),
}
EEG_SITES = ((58, 72), (104, 32), (166, 16), (232, 20), (288, 50), (318, 90))
O1_AT = (323, 128)


def _fill(m: na.SensingModality, key: str, electrode_only: bool) -> tuple[str, float]:
    if electrode_only:
        return "none", 1.0
    level = m.coverage_for(key).level
    if level == na.COVERED:
        return MEASURE_COLOR[m.measurement], 0.42
    if level == na.PARTIAL:
        return "url(#hatch)", 1.0
    return "none", 1.0


def side_view_svg(
    modality_key: str = "eeg",
    selected_system: str | None = None,
    electrode_only: bool = False,
) -> str:
    """
    Side view of the head, facing left.

    With a modality, shading shows coverage and colour shows the kind of
    signal. With electrode_only, the drawing shows the EEG cap and a
    schematic glow around O1 in place of any coverage shading.
    """
    m = na.modality(modality_key)
    if selected_system is not None:
        na.system(selected_system)
    color = MEASURE_COLOR[m.measurement]
    sel = None if electrode_only else selected_system

    def stroke(key: str, dashed: bool = False) -> str:
        width = 2.4 if sel == key else (1.2 if dashed else 1)
        colour = INK if sel == key else (LINE_STRONG if dashed else LINE)
        dash = ' stroke-dasharray="3 2"' if dashed else ""
        return f'stroke="{colour}" stroke-width="{width}"{dash}'

    def shape(key: str, tag: str, attrs: str, dashed: bool = False) -> str:
        fill, opacity = _fill(m, key, electrode_only)
        name = escape(na.system(key).name)
        return (
            f'<{tag} {attrs} fill="{fill}" fill-opacity="{opacity}" {stroke(key, dashed)}>'
            f"<title>{name}</title></{tag}>"
        )

    lobes = "".join(shape(k, "path", f'd="{d}"') for k, (d, _) in LOBES.items())
    label_parts = []
    for _, (x, y, t, r) in LOBES.values():
        rotate = f' transform="rotate({r} {x} {y})"' if r else ""
        size = 8.5 if len(t) > 10 else 10
        label_parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{LINE_STRONG}" text-anchor="middle"{rotate}>{t}</text>'
        )
    labels = "".join(label_parts)

    face = ""
    if not electrode_only and m.key in ("smri", "fmri"):
        face = (
            f'<path d="{HEAD}" clip-path="url(#face-clip)" fill="{FACE}" fill-opacity="0.22" '
            f'stroke="{FACE}" stroke-width="3"/>'
            f'<text x="22" y="150" font-size="10" font-weight="600" fill="{FACE}" text-anchor="middle">Face</text>'
        )

    sensors = ""
    if electrode_only or m.key == "eeg":
        sensors = "".join(
            f'<circle cx="{x}" cy="{y}" r="4" fill="{SURFACE}" stroke="{LINE_STRONG}" stroke-width="1.2"/>'
            for x, y in EEG_SITES
        )
        sensors += (
            f'<circle cx="{O1_AT[0]}" cy="{O1_AT[1]}" r="6.5" fill="{SIGNAL}" stroke="{INK}"/>'
            f'<text x="{O1_AT[0] + 10}" y="{O1_AT[1] + 4}" font-size="10" font-weight="600" fill="{INK}">O1</text>'
        )
    elif m.key == "fnirs":
        sensors = (
            f'<path d="M52,82 Q86,84 76,46" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="3 2"/>'
            f'<rect x="47" y="78" width="9" height="9" fill="{SURFACE}" stroke="{color}" stroke-width="1.4"/>'
            f'<rect x="71" y="38" width="9" height="9" fill="{SURFACE}" stroke="{color}" stroke-width="1.4"/>'
            f'<text x="20" y="64" font-size="10" fill="{LINE_STRONG}">Light</text>'
        )
    elif m.key == "peripheral":
        sensors = f'<text x="232" y="256" font-size="10" font-weight="600" fill="{INK}">To heart and skin</text>'

    autonomic_on = not electrode_only and m.coverage_for("autonomic").level != na.OUT_OF_RANGE
    nerve = (
        f'<path d="M215,238 C214,248 216,256 214,268" fill="none" '
        f'stroke="{color if autonomic_on else LINE}" stroke-width="{3 if autonomic_on else 2}" '
        f'stroke-dasharray="4 3"><title>{escape(na.system("autonomic").name)}</title></path>'
    )
    glow = (
        f'<g clip-path="url(#brain-clip)"><circle cx="{O1_AT[0]}" cy="{O1_AT[1]}" r="78" fill="url(#o1-glow)"/></g>'
        if electrode_only
        else ""
    )
    you = (
        ""
        if electrode_only
        else f'<circle cx="284" cy="128" r="5" fill="{SIGNAL}"/>'
        f'<text x="284" y="146" font-size="10" font-weight="600" fill="{INK}" text-anchor="middle">You</text>'
    )

    if electrode_only:
        description = (
            "Side view of the head facing left. Electrode O1 sits on the scalp over the left "
            "occipital region, with a schematic glow showing that it picks up a blend of nearby sources."
        )
    else:
        parts = "; ".join(
            f"{LEVEL_TEXT[m.coverage_for(s.key).level].lower()}: {s.name}" for s in na.SYSTEMS
        )
        description = (
            f"Side view of the head facing left. {m.short_name} measures "
            f"{MEASURE_LABEL[m.measurement].lower()}. {parts}."
        )

    small = f'font-size="8.5" font-style="italic" fill="{LINE_STRONG}"'
    return f"""<svg viewBox="0 0 360 270" role="img" aria-label="{escape(description)}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:520px;display:block;margin:0 auto;font-family:system-ui,sans-serif">
<defs>
<clipPath id="brain-clip"><path d="{CEREBRUM}"/></clipPath>
<clipPath id="face-clip"><rect x="0" y="84" width="66" height="150"/></clipPath>
<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="{color}" fill-opacity="0.08"/><line x1="0" y1="0" x2="0" y2="6" stroke="{color}" stroke-width="2" stroke-opacity="0.7"/></pattern>
<radialGradient id="o1-glow"><stop offset="0" stop-color="{SIGNAL}" stop-opacity="0.7"/><stop offset="0.55" stop-color="{SIGNAL}" stop-opacity="0.22"/><stop offset="1" stop-color="{SIGNAL}" stop-opacity="0"/></radialGradient>
</defs>
<path d="{HEAD}" fill="none" stroke="{LINE}" stroke-width="1.2"/>
{face}
{nerve}
{shape("brainstem", "path", 'd="M196,166 C200,190 204,214 206,240 L224,240 C222,214 220,190 222,168 Z"')}
{shape("cerebellum", "ellipse", 'cx="250" cy="186" rx="35" ry="18"')}
<path d="{CEREBRUM}" fill="{SURFACE}"/>
<g clip-path="url(#brain-clip)">{lobes}</g>
{glow}
<path d="{CEREBRUM}" fill="none" stroke="{LINE_STRONG}" stroke-width="1.4"/>
{shape("thalamus", "ellipse", 'cx="200" cy="114" rx="15" ry="9"', dashed=True)}
{shape("hippocampus_amygdala", "ellipse", 'cx="178" cy="148" rx="20" ry="6" transform="rotate(-10 178 148)"', dashed=True)}
{labels}
<text x="200" y="104" {small} text-anchor="middle">Thalamus</text>
<text x="178" y="140" {small} text-anchor="middle">Hippocampus</text>
<text x="250" y="190" {small} text-anchor="middle">Cerebellum</text>
<text x="232" y="230" {small}>Brainstem</text>
{you}
{sensors}
</svg>"""


def legend_html(modality_key: str) -> str:
    """Coverage legend and the signal-type swatch for one modality."""
    m = na.modality(modality_key)
    color = MEASURE_COLOR[m.measurement]
    swatch = "display:inline-block;width:11px;height:11px;margin-right:5px;vertical-align:-1px;border-radius:2px"
    hatch = f"background:repeating-linear-gradient(45deg,{color} 0 2px,transparent 2px 5px)"
    return (
        f'<div style="font-family:system-ui,sans-serif;font-size:13px;color:{LINE_STRONG};'
        'display:flex;flex-wrap:wrap;gap:14px;justify-content:center;margin-top:6px">'
        f'<span><i style="{swatch};background:{color};opacity:.6"></i>Signal from here</span>'
        f'<span><i style="{swatch};{hatch}"></i>Partial or weak</span>'
        f'<span><i style="{swatch};border:1px solid {LINE_STRONG}"></i>Out of range</span>'
        f'<span><i style="{swatch};border-radius:50%;background:{color}"></i>Signal type: '
        f"<b>{MEASURE_LABEL[m.measurement]}</b></span></div>"
    )


# Top view of the 10-20 layout, nose up: an outer ring at 36 degree
# steps and an inner cross, the standard textbook drawing.
_RING = ("Fp2", "F8", "T8", "P8", "O2", "O1", "P7", "T7", "F7", "Fp1")
MAP_SITES: tuple[tuple[str, float, float], ...] = tuple(
    (label, math.sin(math.radians(18 + i * 36)) * 0.8, -math.cos(math.radians(18 + i * 36)) * 0.8)
    for i, label in enumerate(_RING)
) + (
    ("Fpz", 0.0, -0.8), ("Oz", 0.0, 0.8), ("Fz", 0.0, -0.4), ("Cz", 0.0, 0.0), ("Pz", 0.0, 0.4),
    ("C3", -0.4, 0.0), ("C4", 0.4, 0.0), ("F3", -0.34, -0.44), ("F4", 0.34, -0.44),
    ("P3", -0.34, 0.44), ("P4", 0.34, 0.44),
)


def electrode_map_svg(chosen: str = na.RECORDING_ELECTRODE) -> str:
    """Top view of the 10-20 positions, with O1 and the chosen electrode marked."""
    chosen = na.electrode_site(chosen).label
    # Classic names are drawn at their modern positions.
    drawn = na.CLASSIC_TO_MODERN.get(chosen, chosen)
    r, c = 100, 120
    dots = []
    for label, x, y in MAP_SITES:
        cx, cy = c + x * r, c + y * r
        is_o1 = label == na.RECORDING_ELECTRODE
        fill = SIGNAL if is_o1 else SURFACE
        edge, width = (MEASURE_COLOR[na.MEASURE_ELECTRICAL], 2.8) if label == drawn else (LINE_STRONG, 1)
        dots.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="11" fill="{fill}" stroke="{edge}" stroke-width="{width}"/>'
            f'<text x="{cx:.1f}" y="{cy + 3.5:.1f}" font-size="7.5" text-anchor="middle" '
            f'fill="{INK}" font-weight="{700 if is_o1 or label == drawn else 400}">{label}</text>'
        )
    small = f'font-size="8.5" font-style="italic" fill="{LINE_STRONG}"'
    return f"""<svg viewBox="0 0 240 240" role="img" aria-label="Top view of the head with the nose at the top, showing the 10-20 electrode positions. O1 is at the back, left of the midline. {escape(chosen)} is outlined." xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:300px;display:block;margin:0 auto;font-family:system-ui,sans-serif">
<path d="M{c - 12},{c - r - 2} L{c},{c - r - 16} L{c + 12},{c - r - 2}" fill="none" stroke="{LINE}" stroke-width="1.2"/>
<ellipse cx="{c - r - 4}" cy="{c}" rx="7" ry="16" fill="none" stroke="{LINE}" stroke-width="1.2"/>
<ellipse cx="{c + r + 4}" cy="{c}" rx="7" ry="16" fill="none" stroke="{LINE}" stroke-width="1.2"/>
<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="{LINE}" stroke-width="1.2"/>
<line x1="{c}" y1="{c - r}" x2="{c}" y2="{c + r}" stroke="{LINE}" stroke-dasharray="2 3"/>
<text x="10" y="232" {small}>Left</text><text x="230" y="232" {small} text-anchor="end">Right</text>
<text x="{c}" y="12" {small} text-anchor="middle">Nose</text>
{"".join(dots)}
</svg>"""
