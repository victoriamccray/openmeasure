"""
SVG drawings for the eyes-closed / eyes-open study, for Streamlit pages.

The facts and numbers come from modules/signal_pipeline/core/resting_state.py
(and the reactivity and chance results a page computes with it). This file
only draws them, in the light theme of the Multimodal Signal Convergence
page. It mirrors the study room in experiences/become-the-signal/main.js;
the two are kept in step by hand.

Pure string builders with no streamlit import.
"""

from __future__ import annotations

import math

from modules.signal_pipeline.core import resting_state as rs

INK = "#1f1e1c"
MUTED = "#52514e"
LINE = "#b9b7b0"
SURFACE = "#fcfcfb"
CLOSED = "#2a78d6"
OPEN = "#898781"
SIGNAL = "#e07a3f"
EXPECTED = "rgba(27,175,122,0.18)"

_BAND_COLOURS = {"Delta": "#c9d3f2", "Theta": "#d9cdf5", "Alpha": SIGNAL, "Beta": "#c8eee4"}
_FONT = "font-family:system-ui,sans-serif"


def _svg(view_box: str, label: str, body: str, max_width: int = 520) -> str:
    return (
        f'<svg viewBox="{view_box}" role="img" aria-label="{label}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:{max_width}px;display:block;margin:0 auto;{_FONT}">{body}</svg>'
    )


def participant_svg(eyes_closed: bool) -> str:
    """A participant seated at rest, with the EEG cap and finger skin conductance electrodes."""
    glow = 0.8 if eyes_closed else 0.25
    eye = (
        f'<path d="M181,84 q5,3.5 10,0" fill="none" stroke="#5a3a28" stroke-width="2" stroke-linecap="round"/>'
        if eyes_closed
        else '<ellipse cx="186" cy="84" rx="5.5" ry="3.6" fill="#fff" stroke="#8a5a40" stroke-width="0.8"/>'
        '<circle cx="188" cy="84" r="2.2" fill="#2a1a10"/>'
    )
    sites = "".join(
        f'<circle cx="{x}" cy="{y}" r="3.2" fill="{SURFACE}" stroke="#2b2f45" stroke-width="1.2"/>'
        for x, y in ((150, 52), (168, 44), (186, 48), (200, 62), (136, 64), (128, 82))
    )
    body_ink = "#5d6b9a"
    state = "Eyes closed" if eyes_closed else "Eyes open"
    label = (
        f"A participant seated at rest with {state.lower()}, wearing an EEG cap with electrode O1 at "
        "the back of the head, and two skin conductance electrodes on the fingers. Cables run to "
        "two amplifiers."
    )
    body = f"""
<defs><radialGradient id="o1g"><stop offset="0" stop-color="{SIGNAL}" stop-opacity="{glow}"/><stop offset="1" stop-color="{SIGNAL}" stop-opacity="0"/></radialGradient></defs>
<rect x="0" y="268" width="420" height="32" fill="#f0efe9"/>
<path d="M92,150 L104,262 M206,196 L214,262 M92,196 H220" fill="none" stroke="#a9a79f" stroke-width="5" stroke-linecap="round"/>
<path d="M86,112 Q80,150 92,200" fill="none" stroke="#a9a79f" stroke-width="5" stroke-linecap="round"/>
<path d="M120,198 C150,200 196,196 222,200 L232,262 L246,262" fill="none" stroke="{body_ink}" stroke-width="16" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M132,118 C110,140 108,178 120,200 L200,200 C206,170 196,138 176,118 Z" fill="#6f7cad" stroke="{body_ink}" stroke-width="2"/>
<path d="M160,124 C176,150 190,170 230,176" fill="none" stroke="{body_ink}" stroke-width="12" stroke-linecap="round"/>
<ellipse cx="236" cy="177" rx="10" ry="6" fill="#d8a988" stroke="#b98a6b"/>
<rect x="242" y="171" width="7" height="4" rx="1.5" fill="#c9d1e8" stroke="#2b2f45"/><rect x="242" y="178" width="7" height="4" rx="1.5" fill="#c9d1e8" stroke="#2b2f45"/>
<path d="M249,173 C290,170 300,214 330,226 M249,180 C286,184 296,220 330,232" fill="none" stroke="#8890b0" stroke-width="1.6"/>
<rect x="330" y="214" width="66" height="34" rx="5" fill="#eceae3" stroke="{MUTED}" stroke-dasharray="4 3"/><text x="363" y="235" font-size="10" fill="{MUTED}" text-anchor="middle">Skin cond.</text>
<text x="363" y="262" font-size="8.5" fill="{MUTED}" text-anchor="middle">published finding</text>
<rect x="150" y="104" width="22" height="18" fill="#d8a988"/>
<circle cx="164" cy="80" r="34" fill="#d8a988" stroke="#b98a6b" stroke-width="1.2"/>
<ellipse cx="160" cy="86" rx="5" ry="8" fill="#d8a988" stroke="#b98a6b"/>
<path d="M196,80 L204,92 L196,95 M188,104 q4,2 8,0 M180,76 q6,-3 12,0" fill="none" stroke="#8a5a40" stroke-width="1.6" stroke-linecap="round"/>
{eye}
<path d="M130,78 C130,36 196,30 200,70" fill="none" stroke="#2b2f45" stroke-width="9" stroke-linecap="round" opacity="0.85"/>
{sites}
<circle cx="131" cy="96" r="22" fill="url(#o1g)"/>
<circle cx="131" cy="96" r="4.5" fill="{SIGNAL}" stroke="{INK}"/><text x="104" y="118" font-size="11" font-weight="600" fill="{INK}">O1</text>
<path d="M130,80 C90,90 60,160 54,214" fill="none" stroke="#8890b0" stroke-width="1.6"/>
<rect x="16" y="214" width="76" height="34" rx="5" fill="#eceae3" stroke="{LINE}"/>
<text x="54" y="229" font-size="10" fill="{MUTED}" text-anchor="middle">EEG amplifier</text>
<text x="54" y="241" font-size="8.5" fill="{MUTED}" text-anchor="middle">500 samples/s</text>
<text x="54" y="262" font-size="8.5" fill="{MUTED}" text-anchor="middle">recorded here</text>
<text x="232" y="34" font-size="11" font-weight="600" fill="{INK}">{state}, at rest</text>
<text x="232" y="50" font-size="8.5" fill="{MUTED}">No task. Sitting still.</text>"""
    return _svg("0 0 420 300", label, body, max_width=560)


def bands_svg() -> str:
    """The conventional bands from 0 to 30 Hz, alpha highlighted."""

    def x(hz: float) -> float:
        return 10 + hz / 30 * 340

    bars = "".join(
        f'<rect x="{x(b.low_hz):.1f}" y="18" width="{x(b.high_hz) - x(b.low_hz):.1f}" height="22" fill="{_BAND_COLOURS[b.name]}"/>'
        f'<text x="{(x(b.low_hz) + x(b.high_hz)) / 2:.1f}" y="33" font-size="9.5" text-anchor="middle" '
        f'fill="{INK}" font-weight="{700 if b.name == "Alpha" else 400}">{b.name}</text>'
        for b in rs.BANDS
    )
    ticks = "".join(
        f'<text x="{x(t):.1f}" y="54" font-size="8.5" text-anchor="middle" fill="{MUTED}">{t:g}</text>'
        for t in (0, 4, 8, 13, 20, 30)
    )
    return _svg(
        "0 0 360 62",
        "Conventional EEG bands from 0 to 30 Hz: delta, theta, alpha from 8 to 13 Hz, and beta.",
        bars + ticks + f'<text x="350" y="12" font-size="8.5" text-anchor="end" fill="{MUTED}">Hz</text>',
    )


def reactivity_svg(result: rs.Reactivity) -> str:
    """The eyes-closed to eyes-open ratio on a log scale, with 1 and the expected side marked."""
    lo, hi = math.log2(0.25), math.log2(32)

    def x(v: float) -> float:
        v = min(max(v, 0.25), 32)
        return 10 + (math.log2(v) - lo) / (hi - lo) * 340

    ticks = "".join(
        f'<text x="{x(t):.1f}" y="62" font-size="8.5" text-anchor="middle" fill="{MUTED}">{t:g}×</text>'
        for t in (0.5, 1, 2, 4, 8, 16, 32)
    )
    body = (
        f'<rect x="{x(1):.1f}" y="20" width="{x(32) - x(1):.1f}" height="18" fill="{EXPECTED}"/>'
        f'<line x1="10" y1="29" x2="350" y2="29" stroke="{LINE}" stroke-width="1.2"/>'
        f'<line x1="{x(1):.1f}" y1="14" x2="{x(1):.1f}" y2="44" stroke="{INK}" stroke-width="1.6"/>'
        f'<text x="{x(1) + 4:.1f}" y="12" font-size="8.5" fill="{MUTED}">1 = no difference</text>'
        f'<text x="{x(16):.1f}" y="12" font-size="8.5" text-anchor="middle" fill="{MUTED}">expected side</text>'
        f'<circle cx="{x(result.ratio):.1f}" cy="29" r="6" fill="{SIGNAL}" stroke="{INK}" stroke-width="1"/>'
        f'<text x="{x(result.ratio):.1f}" y="52" font-size="9.5" font-weight="700" text-anchor="middle" fill="{INK}">{result.ratio:.1f}×</text>'
        + ticks
    )
    return _svg(
        "0 0 360 68",
        f"Ratio scale. 1 means no difference; above 1 is the expected direction. This recording: {result.ratio:.1f} times.",
        body,
    )


def accuracy_svg(result: rs.ChanceAssessment) -> str:
    """Accuracy on 0 to 100%, with chance, the binomial threshold, and this result."""

    def x(p: float) -> float:
        return 10 + p * 340

    pct = lambda p: f"{round(p * 100)}%"  # noqa: E731
    ticks = "".join(
        f'<text x="{x(t):.1f}" y="70" font-size="8.5" text-anchor="middle" fill="{MUTED}">{round(t * 100)}%</text>'
        for t in (0, 0.25, 0.5, 0.75, 1)
    )
    body = (
        f'<rect x="{x(result.threshold_accuracy):.1f}" y="22" width="{x(1) - x(result.threshold_accuracy):.1f}" height="16" fill="{EXPECTED}"/>'
        f'<line x1="10" y1="30" x2="350" y2="30" stroke="{LINE}" stroke-width="1.2"/>'
        f'<line x1="{x(result.chance_accuracy):.1f}" y1="16" x2="{x(result.chance_accuracy):.1f}" y2="44" stroke="{INK}" stroke-width="1.6"/>'
        f'<text x="{x(result.chance_accuracy):.1f}" y="12" font-size="8.5" text-anchor="middle" fill="{MUTED}">chance {pct(result.chance_accuracy)}</text>'
        f'<line x1="{x(result.threshold_accuracy):.1f}" y1="16" x2="{x(result.threshold_accuracy):.1f}" y2="44" stroke="{INK}" stroke-dasharray="3 2"/>'
        f'<text x="{x(result.threshold_accuracy):.1f}" y="57" font-size="8.5" text-anchor="middle" fill="{MUTED}">beats chance from {result.threshold_correct}/{result.n_trials}</text>'
        f'<circle cx="{x(result.accuracy):.1f}" cy="30" r="6" fill="{SIGNAL}" stroke="{INK}" stroke-width="1"/>'
        f'<text x="{x(result.accuracy):.1f}" y="12" font-size="9.5" font-weight="700" text-anchor="middle" fill="{INK}">{result.n_correct}/{result.n_trials}</text>'
        + ticks
    )
    return _svg(
        "0 0 360 74",
        f"Accuracy from 0 to 100 percent. Chance is {pct(result.chance_accuracy)}. With {result.n_trials} "
        f"windows, {result.threshold_correct} correct beats chance. This result: {result.n_correct} of {result.n_trials}.",
        body,
    )


def skin_conductance_svg() -> str:
    """Direction only: skin conductance rises from eyes closed to eyes open."""
    body = (
        f'<text x="20" y="40" font-size="9.5" fill="{MUTED}">Eyes closed</text>'
        '<path d="M100,40 C160,40 200,18 260,16" fill="none" stroke="#1baf7a" stroke-width="2.5"/>'
        '<polygon points="260,10 272,16 260,22" fill="#1baf7a"/>'
        f'<text x="280" y="20" font-size="9.5" fill="{MUTED}">Eyes open</text>'
        f'<text x="180" y="56" font-size="8.5" text-anchor="middle" fill="{MUTED}">direction only: absolute levels vary between people</text>'
    )
    return _svg("0 0 360 62", "Skin conductance rises from eyes closed to eyes open. Direction only, no values.", body)
