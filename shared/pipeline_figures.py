"""
The Multimodal Signal Convergence page's stage graphics, in the style of
the neuroanatomy figure: an illustrated drawing, colour that means signal
category, a legend and caveat under it, and a tap that explains a part.

    pipeline_html()  The Signals -> Feedback pipeline drawn top to bottom,
                     one coloured lane per modality until the stage where
                     signals converge, then one shared trunk, with the
                     feedback loop drawn back to the top. Used with one
                     modality (stage 2) and with several (stages 3 and 4),
                     so the same diagram grows as modalities are added.
    body_html()      An illustrated person with each body-sensed modality's
                     sensor at its real site, the modalities that are never
                     sensed from the body drawn outside the person, and the
                     Build, Privacy, or Agency view drawn over them.

Every label and explanation comes from the caller: the core Modality
records (name, signal_examples, body_region, privacy_exit_point,
agency_control_point), the core agency answers, and the page's stage notes. Nothing
here adds a claim. Pure string builders with no streamlit import; the tap
behaviour runs inside the embedded panel, so tapping never reruns the page.
"""

from __future__ import annotations

import json
import math
from html import escape

INK = "#1f1e1c"
MUTED = "#52514e"
LINE = "#b9b7b0"
SURFACE = "#fcfcfb"
ACCENT = "#e07a3f"
SKIN = "#e2b391"
SKIN_EDGE = "#b98a6b"
SHIRT = "#6f7cad"
SHIRT_EDGE = "#5d6b9a"

# Short lane labels by category (display text only).
SHORT_LABEL = {
    "Neural": "EEG",
    "Autonomic": "ECG",
    "Muscular": "EMG",
    "Behavioral": "Motion",
    "Subjective": "Self-report",
    "Environmental": "Context",
    "Institutional": "Records",
}

_FONT = "font-family:system-ui,-apple-system,'Segoe UI',sans-serif"


def _panel(svg: str, legend: str, caveat: str, info_default: str, payload: dict, max_width: int = 520) -> str:
    """Wrap an SVG with a tap-to-explain line, a legend, and a caveat."""
    import hashlib

    uid = "pf" + hashlib.sha1((svg + info_default).encode()).hexdigest()[:10]
    return f"""<div id="{uid}" style="{_FONT};background:{SURFACE};color:{MUTED};border:1px solid #e1e0d9;border-radius:10px;padding:10px 12px">
<div style="max-width:{max_width}px;margin:0 auto">{svg}</div>
<p class="info" aria-live="polite" style="font-size:13px;line-height:1.45;margin:8px 0 6px;border-left:3px solid {ACCENT};padding-left:8px;color:{INK}">{escape(info_default)}</p>
<div style="font-size:12px;display:flex;flex-wrap:wrap;gap:10px;margin:4px 0">{legend}</div>
<p style="font-size:11.5px;margin:4px 0 0">{escape(caveat)}</p></div>
<style>
[data-key] {{ cursor:pointer; }}
[data-key]:hover .hit, [data-key]:focus .hit {{ stroke:{INK}; stroke-width:2.2; }}
[data-key].on .hit {{ stroke:{ACCENT}; stroke-width:3; }}
[data-key]:focus {{ outline:none; }}
</style>
<script>
(() => {{
const root = document.getElementById("{uid}");
const INFO = {json.dumps(payload)};
const box = root.querySelector(".info");
function pick(el) {{
  root.querySelectorAll("[data-key]").forEach(g => g.classList.toggle("on", g === el));
  box.innerHTML = INFO[el.dataset.key];
}}
root.querySelectorAll("[data-key]").forEach(el => {{
  el.setAttribute("tabindex", "0");
  el.setAttribute("role", "button");
  el.addEventListener("click", () => pick(el));
  el.addEventListener("keydown", e => {{ if (e.key === "Enter" || e.key === " ") {{ e.preventDefault(); pick(el); }} }});
}});
}})();
</script>"""


def _chip(color: str, label: str, shape: str = "circle") -> str:
    radius = "50%" if shape == "circle" else "3px"
    return (
        f'<span style="display:inline-flex;align-items:center;gap:5px"><i style="display:inline-block;width:11px;'
        f'height:11px;border-radius:{radius};background:{color}"></i>{escape(label)}</span>'
    )


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width and line:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines


# ------------------------------------------------------------------
# The pipeline as a story: what the signal looks like at each stage
# ------------------------------------------------------------------

def _illustrative_shape(category: str, n: int = 120) -> list[float] | None:
    """A drawn shape for a modality with no recording here, scaled to [-1, 1]."""
    t = [i / (n - 1) for i in range(n)]
    if category == "Autonomic":  # heartbeats: a sharp peak each cycle
        out = []
        for x in t:
            phase = (x * 3.2) % 1.0
            out.append(1.0 if 0.48 < phase < 0.52 else (-0.35 if 0.53 < phase < 0.58 else 0.08 * math.sin(phase * 6.28)))
        return out
    if category == "Muscular":  # bursts of activity when a muscle contracts
        return [
            (0.9 if 0.2 < x < 0.42 or 0.66 < x < 0.86 else 0.12) * math.sin(97 * x * x + 13 * x) for x in t
        ]
    if category == "Behavioral":  # movement: slow swings with a step
        return [0.6 * math.sin(6.28 * 1.3 * x) + (0.3 if x > 0.55 else -0.3) for x in t]
    return None


_DISCRETE = {
    "Subjective": [(0.18, 0.4), (0.52, 0.8), (0.86, 0.2)],  # ratings the person gives
    "Environmental": [(0.0, 0.33), (0.35, 0.7), (0.72, 0.15)],  # location segments (start, level)
    "Institutional": [(0.22, 1.0), (0.61, 1.0)],  # record entries
}


def _trace_path(values, x0, x1, ymid, half, every=1):
    pts = list(values)[::every]
    n = len(pts)
    if n < 2:
        return ""
    peak = max(abs(v) for v in pts) or 1.0
    return "".join(
        f"{'M' if i == 0 else 'L'}{x0 + (x1 - x0) * i / (n - 1):.1f},{ymid - (v / peak) * half:.1f}"
        for i, v in enumerate(pts)
    )


def _lane(m, eeg_raw, x0, x1, y, color, sampled: bool) -> str:
    """One modality's signal (stage 1) or its sampled points (stage 2)."""
    half = 6.5
    label = SHORT_LABEL.get(m.category, m.category)
    recorded = m.category == "Neural" and eeg_raw
    tag = "recorded" if recorded else "illustrative"
    head = (
        f'<text x="{x0 - 4}" y="{y + 3.5}" font-size="9.5" font-weight="600" fill="{color}" text-anchor="end">{escape(label)}</text>'
        f'<text x="{x1 + 4}" y="{y + 3}" font-size="7.5" fill="{MUTED}">{tag}</text>'
    )
    values = eeg_raw if recorded else _illustrative_shape(m.category)
    if values is not None:
        path = _trace_path(values, x0, x1, y, half)
        body = f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.2" stroke-opacity="{0.35 if sampled else 1}"/>'
        if sampled:
            pts = list(values)
            peak = max(abs(v) for v in pts) or 1.0
            step = max(1, len(pts) // 28)
            body += "".join(
                f'<circle cx="{x0 + (x1 - x0) * i / (len(pts) - 1):.1f}" cy="{y - (pts[i] / peak) * half:.1f}" r="1.6" fill="{color}"/>'
                for i in range(0, len(pts), step)
            )
        return head + f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#e1e0d9"/>' + body
    marks = _DISCRETE.get(m.category, [])
    if m.category == "Environmental":
        segs = marks + [(1.0, 0)]
        body = "".join(
            f'<rect x="{x0 + (x1 - x0) * a:.1f}" y="{y - 5}" width="{(x1 - x0) * (b - a) - 2:.1f}" height="10" rx="2" fill="{color}" fill-opacity="{0.25 + 0.6 * lvl:.2f}"/>'
            for (a, lvl), (b, _) in zip(segs, segs[1:])
        )
    elif m.category == "Subjective":
        body = "".join(
            f'<circle cx="{x0 + (x1 - x0) * a:.1f}" cy="{y + half - 2 * half * lvl:.1f}" r="3.2" fill="{color}"/>' for a, lvl in marks
        )
    else:
        body = "".join(
            f'<rect x="{x0 + (x1 - x0) * a - 3:.1f}" y="{y - 6}" width="6" height="12" rx="1" fill="{color}"/>' for a, _ in marks
        )
    return head + f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#e1e0d9"/>' + body


def pipeline_html(
    modalities,
    stages: tuple[str, ...],
    convergence_stage: str,
    stage_notes: dict[str, str],
    handoffs: dict[str, str],
    loop_text: str,
    colors: dict[str, str],
    eeg: dict | None = None,
) -> str:
    """
    The pipeline as seven illustrated stage cards, each showing the signal
    at that stage. eeg carries recorded numbers computed by core: raw (one
    second of voltage), filtered (the same second band-passed to alpha),
    powers (alpha power per one-second window), threshold, labels (True for
    windows labelled eyes closed), decision_value and decision_label.
    Modalities with no recording here are drawn as labelled illustrative
    shapes.
    """
    if not modalities:
        raise ValueError("At least one modality is required.")
    if convergence_stage not in stages:
        raise ValueError(f"{convergence_stage!r} is not one of the stages.")
    eeg = eeg or {}
    w, x0, x1 = 360, 74, 318
    n = len(modalities)
    neural = colors.get("Neural", "#2a78d6")
    parts: list[str] = []
    payload: dict[str, str] = {}
    y = 10

    def card(i: int, stage: str, height: float, inner: str) -> None:
        nonlocal y
        height += 14  # room for the note line under the title
        key = f"s{i}"
        note = stage_notes.get(stage, "")
        handoff = handoffs.get(stage, "")
        payload[key] = f"<strong>{i + 1}. {escape(stage)}</strong> · {escape(note)}" + (
            f" It passes on {escape(handoff)}." if handoff else f" It loops back as {escape(loop_text)}."
        )
        hl = stage == convergence_stage and n > 1
        parts.append(
            f'<g data-key="{key}"><rect class="hit" x="4" y="{y}" width="{w - 8}" height="{height}" rx="10" '
            f'fill="{"#fbeee5" if hl else "#f6f5f0"}" stroke="{ACCENT if hl else "#e1e0d9"}" stroke-width="{1.6 if hl else 1}"/>'
            f'<circle cx="20" cy="{y + 16}" r="9" fill="{INK}"/><text x="20" y="{y + 19.5}" font-size="10" font-weight="700" fill="{SURFACE}" text-anchor="middle">{i + 1}</text>'
            f'<text x="36" y="{y + 20}" font-size="12.5" font-weight="600" fill="{INK}">{escape(stage)}</text>'
            f'<text x="36" y="{y + 33}" font-size="9.5" fill="{MUTED}">{escape(note)}</text>'
            f'<g transform="translate(0 14)">{inner}</g></g>'
        )
        y += height
        if handoff:
            parts.append(
                f'<line x1="{w / 2}" y1="{y + 2}" x2="{w / 2}" y2="{y + 16}" stroke="{LINE}" stroke-width="1.4" marker-end="url(#arrow)"/>'
                f'<text x="{w / 2 + 8}" y="{y + 12}" font-size="9" font-style="italic" fill="{MUTED}">{escape(handoff)}</text>'
            )
            y += 20

    raw = eeg.get("raw")
    for i, stage in enumerate(stages):
        top = y
        if i in (0, 1):
            rows = "".join(
                _lane(m, raw, x0, x1, top + 40 + 18 * k, colors.get(m.category, MUTED), sampled=(i == 1))
                for k, m in enumerate(modalities)
            )
            caption = (
                "Each modality's own signal, as it exists before any device."
                if i == 0
                else "The device turns each signal into numbers: sampled points it can store and copy."
            )
            inner = rows + f'<text x="20" y="{top + 40 + 18 * n}" font-size="8.5" fill="{MUTED}">{caption}</text>'
            card(i, stage, 50 + 18 * n, inner)
        elif stage == convergence_stage:
            inner = ""
            if eeg.get("filtered"):
                inner += (
                    f'<text x="{x0 - 4}" y="{top + 45}" font-size="9.5" font-weight="600" fill="{neural}" text-anchor="end">EEG</text>'
                    f'<path d="{_trace_path(raw, x0, x1, top + 42, 7)}" fill="none" stroke="{neural}" stroke-opacity="0.25" stroke-width="1"/>'
                    f'<path d="{_trace_path(eeg["filtered"], x0, x1, top + 42, 7)}" fill="none" stroke="{neural}" stroke-width="1.6"/>'
                    f'<text x="{x1 + 4}" y="{top + 45}" font-size="7.5" fill="{MUTED}">recorded</text>'
                    f'<text x="20" y="{top + 62}" font-size="8.5" fill="{MUTED}">Cleaned: the EEG filtered to its 8 to 13 Hz alpha band (pale: before).</text>'
                )
            if n > 1:
                bw = (x1 - x0) / n
                blocks = "".join(
                    f'<rect x="{x0 + k * bw:.1f}" y="{top + 76}" width="{bw - 2:.1f}" height="14" rx="3" fill="{colors.get(m.category, MUTED)}"/>'
                    for k, m in enumerate(modalities)
                )
                inner += (
                    blocks
                    + f'<text x="{x0 - 4}" y="{top + 87}" font-size="9" font-weight="600" fill="{ACCENT}" text-anchor="end">combined</text>'
                    f'<text x="20" y="{top + 104}" font-size="8.5" fill="{ACCENT}" font-weight="600">Signals meet here: every modality’s features join one combined input.</text>'
                )
            card(i, stage, (112 if n > 1 else 72), inner)
        elif stage == "Inference" and eeg.get("powers"):
            powers, labels, thr = eeg["powers"], eeg["labels"], eeg["threshold"]
            peak = max(powers)
            bw = (x1 - x0) / len(powers)
            base = top + 70
            bars = "".join(
                f'<rect x="{x0 + k * bw:.1f}" y="{base - 36 * p / peak:.1f}" width="{bw - 1.5:.1f}" height="{36 * p / peak:.1f}" '
                f'fill="{neural}" fill-opacity="{0.9 if lab else 0.35}"/>'
                for k, (p, lab) in enumerate(zip(powers, labels))
            )
            ty = base - 36 * thr / peak
            inner = (
                bars
                + f'<line x1="{x0}" y1="{ty:.1f}" x2="{x1}" y2="{ty:.1f}" stroke="{INK}" stroke-dasharray="3 2"/>'
                f'<text x="{x0 - 4}" y="{ty + 3:.1f}" font-size="8" fill="{INK}" text-anchor="end">threshold</text>'
                f'<text x="{x1 + 4}" y="{base}" font-size="7.5" fill="{MUTED}">recorded</text>'
                f'<text x="20" y="{base + 14}" font-size="8.5" fill="{MUTED}">Alpha power per second. Dark bars: labelled eyes closed.</text>'
            )
            card(i, stage, 92, inner)
        elif stage == "Decision" and eeg.get("decision_label"):
            v, thr, lab = eeg["decision_value"], eeg["threshold"], eeg["decision_label"]
            scale = max(v, thr) * 1.2
            px = x0 + (x1 - x0 - 90) * v / scale
            tx = x0 + (x1 - x0 - 90) * thr / scale
            inner = (
                f'<line x1="{x0}" y1="{top + 44}" x2="{x1 - 90}" y2="{top + 44}" stroke="{LINE}" stroke-width="4" stroke-linecap="round"/>'
                f'<line x1="{tx:.1f}" y1="{top + 34}" x2="{tx:.1f}" y2="{top + 54}" stroke="{INK}" stroke-dasharray="3 2"/>'
                f'<circle cx="{px:.1f}" cy="{top + 44}" r="6" fill="{neural}"/>'
                f'<rect x="{x1 - 78}" y="{top + 33}" width="86" height="22" rx="11" fill="{neural}"/>'
                f'<text x="{x1 - 35}" y="{top + 48}" font-size="10" font-weight="600" fill="{SURFACE}" text-anchor="middle">{escape(lab)}</text>'
                f'<text x="20" y="{top + 70}" font-size="8.5" fill="{MUTED}">One second: {v:.1f} µV² against a threshold of {thr:.1f} µV².</text>'
            )
            card(i, stage, 78, inner)
        elif stage == "Action":
            inner = (
                f'<rect x="{x0 + 40}" y="{top + 30}" width="120" height="34" rx="5" fill="{SURFACE}" stroke="{MUTED}" stroke-width="1.4"/>'
                f'<path d="M{x0 + 70},{top + 47} h40" stroke="{LINE}" stroke-width="2" stroke-dasharray="3 3" marker-end="url(#arrow)"/>'
                f'<path d="M{x0 + 118},{top + 40} l0,14 l4,-4 l5,7 l3,-2 l-5,-7 l5,-1 z" fill="{INK}"/>'
                f'<text x="{x0 + 176}" y="{top + 50}" font-size="9.5" fill="{INK}">a cursor moves,</text>'
                f'<text x="{x0 + 176}" y="{top + 62}" font-size="9.5" fill="{INK}">a device responds</text>'
                f'<text x="20" y="{top + 78}" font-size="8.5" fill="{MUTED}">The system acts on the decision. Illustrative.</text>'
            )
            card(i, stage, 86, inner)
        elif i == len(stages) - 1:
            inner = (
                f'<ellipse cx="{x0 + 10}" cy="{top + 48}" rx="13" ry="15" fill="{SKIN}" stroke="{SKIN_EDGE}"/>'
                f'<circle cx="{x0 + 15}" cy="{top + 45}" r="1.8" fill="#2a1a10"/>'
                f'<rect x="{x0 + 150}" y="{top + 34}" width="60" height="28" rx="4" fill="{SURFACE}" stroke="{MUTED}" stroke-width="1.4"/>'
                f'<path d="M{x0 + 176},{top + 41} l0,13 l4,-4 l4,6 l3,-2 l-4,-6 l5,-1 z" fill="{INK}"/>'
                f'<path d="M{x0 + 146},{top + 48} C{x0 + 100},{top + 48} {x0 + 70},{top + 48} {x0 + 30},{top + 48}" fill="none" stroke="{ACCENT}" stroke-width="1.6" marker-end="url(#arrowA)"/>'
                f'<text x="{x0 + 88}" y="{top + 42}" font-size="9" fill="{ACCENT}" text-anchor="middle">sees the response</text>'
                f'<text x="20" y="{top + 78}" font-size="8.5" fill="{MUTED}">The person reacts, which changes the next signal.</text>'
            )
            card(i, stage, 86, inner)
        else:
            card(i, stage, 40, "")

    h = y + 8
    label = (
        f"The pipeline from {stages[0]} to {stages[-1]} as {len(stages)} illustrated stages, with "
        f"{n} modalit{'ies' if n > 1 else 'y'} converging at {convergence_stage}."
    )
    svg = (
        f'<svg viewBox="0 0 {w} {h:.0f}" role="img" aria-label="{escape(label)}" style="width:100%;height:auto;display:block">'
        f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M0 0L10 5L0 10z" fill="{LINE}"/></marker><marker id="arrowA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{ACCENT}"/></marker></defs>{"".join(parts)}</svg>'
    )
    legend = "".join(_chip(colors.get(m.category, MUTED), m.name) for m in modalities)
    caveat = (
        "Recorded: the EEG from OpenNeuro ds005420 at channel O1, processed by the same functions as the rest "
        "of the page. Illustrative: shapes drawn for modalities without a recording here. Tap a stage for its note."
    )
    return _panel(svg, legend, caveat, "Tap a stage to read what happens there.", payload, max_width=560)


# ------------------------------------------------------------------
# The person and the sensors
# ------------------------------------------------------------------

# Sensor sites on a front-facing figure in a 360 x 400 view.
_SITES = {
    "Neural": (130, 34),
    "Behavioral": (72, 262),
    "Autonomic": (130, 150),
    "Muscular": (192, 222),
}
_EXTERNAL_Y = {"Subjective": 70, "Environmental": 150, "Institutional": 230}
_ENTRY = (300, 366)
_BOUNDARY = (36, 6, 226, 394)


def _figure() -> str:
    return (
        # legs
        f'<path d="M108,300 L100,386 L120,386 L128,306 Z M152,300 L160,386 L140,386 L132,306 Z" fill="#4a5578"/>'
        # torso and arms (shirt)
        f'<path d="M92,108 C100,100 160,100 168,108 L176,180 L172,300 L88,300 L84,180 Z" fill="{SHIRT}" stroke="{SHIRT_EDGE}"/>'
        f'<path d="M92,110 C78,130 72,190 70,256" fill="none" stroke="{SHIRT}" stroke-width="16" stroke-linecap="round"/>'
        f'<path d="M168,110 C182,130 190,190 192,256" fill="none" stroke="{SHIRT}" stroke-width="16" stroke-linecap="round"/>'
        # hands, neck, head
        f'<circle cx="70" cy="266" r="9" fill="{SKIN}" stroke="{SKIN_EDGE}"/><circle cx="192" cy="266" r="9" fill="{SKIN}" stroke="{SKIN_EDGE}"/>'
        f'<rect x="121" y="84" width="18" height="20" fill="{SKIN}"/>'
        f'<ellipse cx="130" cy="58" rx="28" ry="32" fill="{SKIN}" stroke="{SKIN_EDGE}"/>'
        f'<path d="M104,52 C104,24 156,24 156,52 C150,40 112,40 104,52 Z" fill="#5a3a28"/>'
        f'<circle cx="120" cy="60" r="2.2" fill="#2a1a10"/><circle cx="140" cy="60" r="2.2" fill="#2a1a10"/>'
        f'<path d="M123,74 q7,5 14,0" fill="none" stroke="#8a5a40" stroke-width="1.6" stroke-linecap="round"/>'
    )


def _sensor(category: str, color: str) -> str:
    x, y = _SITES[category]
    if category == "Neural":
        dots = "".join(
            f'<circle cx="{130 + 26 * math.cos(a):.1f}" cy="{56 - 30 * math.sin(a):.1f}" r="3.4" fill="{color}" stroke="{SURFACE}"/>'
            for a in (0.35, 0.95, 1.57, 2.19, 2.79)
        )
        return f'<path d="M102,52 C102,18 158,18 158,52" fill="none" stroke="{color}" stroke-width="5" stroke-opacity="0.55"/>{dots}'
    if category == "Autonomic":
        pads = "".join(f'<circle cx="{px}" cy="{py}" r="5" fill="{color}" stroke="{SURFACE}"/>' for px, py in ((112, 140), (148, 140), (130, 172)))
        return f'<path d="M112,140 L130,172 L148,140" fill="none" stroke="{color}" stroke-width="1.2"/>{pads}'
    if category == "Muscular":
        return f'<rect x="183" y="214" width="20" height="14" rx="4" fill="{color}" stroke="{SURFACE}"/>'
    if category == "Behavioral":
        return (
            f'<rect x="62" y="252" width="18" height="12" rx="3" fill="{color}" stroke="{SURFACE}"/>'
            f'<circle cx="130" cy="98" r="4" fill="{color}" stroke="{SURFACE}"/>'
        )
    return ""


def _external_icon(category: str, x: float, y: float, color: str) -> str:
    if category == "Subjective":
        return f'<path d="M{x},{y} h18 a3,3 0 0 1 3,3 v10 a3,3 0 0 1 -3,3 h-10 l-5,5 v-5 h-3 a3,3 0 0 1 -3,-3 v-10 a3,3 0 0 1 3,-3 z" fill="{color}"/>'
    if category == "Environmental":
        return f'<path d="M{x + 10},{y + 22} C{x + 2},{y + 12} {x + 2},{y + 2} {x + 10},{y + 1} C{x + 18},{y + 2} {x + 18},{y + 12} {x + 10},{y + 22} Z" fill="{color}"/><circle cx="{x + 10}" cy="{y + 9}" r="3" fill="{SURFACE}"/>'
    return f'<rect x="{x + 2}" y="{y}" width="16" height="21" rx="2" fill="{color}"/><rect x="{x + 6}" y="{y - 2}" width="8" height="4" rx="1" fill="{INK}"/>'


_GLYPH = {"yes": "\u2713", "no": "\u2715", "not stated": "?"}


def _mark(cx: float, cy: float, answer: str, color: str) -> str:
    """One answer: filled tick for yes, outlined cross for no, dashed ? for not stated."""
    if answer == "yes":
        shape, ink = f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{color}"/>', SURFACE
    elif answer == "no":
        shape, ink = f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{SURFACE}" stroke="{color}" stroke-width="1.5"/>', color
    else:
        shape, ink = f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{SURFACE}" stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="2 1.6"/>', MUTED
    return f'{shape}<text x="{cx:.1f}" y="{cy + 3.4:.1f}" font-size="9.5" font-weight="700" fill="{ink}" text-anchor="middle">{_GLYPH[answer]}</text>'


_AGENCY_TOP = 408
_AGENCY_ROW = 24
_AGENCY_COLS = (196, 254, 312)
_AGENCY_HEAD = (("Can they", "stop it?"), ("Asked", "each time?"), ("Can they", "undo it?"))


def _agency_table(modalities, agency, colors) -> tuple[str, float]:
    """Rows of modalities by the three questions; returns the markup and its bottom edge."""
    out = [
        f'<line x1="14" y1="{_AGENCY_TOP - 4}" x2="346" y2="{_AGENCY_TOP - 4}" stroke="{MUTED}" stroke-opacity="0.4"/>'
        f'<text x="16" y="{_AGENCY_TOP + 12}" font-size="10.5" font-weight="700" fill="{INK}">How much control</text>'
        f'<text x="16" y="{_AGENCY_TOP + 24}" font-size="10.5" font-weight="700" fill="{INK}">does the person have?</text>'
    ]
    for (a, b), cx in zip(_AGENCY_HEAD, _AGENCY_COLS):
        out.append(
            f'<text x="{cx}" y="{_AGENCY_TOP + 12}" font-size="9.5" font-weight="600" fill="{INK}" text-anchor="middle">{a}</text>'
            f'<text x="{cx}" y="{_AGENCY_TOP + 24}" font-size="9.5" font-weight="600" fill="{INK}" text-anchor="middle">{b}</text>'
        )
    y = _AGENCY_TOP + 46
    for i, m in enumerate(modalities):
        color = colors.get(m.category, MUTED)
        cells = "".join(_mark(cx, y - 4, a.answer, color) for a, cx in zip(agency[m.category].answers, _AGENCY_COLS))
        out.append(
            f'<g data-key="m{i}"><rect class="hit" x="14" y="{y - 16}" width="332" height="{_AGENCY_ROW - 2}" rx="5" fill="{color}" fill-opacity="0.07"/>'
            f'<circle cx="26" cy="{y - 4}" r="4.5" fill="{color}"/>'
            f'<text x="36" y="{y}" font-size="10.5" font-weight="600" fill="{INK}">{escape(SHORT_LABEL.get(m.category, m.category))}</text>{cells}</g>'
        )
        y += _AGENCY_ROW
    return "".join(out), y - 8


def agency_table_height(n_modalities: int) -> float:
    """Extra viewBox height the Agency view's table adds below the figure."""
    return _AGENCY_TOP - 400 + 46 + _AGENCY_ROW * n_modalities


def _agency_list(profile) -> str:
    rows = "".join(
        f"<p style='margin:6px 0 0'><strong>{escape(a.question)}</strong> {escape(a.plain)}"
        + (f"<br><span style='opacity:.75'>Source text: \u201c{escape(a.evidence)}\u201d</span>" if a.evidence else "")
        + "</p>"
        for a in profile.answers
    )
    return rows


def body_html(modalities, perspective: str, colors: dict[str, str], agency=None) -> str:
    """The illustrated person with each modality placed, under one perspective.

    agency maps category to a core AgencyProfile and is required for the
    Agency view: each path then carries three marks (stop, each time,
    correct or retract), each answered only where the cited text answers it.
    """
    if not modalities:
        raise ValueError("At least one modality is required.")
    if perspective not in ("Build", "Privacy", "Agency"):
        raise ValueError(f"perspective must be Build, Privacy, or Agency; got {perspective!r}.")
    if perspective == "Agency":
        missing = [m.category for m in modalities if not agency or m.category not in agency]
        if missing:
            raise ValueError(f"The Agency view needs agency answers for: {', '.join(missing)}.")

    parts = [_figure()]
    overlay: list[str] = []
    marks: list[str] = []
    payload: dict[str, str] = {}

    if perspective == "Privacy":
        x1, y1, x2, y2 = _BOUNDARY
        overlay.append(
            f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" rx="22" fill="none" stroke="{ACCENT}" '
            f'stroke-width="1.6" stroke-dasharray="6 4"/><text x="{(x1 + x2) / 2}" y="{y2 - 6}" font-size="10" '
            f'font-weight="600" fill="{ACCENT}" text-anchor="middle">edge of the person</text>'
        )
    if perspective == "Build":
        ex, ey = _ENTRY
        overlay.append(
            f'<rect x="{ex - 48}" y="{ey - 16}" width="96" height="30" rx="8" fill="{SURFACE}" stroke="{MUTED}" stroke-width="1.4"/>'
            f'<text x="{ex}" y="{ey + 3}" font-size="10.5" font-weight="600" fill="{INK}" text-anchor="middle">Pipeline entry</text>'
        )

    for i, m in enumerate(modalities):
        color = colors.get(m.category, MUTED)
        key = f"m{i}"
        if m.is_body_sensed and m.category in _SITES:
            parts.append(_sensor(m.category, color))
            sx, sy = _SITES[m.category]
            where = f"Sensed at: {escape(str(m.body_region))}."
        else:
            sx, sy = 270, _EXTERNAL_Y.get(m.category, 300)
            marks.append(
                f'<g data-key="{key}"><rect class="hit" x="{sx - 22}" y="{sy - 18}" width="96" height="40" rx="8" fill="{SURFACE}" stroke="{color}" stroke-width="1.4"/>'
                f'{_external_icon(m.category, sx - 16, sy - 10, color)}'
                f'<text x="{sx + 10}" y="{sy + 6}" font-size="10.5" font-weight="600" fill="{INK}">{escape(SHORT_LABEL.get(m.category, m.category))}</text></g>'
            )
            where = "Never sensed from the body."
            sx, sy = sx - 22, sy
        if perspective == "Agency":
            payload[key] = f"<strong>{escape(m.name)}: how much control does the person have?</strong>" + _agency_list(agency[m.category])
        elif perspective == "Privacy":
            payload[key] = (
                f"<strong>{escape(m.name)}</strong> · {escape(m.signal_examples)} {where}"
                f" <em>Where it leaves the person:</em> {escape(m.privacy_exit_point)}"
            )
        else:
            payload[key] = f"<strong>{escape(m.name)}</strong> · {escape(m.signal_examples)} {where}"
        if m.is_body_sensed and m.category in _SITES:
            label = SHORT_LABEL.get(m.category, m.category)
            lx = 6 if sx < 130 else 214
            anchor = "start"
            marks.append(
                f'<g data-key="{key}"><circle class="hit" cx="{sx}" cy="{sy}" r="13" fill="{color}" fill-opacity="0.18" stroke="{color}" stroke-width="1.6"/>'
                f'<line x1="{sx + (-13 if sx < 130 else 13)}" y1="{sy}" x2="{lx + (34 if sx < 130 else 0)}" y2="{sy}" stroke="{color}" stroke-width="1"/>'
                f'<text x="{lx}" y="{sy - 4}" font-size="10.5" font-weight="600" fill="{color}" text-anchor="{anchor}">{escape(label)}</text></g>'
            )
        if perspective in ("Build", "Agency"):
            ex, ey = _ENTRY
            overlay.append(
                f'<path d="M{sx},{sy} C{sx},{(sy + ey) / 2} {ex - 60},{ey} {ex - 48},{ey}" fill="none" stroke="{color}" '
                f'stroke-width="1.6" stroke-opacity="0.75" stroke-dasharray="5 3"/>'
            )
        if perspective == "Privacy" and m.is_body_sensed and m.category in _SITES:
            x1, _, x2, _ = _BOUNDARY
            edge = x2 if sx >= 130 else x1
            overlay.append(
                f'<line x1="{sx}" y1="{sy}" x2="{edge + (10 if edge == x2 else -10)}" y2="{sy}" stroke="{color}" stroke-width="1.8" marker-end="url(#exit)"/>'
            )

    if perspective == "Agency":
        ex, ey = _ENTRY
        overlay.append(
            f'<rect x="{ex - 48}" y="{ey - 16}" width="96" height="30" rx="8" fill="{SURFACE}" stroke="{MUTED}" stroke-width="1.4"/>'
            f'<text x="{ex}" y="{ey + 3}" font-size="10.5" font-weight="600" fill="{INK}" text-anchor="middle">Pipeline entry</text>'
        )

    height = 400
    if perspective == "Agency":
        table, bottom = _agency_table(modalities, agency, colors)
        marks.append(table)
        height = bottom + 8
    names = ", ".join(m.name for m in modalities)
    svg = (
        f'<svg viewBox="0 0 360 {height:.0f}" role="img" aria-label="{escape(perspective)} view of where each signal comes from: {escape(names)}." '
        f'style="width:100%;height:auto;display:block"><defs><marker id="exit" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="{ACCENT}"/></marker></defs>'
        f'{"".join(parts)}{"".join(overlay)}{"".join(marks)}</svg>'
    )
    # The Agency table labels each row with its colour, so a legend would repeat it.
    legend = "" if perspective == "Agency" else "".join(_chip(colors.get(m.category, MUTED), m.name) for m in modalities)
    caveat = {
        "Build": "Build view: every signal, sensed from the body or not, runs into the same pipeline entry.",
        "Privacy": "Privacy view: body-sensed signals cross the edge of the person at their sensor; the others were never inside it.",
        "Agency": "\u2713 yes \u00b7 \u2715 no \u00b7 ? unknown: the cited sources do not say. Tap a row to read each answer in a sentence, with the source text.",
    }[perspective]
    info = "Tap a row to read what each mark means for that signal." if perspective == "Agency" else "Tap a sensor or a card to see what it measures."
    return _panel(svg, legend, caveat, info, payload, max_width=460)
