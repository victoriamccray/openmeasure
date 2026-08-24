"""
HealthRing Worked Example - a guided validation journey, not a workflow.

This page is a prototype of OpenMeasure's immersive worked-example
architecture, not a production Wearables module: it records nothing to
shared/handoff.py, carries no module_key, and is deliberately not a
numbered page, so it needs no entry in shared/catalog.py (see
shared/tests/test_catalog.py, which only requires a catalog entry for
numbered pages) and cannot appear on the overview's progress cards or
stage strip. Explore Real Data and Method Selection are the existing
pages that already establish this "not a workflow" pattern.

Archive I/O lives here, not in modules/healthring/core/, because core is
pure statistics on an already-loaded DataFrame with no I/O of its own
(see docs/design-standards.md section 1). The HealthRing archive
(RingDatasetV2.1_submission.zip, Zenodo record 18426864) is never
bundled or redistributed by this repository, and only this page reads
it. Two ways in feed the same loader below: a local filesystem path
(the original flow, still available when running locally with the
archive already on disk) and a Streamlit file uploader (needed because
the hosted app has no access to a path on the visitor's computer). An
uploaded archive is written to a session-scoped temp file so
_index_zip_entries/_load_subjects, both written against a path, never
need a second, in-memory code path; the temp file is never redistributed
and lives only as long as the hosting process does.

Known upstream issue, carried over from scripts/healthring_prototype.py:
the published archive is missing its central directory and its final
entry is truncated (verified by MD5 match against Zenodo's published
checksum -- this is not a bad download). Standard zipfile.ZipFile cannot
open it, so entries are located by walking local file headers directly.

Security note: this page unpickles data with the standard `pickle`
module, which can execute arbitrary code for a malicious file. Only
point it at a HealthRing archive whose checksum you have verified.

This page does not load or display raw PPG/accelerometer waveforms.
Only per-window summary columns (hr, bvp_hr, quality, Label) are read;
the raw signal is described in prose in the "Understand measurement"
stage, not visualized, because visualizing it would mean loading a much
larger, currently-unused part of each subject's file -- new scope worth
proposing separately rather than adding quietly here.
"""

from __future__ import annotations

import math
import pickle
import struct
import sys
import tempfile
import uuid
import zlib
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from modules.healthring.core import acquisition_robustness as ar
from shared.charts import multiline_time_series_chart
from shared.datasets import DATASETS
from shared.data_handling import disclosure_for, render_data_handling_summary
from shared.report import caveat, flagged_item_note, section_header

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

HEALTHRING_DATASET = next(item for item in DATASETS if item.id == "healthring")

# Each stage unlocks after the reader makes a research decision or
# inspection in the stage before it, rather than all nine-turned-ten
# sections simply being available on scroll. STAGE_KEY holds the highest
# unlocked index; a stage's own content is rendered once
# STAGE_KEY >= its index, and stays visible (and its widgets stay live)
# after later stages unlock too.
STAGE_KEY = "hr_stage"

STAGE_RESEARCH_QUESTION = 0
STAGE_UNDERSTAND_MEASUREMENT = 1
STAGE_SIGNAL_INSPECTION = 2
STAGE_DESIGN_EVALUATION = 3
STAGE_BASELINE = 4
STAGE_MODEL = 5
STAGE_EVALUATE = 6
STAGE_RETENTION = 7
STAGE_CONDITIONS_CHECK = 8
STAGE_CONCLUSION = 9
STAGE_FINISH = 10

JOURNEY_STAGES = (
    "Research question",
    "Understand measurement",
    "Signal inspection",
    "Design evaluation",
    "Establish baseline",
    "Build model",
    "Evaluate",
    "Research decision",
    "Across conditions?",
    "Defend conclusion",
    "Finish study",
)

SPLIT_PARTICIPANT = "participant"
SPLIT_WINDOW = "window"

SPLIT_CHOICE_LABELS = {
    SPLIT_PARTICIPANT: "Participant-level (holds out whole participants)",
    SPLIT_WINDOW: "Window-level (splits by measurement window, ignoring participant)",
}

CONCLUSION_OPTIONS = (
    "The model is ready to use as-is, across conditions",
    (
        "The model works better in some conditions than others, and would "
        "need condition-specific checking before wider use"
    ),
    "The model doesn't clearly beat the raw baseline given this data",
    "Not enough evidence either way from this sample",
)

# v0.1 scope: one ring hardware design only. The archive also contains a
# second ring design (ring2); mixing both into one model would confound
# "does the model generalize" with "do the two rings even measure the
# same thing," so ring2 stays a stated next check instead.
RING_ENTRY = "ring1"

RAW_COLUMNS: tuple[str, ...] = ("Label", "hr", "bvp_hr", "ir-quality", "red-quality")

# Columns for the Signal Inspection stage only. HealthRing's per-window
# raw pickle carries roughly 60 columns (every channel repeated as raw,
# standardized, filtered, difference, welch, and respiratory-rate
# variants); this selects the "-filtered" PPG/ACC waveforms HealthRing
# itself already computed, plus fs (the sampling rate, used to build a
# time axis). No new signal processing happens on this page: these
# columns are used as HealthRing provides them.
SIGNAL_INSPECTION_COLUMNS: tuple[str, ...] = (
    "Label",
    "hr",
    "bvp_hr",
    "ir-quality",
    "red-quality",
    "ir-filtered",
    "red-filtered",
    "ax-filtered",
    "ay-filtered",
    "az-filtered",
    "fs",
)

DEFAULT_ZIP_PATH = ROOT / "RingDatasetV2.1_submission.zip"

# Palette matching shared/report.py-adjacent OpenMeasure chart conventions
# and scripts/healthring_prototype.py's existing choices, validated for
# categorical/sequential use (see the dataviz skill's reference palette).
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
ACCENT = "#2a78d6"
ACCENT_2 = "#eb6834"
ACCENT_3 = "#1baf7a"

_VEGA_CHART_CONFIG = {
    "background": SURFACE,
    "axis": {
        "gridColor": GRIDLINE,
        "domainColor": "#c3c2b7",
        "tickColor": "#c3c2b7",
        "labelColor": INK_MUTED,
        "titleColor": INK_SECONDARY,
    },
    "view": {"stroke": "transparent"},
}

# ---------------------------------------------------------------------
# Original, static icons for the "Understand measurement" stage.
#
# Static on purpose, not animated: a flickering/pulsing version of a
# similar icon set on the GRAND worked example was flagged as both a
# seizure-trigger risk (rapid, high-contrast flicker) and useless to a
# screen-reader user, who already has the same information in the prose
# and unit labels next to each icon. These are original, generic
# pictographs, not real recorded PPG/accelerometer traces or a
# reproduction of any device's actual display.
# ---------------------------------------------------------------------

_MEASUREMENT_ICON_PATHS = {
    "ppg": (
        '<circle cx="5" cy="16" r="3" fill="{c}"/>'
        '<path d="M9 16L13 16L16 8L19 24L22 12L25 16L29 16" '
        'stroke="{c}" stroke-width="1.8" fill="none" stroke-linecap="round"/>'
    ),
    "acc": (
        '<line x1="16" y1="16" x2="27" y2="16" stroke="{c}" stroke-width="1.6"/>'
        '<polygon points="27,16 22.5,13.5 22.5,18.5" fill="{c}"/>'
        '<line x1="16" y1="16" x2="16" y2="5" stroke="{c}" stroke-width="1.6"/>'
        '<polygon points="16,5 13.5,9.5 18.5,9.5" fill="{c}"/>'
        '<line x1="16" y1="16" x2="8" y2="24" stroke="{c}" stroke-width="1.6"/>'
        '<polygon points="8,24 12,21.5 10.5,25.5" fill="{c}"/>'
        '<circle cx="16" cy="16" r="1.6" fill="{c}"/>'
    ),
    "window": (
        '<rect x="3" y="12" width="26" height="8" rx="1" fill="{c}" opacity="0.12"/>'
        '<line x1="10.5" y1="11" x2="10.5" y2="21" stroke="{c}" stroke-width="1"/>'
        '<line x1="16" y1="11" x2="16" y2="21" stroke="{c}" stroke-width="1"/>'
        '<line x1="21.5" y1="11" x2="21.5" y2="21" stroke="{c}" stroke-width="1"/>'
        '<rect x="10.5" y="12" width="5.5" height="8" fill="{c}" opacity="0.45"/>'
    ),
    "hr": (
        '<path d="M3 16L9 16L12 7L16 25L19 11L22 16L29 16" '
        'stroke="{c}" stroke-width="2" fill="none" stroke-linecap="round"/>'
    ),
    "quality": (
        '<path d="M6 23A10 10 0 0 1 26 23" stroke="{c}" stroke-width="2" fill="none"/>'
        '<line x1="16" y1="23" x2="21" y2="15" stroke="{c}" stroke-width="1.6" stroke-linecap="round"/>'
        '<circle cx="16" cy="23" r="1.6" fill="{c}"/>'
    ),
}

MEASUREMENT_LEGEND = (
    ("ppg", "PPG waveform", "light returned from skin"),
    ("acc", "ACC (motion)", "3-axis acceleration"),
    ("window", "Window", "one fixed-length segment"),
    ("hr", "HR", "beats per minute"),
    ("quality", "Signal quality", "0-1 scale"),
)

# "raw" and "filtered" trace two cycles of a PPG pulse's established shape
# (systolic upstroke -> systolic peak -> dicrotic notch -> diastolic wave
# -> decay to baseline; Elgendi, 2012 - see PPG_MORPHOLOGY_CITATION below),
# not an arbitrary squiggle: "raw" draws it as jagged line segments to
# read as noisy, "filtered" draws the same landmarks as a smooth curve.
# Still not a real recorded trace - this page loads no raw waveform (see
# module docstring) - only a more accurately shaped stand-in for one.
_PIPELINE_STAGE_ICON_PATHS = {
    "raw": (
        '<path d="M2,24 L4,20 L5,7 L6,17 L7,12 L9,20 L12,24 '
        'L14,22 L16,20 L17,7 L18,17 L19,12 L21,20 L24,24 L30,24" '
        'stroke="{c}" stroke-width="1.3" fill="none" stroke-linecap="round"/>'
    ),
    "filtered": (
        '<path d="M2,24 C4,10 5,6 6,11 C7,15 9,19 12,24 '
        'C14,22 16,10 17,6 C18,11 19,15 21,19 C22,22 23,24 24,24 L30,24" '
        'stroke="{c}" stroke-width="1.6" fill="none" stroke-linecap="round" '
        'stroke-linejoin="round"/>'
    ),
    "windows": _MEASUREMENT_ICON_PATHS["window"],
    "derived": (
        '<circle cx="16" cy="16" r="10" fill="{c}" opacity="0.14" stroke="{c}" stroke-width="1.3"/>'
        '<line x1="10" y1="16" x2="22" y2="16" stroke="{c}" stroke-width="1.4"/>'
        '<line x1="16" y1="10" x2="16" y2="22" stroke="{c}" stroke-width="1.4"/>'
    ),
}

PIPELINE_STAGES = (
    ("raw", "Raw PPG signal"),
    ("filtered", "Filtering"),
    ("windows", "Windows"),
    ("derived", "Derived measurement"),
)

PIPELINE_STAGE_HOVER = {
    "raw": (
        "Normative shape of a PPG pulse: systolic upstroke, systolic "
        "peak, dicrotic notch, and diastolic wave (Elgendi, 2012)."
    ),
    "filtered": (
        "The same pulse landmarks, smoothed. Filtering removes noise; "
        "it does not remove the underlying physiological shape."
    ),
    "windows": (
        "The smoothed signal cut into fixed-length windows. One "
        "segment (highlighted) becomes one row in this dataset."
    ),
    "derived": "One derived value (e.g. HR) computed from one window.",
}

PPG_MORPHOLOGY_CITATION = (
    "Elgendi, M. (2012). On the analysis of fingertip photoplethysmogram "
    "signals. Current Cardiology Reviews, 8(1), 14-25. "
    "https://doi.org/10.2174/157340312801215782"
)


def _icon_svg(path_template: str, color: str, cx: float, cy: float, scale: float = 1.0) -> str:
    path = path_template.format(c=color)
    tx, ty = cx - 16 * scale, cy - 16 * scale
    return f'<g transform="translate({tx:.1f},{ty:.1f}) scale({scale:.3f})">{path}</g>'


def _arrow_svg(x1: float, y1: float, x2: float, y2: float, color: str) -> str:
    angle = math.atan2(y2 - y1, x2 - x1)
    hx1 = x2 - 6 * math.cos(angle - math.pi / 7)
    hy1 = y2 - 6 * math.sin(angle - math.pi / 7)
    hx2 = x2 - 6 * math.cos(angle + math.pi / 7)
    hy2 = y2 - 6 * math.sin(angle + math.pi / 7)
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="1.5" opacity="0.6"/>'
        f'<polygon points="{x2:.1f},{y2:.1f} {hx1:.1f},{hy1:.1f} {hx2:.1f},{hy2:.1f}" '
        f'fill="{color}" opacity="0.8"/>'
    )


def _measurement_legend_html() -> str:
    """A static, at-a-glance legend for the vocabulary just introduced in
    prose above: what each icon stands for and its real unit or scale."""

    items = []
    for key, label, unit in MEASUREMENT_LEGEND:
        icon = _icon_svg(_MEASUREMENT_ICON_PATHS[key], ACCENT, 16, 16, 1.0)
        items.append(
            f'<div style="display:flex; flex-direction:column; align-items:center; '
            f'width:88px; gap:2px;">'
            f'<svg width="32" height="32" viewBox="0 0 32 32">{icon}</svg>'
            f'<span style="font-size:11px; color:{INK_PRIMARY}; text-align:center;">{label}</span>'
            f'<span style="font-size:9.5px; color:{INK_SECONDARY}; text-align:center;">{unit}</span>'
            f"</div>"
        )

    return f"""
    <div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
                background:{SURFACE}; border-radius:6px; padding:10px 4px;
                display:flex; flex-wrap:wrap; justify-content:center; gap:6px;">
      {"".join(items)}
    </div>
    """


def _signal_pipeline_glyph_html() -> str:
    """
    A static box-and-arrow rendering of the 'raw PPG signal -> filtering
    -> windows -> derived measurement' chain described in prose above.

    Each stage's icon carries a native SVG <title>, so hovering it shows
    PIPELINE_STAGE_HOVER's explanation - no click or animation needed.
    """

    slot_w, gap, badge = 84.0, 26.0, 40.0
    boxes: list[str] = []
    arrows: list[str] = []

    for i, (key, label) in enumerate(PIPELINE_STAGES):
        x = i * (slot_w + gap) + (slot_w - badge) / 2
        cx, cy = x + badge / 2, badge / 2 + 4

        if i > 0:
            prev_x = (i - 1) * (slot_w + gap) + (slot_w - badge) / 2
            arrows.append(_arrow_svg(prev_x + badge, cy, x, cy, GRIDLINE))

        boxes.append(f'<g style="cursor:help;"><title>{PIPELINE_STAGE_HOVER[key]}</title>')
        boxes.append(
            f'<rect x="{x:.0f}" y="4" width="{badge:.0f}" height="{badge:.0f}" rx="8" '
            f'fill="{ACCENT}" opacity="0.1" stroke="{ACCENT}" stroke-width="1.1"/>'
        )
        boxes.append(_icon_svg(_PIPELINE_STAGE_ICON_PATHS[key], ACCENT, cx, cy, badge / 32))
        boxes.append("</g>")
        boxes.append(
            f'<text x="{i * (slot_w + gap) + slot_w / 2:.0f}" y="{badge + 20:.0f}" '
            f'text-anchor="middle" font-size="10" fill="{INK_SECONDARY}">{label}</text>'
        )

    scene_w = len(PIPELINE_STAGES) * (slot_w + gap) - gap + 8

    return f"""
    <div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
                background:{SURFACE}; border-radius:6px; padding:10px 4px;">
      <svg width="100%" height="70" viewBox="-4 0 {scene_w:.0f} 70" preserveAspectRatio="xMidYMid meet">
        {"".join(arrows)}
        {"".join(boxes)}
      </svg>
    </div>
    """


# ---------------------------------------------------------------------
# Archive I/O (page-level; core/ never touches a file)
# ---------------------------------------------------------------------


def _index_zip_entries(zip_path: Path) -> dict[str, tuple[int, int, int]]:
    """
    Map entry name -> (data offset, compressed size, compression method).

    Walks local file headers sequentially rather than using zipfile.ZipFile,
    because the published archive's central directory is missing. General-
    purpose flag bit 3 is unset for every intact entry, so local headers
    carry real compressed/uncompressed sizes and entries can be located
    without it. Stops at the first entry with no recoverable size, which is
    the one entry (00029_ring2) known to be genuinely truncated upstream.
    """

    index: dict[str, tuple[int, int, int]] = {}

    with zip_path.open("rb") as handle:
        handle.seek(0, 2)
        filesize = handle.tell()
        pos = 0

        while pos < filesize - 4:
            handle.seek(pos)

            if handle.read(4) != b"PK\x03\x04":
                break

            (_ver, flags, method, _mtime, _mdate, _crc32, csize, _usize, nlen, elen) = (
                struct.unpack("<HHHHHIIIHH", handle.read(26))
            )
            name = handle.read(nlen).decode("utf-8", errors="replace")
            handle.seek(elen, 1)
            data_offset = handle.tell()

            if flags & 0x8 or csize == 0:
                break

            index[name] = (data_offset, csize, method)
            pos = data_offset + csize

    return index


def _session_temp_dir() -> Path:
    """
    A per-session scratch directory for uploaded archives.

    Keyed on a uuid generated once per Streamlit session (not on anything
    upload-specific), so two visitors uploading the same file never share
    a path, and the archive lives only as long as this session's temp
    files do.
    """

    session_id = st.session_state.setdefault("hr_session_id", uuid.uuid4().hex)
    tmp_dir = Path(tempfile.gettempdir()) / "openmeasure_healthring" / session_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def _save_uploaded_archive(uploaded_file) -> Path:
    """
    Persist an uploaded archive to a session-scoped temp file, keyed by
    content hash, so the same path-based loader below (_available_subject_ids,
    _load_subjects, _load_subject_signal) reads it with no separate
    in-memory code path.

    Keying on the hash rather than always writing means a rerun that still
    has the same file selected in the uploader (Streamlit keeps it in
    widget state across reruns) does not rewrite a potentially
    30-160+ MB archive to disk on every script pass.
    """

    content = uploaded_file.getvalue()
    digest = sha256(content).hexdigest()
    tmp_path = _session_temp_dir() / f"{digest}.zip"

    if not tmp_path.is_file():
        tmp_path.write_bytes(content)

    return tmp_path


def _read_entry(zip_path: Path, offset: int, csize: int, method: int) -> bytes:
    """Read and decompress one entry, given its indexed location."""

    with zip_path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(csize)

    if method == 0:
        return raw

    if method == 8:
        return zlib.decompress(raw, -15)

    raise ValueError(f"Unsupported ZIP compression method {method}.")


@st.cache_data(show_spinner=False)
def _available_subject_ids(zip_path_str: str) -> tuple[int, ...]:
    """Participant IDs with an intact ring1 entry in the archive."""

    index = _index_zip_entries(Path(zip_path_str))
    suffix = f"_{RING_ENTRY}_processed.pkl"

    return tuple(
        sorted(int(name[: -len(suffix)]) for name in index if name.endswith(suffix))
    )


@st.cache_data(show_spinner="Reading participant files from the local archive...")
def _load_subjects(zip_path_str: str, subject_ids: tuple[int, ...]) -> pd.DataFrame:
    """Load and concatenate ring1 windows for the requested participants."""

    zip_path = Path(zip_path_str)
    index = _index_zip_entries(zip_path)

    frames: list[pd.DataFrame] = []

    for subject_id in subject_ids:
        entry_name = f"{subject_id:05d}_{RING_ENTRY}_processed.pkl"

        if entry_name not in index:
            continue

        offset, csize, method = index[entry_name]
        raw = pickle.loads(_read_entry(zip_path, offset, csize, method))

        missing = [column for column in RAW_COLUMNS if column not in raw.columns]

        if missing:
            raise KeyError(f"{entry_name} is missing expected columns: {missing}.")

        subject_frame = raw[list(RAW_COLUMNS)].copy()
        subject_frame["subject_id"] = subject_id
        frames.append(subject_frame)

    if not frames:
        raise ValueError("None of the requested participants were found in the archive.")

    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner="Reading this participant's raw signal...")
def _load_subject_signal(zip_path_str: str, subject_id: int) -> pd.DataFrame:
    """
    Load one participant's per-window PPG/ACC waveforms, for the Signal
    Inspection stage only.

    Deliberately scoped to a single participant: the raw pickle behind
    each participant's ring1 entry is 30-160 MB uncompressed once fully
    unpickled (pickle has no way to deserialize only some columns), so
    loading it for every participant already in the main journey would
    defeat the point of keeping that step bounded. Columns are trimmed
    to SIGNAL_INSPECTION_COLUMNS immediately after unpickling.
    """

    zip_path = Path(zip_path_str)
    index = _index_zip_entries(zip_path)
    entry_name = f"{subject_id:05d}_{RING_ENTRY}_processed.pkl"

    if entry_name not in index:
        raise KeyError(f"{entry_name} was not found in the archive.")

    offset, csize, method = index[entry_name]
    raw = pickle.loads(_read_entry(zip_path, offset, csize, method))

    missing = [column for column in SIGNAL_INSPECTION_COLUMNS if column not in raw.columns]

    if missing:
        raise KeyError(f"{entry_name} is missing expected columns: {missing}.")

    subject_frame = raw[list(SIGNAL_INSPECTION_COLUMNS)].copy()
    subject_frame["subject_id"] = subject_id

    return subject_frame


# ---------------------------------------------------------------------
# Chart builders (Vega-Lite specs; no matplotlib/plotly)
# ---------------------------------------------------------------------


def _bland_altman_chart(
    mean_values: pd.Series,
    diff_values: pd.Series,
    bias: float,
    lower_loa: float,
    upper_loa: float,
) -> dict:
    points = pd.DataFrame({"mean_hr": mean_values, "diff": diff_values}).to_dict("records")

    return {
        "layer": [
            {
                "data": {"values": points},
                "mark": {"type": "point", "filled": True, "size": 45, "opacity": 0.6},
                "encoding": {
                    "x": {
                        "field": "mean_hr",
                        "type": "quantitative",
                        "title": "Mean of predicted and reference HR (bpm)",
                    },
                    "y": {
                        "field": "diff",
                        "type": "quantitative",
                        "title": "Predicted - reference HR (bpm)",
                    },
                    "color": {"value": ACCENT},
                },
            },
            {
                "data": {"values": [{"y": bias}]},
                "mark": {"type": "rule", "color": INK_PRIMARY, "strokeWidth": 2},
                "encoding": {"y": {"field": "y", "type": "quantitative"}},
            },
            {
                "data": {"values": [{"y": upper_loa}, {"y": lower_loa}]},
                "mark": {"type": "rule", "color": INK_SECONDARY, "strokeDash": [4, 4]},
                "encoding": {"y": {"field": "y", "type": "quantitative"}},
            },
        ],
        "width": "container",
        "height": 320,
        "config": _VEGA_CHART_CONFIG,
    }


def _condition_mae_chart(
    breakdown: tuple[ar.ConditionBreakdown, ...],
    aggregate_mae: float,
    group_title: str,
) -> dict:
    rows = [{"group": item.group, "mae": item.mae} for item in breakdown]
    order = [item.group for item in breakdown]

    return {
        "layer": [
            {
                "data": {"values": rows},
                "mark": {"type": "bar", "size": 22, "color": ACCENT},
                "encoding": {
                    "x": {
                        "field": "group",
                        "type": "nominal",
                        "sort": order,
                        "title": group_title,
                    },
                    "y": {"field": "mae", "type": "quantitative", "title": "MAE (bpm)"},
                },
            },
            {
                "data": {"values": [{"y": aggregate_mae}]},
                "mark": {"type": "rule", "color": INK_PRIMARY, "strokeWidth": 2, "strokeDash": [4, 4]},
                "encoding": {"y": {"field": "y", "type": "quantitative"}},
            },
        ],
        "width": "container",
        "height": 280,
        "config": _VEGA_CHART_CONFIG,
    }


def _error_distribution_chart(data: pd.DataFrame, error_col: str, group_col: str) -> dict:
    rows = data[[group_col, error_col]].rename(
        columns={group_col: "group", error_col: "abs_error"}
    ).to_dict("records")

    return {
        "data": {"values": rows},
        "mark": {"type": "boxplot", "extent": "min-max", "color": ACCENT},
        "encoding": {
            "x": {"field": "group", "type": "nominal", "title": group_col},
            "y": {"field": "abs_error", "type": "quantitative", "title": "Absolute error (bpm)"},
        },
        "width": "container",
        "height": 280,
        "config": _VEGA_CHART_CONFIG,
    }


# ---------------------------------------------------------------------
# Stage-gating, pipeline, and plain-language interpretation helpers
# ---------------------------------------------------------------------


def _current_stage() -> int:
    return st.session_state.get(STAGE_KEY, STAGE_RESEARCH_QUESTION)


def _advance_to(stage: int) -> None:
    """Unlock through `stage` and force an immediate rerun.

    A rerun is needed, not just the session_state write, because the
    button click that calls this is already mid-script: without
    rerunning, the rest of this same pass would still read the stale
    frontier and the newly unlocked section would not appear until some
    later, unrelated interaction triggered a rerun on its own.
    """

    st.session_state[STAGE_KEY] = max(_current_stage(), stage)
    st.rerun()


def _fit_and_evaluate(
    split: ar.SplitResult,
) -> tuple[ar.RecalibrationModel, ar.AgreementResult, pd.DataFrame]:
    """
    Fit on split.train_data, predict on split.test_data, and summarize
    agreement -- the same three core calls, reused for the chosen split,
    the "what if" comparison split, and (with a different split) nowhere
    else, so this exists once rather than being copied three times.
    """

    model = ar.fit_recalibration(split.train_data)

    test_data = split.test_data.copy()
    test_data["predicted_hr"] = ar.apply_recalibration(model, test_data["bvp_hr"])
    test_data["pred_abs_error"] = (test_data["predicted_hr"] - test_data["hr"]).abs()
    test_data["pred_diff"] = test_data["predicted_hr"] - test_data["hr"]
    test_data["pred_mean_hr"] = (test_data["predicted_hr"] + test_data["hr"]) / 2

    evaluation = ar.agreement_summary(test_data["predicted_hr"], test_data["hr"])

    return model, evaluation, test_data


# Every MAE, bias, and limits-of-agreement metric on this page is followed
# by one of these sentences, so a number never appears without a plain-
# language reading. Written once here rather than at each of the three
# call sites (baseline, evaluate, retention) that need the same reading.


def _mae_sentence(value: float) -> str:
    return f"Predictions differed from the reference by about {value:.1f} bpm on average."


def _bias_sentence(value: float) -> str:
    if abs(value) < 0.05:
        return (
            "Predictions were about equal to the reference on average, "
            "with no consistent over- or under-estimate."
        )

    direction = "higher" if value > 0 else "lower"
    return f"Predictions were about {abs(value):.1f} bpm {direction} than the reference on average."


def _loa_sentence(lower: float, upper: float) -> str:
    return (
        f"For about 95% of windows, the error is expected to fall between "
        f"{lower:+.1f} and {upper:+.1f} bpm, following the Bland-Altman "
        "convention (bias plus or minus 1.96 standard deviations)."
    )


def _quality_sentence(value: float) -> str:
    if value >= 0.7:
        level = "high"
    elif value >= 0.4:
        level = "middling"
    else:
        level = "low"

    return (
        f"The ring scored this window's own reading as {level} usability "
        f"({value:.2f} on its 0-1 scale). That score does not, by itself, "
        "say what made it that way."
    )


def _render_signal_window(row: pd.Series, slot_key: str) -> None:
    """
    Walk through one real measurement window: activity, movement, PPG,
    signal quality, HR estimate, then (only after a prediction) error.

    slot_key makes every widget key unique so this can render twice in
    one script pass, for the two-window comparison, without Streamlit
    raising a duplicate-widget-ID error.
    """

    st.markdown(f"**Activity: {row['Label']}**")

    fs = float(row["fs"])
    n_samples = len(row["ax-filtered"])
    t = np.arange(n_samples) / fs

    st.caption("Movement (accelerometer, three axes):")
    st.vega_lite_chart(
        multiline_time_series_chart(
            t,
            {"x": row["ax-filtered"], "y": row["ay-filtered"], "z": row["az-filtered"]},
            {"x": ACCENT, "y": ACCENT_2, "z": ACCENT_3},
            "Acceleration (filtered, arbitrary units)",
            config=_VEGA_CHART_CONFIG,
        ),
        theme=None,
        use_container_width=True,
    )

    st.caption("PPG (photoplethysmography), two light channels:")
    st.vega_lite_chart(
        multiline_time_series_chart(
            t,
            {"infrared": row["ir-filtered"], "red": row["red-filtered"]},
            {"infrared": ACCENT, "red": ACCENT_2},
            "PPG signal (filtered, arbitrary units)",
            config=_VEGA_CHART_CONFIG,
        ),
        theme=None,
        use_container_width=True,
    )

    quality = (row["ir-quality"] + row["red-quality"]) / 2
    st.caption(_quality_sentence(quality))

    hr_col1, hr_col2 = st.columns(2)
    hr_col1.metric("Ring estimate (bvp_hr)", f"{row['bvp_hr']:.1f} bpm")
    hr_col2.metric("Reference (hr)", f"{row['hr']:.1f} bpm")

    predict_key = f"hr_signal_predict_{slot_key}"
    reveal_key = f"hr_signal_reveal_{slot_key}"

    st.radio(
        "Before revealing the error: do you expect this window's HR "
        "estimate to be accurate?",
        options=["Yes", "No", "Not sure"],
        index=2,
        key=predict_key,
    )

    if not st.session_state.get(reveal_key, False):
        if st.button("Reveal error", key=f"hr_signal_reveal_button_{slot_key}"):
            st.session_state[reveal_key] = True
            st.rerun()

    if st.session_state.get(reveal_key, False):
        abs_error = abs(row["bvp_hr"] - row["hr"])
        prediction = st.session_state.get(predict_key, "Not sure")
        st.metric("Absolute error", f"{abs_error:.1f} bpm")
        st.caption(f"You predicted: {prediction}. {_mae_sentence(abs_error)}")


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="OpenMeasure - Wearables Research Journey",
    layout="centered",
)

st.title("Wearables Research Journey")
st.caption("This prototype uses the HealthRing dataset specifically.")
st.caption(
    "A guided case study: each stage unlocks after you make a decision "
    "or inspect its consequence."
)

render_data_handling_summary(disclosure_for("pages/HealthRing_Worked_Example.py"))

stage = _current_stage()

# A single wrapped line, not a fixed grid of narrow columns: ten short
# labels do not all fit side by side at "centered" page width, and a
# rigid st.columns() split forced text to overflow its column instead of
# wrapping. Plain text wraps naturally at any width.
_stage_parts = [
    f"**{label}**" if index == stage else label
    for index, label in enumerate(JOURNEY_STAGES)
]

with st.container(border=True):
    st.markdown(" → ".join(_stage_parts))

if stage > STAGE_RESEARCH_QUESTION:
    if st.button("Restart study", icon=":material/restart_alt:"):
        for key in (STAGE_KEY, "healthring_windows", "healthring_n_subjects", "hr_reveal_breakdown"):
            st.session_state.pop(key, None)
        st.rerun()

st.divider()

# -----------------------------------------------------------------
# 0. Research question
# -----------------------------------------------------------------

section_header("Research Question")

st.markdown(
    "### Can We Trust Ring-Derived Heart Rate Across Real-World Conditions?"
)

st.write(
    "A wearable ring reports a heart-rate number with no error bars "
    "attached. This page runs a real validation of one ring's heart-rate "
    "estimate against a reference device, across real activity "
    "conditions. It uses real data, not a demonstration built to make a "
    "model look good, and it explains each term as it comes up."
)

rq_col1, rq_col2 = st.columns(2)
with rq_col1:
    st.badge("Ring estimate", icon=":material/watch:", color="blue")
with rq_col2:
    st.badge("Reference device", icon=":material/monitor_heart:", color="blue")

st.info(
    "\"Smart rings enable unobtrusive monitoring of cardiovascular "
    "vital signs via photoplethysmography (PPG), yet rigorous "
    "validation is limited by the scarcity of open, multi-parameter "
    "datasets.\" HealthRing is that dataset: three synchronized "
    "cohorts from 54 adults, recorded on two custom ring designs "
    "(reflective and transmissive PPG) alongside clinical-grade "
    "reference devices."
)

with st.expander("Dataset access and citation"):
    st.write(
        f"**{HEALTHRING_DATASET.name}** ({HEALTHRING_DATASET.domain}). "
        f"{HEALTHRING_DATASET.description}"
    )
    for source in HEALTHRING_DATASET.sources:
        st.markdown(f"[{source.label}]({source.url})")
    st.caption(f"Access: {HEALTHRING_DATASET.access}")
    st.caption(HEALTHRING_DATASET.citation)
    st.caption(
        "This page never bundles HealthRing data. Bring your own copy of "
        "the archive, either by uploading it or by pointing at a local "
        "path, and it is used only for this session: not modified, "
        "stored beyond the session, or redistributed."
    )

with st.expander("What was found by HealthRing researchers"):
    st.write(
        "On the controlled and daily-life cohorts, the paper's own "
        "physics-based and supervised benchmarks reach mean absolute "
        "errors of 5.33 BPM (heart rate), 2.98 breaths/min (respiratory "
        "rate), 1.72% (SpO2), 12.98 mmHg (systolic blood pressure), and "
        "7.64 mmHg (diastolic blood pressure)."
    )
    st.write(
        "On the treadmill cohort, fine-tuning cuts heart-rate error "
        "from 36.91 to 23.99 BPM, and respiratory-rate error from 5.44 "
        "to 4.61 breaths/min, relative to applying a model with no "
        "retraining on that cohort's motion."
    )
    st.write(
        "Other findings from the paper, useful context before this "
        "walkthrough runs its own, much simpler model: supervised "
        "models consistently beat physics-based methods, but the best "
        "model varied by task; error increases sharply with more "
        "intense motion, roughly tripling versus stationary scenarios; "
        "blood pressure was the hardest of the four vital signs to "
        "estimate; adding extra sensor channels (a second PPG "
        "wavelength, the accelerometer) gave only modest, inconsistent "
        "gains depending on ring design; and their best supervised "
        "models matched or beat two commercial rings (Samsung Galaxy "
        "Ring, Oura Ring) on heart rate."
    )
    st.caption(
        "This page fits one predictor with ordinary least squares, not "
        "the supervised models the paper benchmarks. Treat these as "
        "prior findings from the literature to compare this walkthrough's "
        "own result against, not a bar this page's model is expected to "
        "clear."
    )

if stage < STAGE_UNDERSTAND_MEASUREMENT:
    if st.button("Begin study", type="primary"):
        _advance_to(STAGE_UNDERSTAND_MEASUREMENT)

# -----------------------------------------------------------------
# 1. Understand measurement
# -----------------------------------------------------------------

windows = None
predicted_problems: list[str] = []

if stage >= STAGE_UNDERSTAND_MEASUREMENT:
    section_header(
        "Understand Measurement",
        "What is actually being compared, before any analysis runs",
    )

    st.markdown(
        """
A wearable ring estimates heart rate using **PPG (photoplethysmography)**:
an LED shines light into the skin, and a sensor measures how much light
comes back. Blood volume near the skin rises and falls with each
cardiac cycle, so the returned light varies in step with it, producing
a repeating waveform that can be used to estimate HR.

Many rings also carry an **ACC (accelerometer)**, which measures physical
motion. During activity, its readings become larger and more erratic.
That matters here because motion can shake the sensor against the skin
and distort the PPG waveform it is trying to read.

From that waveform, a ring computes, for each **window** (a short,
fixed-length segment of time; one row in this dataset is one window):

- **HR (heart rate)**, in beats per minute, from how often the waveform
  repeats.
- A **signal quality** score. This reflects how usable the ring judged
  that window's reading to be. A low score does not, by itself, say
  what made the window less usable; this page does not assume a cause
  such as motion or poor contact unless the data shows it.

A related measure, **HRV (heart rate variability)**, describes the
beat-to-beat timing pattern rather than the rate itself. This dataset
does not include HRV, so it is not analyzed on this page.
"""
    )

    components.html(_measurement_legend_html(), height=90)
    st.caption(
        "Original, static icons standing in for the vocabulary above and "
        "its real unit or scale, not a real recorded waveform or a "
        "device's actual display."
    )

    with st.expander("How a raw signal becomes one number per window"):
        st.markdown(
            """
The pipeline behind `hr` and the quality score generally looks like:

**raw PPG signal -> filtering -> windows -> derived measurement**
"""
        )
        components.html(_signal_pipeline_glyph_html(), height=70)
        st.caption(
            "The pulse shape drawn for 'raw' and 'filtered' is the "
            "normative PPG waveform (systolic peak, dicrotic notch, "
            "diastolic wave); this page loads no raw waveform of its "
            "own (see module docstring). 'Windows' shows it split into "
            "fixed-length segments, and 'derived measurement' shows one "
            "number coming out of one window. Hover an icon for detail."
        )
        st.caption(PPG_MORPHOLOGY_CITATION)
        st.markdown(
            """
Filtering here means reducing unwanted parts of the signal, such as slow
drift or high-frequency noise. Filtering does not delete observations;
it reshapes the signal that is still there. A filter tuned to remove
noise can also remove real, useful signal if it is too aggressive.

That is a different operation from a decision later on this page:
choosing a minimum signal-quality threshold to decide which whole
windows to keep or exclude from analysis. That step does drop
observations. Signal filtering and excluding a window are two different
things, and this page keeps them separate.

This page does not load or display the raw PPG/accelerometer waveform
itself, only the per-window summary values described below.
"""
        )

    st.markdown(
        """
Each measurement window in this dataset carries:

- **`hr`**: the reference heart rate. "Reference" (sometimes called
  "ground truth") means the value treated as correct for comparison,
  taken from the study's own separate measurement device, not the ring.
- **`bvp_hr`**: the ring's own heart-rate estimate. This is what a
  deployed ring would actually report.
- **`Label`**: the recorded activity/acquisition condition (for
  example, resting, walking, or treadmill exercise).
- **`ir-quality` / `red-quality`**: the ring's own per-channel signal-
  quality estimate for the window, averaged below into one `quality`
  value.
"""
    )

    st.caption(
        "The options below are hypotheses drawn from the wider PPG "
        "measurement literature, not established facts about this "
        "specific dataset. Skin tone and perfusion in particular are a "
        "debated, actively studied topic in pulse-oximetry research, "
        "and this dataset does not record skin tone, so nothing on this "
        "page can settle that question either way; treat it as an open "
        "validation question, not a known bias."
    )

    predicted_problems = st.multiselect(
        "Before loading any data: which of these do you expect could hurt "
        "ring accuracy?",
        options=[
            "Motion during activity",
            "Poor skin contact or fit",
            "Skin tone or perfusion differences (open question, see note above)",
            "Low ambient light",
            "Session order or fatigue effects",
        ],
        help="Not graded. The 'Does it hold across conditions?' stage comes back to this.",
    )

    # A hosted Streamlit app has no access to a path on the visitor's
    # computer, so "browse" (a file uploader, which opens the browser's own
    # native file picker) has to be an option, not just a path typed in. The
    # default favors whichever is likely to actually work where this
    # script is running: if the archive already sits at DEFAULT_ZIP_PATH,
    # this is presumably a local run with the file on disk, so default to
    # path; otherwise (the hosted case, since the archive is never
    # bundled with this repo) default to browse.
    source_mode = st.radio(
        "How will you provide the archive?",
        options=["upload", "path"],
        format_func=lambda key: {
            "upload": "Browse for the archive on this computer",
            "path": "Enter a local filesystem path already on this machine",
        }[key],
        index=1 if DEFAULT_ZIP_PATH.exists() else 0,
        horizontal=True,
    )

    st.caption(
        "Most people want 'Browse': it opens your browser's own file "
        "picker, works for a file anywhere on disk (including Downloads), "
        "and cannot be mistyped. 'Enter a local filesystem path' only "
        "works if this app is running on the same machine where the "
        "archive already sits; on a hosted copy of this app, a typed "
        "path points at the server, not your computer, and will never "
        "resolve."
    )
    st.info(
        "RingDatasetV2.1_submission.zip is about 2.4 GiB. Streamlit caps "
        "browser uploads at 200 MB by default, so 'Browse' is unlikely "
        "to work for the real archive on a hosted copy of this app "
        "either, regardless of that cap, loading the real archive is "
        "realistically a local-run task: clone this repository, run "
        "`streamlit run Home.py` on the same machine the archive is on, "
        "and use 'Enter a local filesystem path' there."
    )

    zip_path: Path | None = None
    path_input_given = False

    if source_mode == "upload":
        uploaded_archive = st.file_uploader(
            "RingDatasetV2.1_submission.zip",
            type="zip",
            help=(
                "Kept for this session only: it is written to a "
                "session-scoped temp file so the loader below can read "
                "it the same way it reads a local path, and it is never "
                "redistributed."
            ),
        )

        if uploaded_archive is not None:
            zip_path = _save_uploaded_archive(uploaded_archive)
    else:
        zip_path_input = st.text_input(
            "Local path to RingDatasetV2.1_submission.zip",
            value=str(DEFAULT_ZIP_PATH) if DEFAULT_ZIP_PATH.exists() else "",
            help=(
                "The full path to the archive, exactly as your file "
                "browser shows it -- for example "
                "C:\\Users\\you\\Downloads\\RingDatasetV2.1_submission.zip "
                "on Windows, or "
                "/home/you/Downloads/RingDatasetV2.1_submission.zip on "
                "macOS/Linux. Surrounding quotes are stripped "
                "automatically if a 'copy path' action added them."
            ),
        )
        cleaned_input = zip_path_input.strip().strip('"').strip("'")
        path_input_given = bool(cleaned_input)
        zip_path = Path(cleaned_input).expanduser() if cleaned_input else None

        if zip_path is not None and not zip_path.is_file():
            if zip_path.is_dir():
                st.warning(
                    f"'{zip_path}' is a folder, not the archive itself. "
                    "Point at RingDatasetV2.1_submission.zip inside it, "
                    "not the folder it is in."
                )
            elif not zip_path.exists():
                st.warning(
                    f"No file was found at '{zip_path}'. Check the path is "
                    "typed exactly as your file browser shows it (on "
                    "Windows, including the drive letter), or switch to "
                    "'Browse for the archive' above, which cannot be "
                    "mistyped."
                )
            elif zip_path.suffix.lower() != ".zip":
                st.warning(
                    f"'{zip_path}' does not end in .zip. Point at "
                    "RingDatasetV2.1_submission.zip itself, not a folder "
                    "it was extracted into."
                )
            zip_path = None

    if zip_path is None:
        if not path_input_given:
            st.info(
                "Provide the archive above to continue, either by "
                "browsing for it or by entering a local path. See "
                "'Dataset access and citation' above for where to obtain "
                "it."
            )
    else:
        try:
            available_subjects = _available_subject_ids(str(zip_path))
        except OSError as error:
            st.error(f"Could not read the archive: {error}")
            available_subjects = ()

        if not available_subjects:
            st.warning("No usable ring1 entries were found in this archive.")
        else:
            max_subjects = min(20, len(available_subjects))
            default_subjects = min(8, max_subjects)

            st.caption(
                f"{len(available_subjects)} participants have an intact "
                "ring1 entry in this archive. Each participant's file is "
                "30-160 MB uncompressed, so this loads a bounded, "
                "adjustable subset rather than all of them every run."
            )

            n_subjects = st.slider(
                "How many participants to load",
                min_value=4,
                max_value=max_subjects,
                value=default_subjects,
                help=(
                    "A small, non-random subset for responsiveness. This is "
                    "a stated limitation, not a representative sample."
                ),
            )

            if st.button("Load participants", type="primary"):
                subject_ids = available_subjects[:n_subjects]
                try:
                    raw = _load_subjects(str(zip_path), subject_ids)
                    st.session_state["healthring_windows"] = ar.prepare_windows(raw)
                    st.session_state["healthring_n_subjects"] = n_subjects
                except (OSError, KeyError, ValueError) as error:
                    st.error(f"Could not load the requested participants: {error}")
                    st.session_state["healthring_windows"] = None

            windows = st.session_state.get("healthring_windows")

            if windows is not None:
                st.success(
                    f"We found {windows.n_input_windows} measurement "
                    f"windows from "
                    f"{st.session_state.get('healthring_n_subjects')} "
                    f"participants. {windows.n_excluded_windows} were "
                    "missing information needed for this analysis, "
                    f"leaving {windows.n_usable_windows} usable windows."
                )

    if windows is not None and stage < STAGE_SIGNAL_INSPECTION:
        if st.button("Continue to signal inspection", type="primary"):
            _advance_to(STAGE_SIGNAL_INSPECTION)

# -----------------------------------------------------------------
# 2. Signal inspection
# -----------------------------------------------------------------

if stage >= STAGE_SIGNAL_INSPECTION and windows is not None:
    section_header(
        "Signal Inspection",
        "Walk through one real measurement window end to end",
    )

    st.write(
        "This stage follows one real window from activity, to movement, "
        "to the PPG signal, to the ring's own signal-quality score, to "
        "the HR estimate, to the error against reference HR. Everything "
        "shown here is real HealthRing data for the participant and "
        "window you pick, not a simulated waveform."
    )

    loaded_subject_ids = sorted(windows.data["subject_id"].unique().tolist())

    signal_subject_id = st.selectbox(
        "Which participant to inspect",
        options=loaded_subject_ids,
        key="hr_signal_subject",
    )

    try:
        signal_raw = _load_subject_signal(str(zip_path), int(signal_subject_id))
        signal_windows = ar.prepare_windows(signal_raw)
    except (OSError, KeyError, ValueError) as error:
        st.error(f"Could not load this participant's signal: {error}")
        signal_windows = None

    if signal_windows is not None:
        available_labels = signal_windows.condition_order

        if len(available_labels) >= 2:
            compare = st.checkbox(
                "Compare two contrasting windows", value=True, key="hr_signal_compare"
            )
        else:
            compare = False
            st.caption(
                "Only one activity condition is available for this "
                "participant, so a two-window comparison is not "
                "possible here."
            )

        condition_a = st.selectbox(
            "Window A: activity/condition",
            options=available_labels,
            key="hr_signal_condition_a",
        )
        row_a = signal_windows.data.loc[
            signal_windows.data["Label"] == condition_a
        ].iloc[0]

        if compare:
            other_labels = [label for label in available_labels if label != condition_a]
            condition_b = st.selectbox(
                "Window B: activity/condition",
                options=other_labels,
                key="hr_signal_condition_b",
            )
            row_b = signal_windows.data.loc[
                signal_windows.data["Label"] == condition_b
            ].iloc[0]

            col_a, col_b = st.columns(2)
            with col_a:
                _render_signal_window(row_a, "a")
            with col_b:
                _render_signal_window(row_b, "b")
        else:
            _render_signal_window(row_a, "a")

        st.caption(
            "One window is one example, not a pattern: if a higher-"
            "movement window also shows a larger error here, that is "
            "something to investigate, not something this single "
            "comparison proves."
        )

    if stage < STAGE_DESIGN_EVALUATION:
        if st.button("Continue to evaluation design", type="primary"):
            _advance_to(STAGE_DESIGN_EVALUATION)

# -----------------------------------------------------------------
# 3. Design the evaluation
# -----------------------------------------------------------------

chosen_split: ar.SplitResult | None = None
split_choice = SPLIT_PARTICIPANT
split_seed = 0
fitted: dict[str, tuple[ar.RecalibrationModel, ar.AgreementResult, pd.DataFrame]] = {}

if stage >= STAGE_DESIGN_EVALUATION and windows is not None:
    section_header(
        "Design the Evaluation",
        "How you split training and test data decides what the test result can claim",
    )

    st.markdown(
        """
To check whether a model works on someone new, some data has to be set
aside and never used while fitting the model. That set-aside portion is
called **held-out data** (also "test data"): the model never sees it
during training, so it is a fair check of whether the model
generalizes, rather than only fitting patterns specific to the
training data.

There are two common ways to choose what to hold out here:

- **Participant-level**: hold out whole participants. Every window from
  a held-out participant goes to the test set; none of their windows
  are used for training.
- **Window-level**: hold out individual windows at random, regardless
  of which participant they came from. A participant can have some
  windows in training and other windows in test at the same time.

Windows from the same participant are correlated: they share that
person's typical heart rate, skin, and how their ring happened to sit.
If a participant's windows appear on both sides, the training and test
windows are not independent of each other, so a window-level split does
not test performance on a genuinely new participant the way a
participant-level split does. This is called **leakage**.
"""
    )

    split_choice = st.radio(
        "How should training and test data be split?",
        options=[SPLIT_PARTICIPANT, SPLIT_WINDOW],
        format_func=lambda key: SPLIT_CHOICE_LABELS[key],
    )

    split_seed = st.number_input(
        "Split seed",
        min_value=0,
        value=0,
        step=1,
        help=(
            "The seed controls which participants or windows happen to "
            "land in the test set. Changing it re-runs the same split "
            "logic with a different random draw. If the result changes a "
            "lot between seeds, that is a sign this test set is small "
            "enough that chance is doing some of the work."
        ),
    )

    splits: dict[str, ar.SplitResult | None] = {}
    split_errors: dict[str, str] = {}

    try:
        splits[SPLIT_PARTICIPANT] = ar.split_by_subject(
            windows, test_fraction=0.3, seed=int(split_seed)
        )
    except ValueError as error:
        splits[SPLIT_PARTICIPANT] = None
        split_errors[SPLIT_PARTICIPANT] = str(error)

    try:
        splits[SPLIT_WINDOW] = ar.split_by_window(
            windows, test_fraction=0.3, seed=int(split_seed)
        )
    except ValueError as error:
        splits[SPLIT_WINDOW] = None
        split_errors[SPLIT_WINDOW] = str(error)

    for key, result in splits.items():
        if result is not None:
            try:
                fitted[key] = _fit_and_evaluate(result)
            except ValueError as error:
                split_errors[key] = str(error)

    chosen_split = splits.get(split_choice)

    if chosen_split is None:
        st.error(split_errors.get(split_choice, "This split could not be computed."))
    else:
        overlap = set(chosen_split.train_subjects) & set(chosen_split.test_subjects)

        st.caption(
            f"Train: {len(chosen_split.train_subjects)} participants, "
            f"{chosen_split.n_train_windows} windows. "
            f"Test: {len(chosen_split.test_subjects)} participants, "
            f"{chosen_split.n_test_windows} windows. "
            + (
                f"{len(overlap)} participant(s) appear on both sides."
                if overlap
                else "No participant appears on both sides."
            )
        )

        other_key = SPLIT_WINDOW if split_choice == SPLIT_PARTICIPANT else SPLIT_PARTICIPANT

        if split_choice in fitted and other_key in fitted:
            _, chosen_eval, _ = fitted[split_choice]
            _, other_eval, _ = fitted[other_key]

            chosen_name = SPLIT_CHOICE_LABELS[split_choice].split(" (")[0]
            other_name = SPLIT_CHOICE_LABELS[other_key].split(" (")[0]

            st.markdown(
                f"**{chosen_name} split (what you chose):** test MAE = "
                f"{chosen_eval.mae:.2f} bpm."
            )
            st.markdown(
                f"**{other_name} split (the alternative):** test MAE = "
                f"{other_eval.mae:.2f} bpm."
            )

            st.write(
                "Neither number is automatically the 'right' one. The "
                "takeaway is that the split you choose determines what "
                "the test result can legitimately claim. A "
                "participant-level test score describes performance on "
                "people the model has not seen. A window-level test score "
                "can be inflated by leakage when the same participants "
                "appear on both sides, and the gap between the two "
                "numbers above is one way to see how much that matters "
                "for this data."
            )

        caveat(
            "A window-level split lets the same participant's windows "
            "land on both sides, which is leakage (see above). That does "
            "not make its number meaningless to look at; comparing it "
            "against the participant-level number is exactly how the "
            "leakage effect becomes visible. But only the "
            "participant-level split's test score describes performance "
            "on someone new."
        )

    if stage < STAGE_BASELINE:
        if st.button(
            "Continue with this split", type="primary", disabled=chosen_split is None
        ):
            _advance_to(STAGE_BASELINE)

# -----------------------------------------------------------------
# 3. Establish baseline
# -----------------------------------------------------------------

baseline: ar.AgreementResult | None = None
model_prediction = "Not sure"

if stage >= STAGE_BASELINE and chosen_split is not None:
    section_header(
        "Establish Baseline",
        "What agreement looks like on the test set, before introducing any model",
    )

    baseline = ar.agreement_summary(
        chosen_split.test_data["bvp_hr"], chosen_split.test_data["hr"]
    )

    b1, b2, b3 = st.columns(3)
    b1.metric("Baseline MAE", f"{baseline.mae:.2f} bpm")
    b2.metric("Baseline bias", f"{baseline.bias:+.2f} bpm")
    b3.metric("Limits of agreement", f"±{1.96 * baseline.sd:.2f} bpm")

    st.caption(f"**MAE {baseline.mae:.2f} bpm:** {_mae_sentence(baseline.mae)}")
    st.caption(f"**Bias {baseline.bias:+.2f} bpm:** {_bias_sentence(baseline.bias)}")
    st.caption(
        f"**Limits of agreement ±{1.96 * baseline.sd:.2f} bpm:** "
        f"{_loa_sentence(baseline.lower_loa, baseline.upper_loa)}"
    )

    st.write(
        "This treats the ring's own `bvp_hr` estimate as the prediction, "
        "with no fitting at all, measured on the same held-out test data "
        "the model will be evaluated on next. It is the number a model "
        "has to beat, not a null result to dismiss."
    )

    model_prediction = st.radio(
        "Would you model this, or is the baseline already good enough? "
        "Do you expect a simple linear recalibration to meaningfully beat "
        "it?",
        options=["Yes, meaningfully better", "No, about the same or worse", "Not sure"],
        index=2,
    )

    if stage < STAGE_MODEL:
        if st.button("Continue to model", type="primary"):
            _advance_to(STAGE_MODEL)

# -----------------------------------------------------------------
# 4. Build model
# -----------------------------------------------------------------

model: ar.RecalibrationModel | None = None
evaluation: ar.AgreementResult | None = None
test_data: pd.DataFrame | None = None

if stage >= STAGE_MODEL and chosen_split is not None and split_choice in fitted:
    section_header(
        "Build Model",
        "One simple, interpretable model, fit on training data only",
    )

    model, evaluation, test_data = fitted[split_choice]

    m1, m2, m3 = st.columns(3)
    m1.metric("Intercept", f"{model.intercept:+.2f}")
    m2.metric("Slope", f"{model.slope:.3f}")
    m3.metric("R² (train)", f"{model.r_squared:.3f}")

    st.markdown(
        f"**In plain language:** take the ring's own estimate (`bvp_hr`), "
        f"multiply it by {model.slope:.2f}, then add {model.intercept:+.2f}. "
        "A slope near 1 and an intercept near 0 would mean the ring's raw "
        "estimate was already well-calibrated; this fit adjusts for "
        "whatever gap was found in the training data."
    )

    st.caption(
        f"**R² (coefficient of determination) {model.r_squared:.3f}:** on "
        "training data, this fraction of the variation in reference heart "
        "rate lines up with variation in the ring's estimate. It "
        "describes fit on training data only, not how well the model "
        "will do on new participants -- that is what the evaluate stage "
        "checks next."
    )

    st.caption(
        "This is a single linear bias correction, fit by ordinary least "
        f"squares (OLS) on {model.n_train} training windows: deliberately "
        "the simplest model that could work, a teaching model rather than "
        "the best possible one. It cannot correct condition-specific or "
        "nonlinear error, which the 'Does it hold across conditions?' "
        "stage checks separately."
    )

    with st.expander("What mistakes matter? Why we evaluate with MAE"):
        st.markdown(
            """
**A subtlety worth naming:** this model is trained by ordinary least
squares (OLS), which minimizes squared error during fitting. This page
then evaluates it using MAE instead. The fitting objective and the
evaluation metric are different on purpose here, and they don't have to
match: a model can be fit one way and judged another, and it is worth
knowing which is which for any model, including this one.

Model choices encode which mistakes we consider costly.

- **MAE (mean absolute error)**, used to evaluate the model throughout
  this page, treats each extra bpm of error as roughly equally bad,
  whether the total error is 1 bpm or 10 bpm.
- **MSE (mean squared error)**, closer to what OLS actually minimizes
  during fitting, squares each error before averaging, so large errors
  count disproportionately more than small ones.
- **L1** and **L2** are not error metrics; they are **regularization**,
  a penalty applied to properties of the model itself, not to
  prediction errors. L1 penalizes the magnitude of coefficients, which
  can drive some coefficients to exactly zero and produce a sparser
  model. L2 penalizes large coefficients without forcing them to zero,
  encouraging smaller, more stable ones.

None of that regularization is used here: this model has one predictor
and no penalty term, on purpose. The question worth asking of any
model, including this one, is: **which mistakes matter in this
application, and does the modeling choice reflect that?**
"""
        )

    if model_prediction == "Yes, meaningfully better":
        prediction_note = "You predicted a meaningful improvement."
    elif model_prediction == "No, about the same or worse":
        prediction_note = "You predicted little or no improvement."
    else:
        prediction_note = "You weren't sure."

    st.write(
        f"{prediction_note} The evaluate stage next shows what actually "
        "happened on held-out data."
    )

    if stage < STAGE_EVALUATE:
        if st.button("Continue to evaluation", type="primary"):
            _advance_to(STAGE_EVALUATE)

# -----------------------------------------------------------------
# 5. Evaluate
# -----------------------------------------------------------------

if stage >= STAGE_EVALUATE and evaluation is not None and baseline is not None:
    section_header(
        "Evaluate",
        "Agreement on the held-out test set, revealed one layer at a time",
    )

    e1, e2, e3 = st.columns(3)
    e1.metric(
        "Test MAE", f"{evaluation.mae:.2f} bpm",
        delta=f"{evaluation.mae - baseline.mae:+.2f} vs. baseline",
        delta_color="inverse",
    )
    e2.metric("Test bias", f"{evaluation.bias:+.2f} bpm")
    e3.metric("Limits of agreement", f"±{1.96 * evaluation.sd:.2f} bpm")

    st.caption(f"**MAE {evaluation.mae:.2f} bpm:** {_mae_sentence(evaluation.mae)}")
    st.caption(f"**Bias {evaluation.bias:+.2f} bpm:** {_bias_sentence(evaluation.bias)}")
    st.caption(
        f"**Limits of agreement ±{1.96 * evaluation.sd:.2f} bpm:** "
        f"{_loa_sentence(evaluation.lower_loa, evaluation.upper_loa)}"
    )

    if evaluation.mae >= baseline.mae:
        st.info(
            "**Adding the model did not improve performance on new "
            f"participants.** Test MAE ({evaluation.mae:.2f} bpm) was not "
            f"lower than the raw baseline ({baseline.mae:.2f} bpm) on this "
            "held-out test set."
        )
    else:
        st.write(
            f"Test MAE ({evaluation.mae:.2f} bpm) was lower than the raw "
            f"baseline ({baseline.mae:.2f} bpm) on held-out participants."
        )

    st.write(
        "That single MAE number is the one you'll carry forward. Before "
        "trusting it, look at whether it comes from a well-behaved error "
        "distribution or a skewed one:"
    )
    st.vega_lite_chart(
        _error_distribution_chart(test_data, "pred_abs_error", "Label"),
        theme=None,
        use_container_width=True,
    )

    st.write(
        "And whether the ring and reference systematically disagree, or "
        "just disagree noisily. This is a Bland-Altman plot, a standard "
        "way to compare two measurement methods: each point is one "
        "window, plotted by its average of the two readings against "
        "their difference."
    )
    st.vega_lite_chart(
        _bland_altman_chart(
            test_data["pred_mean_hr"],
            test_data["pred_diff"],
            evaluation.bias,
            evaluation.lower_loa,
            evaluation.upper_loa,
        ),
        theme=None,
        use_container_width=True,
    )
    st.caption(
        "Solid line = bias (the average difference). Dashed lines = "
        "limits of agreement (bias plus or minus 1.96 standard "
        "deviations), following Bland & Altman (1986). This is pooled "
        "across every condition, so it describes average agreement; pair "
        "it with the 'Does it hold across conditions?' stage rather than "
        "reading it alone."
    )
    st.caption(
        "Bland-Altman's limits of agreement are typically derived "
        "assuming independent measurement pairs. Here, multiple windows "
        "come from the same participant and are not independent of each "
        "other, so treat these limits as approximate rather than exact."
    )

    if stage < STAGE_RETENTION:
        if st.button("Continue to the research decision", type="primary"):
            _advance_to(STAGE_RETENTION)

# -----------------------------------------------------------------
# 6. Make a research decision (retention)
# -----------------------------------------------------------------

retention: ar.RetentionResult | None = None
quality_threshold = 0.5

if stage >= STAGE_RETENTION and evaluation is not None and test_data is not None:
    section_header(
        "Make a Research Decision",
        "Cleaner signal keeps less data: watch retention and error move together",
    )

    st.write(
        "Every window has a quality score from the ring itself, "
        "described in the 'Understand measurement' stage. Raising the "
        "minimum quality you'll accept excludes lower-quality windows "
        "entirely. This is different from the signal filtering described "
        "earlier, which reshapes a signal without deleting observations: "
        "here, an excluded window is gone from the analysis."
    )

    left, mid, right = st.columns([1, 3, 1])
    with left:
        st.caption("More data")
    with right:
        st.caption("Cleaner signal")
    with mid:
        quality_threshold = st.slider(
            "Minimum signal quality to keep a window",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            label_visibility="collapsed",
        )

    st.caption(f"At a minimum quality of {quality_threshold:.2f}:")

    try:
        retention = ar.filter_by_quality(
            test_data,
            predicted_col="predicted_hr",
            target_col="hr",
            threshold=quality_threshold,
        )

        r1, r2, r3 = st.columns(3)
        r1.metric(
            "Windows retained",
            f"{retention.n_retained_windows} / {retention.n_input_windows}",
            delta=f"{retention.retention_rate:.0%} retained",
        )
        r2.metric(
            "MAE at this threshold",
            f"{retention.agreement.mae:.2f} bpm",
            delta=f"{retention.agreement.mae - evaluation.mae:+.2f} vs. unfiltered",
            delta_color="inverse",
        )
        r3.metric("Bias at this threshold", f"{retention.agreement.bias:+.2f} bpm")

        st.caption(
            f"**MAE {retention.agreement.mae:.2f} bpm:** "
            f"{_mae_sentence(retention.agreement.mae)}"
        )
        st.caption(
            f"**Bias {retention.agreement.bias:+.2f} bpm:** "
            f"{_bias_sentence(retention.agreement.bias)}"
        )

        if retention.n_excluded_windows == 0:
            st.write(
                "At this threshold, every test window already met the "
                "bar, so this particular run does not yet show a "
                "quality-versus-retention tradeoff. Raise the threshold "
                "further to see whether one appears."
            )
        elif retention.retention_rate < 0.5:
            flagged_item_note(
                "Retention",
                f"This threshold keeps only {retention.retention_rate:.0%} "
                "of test windows. An MAE computed on a minority of the "
                "data describes that minority, not the full test set.",
            )
    except ValueError as error:
        st.warning(str(error))
        retention = None

    caveat(
        "A lower MAE after raising the threshold is not, by itself, an "
        "improvement: it can just mean the hardest windows were "
        "excluded. Read the MAE and the percent retained together, "
        "never one without the other."
    )

    if stage < STAGE_CONDITIONS_CHECK:
        if st.button("Continue to the conditions check", type="primary"):
            _advance_to(STAGE_CONDITIONS_CHECK)

# -----------------------------------------------------------------
# 7. Does it hold across conditions?
# -----------------------------------------------------------------

condition_breakdown: tuple[ar.ConditionBreakdown, ...] | None = None
REVEAL_BREAKDOWN_KEY = "hr_reveal_breakdown"

if stage >= STAGE_CONDITIONS_CHECK and evaluation is not None and test_data is not None:
    section_header(
        "Does It Hold Across Conditions?",
        "The pooled result can hide differences that only show up condition by condition",
    )

    st.write(
        f"Your overall test MAE was **{evaluation.mae:.2f} bpm**, pooled "
        "across every activity condition in the test set."
    )

    if not st.session_state.get(REVEAL_BREAKDOWN_KEY, False):
        if st.button("Reveal breakdown by condition"):
            st.session_state[REVEAL_BREAKDOWN_KEY] = True
            st.rerun()

    if st.session_state.get(REVEAL_BREAKDOWN_KEY, False):
        condition_breakdown = ar.breakdown_by_condition(
            test_data, predicted_col="predicted_hr", target_col="hr", group_col="Label"
        )

        st.vega_lite_chart(
            _condition_mae_chart(condition_breakdown, evaluation.mae, "Condition"),
            theme=None,
            use_container_width=True,
        )
        st.caption(
            f"Dashed line = pooled test MAE ({evaluation.mae:.2f} bpm). Any "
            "bar clearing the line has worse-than-aggregate error in that "
            "condition."
        )

        worst = max(condition_breakdown, key=lambda item: item.mae)
        best = min(condition_breakdown, key=lambda item: item.mae)

        if worst.group != best.group:
            st.markdown(
                f"**Overall performance hid substantial differences "
                f"between activities.** MAE was {worst.mae:.2f} bpm "
                f"during {worst.group}, versus {best.mae:.2f} bpm during "
                f"{best.group}, a spread the pooled "
                f"{evaluation.mae:.2f} bpm figure above did not show. "
                "This shows the two conditions differ in measured error; "
                "it does not, by itself, explain why."
            )

        if predicted_problems:
            st.caption(
                "You predicted possible problems from: "
                + ", ".join(predicted_problems)
                + ". This pattern is consistent with some of those "
                "hypotheses, but this analysis does not establish what "
                "caused the difference: condition, participant mix, and "
                "sample size are all confounded here."
            )

        median_quality = float(test_data["quality"].median())
        test_data["quality_bin"] = np.where(
            test_data["quality"] >= median_quality,
            "Higher-quality half",
            "Lower-quality half",
        )

        quality_breakdown = ar.breakdown_by_condition(
            test_data, predicted_col="predicted_hr", target_col="hr", group_col="quality_bin"
        )

        st.markdown(
            "**By signal-quality half.** Splitting the test set at its "
            "median quality score gives two roughly equal-sized groups: "
            "the 'higher-quality half' (the windows the ring itself "
            "scored above the median) and the 'lower-quality half' "
            "(the windows scored below it)."
        )
        st.vega_lite_chart(
            _condition_mae_chart(quality_breakdown, evaluation.mae, "Quality half"),
            theme=None,
            use_container_width=True,
        )

    if stage < STAGE_CONCLUSION:
        if st.button("Continue to your conclusion", type="primary"):
            _advance_to(STAGE_CONCLUSION)

# -----------------------------------------------------------------
# 8. Defend your conclusion
# -----------------------------------------------------------------

if stage >= STAGE_CONCLUSION and evaluation is not None and baseline is not None:
    section_header(
        "Defend Your Conclusion",
        "Based on what you observed, what would you actually trust this measurement to do?",
    )

    conclusion_choice = st.radio(
        "Based on everything above, which best describes what this run "
        "supports?",
        options=CONCLUSION_OPTIONS,
    )

    spread = None
    if condition_breakdown:
        spread = max(c.mae for c in condition_breakdown) - min(
            c.mae for c in condition_breakdown
        )

    model_delta = evaluation.mae - baseline.mae

    supports_lines = [
        f"- On this held-out test set, the model's MAE was "
        f"{evaluation.mae:.2f} bpm versus a {baseline.mae:.2f} bpm "
        f"baseline ({'an improvement' if model_delta < 0 else 'no improvement'})."
    ]

    if spread is not None:
        supports_lines.append(
            f"- Performance differed by {spread:.2f} bpm between the "
            "best- and worst-performing activity conditions in the "
            "breakdown you revealed."
        )

    if retention is not None:
        supports_lines.append(
            f"- At the quality threshold you chose, "
            f"{retention.retention_rate:.0%} of test windows were usable."
        )

    uncertain_lines = [
        "- Why performance differs across conditions: motion, "
        "participant mix, and sample size are all plausible and are not "
        "separated by this analysis.",
        "- Whether this holds for participants outside this small, "
        "non-random sample.",
        "- Whether skin tone or perfusion affects this ring's accuracy: "
        "this dataset does not record skin tone, so nothing here can "
        "answer that.",
    ]

    if spread is None:
        uncertain_lines.insert(
            0,
            "- Whether performance is consistent across conditions: the "
            "per-condition breakdown was never revealed in the previous "
            "stage, so this run has no figure for that.",
        )

    st.markdown("**What the evidence supports:**")
    st.markdown("\n".join(supports_lines))
    st.markdown("**What remains uncertain:**")
    st.markdown("\n".join(uncertain_lines))

    caveat(
        "This is a comparison, not a graded quiz: more than one of the "
        "conclusion options above can be reasonable depending on how much "
        "weight you put on the per-condition spread versus the pooled "
        "average. OpenMeasure states what the evidence shows; deciding "
        "what it is sufficient for is still a judgment call."
    )

    if stage < STAGE_FINISH:
        if st.button("Finish study", type="primary"):
            _advance_to(STAGE_FINISH)

# -----------------------------------------------------------------
# 9. Finish study
# -----------------------------------------------------------------

if stage >= STAGE_FINISH and evaluation is not None and baseline is not None:
    section_header(
        "Finish Study",
        "A compact validation record",
    )

    condition_finding = (
        "The per-condition breakdown was never revealed in the "
        "conditions-check stage, so no spread figure is recorded from "
        "this run."
    )
    if condition_breakdown:
        condition_finding = (
            f"Per-condition MAE ranged from "
            f"{min(c.mae for c in condition_breakdown):.2f} to "
            f"{max(c.mae for c in condition_breakdown):.2f} bpm, so the "
            "pooled test MAE alone would have hidden that spread."
        )

    model_delta = evaluation.mae - baseline.mae

    if retention is None:
        retention_line = (
            "the signal-quality threshold could not be applied on this "
            "run, so there is no retention figure to report"
        )
    elif retention.n_excluded_windows == 0:
        retention_line = (
            f"at a threshold of {quality_threshold:.2f}, every test "
            "window already met the bar, so this run does not "
            "demonstrate a quality-versus-retention tradeoff; a higher "
            "threshold would be needed to see one"
        )
    else:
        retention_line = (
            f"raising the signal-quality threshold to "
            f"{quality_threshold:.2f} retained {retention.retention_rate:.0%} "
            "of test windows, showing the tradeoff directly"
        )

    st.markdown(
        f"""
**Question:** Can we trust this ring's heart-rate estimate, and under
what conditions?

**Decisions:** a {SPLIT_CHOICE_LABELS[split_choice].split(" (")[0].lower()}
train/test split (seed {int(split_seed)}); a single linear recalibration
model fit on training data only; a signal-quality threshold of
{quality_threshold:.2f} for the retention check.

**Checks:** baseline agreement on held-out data; the leakage consequence
of a window-level split versus a participant-level split; agreement on
held-out data overall, after signal-quality filtering, and broken down
by activity condition and by signal-quality half.

**Findings:** baseline MAE was {baseline.mae:.2f} bpm; the recalibrated
model reached {evaluation.mae:.2f} bpm on held-out participants
({'an improvement over' if model_delta < 0 else 'no improvement over'}
baseline). {condition_finding}

**Tradeoffs:** a participant-level split avoids leakage at the cost of
fewer distinct training examples per participant; a window-level split
uses more data per participant but risks leakage. On the signal-quality
side, {retention_line}.

**Limitations:** a small, non-random subset of participants (bounded for
responsiveness, not chosen for representativeness); one ring hardware
design (`ring1`) only; one predictor (`bvp_hr`); the dataset's
`Experiment` column was not used; Bland-Altman agreement is pooled
across conditions rather than computed per condition, and its limits of
agreement assume independent measurement pairs, which repeated windows
within the same participant only approximate.

**Unresolved questions:** whether the same recalibration holds for the
archive's second ring design (`ring2`); whether the `Experiment` column
marks a distinction that matters for agreement; whether the
per-condition MAE spread reflects motion, a genuine physiological
difference across conditions, or small per-condition sample sizes;
whether skin tone or perfusion affects this ring's accuracy, which this
dataset cannot answer.

**Next checks:** repeat this walkthrough with `ring2` and compare; check
whether a model with more than one predictor closes the per-condition
gap or only improves the pooled average; validate against a larger and
more representative participant sample before drawing any conclusion
beyond this dataset.
"""
    )

    st.markdown("### WIGOR Coverage")

    st.caption(
        "WIGOR is a published reporting checklist for wearable-signal "
        "model evaluation, not an OpenMeasure invention: Puszkarski, B. "
        "(2026). Automated analysis of wearable ECG: machine learning "
        "methods, preprocessing pipelines, and benchmark datasets. "
        "Physiological Measurement, 47, 08TR02. "
        "https://doi.org/10.1088/1361-6579/ae8b71."
    )

    wigor_rows = (
        (
            "Independence",
            "Demonstrated",
            ":material/check_circle:",
            "blue",
            "The evaluation-design stage keeps test data separate from "
            "training under a participant-level split, and shows the "
            "leakage cost of the alternative.",
        ),
        (
            "Gating transparency",
            "Demonstrated",
            ":material/check_circle:",
            "blue",
            "The research-decision stage always reports how much data a "
            "signal-quality threshold kept, alongside how performance "
            "changed at that threshold.",
        ),
        (
            "Out-of-distribution validation",
            "Partially introduced",
            ":material/adjust:",
            "gray",
            "The conditions-check stage checks performance across "
            "acquisition conditions within one dataset and population: a "
            "robustness check, not a test against a genuinely external "
            "population or device.",
        ),
        (
            "Relevance",
            "Partially introduced",
            ":material/adjust:",
            "gray",
            "The defend-your-conclusion stage asks whether the result "
            "fits the intended use, but does not carry out a clinical-"
            "relevance study.",
        ),
        (
            "Wearable-deployment cost",
            "Not assessed",
            ":material/radio_button_unchecked:",
            "gray",
            "This page has no battery-life, comfort, or "
            "real-world-deployment data to draw on.",
        ),
    )

    for label, state, icon, color, note in wigor_rows:
        # Stacked, full-width rows rather than a badge/caption column
        # split: the longer badge labels and notes did not fit a narrow
        # column and were getting cut off instead of wrapping.
        with st.container(border=True):
            st.badge(f"{label}: {state}", icon=icon, color=color)
            st.caption(note)
