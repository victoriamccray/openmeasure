"""
fMRI QC Worked Example - a guided validation journey, not a workflow.

Second entry in the "Research Journeys" nav section (see Home.py), a
prototype of the same immersive worked-example architecture as the
Wearables/HealthRing journey, applied to a different domain: does an
existing fMRI quality-control tool's own metrics line up with what
trained human raters decide, and how much do independent raters agree
with each other in the first place?

This page is deliberately smaller than the Wearables journey: it covers
Research question, Understand measurement, Signal inspection (real
subject QC data, plus an optional interrater-agreement view if you
supply rater-decision data yourself), and a bundled simulated-events
comparison. It does not yet build the later stages (a retention
decision, a stress-test-by-reason breakdown, defending a conclusion, a
closing record) that the Wearables journey has -- those are a stated
next step, not an oversight.

Data sources, none of them bundled or redistributed by this repository:

- Real subject QC output: github.com/bwilliams96/cinnqc, which hosts
  pyfMRIqc run on the real fMRI Open QC Project subjects (Williams &
  Lindner, 2023). This repository currently has no stated license, so
  nothing from it is bundled here; it is only ever read at runtime, not
  redistributed. Two ways to read it feed the same parsing/rendering
  code below: fetching the public files directly from GitHub (the
  default, and the only option that works on the hosted app, which has
  no access to a visitor's filesystem) or a local clone you provide
  (useful mainly for running OpenMeasure without a network call).
- Rater decisions (Include/Uncertain/Exclude per subject per rater):
  not available to this page at all unless you upload your own copy,
  in the wide shape core/interrater.py expects (see the Signal
  Inspection stage for the exact contract). No rater-decision data of
  any kind is bundled.
- The three simulated-event comparison images ARE bundled: they are
  pyfMRIqc's own output on pyfMRIqc's own synthetic example files, not
  real research data. See modules/reliability/sample_data/
  pyfmriqc_simulated_examples/README.md for provenance and attribution.

pyfMRIqc itself (Williams & Lindner, 2020) is not reimplemented here:
every metric shown comes from its own textfile/image output, resynthesized
for this page, the same principle the Wearables journey follows for
HealthRing's own precomputed signal columns.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import streamlit as st

from modules.reliability.core import interrater as ir
from shared.charts import multiline_time_series_chart
from shared.data_handling import disclosure_for, render_data_handling_summary
from shared.report import caveat, section_header

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

STAGE_KEY = "fqc_stage"

STAGE_RESEARCH_QUESTION = 0
STAGE_UNDERSTAND_MEASUREMENT = 1
STAGE_SIGNAL_INSPECTION = 2
STAGE_SIMULATED_COMPARISON = 3

JOURNEY_STAGES = (
    "Research question",
    "Understand measurement",
    "Signal inspection",
    "Compare simulated events",
)

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
ACCENT = "#2a78d6"
ACCENT_2 = "#eb6834"

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

SIMULATED_EXAMPLES_DIR = (
    ROOT / "modules" / "reliability" / "sample_data" / "pyfmriqc_simulated_examples"
)

# name -> (plots png, mean png, one-line description of the injected event)
SIMULATED_EXAMPLES: dict[str, tuple[str, str, str]] = {
    "Motion": (
        "motion_plots.png",
        "motion_mean.png",
        "A burst of head motion partway through the scan.",
    ),
    "Local signal loss": (
        "local_signal_loss_plots.png",
        "local_signal_loss_mean.png",
        "A localized dropout affecting a subset of slices, not the whole volume.",
    ),
    "Global intensity loss": (
        "global_intensity_loss_plots.png",
        "global_intensity_loss_mean.png",
        "A sudden drop in mean signal intensity across the whole volume.",
    ),
}

# pyfMRIqc's own stated guidance (Lindner & Williams,
# https://drmichaellindner.github.io/pyfMRIqc/): relative movement over
# 0.1mm "may be acceptable but the data should be checked thoroughly,"
# over 0.5mm "is not good," and over the acquisition voxel size is
# "unacceptable." A stated tool convention, not an OpenMeasure judgment.
MOTION_THRESHOLD_FINE_MM = 0.1
MOTION_THRESHOLD_CONCERNING_MM = 0.5


# ---------------------------------------------------------------------
# Real-subject data I/O (page-level; core/ never touches a file)
# ---------------------------------------------------------------------


def _list_real_subjects(cinnqc_root: Path) -> list[tuple[str, str, Path]]:
    """
    Every (batch, subject_id, pyfmriqc_dir) found under a local cinnqc
    clone's examples/*/derivatives/cinnqc/sub-*/pyfmriqc/ layout.
    """

    found: list[tuple[str, str, Path]] = []

    for batch_dir in sorted((cinnqc_root / "examples").glob("fmri-open-qc-*")):
        cinnqc_dir = batch_dir / "derivatives" / "cinnqc"

        if not cinnqc_dir.is_dir():
            continue

        for subject_dir in sorted(cinnqc_dir.glob("sub-*")):
            pyfmriqc_dir = subject_dir / "pyfmriqc"

            if pyfmriqc_dir.is_dir():
                found.append((batch_dir.name, subject_dir.name, pyfmriqc_dir))

    return found


# ---------------------------------------------------------------------
# Real-subject data I/O, fetched from the public cinnqc repo on GitHub
# ---------------------------------------------------------------------
#
# The hosted app has no filesystem access to a visitor's computer, so a
# local-path-only loader (the original design here, and HealthRing's
# design too) does not work for it. Unlike the HealthRing archive,
# cinnqc's files are public with no access agreement, so instead of an
# upload widget, this fetches them directly from GitHub: nothing to ask
# a visitor to provide at all. _remote_cinnqc_index/_remote_pyfmriqc_files
# mirror _list_real_subjects' layout assumptions exactly, and
# parse_qc_textfile_text below is the same parser parse_qc_textfile
# already used, split out so both a local file and downloaded text feed
# it identically.

_CINNQC_OWNER = "bwilliams96"
_CINNQC_REPO = "cinnqc"
_CINNQC_API_ROOT = f"https://api.github.com/repos/{_CINNQC_OWNER}/{_CINNQC_REPO}"
_CINNQC_RAW_ROOT = f"https://raw.githubusercontent.com/{_CINNQC_OWNER}/{_CINNQC_REPO}"

_CINNQC_SUBJECT_PATTERN = re.compile(
    r"^(examples/(fmri-open-qc-[^/]+)/derivatives/cinnqc/(sub-[^/]+)/pyfmriqc)/"
)


@dataclass(frozen=True)
class RemoteCinnqcIndex:
    """Every file path in the public cinnqc repo, as of one Git Trees call."""

    branch: str
    paths: tuple[str, ...]


def _github_get_json(url: str) -> dict:
    request = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json", "User-Agent": "OpenMeasure"}
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _github_get_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "OpenMeasure"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read()


@st.cache_data(show_spinner="Listing cinnqc's public files on GitHub...", ttl=3600)
def _remote_cinnqc_tree() -> tuple[str, tuple[str, ...]]:
    """
    List every file path in the public cinnqc repo, in one Git Trees API
    call rather than walking batch -> subject -> pyfmriqc directories one
    API call at a time: unauthenticated GitHub API requests are limited
    to 60/hour, and a directory-at-a-time walk would spend most of that
    budget just building the subject picker below.

    Returns a plain (branch, paths) tuple rather than a RemoteCinnqcIndex.
    st.cache_data pickles whatever it stores, and a page script is exec'd
    into its module rather than import-registered, so pickle's
    module-lookup step for a class defined here (like RemoteCinnqcIndex)
    fails with PicklingError. Builtins round-trip with no such lookup;
    RemoteCinnqcIndex is still built fresh from this on every call, just
    never itself passed through the cache.
    """

    repo_info = _github_get_json(_CINNQC_API_ROOT)
    branch = repo_info["default_branch"]
    tree = _github_get_json(f"{_CINNQC_API_ROOT}/git/trees/{branch}?recursive=1")

    if tree.get("truncated"):
        raise ValueError(
            "cinnqc's file tree is too large to list in a single request."
        )

    paths = tuple(item["path"] for item in tree["tree"] if item["type"] == "blob")
    return branch, paths


def _remote_cinnqc_index() -> RemoteCinnqcIndex:
    """RemoteCinnqcIndex built from the cached (branch, paths) tuple."""

    branch, paths = _remote_cinnqc_tree()
    return RemoteCinnqcIndex(branch=branch, paths=paths)


def _list_remote_subjects(paths: tuple[str, ...]) -> list[tuple[str, str, str]]:
    """
    Every (batch, subject_id, pyfmriqc_dir_path) found among cinnqc's
    remote file paths. pyfmriqc_dir_path is a repo-relative path string
    here, where _list_real_subjects returns a local Path -- the two are
    used identically below, just resolved by a different loader.
    """

    found: dict[tuple[str, str], str] = {}

    for path in paths:
        match = _CINNQC_SUBJECT_PATTERN.match(path)
        if match:
            found[(match.group(2), match.group(3))] = match.group(1)

    return sorted(
        (batch, subject, pyfmriqc_dir)
        for (batch, subject), pyfmriqc_dir in found.items()
    )


def _remote_pyfmriqc_files(pyfmriqc_dir: str, paths: tuple[str, ...]) -> list[str]:
    """Every remote file path under one subject's pyfmriqc directory."""

    prefix = pyfmriqc_dir + "/"
    return sorted(path for path in paths if path.startswith(prefix))


@st.cache_data(show_spinner="Downloading this subject's QC report from GitHub...", ttl=3600)
def _fetch_remote_textfile(branch: str, path: str) -> str:
    return _github_get_bytes(f"{_CINNQC_RAW_ROOT}/{branch}/{path}").decode(
        "utf-8", errors="replace"
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_remote_image(branch: str, path: str) -> bytes:
    return _github_get_bytes(f"{_CINNQC_RAW_ROOT}/{branch}/{path}")


@dataclass(frozen=True)
class QCTextfileSummary:
    """The numeric fields pyfMRIqc writes to its per-scan text report."""

    mean: float | None
    mean_masked: float | None
    sd: float | None
    sd_masked: float | None
    slice_snr: tuple[float, ...]
    mean_voxel_snr: float | None
    mean_abs_movement_mm: float | None
    max_abs_movement_mm: float | None
    mean_rel_movement_mm: float | None
    max_rel_movement_mm: float | None
    n_rel_movement_over_fine: int | None
    n_rel_movement_over_concerning: int | None
    n_rel_movement_over_voxelsize: int | None

    @property
    def has_movement_data(self) -> bool:
        return self.mean_abs_movement_mm is not None


def _grab(text: str, pattern: str, cast=float):
    match = re.search(pattern, text, re.MULTILINE)
    return cast(match.group(1)) if match else None


def parse_qc_textfile(path: Path) -> QCTextfileSummary:
    """Parse one local pyfMRIqc_textfile_*.txt report into structured fields."""

    return parse_qc_textfile_text(path.read_text(encoding="utf-8", errors="replace"))


def parse_qc_textfile_text(text: str) -> QCTextfileSummary:
    """
    The actual pyfMRIqc_textfile_*.txt parser, taking text rather than a
    path so a locally-read file and a downloaded-from-GitHub file are
    parsed by the exact same code.
    """

    slice_snr: tuple[float, ...] = ()
    snr_line_match = re.search(r"ALL Slice SNRs:\s*(.+)", text)

    if snr_line_match:
        bracketed = re.findall(r"\[([^\]]*)\]", snr_line_match.group(1))
        slice_snr = tuple(float(value.strip()) for value in bracketed)

    return QCTextfileSummary(
        mean=_grab(text, r"^Mean:\s*([\-\d\.]+)"),
        mean_masked=_grab(text, r"Mean \(mask\):\s*([\-\d\.]+)"),
        sd=_grab(text, r"^SD:\s*([\-\d\.]+)"),
        sd_masked=_grab(text, r"SD \(mask\):\s*([\-\d\.]+)"),
        slice_snr=slice_snr,
        mean_voxel_snr=_grab(text, r"Mean voxel SNR:\s*([\-\d\.]+)"),
        mean_abs_movement_mm=_grab(text, r"Mean absolute Movement:\s*([\-\d\.]+)"),
        max_abs_movement_mm=_grab(text, r"Max absolute Movement:\s*([\-\d\.]+)"),
        mean_rel_movement_mm=_grab(text, r"Mean relative Movement:\s*([\-\d\.]+)"),
        max_rel_movement_mm=_grab(text, r"Max relative Movement:\s*([\-\d\.]+)"),
        n_rel_movement_over_fine=_grab(
            text, r"Relative movements \(>0\.1mm\):\s*(\d+)", cast=int
        ),
        n_rel_movement_over_concerning=_grab(
            text, r"Relative movements \(>0\.5mm\):\s*(\d+)", cast=int
        ),
        n_rel_movement_over_voxelsize=_grab(
            text, r"Relative movements \(>voxelsize\):\s*(\d+)", cast=int
        ),
    )


def _read_rater_table(uploaded) -> pd.DataFrame:
    """Read an uploaded wide-format rater-decision file (CSV or XLSX)."""

    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)

    return pd.read_csv(uploaded)


# ---------------------------------------------------------------------
# Stage-gating helpers
# ---------------------------------------------------------------------


def _current_stage() -> int:
    return st.session_state.get(STAGE_KEY, STAGE_RESEARCH_QUESTION)


def _advance_to(stage: int) -> None:
    st.session_state[STAGE_KEY] = max(_current_stage(), stage)
    st.rerun()


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="OpenMeasure - fMRI QC Worked Example",
    layout="centered",
)

st.title("fMRI QC Worked Example")

render_data_handling_summary(disclosure_for("pages/FMRI_QC_Worked_Example.py"))

stage = _current_stage()

_stage_parts = [
    f"**{label}**" if index == stage else label
    for index, label in enumerate(JOURNEY_STAGES)
]

with st.container(border=True):
    st.markdown(" → ".join(_stage_parts))

if stage > STAGE_RESEARCH_QUESTION:
    if st.button("Restart study", icon=":material/restart_alt:"):
        st.session_state.pop(STAGE_KEY, None)
        st.rerun()

st.divider()

# -----------------------------------------------------------------
# 0. Research question
# -----------------------------------------------------------------

section_header("Research Question")

st.markdown(
    "### Do trained raters agree on which fMRI scans are usable, and "
    "does an existing QC tool's own metrics line up with their reasons?"
)

st.write(
    "Before analyzing fMRI data, someone has to decide whether each scan "
    "is usable. This page walks through one real study of that decision: "
    "four independent, trained raters judged the same scans, and an "
    "existing, unmodified quality-control tool (`pyfMRIqc`) computed its "
    "own metrics for the same scans, independently."
)

with st.expander("Data sources and citations"):
    st.markdown(
        """
- **`pyfMRIqc`**: Williams, B., & Lindner, M. (2020). pyfMRIqc: A
  Software Package for Raw fMRI Data Quality Assurance. *Journal of
  Open Research Software*, 8(1), 23.
  [doi.org/10.5334/jors.280](https://doi.org/10.5334/jors.280)
  ([source](https://github.com/DrMichaelLindner/pyfMRIqc), GPLv3).
- **The rating study**: Williams, B., et al. (2023). Inter-rater
  reliability of functional MRI data quality control assessments: a
  standardised protocol and practical guide using pyfMRIqc.
  [PMC9936142](https://pmc.ncbi.nlm.nih.gov/articles/PMC9936142). Data:
  University of Reading Research Data Archive,
  [doi.org/10.17864/1947.000424](https://doi.org/10.17864/1947.000424).
- **Real per-subject QC output** used in the Signal Inspection stage
  below comes from [github.com/bwilliams96/cinnqc](https://github.com/bwilliams96/cinnqc),
  which hosts `pyfMRIqc` run on the study's real subjects, identified
  only by pseudonymous IDs (e.g. `sub-013`). That repository states no
  license; nothing from it is bundled by this page. It is read directly
  from GitHub at runtime by default, or from a local clone you provide.
"""
    )

if stage < STAGE_UNDERSTAND_MEASUREMENT:
    if st.button("Begin study", type="primary"):
        _advance_to(STAGE_UNDERSTAND_MEASUREMENT)

# -----------------------------------------------------------------
# 1. Understand measurement
# -----------------------------------------------------------------

if stage >= STAGE_UNDERSTAND_MEASUREMENT:
    section_header(
        "Understand Measurement",
        "What pyfMRIqc measures, and what a QC decision means",
    )

    st.markdown(
        """
An fMRI scan is a sequence of 3D brain volumes taken over time (a
"time series" of images, one every few seconds). Several things can
make a volume, or a stretch of volumes, unusable: the participant
moving inside the scanner, a sudden signal dropout in part of the
brain, or a global intensity change across the whole volume.

**`pyfMRIqc`** checks for exactly these things and reports what it
finds. It is meant to run early, before most preprocessing; the real
subject scans this page reads had already had one step (motion
correction) applied before pyfMRIqc ran on them, so "early" here does
not mean "completely unprocessed." It does not decide
whether a scan should be used: per its own documentation, it "gives you
the information to judge for yourself." That is the same stance
OpenMeasure takes throughout this toolkit, so this page is demonstrating
a tool that already shares its design philosophy, not importing a
foreign one.

For each scan, `pyfMRIqc` reports, among other things:

- **Mean intensity and variance** over time, which can reveal signal
  loss or sudden jumps.
- **SNR (signal-to-noise ratio)**, both overall and per slice.
- **Movement**, if head-motion parameters are supplied: mean and
  maximum absolute and relative movement, and how many timepoints
  exceeded stated thresholds
  (>{fine}mm "may be acceptable but the data should be checked
  thoroughly," >{concerning}mm "is not good," and movement larger than
  the acquisition voxel size is "unacceptable" -- pyfMRIqc's own stated
  wording, not an OpenMeasure judgment).

A human rater then looks at a scan (often with tools like this) and
decides: **Include**, **Exclude**, or **Uncertain**. The rating study
this page is built on had four independent raters make exactly that
decision, and measured how much they agreed with each other.
""".format(
            fine=MOTION_THRESHOLD_FINE_MM, concerning=MOTION_THRESHOLD_CONCERNING_MM
        )
    )

    if stage < STAGE_SIGNAL_INSPECTION:
        if st.button("Continue to signal inspection", type="primary"):
            _advance_to(STAGE_SIGNAL_INSPECTION)

# -----------------------------------------------------------------
# 2. Signal inspection
# -----------------------------------------------------------------

if stage >= STAGE_SIGNAL_INSPECTION:
    section_header(
        "Signal Inspection",
        "Real pyfMRIqc output on real study subjects",
    )

    st.markdown(
        "**Load the QC data.** This journey uses publicly available "
        "`pyfMRIqc` outputs from the `cinnqc` repository. On the hosted "
        "OpenMeasure app, the required files are loaded directly from "
        "the public source. If you are running OpenMeasure locally, you "
        "may instead use a local clone."
    )

    st.caption(
        "This journey uses publicly available, derived QC outputs and "
        "pseudonymous subject identifiers (e.g. `sub-013`), fetched at "
        "runtime and never bundled or otherwise redistributed by "
        "OpenMeasure. It reads only these derived images and text "
        "reports, not raw MRI data. `cinnqc` itself states no license, "
        "so this stays read-only for the same reason: publicly "
        "accessible is not the same as freely redistributable."
    )

    source_mode = st.radio(
        "Where should this subject data come from?",
        options=["remote", "local"],
        format_func=lambda key: {
            "remote": "Fetch from the public cinnqc repository on GitHub",
            "local": "Use a local clone of cinnqc (for local OpenMeasure runs)",
        }[key],
        index=0,
        horizontal=True,
        help=(
            "cinnqc's files are public, so they can be read directly "
            "from GitHub with nothing to set up first -- this is the "
            "only option that works on the hosted app, which has no "
            "access to a visitor's filesystem. A local clone is only "
            "useful when running OpenMeasure on your own machine, "
            "typically to avoid a network call."
        ),
    )

    subjects: list[tuple[str, str, object]] = []
    remote_index: RemoteCinnqcIndex | None = None
    search_attempted = False

    if source_mode == "remote":
        try:
            remote_index = _remote_cinnqc_index()
        except (URLError, HTTPError, ValueError, KeyError) as error:
            st.error(f"Could not reach cinnqc on GitHub: {error}")
        else:
            subjects = _list_remote_subjects(remote_index.paths)
            search_attempted = True
    else:
        with st.expander("First time here? Get the data first", expanded=False):
            st.write("Run this in a terminal, once, outside this app:")
            st.code("git clone https://github.com/bwilliams96/cinnqc", language="bash")
            st.write(
                "That creates a `cinnqc` folder. Enter the path to **that "
                "folder** below (not the command above)."
            )

        cinnqc_path_input = st.text_input(
            "Local path to the cinnqc folder you cloned",
            value="",
            placeholder="C:\\Users\\yourname\\cinnqc",
            help="The folder git clone created -- not the git clone command itself.",
        )

        if not cinnqc_path_input:
            st.info("Provide a local path to a cinnqc clone to continue.")
        elif "git clone" in cinnqc_path_input or "://" in cinnqc_path_input:
            st.error(
                "This looks like the `git clone` command, not a folder path. "
                "Run that command in a terminal first, then enter the path to "
                "the folder it created (see the expander above)."
            )
        else:
            cinnqc_root = Path(cinnqc_path_input)

            if not (cinnqc_root / "examples").is_dir():
                st.error(
                    f"No 'examples' directory found under {cinnqc_root}. "
                    "Check that this path points at the root of a cinnqc clone."
                )
            else:
                subjects = _list_real_subjects(cinnqc_root)
                search_attempted = True

    if subjects:
        st.caption(f"{len(subjects)} real subject scans found.")

        labels = [f"{batch} / {subject}" for batch, subject, _ in subjects]
        selected_label = st.selectbox("Which subject to inspect", options=labels)
        selected_index = labels.index(selected_label)
        _, subject_id, pyfmriqc_dir = subjects[selected_index]

        # These two closures are the only place remote vs. local
        # actually differs from here on: everything below reads a
        # QCTextfileSummary and an image source (a local path string or
        # remote bytes, both accepted by st.image), identically either way.
        if source_mode == "remote":
            remote_files = _remote_pyfmriqc_files(pyfmriqc_dir, remote_index.paths)

            def _get_summary() -> QCTextfileSummary | None:
                matches = sorted(
                    p for p in remote_files if re.search(r"pyfMRIqc_textfile_.*\.txt$", p)
                )
                if not matches:
                    return None
                text = _fetch_remote_textfile(remote_index.branch, matches[0])
                return parse_qc_textfile_text(text)

            def _get_image(keyword: str) -> bytes | None:
                matches = sorted(
                    p for p in remote_files if re.search(rf"pyfMRIqc_{keyword}_.*\.png$", p)
                )
                return _fetch_remote_image(remote_index.branch, matches[0]) if matches else None

        else:

            def _get_summary() -> QCTextfileSummary | None:
                matches = sorted(pyfmriqc_dir.glob("pyfMRIqc_textfile_*.txt"))
                return parse_qc_textfile(matches[0]) if matches else None

            def _get_image(keyword: str) -> str | None:
                matches = sorted(pyfmriqc_dir.glob(f"pyfMRIqc_{keyword}_*.png"))
                return str(matches[0]) if matches else None

        summary = _get_summary()

        if summary is None:
            st.warning("No pyfMRIqc text report was found for this subject.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Mean voxel SNR",
                f"{summary.mean_voxel_snr:.1f}" if summary.mean_voxel_snr else "n/a",
            )
            m2.metric(
                "Mean intensity (masked)",
                f"{summary.mean_masked:.1f}" if summary.mean_masked else "n/a",
            )
            m3.metric(
                "SD (masked)",
                f"{summary.sd_masked:.1f}" if summary.sd_masked else "n/a",
            )

            if summary.slice_snr:
                st.caption("Per-slice SNR:")
                st.vega_lite_chart(
                    multiline_time_series_chart(
                        np.arange(len(summary.slice_snr)),
                        {"SNR": np.array(summary.slice_snr)},
                        {"SNR": ACCENT},
                        "SNR",
                        config=_VEGA_CHART_CONFIG,
                        x_title="Slice number",
                    ),
                    theme=None,
                    use_container_width=True,
                )
                st.caption(
                    "Some slices show `nan`: pyfMRIqc's own source "
                    "sets a slice's SNR to nan specifically when that "
                    "slice has zero voxels inside the computed brain "
                    "mask (for example, an edge slice entirely "
                    "outside it), not merely a low value."
                )

            if summary.has_movement_data:
                r1, r2, r3 = st.columns(3)
                r1.metric(
                    "Mean relative movement",
                    f"{summary.mean_rel_movement_mm:.3f} mm",
                )
                r2.metric(
                    f"Timepoints > {MOTION_THRESHOLD_FINE_MM}mm",
                    str(summary.n_rel_movement_over_fine),
                )
                r3.metric(
                    f"Timepoints > {MOTION_THRESHOLD_CONCERNING_MM}mm",
                    str(summary.n_rel_movement_over_concerning),
                )
            else:
                st.caption(
                    "No motion parameters were supplied for this scan, "
                    "so pyfMRIqc reports no movement statistics."
                )

            st.markdown("**pyfMRIqc's own generated images for this scan:**")
            st.caption(
                "These are derived QC images (intensity/mask/variance maps "
                "and summary plots), not raw anatomical images, identified "
                "only by the pseudonymous subject ID above."
            )

            image_cols = st.columns(2)
            image_labels = {
                "MEAN": "Mean intensity",
                "MASK": "Computed brain mask",
                "VARIANCE": "Variance",
                "PLOTS": "QC summary plots",
            }

            shown = 0
            for keyword, caption in image_labels.items():
                image_source = _get_image(keyword)
                if image_source is not None:
                    with image_cols[shown % 2]:
                        st.image(image_source, caption=caption)
                    shown += 1

            with st.expander(
                "Optional: cross-reference with rater decisions "
                "(bring your own file)"
            ):
                st.write(
                    "This page has no rater-decision data of its own. "
                    "If you have your own copy of rater decisions "
                    "(from the University of Reading archive, DOI "
                    "10.17864/1947.000424, or any source), upload it "
                    "here as a wide-format table: one row per subject, "
                    "one column per rater, with a category label "
                    "(e.g. Include/Uncertain/Exclude) in each cell -- "
                    "leave a cell blank if that rater did not rate "
                    "that subject."
                )

                rater_upload = st.file_uploader(
                    "Rater decisions (CSV or XLSX)",
                    type=["csv", "xlsx", "xls"],
                    key="fqc_rater_upload",
                )

                if rater_upload is not None:
                    try:
                        rater_table = _read_rater_table(rater_upload)
                    except Exception as error:
                        st.error(f"Could not read this file: {error}")
                        rater_table = None

                    if rater_table is not None:
                        id_column = rater_table.columns[0]
                        rater_columns = list(rater_table.columns[1:])

                        if len(rater_columns) < 2:
                            st.warning(
                                "At least 2 rater columns are needed "
                                "besides the subject-identifier column."
                            )
                        else:
                            decisions_only = rater_table.set_index(id_column)[
                                rater_columns
                            ]

                            st.write(
                                f"Loaded {len(decisions_only)} subjects, "
                                f"{len(rater_columns)} raters."
                            )

                            try:
                                alpha_result = ir.krippendorff_alpha(decisions_only)
                                st.metric(
                                    "Krippendorff's alpha (all raters, "
                                    "tolerates partial coverage)",
                                    f"{alpha_result.alpha:.3f}",
                                )
                            except ValueError as error:
                                st.warning(str(error))

                            complete_rows = decisions_only.dropna()

                            if len(complete_rows) >= 2 and len(rater_columns) >= 3:
                                try:
                                    kappa_result = ir.fleiss_kappa(complete_rows)
                                    st.caption(
                                        f"Fleiss' kappa on the "
                                        f"{kappa_result.n_items} subjects "
                                        "every rater rated: "
                                        f"{kappa_result.kappa:.3f}"
                                    )
                                except ValueError as error:
                                    st.caption(
                                        f"Fleiss' kappa unavailable: {error}"
                                    )

                            caveat(
                                "Krippendorff's alpha is the recommended "
                                "default here because it tolerates raters "
                                "who did not rate every subject; Fleiss' "
                                "kappa (shown for comparison) only uses "
                                "the subset every rater rated, and "
                                "assumes a fixed rater panel per item."
                            )
    elif search_attempted:
        st.warning("No subject QC output was found for this source.")

# -----------------------------------------------------------------
# 3. Compare simulated events
# -----------------------------------------------------------------

if stage >= STAGE_SIMULATED_COMPARISON:
    section_header(
        "Compare Simulated Events",
        "Three known-cause examples, kept separate from the real subject data above",
    )

    st.write(
        "These three examples are simulated, not real research data: "
        "`pyfMRIqc` was run on its own bundled synthetic example files, "
        "each engineered to show one specific problem. Real fMRI data "
        "rarely has a single, isolated, known cause this cleanly -- that "
        "is exactly why these are useful for learning what a signature "
        "looks like before looking at ambiguous real data."
    )

    guess = st.radio(
        "Before revealing them: which do you expect will show the "
        "clearest single spike in the QC summary plot?",
        options=list(SIMULATED_EXAMPLES.keys()) + ["Not sure"],
        index=len(SIMULATED_EXAMPLES),
    )

    reveal_key = "fqc_reveal_simulated"

    if not st.session_state.get(reveal_key, False):
        if st.button("Reveal all three"):
            st.session_state[reveal_key] = True
            st.rerun()

    if st.session_state.get(reveal_key, False):
        st.caption(f"You guessed: {guess}.")

        example_cols = st.columns(3)

        for column, (name, (plots_file, mean_file, description)) in zip(
            example_cols, SIMULATED_EXAMPLES.items()
        ):
            with column:
                st.markdown(f"**{name}**")
                st.caption(description)
                st.image(str(SIMULATED_EXAMPLES_DIR / plots_file))

        caveat(
            "These three signatures are clean because the cause was "
            "engineered and singular. A real scan's QC plot can show "
            "any of these patterns, several at once, or none clearly -- "
            "matching a real pattern to one of these examples is a "
            "hypothesis to check, not a diagnosis this page can make "
            "for you."
        )
