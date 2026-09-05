"""
How a page takes in data, and what it shows about it once loaded.

Two things live here. render_data_entry() is the uploader plus the
one-click sample loader every numbered analysis page offers, and
render_data_profile() is the per-column structure summary shown
immediately after a file loads, before that page's own column-picking UI.

This is presentation only: modules/data_profile/core does the actual
profiling and flagging (pure functions, no streamlit); this renders that
result the same way on every page, the same reason shared/report.py
exists for verdicts and section headers rather than each page building
its own.

Before render_data_entry() existed, all four analysis pages offered their
bundled example as a download only, so seeing a worked result meant
downloading a CSV and uploading it back. Each page also built its own
path to that file, so nothing checked the four still existed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.data_profile.core.profile import DataProfile, profile_dataframe
from modules.data_profile.core.quality import SEVERITY_WARNING, quality_flags
from shared.catalog import (
    MODULE_FAIRNESS,
    MODULE_PROGRAM_EVALUATION,
    MODULE_RELIABILITY,
    MODULE_TIME_SERIES_QA,
)
from shared.report import flagged_item_note

ROOT = Path(__file__).resolve().parents[1]

# Every bundled example, in one place, keyed by the module constant its
# page already uses. Pages and tests read the same entry, so a renamed or
# deleted sample fails a test rather than turning into a runtime warning
# nobody sees until a reader hits the page.
SAMPLE_DATASETS: dict[str, Path] = {
    MODULE_RELIABILITY: (
        ROOT / "modules" / "reliability" / "sample_data" / "survey_example.csv"
    ),
    MODULE_PROGRAM_EVALUATION: (
        ROOT
        / "modules"
        / "program_evaluation"
        / "sample_data"
        / "program_eval_example.csv"
    ),
    MODULE_FAIRNESS: (
        ROOT / "modules" / "fairness" / "sample_data" / "fairness_example.csv"
    ),
    MODULE_TIME_SERIES_QA: (
        ROOT
        / "modules"
        / "time_series_qa"
        / "sample_data"
        / "time_series_example.csv"
    ),
}

SAMPLE_BUTTON_LABEL = "Load the sample dataset"
SAMPLE_CLEAR_LABEL = "Clear the sample dataset"

# Shown while a sample is loaded, so a reader is never unclear about
# whose numbers are on screen.
SAMPLE_ACTIVE_NOTE = (
    "Showing the bundled sample dataset. Upload a CSV above to use your own."
)

NO_FLAGS_MESSAGE = "No structural quality flags found."

ROLE_GUESS_CAVEAT = (
    '"Looks like" is a heuristic guess from each column\'s values, not a '
    "determination. Confirm it fits before relying on it."
)


@dataclass(frozen=True)
class LoadedData:
    """
    A dataset a page is working on, and where it came from.

    token is a stable identity for the loaded data, used by pages that
    reset a carried-over analysis plan when the data changes. An upload
    contributes Streamlit's own file_id; a sample contributes a constant
    derived from its path, so re-running with the sample loaded does not
    look like a new file each time.
    """

    frame: pd.DataFrame
    name: str
    token: str
    is_sample: bool


def sample_path_for(module_key: str) -> Path:
    """
    The bundled example for a module.

    Raises on an unknown key rather than returning a path that does not
    exist, so a typo fails at the call site instead of rendering a
    "sample could not be found" warning to a reader.
    """
    if module_key not in SAMPLE_DATASETS:
        raise ValueError(
            f"'{module_key}' has no bundled sample dataset. Known modules: "
            f"{', '.join(sorted(SAMPLE_DATASETS))}."
        )

    return SAMPLE_DATASETS[module_key]


def render_data_entry(
    module_key: str,
    *,
    empty_prompt: str,
    about_sample: str | None = None,
    uploader_label: str = "CSV file",
) -> LoadedData | None:
    """
    Render the uploader and the sample loader, and return whatever loaded.

    Returns None when nothing is loaded yet, which is the caller's cue to
    render whatever it wants a visitor to see before committing data (its
    teaching content, its case studies) and then stop.

    An upload always wins over a loaded sample, so a reader who uploads
    while the sample is on screen gets their own data without having to
    clear anything first.

    The sample is read from disk on each rerun rather than held in session
    state. It is a small bundled CSV, and keeping only a flag means
    nothing here retains a reader's data or grows a session over time.

    about_sample is rendered in a collapsed expander when given, which is
    where the fairness and time-series pages already described their
    example's columns.
    """
    path = sample_path_for(module_key)
    session_key = f"{module_key}_sample_loaded"

    uploaded = st.file_uploader(
        uploader_label, type="csv", label_visibility="collapsed"
    )

    if uploaded is not None:
        st.session_state[session_key] = False
        return LoadedData(
            frame=pd.read_csv(uploaded),
            name=uploaded.name,
            token=str(uploaded.file_id),
            is_sample=False,
        )

    if st.session_state.get(session_key):
        st.caption(SAMPLE_ACTIVE_NOTE)
        if st.button(SAMPLE_CLEAR_LABEL, key=f"{session_key}_clear"):
            st.session_state[session_key] = False
            st.rerun()

        return LoadedData(
            frame=pd.read_csv(path),
            name=path.name,
            token=f"sample:{path.name}",
            is_sample=True,
        )

    st.info(empty_prompt)

    if about_sample:
        with st.expander("About the sample dataset"):
            st.markdown(about_sample)

    if not path.exists():
        # A packaging problem rather than anything a reader can act on.
        # shared/tests/test_sample_datasets.py fails on this first.
        st.warning(
            f"The sample dataset could not be found at {path.name}. Upload a "
            "CSV to continue."
        )
        return None

    load_column, download_column = st.columns(2)

    with load_column:
        if st.button(
            SAMPLE_BUTTON_LABEL, type="primary", key=f"{session_key}_load"
        ):
            st.session_state[session_key] = True
            st.rerun()

    with download_column:
        with path.open("rb") as sample_file:
            st.download_button(
                f"Download {path.name}",
                data=sample_file,
                file_name=path.name,
                mime="text/csv",
            )

    return None


def render_data_profile(data: pd.DataFrame, *, expanded: bool = False) -> DataProfile:
    """
    Render a per-column structure summary and any quality flags for data,
    and return the underlying DataProfile so the calling page can use its
    role guesses (e.g. to default a timestamp selectbox to a detected
    datetime-like column) -- a hint for that default, never a filter that
    removes other columns from the choice.
    """

    profile = profile_dataframe(data)
    flags = quality_flags(data, profile)

    label = f"Data profile: {profile.n_rows:,} rows x {profile.n_columns} columns"
    if flags:
        flag_word = "flag" if len(flags) == 1 else "flags"
        label += f" ({len(flags)} quality {flag_word})"

    with st.expander(label, expanded=expanded):
        summary = pd.DataFrame(
            [
                {
                    "Column": column.name,
                    "Type": column.dtype,
                    "Missing": f"{column.n_missing:,} ({column.pct_missing:.0f}%)",
                    "Unique values": f"{column.n_unique:,}",
                    "Looks like": column.role,
                }
                for column in profile.columns
            ]
        )
        st.dataframe(summary, width="stretch", hide_index=True)
        st.caption(ROLE_GUESS_CAVEAT)

        if not flags:
            st.caption(NO_FLAGS_MESSAGE)
        else:
            st.markdown("**Quality flags**")
            for flag in flags:
                if flag.severity == SEVERITY_WARNING:
                    st.warning(
                        f"**{flag.column}**: {flag.message}"
                        if flag.column is not None
                        else flag.message
                    )
                elif flag.column is not None:
                    flagged_item_note(flag.column, flag.message)
                else:
                    st.caption(flag.message)

    return profile
