"""
Automatic data-profile rendering, shown immediately after a file loads on
every upload-taking page, before that page's own column-picking UI.

This is presentation only: modules/data_profile/core does the actual
profiling and flagging (pure functions, no streamlit); this renders that
result the same way on every page, the same reason shared/report.py
exists for verdicts and section headers rather than each page building
its own.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.data_profile.core.profile import DataProfile, profile_dataframe
from modules.data_profile.core.quality import SEVERITY_WARNING, quality_flags
from shared.report import flagged_item_note

NO_FLAGS_MESSAGE = "No structural quality flags found."

ROLE_GUESS_CAVEAT = (
    '"Looks like" is a heuristic guess from each column\'s values, not a '
    "determination. Confirm it fits before relying on it."
)


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
