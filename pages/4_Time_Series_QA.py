"""
OpenMeasure - Time-Series QA module.

Checks whether the time axis and coverage of a single time series can be
trusted before it is analyzed. Core calculations live in
modules/time_series_qa/core; this file handles presentation only.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from the repository root regardless of where Streamlit
# is launched.
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from modules.time_series_qa.core import recommend as rc
from modules.time_series_qa.core.completeness import (
    DEFAULT_MAX_CONSECUTIVE_MISSING,
    DEFAULT_MAX_MISSING_PER_MONTH,
    DEFAULT_PERIOD_COVERAGE_THRESHOLD,
)
from modules.time_series_qa.core.qa import run_time_series_qa
from shared.catalog import MODULE_TIME_SERIES_QA
from shared.handoff import (
    KIND_CELLS_EMPTY,
    KIND_OBSERVATIONS_ABSENT,
    KIND_ROWS_DROPPED,
    ExclusionAccount,
    HandoffStore,
    RetentionItem,
    fingerprint_dataframe,
)
from shared.data_handling import disclosure_for, render_data_handling_summary
from shared.report import (
    Band,
    caveat,
    classify,
    flagged_item_note,
    render_lifecycle_tracker,
    render_verdict,
    section_header,
    show_case_studies,
)
from shared.upload import render_data_profile
from modules.data_profile.core.profile import ROLE_DATETIME


st.set_page_config(
    page_title="OpenMeasure · Time-Series QA",
    layout="centered",
)

ACCENT = "#2a78d6"
ACCENT_2 = "#c0392b"

st.title("Time-Series QA")
st.subheader("Data Validation")
st.caption(
    "Check whether a time series is complete and regularly sampled enough "
    "to analyze, before you analyze it."
)

st.divider()

render_lifecycle_tracker(current_workflow="Time-Series QA")

render_data_handling_summary(disclosure_for("pages/4_Time_Series_QA.py"))


# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------

with st.expander("What does time-series QA check?"):
    st.markdown(
        """
Before a trend, mean, or forecast means anything, the time axis underneath
it has to be trustworthy. This module checks two things.

**Temporal integrity** asks whether the timestamps themselves make sense:

- **Gaps** - expected observations that never arrived.
- **Duplicate timestamps** - more than one observation claiming the same moment.
- **Chronological order** - whether the rows arrived in time order.
- **Interval regularity** - whether one sampling interval dominates.

**Completeness and coverage** asks whether enough usable values are present:

- **Value completeness** - rows that exist but have no value.
- **Longest unavailable run** - the longest stretch with no usable observation.
- **Coverage per period** - how much of each month, day, or year is actually
  populated, measured against how much was expected.

### Two different ways data goes missing

These are easy to conflate and they mean different things:

| | What it looks like | Where it is reported |
|---|---|---|
| **Absent** | No row exists for an expected time | Gaps, coverage per period |
| **Empty** | A row exists but its value is blank | Value completeness |

A month where most days never arrived, and the few that did have values,
is badly covered even though nothing looks empty. Coverage is therefore
always measured against *expected* observations, never against the rows
that happen to be present.
"""
    )


with st.expander("Assumptions and limitations"):
    st.markdown(
        f"""
### Assumptions

- One row per observation, one timestamp column, one value column.
- A single series. Multiple sensors or sites should be checked separately.
- Timestamps use one timezone or offset. Mixed offsets are rejected rather
  than converted, because converting can merge two different values into
  the same instant.

### What this module will not do

- It does **not** modify your data. Nothing is filled, dropped,
  deduplicated, or corrected.
- It does **not** decide whether a finding is an error. A gap may be a
  closure, a planned outage, or simply nothing having happened.
- It does **not** detect outliers, spikes, flatlines, or level shifts.
  Those are value-quality questions and are not part of this version.
- Sentinel values such as `-999` or `NA` are counted as **present**.
  Deciding that a particular value stands for "missing" is a judgment this
  version does not make. Recode them before uploading if they should be
  treated as missing.
- Business-day and holiday calendars are not supported. If a business-day
  pattern is detected, gap detection is declined rather than reporting
  every weekend as a gap.

### About the thresholds

**Every threshold used here is an OpenMeasure project convention, not a
published standard.** No canonical cross-domain completeness standard
exists, and none of these numbers is presented as a citable one.

The coverage threshold per period is editable above
({DEFAULT_PERIOD_COVERAGE_THRESHOLD:.0%} by default), and its value is
shown alongside every result that depends on it. The run flags
({DEFAULT_MAX_CONSECUTIVE_MISSING} or more consecutive unavailable days,
or more than {DEFAULT_MAX_MISSING_PER_MONTH} in a month) are fixed in
this version; their values are shown inline whenever a flag fires. They
apply only to daily data grouped by month and are not extrapolated to
other frequencies. The interval-regularity and other verdict bands used
to classify results on this page are also fixed in this version and are
not currently exposed as adjustable numbers; the underlying
measurements they classify (such as the share of matching intervals)
are shown in full.

If a published standard applies in your field, use its numbers rather than
these defaults.

Above all: the share of data missing does not tell you *why* it is
missing, and the reason usually matters more than the rate. Values are
often missing precisely because conditions were unusual.
"""
    )


# ---------------------------------------------------------------------
# Verdict bands
# ---------------------------------------------------------------------

OCCUPANCY_BANDS = [
    Band(0.00, "Many expected observations are absent", "error"),
    Band(0.75, "Some expected observations are absent", "warning"),
    Band(0.95, "Nearly all expected observations are present", "info"),
    Band(1.00, "Every expected observation is present", "success"),
]

DISTINCT_BANDS = [
    Band(0.00, "Duplicate timestamps are present", "warning"),
    Band(1.00, "Every timestamp is unique", "success"),
]

ORDER_BANDS = [
    Band(0.00, "Rows were not in chronological order", "warning"),
    Band(1.00, "Rows were in chronological order", "success"),
]

REGULARITY_BANDS = [
    Band(0.00, "Sampling is highly irregular", "error"),
    Band(0.50, "Sampling is irregular", "warning"),
    Band(0.90, "Sampling is mostly regular", "info"),
    Band(1.00, "Sampling is perfectly regular", "success"),
]

COMPLETENESS_BANDS = [
    Band(0.00, "Many values are missing", "error"),
    Band(0.90, "Some values are missing", "warning"),
    Band(0.99, "Nearly all values are present", "info"),
    Band(1.00, "No values are missing", "success"),
]

COVERAGE_BANDS = [
    Band(0.00, "Most periods fall below the coverage threshold", "error"),
    Band(0.50, "Several periods fall below the coverage threshold", "warning"),
    Band(0.90, "Nearly all periods meet the coverage threshold", "info"),
    Band(1.00, "Every period meets the coverage threshold", "success"),
]


def render_check(
    title: str,
    value: float | None,
    bands: list[Band],
    recommendation: rc.CheckRecommendation,
    *,
    detail: str | None = None,
) -> None:
    """
    Render one check as a verdict, or explain why no verdict is offered.

    A check that cannot be defensibly interpreted still shows its raw
    numbers; only the verdict is withheld. Hiding the check entirely would
    leave the reader with neither a result nor a reason.

    value is None whenever the measure is indeterminate. It must never be
    passed to classify() in that case: classify() returns the worst band
    for NaN, because every NaN comparison is False.
    """

    st.markdown(f"**{title}**")

    if value is None or not recommendation.defensible:
        st.info(
            "No verdict is offered for this check.\n\n"
            + "\n\n".join(f"- {line}" for line in recommendation.reasoning)
        )
    else:
        render_verdict(classify(value, bands))

    if detail:
        st.caption(detail)


# ---------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------

section_header(
    "1. Upload Your Data",
    "CSV file, one row per observation, with a timestamp column and a value column",
)

uploaded = st.file_uploader(
    "CSV file",
    type="csv",
    label_visibility="collapsed",
)

SAMPLE_PATH = (
    ROOT
    / "modules"
    / "time_series_qa"
    / "sample_data"
    / "time_series_example.csv"
)

if uploaded is None:
    st.info("Upload a CSV to begin, or download the sample dataset below.")

    with st.expander("About the sample dataset"):
        st.markdown(
            """
Daily clinic visit counts from a nightly export, covering four months.

| Column | Description |
|---|---|
| `recorded_at` | Timestamp of the nightly export |
| `visits` | Number of visits recorded that day |

It deliberately contains every finding this module reports: a three-day
outage, a one-day outage, a day that was exported twice with two different
counts, three days where the export ran but recorded no count, one row out
of chronological order, and a few seconds of logging jitter on every
timestamp.
"""
        )

    if SAMPLE_PATH.exists():
        with SAMPLE_PATH.open("rb") as sample_file:
            st.download_button(
                label="Download sample time-series dataset",
                data=sample_file,
                file_name=SAMPLE_PATH.name,
                mime="text/csv",
            )
    else:
        st.warning(
            "Sample dataset could not be found. Add a CSV under "
            "modules/time_series_qa/sample_data."
        )

    show_case_studies("data_validation")
    st.stop()


try:
    frame = pd.read_csv(uploaded)
except Exception as error:
    st.error(f"The CSV could not be read: {error}")
    st.stop()

if frame.empty or len(frame.columns) < 2:
    st.error(
        "The CSV needs at least two columns and one row: a timestamp column "
        "and a value column."
    )
    st.stop()

profile = render_data_profile(frame)

# ---------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------

section_header("2. Select Columns")

columns = list(frame.columns)

# Defaults to the first datetime-like column found, if any -- a hint from
# the profile above, not a decision: every column stays selectable.
datetime_columns = profile.columns_with_role(ROLE_DATETIME)
default_timestamp_index = columns.index(datetime_columns[0]) if datetime_columns else 0

timestamp_col = st.selectbox(
    "Timestamp column",
    options=columns,
    index=default_timestamp_index,
    help=(
        "The column holding the observation time. Numeric epoch columns are "
        "rejected; convert them to datetime strings first, so the unit is "
        "never guessed."
    ),
)

value_columns = [column for column in columns if column != timestamp_col]

value_col = st.selectbox(
    "Value column",
    options=value_columns,
    help="The observed measurement.",
)

with st.expander("Options"):
    coverage_threshold = st.slider(
        "Coverage threshold per period",
        min_value=0.50,
        max_value=1.00,
        value=DEFAULT_PERIOD_COVERAGE_THRESHOLD,
        step=0.01,
        help=(
            "Share of expected observations a period must contain to count "
            "as adequately covered. An OpenMeasure convention, not a "
            "published standard."
        ),
    )

    period_choice = st.selectbox(
        "Group coverage by",
        options=["Automatic", "Day", "Month", "Year"],
        index=0,
        help=(
            "Automatic picks a period coarser than the sampling interval: "
            "days for sub-daily data, months for daily data, years for "
            "monthly data."
        ),
    )

period_freq = {
    "Automatic": None,
    "Day": "D",
    "Month": "M",
    "Year": "Y",
}[period_choice]

if not st.button("Run quality checks", type="primary"):
    show_case_studies("data_validation")
    st.stop()

try:
    result = run_time_series_qa(
        frame,
        timestamp_col,
        value_col,
        period_freq=period_freq,
        coverage_threshold=coverage_threshold,
    )
except (ValueError, TypeError) as error:
    st.error(str(error))
    show_case_studies("data_validation")
    st.stop()

temporal = result.temporal
completeness = result.completeness
frequency = result.frequency
recommendation = result.recommendation


def record_time_series(frame, upload, qa_result) -> None:
    """
    Record this analysis for the Cross-Analysis Implications page.

    Reports absent observations separately from empty values, because an
    observation that never arrived and a row that arrived without a value
    are different problems.
    """
    items = [
        RetentionItem(
            label="Rows with an unusable timestamp",
            count=(
                qa_result.n_null_timestamps
                + qa_result.n_unparseable_timestamps
            ),
            kind=KIND_ROWS_DROPPED,
            mechanism="timestamp missing or unreadable",
        ),
        RetentionItem(
            label="Empty values",
            count=qa_result.completeness.n_missing_values,
            kind=KIND_CELLS_EMPTY,
            mechanism="row present but no value recorded",
        ),
    ]

    if qa_result.temporal.n_missing_observations is not None:
        items.append(
            RetentionItem(
                label="Expected observations absent",
                count=qa_result.temporal.n_missing_observations,
                kind=KIND_OBSERVATIONS_ABSENT,
                mechanism=(
                    "no row exists for an expected time on the inferred "
                    "schedule"
                ),
            )
        )

    HandoffStore(st.session_state).record(
        module=MODULE_TIME_SERIES_QA,
        fingerprint=fingerprint_dataframe(frame, upload.name),
        exclusion=ExclusionAccount(
            module=MODULE_TIME_SERIES_QA,
            analysis_label="Time-Series QA",
            columns_considered=(
                str(qa_result.timestamp_col),
                str(qa_result.value_col),
            ),
            n_input_rows=qa_result.n_input_rows,
            n_retained_rows=qa_result.n_rows_used,
            items=tuple(items),
        ),
        primary_statistics={
            "value_completeness_ratio": float(
                qa_result.completeness.value_completeness_ratio
            ),
        },
    )


record_time_series(frame, uploaded, result)

st.caption(
    "Recorded for the Cross-Analysis Implications page, which shows how much "
    "of your data each analysis used."
)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

section_header("Dataset", "What was loaded, what was excluded, and why")

first_column, second_column, third_column = st.columns(3)
first_column.metric("Rows uploaded", result.n_input_rows)
second_column.metric("Observations used", result.n_rows_used)
third_column.metric(
    "Excluded",
    result.n_input_rows - result.n_rows_used,
)

st.caption(
    f"Span: {temporal.first_timestamp} to {temporal.last_timestamp} "
    f"({temporal.span}). "
    f"Distinct timestamps: {temporal.n_distinct_timestamps}."
)

if result.n_null_timestamps or result.n_unparseable_timestamps:
    st.caption(
        f"Excluded {result.n_null_timestamps} row(s) with an empty timestamp "
        f"and {result.n_unparseable_timestamps} row(s) whose timestamp could "
        "not be read."
    )
    if result.example_unparseable:
        examples = ", ".join(
            f"`{value}`" for value in result.example_unparseable
        )
        flagged_item_note("Unreadable timestamps", examples)

caveat(
    "Rows without a usable timestamp are excluded from every check, because "
    "they cannot be placed on the time axis. No other row is removed."
)

_ROWS_PER_TIMESTAMP_FLAG = 1.5

if temporal.n_distinct_timestamps:
    avg_rows_per_timestamp = result.n_rows_used / temporal.n_distinct_timestamps
    if avg_rows_per_timestamp > _ROWS_PER_TIMESTAMP_FLAG:
        st.warning(
            f"**{temporal.n_distinct_timestamps} distinct timestamps hold "
            f"{result.n_rows_used:,} rows, averaging "
            f"{avg_rows_per_timestamp:.0f} rows per timestamp.** Before "
            "treating this as one series sampled at the interval below, "
            "check whether another column identifies separate "
            "participants, devices, locations, or groups: if so, each "
            "group is its own series and should be checked separately, "
            "rather than as one series with many duplicate timestamps."
        )


# ---------------------------------------------------------------------
# Sampling frequency
# ---------------------------------------------------------------------

section_header(
    "Sampling Frequency",
    "Every gap and coverage figure below depends on this",
)

if frequency.offset is None:
    st.warning("No expected sampling frequency could be established.")
else:
    st.success(f"**Expected interval: {frequency.offset}**")

st.caption(frequency.reason)

if frequency.is_calendar_anchored and frequency.offset is not None:
    st.caption(
        "This is a calendar interval, so its real length varies. Gaps are "
        "measured on a calendar-aware schedule, which keeps month lengths, "
        "leap years, and daylight saving changes from creating false gaps."
    )


# ---------------------------------------------------------------------
# Temporal integrity
# ---------------------------------------------------------------------

section_header("Temporal Integrity", "Is the time axis itself trustworthy?")

if temporal.gaps_assessable:
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Expected observations", temporal.n_expected_observations)
    metric_two.metric("Absent", temporal.n_missing_observations)
    metric_three.metric("Gaps", len(temporal.gaps))

render_check(
    "Gaps",
    temporal.grid_occupancy_ratio,
    OCCUPANCY_BANDS,
    recommendation.for_check(rc.CHECK_GAPS),
    detail=(
        f"{temporal.n_missing_observations} of "
        f"{temporal.n_expected_observations} expected observations are "
        "absent."
        if temporal.gaps_assessable
        else None
    ),
)

render_check(
    "Duplicate timestamps",
    temporal.distinct_timestamp_ratio,
    DISTINCT_BANDS,
    recommendation.for_check(rc.CHECK_DUPLICATES),
    detail=(
        f"{temporal.n_duplicate_timestamp_rows} row(s) share a timestamp "
        f"with another row, across "
        f"{temporal.n_distinct_duplicated_timestamps} timestamp(s). "
        f"{temporal.n_conflicting_duplicate_timestamps} of those hold "
        "conflicting values."
        if temporal.n_duplicate_timestamp_rows
        else None
    ),
)

render_check(
    "Chronological order",
    float(not temporal.input_was_out_of_order),
    ORDER_BANDS,
    recommendation.for_check(rc.CHECK_ORDER),
    detail=(
        f"{temporal.n_out_of_order_steps} row(s) went backwards in time. "
        "The series was sorted before every other check, so this affects no "
        "other result, but it often indicates an export or join problem."
        if temporal.input_was_out_of_order
        else None
    ),
)

render_check(
    "Interval regularity",
    temporal.modal_interval_share if temporal.regularity_assessable else None,
    REGULARITY_BANDS,
    recommendation.for_check(rc.CHECK_REGULARITY),
    detail=(
        f"{temporal.modal_interval_share:.1%} of intervals equal the most "
        f"common interval ({frequency.modal_interval})."
        if temporal.regularity_assessable
        and temporal.modal_interval_share is not None
        else None
    ),
)

if temporal.n_jittered_observations:
    flagged_item_note(
        "Timing jitter",
        f"{temporal.n_jittered_observations} observation(s) did not fall "
        f"exactly on the expected schedule, by up to {temporal.max_jitter}. "
        f"Anything within {temporal.jitter_tolerance} is treated as present "
        "rather than absent.",
    )


# ---------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------

section_header(
    "Completeness and Coverage",
    "Are enough usable values actually present?",
)

completeness_one, completeness_two, completeness_three = st.columns(3)
completeness_one.metric(
    "Values present",
    f"{completeness.value_completeness_ratio:.1%}",
)
completeness_two.metric("Empty values", completeness.n_missing_values)
completeness_three.metric(
    "Longest unavailable run",
    completeness.longest_missing_run,
)

render_check(
    "Value completeness",
    completeness.value_completeness_ratio,
    COMPLETENESS_BANDS,
    recommendation.for_check(rc.CHECK_COMPLETENESS),
    detail=(
        f"{completeness.n_missing_values} of {completeness.n_rows_used} "
        "present rows have no value."
    ),
)

render_check(
    "Coverage per period",
    completeness.share_of_periods_meeting_coverage,
    COVERAGE_BANDS,
    recommendation.for_check(rc.CHECK_PERIOD_COVERAGE),
    detail=(
        f"{completeness.n_periods_assessed} complete period(s) assessed "
        f"against a {completeness.coverage_threshold:.0%} threshold."
        if completeness.n_periods_assessed
        else None
    ),
)

if completeness.longest_missing_run:
    flagged_item_note(
        "Longest unavailable run",
        f"{completeness.longest_missing_run} consecutive expected "
        f"observation(s) had no usable value, from "
        f"{completeness.longest_missing_run_start} to "
        f"{completeness.longest_missing_run_end}. This counts observations "
        "that never arrived as well as rows that arrived empty."
        if completeness.missing_run_basis == "expected_grid"
        else f"{completeness.longest_missing_run} consecutive present "
        "row(s) had no value. Observations that never arrived are not "
        "counted here, because no expected schedule was available.",
    )

if completeness.monthly_run_rule_applicable:
    if completeness.exceeds_max_consecutive_missing:
        flagged_item_note(
            "Consecutive-run flag",
            f"At least {DEFAULT_MAX_CONSECUTIVE_MISSING} consecutive days "
            "are unavailable. This is an OpenMeasure default for reviewing "
            "daily-to-monthly summaries, not a published standard, and it "
            "does not establish that the month is unusable.",
        )
    if completeness.exceeds_max_missing_per_month:
        flagged_item_note(
            "Monthly total flag",
            f"At least one month is missing more than "
            f"{DEFAULT_MAX_MISSING_PER_MONTH} days. This is an OpenMeasure "
            "default, not a published standard.",
        )


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

section_header("Diagnostics", "The detail behind the headline figures")

if temporal.gaps:
    st.markdown("**Gaps**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Last seen": gap.gap_start,
                    "Next seen": gap.gap_end,
                    "Absent observations": gap.n_expected_missing,
                    "Elapsed": str(gap.duration),
                }
                for gap in temporal.gaps
            ]
        ),
        width="stretch",
        hide_index=True,
    )

if temporal.duplicates:
    st.markdown("**Duplicate timestamps**")

    timestamp_counts = (
        pd.Series(result.prepared.timestamps)
        .value_counts()
        .rename_axis("timestamp")
        .reset_index(name="rows")
    )
    timestamp_counts["status"] = timestamp_counts["rows"].apply(
        lambda n: "Duplicate" if n > 1 else "Single"
    )
    st.vega_lite_chart(
        {
            "data": {"values": timestamp_counts.to_dict("records")},
            "mark": {"type": "bar"},
            "encoding": {
                "x": {"field": "timestamp", "type": "temporal", "title": None},
                "y": {"field": "rows", "type": "quantitative", "title": "Rows at this timestamp"},
                "color": {
                    "field": "status",
                    "type": "nominal",
                    "scale": {"domain": ["Single", "Duplicate"], "range": [ACCENT, ACCENT_2]},
                    "legend": {"title": None},
                },
                "tooltip": [
                    {"field": "timestamp", "type": "temporal", "title": "Timestamp"},
                    {"field": "rows", "type": "quantitative", "title": "Rows"},
                    {"field": "status", "type": "nominal"},
                ],
            },
            "width": "container",
            "height": 220,
        },
        use_container_width=True,
    )
    st.caption(
        "One bar per distinct timestamp present in the data. Absent "
        "expected observations (gaps) are not shown here; see Gaps above."
    )

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Timestamp": item.timestamp,
                    "Rows": item.n_rows,
                    "Distinct values": item.n_distinct_values,
                    "Conflicting": "Yes" if item.has_conflicting_values else "No",
                }
                for item in temporal.duplicates
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Duplicates are reported but never merged or removed; see Which "
        "Checks Are Defensible below for why."
    )

if completeness.per_period:
    st.markdown("**Coverage per period**")
    coverage_frame = pd.DataFrame(
        [
            {
                "Period": item.period_label,
                "Expected": item.n_expected,
                "Rows present": item.n_rows_present,
                "Values present": item.n_values_nonmissing,
                "Coverage": round(item.effective_coverage_ratio, 3),
                "Partial": "Yes" if item.is_partial_period else "No",
                "": "OK" if item.meets_threshold else "Below threshold",
            }
            for item in completeness.per_period
        ]
    )
    st.dataframe(coverage_frame, width="stretch", hide_index=True)

    st.caption(
        "Coverage is measured against expected observations, not against "
        "rows present, so observations that never arrived cannot hide. "
        "Partial periods at the start and end of the series are shown but "
        "excluded from the headline figure."
    )

    if len(coverage_frame) > 1:
        st.bar_chart(
            coverage_frame.set_index("Period")[["Coverage"]],
        )
    else:
        st.caption(
            "Only one period was assessed, so no chart is shown for it; "
            "see the table above."
        )

if result.n_rows_used:
    st.markdown("**Series**")
    series_frame = pd.DataFrame(
        {
            "timestamp": result.prepared.timestamps,
            value_col: pd.to_numeric(result.prepared.values, errors="coerce"),
        }
    ).set_index("timestamp")

    if series_frame[value_col].notna().any():
        st.line_chart(series_frame)
    else:
        st.caption("The value column is not numeric, so it is not charted.")


# ---------------------------------------------------------------------
# Which checks are defensible
# ---------------------------------------------------------------------

section_header(
    "Which Checks Are Defensible",
    "What this series can and cannot support, and why",
)

st.caption(
    "This guidance is about which checks can be interpreted. It does not "
    "decide whether any particular finding is an error, and it never "
    "proposes changing your data."
)

for item in recommendation.recommendations:
    label = "Defensible" if item.defensible else "Not defensible"
    with st.expander(f"{item.display_name} - {label}"):
        st.markdown("**Reasoning**")
        for line in item.reasoning:
            st.markdown(f"- {line}")

        st.markdown("**Assumptions**")
        for line in item.assumptions:
            st.markdown(f"- {line}")

        st.markdown("**Tradeoffs**")
        for line in item.tradeoffs:
            st.markdown(f"- {line}")

        st.markdown("**Reasonable alternatives**")
        for line in item.alternatives:
            st.markdown(f"- {line}")

        st.markdown("**Limitations**")
        for line in item.limitations:
            st.markdown(f"- {line}")


# ---------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------

section_header("What This Result Does Not Establish")

st.markdown(
    """
- It does not establish **why** an observation is absent or empty. A gap may
  reflect a closure, a planned outage, a reporting change, or nothing having
  happened.
- It does not determine whether any finding is an **error**. That judgment
  needs knowledge of how the data was collected.
- It does not say whether the **values** are correct. Outliers, stuck
  sensors, unit changes, and level shifts are not examined in this version.
- It does not reveal the **mechanism** behind missing data, which usually
  matters more than the rate. Observations are often missing precisely
  because conditions were unusual, and no completeness figure can detect
  that.
- It does not tell you whether the series is **fit for your purpose**. A
  coverage threshold is a practical convention; what counts as enough
  depends on what you intend to compute.
"""
)

caveat(
    "Interpretive thresholds used here are conventions, not published "
    "standards, and are shown alongside every result so they can be "
    "reviewed and changed."
)


# ---------------------------------------------------------------------
# Research examples
# ---------------------------------------------------------------------

show_case_studies("data_validation")
