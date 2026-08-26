"""
Recommend which quality checks are defensible for a given series.

This module answers one narrow question: given how the observations are
spaced, which checks can be interpreted, and which cannot? It deliberately
does not decide whether any particular finding is an error, and it never
proposes filling, dropping, deduplicating, or otherwise cleaning anything.
Those are judgments for the person who knows why the data looks the way it
does.

That constraint is enforced structurally rather than by convention:

- CheckRecommendation has no field capable of expressing a per-observation
  judgment or an action. There is no severity, no flagged index, no
  suggested fix. A test asserts the exact field set so the guarantee cannot
  be quietly removed.
- recommend_checks accepts only counts and a frequency estimate. It never
  receives the DataFrame or the observed values, so it is structurally
  incapable of pointing at a data point.

A check being "not defensible" does not suppress it. The check still runs
and still reports its raw counts; only the interpretive verdict is withheld,
with the reason shown in its place. Hiding the check would leave the user
with neither a result nor an explanation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .frequency import (
    SOURCE_ALL_IDENTICAL,
    SOURCE_SINGLE_INTERVAL,
    FrequencyEstimate,
)


CHECK_GAPS = "gap_detection"
CHECK_DUPLICATES = "duplicate_timestamps"
CHECK_ORDER = "chronological_order"
CHECK_REGULARITY = "interval_regularity"
CHECK_COMPLETENESS = "value_completeness"
CHECK_PERIOD_COVERAGE = "period_coverage"

ALL_CHECKS = (
    CHECK_GAPS,
    CHECK_DUPLICATES,
    CHECK_ORDER,
    CHECK_REGULARITY,
    CHECK_COMPLETENESS,
    CHECK_PERIOD_COVERAGE,
)


@dataclass(frozen=True)
class CheckRecommendation:
    """
    Whether one check can be defensibly interpreted for this series.

    The field set is deliberately limited to guidance about the check
    itself. Adding any field that identifies an observation, ranks
    severity, or proposes an action would turn this from guidance into a
    verdict about the data, which is outside this module's remit.
    """

    check: str
    display_name: str
    defensible: bool
    reasoning: tuple[str, ...]
    assumptions: tuple[str, ...]
    tradeoffs: tuple[str, ...]
    alternatives: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class QARecommendation:
    """Which checks are defensible, and the frequency reasoning behind it."""

    recommendations: tuple[CheckRecommendation, ...]
    frequency_source: str
    frequency_reason: str

    def for_check(self, check: str) -> CheckRecommendation:
        """Return the recommendation for one check."""

        for recommendation in self.recommendations:
            if recommendation.check == check:
                return recommendation

        valid = ", ".join(ALL_CHECKS)
        raise ValueError(f"Unknown check '{check}'. Valid checks: {valid}.")


_NEVER_ESTABLISHES_CAUSE = (
    "It does not establish why an observation is absent. A gap may reflect "
    "a closure, a planned outage, a reporting change, or simply that "
    "nothing happened."
)


def _gap_recommendation(frequency: FrequencyEstimate) -> CheckRecommendation:
    """Assess whether gap detection can be interpreted."""

    assumptions = (
        "Observations are expected on a repeating schedule.",
        "The expected schedule did not change part-way through the series.",
        "An absent row means an observation was expected and did not "
        "arrive, rather than that none was ever due.",
    )
    tradeoffs = (
        "Gap counts depend on the expected frequency. A different "
        "defensible frequency produces a different gap count.",
        "Observations slightly off their expected time are treated as "
        "present, so systematic drift is reported as jitter rather than as "
        "gaps.",
    )
    limitations = (
        _NEVER_ESTABLISHES_CAUSE,
        "It does not distinguish an absent observation from one that was "
        "recorded elsewhere and never loaded.",
    )

    if frequency.is_business_day:
        return CheckRecommendation(
            check=CHECK_GAPS,
            display_name="Gap detection",
            defensible=False,
            reasoning=(
                "The observations follow a business-day pattern. Checking "
                "them against a calendar-day schedule would report every "
                "weekend as a gap, and business-day calendars are outside "
                "this version's scope.",
            ),
            assumptions=assumptions,
            tradeoffs=tradeoffs,
            alternatives=(
                "Report the observed interval distribution instead.",
                "Restrict the upload to a period without holidays, or "
                "supply the series already aligned to calendar days.",
            ),
            limitations=limitations,
        )

    if frequency.source == SOURCE_SINGLE_INTERVAL:
        return CheckRecommendation(
            check=CHECK_GAPS,
            display_name="Gap detection",
            defensible=False,
            reasoning=(
                "Only two distinct timestamps are present. A single "
                "interval cannot distinguish the sampling frequency from a "
                "gap, so any gap count would be an artifact of that "
                "assumption.",
            ),
            assumptions=assumptions,
            tradeoffs=tradeoffs,
            alternatives=(
                "Upload a longer span so the sampling frequency can be "
                "established.",
            ),
            limitations=limitations,
        )

    if frequency.source == SOURCE_ALL_IDENTICAL:
        return CheckRecommendation(
            check=CHECK_GAPS,
            display_name="Gap detection",
            defensible=False,
            reasoning=(
                "Every observation shares one timestamp, so there is no "
                "time axis along which a gap could exist. The duplicate "
                "timestamp check is the relevant one here.",
            ),
            assumptions=assumptions,
            tradeoffs=tradeoffs,
            alternatives=(
                "Check whether the timestamp column was populated "
                "correctly during export.",
            ),
            limitations=limitations,
        )

    if frequency.offset is None:
        return CheckRecommendation(
            check=CHECK_GAPS,
            display_name="Gap detection",
            defensible=False,
            reasoning=(
                "No repeating interval describes this series well enough to "
                "treat as an expected schedule. This is normal for "
                "event-driven data, where an observation exists because "
                "something happened rather than because a schedule came "
                "due.",
                "For such a series the absence of an observation may itself "
                "be the finding, not a defect.",
            ),
            assumptions=assumptions,
            tradeoffs=tradeoffs,
            alternatives=(
                "Examine the distribution of intervals between "
                "observations rather than counting gaps.",
                "If a schedule genuinely applies, supply the expected "
                "frequency explicitly instead of inferring it.",
            ),
            limitations=limitations,
        )

    return CheckRecommendation(
        check=CHECK_GAPS,
        display_name="Gap detection",
        defensible=True,
        reasoning=(
            f"An expected interval of {frequency.offset} was established "
            f"({frequency.reason})",
            "Absent observations are found by comparing the observed "
            "timestamps against a calendar-aware expected schedule, so "
            "month lengths, leap years, and daylight saving changes do not "
            "create false gaps.",
        ),
        assumptions=assumptions,
        tradeoffs=tradeoffs,
        alternatives=(
            "Supply the expected frequency explicitly if the inferred one "
            "does not match how the data was collected.",
        ),
        limitations=limitations,
    )


def _regularity_recommendation(
    frequency: FrequencyEstimate,
) -> CheckRecommendation:
    """Assess whether interval regularity can be interpreted."""

    shared = dict(
        check=CHECK_REGULARITY,
        display_name="Interval regularity",
        assumptions=(
            "A regular series is expected to have one dominant interval.",
        ),
        tradeoffs=(
            "Regularity is measured on the dominant interval, so a series "
            "alternating evenly between two intervals scores low without "
            "being disordered.",
        ),
        limitations=(
            "It describes spacing only, and says nothing about whether the "
            "values themselves are usable.",
        ),
    )

    if frequency.is_calendar_anchored:
        return CheckRecommendation(
            defensible=False,
            reasoning=(
                "This series uses a calendar frequency, whose real spacing "
                "varies by design: months are 28 to 31 days long. Measuring "
                "regularity against a fixed interval would penalize "
                "correctly spaced data.",
                "Grid occupancy is the meaningful measure for calendar "
                "frequencies.",
            ),
            alternatives=(
                "Use the gap and coverage results, which are calendar-aware.",
            ),
            **shared,
        )

    if frequency.modal_interval_share is None:
        return CheckRecommendation(
            defensible=False,
            reasoning=(
                "At least two intervals are needed before regularity means "
                "anything; a single interval is trivially self-consistent.",
            ),
            alternatives=(
                "Upload a longer span.",
            ),
            **shared,
        )

    return CheckRecommendation(
        defensible=True,
        reasoning=(
            "There are enough intervals, at a fixed expected frequency, for "
            "the share matching the dominant interval to be meaningful.",
        ),
        alternatives=(
            "Inspect the interval distribution directly if the series is "
            "expected to have more than one legitimate cadence.",
        ),
        **shared,
    )


def _coverage_recommendation(
    frequency: FrequencyEstimate,
) -> CheckRecommendation:
    """Assess whether coverage per period can be interpreted."""

    gap_check = _gap_recommendation(frequency)

    shared = dict(
        check=CHECK_PERIOD_COVERAGE,
        display_name="Coverage per period",
        assumptions=(
            "The number of observations expected in a period can be derived "
            "from the expected frequency.",
            "Periods are grouped on local wall-clock calendar boundaries.",
        ),
        tradeoffs=(
            "Coverage is measured against expected observations rather than "
            "against rows present, so a period whose rows never arrived "
            "reports low coverage rather than appearing complete.",
            "Periods only partly spanned by the data are reported but "
            "excluded from the headline figure, since an incomplete first "
            "or last period is not a defect.",
        ),
        limitations=(
            "A coverage threshold is a practical convention. Whether a "
            "period is usable depends on what will be computed from it.",
            _NEVER_ESTABLISHES_CAUSE,
        ),
    )

    if not gap_check.defensible:
        return CheckRecommendation(
            defensible=False,
            reasoning=(
                "Coverage per period counts observed against expected "
                "observations, so it needs the same expected schedule that "
                "gap detection needs, and that could not be established "
                "here.",
            ),
            alternatives=(
                "Report the raw number of observations per period, without "
                "expressing it as a share of an expected total.",
            ),
            **shared,
        )

    return CheckRecommendation(
        defensible=True,
        reasoning=(
            "An expected schedule was established, so the number of "
            "observations expected within each period is known.",
        ),
        alternatives=(
            "Adjust the coverage threshold to match the requirement of the "
            "analysis the data will feed.",
        ),
        **shared,
    )


def recommend_checks(
    *,
    n_rows_used: int,
    n_distinct_timestamps: int,
    frequency: FrequencyEstimate,
) -> QARecommendation:
    """
    Recommend which checks are defensible for a series.

    Accepts counts and a frequency estimate only. The observed values are
    deliberately not a parameter, so this function cannot identify or judge
    an individual observation.

    Parameters
    ----------
    n_rows_used:
        Observations with a usable timestamp.
    n_distinct_timestamps:
        Distinct timestamps among them.
    frequency:
        Output of core.frequency.infer_frequency.

    Returns
    -------
    QARecommendation

    Raises
    ------
    ValueError
        If the counts are not positive, or more distinct timestamps than
        rows are reported.
    """

    if n_rows_used < 1:
        raise ValueError(f"n_rows_used must be at least 1; got {n_rows_used}.")

    if n_distinct_timestamps < 1:
        raise ValueError(
            f"n_distinct_timestamps must be at least 1; got "
            f"{n_distinct_timestamps}."
        )

    if n_distinct_timestamps > n_rows_used:
        raise ValueError(
            f"n_distinct_timestamps ({n_distinct_timestamps}) cannot exceed "
            f"n_rows_used ({n_rows_used})."
        )

    has_duplicates = n_distinct_timestamps < n_rows_used

    duplicates = CheckRecommendation(
        check=CHECK_DUPLICATES,
        display_name="Duplicate timestamps",
        defensible=True,
        reasoning=(
            "Whether two observations share a timestamp is a structural "
            "property of the series and can always be determined.",
            (
                "Duplicated timestamps are present in this series."
                if has_duplicates
                else "No timestamp is repeated in this series."
            ),
        ),
        assumptions=(
            "Each timestamp is intended to identify one observation.",
        ),
        tradeoffs=(
            "Duplicates are reported but never merged or removed, because "
            "choosing which of two conflicting values is correct requires "
            "knowledge this tool does not have.",
        ),
        alternatives=(
            "If the series legitimately holds several measurements per "
            "timestamp, treat it as multiple series and check each "
            "separately.",
        ),
        limitations=(
            "It does not determine which duplicated value is correct.",
            "A repeated timestamp can be legitimate, such as a duplicated "
            "local hour when clocks are set back.",
        ),
    )

    order = CheckRecommendation(
        check=CHECK_ORDER,
        display_name="Chronological order",
        defensible=True,
        reasoning=(
            "Whether the rows arrived in time order is a structural "
            "property of the file and can always be determined.",
        ),
        assumptions=(
            "Row order in the file was intended to be chronological.",
        ),
        tradeoffs=(
            "The series is sorted before every other check, so ordering "
            "affects no other result. It is reported because it often "
            "indicates an export or join problem upstream.",
        ),
        alternatives=(
            "Ignore this check if the source system is known not to "
            "guarantee row order.",
        ),
        limitations=(
            "Out-of-order rows are not themselves an error; many exports "
            "are unordered by design.",
        ),
    )

    completeness = CheckRecommendation(
        check=CHECK_COMPLETENESS,
        display_name="Value completeness",
        defensible=True,
        reasoning=(
            "Counting how many present rows have an empty value requires no "
            "assumption about the sampling schedule.",
        ),
        assumptions=(
            "An empty cell means the value is unknown.",
            "Values such as -999 or NA are counted as present, because "
            "deciding that a particular value stands for missing is a "
            "separate judgment.",
        ),
        tradeoffs=(
            "This counts only rows that exist. Observations that never "
            "arrived at all are counted by the gap and coverage checks, so "
            "the two figures should be read together.",
        ),
        alternatives=(
            "Recode sentinel values as empty before uploading if they "
            "should be treated as missing.",
        ),
        limitations=(
            "The share of missing values does not reveal why they are "
            "missing, and the reason usually matters more than the rate. "
            "Values may be missing precisely because conditions were "
            "unusual, which no completeness figure can detect.",
        ),
    )

    return QARecommendation(
        recommendations=(
            _gap_recommendation(frequency),
            duplicates,
            order,
            _regularity_recommendation(frequency),
            completeness,
            _coverage_recommendation(frequency),
        ),
        frequency_source=frequency.source,
        frequency_reason=frequency.reason,
    )
