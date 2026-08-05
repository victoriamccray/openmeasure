"""
Unit tests for core/completeness.py

Run with: pytest modules/time_series_qa/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.completeness import (  # noqa: E402
    BASIS_EXPECTED_GRID,
    BASIS_PRESENT_ROWS,
    check_completeness,
)
from core.frequency import infer_frequency  # noqa: E402
from core.prepare import prepare_series  # noqa: E402


def analyze(timestamps, values=None, **kwargs):
    if values is None:
        values = list(range(len(timestamps)))
    data = pd.DataFrame({"ts": timestamps, "val": values})
    prepared = prepare_series(data, "ts", "val")
    return check_completeness(
        prepared, infer_frequency(prepared.timestamps), **kwargs
    )


class TestValueCompleteness(unittest.TestCase):
    def test_counts_missing_values(self):
        result = analyze(
            pd.date_range("2020-01-01", periods=10),
            [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
        )

        self.assertEqual(result.n_missing_values, 2)
        self.assertAlmostEqual(result.value_completeness_ratio, 0.8)

    def test_complete_series_scores_one(self):
        result = analyze(pd.date_range("2020-01-01", periods=5))

        self.assertEqual(result.n_missing_values, 0)
        self.assertEqual(result.value_completeness_ratio, 1.0)

    def test_all_values_missing_scores_zero(self):
        result = analyze(pd.date_range("2020-01-01", periods=5), [np.nan] * 5)

        self.assertEqual(result.value_completeness_ratio, 0.0)

    def test_sentinel_values_count_as_present(self):
        # Deciding that -999 means "missing" is a value-anomaly judgment,
        # which is out of scope for this version.
        result = analyze(
            pd.date_range("2020-01-01", periods=5), [-999, -999, 3, 4, 5]
        )

        self.assertEqual(result.n_missing_values, 0)
        self.assertEqual(result.value_completeness_ratio, 1.0)

    def test_empty_strings_count_as_present(self):
        result = analyze(
            pd.date_range("2020-01-01", periods=4), ["", "a", "b", "c"]
        )

        self.assertEqual(result.n_missing_values, 0)


class TestMissingRuns(unittest.TestCase):
    def test_longest_run_of_missing_values(self):
        result = analyze(
            pd.date_range("2020-01-01", periods=10),
            [1, 2, np.nan, np.nan, np.nan, 6, 7, np.nan, 9, 10],
        )

        self.assertEqual(result.longest_missing_run, 3)
        self.assertEqual(
            result.longest_missing_run_start, pd.Timestamp("2020-01-03")
        )
        self.assertEqual(
            result.longest_missing_run_end, pd.Timestamp("2020-01-05")
        )

    def test_run_counts_absent_observations_too(self):
        # An observation that never arrived is as unavailable as one that
        # arrived empty, and both matter for computing a period mean.
        result = analyze(
            ["2020-01-06", "2020-01-07", "2020-01-11", "2020-01-12"]
        )

        self.assertEqual(result.missing_run_basis, BASIS_EXPECTED_GRID)
        self.assertEqual(result.longest_missing_run, 3)

    def test_run_at_the_start_is_found(self):
        result = analyze(
            pd.date_range("2020-01-01", periods=6),
            [np.nan, np.nan, 3, 4, 5, 6],
        )

        self.assertEqual(result.longest_missing_run, 2)

    def test_run_at_the_end_is_found(self):
        result = analyze(
            pd.date_range("2020-01-01", periods=6),
            [1, 2, 3, 4, np.nan, np.nan],
        )

        self.assertEqual(result.longest_missing_run, 2)

    def test_no_missing_values_reports_zero_run(self):
        result = analyze(pd.date_range("2020-01-01", periods=5))

        self.assertEqual(result.longest_missing_run, 0)
        self.assertIsNone(result.longest_missing_run_start)


class TestPeriodCoverage(unittest.TestCase):
    def test_absent_observations_cannot_hide_behind_present_rows(self):
        # The most important test in this file. January's span is covered
        # but only 4 of its 31 days arrived, and all 4 have values.
        # Measuring coverage against rows present would call January 100%
        # complete; measuring against expected observations calls it 4/31.
        timestamps = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-31"]
        timestamps += list(
            pd.date_range("2020-02-01", "2020-02-29").astype(str)
        )

        result = analyze(timestamps)

        january = result.per_period[0]
        self.assertEqual(january.period_label, "2020-01")
        self.assertEqual(january.n_expected, 31)
        self.assertEqual(january.n_rows_present, 4)
        self.assertEqual(january.n_values_nonmissing, 4)
        self.assertAlmostEqual(january.effective_coverage_ratio, 4 / 31)
        self.assertFalse(january.meets_threshold)

    def test_full_months_are_fully_covered(self):
        result = analyze(pd.date_range("2020-01-01", periods=91))

        self.assertEqual(result.n_periods, 3)
        self.assertEqual(
            [item.n_expected for item in result.per_period], [31, 29, 31]
        )
        for item in result.per_period:
            self.assertEqual(item.effective_coverage_ratio, 1.0)

    def test_coverage_never_exceeds_one_with_duplicates(self):
        # Availability is per expected observation, not per row, so two
        # rows sharing a timestamp cannot report more values than expected.
        result = analyze(
            ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"],
            [1, 2, 99, 4],
        )

        for item in result.per_period:
            self.assertLessEqual(item.effective_coverage_ratio, 1.0)

    def test_missing_values_lower_period_coverage(self):
        values = [1] * 31
        values[0] = np.nan
        values[1] = np.nan

        result = analyze(pd.date_range("2020-01-01", periods=31), values)

        january = result.per_period[0]
        self.assertEqual(january.n_expected, 31)
        self.assertEqual(january.n_values_nonmissing, 29)
        self.assertAlmostEqual(january.effective_coverage_ratio, 29 / 31)


class TestPartialPeriods(unittest.TestCase):
    def test_whole_months_are_not_partial(self):
        # Regression: comparing the observed span to wall-clock period
        # boundaries marked the final period partial for every daily
        # series, because a day's period runs until midnight.
        result = analyze(pd.date_range("2020-01-01", periods=91))

        self.assertEqual(result.n_periods_assessed, 3)
        for item in result.per_period:
            self.assertFalse(item.is_partial_period)

    def test_first_period_is_partial_when_series_starts_mid_month(self):
        result = analyze(pd.date_range("2020-01-15", "2020-03-31"))

        self.assertTrue(result.per_period[0].is_partial_period)
        self.assertFalse(result.per_period[1].is_partial_period)
        self.assertEqual(result.n_periods_assessed, 2)

    def test_partial_periods_are_excluded_from_the_headline_share(self):
        # January is partial and poorly covered; February is complete. The
        # headline share should reflect February only.
        result = analyze(pd.date_range("2020-01-30", "2020-02-29"))

        self.assertTrue(result.per_period[0].is_partial_period)
        self.assertEqual(result.share_of_periods_meeting_coverage, 1.0)

    def test_share_is_none_when_every_period_is_partial(self):
        result = analyze(pd.date_range("2020-01-10", periods=5))

        self.assertIsNone(result.share_of_periods_meeting_coverage)
        self.assertTrue(result.reason_not_assessable)


class TestHeadlineShare(unittest.TestCase):
    def test_share_counts_periods_meeting_the_threshold(self):
        timestamps = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-31"]
        timestamps += list(
            pd.date_range("2020-02-01", "2020-02-29").astype(str)
        )

        result = analyze(timestamps)

        # January fails, February passes.
        self.assertEqual(result.n_periods_assessed, 2)
        self.assertAlmostEqual(result.share_of_periods_meeting_coverage, 0.5)

    def test_threshold_is_configurable(self):
        values = [1] * 31
        values[:5] = [np.nan] * 5

        strict = analyze(
            pd.date_range("2020-01-01", periods=31), values,
            coverage_threshold=0.95,
        )
        lenient = analyze(
            pd.date_range("2020-01-01", periods=31), values,
            coverage_threshold=0.50,
        )

        self.assertFalse(strict.per_period[0].meets_threshold)
        self.assertTrue(lenient.per_period[0].meets_threshold)

    def test_invalid_threshold_raises(self):
        with self.assertRaises(ValueError):
            analyze(pd.date_range("2020-01-01", periods=5), coverage_threshold=1.5)

        with self.assertRaises(ValueError):
            analyze(pd.date_range("2020-01-01", periods=5), coverage_threshold=-0.1)


class TestNotAssessable(unittest.TestCase):
    def test_irregular_series_reports_no_period_coverage(self):
        result = analyze(
            ["2020-01-01", "2020-01-05", "2020-01-11", "2020-01-20"]
        )

        self.assertEqual(result.per_period, ())
        self.assertIsNone(result.share_of_periods_meeting_coverage)
        self.assertEqual(result.missing_run_basis, BASIS_PRESENT_ROWS)
        self.assertTrue(result.reason_not_assessable)

    def test_value_completeness_still_reported_without_a_frequency(self):
        result = analyze(
            ["2020-01-01", "2020-01-05", "2020-01-11", "2020-01-20"],
            [1, np.nan, 3, 4],
        )

        self.assertEqual(result.n_missing_values, 1)
        self.assertAlmostEqual(result.value_completeness_ratio, 0.75)


class TestMonthlyRunRule(unittest.TestCase):
    def test_rule_applies_to_daily_data_grouped_by_month(self):
        result = analyze(pd.date_range("2020-01-01", periods=91))

        self.assertTrue(result.monthly_run_rule_applicable)
        self.assertFalse(result.exceeds_max_missing_per_month)
        self.assertFalse(result.exceeds_max_consecutive_missing)

    def test_rule_flags_a_badly_covered_month(self):
        timestamps = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-31"]
        timestamps += list(
            pd.date_range("2020-02-01", "2020-02-29").astype(str)
        )

        result = analyze(timestamps)

        self.assertTrue(result.exceeds_max_missing_per_month)
        self.assertTrue(result.exceeds_max_consecutive_missing)

    def test_rule_does_not_apply_to_monthly_data(self):
        result = analyze(
            pd.date_range("2020-01-15", periods=14, freq=pd.DateOffset(months=1))
        )

        self.assertFalse(result.monthly_run_rule_applicable)
        self.assertIsNone(result.exceeds_max_missing_per_month)


if __name__ == "__main__":
    unittest.main()
