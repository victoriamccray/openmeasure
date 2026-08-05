"""
Unit tests for core/temporal.py

Run with: pytest modules/time_series_qa/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from core.frequency import infer_frequency  # noqa: E402
from core.prepare import prepare_series  # noqa: E402
from core.temporal import check_temporal_integrity  # noqa: E402


def analyze(timestamps, values=None):
    if values is None:
        values = list(range(len(timestamps)))
    data = pd.DataFrame({"ts": timestamps, "val": values})
    prepared = prepare_series(data, "ts", "val")
    return check_temporal_integrity(prepared, infer_frequency(prepared.timestamps))


class TestGapDetection(unittest.TestCase):
    def test_single_two_day_gap(self):
        # Ten expected days, Jan 8 and 9 absent. Mid-week so the series is
        # not mistaken for a business-day cadence.
        result = analyze(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-10",
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
            ]
        )

        self.assertTrue(result.gaps_assessable)
        self.assertEqual(len(result.gaps), 1)
        self.assertEqual(result.gaps[0].n_expected_missing, 2)
        self.assertEqual(result.n_expected_observations, 10)
        self.assertEqual(result.n_missing_observations, 2)
        self.assertAlmostEqual(result.grid_occupancy_ratio, 0.8)

    def test_gap_is_bracketed_by_observed_timestamps(self):
        result = analyze(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-10",
                "2020-01-11",
                "2020-01-12",
            ]
        )

        gap = result.gaps[0]
        self.assertEqual(gap.gap_start, pd.Timestamp("2020-01-07"))
        self.assertEqual(gap.gap_end, pd.Timestamp("2020-01-10"))
        self.assertEqual(gap.duration, pd.Timedelta("3D"))

    def test_two_separate_gaps(self):
        result = analyze(
            [
                "2020-01-06",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
            ]
        )

        self.assertEqual(len(result.gaps), 2)
        self.assertEqual([gap.n_expected_missing for gap in result.gaps], [1, 1])

    def test_complete_series_has_no_gaps(self):
        result = analyze(pd.date_range("2020-01-01", periods=20))

        self.assertEqual(result.gaps, ())
        self.assertEqual(result.grid_occupancy_ratio, 1.0)

    def test_monthly_gap_across_leap_february(self):
        # One absent month is one grid step regardless of its day count.
        result = analyze(
            ["2020-01-15", "2020-02-15", "2020-04-15", "2020-05-15"]
        )

        self.assertEqual(len(result.gaps), 1)
        self.assertEqual(result.gaps[0].n_expected_missing, 1)
        self.assertEqual(result.n_expected_observations, 5)


class TestDaylightSaving(unittest.TestCase):
    def test_complete_series_across_spring_forward_has_no_gaps(self):
        # Regression: a daily series spans 23 absolute hours across the
        # transition, so difference-based gap detection reports a false gap
        # at every DST change. Grid differencing must not.
        result = analyze(
            pd.date_range(
                "2021-03-12", periods=6, freq="D", tz="America/New_York"
            )
        )

        self.assertEqual(result.gaps, ())
        self.assertEqual(result.grid_occupancy_ratio, 1.0)

    def test_real_gap_across_spring_forward_is_found(self):
        timestamps = list(
            pd.date_range(
                "2021-03-12", periods=6, freq="D", tz="America/New_York"
            )
        )
        removed = timestamps.pop(2)

        result = analyze(timestamps)

        self.assertEqual(len(result.gaps), 1)
        self.assertEqual(result.gaps[0].n_expected_missing, 1)
        self.assertNotIn(removed, result.gaps)

    def test_complete_series_across_fall_back_has_no_gaps(self):
        result = analyze(
            pd.date_range(
                "2021-10-31", periods=6, freq="D", tz="America/New_York"
            )
        )

        self.assertEqual(result.gaps, ())


class TestJitter(unittest.TestCase):
    def test_jittered_series_reports_only_the_real_gap(self):
        # Daily data logged a couple of seconds apart is daily data.
        result = analyze(
            [
                "2020-01-01 09:00:01",
                "2020-01-02 09:00:03",
                "2020-01-04 09:00:02",
                "2020-01-05 09:00:00",
                "2020-01-06 09:00:01",
            ]
        )

        self.assertEqual(len(result.gaps), 1)
        self.assertEqual(result.gaps[0].n_expected_missing, 1)

    def test_jitter_is_reported(self):
        result = analyze(
            [
                "2020-01-01 09:00:00",
                "2020-01-02 09:00:05",
                "2020-01-03 09:00:00",
                "2020-01-04 09:00:00",
            ]
        )

        self.assertGreater(result.n_jittered_observations, 0)
        self.assertEqual(result.max_jitter, pd.Timedelta("5s"))

    def test_exact_series_reports_no_jitter(self):
        result = analyze(pd.date_range("2020-01-01", periods=10))

        self.assertEqual(result.n_jittered_observations, 0)
        self.assertEqual(result.max_jitter, pd.Timedelta(0))


class TestDuplicates(unittest.TestCase):
    def test_conflicting_duplicate_is_identified(self):
        result = analyze(
            ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"],
            [1, 2, 99, 4],
        )

        self.assertEqual(result.n_duplicate_timestamp_rows, 1)
        self.assertEqual(result.n_distinct_duplicated_timestamps, 1)
        self.assertEqual(result.n_conflicting_duplicate_timestamps, 1)
        self.assertAlmostEqual(result.distinct_timestamp_ratio, 0.75)

    def test_exact_duplicate_is_not_conflicting(self):
        result = analyze(
            ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"],
            [1, 2, 2, 4],
        )

        self.assertEqual(result.n_duplicate_timestamp_rows, 1)
        self.assertEqual(result.n_conflicting_duplicate_timestamps, 0)

    def test_duplicates_do_not_break_grid_matching(self):
        # Regression: DatetimeIndex.get_indexer raises InvalidIndexError on
        # a duplicated index, so duplicates must be collapsed for matching.
        result = analyze(
            ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-04"],
            [1, 2, 3, 4],
        )

        self.assertTrue(result.gaps_assessable)
        self.assertEqual(len(result.gaps), 1)

    def test_no_duplicates_reports_clean(self):
        result = analyze(pd.date_range("2020-01-01", periods=5))

        self.assertEqual(result.n_duplicate_timestamp_rows, 0)
        self.assertEqual(result.duplicates, ())
        self.assertEqual(result.distinct_timestamp_ratio, 1.0)

    def test_all_timestamps_identical_reports_duplicates_not_gaps(self):
        result = analyze(["2020-01-01"] * 4, [1, 2, 3, 4])

        self.assertEqual(result.n_duplicate_timestamp_rows, 3)
        self.assertFalse(result.gaps_assessable)
        self.assertEqual(result.gaps, ())
        self.assertIsNone(result.grid_occupancy_ratio)


class TestNotAssessable(unittest.TestCase):
    def test_irregular_series_declines_gap_detection(self):
        result = analyze(
            ["2020-01-01", "2020-01-05", "2020-01-11", "2020-01-20"]
        )

        self.assertFalse(result.gaps_assessable)
        self.assertIsNone(result.grid_occupancy_ratio)
        self.assertTrue(result.reason_not_assessable)

    def test_business_day_series_declines_gap_detection(self):
        result = analyze(pd.bdate_range("2020-01-06", periods=15))

        self.assertFalse(result.gaps_assessable)
        self.assertIn("business-day", result.reason_not_assessable)

    def test_calendar_frequency_declines_regularity(self):
        result = analyze(
            pd.date_range("2020-01-15", periods=8, freq=pd.DateOffset(months=1))
        )

        self.assertFalse(result.regularity_assessable)

    def test_fixed_frequency_allows_regularity(self):
        result = analyze(pd.date_range("2020-01-01", periods=10))

        self.assertTrue(result.regularity_assessable)
        self.assertEqual(result.modal_interval_share, 1.0)


class TestCarriedThroughFacts(unittest.TestCase):
    def test_out_of_order_flag_survives_sorting(self):
        # Recomputing this on the sorted timestamps would always be False.
        result = analyze(["2020-01-03", "2020-01-01", "2020-01-02"])

        self.assertTrue(result.input_was_out_of_order)
        self.assertEqual(result.n_out_of_order_steps, 1)

    def test_exclusion_counts_are_carried_through(self):
        result = analyze(
            ["2020-01-01", None, "junk", "2020-01-02", "2020-01-03"],
            [1, 2, 3, 4, 5],
        )

        self.assertEqual(result.n_input_rows, 5)
        self.assertEqual(result.n_rows_used, 3)
        self.assertEqual(result.n_null_timestamps, 1)
        self.assertEqual(result.n_unparseable_timestamps, 1)

    def test_span_is_reported(self):
        result = analyze(pd.date_range("2020-01-01", periods=11))

        self.assertEqual(result.first_timestamp, pd.Timestamp("2020-01-01"))
        self.assertEqual(result.last_timestamp, pd.Timestamp("2020-01-11"))
        self.assertEqual(result.span, pd.Timedelta("10D"))


class TestDegenerateInput(unittest.TestCase):
    def test_single_observation_raises(self):
        data = pd.DataFrame({"ts": ["2020-01-01"], "val": [1]})
        prepared = prepare_series(data, "ts", "val")

        with self.assertRaises(ValueError) as context:
            check_temporal_integrity(
                prepared, infer_frequency(prepared.timestamps)
            )

        self.assertIn("At least 2", str(context.exception))


if __name__ == "__main__":
    unittest.main()
