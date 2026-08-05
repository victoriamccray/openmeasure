"""
Unit tests for core/qa.py

Run with: pytest modules/time_series_qa/tests/ -v
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core import completeness, frequency, prepare, qa, recommend, temporal  # noqa: E402
from core.qa import run_time_series_qa  # noqa: E402


class TestEndToEnd(unittest.TestCase):
    """
    One fixture with a known gap, a known conflicting duplicate, a known
    missing run, and a known out-of-order row. Every figure below is
    computed by hand from it.
    """

    def setUp(self):
        # Expected daily grid: Jan 6 to Jan 20 inclusive = 15 days.
        # Present: Jan 6,7,8,9,10 (Jan 9 value missing), then Jan 13..20.
        # Absent: Jan 11, Jan 12 -> one gap of 2.
        # Jan 8 appears twice with conflicting values.
        # Jan 7 is listed after Jan 8 to make the input out of order.
        timestamps = [
            "2020-01-06",
            "2020-01-08",
            "2020-01-07",
            "2020-01-08",
            "2020-01-09",
            "2020-01-10",
            "2020-01-13",
            "2020-01-14",
            "2020-01-15",
            "2020-01-16",
            "2020-01-17",
            "2020-01-18",
            "2020-01-19",
            "2020-01-20",
        ]
        values = [1, 2, 3, 99, np.nan, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        self.data = pd.DataFrame({"ts": timestamps, "val": values})
        self.result = run_time_series_qa(self.data, "ts", "val")

    def test_ingest_accounting(self):
        self.assertEqual(self.result.n_input_rows, 14)
        self.assertEqual(self.result.n_rows_used, 14)
        self.assertEqual(self.result.n_null_timestamps, 0)
        self.assertEqual(self.result.n_unparseable_timestamps, 0)

    def test_gap_is_found_with_the_right_size(self):
        self.assertTrue(self.result.temporal.gaps_assessable)
        self.assertEqual(len(self.result.temporal.gaps), 1)
        self.assertEqual(self.result.temporal.gaps[0].n_expected_missing, 2)

    def test_grid_occupancy_is_hand_calculable(self):
        # 13 distinct observed days out of 15 expected.
        self.assertEqual(self.result.temporal.n_expected_observations, 15)
        self.assertEqual(self.result.temporal.n_missing_observations, 2)
        self.assertAlmostEqual(
            self.result.temporal.grid_occupancy_ratio, 13 / 15
        )

    def test_duplicate_is_found_and_flagged_as_conflicting(self):
        self.assertEqual(self.result.temporal.n_duplicate_timestamp_rows, 1)
        self.assertEqual(
            self.result.temporal.n_conflicting_duplicate_timestamps, 1
        )
        self.assertAlmostEqual(
            self.result.temporal.distinct_timestamp_ratio, 13 / 14
        )

    def test_out_of_order_input_is_reported(self):
        self.assertTrue(self.result.temporal.input_was_out_of_order)

    def test_value_completeness_is_hand_calculable(self):
        # One missing value among 14 present rows.
        self.assertEqual(self.result.completeness.n_missing_values, 1)
        self.assertAlmostEqual(
            self.result.completeness.value_completeness_ratio, 13 / 14
        )

    def test_longest_missing_run_spans_the_gap_and_the_empty_value(self):
        # Jan 9 has no value, and Jan 11 and 12 never arrived. Jan 10 has a
        # value, so the longest unavailable run is Jan 11 to Jan 12.
        self.assertEqual(self.result.completeness.longest_missing_run, 2)

    def test_recommender_supports_gap_detection_for_this_series(self):
        self.assertTrue(
            self.result.recommendation.for_check(
                recommend.CHECK_GAPS
            ).defensible
        )


class TestBandedValuesAreSafeToClassify(unittest.TestCase):
    """
    shared.report.classify() returns the worst band for NaN, because every
    NaN comparison is False. Any metric intended for a verdict must
    therefore be a real number in [0, 1] or None, never NaN.
    """

    FIXTURES = [
        pd.date_range("2020-01-01", periods=40),
        ["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"],
        ["2020-01-01", "2020-01-05", "2020-01-11", "2020-01-20"],
        ["2020-01-01"] * 4,
        ["2020-01-01", "2020-01-02"],
        pd.bdate_range("2020-01-06", periods=15),
        pd.date_range("2020-01-15", periods=14, freq=pd.DateOffset(months=1)),
        pd.date_range("2021-03-12", periods=6, freq="D", tz="America/New_York"),
    ]

    def test_every_banded_metric_is_a_ratio_or_none(self):
        for timestamps in self.FIXTURES:
            data = pd.DataFrame(
                {"ts": timestamps, "val": range(len(timestamps))}
            )
            result = run_time_series_qa(data, "ts", "val")

            banded = {
                "grid_occupancy_ratio": result.temporal.grid_occupancy_ratio,
                "distinct_timestamp_ratio": (
                    result.temporal.distinct_timestamp_ratio
                ),
                "modal_interval_share": result.temporal.modal_interval_share,
                "value_completeness_ratio": (
                    result.completeness.value_completeness_ratio
                ),
                "share_of_periods_meeting_coverage": (
                    result.completeness.share_of_periods_meeting_coverage
                ),
            }

            for name, value in banded.items():
                with self.subTest(n=len(timestamps), metric=name):
                    if value is None:
                        continue
                    self.assertFalse(
                        math.isnan(value), f"{name} is NaN"
                    )
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

    def test_period_coverage_ratios_are_ratios(self):
        for timestamps in self.FIXTURES:
            data = pd.DataFrame(
                {"ts": timestamps, "val": range(len(timestamps))}
            )
            result = run_time_series_qa(data, "ts", "val")

            for item in result.completeness.per_period:
                with self.subTest(period=item.period_label):
                    self.assertFalse(
                        math.isnan(item.effective_coverage_ratio)
                    )
                    self.assertGreaterEqual(item.effective_coverage_ratio, 0.0)
                    self.assertLessEqual(item.effective_coverage_ratio, 1.0)


class TestResultObjectConventions(unittest.TestCase):
    def test_every_result_dataclass_is_frozen(self):
        modules = [
            prepare,
            frequency,
            temporal,
            completeness,
            recommend,
            qa,
        ]

        found = 0
        for module in modules:
            for name in dir(module):
                candidate = getattr(module, name)
                if not dataclasses.is_dataclass(candidate):
                    continue
                if not isinstance(candidate, type):
                    continue
                found += 1
                with self.subTest(dataclass=f"{module.__name__}.{name}"):
                    self.assertTrue(
                        candidate.__dataclass_params__.frozen,
                        f"{name} must be frozen",
                    )

        self.assertGreater(found, 5)


class TestPipelineGuards(unittest.TestCase):
    def test_single_observation_raises(self):
        data = pd.DataFrame({"ts": ["2020-01-01"], "val": [1]})

        with self.assertRaises(ValueError) as context:
            run_time_series_qa(data, "ts", "val")

        self.assertIn("At least 2", str(context.exception))

    def test_series_with_one_usable_timestamp_raises(self):
        data = pd.DataFrame(
            {"ts": ["2020-01-01", "junk", None], "val": [1, 2, 3]}
        )

        with self.assertRaises(ValueError):
            run_time_series_qa(data, "ts", "val")

    def test_non_dataframe_raises_typeerror(self):
        with self.assertRaises(TypeError):
            run_time_series_qa("not a frame", "ts", "val")

    def test_column_names_are_echoed_back(self):
        data = pd.DataFrame(
            {"when": pd.date_range("2020-01-01", periods=5), "reading": range(5)}
        )

        result = run_time_series_qa(data, "when", "reading")

        self.assertEqual(result.timestamp_col, "when")
        self.assertEqual(result.value_col, "reading")

    def test_coverage_threshold_is_passed_through(self):
        data = pd.DataFrame(
            {"ts": pd.date_range("2020-01-01", periods=60), "val": range(60)}
        )

        result = run_time_series_qa(
            data, "ts", "val", coverage_threshold=0.5
        )

        self.assertEqual(result.completeness.coverage_threshold, 0.5)


if __name__ == "__main__":
    unittest.main()
