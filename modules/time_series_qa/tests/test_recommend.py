"""
Unit tests for core/recommend.py

Run with: pytest modules/time_series_qa/tests/ -v
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from core import recommend as rc  # noqa: E402
from core.frequency import infer_frequency  # noqa: E402


def recommend_for(timestamps, n_rows_used=None):
    index = pd.DatetimeIndex(pd.to_datetime(pd.Series(timestamps)))
    n_distinct = len(index.unique())
    return rc.recommend_checks(
        n_rows_used=n_rows_used or len(index),
        n_distinct_timestamps=n_distinct,
        frequency=infer_frequency(index),
    )


class TestStructuralConstraints(unittest.TestCase):
    """
    The recommender must be incapable of judging an individual observation
    or proposing a fix. These tests encode that as a contract rather than
    trusting future contributors to remember it.
    """

    def test_recommendation_has_exactly_the_permitted_fields(self):
        expected = {
            "check",
            "display_name",
            "defensible",
            "reasoning",
            "assumptions",
            "tradeoffs",
            "alternatives",
            "limitations",
        }
        actual = {
            field.name
            for field in dataclasses.fields(rc.CheckRecommendation)
        }

        self.assertEqual(actual, expected)

    def test_recommendation_cannot_express_a_judgment_or_an_action(self):
        forbidden = {
            "flagged_indices",
            "flagged_timestamps",
            "severity",
            "suggested_action",
            "fill_method",
            "imputation",
            "is_error",
            "values",
            "data",
        }
        actual = {
            field.name
            for field in dataclasses.fields(rc.CheckRecommendation)
        }

        self.assertEqual(actual & forbidden, set())

    def test_recommender_never_receives_the_observed_values(self):
        # Passing only counts makes it structurally impossible for this
        # function to point at a data point.
        parameters = set(
            inspect.signature(rc.recommend_checks).parameters
        )

        self.assertEqual(
            parameters,
            {"n_rows_used", "n_distinct_timestamps", "frequency"},
        )

    def test_recommendations_are_deterministic(self):
        first = recommend_for(pd.date_range("2020-01-01", periods=10))
        second = recommend_for(pd.date_range("2020-01-01", periods=10))

        self.assertEqual(first, second)


class TestCoverageOfChecks(unittest.TestCase):
    def test_every_check_is_covered(self):
        result = recommend_for(pd.date_range("2020-01-01", periods=10))

        covered = {item.check for item in result.recommendations}
        self.assertEqual(covered, set(rc.ALL_CHECKS))

    def test_every_recommendation_is_fully_documented(self):
        result = recommend_for(pd.date_range("2020-01-01", periods=10))

        for item in result.recommendations:
            with self.subTest(check=item.check):
                self.assertTrue(item.display_name)
                self.assertTrue(item.reasoning)
                self.assertTrue(item.assumptions)
                self.assertTrue(item.tradeoffs)
                self.assertTrue(item.alternatives)
                self.assertTrue(item.limitations)

    def test_unknown_check_raises_and_lists_valid_checks(self):
        result = recommend_for(pd.date_range("2020-01-01", periods=5))

        with self.assertRaises(ValueError) as context:
            result.for_check("not_a_real_check")

        for check in rc.ALL_CHECKS:
            self.assertIn(check, str(context.exception))


class TestDefensibility(unittest.TestCase):
    def test_regular_daily_supports_every_check(self):
        result = recommend_for(pd.date_range("2020-01-01", periods=30))

        for check in rc.ALL_CHECKS:
            with self.subTest(check=check):
                self.assertTrue(result.for_check(check).defensible)

    def test_irregular_series_declines_gaps_and_coverage(self):
        result = recommend_for(
            ["2020-01-01", "2020-01-05", "2020-01-11", "2020-01-20"]
        )

        self.assertFalse(result.for_check(rc.CHECK_GAPS).defensible)
        self.assertFalse(result.for_check(rc.CHECK_PERIOD_COVERAGE).defensible)

    def test_irregular_series_offers_an_interval_alternative(self):
        result = recommend_for(
            ["2020-01-01", "2020-01-05", "2020-01-11", "2020-01-20"]
        )

        alternatives = " ".join(
            result.for_check(rc.CHECK_GAPS).alternatives
        ).lower()
        self.assertIn("interval", alternatives)

    def test_irregular_series_notes_absence_may_be_data(self):
        # For an event-driven process, no observation can be the finding
        # rather than a defect. Saying so is the point of the recommender.
        result = recommend_for(
            ["2020-01-01", "2020-01-05", "2020-01-11", "2020-01-20"]
        )

        reasoning = " ".join(
            result.for_check(rc.CHECK_GAPS).reasoning
        ).lower()
        self.assertIn("event-driven", reasoning)

    def test_two_observations_decline_gap_detection(self):
        result = recommend_for(["2020-01-01", "2020-01-02"])

        self.assertFalse(result.for_check(rc.CHECK_GAPS).defensible)

    def test_identical_timestamps_decline_gaps_but_allow_duplicates(self):
        result = recommend_for(["2020-01-01"] * 4)

        self.assertFalse(result.for_check(rc.CHECK_GAPS).defensible)
        self.assertTrue(result.for_check(rc.CHECK_DUPLICATES).defensible)

    def test_business_day_series_declines_gaps_with_weekend_reasoning(self):
        result = recommend_for(pd.bdate_range("2020-01-06", periods=15))

        gaps = result.for_check(rc.CHECK_GAPS)
        self.assertFalse(gaps.defensible)
        self.assertIn("weekend", " ".join(gaps.reasoning).lower())

    def test_calendar_frequency_declines_interval_regularity(self):
        result = recommend_for(
            pd.date_range("2020-01-15", periods=10, freq=pd.DateOffset(months=1))
        )

        self.assertFalse(result.for_check(rc.CHECK_REGULARITY).defensible)
        self.assertTrue(result.for_check(rc.CHECK_GAPS).defensible)

    def test_structural_checks_are_always_defensible(self):
        for timestamps in [
            ["2020-01-01"] * 3,
            ["2020-01-01", "2020-01-05", "2020-01-11", "2020-01-20"],
            pd.bdate_range("2020-01-06", periods=10),
            pd.date_range("2020-01-01", periods=10),
        ]:
            result = recommend_for(timestamps)
            for check in [
                rc.CHECK_DUPLICATES,
                rc.CHECK_ORDER,
                rc.CHECK_COMPLETENESS,
            ]:
                with self.subTest(check=check, n=len(timestamps)):
                    self.assertTrue(result.for_check(check).defensible)

    def test_duplicate_reasoning_reflects_whether_duplicates_exist(self):
        with_duplicates = rc.recommend_checks(
            n_rows_used=5,
            n_distinct_timestamps=4,
            frequency=infer_frequency(pd.date_range("2020-01-01", periods=4)),
        )
        without = rc.recommend_checks(
            n_rows_used=4,
            n_distinct_timestamps=4,
            frequency=infer_frequency(pd.date_range("2020-01-01", periods=4)),
        )

        self.assertIn(
            "are present",
            " ".join(with_duplicates.for_check(rc.CHECK_DUPLICATES).reasoning),
        )
        self.assertIn(
            "No timestamp is repeated",
            " ".join(without.for_check(rc.CHECK_DUPLICATES).reasoning),
        )


class TestInvalidCounts(unittest.TestCase):
    def setUp(self):
        self.frequency = infer_frequency(pd.date_range("2020-01-01", periods=5))

    def test_zero_rows_raises(self):
        with self.assertRaises(ValueError):
            rc.recommend_checks(
                n_rows_used=0,
                n_distinct_timestamps=1,
                frequency=self.frequency,
            )

    def test_zero_distinct_timestamps_raises(self):
        with self.assertRaises(ValueError):
            rc.recommend_checks(
                n_rows_used=5,
                n_distinct_timestamps=0,
                frequency=self.frequency,
            )

    def test_more_distinct_than_rows_raises(self):
        with self.assertRaises(ValueError):
            rc.recommend_checks(
                n_rows_used=3,
                n_distinct_timestamps=4,
                frequency=self.frequency,
            )


if __name__ == "__main__":
    unittest.main()
