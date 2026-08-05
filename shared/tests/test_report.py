"""
Unit tests for shared/report.py

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest

from shared.report import Band, classify


BANDS = [
    Band(0.00, "Unacceptable", "error"),
    Band(0.50, "Poor", "error"),
    Band(0.70, "Acceptable", "info"),
    Band(0.90, "Excellent", "success"),
]


class TestClassify(unittest.TestCase):
    def test_returns_highest_band_whose_threshold_is_met(self):
        self.assertEqual(classify(0.82, BANDS).label, "Acceptable")

    def test_value_exactly_on_a_threshold_takes_that_band(self):
        self.assertEqual(classify(0.90, BANDS).label, "Excellent")

    def test_value_below_every_threshold_takes_the_lowest_band(self):
        self.assertEqual(classify(-1.0, BANDS).label, "Unacceptable")

    def test_value_above_every_threshold_takes_the_highest_band(self):
        self.assertEqual(classify(1.5, BANDS).label, "Excellent")

    def test_unsorted_bands_are_handled(self):
        shuffled = [BANDS[2], BANDS[0], BANDS[3], BANDS[1]]

        self.assertEqual(classify(0.82, shuffled).label, "Acceptable")

    def test_nan_silently_returns_the_worst_band(self):
        """
        Pins a trap rather than endorsing it.

        Every comparison against NaN is False, so a NaN never meets any
        threshold and classify falls back to the lowest band. A caller that
        passes an indeterminate metric therefore renders the most alarming
        possible verdict with no indication anything went wrong.

        Callers must pass None and skip rendering instead. This test exists
        so the behavior is documented and cannot change unnoticed.
        """
        self.assertEqual(classify(float("nan"), BANDS).label, "Unacceptable")


if __name__ == "__main__":
    unittest.main()
