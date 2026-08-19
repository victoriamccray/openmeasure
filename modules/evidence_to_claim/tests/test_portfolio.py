"""
Unit tests for core/portfolio.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import portfolio as p  # noqa: E402


def _portfolio_frame():
    return pd.DataFrame(
        {
            "grantee_id": ["G1", "G2", "G3", "G4", "G5"],
            "indicator_id": ["IND-1"] * 5,
            "result_value": [10, 20, 30, 40, 50],
            "unit": ["percent"] * 5,
        }
    )


class TestSummarizePortfolioIndicator(unittest.TestCase):
    def test_tukey_fences_match_hand_calculation(self):
        # [10, 20, 30, 40, 50] -> Q1=20, median=30, Q3=40 (linear
        # interpolation, matching numpy.percentile's default method).
        # IQR = 20, lower fence = 20 - 1.5*20 = -10, upper fence = 40 + 1.5*20 = 70.
        frame = _portfolio_frame()
        result = p.summarize_portfolio_indicator(frame, "IND-1", this_value=30)

        values = frame["result_value"].to_numpy(dtype=float)
        expected_q1, expected_median, expected_q3 = np.percentile(values, [25, 50, 75])

        self.assertAlmostEqual(result.q1, expected_q1)
        self.assertAlmostEqual(result.median_value, expected_median)
        self.assertAlmostEqual(result.q3, expected_q3)
        self.assertAlmostEqual(result.lower_fence, -10.0)
        self.assertAlmostEqual(result.upper_fence, 70.0)
        self.assertEqual(result.position, p.POSITION_WITHIN_RANGE)

    def test_value_above_upper_fence_is_flagged_above_range(self):
        frame = _portfolio_frame()
        result = p.summarize_portfolio_indicator(frame, "IND-1", this_value=100)
        self.assertEqual(result.position, p.POSITION_ABOVE_RANGE)

    def test_value_below_lower_fence_is_flagged_below_range(self):
        frame = _portfolio_frame()
        result = p.summarize_portfolio_indicator(frame, "IND-1", this_value=-50)
        self.assertEqual(result.position, p.POSITION_BELOW_RANGE)

    def test_raises_on_absent_indicator_id(self):
        frame = _portfolio_frame()
        with self.assertRaises(ValueError):
            p.summarize_portfolio_indicator(frame, "IND-DOES-NOT-EXIST", this_value=30)


class TestCheckComparability(unittest.TestCase):
    def test_no_flag_when_units_agree(self):
        frame = _portfolio_frame()
        flags = p.check_comparability(frame)
        self.assertEqual(flags, ())

    def test_flag_when_indicator_reported_in_two_units(self):
        frame = pd.DataFrame(
            {
                "grantee_id": ["G1", "G2"],
                "indicator_id": ["IND-1", "IND-1"],
                "result_value": [10, 20],
                "unit": ["percent", "count"],
            }
        )
        flags = p.check_comparability(frame)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].indicator_id, "IND-1")


if __name__ == "__main__":
    unittest.main()
