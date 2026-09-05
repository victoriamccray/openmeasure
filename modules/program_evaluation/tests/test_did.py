"""
Unit tests for core/did.py

Run with: pytest modules/program_evaluation/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import did  # noqa: E402


def _hand_calculable_frame() -> pd.DataFrame:
    """
    Four units with changes chosen so every reported quantity can be
    worked out by hand.

    Treated changes are 6 and 10, so the mean change is 8 and the
    variance is 8. Comparison changes are 3 and 5, so the mean change is
    4 and the variance is 2. The estimate is 8 - 4 = 4, and the standard
    error is sqrt(8/2 + 2/2) = sqrt(5).
    """
    return pd.DataFrame(
        {
            "clinic": ["treated", "treated", "control", "control"],
            "pre": [10.0, 20.0, 30.0, 40.0],
            "post": [16.0, 30.0, 33.0, 45.0],
        }
    )


def _random_panel(seed: int = 7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n_treated, n_comparison = 40, 55

    pre_treated = rng.normal(50, 8, n_treated)
    post_treated = pre_treated + rng.normal(6, 5, n_treated)
    pre_comparison = rng.normal(48, 12, n_comparison)
    post_comparison = pre_comparison + rng.normal(2, 9, n_comparison)

    return pd.DataFrame(
        {
            "arm": ["treated"] * n_treated + ["comparison"] * n_comparison,
            "pre": np.r_[pre_treated, pre_comparison],
            "post": np.r_[post_treated, post_comparison],
        }
    )


class TestKnownValues(unittest.TestCase):
    def test_matches_hand_calculated_estimate_and_standard_error(self):
        result = did.estimate_did(
            _hand_calculable_frame(),
            "clinic",
            "pre",
            "post",
            treated_label="treated",
        )

        self.assertAlmostEqual(result.change_treated, 8.0, places=12)
        self.assertAlmostEqual(result.change_comparison, 4.0, places=12)
        self.assertAlmostEqual(result.did_estimate, 4.0, places=12)
        self.assertAlmostEqual(result.standard_error, np.sqrt(5.0), places=12)

    def test_reports_the_four_cell_means_behind_the_estimate(self):
        result = did.estimate_did(
            _hand_calculable_frame(),
            "clinic",
            "pre",
            "post",
            treated_label="treated",
        )

        self.assertAlmostEqual(result.mean_pre_treated, 15.0, places=12)
        self.assertAlmostEqual(result.mean_post_treated, 23.0, places=12)
        self.assertAlmostEqual(result.mean_pre_comparison, 35.0, places=12)
        self.assertAlmostEqual(result.mean_post_comparison, 39.0, places=12)

        # The estimate must be exactly the difference of the two
        # differences the means describe, not merely close to it.
        self.assertAlmostEqual(
            result.did_estimate,
            (result.mean_post_treated - result.mean_pre_treated)
            - (result.mean_post_comparison - result.mean_pre_comparison),
            places=12,
        )

    def test_baseline_difference_is_treated_minus_comparison(self):
        result = did.estimate_did(
            _hand_calculable_frame(),
            "clinic",
            "pre",
            "post",
            treated_label="treated",
        )

        self.assertAlmostEqual(result.baseline_difference, -20.0, places=12)

    def test_welch_degrees_of_freedom_match_the_hand_calculation(self):
        # (8/2 + 2/2)^2 / ((8/2)^2 / 1 + (2/2)^2 / 1) = 25 / 17
        result = did.estimate_did(
            _hand_calculable_frame(),
            "clinic",
            "pre",
            "post",
            treated_label="treated",
        )

        self.assertAlmostEqual(result.degrees_of_freedom, 25.0 / 17.0, places=12)


class TestAgainstStatsmodels(unittest.TestCase):
    """
    Cross-implementation checks against the two regressions a
    difference-in-differences estimate is usually written as.

    did.py computes neither of them: it works directly on change scores
    with scipy, matching the rest of the module. These tests are what
    make that an equivalent implementation rather than an assertion in a
    docstring.
    """

    def test_estimate_matches_ols_interaction_coefficient(self):
        frame = _random_panel()
        result = did.estimate_did(
            frame, "arm", "pre", "post", treated_label="treated"
        )

        unit = np.arange(len(frame))
        long = pd.DataFrame(
            {
                "y": np.r_[frame["pre"].to_numpy(), frame["post"].to_numpy()],
                "treated": np.r_[
                    (frame["arm"] == "treated").astype(float),
                    (frame["arm"] == "treated").astype(float),
                ],
                "post": np.r_[np.zeros(len(frame)), np.ones(len(frame))],
                "unit": np.r_[unit, unit],
            }
        )
        fitted = smf.ols("y ~ treated + post + treated:post", data=long).fit(
            cov_type="cluster", cov_kwds={"groups": long["unit"]}
        )

        self.assertAlmostEqual(
            result.did_estimate, fitted.params["treated:post"], places=10
        )

    def test_standard_error_matches_hc2_on_the_change_regression(self):
        """
        For a regression on a single two-level indicator, the HC2 robust
        standard error is exactly the unequal-variance (Welch) standard
        error. Pinning the equality here is what lets did.py describe its
        interval as robust to unequal group variances.
        """
        frame = _random_panel()
        result = did.estimate_did(
            frame, "arm", "pre", "post", treated_label="treated"
        )

        changes = pd.DataFrame(
            {
                "change": frame["post"] - frame["pre"],
                "treated": (frame["arm"] == "treated").astype(float),
            }
        )
        fitted = smf.ols("change ~ treated", data=changes).fit(cov_type="HC2")

        self.assertAlmostEqual(
            result.did_estimate, fitted.params["treated"], places=10
        )
        self.assertAlmostEqual(
            result.standard_error, fitted.bse["treated"], places=10
        )


    def test_t_and_p_match_scipy_welch_ttest_on_the_change_scores(self):
        """
        did.py derives t and p from the estimate, its standard error, and
        the Welch degrees of freedom rather than calling scipy, to avoid
        a spurious cancellation warning when one group's changes are
        identical. This pins that the two routes agree.
        """
        frame = _random_panel()
        result = did.estimate_did(
            frame, "arm", "pre", "post", treated_label="treated"
        )

        change = frame["post"] - frame["pre"]
        t_expected, p_expected = scipy_stats.ttest_ind(
            change[frame["arm"] == "treated"],
            change[frame["arm"] == "comparison"],
            equal_var=False,
        )

        self.assertAlmostEqual(result.t_statistic, t_expected, places=10)
        self.assertAlmostEqual(result.p_value, p_expected, places=10)


class TestTreatedLabel(unittest.TestCase):
    def test_swapping_the_treated_label_flips_the_sign(self):
        frame = _hand_calculable_frame()

        treated = did.estimate_did(
            frame, "clinic", "pre", "post", treated_label="treated"
        )
        flipped = did.estimate_did(
            frame, "clinic", "pre", "post", treated_label="control"
        )

        self.assertAlmostEqual(
            flipped.did_estimate, -treated.did_estimate, places=12
        )
        self.assertAlmostEqual(
            flipped.standard_error, treated.standard_error, places=12
        )
        self.assertEqual(flipped.treated_label, "control")
        self.assertEqual(flipped.comparison_label, "treated")

    def test_rejects_a_treated_label_not_in_the_group_column(self):
        with self.assertRaises(ValueError) as raised:
            did.estimate_did(
                _hand_calculable_frame(),
                "clinic",
                "pre",
                "post",
                treated_label="not_a_group",
            )

        self.assertIn("not_a_group", str(raised.exception))


class TestExclusionAccounting(unittest.TestCase):
    def test_incomplete_units_are_excluded_and_counted(self):
        frame = _hand_calculable_frame()
        frame.loc[len(frame)] = ["treated", np.nan, 25.0]
        frame.loc[len(frame)] = ["control", 30.0, np.nan]

        result = did.estimate_did(
            frame, "clinic", "pre", "post", treated_label="treated"
        )

        self.assertEqual(result.n_input_rows, 6)
        self.assertEqual(result.n_rows_used, 4)
        self.assertEqual(result.n_excluded_rows, 2)
        self.assertEqual(result.exclusion_reason, did.MISSING_GROUP_PRE_OR_POST)

        # The dropped rows must not have moved the estimate.
        self.assertAlmostEqual(result.did_estimate, 4.0, places=12)

    def test_small_groups_are_flagged_rather_than_refused(self):
        result = did.estimate_did(
            _hand_calculable_frame(),
            "clinic",
            "pre",
            "post",
            treated_label="treated",
        )

        self.assertEqual(len(result.small_groups_flagged), 2)
        self.assertTrue(
            any("treated" in flag for flag in result.small_groups_flagged)
        )

    def test_large_groups_are_not_flagged(self):
        result = did.estimate_did(
            _random_panel(), "arm", "pre", "post", treated_label="treated"
        )

        self.assertEqual(result.small_groups_flagged, ())


class TestDegenerateInput(unittest.TestCase):
    def test_rejects_non_dataframe(self):
        with self.assertRaises(TypeError):
            did.estimate_did(
                [1, 2, 3], "clinic", "pre", "post", treated_label="treated"
            )

    def test_rejects_missing_column(self):
        with self.assertRaises(ValueError) as raised:
            did.estimate_did(
                _hand_calculable_frame(),
                "clinic",
                "baseline",
                "post",
                treated_label="treated",
            )

        self.assertIn("baseline", str(raised.exception))

    def test_rejects_repeated_column(self):
        with self.assertRaises(ValueError) as raised:
            did.estimate_did(
                _hand_calculable_frame(),
                "clinic",
                "pre",
                "pre",
                treated_label="treated",
            )

        self.assertIn("three different", str(raised.exception))

    def test_rejects_three_groups(self):
        frame = _hand_calculable_frame()
        frame.loc[len(frame)] = ["third_arm", 12.0, 15.0]
        frame.loc[len(frame)] = ["third_arm", 14.0, 19.0]

        with self.assertRaises(ValueError) as raised:
            did.estimate_did(
                frame, "clinic", "pre", "post", treated_label="treated"
            )

        self.assertIn("exactly 2 groups", str(raised.exception))

    def test_rejects_a_group_with_only_one_complete_unit(self):
        frame = pd.DataFrame(
            {
                "clinic": ["treated", "control", "control"],
                "pre": [10.0, 30.0, 40.0],
                "post": [16.0, 33.0, 45.0],
            }
        )

        with self.assertRaises(ValueError) as raised:
            did.estimate_did(
                frame, "clinic", "pre", "post", treated_label="treated"
            )

        self.assertIn("at least 2 units", str(raised.exception))

    def test_rejects_zero_variance_in_both_groups(self):
        """
        Every unit changing by an identical amount leaves a standard
        error of zero, which would print an interval of zero width and a
        p-value of nan rather than failing.
        """
        frame = pd.DataFrame(
            {
                "clinic": ["treated", "treated", "control", "control"],
                "pre": [10.0, 20.0, 30.0, 40.0],
                "post": [15.0, 25.0, 32.0, 42.0],
            }
        )

        with self.assertRaises(ValueError) as raised:
            did.estimate_did(
                frame, "clinic", "pre", "post", treated_label="treated"
            )

        self.assertIn("standard error is zero", str(raised.exception))

    def test_allows_zero_variance_in_only_one_group(self):
        """One group with identical changes still leaves a usable standard error."""
        frame = pd.DataFrame(
            {
                "clinic": ["treated", "treated", "control", "control"],
                "pre": [10.0, 20.0, 30.0, 40.0],
                "post": [15.0, 25.0, 33.0, 45.0],
            }
        )

        result = did.estimate_did(
            frame, "clinic", "pre", "post", treated_label="treated"
        )

        self.assertGreater(result.standard_error, 0)
        self.assertAlmostEqual(result.did_estimate, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
