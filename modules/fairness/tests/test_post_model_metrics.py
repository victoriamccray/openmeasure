"""
Unit tests for core/post_model_metrics.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from core import post_model_metrics as pmm  # noqa: E402


def _make_df(rows):
    """rows is a list of (true_label, predicted_label, group) tuples."""
    return pd.DataFrame(rows, columns=["true_label", "predicted_label", "group"])


def _confusion_dataset():
    # Group A: actual positives=6 (TP=4, FN=2), actual negatives=4 (FP=1, TN=3)
    #   TPR = 4/6, FPR = 1/4, PPV = 4/5
    # Group B: actual positives=5 (TP=2, FN=3), actual negatives=5 (FP=2, TN=3)
    #   TPR = 2/5, FPR = 2/5, PPV = 2/4
    rows = (
        [(1, 1, "A")] * 4
        + [(1, 0, "A")] * 2
        + [(0, 1, "A")] * 1
        + [(0, 0, "A")] * 3
        + [(1, 1, "B")] * 2
        + [(1, 0, "B")] * 3
        + [(0, 1, "B")] * 2
        + [(0, 0, "B")] * 3
    )
    return _make_df(rows)


class TestComputeGroupConfusionRates(unittest.TestCase):
    def setUp(self):
        self.data = _confusion_dataset()

    def test_known_rate_calculations(self):
        results = {
            r.group: r
            for r in pmm.compute_group_confusion_rates(
                self.data,
                "true_label",
                "predicted_label",
                "group",
                positive_label=1,
            )
        }

        self.assertEqual(results["A"].n, 10)
        self.assertEqual((results["A"].true_positive, results["A"].false_positive), (4, 1))
        self.assertAlmostEqual(results["A"].true_positive_rate, 4 / 6, places=9)
        self.assertAlmostEqual(results["A"].false_positive_rate, 1 / 4, places=9)
        self.assertAlmostEqual(results["A"].positive_predictive_value, 4 / 5, places=9)

        self.assertEqual(results["B"].n, 10)
        self.assertAlmostEqual(results["B"].true_positive_rate, 2 / 5, places=9)
        self.assertAlmostEqual(results["B"].false_positive_rate, 2 / 5, places=9)
        self.assertAlmostEqual(results["B"].positive_predictive_value, 2 / 4, places=9)

    def test_flags_small_groups(self):
        data = pd.concat(
            [self.data, _make_df([(1, 1, "C"), (0, 0, "C")])],
            ignore_index=True,
        )
        results = {
            r.group: r
            for r in pmm.compute_group_confusion_rates(
                data, "true_label", "predicted_label", "group", positive_label=1
            )
        }
        self.assertTrue(results["C"].small_sample)
        self.assertFalse(results["A"].small_sample)

    def test_undefined_rate_reported_as_none_not_raised(self):
        # Group C has no actual positives, so true_positive_rate must be
        # None rather than a fabricated 0.0 or a division error.
        data = pd.concat(
            [self.data, _make_df([(0, 0, "C")] * 5)],
            ignore_index=True,
        )
        results = {
            r.group: r
            for r in pmm.compute_group_confusion_rates(
                data, "true_label", "predicted_label", "group", positive_label=1
            )
        }
        self.assertIsNone(results["C"].true_positive_rate)
        self.assertIsNotNone(results["C"].false_positive_rate)

    def test_raises_on_non_binary_true_label(self):
        data = pd.concat(
            [self.data, _make_df([(2, 1, "A")])],
            ignore_index=True,
        )
        with self.assertRaises(ValueError):
            pmm.compute_group_confusion_rates(
                data, "true_label", "predicted_label", "group", positive_label=1
            )


class TestComparePostModelBias(unittest.TestCase):
    def setUp(self):
        self.data = _confusion_dataset()

    def test_known_gap_calculations(self):
        result = pmm.compare_post_model_bias(
            self.data,
            "true_label",
            "predicted_label",
            "group",
            positive_label=1,
            privileged_group="A",
            unprivileged_group="B",
        )

        self.assertAlmostEqual(result.privileged_true_positive_rate, 4 / 6, places=9)
        self.assertAlmostEqual(result.unprivileged_true_positive_rate, 2 / 5, places=9)
        self.assertAlmostEqual(
            result.equal_opportunity_difference, (2 / 5) - (4 / 6), places=9
        )

        self.assertAlmostEqual(result.privileged_false_positive_rate, 1 / 4, places=9)
        self.assertAlmostEqual(result.unprivileged_false_positive_rate, 2 / 5, places=9)
        self.assertAlmostEqual(
            result.predictive_equality_difference, (2 / 5) - (1 / 4), places=9
        )

        self.assertAlmostEqual(
            result.calibration_within_groups_difference, (2 / 4) - (4 / 5), places=9
        )
        self.assertEqual(result.n_rows_used, 20)
        self.assertEqual(result.n_excluded_rows, 0)

    def test_raises_when_a_group_has_no_actual_positives(self):
        data = pd.concat(
            [self.data, _make_df([(0, 0, "C")] * 5 + [(0, 1, "C")])],
            ignore_index=True,
        )
        with self.assertRaises(ValueError):
            pmm.compare_post_model_bias(
                data,
                "true_label",
                "predicted_label",
                "group",
                positive_label=1,
                privileged_group="A",
                unprivileged_group="C",
            )

    def test_raises_on_same_group_selected_twice(self):
        with self.assertRaises(ValueError):
            pmm.compare_post_model_bias(
                self.data,
                "true_label",
                "predicted_label",
                "group",
                positive_label=1,
                privileged_group="A",
                unprivileged_group="A",
            )

    def test_raises_when_true_and_predicted_columns_are_the_same(self):
        with self.assertRaises(ValueError):
            pmm.compare_post_model_bias(
                self.data,
                "true_label",
                "true_label",
                "group",
                positive_label=1,
                privileged_group="A",
                unprivileged_group="B",
            )


if __name__ == "__main__":
    unittest.main()
