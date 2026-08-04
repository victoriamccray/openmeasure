"""
Unit tests for core/pre_model_metrics.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from core import pre_model_metrics as pm  # noqa: E402


def _make_df(group_col, label_col, rows):
    """rows is a list of (group_value, label_value) tuples."""
    return pd.DataFrame(rows, columns=[group_col, label_col])


class TestComputeGroupRates(unittest.TestCase):
    def setUp(self):
        # Group A: 4 favorable, 4 unfavorable, n=8, rate=0.5
        # Group B: 2 favorable, 8 unfavorable, n=10, rate=0.2
        rows = (
            [("A", 1)] * 4
            + [("A", 0)] * 4
            + [("B", 1)] * 2
            + [("B", 0)] * 8
        )
        self.data = _make_df("group", "label", rows)

    def test_known_favorable_rate_calculations(self):
        results = {
            r.group: r
            for r in pm.compute_group_rates(
                self.data, "label", "group", favorable_label=1
            )
        }

        self.assertEqual(results["A"].n, 8)
        self.assertEqual(results["A"].favorable_count, 4)
        self.assertEqual(results["A"].unfavorable_count, 4)
        self.assertAlmostEqual(results["A"].favorable_rate, 0.5)

        self.assertEqual(results["B"].n, 10)
        self.assertEqual(results["B"].favorable_count, 2)
        self.assertEqual(results["B"].unfavorable_count, 8)
        self.assertAlmostEqual(results["B"].favorable_rate, 0.2)

    def test_alphabetical_group_ordering(self):
        rows = [("Z", 1), ("Z", 0), ("A", 1), ("A", 0), ("M", 1), ("M", 0)]
        data = _make_df("group", "label", rows)

        results = pm.compute_group_rates(data, "label", "group", favorable_label=1)

        self.assertEqual([r.group for r in results], ["A", "M", "Z"])

    def test_small_group_warning_with_default_threshold(self):
        rows = [("Small", 1), ("Small", 0), ("Small", 1)] + [("Big", 1)] * 6
        data = _make_df("group", "label", rows)

        results = {
            r.group: r
            for r in pm.compute_group_rates(data, "label", "group", favorable_label=1)
        }

        self.assertTrue(results["Small"].small_sample)
        self.assertFalse(results["Big"].small_sample)

    def test_small_group_warning_with_custom_threshold(self):
        rows = [("A", 1)] * 3 + [("A", 0)] * 3 + [("B", 1)] * 3 + [("B", 0)] * 3
        data = _make_df("group", "label", rows)

        results = {
            r.group: r
            for r in pm.compute_group_rates(
                data, "label", "group", favorable_label=1, minimum_group_size=10
            )
        }

        self.assertTrue(results["A"].small_sample)
        self.assertTrue(results["B"].small_sample)

    def test_missing_values_excluded_before_rates_are_computed(self):
        rows = [("A", 1), ("A", 0), ("A", None), ("A", 1)]
        data = _make_df("group", "label", rows)

        results = {
            r.group: r
            for r in pm.compute_group_rates(data, "label", "group", favorable_label=1)
        }

        self.assertEqual(results["A"].n, 3)
        self.assertEqual(results["A"].favorable_count, 2)

    def test_missing_group_values_excluded(self):
        rows = [("A", 1), (None, 1), ("A", 0), ("A", 1)]
        data = _make_df("group", "label", rows)

        results = {
            r.group: r
            for r in pm.compute_group_rates(data, "label", "group", favorable_label=1)
        }

        self.assertEqual(set(results.keys()), {"A"})
        self.assertEqual(results["A"].n, 3)

    def test_missing_label_column_raises(self):
        with self.assertRaises(ValueError):
            pm.compute_group_rates(self.data, "not_a_column", "group", favorable_label=1)

    def test_missing_group_column_raises(self):
        with self.assertRaises(ValueError):
            pm.compute_group_rates(self.data, "label", "not_a_column", favorable_label=1)

    def test_nonbinary_label_raises(self):
        rows = [("A", 1), ("A", 2), ("A", 3)]
        data = _make_df("group", "label", rows)

        with self.assertRaises(ValueError) as ctx:
            pm.compute_group_rates(data, "label", "group", favorable_label=1)

        self.assertIn("exactly two nonmissing values", str(ctx.exception))

    def test_empty_dataframe_raises(self):
        empty = pd.DataFrame(columns=["group", "label"])

        with self.assertRaises(ValueError) as ctx:
            pm.compute_group_rates(empty, "label", "group", favorable_label=1)

        self.assertIn("no rows", str(ctx.exception))

    def test_all_missing_values_raises(self):
        rows = [("A", None), (None, 1)]
        data = _make_df("group", "label", rows)

        with self.assertRaises(ValueError) as ctx:
            pm.compute_group_rates(data, "label", "group", favorable_label=1)

        self.assertIn("No complete observations remain", str(ctx.exception))


class TestComputePreModelBias(unittest.TestCase):
    def setUp(self):
        # Privileged group "A": rate 0.5 (4 of 8)
        # Unprivileged group "B": rate 0.2 (2 of 10)
        rows = (
            [("A", 1)] * 4
            + [("A", 0)] * 4
            + [("B", 1)] * 2
            + [("B", 0)] * 8
        )
        self.data = _make_df("group", "label", rows)

    def test_disparate_impact(self):
        result = pm.compute_pre_model_bias(
            self.data,
            "label",
            "group",
            favorable_label=1,
            privileged_group="A",
            unprivileged_group="B",
        )

        self.assertAlmostEqual(result.disparate_impact, 0.2 / 0.5)

    def test_statistical_parity_difference(self):
        result = pm.compute_pre_model_bias(
            self.data,
            "label",
            "group",
            favorable_label=1,
            privileged_group="A",
            unprivileged_group="B",
        )

        self.assertAlmostEqual(result.statistical_parity_difference, 0.2 - 0.5)

    def test_parity_case(self):
        rows = [("C", 1)] * 4 + [("C", 0)] * 4 + [("D", 1)] * 5 + [("D", 0)] * 5
        data = _make_df("group", "label", rows)

        result = pm.compute_pre_model_bias(
            data,
            "label",
            "group",
            favorable_label=1,
            privileged_group="C",
            unprivileged_group="D",
        )

        self.assertAlmostEqual(result.disparate_impact, 1.0)
        self.assertAlmostEqual(result.statistical_parity_difference, 0.0)

    def test_zero_privileged_rate_raises(self):
        rows = [("P", 0)] * 5 + [("Q", 1)] * 3 + [("Q", 0)] * 2
        data = _make_df("group", "label", rows)

        with self.assertRaises(ValueError) as ctx:
            pm.compute_pre_model_bias(
                data,
                "label",
                "group",
                favorable_label=1,
                privileged_group="P",
                unprivileged_group="Q",
            )

        self.assertIn("disparate impact is undefined", str(ctx.exception))

    def test_missing_columns_raise(self):
        with self.assertRaises(ValueError):
            pm.compute_pre_model_bias(
                self.data,
                "not_a_column",
                "group",
                favorable_label=1,
                privileged_group="A",
                unprivileged_group="B",
            )

        with self.assertRaises(ValueError):
            pm.compute_pre_model_bias(
                self.data,
                "label",
                "not_a_column",
                favorable_label=1,
                privileged_group="A",
                unprivileged_group="B",
            )

    def test_missing_groups_raise(self):
        with self.assertRaises(ValueError) as ctx:
            pm.compute_pre_model_bias(
                self.data,
                "label",
                "group",
                favorable_label=1,
                privileged_group="does_not_exist",
                unprivileged_group="B",
            )
        self.assertIn("was not found in column", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            pm.compute_pre_model_bias(
                self.data,
                "label",
                "group",
                favorable_label=1,
                privileged_group="A",
                unprivileged_group="does_not_exist",
            )
        self.assertIn("was not found in column", str(ctx.exception))

    def test_nonbinary_labels_raise(self):
        rows = [("A", 1), ("A", 2), ("A", 3), ("B", 1)]
        data = _make_df("group", "label", rows)

        with self.assertRaises(ValueError) as ctx:
            pm.compute_pre_model_bias(
                data,
                "label",
                "group",
                favorable_label=1,
                privileged_group="A",
                unprivileged_group="B",
            )

        self.assertIn("exactly two nonmissing values", str(ctx.exception))

    def test_missing_values_excluded_before_comparison(self):
        rows = [("A", 1), ("A", 0), ("A", None), ("B", 1), ("B", 0), ("B", 0)]
        data = _make_df("group", "label", rows)

        result = pm.compute_pre_model_bias(
            data,
            "label",
            "group",
            favorable_label=1,
            privileged_group="A",
            unprivileged_group="B",
        )

        self.assertEqual(result.privileged_n, 2)
        self.assertEqual(result.unprivileged_n, 3)

    def test_identical_privileged_and_unprivileged_group_raises(self):
        with self.assertRaises(ValueError) as ctx:
            pm.compute_pre_model_bias(
                self.data,
                "label",
                "group",
                favorable_label=1,
                privileged_group="A",
                unprivileged_group="A",
            )

        self.assertIn("must be different", str(ctx.exception))

    def test_empty_dataframe_raises(self):
        empty = pd.DataFrame(columns=["group", "label"])

        with self.assertRaises(ValueError) as ctx:
            pm.compute_pre_model_bias(
                empty,
                "label",
                "group",
                favorable_label=1,
                privileged_group="A",
                unprivileged_group="B",
            )

        self.assertIn("no rows", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
