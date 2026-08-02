"""
Unit tests for core/recommend.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import recommend as rec  # noqa: E402


class TestInputValidation(unittest.TestCase):
    def test_rejects_non_dataframe(self):
        with self.assertRaises(TypeError):
            rec.recommend_method([1, 2, 3], outcome_col="x", group_col="y")

    def test_rejects_empty_dataframe(self):
        with self.assertRaises(ValueError):
            rec.recommend_method(pd.DataFrame(), outcome_col="x", group_col="y")

    def test_rejects_both_designs_given(self):
        data = pd.DataFrame({"x": [1, 2], "y": ["a", "b"], "pre": [1, 2], "post": [2, 3]})
        with self.assertRaises(ValueError):
            rec.recommend_method(data, outcome_col="x", group_col="y", pre_col="pre", post_col="post")

    def test_rejects_neither_design_given(self):
        data = pd.DataFrame({"x": [1, 2]})
        with self.assertRaises(ValueError):
            rec.recommend_method(data)

    def test_rejects_outcome_without_group(self):
        data = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        with self.assertRaises(ValueError):
            rec.recommend_method(data, outcome_col="x")

    def test_rejects_group_without_outcome(self):
        data = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        with self.assertRaises(ValueError):
            rec.recommend_method(data, group_col="y")


class TestGroupComparisonRecommendation(unittest.TestCase):
    def test_likert_outcome_two_groups_recommends_welch_ttest(self):
        """A 1-5 Likert outcome must not be misclassified as categorical."""
        rng = np.random.RandomState(0)
        data = pd.DataFrame({
            "format": ["online"] * 30 + ["in_person"] * 30,
            "satisfaction": rng.randint(1, 6, 60),
        })
        result = rec.recommend_method(data, outcome_col="satisfaction", group_col="format")
        self.assertEqual(result.method, "compare_two_groups")
        self.assertTrue(result.supported)

    def test_string_categorical_outcome_two_groups_recommends_chi_square(self):
        data = pd.DataFrame({
            "format": ["online"] * 30 + ["in_person"] * 30,
            "applied": ["yes"] * 15 + ["no"] * 15 + ["yes"] * 20 + ["no"] * 10,
        })
        result = rec.recommend_method(data, outcome_col="applied", group_col="format")
        self.assertEqual(result.method, "compare_categorical")

    def test_numeric_binary_outcome_flags_warning(self):
        data = pd.DataFrame({
            "format": ["online"] * 30 + ["in_person"] * 30,
            "applied_binary": [0] * 15 + [1] * 15 + [0] * 10 + [1] * 20,
        })
        result = rec.recommend_method(data, outcome_col="applied_binary", group_col="format")
        self.assertEqual(result.method, "compare_categorical")
        self.assertTrue(any("binary categorical" in w for w in result.warnings))

    def test_likert_outcome_three_groups_recommends_welch_anova(self):
        rng = np.random.RandomState(1)
        data = pd.DataFrame({
            "race": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "inclusiveness": rng.randint(1, 6, 60),
        })
        result = rec.recommend_method(data, outcome_col="inclusiveness", group_col="race")
        self.assertEqual(result.method, "compare_multiple_groups_welch")
        self.assertIn("Games-Howell", result.display_name)

    def test_many_groups_adds_pairwise_count_warning(self):
        rng = np.random.RandomState(2)
        labels = [f"G{i}" for i in range(7)]
        data = pd.DataFrame({
            "race": [label for label in labels for _ in range(10)],
            "score": rng.normal(4, 0.5, 70),
        })
        result = rec.recommend_method(data, outcome_col="score", group_col="race")
        self.assertTrue(any("21 pairwise" in w for w in result.warnings))

    def test_multiselect_group_recommends_sensitivity_analysis(self):
        rng = np.random.RandomState(3)
        data = pd.DataFrame({
            "race": ["A", "A, B", "C"] * 10,
            "score": rng.normal(4, 0.5, 30),
        })
        result = rec.recommend_method(
            data, outcome_col="score", group_col="race", is_multiselect_group=True
        )
        self.assertEqual(result.method, "sensitivity_analysis")

    def test_raises_on_missing_outcome_column(self):
        data = pd.DataFrame({"group": ["A", "B"]})
        with self.assertRaises(ValueError):
            rec.recommend_method(data, outcome_col="missing", group_col="group")

    def test_raises_on_missing_group_column(self):
        data = pd.DataFrame({"outcome": [1, 2]})
        with self.assertRaises(ValueError):
            rec.recommend_method(data, outcome_col="outcome", group_col="missing")

    def test_raises_on_fewer_than_two_groups(self):
        data = pd.DataFrame({"group": ["A", "A", "A"], "outcome": [1, 2, 3]})
        with self.assertRaises(ValueError):
            rec.recommend_method(data, outcome_col="outcome", group_col="group")

    def test_randomization_caveat_present_for_two_group_comparison(self):
        rng = np.random.RandomState(4)
        data = pd.DataFrame({
            "format": ["online"] * 20 + ["in_person"] * 20,
            "satisfaction": rng.randint(1, 6, 40),
        })
        result = rec.recommend_method(data, outcome_col="satisfaction", group_col="format")
        self.assertTrue(any("not randomized" in w for w in result.warnings))


class TestPrePostRecommendation(unittest.TestCase):
    def test_continuous_pre_post_recommends_paired_ttest(self):
        data = pd.DataFrame({"pre": [1, 2, 3, 4, 5], "post": [2, 3, 4, 5, 6]})
        result = rec.recommend_method(data, pre_col="pre", post_col="post")
        self.assertEqual(result.method, "compare_pre_post")
        self.assertTrue(result.supported)

    def test_both_categorical_recommends_mcnemar_unsupported(self):
        data = pd.DataFrame({
            "pre": ["yes", "no", "yes", "no", "yes", "yes"],
            "post": ["yes", "yes", "no", "no", "yes", "no"],
        })
        result = rec.recommend_method(data, pre_col="pre", post_col="post")
        self.assertEqual(result.method, "compare_paired_categorical")
        self.assertFalse(result.supported)

    def test_mismatched_types_recommends_review_unsupported(self):
        data = pd.DataFrame({
            "pre": [1, 2, 3, 4, 5, 6],
            "post": ["yes", "no", "yes", "no", "yes", "no"],
        })
        result = rec.recommend_method(data, pre_col="pre", post_col="post")
        self.assertEqual(result.method, "review_pre_post_coding")
        self.assertFalse(result.supported)

    def test_rejects_identical_pre_post_columns(self):
        data = pd.DataFrame({"pre": [1, 2, 3, 4]})
        with self.assertRaises(ValueError):
            rec.recommend_method(data, pre_col="pre", post_col="pre")

    def test_rejects_too_few_complete_pairs(self):
        data = pd.DataFrame({
            "pre": [1, np.nan, np.nan],
            "post": [2, np.nan, np.nan],
        })
        with self.assertRaises(ValueError):
            rec.recommend_method(data, pre_col="pre", post_col="post")

    def test_maturation_caveat_present_for_pre_post(self):
        data = pd.DataFrame({"pre": [1, 2, 3, 4, 5], "post": [2, 3, 4, 5, 6]})
        result = rec.recommend_method(data, pre_col="pre", post_col="post")
        self.assertTrue(any("maturation" in w for w in result.warnings))


class TestIdentifierColumnSafeguard(unittest.TestCase):
    def test_sequential_integer_group_column_flagged_as_identifier(self):
        rng = np.random.RandomState(5)
        data = pd.DataFrame({
            "participant_id": range(1, 21),
            "score": rng.normal(4, 0.5, 20),
        })
        result = rec.recommend_method(data, outcome_col="score", group_col="participant_id")
        self.assertTrue(any("row identifier" in w for w in result.warnings))

    def test_sequential_integer_outcome_column_flagged_as_identifier(self):
        rng = np.random.RandomState(6)
        data = pd.DataFrame({
            "participant_id": range(1, 61),
            "format": ["a"] * 30 + ["b"] * 30,
        })
        result = rec.recommend_method(data, outcome_col="participant_id", group_col="format")
        self.assertTrue(any("identifier" in w for w in result.warnings))

    def test_legitimate_columns_produce_no_identifier_warning(self):
        rng = np.random.RandomState(7)
        data = pd.DataFrame({
            "format": ["a"] * 30 + ["b"] * 30,
            "satisfaction": rng.randint(1, 6, 60),
        })
        result = rec.recommend_method(data, outcome_col="satisfaction", group_col="format")
        self.assertFalse(any("identifier" in w for w in result.warnings))

    def test_name_based_id_detection_without_sequential_values(self):
        """A column named like an ID should be flagged even if values
        aren't a clean sequential run, as long as every value is unique."""
        rng = np.random.RandomState(8)
        data = pd.DataFrame({
            "record_id": rng.choice(range(1000, 9999), size=20, replace=False),
            "score": rng.normal(4, 0.5, 20),
        })
        result = rec.recommend_method(data, outcome_col="score", group_col="record_id")
        self.assertTrue(any("identifier" in w for w in result.warnings))


class TestMultiselectDelimiterSafeguard(unittest.TestCase):
    def test_raises_when_delimiter_never_appears(self):
        rng = np.random.RandomState(9)
        data = pd.DataFrame({
            "format": ["online"] * 20 + ["in_person"] * 20,
            "score": rng.normal(4, 0.5, 40),
        })
        with self.assertRaises(ValueError):
            rec.recommend_method(
                data, outcome_col="score", group_col="format", is_multiselect_group=True
            )

    def test_succeeds_when_delimiter_present_in_at_least_one_value(self):
        rng = np.random.RandomState(10)
        data = pd.DataFrame({
            "race": ["A"] * 15 + ["B"] * 15 + ["A, B"] * 3,
            "score": rng.normal(4, 0.5, 33),
        })
        result = rec.recommend_method(
            data, outcome_col="score", group_col="race", is_multiselect_group=True
        )
        self.assertEqual(result.method, "sensitivity_analysis")

    def test_respects_custom_delimiter(self):
        rng = np.random.RandomState(11)
        data = pd.DataFrame({
            "race": ["A"] * 15 + ["B"] * 15 + ["A; B"] * 3,
            "score": rng.normal(4, 0.5, 33),
        })
        # default delimiter "," won't match "; ", should raise
        with self.assertRaises(ValueError):
            rec.recommend_method(
                data, outcome_col="score", group_col="race", is_multiselect_group=True
            )
        # correct delimiter should succeed
        result = rec.recommend_method(
            data,
            outcome_col="score",
            group_col="race",
            is_multiselect_group=True,
            multiselect_delimiter=";",
        )
        self.assertEqual(result.method, "sensitivity_analysis")


if __name__ == "__main__":
    unittest.main()
