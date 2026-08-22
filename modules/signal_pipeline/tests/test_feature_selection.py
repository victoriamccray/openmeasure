"""
Unit tests for core/feature_selection.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import feature_selection as fs  # noqa: E402


def _result(name, n_features, performance, standard_error=0.05):
    return fs.FeatureSetResult(
        name=name,
        n_features=n_features,
        performance=performance,
        standard_error=standard_error,
    )


class TestFeatureSetResult(unittest.TestCase):
    def test_valid_result_constructs(self):
        result = _result("A", 1, 0.5)
        self.assertEqual(result.name, "A")

    def test_raises_on_empty_name(self):
        with self.assertRaises(ValueError):
            _result("", 1, 0.5)

    def test_raises_on_non_positive_n_features(self):
        with self.assertRaises(ValueError):
            _result("A", 0, 0.5)

    def test_raises_on_negative_standard_error(self):
        with self.assertRaises(ValueError):
            _result("A", 1, 0.5, standard_error=-0.01)


class TestSelectNecessaryFeatureSet(unittest.TestCase):
    def test_raises_on_empty_feature_sets(self):
        with self.assertRaises(ValueError):
            fs.select_necessary_feature_set(())

    def test_raises_on_duplicate_names(self):
        with self.assertRaises(ValueError):
            fs.select_necessary_feature_set(
                (_result("A", 1, 0.5), _result("A", 2, 0.6))
            )

    def test_single_feature_set_is_both_best_and_necessary(self):
        result = fs.select_necessary_feature_set((_result("A", 1, 0.5),))
        self.assertEqual(result.best, "A")
        self.assertEqual(result.necessary, "A")

    def test_best_is_the_highest_performance(self):
        result = fs.select_necessary_feature_set(
            (_result("A", 1, 0.3), _result("B", 2, 0.6), _result("C", 3, 0.5))
        )
        self.assertEqual(result.best, "B")

    def test_necessary_is_smaller_set_within_one_standard_error_of_best(self):
        # Best (B) performance 0.60, SE 0.05 -> threshold 0.55. A's 0.56 is
        # within that threshold and has fewer features, so A is necessary.
        a = _result("A", 1, 0.56, standard_error=0.05)
        b = _result("B", 3, 0.60, standard_error=0.05)
        result = fs.select_necessary_feature_set((a, b))
        self.assertEqual(result.best, "B")
        self.assertEqual(result.necessary, "A")
        self.assertEqual(result.within_one_se_of_best, {"A": True, "B": True})

    def test_necessary_equals_best_when_no_smaller_set_is_within_one_standard_error(self):
        # Best (B) performance 0.60, SE 0.01 -> threshold 0.59. A's 0.40 is
        # well below that, so A is excluded and B is its own necessary set.
        a = _result("A", 1, 0.40, standard_error=0.05)
        b = _result("B", 3, 0.60, standard_error=0.01)
        result = fs.select_necessary_feature_set((a, b))
        self.assertEqual(result.best, "B")
        self.assertEqual(result.necessary, "B")
        self.assertEqual(result.within_one_se_of_best, {"A": False, "B": True})

    def test_hand_calculated_three_feature_sets(self):
        # threshold = 0.70 - 0.04 = 0.66. C (0.68) and B (0.70) qualify;
        # A (0.50) does not. Necessary is the smaller of B, C -> C (2 features).
        a = _result("A", 1, 0.50, standard_error=0.05)
        b = _result("B", 4, 0.70, standard_error=0.04)
        c = _result("C", 2, 0.68, standard_error=0.03)
        result = fs.select_necessary_feature_set((a, b, c))
        self.assertEqual(result.best, "B")
        self.assertEqual(result.necessary, "C")
        self.assertEqual(
            result.within_one_se_of_best, {"A": False, "B": True, "C": True}
        )


if __name__ == "__main__":
    unittest.main()
