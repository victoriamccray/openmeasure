"""
Unit tests for core/reliability.py

Run with:
    pytest tests/

or:
    python -m unittest tests/test_reliability.py
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    ),
)

from core import reliability as rel  # noqa: E402


class TestCronbachAlpha(unittest.TestCase):
    def test_perfect_correlation_gives_alpha_one(self) -> None:
        """Identical items with participant variance should produce alpha = 1."""
        base = np.array(
            [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            dtype=float,
        )

        data = pd.DataFrame(
            {
                f"item_{index}": base
                for index in range(5)
            }
        )

        alpha = rel.cronbach_alpha(data)

        self.assertAlmostEqual(alpha, 1.0, places=6)

    def test_known_value_regression(self) -> None:
        """A fixed dataset should continue to produce the same alpha."""
        data = pd.DataFrame(
            {
                "q1": [4, 3, 5, 2, 4, 5, 3, 4],
                "q2": [5, 2, 5, 2, 4, 5, 3, 4],
                "q3": [4, 3, 4, 1, 5, 4, 2, 5],
                "q4": [5, 3, 5, 2, 4, 5, 3, 4],
            }
        )

        expected_alpha = 0.9479436295657175

        alpha = rel.cronbach_alpha(data)

        self.assertAlmostEqual(
            alpha,
            expected_alpha,
            places=6,
        )

    def test_raises_on_single_item(self) -> None:
        data = pd.DataFrame(
            {
                "q1": [1, 2, 3, 4],
            }
        )

        with self.assertRaises(ValueError):
            rel.cronbach_alpha(data)

    def test_raises_on_zero_variance_total(self) -> None:
        """Alpha is undefined when all total scores are identical."""
        data = pd.DataFrame(
            {
                "q1": [3, 3, 3],
                "q2": [3, 3, 3],
            }
        )

        with self.assertRaises(ValueError):
            rel.cronbach_alpha(data)

    def test_rejects_nonnumeric_columns(self) -> None:
        data = pd.DataFrame(
            {
                "q1": [1, 2, 3],
                "q2": ["low", "medium", "high"],
            }
        )

        with self.assertRaises(TypeError):
            rel.cronbach_alpha(data)

    def test_rejects_duplicate_column_names(self) -> None:
        data = pd.DataFrame(
            [
                [1, 2],
                [2, 3],
                [3, 4],
            ],
            columns=["q1", "q1"],
        )

        with self.assertRaises(ValueError):
            rel.cronbach_alpha(data)

    def test_rejects_infinite_values(self) -> None:
        data = pd.DataFrame(
            {
                "q1": [1, 2, np.inf],
                "q2": [2, 3, 4],
            }
        )

        with self.assertRaises(ValueError):
            rel.cronbach_alpha(data)


class TestItemTotalCorrelations(unittest.TestCase):
    def test_returns_one_value_per_item(self) -> None:
        data = pd.DataFrame(
            {
                "q1": [4, 3, 5, 2, 4],
                "q2": [5, 2, 5, 2, 4],
                "q3": [4, 3, 4, 1, 5],
            }
        )

        correlations = rel.item_total_correlations(data)

        self.assertEqual(len(correlations), 3)
        self.assertListEqual(
            list(correlations.index),
            ["q1", "q2", "q3"],
        )

        finite_values = correlations.dropna()

        self.assertTrue(
            all(
                -1.0 <= value <= 1.0
                for value in finite_values
            )
        )

    def test_zero_variance_item_gives_nan(self) -> None:
        data = pd.DataFrame(
            {
                "q1": [3, 3, 3, 3],
                "q2": [1, 2, 3, 4],
                "q3": [4, 3, 2, 1],
            }
        )

        correlations = rel.item_total_correlations(data)

        self.assertTrue(np.isnan(correlations["q1"]))

    def test_item_is_excluded_from_its_corrected_total(self) -> None:
        """The implementation should correlate each item with all other items."""
        data = pd.DataFrame(
            {
                "q1": [1, 2, 3, 4, 5],
                "q2": [2, 4, 6, 8, 10],
                "q3": [5, 4, 3, 2, 1],
            }
        )

        correlations = rel.item_total_correlations(data)

        expected_q1 = data["q1"].corr(
            data[["q2", "q3"]].sum(axis=1)
        )

        self.assertAlmostEqual(
            correlations["q1"],
            expected_q1,
            places=6,
        )


class TestAlphaIfItemDropped(unittest.TestCase):
    def test_returns_one_value_per_item(self) -> None:
        data = pd.DataFrame(
            {
                "q1": [1, 2, 3, 4, 5],
                "q2": [2, 3, 4, 5, 6],
                "q3": [1, 3, 2, 4, 5],
                "q4": [2, 4, 3, 5, 6],
            }
        )

        dropped = rel.alpha_if_item_dropped(data)

        self.assertEqual(len(dropped), 4)
        self.assertListEqual(
            list(dropped.index),
            ["q1", "q2", "q3", "q4"],
        )

    def test_two_item_scale_returns_nan_after_dropping(self) -> None:
        data = pd.DataFrame(
            {
                "q1": [1, 2, 3, 4],
                "q2": [2, 3, 4, 5],
            }
        )

        dropped = rel.alpha_if_item_dropped(data)

        self.assertTrue(dropped.isna().all())


class TestSplitHalf(unittest.TestCase):
    def test_requires_at_least_four_items(self) -> None:
        data = pd.DataFrame(
            {
                "q1": [1, 2, 3],
                "q2": [2, 3, 4],
                "q3": [1, 2, 3],
            }
        )

        with self.assertRaises(ValueError):
            rel.split_half_reliability(data)

    def test_default_allows_odd_number_of_items(self) -> None:
        """By default (require_equal_halves=False), an odd item count
        should succeed using unequal-sized halves rather than raising."""
        data = pd.DataFrame(
            np.arange(50, dtype=float).reshape(10, 5),
            columns=[f"q{index}" for index in range(5)],
        )

        correlation, corrected = rel.split_half_reliability(data)

        self.assertIsInstance(correlation, float)
        self.assertIsInstance(corrected, float)

    def test_require_equal_halves_rejects_odd_number_of_items(self) -> None:
        """With require_equal_halves=True, an odd item count should raise,
        since the two halves would be unequal length."""
        data = pd.DataFrame(
            np.arange(50, dtype=float).reshape(10, 5),
            columns=[f"q{index}" for index in range(5)],
        )

        with self.assertRaises(ValueError):
            rel.split_half_reliability(data, require_equal_halves=True)

    def test_require_equal_halves_accepts_even_number_of_items(self) -> None:
        data = pd.DataFrame(
            np.arange(60, dtype=float).reshape(10, 6),
            columns=[f"q{index}" for index in range(6)],
        )

        correlation, corrected = rel.split_half_reliability(
            data, require_equal_halves=True
        )

        self.assertIsInstance(correlation, float)
        self.assertIsInstance(corrected, float)

    def test_returns_two_floats_for_valid_input(self) -> None:
        rng = np.random.RandomState(0)

        data = pd.DataFrame(
            rng.randint(1, 6, size=(30, 6)),
            columns=[f"q{index}" for index in range(6)],
        )

        correlation, corrected = rel.split_half_reliability(data)

        self.assertIsInstance(correlation, float)
        self.assertIsInstance(corrected, float)
        self.assertTrue(np.isfinite(correlation))
        self.assertTrue(np.isfinite(corrected))

    def test_uses_odd_even_item_positions(self) -> None:
        data = pd.DataFrame(
            {
                "q1": [1, 2, 3, 4, 5],
                "q2": [2, 4, 6, 8, 10],
                "q3": [3, 6, 9, 12, 15],
                "q4": [4, 8, 12, 16, 20],
            }
        )

        correlation, corrected = rel.split_half_reliability(data)

        odd_scores = data[["q1", "q3"]].sum(axis=1)
        even_scores = data[["q2", "q4"]].sum(axis=1)

        expected_correlation = odd_scores.corr(even_scores)
        expected_corrected = (
            2 * expected_correlation
            / (1 + expected_correlation)
        )

        self.assertAlmostEqual(
            correlation,
            expected_correlation,
            places=6,
        )
        self.assertAlmostEqual(
            corrected,
            expected_corrected,
            places=6,
        )

    def test_uses_unequal_odd_even_item_positions_by_default(self) -> None:
        """For an odd item count, the odd half should have one more item
        than the even half (positions 1,3,5 vs 2,4), and the default
        (require_equal_halves=False) should compute from exactly those
        unequal-sized groups."""
        data = pd.DataFrame(
            {
                "q1": [1, 2, 3, 4, 5],
                "q2": [2, 4, 6, 8, 10],
                "q3": [3, 6, 9, 12, 15],
                "q4": [4, 8, 12, 16, 20],
                "q5": [5, 10, 15, 20, 25],
            }
        )

        correlation, corrected = rel.split_half_reliability(data)

        odd_scores = data[["q1", "q3", "q5"]].sum(axis=1)
        even_scores = data[["q2", "q4"]].sum(axis=1)

        expected_correlation = odd_scores.corr(even_scores)
        expected_corrected = (
            2 * expected_correlation
            / (1 + expected_correlation)
        )

        self.assertAlmostEqual(correlation, expected_correlation, places=6)
        self.assertAlmostEqual(corrected, expected_corrected, places=6)

    def test_raises_when_one_half_has_zero_variance(self) -> None:
        data = pd.DataFrame(
            {
                "q1": [1, 1, 1, 1],
                "q2": [1, 2, 3, 4],
                "q3": [2, 2, 2, 2],
                "q4": [4, 3, 2, 1],
            }
        )

        with self.assertRaises(ValueError):
            rel.split_half_reliability(data)


class TestAnalyzePipeline(unittest.TestCase):
    def test_full_pipeline_runs_end_to_end(self) -> None:
        rng = np.random.RandomState(1)

        data = pd.DataFrame(
            rng.randint(1, 6, size=(50, 6)),
            columns=[f"q{index}" for index in range(6)],
        )

        result = rel.analyze(data)

        self.assertEqual(result.n_participants, 50)
        self.assertEqual(result.n_items, 6)
        self.assertEqual(result.n_complete_cases, 50)
        self.assertEqual(result.n_excluded_cases, 0)
        self.assertEqual(result.pct_excluded_cases, 0.0)
        self.assertEqual(result.pct_missing_cells, 0.0)
        self.assertEqual(len(result.item_diagnostics), 6)
        self.assertTrue(np.isfinite(result.cronbach_alpha))
        self.assertIsNotNone(result.split_half_correlation)
        self.assertIsNotNone(result.spearman_brown)

    def test_missing_data_uses_listwise_deletion(self) -> None:
        data = pd.DataFrame(
            {
                "q1": [4, 3, np.nan, 2, 4, 5],
                "q2": [5, 2, 5, 2, 4, 5],
                "q3": [4, 3, 4, 1, 5, 4],
                "q4": [5, 3, 5, 2, 4, 5],
            }
        )

        result = rel.analyze(data)

        self.assertEqual(result.n_participants, 6)
        self.assertEqual(result.n_complete_cases, 5)
        self.assertEqual(result.n_excluded_cases, 1)
        self.assertAlmostEqual(
            result.pct_excluded_cases,
            100 / 6,
            places=6,
        )
        self.assertAlmostEqual(
            result.pct_missing_cells,
            100 / 24,
            places=6,
        )

    def test_raises_when_too_few_complete_cases_remain(self) -> None:
        data = pd.DataFrame(
            {
                "q1": [np.nan, np.nan, 1],
                "q2": [np.nan, np.nan, 2],
            }
        )

        with self.assertRaises(ValueError):
            rel.analyze(data)

    def test_split_half_is_optional_for_two_item_scale(self) -> None:
        data = pd.DataFrame(
            {
                "q1": [1, 2, 3, 4, 5],
                "q2": [2, 3, 4, 5, 6],
            }
        )

        result = rel.analyze(data)

        self.assertIsNone(result.split_half_correlation)
        self.assertIsNone(result.spearman_brown)
        self.assertEqual(len(result.item_diagnostics), 2)

    def test_split_half_available_by_default_for_odd_item_count(self) -> None:
        """analyze() with default settings (require_equal_halves=False)
        should populate split-half values even for an odd number of items,
        using unequal-sized halves."""
        data = pd.DataFrame(
            {
                "q1": [1, 2, 3, 4, 5],
                "q2": [2, 3, 4, 5, 6],
                "q3": [1, 3, 2, 4, 5],
                "q4": [2, 4, 3, 5, 6],
                "q5": [3, 5, 4, 6, 7],
            }
        )

        result = rel.analyze(data)

        self.assertIsNotNone(result.split_half_correlation)
        self.assertIsNotNone(result.spearman_brown)

    def test_split_half_none_when_equal_halves_required_for_odd_item_count(self) -> None:
        """analyze() with require_equal_halves=True should return None for
        split-half values when the item count is odd, rather than raising
        out of the whole analysis."""
        data = pd.DataFrame(
            {
                "q1": [1, 2, 3, 4, 5],
                "q2": [2, 3, 4, 5, 6],
                "q3": [1, 3, 2, 4, 5],
                "q4": [2, 4, 3, 5, 6],
                "q5": [3, 5, 4, 6, 7],
            }
        )

        result = rel.analyze(data, require_equal_halves=True)

        self.assertIsNone(result.split_half_correlation)
        self.assertIsNone(result.spearman_brown)

    def test_low_item_total_correlation_is_flagged(self) -> None:
        data = pd.DataFrame(
            {
                "q1": [1, 2, 3, 4, 5, 6],
                "q2": [1, 2, 3, 4, 5, 6],
                "q3": [6, 1, 5, 2, 4, 3],
                "q4": [1, 2, 3, 4, 5, 6],
            }
        )

        result = rel.analyze(data)

        diagnostic_by_item = {
            diagnostic.item: diagnostic
            for diagnostic in result.item_diagnostics
        }

        self.assertTrue(diagnostic_by_item["q3"].flagged)

    def test_split_half_reason_is_none_when_available(self) -> None:
        rng = np.random.RandomState(2)

        data = pd.DataFrame(
            rng.randint(1, 6, size=(30, 6)),
            columns=[f"q{index}" for index in range(6)],
        )

        result = rel.analyze(data)

        self.assertIsNotNone(result.split_half_correlation)
        self.assertIsNone(result.split_half_unavailable_reason)

    def test_split_half_reason_names_the_selected_item_count(self) -> None:
        data = pd.DataFrame(
            {
                "q1": [1, 2, 3, 4, 5],
                "q2": [2, 3, 4, 5, 6],
            }
        )

        result = rel.analyze(data)

        self.assertIsNone(result.split_half_correlation)
        self.assertEqual(
            result.split_half_unavailable_reason,
            "Split-half reliability requires at least 4 items - 2 selected.",
        )

    def test_split_half_reason_names_the_zero_variance_half(self) -> None:
        # q1 + q3 (the odd half) is constant across rows, so that half has
        # zero variance, while the total across all four items still varies
        # so that cronbach_alpha itself succeeds and the pipeline reaches
        # the split-half step.
        data = pd.DataFrame(
            {
                "q1": [1, 2, 3, 4],
                "q2": [1, 2, 1, 2],
                "q3": [4, 3, 2, 1],
                "q4": [1, 2, 3, 1],
            }
        )

        result = rel.analyze(data)

        self.assertIsNone(result.split_half_correlation)
        self.assertIsNotNone(result.split_half_unavailable_reason)
        self.assertIn("variance", result.split_half_unavailable_reason)

    def test_split_half_reason_names_the_equal_halves_requirement(self) -> None:
        data = pd.DataFrame(
            {
                "q1": [1, 2, 3, 4, 5],
                "q2": [2, 3, 4, 5, 6],
                "q3": [1, 3, 2, 4, 5],
                "q4": [2, 4, 3, 5, 6],
                "q5": [3, 5, 4, 6, 7],
            }
        )

        result = rel.analyze(data, require_equal_halves=True)

        self.assertIsNone(result.split_half_correlation)
        self.assertIsNotNone(result.split_half_unavailable_reason)
        self.assertIn("even number of items", result.split_half_unavailable_reason)

    def test_result_uses_original_participant_count(self) -> None:
        data = pd.DataFrame(
            {
                "q1": [1, 2, np.nan, 4],
                "q2": [2, 3, 4, 5],
                "q3": [3, 4, 5, 6],
                "q4": [4, 5, 6, 7],
            }
        )

        result = rel.analyze(data)

        self.assertEqual(result.n_participants, 4)
        self.assertEqual(result.n_complete_cases, 3)


if __name__ == "__main__":
    unittest.main()
