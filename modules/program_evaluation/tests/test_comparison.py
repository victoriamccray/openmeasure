"""
Unit tests for core/comparison.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import comparison as pe  # noqa: E402


class TestCompareTwoGroups(unittest.TestCase):
    def test_matches_scipy_welch_ttest(self):
        rng = np.random.RandomState(0)
        group_a = rng.normal(50, 10, 30)
        group_b = rng.normal(55, 12, 25)
        data = pd.DataFrame({
            "group": ["A"] * 30 + ["B"] * 25,
            "score": list(group_a) + list(group_b),
        })

        result = pe.compare_two_groups(data, "group", "score")
        t_expected, p_expected = scipy_stats.ttest_ind(group_a, group_b, equal_var=False)

        self.assertAlmostEqual(result.t_statistic, t_expected, places=9)
        self.assertAlmostEqual(result.p_value, p_expected, places=9)

    def test_raises_on_more_than_two_groups(self):
        data = pd.DataFrame({
            "group": ["A", "B", "C", "A", "B", "C"],
            "score": [1, 2, 3, 4, 5, 6],
        })
        with self.assertRaises(ValueError):
            pe.compare_two_groups(data, "group", "score")

    def test_cohens_d_sign_reflects_direction(self):
        data = pd.DataFrame({
            "group": ["A"] * 10 + ["B"] * 10,
            "score": [10] * 10 + [5] * 10,
        })
        # add tiny noise so variance isn't exactly zero
        data["score"] = data["score"] + np.array([0.01 * i for i in range(20)])
        result = pe.compare_two_groups(data, "group", "score")
        self.assertGreater(result.cohens_d, 0)  # A > B


class TestCompareMultipleGroups(unittest.TestCase):
    def test_f_statistic_matches_scipy(self):
        rng = np.random.RandomState(1)
        a = rng.normal(4.5, 0.5, 20)
        b = rng.normal(4.0, 0.6, 18)
        c = rng.normal(4.8, 0.4, 22)
        data = pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 18 + ["C"] * 22,
            "score": list(a) + list(b) + list(c),
        })

        f_expected, p_expected = scipy_stats.f_oneway(a, b, c)

        try:
            result = pe.compare_multiple_groups(data, "group", "score")
        except ImportError:
            self.skipTest("statsmodels not installed")
            return

        self.assertAlmostEqual(result.f_statistic, f_expected, places=6)
        self.assertAlmostEqual(result.p_value, p_expected, places=6)

    def test_raises_on_fewer_than_three_groups(self):
        data = pd.DataFrame({
            "group": ["A", "B", "A", "B"],
            "score": [1, 2, 3, 4],
        })
        with self.assertRaises(ValueError):
            pe.compare_multiple_groups(data, "group", "score")

    def test_flags_small_groups(self):
        rng = np.random.RandomState(2)
        data = pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 3,
            "score": list(rng.normal(4, 1, 43)),
        })
        try:
            result = pe.compare_multiple_groups(data, "group", "score")
        except ImportError:
            self.skipTest("statsmodels not installed")
            return
        self.assertIn("C", result.small_groups_flagged)


class TestCompareCategorical(unittest.TestCase):
    def test_matches_scipy_chi_square(self):
        data = pd.DataFrame({
            "group": ["A"] * 50 + ["B"] * 50,
            "outcome": ["yes"] * 30 + ["no"] * 20 + ["yes"] * 20 + ["no"] * 30,
        })
        result = pe.compare_categorical(data, "group", "outcome")

        table = pd.crosstab(data["group"], data["outcome"])
        chi2_expected, p_expected, _, _ = scipy_stats.chi2_contingency(table)

        self.assertAlmostEqual(result.chi2_statistic, chi2_expected, places=9)
        self.assertAlmostEqual(result.p_value, p_expected, places=9)

    def test_flags_low_expected_frequency(self):
        data = pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 2,
            "outcome": ["yes"] * 19 + ["no"] * 1 + ["yes"] * 1 + ["no"] * 1,
        })
        result = pe.compare_categorical(data, "group", "outcome")
        self.assertTrue(result.low_expected_frequency_warning)


class TestComparePrePost(unittest.TestCase):
    def test_matches_scipy_paired_ttest(self):
        rng = np.random.RandomState(3)
        pre = pd.Series(rng.normal(3.5, 0.8, 40))
        post = pre + pd.Series(rng.normal(0.4, 0.5, 40))

        result = pe.compare_pre_post(pre, post)
        t_expected, p_expected = scipy_stats.ttest_rel(post, pre)

        self.assertAlmostEqual(result.t_statistic, t_expected, places=9)
        self.assertAlmostEqual(result.p_value, p_expected, places=9)

    def test_raises_on_mismatched_lengths(self):
        pre = pd.Series([1, 2, 3])
        post = pd.Series([1, 2])
        with self.assertRaises(ValueError):
            pe.compare_pre_post(pre, post)

    def test_raises_on_identical_pre_post(self):
        pre = pd.Series([3, 3, 3, 3])
        post = pd.Series([3, 3, 3, 3])
        with self.assertRaises(ValueError):
            pe.compare_pre_post(pre, post)


class TestMultiselectCoding(unittest.TestCase):
    def test_expand_multiselect_creates_one_row_per_category(self):
        data = pd.DataFrame({
            "race": ["A", "A, B", "C"],
            "score": [1.0, 2.0, 3.0],
        })
        expanded = pe.expand_multiselect(data, "race")
        self.assertEqual(len(expanded), 4)  # A, A, B, C
        self.assertEqual(sorted(expanded["race"].tolist()), ["A", "A", "B", "C"])

    def test_single_select_keeps_first_category(self):
        data = pd.DataFrame({
            "race": ["A", "A, B", "C"],
            "score": [1.0, 2.0, 3.0],
        })
        single = pe.single_select_multiselect(data, "race")
        self.assertEqual(single["race"].tolist(), ["A", "A", "C"])

    def test_combined_category_relabels_multi_selections(self):
        data = pd.DataFrame({
            "race": ["A", "A, B", "C"],
            "score": [1.0, 2.0, 3.0],
        })
        combined = pe.combined_category_multiselect(data, "race")
        self.assertEqual(
            combined["race"].tolist(),
            ["A", "Multiple categories", "C"],
        )


class TestSensitivityAnalysis(unittest.TestCase):
    def test_runs_all_three_codings(self):
        rng = np.random.RandomState(4)
        data = pd.DataFrame({
            "race": (
                ["A"] * 15 + ["B"] * 15 + ["C"] * 15
                + ["A, B"] * 3
            ),
            "score": list(rng.normal(4, 0.5, 48)),
        })
        try:
            result = pe.sensitivity_analysis(data, "race", "score")
        except ImportError:
            self.skipTest("statsmodels not installed")
            return

        self.assertEqual(
            set(result.coding_results.keys()),
            {"expanded", "single_selection", "combined_category"},
        )
        self.assertEqual(len(result.p_values_by_coding), 3)
        self.assertIsInstance(result.consistent_conclusion, bool)


class TestCompareMultipleGroupsWelch(unittest.TestCase):
    def test_f_statistic_and_df_match_manual_welch_formula(self):
        """Cross-check the F-statistic and fractional df against an
        independently written implementation of Welch's ANOVA (1951),
        not just against the module's own internal consistency."""
        rng = np.random.RandomState(0)
        a = rng.normal(4.0, 0.5, 25)
        b = rng.normal(4.5, 1.2, 18)
        c = rng.normal(5.0, 0.3, 22)
        data = pd.DataFrame({
            "group": ["A"] * 25 + ["B"] * 18 + ["C"] * 22,
            "score": list(a) + list(b) + list(c),
        })

        def welch_anova_manual(groups):
            k = len(groups)
            ns = np.array([len(g) for g in groups], dtype=float)
            means = np.array([np.mean(g) for g in groups])
            variances = np.array([np.var(g, ddof=1) for g in groups])
            weights = ns / variances
            sum_w = weights.sum()
            grand_mean = (weights * means).sum() / sum_w
            numerator = (weights * (means - grand_mean) ** 2).sum() / (k - 1)
            term = ((1 - weights / sum_w) ** 2 / (ns - 1)).sum()
            denominator = 1 + (2 * (k - 2) / (k ** 2 - 1)) * term
            F = numerator / denominator
            df2 = (k ** 2 - 1) / (3 * term)
            p = scipy_stats.f.sf(F, k - 1, df2)
            return F, k - 1, df2, p

        F_expected, df1_expected, df2_expected, p_expected = welch_anova_manual([a, b, c])

        try:
            result = pe.compare_multiple_groups_welch(data, "group", "score")
        except ImportError:
            self.skipTest("statsmodels not installed")
            return

        self.assertAlmostEqual(result.f_statistic, F_expected, places=9)
        self.assertAlmostEqual(result.df_between, df1_expected, places=9)
        self.assertAlmostEqual(result.df_within, df2_expected, places=9)
        self.assertAlmostEqual(result.p_value, p_expected, places=9)

    def test_raises_on_fewer_than_three_groups(self):
        data = pd.DataFrame({"group": ["A", "B"], "score": [1, 2]})
        with self.assertRaises(ValueError):
            pe.compare_multiple_groups_welch(data, "group", "score")

    def test_raises_on_zero_variance_group(self):
        data = pd.DataFrame({
            "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "score": [3, 3, 3, 3, 3] + [1, 2, 3, 4, 5] + [2, 3, 4, 5, 6],
        })
        with self.assertRaises(ValueError):
            pe.compare_multiple_groups_welch(data, "group", "score")

    def test_produces_correct_number_of_pairwise_comparisons(self):
        rng = np.random.RandomState(1)
        data = pd.DataFrame({
            "group": ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10,
            "score": list(rng.normal(4, 1, 40)),
        })
        try:
            result = pe.compare_multiple_groups_welch(data, "group", "score")
        except ImportError:
            self.skipTest("statsmodels not installed")
            return
        # 4 groups -> C(4,2) = 6 pairwise comparisons
        self.assertEqual(len(result.pairwise_comparisons), 6)

    def test_flags_small_groups(self):
        rng = np.random.RandomState(2)
        data = pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 3,
            "score": list(rng.normal(4, 1, 43)),
        })
        try:
            result = pe.compare_multiple_groups_welch(data, "group", "score")
        except ImportError:
            self.skipTest("statsmodels not installed")
            return
        self.assertIn("C", result.small_groups_flagged)

    def test_pairwise_p_values_are_between_zero_and_one(self):
        """Sanity check on the actual psturng lookup, not just the
        surrounding arithmetic: every p-value should be a valid probability."""
        rng = np.random.RandomState(3)
        data = pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "score": list(rng.normal(4, 0.5, 20)) + list(rng.normal(5, 1.5, 20)) + list(rng.normal(4.2, 0.3, 20)),
        })
        try:
            result = pe.compare_multiple_groups_welch(data, "group", "score")
        except ImportError:
            self.skipTest("statsmodels not installed")
            return
        for p in result.pairwise_comparisons:
            self.assertGreaterEqual(p.p_value, 0.0)
            self.assertLessEqual(p.p_value, 1.0)


if __name__ == "__main__":
    unittest.main()
