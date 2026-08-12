"""
Unit tests for core/interrater.py

Run with:
    pytest tests/
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

from core import interrater as ir  # noqa: E402


class TestCohensKappa(unittest.TestCase):
    def setUp(self) -> None:
        # Hand-calculable: 5 items.
        # Confusion matrix (rows=A, cols=B), categories [No, Yes]:
        #   No/No=2, No/Yes=0, Yes/No=1, Yes/Yes=2
        # Po = (2+2)/5 = 0.8
        # marginals: A: No=2 Yes=3; B: No=3 Yes=2
        # Pe = (2/5*3/5) + (3/5*2/5) = 0.48
        # kappa = (0.8-0.48)/(1-0.48) = 0.32/0.52 = 0.615384...
        self.data = pd.DataFrame(
            {
                "rater_a": ["Yes", "Yes", "No", "No", "Yes"],
                "rater_b": ["Yes", "No", "No", "No", "Yes"],
            }
        )

    def test_known_value(self) -> None:
        result = ir.cohens_kappa(self.data, "rater_a", "rater_b")

        self.assertEqual(result.n_items, 5)
        self.assertAlmostEqual(result.kappa, 0.32 / 0.52, places=6)

    def test_missing_ratings_are_excluded_before_pairing(self) -> None:
        data = self.data.copy()
        data.loc[0, "rater_b"] = None

        result = ir.cohens_kappa(data, "rater_a", "rater_b")

        self.assertEqual(result.n_items, 4)

    def test_unknown_rater_column_raises(self) -> None:
        with self.assertRaises(ValueError):
            ir.cohens_kappa(self.data, "rater_a", "not_a_column")

    def test_identical_rater_columns_raises(self) -> None:
        with self.assertRaises(ValueError):
            ir.cohens_kappa(self.data, "rater_a", "rater_a")

    def test_too_few_shared_items_raises(self) -> None:
        data = pd.DataFrame({"rater_a": ["Yes", None], "rater_b": [None, "No"]})

        with self.assertRaises(ValueError):
            ir.cohens_kappa(data, "rater_a", "rater_b")


class TestFleissKappa(unittest.TestCase):
    def test_known_value(self) -> None:
        # Hand-calculable: 4 subjects, 3 raters, categories {A, B}.
        # P_i = [sum_j n_ij(n_ij-1)] / [m(m-1)], m=3:
        #   subj1 (A,A,A): n_A=3,n_B=0 -> (3*2+0)/6 = 1.0
        #   subj2 (A,A,B): n_A=2,n_B=1 -> (2*1+0)/6 = 0.33333
        #   subj3 (B,B,B): n_A=0,n_B=3 -> (0+3*2)/6 = 1.0
        #   subj4 (A,B,B): n_A=1,n_B=2 -> (0+2*1)/6 = 0.33333
        # P_bar = mean = 0.666667
        # p_A = 6/12 = 0.5, p_B = 6/12 = 0.5 -> P_bar_e = 0.25+0.25 = 0.5
        # kappa = (0.666667-0.5)/(1-0.5) = 0.333333
        data = pd.DataFrame(
            [
                ["A", "A", "A"],
                ["A", "A", "B"],
                ["B", "B", "B"],
                ["A", "B", "B"],
            ],
            columns=["rater1", "rater2", "rater3"],
        )

        result = ir.fleiss_kappa(data)

        self.assertEqual(result.n_items, 4)
        self.assertEqual(result.n_raters, 3)
        self.assertAlmostEqual(result.kappa, 1 / 3, places=6)

    def test_unanimous_agreement_gives_kappa_one(self) -> None:
        data = pd.DataFrame(
            [["A", "A", "A"], ["B", "B", "B"], ["A", "A", "A"]],
            columns=["r1", "r2", "r3"],
        )

        result = ir.fleiss_kappa(data)

        self.assertAlmostEqual(result.kappa, 1.0, places=6)

    def test_fewer_than_three_raters_raises(self) -> None:
        data = pd.DataFrame({"r1": ["A", "B"], "r2": ["A", "B"]})

        with self.assertRaises(ValueError):
            ir.fleiss_kappa(data)

    def test_missing_values_raise(self) -> None:
        data = pd.DataFrame(
            [["A", "A", "A"], ["A", None, "B"]],
            columns=["r1", "r2", "r3"],
        )

        with self.assertRaises(ValueError) as ctx:
            ir.fleiss_kappa(data)

        self.assertIn("every item to be rated by every rater", str(ctx.exception))


class TestKrippendorffAlpha(unittest.TestCase):
    def test_known_value_from_package_docstring_example(self) -> None:
        # From the krippendorff package's own documented example
        # (nominal level of measurement), reproduced here transposed
        # into this module's (units x raters) input shape. Expected
        # alpha = 0.691358.
        raters_by_units = [
            [np.nan, np.nan, np.nan, np.nan, np.nan, 3, 4, 1, 2, 1, 1, 3, 3, np.nan, 3],
            [1, np.nan, 2, 1, 3, 3, 4, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 2, 1, 3, 4, 4, np.nan, 2, 1, 1, 3, 3, np.nan, 4],
        ]
        data = pd.DataFrame(
            np.array(raters_by_units, dtype=object).T,
            columns=["rater1", "rater2", "rater3"],
        )

        result = ir.krippendorff_alpha(data)

        self.assertEqual(result.n_items, 15)
        self.assertEqual(result.n_raters, 3)
        self.assertAlmostEqual(result.alpha, 0.691358, places=6)

    def test_unanimous_agreement_gives_alpha_one(self) -> None:
        data = pd.DataFrame(
            [["A", "A", None], ["B", "B", "B"], ["A", None, "A"]],
            columns=["r1", "r2", "r3"],
        )

        result = ir.krippendorff_alpha(data)

        self.assertAlmostEqual(result.alpha, 1.0, places=6)

    def test_tolerates_partial_rater_coverage(self) -> None:
        # Unlike fleiss_kappa, missing values must not raise here.
        data = pd.DataFrame(
            [["A", "A", None], ["B", "B", "B"], ["A", None, "A"]],
            columns=["r1", "r2", "r3"],
        )

        result = ir.krippendorff_alpha(data)

        self.assertIsInstance(result.alpha, float)

    def test_fewer_than_two_raters_raises(self) -> None:
        data = pd.DataFrame({"r1": ["A", "B"]})

        with self.assertRaises(ValueError):
            ir.krippendorff_alpha(data)

    def test_too_few_pairable_items_raises(self) -> None:
        data = pd.DataFrame(
            [["A", None], [None, "B"]],
            columns=["r1", "r2"],
        )

        with self.assertRaises(ValueError):
            ir.krippendorff_alpha(data)


if __name__ == "__main__":
    unittest.main()
