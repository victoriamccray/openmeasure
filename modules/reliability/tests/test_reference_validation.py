"""
Regression tests pinning OpenMeasure's Reliability statistics to values
independently confirmed against R's psych::alpha() and a direct hand
formula (not OpenMeasure's own code) - see
docs/validation/reference-validation.md for the full cross-implementation
comparison, the qualitative purpose of each dataset, and how these
numbers were generated.

Unlike test_reliability.py's own regression tests, the expected values
here were NOT produced by calling OpenMeasure and pinning whatever it
returned. They were produced by an independent implementation
(scripts/validation/reliability_reference.R, R psych::alpha()) and a
from-scratch formula derivation
(scripts/validation/reliability_reference.py), then hardcoded here so
this comparison stays reproducible without requiring R or pingouin as
test-time dependencies.

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

from core import reliability as rel  # noqa: E402


# Dataset A: 3 items x 4 participants (docs/validation/reference-validation.md)
DATASET_A = pd.DataFrame(
    {
        "item1": [2, 4, 6, 4],
        "item2": [3, 5, 5, 3],
        "item3": [1, 3, 7, 5],
    },
    dtype=float,
)

# Dataset B: 4 items x 8 participants, same fixed data pinned in
# test_reliability.py::TestCronbachAlpha::test_known_value_regression.
DATASET_B = pd.DataFrame(
    {
        "q1": [4, 3, 5, 2, 4, 5, 3, 4],
        "q2": [5, 2, 5, 2, 4, 5, 3, 4],
        "q3": [4, 3, 4, 1, 5, 4, 2, 5],
        "q4": [5, 3, 5, 2, 4, 5, 3, 4],
    },
    dtype=float,
)

# Dataset C: high reliability. 5 items x 8 participants that move together
# strongly.
DATASET_C = pd.DataFrame(
    {
        "item1": [2, 3, 4, 5, 6, 7, 8, 9],
        "item2": [2, 3, 5, 5, 7, 7, 9, 9],
        "item3": [3, 4, 4, 6, 6, 8, 8, 10],
        "item4": [2, 4, 4, 5, 7, 7, 8, 10],
        "item5": [3, 3, 5, 6, 6, 8, 9, 9],
    },
    dtype=float,
)

# Dataset D: low/negative reliability. 4 items x 8 participants, each
# independently shuffled so items do not move together.
DATASET_D = pd.DataFrame(
    {
        "item1": [5, 3, 6, 2, 7, 1, 4, 8],
        "item2": [1, 7, 2, 8, 3, 6, 5, 4],
        "item3": [8, 2, 5, 3, 6, 1, 7, 4],
        "item4": [4, 8, 1, 7, 2, 5, 3, 6],
    },
    dtype=float,
)

# Dataset E: problematic/reverse item. item1-item4 move together; item5 is
# the reverse pattern of item1, not recoded before analysis.
DATASET_E = pd.DataFrame(
    {
        "item1": [2, 3, 4, 5, 6, 7, 8, 9],
        "item2": [2, 4, 4, 6, 6, 8, 8, 10],
        "item3": [3, 3, 5, 5, 7, 7, 9, 9],
        "item4": [2, 4, 5, 5, 7, 8, 8, 10],
        "item5": [9, 8, 7, 6, 5, 4, 3, 2],
    },
    dtype=float,
)

# Dataset F: missing data. 4 items x 10 participants; participant 8 is
# missing q3, participant 9 is missing q4.
DATASET_F = pd.DataFrame(
    {
        "q1": [4, 3, 5, 2, 4, 5, 3, 4, 6, 2],
        "q2": [5, 2, 5, 2, 4, 5, 3, 4, 6, 3],
        "q3": [4, 3, 4, 1, 5, 4, 2, np.nan, 6, 2],
        "q4": [5, 3, 5, 2, 4, 5, 3, 4, np.nan, 3],
    },
    dtype=float,
)

# Dataset G: small/edge case. The minimum allowed item count (2) with a
# small sample (5 participants).
DATASET_G = pd.DataFrame(
    {
        "item1": [1, 2, 3, 4, 5],
        "item2": [2, 3, 3, 5, 4],
    },
    dtype=float,
)


class TestCronbachAlphaAgainstReference(unittest.TestCase):
    def test_dataset_a_matches_r_psych_alpha(self) -> None:
        self.assertAlmostEqual(
            rel.cronbach_alpha(DATASET_A), 0.833333333333, places=9
        )

    def test_dataset_b_matches_r_psych_alpha(self) -> None:
        self.assertAlmostEqual(
            rel.cronbach_alpha(DATASET_B), 0.947943629566, places=9
        )

    def test_dataset_c_high_reliability_matches_r_psych_alpha(self) -> None:
        self.assertAlmostEqual(
            rel.cronbach_alpha(DATASET_C), 0.991071428571, places=9
        )

    def test_dataset_d_negative_reliability_matches_r_psych_alpha(self) -> None:
        """psych::alpha() computes and reports the same negative value;
        neither implementation clamps alpha to a minimum of 0."""
        self.assertAlmostEqual(
            rel.cronbach_alpha(DATASET_D), -2.060606060606, places=9
        )

    def test_dataset_e_reverse_item_matches_r_psych_alpha(self) -> None:
        self.assertAlmostEqual(
            rel.cronbach_alpha(DATASET_E), 0.546448087432, places=9
        )

    def test_dataset_f_listwise_deleted_matches_r_psych_complete_obs(self) -> None:
        """analyze()'s documented listwise deletion (2 of 10 rows
        excluded) matches R's use="complete.obs", not its use="pairwise"
        default - see docs/validation/reference-validation.md."""
        result = rel.analyze(DATASET_F)

        self.assertEqual(result.n_excluded_cases, 2)
        self.assertAlmostEqual(
            result.cronbach_alpha, 0.959915611814, places=9
        )

    def test_dataset_g_small_edge_case_matches_r_psych_alpha(self) -> None:
        self.assertAlmostEqual(
            rel.cronbach_alpha(DATASET_G), 0.882352941176, places=9
        )


class TestItemTotalCorrelationsAgainstReference(unittest.TestCase):
    def test_dataset_a_matches_r_psych_r_drop(self) -> None:
        expected = {
            "item1": 1.000000000000,
            "item2": 0.554700196225,
            "item3": 0.800000000000,
        }
        actual = rel.item_total_correlations(DATASET_A)

        for item, value in expected.items():
            self.assertAlmostEqual(actual[item], value, places=9)

    def test_dataset_b_matches_r_psych_r_drop(self) -> None:
        expected = {
            "q1": 0.939271844935,
            "q2": 0.904790225656,
            "q3": 0.762554216828,
            "q4": 0.952246804771,
        }
        actual = rel.item_total_correlations(DATASET_B)

        for item, value in expected.items():
            self.assertAlmostEqual(actual[item], value, places=9)

    def test_dataset_c_high_reliability_matches_r_psych_r_drop(self) -> None:
        expected = {
            "item1": 1.000000000000,
            "item2": 0.968688257650,
            "item3": 0.965288580493,
            "item4": 0.968688257650,
            "item5": 0.965288580493,
        }
        actual = rel.item_total_correlations(DATASET_C)

        for item, value in expected.items():
            self.assertAlmostEqual(actual[item], value, places=9)

    def test_dataset_d_negative_reliability_matches_r_psych_r_drop(self) -> None:
        expected = {
            "item1": -0.448435002084,
            "item2": -0.494158678028,
            "item3": -0.546966034275,
            "item4": -0.299572344758,
        }
        actual = rel.item_total_correlations(DATASET_D)

        for item, value in expected.items():
            self.assertAlmostEqual(actual[item], value, places=9)

    def test_dataset_e_reverse_item_matches_r_psych_r_drop(self) -> None:
        """item5's strongly negative item-total correlation is the
        intended signal this dataset is constructed to surface."""
        expected = {
            "item1": 0.994808332616,
            "item2": 0.955201842586,
            "item3": 0.936870189785,
            "item4": 0.980378935485,
            "item5": -0.998671303489,
        }
        actual = rel.item_total_correlations(DATASET_E)

        for item, value in expected.items():
            self.assertAlmostEqual(actual[item], value, places=9)

    def test_dataset_g_small_edge_case_matches_r_psych_r_drop(self) -> None:
        expected = {
            "item1": 0.832050294338,
            "item2": 0.832050294338,
        }
        actual = rel.item_total_correlations(DATASET_G)

        for item, value in expected.items():
            self.assertAlmostEqual(actual[item], value, places=9)


class TestAlphaIfItemDroppedAgainstReference(unittest.TestCase):
    def test_dataset_a_matches_r_psych_alpha_drop(self) -> None:
        expected = {
            "item1": 0.500000000000,
            "item2": 0.923076923077,
            "item3": 0.800000000000,
        }
        actual = rel.alpha_if_item_dropped(DATASET_A)

        for item, value in expected.items():
            self.assertAlmostEqual(actual[item], value, places=9)

    def test_dataset_b_matches_r_psych_alpha_drop(self) -> None:
        expected = {
            "q1": 0.919831223629,
            "q2": 0.922345483360,
            "q3": 0.976525821596,
            "q4": 0.910714285714,
        }
        actual = rel.alpha_if_item_dropped(DATASET_B)

        for item, value in expected.items():
            self.assertAlmostEqual(actual[item], value, places=9)

    def test_dataset_c_high_reliability_matches_r_psych_alpha_drop(self) -> None:
        expected = {
            "item1": 0.985119047619,
            "item2": 0.989612842304,
            "item3": 0.989976307636,
            "item4": 0.989612842304,
            "item5": 0.989976307636,
        }
        actual = rel.alpha_if_item_dropped(DATASET_C)

        for item, value in expected.items():
            self.assertAlmostEqual(actual[item], value, places=9)

    def test_dataset_d_negative_reliability_matches_r_psych_alpha_drop(self) -> None:
        expected = {
            "item1": -1.054054054054,
            "item2": -0.804878048780,
            "item3": -0.554347826087,
            "item4": -2.134615384615,
        }
        actual = rel.alpha_if_item_dropped(DATASET_D)

        for item, value in expected.items():
            self.assertAlmostEqual(actual[item], value, places=9)

    def test_dataset_e_reverse_item_recovers_alpha_when_dropped(self) -> None:
        """Dropping item5 (the reverse item) raises alpha from 0.546 to
        0.989 - the textbook signature of a single un-recoded reverse
        item, confirmed against R psych::alpha()'s alpha.drop table."""
        expected = {
            "item1": -0.007581141910,
            "item2": -0.015790772267,
            "item3": 0.057079152731,
            "item4": -0.031746031746,
            "item5": 0.989490310431,
        }
        actual = rel.alpha_if_item_dropped(DATASET_E)

        for item, value in expected.items():
            self.assertAlmostEqual(actual[item], value, places=9)

    def test_dataset_g_alpha_if_dropped_intentionally_diverges_from_r_psych(
        self,
    ) -> None:
        """Investigated discrepancy (see
        docs/validation/reference-validation.md, "Requiring human
        review"): dropping either item from this 2-item scale leaves a
        1-item "scale", for which raw alpha's k/(k-1) term is 1/0 -
        mathematically undefined. OpenMeasure returns NaN for this case
        by design (see test_reliability.py's
        test_two_item_scale_returns_nan_after_dropping). R's
        psych::alpha() does not special-case it and reports finite but
        internally inconsistent numbers instead (raw_alpha > 1, which is
        not a valid value for a real reliability coefficient); calling
        psych::alpha() directly on a genuine single-column dataset
        produces a warning rather than a usable result, confirming those
        alpha.drop numbers are an artifact, not a meaningful reference
        value. OpenMeasure's NaN is retained; this test pins that choice
        rather than R's output.
        """
        actual = rel.alpha_if_item_dropped(DATASET_G)

        self.assertTrue(np.isnan(actual["item1"]))
        self.assertTrue(np.isnan(actual["item2"]))


class TestSplitHalfAgainstReference(unittest.TestCase):
    def test_dataset_b_matches_independently_computed_r_split(self) -> None:
        """R script reproduces OpenMeasure's exact odd/even positional
        split independently (not psych::splitHalf()'s random split),
        so this compares the same defined quantity."""
        correlation, spearman_brown = rel.split_half_reliability(DATASET_B)

        self.assertAlmostEqual(correlation, 0.872786048178, places=9)
        self.assertAlmostEqual(spearman_brown, 0.932072351806, places=9)

    def test_dataset_c_high_reliability_matches_independently_computed_r_split(
        self,
    ) -> None:
        correlation, spearman_brown = rel.split_half_reliability(DATASET_C)

        self.assertAlmostEqual(correlation, 0.973795464575, places=9)
        self.assertAlmostEqual(spearman_brown, 0.986723783748, places=9)

    def test_dataset_d_negative_reliability_matches_independently_computed_r_split(
        self,
    ) -> None:
        correlation, spearman_brown = rel.split_half_reliability(DATASET_D)

        self.assertAlmostEqual(correlation, -0.762151300496, places=9)
        self.assertAlmostEqual(spearman_brown, -6.408706897148, places=9)

    def test_dataset_e_reverse_item_matches_independently_computed_r_split(
        self,
    ) -> None:
        correlation, spearman_brown = rel.split_half_reliability(DATASET_E)

        self.assertAlmostEqual(correlation, 0.936870189785, places=9)
        self.assertAlmostEqual(spearman_brown, 0.967406277123, places=9)


if __name__ == "__main__":
    unittest.main()
