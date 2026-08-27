"""
Regression tests pinning OpenMeasure's Reliability statistics to values
independently confirmed against R's psych::alpha() and a direct hand
formula (not OpenMeasure's own code) - see
docs/validation/reference-validation.md for the full cross-implementation
comparison and how these numbers were generated.

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


class TestCronbachAlphaAgainstReference(unittest.TestCase):
    def test_dataset_a_matches_r_psych_alpha(self) -> None:
        self.assertAlmostEqual(
            rel.cronbach_alpha(DATASET_A), 0.833333333333, places=9
        )

    def test_dataset_b_matches_r_psych_alpha(self) -> None:
        self.assertAlmostEqual(
            rel.cronbach_alpha(DATASET_B), 0.947943629566, places=9
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


class TestSplitHalfAgainstReference(unittest.TestCase):
    def test_dataset_b_matches_independently_computed_r_split(self) -> None:
        """R script reproduces OpenMeasure's exact odd/even positional
        split independently (not psych::splitHalf()'s random split),
        so this compares the same defined quantity."""
        correlation, spearman_brown = rel.split_half_reliability(DATASET_B)

        self.assertAlmostEqual(correlation, 0.872786048178, places=9)
        self.assertAlmostEqual(spearman_brown, 0.932072351806, places=9)


if __name__ == "__main__":
    unittest.main()
