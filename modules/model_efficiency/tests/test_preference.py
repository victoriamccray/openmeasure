"""
Unit tests for core/preference.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import models as m  # noqa: E402
from core import preference as p  # noqa: E402


def _profile(name, performance_value, resource_value):
    return m.ModelProfile(
        name=name,
        performance_metric_name="MSE",
        performance_value=performance_value,
        performance_is_approximate=False,
        n_parameters=1000,
        resource_metric_name="CO2 (kg)",
        resource_value=resource_value,
        resource_is_approximate=False,
    )


class TestRankByPreference(unittest.TestCase):
    def setUp(self):
        # A has the best (lowest) performance value, worst resource value.
        # B has the best resource value, worst performance value. Min-max
        # normalization over just these two profiles puts each at exactly
        # 0 or 1 on each axis, making the weighted sum hand-checkable.
        self.a = _profile("A", performance_value=0, resource_value=10)
        self.b = _profile("B", performance_value=10, resource_value=0)

    def test_pure_resource_weight_favors_the_better_resource_model(self):
        result = p.rank_by_preference((self.a, self.b), performance_weight=0.0)
        self.assertAlmostEqual(result.scores["A"], 1.0)
        self.assertAlmostEqual(result.scores["B"], 0.0)
        self.assertEqual(result.favored_by_weights, "B")

    def test_pure_performance_weight_favors_the_better_performance_model(self):
        result = p.rank_by_preference((self.a, self.b), performance_weight=1.0)
        self.assertAlmostEqual(result.scores["A"], 0.0)
        self.assertAlmostEqual(result.scores["B"], 1.0)
        self.assertEqual(result.favored_by_weights, "A")

    def test_intermediate_weight_matches_hand_calculation(self):
        # weight=0.3: score_A = 0.3*0 + 0.7*1 = 0.7; score_B = 0.3*1 + 0.7*0 = 0.3
        result = p.rank_by_preference((self.a, self.b), performance_weight=0.3)
        self.assertAlmostEqual(result.scores["A"], 0.7)
        self.assertAlmostEqual(result.scores["B"], 0.3)
        self.assertEqual(result.favored_by_weights, "B")

    def test_result_never_uses_the_word_preferred_as_a_field(self):
        result = p.rank_by_preference((self.a, self.b), performance_weight=0.5)
        self.assertFalse(hasattr(result, "preferred"))
        self.assertTrue(hasattr(result, "favored_by_weights"))

    def test_raises_on_out_of_range_weight(self):
        with self.assertRaises(ValueError):
            p.rank_by_preference((self.a, self.b), performance_weight=1.5)

    def test_raises_on_empty_profiles(self):
        with self.assertRaises(ValueError):
            p.rank_by_preference((), performance_weight=0.5)

    def test_raises_when_performance_values_are_identical(self):
        tied = _profile("C", performance_value=5, resource_value=1)
        other = _profile("D", performance_value=5, resource_value=2)
        with self.assertRaises(ValueError):
            p.rank_by_preference((tied, other), performance_weight=0.5)


if __name__ == "__main__":
    unittest.main()
