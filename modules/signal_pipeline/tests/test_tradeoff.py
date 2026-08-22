"""
Unit tests for core/tradeoff.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import modality as mod  # noqa: E402
from core import tradeoff as t  # noqa: E402


def _modality(name, interpretive_gain, privacy_cost, security_cost, agency_cost):
    return mod.Modality(
        name=name,
        category="Neural",
        signal_examples="example signal",
        interpretive_gain=interpretive_gain,
        privacy_cost=privacy_cost,
        security_cost=security_cost,
        agency_cost=agency_cost,
        citation="Example citation (2026).",
        is_body_sensed=True,
        body_region="Example region",
        privacy_exit_point="Example privacy exit point.",
        agency_control_point="Example agency control point.",
    )


class TestCombineCosts(unittest.TestCase):
    def setUp(self):
        self.a = _modality(
            "A", interpretive_gain=0.5, privacy_cost=1.0, security_cost=0.0, agency_cost=0.0
        )

    def test_raises_on_empty_modalities(self):
        with self.assertRaises(ValueError):
            t.combine_costs((), 1.0, 1.0, 1.0)

    def test_raises_on_negative_weight(self):
        with self.assertRaises(ValueError):
            t.combine_costs((self.a,), -1.0, 1.0, 1.0)

    def test_raises_when_all_weights_are_zero(self):
        with self.assertRaises(ValueError):
            t.combine_costs((self.a,), 0.0, 0.0, 0.0)

    def test_pure_privacy_weight_matches_hand_calculation(self):
        result = t.combine_costs((self.a,), 1.0, 0.0, 0.0)
        self.assertAlmostEqual(result.combined_cost["A"], 1.0)
        self.assertAlmostEqual(result.privacy_weight, 1.0)
        self.assertAlmostEqual(result.security_weight, 0.0)

    def test_unnormalized_weights_are_normalized_to_sum_to_one(self):
        # Raw weights (2, 2, 0) normalize to (0.5, 0.5, 0.0); with
        # privacy_cost=1.0 and security_cost=0.0 the combined cost is
        # 0.5*1.0 + 0.5*0.0 = 0.5.
        result = t.combine_costs((self.a,), 2.0, 2.0, 0.0)
        self.assertAlmostEqual(result.privacy_weight, 0.5)
        self.assertAlmostEqual(result.security_weight, 0.5)
        self.assertAlmostEqual(result.combined_cost["A"], 0.5)


class TestComputeGainCostFrontier(unittest.TestCase):
    def test_dominated_modality_is_flagged_inefficient(self):
        # A gains more and costs less than B on every dimension, so B is
        # strictly dominated. C gains less than A but costs less too, so
        # neither A nor C dominates the other.
        a = _modality(
            "A", interpretive_gain=0.8, privacy_cost=0.2, security_cost=0.2, agency_cost=0.2
        )
        b = _modality(
            "B", interpretive_gain=0.5, privacy_cost=0.6, security_cost=0.6, agency_cost=0.6
        )
        c = _modality(
            "C", interpretive_gain=0.3, privacy_cost=0.1, security_cost=0.1, agency_cost=0.1
        )
        modalities = (a, b, c)
        weighted_cost = t.combine_costs(modalities, 1.0, 1.0, 1.0)

        result = t.compute_gain_cost_frontier(modalities, weighted_cost)

        self.assertEqual(
            result.is_efficient, {"A": True, "B": False, "C": True}
        )

    def test_raises_on_empty_modalities(self):
        with self.assertRaises(ValueError):
            t.compute_gain_cost_frontier((), t.WeightedCostResult({}, 1.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
