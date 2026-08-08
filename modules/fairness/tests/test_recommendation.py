"""
Unit tests for core/recommend.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import recommend as gr  # noqa: E402


class TestFairnessGoals(unittest.TestCase):
    def test_exactly_four_goals_defined(self):
        self.assertEqual(len(gr.FAIRNESS_GOALS), 4)

    def test_all_goal_keys_present(self):
        expected = {
            "opportunity_access",
            "avoid_missed_need",
            "avoid_wrong_flags",
            "comparable_risk_scores",
        }
        self.assertEqual(set(gr.FAIRNESS_GOALS.keys()), expected)

    def test_all_metrics_are_distinct(self):
        metrics = [rec.metric for rec in gr.FAIRNESS_GOALS.values()]
        self.assertEqual(len(metrics), len(set(metrics)))

    def test_every_recommendation_has_nonempty_reasoning(self):
        for goal, rec in gr.FAIRNESS_GOALS.items():
            self.assertGreater(len(rec.reasoning), 0, f"{goal} has no reasoning")

    def test_every_recommendation_has_at_least_one_tradeoff(self):
        """Every goal's recommendation should name at least one limitation,
        since no fairness metric is a complete guarantee on its own."""
        for goal, rec in gr.FAIRNESS_GOALS.items():
            self.assertGreater(len(rec.tradeoffs), 0, f"{goal} has no tradeoffs listed")

    def test_every_recommendation_has_at_least_one_alternative(self):
        for goal, rec in gr.FAIRNESS_GOALS.items():
            self.assertGreater(len(rec.alternatives), 0, f"{goal} has no alternatives listed")


class TestRecommendFairnessMetric(unittest.TestCase):
    def test_opportunity_access_maps_to_demographic_parity(self):
        rec = gr.recommend_fairness_metric("opportunity_access")
        self.assertEqual(rec.metric, "demographic_parity")

    def test_avoid_missed_need_maps_to_equal_opportunity(self):
        rec = gr.recommend_fairness_metric("avoid_missed_need")
        self.assertEqual(rec.metric, "equal_opportunity")

    def test_avoid_wrong_flags_maps_to_predictive_equality(self):
        rec = gr.recommend_fairness_metric("avoid_wrong_flags")
        self.assertEqual(rec.metric, "predictive_equality")

    def test_comparable_risk_scores_maps_to_calibration(self):
        rec = gr.recommend_fairness_metric("comparable_risk_scores")
        self.assertEqual(rec.metric, "calibration")

    def test_raises_on_unknown_goal(self):
        with self.assertRaises(ValueError):
            gr.recommend_fairness_metric("not_a_real_goal")

    def test_error_message_lists_valid_goals(self):
        try:
            gr.recommend_fairness_metric("not_a_real_goal")
            self.fail("Expected ValueError")
        except ValueError as e:
            for goal in gr.FAIRNESS_GOALS:
                self.assertIn(goal, str(e))

    def test_every_recommendation_has_applicable_domains(self):
        for goal, rec in gr.FAIRNESS_GOALS.items():
            self.assertGreater(
                len(rec.applicable_domains), 0, f"{goal} has no applicable_domains"
            )

    def test_every_domain_context_has_a_domain_and_relevance(self):
        for goal, rec in gr.FAIRNESS_GOALS.items():
            for context in rec.applicable_domains:
                with self.subTest(goal=goal, domain=context.domain):
                    self.assertTrue(context.domain)
                    self.assertTrue(context.relevance)

    def test_domain_context_is_frozen(self):
        self.assertTrue(gr.DomainContext.__dataclass_params__.frozen)

    def test_no_domain_example_reads_as_a_universal_prescription(self):
        """
        A domain example should show where a goal can matter, not claim the
        domain always requires this goal's metric. Mirrors the phrasing
        guard in shared/case_studies.py's TestNoPositionalReferences.
        """
        forbidden = ("always use", "must use", "the only", "requires this metric")

        for goal, rec in gr.FAIRNESS_GOALS.items():
            for context in rec.applicable_domains:
                combined = context.relevance.lower()
                for phrase in forbidden:
                    with self.subTest(goal=goal, domain=context.domain, phrase=phrase):
                        self.assertNotIn(phrase, combined)

    def test_calibration_tradeoff_mentions_base_rate_conflict(self):
        """This is the specific incompatibility proven by Kleinberg,
        Mullainathan, & Raghavan (2017) and Chouldechova (2017); confirm
        the tradeoff text actually names the base-rate condition, not
        just a vague caveat."""
        rec = gr.recommend_fairness_metric("comparable_risk_scores")
        combined_tradeoffs = " ".join(rec.tradeoffs).lower()
        self.assertIn("base rate", combined_tradeoffs)


if __name__ == "__main__":
    unittest.main()
