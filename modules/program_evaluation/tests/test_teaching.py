"""
Unit tests for core/teaching.py

The scenario's arithmetic is deliberately simple, so these tests pin the
exact numbers a reader is shown. If the scenario's fixed values are ever
edited, these fail rather than the teaching text quietly describing
different numbers than the ones on screen.

Run with: pytest modules/program_evaluation/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import teaching  # noqa: E402


class TestScenarioContent(unittest.TestCase):
    def test_the_treated_group_rises_twelve_points(self):
        scenario = teaching.DID_TEACHING_SCENARIO

        self.assertAlmostEqual(scenario.pre_treated, 60.0, places=12)
        self.assertAlmostEqual(scenario.post_treated, 72.0, places=12)
        self.assertAlmostEqual(scenario.change_treated, 12.0, places=12)

    def test_every_narrative_field_is_populated(self):
        scenario = teaching.DID_TEACHING_SCENARIO

        for field_name in ("scenario", "question", "method_fit"):
            with self.subTest(field=field_name):
                self.assertTrue(getattr(scenario, field_name).strip())

        for field_name in ("assumptions", "can_conclude", "cannot_conclude"):
            with self.subTest(field=field_name):
                self.assertTrue(getattr(scenario, field_name))

    def test_the_scenario_states_the_twelve_point_rise_it_computes(self):
        """
        The narrative quotes 60% and 72%; the arithmetic reads them from
        the same dataclass. This checks the prose was not edited away
        from the numbers underneath it.
        """
        scenario = teaching.DID_TEACHING_SCENARIO

        self.assertIn("60%", scenario.scenario)
        self.assertIn("72%", scenario.scenario)


class TestKnownValues(unittest.TestCase):
    def test_a_flat_comparison_group_reproduces_the_before_after_number(self):
        outcome = teaching.teaching_did(0.0)

        self.assertAlmostEqual(outcome.did_estimate, 12.0, places=12)
        self.assertAlmostEqual(outcome.before_after_estimate, 12.0, places=12)
        self.assertAlmostEqual(outcome.post_comparison, 58.0, places=12)

    def test_a_rising_comparison_group_shrinks_the_estimate(self):
        outcome = teaching.teaching_did(5.0)

        self.assertAlmostEqual(outcome.did_estimate, 7.0, places=12)
        self.assertAlmostEqual(outcome.post_comparison, 63.0, places=12)

    def test_a_matching_comparison_group_zeroes_the_estimate(self):
        outcome = teaching.teaching_did(12.0)

        self.assertAlmostEqual(outcome.did_estimate, 0.0, places=12)

    def test_a_faster_comparison_group_turns_the_estimate_negative(self):
        outcome = teaching.teaching_did(20.0)

        self.assertAlmostEqual(outcome.did_estimate, -8.0, places=12)

    def test_a_falling_comparison_group_enlarges_the_estimate(self):
        outcome = teaching.teaching_did(-4.0)

        self.assertAlmostEqual(outcome.did_estimate, 16.0, places=12)

    def test_the_before_after_number_never_moves(self):
        """
        The whole point of the example: one estimate responds to the
        comparison group and the other cannot see it at all.
        """
        estimates = {
            teaching.teaching_did(change).before_after_estimate
            for change in (-10.0, -4.0, 0.0, 5.0, 12.0, 20.0)
        }

        self.assertEqual(estimates, {12.0})


class TestReading(unittest.TestCase):
    def test_the_flat_case_names_the_assumption_behind_the_agreement(self):
        reading = teaching.teaching_did(0.0).reading

        self.assertIn("assumption", reading)

    def test_the_matching_case_reports_a_zero_estimate(self):
        reading = teaching.teaching_did(12.0).reading

        self.assertIn("0.0", reading)

    def test_the_partial_case_reports_the_share_absorbed(self):
        reading = teaching.teaching_did(6.0).reading

        self.assertIn("50%", reading)

    def test_the_negative_case_is_described_as_negative(self):
        reading = teaching.teaching_did(20.0).reading

        self.assertIn("-8.0", reading)

    def test_every_supported_value_produces_a_nonempty_reading(self):
        for change in (-10.0, -0.5, 0.0, 0.5, 6.0, 12.0, 12.5, 20.0):
            with self.subTest(comparison_change=change):
                self.assertTrue(teaching.teaching_did(change).reading.strip())


class TestDegenerateInput(unittest.TestCase):
    def test_rejects_a_non_numeric_change(self):
        with self.assertRaises(TypeError):
            teaching.teaching_did("5")

    def test_rejects_a_boolean_change(self):
        """bool is a subclass of int, so it would otherwise pass as 0 or 1."""
        with self.assertRaises(TypeError):
            teaching.teaching_did(True)

    def test_rejects_a_nonfinite_change(self):
        for bad in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    teaching.teaching_did(bad)


if __name__ == "__main__":
    unittest.main()
