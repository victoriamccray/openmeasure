"""
Unit tests for core/teaching.py

The scenario's arithmetic is deliberately simple, so these tests pin the
exact numbers a reader is shown. If the scenario's fixed values are ever
edited, these fail rather than the teaching text quietly describing
different numbers than the ones on screen.

Run with: pytest modules/program_evaluation/tests/ -v
"""

from __future__ import annotations

import dataclasses
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


class TestDomainVariants(unittest.TestCase):
    """
    The same lesson told in several fields' terms, with the same numbers.
    The arithmetic invariance itself is pinned in test_domains.py.
    """

    OWN_SCENARIO = ("public_health", "education", "workforce")

    def test_the_three_seeded_domains_have_their_own_scenario(self):
        for domain_id in self.OWN_SCENARIO:
            with self.subTest(domain=domain_id):
                self.assertTrue(teaching.has_own_did_scenario(domain_id))

    def test_the_remaining_domains_fall_back_to_the_canonical(self):
        for domain_id in ("social_programs", "criminal_justice", "digital", "other"):
            with self.subTest(domain=domain_id):
                self.assertFalse(teaching.has_own_did_scenario(domain_id))
                self.assertIs(
                    teaching.did_scenario_for(domain_id),
                    teaching.CANONICAL_DID_SCENARIO,
                )

    def test_no_domain_chosen_yet_gets_the_canonical(self):
        self.assertIs(
            teaching.did_scenario_for(None), teaching.CANONICAL_DID_SCENARIO
        )

    def test_an_unknown_domain_raises_rather_than_falling_back(self):
        """
        A typo must surface. Falling back would silently show the least
        tailored example and look like a domain awaiting its variant.
        """
        with self.assertRaises(ValueError):
            teaching.did_scenario_for("helthcare")

        with self.assertRaises(ValueError):
            teaching.has_own_did_scenario("helthcare")

    def test_each_variant_tells_a_genuinely_different_story(self):
        """
        Catches a variant added by copying another and changing a label,
        which would present as tailored without being so.
        """
        narratives = [
            teaching.did_scenario_for(domain_id).scenario
            for domain_id in self.OWN_SCENARIO
        ]

        self.assertEqual(len(narratives), len(set(narratives)))

    def test_each_variant_names_its_own_field_vocabulary(self):
        for domain_id, expected in (
            ("public_health", "clinic"),
            ("education", "school"),
            ("workforce", "region"),
        ):
            scenario = teaching.did_scenario_for(domain_id)
            with self.subTest(domain=domain_id):
                self.assertIn(expected, scenario.scenario.lower())

    def test_every_variant_states_the_same_sixty_to_seventy_two_rise(self):
        for domain_id in self.OWN_SCENARIO:
            scenario = teaching.did_scenario_for(domain_id)
            with self.subTest(domain=domain_id):
                self.assertIn("60%", scenario.scenario)
                self.assertIn("72%", scenario.scenario)

    def test_every_variant_populates_the_labels_the_page_renders(self):
        for domain_id in self.OWN_SCENARIO:
            scenario = teaching.did_scenario_for(domain_id)
            for field_name in (
                "program_label",
                "period_label",
                "pre_period_label",
                "post_period_label",
                "unit_label",
                "outcome_label",
            ):
                with self.subTest(domain=domain_id, field=field_name):
                    self.assertTrue(getattr(scenario, field_name).strip())

    def test_a_scenario_naming_an_unknown_domain_is_rejected(self):
        canonical = teaching.CANONICAL_DID_SCENARIO

        with self.assertRaises(ValueError):
            dataclasses.replace(canonical, domain_id="not_a_domain")

    def test_the_reading_uses_each_variants_own_program_word(self):
        for domain_id, expected in (
            ("public_health", "reminder program"),
            ("education", "tutoring program"),
            ("workforce", "training program"),
        ):
            scenario = teaching.did_scenario_for(domain_id)
            reading = teaching.teaching_did(6.0, scenario=scenario).reading
            with self.subTest(domain=domain_id):
                self.assertIn(expected, reading)


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
