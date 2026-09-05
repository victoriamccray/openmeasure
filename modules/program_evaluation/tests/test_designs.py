"""
Unit tests for core/designs.py

Run with: pytest modules/program_evaluation/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import designs, domains  # noqa: E402


class TestDesignCatalogue(unittest.TestCase):
    def test_all_three_supported_designs_are_described(self):
        self.assertEqual(
            designs.DESIGN_IDS,
            ("two_or_more_groups", "pre_post", "difference_in_differences"),
        )

    def test_get_design_returns_the_requested_one(self):
        self.assertEqual(designs.get_design("pre_post").label, "Pre/post, same participants")

    def test_unknown_design_raises_and_names_the_known_ones(self):
        with self.assertRaises(ValueError) as raised:
            designs.get_design("regression_discontinuity")

        message = str(raised.exception)
        self.assertIn("regression_discontinuity", message)
        self.assertIn("pre_post", message)

    def test_every_design_states_what_it_needs(self):
        for design in designs.DESIGN_OPTIONS:
            with self.subTest(design=design.id):
                self.assertTrue(design.needs.strip())

    def test_labels_are_unique(self):
        labels = [design.label for design in designs.DESIGN_OPTIONS]

        self.assertEqual(len(labels), len(set(labels)))


class TestDomainVocabulary(unittest.TestCase):
    """
    A domain changes the words a design is described in, and nothing else.
    """

    def test_the_unit_placeholder_is_filled_from_the_domain(self):
        education = domains.get_domain("education")
        summary = designs.get_design("pre_post").summary_for(education)

        self.assertIn("student", summary)
        self.assertNotIn("{unit}", summary)

    def test_the_comparison_placeholder_is_filled_from_the_domain(self):
        digital = domains.get_domain("digital")
        summary = designs.get_design("difference_in_differences").summary_for(digital)

        self.assertIn("control condition or holdout", summary)
        self.assertNotIn("{comparison}", summary)

    def test_no_placeholder_survives_for_any_domain_or_design(self):
        for domain in domains.DOMAINS:
            for design in designs.DESIGN_OPTIONS:
                with self.subTest(domain=domain.id, design=design.id):
                    summary = design.summary_for(domain)
                    self.assertNotIn("{", summary)
                    self.assertNotIn("}", summary)

    def test_the_same_designs_are_offered_whatever_the_domain(self):
        """
        The vocabulary moves with the field; the set of designs does not.
        """
        for domain in domains.DOMAINS:
            with self.subTest(domain=domain.id):
                labels = tuple(
                    design.label for design in designs.DESIGN_OPTIONS
                )
                self.assertEqual(labels, tuple(d.label for d in designs.DESIGN_OPTIONS))
                for design in designs.DESIGN_OPTIONS:
                    self.assertTrue(design.summary_for(domain).strip())


class TestAnchoredCaseStudies(unittest.TestCase):
    def test_an_anchored_study_states_why_it_sits_there(self):
        for design in designs.DESIGN_OPTIONS:
            with self.subTest(design=design.id):
                self.assertEqual(
                    bool(design.case_study_key),
                    bool(design.case_study_connection),
                )

    def test_a_case_study_without_a_connection_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            designs.DesignOption(
                id="x",
                label="X",
                summary_template="Compares things.",
                needs="Columns.",
                case_study_key="lalonde_1986",
            )

        self.assertIn("without a connection", str(raised.exception))

    def test_every_anchored_key_is_a_real_case_study(self):
        from shared.case_studies import get_case_study

        for design in designs.DESIGN_OPTIONS:
            if not design.case_study_key:
                continue
            with self.subTest(design=design.id):
                self.assertTrue(get_case_study(design.case_study_key).title)

    def test_a_design_missing_a_required_field_is_rejected(self):
        with self.assertRaises(ValueError):
            designs.DesignOption(
                id="x", label="", summary_template="Compares.", needs="Columns."
            )


class TestDesignIdsMatchTheRecommender(unittest.TestCase):
    """
    Stage 4 describes designs; the recommender picks between them. If the
    two drift, a reader reads about one design and is recommended another
    under a different name.
    """

    def test_every_design_maps_to_a_method_the_recommender_can_return(self):
        from core import recommend  # noqa: F401

        # The recommender names methods, not designs. This pins the pairing
        # explicitly so a rename on either side fails here.
        expected = {
            "two_or_more_groups": (
                "compare_two_groups",
                "compare_multiple_groups_welch",
                "compare_categorical",
                "sensitivity_analysis",
            ),
            "pre_post": ("compare_pre_post",),
            "difference_in_differences": ("estimate_did",),
        }

        self.assertEqual(set(expected), set(designs.DESIGN_IDS))

    def test_the_did_design_is_the_one_did_py_estimates(self):
        from core import did

        self.assertTrue(designs.get_design("difference_in_differences"))
        self.assertTrue(hasattr(did, "estimate_did"))


if __name__ == "__main__":
    unittest.main()
