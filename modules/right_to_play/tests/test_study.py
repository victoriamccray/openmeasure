"""
Unit tests for modules/right_to_play/core/study.py

Run with: pytest modules/right_to_play/tests/ -v

This journey is about a boundary, so most of what is tested here is that
the boundary holds: that nothing is recorded without a source, that the
things the public artifacts cannot settle stay unsettled, and that no
figure appears which would have to have been invented.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from core import study  # noqa: E402

ALL_FACTS = study.DESIGN + study.MEASUREMENT + study.ARTIFACTS


class TestEveryFactCarriesItsProvenance(unittest.TestCase):
    def test_every_fact_declares_a_known_state(self):
        for fact in ALL_FACTS:
            with self.subTest(fact=fact.label):
                self.assertIn(fact.provenance, study.PROVENANCE_STATES)

    def test_every_fact_names_a_source(self):
        for fact in ALL_FACTS:
            with self.subTest(fact=fact.label):
                self.assertTrue(fact.source.strip())

    def test_a_fact_without_a_source_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            study.StudyFact(
                label="Something",
                value="A value",
                provenance=study.REPORTED,
                source="",
            )

        self.assertIn("not a fact this module records", str(raised.exception))

    def test_an_unknown_provenance_state_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            study.StudyFact(
                label="Something",
                value="A value",
                provenance="Probably true",
                source="Somewhere",
            )

        self.assertIn("not one of the declared states", str(raised.exception))


class TestTheBoundaryHolds(unittest.TestCase):
    """
    The public supplementary file is the baseline wave. Everything that
    needs the 24-month outcomes has to stay unresolved.
    """

    def setUp(self):
        self.boundary = study.replication_boundary()

    def test_the_reported_effect_cannot_be_recomputed(self):
        unresolved = " ".join(fact.label for fact in self.boundary.unresolved)

        self.assertIn("independently recomputed", unresolved)

    def test_follow_up_microdata_are_marked_unavailable(self):
        unresolved = " ".join(fact.label for fact in self.boundary.unresolved)

        self.assertIn("Follow-up participant microdata", unresolved)

    def test_the_power_calculation_is_not_claimed_as_reproducible(self):
        power = next(
            fact for fact in study.ARTIFACTS if "power calculation" in fact.label
        )

        self.assertEqual(power.provenance, study.NOT_ESTABLISHED)

    def test_baseline_microdata_are_established(self):
        established = " ".join(fact.label for fact in self.boundary.established)

        self.assertIn("Baseline participant microdata", established)

    def test_both_sides_of_the_boundary_are_populated(self):
        """
        A boundary with nothing on one side is not a boundary. The lesson
        is that the same study is open on one side and closed on the
        other.
        """
        self.assertTrue(self.boundary.established)
        self.assertTrue(self.boundary.unresolved)

    def test_the_lesson_is_about_interpretability_without_reproducibility(self):
        lesson = self.boundary.lesson.lower()

        self.assertIn("interpretable", lesson)
        self.assertIn("without being reproducible", lesson)


class TestNothingIsCalculatedYet(unittest.TestCase):
    """
    Nothing here is derived from the data file, because OpenMeasure
    cannot open an SPSS .sav today. A fact marked CALCULATED would have
    to have come from somewhere, and there is nowhere for it to have come
    from.
    """

    def test_no_fact_claims_to_have_been_calculated(self):
        for fact in ALL_FACTS:
            with self.subTest(fact=fact.label):
                self.assertNotEqual(fact.provenance, study.CALCULATED)

    def test_no_effect_size_or_p_value_appears_anywhere(self):
        """
        The trial's reported findings are cited, never restated as
        numbers. Restating one would put a figure on the page that this
        repository has not verified.
        """
        text = " ".join(f"{fact.label} {fact.value}" for fact in ALL_FACTS)

        for token in ("p =", "p<", "p >", "Cohen", "95% CI", "odds ratio"):
            with self.subTest(token=token):
                self.assertNotIn(token, text)


class TestStatedLimits(unittest.TestCase):
    def test_the_clustering_limit_names_the_direction_of_the_error(self):
        """
        Getting clustering wrong is not neutral. It makes a result look
        more certain than the design supports, and saying only that the
        assumption is violated leaves a reader unable to tell which way.
        """
        limit = study.CLUSTERING_LIMIT.lower()

        self.assertIn("narrower", limit)
        self.assertIn("smaller", limit)

    def test_the_clustering_limit_does_not_promise_cluster_robust_support(self):
        self.assertIn("not something this toolkit does yet", study.CLUSTERING_LIMIT)

    def test_the_file_support_note_says_the_format_cannot_be_read(self):
        self.assertIn(".sav", study.BASELINE_FILE_SUPPORT)
        self.assertIn("cannot read", study.BASELINE_FILE_SUPPORT)


class TestCitationsMatchTheCatalog(unittest.TestCase):
    def test_the_baseline_citation_matches_the_catalog_entry(self):
        from shared.datasets import get_dataset

        catalog = get_dataset("right_to_play_baseline").citation

        self.assertIn("Karmaliani", catalog)
        self.assertIn("e0180833", study.BASELINE_CITATION)
        self.assertIn("e0180833", catalog)

    def test_the_trial_citation_is_the_separate_results_article(self):
        self.assertIn("Global Health Action", study.TRIAL_CITATION)
        self.assertNotIn("PLOS ONE", study.TRIAL_CITATION)


if __name__ == "__main__":
    unittest.main()
