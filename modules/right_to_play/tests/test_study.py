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


class TestProvenanceBadges(unittest.TestCase):
    """
    The short form, for sitting beside a fact rather than under it.
    """

    def test_every_state_has_a_badge(self):
        for state in study.PROVENANCE_STATES:
            with self.subTest(state=state):
                self.assertIn(state, study.PROVENANCE_BADGES)

    def test_a_badge_is_short_enough_to_sit_beside_a_fact(self):
        for badge in study.PROVENANCE_BADGES.values():
            with self.subTest(badge=badge):
                self.assertLessEqual(len(badge), 16)

    def test_the_badges_stay_distinguishable(self):
        self.assertEqual(
            len(set(study.PROVENANCE_BADGES.values())),
            len(study.PROVENANCE_STATES),
        )

    def test_every_fact_can_produce_one(self):
        for fact in ALL_FACTS:
            with self.subTest(fact=fact.label):
                self.assertTrue(fact.badge)


class TestMeasurementMap(unittest.TestCase):
    """
    Peer violence was measured as two separate things, on two separate
    scales. A list of three instruments hides that.
    """

    def test_peer_violence_splits_into_two_facets(self):
        violence = next(
            construct
            for construct in study.MEASUREMENT_MAP
            if construct.name == "Peer violence"
        )

        self.assertEqual(len(violence.facets), 2)

    def test_the_two_facets_use_different_instruments(self):
        violence = next(
            construct
            for construct in study.MEASUREMENT_MAP
            if construct.name == "Peer violence"
        )
        instruments = {facet.instrument for facet in violence.facets}

        self.assertEqual(len(instruments), 2)

    def test_every_instrument_named_also_appears_in_the_facts(self):
        """
        So the diagram and the cited list cannot drift apart.
        """
        recorded = " ".join(fact.value for fact in study.MEASUREMENT)

        for construct in study.MEASUREMENT_MAP:
            for facet in construct.facets:
                with self.subTest(instrument=facet.instrument):
                    self.assertIn(facet.instrument, recorded)

    def test_a_facet_without_an_instrument_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            study.MeasuredFacet("Something", "")

        self.assertIn("was not measured", str(raised.exception))

    def test_a_construct_with_no_facets_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            study.MeasuredConstruct("Something", ())

        self.assertIn("names no facets", str(raised.exception))


class TestDiagramLabelsFit(unittest.TestCase):
    """
    Clipping produced "The reported treatment effect..." on the one line
    of the journey that has to be read, and a reader cannot tell a
    shortened label from one that happens to end there.
    """

    def test_every_artifact_label_fits_the_boundary_diagram(self):
        for fact in study.ARTIFACTS:
            with self.subTest(fact=fact.label):
                self.assertLessEqual(
                    len(fact.for_diagram), study.BOUNDARY_LABEL_LIMIT
                )

    def test_a_fact_that_does_not_fit_is_rejected_rather_than_clipped(self):
        with self.assertRaises(ValueError) as raised:
            study.StudyFact(
                label="A label far longer than any column could ever hold",
                value="A value",
                provenance=study.REPORTED,
                source="Somewhere",
            )

        self.assertIn("rather than letting it be clipped", str(raised.exception))

    def test_a_short_form_satisfies_the_limit(self):
        fact = study.StudyFact(
            label="A label far longer than any column could ever hold",
            short_label="A short label",
            value="A value",
            provenance=study.REPORTED,
            source="Somewhere",
        )

        self.assertEqual(fact.for_diagram, "A short label")

    def test_the_full_label_is_what_the_expander_still_shows(self):
        """
        The short form is for the diagram only. The full name stays the
        fact's label, so the list underneath is unaffected.
        """
        recomputed = next(
            fact for fact in study.ARTIFACTS if "recomputed" in fact.label
        )

        self.assertIn("independently recomputed", recomputed.label)
        self.assertNotEqual(recomputed.label, recomputed.for_diagram)


if __name__ == "__main__":
    unittest.main()
