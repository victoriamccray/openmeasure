"""
Unit tests for core/lexicon.py

Run with: pytest modules/research_design/tests/ -v

The property under test is restraint. Recognising a phrase that is in the
list is easy; the tests that matter are the ones checking nothing is
offered when nothing was recognised, and that a suggestion always says
which wording produced it.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import lexicon, ontology  # noqa: E402


class TestItRecognisesRatherThanGuesses(unittest.TestCase):
    def test_an_unrecognised_question_yields_nothing(self):
        """
        The whole difference between recognising and inventing. A
        plausible suggestion here would be a construct nobody chose,
        entering a study looking as though they had.
        """
        self.assertEqual(
            lexicon.suggest_concepts("Something entirely novel about widgets"),
            (),
        )

    def test_the_empty_case_is_said_plainly(self):
        note = lexicon.NOTHING_RECOGNISED.lower()

        self.assertIn("gap in the list", note)
        self.assertIn("name the concepts yourself", note)

    def test_every_suggestion_carries_the_phrase_that_produced_it(self):
        """
        So a researcher can see why something was offered, and throw it
        away on that basis.
        """
        for suggestion in lexicon.suggest_concepts(
            "financial security and depression among adults"
        ):
            with self.subTest(concept=suggestion.concept):
                self.assertTrue(suggestion.matched_phrase.strip())

    def test_a_phrase_matches_only_as_a_whole_word(self):
        """
        Without this, "debt" matches "indebtedness" and "pain" matches
        "painting", and the suggestions stop being trustworthy.
        """
        self.assertEqual(lexicon.suggest_concepts("a painting of a train"), ())


class TestTheColdTestQuestions(unittest.TestCase):
    """
    The three questions this rebuild was tested against, plus the one
    that started it.
    """

    def _concepts(self, question):
        return [s.concept for s in lexicon.suggest_concepts(question)]

    def test_financial_security(self):
        self.assertIn(
            "Financial security",
            self._concepts("increase in financial security among adults"),
        )

    def test_mental_models(self):
        self.assertIn(
            "Mental-model structure",
            self._concepts(
                "How can researchers assess mental models that affect "
                "implementation of an intervention?"
            ),
        )

    def test_chronic_pain(self):
        concepts = self._concepts(
            "Does coupling between subjective pain and physiological arousal "
            "change with pain distribution?"
        )

        self.assertIn("Pain experience", concepts)
        self.assertIn("Physiological arousal", concepts)

    def test_each_suggested_concept_leads_somewhere(self):
        """
        A suggestion whose kind no measure can observe would be an offer
        with nothing behind it.
        """
        for question in (
            "increase in financial security among adults",
            "assessing mental models in implementation",
            "subjective pain and physiological arousal",
            "gene expression and inflammation",
        ):
            for suggestion in lexicon.suggest_concepts(question):
                with self.subTest(concept=suggestion.concept):
                    self.assertTrue(ontology.measures_for(suggestion.kind))


class TestOverlappingPhrases(unittest.TestCase):
    def test_a_longer_phrase_wins_over_one_inside_it(self):
        concepts = [
            s.concept
            for s in lexicon.suggest_concepts("improving financial well-being")
        ]

        self.assertEqual(concepts, ["Financial well-being"])

    def test_shared_mental_model_beats_mental_model(self):
        concepts = [
            s.concept
            for s in lexicon.suggest_concepts("measuring shared mental models")
        ]

        self.assertIn("Shared mental models", concepts)
        self.assertNotIn("Mental-model structure", concepts)

    def test_two_phrases_for_one_concept_suggest_it_once(self):
        concepts = [
            s.concept
            for s in lexicon.suggest_concepts("depression and depressive symptoms")
        ]

        self.assertEqual(concepts.count("Depressive symptoms"), 1)

    def test_suggestions_follow_the_order_of_the_question(self):
        concepts = [
            s.concept
            for s in lexicon.suggest_concepts("attendance, then test scores")
        ]

        self.assertEqual(concepts, ["Attendance", "Academic achievement"])


class TestTheListIsWellFormed(unittest.TestCase):
    def test_every_entry_maps_to_a_declared_kind(self):
        for entry in lexicon.LEXICON:
            with self.subTest(phrase=entry.phrase):
                self.assertIn(entry.kind, ontology.CONCEPT_KINDS)

    def test_every_entry_is_lowercase(self):
        """
        Matching is case-insensitive against a lowered question, so a
        mixed-case entry would never be reached.
        """
        for entry in lexicon.LEXICON:
            with self.subTest(phrase=entry.phrase):
                self.assertEqual(entry.phrase, entry.phrase.lower())

    def test_a_mixed_case_entry_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            lexicon.LexiconEntry(
                "Financial Security", "Financial security", ontology.KIND_ECONOMIC_STATE
            )

        self.assertIn("never be reached", str(raised.exception))

    def test_an_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            lexicon.LexiconEntry("widgets", "Widgets", "A vibe")

    def test_no_phrase_is_duplicated(self):
        phrases = [entry.phrase for entry in lexicon.LEXICON]

        self.assertEqual(len(phrases), len(set(phrases)))

    def test_every_kind_the_list_can_reach_has_measures(self):
        for kind in {entry.kind for entry in lexicon.LEXICON}:
            with self.subTest(kind=kind):
                self.assertTrue(ontology.measures_for(kind))


if __name__ == "__main__":
    unittest.main()


class TestCategoryCompatibilityIsNotConstructValidity(unittest.TestCase):
    """
    The distinction this module exists to keep, at the measure level.

    It kept it at the concept level from the start: a phrase outside the
    curated list is not guessed at. Then it lost it one layer down. A
    concept nobody had curated fell back to every measure whose own
    record says it can observe that kind of thing, and the page rendered
    that join in the same gallery it rendered a curated match in.

    The case that exposed it: functional neuroimaging declares that it
    observes "something a person or system does", which is true, so
    OpenMeasure offered it as a way to measure violent behaviour among
    students. It can study neural processes associated with a behaviour.
    It does not measure the behaviour.
    """

    def test_a_curated_concept_says_it_was_matched_to_the_concept(self):
        found = lexicon.measures_for_concept(
            "Pain experience", ontology.KIND_EXPERIENCE
        )

        self.assertEqual(found.basis, lexicon.BASIS_CONCEPT)
        self.assertTrue(found.validated_for_the_concept)

    def test_an_uncurated_concept_says_it_is_only_category_level(self):
        found = lexicon.measures_for_concept(
            "Violent behaviour", ontology.KIND_BEHAVIOR
        )

        self.assertEqual(found.basis, lexicon.BASIS_CATEGORY)
        self.assertFalse(found.validated_for_the_concept)

    def test_the_category_note_refuses_the_appropriateness_claim(self):
        found = lexicon.measures_for_concept(
            "Violent behaviour", ontology.KIND_BEHAVIOR
        )

        self.assertIn("has not established", found.note)
        self.assertIn("Some will not be", found.note)

    def test_neuroimaging_for_violence_is_never_called_a_match(self):
        """
        Pinned by name, because this is the specific thing that read as a
        recommendation. If a later change curates a list for violent
        behaviour, this test should be updated deliberately rather than
        pass by accident.
        """
        found = lexicon.measures_for_concept(
            "Violent behaviour", ontology.KIND_BEHAVIOR
        )
        names = [measure.name for measure in found.measures]

        self.assertIn("Functional neuroimaging", names)
        self.assertFalse(found.validated_for_the_concept)

    def test_every_uncurated_concept_is_category_level(self):
        """
        No concept reaches a concept-specific basis without someone
        having written it down.
        """
        for entry in lexicon.LEXICON:
            found = lexicon.measures_for_concept(entry.concept, entry.kind)
            curated = entry.concept in lexicon.CONCEPT_MEASURES

            with self.subTest(concept=entry.concept):
                self.assertEqual(found.validated_for_the_concept, curated)

    def test_a_basis_outside_the_two_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            lexicon.MeasureCandidates(
                concept="Anything",
                kind=ontology.KIND_EXPERIENCE,
                measures=(),
                basis="Probably fine",
            )

        self.assertIn("is not a basis", str(raised.exception))


class TestCoverageIsReportedRatherThanAssumed(unittest.TestCase):
    """
    Roughly half the known concepts have no curated list. A reader
    deciding how much to trust this step is entitled to the number.
    """

    def test_coverage_counts_curated_concepts_against_known_ones(self):
        curated, known = lexicon.curated_coverage()

        self.assertGreater(known, 0)
        self.assertLessEqual(curated, known)

    def test_it_matches_the_two_tables_it_reports_on(self):
        curated, known = lexicon.curated_coverage()
        concepts = {entry.concept for entry in lexicon.LEXICON}

        self.assertEqual(known, len(concepts))
        self.assertEqual(curated, len(concepts & set(lexicon.CONCEPT_MEASURES)))

    def test_it_does_not_claim_full_coverage(self):
        """
        A guard against a future change that silently reports complete
        curation because it started counting the wrong set.
        """
        curated, known = lexicon.curated_coverage()

        self.assertLess(curated, known)
