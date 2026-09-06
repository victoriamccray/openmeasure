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
