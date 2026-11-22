"""
Tests for core/agency.py: every yes or no is backed by a phrase that
really appears in that modality's cited control text.

Run with: pytest modules/signal_pipeline/tests/ -v
"""

import unittest
from pathlib import Path

import pandas as pd

from core import agency

PROFILES = pd.read_csv(Path(__file__).resolve().parents[1] / "sample_data" / "modality_profiles.csv")
CONTROL = dict(zip(PROFILES["category"], PROFILES["agency_control_point"]))


class TestEveryAnswerHasItsSource(unittest.TestCase):
    def test_every_category_in_the_data_has_answers(self):
        self.assertEqual(set(agency.CATEGORIES), set(CONTROL))

    def test_each_yes_or_no_quotes_the_control_text_verbatim(self):
        for category in agency.CATEGORIES:
            for a in agency.agency_profile(category).answers:
                with self.subTest(category=category, question=a.key):
                    self.assertIn(a.answer, agency.ANSWERS)
                    if a.answer == agency.NOT_STATED:
                        self.assertEqual(a.evidence, "")
                    else:
                        self.assertIn(a.evidence, CONTROL[category])


class TestKnownAnswers(unittest.TestCase):
    def test_eeg_can_be_stopped_but_not_retracted(self):
        p = {a.key: a.answer for a in agency.agency_profile("Neural").answers}
        self.assertEqual(p, {"stop": "yes", "each_time": "not stated", "undo": "no"})

    def test_self_report_needs_the_person_each_time(self):
        p = {a.key: a.answer for a in agency.agency_profile("Subjective").answers}
        self.assertEqual(p["each_time"], "yes")

    def test_answers_follow_the_question_order(self):
        keys = tuple(a.key for a in agency.agency_profile("Muscular").answers)
        self.assertEqual(keys, ("stop", "each_time", "undo"))


class TestPlainReading(unittest.TestCase):
    def test_every_answer_reads_as_a_sentence_that_leads_with_it(self):
        lead = {"yes": "Yes.", "no": "No.", "not stated": "Unknown."}
        for category in agency.CATEGORIES:
            for a in agency.agency_profile(category).answers:
                with self.subTest(category=category, question=a.key):
                    self.assertTrue(a.plain.startswith(lead[a.answer]))


class TestUnknownCategory(unittest.TestCase):
    def test_an_unknown_category_raises_a_typed_error(self):
        with self.assertRaises(agency.UnknownCategoryError):
            agency.agency_profile("Astral")


if __name__ == "__main__":
    unittest.main()
