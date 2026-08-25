"""
Unit tests for core/screening.py

Run with: pytest modules/evidence_review/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import screening  # noqa: E402


class TestScreeningDecisionValidation(unittest.TestCase):
    def test_unknown_decision_raises(self):
        with self.assertRaises(ValueError):
            screening.ScreeningDecision(record_title="A Study", decision="Maybe")

    def test_empty_record_title_raises(self):
        with self.assertRaises(ValueError):
            screening.ScreeningDecision(
                record_title="", decision=screening.DECISION_INCLUDE
            )


class TestSummarizeScreening(unittest.TestCase):
    def test_hand_calculable_counts(self):
        decisions = (
            screening.ScreeningDecision("A", screening.DECISION_INCLUDE),
            screening.ScreeningDecision("B", screening.DECISION_INCLUDE),
            screening.ScreeningDecision("C", screening.DECISION_EXCLUDE),
            screening.ScreeningDecision("D", screening.DECISION_UNCERTAIN),
        )
        summary = screening.summarize_screening(n_found=4, decisions=decisions)

        self.assertEqual(summary.n_found, 4)
        self.assertEqual(summary.n_included, 2)
        self.assertEqual(summary.n_excluded, 1)
        self.assertEqual(summary.n_uncertain, 1)

    def test_undecided_results_count_as_uncertain(self):
        decisions = (screening.ScreeningDecision("A", screening.DECISION_INCLUDE),)
        summary = screening.summarize_screening(n_found=3, decisions=decisions)

        self.assertEqual(summary.n_included, 1)
        self.assertEqual(summary.n_excluded, 0)
        self.assertEqual(summary.n_uncertain, 2)

    def test_zero_results_found_is_a_valid_all_zero_summary(self):
        summary = screening.summarize_screening(n_found=0, decisions=())

        self.assertEqual(summary.n_found, 0)
        self.assertEqual(summary.n_included, 0)

    def test_negative_n_found_raises(self):
        with self.assertRaises(ValueError):
            screening.summarize_screening(n_found=-1, decisions=())

    def test_more_decisions_than_results_found_raises(self):
        decisions = (
            screening.ScreeningDecision("A", screening.DECISION_INCLUDE),
            screening.ScreeningDecision("B", screening.DECISION_INCLUDE),
        )
        with self.assertRaises(ValueError):
            screening.summarize_screening(n_found=1, decisions=decisions)


class TestScreeningSummaryValidation(unittest.TestCase):
    def test_counts_not_summing_to_n_found_raises(self):
        with self.assertRaises(ValueError):
            screening.ScreeningSummary(
                n_found=5, n_included=1, n_excluded=1, n_uncertain=1
            )

    def test_negative_count_raises(self):
        with self.assertRaises(ValueError):
            screening.ScreeningSummary(
                n_found=1, n_included=-1, n_excluded=1, n_uncertain=1
            )


if __name__ == "__main__":
    unittest.main()
