"""
Unit tests for core/eligibility.py

Run with: pytest modules/evidence_review/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import eligibility  # noqa: E402
from core.record import LiteratureRecord  # noqa: E402


def _record(title="A Study", year=2020, abstract="about anxiety in adults"):
    return LiteratureRecord(
        title=title,
        authors=("A. Author",),
        year=year,
        venue="Journal of Testing",
        doi=None,
        url="https://openalex.org/W1",
        abstract=abstract,
        citation_count=5,
        source_api="OpenAlex",
    )


class TestEligibilityCriteriaValidation(unittest.TestCase):
    def test_negative_min_year_raises(self):
        with self.assertRaises(ValueError):
            eligibility.EligibilityCriteria(min_year=-1)

    def test_blank_required_term_raises(self):
        with self.assertRaises(ValueError):
            eligibility.EligibilityCriteria(required_terms=("  ",))

    def test_is_empty_true_with_no_criteria(self):
        self.assertTrue(eligibility.EligibilityCriteria().is_empty)

    def test_is_empty_false_with_a_criterion(self):
        self.assertFalse(eligibility.EligibilityCriteria(min_year=2015).is_empty)


class TestAssessEligibility(unittest.TestCase):
    def test_no_criteria_means_everything_eligible(self):
        results = (_record(year=1990), _record(year=2020))
        summary = eligibility.assess_eligibility(
            results, eligibility.EligibilityCriteria()
        )
        self.assertEqual(summary.n_eligible, 2)
        self.assertEqual(summary.n_ineligible, 0)

    def test_min_year_excludes_older_results(self):
        results = (_record(title="Old", year=2000), _record(title="New", year=2022))
        summary = eligibility.assess_eligibility(
            results, eligibility.EligibilityCriteria(min_year=2015)
        )
        self.assertEqual(summary.n_eligible, 1)
        self.assertEqual(summary.n_ineligible, 1)

        by_title = {a.record_title: a for a in summary.assessments}
        self.assertFalse(by_title["Old"].eligible)
        self.assertTrue(by_title["New"].eligible)
        self.assertIn("2000", by_title["Old"].reasons_excluded[0])

    def test_missing_year_treated_as_not_meeting_min_year(self):
        results = (_record(title="Unknown year", year=None),)
        summary = eligibility.assess_eligibility(
            results, eligibility.EligibilityCriteria(min_year=2000)
        )
        self.assertEqual(summary.n_eligible, 0)
        self.assertIn(
            "no recorded publication year",
            summary.assessments[0].reasons_excluded[0],
        )

    def test_required_term_checks_title_and_abstract(self):
        results = (
            _record(title="Anxiety in postpartum women", abstract="unrelated text"),
            _record(title="Unrelated title", abstract="mentions anxiety directly"),
            _record(title="Nothing here", abstract="still nothing"),
        )
        summary = eligibility.assess_eligibility(
            results, eligibility.EligibilityCriteria(required_terms=("anxiety",))
        )
        self.assertEqual(summary.n_eligible, 2)
        self.assertEqual(summary.n_ineligible, 1)

    def test_n_found_matches_input_length(self):
        results = (_record(), _record(), _record())
        summary = eligibility.assess_eligibility(
            results, eligibility.EligibilityCriteria()
        )
        self.assertEqual(summary.n_found, 3)


if __name__ == "__main__":
    unittest.main()
