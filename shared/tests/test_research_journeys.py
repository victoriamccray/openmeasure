"""
Unit tests for shared/research_journeys.py

Drift tests, same purpose as shared/tests/test_catalog.py: fail when the
domain grouping and the repository disagree, or when a journey's url_path
would break a link someone already has.

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from shared.research_journeys import (
    JOURNEY_DOMAINS,
    JOURNEYS,
    ResearchJourney,
    journeys_by_domain,
)

ROOT = Path(__file__).resolve().parents[2]

# Every url_path already linked to before domain grouping existed. A
# journey's page and title may change; this must not, or an existing link
# such as /HealthRing_Worked_Example breaks silently.
PREEXISTING_URL_PATHS: frozenset[str] = frozenset(
    {
        "HealthRing_Worked_Example",
        "FMRI_QC_Worked_Example",
        "Portfolio_Impact_Analysis",
        "GAIA_Worked_Example",
        "GRAND_Worked_Example",
        "Multimodal_Signal_Convergence",
    }
)


class TestJourneyFields(unittest.TestCase):
    def test_required_fields_are_populated(self):
        for journey in JOURNEYS:
            with self.subTest(journey=journey.title):
                self.assertTrue(journey.title)
                self.assertTrue(journey.domain)
                self.assertTrue(journey.subdomain)
                self.assertTrue(journey.summary)
                self.assertTrue(journey.page)

    def test_empty_field_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            ResearchJourney(
                title="",
                domain="Responsible AI",
                subdomain="Something",
                summary="A summary.",
                page="pages/9_X.py",
            )

    def test_unknown_domain_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            ResearchJourney(
                title="Orphan",
                domain="Not A Domain",
                subdomain="Something",
                summary="A summary.",
                page="pages/9_Orphan.py",
            )

        self.assertIn("not in JOURNEY_DOMAINS", str(context.exception))

    def test_multiline_summary_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            ResearchJourney(
                title="Wordy",
                domain="Responsible AI",
                subdomain="Something",
                summary="Line one.\nLine two.",
                page="pages/9_X.py",
            )

        self.assertIn("single line", str(context.exception))

    def test_entries_are_frozen(self):
        self.assertTrue(ResearchJourney.__dataclass_params__.frozen)

    def test_titles_are_unique(self):
        titles = [journey.title for journey in JOURNEYS]

        self.assertEqual(len(titles), len(set(titles)))

    def test_every_journey_page_exists_on_disk(self):
        missing = {
            journey.page
            for journey in JOURNEYS
            if not (ROOT / journey.page).is_file()
        }

        self.assertEqual(missing, set())


class TestUrlPathsPreserved(unittest.TestCase):
    def test_url_paths_are_unique(self):
        paths = [journey.url_path for journey in JOURNEYS]

        self.assertEqual(len(paths), len(set(paths)))

    def test_url_paths_strip_the_numeric_prefix(self):
        for journey in JOURNEYS:
            with self.subTest(journey=journey.title):
                self.assertFalse(re.match(r"^\d+_", journey.url_path))
                self.assertNotIn("/", journey.url_path)

    def test_every_preexisting_url_path_still_resolves(self):
        current = {journey.url_path for journey in JOURNEYS}

        missing = PREEXISTING_URL_PATHS - current
        self.assertEqual(
            missing,
            set(),
            f"These url_paths were reachable before domain grouping and no "
            f"longer are, which breaks an existing link: {sorted(missing)}.",
        )


class TestDomainGrouping(unittest.TestCase):
    def test_grouping_covers_every_journey_exactly_once(self):
        grouped = journeys_by_domain()
        seen = [
            journey.title
            for journeys in grouped.values()
            for journey in journeys
        ]

        self.assertEqual(sorted(seen), sorted(j.title for j in JOURNEYS))
        self.assertEqual(len(seen), len(set(seen)))

    def test_domains_appear_in_declared_order(self):
        grouped = journeys_by_domain()
        expected = [
            domain for domain in JOURNEY_DOMAINS if domain in grouped
        ]

        self.assertEqual(list(grouped.keys()), expected)

    def test_no_domain_group_is_empty(self):
        for domain, journeys in journeys_by_domain().items():
            with self.subTest(domain=domain):
                self.assertTrue(journeys)


if __name__ == "__main__":
    unittest.main()
