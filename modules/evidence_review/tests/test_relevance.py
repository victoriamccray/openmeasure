"""
Unit tests for core/relevance.py

Run with: pytest modules/evidence_review/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import relevance  # noqa: E402
from core.record import LiteratureRecord  # noqa: E402


def _record(title: str, abstract: str = "") -> LiteratureRecord:
    return LiteratureRecord(
        title=title,
        authors=(),
        year=2020,
        venue="A Journal",
        doi=None,
        url="https://openalex.org/W1",
        abstract=abstract,
        citation_count=5,
        source_api="OpenAlex",
    )


class TestScoreRelevance(unittest.TestCase):
    def test_shared_keywords_are_matched(self):
        item = _record(
            "Reliability of Wearable Heart-Rate Sensors",
            abstract="Consumer wearables show moderate agreement with reference devices.",
        )
        score = relevance.score_relevance(
            "How reliable are wearable heart-rate sensors compared to reference devices?",
            item,
        )

        self.assertIn("wearable", score.matched_keywords)
        self.assertIn("heart", score.matched_keywords)
        self.assertIn("reference", score.matched_keywords)
        self.assertIn("devices", score.matched_keywords)
        self.assertEqual(score.overlap_count, len(score.matched_keywords))
        self.assertEqual(score.citation_count, 5)

    def test_stopwords_are_excluded_from_matching(self):
        item = _record("The Study of the Thing", abstract="")
        score = relevance.score_relevance("the and of for", item)

        self.assertEqual(score.matched_keywords, ())
        self.assertEqual(score.overlap_count, 0)

    def test_no_overlap_is_a_valid_zero_result_not_an_error(self):
        item = _record("Astrophysics of Distant Galaxies", abstract="")
        score = relevance.score_relevance("wearable heart rate sensor reliability", item)

        self.assertEqual(score.overlap_count, 0)

    def test_empty_finding_text_does_not_raise(self):
        item = _record("Anything", abstract="")
        score = relevance.score_relevance("", item)

        self.assertEqual(score.overlap_count, 0)


class TestRelevanceScoreValidation(unittest.TestCase):
    def test_mismatched_overlap_count_raises(self):
        with self.assertRaises(ValueError):
            relevance.RelevanceScore(
                matched_keywords=("wearable",), overlap_count=2, citation_count=0
            )

    def test_negative_overlap_count_raises(self):
        with self.assertRaises(ValueError):
            relevance.RelevanceScore(
                matched_keywords=(), overlap_count=-1, citation_count=0
            )

    def test_negative_citation_count_raises(self):
        with self.assertRaises(ValueError):
            relevance.RelevanceScore(
                matched_keywords=(), overlap_count=0, citation_count=-1
            )


if __name__ == "__main__":
    unittest.main()
