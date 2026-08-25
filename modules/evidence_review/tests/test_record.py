"""
Unit tests for core/record.py

Run with: pytest modules/evidence_review/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import record  # noqa: E402

_SAMPLE_WORK = {
    "display_name": "Reliability of Wearable Heart-Rate Sensors",
    "publication_year": 2022,
    "cited_by_count": 41,
    "doi": "https://doi.org/10.1000/example",
    "id": "https://openalex.org/W123456789",
    "authorships": [
        {"author": {"display_name": "A. Smith"}},
        {"author": {"display_name": "B. Jones"}},
    ],
    "primary_location": {"source": {"display_name": "Journal of Sensors"}},
    "abstract_inverted_index": {
        "Wearable": [0],
        "sensors": [1],
        "show": [2],
        "moderate": [3],
        "agreement": [4],
    },
}


class TestFromOpenalexWork(unittest.TestCase):
    def test_maps_every_field(self):
        result = record.from_openalex_work(_SAMPLE_WORK)

        self.assertEqual(result.title, "Reliability of Wearable Heart-Rate Sensors")
        self.assertEqual(result.authors, ("A. Smith", "B. Jones"))
        self.assertEqual(result.year, 2022)
        self.assertEqual(result.venue, "Journal of Sensors")
        self.assertEqual(result.doi, "https://doi.org/10.1000/example")
        self.assertEqual(result.url, "https://openalex.org/W123456789")
        self.assertEqual(result.abstract, "Wearable sensors show moderate agreement")
        self.assertEqual(result.citation_count, 41)
        self.assertEqual(result.source_api, "OpenAlex")

    def test_missing_optional_fields_fall_back_rather_than_raise(self):
        result = record.from_openalex_work({"id": "https://openalex.org/W1"})

        self.assertEqual(result.title, record.UNTITLED)
        self.assertEqual(result.authors, ())
        self.assertIsNone(result.year)
        self.assertEqual(result.venue, record.UNKNOWN_VENUE)
        self.assertIsNone(result.doi)
        self.assertEqual(result.abstract, "")

    def test_no_abstract_inverted_index_gives_empty_abstract(self):
        result = record.from_openalex_work({"id": "https://openalex.org/W2", "display_name": "X"})

        self.assertEqual(result.abstract, "")


class TestLiteratureRecordValidation(unittest.TestCase):
    def _valid_kwargs(self, **overrides):
        kwargs = dict(
            title="A Study",
            authors=("A. Author",),
            year=2020,
            venue="A Journal",
            doi=None,
            url="https://openalex.org/W1",
            abstract="",
            citation_count=0,
            source_api="OpenAlex",
        )
        kwargs.update(overrides)
        return kwargs

    def test_empty_title_raises(self):
        with self.assertRaises(ValueError):
            record.LiteratureRecord(**self._valid_kwargs(title=""))

    def test_empty_url_raises(self):
        with self.assertRaises(ValueError):
            record.LiteratureRecord(**self._valid_kwargs(url=""))

    def test_negative_year_raises(self):
        with self.assertRaises(ValueError):
            record.LiteratureRecord(**self._valid_kwargs(year=-1))

    def test_negative_citation_count_raises(self):
        with self.assertRaises(ValueError):
            record.LiteratureRecord(**self._valid_kwargs(citation_count=-1))

    def test_author_summary_single_author(self):
        item = record.LiteratureRecord(**self._valid_kwargs(authors=("Only Author",)))
        self.assertEqual(item.author_summary, "Only Author")

    def test_author_summary_multiple_authors(self):
        item = record.LiteratureRecord(
            **self._valid_kwargs(authors=("First Author", "Second Author"))
        )
        self.assertEqual(item.author_summary, "First Author et al.")

    def test_author_summary_no_authors(self):
        item = record.LiteratureRecord(**self._valid_kwargs(authors=()))
        self.assertEqual(item.author_summary, "Unknown author")


if __name__ == "__main__":
    unittest.main()
