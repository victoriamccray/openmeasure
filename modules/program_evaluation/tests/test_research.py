"""
Unit tests for core/research.py

Run with: pytest modules/program_evaluation/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import research  # noqa: E402


def _work(title, year=2020, venue="A Journal", abstract_words=None, doi="10.1/x"):
    """One OpenAlex work, shaped as the API returns it."""
    inverted = None
    if abstract_words:
        inverted = {word: [index] for index, word in enumerate(abstract_words)}

    return {
        "title": title,
        "publication_year": year,
        "doi": f"https://doi.org/{doi}",
        "cited_by_count": 3,
        "primary_location": {"source": {"display_name": venue}},
        "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
        "abstract_inverted_index": inverted,
    }


class TestResearchRows(unittest.TestCase):
    def test_one_row_per_returned_work(self):
        rows = research.research_rows(
            [_work("First study"), _work("Second study")], "appointment reminders"
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].record.title, "First study")

    def test_overlap_counts_words_shared_with_the_question(self):
        rows = research.research_rows(
            [_work("Appointment reminders in clinics")], "appointment reminders"
        )

        self.assertGreaterEqual(rows[0].overlap_count, 2)
        self.assertIn("appointment", [w.lower() for w in rows[0].matched_keywords])

    def test_a_result_sharing_nothing_scores_zero(self):
        rows = research.research_rows(
            [_work("Volcanic ash dispersal modelling")], "appointment reminders"
        )

        self.assertEqual(rows[0].overlap_count, 0)
        self.assertEqual(rows[0].matched_keywords, ())

    def test_order_is_the_order_returned_not_the_overlap_ranking(self):
        """
        Overlap counts shared words; it is not a relevance ranking, and
        sorting by it would present it as one.
        """
        works = [
            _work("Volcanic ash dispersal modelling"),
            _work("Appointment reminders in clinics"),
        ]
        rows = research.research_rows(works, "appointment reminders")

        self.assertEqual(rows[0].record.title, "Volcanic ash dispersal modelling")
        self.assertLess(rows[0].overlap_count, rows[1].overlap_count)

    def test_no_works_gives_no_rows(self):
        self.assertEqual(research.research_rows([], "anything"), ())

    def test_rows_are_frozen(self):
        self.assertTrue(research.ResearchRow.__dataclass_params__.frozen)


class TestDisplayRow(unittest.TestCase):
    def test_a_display_row_carries_the_columns_a_table_shows(self):
        rows = research.research_rows([_work("A study", year=2019)], "study")
        row = rows[0].as_display_row()

        self.assertEqual(row["Title"], "A study")
        self.assertEqual(row["Year"], 2019)
        self.assertIn(research.OVERLAP_COLUMN, row)

    def test_every_row_has_the_same_keys(self):
        rows = research.research_rows(
            [_work("One"), _work("Two", year=None)], "one"
        )
        keys = {tuple(sorted(row.as_display_row())) for row in rows}

        self.assertEqual(len(keys), 1)


class TestSelectableTitles(unittest.TestCase):
    def test_titles_are_offered_in_the_order_shown(self):
        rows = research.research_rows(
            [_work("First"), _work("Second")], "anything"
        )

        self.assertEqual(research.selectable_titles(rows), ["First", "Second"])

    def test_a_repeated_title_is_offered_once(self):
        """
        OpenAlex can return a preprint and its published version under the
        same title, and a picker with two identical options cannot tell a
        reader which one they chose.
        """
        rows = research.research_rows(
            [_work("Same title", doi="10.1/a"), _work("Same title", doi="10.1/b")],
            "anything",
        )

        self.assertEqual(research.selectable_titles(rows), ["Same title"])

    def test_no_rows_gives_no_titles(self):
        self.assertEqual(research.selectable_titles(()), [])


if __name__ == "__main__":
    unittest.main()
