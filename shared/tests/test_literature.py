"""
Unit tests for shared/literature.py

Run with: pytest shared/tests/ -v

Only the pure half is covered here. search_openalex makes a network call
and is wrapped in st.cache_data; what is testable without a network is
how a researcher's wording becomes a query, which is where this was
failing.
"""

from __future__ import annotations

import unittest

from shared.literature import (
    OPENALEX_SEARCH_FIELD,
    searchable_terms,
)


class TestSearchableTerms(unittest.TestCase):
    """
    OpenAlex reads some punctuation as query syntax. Left in, it either
    rejects the request or quietly searches for something else.
    """

    def test_a_question_mark_is_removed(self):
        """
        The stage that uses this asks for an evaluation question, so the
        most natural thing a reader can type ended in the one character
        that guaranteed an HTTP 400 from a stemmed field.
        """
        self.assertEqual(
            searchable_terms("Did the program improve attendance?"),
            "Did the program improve attendance",
        )

    def test_an_asterisk_is_removed(self):
        self.assertEqual(searchable_terms("attend* program"), "attend program")

    def test_a_comma_becomes_a_space_rather_than_vanishing(self):
        """
        A comma separates filters, so it has to go; the words on either
        side of it are still the reader's words and stay.
        """
        self.assertEqual(
            searchable_terms("peer support, patient confidence"),
            "peer support patient confidence",
        )

    def test_filter_separators_become_spaces(self):
        for separator in (":", "|", "!"):
            with self.subTest(separator=separator):
                self.assertEqual(
                    searchable_terms(f"trial{separator}phase two"),
                    "trial phase two",
                )

    def test_ordinary_wording_is_left_alone(self):
        """
        No stemming, no stop-word removal, no reordering. What a reader
        typed is what is searched for.
        """
        question = "Did our peer support program improve patient confidence"

        self.assertEqual(searchable_terms(question), question)

    def test_whitespace_is_collapsed(self):
        self.assertEqual(searchable_terms("  a   b  "), "a b")

    def test_a_query_with_nothing_left_raises(self):
        """
        Rather than sending an empty query, which returns whatever
        OpenAlex considers the most cited works in existence.
        """
        with self.assertRaises(ValueError) as raised:
            searchable_terms("???")

        self.assertIn("no searchable words", str(raised.exception))

    def test_an_empty_query_raises(self):
        with self.assertRaises(ValueError):
            searchable_terms("   ")


class TestSearchField(unittest.TestCase):
    def test_the_search_is_restricted_to_title_and_abstract(self):
        """
        The bare `search` parameter also matches full text, which
        returned heavily cited clinical guidelines ahead of any study of
        the thing that was asked about. Pinned because the difference is
        invisible in the code and enormous in the results.
        """
        self.assertEqual(OPENALEX_SEARCH_FIELD, "title_and_abstract.search")
