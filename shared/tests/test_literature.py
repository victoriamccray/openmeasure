"""
Unit tests for shared/literature.py

Run with: pytest shared/tests/ -v

Only the pure half is covered here. search_openalex makes a network call
and is wrapped in st.cache_data; what is testable without a network is
how a researcher's wording becomes a query, which is where this was
failing.
"""

from __future__ import annotations

import pathlib
import unittest

from shared import literature
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


class TestDerivingAQueryFromAFinding(unittest.TestCase):
    """
    A researcher types a finding; OpenAlex is a keyword index over titles
    and abstracts. "An increase in financial security" sent whole matched
    securities markets and bank security, because three of its five words
    carry no topic and one of the remaining two belongs to two
    literatures.

    What is under test is restraint. The derivation drops words from two
    named lists and reports every one; it does not stem, reorder, expand
    synonyms, or rank by anything. A page that silently rewrote a query
    would be worse than the noisy one, because a reader could no longer
    tell why the results looked like that.
    """

    def test_it_drops_function_words_and_keeps_the_topic(self):
        derived = literature.derive_query("an increase in financial security")

        self.assertEqual(derived.terms, "financial security")

    def test_it_names_the_framing_words_it_dropped(self):
        derived = literature.derive_query(
            "Does the program increase financial security among adults?"
        )

        self.assertIn("program", derived.dropped_framing_words)
        self.assertIn("increase", derived.dropped_framing_words)

    def test_it_separates_framing_from_function_words(self):
        """
        Dropped for different reasons, so reported separately: one is
        grammar and the other is how a research question is phrased.
        """
        derived = literature.derive_query(
            "Does the program increase financial security"
        )

        self.assertIn("Does", derived.dropped_function_words)
        self.assertIn("increase", derived.dropped_framing_words)
        self.assertNotIn("increase", derived.dropped_function_words)

    def test_it_keeps_the_finding_as_it_was_written(self):
        finding = "Our intervention reduced violence among grade 6 students"
        derived = literature.derive_query(finding)

        self.assertEqual(derived.finding, finding)

    def test_it_warns_where_a_surviving_word_spans_literatures(self):
        derived = literature.derive_query("an increase in financial security")
        flagged = [term for term, _ in derived.ambiguous]

        self.assertEqual(flagged, ["security"])
        self.assertTrue(
            any("financial securities" in note for note in derived.notes())
        )

    def test_it_warns_when_too_little_survives_to_match_on(self):
        derived = literature.derive_query("treatment adherence improved")

        self.assertTrue(derived.is_thin)
        self.assertTrue(
            any("will be chance" in note for note in derived.notes()),
            derived.notes(),
        )

    def test_a_longer_finding_is_not_called_thin(self):
        derived = literature.derive_query(
            "violence among grade 6 students in single-sex schools"
        )

        self.assertFalse(derived.is_thin)

    def test_it_refuses_a_finding_with_no_topic_in_it(self):
        with self.assertRaises(ValueError) as raised:
            literature.derive_query("did the effect increase")

        self.assertIn("nothing topical left", str(raised.exception))

    def test_it_adds_nothing_the_researcher_did_not_write(self):
        """
        No synonym expansion and no reordering: every surviving term is
        one of the original words, in the order they were written.
        """
        finding = "reading comprehension in bilingual children"
        derived = literature.derive_query(finding)
        surviving = [term.lower() for term in derived.terms.split()]

        # Every surviving term is one of the words written, in the order
        # written, and "in" is gone rather than something new added.
        self.assertEqual(
            surviving, ["reading", "comprehension", "bilingual", "children"]
        )
        self.assertTrue(
            set(surviving) <= set(finding.lower().split()), surviving
        )

    def test_the_ambiguous_list_is_presented_as_incomplete(self):
        """
        A word missing from it produces a noisy search, which is a
        smaller problem than a claim that a search is clean.
        """
        source = (
            pathlib.Path(literature.__file__).read_text(encoding="utf-8")
        )

        self.assertIn("certainly", source)
        self.assertIn("incomplete", source)
