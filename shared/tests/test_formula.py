"""
Unit tests for shared/formula.py

The model's job is to make a formula explanation that cannot lie: the
arithmetic on screen is assembled from the same terms the anatomy lists,
so the two cannot disagree. These cover the validation that enforces it.

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest

from shared.formula import FormulaExplanation, FormulaTerm


def _term(key="a", value="1.0", **overrides):
    fields = {
        "key": key,
        "symbol": "x",
        "plain_name": "a quantity",
        "meaning": "What it is.",
        "display_value": value,
    }
    fields.update(overrides)
    return FormulaTerm(**fields)


def _explanation(**overrides):
    fields = {
        "name": "Test statistic",
        "blocks": ("Numerator", "/", "Denominator", "=", "Result"),
        "substitution_template": "{a} / {b}",
        "formal_latex": "t = a / b",
        "terms": (_term("a", "6.0"), _term("b", "3.0")),
        "result_display": "2.00",
        "reading": "Twice as large.",
    }
    fields.update(overrides)
    return FormulaExplanation(**fields)


class TestSubstitution(unittest.TestCase):
    def test_the_substituted_line_is_built_from_the_terms(self):
        self.assertEqual(_explanation().substituted, "6.0 / 3.0 = 2.00")

    def test_changing_a_term_changes_the_line(self):
        """
        The point of generating the line: a value cannot be shown in the
        arithmetic that disagrees with the term it came from.
        """
        explanation = _explanation(terms=(_term("a", "9.0"), _term("b", "3.0")))

        self.assertIn("9.0", explanation.substituted)
        self.assertNotIn("6.0", explanation.substituted)

    def test_the_concept_line_reads_as_words_with_no_notation(self):
        self.assertEqual(
            _explanation().concept, "Numerator / Denominator = Result"
        )


class TestTermLookup(unittest.TestCase):
    def test_a_term_can_be_fetched_by_key(self):
        self.assertEqual(_explanation().term("b").display_value, "3.0")

    def test_an_unknown_term_raises_and_names_the_known_ones(self):
        with self.assertRaises(ValueError) as raised:
            _explanation().term("c")

        message = str(raised.exception)
        self.assertIn("'c'", message)
        self.assertIn("a", message)


class TestValidation(unittest.TestCase):
    def test_a_placeholder_with_no_term_is_rejected(self):
        """
        A number in the arithmetic that a reader cannot look up is the
        thing this model exists to prevent.
        """
        with self.assertRaises(ValueError) as raised:
            _explanation(substitution_template="{a} / {b} + {c}")

        self.assertIn("no term defines", str(raised.exception))

    def test_a_term_that_never_appears_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            _explanation(
                terms=(_term("a", "6.0"), _term("b", "3.0"), _term("c", "1.0"))
            )

        self.assertIn("never use", str(raised.exception))

    def test_a_repeated_term_key_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            _explanation(terms=(_term("a", "6.0"), _term("a", "3.0")))

        self.assertIn("same term key twice", str(raised.exception))

    def test_no_terms_is_rejected(self):
        with self.assertRaises(ValueError):
            _explanation(terms=())

    def test_no_concept_blocks_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            _explanation(blocks=())

        self.assertIn("no concept blocks", str(raised.exception))

    def test_every_required_field_must_be_populated(self):
        for field_name in (
            "name",
            "substitution_template",
            "formal_latex",
            "result_display",
            "reading",
        ):
            with self.subTest(field=field_name):
                with self.assertRaises(ValueError):
                    _explanation(**{field_name: ""})

    def test_a_term_missing_a_required_field_is_rejected(self):
        for field_name in (
            "key",
            "symbol",
            "plain_name",
            "meaning",
            "display_value",
        ):
            with self.subTest(field=field_name):
                with self.assertRaises(ValueError):
                    _term(**{field_name: ""})

    def test_source_is_optional(self):
        """
        Provenance is prose for a reader and not every term has a place to
        point at. Structured provenance for replication is a separate
        concern and deliberately not this field.
        """
        self.assertEqual(_term().source, "")


class TestFrozen(unittest.TestCase):
    def test_both_models_are_frozen(self):
        self.assertTrue(FormulaTerm.__dataclass_params__.frozen)
        self.assertTrue(FormulaExplanation.__dataclass_params__.frozen)


class TestNoFrameworkDependency(unittest.TestCase):
    def test_the_model_does_not_import_streamlit(self):
        """
        Module core/ files import this to describe their own statistics,
        and core/ must stay framework-independent. The rendering half
        lives in shared/report.py instead.
        """
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "formula.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("import streamlit", source)
        self.assertNotIn("from streamlit", source)


if __name__ == "__main__":
    unittest.main()
