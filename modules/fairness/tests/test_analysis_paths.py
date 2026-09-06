"""
Unit tests for modules/fairness/core/analysis_paths.py

Run with: pytest modules/fairness/tests/ -v

The thing being tested is a boundary: which of two genuinely different
analyses a set of columns can support. Most of these check that the
boundary holds where it should, because the failure mode is a page
offering an analysis the data cannot carry and rejecting the reader's
selection two steps later.
"""

from __future__ import annotations

import os
import sys
import unittest

# The module's own core, then the repository root, matching the other
# test modules here.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import pandas as pd  # noqa: E402

from core import analysis_paths  # noqa: E402
from modules.data_profile.core.profile import profile_dataframe  # noqa: E402


def _read(frame):
    return analysis_paths.read_paths(profile_dataframe(frame))


class TestBothPathsAvailable(unittest.TestCase):
    """
    The module's own sample: an observed label, a model's decision and a
    group, all with two values.
    """

    def setUp(self):
        self.reading = _read(
            pd.DataFrame(
                {
                    "true_label": [i % 2 for i in range(60)],
                    "predicted_label": [(i + 1) % 2 for i in range(60)],
                    "predicted_probability": [i / 60 for i in range(60)],
                    "sex": ["Female", "Male"] * 30,
                }
            )
        )

    def test_both_paths_are_open(self):
        self.assertTrue(self.reading.pre_model.available)
        self.assertTrue(self.reading.post_model.available)

    def test_a_many_valued_numeric_column_is_a_score_not_a_decision(self):
        """
        predicted_probability had 130 distinct values in the real sample
        and was still offered as the model's binary decision, which the
        analysis then rejected.
        """
        self.assertIn("predicted_probability", self.reading.shapes.scores)
        self.assertNotIn("predicted_probability", self.reading.shapes.binary)

    def test_the_score_is_named_and_explained_rather_than_hidden(self):
        explanation = self.reading.post_model.explanation

        self.assertIn("predicted_probability", explanation)
        self.assertIn("threshold", explanation)

    def test_an_available_decision_column_is_not_claimed_to_be_one(self):
        """
        Shape cannot tell a model's output from anything else recorded
        with two values, and the page must not imply otherwise.
        """
        self.assertIn(
            analysis_paths.DECISION_PROVENANCE_NOTE,
            self.reading.post_model.explanation,
        )


class TestOnlyObservedData(unittest.TestCase):
    """
    A file recording what happened to real people and no predictions.
    This is the Diabetes 130-hospitals shape, and the correct answer is
    one path open and one closed rather than a way to invent the other.
    """

    def setUp(self):
        self.reading = _read(
            pd.DataFrame(
                {
                    "readmitted": [i % 2 for i in range(60)],
                    "race": ["A", "B", "C"] * 20,
                    "time_in_hospital": [i % 14 for i in range(60)],
                }
            )
        )

    def test_the_pre_model_path_is_open(self):
        self.assertTrue(self.reading.pre_model.available)

    def test_the_post_model_path_is_closed(self):
        self.assertFalse(self.reading.post_model.available)

    def test_the_closed_path_says_what_is_missing(self):
        explanation = self.reading.post_model.explanation.lower()

        self.assertIn("not established from these data", explanation)
        self.assertIn("two-valued column", explanation)

    def test_a_closed_path_is_not_described_as_a_fault_in_the_data(self):
        text = " ".join(
            (
                self.reading.post_model.status_label,
                self.reading.post_model.explanation,
            )
        ).lower()

        for blame in ("invalid", "unsuitable", "insufficient", "poor", "bad"):
            with self.subTest(word=blame):
                self.assertNotIn(blame, text)


class TestNeitherPathAvailable(unittest.TestCase):
    def test_one_column_supports_nothing(self):
        reading = _read(pd.DataFrame({"approved": [i % 2 for i in range(40)]}))

        self.assertFalse(reading.pre_model.available)
        self.assertFalse(reading.post_model.available)

    def test_the_two_paths_word_an_absence_differently(self):
        """
        Columns that cannot support a comparison is a fact about the
        columns; a model comparison the data do not carry is a claim the
        data do not make. Flattening both into "unavailable" loses that.
        """
        reading = _read(pd.DataFrame({"approved": [i % 2 for i in range(40)]}))

        self.assertNotEqual(
            reading.pre_model.status_label, reading.post_model.status_label
        )


class TestShapesOverlapCorrectly(unittest.TestCase):
    def test_a_two_valued_group_counts_as_both_binary_and_grouping(self):
        reading = _read(
            pd.DataFrame({"approved": [i % 2 for i in range(40)], "sex": ["F", "M"] * 20})
        )

        self.assertIn("sex", reading.shapes.binary)
        self.assertIn("sex", reading.shapes.grouping)

    def test_two_columns_that_are_the_same_two_do_not_open_the_post_path(self):
        """
        Counting the lists would say two binary columns and two grouping
        candidates is enough for three roles. Assigning them shows it is
        not.
        """
        reading = _read(
            pd.DataFrame({"approved": [i % 2 for i in range(40)], "sex": ["F", "M"] * 20})
        )

        self.assertTrue(reading.pre_model.available)
        self.assertFalse(reading.post_model.available)

    def test_an_identifier_fills_no_role(self):
        reading = _read(
            pd.DataFrame(
                {
                    "participant_id": range(1, 41),
                    "approved": [i % 2 for i in range(40)],
                    "sex": ["F", "M"] * 20,
                }
            )
        )

        for shapes in (
            reading.shapes.binary,
            reading.shapes.grouping,
            reading.shapes.scores,
        ):
            self.assertNotIn("participant_id", shapes)

    def test_a_three_valued_column_is_not_binary(self):
        """
        Treating three values as binary would mean dropping or merging
        one of them without saying so.
        """
        reading = _read(
            pd.DataFrame({"outcome": ["yes", "no", "maybe"] * 20, "sex": ["F", "M"] * 30})
        )

        self.assertNotIn("outcome", reading.shapes.binary)
        self.assertIn("outcome", reading.shapes.grouping)


class TestPathObjects(unittest.TestCase):
    def test_a_path_missing_its_wording_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            analysis_paths.AnalysisPath(
                id="x",
                label="X",
                question="A question?",
                needs=("Something",),
                available=True,
                status_label="",
                explanation="Because.",
            )

        self.assertIn("status_label", str(raised.exception))

    def test_a_path_that_needs_nothing_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            analysis_paths.AnalysisPath(
                id="x",
                label="X",
                question="A question?",
                needs=(),
                available=True,
                status_label="Available",
                explanation="Because.",
            )

        self.assertIn("names nothing it needs", str(raised.exception))

    def test_the_reading_lists_pre_model_first(self):
        reading = _read(
            pd.DataFrame({"approved": [i % 2 for i in range(40)], "sex": ["F", "M"] * 20})
        )

        self.assertEqual(
            [path.id for path in reading.paths],
            [analysis_paths.PATH_PRE_MODEL, analysis_paths.PATH_POST_MODEL],
        )


if __name__ == "__main__":
    unittest.main()
