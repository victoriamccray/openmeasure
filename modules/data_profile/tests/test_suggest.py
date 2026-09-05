"""
Unit tests for core/suggest.py

Run with: pytest modules/data_profile/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from core import profile, suggest  # noqa: E402


def _suggested_workflows(df: pd.DataFrame) -> set[str]:
    result = suggest.suggest_workflows(profile.profile_dataframe(df))
    return {s.workflow for s in result}


class TestSuggestTimeSeriesQA(unittest.TestCase):
    def test_a_datetime_like_column_suggests_time_series_qa(self):
        df = pd.DataFrame({
            "recorded_at": pd.date_range("2024-01-01", periods=10),
            "visits": range(10),
        })
        self.assertIn(suggest.WORKFLOW_TIME_SERIES_QA, _suggested_workflows(df))

    def test_no_datetime_column_does_not_suggest_it(self):
        df = pd.DataFrame({"a": range(10), "b": range(10, 20)})
        self.assertNotIn(suggest.WORKFLOW_TIME_SERIES_QA, _suggested_workflows(df))


class TestSuggestReliability(unittest.TestCase):
    def test_three_or_more_scale_like_columns_suggest_reliability(self):
        df = pd.DataFrame({
            "q1": [1, 2, 3, 4, 5, 1, 2, 3],
            "q2": [2, 3, 4, 5, 1, 2, 3, 4],
            "q3": [1, 1, 2, 2, 3, 3, 4, 4],
            "participant_id": range(8),
        })
        self.assertIn(suggest.WORKFLOW_RELIABILITY, _suggested_workflows(df))

    def test_two_scale_like_columns_is_not_enough(self):
        df = pd.DataFrame({
            "q1": [1, 2, 3, 4, 5, 1, 2, 3],
            "q2": [2, 3, 4, 5, 1, 2, 3, 4],
            "participant_id": range(8),
        })
        self.assertNotIn(suggest.WORKFLOW_RELIABILITY, _suggested_workflows(df))


class TestSuggestImpactEvaluation(unittest.TestCase):
    def test_categorical_group_plus_continuous_outcome_suggests_it(self):
        df = pd.DataFrame({
            "group": ["treatment", "control"] * 10,
            "score": [float(i) + 0.5 for i in range(20)],
        })
        self.assertIn(suggest.WORKFLOW_IMPACT_EVALUATION, _suggested_workflows(df))

    def test_no_continuous_column_does_not_suggest_it(self):
        df = pd.DataFrame({
            "group": ["treatment", "control"] * 10,
            "outcome": ["yes", "no"] * 10,
        })
        self.assertNotIn(suggest.WORKFLOW_IMPACT_EVALUATION, _suggested_workflows(df))


class TestSuggestFairness(unittest.TestCase):
    def test_group_plus_binary_outcome_suggests_fairness(self):
        df = pd.DataFrame({
            "sex": ["A", "B"] * 10,
            "approved": ["yes", "no"] * 10,
        })
        self.assertIn(suggest.WORKFLOW_FAIRNESS, _suggested_workflows(df))

    def test_single_categorical_column_does_not_suggest_fairness(self):
        df = pd.DataFrame({"approved": ["yes", "no"] * 10})
        self.assertNotIn(suggest.WORKFLOW_FAIRNESS, _suggested_workflows(df))


class TestBothImpactAndFairnessCanCoOccur(unittest.TestCase):
    def test_a_group_binary_outcome_and_continuous_column_matches_both(self):
        # This is the deliberately ambiguous shape: column dtypes alone
        # cannot tell "was this a program effect" from "is this a
        # fairness concern" apart, so both should be suggested together.
        # score's values are deliberately non-sequential (not a plain
        # 0..N run), so the profiler reads it as continuous rather than
        # coincidentally identifier-like.
        df = pd.DataFrame({
            "group": ["A", "B"] * 10,
            "approved": ["yes", "no"] * 10,
            "score": [round((i * 1.37) % 23 + 0.5, 2) for i in range(20)],
        })
        workflows = _suggested_workflows(df)
        self.assertIn(suggest.WORKFLOW_IMPACT_EVALUATION, workflows)
        self.assertIn(suggest.WORKFLOW_FAIRNESS, workflows)


class TestNoMatch(unittest.TestCase):
    def test_a_single_plain_numeric_column_matches_nothing(self):
        df = pd.DataFrame({"id": range(20)})
        self.assertEqual(suggest.suggest_workflows(profile.profile_dataframe(df)), ())


class TestWorkflowSuggestionValidation(unittest.TestCase):
    def test_missing_reasoning_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            suggest.WorkflowSuggestion(
                workflow="Reliability", reasoning="", matched_columns=("q1",)
            )

    def test_no_matched_columns_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            suggest.WorkflowSuggestion(
                workflow="Reliability", reasoning="Some reason.", matched_columns=()
            )


if __name__ == "__main__":
    unittest.main()


class TestDefaultColumns(unittest.TestCase):
    """
    Which column a picker opens on.

    The ordinary case matters less than the awkward ones: these are what
    decide whether a zero-friction visitor lands on a meaningful analysis
    or on a row number compared against itself.
    """

    @staticmethod
    def _profile(frame):
        return profile.profile_dataframe(frame)

    def test_prefers_a_numeric_outcome_over_an_identifier(self):
        frame = pd.DataFrame({
            "participant_id": [1, 2, 3, 4, 5, 6],
            "arm": ["a", "b"] * 3,
            "score": [3, 5, 4, 2, 5, 1],
        })
        prof = self._profile(frame)

        self.assertEqual(
            suggest.default_outcome_column(prof, list(frame.columns)), "score"
        )

    def test_prefers_a_nonnumeric_grouping_over_a_numeric_one(self):
        frame = pd.DataFrame({
            "participant_id": [1, 2, 3, 4, 5, 6],
            "wave": [1, 2, 1, 2, 1, 2],
            "arm": ["a", "b"] * 3,
        })
        prof = self._profile(frame)

        self.assertEqual(
            suggest.default_group_column(prof, list(frame.columns)), "arm"
        )

    def test_a_likert_outcome_is_preferred_though_the_profile_calls_it_categorical(self):
        """
        profile_dataframe marks a 1-to-5 Likert column categorical-like,
        while recommend.py treats any numeric column with three or more
        distinct values as continuous-like. For an outcome the recommender
        is the right authority, since it is what decides which test runs.
        """
        frame = pd.DataFrame({
            "arm": ["a", "b"] * 4,
            "satisfaction": [1, 2, 3, 4, 5, 4, 3, 2],
        })
        prof = self._profile(frame)

        self.assertEqual(
            prof.columns[1].role, profile.ROLE_CATEGORICAL, "test premise changed"
        )
        self.assertEqual(
            suggest.default_outcome_column(prof, list(frame.columns)), "satisfaction"
        )

    def test_every_numeric_column_being_an_identifier_still_returns_something(self):
        """
        Nothing here is a good outcome. Returning None would leave the
        picker with no default at all; the recommender warns about an
        identifier-like selection separately.
        """
        frame = pd.DataFrame({
            "participant_id": [1, 2, 3, 4],
            "record_id": [10, 11, 12, 13],
            "site": ["x", "y", "x", "y"],
        })
        prof = self._profile(frame)
        chosen = suggest.default_outcome_column(prof, list(frame.columns))

        self.assertIsNotNone(chosen)
        self.assertEqual(chosen, "site")

    def test_all_columns_identifier_like_falls_back_to_the_first_option(self):
        frame = pd.DataFrame({
            "participant_id": [1, 2, 3, 4],
            "record_id": [10, 11, 12, 13],
        })
        prof = self._profile(frame)

        self.assertEqual(
            suggest.default_outcome_column(prof, list(frame.columns)),
            "participant_id",
        )

    def test_several_identifier_like_columns_are_all_skipped(self):
        frame = pd.DataFrame({
            "participant_id": [1, 2, 3, 4],
            "uuid": ["a", "b", "c", "d"],
            "row_index": [0, 1, 2, 3],
            "arm": ["x", "y", "x", "y"],
            "score": [5, 3, 5, 2],
        })
        prof = self._profile(frame)
        options = list(frame.columns)

        self.assertEqual(suggest.default_outcome_column(prof, options), "score")
        self.assertEqual(suggest.default_group_column(prof, options), "arm")

    def test_a_binary_outcome_is_not_preferred_as_an_outcome(self):
        """Two distinct values is a grouping, not a quantity to compare."""
        frame = pd.DataFrame({
            "applied": ["yes", "no"] * 4,
            "flag": [0, 1] * 4,
            "score": [3, 5, 4, 2, 5, 1, 4, 2],
        })
        prof = self._profile(frame)

        self.assertEqual(
            suggest.default_outcome_column(prof, list(frame.columns)), "score"
        )

    def test_a_sequential_integer_column_is_not_preferred_as_an_outcome(self):
        """
        An all-unique run of consecutive integers is the signature of a
        row number, and profile_dataframe classifies it identifier-like
        whatever it is called. A real outcome in the same frame wins.
        """
        frame = pd.DataFrame({
            "row": [1, 2, 3, 4, 5, 6],
            "score": [3, 5, 4, 2, 5, 1],
        })
        prof = self._profile(frame)

        self.assertEqual(prof.columns[0].role, profile.ROLE_IDENTIFIER)
        self.assertEqual(
            suggest.default_outcome_column(prof, list(frame.columns)), "score"
        )

    def test_empty_options_return_none(self):
        prof = self._profile(pd.DataFrame({"a": [1, 2, 3]}))

        self.assertIsNone(suggest.default_outcome_column(prof, []))
        self.assertIsNone(suggest.default_group_column(prof, []))

    def test_a_column_absent_from_the_profile_is_never_chosen(self):
        """A picker's options can be a subset; an unknown name is not one."""
        frame = pd.DataFrame({"arm": ["a", "b"] * 3, "score": [1, 2, 3, 4, 5, 6]})
        prof = self._profile(frame)

        self.assertEqual(suggest.default_outcome_column(prof, ["score"]), "score")


class TestDefaultPrePostColumns(unittest.TestCase):
    def test_matches_a_named_pair_rather_than_two_unrelated_columns(self):
        frame = pd.DataFrame({
            "participant_id": [1, 2, 3, 4],
            "satisfaction": [5, 3, 4, 2],
            "pre_confidence": [7, 5, 6, 4],
            "post_confidence": [8, 6, 7, 5],
        })
        prof = profile.profile_dataframe(frame)

        self.assertEqual(
            suggest.default_prepost_columns(prof, list(frame.columns)),
            ("pre_confidence", "post_confidence"),
        )

    def test_a_shared_remainder_is_required_not_just_the_prefixes(self):
        """
        pretest_score and post_weight share a prefix family but not a
        remainder, so they are not a repeated measure of one thing. The
        genuine pair in the same frame must win over them.
        """
        frame = pd.DataFrame({
            "pretest_score": [3, 5, 3, 2],
            "post_weight": [8, 6, 8, 5],
            "pre_score": [4, 6, 4, 3],
            "post_score": [7, 9, 7, 6],
        })
        prof = profile.profile_dataframe(frame)
        baseline, follow_up = suggest.default_prepost_columns(prof, list(frame.columns))

        self.assertEqual((baseline, follow_up), ("pre_score", "post_score"))

    def test_baseline_prefix_is_also_matched(self):
        frame = pd.DataFrame({
            "baseline_score": [3, 5, 3, 2],
            "followup_score": [8, 6, 8, 5],
        })
        prof = profile.profile_dataframe(frame)

        self.assertEqual(
            suggest.default_prepost_columns(prof, list(frame.columns)),
            ("baseline_score", "followup_score"),
        )

    def test_falls_back_to_two_different_outcome_like_columns(self):
        frame = pd.DataFrame({
            "participant_id": [1, 2, 3, 4],
            "first": [3, 5, 3, 2],
            "second": [8, 6, 8, 5],
        })
        prof = profile.profile_dataframe(frame)
        baseline, follow_up = suggest.default_prepost_columns(prof, list(frame.columns))

        self.assertNotEqual(baseline, follow_up)
        self.assertNotIn("participant_id", (baseline, follow_up))

    def test_a_single_column_yields_no_second_choice(self):
        frame = pd.DataFrame({"only": [3, 5, 3, 2]})
        prof = profile.profile_dataframe(frame)

        self.assertEqual(
            suggest.default_prepost_columns(prof, ["only"]), ("only", None)
        )
