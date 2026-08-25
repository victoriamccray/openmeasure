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
