"""
Unit tests for core/retention.py

Run with: pytest modules/validation_chain/tests/ -v
"""

from __future__ import annotations

import dataclasses
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.retention import (  # noqa: E402
    DatasetRetention,
    RetentionSummary,
    summarize_retention,
)
from shared.handoff import (  # noqa: E402
    KIND_CELLS_EMPTY,
    KIND_ROWS_DROPPED,
    DatasetFingerprint,
    ExclusionAccount,
    RetentionItem,
)


def fingerprint(digest="abc123", filename="data.csv"):
    return DatasetFingerprint(
        digest=digest,
        filename=filename,
        n_rows=100,
        n_columns=3,
        column_names=("a", "b", "c"),
    )


def account(label, used, total=100, **kwargs):
    return ExclusionAccount(
        module=label.lower().replace(" ", "_"),
        analysis_label=label,
        columns_considered=("a", "b"),
        n_input_rows=total,
        n_retained_rows=used,
        **kwargs,
    )


class TestRetentionArithmetic(unittest.TestCase):
    def test_counts_are_reported_per_analysis(self):
        summary = summarize_retention(
            {
                fingerprint(): (
                    account("Reliability", used=90),
                    account("Group comparison", used=65),
                )
            }
        )

        self.assertEqual(summary.n_analyses, 2)
        self.assertEqual(summary.n_datasets, 1)

        accounts = summary.datasets[0].accounts
        self.assertEqual(accounts[0].n_retained_rows, 90)
        self.assertEqual(accounts[0].n_excluded_rows, 10)
        self.assertEqual(accounts[1].n_retained_rows, 65)
        self.assertEqual(accounts[1].n_excluded_rows, 35)

    def test_analysis_retaining_everything(self):
        summary = summarize_retention(
            {fingerprint(): (account("Full", used=100),)}
        )

        self.assertEqual(summary.datasets[0].accounts[0].n_excluded_rows, 0)
        self.assertEqual(summary.datasets[0].smallest_retained_n, 100)

    def test_analysis_retaining_nothing(self):
        summary = summarize_retention(
            {fingerprint(): (account("Empty", used=0),)}
        )

        self.assertEqual(summary.datasets[0].accounts[0].n_excluded_rows, 100)
        self.assertEqual(summary.datasets[0].smallest_retained_n, 0)

    def test_smallest_retained_identifies_the_right_analysis(self):
        summary = summarize_retention(
            {
                fingerprint(): (
                    account("Reliability", used=90),
                    account("Group comparison", used=65),
                    account("Fairness", used=78),
                )
            }
        )

        dataset = summary.datasets[0]
        self.assertEqual(dataset.smallest_retained_n, 65)
        self.assertEqual(dataset.smallest_retained_analysis, "Group comparison")

    def test_retention_items_are_preserved_with_their_mechanisms(self):
        summary = summarize_retention(
            {
                fingerprint(): (
                    account(
                        "Reliability",
                        used=88,
                        items=(
                            RetentionItem(
                                "Rows dropped", 12, KIND_ROWS_DROPPED,
                                "missing value in a selected item",
                            ),
                            RetentionItem(
                                "Empty cells", 5, KIND_CELLS_EMPTY,
                                "blank cell within a retained row",
                            ),
                        ),
                    ),
                )
            }
        )

        items = summary.datasets[0].accounts[0].items
        self.assertEqual(len(items), 2)
        # Different kinds must stay distinguishable rather than being summed.
        self.assertNotEqual(items[0].kind, items[1].kind)


def expanded_account(label="Sensitivity analysis", total=100, expanded=140):
    """An analysis with expanded observations and no retained-participant count."""
    return ExclusionAccount(
        module="program_evaluation",
        analysis_label=label,
        columns_considered=("race", "y"),
        n_input_rows=total,
        n_expanded_observations=expanded,
        rows_can_exceed_participants=True,
    )


class TestRowExpandingAnalyses(unittest.TestCase):
    def test_expanding_analysis_is_excluded_from_the_comparison(self):
        summary = summarize_retention(
            {
                fingerprint(): (
                    account("Group comparison", used=80),
                    expanded_account(),
                )
            }
        )

        dataset = summary.datasets[0]
        self.assertEqual(dataset.smallest_retained_n, 80)
        self.assertEqual(dataset.smallest_retained_analysis, "Group comparison")
        self.assertEqual(
            dataset.incomparable_analyses, ("Sensitivity analysis",)
        )

    def test_expanding_analysis_is_still_reported(self):
        summary = summarize_retention({fingerprint(): (expanded_account(),)})

        self.assertEqual(summary.n_analyses, 1)
        self.assertEqual(len(summary.datasets[0].accounts), 1)

    def test_expanded_observation_count_survives_the_summary(self):
        summary = summarize_retention(
            {fingerprint(): (expanded_account(total=100, expanded=137),)}
        )

        reported = summary.datasets[0].accounts[0]
        self.assertEqual(reported.n_expanded_observations, 137)
        self.assertIsNone(reported.n_retained_rows)
        self.assertIsNone(reported.n_excluded_rows)

    def test_only_expanding_analyses_leaves_no_comparable_smallest(self):
        summary = summarize_retention({fingerprint(): (expanded_account(),)})

        dataset = summary.datasets[0]
        self.assertIsNone(dataset.smallest_retained_n)
        self.assertIsNone(dataset.smallest_retained_analysis)

    def test_expanding_analysis_never_appears_as_the_smallest(self):
        # Even when its observation count is lower than another analysis's
        # retained count, it is a different quantity and must not be ranked.
        summary = summarize_retention(
            {
                fingerprint(): (
                    account("Group comparison", used=90),
                    expanded_account(total=20, expanded=25),
                )
            }
        )

        self.assertEqual(
            summary.datasets[0].smallest_retained_analysis, "Group comparison"
        )
        self.assertEqual(summary.datasets[0].smallest_retained_n, 90)


class TestMultipleDatasets(unittest.TestCase):
    def test_datasets_are_summarized_separately(self):
        summary = summarize_retention(
            {
                fingerprint(digest="one", filename="survey.csv"): (
                    account("Reliability", used=90),
                ),
                fingerprint(digest="two", filename="series.csv"): (
                    account("Time-Series QA", used=118, total=121),
                ),
            }
        )

        self.assertEqual(summary.n_datasets, 2)
        self.assertEqual(summary.n_analyses, 2)

        labels = [item.fingerprint.filename for item in summary.datasets]
        self.assertEqual(sorted(labels), ["series.csv", "survey.csv"])

    def test_smallest_is_computed_within_each_dataset(self):
        # A small analysis on one upload must not be reported as the smallest
        # for a different upload.
        summary = summarize_retention(
            {
                fingerprint(digest="one"): (account("Big", used=95),),
                fingerprint(digest="two", filename="other.csv"): (
                    account("Small", used=10),
                ),
            }
        )

        by_digest = {
            item.fingerprint.digest: item.smallest_retained_n
            for item in summary.datasets
        }
        self.assertEqual(by_digest["one"], 95)
        self.assertEqual(by_digest["two"], 10)


class TestNoCompositeFigure(unittest.TestCase):
    """
    The summary must not collapse retention into a single figure. Rows
    dropped, empty cells, and absent observations have no shared
    denominator, and an overall percentage would be a composite score.
    """

    FORBIDDEN = ("score", "composite", "overall", "grade", "pass", "rating")

    def test_no_result_field_implies_a_composite_verdict(self):
        for cls in (RetentionSummary, DatasetRetention):
            for field in dataclasses.fields(cls):
                for term in self.FORBIDDEN:
                    with self.subTest(cls=cls.__name__, field=field.name):
                        self.assertNotIn(term, field.name.lower())

    def test_no_combined_exclusion_rate_is_reported(self):
        names = {field.name for field in dataclasses.fields(RetentionSummary)}

        self.assertNotIn("exclusion_rate", names)
        self.assertNotIn("retention_rate", names)
        self.assertNotIn("total_excluded", names)


class TestNarrative(unittest.TestCase):
    def test_shared_implication_is_present(self):
        summary = summarize_retention(
            {fingerprint(): (account("Reliability", used=90),)}
        )

        self.assertTrue(summary.shared_implication)

    def test_limitations_name_the_mechanism_and_the_overlap_gap(self):
        summary = summarize_retention(
            {fingerprint(): (account("Reliability", used=90),)}
        )

        combined = " ".join(summary.limitations).lower()
        self.assertIn("why", combined)
        self.assertIn("overlap", combined)


class TestDegenerateInput(unittest.TestCase):
    def test_no_recorded_analyses_raises(self):
        with self.assertRaises(ValueError) as context:
            summarize_retention({})

        self.assertIn("At least one", str(context.exception))

    def test_dataset_with_no_analyses_raises(self):
        with self.assertRaises(ValueError) as context:
            summarize_retention({fingerprint(): ()})

        self.assertIn("no recorded analyses", str(context.exception))


if __name__ == "__main__":
    unittest.main()
