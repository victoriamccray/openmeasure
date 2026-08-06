"""
Unit tests for shared/handoff.py

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest

import pandas as pd

from shared.handoff import (
    HANDOFF_SCHEMA_VERSION,
    KIND_CELLS_EMPTY,
    KIND_ROWS_DROPPED,
    STORE_KEY,
    DatasetFingerprint,
    ExclusionAccount,
    HandoffEntry,
    HandoffStore,
    RetentionItem,
    fingerprint_dataframe,
    group_by_dataset,
)


def account(module="reliability", label="Reliability", used=90, total=100, **kwargs):
    return ExclusionAccount(
        module=module,
        analysis_label=label,
        columns_considered=("a", "b"),
        n_input_rows=total,
        n_retained_rows=used,
        **kwargs,
    )


def fingerprint(digest="abc123", filename="data.csv"):
    return DatasetFingerprint(
        digest=digest,
        filename=filename,
        n_rows=100,
        n_columns=2,
        column_names=("a", "b"),
    )


class TestFingerprintDataframe(unittest.TestCase):
    def test_identical_frames_share_a_digest(self):
        first = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        second = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        self.assertEqual(
            fingerprint_dataframe(first).digest,
            fingerprint_dataframe(second).digest,
        )

    def test_changed_value_changes_the_digest(self):
        first = pd.DataFrame({"a": [1, 2, 3]})
        second = pd.DataFrame({"a": [1, 2, 4]})

        self.assertNotEqual(
            fingerprint_dataframe(first).digest,
            fingerprint_dataframe(second).digest,
        )

    def test_renamed_column_changes_the_digest(self):
        first = pd.DataFrame({"a": [1, 2, 3]})
        second = pd.DataFrame({"renamed": [1, 2, 3]})

        self.assertNotEqual(
            fingerprint_dataframe(first).digest,
            fingerprint_dataframe(second).digest,
        )

    def test_filename_does_not_affect_the_digest(self):
        # The digest identifies the data, so the same data saved under two
        # names must group together.
        frame = pd.DataFrame({"a": [1, 2, 3]})

        self.assertEqual(
            fingerprint_dataframe(frame, "one.csv").digest,
            fingerprint_dataframe(frame, "two.csv").digest,
        )

    def test_records_shape_and_columns(self):
        frame = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        result = fingerprint_dataframe(frame, "data.csv")

        self.assertEqual(result.n_rows, 3)
        self.assertEqual(result.n_columns, 2)
        self.assertEqual(result.column_names, ("a", "b"))
        self.assertEqual(result.filename, "data.csv")

    def test_label_and_short_digest_are_display_friendly(self):
        result = fingerprint_dataframe(pd.DataFrame({"a": [1]}), "study.csv")

        self.assertEqual(len(result.short_digest), 8)
        self.assertIn("study.csv", result.label)


class TestExclusionAccount(unittest.TestCase):
    def test_excluded_rows_are_derived(self):
        self.assertEqual(account(used=90, total=100).n_excluded_rows, 10)

    def test_retaining_everything_excludes_nothing(self):
        self.assertEqual(account(used=100, total=100).n_excluded_rows, 0)

    def test_retaining_nothing_excludes_everything(self):
        self.assertEqual(account(used=0, total=100).n_excluded_rows, 100)

    def test_negative_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            account(used=-1, total=100)

        with self.assertRaises(ValueError):
            account(used=10, total=-100)

    def test_retaining_more_than_received_is_rejected(self):
        with self.assertRaises(ValueError) as context:
            account(used=120, total=100)

        self.assertIn("cannot retain more rows", str(context.exception))

    def test_missing_retained_count_is_rejected(self):
        with self.assertRaises(ValueError) as context:
            ExclusionAccount(
                module="m",
                analysis_label="Some analysis",
                columns_considered=("a",),
                n_input_rows=100,
            )

        self.assertIn("must report n_retained_rows", str(context.exception))


class TestRowExpandingAccount(unittest.TestCase):
    """
    A row-expanding analysis has no retained-participant count. It reports
    expanded observations instead, so nothing has to be invented and it is
    never silently reported as having excluded nothing.
    """

    def expanded(self, **kwargs):
        defaults = dict(
            module="program_evaluation",
            analysis_label="Sensitivity analysis",
            columns_considered=("race", "y"),
            n_input_rows=100,
            n_expanded_observations=137,
            rows_can_exceed_participants=True,
        )
        defaults.update(kwargs)
        return ExclusionAccount(**defaults)

    def test_reports_expanded_observations(self):
        result = self.expanded()

        self.assertEqual(result.n_expanded_observations, 137)
        self.assertEqual(result.n_input_rows, 100)

    def test_has_no_retained_or_excluded_count(self):
        result = self.expanded()

        self.assertIsNone(result.n_retained_rows)
        self.assertIsNone(result.n_excluded_rows)

    def test_expanded_count_may_exceed_participants(self):
        # One participant selecting three categories becomes three rows.
        self.assertEqual(
            self.expanded(n_input_rows=10, n_expanded_observations=30)
            .n_expanded_observations,
            30,
        )

    def test_expanding_without_an_observation_count_is_rejected(self):
        with self.assertRaises(ValueError) as context:
            self.expanded(n_expanded_observations=None)

        self.assertIn("must report n_expanded_observations", str(context.exception))

    def test_expanding_with_a_retained_count_is_rejected(self):
        # Claiming both would reintroduce the fabricated retained figure.
        with self.assertRaises(ValueError) as context:
            self.expanded(n_retained_rows=100)

        self.assertIn("must leave", str(context.exception))

    def test_negative_expanded_count_is_rejected(self):
        with self.assertRaises(ValueError):
            self.expanded(n_expanded_observations=-5)

    def test_negative_item_count_is_rejected(self):
        with self.assertRaises(ValueError):
            account(
                items=(
                    RetentionItem("bad", -3, KIND_ROWS_DROPPED, "why"),
                )
            )

    def test_unknown_item_kind_is_rejected(self):
        with self.assertRaises(ValueError) as context:
            account(items=(RetentionItem("x", 1, "NOT_A_KIND", "why"),))

        self.assertIn("unknown kind", str(context.exception))

    def test_valid_items_are_accepted(self):
        result = account(
            items=(
                RetentionItem("dropped", 10, KIND_ROWS_DROPPED, "missing group"),
                RetentionItem("empty", 4, KIND_CELLS_EMPTY, "blank cell"),
            )
        )

        self.assertEqual(len(result.items), 2)


class TestHandoffStore(unittest.TestCase):
    def setUp(self):
        self.mapping = {}
        self.store = HandoffStore(self.mapping)

    def test_records_round_trip(self):
        self.store.record("reliability", fingerprint(), account())

        entries = self.store.entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].module, "reliability")
        self.assertEqual(entries[0].exclusion.n_retained_rows, 90)

    def test_empty_store_has_no_entries(self):
        self.assertEqual(self.store.entries(), ())
        self.assertFalse(self.store.has("reliability"))

    def test_has_reports_recorded_modules(self):
        self.store.record("reliability", fingerprint(), account())

        self.assertTrue(self.store.has("reliability"))
        self.assertFalse(self.store.has("fairness"))

    def test_rerunning_a_module_replaces_its_record(self):
        # A stale result from an earlier run of the same module is never what
        # the user wants shown.
        self.store.record("reliability", fingerprint(), account(used=90))
        self.store.record("reliability", fingerprint(), account(used=50))

        entries = self.store.entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].exclusion.n_retained_rows, 50)

    def test_sequence_increases_monotonically(self):
        self.store.record("reliability", fingerprint(), account())
        self.store.record("fairness", fingerprint(), account(module="fairness"))
        self.store.record(
            "time_series_qa", fingerprint(), account(module="time_series_qa")
        )

        sequences = [entry.sequence for entry in self.store.entries()]
        self.assertEqual(sequences, sorted(sequences))
        self.assertEqual(len(set(sequences)), 3)

    def test_entries_are_returned_in_recording_order(self):
        self.store.record("second", fingerprint(), account(module="second"))
        self.store.record("first", fingerprint(), account(module="first"))

        self.assertEqual(
            [entry.module for entry in self.store.entries()],
            ["second", "first"],
        )

    def test_primary_statistics_are_stored_as_primitives(self):
        self.store.record(
            "reliability",
            fingerprint(),
            account(),
            {"cronbach_alpha": 0.82},
        )

        stats = self.store.entries()[0].primary_statistics
        self.assertEqual(stats["cronbach_alpha"], 0.82)

    def test_entry_from_a_different_schema_is_discarded(self):
        stale = HandoffEntry(
            module="reliability",
            fingerprint=fingerprint(),
            exclusion=account(),
            schema_version=HANDOFF_SCHEMA_VERSION + 1,
        )
        self.mapping[STORE_KEY] = {"reliability": stale}

        self.assertEqual(self.store.entries(), ())
        self.assertFalse(self.store.has("reliability"))

    def test_unrecognized_store_contents_are_ignored(self):
        self.mapping[STORE_KEY] = "not a dict"

        self.assertEqual(self.store.entries(), ())

    def test_non_entry_values_are_ignored(self):
        self.mapping[STORE_KEY] = {"reliability": {"module": "reliability"}}

        self.assertEqual(self.store.entries(), ())

    def test_clear_discards_everything(self):
        self.store.record("reliability", fingerprint(), account())
        self.store.clear()

        self.assertEqual(self.store.entries(), ())
        self.assertNotIn(STORE_KEY, self.mapping)

    def test_clearing_an_empty_store_is_safe(self):
        self.store.clear()

        self.assertEqual(self.store.entries(), ())


class TestGroupByDataset(unittest.TestCase):
    def test_same_dataset_groups_together(self):
        store = HandoffStore({})
        shared = fingerprint(digest="same")
        store.record("reliability", shared, account())
        store.record("program_evaluation", shared, account(module="program_evaluation"))

        grouped = group_by_dataset(store.entries())

        self.assertEqual(len(grouped), 1)
        self.assertEqual(len(grouped["same"]), 2)

    def test_different_datasets_group_separately(self):
        store = HandoffStore({})
        store.record("reliability", fingerprint(digest="one"), account())
        store.record(
            "time_series_qa",
            fingerprint(digest="two", filename="series.csv"),
            account(module="time_series_qa"),
        )

        grouped = group_by_dataset(store.entries())

        self.assertEqual(len(grouped), 2)
        self.assertEqual(len(grouped["one"]), 1)
        self.assertEqual(len(grouped["two"]), 1)

    def test_grouping_nothing_gives_nothing(self):
        self.assertEqual(group_by_dataset(()), {})


if __name__ == "__main__":
    unittest.main()
