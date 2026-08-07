"""
Unit tests for shared/progress.py

Two things here can be silently wrong, and both would be invisible in the UI.

A broken join makes every card read "Not assessed" forever, which looks like a
working feature reporting that you have done nothing. And the status wording
can drift into implying that a validation stage is finished, which would
contradict the caveats every module page attaches to its results. The first is
covered by joining real records; the second by asserting the state set is
closed rather than by searching for forbidden words.

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest

import pandas as pd

from shared.catalog import (
    MODULE_FAIRNESS,
    MODULE_KEYS,
    MODULE_RELIABILITY,
    WORKFLOWS,
    Workflow,
)
from shared.handoff import (
    KIND_ROWS_DROPPED,
    ExclusionAccount,
    HandoffStore,
    RetentionItem,
    fingerprint_dataframe,
)
from shared.progress import (
    ALLOWED_STATES,
    READS_OTHER_RECORDS,
    STATE_NOT_ASSESSED,
    STATE_RECORDED,
    WorkflowProgress,
    has_any_records,
    status_caption,
    workflow_progress,
)


def frame(n_rows: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "q1": list(range(n_rows)),
            "q2": list(range(n_rows)),
        }
    )


def record(store: HandoffStore, module: str, filename: str, n_rows: int = 10):
    """Record one analysis the way a page does."""

    data = frame(n_rows)

    return store.record(
        module=module,
        fingerprint=fingerprint_dataframe(data, filename),
        exclusion=ExclusionAccount(
            module=module,
            analysis_label="Test analysis",
            columns_considered=("q1", "q2"),
            n_input_rows=n_rows,
            n_retained_rows=n_rows - 1,
            items=(
                RetentionItem(
                    label="Rows excluded",
                    count=1,
                    kind=KIND_ROWS_DROPPED,
                    mechanism="Incomplete responses",
                ),
            ),
        ),
        primary_statistics={"cronbach_alpha": 0.82},
    )


def progress_for(module_key: str, entries) -> WorkflowProgress:
    """The progress entry for one workflow, found by its module key."""

    matches = [
        item
        for item in workflow_progress(entries)
        if item.workflow.module_key == module_key
    ]

    assert len(matches) == 1, f"Expected one workflow for {module_key}."

    return matches[0]


class TestJoiningRecordsToWorkflows(unittest.TestCase):
    def test_a_workflow_with_a_record_is_reported_as_recorded(self):
        store = HandoffStore({})
        record(store, MODULE_RELIABILITY, "survey.csv")

        item = progress_for(MODULE_RELIABILITY, store.entries())

        self.assertEqual(item.state, STATE_RECORDED)
        self.assertEqual(item.dataset, "survey.csv")
        self.assertTrue(item.is_recorded)

    def test_a_workflow_without_a_record_is_not_assessed(self):
        store = HandoffStore({})
        record(store, MODULE_RELIABILITY, "survey.csv")

        item = progress_for(MODULE_FAIRNESS, store.entries())

        self.assertEqual(item.state, STATE_NOT_ASSESSED)
        self.assertIsNone(item.dataset)
        self.assertFalse(item.is_recorded)

    def test_every_workflow_appears_exactly_once(self):
        # A workflow missing from the result would render a card with no
        # status while its neighbours had one, which reads as a bug.
        items = workflow_progress(())

        self.assertEqual(len(items), len(WORKFLOWS))
        self.assertEqual(
            [item.workflow.workflow for item in items],
            [workflow.workflow for workflow in WORKFLOWS],
        )

    def test_records_from_two_datasets_are_attributed_correctly(self):
        # The point of naming the dataset on the card is telling two
        # analyses apart, so the wrong filename would defeat the feature.
        store = HandoffStore({})
        record(store, MODULE_RELIABILITY, "wave_one.csv")
        record(store, MODULE_FAIRNESS, "wave_two.csv")

        entries = store.entries()

        self.assertEqual(
            progress_for(MODULE_RELIABILITY, entries).dataset, "wave_one.csv"
        )
        self.assertEqual(
            progress_for(MODULE_FAIRNESS, entries).dataset, "wave_two.csv"
        )

    def test_rerunning_a_module_reports_the_newer_dataset(self):
        # HandoffStore keeps one record per module. The status must follow it
        # rather than showing whichever record happened to be found first.
        store = HandoffStore({})
        record(store, MODULE_RELIABILITY, "first.csv")
        record(store, MODULE_RELIABILITY, "second.csv")

        item = progress_for(MODULE_RELIABILITY, store.entries())

        self.assertEqual(item.dataset, "second.csv")

    def test_a_record_matching_no_workflow_is_ignored(self):
        # There is no card to attach it to, so reporting it is not possible.
        store = HandoffStore({})
        record(store, "not_a_module", "survey.csv")

        states = {item.state for item in workflow_progress(store.entries())}

        self.assertNotIn(STATE_RECORDED, states)

    def test_every_declared_key_can_be_joined(self):
        # Guards against a constant existing with no workflow using it, which
        # would mean a page recording into a status nothing displays.
        store = HandoffStore({})

        for key in MODULE_KEYS:
            record(store, key, f"{key}.csv")

        entries = store.entries()

        for key in MODULE_KEYS:
            with self.subTest(module_key=key):
                self.assertEqual(
                    progress_for(key, entries).state, STATE_RECORDED
                )


class TestTheReaderWorkflow(unittest.TestCase):
    """
    Cross-Analysis Implications reads records rather than producing one.
    """

    def test_a_workflow_without_a_module_key_has_no_state(self):
        readers = [
            item
            for item in workflow_progress(())
            if item.workflow.module_key is None
        ]

        self.assertTrue(readers, "No reader workflow found in the catalog.")

        for item in readers:
            with self.subTest(workflow=item.workflow.workflow):
                self.assertIsNone(item.state)

    def test_a_state_of_none_is_rejected_for_a_recording_workflow(self):
        recording = next(
            workflow for workflow in WORKFLOWS if workflow.module_key
        )

        with self.assertRaises(ValueError) as context:
            WorkflowProgress(workflow=recording, state=None)

        self.assertIn("must have a state", str(context.exception))

    def test_the_reader_explains_itself_instead_of_showing_a_status(self):
        reader = next(
            workflow for workflow in WORKFLOWS if workflow.module_key is None
        )

        caption = status_caption(WorkflowProgress(workflow=reader))

        self.assertEqual(caption, READS_OTHER_RECORDS)


class TestTheStatesAreAClosedSet(unittest.TestCase):
    """
    OpenMeasure records that an analysis was performed. It must not imply that
    the validation stage has been fully validated.

    Asserted as an allowlist rather than a list of banned words: a blocklist
    only catches the phrasings someone thought of, while a closed set makes
    adding a third state a deliberate edit to a named constant.
    """

    def test_only_two_states_exist(self):
        self.assertEqual(
            ALLOWED_STATES, frozenset({"Recorded", "Not assessed"})
        )

    def test_every_produced_state_is_allowed(self):
        store = HandoffStore({})
        record(store, MODULE_RELIABILITY, "survey.csv")

        for item in workflow_progress(store.entries()):
            with self.subTest(workflow=item.workflow.workflow):
                if item.state is None:
                    self.assertIsNone(item.workflow.module_key)
                else:
                    self.assertIn(item.state, ALLOWED_STATES)

    def test_any_other_state_is_rejected_at_construction(self):
        recording = next(
            workflow for workflow in WORKFLOWS if workflow.module_key
        )

        # "Complete" is the specific word this guards against, since it is
        # the natural thing to reach for and the wrong claim to make.
        for state in ("Complete", "Incomplete", "Validated", "Passed", "100%"):
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    WorkflowProgress(workflow=recording, state=state)

    def test_the_caption_leads_with_the_state(self):
        # A reader scans for the status word, so it cannot be buried behind
        # the filename.
        store = HandoffStore({})
        record(store, MODULE_RELIABILITY, "survey.csv")

        item = progress_for(MODULE_RELIABILITY, store.entries())

        self.assertTrue(status_caption(item).startswith(STATE_RECORDED))

    def test_the_unassessed_caption_is_exactly_the_state(self):
        item = progress_for(MODULE_RELIABILITY, ())

        self.assertEqual(status_caption(item), STATE_NOT_ASSESSED)


class TestTheCaption(unittest.TestCase):
    def test_the_caption_names_the_dataset(self):
        store = HandoffStore({})
        record(store, MODULE_RELIABILITY, "survey.csv")

        caption = status_caption(progress_for(MODULE_RELIABILITY, store.entries()))

        self.assertIn("survey.csv", caption)

    def test_the_caption_labels_the_time_as_utc(self):
        # Records store UTC. Showing it as local time would be wrong for
        # every reader whose timezone is not the server's.
        store = HandoffStore({})
        record(store, MODULE_RELIABILITY, "survey.csv")

        caption = status_caption(progress_for(MODULE_RELIABILITY, store.entries()))

        self.assertIn("UTC", caption)

    def test_the_caption_omits_an_unparseable_time(self):
        # Rather than printing a raw stored string at the user.
        recording = next(
            workflow for workflow in WORKFLOWS if workflow.module_key
        )

        caption = status_caption(
            WorkflowProgress(
                workflow=recording,
                state=STATE_RECORDED,
                dataset="survey.csv",
                recorded_at="not a timestamp",
            )
        )

        self.assertEqual(caption, "Recorded, survey.csv")

    def test_the_caption_omits_a_missing_time(self):
        recording = next(
            workflow for workflow in WORKFLOWS if workflow.module_key
        )

        caption = status_caption(
            WorkflowProgress(
                workflow=recording,
                state=STATE_RECORDED,
                dataset="survey.csv",
                recorded_at="",
            )
        )

        self.assertEqual(caption, "Recorded, survey.csv")

    def test_no_caption_reports_a_statistic(self):
        # A bare number on a card is severed from the interpretation band and
        # caveats that make it mean anything. The record holds one, and the
        # card must not show it.
        store = HandoffStore({})
        entry = record(store, MODULE_RELIABILITY, "survey.csv")

        self.assertTrue(
            entry.primary_statistics,
            "The fixture must record a statistic for this test to bite.",
        )

        caption = status_caption(progress_for(MODULE_RELIABILITY, store.entries()))

        for name, value in entry.primary_statistics.items():
            with self.subTest(statistic=name):
                self.assertNotIn(str(value), caption)
                self.assertNotIn(name, caption)


class TestGatingTheDisplay(unittest.TestCase):
    """
    The whole status display is hidden until something is recorded, so this
    gate decides whether a first visit is unchanged.
    """

    def test_no_records_means_nothing_to_show(self):
        self.assertFalse(has_any_records(()))
        self.assertFalse(has_any_records(HandoffStore({}).entries()))

    def test_one_record_is_enough_to_show_statuses(self):
        store = HandoffStore({})
        record(store, MODULE_RELIABILITY, "survey.csv")

        self.assertTrue(has_any_records(store.entries()))

    def test_clearing_records_hides_the_display_again(self):
        store = HandoffStore({})
        record(store, MODULE_RELIABILITY, "survey.csv")
        store.clear()

        self.assertFalse(has_any_records(store.entries()))


class TestProgressEntries(unittest.TestCase):
    def test_entries_are_frozen(self):
        self.assertTrue(WorkflowProgress.__dataclass_params__.frozen)

    def test_an_explicit_workflow_list_is_honoured(self):
        # The overview passes the catalog, but the join must not be welded to
        # it, otherwise this file could only ever be tested against live data.
        only = (
            Workflow(
                workflow="Solo",
                category="Data Validation",
                stage="Data",
                version="0.1",
                summary="A summary.",
                page="pages/9_Solo.py",
                module_key=MODULE_RELIABILITY,
            ),
        )

        store = HandoffStore({})
        record(store, MODULE_RELIABILITY, "survey.csv")

        items = workflow_progress(store.entries(), only)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].state, STATE_RECORDED)


if __name__ == "__main__":
    unittest.main()
