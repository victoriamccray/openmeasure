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
    LIFECYCLE_STAGES,
    MODULE_FAIRNESS,
    MODULE_KEYS,
    MODULE_PROGRAM_EVALUATION,
    MODULE_RELIABILITY,
    STAGES_WITHOUT_WORKFLOWS,
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
    ALLOWED_STAGE_STATES,
    ALLOWED_STATES,
    READS_OTHER_RECORDS,
    STAGE_NO_MODULE,
    STAGE_NOT_ASSESSED,
    STAGE_PARTLY_RECORDED,
    STAGE_READS_RECORDS,
    STAGE_RECORDED,
    STATE_NOT_ASSESSED,
    STATE_RECORDED,
    StageProgress,
    WorkflowProgress,
    has_any_records,
    stage_progress,
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


def stage_named(name: str, entries, workflows: tuple[Workflow, ...] = WORKFLOWS) -> StageProgress:
    """The strip entry for one stage."""

    matches = [
        item for item in stage_progress(entries, workflows) if item.stage == name
    ]

    assert len(matches) == 1, f"Expected one entry for stage {name}."

    return matches[0]


# A synthetic stage holding exactly one reader (a workflow with no
# module_key), used by the reader-only tests below. Built explicitly rather
# than pointed at the live Interpretation stage, so these tests do not
# depend on whatever the catalog's Interpretation stage happens to contain
# today -- a future workflow could join it without breaking this fixture.
READER_ONLY_STAGE: tuple[Workflow, ...] = (
    Workflow(
        workflow="Reader Only",
        category="Cross-cutting validation",
        stage="Interpretation",
        version="0.1",
        summary="A summary.",
        page="pages/9_Reader_Only.py",
    ),
)


class TestTheStageStrip(unittest.TestCase):
    """
    The strip aggregates workflow records up to lifecycle stages.

    The risk it carries is different from the cards': one glance summarizes
    several workflows, so a stage with half its workflows recorded must not
    read as covered.
    """

    def test_every_stage_appears_in_order(self):
        # A stage omitted from the strip would imply it is not part of a
        # research lifecycle at all.
        stages = stage_progress(())

        self.assertEqual(
            tuple(item.stage for item in stages), LIFECYCLE_STAGES
        )

    def test_a_stage_with_no_workflow_says_so(self):
        # Research Question has no module. "Not assessed" would blame the user
        # for something the toolkit does not offer.
        for stage in STAGES_WITHOUT_WORKFLOWS:
            with self.subTest(stage=stage):
                item = stage_named(stage, ())

                self.assertEqual(item.state, STAGE_NO_MODULE)
                self.assertEqual(item.n_workflows, 0)

    def test_a_stage_with_nothing_recorded_is_not_assessed(self):
        item = stage_named("Measurement", ())

        self.assertEqual(item.state, STAGE_NOT_ASSESSED)
        self.assertEqual(item.n_recorded, 0)

    def test_a_fully_recorded_stage_is_recorded(self):
        store = HandoffStore({})
        record(store, MODULE_RELIABILITY, "survey.csv")

        item = stage_named("Measurement", store.entries())

        self.assertEqual(item.state, STAGE_RECORDED)
        self.assertEqual(item.n_recorded, item.n_workflows)

    def test_a_partly_recorded_stage_does_not_read_as_recorded(self):
        """
        The assertion this whole class exists for.

        Analysis holds Impact Evaluation and Fairness. Recording one must not
        report the stage as recorded, or a glance at the strip suggests the
        stage was covered when half of it was not.
        """
        store = HandoffStore({})
        record(store, MODULE_PROGRAM_EVALUATION, "trial.csv")

        item = stage_named("Analysis", store.entries())

        self.assertEqual(item.state, STAGE_PARTLY_RECORDED)
        self.assertNotEqual(item.state, STAGE_RECORDED)
        self.assertEqual(item.n_recorded, 1)
        self.assertEqual(item.n_workflows, 2)

    def test_recording_every_workflow_in_a_stage_promotes_it(self):
        store = HandoffStore({})
        record(store, MODULE_PROGRAM_EVALUATION, "trial.csv")
        record(store, MODULE_FAIRNESS, "trial.csv")

        item = stage_named("Analysis", store.entries())

        self.assertEqual(item.state, STAGE_RECORDED)

    def test_a_stage_holding_only_a_reader_says_it_reads(self):
        """
        A stage whose only workflow records nothing is not assessable.

        Two wrong answers this rules out. Counting such a workflow as an
        unrecorded one would leave the stage permanently unassessed. Dropping
        it would report "No module yet" for a stage that has a module the
        user can open from the cards below.
        """
        item = stage_named("Interpretation", (), workflows=READER_ONLY_STAGE)

        self.assertEqual(item.state, STAGE_READS_RECORDS)
        self.assertEqual(item.n_workflows, 1)
        self.assertEqual(item.n_recording, 0)

    def test_a_reader_stage_never_becomes_assessable(self):
        # Recording every workflow that can record must not change it, since
        # the stage has nothing of its own to record.
        store = HandoffStore({})
        for key in MODULE_KEYS:
            record(store, key, f"{key}.csv")

        item = stage_named("Interpretation", store.entries(), workflows=READER_ONLY_STAGE)

        self.assertEqual(item.state, STAGE_READS_RECORDS)

    def test_no_module_and_reads_records_are_distinguished(self):
        # These look alike on the strip but are different facts: one stage has
        # nothing, the other has something that does not record.
        empty = stage_named("Research Question", ())
        reader = stage_named("Interpretation", (), workflows=READER_ONLY_STAGE)

        self.assertEqual(empty.state, STAGE_NO_MODULE)
        self.assertEqual(reader.state, STAGE_READS_RECORDS)
        self.assertNotEqual(empty.state, reader.state)

    def test_recorded_never_exceeds_the_workflow_count(self):
        store = HandoffStore({})
        for key in MODULE_KEYS:
            record(store, key, f"{key}.csv")

        for item in stage_progress(store.entries()):
            with self.subTest(stage=item.stage):
                self.assertLessEqual(item.n_recorded, item.n_recording)
                self.assertLessEqual(item.n_recording, item.n_workflows)

    def test_an_impossible_count_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            StageProgress(
                stage="Analysis",
                state=STAGE_RECORDED,
                n_workflows=1,
                n_recording=1,
                n_recorded=2,
            )

        self.assertIn("more than it has", str(context.exception))

    def test_an_unknown_stage_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            StageProgress(
                stage="Not A Stage",
                state=STAGE_RECORDED,
                n_workflows=1,
                n_recording=1,
                n_recorded=1,
            )


class TestStageStatesAreAClosedSet(unittest.TestCase):
    """
    The same closed-set rule as the workflow states, one level up.

    A separate set because a stage has cases a single workflow does not, but
    the constraint is identical: nothing may imply the stage is finished.
    """

    def test_only_five_stage_states_exist(self):
        self.assertEqual(
            ALLOWED_STAGE_STATES,
            frozenset(
                {
                    "Recorded",
                    "Partly recorded",
                    "Not assessed",
                    "Reads records",
                    "No module yet",
                }
            ),
        )

    def test_every_produced_stage_state_is_allowed(self):
        store = HandoffStore({})
        record(store, MODULE_PROGRAM_EVALUATION, "trial.csv")

        for item in stage_progress(store.entries()):
            with self.subTest(stage=item.stage):
                self.assertIn(item.state, ALLOWED_STAGE_STATES)

    def test_any_other_stage_state_is_rejected_at_construction(self):
        for state in ("Complete", "Validated", "Passed", "50%", "1 of 2"):
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    StageProgress(
                        stage="Analysis",
                        state=state,
                        n_workflows=2,
                        n_recording=2,
                        n_recorded=1,
                    )

    def test_no_stage_state_is_a_counter(self):
        # A "1 of 2" reading would imply 2 of 2 is the goal, which is the
        # completeness claim the wording exists to avoid.
        for state in ALLOWED_STAGE_STATES:
            with self.subTest(state=state):
                self.assertFalse(any(char.isdigit() for char in state))
                self.assertNotIn("%", state)

    def test_entries_are_frozen(self):
        self.assertTrue(StageProgress.__dataclass_params__.frozen)


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
