"""
Smoke tests: every page loads, and the environment can run it.

These exist because the rest of the suite cannot catch a deployment
failure. Everything else here tests pure functions, so the suite passed
green while the app itself could have been unable to start. Pages were
verified by hand instead, which is not a guarantee.

Two distinct failure modes are covered.

1. A page raises on load. Import errors, a missing name, a bad call at
   module level. Pages are discovered from the directory rather than
   listed, so a new page is covered the moment it is added.

2. The installed environment is older than the app requires. This is the
   deployment case specifically: requirements.txt was raised to
   streamlit>=1.49 and pandas>=2.2, and a host that caches its environment
   will happily run new code against old dependencies. That does not look
   stale, it raises TypeError on roughly seventeen st.dataframe calls and
   AttributeError on st.badge. Asserting the capabilities directly is
   faster and clearer than driving the UI far enough to trip over them.

Every page is driven through the entrypoint (Home.py) via switch_page,
rather than executed as a standalone script. Once shared/report.py's
lifecycle tracker started calling st.page_link on every validation page,
every page needed the navigation Home.py declares: st.page_link resolves
its target against that navigation and raises KeyError: 'url_pathname'
without it. Overview.py already needed this treatment for the same
reason; now the rest of the pages do too, so one mechanism covers all of
them instead of a standalone/navigation-context split.

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from shared.catalog import LIFECYCLE_STAGES, MODULE_RELIABILITY, WORKFLOWS
from shared.handoff import (
    STORE_KEY,
    ExclusionAccount,
    HandoffStore,
    fingerprint_dataframe,
)
from shared.progress import (
    ALLOWED_STAGE_STATES,
    STAGE_NO_MODULE,
    STAGE_READS_RECORDS,
    STAGE_RECORDED,
    STATE_NOT_ASSESSED,
    STATE_RECORDED,
    stage_progress,
    status_caption,
    workflow_progress,
)
from shared.report import CASE_STUDIES_HEADING, CURRENT_STAGE_MARKER

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "pages"
ENTRYPOINT = ROOT / "Home.py"

# Generous, because a cold AppTest start is slow and CI runners vary.
LOAD_TIMEOUT_SECONDS = 180

def workflow_page_names() -> set[str]:
    """Filenames of the pages the catalog treats as workflows."""

    return {Path(workflow.page).name for workflow in WORKFLOWS}


def all_page_names() -> list[str]:
    """Every page file, discovered from the directory rather than listed."""

    return sorted(path.name for path in PAGES.glob("*.py"))


class TestEveryPageLoads(unittest.TestCase):
    def test_no_page_raises_on_load(self):
        names = all_page_names()

        # Guard the guard: if discovery silently returned nothing, the test
        # below would pass while checking nothing at all.
        self.assertGreater(len(names), 1, "No page scripts were discovered.")

        app = AppTest.from_file(
            str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS
        )
        app.run()

        if app.exception:
            self.fail(f"The entrypoint raised on load: {app.exception}")

        for name in names:
            if name == "Overview.py":
                continue  # already loaded above, as the default page

            with self.subTest(page=name):
                app.switch_page(f"pages/{name}")
                app.run()

                if app.exception:
                    messages = "; ".join(
                        str(item.value)[:400] for item in app.exception
                    )
                    self.fail(f"{name} raised on load: {messages}")

    def test_the_entrypoint_exists(self):
        self.assertTrue(
            ENTRYPOINT.is_file(),
            "Home.py is the deployed entrypoint and must exist.",
        )

    def test_the_entrypoint_renders_the_default_page(self):
        # Overview cannot be executed standalone, so this is what covers it.
        app = AppTest.from_file(
            str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS
        )
        app.run()

        self.assertFalse(
            app.exception,
            "The entrypoint failed to render its default page.",
        )
        self.assertIn(
            "Where Each Module Fits",
            [str(item.value) for item in app.subheader],
            "The entrypoint ran but did not render the overview content.",
        )

    def test_every_workflow_page_names_its_examples_section(self):
        """
        The examples section must be labelled, not left to be inferred.

        Scoped to the pages the catalog calls workflows, not to every file
        in the directory. The property is that a page showing case studies
        names the section, and the overview shows none.

        There are nine show_case_studies call sites across the workflow
        pages, including early-return branches, and the heading used to be
        applied by each page individually. Only three sites had one, so two
        pages showed the panel with no heading at all.
        """
        names = workflow_page_names()

        self.assertTrue(names, "The catalog lists no workflow pages.")

        app = AppTest.from_file(
            str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS
        )
        app.run()

        for name in sorted(names):
            with self.subTest(page=name):
                app.switch_page(f"pages/{name}")
                app.run()

                headings = [
                    str(item.value) for item in app.subheader
                ]

                self.assertIn(
                    CASE_STUDIES_HEADING,
                    headings,
                    f"{name} renders case studies without naming the "
                    f"section. Headings found: {headings}",
                )


class TestOverviewProgressStatus(unittest.TestCase):
    """
    The status line on the overview cards.

    shared/tests/test_progress.py covers the join. This covers the part that
    only exists once rendered: that the status appears on the page at all, and
    that a first visit is unchanged by the feature.
    """

    @staticmethod
    def _recorded_store() -> dict:
        """A session store holding one recorded reliability analysis."""

        data = pd.DataFrame({"q1": [1, 2, 3], "q2": [2, 3, 4]})
        mapping: dict = {}

        HandoffStore(mapping).record(
            module=MODULE_RELIABILITY,
            fingerprint=fingerprint_dataframe(data, "survey.csv"),
            exclusion=ExclusionAccount(
                module=MODULE_RELIABILITY,
                analysis_label="Reliability",
                columns_considered=("q1", "q2"),
                n_input_rows=3,
                n_retained_rows=3,
            ),
        )

        return mapping

    def _run_entrypoint(self, session: dict | None = None) -> AppTest:
        app = AppTest.from_file(
            str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS
        )

        if session:
            app.session_state[STORE_KEY] = session[STORE_KEY]

        app.run()

        self.assertFalse(
            app.exception,
            "The entrypoint raised while rendering the overview.",
        )

        return app

    def test_no_status_appears_before_anything_is_recorded(self):
        # The status is deliberately absent on a first visit. A wall of "Not
        # assessed" before a user has had the chance to do anything reads as a
        # scolding rather than as guidance.
        app = self._run_entrypoint()

        captions = [str(item.value) for item in app.caption]

        self.assertNotIn(STATE_NOT_ASSESSED, captions)
        for caption in captions:
            self.assertFalse(caption.startswith(STATE_RECORDED))

    def test_recording_an_analysis_shows_a_status_on_every_card(self):
        """
        The page renders exactly what the data layer reports.

        Asserted against shared/progress.py rather than against hardcoded
        counts. Both the strip and the cards emit status captions, and some
        strings are legitimately shared between them, so counting occurrences
        of "Recorded" by hand goes stale the moment either grows a row.
        """
        store = self._recorded_store()
        entries = HandoffStore(store).entries()

        app = self._run_entrypoint(store)

        captions = Counter(str(item.value) for item in app.caption)

        expected = Counter(
            [stage.state for stage in stage_progress(entries)]
            + [status_caption(item) for item in workflow_progress(entries)]
        )

        for value, count in expected.items():
            with self.subTest(status=value):
                self.assertEqual(
                    captions[value],
                    count,
                    f"Expected '{value}' {count} time(s), found "
                    f"{captions[value]}.",
                )

    def test_the_recorded_card_names_its_dataset(self):
        # The card carries the filename; the strip's bare state does not. This
        # is what distinguishes the two, so it is worth asserting directly.
        app = self._run_entrypoint(self._recorded_store())

        detailed = [
            str(item.value)
            for item in app.caption
            if str(item.value).startswith(f"{STATE_RECORDED},")
        ]

        self.assertEqual(len(detailed), 1, f"Found: {detailed}")
        self.assertIn("survey.csv", detailed[0])

    def test_every_workflow_that_records_nothing_yet_says_so(self):
        # The whole point of the feature: showing what has not been looked at.
        app = self._run_entrypoint(self._recorded_store())

        captions = [str(item.value) for item in app.caption]
        unassessed_cards = [
            workflow
            for workflow in WORKFLOWS
            if workflow.module_key and workflow.module_key != MODULE_RELIABILITY
        ]

        self.assertTrue(unassessed_cards, "No unassessed workflow to check.")
        self.assertGreaterEqual(
            captions.count(STATE_NOT_ASSESSED), len(unassessed_cards)
        )

    def test_the_stage_strip_names_every_stage_before_anything_is_recorded(self):
        # The strip is a map of the lifecycle, so it is shown from the start.
        # Only its state labels wait for a record.
        app = self._run_entrypoint()

        rendered = " ".join(str(item.value) for item in app.markdown)

        for stage in LIFECYCLE_STAGES:
            with self.subTest(stage=stage):
                self.assertIn(stage, rendered)

    def test_the_stage_strip_shows_states_once_something_is_recorded(self):
        app = self._run_entrypoint(self._recorded_store())

        captions = [str(item.value) for item in app.caption]

        # Measurement holds only Reliability, which was recorded.
        self.assertIn(STAGE_RECORDED, captions)
        # Research Question has no module, and must not read as a failing.
        self.assertIn(STAGE_NO_MODULE, captions)
        # Interpretation holds Cross-Analysis Implications (reads other
        # workflows' records, module_key=None) and Evidence Review (records
        # its own, module_key set). Only Evidence Review counts toward the
        # stage's state, and this fixture never records it, so the stage
        # reads Not assessed rather than Reads records.
        self.assertIn(STATE_NOT_ASSESSED, captions)

    def test_the_stage_strip_hides_states_before_anything_is_recorded(self):
        app = self._run_entrypoint()

        captions = [str(item.value) for item in app.caption]

        for state in (STAGE_RECORDED, STAGE_NO_MODULE, STAGE_READS_RECORDS):
            with self.subTest(state=state):
                self.assertNotIn(state, captions)

    def test_the_status_never_reports_the_statistic(self):
        # The record holds primary_statistics. A bare number on a card is
        # severed from the interpretation band that makes it mean anything.
        store = self._recorded_store()
        entry = store[STORE_KEY][MODULE_RELIABILITY]
        store[STORE_KEY][MODULE_RELIABILITY] = dataclasses.replace(
            entry, primary_statistics={"cronbach_alpha": 0.8231}
        )

        app = self._run_entrypoint(store)

        captions = " ".join(str(item.value) for item in app.caption)

        self.assertNotIn("0.8231", captions)
        self.assertNotIn("cronbach_alpha", captions)


def _popover_labels(node) -> list[str]:
    """
    Every st.popover trigger label under an element-tree node.

    AppTest has no dedicated collection for popovers the way it does for
    app.button or app.caption, so this walks the tree directly. A stage's
    name lives here, as the label of the popover that reveals its
    workflow(s), rather than as a markdown heading.
    """
    labels: list[str] = []
    proto = getattr(node, "proto", None)

    if proto is not None:
        try:
            has_popover = proto.HasField("popover")
        except ValueError:
            # Leaf element protos (Title, Markdown, ...) do not declare a
            # "popover" field at all, and HasField raises rather than
            # returning False for a field the message type does not have.
            has_popover = False

        if has_popover:
            labels.append(proto.popover.label)

    for child in getattr(node, "children", {}).values():
        labels.extend(_popover_labels(child))

    return labels


class TestValidationPageLifecycleTracker(unittest.TestCase):
    """
    The compact tracker shown on every validation page: every stage is
    reachable, the page's own stage is marked, and none of the Recorded /
    Not assessed vocabulary leaks in, since that is reserved for the
    richer, status-aware tracker on Home.
    """

    def test_reliability_marks_its_own_stage_with_no_status_words(self):
        app = AppTest.from_file(
            str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS
        )
        app.run()
        app.switch_page("pages/1_Reliability.py")
        app.run()

        self.assertFalse(app.exception)

        labels = _popover_labels(app.main)
        for stage in LIFECYCLE_STAGES:
            with self.subTest(stage=stage):
                self.assertIn(stage, labels)

        # st.badge renders as markdown using :color-badge[...] syntax
        # rather than its own element collection.
        markdown = " ".join(str(item.value) for item in app.markdown)
        self.assertIn(CURRENT_STAGE_MARKER, markdown)

        captions = [str(item.value) for item in app.caption]
        for state in ALLOWED_STAGE_STATES:
            with self.subTest(state=state):
                self.assertNotIn(state, captions)


METHOD_SELECTION_BACK_LINK = "Not sure this is the right workflow? Open Method Selection"


class TestFrontDoorReverseLinks(unittest.TestCase):
    """
    A reader can land on a workflow page or Research Journeys directly (a
    bookmark, a search result) without having gone through Method
    Selection first. Every numbered workflow page links back to it via
    render_lifecycle_tracker(current_workflow=...), and Research Journeys
    links back to it directly, so that reader is never stuck with no way
    back to the research-question-first entry point.
    """

    def test_every_workflow_page_links_back_to_method_selection(self):
        app = AppTest.from_file(
            str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS
        )
        app.run()

        for workflow in WORKFLOWS:
            with self.subTest(workflow=workflow.workflow):
                app.switch_page(workflow.page)
                app.run()

                self.assertFalse(app.exception)

                labels = [link.label for link in app.get("page_link")]
                self.assertIn(METHOD_SELECTION_BACK_LINK, labels)

    def test_overview_does_not_show_the_workflow_back_link(self):
        # Overview's tracker has no current_workflow, and already has its
        # own, differently-worded Method Selection link (see
        # pages/Overview.py's Getting Started section), so the generic
        # per-workflow link must not also appear there.
        app = AppTest.from_file(
            str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS
        )
        app.run()

        labels = [link.label for link in app.get("page_link")]
        self.assertNotIn(METHOD_SELECTION_BACK_LINK, labels)

    def test_research_journeys_links_back_to_method_selection(self):
        app = AppTest.from_file(
            str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS
        )
        app.run()
        app.switch_page("pages/Research_Journeys.py")
        app.run()

        self.assertFalse(app.exception)

        links = [link for link in app.get("page_link") if link.page == "Method_Selection"]
        self.assertEqual(len(links), 1)


class TestCrossAnalysisWhatThisMeans(unittest.TestCase):
    """
    The "What this means" section is Interpretation -> Implications ->
    Real-world takeaway (Interpretation/Implications via shared/report.py's
    interpretation_note()/implications(), consistent with every other
    page's guidance labels, rather than this page's own former "Observation"
    / "Why it matters" headings). This covers that the labels and the
    underlying content actually render, since shared/tests/test_retention.py
    only covers that the core summary carries the text.
    """

    @staticmethod
    def _recorded_store() -> dict:
        data = pd.DataFrame({"q1": [1, 2, 3], "q2": [2, 3, 4]})
        mapping: dict = {}

        HandoffStore(mapping).record(
            module=MODULE_RELIABILITY,
            fingerprint=fingerprint_dataframe(data, "survey.csv"),
            exclusion=ExclusionAccount(
                module=MODULE_RELIABILITY,
                analysis_label="Reliability",
                columns_considered=("q1", "q2"),
                n_input_rows=3,
                n_retained_rows=3,
            ),
        )

        return mapping

    def _run_cross_analysis_page(self, store: dict) -> AppTest:
        # Cross-Analysis Implications now calls st.page_link via the
        # lifecycle tracker, so it needs the navigation Home.py declares,
        # the same reason Overview and Explore Real Data already do.
        app = AppTest.from_file(
            str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS
        )
        app.session_state[STORE_KEY] = store[STORE_KEY]
        app.run()
        app.switch_page("pages/5_Cross_Analysis_Implications.py")
        app.run()

        return app

    def test_the_three_labels_and_the_takeaway_render(self):
        app = self._run_cross_analysis_page(self._recorded_store())

        self.assertFalse(app.exception)

        rendered = " ".join(str(item.value) for item in app.markdown)
        self.assertIn("Interpretation", rendered)
        self.assertIn("Implications", rendered)
        self.assertIn("Real-world takeaway", rendered)

        infos = " ".join(str(item.value) for item in app.info)
        self.assertIn("165", infos)  # from REAL_WORLD_TAKEAWAY

    def test_technical_caveats_are_preserved(self):
        app = self._run_cross_analysis_page(self._recorded_store())

        captions = " ".join(str(item.value) for item in app.caption)
        self.assertIn("usually matters more than the rate", captions)


class TestCrossAnalysisOtherRecordedSignals(unittest.TestCase):
    """
    Cross-Analysis Implications surfaces whatever each analysis recorded
    in HandoffEntry.primary_statistics (e.g. a Fairness disparity measure)
    alongside retention, as raw numbers with no combined verdict --
    per validation_chain's own stated aspiration to connect further
    findings across modules.
    """

    @staticmethod
    def _run_cross_analysis_page(store: dict) -> AppTest:
        app = AppTest.from_file(
            str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS
        )
        app.session_state[STORE_KEY] = store[STORE_KEY]
        app.run()
        app.switch_page("pages/5_Cross_Analysis_Implications.py")
        app.run()
        return app

    def test_a_recorded_primary_statistic_is_surfaced(self):
        data = pd.DataFrame({"a": [1, 2, 3], "b": [1, 0, 1]})
        mapping: dict = {}
        HandoffStore(mapping).record(
            module="fairness",
            fingerprint=fingerprint_dataframe(data, "applicants.csv"),
            exclusion=ExclusionAccount(
                module="fairness",
                analysis_label="Fairness (pre-model)",
                columns_considered=("a", "b"),
                n_input_rows=3,
                n_retained_rows=3,
            ),
            primary_statistics={"disparate_impact": 0.62},
        )

        app = self._run_cross_analysis_page(mapping)

        self.assertFalse(app.exception)

        rendered = " ".join(str(item.value) for item in app.markdown)
        self.assertIn("Other recorded signals", rendered)

        dataframe_values = [
            str(cell)
            for df in app.dataframe
            for row in df.value.to_dict("records")
            for cell in row.values()
        ]
        self.assertIn("disparate_impact", dataframe_values)
        self.assertIn("0.62", dataframe_values)

    def test_no_section_when_nothing_recorded_a_primary_statistic(self):
        data = pd.DataFrame({"q1": [1, 2, 3], "q2": [2, 3, 4]})
        mapping: dict = {}
        HandoffStore(mapping).record(
            module=MODULE_RELIABILITY,
            fingerprint=fingerprint_dataframe(data, "survey.csv"),
            exclusion=ExclusionAccount(
                module=MODULE_RELIABILITY,
                analysis_label="Reliability",
                columns_considered=("q1", "q2"),
                n_input_rows=3,
                n_retained_rows=3,
            ),
        )

        app = self._run_cross_analysis_page(mapping)

        self.assertFalse(app.exception)

        rendered = " ".join(str(item.value) for item in app.markdown)
        self.assertNotIn("Other recorded signals", rendered)


class TestExploreRealDataPage(unittest.TestCase):
    """
    Explore Real Data calls st.page_link and so needs navigation context,
    the same reason every page is now driven through the entrypoint above
    rather than run as a standalone script.
    """

    def _run_explore_page(self) -> AppTest:
        app = AppTest.from_file(str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS)
        app.run()
        app.switch_page("pages/Explore_Real_Data.py")
        app.run()

        self.assertFalse(
            app.exception,
            "Explore Real Data raised on load.",
        )

        return app

    def test_the_page_is_reachable_from_the_entrypoint(self):
        app = self._run_explore_page()

        self.assertIn(
            "Explore Real Data",
            [str(item.value) for item in app.title],
        )

    def test_every_dataset_name_is_rendered(self):
        from shared.datasets import DATASETS

        app = self._run_explore_page()

        rendered = " ".join(str(item.value) for item in app.markdown)

        for dataset in DATASETS:
            with self.subTest(dataset=dataset.id):
                self.assertIn(dataset.name, rendered)

    def test_the_page_records_nothing_to_the_handoff_store(self):
        # The whole point of keeping this page outside the catalog: reading
        # about a dataset must never look like an analysis was run.
        app = self._run_explore_page()

        self.assertNotIn(STORE_KEY, app.session_state)


class TestMethodSelectionPage(unittest.TestCase):
    """
    The Method Selection Decision Tree: one guided question, six branches
    (five to workflows, one to a research journey), each with its own
    Try/Why/You'll learn/Limitations content.
    """

    def _run_method_selection_page(self) -> AppTest:
        app = AppTest.from_file(
            str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS
        )
        app.run()
        app.switch_page("pages/Method_Selection.py")
        app.run()

        self.assertFalse(
            app.exception,
            "Method Selection raised on load.",
        )

        return app

    def test_the_page_is_reachable_from_the_entrypoint(self):
        app = self._run_method_selection_page()

        self.assertIn(
            "Method Selection",
            [str(item.value) for item in app.title],
        )

    def _branch_radio(self, app: AppTest):
        """
        The branch-selector radio, found by its options rather than by
        position: this page also has a top-level "Choose an analysis" /
        "Design a study" mode radio, so the branch radio is not
        reliably app.radio[0].
        """

        from shared.method_guide import BRANCHES

        branch_questions = {b.question for b in BRANCHES}
        for radio in app.radio:
            if set(radio.options) == branch_questions:
                return radio

        raise AssertionError("No radio widget offered exactly the branch questions.")

    def test_every_question_is_offered(self):
        from shared.method_guide import BRANCHES

        app = self._run_method_selection_page()

        # .options reflects the formatted display label (format_func), the
        # plain-language question text, not the underlying branch id.
        self.assertEqual(
            set(self._branch_radio(app).options), {b.question for b in BRANCHES}
        )

    def test_each_branch_renders_its_own_try_why_and_learn(self):
        from shared.method_guide import BRANCHES

        app = self._run_method_selection_page()

        for branch in BRANCHES:
            with self.subTest(branch=branch.id):
                self._branch_radio(app).set_value(branch.id)
                app.run()

                self.assertFalse(app.exception)

                rendered = " ".join(str(item.value) for item in app.markdown)
                self.assertIn(f"Try: {branch.destination}", rendered)

                # branch.why is rendered via st.write, which emits Markdown
                # for a plain string.
                self.assertIn(branch.why, rendered)

                for limitation in branch.limitations:
                    self.assertIn(limitation, rendered)

    def test_the_page_records_nothing_to_the_handoff_store(self):
        # A guidance page that routes to a workflow must never look like
        # that workflow's analysis was already run.
        app = self._run_method_selection_page()

        self.assertNotIn(STORE_KEY, app.session_state)

    def test_the_pictograph_has_one_icon_card_per_destination(self):
        # The pictograph is a decorative supplement (see
        # pages/Method_Selection.py's docstring on why it is static, not
        # animated), so this only pins that every destination still gets
        # a card and none silently drops out of the srcdoc, not the SVG
        # markup itself.
        from shared.method_guide import BRANCHES

        app = self._run_method_selection_page()

        iframes = app.get("iframe")
        self.assertEqual(len(iframes), 1)

        srcdoc = iframes[0].proto.srcdoc
        for branch in BRANCHES:
            with self.subTest(branch=branch.id):
                self.assertIn(branch.destination, srcdoc)

    def test_uploading_a_time_series_shaped_file_suggests_time_series_qa(self):
        app = self._run_method_selection_page()

        csv_bytes = (
            b"recorded_at,visits\n"
            b"2024-01-01,10\n2024-01-02,12\n2024-01-03,9\n"
            b"2024-01-04,14\n2024-01-05,11\n"
        )
        app.file_uploader[0].upload("visits.csv", csv_bytes, "text/csv")
        app.run()

        self.assertFalse(app.exception)

        rendered = " ".join(str(item.value) for item in app.markdown)
        self.assertIn("Time-Series QA", rendered)
        self.assertIn("timestamp column", rendered)

    def test_uploading_an_unrecognizable_shape_says_so_plainly(self):
        app = self._run_method_selection_page()

        csv_bytes = b"a\n1\n2\n3\n4\n5\n"
        app.file_uploader[0].upload("plain.csv", csv_bytes, "text/csv")
        app.run()

        self.assertFalse(app.exception)

        info_messages = " ".join(str(item.value) for item in app.info)
        self.assertIn("didn't clearly match", info_messages)

    def test_the_upload_path_records_nothing_to_the_handoff_store(self):
        app = self._run_method_selection_page()

        csv_bytes = b"recorded_at,visits\n2024-01-01,10\n2024-01-02,12\n"
        app.file_uploader[0].upload("visits.csv", csv_bytes, "text/csv")
        app.run()

        self.assertNotIn(STORE_KEY, app.session_state)

    def test_switching_to_design_mode_shows_the_research_question(self):
        # The mode selector is always the first radio on the page,
        # rendered unconditionally before either mode's own content.
        app = self._run_method_selection_page()

        app.radio[0].set_value("design")
        app.run()

        self.assertFalse(app.exception)
        rendered = " ".join(str(item.value) for item in app.markdown)
        rendered += " ".join(str(item.value) for item in app.caption)
        self.assertIn("Enter your own, or load the built-in example", rendered)

        # The hypothesis field starts empty (user-editable, per Design
        # mode's actual purpose); "Load pain example" populates it.
        load_example_buttons = [b for b in app.button if b.label == "Load pain example"]
        self.assertEqual(len(load_example_buttons), 1)
        load_example_buttons[0].click()
        app.run()

        self.assertFalse(app.exception)
        hypothesis_field = next(
            ta for ta in app.text_area if ta.key == "rq_hypothesis"
        )
        self.assertIn("coupling", hypothesis_field.value.lower())

    def test_design_mode_reaches_the_method_selection_handoff(self):
        # Walks Research question -> ... -> Simulate the design, where
        # the simulated measurement plan is handed to the same
        # suggest_workflows() the upload branch above uses.
        app = self._run_method_selection_page()

        app.radio[0].set_value("design")
        app.run()

        continue_labels = (
            "Continue to explore measures",
            "Continue to timing & synchronization",
            "Continue to simulate the design",
            "Continue to reveal terminology & implications",
        )
        for label in continue_labels:
            matches = [b for b in app.button if b.label == label]
            self.assertEqual(len(matches), 1, f"Expected exactly one '{label}' button.")
            matches[0].click()
            app.run()
            self.assertFalse(app.exception)

        rendered = " ".join(str(item.value) for item in app.markdown)
        self.assertIn("Time-Series QA", rendered)
        self.assertIn("Impact Evaluation", rendered)


class TestDataProfileWiredIntoUploadPages(unittest.TestCase):
    """
    shared/upload.py's render_data_profile() is called on every page that
    accepts a file upload. This drives an actual upload through two of
    them end to end (not just a unit test of render_data_profile in
    isolation) to confirm the wiring holds up inside each page's own
    session-state and import context.
    """

    def test_reliability_shows_the_data_profile_after_upload(self):
        sample_path = (
            ROOT / "modules" / "reliability" / "sample_data" / "survey_example.csv"
        )

        app = AppTest.from_file(
            str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS
        )
        app.run()
        app.switch_page("pages/1_Reliability.py")
        app.run()

        app.file_uploader[0].upload(
            "survey_example.csv", sample_path.read_bytes(), "text/csv"
        )
        app.run()

        self.assertFalse(app.exception)
        profile_expanders = [e for e in app.expander if "Data profile" in e.label]
        self.assertEqual(len(profile_expanders), 1)

    def test_time_series_qa_defaults_the_timestamp_selectbox_to_a_datetime_column(self):
        sample_path = (
            ROOT
            / "modules"
            / "time_series_qa"
            / "sample_data"
            / "time_series_example.csv"
        )

        app = AppTest.from_file(
            str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS
        )
        app.run()
        app.switch_page("pages/4_Time_Series_QA.py")
        app.run()

        app.file_uploader[0].upload(
            "time_series_example.csv", sample_path.read_bytes(), "text/csv"
        )
        app.run()

        self.assertFalse(app.exception)
        profile_expanders = [e for e in app.expander if "Data profile" in e.label]
        self.assertEqual(len(profile_expanders), 1)

        timestamp_selectbox = next(
            sb for sb in app.selectbox if sb.label == "Timestamp column"
        )
        self.assertEqual(timestamp_selectbox.value, "recorded_at")


class TestResourcesPage(unittest.TestCase):
    """Resources: every entry's name renders, and nothing is recorded."""

    def _run_resources_page(self) -> AppTest:
        app = AppTest.from_file(str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS)
        app.run()
        app.switch_page("pages/Resources.py")
        app.run()

        self.assertFalse(
            app.exception,
            "Resources raised on load.",
        )

        return app

    def test_the_page_is_reachable_from_the_entrypoint(self):
        app = self._run_resources_page()

        self.assertIn(
            "Resources",
            [str(item.value) for item in app.title],
        )

    def test_every_resource_name_is_rendered(self):
        from shared.resources import RESOURCES

        app = self._run_resources_page()

        rendered = " ".join(str(item.value) for item in app.markdown)

        for resource in RESOURCES:
            with self.subTest(resource=resource.name):
                self.assertIn(resource.name, rendered)

    def test_the_page_records_nothing_to_the_handoff_store(self):
        app = self._run_resources_page()

        self.assertNotIn(STORE_KEY, app.session_state)


class TestQualityAndValidationPage(unittest.TestCase):
    """Quality & Validation: the page loads, names Reliability's status,
    and records nothing to the handoff store."""

    def _run_quality_and_validation_page(self) -> AppTest:
        app = AppTest.from_file(str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS)
        app.run()
        app.switch_page("pages/Quality_and_Validation.py")
        app.run()

        self.assertFalse(
            app.exception,
            "Quality & Validation raised on load.",
        )

        return app

    def test_the_page_is_reachable_from_the_entrypoint(self):
        app = self._run_quality_and_validation_page()

        self.assertIn(
            "Quality & Validation",
            [str(item.value) for item in app.title],
        )

    def test_reliability_status_is_shown(self):
        app = self._run_quality_and_validation_page()

        markdown = " ".join(str(item.value) for item in app.markdown)
        self.assertIn("Reference validated", markdown)

        dataframe_values = [
            str(cell)
            for df in app.dataframe
            for row in df.value.to_dict("records")
            for cell in row.values()
        ]
        self.assertIn("Reliability", dataframe_values)
        self.assertIn("Match", dataframe_values)

    def test_the_page_records_nothing_to_the_handoff_store(self):
        app = self._run_quality_and_validation_page()

        self.assertNotIn(STORE_KEY, app.session_state)


class TestResearchJourneysPage(unittest.TestCase):
    """
    The landing page for Research Journeys: every journey's title renders,
    and each journey page is still reachable even though it is hidden from
    the sidebar (Home.py declares them with visibility="hidden").
    """

    def _run_research_journeys_page(self) -> AppTest:
        app = AppTest.from_file(str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS)
        app.run()
        app.switch_page("pages/Research_Journeys.py")
        app.run()

        self.assertFalse(
            app.exception,
            "Research Journeys raised on load.",
        )

        return app

    def test_the_page_is_reachable_from_the_entrypoint(self):
        app = self._run_research_journeys_page()

        self.assertIn(
            "Research Journeys",
            [str(item.value) for item in app.title],
        )

    def test_every_journey_title_is_rendered(self):
        from shared.research_journeys import JOURNEYS

        app = self._run_research_journeys_page()

        rendered = " ".join(str(item.value) for item in app.markdown)

        for journey in JOURNEYS:
            with self.subTest(journey=journey.title):
                self.assertIn(journey.title, rendered)

    def test_a_hidden_journey_page_is_still_reachable_directly(self):
        from shared.research_journeys import JOURNEYS

        app = AppTest.from_file(str(ENTRYPOINT), default_timeout=LOAD_TIMEOUT_SECONDS)
        app.run()

        for journey in JOURNEYS:
            with self.subTest(journey=journey.title):
                app.switch_page(journey.page)
                app.run()

                self.assertFalse(
                    app.exception,
                    f"{journey.page} raised when reached directly, despite "
                    "being hidden from the sidebar.",
                )

    def test_the_page_records_nothing_to_the_handoff_store(self):
        app = self._run_research_journeys_page()

        self.assertNotIn(STORE_KEY, app.session_state)


class TestEnvironmentCanRunTheApp(unittest.TestCase):
    """
    The app calls APIs that older Streamlit releases do not have. A host
    running a cached environment against current code fails at runtime, so
    these assertions state the requirement plainly.
    """

    def test_streamlit_meets_the_declared_minimum(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        match = re.search(r"^streamlit>=([\d.]+)", requirements, re.MULTILINE)

        self.assertIsNotNone(
            match, "requirements.txt does not declare a streamlit floor."
        )

        required = tuple(int(part) for part in match.group(1).split("."))
        installed = tuple(
            int(part) for part in st.__version__.split(".")[: len(required)]
        )

        self.assertGreaterEqual(
            installed,
            required,
            f"streamlit {st.__version__} is older than the declared minimum "
            f"{match.group(1)}. The app calls APIs this version lacks.",
        )

    def test_dataframe_accepts_the_width_argument(self):
        # Seventeen call sites pass width="stretch". On streamlit below 1.49
        # this raises TypeError, which is a blank page rather than a stale one.
        self.assertIn("width", inspect.signature(st.dataframe).parameters)

    def test_container_accepts_the_border_argument(self):
        # Used by the case study panel on every page.
        self.assertIn("border", inspect.signature(st.container).parameters)

    def test_badge_exists(self):
        # Used for the category chip on the landing page cards.
        self.assertTrue(hasattr(st, "badge"))

    def test_page_link_exists(self):
        # Used for the landing page navigation cards.
        self.assertTrue(hasattr(st, "page_link"))

    def test_pandas_meets_the_declared_minimum(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        match = re.search(r"^pandas>=([\d.]+)", requirements, re.MULTILINE)

        self.assertIsNotNone(
            match, "requirements.txt does not declare a pandas floor."
        )

        required = tuple(int(part) for part in match.group(1).split("."))
        installed = tuple(
            int(part) for part in pd.__version__.split(".")[: len(required)]
        )

        self.assertGreaterEqual(
            installed,
            required,
            f"pandas {pd.__version__} is older than the declared minimum "
            f"{match.group(1)}. Time-Series QA relies on frequency aliases "
            "renamed in 2.2.",
        )

    def test_frequency_aliases_the_module_relies_on_are_valid(self):
        # Time-Series QA rounds intervals using these aliases. The lowercase
        # forms were renamed in pandas 2.2, so an older pandas rejects them.
        series = pd.Series([pd.Timedelta("1D") + pd.Timedelta("3s")])

        for alias in ("D", "h", "min", "s", "ms", "us"):
            with self.subTest(alias=alias):
                series.dt.round(alias)


if __name__ == "__main__":
    unittest.main()
