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
    STAGE_NO_MODULE,
    STAGE_READS_RECORDS,
    STAGE_RECORDED,
    STATE_NOT_ASSESSED,
    STATE_RECORDED,
    stage_progress,
    status_caption,
    workflow_progress,
)
from shared.report import CASE_STUDIES_HEADING

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "pages"
ENTRYPOINT = ROOT / "Home.py"

# Generous, because a cold AppTest start is slow and CI runners vary.
LOAD_TIMEOUT_SECONDS = 180

# Pages that cannot be executed on their own because they depend on the
# app's navigation existing.
#
# Overview.py calls st.page_link, which resolves its target against the
# navigation declared in the entrypoint. Run in isolation there is no
# navigation, so Streamlit raises KeyError: 'url_pathname'. This is a real
# property of the page rather than a test inconvenience, and the page is
# still covered: the entrypoint test below runs Home.py, which renders
# Overview as the default page, exactly as it runs in production.
REQUIRES_NAVIGATION_CONTEXT: frozenset[str] = frozenset({"Overview.py"})


def workflow_page_names() -> set[str]:
    """Filenames of the pages the catalog treats as workflows."""

    return {Path(workflow.page).name for workflow in WORKFLOWS}


def standalone_scripts() -> list[Path]:
    """
    Scripts that can be executed directly.

    The entrypoint plus every page except those needing navigation context.
    Discovered rather than listed, so a new page is covered as soon as it
    exists.
    """

    return [ENTRYPOINT] + sorted(
        path
        for path in PAGES.glob("*.py")
        if path.name not in REQUIRES_NAVIGATION_CONTEXT
    )


class TestEveryPageLoads(unittest.TestCase):
    def test_no_page_raises_on_load(self):
        scripts = standalone_scripts()

        # Guard the guard: if discovery silently returned nothing, the test
        # below would pass while checking nothing at all.
        self.assertGreater(len(scripts), 1, "No page scripts were discovered.")

        for script in scripts:
            with self.subTest(page=script.name):
                app = AppTest.from_file(
                    str(script), default_timeout=LOAD_TIMEOUT_SECONDS
                )
                app.run()

                if app.exception:
                    messages = "; ".join(
                        str(item.value)[:400] for item in app.exception
                    )
                    self.fail(f"{script.name} raised on load: {messages}")

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
            "Where each module fits",
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

        for script in sorted(PAGES.glob("*.py")):
            if script.name not in names:
                continue

            with self.subTest(page=script.name):
                app = AppTest.from_file(
                    str(script), default_timeout=LOAD_TIMEOUT_SECONDS
                )
                app.run()

                headings = [
                    str(item.value) for item in app.subheader
                ]

                self.assertIn(
                    CASE_STUDIES_HEADING,
                    headings,
                    f"{script.name} renders case studies without naming the "
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
        # Interpretation has a module that reads rather than records.
        self.assertIn(STAGE_READS_RECORDS, captions)

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
