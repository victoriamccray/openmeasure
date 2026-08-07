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

import inspect
import re
import unittest
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from shared.catalog import WORKFLOWS
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
        import pandas as pd

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
        import pandas as pd

        series = pd.Series([pd.Timedelta("1D") + pd.Timedelta("3s")])

        for alias in ("D", "h", "min", "s", "ms", "us"):
            with self.subTest(alias=alias):
                series.dt.round(alias)


if __name__ == "__main__":
    unittest.main()
