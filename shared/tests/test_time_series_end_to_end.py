"""
Time-Series QA from load to final interpretation.

Run with: pytest shared/tests/test_time_series_end_to_end.py -v

The page-load smoke test opens this page and passes. It cannot fail on
what actually broke here, because the timeline health map and everything
after it only render once a dataset is loaded and the quality checks have
been run: `inspect_note` was called at line 771 and never imported, and
608 passing tests said nothing about it.

So this drives the whole path. Load the bundled sample, pick the columns,
run the checks, and assert the later sections are on screen and nothing
raised. The point is coverage of the stretch of page that only exists
after two button presses, not a new assertion about the statistics, which
their own module tests own.

One structural check comes with it: every helper a page calls from
shared/report.py has to be imported there. That is the class of fault
rather than the instance, and it costs nothing to check across all
pages.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from shared.upload import SAMPLE_BUTTON_LABEL

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = str(ROOT / "Home.py")
PAGE = "pages/4_Time_Series_QA.py"
LOAD_TIMEOUT_SECONDS = 300

RUN_CHECKS_LABEL = "Run quality checks"

# The sections that only exist after the checks have run, in the order
# the page renders them.
LATER_SECTIONS = (
    "Sampling Frequency",
    "Temporal Integrity",
    "Completeness and Coverage",
    "Diagnostics",
    "Which Checks Are Defensible",
    "What This Result Does Not Establish",
)


def _click(app: AppTest, label: str) -> AppTest:
    matches = [button for button in app.button if button.label == label]

    if not matches:
        raise AssertionError(
            f"no {label!r} button; found {[str(b.label) for b in app.button]}"
        )

    matches[0].click()
    app.run()

    return app


class TestTheWholePathRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """One walk, shared by the assertions about it."""
        app = AppTest.from_file(ENTRYPOINT, default_timeout=LOAD_TIMEOUT_SECONDS)
        app.run()
        app.switch_page(PAGE).run()

        app = _click(app, SAMPLE_BUTTON_LABEL)
        cls.after_load = app.exception

        app = _click(app, RUN_CHECKS_LABEL)

        cls.app = app
        cls.exception = app.exception
        cls.rendered = " ".join(str(item.value) for item in app.markdown)
        cls.headings = [str(item.value) for item in app.subheader]

    def test_loading_the_sample_raises_nothing(self):
        self.assertFalse(self.after_load)

    def test_running_the_checks_raises_nothing(self):
        """
        The definition of done: load, checks, complete results, no
        exception.
        """
        self.assertFalse(self.exception)

    def test_every_later_section_is_reached(self):
        for heading in LATER_SECTIONS:
            with self.subTest(section=heading):
                self.assertIn(heading, self.headings)

    def test_the_timeline_map_is_drawn(self):
        self.assertIn("<svg", self.rendered)

    def test_the_line_that_was_missing_its_import_runs(self):
        """
        inspect_note renders through st.write, so its text appears as
        markdown. This asserts the call executed rather than that the
        name resolves, because the name resolving is not the property
        that matters.
        """
        self.assertIn("**What to inspect**: Hollow marks", self.rendered)

    def test_the_page_reports_the_numbers_the_checks_produced(self):
        labels = [str(item.label) for item in self.app.metric]

        for expected in ("Rows loaded", "Observations used", "Gaps"):
            with self.subTest(metric=expected):
                self.assertIn(expected, labels)


class TestEveryPageImportsTheHelpersItCalls(unittest.TestCase):
    """
    The class of fault, not the instance.

    A helper called but not imported is a NameError on whichever branch
    calls it, which may be several button presses deep. Checked
    statically across every page, since that is cheaper than driving
    every branch of every page.
    """

    # Callables shared/report.py exports that pages use bare.
    HELPERS = (
        "inspect_note",
        "interpretation_note",
        "implications",
        "caveat",
        "flagged_item_note",
        "section_header",
        "render_verdict",
        "classify",
        "show_case_studies",
        "render_lifecycle_tracker",
        "render_formula",
    )

    def _missing(self, path):
        """Helpers this page calls without importing or defining."""
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        bound = set()
        called = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    bound.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    bound.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bound.add(target.id)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.add(node.func.id)

        return sorted((called & set(self.HELPERS)) - bound)

    def test_the_check_finds_pages_that_call_these_helpers(self):
        """So a pass cannot come from matching nothing."""
        calling = [
            path.name for path in sorted((ROOT / "pages").glob("*.py"))
            if self._missing(path) is not None
            and any(
                f"{helper}(" in path.read_text(encoding="utf-8")
                for helper in self.HELPERS
            )
        ]

        self.assertGreaterEqual(len(calling), 8)

    def test_no_page_calls_a_report_helper_it_did_not_import(self):
        for path in sorted((ROOT / "pages").glob("*.py")):
            missing = self._missing(path)

            with self.subTest(page=path.name):
                self.assertEqual(
                    missing,
                    [],
                    f"{path.name} calls {', '.join(missing)} without "
                    "importing it. That is a NameError on whichever "
                    "branch calls it, which may be several button "
                    "presses deep.",
                )


if __name__ == "__main__":
    unittest.main()
