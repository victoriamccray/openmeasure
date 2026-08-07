"""
Unit tests for shared/catalog.py

These are drift tests. The landing page's module list was previously
hand-written and fell out of date, so the point of this file is to fail when
the catalog and the repository disagree.

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import dataclasses
import re
import unittest
from pathlib import Path

from shared.case_studies import CASE_STUDIES
from shared.catalog import (
    CATEGORY_ORDER,
    LIFECYCLE_STAGES,
    STAGE_QUESTIONS,
    STAGES_WITHOUT_WORKFLOWS,
    WORKFLOWS,
    Workflow,
    workflows_by_category,
    workflows_by_stage,
)

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "pages"

# Streamlit orders sidebar entries by a numeric filename prefix, and every
# workflow page uses one. Only those pages are required to appear in the
# catalog, so an unnumbered page added later (About.py, Documentation.py)
# needs no change here at all.
NUMBERED_PAGE = re.compile(r"^\d+_.*\.py$")

# Numbered pages that are deliberately not workflow cards. Empty today.
# It exists so that a genuine exception is a visible one-line edit rather
# than a reason to loosen the rule above.
NON_WORKFLOW_PAGES: frozenset[str] = frozenset()


def numbered_pages() -> set[str]:
    """Numbered page files, as paths relative to the repository root."""

    return {
        f"pages/{path.name}"
        for path in PAGES.glob("*.py")
        if NUMBERED_PAGE.match(path.name)
        and f"pages/{path.name}" not in NON_WORKFLOW_PAGES
    }


class TestCatalogMatchesPages(unittest.TestCase):
    """
    The assertions that would have caught the drift this catalog replaces.
    """

    def test_every_numbered_page_has_a_catalog_entry(self):
        catalogued = {item.page for item in WORKFLOWS}
        missing = numbered_pages() - catalogued

        self.assertEqual(
            missing,
            set(),
            f"These pages exist but are not in shared/catalog.py, so they "
            f"would be missing from the landing page: {sorted(missing)}. Add "
            f"a Workflow entry, or list the page in NON_WORKFLOW_PAGES if it "
            f"is deliberately not a workflow.",
        )

    def test_every_catalog_page_exists_on_disk(self):
        missing = {
            item.page for item in WORKFLOWS if not (ROOT / item.page).is_file()
        }

        self.assertEqual(
            missing,
            set(),
            f"The catalog points at pages that do not exist, so the landing "
            f"page would render a dead link: {sorted(missing)}.",
        )

    def test_no_page_is_claimed_by_two_workflows(self):
        pages = [item.page for item in WORKFLOWS]

        self.assertEqual(len(pages), len(set(pages)))

    def test_unnumbered_pages_are_not_required(self):
        # Guards the rule itself: a future About.py must not be demanded of
        # the catalog. Asserted on the matcher rather than by creating a file.
        self.assertFalse(NUMBERED_PAGE.match("About.py"))
        self.assertFalse(NUMBERED_PAGE.match("Documentation.py"))
        self.assertTrue(NUMBERED_PAGE.match("1_Reliability.py"))
        self.assertTrue(NUMBERED_PAGE.match("10_Something_Later.py"))


class TestStages(unittest.TestCase):
    def test_every_workflow_has_a_known_stage(self):
        for item in WORKFLOWS:
            with self.subTest(workflow=item.workflow):
                self.assertIn(item.stage, LIFECYCLE_STAGES)

    def test_unknown_stage_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            Workflow(
                workflow="Bogus",
                category="Some Category",
                stage="Not A Stage",
                version="0.1",
                summary="A summary.",
                page="pages/9_Bogus.py",
            )

        self.assertIn("not a known lifecycle stage", str(context.exception))

    def test_grouping_covers_every_stage_in_order(self):
        grouped = workflows_by_stage()

        self.assertEqual(tuple(grouped.keys()), LIFECYCLE_STAGES)

    def test_grouping_accounts_for_every_workflow(self):
        grouped = workflows_by_stage()
        total = sum(len(items) for items in grouped.values())

        self.assertEqual(total, len(WORKFLOWS))

    def test_only_the_declared_stages_are_empty(self):
        # A stage emptying unintentionally is drift too: it would silently
        # disappear from the page's coverage.
        grouped = workflows_by_stage()
        empty = tuple(
            stage for stage, items in grouped.items() if not items
        )

        self.assertEqual(empty, STAGES_WITHOUT_WORKFLOWS)

    def test_every_stage_has_a_question(self):
        for stage in LIFECYCLE_STAGES:
            with self.subTest(stage=stage):
                self.assertTrue(STAGE_QUESTIONS.get(stage))


class TestNavigation(unittest.TestCase):
    """
    The sidebar is built from the catalog, so a workflow missing from the
    grouping is a workflow the user cannot reach.
    """

    def test_every_workflow_has_a_known_category(self):
        for workflow in WORKFLOWS:
            with self.subTest(workflow=workflow.workflow):
                self.assertIn(workflow.category, CATEGORY_ORDER)

    def test_unknown_category_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            Workflow(
                workflow="Orphan",
                category="Not A Category",
                stage="Data",
                version="0.1",
                summary="A summary.",
                page="pages/9_Orphan.py",
            )

        self.assertIn("not in CATEGORY_ORDER", str(context.exception))

    def test_grouping_covers_every_workflow_exactly_once(self):
        grouped = workflows_by_category()
        seen = [
            workflow.workflow
            for workflows in grouped.values()
            for workflow in workflows
        ]

        self.assertEqual(sorted(seen), sorted(w.workflow for w in WORKFLOWS))
        self.assertEqual(len(seen), len(set(seen)))

    def test_categories_appear_in_declared_order(self):
        grouped = workflows_by_category()
        expected = [
            category for category in CATEGORY_ORDER if category in grouped
        ]

        self.assertEqual(list(grouped.keys()), expected)

    def test_no_category_group_is_empty(self):
        # An empty navigation section is a dead heading, unlike an empty
        # lifecycle stage which the overview states as a gap on purpose.
        for category, workflows in workflows_by_category().items():
            with self.subTest(category=category):
                self.assertTrue(workflows)

    def test_url_paths_are_unique(self):
        # A collision would make two sidebar entries resolve to one page.
        paths = [workflow.url_path for workflow in WORKFLOWS]

        self.assertEqual(len(paths), len(set(paths)))

    def test_url_paths_strip_the_numeric_prefix(self):
        # This is what preserves links such as /Reliability that Streamlit's
        # file-based routing produced before navigation became explicit.
        for workflow in WORKFLOWS:
            with self.subTest(workflow=workflow.workflow):
                self.assertFalse(re.match(r"^\d+_", workflow.url_path))
                self.assertNotIn("/", workflow.url_path)
                self.assertTrue(workflow.url_path)


class TestTaxonomyKeys(unittest.TestCase):
    def test_taxonomy_keys_match_the_case_study_tags(self):
        # Catches drift in the other direction: a key that no case study
        # uses would render an empty research-examples section.
        tagged = {
            key
            for study in CASE_STUDIES.values()
            for key in study.modules
        }

        for item in WORKFLOWS:
            if item.taxonomy_key is None:
                continue
            with self.subTest(workflow=item.workflow):
                self.assertIn(item.taxonomy_key, tagged)


class TestWorkflowFields(unittest.TestCase):
    def test_required_fields_are_populated(self):
        for item in WORKFLOWS:
            with self.subTest(workflow=item.workflow):
                self.assertTrue(item.workflow)
                self.assertTrue(item.category)
                self.assertTrue(item.version)
                self.assertTrue(item.summary)
                self.assertTrue(item.page)

    def test_empty_field_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            Workflow(
                workflow="Nameless",
                category="",
                stage="Data",
                version="0.1",
                summary="A summary.",
                page="pages/9_X.py",
            )

    def test_summaries_are_a_single_line(self):
        for item in WORKFLOWS:
            with self.subTest(workflow=item.workflow):
                self.assertNotIn("\n", item.summary)

    def test_multiline_summary_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            Workflow(
                workflow="Wordy",
                category="Some Category",
                stage="Data",
                version="0.1",
                summary="Line one.\nLine two.",
                page="pages/9_X.py",
            )

        self.assertIn("single line", str(context.exception))

    def test_workflow_names_are_unique(self):
        names = [item.workflow for item in WORKFLOWS]

        self.assertEqual(len(names), len(set(names)))

    def test_entries_are_frozen(self):
        self.assertTrue(Workflow.__dataclass_params__.frozen)

    def test_no_field_implies_a_score_or_status(self):
        # Status indicators are out of scope for this version, and any added
        # later must ship as icon plus text rather than color alone.
        forbidden = ("score", "grade", "rating", "colour", "color")
        names = {field.name for field in dataclasses.fields(Workflow)}

        for term in forbidden:
            for name in names:
                with self.subTest(field=name, term=term):
                    self.assertNotIn(term, name.lower())


if __name__ == "__main__":
    unittest.main()
