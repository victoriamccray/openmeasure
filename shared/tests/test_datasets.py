"""
Unit tests for shared/datasets.py

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest

from shared.catalog import WORKFLOWS
from shared.datasets import (
    ACCESS_LEVELS,
    DATASETS,
    DataSource,
    RealDataset,
)

_WORKFLOW_NAMES = {item.workflow for item in WORKFLOWS}


class TestDatasetFields(unittest.TestCase):
    def test_ids_are_unique(self):
        ids = [dataset.id for dataset in DATASETS]

        self.assertEqual(len(ids), len(set(ids)))

    def test_required_fields_are_populated(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset.id):
                self.assertTrue(dataset.name)
                self.assertTrue(dataset.domain)
                self.assertTrue(dataset.description)
                self.assertTrue(dataset.explore_question)
                self.assertTrue(dataset.try_with)
                self.assertTrue(dataset.sources)

    def test_empty_field_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            RealDataset(
                id="",
                name="Nameless",
                domain="Some domain",
                description="A description.",
                try_with=("Reliability",),
                explore_question="A question?",
                access="Open",
                sources=(DataSource(label="Source", url="https://example.org"),),
            )

    def test_entries_are_frozen(self):
        self.assertTrue(RealDataset.__dataclass_params__.frozen)
        self.assertTrue(DataSource.__dataclass_params__.frozen)


class TestAccessLevels(unittest.TestCase):
    def test_every_dataset_has_a_declared_access_level(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset.id):
                self.assertIn(dataset.access, ACCESS_LEVELS)

    def test_unknown_access_level_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            RealDataset(
                id="bogus",
                name="Bogus",
                domain="Some domain",
                description="A description.",
                try_with=("Reliability",),
                explore_question="A question?",
                access="Free for all",
                sources=(DataSource(label="Source", url="https://example.org"),),
            )

        self.assertIn("not one of the declared access levels", str(context.exception))


class TestWorkflowReferences(unittest.TestCase):
    def test_every_try_with_entry_matches_a_known_workflow(self):
        for dataset in DATASETS:
            for workflow_name in dataset.try_with:
                with self.subTest(dataset=dataset.id, workflow=workflow_name):
                    self.assertIn(workflow_name, _WORKFLOW_NAMES)

    def test_unknown_workflow_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            RealDataset(
                id="orphan",
                name="Orphan",
                domain="Some domain",
                description="A description.",
                try_with=("Not A Real Workflow",),
                explore_question="A question?",
                access="Open",
                sources=(DataSource(label="Source", url="https://example.org"),),
            )

        self.assertIn("do not match any workflow", str(context.exception))

    def test_no_try_with_entries_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            RealDataset(
                id="aimless",
                name="Aimless",
                domain="Some domain",
                description="A description.",
                try_with=(),
                explore_question="A question?",
                access="Open",
                sources=(DataSource(label="Source", url="https://example.org"),),
            )

        self.assertIn("does not name any workflow", str(context.exception))


class TestSources(unittest.TestCase):
    def test_every_source_is_an_https_link(self):
        for dataset in DATASETS:
            for source in dataset.sources:
                with self.subTest(dataset=dataset.id, source=source.label):
                    self.assertTrue(source.url.startswith("https://"))

    def test_non_https_source_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            DataSource(label="Insecure", url="http://example.org")

        self.assertIn("not an https link", str(context.exception))

    def test_no_sources_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            RealDataset(
                id="sourceless",
                name="Sourceless",
                domain="Some domain",
                description="A description.",
                try_with=("Reliability",),
                explore_question="A question?",
                access="Open",
                sources=(),
            )

        self.assertIn("lists no sources", str(context.exception))


class TestWastewaterSourceIsTheActualNetwork(unittest.TestCase):
    """
    v0.2C audit fix: the wastewater entry originally pointed only at CDC's
    national NWSS dashboard, but the equity study it's paired with analyzed
    New York State's own statewide network. Pins the correction: the more
    specific NY source must stay listed, so a future edit can't quietly
    drop back to citing only the generic national aggregator.
    """

    def test_wastewater_entry_lists_the_ny_state_source(self):
        dataset = next(
            d for d in DATASETS if d.id == "wastewater_surveillance_equity"
        )
        source_urls = [source.url for source in dataset.sources]

        self.assertTrue(
            any("health.data.ny.gov" in url for url in source_urls),
            f"Expected a health.data.ny.gov source; got: {source_urls}",
        )


class TestDoesNotLeakIntoValidationLifecycle(unittest.TestCase):
    """
    Explore Real Data must never look like a validation workflow.

    A dataset gaining a module_key or a lifecycle stage would let it appear
    on the overview's progress cards or stage strip, which would misstate
    what this page does: point at data, not record an analysis of it.
    """

    def test_real_dataset_has_no_module_key_field(self):
        field_names = {
            field.name for field in RealDataset.__dataclass_fields__.values()
        }

        self.assertNotIn("module_key", field_names)
        self.assertNotIn("stage", field_names)
        self.assertNotIn("category", field_names)


if __name__ == "__main__":
    unittest.main()
