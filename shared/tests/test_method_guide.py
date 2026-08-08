"""
Unit tests for shared/method_guide.py

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest

from shared.catalog import WORKFLOWS
from shared.method_guide import BRANCHES, MethodBranch

_WORKFLOW_NAMES = {item.workflow for item in WORKFLOWS}

EXPECTED_WORKFLOWS = {
    "Reliability",
    "Time-Series QA",
    "Impact Evaluation",
    "Fairness",
    "Cross-Analysis Implications",
}


class TestBranches(unittest.TestCase):
    def test_exactly_one_branch_per_expected_workflow(self):
        workflows = {branch.workflow for branch in BRANCHES}

        self.assertEqual(workflows, EXPECTED_WORKFLOWS)
        self.assertEqual(len(BRANCHES), len(EXPECTED_WORKFLOWS))

    def test_ids_are_unique(self):
        ids = [branch.id for branch in BRANCHES]

        self.assertEqual(len(ids), len(set(ids)))

    def test_required_fields_are_populated(self):
        for branch in BRANCHES:
            with self.subTest(branch=branch.id):
                self.assertTrue(branch.situation)
                self.assertTrue(branch.workflow)
                self.assertTrue(branch.why)
                self.assertTrue(branch.youll_learn)
                self.assertTrue(branch.limitations)

    def test_entries_are_frozen(self):
        self.assertTrue(MethodBranch.__dataclass_params__.frozen)

    def test_empty_field_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            MethodBranch(
                id="nameless",
                situation="",
                workflow="Reliability",
                why="A reason.",
                youll_learn="Something.",
                limitations=("A limitation.",),
            )


class TestWorkflowReferences(unittest.TestCase):
    def test_every_branch_workflow_matches_a_known_workflow(self):
        for branch in BRANCHES:
            with self.subTest(branch=branch.id):
                self.assertIn(branch.workflow, _WORKFLOW_NAMES)

    def test_unknown_workflow_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            MethodBranch(
                id="orphan",
                situation="Some situation",
                workflow="Not A Real Workflow",
                why="A reason.",
                youll_learn="Something.",
                limitations=("A limitation.",),
            )

        self.assertIn("does not match any workflow", str(context.exception))


class TestLimitations(unittest.TestCase):
    def test_no_limitations_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            MethodBranch(
                id="unqualified",
                situation="Some situation",
                workflow="Reliability",
                why="A reason.",
                youll_learn="Something.",
                limitations=(),
            )

        self.assertIn("lists no limitations", str(context.exception))


class TestDoesNotLeakIntoValidationLifecycle(unittest.TestCase):
    """
    Method Selection must never look like a validation workflow: it routes
    to one, but running it is not itself an analysis.
    """

    def test_method_branch_has_no_module_key_field(self):
        field_names = {
            field.name for field in MethodBranch.__dataclass_fields__.values()
        }

        self.assertNotIn("module_key", field_names)
        self.assertNotIn("stage", field_names)
        self.assertNotIn("category", field_names)


if __name__ == "__main__":
    unittest.main()
