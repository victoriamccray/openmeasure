"""
Unit tests for shared/method_guide.py

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest

from shared.catalog import WORKFLOWS
from shared.method_guide import BRANCHES, MethodBranch
from shared.research_journeys import JOURNEYS

_WORKFLOW_NAMES = {item.workflow for item in WORKFLOWS}
_JOURNEY_TITLES = {item.title for item in JOURNEYS}

EXPECTED_WORKFLOWS = {
    "Reliability",
    "Time-Series QA",
    "Impact Evaluation",
    "Fairness",
    "Cross-Analysis Implications",
}

EXPECTED_JOURNEYS = {
    "Portfolio Impact Analysis",
}


class TestBranches(unittest.TestCase):
    def test_exactly_one_branch_per_expected_workflow(self):
        workflows = {
            branch.workflow for branch in BRANCHES if branch.workflow is not None
        }

        self.assertEqual(workflows, EXPECTED_WORKFLOWS)
        self.assertEqual(
            len([b for b in BRANCHES if b.workflow is not None]),
            len(EXPECTED_WORKFLOWS),
        )

    def test_exactly_one_branch_per_expected_journey(self):
        journeys = {
            branch.journey for branch in BRANCHES if branch.journey is not None
        }

        self.assertEqual(journeys, EXPECTED_JOURNEYS)
        self.assertEqual(
            len([b for b in BRANCHES if b.journey is not None]),
            len(EXPECTED_JOURNEYS),
        )

    def test_ids_are_unique(self):
        ids = [branch.id for branch in BRANCHES]

        self.assertEqual(len(ids), len(set(ids)))

    def test_required_fields_are_populated(self):
        for branch in BRANCHES:
            with self.subTest(branch=branch.id):
                self.assertTrue(branch.question)
                self.assertTrue(branch.workflow or branch.journey)
                self.assertTrue(branch.why)
                self.assertTrue(branch.youll_learn)
                self.assertTrue(branch.limitations)

    def test_entries_are_frozen(self):
        self.assertTrue(MethodBranch.__dataclass_params__.frozen)

    def test_empty_field_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            MethodBranch(
                id="nameless",
                question="",
                workflow="Reliability",
                why="A reason.",
                youll_learn="Something.",
                limitations=("A limitation.",),
            )


class TestDestinationIsExactlyOne(unittest.TestCase):
    def test_neither_workflow_nor_journey_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            MethodBranch(
                id="nowhere",
                question="Some question?",
                why="A reason.",
                youll_learn="Something.",
                limitations=("A limitation.",),
            )

        self.assertIn("exactly one", str(context.exception))

    def test_both_workflow_and_journey_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            MethodBranch(
                id="everywhere",
                question="Some question?",
                workflow="Reliability",
                journey="Portfolio Impact Analysis",
                why="A reason.",
                youll_learn="Something.",
                limitations=("A limitation.",),
            )

        self.assertIn("exactly one", str(context.exception))


class TestWorkflowReferences(unittest.TestCase):
    def test_every_branch_workflow_matches_a_known_workflow(self):
        for branch in BRANCHES:
            if branch.workflow is None:
                continue
            with self.subTest(branch=branch.id):
                self.assertIn(branch.workflow, _WORKFLOW_NAMES)

    def test_unknown_workflow_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            MethodBranch(
                id="orphan",
                question="Some question?",
                workflow="Not A Real Workflow",
                why="A reason.",
                youll_learn="Something.",
                limitations=("A limitation.",),
            )

        self.assertIn("does not match any workflow", str(context.exception))


class TestJourneyReferences(unittest.TestCase):
    def test_every_branch_journey_matches_a_known_journey(self):
        for branch in BRANCHES:
            if branch.journey is None:
                continue
            with self.subTest(branch=branch.id):
                self.assertIn(branch.journey, _JOURNEY_TITLES)

    def test_unknown_journey_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            MethodBranch(
                id="orphan",
                question="Some question?",
                journey="Not A Real Journey",
                why="A reason.",
                youll_learn="Something.",
                limitations=("A limitation.",),
            )

        self.assertIn("does not match any journey", str(context.exception))


class TestLimitations(unittest.TestCase):
    def test_no_limitations_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            MethodBranch(
                id="unqualified",
                question="Some question?",
                workflow="Reliability",
                why="A reason.",
                youll_learn="Something.",
                limitations=(),
            )

        self.assertIn("lists no limitations", str(context.exception))


class TestDestinationHelpers(unittest.TestCase):
    def test_destination_returns_whichever_field_is_set(self):
        for branch in BRANCHES:
            with self.subTest(branch=branch.id):
                expected = (
                    branch.workflow if branch.workflow is not None else branch.journey
                )
                self.assertEqual(branch.destination, expected)

    def test_destination_kind_matches_which_field_is_set(self):
        for branch in BRANCHES:
            with self.subTest(branch=branch.id):
                if branch.workflow is not None:
                    self.assertEqual(branch.destination_kind, "workflow")
                else:
                    self.assertEqual(branch.destination_kind, "research journey")


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
