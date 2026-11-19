"""
Unit tests for core/decision_record.py

Each built record is validated against the published schema in
docs/decision-record/, so a mapping that produces a record the schema
rejects fails here.

Run with: pytest modules/program_evaluation/tests/ -v
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator, FormatChecker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import comparison, decision_record, did, interpret  # noqa: E402
from core import recommend as rec  # noqa: E402
from shared.decision_record import EvidenceItem, ResearcherChoice  # noqa: E402

SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "decision-record"
    / "decision_record.schema.json"
)
VALIDATOR = Draft202012Validator(
    json.loads(SCHEMA_PATH.read_text(encoding="utf-8")),
    format_checker=FormatChecker(),
)

DATASET = EvidenceItem(kind="dataset", label="test.csv", status="synthetic")

# Two groups with hand-calculable means: A = 2, B = 5, difference -3.
# The last row has no outcome and must be excluded.
TWO_GROUP_DATA = pd.DataFrame(
    {
        "group": ["A", "A", "A", "B", "B", "B", "B"],
        "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan],
    }
)
TWO_GROUP_COLUMNS = {"outcome_col": "score", "group_col": "group"}

# Treated changes 2, 3, 4 (mean 3); comparison changes 0, 1, 2 (mean 1).
# The difference in differences is 3 - 1 = 2.
DID_DATA = pd.DataFrame(
    {
        "arm": ["program"] * 3 + ["control"] * 3,
        "pre": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "post": [3.0, 5.0, 7.0, 1.0, 3.0, 5.0],
    }
)
DID_COLUMNS = {"group_col": "arm", "pre_col": "pre", "post_col": "post"}


def build(result, recommendation, columns, **overrides):
    values = dict(
        record_id="test",
        result=result,
        recommendation=recommendation,
        columns=columns,
        dataset=DATASET,
        software_version="abc1234",
        generated_at="2026-11-19T12:00:00+00:00",
    )
    values.update(overrides)
    return decision_record.build_decision_record(**values)


def two_group_record(**overrides):
    result = comparison.compare_two_groups(TWO_GROUP_DATA, "group", "score")
    recommendation = rec.recommend_method(
        TWO_GROUP_DATA, outcome_col="score", group_col="group"
    )
    return build(result, recommendation, TWO_GROUP_COLUMNS, **overrides)


def did_record():
    result = did.estimate_did(DID_DATA, "arm", "pre", "post", treated_label="program")
    recommendation = rec.recommend_method(
        DID_DATA, group_col="arm", pre_col="pre", post_col="post"
    )
    return build(result, recommendation, DID_COLUMNS)


def quantity(record, name):
    return next(q for q in record.result.quantities if q.name == name)


class TestTwoGroupRecord(unittest.TestCase):
    def test_conforms_to_schema(self):
        errors = [e.message for e in VALIDATOR.iter_errors(two_group_record().to_dict())]

        self.assertEqual(errors, [])

    def test_difference_matches_hand_calculation(self):
        difference = quantity(two_group_record(), "Difference in means (A minus B)")

        self.assertAlmostEqual(difference.value, -3.0)
        self.assertEqual(difference.n, 6)
        self.assertEqual(difference.interval.level, 0.95)

    def test_excluded_row_is_accounted_for(self):
        handling = two_group_record().provenance.data_handling[0]

        self.assertEqual(handling.n_input_rows, 7)
        self.assertEqual(handling.n_retained_rows, 6)
        self.assertEqual(handling.retention[0].count, 1)
        self.assertEqual(handling.columns_used, ("score", "group"))

    def test_causal_assumptions_match_the_boundary_diagram(self):
        # The page draws these names on its support-boundary figure. The
        # record must name the same conditions, or the two would describe
        # different designs.
        names = tuple(a.name for a in decision_record._GROUP_COMPARISON_CAUSAL_ASSUMPTIONS)

        self.assertEqual(names, interpret.support_boundary_conditions("compare_two_groups"))

    def test_causal_claim_is_not_supported(self):
        record = two_group_record()

        self.assertTrue(
            any("caused" in entry for entry in record.does_not_support)
        )

    def test_mann_whitney_is_recorded_as_an_alternative(self):
        options = [a.option for a in two_group_record().decision.alternatives]

        self.assertEqual(options, ["Mann-Whitney U test"])

    def test_recommender_warnings_are_kept_as_unplaced(self):
        recommendation = rec.recommend_method(
            TWO_GROUP_DATA, outcome_col="score", group_col="group"
        )
        unplaced = [u.content for u in two_group_record().unplaced]

        self.assertEqual(unplaced, list(recommendation.warnings))

    def test_researcher_choices_pass_through(self):
        choice = ResearcherChoice(choice="Outcome column", value="score", origin="researcher")

        record = two_group_record(researcher_choices=(choice,))

        self.assertEqual(record.provenance.researcher_choices, (choice,))


class TestDiDRecord(unittest.TestCase):
    def test_conforms_to_schema(self):
        errors = [e.message for e in VALIDATOR.iter_errors(did_record().to_dict())]

        self.assertEqual(errors, [])

    def test_estimate_matches_hand_calculation(self):
        estimate = quantity(did_record(), "Difference-in-differences estimate")

        self.assertAlmostEqual(estimate.value, 2.0)

    def test_assumptions_come_from_interpret(self):
        record = did_record()

        self.assertEqual(
            tuple(a.name for a in record.assumptions),
            tuple(a.name for a in interpret.did_assumptions()),
        )
        parallel_trends = record.assumptions[0]
        self.assertEqual(parallel_trends.status, "not_testable_with_these_data")

    def test_supports_come_from_the_existing_reading(self):
        result = did.estimate_did(DID_DATA, "arm", "pre", "post", treated_label="program")
        reading = interpret.interpret_did(result)
        record = did_record()

        self.assertEqual(record.supports, reading.supports)
        self.assertEqual(record.does_not_support, reading.does_not_support)
        self.assertEqual(record.result.summary, reading.headline)

    def test_estimand_is_kept_as_unplaced(self):
        result = did.estimate_did(DID_DATA, "arm", "pre", "post", treated_label="program")
        estimand = interpret.interpret_did(result).estimand

        self.assertIn(estimand, [u.content for u in did_record().unplaced])


class TestEdgeCases(unittest.TestCase):
    def test_unmapped_method_raises(self):
        data = pd.DataFrame({"pre": [1.0, 2.0, 3.0], "post": [2.0, 4.0, 5.0]})
        result = comparison.compare_pre_post(data["pre"], data["post"])
        recommendation = rec.recommend_method(data, pre_col="pre", post_col="post")

        with self.assertRaisesRegex(ValueError, "not mapped onto the Decision Record yet"):
            build(result, recommendation, {"pre_col": "pre", "post_col": "post"})

    def test_result_from_another_method_raises(self):
        did_result = did.estimate_did(DID_DATA, "arm", "pre", "post", treated_label="program")
        recommendation = rec.recommend_method(
            TWO_GROUP_DATA, outcome_col="score", group_col="group"
        )

        with self.assertRaisesRegex(TypeError, "needs a TwoGroupResult"):
            build(did_result, recommendation, TWO_GROUP_COLUMNS)

    def test_non_finite_value_is_recorded_as_null_with_a_note(self):
        recorded = decision_record._quantity("t-statistic", float("inf"))

        self.assertIsNone(recorded.value)
        self.assertIn("not a finite number", recorded.value_note)


if __name__ == "__main__":
    unittest.main()
