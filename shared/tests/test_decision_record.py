"""
Tests for shared/decision_record.py.

Every serialized record is validated against
docs/decision-record/decision_record.schema.json, so the Python objects and
the published schema cannot drift apart without a test failing.

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.decision_record import (  # noqa: E402
    Convention,
    DataHandling,
    Decision,
    DecisionRecord,
    EvidenceItem,
    Method,
    Provenance,
    Quantity,
    Question,
    Reading,
    RecordedAssumption,
    Result,
    RetentionEntry,
    Software,
    to_markdown,
)

SCHEMA_DIR = ROOT / "docs" / "decision-record"
SCHEMA = json.loads((SCHEMA_DIR / "decision_record.schema.json").read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())


def schema_errors(record: dict) -> list[str]:
    return [error.message for error in VALIDATOR.iter_errors(record)]


def minimal_record(**overrides) -> DecisionRecord:
    """A small record with every required field and nothing optional."""

    values = dict(
        record_id="test-1",
        question=Question(text="Did the groups differ?"),
        evidence=(EvidenceItem(kind="dataset", label="data.csv", status="synthetic"),),
        decision=Decision(statement="Compare two groups.", origin="researcher"),
        method=Method(name="Welch's t-test", parameters={"confidence_level": 0.95}),
        assumptions=(),
        result=Result(
            summary="The means differed by 3.",
            quantities=(Quantity(name="Difference in means", value=3.0),),
        ),
        supports=("The groups differed on this outcome.",),
        does_not_support=("That the program caused the difference.",),
        provenance=Provenance(
            software=Software(name="OpenMeasure", version="abc1234"),
            generated_at="2026-11-19T12:00:00+00:00",
            data_handling=(
                DataHandling(
                    evidence_label="data.csv",
                    n_input_rows=7,
                    n_retained_rows=6,
                    retention=(
                        RetentionEntry(
                            label="Rows excluded",
                            count=1,
                            kind="rows_dropped",
                            mechanism="Missing outcome",
                        ),
                    ),
                ),
            ),
            researcher_choices=(),
        ),
    )
    values.update(overrides)
    return DecisionRecord(**values)


class TestSchemaItself(unittest.TestCase):
    def test_schema_is_a_valid_json_schema(self):
        Draft202012Validator.check_schema(SCHEMA)

    def test_specification_example_conforms(self):
        # The example in the specification is what a reader copies, so it
        # is held to the schema it illustrates.
        spec = (SCHEMA_DIR / "specification.md").read_text(encoding="utf-8")
        example = json.loads(re.search(r"```json\n(.*?)```", spec, re.S).group(1))

        self.assertEqual(schema_errors(example), [])


class TestSerialization(unittest.TestCase):
    def test_minimal_record_conforms_to_schema(self):
        self.assertEqual(schema_errors(minimal_record().to_dict()), [])

    def test_version_is_written(self):
        self.assertEqual(minimal_record().to_dict()["record_version"], "0.1")

    def test_unset_optional_fields_are_left_out(self):
        record = minimal_record().to_dict()

        self.assertNotIn("origin", record["question"])
        self.assertNotIn("component", record["provenance"]["software"])

    def test_null_value_is_kept_with_its_note(self):
        quantity = Quantity(name="p-value", value=None, value_note="Variance was zero.")
        record = minimal_record(
            result=Result(summary="Could not test.", quantities=(quantity,))
        ).to_dict()

        serialized = record["result"]["quantities"][0]
        self.assertIsNone(serialized["value"])
        self.assertEqual(serialized["value_note"], "Variance was zero.")
        self.assertEqual(schema_errors(record), [])

    def test_numpy_scalars_become_json_numbers(self):
        record = minimal_record(
            result=Result(
                summary="The means differed by 3.",
                quantities=(Quantity(name="Difference", value=np.float64(3.0), n=np.int64(6)),),
            )
        ).to_dict()

        # json.dumps fails on numpy integers, so this is the real check.
        round_tripped = json.loads(json.dumps(record))
        self.assertEqual(round_tripped["result"]["quantities"][0]["n"], 6)
        self.assertEqual(schema_errors(record), [])

    def test_tuples_become_lists(self):
        record = minimal_record().to_dict()

        self.assertIsInstance(record["supports"], list)


class TestConstructionRules(unittest.TestCase):
    def test_empty_does_not_support_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does_not_support"):
            minimal_record(does_not_support=())

    def test_blank_supports_entry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "supports"):
            minimal_record(supports=("   ",))

    def test_dataset_without_status_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "real, synthetic"):
            EvidenceItem(kind="dataset", label="data.csv")

    def test_checked_assumption_without_check_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "how it was checked"):
            RecordedAssumption(name="Normality", statement="Roughly normal.", status="checked_consistent")

    def test_unknown_assumption_status_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "status 'holds'"):
            RecordedAssumption(name="Normality", statement="Roughly normal.", status="holds")

    def test_null_quantity_without_note_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no value_note"):
            Quantity(name="p-value", value=None)

    def test_reading_of_missing_quantity_is_rejected(self):
        convention = Convention(name="Conventional threshold", citation="Fisher (1925)")

        with self.assertRaisesRegex(ValueError, "not among the result's quantities"):
            Result(
                summary="A result.",
                quantities=(Quantity(name="Difference", value=1.0),),
                readings=(Reading(quantity="p-value", text="Below 0.05.", convention=convention),),
            )

    def test_data_handling_for_unknown_dataset_is_rejected(self):
        base = minimal_record()
        handling = replace(base.provenance.data_handling[0], evidence_label="other.csv")

        with self.assertRaisesRegex(ValueError, "other.csv"):
            minimal_record(provenance=replace(base.provenance, data_handling=(handling,)))

    def test_retaining_more_rows_than_received_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "retains 8 rows out of 7"):
            DataHandling(evidence_label="data.csv", n_input_rows=7, n_retained_rows=8, retention=())

    def test_unknown_origin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "origin 'automatic'"):
            Decision(statement="Compare two groups.", origin="automatic")


class TestMarkdown(unittest.TestCase):
    def test_every_section_appears_in_schema_order(self):
        text = to_markdown(minimal_record())
        headings = [line[5:] for line in text.splitlines() if line.startswith("#### ")]

        self.assertEqual(
            headings,
            [
                "Question",
                "Evidence",
                "Decision",
                "Method",
                "Assumptions",
                "Result",
                "Supports",
                "Does Not Support",
                "Provenance",
            ],
        )

    def test_claims_and_counts_are_carried_over(self):
        text = to_markdown(minimal_record())

        self.assertIn("- The groups differed on this outcome.", text)
        self.assertIn("- That the program caused the difference.", text)
        self.assertIn("6 of 7 rows used", text)
        self.assertIn("Difference in means: 3", text)

    def test_empty_lists_say_none_recorded(self):
        text = to_markdown(minimal_record())
        assumptions = text.split("#### Assumptions")[1].split("####")[0]

        self.assertIn("None recorded.", assumptions)

    def test_question_without_origin_says_it_was_implied(self):
        self.assertIn("not stated by the researcher", to_markdown(minimal_record()))

    def test_dollar_signs_are_escaped_so_they_do_not_render_as_math(self):
        record = minimal_record(question=Question(text="Did cost_$ fall by $5?"))

        self.assertIn("cost\_\$ fall by \$5", to_markdown(record))

    def test_unplaced_section_only_when_there_is_something_in_it(self):
        self.assertNotIn("Recorded Outside the Schema", to_markdown(minimal_record()))


if __name__ == "__main__":
    unittest.main()
