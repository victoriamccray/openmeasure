"""
Unit tests for modules/cross_analysis/core/compare.py

Run with: pytest modules/cross_analysis/tests/ -v

The behaviour under test is mostly a refusal: two analyses reporting
different kinds of quantity must not be reported as agreeing or
disagreeing, and nothing here may produce a single number saying how much
two analyses agree.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from core import compare  # noqa: E402

from shared.handoff import (  # noqa: E402
    QUANTITY_GROUP_DIFFERENCE,
    QUANTITY_INTERNAL_CONSISTENCY,
    AssumptionRecord,
    DatasetFingerprint,
    ExclusionAccount,
    Finding,
    HandoffEntry,
)


def _entry(module, findings=(), assumptions=(), retained=40):
    return HandoffEntry(
        module=module,
        fingerprint=DatasetFingerprint(
            digest="abc123",
            filename="study.csv",
            n_rows=40,
            n_columns=5,
            column_names=("a", "b", "c", "d", "e"),
        ),
        exclusion=ExclusionAccount(
            module=module,
            analysis_label=module,
            columns_considered=("a", "b"),
            n_input_rows=40,
            n_retained_rows=retained,
        ),
        findings=findings,
        assumptions=assumptions,
    )


def _alpha(reading="Acceptable internal consistency", value=0.75):
    return Finding(
        label="Cronbach's alpha",
        quantity=QUANTITY_INTERNAL_CONSISTENCY,
        value=value,
        reading=reading,
        statement=f"alpha = {value}",
    )


def _difference(reading="Difference detected at the conventional threshold"):
    return Finding(
        label="Welch's t-test",
        quantity=QUANTITY_GROUP_DIFFERENCE,
        value=0.4,
        reading=reading,
        statement="p = 0.01",
    )


class TestDifferentQuantitiesAreNotComparable(unittest.TestCase):
    """
    The state the whole redesign turns on. An alpha and a group
    difference are both numbers about one upload, and neither agreement
    nor disagreement between them is defined.
    """

    def setUp(self):
        self.pairs = compare.compare_findings(
            (
                _entry("Reliability", findings=(_alpha(),)),
                _entry("Impact Evaluation", findings=(_difference(),)),
            )
        )

    def test_the_pair_is_marked_not_comparable(self):
        self.assertEqual(len(self.pairs), 1)
        self.assertEqual(self.pairs[0].comparison, compare.NOT_COMPARABLE)

    def test_it_names_both_quantities_rather_than_saying_only_no(self):
        explanation = self.pairs[0].explanation.lower()

        self.assertIn("internal consistency", explanation)
        self.assertIn("difference between groups", explanation)

    def test_an_incomparable_pair_is_still_reported(self):
        """
        Dropping it would leave a reader unable to tell a comparison that
        was attempted and could not be made from one never attempted.
        """
        self.assertTrue(self.pairs)


class TestSameQuantity(unittest.TestCase):
    def test_matching_readings_converge(self):
        pairs = compare.compare_findings(
            (
                _entry("Reliability", findings=(_alpha(),)),
                _entry("Reliability (subset)", findings=(_alpha(),)),
            )
        )

        self.assertEqual(pairs[0].comparison, compare.CONVERGE)

    def test_convergence_is_not_called_confirmation(self):
        """
        Two analyses of one upload agreeing is not independent evidence,
        and the explanation has to say so where the reader sees it.
        """
        pairs = compare.compare_findings(
            (
                _entry("Reliability", findings=(_alpha(),)),
                _entry("Reliability (subset)", findings=(_alpha(),)),
            )
        )

        self.assertIn("not independent confirmation", pairs[0].explanation)

    def test_differing_readings_diverge(self):
        pairs = compare.compare_findings(
            (
                _entry("Reliability", findings=(_alpha(),)),
                _entry(
                    "Reliability (subset)",
                    findings=(_alpha(reading="Questionable internal consistency", value=0.62),),
                ),
            )
        )

        self.assertEqual(pairs[0].comparison, compare.DIVERGE)

    def test_divergence_does_not_declare_a_winner(self):
        pairs = compare.compare_findings(
            (
                _entry("Reliability", findings=(_alpha(),)),
                _entry(
                    "Reliability (subset)",
                    findings=(_alpha(reading="Questionable internal consistency", value=0.62),),
                ),
            )
        )
        explanation = pairs[0].explanation.lower()

        self.assertIn("not settled", explanation)
        for verdict in ("correct", "wrong", "better", "should use"):
            with self.subTest(word=verdict):
                self.assertNotIn(verdict, explanation)


class TestRetentionQualifiesEveryPair(unittest.TestCase):
    """
    The page's strongest existing idea, previously buried in prose: two
    analyses of one upload can keep different rows, and equal counts do
    not establish the same subset either.
    """

    def test_differing_retention_is_flagged(self):
        pairs = compare.compare_findings(
            (
                _entry("Reliability", findings=(_alpha(),), retained=38),
                _entry("Impact Evaluation", findings=(_difference(),), retained=35),
            )
        )

        self.assertFalse(pairs[0].same_retained_count)

    def test_equal_retention_is_reported_as_equal_counts_only(self):
        pairs = compare.compare_findings(
            (
                _entry("Reliability", findings=(_alpha(),), retained=38),
                _entry("Impact Evaluation", findings=(_difference(),), retained=38),
            )
        )

        self.assertTrue(pairs[0].same_retained_count)
        self.assertIn("do not establish", compare.RETENTION_MATCHES_NOTE)


class TestAssumptions(unittest.TestCase):
    def test_a_shared_condition_is_listed_first(self):
        reading = compare.read_across(
            (
                _entry(
                    "Reliability",
                    findings=(_alpha(),),
                    assumptions=(
                        AssumptionRecord(name="Shared thing", status="Stated"),
                        AssumptionRecord(name="Only mine", status="Stated"),
                    ),
                ),
                _entry(
                    "Impact Evaluation",
                    findings=(_difference(),),
                    assumptions=(
                        AssumptionRecord(name="Shared thing", status="Stated"),
                    ),
                ),
            )
        )

        self.assertTrue(reading.assumptions[0].shared)
        self.assertEqual(reading.assumptions[0].name, "Shared thing")

    def test_a_condition_records_which_analyses_rest_on_it(self):
        reading = compare.read_across(
            (
                _entry(
                    "Reliability",
                    findings=(_alpha(),),
                    assumptions=(AssumptionRecord(name="Shared thing", status="Stated"),),
                ),
                _entry(
                    "Impact Evaluation",
                    findings=(_difference(),),
                    assumptions=(AssumptionRecord(name="Shared thing", status="Stated"),),
                ),
            )
        )

        self.assertEqual(
            reading.assumptions[0].modules, ("Reliability", "Impact Evaluation")
        )


class TestModulesWithoutFindings(unittest.TestCase):
    def test_a_module_recording_no_finding_is_named_not_omitted(self):
        """
        A module wired for retention and not yet wired for findings is
        absent from the comparison for a reason a reader should see.
        """
        reading = compare.read_across(
            (
                _entry("Reliability", findings=(_alpha(),)),
                _entry("Time-Series QA"),
            )
        )

        self.assertEqual(reading.modules_without_findings, ("Time-Series QA",))
        self.assertIn("Time-Series QA", reading.modules)

    def test_one_finding_alone_produces_no_pairs(self):
        reading = compare.read_across((_entry("Reliability", findings=(_alpha(),)),))

        self.assertEqual(reading.pairs, ())
        self.assertFalse(reading.has_comparable_pair)


class TestNoAgreementScore(unittest.TestCase):
    """
    The one output this module must not produce. A number saying how much
    two analyses agree would collapse a judgment that depends on knowing
    what the analyses are into something that looks like a measurement.
    """

    def test_the_reading_exposes_no_overall_score(self):
        reading = compare.read_across(
            (
                _entry("Reliability", findings=(_alpha(),)),
                _entry("Impact Evaluation", findings=(_difference(),)),
            )
        )

        for forbidden in ("score", "agreement", "overall", "confidence"):
            with self.subTest(attribute=forbidden):
                self.assertFalse(
                    any(forbidden in name.lower() for name in vars(reading))
                )


class TestPairValidation(unittest.TestCase):
    def test_an_unknown_comparison_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            compare.FindingPair(
                module_a="A",
                module_b="B",
                label_a="a",
                label_b="b",
                reading_a="r",
                reading_b="r",
                comparison="Mostly agrees",
                explanation="Because.",
                same_retained_count=True,
            )

        self.assertIn("not a known comparison", str(raised.exception))

    def test_a_pair_without_an_explanation_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            compare.FindingPair(
                module_a="A",
                module_b="B",
                label_a="a",
                label_b="b",
                reading_a="r",
                reading_b="r",
                comparison=compare.CONVERGE,
                explanation="",
                same_retained_count=True,
            )

        self.assertIn("is a verdict", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
