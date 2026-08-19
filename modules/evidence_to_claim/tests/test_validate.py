"""
Unit tests for core/validate.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import evidence as e  # noqa: E402
from core import validate as v  # noqa: E402


def _item(**overrides):
    defaults = dict(
        source="Source A",
        finding_text="Some finding.",
        indicator_id="IND-1",
        sample_size=50,
        has_comparison_group=True,
        collection_method="survey",
        time_lag_days=100,
    )
    defaults.update(overrides)
    return e.EvidenceItem(**defaults)


class TestValidateEvidence(unittest.TestCase):
    def test_all_checks_pass_with_known_inputs(self):
        items = [
            _item(source="Source A"),
            _item(source="Source B"),
        ]
        bundle = e.summarize_evidence(items, claim_id="CLAIM-1")
        result = v.validate_evidence(bundle)

        self.assertEqual(result.n_checks_passed, 4)
        self.assertEqual(result.n_checks_total, 4)
        self.assertTrue(result.meets_minimum_bar)

    def test_small_sample_flips_only_sample_size_check(self):
        items = [
            _item(source="Source A", sample_size=5),
            _item(source="Source B", sample_size=5),
        ]
        bundle = e.summarize_evidence(items, claim_id="CLAIM-1")
        result = v.validate_evidence(bundle)

        checks_by_criterion = {c.criterion: c for c in result.checks}
        self.assertFalse(checks_by_criterion[v.CRITERION_SAMPLE_SIZE].passed)
        self.assertTrue(checks_by_criterion[v.CRITERION_COMPARISON_GROUP].passed)
        self.assertTrue(checks_by_criterion[v.CRITERION_CORROBORATION].passed)
        self.assertTrue(checks_by_criterion[v.CRITERION_TIME_LAG].passed)
        # meets_minimum_bar only depends on the blocking (comparison_group)
        # check, so a small sample alone does not fail it.
        self.assertTrue(result.meets_minimum_bar)

    def test_no_comparison_group_fails_minimum_bar(self):
        items = [_item(source="Source A", has_comparison_group=False)]
        bundle = e.summarize_evidence(items, claim_id="CLAIM-1")
        result = v.validate_evidence(bundle)

        self.assertFalse(result.meets_minimum_bar)

    def test_raises_on_non_positive_min_sample_size(self):
        items = [_item()]
        bundle = e.summarize_evidence(items, claim_id="CLAIM-1")
        with self.assertRaises(ValueError):
            v.validate_evidence(bundle, min_sample_size=0)


if __name__ == "__main__":
    unittest.main()
