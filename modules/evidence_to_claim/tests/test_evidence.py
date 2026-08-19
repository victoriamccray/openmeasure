"""
Unit tests for core/evidence.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import evidence as e  # noqa: E402


def _item(**overrides):
    defaults = dict(
        source="Source A",
        finding_text="Some finding.",
        indicator_id="IND-1",
        sample_size=None,
        has_comparison_group=False,
        collection_method="survey",
        time_lag_days=None,
    )
    defaults.update(overrides)
    return e.EvidenceItem(**defaults)


class TestSummarizeEvidence(unittest.TestCase):
    def test_hand_calculated_aggregation(self):
        items = [
            _item(source="Source A", sample_size=10),
            _item(source="Source B", sample_size=50),
            _item(source="Source A", sample_size=None),
        ]
        bundle = e.summarize_evidence(items, claim_id="CLAIM-1")

        self.assertEqual(bundle.min_sample_size, 10)
        self.assertEqual(bundle.n_sources, 2)
        self.assertEqual(bundle.n_usable_items, 3)
        self.assertEqual(bundle.n_input_items, 3)
        self.assertEqual(bundle.n_excluded_items, 0)

    def test_items_missing_finding_text_are_excluded_and_counted(self):
        items = [
            _item(finding_text="A real finding."),
            _item(finding_text="   "),
            _item(indicator_id=None),
        ]
        bundle = e.summarize_evidence(items, claim_id="CLAIM-1")

        self.assertEqual(bundle.n_input_items, 3)
        self.assertEqual(bundle.n_usable_items, 1)
        self.assertEqual(bundle.n_excluded_items, 2)
        self.assertEqual(bundle.exclusion_reason, e.MISSING_FINDING_OR_INDICATOR)

    def test_raises_on_zero_usable_items(self):
        items = [_item(finding_text=""), _item(indicator_id=None)]
        with self.assertRaises(ValueError):
            e.summarize_evidence(items, claim_id="CLAIM-1")

    def test_raises_on_empty_claim_id(self):
        with self.assertRaises(ValueError):
            e.summarize_evidence([_item()], claim_id="")


if __name__ == "__main__":
    unittest.main()
