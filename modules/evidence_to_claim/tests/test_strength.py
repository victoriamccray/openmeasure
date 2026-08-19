"""
Unit tests for core/strength.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import claim as c  # noqa: E402
from core import evidence as e  # noqa: E402
from core import strength as s  # noqa: E402
from core import validate as v  # noqa: E402


def _item(**overrides):
    defaults = dict(
        source="Source A",
        finding_text="Some finding.",
        indicator_id="IND-1",
        sample_size=50,
        has_comparison_group=False,
        collection_method="survey",
        time_lag_days=100,
    )
    defaults.update(overrides)
    return e.EvidenceItem(**defaults)


def _claim(claim_type="outcome"):
    return c.ClaimDraft(
        claim_id="CLAIM-1",
        claim_text="Participants improved.",
        claim_type=claim_type,
        level="grantee",
        grantee_id="G1",
    )


class TestDetermineSupportedClaim(unittest.TestCase):
    def test_no_comparison_group_reaches_level_2(self):
        bundle = e.summarize_evidence(
            [_item(has_comparison_group=False)], claim_id="CLAIM-1"
        )
        validation = v.validate_evidence(bundle)
        result = s.determine_supported_claim(_claim(), bundle, validation)
        self.assertEqual(result.nesta_level.level, 2)

    def test_comparison_group_one_source_reaches_level_3(self):
        bundle = e.summarize_evidence(
            [_item(source="Source A", has_comparison_group=True)],
            claim_id="CLAIM-1",
        )
        validation = v.validate_evidence(bundle)
        result = s.determine_supported_claim(_claim(), bundle, validation)
        self.assertEqual(result.nesta_level.level, 3)

    def test_comparison_group_two_sources_reaches_level_4(self):
        bundle = e.summarize_evidence(
            [
                _item(source="Source A", has_comparison_group=True),
                _item(source="Source B", has_comparison_group=True),
            ],
            claim_id="CLAIM-1",
        )
        validation = v.validate_evidence(bundle)
        result = s.determine_supported_claim(_claim(), bundle, validation)
        self.assertEqual(result.nesta_level.level, 4)

    def test_impact_claim_below_level_3_gets_alignment_warning(self):
        bundle = e.summarize_evidence(
            [_item(has_comparison_group=False)], claim_id="CLAIM-1"
        )
        validation = v.validate_evidence(bundle)
        result = s.determine_supported_claim(
            _claim(claim_type="impact"), bundle, validation
        )
        self.assertIsNotNone(result.claim_type_alignment_warning)

    def test_impact_claim_at_level_3_has_no_alignment_warning(self):
        bundle = e.summarize_evidence(
            [_item(source="Source A", has_comparison_group=True)],
            claim_id="CLAIM-1",
        )
        validation = v.validate_evidence(bundle)
        result = s.determine_supported_claim(
            _claim(claim_type="impact"), bundle, validation
        )
        self.assertIsNone(result.claim_type_alignment_warning)

    def test_next_level_hint_none_at_max_level(self):
        bundle = e.summarize_evidence(
            [
                _item(source="Source A", has_comparison_group=True),
                _item(source="Source B", has_comparison_group=True),
            ],
            claim_id="CLAIM-1",
        )
        validation = v.validate_evidence(bundle)
        result = s.determine_supported_claim(_claim(), bundle, validation)
        self.assertEqual(result.nesta_level.level, 4)
        self.assertIsNone(result.next_level_hint)

    def test_next_level_hint_present_below_max_level(self):
        bundle = e.summarize_evidence(
            [_item(has_comparison_group=False)], claim_id="CLAIM-1"
        )
        validation = v.validate_evidence(bundle)
        result = s.determine_supported_claim(_claim(), bundle, validation)
        self.assertEqual(result.nesta_level.level, 2)
        self.assertIsNotNone(result.next_level_hint)

    def test_level_5_note_always_present(self):
        bundle = e.summarize_evidence(
            [_item(has_comparison_group=False)], claim_id="CLAIM-1"
        )
        validation = v.validate_evidence(bundle)
        result = s.determine_supported_claim(_claim(), bundle, validation)
        self.assertEqual(result.level_5_note, s.LEVEL_5_NOTE)

    def test_raises_on_claim_id_mismatch(self):
        bundle = e.summarize_evidence(
            [_item(has_comparison_group=False)], claim_id="CLAIM-1"
        )
        validation = v.validate_evidence(bundle)
        mismatched_claim = c.ClaimDraft(
            claim_id="CLAIM-OTHER",
            claim_text="A different claim.",
            claim_type="outcome",
            level="grantee",
            grantee_id="G1",
        )
        with self.assertRaises(ValueError):
            s.determine_supported_claim(mismatched_claim, bundle, validation)


if __name__ == "__main__":
    unittest.main()
