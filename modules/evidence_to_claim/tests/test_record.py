"""
Unit tests for core/record.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import claim as c  # noqa: E402
from core import evidence as e  # noqa: E402
from core import limitations as lim  # noqa: E402
from core import record as r  # noqa: E402
from core import strength as s  # noqa: E402
from core import validate as v  # noqa: E402


def _build_record():
    claim = c.ClaimDraft(
        claim_id="CLAIM-1",
        claim_text="Participants improved job readiness.",
        claim_type="outcome",
        level="grantee",
        grantee_id="G1",
    )
    items = [
        e.EvidenceItem(
            source="Source A",
            finding_text="Found X",
            indicator_id="IND-1",
            sample_size=50,
            has_comparison_group=True,
            collection_method="survey",
            time_lag_days=100,
        ),
        e.EvidenceItem(
            source="Source B",
            finding_text="Found X confirmed",
            indicator_id="IND-1",
            sample_size=60,
            has_comparison_group=True,
            collection_method="administrative_record",
            time_lag_days=120,
        ),
    ]
    bundle = e.summarize_evidence(items, claim_id="CLAIM-1")
    validation = v.validate_evidence(bundle)
    supported = s.determine_supported_claim(claim, bundle, validation)
    limitations = lim.examine_limitations(bundle, validation)

    record = r.EvidenceRecord(
        claim=claim,
        evidence=bundle,
        validation=validation,
        supported_claim=supported,
        limitations=limitations,
        portfolio_context=None,
    )
    return claim, bundle, validation, supported, limitations, record


class TestBuildLeadershipSummary(unittest.TestCase):
    def test_exact_summary_string_for_fixed_fixture(self):
        *_, record = _build_record()
        summary = r.build_leadership_summary(record)

        expected = (
            "Participants improved job readiness. "
            "Based on 2 independent source(s), the evidence reaches Nesta "
            "Standards of Evidence Level 4 (One or more independent "
            "replications confirm the finding.) This meets the rigor "
            "conventionally expected for a claim of this type ('outcome')."
        )
        self.assertEqual(summary, expected)

    def test_raises_on_claim_id_mismatch_across_sub_results(self):
        claim, bundle, validation, supported, limitations, _ = _build_record()

        other_claim = c.ClaimDraft(
            claim_id="CLAIM-OTHER",
            claim_text="A different claim.",
            claim_type="outcome",
            level="grantee",
            grantee_id="G1",
        )

        with self.assertRaises(ValueError):
            r.EvidenceRecord(
                claim=other_claim,
                evidence=bundle,
                validation=validation,
                supported_claim=supported,
                limitations=limitations,
                portfolio_context=None,
            )


if __name__ == "__main__":
    unittest.main()
