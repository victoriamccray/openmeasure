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

        # Two evidence sources describing one program. Comparison-group
        # evidence takes this to Level 3; nothing here reproduces the
        # finding on an independent sample, so it stops there and says
        # so.
        expected = (
            "Participants improved job readiness. "
            "Drawing on 2 evidence source(s), the evidence reaches Nesta "
            "Standards of Evidence Level 3 (Causality is demonstrated "
            "using a comparison group.) Independent replication not "
            "established from the supplied evidence. 2 sources describe "
            "this program, which is a count of sources rather than of "
            "studies that reproduced the finding. This meets the rigor "
            "conventionally expected for a claim of this type ('outcome')."
        )
        self.assertEqual(summary, expected)

    def test_the_summary_does_not_call_sources_independent(self):
        """
        The phrase a reader is most likely to quote. A source is a place
        evidence came from, and calling it independent invites the
        replication reading.
        """
        *_, record = _build_record()

        self.assertNotIn("independent source", r.build_leadership_summary(record))

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
