"""
Unit tests for core/claim.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import claim as c  # noqa: E402


class TestClaimDraft(unittest.TestCase):
    def test_valid_claim_constructs(self):
        draft = c.ClaimDraft(
            claim_id="CLAIM-1",
            claim_text="Participants improved their job readiness.",
            claim_type="outcome",
            level="grantee",
            grantee_id="G1",
        )
        self.assertEqual(draft.claim_type, "outcome")
        self.assertEqual(draft.grantee_id, "G1")

    def test_raises_on_unknown_claim_type(self):
        with self.assertRaises(ValueError):
            c.ClaimDraft(
                claim_id="CLAIM-1",
                claim_text="Some claim.",
                claim_type="not_a_real_type",
                level="grantee",
                grantee_id="G1",
            )

    def test_raises_on_unknown_level(self):
        with self.assertRaises(ValueError):
            c.ClaimDraft(
                claim_id="CLAIM-1",
                claim_text="Some claim.",
                claim_type="output",
                level="not_a_real_level",
                grantee_id="G1",
            )

    def test_raises_when_grantee_level_missing_grantee_id(self):
        with self.assertRaises(ValueError):
            c.ClaimDraft(
                claim_id="CLAIM-1",
                claim_text="Some claim.",
                claim_type="output",
                level="grantee",
                grantee_id=None,
            )

    def test_raises_on_empty_claim_text(self):
        with self.assertRaises(ValueError):
            c.ClaimDraft(
                claim_id="CLAIM-1",
                claim_text="   ",
                claim_type="output",
                level="grantee",
                grantee_id="G1",
            )


if __name__ == "__main__":
    unittest.main()
