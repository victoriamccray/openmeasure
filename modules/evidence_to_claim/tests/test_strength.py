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

    def test_two_sources_alone_do_not_reach_level_4(self):
        """
        The defect this replaces. Two distinct source names reporting the
        same finding used to promote a bundle to independent replication.
        A grants export and a follow-up survey are two sources describing
        one program, and neither reproduces a causal result elsewhere.
        """
        bundle = e.summarize_evidence(
            [
                _item(source="Source A", has_comparison_group=True),
                _item(source="Source B", has_comparison_group=True),
            ],
            claim_id="CLAIM-1",
        )
        validation = v.validate_evidence(bundle)
        result = s.determine_supported_claim(_claim(), bundle, validation)

        self.assertEqual(result.nesta_level.level, 3)
        self.assertFalse(result.replication_established)
        self.assertIn(s.REPLICATION_NOT_ESTABLISHED, result.replication_note)

    def test_three_collection_methods_are_triangulation_not_replication(self):
        """
        The reported case: a grants-management export, employer
        verification calls and a participant follow-up survey.
        """
        bundle = e.summarize_evidence(
            [
                _item(
                    source="GivingData export",
                    collection_method="admin record",
                    has_comparison_group=True,
                ),
                _item(
                    source="Employer verification", collection_method="phone call"
                ),
                _item(source="Participant follow-up", collection_method="survey"),
            ],
            claim_id="CLAIM-1",
        )
        validation = v.validate_evidence(bundle)
        result = s.determine_supported_claim(_claim(), bundle, validation)

        self.assertTrue(result.triangulated)
        self.assertFalse(result.replication_established)
        self.assertEqual(result.nesta_level.level, 3)

    def test_the_note_says_sources_are_not_studies(self):
        bundle = e.summarize_evidence(
            [
                _item(source="Source A", has_comparison_group=True),
                _item(source="Source B"),
            ],
            claim_id="CLAIM-1",
        )
        validation = v.validate_evidence(bundle)
        result = s.determine_supported_claim(_claim(), bundle, validation)

        self.assertIn("count of sources", result.replication_note)

    def test_an_identified_independent_evaluation_reaches_level_4(self):
        bundle = e.summarize_evidence(
            [
                _item(
                    source="Original evaluation",
                    has_comparison_group=True,
                    study_identity="Ramirez et al. 2021",
                ),
                _item(
                    source="Replication",
                    has_comparison_group=True,
                    study_identity="Okafor et al. 2024",
                    independent_sample=True,
                    same_intervention=True,
                    causal_design=True,
                ),
            ],
            claim_id="CLAIM-1",
        )
        validation = v.validate_evidence(bundle)
        result = s.determine_supported_claim(_claim(), bundle, validation)

        self.assertEqual(result.nesta_level.level, 4)
        self.assertTrue(result.replication_established)

    def test_one_study_cannot_replicate_itself(self):
        """
        An item flagged as an independent evaluation, with no second
        identified study for it to be independent of.
        """
        bundle = e.summarize_evidence(
            [
                _item(
                    source="Only study",
                    has_comparison_group=True,
                    study_identity="Okafor et al. 2024",
                    independent_sample=True,
                    same_intervention=True,
                    causal_design=True,
                ),
            ],
            claim_id="CLAIM-1",
        )
        validation = v.validate_evidence(bundle)
        result = s.determine_supported_claim(_claim(), bundle, validation)

        self.assertEqual(result.nesta_level.level, 3)
        self.assertFalse(result.replication_established)

    def test_a_replication_without_a_causal_design_does_not_count(self):
        bundle = e.summarize_evidence(
            [
                _item(
                    source="Original",
                    has_comparison_group=True,
                    study_identity="Ramirez et al. 2021",
                ),
                _item(
                    source="Other study",
                    study_identity="Okafor et al. 2024",
                    independent_sample=True,
                    same_intervention=True,
                    causal_design=False,
                ),
            ],
            claim_id="CLAIM-1",
        )
        validation = v.validate_evidence(bundle)
        result = s.determine_supported_claim(_claim(), bundle, validation)

        self.assertEqual(result.nesta_level.level, 3)


class TestFrameworkAndRuleAreDisclosedSeparately(unittest.TestCase):
    """
    The levels are Nesta's. The rule mapping a bundle onto them is
    OpenMeasure's, and citing Nesta for it would put their name on an
    algorithm they did not write.
    """

    def _result(self):
        bundle = e.summarize_evidence([_item()], claim_id="CLAIM-1")
        return s.determine_supported_claim(
            _claim(), bundle, v.validate_evidence(bundle)
        )

    def test_the_citation_names_the_framework(self):
        self.assertIn("Nesta", self._result().framework_citation)

    def test_the_rule_is_named_as_openmeasures_own(self):
        note = self._result().openmeasure_rule_note

        self.assertIn("OpenMeasure", note)
        self.assertIn("not part of the cited framework", note)

    def test_the_rule_itself_is_stated_rather_than_only_referred_to(self):
        statement = self._result().openmeasure_rule_statement

        self.assertIn("Level 4", statement)
        self.assertIn("independent sample", statement)

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
                _item(
                    source="Original evaluation",
                    has_comparison_group=True,
                    study_identity="Ramirez et al. 2021",
                ),
                _item(
                    source="Replication",
                    has_comparison_group=True,
                    study_identity="Okafor et al. 2024",
                    independent_sample=True,
                    same_intervention=True,
                    causal_design=True,
                ),
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
