"""
Unit tests for core/limitations.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import evidence as e  # noqa: E402
from core import limitations as lim  # noqa: E402
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


class TestExamineLimitations(unittest.TestCase):
    def test_zero_flags_when_all_checks_pass_and_methods_vary(self):
        items = [
            _item(source="Source A", collection_method="survey"),
            _item(source="Source B", collection_method="administrative_record"),
        ]
        bundle = e.summarize_evidence(items, claim_id="CLAIM-1")
        validation = v.validate_evidence(bundle)
        result = lim.examine_limitations(bundle, validation)

        self.assertEqual(result.n_flags, 0)

    def test_exactly_one_comparison_group_flag_when_only_that_check_fails(self):
        items = [
            _item(source="Source A", has_comparison_group=False, collection_method="survey"),
            _item(source="Source B", has_comparison_group=False, collection_method="administrative_record"),
        ]
        bundle = e.summarize_evidence(items, claim_id="CLAIM-1")
        validation = v.validate_evidence(bundle)
        result = lim.examine_limitations(bundle, validation)

        categories = [flag.category for flag in result.flags]
        self.assertEqual(categories, [v.CRITERION_COMPARISON_GROUP])

    def test_method_bias_flag_when_all_items_share_one_method(self):
        items = [
            _item(source="Source A", collection_method="survey"),
            _item(source="Source B", collection_method="survey"),
        ]
        bundle = e.summarize_evidence(items, claim_id="CLAIM-1")
        validation = v.validate_evidence(bundle)
        result = lim.examine_limitations(bundle, validation)

        categories = [flag.category for flag in result.flags]
        self.assertIn(lim.CATEGORY_METHOD_BIAS, categories)

    def test_raises_on_claim_id_mismatch(self):
        bundle = e.summarize_evidence([_item()], claim_id="CLAIM-1")
        other_bundle = e.summarize_evidence([_item()], claim_id="CLAIM-2")
        validation = v.validate_evidence(other_bundle)

        with self.assertRaises(ValueError):
            lim.examine_limitations(bundle, validation)


if __name__ == "__main__":
    unittest.main()
