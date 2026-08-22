"""
Unit tests for core/modality.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import modality as mod  # noqa: E402


def _modality(**overrides):
    defaults = dict(
        name="EEG",
        category="Neural",
        signal_examples="EEG (scalp electrodes)",
        interpretive_gain=0.9,
        privacy_cost=0.9,
        security_cost=0.6,
        agency_cost=0.7,
        citation="Example citation (2026).",
        is_body_sensed=True,
        body_region="Scalp",
        privacy_exit_point="Leaves the scalp as a weak electrical signal.",
        agency_control_point="The cap can be removed at any time.",
    )
    defaults.update(overrides)
    return mod.Modality(**defaults)


class TestModality(unittest.TestCase):
    def test_valid_modality_constructs(self):
        modality = _modality()
        self.assertEqual(modality.name, "EEG")
        self.assertEqual(modality.category, "Neural")

    def test_raises_on_empty_name(self):
        with self.assertRaises(ValueError):
            _modality(name="")

    def test_raises_on_empty_category(self):
        with self.assertRaises(ValueError):
            _modality(category="")

    def test_raises_on_empty_citation(self):
        with self.assertRaises(ValueError):
            _modality(citation="")

    def test_raises_on_interpretive_gain_above_max(self):
        with self.assertRaises(ValueError):
            _modality(interpretive_gain=1.1)

    def test_raises_on_negative_privacy_cost(self):
        with self.assertRaises(ValueError):
            _modality(privacy_cost=-0.1)

    def test_raises_on_negative_security_cost(self):
        with self.assertRaises(ValueError):
            _modality(security_cost=-0.1)

    def test_raises_on_agency_cost_above_max(self):
        with self.assertRaises(ValueError):
            _modality(agency_cost=1.5)

    def test_boundary_ratings_of_zero_and_one_are_valid(self):
        modality = _modality(
            interpretive_gain=0.0,
            privacy_cost=1.0,
            security_cost=0.0,
            agency_cost=1.0,
        )
        self.assertEqual(modality.interpretive_gain, 0.0)
        self.assertEqual(modality.privacy_cost, 1.0)

    def test_raises_on_empty_body_region_when_body_sensed(self):
        with self.assertRaises(ValueError):
            _modality(is_body_sensed=True, body_region="")

    def test_empty_body_region_is_allowed_when_not_body_sensed(self):
        modality = _modality(is_body_sensed=False, body_region="")
        self.assertEqual(modality.body_region, "")

    def test_raises_on_empty_privacy_exit_point(self):
        with self.assertRaises(ValueError):
            _modality(privacy_exit_point="")

    def test_raises_on_empty_agency_control_point(self):
        with self.assertRaises(ValueError):
            _modality(agency_control_point="")


if __name__ == "__main__":
    unittest.main()
