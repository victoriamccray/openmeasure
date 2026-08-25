"""
Unit tests for core/interpret.py

Run with: pytest modules/healthring/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import interpret  # noqa: E402


class TestMaeSentence(unittest.TestCase):
    def test_states_the_value_to_one_decimal(self):
        self.assertIn("2.3 bpm", interpret.mae_sentence(2.3))

    def test_zero_is_still_stated_plainly(self):
        self.assertIn("0.0 bpm", interpret.mae_sentence(0.0))


class TestBiasSentence(unittest.TestCase):
    def test_near_zero_bias_reads_as_no_consistent_direction(self):
        self.assertIn("no consistent over- or under-estimate", interpret.bias_sentence(0.02))

    def test_positive_bias_reads_as_higher(self):
        sentence = interpret.bias_sentence(2.5)
        self.assertIn("2.5 bpm higher", sentence)

    def test_negative_bias_reads_as_lower(self):
        sentence = interpret.bias_sentence(-1.8)
        self.assertIn("1.8 bpm lower", sentence)


class TestLoaSentence(unittest.TestCase):
    def test_states_both_bounds_with_sign(self):
        sentence = interpret.loa_sentence(-4.2, 6.1)
        self.assertIn("-4.2", sentence)
        self.assertIn("+6.1", sentence)
        self.assertIn("Bland-Altman", sentence)


class TestQualitySentence(unittest.TestCase):
    def test_high_quality_threshold(self):
        # 0.7 is the hand-calculable boundary stated in the function.
        self.assertIn("high usability", interpret.quality_sentence(0.7))

    def test_middling_quality_band(self):
        self.assertIn("middling usability", interpret.quality_sentence(0.55))

    def test_low_quality_band(self):
        self.assertIn("low usability", interpret.quality_sentence(0.1))


if __name__ == "__main__":
    unittest.main()
