"""
Unit tests for core/source_lesson.py

Run with: pytest modules/signal_pipeline/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from core import source_lesson as s  # noqa: E402


class TestForwardModel(unittest.TestCase):
    def test_hand_values(self):
        # Centre source alone: (1.0*0 + 0.6*1 + 0.2*0, 0.2*0 + 0.6*1 + 1.0*0) = (0.6, 0.6).
        self.assertEqual(s.forward((0.0, 1.0, 0.0)), (0.6, 0.6))
        # Outer sources at 0.5 each: (0.5 + 0.1, 0.1 + 0.5) = (0.6, 0.6).
        self.assertAlmostEqual(s.forward((0.5, 0.0, 0.5))[0], 0.6)

    def test_null_direction_changes_nothing(self):
        self.assertTrue(np.allclose(s.forward(s.NULL_DIRECTION), (0.0, 0.0)))


class TestAmbiguity(unittest.TestCase):
    def test_three_different_configurations_give_identical_readings(self):
        result = s.ambiguity()
        strengths = {c.strengths for c in result.configurations}

        self.assertEqual(len(strengths), 3)
        for config in result.configurations:
            with self.subTest(config=config.label):
                self.assertTrue(np.allclose(config.readings, (0.6, 0.6)))

    def test_minimum_norm_is_one_third_each_and_reproduces_the_readings(self):
        # Smallest norm among (0,1,0) + t(1,-2,1): t = 2/6 = 1/3, giving (1/3, 1/3, 1/3).
        result = s.ambiguity()

        self.assertTrue(np.allclose(result.minimum_norm.strengths, (1 / 3, 1 / 3, 1 / 3)))
        self.assertTrue(np.allclose(result.minimum_norm.readings, result.readings))
        norms = [np.linalg.norm(c.strengths) for c in result.configurations]
        self.assertLess(np.linalg.norm(result.minimum_norm.strengths), min(norms))

    def test_invalid_inputs_raise(self):
        for bad in ((1.0, 2.0), (1.0, np.nan, 0.0)):
            with self.subTest(bad=bad), self.assertRaises(s.SourceLessonError):
                s.forward(bad)
        with self.assertRaises(s.SourceLessonError):
            s.minimum_norm((1.0, 2.0, 3.0))

    def test_every_symbol_of_the_formula_is_defined(self):
        self.assertEqual(set(s.SYMBOLS), {"y", "L", "x", "epsilon"})

    def test_prose_follows_the_style_rule(self):
        import re

        pattern = re.compile(r" - |–|—|, not |not an? |rather than", re.I)
        texts = [s.LESSON["question"], s.LESSON["obstacle"], s.LESSON["formula_plain"], *s.LESSON["limitations"], *s.SYMBOLS.values()]
        self.assertEqual([t for t in texts if pattern.search(t)], [])


if __name__ == "__main__":
    unittest.main()
