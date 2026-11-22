"""
Unit tests for core/correlation_lesson.py

Run with: pytest modules/signal_pipeline/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from core import correlation_lesson as c  # noqa: E402


class TestPearson(unittest.TestCase):
    def test_hand_values(self):
        self.assertAlmostEqual(c.pearson_r([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(c.pearson_r([1, 2, 3], [3, 2, 1]), -1.0)
        # x = [1, 2, 3, 4], y = [1, 3, 2, 4]: cov = 1.0, var x = var y = 1.25, r = 0.8
        self.assertAlmostEqual(c.pearson_r([1, 2, 3, 4], [1, 3, 2, 4]), 0.8)

    def test_matches_numpy(self):
        rng = np.random.default_rng(0)
        x, y = rng.normal(size=50), rng.normal(size=50)
        self.assertAlmostEqual(c.pearson_r(x, y), float(np.corrcoef(x, y)[0, 1]), places=12)

    def test_degenerate_inputs_raise(self):
        for x, y in (([1, 1, 1], [1, 2, 3]), ([1, 2], [1, 2]), ([1, 2, 3], [1, 2]), ([1, np.nan, 3], [1, 2, 3])):
            with self.subTest(x=x, y=y), self.assertRaises(c.CorrelationLessonError):
                c.pearson_r(x, y)


class TestClosedForms(unittest.TestCase):
    def test_hand_values(self):
        # direct b=1, s=1: 1/sqrt(2); common a=1, s=1: 1/2; shared w1=1, w2=0.5, s=1:
        # 0.5 / sqrt(2 * 1.25) = 0.3162
        self.assertAlmostEqual(c.expected_r(c.WIRING_DIRECT, 1.0, 1.0), 1 / np.sqrt(2))
        self.assertAlmostEqual(c.expected_r(c.WIRING_COMMON, 1.0, 1.0), 0.5)
        self.assertAlmostEqual(c.expected_r(c.WIRING_SHARED, 1.0, 1.0, 0.5), 0.5 / np.sqrt(2.5))

    def test_no_link_and_no_shared_source_means_zero(self):
        self.assertEqual(c.expected_r(c.WIRING_COMMON, 0.0, 1.0), 0.0)

    def test_simulated_r_lands_near_the_closed_form(self):
        # Sampling SD of r is about (1 - r^2) / sqrt(n); allow 4 of them.
        for wiring in c.WIRINGS:
            for strength in (0.3, 1.0, 2.0):
                sim = c.simulate(wiring, strength, n=2000)
                sd = (1 - sim.expected_r**2) / np.sqrt(2000)
                with self.subTest(wiring=wiring, strength=strength):
                    self.assertLess(abs(sim.r - sim.expected_r), 4 * sd + 1e-9)

    def test_a_direct_link_and_a_common_input_can_share_one_r(self):
        # The lesson's central claim: r alone cannot tell the wirings apart.
        b = c.matching_direct_strength(1.0, 1.0)
        self.assertAlmostEqual(b, 1 / np.sqrt(3))
        self.assertAlmostEqual(c.expected_r(c.WIRING_DIRECT, b, 1.0), c.expected_r(c.WIRING_COMMON, 1.0, 1.0))
        self.assertTrue(c.simulate(c.WIRING_DIRECT, b).direct_link)
        self.assertFalse(c.simulate(c.WIRING_COMMON, 1.0).direct_link)

    def test_simulation_is_deterministic(self):
        self.assertEqual(c.simulate(c.WIRING_COMMON, 1.0).x, c.simulate(c.WIRING_COMMON, 1.0).x)

    def test_invalid_inputs_raise(self):
        for args in (("telepathy", 1.0, 1.0), (c.WIRING_DIRECT, -1.0, 1.0), (c.WIRING_DIRECT, 1.0, 0.0)):
            with self.subTest(args=args), self.assertRaises(c.CorrelationLessonError):
                c.expected_r(*args)
        with self.assertRaises(c.CorrelationLessonError):
            c.simulate(c.WIRING_DIRECT, 1.0, n=2)


class TestLessonText(unittest.TestCase):
    def test_prose_follows_the_style_rule(self):
        import re

        pattern = re.compile(r" - |–|—|, not |not an? |rather than", re.I)
        texts = [c.LESSON["question"], c.LESSON["obstacle"], c.LESSON["formula_plain"], *c.LESSON["limitations"]]
        self.assertEqual([t for t in texts if pattern.search(t)], [])


if __name__ == "__main__":
    unittest.main()
