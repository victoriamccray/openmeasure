"""
Unit tests for core/hemodynamics.py

Run with: pytest modules/signal_pipeline/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from core import hemodynamics as h  # noqa: E402


class TestResponseFunction(unittest.TestCase):
    def test_peaks_at_the_gamma_mode(self):
        # A gamma density with shape 6 and scale 1 has its mode at 6 - 1 = 5 s;
        # the small undershoot term barely moves it.
        self.assertAlmostEqual(h.response_function(rate_hz=100).peak_time, 5.0, delta=0.05)

    def test_undershoot_comes_after_the_peak_and_dips_below_zero(self):
        rf = h.response_function()
        self.assertGreater(rf.undershoot_time, 12.0)
        self.assertLess(min(rf.values), 0.0)
        self.assertAlmostEqual(max(rf.values), 1.0)

    def test_bad_rates_raise(self):
        for rate in (0, -1, float("nan")):
            with self.subTest(rate=rate), self.assertRaises(h.HemodynamicsError):
                h.response_function(rate)


class TestConvolution(unittest.TestCase):
    def test_single_event_peaks_five_seconds_later(self):
        result = h.hemodynamic_response([3.0])
        self.assertAlmostEqual(result.first_peak_delay, 5.0, delta=0.1)
        self.assertTrue(result.separable)

    def test_responses_add_linearly(self):
        # Convolution is linear: two events give the sum of two shifted
        # single responses, sample for sample.
        both = np.array(h.hemodynamic_response([2.0, 9.0]).response)
        one = np.array(h.hemodynamic_response([2.0]).response)
        two = np.array(h.hemodynamic_response([9.0]).response)
        self.assertTrue(np.allclose(both, one + two))

    def test_events_far_apart_separate_and_close_events_merge(self):
        self.assertTrue(h.hemodynamic_response([5.0, 15.0]).separable)
        self.assertFalse(h.hemodynamic_response([5.0, 6.0]).separable)

    def test_merged_events_peak_later_than_a_single_one(self):
        merged = h.hemodynamic_response([5.0, 7.0])
        self.assertFalse(merged.separable)
        self.assertGreater(merged.first_peak_delay, 5.5)

    def test_neural_train_marks_each_event(self):
        result = h.hemodynamic_response([1.0, 4.0], rate_hz=10)
        self.assertEqual(sum(result.neural), 2.0)
        self.assertEqual(result.neural[10], 1.0)

    def test_event_in_the_last_half_sample_snaps_to_the_last_sample(self):
        # Found by the independent tests: 39.96 s at 10 Hz rounded to
        # sample 400 of a 400-sample window and crashed with IndexError.
        for event in (39.95, 39.96, 39.99999):
            with self.subTest(event=event):
                result = h.hemodynamic_response([event], duration=40.0, rate_hz=10.0)
                self.assertEqual(len(result.neural), 400)
                self.assertEqual(result.neural[-1], 1.0)

    def test_a_window_with_no_samples_raises_a_typed_error(self):
        with self.assertRaises(h.HemodynamicsError):
            h.hemodynamic_response([0.01], duration=0.04, rate_hz=10.0)

    def test_invalid_events_raise(self):
        for events in ([], [-1.0], [50.0], [float("nan")]):
            with self.subTest(events=events), self.assertRaises(h.HemodynamicsError):
                h.hemodynamic_response(events, duration=40.0)


class TestComponents(unittest.TestCase):
    def test_components_add_up_to_the_response(self):
        parts = np.array(h.response_components(h.STEP_EVENTS))
        whole = np.array(h.hemodynamic_response(h.STEP_EVENTS).response)

        self.assertEqual(parts.shape[0], 2)
        self.assertTrue(np.allclose(parts.sum(axis=0), whole))

    def test_step_events_three_seconds_apart_merge(self):
        # The step-through's closing claim.
        self.assertEqual(h.STEP_EVENTS[1] - h.STEP_EVENTS[0], 3.0)
        self.assertFalse(h.hemodynamic_response(h.STEP_EVENTS).separable)
        self.assertEqual(len(h.CONVOLUTION_STEPS), 5)


class TestLessonText(unittest.TestCase):
    def test_limitations_cite_the_linearity_bound(self):
        self.assertTrue(any("4 s" in item for item in h.LESSON["limitations"]))

    def test_prose_follows_the_style_rule(self):
        import re

        pattern = re.compile(r" - |–|—|, not |not an? |rather than", re.I)
        texts = [h.LESSON["question"], h.LESSON["obstacle"], h.LESSON["formula_plain"], *h.LESSON["limitations"]]
        texts += [t for step in h.CONVOLUTION_STEPS for t in step] + list(h.SYMBOLS.values())
        self.assertEqual([t for t in texts if pattern.search(t)], [])


if __name__ == "__main__":
    unittest.main()
