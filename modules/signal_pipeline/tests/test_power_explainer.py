"""
Unit tests for core/power_explainer.py

Run with: pytest modules/signal_pipeline/tests/ -v
"""

from __future__ import annotations

import dataclasses
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core import intervention, power_explainer as pe  # noqa: E402

SNIPPET = os.path.join(os.path.dirname(__file__), "..", "sample_data", "ds005420_sub-5_O1_eyes_snippet.csv")


def _recorded(condition: str, start: int = 0) -> np.ndarray:
    frame = pd.read_csv(SNIPPET)
    return frame.loc[frame["condition"] == condition, "voltage_uv"].to_numpy()[start : start + 500]


class TestMeanSquare(unittest.TestCase):
    def test_amplitude_two_gives_mean_zero_and_mean_square_two(self):
        # Hand value: A^2 / 2 = 4 / 2 = 2.0 uV^2, and whole cycles average to 0.
        result = pe.mean_and_mean_square(2.0)

        self.assertAlmostEqual(result.mean, 0.0, places=12)
        self.assertAlmostEqual(result.mean_square, 2.0, places=12)
        self.assertEqual(result.expected_mean_square, 2.0)

    def test_mean_square_is_half_amplitude_squared_across_amplitudes(self):
        for amplitude in (0.0, 0.5, 1.0, 3.5, 40.0):
            with self.subTest(amplitude=amplitude):
                self.assertAlmostEqual(pe.mean_and_mean_square(amplitude).mean_square, amplitude**2 / 2, places=9)

    def test_partial_cycles_and_bad_inputs_raise(self):
        with self.assertRaises(pe.PowerExplainerError):
            pe.mean_and_mean_square(1.0, frequency_hz=2.5, seconds=1.0)
        for amplitude in (-1.0, float("nan"), float("inf")):
            with self.subTest(amplitude=amplitude), self.assertRaises(pe.PowerExplainerError):
                pe.mean_and_mean_square(amplitude)


class TestWindowSpectrum(unittest.TestCase):
    def test_alpha_area_equals_the_pipeline_on_recorded_windows(self):
        # The explainer must teach the calculation the journey runs.
        for condition in ("eyes_closed", "eyes_open"):
            for start in (0, 3000, 7000):
                with self.subTest(condition=condition, start=start):
                    window = _recorded(condition, start)
                    self.assertTrue(pe.agrees_with_pipeline(window))
                    self.assertAlmostEqual(
                        pe.window_spectrum(window).alpha_power,
                        intervention.compute_alpha_power(window, intervention.SAMPLING_RATE_HZ),
                        places=9,
                    )

    def test_parseval_total_area_equals_tapered_mean_square(self):
        # Exact identity for a Hann-windowed periodogram with density scaling.
        for condition in ("eyes_closed", "eyes_open"):
            spectrum = pe.window_spectrum(_recorded(condition))
            with self.subTest(condition=condition):
                self.assertAlmostEqual(spectrum.total_power / spectrum.tapered_mean_square, 1.0, places=9)

    def test_steady_noise_area_matches_plain_mean_square_on_average(self):
        # For stationary white noise the taper changes nothing on average:
        # over 2000 seeded windows the ratio is within 1% of 1.
        rng = np.random.default_rng(1)
        ratios = []
        for _ in range(2000):
            spectrum = pe.window_spectrum(rng.normal(0, 10, 500))
            ratios.append(spectrum.total_power / spectrum.mean_square)

        self.assertAlmostEqual(float(np.mean(ratios)), 1.0, delta=0.01)

    def test_first_recorded_second_matches_the_published_baseline(self):
        # data.json's first eyes-closed window power, 240.887 uV^2.
        self.assertAlmostEqual(pe.window_spectrum(_recorded("eyes_closed")).alpha_power, 240.887, places=3)

    def test_bad_windows_raise(self):
        for window in ([1.0], [[1.0, 2.0], [3.0, 4.0]], [1.0, float("nan"), 2.0]):
            with self.subTest(window=window), self.assertRaises(pe.PowerExplainerError):
                pe.window_spectrum(window)


class TestMixture(unittest.TestCase):
    def test_changing_alpha_moves_only_the_alpha_band(self):
        quiet, loud = pe.mixture_spectrum(0.0).spectrum, pe.mixture_spectrum(20.0).spectrum
        freqs = np.array(quiet.freqs_hz)
        outside = (freqs < 6) | ((freqs > 15) & (freqs <= 30))

        self.assertTrue(np.allclose(np.array(quiet.density)[outside], np.array(loud.density)[outside], rtol=1e-6, atol=1e-9))
        self.assertGreater(loud.alpha_power, 100 * quiet.alpha_power)

    def test_alpha_power_follows_the_amplitude(self):
        # A 20 uV alpha tone carries 200 uV^2; noise and leakage from theta
        # and beta add a little and the 10 Hz cross term with noise moves
        # it slightly, so the band reads within 5% of 200.
        self.assertAlmostEqual(pe.mixture_spectrum(20.0).spectrum.alpha_power / 200.0, 1.0, delta=0.05)

    def test_the_mixture_is_deterministic_and_frozen(self):
        self.assertEqual(pe.mixture_spectrum(5.0).values, pe.mixture_spectrum(5.0).values)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            pe.mixture_spectrum(5.0).alpha_amplitude = 1.0

    def test_negative_alpha_raises(self):
        with self.assertRaises(pe.PowerExplainerError):
            pe.mixture_spectrum(-1.0)


class TestSteps(unittest.TestCase):
    def test_three_steps_in_order(self):
        self.assertEqual([s.key for s in pe.STEPS], ["square", "separate", "integrate"])

    def test_prose_follows_the_style_rule(self):
        import re

        pattern = re.compile(r" - |–|—|, not |not an? |rather than", re.I)
        texts = [t for s in pe.STEPS for t in (s.title, s.question, s.explanation, s.takeaway)]

        self.assertEqual([t for t in texts if pattern.search(t)], [])


if __name__ == "__main__":
    unittest.main()
