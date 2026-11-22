"""
Unit tests for core/filtering.py

Run with: pytest modules/signal_pipeline/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from core import filtering as f  # noqa: E402

FS = 500.0
T = np.arange(2000) / FS


def _settled_amplitude(y) -> float:
    # RMS of the middle second times sqrt(2) = amplitude of a settled sine.
    return float(np.std(np.asarray(y)[500:1500]) * np.sqrt(2))


class TestGain(unittest.TestCase):
    def test_band_centre_passes_and_far_frequencies_are_blocked(self):
        for freq, expected, places in ((10.0, 1.0, 2), (2.0, 0.0, 3), (30.0, 0.0, 3)):
            with self.subTest(freq=freq):
                y = f.band_pass(np.sin(2 * np.pi * freq * T), FS).filtered
                self.assertAlmostEqual(_settled_amplitude(y), expected, places=places)

    def test_band_edges_pass_half_amplitude_when_zero_phase(self):
        # A Butterworth filter is down to 1/sqrt(2) at its cutoffs; running it
        # forward and backward squares that to 1/2.
        for freq in (8.0, 13.0):
            with self.subTest(freq=freq):
                self.assertAlmostEqual(f.amplitude_gain(freq, FS), 0.5, places=6)
                self.assertAlmostEqual(f.amplitude_gain(freq, FS, zero_phase=False), 1 / np.sqrt(2), places=6)

    def test_measured_gain_matches_the_designed_gain(self):
        for freq in (9.0, 11.0, 12.0, 15.0):
            with self.subTest(freq=freq):
                y = f.band_pass(np.sin(2 * np.pi * freq * T), FS).filtered
                self.assertAlmostEqual(_settled_amplitude(y), f.amplitude_gain(freq, FS), places=2)


class TestTiming(unittest.TestCase):
    def test_zero_phase_keeps_a_burst_in_place_and_causal_delays_it(self):
        # A steady sine at the band centre has no phase shift through a
        # Butterworth band-pass, so the delay shows in a burst's envelope
        # (group delay): a 10 Hz burst centred at 2.0 s.
        from scipy.signal import hilbert

        t = np.arange(2000) / FS
        burst = np.exp(-((t - 2.0) ** 2) / (2 * 0.1**2)) * np.sin(2 * np.pi * 10 * t)

        def envelope_peak(y):
            return float(t[np.argmax(np.abs(hilbert(np.asarray(y))))])

        self.assertAlmostEqual(envelope_peak(f.band_pass(burst, FS).filtered), 2.0, delta=0.01)
        self.assertGreater(envelope_peak(f.band_pass(burst, FS, zero_phase=False).filtered), 2.03)


class TestCausalDelay(unittest.TestCase):
    def test_forward_only_delay_matches_the_burst_test(self):
        delay = f.causal_burst_delay(FS)
        self.assertGreater(delay, 0.1)
        self.assertLess(delay, 0.3)


class TestConstantSignal(unittest.TestCase):
    def test_constant_signal_keeps_zero_variance(self):
        # Found by the independent tests: 3.7 repeated left rounding residue
        # in the variance and reported 0.0018 kept.
        for value in (3.7, -12.25, 1e6):
            with self.subTest(value=value):
                self.assertEqual(f.band_pass(np.full(500, value), FS).kept_variance_share, 0.0)


class TestRecorded(unittest.TestCase):
    def test_recorded_eyes_closed_signal_keeps_a_share_of_its_variance(self):
        import pandas as pd

        frame = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "sample_data", "ds005420_sub-5_O1_eyes_snippet.csv"))
        closed = frame.loc[frame["condition"] == "eyes_closed", "voltage_uv"].to_numpy()
        result = f.band_pass(closed, FS)

        self.assertTrue(0 < result.kept_variance_share < 1)
        self.assertEqual(len(result.filtered), len(closed))


class TestInvalid(unittest.TestCase):
    def test_bad_inputs_raise(self):
        x = np.sin(2 * np.pi * 10 * T)
        # order=True was accepted as 1 until the independent tests flagged it.
        for kwargs in ({"low_hz": 13.0, "high_hz": 8.0}, {"high_hz": 300.0}, {"order": 0}, {"order": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(f.FilteringError):
                f.band_pass(x, FS, **kwargs)
        for signal in (np.ones((2, 100)), np.r_[x[:-1], np.nan], np.ones(10)):
            with self.subTest(shape=np.shape(signal)), self.assertRaises(f.FilteringError):
                f.band_pass(signal, FS)
        with self.assertRaises(f.FilteringError):
            f.band_pass(x, 0.0)

    def test_prose_follows_the_style_rule(self):
        import re

        pattern = re.compile(r" - |–|—|, not |not an? |rather than", re.I)
        texts = [f.LESSON["question"], f.LESSON["what_it_does"], *f.LESSON["costs"]]
        self.assertEqual([t for t in texts if pattern.search(t)], [])


if __name__ == "__main__":
    unittest.main()
