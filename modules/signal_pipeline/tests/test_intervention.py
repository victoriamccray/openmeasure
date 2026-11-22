"""
Unit tests for core/intervention.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from core import intervention as iv  # noqa: E402

SAMPLING_RATE_HZ = 500.0


def _sine_window(frequency_hz: float, sampling_rate: float = SAMPLING_RATE_HZ) -> np.ndarray:
    t = np.arange(int(sampling_rate)) / sampling_rate
    return np.sin(2 * np.pi * frequency_hz * t)


class TestComputeAlphaPower(unittest.TestCase):
    def test_pure_alpha_tone_matches_analytic_power(self):
        # A unit-amplitude sine's power is exactly 0.5 (Parseval), and a
        # 10 Hz tone sits entirely inside the 8-13 Hz band, so Welch's
        # band-integrated estimate should land close to 0.5.
        power = iv.compute_alpha_power(_sine_window(10.0), SAMPLING_RATE_HZ)
        self.assertAlmostEqual(power, 0.5, places=1)

    def test_out_of_band_tone_has_near_zero_alpha_power(self):
        power = iv.compute_alpha_power(_sine_window(2.0), SAMPLING_RATE_HZ)
        self.assertLess(power, 1e-6)

    def test_matches_an_independent_periodogram_on_recorded_eeg(self):
        # Cross-implementation check: a Hann-windowed periodogram written
        # directly with numpy's FFT (one-sided PSD in uV^2/Hz, integrated
        # over 8-13 Hz) must agree with the scipy-based core function on
        # the bundled recording. One 1-second segment of 500 samples means
        # Welch's method reduces to this periodogram: no overlap, no
        # averaging, 1 Hz bins.
        import pandas as pd

        frame = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "..", "sample_data", "ds005420_sub-5_O1_eyes_snippet.csv")
        )
        fs = iv.SAMPLING_RATE_HZ

        def numpy_alpha(window):
            centered = window - window.mean()
            hann = np.hanning(window.size + 1)[:-1]  # periodic Hann, scipy's default
            spectrum = np.abs(np.fft.rfft(centered * hann)) ** 2 / (fs * np.sum(hann**2))
            spectrum[1:-1] *= 2
            freqs = np.fft.rfftfreq(window.size, 1 / fs)
            band = (freqs >= 8) & (freqs <= 13)
            return float(np.trapezoid(spectrum[band], freqs[band])) if hasattr(np, "trapezoid") else float(np.trapz(spectrum[band], freqs[band]))

        for condition in ("eyes_closed", "eyes_open"):
            signal = frame.loc[frame["condition"] == condition, "voltage_uv"].to_numpy()
            for start in (0, 2500, 7000):
                window = signal[start : start + 500]
                with self.subTest(condition=condition, start=start):
                    self.assertAlmostEqual(
                        iv.compute_alpha_power(window, fs), numpy_alpha(window), places=6
                    )

    def test_off_bin_alpha_tone_keeps_its_power(self):
        # A 10.5 Hz tone falls between 1 Hz bins; the band integral should
        # still recover A^2/2 = 200 uV^2 to within 0.5%.
        t = np.arange(500) / 500.0
        power = iv.compute_alpha_power(20 * np.sin(2 * np.pi * 10.5 * t), 500.0)

        self.assertAlmostEqual(power / 200.0, 1.0, delta=0.005)

    def test_raises_when_nyquist_cannot_resolve_the_band(self):
        with self.assertRaises(ValueError):
            iv.compute_alpha_power(_sine_window(10.0, sampling_rate=10.0), 10.0)

    def test_raises_on_window_with_fewer_than_two_samples(self):
        with self.assertRaises(ValueError):
            iv.compute_alpha_power(np.array([1.0]), SAMPLING_RATE_HZ)


class TestClassifyEyesState(unittest.TestCase):
    def test_separable_conditions_classify_perfectly(self):
        # Ten 1-second windows of a clear in-band tone (eyes closed) vs.
        # ten windows of an out-of-band tone (eyes open): a threshold
        # classifier should separate these completely.
        eyes_closed = np.concatenate([_sine_window(10.0) for _ in range(10)])
        eyes_open = np.concatenate([_sine_window(2.0) for _ in range(10)])

        result = iv.classify_eyes_state(eyes_closed, eyes_open, SAMPLING_RATE_HZ)

        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.n_windows_eyes_closed, 10)
        self.assertEqual(result.n_windows_eyes_open, 10)

    def test_raises_when_a_condition_has_no_full_window(self):
        with self.assertRaises(ValueError):
            iv.classify_eyes_state(
                np.zeros(10), _sine_window(10.0), SAMPLING_RATE_HZ
            )


class TestApplyNoise(unittest.TestCase):
    def test_same_seed_is_deterministic(self):
        signal = _sine_window(10.0)
        first = iv.apply_noise(signal, sd=5.0, seed=42)
        second = iv.apply_noise(signal, sd=5.0, seed=42)
        np.testing.assert_array_equal(first, second)

    def test_zero_sd_returns_signal_unchanged_in_expectation(self):
        signal = _sine_window(10.0)
        noised = iv.apply_noise(signal, sd=0.0, seed=1)
        np.testing.assert_allclose(noised, signal)

    def test_raises_on_negative_sd(self):
        with self.assertRaises(ValueError):
            iv.apply_noise(_sine_window(10.0), sd=-1.0, seed=1)


class TestApplyTemporalDegradation(unittest.TestCase):
    def test_factor_one_is_a_no_op(self):
        signal = _sine_window(10.0)
        degraded, new_rate = iv.apply_temporal_degradation(signal, SAMPLING_RATE_HZ, factor=1)
        np.testing.assert_array_equal(degraded, signal)
        self.assertEqual(new_rate, SAMPLING_RATE_HZ)

    def test_effective_sampling_rate_divides_by_factor(self):
        signal = np.concatenate([_sine_window(10.0) for _ in range(3)])
        _, new_rate = iv.apply_temporal_degradation(signal, SAMPLING_RATE_HZ, factor=10)
        self.assertEqual(new_rate, 50.0)

    def test_raises_on_factor_below_one(self):
        with self.assertRaises(ValueError):
            iv.apply_temporal_degradation(_sine_window(10.0), SAMPLING_RATE_HZ, factor=0)


class TestStripMetadata(unittest.TestCase):
    def test_removes_only_the_illustrative_fields(self):
        fields = {
            "device_serial_number": "SN-1",
            "recording_timestamp": "2026-01-01T00:00:00Z",
            "subject_session_code": "sub-5",
            "task": "eyes_closed",
        }

        stripped = iv.strip_metadata(fields)

        self.assertEqual(stripped, {"task": "eyes_closed"})

    def test_empty_fields_returns_empty(self):
        self.assertEqual(iv.strip_metadata({}), {})


if __name__ == "__main__":
    unittest.main()
