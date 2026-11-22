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
            # The module's own alias, which works on NumPy 1 and 2.
            return float(iv._trapezoid(spectrum[band], freqs[band]))

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


# ---------------------------------------------------------------------
# Hardening: scaling, noise, band edges, precision, invalid input, and
# windows with NaN or infinite samples. Seeds are fixed, so every run
# checks the same inputs.
# ---------------------------------------------------------------------

_T = np.arange(500) / SAMPLING_RATE_HZ


def _tone(freq_hz: float, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq_hz * _T)


class TestAmplitudeScaling(unittest.TestCase):
    def test_power_is_half_the_squared_amplitude_across_a_seeded_sweep(self):
        # Parseval: a sine of amplitude A carries power A^2 / 2, and a 10 Hz
        # tone sits on a bin in the middle of the band, so all of it counts.
        rng = np.random.default_rng(7)
        for amplitude in rng.uniform(0.01, 500.0, size=200):
            with self.subTest(amplitude=float(amplitude)):
                power = iv.compute_alpha_power(_tone(10.0, amplitude), SAMPLING_RATE_HZ)
                self.assertAlmostEqual(power / (amplitude**2 / 2), 1.0, places=9)

    def test_scaling_any_signal_by_k_scales_power_by_k_squared(self):
        # Holds for any window, here random EEG-like noise, because power is
        # quadratic in the signal.
        rng = np.random.default_rng(11)
        for _ in range(50):
            window = rng.normal(0, 20, 500)
            k = float(rng.uniform(0.1, 10))
            with self.subTest(k=k):
                base = iv.compute_alpha_power(window, SAMPLING_RATE_HZ)
                scaled = iv.compute_alpha_power(k * window, SAMPLING_RATE_HZ)
                self.assertAlmostEqual(scaled / base, k**2, places=9)


class TestNoise(unittest.TestCase):
    def test_white_noise_band_power_matches_its_share_of_the_variance(self):
        # White noise of variance s^2 spreads it evenly over 0 to fs/2, so
        # 8 to 13 Hz holds s^2 * 5 / 250 = 2.0 uV^2 for s = 10 uV. Each
        # window's estimate varies by about 59%; over 2000 windows the
        # standard error is about 1.3%, so +/-5% is about four of them.
        rng = np.random.default_rng(20261122)
        powers = [iv.compute_alpha_power(rng.normal(0, 10, 500), SAMPLING_RATE_HZ) for _ in range(2000)]

        self.assertAlmostEqual(float(np.mean(powers)) / 2.0, 1.0, delta=0.05)

    def test_added_noise_raises_band_power_by_about_its_own(self):
        # Independent noise adds power: tone (0.5) + noise (about 0.02 * 1)
        # averages about 0.52 for unit tone and unit-variance noise.
        rng = np.random.default_rng(3)
        powers = [
            iv.compute_alpha_power(_tone(10.0) + rng.normal(0, 1, 500), SAMPLING_RATE_HZ)
            for _ in range(2000)
        ]

        self.assertAlmostEqual(float(np.mean(powers)), 0.52, delta=0.01)


class TestFrequencyBoundaries(unittest.TestCase):
    # Weights of a unit tone's power, A^2 / 2, that land in the band, for a
    # 1-second Hann window at 500 Hz with trapezoid integration over the
    # 8..13 Hz bins (see compute_alpha_power's docstring).
    EXPECTED = {6.0: 0.0, 7.0: 1 / 12, 8.0: 0.5, 10.0: 1.0, 13.0: 0.5, 14.0: 1 / 12, 15.0: 0.0}

    def test_tone_weights_at_and_around_the_band_edges(self):
        for freq, weight in self.EXPECTED.items():
            with self.subTest(freq=freq):
                power = iv.compute_alpha_power(_tone(freq), SAMPLING_RATE_HZ)
                self.assertAlmostEqual(power / 0.5, weight, places=6)

    def test_band_response_is_symmetric_about_its_centre(self):
        # 8..13 Hz is symmetric about 10.5 Hz, so tones equally far from the
        # centre count equally.
        for offset in (0.5, 1.5, 2.5, 3.5):
            with self.subTest(offset=offset):
                low = iv.compute_alpha_power(_tone(10.5 - offset), SAMPLING_RATE_HZ)
                high = iv.compute_alpha_power(_tone(10.5 + offset), SAMPLING_RATE_HZ)
                self.assertAlmostEqual(low, high, places=9)

    def test_lowest_rate_that_resolves_the_band(self):
        # Nyquist must sit above 8 Hz: 16 Hz sampling is refused, 18 Hz is not.
        with self.assertRaises(ValueError):
            iv.compute_alpha_power(np.ones(16), 16.0)
        self.assertGreaterEqual(iv.compute_alpha_power(np.sin(np.arange(18)), 18.0), 0.0)


class TestNumericalPrecision(unittest.TestCase):
    def test_tiny_and_huge_amplitudes_keep_relative_precision(self):
        for amplitude in (1e-9, 1e-6, 1e3, 1e6):
            with self.subTest(amplitude=amplitude):
                power = iv.compute_alpha_power(_tone(10.0, amplitude), SAMPLING_RATE_HZ)
                self.assertAlmostEqual(power / (amplitude**2 / 2), 1.0, places=9)

    def test_float32_input_agrees_with_float64_to_single_precision(self):
        power32 = iv.compute_alpha_power(_tone(10.0, 20.0).astype(np.float32), SAMPLING_RATE_HZ)
        power64 = iv.compute_alpha_power(_tone(10.0, 20.0), SAMPLING_RATE_HZ)

        self.assertAlmostEqual(power32 / power64, 1.0, places=6)

    def test_large_dc_offset_does_not_change_band_power(self):
        # The mean is removed before the spectrum; an offset 10^4 times the
        # tone must leave the result unchanged to near machine precision.
        with_offset = iv.compute_alpha_power(_tone(10.0) + 1e4, SAMPLING_RATE_HZ)

        self.assertAlmostEqual(with_offset, 0.5, places=8)

    def test_sign_flip_leaves_power_unchanged(self):
        # Power is quadratic, so negating every sample changes nothing. Time
        # reversal is deliberately not tested as exact: scipy's default
        # periodic Hann window is offset by one sample, so a reversed
        # window's spectrum differs slightly (about 0.3% on this window).
        rng = np.random.default_rng(5)
        window = rng.normal(0, 30, 500)
        base = iv.compute_alpha_power(window, SAMPLING_RATE_HZ)

        self.assertAlmostEqual(iv.compute_alpha_power(-window, SAMPLING_RATE_HZ), base, places=9)


class TestInvalidInput(unittest.TestCase):
    def test_bad_sampling_rates_raise(self):
        for rate in (0, -500.0, float("nan"), float("inf"), "500"):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                iv.compute_alpha_power(_tone(10.0), rate)

    def test_bad_windows_raise_value_error(self):
        for window in (np.vstack([_tone(10.0)] * 2), np.array([]), ["a", "b", "c"], [[1, 2], [3, 4]]):
            with self.subTest(window=str(window)[:30]), self.assertRaises(ValueError):
                iv.compute_alpha_power(window, SAMPLING_RATE_HZ)

    def test_a_plain_list_is_accepted(self):
        self.assertAlmostEqual(iv.compute_alpha_power(list(_tone(10.0)), SAMPLING_RATE_HZ), 0.5, places=9)

    def test_nan_or_infinite_samples_raise_a_typed_error(self):
        for bad in (np.nan, np.inf, -np.inf):
            window = _tone(10.0)
            window[250] = bad
            with self.subTest(bad=bad), self.assertRaises(iv.NonFiniteWindowError):
                iv.compute_alpha_power(window, SAMPLING_RATE_HZ)
        self.assertTrue(issubclass(iv.NonFiniteWindowError, ValueError))


class TestNonFiniteWindowsAreSkipped(unittest.TestCase):
    def _signals(self):
        # 3 seconds per condition: strong alpha closed, weak alpha open.
        t = np.arange(1500) / SAMPLING_RATE_HZ
        return 20 * np.sin(2 * np.pi * 10 * t), 2 * np.sin(2 * np.pi * 10 * t)

    def test_one_bad_sample_drops_only_its_window(self):
        closed, opened = self._signals()
        closed[700] = np.nan  # inside the second 1-second window

        result = iv.classify_eyes_state(closed, opened, SAMPLING_RATE_HZ)

        self.assertEqual((result.n_dropped_eyes_closed, result.n_dropped_eyes_open), (1, 0))
        self.assertEqual((result.n_windows_eyes_closed, result.n_windows_eyes_open), (2, 3))
        self.assertEqual(result.accuracy, 1.0)

    def test_threshold_uses_only_kept_windows(self):
        # Kept closed windows each have power 200, open windows 2, so the
        # midpoint is (200 + 2) / 2 = 101 regardless of the dropped window.
        closed, opened = self._signals()
        closed[0] = np.inf

        result = iv.classify_eyes_state(closed, opened, SAMPLING_RATE_HZ)

        self.assertAlmostEqual(result.threshold, 101.0, places=6)

    def test_clean_signals_drop_nothing(self):
        result = iv.classify_eyes_state(*self._signals(), SAMPLING_RATE_HZ)

        self.assertEqual((result.n_dropped_eyes_closed, result.n_dropped_eyes_open), (0, 0))

    def test_a_condition_with_every_window_dropped_raises(self):
        closed, opened = self._signals()
        opened[::400] = np.nan  # one bad sample in each of the three windows

        with self.assertRaises(ValueError) as caught:
            iv.classify_eyes_state(closed, opened, SAMPLING_RATE_HZ)
        self.assertIn("3 dropped", str(caught.exception))


class TestThresholdTies(unittest.TestCase):
    def test_identical_conditions_score_exactly_chance(self):
        # Every window has the same power P, so the threshold is P. Closed
        # windows need power > P (all wrong); open windows need power <= P
        # (all right): accuracy is exactly 0.5. This pins the tie rule, so
        # an open window at the threshold counts as eyes open.
        # Two copies of one window: (P + P) / 2 == P exactly in floating
        # point, so each window sits exactly on the threshold. (Windows cut
        # from one long sine differ in the last bit and never tie.)
        window = 10 * np.sin(2 * np.pi * 10 * np.arange(500) / SAMPLING_RATE_HZ)
        signal = np.tile(window, 2)

        result = iv.classify_eyes_state(signal, signal.copy(), SAMPLING_RATE_HZ)

        self.assertEqual(result.threshold, result.alpha_power_eyes_open[0])
        self.assertEqual(result.accuracy, 0.5)
