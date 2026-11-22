"""Independent, spec-derived adversarial tests for core.intervention.

Written from the written spec only, without reading the implementation or the
sibling test file. Hand-calculated values derive from the periodic Hann window
(the window scipy.signal.welch uses):

For an N-sample, 1 s window (fs = N, so 1 Hz bin spacing) a sine of amplitude A
on an exact bin k has Hann DFT coefficients |X[k]| = A N / 4 and
|X[k +/- 1]| = A N / 8, and sum(w^2) = 3N/8. One-sided density
2|X|^2 / (fs * sum(w^2)) gives PSD[k] = A^2/3 and PSD[k +/- 1] = A^2/12, zero
elsewhere. Trapezoid integration over 8..13 Hz with df = 1 then gives:
  tone at 10 Hz : A^2/12 + A^2/3 + A^2/12           = A^2/2
  tone at  8 Hz : 0.5*A^2/3 + A^2/12                 = A^2/4  (half)
  tone at  7 Hz : 0.5*A^2/12                         = A^2/24 (1/12 of A^2/2)
  tone at  6 Hz : 0
"""

import dataclasses
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import intervention as iv  # noqa: E402

FS = 500.0
N = 500


def tone(freq, amp=1.0, n=N, fs=FS, phase=0.0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq * t + phase)


# ---------------------------------------------------------------- constants

def test_module_constants():
    # Spec: ALPHA_BAND_HZ == (8.0, 13.0), SAMPLING_RATE_HZ == 500.0, WINDOW_SECONDS == 1.0
    assert tuple(iv.ALPHA_BAND_HZ) == (8.0, 13.0)
    assert iv.SAMPLING_RATE_HZ == 500.0
    assert iv.WINDOW_SECONDS == 1.0


def test_nonfinite_error_is_valueerror_subclass():
    # Spec: NonFiniteWindowError is a subclass of ValueError
    assert issubclass(iv.NonFiniteWindowError, ValueError)


# ------------------------------------------------------- hand-calculated values

def test_unit_sine_10hz_equals_half():
    # Spec: pure sine amplitude A at 10 Hz, 1 s at 500 Hz -> A^2/2 (Parseval)
    p = iv.compute_alpha_power(tone(10.0, 1.0), FS)
    assert isinstance(p, float)
    assert p == pytest.approx(0.5, rel=1e-10)


@pytest.mark.parametrize("phase", [0.0, 0.3, np.pi / 2, 2.0, np.pi])
def test_sine_10hz_phase_independent(phase):
    # Spec: A^2/2 for a pure 10 Hz sine; integer-cycle tone so phase must not matter
    p = iv.compute_alpha_power(tone(10.0, 3.0, phase=phase), FS)
    assert p == pytest.approx(4.5, rel=1e-9)


def test_interior_band_tone_11hz_full_power():
    # Spec: Parseval A^2/2 for an in-band tone; 11 Hz leaks only into 10 and 12
    # Hz, both interior trapezoid nodes, so the hand calc gives exactly A^2/2
    assert iv.compute_alpha_power(tone(11.0, 2.0), FS) == pytest.approx(2.0, rel=1e-9)


@pytest.mark.parametrize("freq", [9.0, 12.0])
def test_tone_adjacent_to_edge_trapezoid_weight(freq):
    # Spec: trapezoid rule over bins 8..13. A 9 Hz (12 Hz) tone leaks A^2/12 into
    # the 8 Hz (13 Hz) endpoint bin, which the trapezoid weights by 1/2:
    # 0.5*A^2/12 + A^2/3 + A^2/12 = 11/24 A^2  (A=2 -> 11/6)
    # [Test corrected after first run: originally asserted A^2/2, which forgot
    # the half endpoint weight.]
    assert iv.compute_alpha_power(tone(freq, 2.0), FS) == pytest.approx(11 / 6, rel=1e-9)


def test_amplitude_scaling_seeded_sweep():
    # Spec: result scales exactly with A^2
    rng = np.random.default_rng(12345)
    amps = rng.uniform(0.01, 1000.0, size=60)
    for a in amps:
        assert iv.compute_alpha_power(tone(10.0, a), FS) == pytest.approx(a * a / 2, rel=1e-9)


def test_amplitude_scaling_on_arbitrary_signal():
    # Spec: result units are input units squared -> scaling a signal by c
    # multiplies the power by c^2 (checked on random, non-tonal signals)
    rng = np.random.default_rng(7)
    for _ in range(30):
        x = rng.normal(size=N)
        c = rng.uniform(-50, 50)
        base = iv.compute_alpha_power(x, FS)
        assert iv.compute_alpha_power(c * x, FS) == pytest.approx(c * c * base, rel=1e-9)


def test_superposition_of_inband_and_outband_tone():
    # Spec: integrated 8..13 Hz density; a 40 Hz tone (far outside band,
    # orthogonal to 10 Hz bin) adds nothing to the 10 Hz result
    x = tone(10.0, 2.0) + tone(40.0, 5.0)
    assert iv.compute_alpha_power(x, FS) == pytest.approx(2.0, rel=1e-9)


def test_two_inband_tones_add():
    # Spec: power is integral of PSD; tones at 9 and 12 Hz have non-overlapping
    # Hann leakage (8-10 and 11-13), so powers add. Each loses half its endpoint
    # leakage bin (see trapezoid test): 11/24*1 + 11/24*4 = 55/24
    # [Test corrected after first run: originally expected 2.5.]
    x = tone(9.0, 1.0) + tone(12.0, 2.0)
    assert iv.compute_alpha_power(x, FS) == pytest.approx(55 / 24, rel=1e-9)


def test_zero_signal_is_zero():
    # Spec: power of a signal with nothing in band is 0
    assert iv.compute_alpha_power(np.zeros(N), FS) == 0.0


def test_constant_signal_is_zero():
    # Spec: mean removed -> a pure constant has zero alpha power
    assert iv.compute_alpha_power(np.full(N, 123.4), FS) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------- band edges

@pytest.mark.parametrize("freq", [8.0, 13.0])
def test_edge_tone_gives_half(freq):
    # Spec: unit tone exactly at 8 or 13 Hz yields half of A^2/2
    assert iv.compute_alpha_power(tone(freq), FS) == pytest.approx(0.25, rel=1e-9)


@pytest.mark.parametrize("freq", [7.0, 14.0])
def test_one_bin_outside_gives_one_twelfth(freq):
    # Spec: 7 or 14 Hz -> about 0.083 of A^2/2 (hand calc: exactly 1/12)
    p = iv.compute_alpha_power(tone(freq), FS)
    assert p / 0.5 == pytest.approx(1 / 12, rel=1e-6)
    assert p / 0.5 == pytest.approx(0.083, abs=5e-4)


@pytest.mark.parametrize("freq", [6.0, 15.0, 2.0, 25.0, 100.0, 200.0])
def test_far_outside_gives_zero(freq):
    # Spec: 6 or 15 Hz -> about 0 (extended to tones further away)
    assert iv.compute_alpha_power(tone(freq), FS) == pytest.approx(0.0, abs=1e-12)


def test_edge_scaling_with_amplitude():
    # Spec: band-edge fractions are fractions of A^2/2 -> scale with A^2
    for a in (0.5, 4.0, 77.0):
        assert iv.compute_alpha_power(tone(8.0, a), FS) == pytest.approx(a * a / 4, rel=1e-9)
        assert iv.compute_alpha_power(tone(14.0, a), FS) == pytest.approx(a * a / 24, rel=1e-6)


def test_edge_symmetry_low_vs_high():
    # Spec: documented edge behaviour is symmetric (8 vs 13, 7 vs 14, 6 vs 15)
    for lo, hi in [(8.0, 13.0), (7.0, 14.0)]:
        assert iv.compute_alpha_power(tone(lo), FS) == pytest.approx(
            iv.compute_alpha_power(tone(hi), FS), rel=1e-6)


# ---------------------------------------------------------- white noise

def test_white_noise_expected_band_power():
    # Spec: white noise of variance s2 -> band power ~ s2 * (13-8)/(fs/2) on
    # average. For s2 = 4: 4 * 5/250 = 0.08. With 3000 independent windows the
    # standard error of the mean is estimated from the data; we require the
    # mean to be within 5 SE (and within 5% relative, which 5 SE is well under).
    rng = np.random.default_rng(2024)
    s2 = 4.0
    reps = 3000
    vals = np.array([iv.compute_alpha_power(rng.normal(0, math.sqrt(s2), N), FS)
                     for _ in range(reps)])
    expected = s2 * 5.0 / (FS / 2)
    se = vals.std(ddof=1) / math.sqrt(reps)
    assert abs(vals.mean() - expected) < 5 * se, (vals.mean(), expected, se)
    assert vals.mean() == pytest.approx(expected, rel=0.05)
    assert np.all(vals >= 0)


def test_white_noise_scales_with_variance():
    # Spec: expected band power proportional to variance (same seed, scaled noise)
    rng = np.random.default_rng(99)
    base = [rng.normal(size=N) for _ in range(200)]
    m1 = np.mean([iv.compute_alpha_power(b, FS) for b in base])
    m9 = np.mean([iv.compute_alpha_power(3 * b, FS) for b in base])
    assert m9 == pytest.approx(9 * m1, rel=1e-9)


# ------------------------------------------------------- numerical precision

@pytest.mark.parametrize("amp", [1e-100, 1e-20, 1e-6, 1e6, 1e20, 1e100])
def test_tiny_and_huge_amplitudes(amp):
    # Spec: A^2/2 and exact A^2 scaling at extreme magnitudes
    p = iv.compute_alpha_power(tone(10.0, amp), FS)
    assert math.isfinite(p)
    assert p == pytest.approx(amp * amp / 2, rel=1e-9)


def test_float32_matches_float64():
    # Spec: accepts any 1-D numeric sequence; float32 input should agree with
    # float64 to float32 precision
    rng = np.random.default_rng(5)
    for _ in range(10):
        x = rng.normal(size=N) + tone(10.0, 2.0)
        p64 = iv.compute_alpha_power(x.astype(np.float64), FS)
        p32 = iv.compute_alpha_power(x.astype(np.float32), FS)
        assert isinstance(p32, float)
        assert p32 == pytest.approx(p64, rel=1e-4)


def test_integer_input_accepted():
    # Spec: accepts any 1-D sequence of numbers (ints included)
    # Integer input must give the same answer as the same values as float64.
    # [Test corrected after first run: originally compared to A^2/2 at rel 1e-4,
    # but rounding error is correlated with the tone, and the cross term
    # (~2*sqrt(P_signal*P_err) ~ 60) exceeds that tolerance.]
    x = np.round(tone(10.0, 1000.0)).astype(np.int64)
    p = iv.compute_alpha_power(x, FS)
    assert isinstance(p, float)
    assert p == pytest.approx(iv.compute_alpha_power(x.astype(np.float64), FS), rel=1e-12)
    assert p == pytest.approx(1000.0 ** 2 / 2, rel=1e-3)
    assert iv.compute_alpha_power(list(range(N)), FS) >= 0.0


def test_large_dc_offset_precision():
    # Spec: DC offset must not change result; a huge offset relative to the
    # signal stresses mean removal (1e6 offset on unit sine -> rel err ~1e-6 ok)
    p = iv.compute_alpha_power(tone(10.0, 1.0) + 1e6, FS)
    assert p == pytest.approx(0.5, rel=1e-4)


# ---------------------------------------------------------- invariants

def test_dc_offset_invariance_sweep():
    # Spec: adding a constant (DC offset) must not change the result
    rng = np.random.default_rng(11)
    for _ in range(40):
        x = rng.normal(size=N)
        c = rng.uniform(-1000, 1000)
        assert iv.compute_alpha_power(x + c, FS) == pytest.approx(
            iv.compute_alpha_power(x, FS), rel=1e-6, abs=1e-12)


def test_sign_flip_invariance():
    # Spec: power is units squared (|.|^2 of the spectrum) -> x and -x identical
    rng = np.random.default_rng(13)
    for _ in range(20):
        x = rng.normal(size=N)
        assert iv.compute_alpha_power(-x, FS) == pytest.approx(
            iv.compute_alpha_power(x, FS), rel=1e-12)


def test_time_reversal_tone_exact_value():
    # Spec: A^2/2 for a pure 10 Hz sine; a reversed integer-cycle sine is still
    # a 10 Hz sine (phase-shifted), so must also give A^2/2
    x = tone(10.0, 2.0, phase=0.4)
    assert iv.compute_alpha_power(x[::-1], FS) == pytest.approx(2.0, rel=1e-9)


def test_time_reversal_noise_approximate():
    # Spec: one-sided PSD (magnitude only). Time reversal preserves |DFT|, but
    # the periodic Hann window is not exactly symmetric under reversal (it is
    # shifted by one sample), so invariance is only approximate: allow 5%.
    rng = np.random.default_rng(17)
    for _ in range(20):
        x = rng.normal(size=N)
        assert iv.compute_alpha_power(x[::-1], FS) == pytest.approx(
            iv.compute_alpha_power(x, FS), rel=0.05)


def test_list_input_equals_array_input():
    # Spec: accepts any 1-D sequence of numbers (lists included)
    x = tone(10.0, 1.5)
    assert iv.compute_alpha_power(list(x), FS) == iv.compute_alpha_power(x, FS)
    assert iv.compute_alpha_power(tuple(x), FS) == iv.compute_alpha_power(x, FS)


def test_does_not_mutate_input():
    # Spec: pure function (CLAUDE.md core rule); mean removal must not modify caller data
    x = tone(10.0) + 5.0
    copy = x.copy()
    iv.compute_alpha_power(x, FS)
    np.testing.assert_array_equal(x, copy)


def test_non_default_sampling_rate():
    # Spec: sampling_rate parameter is honoured: 1 s at 250 Hz, 10 Hz tone -> A^2/2
    x = tone(10.0, 2.0, n=250, fs=250.0)
    assert iv.compute_alpha_power(x, 250.0) == pytest.approx(2.0, rel=1e-9)


def test_wrong_sampling_rate_changes_result():
    # Spec: frequency axis comes from sampling_rate; telling it fs=1000 makes the
    # 10 Hz-at-500 tone a 20 Hz tone -> out of band -> ~0
    assert iv.compute_alpha_power(tone(10.0), 1000.0) == pytest.approx(0.0, abs=1e-12)


def test_determinism_compute():
    # Spec: pure function -> identical output on repeated calls
    x = np.random.default_rng(3).normal(size=N)
    assert iv.compute_alpha_power(x, FS) == iv.compute_alpha_power(x, FS)


# ---------------------------------------------------------- invalid inputs

@pytest.mark.parametrize("bad", [
    np.zeros((10, 50)),          # 2-D
    np.zeros((N, 1)),            # column vector is still 2-D
    np.float64(3.0),             # 0-D scalar array
    5.0,                         # bare scalar
])
def test_non_1d_raises(bad):
    # Spec: ValueError for a non-1-D input
    with pytest.raises(ValueError):
        iv.compute_alpha_power(bad, FS)


@pytest.mark.parametrize("bad", [[], [1.0], np.array([2.0])])
def test_fewer_than_two_samples_raises(bad):
    # Spec: ValueError for fewer than 2 samples
    with pytest.raises(ValueError):
        iv.compute_alpha_power(bad, FS)


@pytest.mark.parametrize("bad", [["a", "b", "c"], [1.0, "x", 2.0], [1.0, None, 2.0],
                                 [object(), object()]])
def test_non_numeric_raises_valueerror(bad):
    # Spec: ValueError for non-numeric content (not TypeError)
    with pytest.raises(ValueError):
        iv.compute_alpha_power(bad, FS)


@pytest.mark.parametrize("fs", [0.0, 0, -1.0, -500.0, float("nan"), float("inf"),
                                float("-inf")])
def test_bad_sampling_rate_raises(fs):
    # Spec: ValueError when sampling_rate is not a positive finite number
    with pytest.raises(ValueError):
        iv.compute_alpha_power(tone(10.0), fs)


def test_non_numeric_sampling_rate_raises_valueerror():
    # Spec: ValueError when sampling_rate is not a positive finite number
    # (a string is not a number at all)
    with pytest.raises(ValueError):
        iv.compute_alpha_power(tone(10.0), "500")


@pytest.mark.parametrize("fs", [16.0, 10.0, 1.0, 15.999])
def test_nyquist_at_or_below_band_low_raises(fs):
    # Spec: ValueError when Nyquist (fs/2) is at or below 8 Hz (fs=16 is the
    # boundary: Nyquist exactly 8 must raise)
    with pytest.raises(ValueError):
        iv.compute_alpha_power(np.random.default_rng(0).normal(size=64), fs)


def test_nyquist_just_above_band_low_accepted():
    # Spec: only Nyquist <= 8 raises; fs=17 (Nyquist 8.5) must be accepted and
    # return a finite non-negative power
    x = np.random.default_rng(1).normal(size=17)
    p = iv.compute_alpha_power(x, 17.0)
    assert math.isfinite(p) and p >= 0


@pytest.mark.parametrize("val", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("pos", [0, 250, N - 1])
def test_nonfinite_sample_raises_specific_error(val, pos):
    # Spec: NonFiniteWindowError when any sample is NaN or +/-inf
    x = tone(10.0)
    x[pos] = val
    with pytest.raises(iv.NonFiniteWindowError):
        iv.compute_alpha_power(x, FS)


def test_nonfinite_in_list_raises_specific_error():
    # Spec: NonFiniteWindowError for NaN, also via list input
    x = list(tone(10.0))
    x[3] = float("nan")
    with pytest.raises(iv.NonFiniteWindowError):
        iv.compute_alpha_power(x, FS)


# ================================================== classify_eyes_state

def windows_of(amps, freq=10.0):
    """Concatenate 1 s windows of 10 Hz tones with given amplitudes; each
    window's alpha power is exactly amp^2/2."""
    return np.concatenate([tone(freq, a) for a in amps])


def test_classify_hand_calculated():
    # Spec: threshold = (mean closed + mean open)/2; closed correct iff > thr;
    # open correct iff <= thr; accuracy = correct/total.
    # closed amps [2,4] -> powers [2,8] mean 5; open amps [1,3] -> [0.5,4.5] mean 2.5
    # threshold 3.75; closed correct: 8 only; open correct: 0.5 only -> 2/4
    r = iv.classify_eyes_state(windows_of([2, 4]), windows_of([1, 3]), FS)
    assert r.threshold == pytest.approx(3.75, rel=1e-9)
    assert r.accuracy == pytest.approx(0.5)
    assert r.alpha_power_eyes_closed == pytest.approx((2.0, 8.0), rel=1e-9)
    assert r.alpha_power_eyes_open == pytest.approx((0.5, 4.5), rel=1e-9)
    assert isinstance(r.alpha_power_eyes_closed, tuple)
    assert isinstance(r.alpha_power_eyes_open, tuple)
    assert r.n_windows_eyes_closed == 2 and r.n_windows_eyes_open == 2
    assert r.n_dropped_eyes_closed == 0 and r.n_dropped_eyes_open == 0


def test_classify_perfect_separation():
    # Spec: accuracy = correct/total; well-separated conditions -> 1.0
    r = iv.classify_eyes_state(windows_of([10, 11, 12]), windows_of([1, 1.5, 2, 0.5]), FS)
    assert r.accuracy == 1.0
    assert r.n_windows_eyes_closed == 3 and r.n_windows_eyes_open == 4


def test_classify_inverted_conditions_zero_accuracy():
    # Spec: closed correct only when > threshold; if "closed" has less alpha
    # than "open" every window is wrong -> accuracy 0 (no auto flipping)
    r = iv.classify_eyes_state(windows_of([1, 1]), windows_of([10, 10]), FS)
    assert r.accuracy == 0.0


def test_classify_unequal_counts_accuracy_denominator():
    # Spec: accuracy = correct / total kept windows (pooled, not per-class mean)
    # closed amps [4] -> [8]; open amps [1,1,1,4] -> [0.5,0.5,0.5,8], mean 2.375
    # thr = (8 + 2.375)/2 = 5.1875; closed 8 > thr ok; open 0.5 x3 ok, 8 wrong -> 4/5
    r = iv.classify_eyes_state(windows_of([4]), windows_of([1, 1, 1, 4]), FS)
    assert r.threshold == pytest.approx(5.1875, rel=1e-9)
    assert r.accuracy == pytest.approx(4 / 5)


def test_classify_tie_at_threshold():
    # Spec: closed correct iff power > threshold (strict); open correct iff
    # power <= threshold. Identical single windows -> threshold equals both
    # powers exactly -> closed wrong, open right -> 0.5
    x = tone(10.0, 2.0) + np.random.default_rng(4).normal(size=N)
    r = iv.classify_eyes_state(x.copy(), x.copy(), FS)
    assert r.threshold == r.alpha_power_eyes_closed[0] == r.alpha_power_eyes_open[0]
    assert r.accuracy == 0.5


def test_classify_trailing_partial_window_discarded():
    # Spec: consecutive non-overlapping 1 s windows; trailing partial discarded.
    # 2.5 s closed -> 2 windows, 1.999 s open -> 1 window. The trailing partial
    # carries a huge tone that would dominate the power if it were used.
    closed = np.concatenate([windows_of([3, 3]), tone(10.0, 1000.0, n=250)])
    opened = np.concatenate([windows_of([1]), tone(10.0, 1000.0, n=499)])
    r = iv.classify_eyes_state(closed, opened, FS)
    assert r.n_windows_eyes_closed == 2
    assert r.n_windows_eyes_open == 1
    assert r.alpha_power_eyes_closed == pytest.approx((4.5, 4.5), rel=1e-9)
    assert r.alpha_power_eyes_open == pytest.approx((0.5,), rel=1e-9)


def test_classify_nan_in_trailing_partial_not_counted():
    # Spec: n_dropped counts skipped *windows* containing NaN/inf; the trailing
    # partial is discarded before windowing, so NaN there is neither a kept
    # window, a dropped window, nor an error (interpretation; see report)
    closed = np.concatenate([windows_of([3]), np.full(100, np.nan)])
    r = iv.classify_eyes_state(closed, windows_of([1]), FS)
    assert r.n_windows_eyes_closed == 1
    assert r.n_dropped_eyes_closed == 0


def test_classify_windows_are_consecutive_and_aligned():
    # Spec: consecutive non-overlapping windows starting at sample 0 -> per-window
    # powers equal compute_alpha_power on each 500-sample slice, in order
    rng = np.random.default_rng(21)
    closed = rng.normal(size=5 * N) + np.tile(tone(10.0, 3.0), 5)
    opened = rng.normal(size=3 * N + 123)
    r = iv.classify_eyes_state(closed, opened, FS)
    exp_c = tuple(iv.compute_alpha_power(closed[i * N:(i + 1) * N], FS) for i in range(5))
    exp_o = tuple(iv.compute_alpha_power(opened[i * N:(i + 1) * N], FS) for i in range(3))
    assert r.alpha_power_eyes_closed == pytest.approx(exp_c, rel=1e-12)
    assert r.alpha_power_eyes_open == pytest.approx(exp_o, rel=1e-12)
    thr = (np.mean(exp_c) + np.mean(exp_o)) / 2
    assert r.threshold == pytest.approx(thr, rel=1e-12)
    correct = sum(p > thr for p in exp_c) + sum(p <= thr for p in exp_o)
    assert r.accuracy == pytest.approx(correct / 8)


def test_classify_window_length_follows_sampling_rate():
    # Spec: window = WINDOW_SECONDS at the given sampling rate. At 250 Hz,
    # 1000 samples -> 4 windows (not 2 as with a hard-coded 500)
    closed = np.concatenate([tone(10.0, 3.0, n=250, fs=250.0)] * 4)
    opened = np.concatenate([tone(10.0, 1.0, n=250, fs=250.0)] * 4)
    r = iv.classify_eyes_state(closed, opened, 250.0)
    assert r.n_windows_eyes_closed == 4 and r.n_windows_eyes_open == 4
    assert r.alpha_power_eyes_closed == pytest.approx((4.5,) * 4, rel=1e-9)
    assert r.accuracy == 1.0


def test_classify_nan_dropping_counts_and_accuracy():
    # Spec: NaN/inf windows skipped and counted in matching n_dropped; threshold
    # and accuracy use only kept windows.
    # closed amps [2,4,1] -> [2,8,0.5]; open amps [1,3] -> [0.5,4.5].
    # Without drops: mean c = 3.5, mean o = 2.5, thr 3.0; correct: c 8; o 0.5 -> 2/5.
    # Drop closed window 3 (amp 1) with NaN and open window 2 (amp 3) with inf:
    # kept c [2,8] mean 5, kept o [0.5], thr 2.75 -> c 8 ok, 2 wrong; o ok -> 2/3
    closed = windows_of([2, 4, 1])
    opened = windows_of([1, 3])
    full = iv.classify_eyes_state(closed, opened, FS)
    assert full.threshold == pytest.approx(3.0, rel=1e-9)
    assert full.accuracy == pytest.approx(2 / 5)

    closed[2 * N + 17] = np.nan
    opened[N + 400] = np.inf
    r = iv.classify_eyes_state(closed, opened, FS)
    assert r.n_dropped_eyes_closed == 1
    assert r.n_dropped_eyes_open == 1
    assert r.n_windows_eyes_closed == 2
    assert r.n_windows_eyes_open == 1
    assert r.alpha_power_eyes_closed == pytest.approx((2.0, 8.0), rel=1e-9)
    assert r.alpha_power_eyes_open == pytest.approx((0.5,), rel=1e-9)
    assert r.threshold == pytest.approx(2.75, rel=1e-9)
    assert r.accuracy == pytest.approx(2 / 3)
    assert math.isfinite(r.threshold) and math.isfinite(r.accuracy)


def test_classify_multiple_nonfinite_in_one_window_counts_once():
    # Spec: n_dropped counts windows, not bad samples
    closed = windows_of([5, 5, 5])
    closed[N:N + 50] = np.nan
    closed[N + 100] = -np.inf
    r = iv.classify_eyes_state(closed, windows_of([1]), FS)
    assert r.n_dropped_eyes_closed == 1
    assert r.n_windows_eyes_closed == 2
    assert r.n_dropped_eyes_open == 0


def test_classify_dropped_seeded_sweep():
    # Spec: n_dropped equals number of windows with any non-finite sample,
    # and n_windows + n_dropped equals the number of full windows
    rng = np.random.default_rng(31)
    for _ in range(25):
        nc, no = rng.integers(2, 8, size=2)
        closed = rng.normal(size=nc * N + rng.integers(0, N))
        opened = rng.normal(size=no * N + rng.integers(0, N))
        bad_c = rng.choice(nc, size=rng.integers(0, nc), replace=False)
        bad_o = rng.choice(no, size=rng.integers(0, no), replace=False)
        for w in bad_c:
            closed[w * N + rng.integers(0, N)] = rng.choice([np.nan, np.inf, -np.inf])
        for w in bad_o:
            opened[w * N + rng.integers(0, N)] = rng.choice([np.nan, np.inf, -np.inf])
        r = iv.classify_eyes_state(closed, opened, FS)
        assert r.n_dropped_eyes_closed == len(bad_c)
        assert r.n_dropped_eyes_open == len(bad_o)
        assert r.n_windows_eyes_closed == nc - len(bad_c)
        assert r.n_windows_eyes_open == no - len(bad_o)
        assert all(math.isfinite(p) for p in r.alpha_power_eyes_closed + r.alpha_power_eyes_open)
        assert 0.0 <= r.accuracy <= 1.0


def test_classify_properties_match_tuple_lengths():
    # Spec: n_windows_* properties = lengths of the power tuples
    r = iv.classify_eyes_state(windows_of([2, 3, 4]), windows_of([1, 1]), FS)
    assert r.n_windows_eyes_closed == len(r.alpha_power_eyes_closed) == 3
    assert r.n_windows_eyes_open == len(r.alpha_power_eyes_open) == 2


def test_classify_result_is_frozen_dataclass():
    # Spec: result is a frozen dataclass with the listed fields
    r = iv.classify_eyes_state(windows_of([3]), windows_of([1]), FS)
    assert dataclasses.is_dataclass(r)
    names = {f.name for f in dataclasses.fields(r)}
    assert {"accuracy", "threshold", "alpha_power_eyes_closed", "alpha_power_eyes_open",
            "n_dropped_eyes_closed", "n_dropped_eyes_open"} <= names
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.accuracy = 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.threshold = 1.0
    assert isinstance(r, iv.ClassificationResult)


def test_classify_accepts_lists():
    # Spec (via compute_alpha_power): any 1-D numeric sequence, lists included
    a = iv.classify_eyes_state(list(windows_of([3, 2])), list(windows_of([1])), FS)
    b = iv.classify_eyes_state(windows_of([3, 2]), windows_of([1]), FS)
    assert a == b


def test_classify_determinism_and_no_mutation():
    # Spec: pure, deterministic; inputs not modified
    rng = np.random.default_rng(41)
    closed = rng.normal(size=4 * N) + np.tile(tone(10.0, 2.0), 4)
    opened = rng.normal(size=4 * N)
    closed[N + 3] = np.nan
    c0, o0 = closed.copy(), opened.copy()
    r1 = iv.classify_eyes_state(closed, opened, FS)
    r2 = iv.classify_eyes_state(closed, opened, FS)
    assert r1 == r2
    np.testing.assert_array_equal(closed, c0)
    np.testing.assert_array_equal(opened, o0)


def test_classify_threshold_dc_invariant():
    # Spec: compute_alpha_power is DC invariant, so per-condition DC offsets
    # must not move threshold or accuracy
    rng = np.random.default_rng(51)
    closed = rng.normal(size=3 * N) + np.tile(tone(10.0, 2.0), 3)
    opened = rng.normal(size=3 * N)
    a = iv.classify_eyes_state(closed, opened, FS)
    b = iv.classify_eyes_state(closed + 40.0, opened - 7.0, FS)
    assert b.threshold == pytest.approx(a.threshold, rel=1e-6)
    assert b.accuracy == a.accuracy


@pytest.mark.parametrize("short_closed", [True, False])
def test_classify_too_short_raises(short_closed):
    # Spec: ValueError when either condition has no usable full window (too short)
    short = tone(10.0, n=N - 1)
    ok = windows_of([2, 2])
    args = (short, ok) if short_closed else (ok, short)
    with pytest.raises(ValueError):
        iv.classify_eyes_state(*args, FS)


@pytest.mark.parametrize("short_closed", [True, False])
def test_classify_empty_raises(short_closed):
    # Spec: ValueError when either condition has no usable full window
    ok = windows_of([2])
    args = (np.array([]), ok) if short_closed else (ok, np.array([]))
    with pytest.raises(ValueError):
        iv.classify_eyes_state(*args, FS)


@pytest.mark.parametrize("which", ["closed", "open"])
def test_classify_all_windows_dropped_raises(which):
    # Spec: ValueError when every window of a condition is dropped
    bad = windows_of([2, 2])
    bad[10] = np.nan
    bad[N + 10] = np.inf
    ok = windows_of([1, 1])
    args = (bad, ok) if which == "closed" else (ok, bad)
    with pytest.raises(ValueError):
        iv.classify_eyes_state(*args, FS)


def test_classify_one_kept_window_suffices():
    # Spec: raises only when there is NO usable window; one kept window is enough
    closed = windows_of([4, 4])
    closed[5] = np.nan
    r = iv.classify_eyes_state(closed, windows_of([1]), FS)
    assert r.n_windows_eyes_closed == 1 and r.n_dropped_eyes_closed == 1
    assert r.accuracy == 1.0


def test_classify_noise_vs_alpha_seeded():
    # Spec: closed windows with a strong 10 Hz component (A=1, power 0.5) over
    # unit white noise (band power ~0.02) vs pure unit noise: separation is
    # ~25x the noise band power, so accuracy must be 1.0 across seeds
    for seed in range(10):
        rng = np.random.default_rng(seed)
        closed = rng.normal(size=10 * N) + np.tile(tone(10.0, 1.0), 10)
        opened = rng.normal(size=10 * N)
        r = iv.classify_eyes_state(closed, opened, FS)
        assert r.accuracy == 1.0
        assert 0.02 < r.threshold < 0.52
