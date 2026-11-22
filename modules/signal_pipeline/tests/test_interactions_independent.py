"""Independent, spec-derived adversarial tests for filtering, source_lesson,
power_explainer.welch_steps and hemodynamics.response_components.

Written from the specification only; the implementation source was not read.
"""
import dataclasses
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import signal as sps
from scipy.integrate import trapezoid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import filtering as f, source_lesson as s, power_explainer as pe, hemodynamics as h, intervention as iv  # noqa: E402

SAMPLE = os.path.join(
    os.path.dirname(__file__), "..", "sample_data", "ds005420_sub-5_O1_eyes_snippet.csv"
)
FS_REC = 500.0


@pytest.fixture(scope="module")
def recorded():
    df = pd.read_csv(SAMPLE)
    return {
        cond: df.loc[df["condition"] == cond, "voltage_uv"].to_numpy(dtype=float)
        for cond in ("eyes_closed", "eyes_open")
    }


def _assert_frozen(obj):
    assert dataclasses.is_dataclass(obj)
    field = dataclasses.fields(obj)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, field, None)


def _sos(fs, low, high, order):
    return sps.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def _fit_amplitude(x, freq, fs, start):
    """Least-squares amplitude of a sinusoid at freq in x[start:]."""
    t = np.arange(len(x))[start:] / fs
    A = np.column_stack([np.sin(2 * np.pi * freq * t), np.cos(2 * np.pi * freq * t)])
    coef, *_ = np.linalg.lstsq(A, x[start:], rcond=None)
    return float(np.hypot(*coef))


# ---------------------------------------------------------------------------
# filtering.band_pass
# ---------------------------------------------------------------------------

# Spec: band_pass -> frozen dataclass with raw, filtered (tuples, same length), params echoed.
def test_band_pass_result_fields_and_frozen():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    r = f.band_pass(x, 250.0, low_hz=7.0, high_hz=14.0, order=3, zero_phase=False)
    assert isinstance(r.raw, tuple) and isinstance(r.filtered, tuple)
    assert len(r.raw) == len(r.filtered) == 1000
    np.testing.assert_allclose(r.raw, x)
    assert (r.low_hz, r.high_hz, r.order, r.zero_phase, r.sampling_rate) == (7.0, 14.0, 3, False, 250.0)
    _assert_frozen(r)


# Spec: defaults low_hz=8, high_hz=13, order=4, zero_phase=True.
def test_band_pass_defaults():
    x = np.random.default_rng(1).normal(size=600)
    r = f.band_pass(x, 200.0)
    assert (r.low_hz, r.high_hz, r.order, r.zero_phase) == (8.0, 13.0, 4, True)
    ref = sps.sosfiltfilt(_sos(200.0, 8.0, 13.0, 4), x)
    np.testing.assert_allclose(r.filtered, ref, atol=1e-10)


# Spec: Butterworth SOS applied with sosfiltfilt when zero_phase else sosfilt (seeded sweep).
@pytest.mark.parametrize("seed", range(8))
def test_band_pass_matches_scipy_reference(seed):
    rng = np.random.default_rng(seed)
    fs = float(rng.choice([100.0, 250.0, 500.0, 1000.0]))
    low = float(rng.uniform(0.5, fs / 4))
    high = float(rng.uniform(low + 0.5, fs / 2 - 1))
    order = int(rng.integers(1, 6))
    x = rng.normal(size=int(rng.integers(300, 2000))) * 10 + rng.normal() * 50
    sos = _sos(fs, low, high, order)
    for zp, ref in ((True, sps.sosfiltfilt(sos, x)), (False, sps.sosfilt(sos, x))):
        r = f.band_pass(x, fs, low_hz=low, high_hz=high, order=order, zero_phase=zp)
        np.testing.assert_allclose(r.filtered, ref, rtol=1e-9, atol=1e-9)


# Spec: kept_variance_share = mean(filtered^2) / mean((raw - mean(raw))^2).
@pytest.mark.parametrize("zp", [True, False])
def test_kept_variance_share_formula_on_recorded(recorded, zp):
    x = recorded["eyes_closed"]
    r = f.band_pass(x, FS_REC, zero_phase=zp)
    filt = np.asarray(r.filtered)
    expected = np.mean(filt ** 2) / np.mean((x - x.mean()) ** 2)
    assert r.kept_variance_share == pytest.approx(expected, rel=1e-12)
    assert 0.0 < r.kept_variance_share < 1.0


# Spec: kept_variance_share is 0 when the denominator is 0 (constant signal).
# Constants whose float mean is exact, so the denominator is exactly 0.
@pytest.mark.parametrize("c", [0.0, 2.0, -1e6])
def test_kept_variance_share_zero_for_constant(c):
    r = f.band_pass(np.full(500, c), 250.0)
    assert r.kept_variance_share == 0.0
    assert all(math.isfinite(v) for v in r.filtered)


# Spec: kept_variance_share is 0 if the denominator is 0. For a constant like 3.7 the float
# mean is off by ~1e-15, so the denominator is ~1e-31 rather than 0 and the ratio of two
# rounding residues is returned (~0.0018). Literal spec is met; the intent (constant -> 0)
# is not. Was a strict xfail documenting the gap; the implementation now
# treats rounding residue as zero, so it is a normal test.
def test_kept_variance_share_constant_with_inexact_mean():
    r = f.band_pass(np.full(500, 3.7), 250.0)
    assert r.kept_variance_share == pytest.approx(0.0, abs=1e-9)


# Spec: kept_variance_share definition; band-dependence on recorded data: eyes-closed
# alpha should keep a larger share in 8-13 Hz than eyes-open (Berger effect, O1).
def test_kept_variance_share_eyes_closed_exceeds_eyes_open(recorded):
    closed = f.band_pass(recorded["eyes_closed"], FS_REC).kept_variance_share
    opened = f.band_pass(recorded["eyes_open"], FS_REC).kept_variance_share
    assert closed > opened


# Spec: filter is a linear (LTI) Butterworth filter -> superposition and scaling hold.
@pytest.mark.parametrize("zp", [True, False])
def test_band_pass_linearity(recorded, zp):
    rng = np.random.default_rng(42)
    a = recorded["eyes_closed"][:3000]
    b = rng.normal(size=3000) * 20
    for alpha, beta in [(2.0, -3.0), (0.5, 0.0), (-1.0, 7.25)]:
        fa = np.asarray(f.band_pass(a, FS_REC, zero_phase=zp).filtered)
        fb = np.asarray(f.band_pass(b, FS_REC, zero_phase=zp).filtered)
        fab = np.asarray(f.band_pass(alpha * a + beta * b, FS_REC, zero_phase=zp).filtered)
        np.testing.assert_allclose(fab, alpha * fa + beta * fb, atol=1e-8 * (1 + np.abs(fab).max()))


# Spec: sosfilt (causal) when zero_phase=False -> output at n depends only on input <= n.
def test_causal_filter_does_not_see_the_future():
    rng = np.random.default_rng(3)
    x = rng.normal(size=800)
    y = x.copy()
    y[500:] += rng.normal(size=300) * 100
    fx = np.asarray(f.band_pass(x, 250.0, zero_phase=False).filtered)
    fy = np.asarray(f.band_pass(y, 250.0, zero_phase=False).filtered)
    np.testing.assert_array_equal(fx[:500], fy[:500])
    assert not np.allclose(fx[500:], fy[500:])


# Spec: band_pass is a pure function -> deterministic, input not mutated.
def test_band_pass_deterministic_and_non_mutating():
    x = np.random.default_rng(5).normal(size=700)
    copy = x.copy()
    r1 = f.band_pass(x, 250.0)
    r2 = f.band_pass(x, 250.0)
    assert r1 == r2
    np.testing.assert_array_equal(x, copy)


# Spec: list input accepted as 1-D signal (tuples out regardless of input type).
def test_band_pass_accepts_list():
    x = list(np.random.default_rng(6).normal(size=400))
    r = f.band_pass(x, 250.0)
    assert len(r.filtered) == 400


# Spec: FilteringError is a ValueError subclass.
def test_filtering_error_is_value_error():
    assert issubclass(f.FilteringError, ValueError)


# Spec: raises when sampling_rate not positive finite.
@pytest.mark.parametrize("fs", [0.0, -250.0, float("nan"), float("inf"), -float("inf")])
def test_band_pass_bad_sampling_rate(fs):
    with pytest.raises(f.FilteringError):
        f.band_pass(np.zeros(500), fs)


# Spec: raises unless 0 < low < high < fs/2.
@pytest.mark.parametrize(
    "low,high",
    [(0.0, 13.0), (-1.0, 13.0), (13.0, 13.0), (14.0, 13.0), (8.0, 125.0), (8.0, 130.0),
     (float("nan"), 13.0), (8.0, float("nan")), (8.0, float("inf"))],
)
def test_band_pass_bad_band(low, high):
    with pytest.raises(f.FilteringError):
        f.band_pass(np.zeros(500), 250.0, low_hz=low, high_hz=high)


# Spec: 0 < low < high < fs/2 boundary just inside is accepted.
def test_band_pass_band_just_inside_nyquist():
    r = f.band_pass(np.random.default_rng(0).normal(size=500), 250.0, low_hz=1.0, high_hz=124.9)
    assert len(r.filtered) == 500


# Spec: raises when order not an int >= 1.
@pytest.mark.parametrize("order", [0, -2, 2.5, 4.0, "4", None])
def test_band_pass_bad_order(order):
    with pytest.raises(f.FilteringError):
        f.band_pass(np.zeros(500), 250.0, order=order)


# Spec: raises when signal not 1-D.
@pytest.mark.parametrize("sig", [np.zeros((10, 50)), np.zeros((500, 1)), np.float64(1.0)])
def test_band_pass_not_1d(sig):
    with pytest.raises(f.FilteringError):
        f.band_pass(sig, 250.0)


# Spec: raises on NaN/inf in signal.
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_band_pass_non_finite(bad):
    x = np.zeros(500)
    x[250] = bad
    with pytest.raises(f.FilteringError):
        f.band_pass(x, 250.0)


# Spec: raises when length <= 3*(2*n_sections + 1); accepts one more sample.
@pytest.mark.parametrize("order", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("zp", [True, False])
def test_band_pass_length_threshold(order, zp):
    n_sections = _sos(250.0, 8.0, 13.0, order).shape[0]
    limit = 3 * (2 * n_sections + 1)
    rng = np.random.default_rng(order)
    with pytest.raises(f.FilteringError):
        f.band_pass(rng.normal(size=limit), 250.0, order=order, zero_phase=zp)
    r = f.band_pass(rng.normal(size=limit + 1), 250.0, order=order, zero_phase=zp)
    assert len(r.filtered) == limit + 1


# Spec: empty signal is a degenerate input -> length rule raises FilteringError.
def test_band_pass_empty():
    with pytest.raises(f.FilteringError):
        f.band_pass([], 250.0)


# ---------------------------------------------------------------------------
# filtering.amplitude_gain
# ---------------------------------------------------------------------------

# Spec: at the cutoffs gain is 1/sqrt(2) single pass and 1/2 zero-phase.
@pytest.mark.parametrize("order", [1, 2, 4, 6])
@pytest.mark.parametrize("fs,low,high", [(500.0, 8.0, 13.0), (250.0, 1.0, 40.0), (1000.0, 70.0, 150.0)])
def test_amplitude_gain_at_cutoffs(order, fs, low, high):
    for fc in (low, high):
        assert f.amplitude_gain(fc, fs, low, high, order, zero_phase=False) == pytest.approx(1 / math.sqrt(2), abs=1e-6)
        assert f.amplitude_gain(fc, fs, low, high, order, zero_phase=True) == pytest.approx(0.5, abs=1e-6)


# Spec: |H| from sosfreqz, squared when zero_phase (identity across a sweep).
def test_amplitude_gain_zero_phase_is_square_of_single_pass():
    for freq in np.linspace(0.5, 120, 60):
        single = f.amplitude_gain(freq, 250.0, 8.0, 13.0, 4, zero_phase=False)
        double = f.amplitude_gain(freq, 250.0, 8.0, 13.0, 4, zero_phase=True)
        assert double == pytest.approx(single ** 2, rel=1e-9, abs=1e-15)
        assert 0.0 <= single <= 1.0 + 1e-9


# Spec: |H(freq)| from sosfreqz of butter(order, [low, high], bandpass, fs).
def test_amplitude_gain_matches_sosfreqz():
    sos = _sos(500.0, 8.0, 13.0, 4)
    for freq in [1.0, 5.0, 8.0, 10.0, 10.2, 13.0, 20.0, 60.0]:
        _, hh = sps.sosfreqz(sos, worN=[freq], fs=500.0)
        assert f.amplitude_gain(freq, 500.0, 8.0, 13.0, 4, zero_phase=False) == pytest.approx(abs(hh[0]), rel=1e-9)


# Spec: amplitude_gain vs measured settled amplitude of a filtered sinusoid (frequency sweep).
@pytest.mark.parametrize("zp", [True, False])
def test_amplitude_gain_frequency_sweep_matches_measured(zp):
    fs, low, high, order = 250.0, 8.0, 13.0, 4
    n = int(40 * fs)
    t = np.arange(n) / fs
    for freq in [2.0, 5.0, 7.0, 8.0, 9.0, 10.5, 12.0, 13.0, 14.0, 16.0, 20.0, 35.0, 60.0]:
        x = 3.0 * np.sin(2 * np.pi * freq * t + 0.3)
        y = np.asarray(f.band_pass(x, fs, low, high, order, zero_phase=zp).filtered)
        # zero-phase: avoid both edges; causal: skip the initial transient
        seg = y[: int(30 * fs)] if zp else y
        start = int(10 * fs)
        measured = _fit_amplitude(seg, freq, fs, start) / 3.0
        predicted = f.amplitude_gain(freq, fs, low, high, order, zero_phase=zp)
        assert measured == pytest.approx(predicted, abs=2e-3), freq


# Spec: amplitude_gain - passband centre near unity, far stopband near zero.
def test_amplitude_gain_shape():
    centre = math.sqrt(8.0 * 13.0)  # Butterworth band-pass peak is at the geometric centre
    assert f.amplitude_gain(centre, 500.0, 8.0, 13.0, 4) == pytest.approx(1.0, abs=1e-6)
    assert f.amplitude_gain(1.0, 500.0, 8.0, 13.0, 4) < 1e-4
    assert f.amplitude_gain(100.0, 500.0, 8.0, 13.0, 4) < 1e-4


# Spec: amplitude_gain validates like band_pass (bad band / fs / order raise FilteringError).
@pytest.mark.parametrize(
    "args",
    [(10.0, 0.0, 8.0, 13.0, 4), (10.0, 250.0, 13.0, 8.0, 4), (10.0, 250.0, 8.0, 200.0, 4), (10.0, 250.0, 8.0, 13.0, 0)],
)
def test_amplitude_gain_invalid(args):
    with pytest.raises(f.FilteringError):
        f.amplitude_gain(*args)


# ---------------------------------------------------------------------------
# filtering.causal_burst_delay
# ---------------------------------------------------------------------------

def _reference_delay(fs, low, high, order, centre):
    n = int(round(4 * fs))
    t = np.arange(n) / fs
    burst = np.exp(-0.5 * ((t - 2.0) / 0.1) ** 2) * np.sin(2 * np.pi * centre * t)
    out = sps.sosfilt(_sos(fs, low, high, order), burst)
    return (np.argmax(np.abs(sps.hilbert(out))) - np.argmax(np.abs(sps.hilbert(burst)))) / fs


# Spec: causal_burst_delay is positive (forward-only filter delays the envelope).
@pytest.mark.parametrize("fs,low,high,order", [(500.0, 8.0, 13.0, 4), (250.0, 8.0, 13.0, 2), (500.0, 15.0, 30.0, 6), (1000.0, 4.0, 8.0, 3)])
def test_causal_burst_delay_positive_and_matches_reference(fs, low, high, order):
    d = f.causal_burst_delay(fs, low, high, order)
    assert d > 0
    # Band centre is ambiguous (arithmetic vs geometric); accept either within 2 samples.
    refs = [_reference_delay(fs, low, high, order, c) for c in ((low + high) / 2, math.sqrt(low * high))]
    assert min(abs(d - r) for r in refs) <= 2.0 / fs


# Spec: causal_burst_delay -> envelope shift grows with filter order (Butterworth group delay).
def test_causal_burst_delay_increases_with_order():
    delays = [f.causal_burst_delay(500.0, 8.0, 13.0, k) for k in (1, 2, 4, 6)]
    assert all(b > a for a, b in zip(delays, delays[1:]))


# Spec: causal_burst_delay deterministic.
def test_causal_burst_delay_deterministic():
    assert f.causal_burst_delay(500.0, 8.0, 13.0, 4) == f.causal_burst_delay(500.0, 8.0, 13.0, 4)


# Spec: causal_burst_delay shares band/fs/order validation -> FilteringError.
@pytest.mark.parametrize("args", [(0.0, 8.0, 13.0, 4), (500.0, 13.0, 8.0, 4), (500.0, 8.0, 13.0, 0)])
def test_causal_burst_delay_invalid(args):
    with pytest.raises(f.FilteringError):
        f.causal_burst_delay(*args)


# ---------------------------------------------------------------------------
# source_lesson
# ---------------------------------------------------------------------------

L = np.array([[1.0, 0.6, 0.2], [0.2, 0.6, 1.0]])
NULL = np.array([1.0, -2.0, 1.0])


# Spec: LEAD_FIELD, NULL_DIRECTION constants; three SOURCES, two ELECTRODES.
def test_source_constants():
    assert s.LEAD_FIELD == ((1.0, 0.6, 0.2), (0.2, 0.6, 1.0))
    assert s.NULL_DIRECTION == (1.0, -2.0, 1.0)
    assert len(s.SOURCES) == 3 and len(s.ELECTRODES) == 2
    # hand check: L @ null = (1 - 1.2 + 0.2, 0.2 - 1.2 + 1) = (0, 0)
    np.testing.assert_allclose(L @ NULL, [0.0, 0.0], atol=1e-15)


# Spec: forward(strengths) -> tuple y = L x (hand-calculated values).
@pytest.mark.parametrize(
    "x,y",
    [((0, 1, 0), (0.6, 0.6)), ((1, 0, 0), (1.0, 0.2)), ((0, 0, 1), (0.2, 1.0)), ((1, 1, 1), (1.8, 1.8)), ((2, 0, -1), (1.8, -0.6))],
)
def test_forward_hand_values(x, y):
    out = s.forward(x)
    assert isinstance(out, tuple) and len(out) == 2
    np.testing.assert_allclose(out, y, atol=1e-12)


# Spec: forward is y = L x; null-direction invariance across many multiples (seeded).
def test_forward_null_space_invariance_many_multiples():
    rng = np.random.default_rng(11)
    for _ in range(200):
        x = rng.normal(size=3) * rng.choice([1e-3, 1, 1e3])
        k = rng.normal() * rng.choice([1e-3, 1, 1e3])
        base = np.asarray(s.forward(tuple(x)))
        moved = np.asarray(s.forward(tuple(x + k * NULL)))
        np.testing.assert_allclose(moved, base, atol=1e-9 * (1 + abs(k) + np.abs(x).max()))
        np.testing.assert_allclose(base, L @ x, atol=1e-12 * (1 + np.abs(x).max()))


# Spec: forward linearity (y = L x).
def test_forward_linearity():
    rng = np.random.default_rng(12)
    for _ in range(50):
        a, b = rng.normal(size=3), rng.normal(size=3)
        p, q = rng.normal(size=2)
        lhs = np.asarray(s.forward(tuple(p * a + q * b)))
        rhs = p * np.asarray(s.forward(tuple(a))) + q * np.asarray(s.forward(tuple(b)))
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)


# Spec: forward raises SourceLessonError unless exactly 3 finite values.
@pytest.mark.parametrize("bad", [(), (1.0,), (1.0, 2.0), (1, 2, 3, 4), (1, np.nan, 0), (np.inf, 0, 0), ((1, 2, 3),)])
def test_forward_invalid(bad):
    with pytest.raises(s.SourceLessonError):
        s.forward(bad)


# Spec: minimum_norm(readings) -> pinv(L) y as a tuple; hand value (0.6,0.6) -> (1/3,1/3,1/3).
def test_minimum_norm_hand_value():
    out = s.minimum_norm((0.6, 0.6))
    assert isinstance(out, tuple) and len(out) == 3
    np.testing.assert_allclose(out, (1 / 3, 1 / 3, 1 / 3), atol=1e-12)


# Spec: minimum_norm = pinv(L) y; reproduces readings and is orthogonal to the null direction.
def test_minimum_norm_properties_seeded():
    rng = np.random.default_rng(13)
    P = np.linalg.pinv(L)
    for _ in range(100):
        y = rng.normal(size=2) * 5
        x = np.asarray(s.minimum_norm(tuple(y)))
        np.testing.assert_allclose(x, P @ y, atol=1e-12)
        np.testing.assert_allclose(L @ x, y, atol=1e-10)
        assert abs(x @ NULL) < 1e-10
        # any other solution x + k*null has a larger norm
        for k in (-1.0, 0.1, 2.0):
            assert np.linalg.norm(x + k * NULL) > np.linalg.norm(x)


# Spec: minimum_norm raises SourceLessonError unless exactly 2 finite values.
@pytest.mark.parametrize("bad", [(), (1.0,), (1, 2, 3), (np.nan, 1.0), (1.0, -np.inf)])
def test_minimum_norm_invalid(bad):
    with pytest.raises(s.SourceLessonError):
        s.minimum_norm(bad)


# Spec: SourceLessonError is a typed exception (ValueError family expected per project convention).
def test_source_lesson_error_type():
    assert issubclass(s.SourceLessonError, Exception)


# Spec: ambiguity() -> frozen dataclass with lead_field, three configurations, readings, minimum_norm, null_direction.
def test_ambiguity_structure_and_values():
    a = s.ambiguity()
    _assert_frozen(a)
    assert a.lead_field == s.LEAD_FIELD
    assert tuple(a.null_direction) == s.NULL_DIRECTION
    np.testing.assert_allclose(a.readings, (0.6, 0.6), atol=1e-12)
    assert len(a.configurations) == 3
    expected = [(0, 1, 0), (0.25, 0.5, 0.25), (0.5, 0, 0.5)]
    labels = []
    for cfg, strengths in zip(a.configurations, expected):
        np.testing.assert_allclose(cfg.strengths, strengths, atol=1e-12)
        np.testing.assert_allclose(cfg.readings, s.forward(cfg.strengths), atol=1e-12)
        np.testing.assert_allclose(cfg.readings, (0.6, 0.6), atol=1e-12)
        assert isinstance(cfg.label, str) and cfg.label
        labels.append(cfg.label)
    assert len(set(labels)) == 3
    # configurations differ by multiples of the null direction (0.25 steps)
    for cfg, k in zip(a.configurations, (0.0, 0.25, 0.5)):
        np.testing.assert_allclose(np.asarray(cfg.strengths) - np.array([0, 1, 0]), k * NULL, atol=1e-12)


# Spec: ambiguity().minimum_norm strengths ~ (1/3,1/3,1/3) with readings = forward of them.
def test_ambiguity_minimum_norm():
    mn = s.ambiguity().minimum_norm
    np.testing.assert_allclose(mn.strengths, (1 / 3, 1 / 3, 1 / 3), atol=1e-9)
    np.testing.assert_allclose(mn.readings, s.forward(mn.strengths), atol=1e-12)
    np.testing.assert_allclose(mn.readings, (0.6, 0.6), atol=1e-9)
    # minimum-norm solution has smaller norm than every listed configuration
    for cfg in s.ambiguity().configurations:
        assert np.linalg.norm(mn.strengths) < np.linalg.norm(cfg.strengths)


# Spec: ambiguity() deterministic.
def test_ambiguity_deterministic():
    assert s.ambiguity() == s.ambiguity()


# ---------------------------------------------------------------------------
# power_explainer.welch_steps
# ---------------------------------------------------------------------------

def _check_welch(window, fs):
    w = np.asarray(window, dtype=float)
    n = len(w)
    r = pe.welch_steps(window, fs)
    np.testing.assert_allclose(r.times, np.arange(n) / fs, atol=1e-12)
    np.testing.assert_allclose(r.centered, w - w.mean(), atol=1e-9)
    taper = np.hanning(n + 1)[:-1]
    np.testing.assert_allclose(r.taper, taper, atol=1e-12)
    np.testing.assert_allclose(r.tapered, (w - w.mean()) * taper, atol=1e-9)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    np.testing.assert_allclose(r.freqs_hz, freqs, atol=1e-9)
    X = np.fft.rfft((w - w.mean()) * taper)
    np.testing.assert_allclose(r.fft_magnitude, np.abs(X), rtol=1e-9, atol=1e-9)
    dens = np.abs(X) ** 2 / (fs * np.sum(taper ** 2))
    if n % 2 == 0:
        dens[1:-1] *= 2
    else:
        dens[1:] *= 2
    np.testing.assert_allclose(r.density, dens, rtol=1e-9, atol=1e-12)
    _, scipy_d = sps.welch(w, fs, nperseg=n)
    np.testing.assert_allclose(r.density, scipy_d, rtol=1e-8, atol=1e-12)
    band = (freqs >= 8) & (freqs <= 13)
    np.testing.assert_array_equal(np.asarray(r.in_band, dtype=bool), band)
    area = trapezoid(dens[band], freqs[band]) if band.sum() >= 2 else 0.0
    assert r.alpha_power == pytest.approx(area, rel=1e-9, abs=1e-12)
    assert r.alpha_power == pytest.approx(iv.compute_alpha_power(w, fs), rel=1e-9, abs=1e-12)
    return r


# Spec: welch_steps fields, identities with scipy welch and iv.compute_alpha_power, on recorded windows.
@pytest.mark.parametrize("cond", ["eyes_closed", "eyes_open"])
@pytest.mark.parametrize("start,length", [(0, 500), (1000, 1000), (2501, 999), (5000, 2500), (7000, 500)])
def test_welch_steps_on_recorded(recorded, cond, start, length):
    r = _check_welch(recorded[cond][start:start + length], FS_REC)
    _assert_frozen(r)


# Spec: welch_steps identities on seeded random inputs, odd/even n, several sampling rates.
@pytest.mark.parametrize("seed", range(10))
def test_welch_steps_seeded(seed):
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(64, 1500))
    fs = float(rng.choice([100.0, 128.0, 250.0, 500.0, 1000.0]))
    x = rng.normal(size=n) * 10 + rng.normal() * 100
    _check_welch(x, fs)


# Spec: density / alpha_power hand check - a pure 10 Hz sine of amplitude A carries
# power ~ A^2/2 concentrated in the alpha band (Parseval for the density estimate).
def test_welch_steps_pure_alpha_sine_power():
    fs, n, A = 500.0, 1000, 4.0
    t = np.arange(n) / fs
    r = pe.welch_steps(A * np.sin(2 * np.pi * 10 * t), fs)
    total = trapezoid(np.asarray(r.density), np.asarray(r.freqs_hz))
    assert total == pytest.approx(A ** 2 / 2, rel=0.02)
    assert r.alpha_power == pytest.approx(A ** 2 / 2, rel=0.02)


# Spec: centered = window - mean -> DC offset does not change density or alpha_power.
def test_welch_steps_offset_invariance(recorded):
    w = recorded["eyes_closed"][:1000]
    a = pe.welch_steps(w, FS_REC)
    b = pe.welch_steps(w + 1234.5, FS_REC)
    np.testing.assert_allclose(a.density, b.density, rtol=1e-6, atol=1e-9)
    assert a.alpha_power == pytest.approx(b.alpha_power, rel=1e-6)


# Spec: density ~ |X|^2 -> scaling the window by c scales alpha_power by c^2.
def test_welch_steps_quadratic_scaling(recorded):
    w = recorded["eyes_open"][:1000]
    assert pe.welch_steps(3 * w, FS_REC).alpha_power == pytest.approx(9 * pe.welch_steps(w, FS_REC).alpha_power, rel=1e-9)


# Spec: default sampling_rate=500.0.
def test_welch_steps_default_rate(recorded):
    w = recorded["eyes_closed"][:500]
    assert pe.welch_steps(w) == pe.welch_steps(w, 500.0)


# Spec: welch_steps with 2 samples (minimum allowed) still satisfies scipy identity.
def test_welch_steps_two_samples():
    r = pe.welch_steps([1.0, 3.0], 500.0)
    _, d = sps.welch(np.array([1.0, 3.0]), 500.0, nperseg=2)
    np.testing.assert_allclose(r.density, d, atol=1e-12)
    assert r.alpha_power == 0.0


# Spec: raises PowerExplainerError for non-1-D, < 2 samples, NaN/inf, bad sampling rate.
@pytest.mark.parametrize(
    "window,fs",
    [
        (np.zeros((4, 100)), 500.0),
        ([], 500.0),
        ([1.0], 500.0),
        ([1.0, np.nan, 2.0], 500.0),
        ([1.0, np.inf, 2.0], 500.0),
        (np.ones(100), 0.0),
        (np.ones(100), -500.0),
        (np.ones(100), float("nan")),
        (np.ones(100), float("inf")),
    ],
)
def test_welch_steps_invalid(window, fs):
    with pytest.raises(pe.PowerExplainerError):
        pe.welch_steps(window, fs)


def _step_keys(steps):
    if isinstance(steps, dict):
        return list(steps.keys())
    keys = []
    for st in steps:
        if isinstance(st, dict):
            keys.append(st["key"])
        elif isinstance(st, (tuple, list)) and not hasattr(st, "key"):
            keys.append(st[0])
        else:
            keys.append(st.key)
    return keys


# Spec: WELCH_STEPS has five steps with keys record, taper, fft, power, area.
def test_welch_steps_constant():
    assert _step_keys(pe.WELCH_STEPS) == ["record", "taper", "fft", "power", "area"]


# ---------------------------------------------------------------------------
# hemodynamics.response_components
# ---------------------------------------------------------------------------

def _resp(events, duration=40.0, rate_hz=10.0):
    return np.asarray(h.hemodynamic_response(events, duration, rate_hz).response, dtype=float)


# Spec: components sum elementwise to hemodynamic_response(...).response (seeded sweep).
@pytest.mark.parametrize("seed", range(10))
def test_components_sum_to_response(seed):
    rng = np.random.default_rng(200 + seed)
    duration = float(rng.choice([20.0, 40.0, 60.0]))
    rate = float(rng.choice([2.0, 10.0, 20.0]))
    events = tuple(float(e) for e in rng.uniform(0, duration * 0.8, size=int(rng.integers(1, 6))))
    comps = h.response_components(events, duration, rate)
    assert isinstance(comps, tuple) and len(comps) == len(events)
    total = np.sum([np.asarray(c, dtype=float) for c in comps], axis=0)
    np.testing.assert_allclose(total, _resp(events, duration, rate), atol=1e-10)


# Spec: each component is the response to that event alone, in sorted event order.
def test_components_are_single_event_responses_sorted():
    events = (20.0, 3.0, 11.5)
    comps = h.response_components(events)
    for comp, e in zip(comps, sorted(events)):
        np.testing.assert_allclose(np.asarray(comp, dtype=float), _resp((e,)), atol=1e-12)


# Spec: defaults duration=40, rate_hz=10 -> same as explicit call; components length = response length.
def test_components_defaults():
    comps = h.response_components((5.0,))
    np.testing.assert_allclose(comps[0], h.response_components((5.0,), 40.0, 10.0)[0])
    assert len(comps[0]) == len(_resp((5.0,)))


# Spec: STEP_EVENTS == (5.0, 8.0) and their response is not separable (responses overlap in time).
def test_step_events_overlap():
    assert tuple(h.STEP_EVENTS) == (5.0, 8.0)
    c1, c2 = (np.asarray(c, dtype=float) for c in h.response_components(h.STEP_EVENTS))
    overlap = np.abs(c1) * np.abs(c2)
    peak = max(np.abs(c1).max(), np.abs(c2).max())
    # substantial overlap: at some sample both components exceed 10% of the peak
    assert np.any((np.abs(c1) > 0.1 * peak) & (np.abs(c2) > 0.1 * peak))
    assert overlap.max() > 0
    np.testing.assert_allclose(c1 + c2, _resp(h.STEP_EVENTS), atol=1e-12)


# Spec: response_components deterministic and order-independent of input ordering.
def test_components_order_independent():
    a = h.response_components((8.0, 5.0))
    b = h.response_components((5.0, 8.0))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
