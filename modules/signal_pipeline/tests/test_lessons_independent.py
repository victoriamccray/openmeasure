"""Independent, spec-derived adversarial tests for hemodynamics and correlation_lesson.

Written from the written spec only, without reading the implementation or its
sibling tests. Reference values are either hand-calculated or recomputed here
with numpy/scipy directly from the spec's formulas.
"""

import dataclasses
import math
import os
import sys

import numpy as np
import pytest
from scipy.stats import gamma

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import hemodynamics as h, correlation_lesson as c  # noqa: E402


# ---------------------------------------------------------------------------
# Reference implementations built from the spec text
# ---------------------------------------------------------------------------

def ref_kernel(rate_hz=10.0, seconds=32.0):
    n = int(math.ceil(seconds * rate_hz))
    t = np.arange(n) / rate_hz
    t = t[t < seconds]
    v = gamma.pdf(t, 6, scale=1) - gamma.pdf(t, 16, scale=1) / 6.0
    return t, v / v.max()


def ref_hemo(events, duration=40.0, rate_hz=10.0):
    n = int(round(duration * rate_hz))
    times = np.arange(n) / rate_hz
    neural = np.zeros(n)
    for e in events:
        neural[int(round(e * rate_hz))] += 1.0
    _, k = ref_kernel(rate_hz)
    resp = np.convolve(neural, k)[:n]
    return times, neural, resp


def ref_peak_indices(v):
    v = np.asarray(v)
    return [i for i in range(1, len(v) - 1)
            if v[i] > v[i - 1] and v[i] >= v[i + 1] and v[i] > 0]


def ref_separable(resp, n_events):
    if n_events == 1:
        return True
    idx = ref_peak_indices(resp)
    if len(idx) < n_events:
        return False
    resp = np.asarray(resp)
    for a, b in zip(idx[:-1], idx[1:]):
        if resp[a:b + 1].min() > 0.75 * min(resp[a], resp[b]):
            return False
    return True


def assert_frozen(obj, field):
    assert dataclasses.is_dataclass(obj)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, field, None)


# ===========================================================================
# hemodynamics.response_function
# ===========================================================================

def test_response_function_matches_double_gamma_formula_default():
    # Spec: values = gammapdf(t,6,1) - (1/6) gammapdf(t,16,1), t = 0,1/rate,...<seconds, / max.
    r = h.response_function()
    t, v = ref_kernel(10.0, 32.0)
    assert len(r.times) == 320 == len(r.values)
    np.testing.assert_allclose(r.times, t, atol=1e-12)
    np.testing.assert_allclose(r.values, v, rtol=1e-9, atol=1e-12)
    assert isinstance(r.times, tuple) and isinstance(r.values, tuple)


def test_response_function_hand_values():
    # Spec: sampled from t=0 (gamma(6) pdf at 0 is 0) and normalised so max == 1.
    r = h.response_function()
    assert r.values[0] == pytest.approx(0.0, abs=1e-15)
    assert max(r.values) == pytest.approx(1.0, abs=1e-12)
    # Hand: at t=5, unnormalised = 5^5 e^-5/120 - (1/6) 5^15 e^-5/15!
    raw5 = 5**5 * math.exp(-5) / 120 - (5**15 * math.exp(-5) / math.factorial(15)) / 6
    raw_max = max(gamma.pdf(np.arange(320) / 10, 6) - gamma.pdf(np.arange(320) / 10, 16) / 6)
    assert r.values[50] == pytest.approx(raw5 / raw_max, rel=1e-9)


@pytest.mark.parametrize("rate,seconds", [(10.0, 32.0), (1.0, 32.0), (4.0, 25.0), (20.0, 30.0), (3.0, 7.5)])
def test_response_function_peak_and_undershoot_times(rate, seconds):
    # Spec: peak_time = time of the max; undershoot_time = time of the min.
    r = h.response_function(rate_hz=rate, seconds=seconds)
    v = np.array(r.values)
    assert r.peak_time == pytest.approx(r.times[int(np.argmax(v))])
    assert r.undershoot_time == pytest.approx(r.times[int(np.argmin(v))])
    t, rv = ref_kernel(rate, seconds)
    assert len(r.values) == len(rv)
    np.testing.assert_allclose(r.values, rv, rtol=1e-9, atol=1e-12)


def test_response_function_peak_near_five_seconds_undershoot_later():
    # Spec: double gamma (shape 6 mode = 5 s); undershoot from shape-16 term (mode 15 s).
    # Fine-grid analytic optimum, then the sampled peak must lie within one sample of it.
    r = h.response_function(rate_hz=10.0)
    tf = np.linspace(0, 32, 320001)
    vf = gamma.pdf(tf, 6) - gamma.pdf(tf, 16) / 6
    assert abs(r.peak_time - tf[np.argmax(vf)]) <= 0.1 + 1e-9
    assert abs(r.undershoot_time - tf[np.argmin(vf)]) <= 0.1 + 1e-9
    assert 4.0 < r.peak_time < 6.0
    assert 12.0 < r.undershoot_time < 18.0
    assert min(r.values) < 0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_response_function_invalid_rate_and_seconds(bad):
    # Spec: raises HemodynamicsError for rate_hz or seconds not positive finite.
    assert issubclass(h.HemodynamicsError, ValueError)
    with pytest.raises(h.HemodynamicsError):
        h.response_function(rate_hz=bad)
    with pytest.raises(h.HemodynamicsError):
        h.response_function(seconds=bad)


def test_response_function_frozen():
    # Spec: returns a frozen dataclass.
    assert_frozen(h.response_function(), "peak_time")


# ===========================================================================
# hemodynamics.hemodynamic_response
# ===========================================================================

def test_hemodynamic_single_event_matches_reference_convolution():
    # Spec: neural impulse at round(e*rate); response = full convolution truncated to len(times).
    res = h.hemodynamic_response([3.0])
    t, nn, rr = ref_hemo([3.0])
    assert len(res.times) == 400
    np.testing.assert_allclose(res.times, t, atol=1e-12)
    np.testing.assert_allclose(res.neural, nn)
    np.testing.assert_allclose(res.response, rr, rtol=1e-9, atol=1e-12)
    assert res.neural[30] == 1.0 and sum(res.neural) == 1.0


def test_hemodynamic_single_event_is_shifted_kernel():
    # Spec: convolution of a unit impulse at sample k with the kernel is the kernel delayed by k samples.
    k = h.response_function().values
    res = h.hemodynamic_response([2.0])
    np.testing.assert_allclose(res.response[:20], 0.0, atol=1e-15)
    # Kernel is 320 samples long, so it fills samples 20..339 and the rest is exactly zero.
    np.testing.assert_allclose(res.response[20:340], k, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(res.response[340:], 0.0, atol=1e-15)
    # Hand: first_peak_delay equals kernel peak time; single event is separable.
    assert res.first_peak_delay == pytest.approx(h.response_function().peak_time)
    assert res.separable is True


def test_hemodynamic_sorts_events_and_first_peak_uses_earliest():
    # Spec: event_times is a sorted tuple; first_peak_delay measured from the first event.
    res = h.hemodynamic_response([20.0, 4.0])
    assert res.event_times == (4.0, 20.0)
    assert isinstance(res.event_times, tuple)
    assert res.first_peak_delay == pytest.approx(h.response_function().peak_time)


def test_hemodynamic_off_grid_event_rounds_sample_but_delay_uses_true_time():
    # Spec: impulse at sample round(e*rate); first_peak_delay = first peak time - first event (raw).
    e = 1.03  # -> sample 10 (t=1.0)
    res = h.hemodynamic_response([e])
    assert res.neural[10] == 1.0
    assert res.first_peak_delay == pytest.approx(1.0 + h.response_function().peak_time - e)


def test_hemodynamic_coincident_events_add():
    # Spec: coincident events add, so two events at one sample double neural and response.
    one = h.hemodynamic_response([5.0])
    two = h.hemodynamic_response([5.0, 5.0])
    assert two.neural[50] == 2.0
    np.testing.assert_allclose(two.response, 2 * np.array(one.response), rtol=1e-12, atol=1e-15)
    # Events within half a sample of each other also share a sample.
    near = h.hemodynamic_response([5.0, 5.02])
    assert near.neural[50] == 2.0


def test_hemodynamic_superposition_sweep():
    # Spec: response is a linear convolution of the summed impulses -> superposition holds.
    rng = np.random.default_rng(11)
    for _ in range(40):
        k = int(rng.integers(2, 6))
        ev = list(np.round(rng.uniform(0, 39.9, size=k), 1))
        total = np.array(h.hemodynamic_response(ev).response)
        parts = sum(np.array(h.hemodynamic_response([e]).response) for e in ev)
        np.testing.assert_allclose(total, parts, rtol=1e-10, atol=1e-12)
        _, _, rr = ref_hemo(ev)
        np.testing.assert_allclose(total, rr, rtol=1e-9, atol=1e-12)


def test_hemodynamic_time_shift_invariance_sweep():
    # Spec: convolution is shift-invariant: delaying all events by d samples delays response by d.
    rng = np.random.default_rng(12)
    for _ in range(30):
        ev = list(np.round(rng.uniform(0, 15, size=int(rng.integers(1, 4))), 1))
        d = int(rng.integers(1, 150))
        a = np.array(h.hemodynamic_response(ev).response)
        b = np.array(h.hemodynamic_response([e + d / 10 for e in ev]).response)
        np.testing.assert_allclose(b[:d], 0.0, atol=1e-15)
        np.testing.assert_allclose(b[d:], a[:400 - d], rtol=1e-10, atol=1e-12)


def test_hemodynamic_peak_times_rule():
    # Spec: peak_times = times where v[i] > v[i-1], v[i] >= v[i+1], v[i] > 0.
    rng = np.random.default_rng(13)
    for _ in range(30):
        ev = list(np.round(rng.uniform(0, 35, size=int(rng.integers(1, 5))), 1))
        res = h.hemodynamic_response(ev)
        expect = [res.times[i] for i in ref_peak_indices(res.response)]
        np.testing.assert_allclose(res.peak_times, expect, atol=1e-12)


def test_hemodynamic_peak_times_plateau_takes_first_sample():
    # Spec: strict rise then >= : a flat-topped maximum is reported at its first sample only.
    # Two events 0.1 s apart at the kernel peak sample produce no plateau, so check the rule on the
    # reported list directly: no two consecutive samples may both be peaks.
    res = h.hemodynamic_response([0.0, 0.1, 0.2])
    idx = [int(round(p * 10)) for p in res.peak_times]
    assert all(b - a > 1 for a, b in zip(idx[:-1], idx[1:]))


def test_hemodynamic_separable_far_and_close():
    # Spec: several events separable iff >= as many peaks as events and each dip <= 0.75 * smaller peak.
    far = h.hemodynamic_response([2.0, 22.0])
    assert far.separable is True
    close = h.hemodynamic_response([2.0, 3.0])
    assert close.separable is False
    # Hand reason for 'close': kernels 1 s apart merge into one peak (< 2 peaks).
    assert len(close.peak_times) < 2


def test_hemodynamic_separable_matches_rule_across_gap_sweep():
    # Spec: separable rule, checked around the boundary over a fine sweep of gaps.
    seen = set()
    for gap10 in range(1, 300):
        gap = gap10 / 10
        res = h.hemodynamic_response([1.0, 1.0 + gap])
        exp = ref_separable(res.response, 2)
        assert res.separable is exp, gap
        seen.add(exp)
    assert seen == {True, False}  # the sweep actually crosses the boundary


def test_hemodynamic_separable_random_multi_event_sweep():
    # Spec: separable rule for 2-5 events at random positions.
    rng = np.random.default_rng(14)
    outcomes = set()
    for _ in range(80):
        k = int(rng.integers(2, 6))
        ev = list(np.round(rng.uniform(0, 30, size=k), 1))
        res = h.hemodynamic_response(ev)
        exp = ref_separable(res.response, k)
        assert res.separable is exp, ev
        outcomes.add(exp)
    assert outcomes == {True, False}


def test_hemodynamic_separable_coincident_events_not_separable():
    # Spec: needs at least as many peaks as events; two coincident events give one peak.
    res = h.hemodynamic_response([5.0, 5.0])
    assert len(res.peak_times) == 1
    assert res.separable is False


@pytest.mark.parametrize("rate", [1.0, 2.0, 5.0, 20.0])
def test_hemodynamic_other_rates(rate):
    # Spec: times = arange(round(duration*rate))/rate; kernel resampled at the same rate.
    res = h.hemodynamic_response([4.0, 25.0], duration=40.0, rate_hz=rate)
    t, nn, rr = ref_hemo([4.0, 25.0], 40.0, rate)
    np.testing.assert_allclose(res.times, t, atol=1e-12)
    np.testing.assert_allclose(res.response, rr, rtol=1e-9, atol=1e-12)


def test_hemodynamic_duration_shorter_than_kernel_truncates():
    # Spec: response truncated to len(times) even when duration < kernel length.
    res = h.hemodynamic_response([1.0], duration=8.0)
    assert len(res.response) == len(res.times) == 80
    np.testing.assert_allclose(res.response[10:], h.response_function().values[:70], rtol=1e-12)


@pytest.mark.parametrize("events", [[], [-0.1], [40.0], [41.0], [float("nan")], [float("inf")], [1.0, -1.0]])
def test_hemodynamic_invalid_events(events):
    # Spec: raises for empty events, any event < 0, >= duration, or non-finite.
    with pytest.raises(h.HemodynamicsError):
        h.hemodynamic_response(events, duration=40.0)


@pytest.mark.parametrize("kw", [{"duration": 0.0}, {"duration": -5.0}, {"duration": float("nan")},
                                {"duration": float("inf")}, {"rate_hz": 0.0}, {"rate_hz": -1.0},
                                {"rate_hz": float("nan")}, {"rate_hz": float("inf")}])
def test_hemodynamic_invalid_duration_rate(kw):
    # Spec: raises for non-positive/non-finite duration or rate.
    with pytest.raises(h.HemodynamicsError):
        h.hemodynamic_response([0.5], **kw)


def test_hemodynamic_event_just_below_duration_accepted_without_index_error():
    # Spec: events < duration are valid. round((40 - 1e-9) * 10) = 400 = len(times): must not crash
    # with a raw IndexError; the in-window samples receive no impulse.
    res = h.hemodynamic_response([39.99999])
    assert len(res.neural) == 400
    assert sum(res.neural) in (0.0, 1.0)
    assert res.event_times == (39.99999,)


def test_hemodynamic_zero_event_accepted():
    # Spec: event = 0 is allowed (only < 0 rejected); impulse lands at sample 0.
    res = h.hemodynamic_response([0.0])
    assert res.neural[0] == 1.0
    np.testing.assert_allclose(res.response[:320], h.response_function().values, rtol=1e-12)


def test_hemodynamic_deterministic_and_frozen():
    # Spec: frozen dataclass; pure function of its inputs.
    a = h.hemodynamic_response([3.0, 17.0])
    b = h.hemodynamic_response([3.0, 17.0])
    assert a == b
    assert_frozen(a, "separable")
    assert isinstance(a.times, tuple) and isinstance(a.response, tuple)


# ===========================================================================
# correlation_lesson.pearson_r
# ===========================================================================

def test_wiring_constants():
    # Spec: constants and WIRINGS holds the three.
    assert c.WIRING_DIRECT == "direct"
    assert c.WIRING_COMMON == "common input"
    assert c.WIRING_SHARED == "shared source"
    assert set(c.WIRINGS) == {"direct", "common input", "shared source"} and len(c.WIRINGS) == 3


def test_pearson_hand_values():
    # Spec: r = mean((x-xbar)(y-ybar)) / (sd_x sd_y), population sd.
    assert c.pearson_r([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert c.pearson_r([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    # x=[1,2,3,4], y=[1,3,2,4]: cov_pop = (1.5*1.5 + 0.5*(-0.5) + (-0.5)*0.5 ... ) -> r = 0.8
    assert c.pearson_r([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    # orthogonal: x=[-1,0,1], y=[1,-2,1] -> r = 0
    assert c.pearson_r([-1, 0, 1], [1, -2, 1]) == pytest.approx(0.0, abs=1e-15)


def test_pearson_matches_numpy_and_symmetries_sweep():
    # Spec: pearson_r definition; swap x/y, positive affine x unchanged, negating one flips sign.
    rng = np.random.default_rng(21)
    for _ in range(100):
        n = int(rng.integers(3, 200))
        x = rng.normal(size=n)
        y = 0.5 * x + rng.normal(size=n)
        r = c.pearson_r(x, y)
        assert r == pytest.approx(np.corrcoef(x, y)[0, 1], abs=1e-12)
        assert c.pearson_r(y, x) == pytest.approx(r, abs=1e-12)
        k, s = rng.uniform(0.01, 100), rng.uniform(-1e3, 1e3)
        assert c.pearson_r(k * x + s, y) == pytest.approx(r, abs=1e-9)
        assert c.pearson_r(-x, y) == pytest.approx(-r, abs=1e-12)
        assert c.pearson_r(x, -y) == pytest.approx(-r, abs=1e-12)
        assert -1.0 - 1e-12 <= r <= 1.0 + 1e-12


@pytest.mark.parametrize("x,y", [
    ([1, 2], [1, 2]),                        # shorter than 3
    ([1, 2, 3], [1, 2, 3, 4]),               # length mismatch
    ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),    # not 1-D
    ([1, 2, float("nan")], [1, 2, 3]),       # NaN
    ([1, 2, 3], [1, float("inf"), 3]),       # inf
    ([2, 2, 2], [1, 2, 3]),                  # constant x
    ([1, 2, 3], [5, 5, 5]),                  # constant y
    ([], []),
])
def test_pearson_invalid(x, y):
    # Spec: raises CorrelationLessonError for not 1-D, length mismatch, < 3, NaN/inf, constant.
    assert issubclass(c.CorrelationLessonError, ValueError)
    with pytest.raises(c.CorrelationLessonError):
        c.pearson_r(x, y)


# ===========================================================================
# correlation_lesson.expected_r
# ===========================================================================

def test_expected_r_hand_values():
    # Spec: direct b/sqrt(b^2+s^2); common a^2/(a^2+s^2); shared w1 w2/sqrt((w1^2+s^2)(w2^2+s^2)).
    assert c.expected_r("direct", 1.0, 1.0) == pytest.approx(1 / math.sqrt(2))
    assert c.expected_r("direct", 3.0, 4.0) == pytest.approx(0.6)
    assert c.expected_r("common input", 1.0, 1.0) == pytest.approx(0.5)
    assert c.expected_r("common input", 2.0, 1.0) == pytest.approx(0.8)
    assert c.expected_r("shared source", 2.0, 1.0, second_weight=1.0) == pytest.approx(2 / math.sqrt(10))
    # second_weight omitted -> w2 = strength -> equals common input value.
    assert c.expected_r("shared source", 2.0, 1.0) == pytest.approx(0.8)
    for w in c.WIRINGS:
        assert c.expected_r(w, 0.0, 1.0) == pytest.approx(0.0, abs=1e-15)


def test_expected_r_sweep_against_formula():
    # Spec: closed forms, over a grid of strengths and noise SDs.
    for b in [0.0, 0.1, 0.5, 1, 2, 10]:
        for s in [0.1, 0.5, 1, 3]:
            assert c.expected_r(c.WIRING_DIRECT, b, s) == pytest.approx(b / math.sqrt(b * b + s * s))
            assert c.expected_r(c.WIRING_COMMON, b, s) == pytest.approx(b * b / (b * b + s * s))
            for w2 in [0.3, 1.0, 4.0]:
                assert c.expected_r(c.WIRING_SHARED, b, s, second_weight=w2) == pytest.approx(
                    b * w2 / math.sqrt((b * b + s * s) * (w2 * w2 + s * s)))


@pytest.mark.parametrize("args", [
    ("bogus", 1.0, 1.0), ("Direct", 1.0, 1.0),
    ("direct", -0.1, 1.0), ("direct", float("nan"), 1.0), ("direct", float("inf"), 1.0),
    ("direct", 1.0, 0.0), ("direct", 1.0, -1.0), ("direct", 1.0, float("nan")), ("direct", 1.0, float("inf")),
    ("common input", -1.0, 1.0), ("shared source", 1.0, 0.0),
])
def test_expected_r_invalid(args):
    # Spec: raises for unknown wiring, strength < 0 or non-finite, noise_sd <= 0 or non-finite.
    with pytest.raises(c.CorrelationLessonError):
        c.expected_r(*args)


# ===========================================================================
# correlation_lesson.simulate
# ===========================================================================

def test_simulate_fields_and_flags():
    # Spec: fields wiring, strength, noise_sd, x, y (len n tuples), r = pearson_r(x,y), expected_r, direct_link.
    for w in c.WIRINGS:
        res = c.simulate(w, 0.8, noise_sd=1.3, n=123, seed=5)
        assert res.wiring == w and res.strength == 0.8 and res.noise_sd == 1.3
        assert isinstance(res.x, tuple) and isinstance(res.y, tuple)
        assert len(res.x) == len(res.y) == 123
        assert res.r == pytest.approx(c.pearson_r(res.x, res.y), abs=1e-12)
        assert res.expected_r == pytest.approx(c.expected_r(w, 0.8, 1.3))
        assert res.direct_link is (w == c.WIRING_DIRECT)


def test_simulate_shared_second_weight_passes_through():
    # Spec: shared uses w2 = second_weight; expected_r reflects it.
    res = c.simulate(c.WIRING_SHARED, 2.0, noise_sd=1.0, second_weight=0.5)
    assert res.expected_r == pytest.approx(1.0 / math.sqrt(5 * 1.25))


def test_simulate_deterministic_and_seed_sensitive():
    # Spec: deterministic for a given seed.
    for w in c.WIRINGS:
        a = c.simulate(w, 1.0, seed=42)
        b = c.simulate(w, 1.0, seed=42)
        assert a.x == b.x and a.y == b.y and a.r == b.r
        d = c.simulate(w, 1.0, seed=43)
        assert d.x != a.x


def test_simulate_default_seed_and_n():
    # Spec: defaults noise_sd=1.0, n=400, seed=1997.
    a = c.simulate(c.WIRING_DIRECT, 1.0)
    b = c.simulate(c.WIRING_DIRECT, 1.0, noise_sd=1.0, n=400, seed=1997)
    assert len(a.x) == 400 and a.x == b.x and a.y == b.y


def test_simulate_direct_residual_structure():
    # Spec: direct y = b*x + noise_sd*N(0,1) with x a standard normal independent of noise.
    rng_seeds = range(30)
    for seed in rng_seeds:
        b, s, n = 1.7, 0.6, 2000
        res = c.simulate(c.WIRING_DIRECT, b, noise_sd=s, n=n, seed=seed)
        x, y = np.array(res.x), np.array(res.y)
        resid = y - b * x
        # Var(x)=1, var(resid)=s^2; SE of sample variance ~ sigma^2 sqrt(2/n); allow 5 SE.
        assert abs(x.var() - 1.0) < 5 * math.sqrt(2 / n)
        assert abs(resid.var() - s * s) < 5 * s * s * math.sqrt(2 / n)
        # corr(resid, x) ~ N(0, 1/sqrt(n)); allow 5 SE.
        assert abs(np.corrcoef(resid, x)[0, 1]) < 5 / math.sqrt(n)


@pytest.mark.parametrize("wiring", [c.WIRING_COMMON, c.WIRING_SHARED])
def test_simulate_moments_common_and_shared(wiring):
    # Spec: common x=aZ+n1, y=aZ+n2; shared x=w1 S+n1, y=w2 S+n2; independent standard normals.
    a, w2, s, n = 1.5, 0.7, 0.8, 4000
    kw = {"second_weight": w2} if wiring == c.WIRING_SHARED else {}
    wy = w2 if wiring == c.WIRING_SHARED else a
    vx, vy, cov = a * a + s * s, wy * wy + s * s, a * wy
    for seed in range(20):
        res = c.simulate(wiring, a, noise_sd=s, n=n, seed=seed, **kw)
        x, y = np.array(res.x), np.array(res.y)
        assert abs(x.var() - vx) < 5 * vx * math.sqrt(2 / n)
        assert abs(y.var() - vy) < 5 * vy * math.sqrt(2 / n)
        # SE of sample covariance = sqrt((vx vy + cov^2)/n)
        assert abs(np.mean((x - x.mean()) * (y - y.mean())) - cov) < 5 * math.sqrt((vx * vy + cov * cov) / n)
        assert abs(x.mean()) < 5 * math.sqrt(vx / n)


def test_simulate_r_tracks_expected_across_seeds_and_strengths():
    # Spec: r is a sample estimate of expected_r. Sampling SD of r ~ (1 - rho^2)/sqrt(n).
    # Per draw allow 5 SD (two-sided tail ~6e-7, x ~600 draws -> ~4e-4 false-fail chance; seeds fixed).
    # Mean over K seeds has SD (1-rho^2)/sqrt(nK); allow 5 of those plus a small-sample bias
    # allowance |rho|(1-rho^2)/(2n) from the known downward bias of r.
    n, K = 400, 50
    for w in c.WIRINGS:
        for strength in [0.0, 0.3, 1.0, 2.5]:
            for s in [0.5, 1.0, 2.0]:
                rho = c.expected_r(w, strength, s)
                sd = (1 - rho * rho) / math.sqrt(n)
                rs = []
                for seed in range(K):
                    r = c.simulate(w, strength, noise_sd=s, n=n, seed=1000 + seed).r
                    assert abs(r - rho) < 5 * sd + 1e-12, (w, strength, s, seed, r, rho)
                    rs.append(r)
                bias = abs(rho) * (1 - rho * rho) / (2 * n)
                assert abs(np.mean(rs) - rho) < 5 * sd / math.sqrt(K) + bias + 1e-12, (w, strength, s)


def test_simulate_shared_asymmetric_weights_r_tracks_expected():
    # Spec: shared with different w1, w2 follows w1 w2 / sqrt((w1^2+s^2)(w2^2+s^2)).
    n = 400
    for w1, w2 in [(3.0, 0.4), (0.4, 3.0), (1.0, 2.0)]:
        rho = c.expected_r(c.WIRING_SHARED, w1, 1.0, second_weight=w2)
        rs = [c.simulate(c.WIRING_SHARED, w1, n=n, seed=k, second_weight=w2).r for k in range(60)]
        assert abs(np.mean(rs) - rho) < 5 * (1 - rho**2) / math.sqrt(n * 60) + rho * (1 - rho**2) / (2 * n)


@pytest.mark.parametrize("n", [2, 0, -5, 3.5, 10.0, "10"])
def test_simulate_invalid_n(n):
    # Spec: raises for n < 3 or non-int.
    with pytest.raises(c.CorrelationLessonError):
        c.simulate(c.WIRING_DIRECT, 1.0, n=n)


def test_simulate_n_three_ok():
    # Spec: n >= 3 allowed; boundary n = 3 gives length-3 samples.
    res = c.simulate(c.WIRING_DIRECT, 1.0, n=3)
    assert len(res.x) == 3 and res.r == pytest.approx(c.pearson_r(res.x, res.y))


@pytest.mark.parametrize("args", [("nope", 1.0), ("direct", -1.0), ("direct", 1.0, 0.0),
                                  ("common input", float("nan")), ("shared source", 1.0, float("inf"))])
def test_simulate_propagates_expected_r_errors(args):
    # Spec: simulate raises anything expected_r raises.
    with pytest.raises(c.CorrelationLessonError):
        c.simulate(*args)


def test_simulate_frozen():
    # Spec: frozen dataclass.
    assert_frozen(c.simulate(c.WIRING_COMMON, 1.0), "r")


# ===========================================================================
# correlation_lesson.matching_direct_strength
# ===========================================================================

def test_matching_direct_strength_hand_value():
    # Spec: b = s r / sqrt(1 - r^2), r = a^2/(a^2+s^2). a=1,s=1 -> r=0.5 -> b = 1/sqrt(3).
    assert c.matching_direct_strength(1.0) == pytest.approx(1 / math.sqrt(3))
    # a=2, s=1 -> r=0.8 -> b = 0.8/0.6 = 4/3
    assert c.matching_direct_strength(2.0, noise_sd=1.0) == pytest.approx(4 / 3)
    # scaling: s=2, a=2 -> r=0.5 -> b = 2/sqrt(3)
    assert c.matching_direct_strength(2.0, noise_sd=2.0) == pytest.approx(2 / math.sqrt(3))
    assert c.matching_direct_strength(0.0) == pytest.approx(0.0, abs=1e-15)


def test_matching_direct_strength_equalises_expected_r_sweep():
    # Spec: expected_r(direct, b, s) == expected_r(common, a, s).
    for a in [0.05, 0.3, 1.0, 2.0, 5.0]:
        for s in [0.2, 1.0, 3.0]:
            b = c.matching_direct_strength(a, noise_sd=s)
            assert b >= 0
            assert c.expected_r(c.WIRING_DIRECT, b, s) == pytest.approx(c.expected_r(c.WIRING_COMMON, a, s), rel=1e-9)
