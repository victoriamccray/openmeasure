"""
Band-pass filtering an EEG signal to the alpha band, for a "compare two
worlds" view: the recorded signal next to the same signal filtered.

A Butterworth band-pass, applied as second-order sections. zero_phase=True
runs it forward and backward (no time shift, but it uses samples from the
future, so it only works offline). zero_phase=False runs it forward only,
as a real-time device would, which delays the output.

Filters change signals in ways that can mislead: they smooth, ring at
sharp changes and at the edges of a recording, and shift timing when
causal (Widmann et al., 2015; de Cheveigné and Nelken, 2019). The lesson
shows those costs beside the gain.

Pure functions returning frozen dataclasses. No streamlit, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, sosfreqz

DEFAULT_BAND_HZ = (8.0, 13.0)
DEFAULT_ORDER = 4


class FilteringError(ValueError):
    """Inputs the filter cannot use."""


@dataclass(frozen=True)
class FilterResult:
    raw: tuple[float, ...]
    filtered: tuple[float, ...]
    low_hz: float
    high_hz: float
    order: int
    zero_phase: bool
    sampling_rate: float
    kept_variance_share: float


def _design(sampling_rate: float, low_hz: float, high_hz: float, order: int):
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise FilteringError(f"sampling_rate must be a positive finite number; got {sampling_rate!r}.")
    if not (0 < low_hz < high_hz < sampling_rate / 2):
        raise FilteringError(
            f"The band must satisfy 0 < low < high < Nyquist ({sampling_rate / 2} Hz); got {low_hz} to {high_hz} Hz."
        )
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise FilteringError("order must be a whole number of at least 1.")
    return butter(order, [low_hz, high_hz], btype="bandpass", fs=sampling_rate, output="sos")


def band_pass(
    signal,
    sampling_rate: float,
    low_hz: float = DEFAULT_BAND_HZ[0],
    high_hz: float = DEFAULT_BAND_HZ[1],
    order: int = DEFAULT_ORDER,
    zero_phase: bool = True,
) -> FilterResult:
    """Filter a 1-D signal to [low_hz, high_hz] and report what it kept."""
    sos = _design(sampling_rate, low_hz, high_hz, order)
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise FilteringError("signal must be one-dimensional.")
    if not np.all(np.isfinite(x)):
        raise FilteringError("signal cannot hold NaN or infinite samples.")
    padlen = 3 * (2 * len(sos) + 1)
    if x.size <= padlen:
        raise FilteringError(f"signal needs more than {padlen} samples for this filter; got {x.size}.")
    y = sosfiltfilt(sos, x) if zero_phase else sosfilt(sos, x)
    centered = x - x.mean()
    total = float(np.mean(centered**2))
    # A constant signal leaves rounding residue of order eps * mean^2 in
    # the variance (found by the independent tests); treat it as zero.
    floor = (np.finfo(float).eps * max(float(np.mean(x**2)), 1.0)) ** 2 * 1e4
    kept = float(np.mean(y**2)) / total if total > floor else 0.0
    return FilterResult(
        raw=tuple(float(v) for v in x),
        filtered=tuple(float(v) for v in y),
        low_hz=float(low_hz),
        high_hz=float(high_hz),
        order=order,
        zero_phase=zero_phase,
        sampling_rate=float(sampling_rate),
        kept_variance_share=kept,
    )


def amplitude_gain(
    freq_hz: float,
    sampling_rate: float,
    low_hz: float = DEFAULT_BAND_HZ[0],
    high_hz: float = DEFAULT_BAND_HZ[1],
    order: int = DEFAULT_ORDER,
    zero_phase: bool = True,
) -> float:
    """
    How much a sine at freq_hz is scaled by the filter, once settled.

    A forward-backward pass applies the filter twice, so its gain is the
    single pass's gain squared.
    """
    sos = _design(sampling_rate, low_hz, high_hz, order)
    _, h = sosfreqz(sos, worN=[freq_hz], fs=sampling_rate)
    gain = float(abs(h[0]))
    return gain**2 if zero_phase else gain


def causal_burst_delay(
    sampling_rate: float,
    low_hz: float = DEFAULT_BAND_HZ[0],
    high_hz: float = DEFAULT_BAND_HZ[1],
    order: int = DEFAULT_ORDER,
) -> float:
    """
    How late a forward-only filter puts the peak of a brief burst.

    A Gaussian-windowed burst at the band's centre frequency, centred in a
    4-second signal; the delay is the shift of its envelope peak (from the
    analytic signal) after forward-only filtering. A zero-phase pass gives
    zero by construction.
    """
    from scipy.signal import hilbert

    _design(sampling_rate, low_hz, high_hz, order)
    t = np.arange(int(round(4 * sampling_rate))) / sampling_rate
    centre = (low_hz + high_hz) / 2
    burst = np.exp(-((t - 2.0) ** 2) / (2 * 0.1**2)) * np.sin(2 * np.pi * centre * t)
    y = np.asarray(band_pass(burst, sampling_rate, low_hz, high_hz, order, zero_phase=False).filtered)
    return float(t[np.argmax(np.abs(hilbert(y)))] - t[np.argmax(np.abs(hilbert(burst)))])


LESSON = {
    "question": "Raw EEG mixes the alpha rhythm with slower drifts and faster activity. What does "
    "the alpha rhythm look like on its own?",
    "what_it_does": "A band-pass filter keeps the 8 to 13 Hz part of the signal and turns down the "
    "rest. The filtered trace shows the alpha rhythm rising and falling in strength.",
    "costs": (
        "Run forward and backward, the filter adds no delay, but it uses later samples to compute "
        "earlier ones, so a live device cannot run it that way. Run forward only, it delays the "
        "output (de Cheveigné and Nelken, 2019).",
        "Filters ring after sharp changes and at the edges of a recording, which can look like an "
        "oscillation that was never there (Widmann et al., 2015).",
        "A narrow band-pass makes almost any input look rhythmic within the band. The filtered trace "
        "shows the band; it does not prove a rhythm was present.",
    ),
    "refs": ("widmann", "decheveigne"),
}

REFERENCES: dict[str, tuple[str, str]] = {
    "widmann": (
        "Widmann, A., Schröger, E., & Maess, B. (2015). Digital filter design for electrophysiological "
        "data: A practical approach. Journal of Neuroscience Methods, 250, 34-46.",
        "https://doi.org/10.1016/j.jneumeth.2014.08.002",
    ),
    "decheveigne": (
        "de Cheveigné, A., & Nelken, I. (2019). Filters: When, why, and how (not) to use them. "
        "Neuron, 102(2), 280-293.",
        "https://doi.org/10.1016/j.neuron.2019.02.039",
    ),
}
