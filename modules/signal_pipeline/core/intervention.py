"""
Real-signal intervention and downstream-utility measurement.

Unlike modality.py's Modality (an illustrative, author-assigned rating
per signal category), every value this module produces is computed from
a real, bundled EEG excerpt (see
sample_data/ds005420_sub-5_O1_eyes_snippet.csv - one subject, one
channel, from OpenNeuro ds005420, CC0). classify_eyes_state() measures a
real classification accuracy; it is not an illustrative score.

apply_noise() is exploratory signal perturbation only. It carries no
formal privacy guarantee and must never be described as "differential
privacy" or any other named mechanism, here or in any caller - it is
useful only to show that added noise degrades this particular inference,
nothing more specific than that.

apply_temporal_degradation() uses a proper anti-aliased downsample
(scipy.signal.decimate), not a naive hold-and-repeat: empirically, a
naive hold-and-repeat barely changes classification accuracy even at
extreme factors, because it does not actually remove the alpha band's
energy the way a real reduction in acquisition rate would. A real
downstream consequence requires an honest downsample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.signal import decimate, welch

ALPHA_BAND_HZ: tuple[float, float] = (8.0, 13.0)
WINDOW_SECONDS = 1.0

# The bundled sample_data/ds005420_sub-5_O1_eyes_snippet.csv's real
# acquisition rate (OpenNeuro ds005420's own eeg.json sidecar). Not
# stored per-row in that CSV - callers pass this constant alongside it.
SAMPLING_RATE_HZ = 500.0

# A small, illustrative set of fields a real recording/sharing pipeline
# would attach to a session - not a claim that these specific fields are
# present in the bundled CSV, whose own BIDS sidecars are mostly blank
# already (see README.md). Stands in for "what metadata removal strips
# before storage/sharing" in general, per Rothstein (2010) and Bagley et
# al. (2026) section 3.1.5 (Neural Data Stewardship).
ILLUSTRATIVE_METADATA_FIELDS: tuple[str, ...] = (
    "device_serial_number",
    "recording_timestamp",
    "subject_session_code",
)


def compute_alpha_power(window: np.ndarray, sampling_rate: float) -> float:
    """
    Alpha-band (8-13 Hz) power of one window, via Welch's method.

    Raises ValueError if sampling_rate cannot resolve the alpha band at
    all (Nyquist frequency at or below the band's lower edge) - the same
    condition that makes temporal degradation a cliff rather than a
    smooth cost: past this point there is no frequency content left to
    integrate, not just a noisier estimate of it.
    """

    if sampling_rate / 2 <= ALPHA_BAND_HZ[0]:
        raise ValueError(
            f"sampling_rate={sampling_rate} Hz cannot resolve the "
            f"{ALPHA_BAND_HZ[0]}-{ALPHA_BAND_HZ[1]} Hz alpha band: its "
            f"Nyquist frequency ({sampling_rate / 2} Hz) is at or below "
            f"the band's lower edge."
        )

    if window.size < 2:
        raise ValueError(f"window must have at least 2 samples; got {window.size}.")

    freqs, power_spectrum = welch(window, fs=sampling_rate, nperseg=window.size)
    in_band = (freqs >= ALPHA_BAND_HZ[0]) & (freqs <= ALPHA_BAND_HZ[1])

    return float(np.trapz(power_spectrum[in_band], freqs[in_band]))


@dataclass(frozen=True)
class ClassificationResult:
    """
    Result of classifying eyes-closed vs. eyes-open windows by alpha power.

    threshold is the midpoint between the two classes' mean alpha power -
    the simplest possible classifier, deliberately not a fitted model, so
    the result reflects the separation in the data rather than a
    classifier's own capacity to find it.
    """

    accuracy: float
    threshold: float
    alpha_power_eyes_closed: tuple[float, ...]
    alpha_power_eyes_open: tuple[float, ...]

    @property
    def n_windows_eyes_closed(self) -> int:
        return len(self.alpha_power_eyes_closed)

    @property
    def n_windows_eyes_open(self) -> int:
        return len(self.alpha_power_eyes_open)


def _windows(signal: np.ndarray, sampling_rate: float) -> list[np.ndarray]:
    window_size = int(round(sampling_rate * WINDOW_SECONDS))

    if window_size < 2:
        raise ValueError(
            f"sampling_rate={sampling_rate} Hz gives a {WINDOW_SECONDS}s "
            f"window of fewer than 2 samples."
        )

    return [
        signal[start : start + window_size]
        for start in range(0, len(signal) - window_size + 1, window_size)
    ]


def classify_eyes_state(
    eyes_closed: np.ndarray, eyes_open: np.ndarray, sampling_rate: float
) -> ClassificationResult:
    """
    Classify 1-second windows of each condition by alpha power, using a
    midpoint-of-class-means threshold, and report accuracy across both.
    """

    closed_power = tuple(
        compute_alpha_power(w, sampling_rate) for w in _windows(eyes_closed, sampling_rate)
    )
    open_power = tuple(
        compute_alpha_power(w, sampling_rate) for w in _windows(eyes_open, sampling_rate)
    )

    if not closed_power or not open_power:
        raise ValueError("Each condition needs at least one full window to classify.")

    threshold = (float(np.mean(closed_power)) + float(np.mean(open_power))) / 2.0

    correct = sum(p > threshold for p in closed_power) + sum(p <= threshold for p in open_power)
    accuracy = correct / (len(closed_power) + len(open_power))

    return ClassificationResult(
        accuracy=accuracy,
        threshold=threshold,
        alpha_power_eyes_closed=closed_power,
        alpha_power_eyes_open=open_power,
    )


def apply_noise(signal: np.ndarray, sd: float, seed: int) -> np.ndarray:
    """
    Add zero-mean Gaussian noise of the given standard deviation.

    Exploratory signal perturbation only - not a differential-privacy
    mechanism and not a claim of any specific privacy guarantee.
    Deterministic given seed, so the same seed and sd reproduce the same
    perturbed signal exactly.
    """

    if sd < 0:
        raise ValueError(f"sd must be non-negative; got {sd}.")

    rng = np.random.default_rng(seed)

    return signal + rng.normal(0.0, sd, size=signal.shape)


def apply_temporal_degradation(
    signal: np.ndarray, sampling_rate: float, factor: int
) -> tuple[np.ndarray, float]:
    """
    Reduce the effective sampling rate by `factor` via an anti-aliased
    downsample (not a naive hold-and-repeat, which does not actually
    remove the energy a lower acquisition rate would never have
    captured). Returns the degraded signal and its new effective
    sampling rate, for the caller to pass into compute_alpha_power.
    """

    if factor < 1:
        raise ValueError(f"factor must be at least 1; got {factor}.")

    if factor == 1:
        return signal, sampling_rate

    return decimate(signal, factor, ftype="fir"), sampling_rate / factor


def strip_metadata(fields: Mapping[str, str]) -> Mapping[str, str]:
    """
    Return `fields` with every key in ILLUSTRATIVE_METADATA_FIELDS removed.

    Never touches the signal itself: this is why metadata removal, unlike
    the other two interventions, leaves classify_eyes_state's accuracy
    unchanged by construction.
    """

    return {
        key: value for key, value in fields.items() if key not in ILLUSTRATIVE_METADATA_FIELDS
    }
