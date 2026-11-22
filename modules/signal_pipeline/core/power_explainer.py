"""
Why alpha power is computed the way it is, in three worked steps.

The Study Room's "Why the Math Works" tab walks a learner from one
puzzle to the measurement the journey uses:

    1. mean_and_mean_square()   A sine's mean is zero; its mean square is
                                A^2 / 2. Squaring stops the halves of each
                                cycle from cancelling.
    2. mixture_spectrum()       A signal built from several rhythms looks
                                tangled in time, but its spectrum separates
                                them; changing only the alpha component
                                changes only the 8 to 13 Hz part.
    3. window_spectrum()        One recorded second's spectrum. The area
                                from 8 to 13 Hz is that second's alpha
                                power. The whole area equals the window's
                                mean square after the Hann taper (Parseval),
                                which matches the plain mean square on
                                average for a steady signal.

Every spectrum here uses the settings intervention.compute_alpha_power
uses (one Hann-windowed segment per window, density scaling, mean
removed), and the tests check that the two agree, so the explainer can
never teach a different calculation from the one the journey runs.

Pure functions returning frozen dataclasses. No streamlit, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from .intervention import ALPHA_BAND_HZ, SAMPLING_RATE_HZ, compute_alpha_power

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# Step 2's teaching signal: one second at the journey's sampling rate.
MIXTURE_COMPONENTS_HZ_UV = ((6.0, 8.0), (20.0, 5.0))  # theta and beta, fixed
MIXTURE_ALPHA_HZ = 10.0
MIXTURE_NOISE_SD_UV = 3.0
MIXTURE_SEED = 2007
DISPLAY_MAX_HZ = 30.0


class PowerExplainerError(ValueError):
    """Inputs the explainer cannot use."""


@dataclass(frozen=True)
class MeanSquare:
    """Step 1: a sine's mean and mean square."""

    amplitude: float
    frequency_hz: float
    mean: float
    mean_square: float
    expected_mean_square: float
    times: tuple[float, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class Spectrum:
    """A one-sided power spectral density and what it adds up to."""

    freqs_hz: tuple[float, ...]
    density: tuple[float, ...]
    alpha_power: float
    total_power: float
    tapered_mean_square: float
    mean_square: float


@dataclass(frozen=True)
class MixtureSpectrum:
    """Step 2: a tangled signal and the spectrum that separates it."""

    alpha_amplitude: float
    times: tuple[float, ...]
    values: tuple[float, ...]
    spectrum: Spectrum


def _validated(values, sampling_rate: float) -> np.ndarray:
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise PowerExplainerError(f"sampling_rate must be a positive finite number; got {sampling_rate!r}.")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise PowerExplainerError("A signal needs to be one-dimensional with at least 2 samples.")
    if not np.all(np.isfinite(array)):
        raise PowerExplainerError("A signal cannot hold NaN or infinite samples.")
    return array


def mean_and_mean_square(
    amplitude: float,
    frequency_hz: float = 2.0,
    seconds: float = 1.0,
    sampling_rate: float = SAMPLING_RATE_HZ,
) -> MeanSquare:
    """
    The mean and mean square of a sine over whole cycles.

    Over a whole number of cycles the mean is 0 and the mean square is
    A^2 / 2: the hand result the explainer states, checked numerically.
    """
    if not np.isfinite(amplitude) or amplitude < 0:
        raise PowerExplainerError(f"amplitude must be a finite number of at least 0; got {amplitude!r}.")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise PowerExplainerError(f"frequency_hz must be positive; got {frequency_hz!r}.")
    cycles = frequency_hz * seconds
    if abs(cycles - round(cycles)) > 1e-9:
        raise PowerExplainerError("seconds must hold a whole number of cycles, or the mean is not zero.")
    n = int(round(seconds * sampling_rate))
    t = np.arange(n) / sampling_rate
    x = amplitude * np.sin(2 * np.pi * frequency_hz * t)
    return MeanSquare(
        amplitude=float(amplitude),
        frequency_hz=float(frequency_hz),
        mean=float(np.mean(x)),
        mean_square=float(np.mean(x**2)),
        expected_mean_square=float(amplitude**2 / 2),
        times=tuple(float(v) for v in t),
        values=tuple(float(v) for v in x),
    )


def window_spectrum(window, sampling_rate: float = SAMPLING_RATE_HZ) -> Spectrum:
    """
    The spectrum compute_alpha_power integrates, and what it adds up to.

    alpha_power: the 8 to 13 Hz area by the trapezoid rule, exactly as
        compute_alpha_power computes it.
    total_power: the sum of every bin times the bin width. By Parseval's
        theorem this equals tapered_mean_square exactly.
    tapered_mean_square: the mean-removed window's mean square with each
        sample weighted by the Hann taper, sum((x w)^2) / sum(w^2). The
        taper weighs the middle of the window more than its edges.
    mean_square: the plain mean square of the mean-removed window. For a
        steady signal it matches total_power on average; for one second
        of real EEG, whose strength changes within the second, the two
        can differ.
    """
    x = _validated(window, sampling_rate)
    freqs, density = welch(x, fs=sampling_rate, nperseg=x.size)
    band = (freqs >= ALPHA_BAND_HZ[0]) & (freqs <= ALPHA_BAND_HZ[1])
    centered = x - x.mean()
    taper = np.hanning(x.size + 1)[:-1]  # periodic Hann, scipy's default
    return Spectrum(
        freqs_hz=tuple(float(f) for f in freqs),
        density=tuple(float(d) for d in density),
        alpha_power=float(_trapezoid(density[band], freqs[band])),
        total_power=float(np.sum(density) * (freqs[1] - freqs[0])),
        tapered_mean_square=float(np.sum((centered * taper) ** 2) / np.sum(taper**2)),
        mean_square=float(np.mean(centered**2)),
    )


def mixture_spectrum(alpha_amplitude: float, sampling_rate: float = SAMPLING_RATE_HZ) -> MixtureSpectrum:
    """
    One second of theta + alpha + beta + noise, and its spectrum.

    Theta, beta, and the seeded noise stay fixed, so changing the alpha
    amplitude changes only the alpha part of the spectrum.
    """
    if not np.isfinite(alpha_amplitude) or alpha_amplitude < 0:
        raise PowerExplainerError(f"alpha_amplitude must be a finite number of at least 0; got {alpha_amplitude!r}.")
    n = int(round(sampling_rate))
    t = np.arange(n) / sampling_rate
    x = alpha_amplitude * np.sin(2 * np.pi * MIXTURE_ALPHA_HZ * t)
    for freq, amp in MIXTURE_COMPONENTS_HZ_UV:
        x = x + amp * np.sin(2 * np.pi * freq * t)
    x = x + np.random.default_rng(MIXTURE_SEED).normal(0, MIXTURE_NOISE_SD_UV, n)
    return MixtureSpectrum(
        alpha_amplitude=float(alpha_amplitude),
        times=tuple(float(v) for v in t),
        values=tuple(float(v) for v in x),
        spectrum=window_spectrum(x, sampling_rate),
    )


def agrees_with_pipeline(window, sampling_rate: float = SAMPLING_RATE_HZ) -> bool:
    """True when the explainer's alpha area equals the pipeline's alpha power."""
    return bool(np.isclose(window_spectrum(window, sampling_rate).alpha_power, compute_alpha_power(window, sampling_rate)))


# ------------------------------------------------------------------
# The lesson's words, one entry per step
# ------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    question: str
    explanation: str
    takeaway: str


STEPS: tuple[Step, ...] = (
    Step(
        key="square",
        title="Why square the signal?",
        question="A rhythm can be strong, yet its average voltage is zero. How do we measure its strength?",
        explanation="Over each cycle, the positive half and the negative half cancel, so the mean "
        "voltage stays at zero whatever the amplitude. Squaring makes every sample positive, so "
        "both halves add up. For a sine of amplitude A, the mean square is A² / 2.",
        takeaway="Power is the mean of the squared signal. That is why its unit is µV².",
    ),
    Step(
        key="separate",
        title="Why split the signal into frequencies?",
        question="Real EEG mixes several rhythms and noise. How do we measure only the alpha rhythm?",
        explanation="Fourier analysis splits a signal into sine waves of different frequencies. The "
        "power spectral density shows how the mean square is shared among them. Raise the alpha "
        "amplitude: the waveform barely changes to the eye, while the 8 to 13 Hz part of the "
        "spectrum grows and the rest stays put.",
        takeaway="The spectrum separates rhythms that overlap in time.",
    ),
    Step(
        key="integrate",
        title="Why add up the area from 8 to 13 Hz?",
        question="How does a spectrum become one number for one second of recording?",
        explanation="Each point of the spectrum is power per hertz. Adding up the area between 8 and "
        "13 Hz gives the power that belongs to the alpha band, in µV². The whole area equals the "
        "window's mean square after the Hann taper, which weighs the middle of the second more "
        "than its edges. For a steady signal that matches the plain mean square on average. Real "
        "EEG changes within a second, so here the two can differ.",
        takeaway="Alpha power is the alpha band's share of the window's mean square.",
    ),
)


# ------------------------------------------------------------------
# Welch's method, one operation at a time, for the step-through view
# ------------------------------------------------------------------


@dataclass(frozen=True)
class WelchSteps:
    """Each intermediate result between one window and its alpha power."""

    times: tuple[float, ...]
    centered: tuple[float, ...]
    taper: tuple[float, ...]
    tapered: tuple[float, ...]
    freqs_hz: tuple[float, ...]
    fft_magnitude: tuple[float, ...]
    density: tuple[float, ...]
    in_band: tuple[bool, ...]
    alpha_power: float


def welch_steps(window, sampling_rate: float = SAMPLING_RATE_HZ) -> WelchSteps:
    """
    The operations compute_alpha_power performs, exposed in order:

    1. remove the mean; 2. multiply by a periodic Hann taper; 3. take the
    FFT magnitude; 4. square and scale it to a one-sided power spectral
    density, |X|^2 / (fs * sum(w^2)), doubled except at 0 Hz and Nyquist;
    5. integrate the 8 to 13 Hz bins with the trapezoid rule.
    The density and alpha power equal window_spectrum's, which the tests
    check.
    """
    x = _validated(window, sampling_rate)
    centered = x - x.mean()
    taper = np.hanning(x.size + 1)[:-1]
    tapered = centered * taper
    spectrum = np.fft.rfft(tapered)
    freqs = np.fft.rfftfreq(x.size, 1 / sampling_rate)
    density = np.abs(spectrum) ** 2 / (sampling_rate * np.sum(taper**2))
    if x.size % 2 == 0:
        density[1:-1] *= 2
    else:
        density[1:] *= 2
    band = (freqs >= ALPHA_BAND_HZ[0]) & (freqs <= ALPHA_BAND_HZ[1])
    return WelchSteps(
        times=tuple(float(v) for v in np.arange(x.size) / sampling_rate),
        centered=tuple(float(v) for v in centered),
        taper=tuple(float(v) for v in taper),
        tapered=tuple(float(v) for v in tapered),
        freqs_hz=tuple(float(f) for f in freqs),
        fft_magnitude=tuple(float(v) for v in np.abs(spectrum)),
        density=tuple(float(d) for d in density),
        in_band=tuple(bool(b) for b in band),
        alpha_power=float(_trapezoid(density[band], freqs[band])),
    )


WELCH_STEPS: tuple[Step, ...] = (
    Step("record", "One recorded second", "Start with one second of the recording, its mean removed.",
         "500 voltage samples, centred on zero.", "The window is the raw material."),
    Step("taper", "Taper the edges", "Multiply by a Hann window, which fades the edges to zero.",
         "A cut-off window has sharp edges, which spread power into frequencies that are absent "
         "from the signal. The taper softens them.", "The middle of the second now counts most."),
    Step("fft", "Split into frequencies", "The Fourier transform finds how much of each frequency the "
         "tapered window holds.", "One value per frequency, in steps of 1 Hz.", "Each bar is one frequency's share."),
    Step("power", "Square to get power", "Square each magnitude and scale it, so the area under the "
         "curve is power in µV².", "Squaring is the same step as in the first lesson: it keeps "
         "every contribution positive.", "The curve is the power spectral density, in µV² per Hz."),
    Step("area", "Add up the alpha band", "Add the area under the curve from 8 to 13 Hz.",
         "That area is the alpha power of this second.", "One number for one second of recording."),
)

SYMBOLS = {
    "P_alpha": "Alpha power: the power in the 8 to 13 Hz band for one window, in µV².",
    "S_xx": "The power spectral density: how much power the signal has at each frequency, in µV² per Hz.",
    "f": "Frequency, in hertz (cycles per second).",
    "band": "The limits 8 and 13 Hz: the alpha band, a convention.",
    "df": "A small step of frequency: adding up area means summing density times this width.",
    "x_t": "The recorded voltage at time t, in µV.",
    "T": "The length of the window, here one second.",
    "dt": "A small step of time: averaging means summing over it and dividing by T.",
    "square": "Squaring: makes negative and positive halves both count as strength.",
    "A": "The sine's amplitude: its peak voltage, in µV.",
    "xbar": "The window's mean voltage, removed first so it does not count as power at 0 Hz.",
    "w": "The Hann taper: 0 at the window's edges, 1 in the middle.",
    "n": "The sample number within the window, from 0 to N minus 1.",
    "N": "The number of samples in the window: 500 for one second at 500 Hz.",
    "X_k": "The Fourier coefficient: how much of the k-th frequency the tapered window holds.",
    "fs": "The sampling rate: 500 samples per second.",
}
