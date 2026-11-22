"""
Why fNIRS (and fMRI) respond more slowly than EEG: the linear convolution
model of the hemodynamic response.

EEG follows neural activity within milliseconds. Blood-oxygen measures
follow it through the hemodynamic response, which this lesson models as a
convolution h(t) = (n * k)(t): a train of brief neural events n(t)
convolved with a response function k(t) that rises and falls over
seconds (Boynton et al., 1996; Friston et al., 1998).

The response function is a double gamma of the shape standard fMRI
analysis uses: a gamma density with shape 6 (mode 5 s) minus one sixth of
a gamma density with shape 16 (mode 15 s), the post-response undershoot.
In this model a brief event's response peaks about 5 s later. Measured
responses differ across regions and people (Lindquist et al., 2009), and
the linear model breaks down for long or closely spaced stimuli, holding
when events are separated by at least about 4 s (Glover, 1999). Applying
a BOLD-shaped response to fNIRS oxygenated hemoglobin is a common
modeling assumption; fNIRS signals also carry heartbeat, breathing, and
scalp blood flow (Yücel et al., 2021).

Pure functions returning frozen dataclasses. No streamlit, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import gamma

RESPONSE_SHAPE = 6.0
UNDERSHOOT_SHAPE = 16.0
UNDERSHOOT_RATIO = 1.0 / 6.0
RESPONSE_SECONDS = 32.0
DEFAULT_RATE_HZ = 10.0

# Two responses count as separate when the dip between their peaks falls
# below this fraction of the smaller peak. A teaching rule of thumb, stated
# as such on the page.
SEPARATION_DIP = 0.75


class HemodynamicsError(ValueError):
    """Inputs the hemodynamic lesson cannot use."""


@dataclass(frozen=True)
class ResponseFunction:
    times: tuple[float, ...]
    values: tuple[float, ...]
    peak_time: float
    undershoot_time: float


@dataclass(frozen=True)
class HemodynamicResponse:
    """A train of brief neural events and the slow response it produces."""

    event_times: tuple[float, ...]
    times: tuple[float, ...]
    neural: tuple[float, ...]
    response: tuple[float, ...]
    peak_times: tuple[float, ...]
    first_peak_delay: float
    separable: bool


def _check_rate(rate_hz: float) -> None:
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise HemodynamicsError(f"rate_hz must be a positive finite number; got {rate_hz!r}.")


def response_function(rate_hz: float = DEFAULT_RATE_HZ, seconds: float = RESPONSE_SECONDS) -> ResponseFunction:
    """The double-gamma response to one brief event, scaled to a peak of 1."""
    _check_rate(rate_hz)
    if not np.isfinite(seconds) or seconds <= 0:
        raise HemodynamicsError(f"seconds must be positive; got {seconds!r}.")
    t = np.arange(0, seconds, 1 / rate_hz)
    k = gamma.pdf(t, RESPONSE_SHAPE) - UNDERSHOOT_RATIO * gamma.pdf(t, UNDERSHOOT_SHAPE)
    k = k / k.max()
    return ResponseFunction(
        times=tuple(float(v) for v in t),
        values=tuple(float(v) for v in k),
        peak_time=float(t[np.argmax(k)]),
        undershoot_time=float(t[np.argmin(k)]),
    )


def _local_peaks(values: np.ndarray) -> list[int]:
    return [i for i in range(1, len(values) - 1) if values[i] > values[i - 1] and values[i] >= values[i + 1] and values[i] > 0]


def hemodynamic_response(
    event_times,
    duration: float = 40.0,
    rate_hz: float = DEFAULT_RATE_HZ,
) -> HemodynamicResponse:
    """
    Convolve brief neural events with the response function.

    event_times are in seconds from the start. Each event is a unit
    impulse at the nearest sample; an event in the window's last half
    sample snaps to the last sample, within the sampling resolution.
    first_peak_delay is NaN when no peak falls at or after the first event
    inside the window. separable is True when, for every pair
    of neighbouring events, the response shows two peaks with a dip below
    SEPARATION_DIP of the smaller one between them.
    """
    _check_rate(rate_hz)
    if not np.isfinite(duration) or duration <= 0:
        raise HemodynamicsError(f"duration must be positive; got {duration!r}.")
    events = sorted(float(e) for e in event_times)
    if not events:
        raise HemodynamicsError("At least one event time is required.")
    if any(not np.isfinite(e) or e < 0 or e >= duration for e in events):
        raise HemodynamicsError("Event times must lie within [0, duration).")

    n_samples = int(round(duration * rate_hz))
    if n_samples < 1:
        raise HemodynamicsError(
            f"duration {duration} s at {rate_hz} Hz holds no samples; use a longer duration or a higher rate."
        )
    t = np.arange(n_samples) / rate_hz
    neural = np.zeros(n_samples)
    for e in events:
        neural[min(int(round(e * rate_hz)), n_samples - 1)] += 1.0
    k = np.array(response_function(rate_hz).values)
    response = np.convolve(neural, k)[:n_samples]

    peaks = _local_peaks(response)
    peak_times = tuple(float(t[i]) for i in peaks)
    first_peak = next((p for p in peak_times if p >= events[0]), float("nan"))

    separable = True
    if len(events) > 1:
        separable = len(peaks) >= len(events)
        if separable:
            for a, b in zip(peaks, peaks[1:]):
                dip = float(response[a:b + 1].min())
                if dip > SEPARATION_DIP * min(response[a], response[b]):
                    separable = False
                    break

    return HemodynamicResponse(
        event_times=tuple(events),
        times=tuple(float(v) for v in t),
        neural=tuple(float(v) for v in neural),
        response=tuple(float(v) for v in response),
        peak_times=peak_times,
        first_peak_delay=float(first_peak - events[0]),
        separable=separable,
    )


LESSON = {
    "question": "When a person sees a flash of light, EEG shows the response within a fraction of "
    "a second. fNIRS shows it seconds later. Are they measuring the same event?",
    "obstacle": "fNIRS measures blood oxygen, which changes only after neurons signal nearby blood "
    "vessels. The blood response is slow and smooth, so brief neural events blur together.",
    "formula_plain": "The blood response is the neural events convolved with a response function: "
    "each brief event adds one copy of the slow response, starting at that moment.",
    "limitations": (
        "The response is linear only approximately. It breaks down for long or closely spaced "
        "stimuli, and holds when events are at least about 4 s apart (Glover, 1999).",
        "The response function differs between brain regions and between people, and a wrong "
        "shape can bias estimates of timing and size (Lindquist et al., 2009).",
        "Using a BOLD-shaped response for fNIRS is a modeling assumption. fNIRS also carries "
        "heartbeat, breathing, and scalp blood flow (Yücel et al., 2021).",
    ),
    "refs": ("boynton", "friston_1998", "glover", "lindquist", "yucel", "logothetis"),
}

# References this lesson adds; yucel and logothetis come from
# core/neuroanatomy.py's REFERENCES, which the page already lists.
REFERENCES: dict[str, tuple[str, str]] = {
    "boynton": (
        "Boynton, G. M., Engel, S. A., Glover, G. H., & Heeger, D. J. (1996). Linear systems analysis "
        "of functional magnetic resonance imaging in human V1. Journal of Neuroscience, 16(13), 4207-4221.",
        "https://doi.org/10.1523/JNEUROSCI.16-13-04207.1996",
    ),
    "friston_1998": (
        "Friston, K. J., Fletcher, P., Josephs, O., Holmes, A., Rugg, M. D., & Turner, R. (1998). "
        "Event-related fMRI: Characterizing differential responses. NeuroImage, 7(1), 30-40.",
        "https://doi.org/10.1006/nimg.1997.0306",
    ),
    "glover": (
        "Glover, G. H. (1999). Deconvolution of impulse response in event-related BOLD fMRI. "
        "NeuroImage, 9(4), 416-429.",
        "https://doi.org/10.1006/nimg.1998.0419",
    ),
    "lindquist": (
        "Lindquist, M. A., Meng Loh, J., Atlas, L. Y., & Wager, T. D. (2009). Modeling the hemodynamic "
        "response function in fMRI: Efficiency, bias and mis-modeling. NeuroImage, 45(1 Suppl), S187-S198.",
        "https://doi.org/10.1016/j.neuroimage.2008.10.065",
    ),
}


def response_components(
    event_times,
    duration: float = 40.0,
    rate_hz: float = DEFAULT_RATE_HZ,
) -> tuple[tuple[float, ...], ...]:
    """
    One copy of the response per event, in event order.

    Their sum is hemodynamic_response(...).response: convolution adds one
    shifted copy of k per event, which the step-through shows one at a time.
    """
    combined = hemodynamic_response(event_times, duration, rate_hz)
    return tuple(
        hemodynamic_response([e], duration, rate_hz).response for e in combined.event_times
    )


CONVOLUTION_STEPS = (
    ("First event", "A brief burst of neural activity. EEG records it as it happens."),
    ("Its blood response", "One copy of the slow response k(t) starts at the event and peaks 5 s later."),
    ("Second event", "A second burst, 3 s later. EEG records it as it happens too."),
    ("Its blood response", "A second copy of k(t) starts at the second event."),
    ("Add them up", "The blood response h(t) is the sum of the copies. Events 3 s apart blur into one "
     "broad bump, while EEG kept them apart."),
)
STEP_EVENTS = (5.0, 8.0)

SYMBOLS = {
    "h": "The blood-oxygen response that fNIRS or fMRI records, over time.",
    "n": "The neural events: brief bursts of activity, the part EEG sees directly.",
    "k": "The response function: the slow blood response to one brief event.",
    "conv": "Convolution: place a copy of k at every event, scaled by its size, and add the copies.",
    "tau": "The time of an event; the integral sums over every event time.",
    "t": "The time at which the response is read.",
}
