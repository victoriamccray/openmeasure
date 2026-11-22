"""
How correlation can mislead: two signals can correlate strongly with no
direct link between them.

The lesson simulates two recorded signals in three wiring diagrams:

    direct        Y = b X + noise          X drives Y
    common input  X = a Z + noise_x,       a third source Z drives both
                  Y = a Z + noise_y
    shared source x1 = w1 S + noise_1,     two electrodes see one source
                  x2 = w2 S + noise_2      through volume conduction

With unit-variance sources and noise of standard deviation s, the
expected correlations have closed forms, which the tests check:

    direct:        r = b / sqrt(b^2 + s^2)
    common input:  r = a^2 / (a^2 + s^2)
    shared source: r = w1 w2 / sqrt((w1^2 + s^2)(w2^2 + s^2))

The point the page makes: the direct and common-input wirings can give
the same r, so r alone cannot say which wiring produced it (Bastos and
Schoffelen, 2016; Nunez et al., 1997).

Pure functions returning frozen dataclasses. No streamlit, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WIRING_DIRECT = "direct"
WIRING_COMMON = "common input"
WIRING_SHARED = "shared source"
WIRINGS = (WIRING_DIRECT, WIRING_COMMON, WIRING_SHARED)

DEFAULT_SAMPLES = 400
DEFAULT_SEED = 1997


class CorrelationLessonError(ValueError):
    """Inputs the correlation lesson cannot use."""


@dataclass(frozen=True)
class Simulation:
    wiring: str
    strength: float
    noise_sd: float
    x: tuple[float, ...]
    y: tuple[float, ...]
    r: float
    expected_r: float
    direct_link: bool


def pearson_r(x, y) -> float:
    """Pearson correlation: Cov(X, Y) / (sd X sd Y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 3:
        raise CorrelationLessonError("x and y must be 1-D, the same length, and at least 3 long.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise CorrelationLessonError("x and y cannot hold NaN or infinite values.")
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        raise CorrelationLessonError("A constant signal has no correlation.")
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def expected_r(wiring: str, strength: float, noise_sd: float, second_weight: float | None = None) -> float:
    """The closed-form correlation for each wiring, unit-variance sources."""
    if wiring not in WIRINGS:
        raise CorrelationLessonError(f"wiring must be one of {WIRINGS}; got {wiring!r}.")
    if not np.isfinite(strength) or strength < 0:
        raise CorrelationLessonError("strength must be a finite number of at least 0.")
    if not np.isfinite(noise_sd) or noise_sd <= 0:
        raise CorrelationLessonError("noise_sd must be positive.")
    s2 = noise_sd**2
    if wiring == WIRING_DIRECT:
        return float(strength / np.sqrt(strength**2 + s2))
    if wiring == WIRING_COMMON:
        return float(strength**2 / (strength**2 + s2))
    w2 = strength if second_weight is None else second_weight
    return float(strength * w2 / np.sqrt((strength**2 + s2) * (w2**2 + s2)))


def simulate(
    wiring: str,
    strength: float,
    noise_sd: float = 1.0,
    n: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    second_weight: float | None = None,
) -> Simulation:
    """
    Simulate two signals under one wiring, with a fixed seed.

    strength is b (direct), a (common input), or w1 (shared source; w2
    defaults to the same weight). Sources and noise are independent
    standard normals, noise scaled by noise_sd.
    """
    target = expected_r(wiring, strength, noise_sd, second_weight)
    if not isinstance(n, int) or n < 3:
        raise CorrelationLessonError("n must be a whole number of at least 3.")
    rng = np.random.default_rng(seed)
    source = rng.standard_normal(n)
    noise_a = noise_sd * rng.standard_normal(n)
    noise_b = noise_sd * rng.standard_normal(n)
    if wiring == WIRING_DIRECT:
        x = source
        y = strength * x + noise_a
    elif wiring == WIRING_COMMON:
        x = strength * source + noise_a
        y = strength * source + noise_b
    else:
        w2 = strength if second_weight is None else second_weight
        x = strength * source + noise_a
        y = w2 * source + noise_b
    return Simulation(
        wiring=wiring,
        strength=float(strength),
        noise_sd=float(noise_sd),
        x=tuple(float(v) for v in x),
        y=tuple(float(v) for v in y),
        r=pearson_r(x, y),
        expected_r=target,
        direct_link=wiring == WIRING_DIRECT,
    )


def matching_direct_strength(common_strength: float, noise_sd: float = 1.0) -> float:
    """
    The direct-link b whose expected r equals a common input of strength a.

    Solves b / sqrt(b^2 + s^2) = a^2 / (a^2 + s^2) for b: with target r,
    b = s r / sqrt(1 - r^2). Lets the page show two wirings with one r.
    """
    r = expected_r(WIRING_COMMON, common_strength, noise_sd)
    if r >= 1:
        raise CorrelationLessonError("A perfect correlation has no finite matching strength.")
    return float(noise_sd * r / np.sqrt(1 - r**2))


LESSON = {
    "question": "Two brain regions' signals rise and fall together. Are the regions communicating?",
    "obstacle": "Correlation measures whether two signals move together. It cannot tell whether one "
    "drives the other, whether a third source drives both, or whether two electrodes pick up one "
    "source through the tissue between them.",
    "formula_plain": "Correlation is the covariance of the two signals divided by the product of "
    "their standard deviations. It runs from minus 1 to 1.",
    "limitations": (
        "A common input, such as a deep source driving two cortical regions, correlates them with "
        "no link between them (Bastos and Schoffelen, 2016).",
        "Volume conduction lets nearby electrodes record the same source, which inflates "
        "correlation between them (Nunez et al., 1997).",
        "A shared reference electrode or shared noise adds the same signal to every channel and "
        "has the same effect (Bastos and Schoffelen, 2016).",
        "Correlation also misses links that are nonlinear or delayed.",
    ),
    "refs": ("bastos", "nunez"),
}

REFERENCES: dict[str, tuple[str, str]] = {
    "bastos": (
        "Bastos, A. M., & Schoffelen, J.-M. (2016). A tutorial review of functional connectivity "
        "analysis methods and their interpretational pitfalls. Frontiers in Systems Neuroscience, 9, 175.",
        "https://doi.org/10.3389/fnsys.2015.00175",
    ),
    "nunez": (
        "Nunez, P. L., Srinivasan, R., Westdorp, A. F., Wijesinghe, R. S., Tucker, D. M., Silberstein, "
        "R. B., & Cadusch, P. J. (1997). EEG coherency I: Statistics, reference electrode, volume "
        "conduction, Laplacians, cortical imaging, and interpretation at multiple scales. "
        "Electroencephalography and Clinical Neurophysiology, 103(5), 499-515.",
        "https://doi.org/10.1016/s0013-4694(97)00066-7",
    ),
}
