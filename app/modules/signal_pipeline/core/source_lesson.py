"""
Why EEG source localization is ambiguous: different source configurations
can produce identical scalp readings.

A teaching model with three sources (left, centre, right) and two scalp
electrodes. The forward model y = L x maps source strengths x to
electrode readings y through a lead field L:

    L = [[1.0, 0.6, 0.2],     electrode 1 (left)
         [0.2, 0.6, 1.0]]     electrode 2 (right)

With more sources than electrodes, L has a null space. Here it is the
direction (1, -2, 1): adding any multiple of it changes the sources and
leaves the readings unchanged. So (0, 1, 0), (0.25, 0.5, 0.25), and
(0.5, 0, 0.5) all give y = (0.6, 0.6), hand-checkable.

Picking one answer needs an assumption. The minimum-norm estimate picks
the solution with the smallest total strength (Hämäläinen and Ilmoniemi,
1994); here that is (1/3, 1/3, 1/3), different from each of the three
configurations shown. Real inverse methods add constraints of this kind
(Michel et al., 2004; Grech et al., 2008). Real heads have thousands of
candidate sources and dozens to hundreds of electrodes, so the null space
is far larger than in this model.

Pure functions returning frozen dataclasses. No streamlit, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOURCES = ("Left source", "Centre source", "Right source")
ELECTRODES = ("Electrode 1 (left)", "Electrode 2 (right)")
LEAD_FIELD = ((1.0, 0.6, 0.2), (0.2, 0.6, 1.0))
NULL_DIRECTION = (1.0, -2.0, 1.0)
BASE_CONFIGURATION = (0.0, 1.0, 0.0)
SHOWN_STEPS = (0.0, 0.25, 0.5)


class SourceLessonError(ValueError):
    """Inputs the source model cannot use."""


@dataclass(frozen=True)
class Configuration:
    label: str
    strengths: tuple[float, ...]
    readings: tuple[float, ...]


@dataclass(frozen=True)
class Ambiguity:
    lead_field: tuple[tuple[float, ...], ...]
    configurations: tuple[Configuration, ...]
    readings: tuple[float, ...]
    minimum_norm: Configuration
    null_direction: tuple[float, ...]


def forward(strengths) -> tuple[float, ...]:
    """Electrode readings y = L x for source strengths x."""
    x = np.asarray(strengths, dtype=float)
    if x.shape != (len(SOURCES),):
        raise SourceLessonError(f"strengths needs exactly {len(SOURCES)} values.")
    if not np.all(np.isfinite(x)):
        raise SourceLessonError("strengths cannot hold NaN or infinite values.")
    return tuple(float(v) for v in np.asarray(LEAD_FIELD) @ x)


def minimum_norm(readings) -> tuple[float, ...]:
    """The smallest-total-strength sources that reproduce the readings exactly."""
    y = np.asarray(readings, dtype=float)
    if y.shape != (len(ELECTRODES),) or not np.all(np.isfinite(y)):
        raise SourceLessonError(f"readings needs exactly {len(ELECTRODES)} finite values.")
    return tuple(float(v) for v in np.linalg.pinv(np.asarray(LEAD_FIELD)) @ y)


def ambiguity() -> Ambiguity:
    """Three source configurations with identical readings, and the minimum-norm pick."""
    base = np.asarray(BASE_CONFIGURATION)
    null = np.asarray(NULL_DIRECTION)
    labels = ("One central source", "Spread across all three", "Two outer sources")
    configs = []
    for label, t in zip(labels, SHOWN_STEPS):
        x = base + t * null
        configs.append(Configuration(label, tuple(float(v) for v in x), forward(x)))
    readings = configs[0].readings
    mn = minimum_norm(readings)
    return Ambiguity(
        lead_field=LEAD_FIELD,
        configurations=tuple(configs),
        readings=readings,
        minimum_norm=Configuration("Minimum-norm estimate", mn, forward(mn)),
        null_direction=NULL_DIRECTION,
    )


LESSON = {
    "question": "Electrode O1 picked up an alpha rhythm. Which brain sources produced it?",
    "obstacle": "Each electrode records a weighted sum of many sources. With more possible sources "
    "than electrodes, different patterns of source activity can produce exactly the same readings.",
    "formula_plain": "The readings y equal the lead field L times the source strengths x, plus noise. "
    "L says how strongly each source reaches each electrode.",
    "limitations": (
        "A three-source, two-electrode model is a teaching simplification. Real heads have thousands "
        "of candidate sources, which makes the ambiguity larger.",
        "In this model each electrode is most sensitive to the source nearest it. In a real head, "
        "orientation and depth matter too, so the cortex beneath an electrode does not always "
        "contribute most.",
        "Choosing one answer needs an assumption, such as the smallest total strength or smooth "
        "activity. Different assumptions give different answers (Grech et al., 2008).",
        "Methods combine many electrodes, a head model from MRI, and such assumptions to make the "
        "estimate useful (Michel et al., 2004).",
    ),
    "refs": ("hamalainen", "michel", "grech"),
}

SYMBOLS = {
    "y": "The electrode readings: one voltage per electrode.",
    "L": "The lead field: how strongly each source reaches each electrode, set by distance, "
    "orientation, and the tissue between them.",
    "x": "The source strengths inside the brain: the unknowns.",
    "epsilon": "Noise: everything recorded that the sources do not explain.",
}

REFERENCES: dict[str, tuple[str, str]] = {
    "hamalainen": (
        "Hämäläinen, M. S., & Ilmoniemi, R. J. (1994). Interpreting magnetic fields of the brain: "
        "Minimum norm estimates. Medical & Biological Engineering & Computing, 32(1), 35-42.",
        "https://doi.org/10.1007/BF02512476",
    ),
    "michel": (
        "Michel, C. M., Murray, M. M., Lantz, G., Gonzalez, S., Spinelli, L., & Grave de Peralta, R. "
        "(2004). EEG source imaging. Clinical Neurophysiology, 115(10), 2195-2222.",
        "https://doi.org/10.1016/j.clinph.2004.06.001",
    ),
    "grech": (
        "Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G., Zervakis, M., Xanthopoulos, P., "
        "Sakkalis, V., & Vanrumste, B. (2008). Review on solving the inverse problem in EEG source "
        "analysis. Journal of NeuroEngineering and Rehabilitation, 5, 25.",
        "https://doi.org/10.1186/1743-0003-5-25",
    ),
}
