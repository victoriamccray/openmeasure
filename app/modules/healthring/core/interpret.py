"""
Turns raw HealthRing agreement statistics into plain-language sentences.

Every MAE, bias, limits-of-agreement, and per-window quality value on the
Wearables Research Journey is followed by one of these sentences, so a
number never appears without a plain-language reading. Written once here,
alongside modules/reliability/core/interpret.py's equivalent role for
Cronbach's alpha, rather than as page-local functions in
pages/HealthRing_Worked_Example.py.
"""

from __future__ import annotations

# Bland & Altman (1986): the limits of agreement span the mean difference
# plus or minus 1.96 sample standard deviations, covering ~95% of
# differences under a normal-difference assumption. Matches
# acquisition_robustness.py's LOA_MULTIPLIER.
LOA_COVERAGE_PCT = 95


def mae_sentence(value: float) -> str:
    return f"Predictions differed from the reference by about {value:.1f} bpm on average."


def bias_sentence(value: float) -> str:
    if abs(value) < 0.05:
        return (
            "Predictions were about equal to the reference on average, "
            "with no consistent over- or under-estimate."
        )

    direction = "higher" if value > 0 else "lower"
    return f"Predictions were about {abs(value):.1f} bpm {direction} than the reference on average."


def loa_sentence(lower: float, upper: float) -> str:
    return (
        f"For about {LOA_COVERAGE_PCT}% of windows, the error is expected "
        f"to fall between {lower:+.1f} and {upper:+.1f} bpm, following the "
        "Bland-Altman convention (bias plus or minus 1.96 standard "
        "deviations)."
    )


def quality_sentence(value: float) -> str:
    if value >= 0.7:
        level = "high"
    elif value >= 0.4:
        level = "middling"
    else:
        level = "low"

    return (
        f"The ring scored this window's own reading as {level} usability "
        f"({value:.2f} on its 0-1 scale). That score does not, by itself, "
        "say what made it that way."
    )
