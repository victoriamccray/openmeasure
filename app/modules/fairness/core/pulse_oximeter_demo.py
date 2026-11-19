"""
Worked example: racial bias in pulse oximetry (Sjoding et al., 2020).

Generates a synthetic cohort calibrated to the occult-hypoxemia rates
Sjoding et al. (2020, NEJM) reported for patients whose pulse-oximeter
reading fell in the "reassuring" 92-96% band, then evaluates an
adjustable device-alarm threshold through the fairness module's own
post-model confusion-matrix metrics (compare_post_model_bias) rather
than introducing a second, bespoke statistic.

Two things are drawn from the cited study and two are not, and the
distinction matters for how the results should be read:

- Drawn from the study: each group's sample size and occult-hypoxemia
  rate within the 92-96% reading band (PUBLISHED_COHORTS below).
- Not drawn from the study, and disclosed as illustrative: the shape of
  each group's reading distribution within that band. The paper reports
  the overall rate, not how far into the band a hypoxemic patient's
  reading typically falls. This module assumes the group with the higher
  published rate also has its hypoxemic patients' readings concentrated
  closer to the top of the band (harder to catch by raising the alarm
  threshold), a plausible but unverified mechanism, so that the
  threshold slider has something to demonstrate. Every SyntheticCohort
  carries this assumption in plain language via its `assumptions` field.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.fairness.core.post_model_metrics import PostModelBiasResult, compare_post_model_bias

# The pulse-oximeter reading band Sjoding et al. define occult hypoxemia
# within: a reading of 92-96% taken as reassuring, paired with an
# arterial blood gas reading below 88%.
BAND_LOWER = 92.0
BAND_UPPER = 96.0

# Illustrative concentration parameters for the Beta distribution used to
# place a hypoxemic patient's reading within the band (see module
# docstring). Not derived from the cited study.
BETA_SKEW_LOW = 1.5
BETA_SKEW_HIGH = 4.0


@dataclass(frozen=True)
class PublishedGroupRate:
    """One group's reported sample size and occult-hypoxemia rate."""

    group: str
    n: int
    occult_hypoxemia_rate: float

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError(f"'{self.group}' must have a positive sample size.")
        if not 0.0 <= self.occult_hypoxemia_rate <= 1.0:
            raise ValueError(
                f"'{self.group}' occult_hypoxemia_rate must be between 0 and "
                f"1; got {self.occult_hypoxemia_rate}."
            )


@dataclass(frozen=True)
class PublishedCohort:
    """One of the two cohorts Sjoding et al. (2020) report."""

    cohort_id: str
    label: str
    citation: str
    group_rates: tuple[PublishedGroupRate, ...]

    def __post_init__(self) -> None:
        if len(self.group_rates) != 2:
            raise ValueError(
                f"'{self.cohort_id}' must name exactly two groups to support "
                "a privileged/unprivileged comparison; "
                f"got {len(self.group_rates)}."
            )


_CITATION = (
    "Sjoding, M. W., Dickson, R. P., Iwashyna, T. J., Gay, S. E., & "
    "Valley, T. S. (2020). Racial bias in pulse oximetry measurement "
    "[letter]. New England Journal of Medicine, 383(25), 2477-2478."
)

PUBLISHED_COHORTS: dict[str, PublishedCohort] = {
    "michigan": PublishedCohort(
        cohort_id="michigan",
        label="University of Michigan Hospital",
        citation=_CITATION,
        group_rates=(
            PublishedGroupRate(group="White", n=1296, occult_hypoxemia_rate=0.04),
            PublishedGroupRate(group="Black", n=269, occult_hypoxemia_rate=0.12),
        ),
    ),
    "multicenter": PublishedCohort(
        cohort_id="multicenter",
        label="Multicenter ICU database (~178 hospitals, 2014-2015)",
        citation=_CITATION,
        group_rates=(
            PublishedGroupRate(group="White", n=7342, occult_hypoxemia_rate=0.06),
            PublishedGroupRate(group="Black", n=1050, occult_hypoxemia_rate=0.17),
        ),
    ),
}


@dataclass(frozen=True)
class SyntheticCohort:
    """
    A synthetic patient-level cohort calibrated to a PublishedCohort.

    data has one row per synthetic patient with columns "group" (str),
    "hypoxemic" (bool, True if this patient's true arterial oxygen
    saturation was below 88%), and "device_reading" (float, the
    synthetic pulse-oximeter reading, always within [BAND_LOWER,
    BAND_UPPER)). This is illustrative simulation, not study
    participant data: see the module docstring for what is and is not
    drawn from the cited study.
    """

    cohort_id: str
    data: pd.DataFrame
    calibration_target: tuple[PublishedGroupRate, ...]
    assumptions: tuple[str, ...]


def generate_synthetic_cohort(cohort_id: str, *, seed: int = 0) -> SyntheticCohort:
    """
    Generate a synthetic cohort calibrated to a published cohort's
    occult-hypoxemia rate.

    Each group's count of hypoxemic patients is set deterministically
    (round(n * rate)), so the generated rate matches the cited rate up
    to rounding, regardless of seed. seed only controls where, within
    the reading band, each synthetic patient's reading falls.
    """

    if cohort_id not in PUBLISHED_COHORTS:
        raise ValueError(
            f"'{cohort_id}' is not a known cohort. Known cohorts: "
            f"{', '.join(sorted(PUBLISHED_COHORTS))}."
        )

    published = PUBLISHED_COHORTS[cohort_id]
    rng = np.random.default_rng(seed)

    # The group with the higher published rate is assumed to also have
    # its hypoxemic patients concentrated nearer the top of the
    # reassuring band, so it takes a higher alarm threshold to catch
    # them (see module docstring: this shape is illustrative, not
    # itself drawn from the cited study).
    ranked = sorted(published.group_rates, key=lambda gr: gr.occult_hypoxemia_rate)
    skew_by_group = {ranked[0].group: BETA_SKEW_LOW, ranked[-1].group: BETA_SKEW_HIGH}

    band_width = BAND_UPPER - BAND_LOWER
    frames: list[pd.DataFrame] = []

    for group_rate in published.group_rates:
        n_hypoxemic = round(group_rate.n * group_rate.occult_hypoxemia_rate)
        n_normal = group_rate.n - n_hypoxemic
        skew = skew_by_group[group_rate.group]

        hypoxemic_readings = BAND_LOWER + band_width * rng.beta(skew, 1.0, size=n_hypoxemic)
        normal_readings = BAND_LOWER + band_width * rng.beta(2.0, 2.0, size=n_normal)

        frames.append(
            pd.DataFrame(
                {
                    "group": group_rate.group,
                    "hypoxemic": [True] * n_hypoxemic + [False] * n_normal,
                    "device_reading": np.concatenate([hypoxemic_readings, normal_readings]),
                }
            )
        )

    data = pd.concat(frames, ignore_index=True)

    assumptions = (
        f"Each group's sample size and occult-hypoxemia rate within the "
        f"{BAND_LOWER:.0f}-{BAND_UPPER:.0f}% pulse-oximeter reading band "
        f"are calibrated to {published.citation}",
        (
            "The shape of each group's reading distribution within that "
            "band (how close a hypoxemic patient's reading sits to the "
            "top of the band, and therefore how high an alarm threshold "
            "would need to be to catch them) is an illustrative modeling "
            "assumption, not itself reported in the cited study."
        ),
    )

    return SyntheticCohort(
        cohort_id=cohort_id,
        data=data,
        calibration_target=published.group_rates,
        assumptions=assumptions,
    )


def evaluate_action_threshold(
    cohort: SyntheticCohort,
    threshold: float,
    *,
    privileged_group: str,
    unprivileged_group: str,
) -> PostModelBiasResult:
    """
    Flag every synthetic patient whose device reading falls below
    threshold, then compare detection behavior between two groups using
    the fairness module's own compare_post_model_bias.

    threshold must fall strictly inside (BAND_LOWER, BAND_UPPER): at or
    below BAND_LOWER nobody is flagged, and at or above BAND_UPPER
    everybody is, either of which collapses the predicted-label column
    to a single value and makes a group comparison undefined.
    """

    if not BAND_LOWER < threshold < BAND_UPPER:
        raise ValueError(
            f"threshold must be strictly between {BAND_LOWER:.0f} and "
            f"{BAND_UPPER:.0f} (the pulse-oximeter reading band this "
            f"cohort was calibrated to); got {threshold}."
        )

    frame = cohort.data.copy()
    frame["flagged"] = frame["device_reading"] < threshold

    return compare_post_model_bias(
        frame,
        true_label_col="hypoxemic",
        predicted_label_col="flagged",
        group_col="group",
        positive_label=True,
        privileged_group=privileged_group,
        unprivileged_group=unprivileged_group,
    )
