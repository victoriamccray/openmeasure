"""
An illustrative analysis of one simulated naturalistic pain study.

This estimates a within-person coupling difference: for each participant
with enough observations in both pain states, the within-person
correlation between pain rating and physio_signal is computed separately
for localized and distributed observations, and the two are subtracted.
The reported estimate is the average of those per-participant
differences, with its standard error.

This is deliberately not a "precision score," a study-quality rating, or
a recommended analysis for a real version of this study: it is one
illustrative analysis aligned to the specific simulated design in
simulate.py, meant to show how the design's assumptions (participant
count, adherence, noise, imbalance) move an estimate and its
uncertainty, not to grade the design as good or bad.

A participant contributes to the estimate only if they have at least
min_observations_per_state retained observations in *both* pain states,
and if pain rating or physio_signal is constant within a state for
them (making a correlation undefined). Excluding a participant here is
a real, visible consequence of the simulated design (low adherence, or
an extreme pain_state_prevalence, can leave too few observations of one
state per person), not a computational failure: `estimate_coupling_difference`
never raises for this reason, only for malformed arguments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from .simulate import PAIN_STATE_DISTRIBUTED, PAIN_STATE_LOCALIZED, SimulatedStudy

DEFAULT_MIN_OBSERVATIONS_PER_STATE = 3


@dataclass(frozen=True)
class CouplingEstimate:
    """
    The illustrative distributed-minus-localized coupling-difference
    estimate for one simulated study.

    estimated_difference and standard_error are None when fewer than two
    participants had enough usable data to contribute a per-participant
    difference - not an error, a real consequence of the simulated
    design worth showing as-is.
    """

    estimated_difference: float | None
    standard_error: float | None
    n_participants_used: int
    n_participants_excluded_insufficient_data: int
    per_participant_differences: tuple[float, ...]


def _within_person_correlation(observations: pd.DataFrame) -> float | None:
    if observations["pain_rating"].nunique() < 2 or observations["physio_signal"].nunique() < 2:
        return None

    correlation = np.corrcoef(observations["pain_rating"], observations["physio_signal"])[0, 1]
    return None if np.isnan(correlation) else float(correlation)


def estimate_coupling_difference(
    study: SimulatedStudy,
    *,
    min_observations_per_state: int = DEFAULT_MIN_OBSERVATIONS_PER_STATE,
) -> CouplingEstimate:
    """
    Compute the illustrative coupling-difference estimate for one
    simulated study (see module docstring for the full definition).
    """

    if min_observations_per_state < 2:
        raise ValueError(
            "min_observations_per_state must be at least 2 (a correlation "
            f"needs at least 2 points); got {min_observations_per_state}."
        )

    differences: list[float] = []
    n_excluded = 0

    for _, participant_data in study.data.groupby("participant_id"):
        localized = participant_data.loc[participant_data["pain_state"] == PAIN_STATE_LOCALIZED]
        distributed = participant_data.loc[participant_data["pain_state"] == PAIN_STATE_DISTRIBUTED]

        if len(localized) < min_observations_per_state or len(distributed) < min_observations_per_state:
            n_excluded += 1
            continue

        r_localized = _within_person_correlation(localized)
        r_distributed = _within_person_correlation(distributed)

        if r_localized is None or r_distributed is None:
            n_excluded += 1
            continue

        differences.append(r_distributed - r_localized)

    if len(differences) < 2:
        return CouplingEstimate(
            estimated_difference=differences[0] if differences else None,
            standard_error=None,
            n_participants_used=len(differences),
            n_participants_excluded_insufficient_data=n_excluded,
            per_participant_differences=tuple(differences),
        )

    diffs = np.array(differences)
    standard_error = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))

    return CouplingEstimate(
        estimated_difference=float(diffs.mean()),
        standard_error=standard_error,
        n_participants_used=len(differences),
        n_participants_excluded_insufficient_data=n_excluded,
        per_participant_differences=tuple(differences),
    )
