"""
Generate a synthetic naturalistic pain study from a set of design
assumptions.

The scenario: does the coupling between subjective pain and a wearable
physiological signal change when chronic pain is localized versus
spatially distributed/referred/radiating? A naturalistic, one-week,
repeated-measures design: participants log a pain rating (paired with a
digital body map indicating localized vs. distributed pain, not modeled
directly here) alongside a wearable physiological reading, several times
a day.

This is illustrative simulation, not study participant data - see
DesignAssumptions' docstring for exactly which knobs this model exposes
and which real-world confounders (medication, activity, sleep, stress)
it deliberately does not model in v0.1.

v0.1 also simplifies "physiological signals (EDA and HR/HRV)" from the
motivating hypothesis to a single generic physio_signal channel, so the
estimator in estimate.py has one coupling to compute rather than two
near-duplicate pipelines. Extending to multiple channels is a stated
next step, not an oversight.

Generative model, per planned observation:

- pain_state: "distributed" with probability pain_state_prevalence,
  else "localized".
- true_pain_rating: Uniform(0, 10), the value that actually coincides
  with the physiological reading.
- recorded_pain_rating: true_pain_rating plus Gaussian noise scaled by
  temporal_misalignment_minutes (larger misalignment means the logged
  rating is a noisier stand-in for what was actually happening
  physiologically), clipped to [0, 10]. This is what a real analyst
  would see; true_pain_rating is not returned.
- Each participant has their own localized-state coupling slope, drawn
  around a fixed baseline with between_person_sd; their distributed-
  state slope is that same participant's localized slope plus
  effect_magnitude (the effect this design is meant to detect).
- true_physio = slope(participant, pain_state) * true_pain_rating,
  plus within_person_sd noise (real physiological variability the
  slope does not explain).
- observed_physio = true_physio plus sensor_noise_sd noise (wearable
  measurement error, a distinct source from within_person_sd).
- The observation is then kept with probability adherence_rate, else
  dropped (missing, not imputed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .design import DesignAssumptions

# Fixed, documented modeling constants, not derived from any cited
# study: v0.1's generative model needs a reference slope and a rating
# scale to be concrete, and these values are chosen only to produce
# legible, plausible-looking simulated numbers.
BASELINE_COUPLING_SLOPE = 1.0
PAIN_RATING_MIN = 0.0
PAIN_RATING_MAX = 10.0
MISALIGNMENT_NOISE_PER_MINUTE = 0.05

PAIN_STATE_LOCALIZED = "localized"
PAIN_STATE_DISTRIBUTED = "distributed"


@dataclass(frozen=True)
class SimulatedStudy:
    """
    One simulated run of the naturalistic pain study.

    data has one row per retained (non-missing) observation, columns:
    participant_id, day, observation_index, pain_state,
    pain_rating, physio_signal. Rows dropped for "adherence" are not
    included at all, matching how a real naturalistic study would only
    ever see the observations that were actually captured.
    """

    assumptions: DesignAssumptions
    data: pd.DataFrame
    n_observations_planned: int
    n_observations_retained: int
    n_localized_observations: int
    n_distributed_observations: int

    @property
    def pct_missing(self) -> float:
        if self.n_observations_planned == 0:
            raise ValueError("No observations were planned.")
        missing = self.n_observations_planned - self.n_observations_retained
        return missing / self.n_observations_planned


def generate_naturalistic_pain_study(assumptions: DesignAssumptions) -> SimulatedStudy:
    """
    Generate one synthetic naturalistic pain study.

    Deterministic given assumptions: identical DesignAssumptions
    (including its seed field) always produce an identical SimulatedStudy.
    """

    rng = np.random.default_rng(assumptions.seed)

    n_planned = assumptions.n_observations_planned

    participant_ids = np.repeat(
        np.arange(assumptions.n_participants),
        assumptions.observations_per_day * assumptions.duration_days,
    )
    days = np.tile(
        np.repeat(np.arange(assumptions.duration_days), assumptions.observations_per_day),
        assumptions.n_participants,
    )
    observation_index = np.tile(
        np.arange(assumptions.observations_per_day),
        assumptions.n_participants * assumptions.duration_days,
    )

    # One localized-state coupling slope per participant, plus that same
    # participant's distributed-state slope, applied per observation.
    participant_localized_slope = BASELINE_COUPLING_SLOPE + rng.normal(
        0.0, assumptions.between_person_sd, size=assumptions.n_participants
    )
    participant_distributed_slope = participant_localized_slope + assumptions.effect_magnitude

    is_distributed = rng.random(n_planned) < assumptions.pain_state_prevalence
    pain_state = np.where(is_distributed, PAIN_STATE_DISTRIBUTED, PAIN_STATE_LOCALIZED)

    slope_by_observation = np.where(
        is_distributed,
        participant_distributed_slope[participant_ids],
        participant_localized_slope[participant_ids],
    )

    true_pain_rating = rng.uniform(PAIN_RATING_MIN, PAIN_RATING_MAX, size=n_planned)

    misalignment_noise_sd = MISALIGNMENT_NOISE_PER_MINUTE * assumptions.temporal_misalignment_minutes
    recorded_pain_rating = np.clip(
        true_pain_rating + rng.normal(0.0, misalignment_noise_sd, size=n_planned)
        if misalignment_noise_sd > 0
        else true_pain_rating,
        PAIN_RATING_MIN,
        PAIN_RATING_MAX,
    )

    true_physio = slope_by_observation * true_pain_rating + rng.normal(
        0.0, assumptions.within_person_sd, size=n_planned
    )
    observed_physio = true_physio + rng.normal(0.0, assumptions.sensor_noise_sd, size=n_planned)

    retained = rng.random(n_planned) < assumptions.adherence_rate

    data = pd.DataFrame(
        {
            "participant_id": participant_ids,
            "day": days,
            "observation_index": observation_index,
            "pain_state": pain_state,
            "pain_rating": recorded_pain_rating,
            "physio_signal": observed_physio,
        }
    ).loc[retained]

    return SimulatedStudy(
        assumptions=assumptions,
        data=data.reset_index(drop=True),
        n_observations_planned=n_planned,
        n_observations_retained=int(retained.sum()),
        n_localized_observations=int((data["pain_state"] == PAIN_STATE_LOCALIZED).sum()),
        n_distributed_observations=int((data["pain_state"] == PAIN_STATE_DISTRIBUTED).sum()),
    )
