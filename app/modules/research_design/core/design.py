"""
Design assumptions for a simulated study, before any real data exists.

DesignAssumptions is the complete input to modules/research_design/core/
simulate.py: every knob a reader can turn in the "Design Assumptions" and
"Interactive Exploration" stages is a field here, and nothing here is
inferred from data, because there is none yet. Deliberately narrow for
v0.1 (Aug 2026): no medication, activity, sleep, stress, or other
confounders. Those can be added once the basic
design -> simulated data -> method-selection flow is validated, not
before.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DesignAssumptions:
    """
    One fully-specified set of design choices for the naturalistic pain
    study (see simulate.py's module docstring for the scenario this
    parameterizes).

    n_participants: number of people enrolled.
    observations_per_day: planned wearable+pain-rating observations per
        participant per day.
    duration_days: length of the naturalistic observation period.
    adherence_rate: probability any single planned observation is
        actually captured (the rest are missing, not fabricated).
    sensor_noise_sd: standard deviation of wearable measurement error,
        added on top of the true physiological signal.
    within_person_sd: standard deviation of a participant's own
        moment-to-moment physiological variability that the pain-rating
        coupling does not explain, separate from sensor_noise_sd.
    between_person_sd: standard deviation, across participants, of each
        participant's own baseline coupling between pain rating and
        physiological signal.
    effect_magnitude: the true, population-average difference in
        coupling between the distributed and localized pain states -
        the effect this design is meant to detect.
    pain_state_prevalence: probability any given observation occurs
        during a "distributed" pain state rather than "localized"
        (within-person imbalance between the two states).
    temporal_misalignment_minutes: how far apart, on average, the logged
        pain rating and the wearable reading are in time; larger values
        add more noise to the recorded pain rating relative to what
        actually drove the physiological signal at that moment.
    seed: random seed. Identical DesignAssumptions (including this
        field) must produce identical simulated data - see
        test_simulate.py's determinism test.
    """

    n_participants: int
    observations_per_day: int
    duration_days: int
    adherence_rate: float
    sensor_noise_sd: float
    within_person_sd: float
    between_person_sd: float
    effect_magnitude: float
    pain_state_prevalence: float
    temporal_misalignment_minutes: float
    seed: int

    def __post_init__(self) -> None:
        if self.n_participants <= 0:
            raise ValueError("n_participants must be positive.")
        if self.observations_per_day <= 0:
            raise ValueError("observations_per_day must be positive.")
        if self.duration_days <= 0:
            raise ValueError("duration_days must be positive.")
        if not 0.0 < self.adherence_rate <= 1.0:
            raise ValueError(
                f"adherence_rate must be in (0, 1]; got {self.adherence_rate}. "
                "A rate of exactly 0 would produce zero observations."
            )
        if self.sensor_noise_sd < 0:
            raise ValueError("sensor_noise_sd cannot be negative.")
        if self.within_person_sd < 0:
            raise ValueError("within_person_sd cannot be negative.")
        if self.between_person_sd < 0:
            raise ValueError("between_person_sd cannot be negative.")
        if not 0.0 < self.pain_state_prevalence < 1.0:
            raise ValueError(
                "pain_state_prevalence must be strictly between 0 and 1, "
                "so both pain states actually occur in the simulated "
                f"data; got {self.pain_state_prevalence}."
            )
        if self.temporal_misalignment_minutes < 0:
            raise ValueError("temporal_misalignment_minutes cannot be negative.")

    @property
    def n_observations_planned(self) -> int:
        """Total planned observations, before adherence is applied."""

        return self.n_participants * self.observations_per_day * self.duration_days
