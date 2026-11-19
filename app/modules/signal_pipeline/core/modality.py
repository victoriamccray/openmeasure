"""
Modality profiles - the domain-agnostic unit this module compares.

A Modality pairs an illustrative interpretive-gain rating with three
illustrative cost ratings (privacy, security, agency) for one category of
physiological, behavioral, or institutional signal. Unlike GAIA's
ModelProfile (modules/model_efficiency/core/models.py), which reports a
single study's own measured values, every rating here is an illustrative,
author-assigned score informed by the cited literature's directional
findings - none of the four numeric ratings is a value any cited source
reports directly, and a citation here must never be read as "this paper
found this exact number." The worked example's page and README state
this once, prominently, rather than repeating a per-field flag on every
Modality the way ModelProfile's is_approximate flags do.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_RATING = 0.0
MAX_RATING = 1.0


@dataclass(frozen=True)
class Modality:
    """
    One category of signal considered for a multimodal pipeline.

    interpretive_gain is higher-is-better (0 = adds nothing beyond
    signals already in the pipeline, 1 = substantial independent
    information). privacy_cost, security_cost, and agency_cost are each
    higher-is-worse (0 = negligible added cost, 1 = severe added cost)
    and are kept as three separate ratings rather than one "risk" score,
    since a signal can be high on one and low on another - self-report is
    low-privacy-cost but can carry real agency cost if the prompt to
    respond is not genuinely optional.

    is_body_sensed distinguishes a signal actually measured at a body
    region (Neural, Autonomic, Muscular, Behavioral) from one that is
    not (Subjective, Environmental, Institutional): a self-report, an
    ambient location trace, or a clinician's note is about the person,
    but is not picked up from their body by a sensor, and the worked
    example's anatomy figure must not blur that distinction by drawing a
    body-region dot for a signal that was never sensed from the body.
    body_region is only meaningful when is_body_sensed is True.
    privacy_exit_point and agency_control_point are plain-language,
    non-numeric facts about this modality - not additional illustrative
    ratings - describing where the signal physically or practically
    leaves the person, and where the person retains or loses the
    ability to understand, authorize, override, or stop its collection.
    """

    name: str
    category: str
    signal_examples: str
    interpretive_gain: float
    privacy_cost: float
    security_cost: float
    agency_cost: float
    citation: str
    is_body_sensed: bool
    body_region: str
    privacy_exit_point: str
    agency_control_point: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name cannot be empty.")

        if not self.category:
            raise ValueError("category cannot be empty.")

        if not self.citation:
            raise ValueError(f"{self.name}: citation cannot be empty.")

        if self.is_body_sensed and not self.body_region:
            raise ValueError(
                f"{self.name}: body_region cannot be empty when is_body_sensed is True."
            )

        if not self.privacy_exit_point:
            raise ValueError(f"{self.name}: privacy_exit_point cannot be empty.")

        if not self.agency_control_point:
            raise ValueError(f"{self.name}: agency_control_point cannot be empty.")

        for field_name, value in (
            ("interpretive_gain", self.interpretive_gain),
            ("privacy_cost", self.privacy_cost),
            ("security_cost", self.security_cost),
            ("agency_cost", self.agency_cost),
        ):
            if not (MIN_RATING <= value <= MAX_RATING):
                raise ValueError(
                    f"{self.name}: {field_name} must be between "
                    f"{MIN_RATING} and {MAX_RATING}; got {value}."
                )
