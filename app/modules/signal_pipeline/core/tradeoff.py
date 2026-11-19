"""
Weighted cost combination and gain-vs-cost frontier for a set of modalities.

Mirrors modules/model_efficiency/core/preference.py and frontier.py: a
weighted combination of privacy, security, and agency cost is only ever
as meaningful as the weights behind it, so the result is named after
those weights ("combined cost under these weights"), never treated as a
single "risk score" this module asserts on its own. The gain-vs-cost
frontier then uses that weighted cost as one axis and shared/pareto.py's
dominance rule as its computation - the same rule modules/model_efficiency
uses for its own performance-vs-resource frontier.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.pareto import ParetoPoint, compute_pareto_efficiency

from .modality import Modality


@dataclass(frozen=True)
class WeightedCostResult:
    """Each modality's privacy/security/agency cost, combined under one set of weights."""

    combined_cost: dict[str, float]
    privacy_weight: float
    security_weight: float
    agency_weight: float


def combine_costs(
    modalities: tuple[Modality, ...],
    privacy_weight: float,
    security_weight: float,
    agency_weight: float,
) -> WeightedCostResult:
    """
    Combine privacy/security/agency cost into one weighted score per
    modality. Weights must be non-negative and are normalized to sum to
    1 here, so three independently set sliders never need to be made to
    add up exactly by the caller.
    """

    if not modalities:
        raise ValueError("modalities cannot be empty.")

    raw_weights = (privacy_weight, security_weight, agency_weight)

    if any(weight < 0 for weight in raw_weights):
        raise ValueError(f"weights must be non-negative; got {raw_weights}.")

    total = sum(raw_weights)
    if total <= 0:
        raise ValueError("At least one weight must be positive.")

    privacy_weight, security_weight, agency_weight = (
        weight / total for weight in raw_weights
    )

    combined_cost = {
        modality.name: (
            privacy_weight * modality.privacy_cost
            + security_weight * modality.security_cost
            + agency_weight * modality.agency_cost
        )
        for modality in modalities
    }

    return WeightedCostResult(
        combined_cost=combined_cost,
        privacy_weight=privacy_weight,
        security_weight=security_weight,
        agency_weight=agency_weight,
    )


@dataclass(frozen=True)
class GainCostFrontierResult:
    """Which modalities are not dominated on gain versus weighted cost."""

    modalities: tuple[Modality, ...]
    is_efficient: dict[str, bool]


def compute_gain_cost_frontier(
    modalities: tuple[Modality, ...], weighted_cost: WeightedCostResult
) -> GainCostFrontierResult:
    """
    A modality is on the frontier if no other modality in the set both
    costs no more (under these weights) and gains no less, with at least
    one dimension strictly better. interpretive_gain is higher-is-better
    while combined_cost is lower-is-better, so gain is negated before the
    shared lower-is-better dominance check runs.
    """

    if not modalities:
        raise ValueError("modalities cannot be empty.")

    points = tuple(
        ParetoPoint(
            name=modality.name,
            x_value=weighted_cost.combined_cost[modality.name],
            y_value=-modality.interpretive_gain,
        )
        for modality in modalities
    )

    return GainCostFrontierResult(
        modalities=modalities,
        is_efficient=compute_pareto_efficiency(points),
    )
