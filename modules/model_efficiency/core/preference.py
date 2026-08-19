"""
Weighted comparison across performance and resource use.

This is deliberately not a "best model" verdict: a weighted score is one
way to combine two incommensurable quantities into a single number, and
that combination is only ever as meaningful as the weight behind it. The
weight is supplied by whoever is using the comparison, not by this
module, so the result is named after the weights that produced it
("favored by these weights"), never "preferred" or "recommended" - this
module has no opinion of its own on how performance and resource use
should be traded off.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import ModelProfile


@dataclass(frozen=True)
class PreferenceResult:
    """
    Which model a given performance/resource weighting favors.

    scores are min-max-normalized within the given profiles, weighted, and
    summed; lower is more favored. Because normalization is relative to
    whichever profiles were passed in, adding or removing a model changes
    every score, not just the new one.
    """

    scores: dict[str, float]
    favored_by_weights: str
    performance_weight: float
    resource_weight: float


def _min_max_normalize(values: dict[str, float]) -> dict[str, float]:
    lowest = min(values.values())
    highest = max(values.values())
    if highest == lowest:
        raise ValueError(
            "Cannot normalize when every profile reports the same value "
            f"({lowest}) for this metric."
        )
    return {name: (value - lowest) / (highest - lowest) for name, value in values.items()}


def rank_by_preference(
    profiles: tuple[ModelProfile, ...], performance_weight: float
) -> PreferenceResult:
    """
    Combine performance and resource use into one weighted score per model,
    given a performance_weight in [0, 1] (the rest is resource weight).
    """

    if not profiles:
        raise ValueError("profiles cannot be empty.")

    if not (0.0 <= performance_weight <= 1.0):
        raise ValueError(
            f"performance_weight must be between 0 and 1; got {performance_weight}."
        )

    resource_weight = 1.0 - performance_weight

    performance_norm = _min_max_normalize(
        {p.name: p.performance_value for p in profiles}
    )
    resource_norm = _min_max_normalize({p.name: p.resource_value for p in profiles})

    scores = {
        p.name: performance_weight * performance_norm[p.name]
        + resource_weight * resource_norm[p.name]
        for p in profiles
    }

    favored_by_weights = min(scores, key=scores.get)

    return PreferenceResult(
        scores=scores,
        favored_by_weights=favored_by_weights,
        performance_weight=performance_weight,
        resource_weight=resource_weight,
    )
