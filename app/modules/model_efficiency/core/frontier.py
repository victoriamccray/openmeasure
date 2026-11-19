"""
Performance-efficiency frontier - which models are not strictly beaten.

A model is on the frontier ("efficient" in the Pareto sense) if no other
model in the comparison set matches or beats it on both performance and
resource use, with at least one dimension strictly better. A dominated
model is one some other model beats outright on both axes at once - never
a defensible choice regardless of how performance and resource use are
weighted against each other.

The dominance check itself lives in shared/pareto.py, shared with
modules/signal_pipeline's gain-vs-cost frontier; this module keeps its
own ModelProfile-specific FrontierResult and vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.pareto import ParetoPoint, compute_pareto_efficiency

from .models import ModelProfile


@dataclass(frozen=True)
class FrontierResult:
    """Which models in a comparison set are not dominated by another."""

    profiles: tuple[ModelProfile, ...]
    is_efficient: dict[str, bool]


def compute_frontier(profiles: tuple[ModelProfile, ...]) -> FrontierResult:
    """Determine which profiles are not dominated by any other profile."""

    if not profiles:
        raise ValueError("profiles cannot be empty.")

    points = tuple(
        ParetoPoint(
            name=profile.name,
            x_value=profile.performance_value,
            y_value=profile.resource_value,
        )
        for profile in profiles
    )

    return FrontierResult(
        profiles=profiles, is_efficient=compute_pareto_efficiency(points)
    )
