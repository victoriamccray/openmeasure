"""
Performance-efficiency frontier - which models are not strictly beaten.

A model is on the frontier ("efficient" in the Pareto sense) if no other
model in the comparison set matches or beats it on both performance and
resource use, with at least one dimension strictly better. A dominated
model is one some other model beats outright on both axes at once - never
a defensible choice regardless of how performance and resource use are
weighted against each other.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    is_efficient: dict[str, bool] = {}

    for candidate in profiles:
        dominated = False
        for other in profiles:
            if other is candidate:
                continue
            not_worse_on_both = (
                other.performance_value <= candidate.performance_value
                and other.resource_value <= candidate.resource_value
            )
            strictly_better_on_one = (
                other.performance_value < candidate.performance_value
                or other.resource_value < candidate.resource_value
            )
            if not_worse_on_both and strictly_better_on_one:
                dominated = True
                break
        is_efficient[candidate.name] = not dominated

    return FrontierResult(profiles=profiles, is_efficient=is_efficient)
