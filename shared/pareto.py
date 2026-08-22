"""
Generic Pareto-dominance check over two lower-is-better axes.

Extracted here once a second module (modules/signal_pipeline, alongside
modules/model_efficiency) needed the exact same dominance rule, per this
project's own "only extract what is actually duplicated" convention (see
CLAUDE.md). Each module keeps its own domain-specific result dataclass
and vocabulary - a "model" is not a "modality" - only the dominance
check itself, which has no domain content, lives here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParetoPoint:
    """One named item's position on two lower-is-better axes."""

    name: str
    x_value: float
    y_value: float


def compute_pareto_efficiency(points: tuple[ParetoPoint, ...]) -> dict[str, bool]:
    """
    Which points are not dominated by another point in the same set.

    A point is dominated when some other point matches or beats it on
    both axes, with at least one axis strictly better. Both axes are
    assumed lower-is-better; a caller with a higher-is-better axis must
    negate it before constructing a ParetoPoint.
    """

    if not points:
        raise ValueError("points cannot be empty.")

    names = [point.name for point in points]
    if len(names) != len(set(names)):
        raise ValueError("points must have unique names.")

    is_efficient: dict[str, bool] = {}

    for candidate in points:
        dominated = False
        for other in points:
            if other is candidate:
                continue
            not_worse_on_both = (
                other.x_value <= candidate.x_value
                and other.y_value <= candidate.y_value
            )
            strictly_better_on_one = (
                other.x_value < candidate.x_value
                or other.y_value < candidate.y_value
            )
            if not_worse_on_both and strictly_better_on_one:
                dominated = True
                break
        is_efficient[candidate.name] = not dominated

    return is_efficient
