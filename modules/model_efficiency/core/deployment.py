"""
Deployment-scale projection - how a per-unit resource difference
accumulates over many uses.

Takes an already-known per-unit difference (e.g. a source's own reported
resource saving per deployment) rather than deriving one from two
ModelProfile values, since a per-unit difference is sometimes reported
directly by a source without the two absolute values it was computed
from being reported anywhere. Assumes the difference scales linearly with
deployment count; that assumption should be attributed to whichever
source's figure is being projected, not treated as a fact this module
establishes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeploymentProjection:
    """A per-unit resource difference, scaled to a chosen deployment count."""

    resource_metric_name: str
    per_unit_difference: float
    n_deployments: int
    projected_difference: float


def project_deployment_savings(
    per_unit_difference: float, n_deployments: int, resource_metric_name: str
) -> DeploymentProjection:
    """Scale a per-unit resource difference to n_deployments, linearly."""

    if n_deployments <= 0:
        raise ValueError(f"n_deployments must be positive; got {n_deployments}.")

    if not resource_metric_name:
        raise ValueError("resource_metric_name cannot be empty.")

    return DeploymentProjection(
        resource_metric_name=resource_metric_name,
        per_unit_difference=per_unit_difference,
        n_deployments=n_deployments,
        projected_difference=per_unit_difference * n_deployments,
    )
