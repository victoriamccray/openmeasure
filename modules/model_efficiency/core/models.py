"""
Model profiles - the domain-agnostic unit this module compares.

A ModelProfile pairs one predictive-performance measurement with one
resource-use measurement for a single model. Nothing here is specific to
any domain: the worked example (GAIA, a medical-imaging knowledge
distillation study) supplies real ModelProfile values, but any set of
models compared on one performance metric and one resource metric can be
run through this module's functions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    """
    One model's reported performance and resource use.

    Both performance_value and resource_value are assumed lower-is-better
    (e.g. an error metric, and an energy/CO2/cost metric). If a source
    reports a higher-is-better performance metric (e.g. accuracy), convert
    it to a lower-is-better form (e.g. 1 - accuracy, or error rate) before
    constructing a ModelProfile, since this module never assumes a metric's
    direction beyond "lower is better".

    performance_is_approximate and resource_is_approximate mark whether the
    value was read directly from a source's stated number, or estimated
    (e.g. visually read from a published figure that gives no exact value).
    A page rendering a ModelProfile must surface this distinction, never
    presenting an approximate value as if it were exact.
    """

    name: str
    performance_metric_name: str
    performance_value: float
    performance_is_approximate: bool
    n_parameters: int
    resource_metric_name: str
    resource_value: float
    resource_is_approximate: bool
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name cannot be empty.")

        if not self.performance_metric_name:
            raise ValueError("performance_metric_name cannot be empty.")

        if self.performance_value < 0:
            raise ValueError(
                f"performance_value cannot be negative; got {self.performance_value}."
            )

        if not self.resource_metric_name:
            raise ValueError("resource_metric_name cannot be empty.")

        if self.resource_value < 0:
            raise ValueError(
                f"resource_value cannot be negative; got {self.resource_value}."
            )

        if self.n_parameters <= 0:
            raise ValueError(
                f"n_parameters must be positive; got {self.n_parameters}."
            )
