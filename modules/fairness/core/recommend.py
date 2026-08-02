"""
Recommend fairness metrics based on the user's evaluation goal.

Recommendations are guides rather than declarations of the single correct
definition of fairness. The UI should show the metric, rationale,
assumptions, tradeoffs, and reasonable alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FairnessRecommendation:
    metric: str
    display_name: str
    reasoning: list[str]
    assumptions: list[str] = field(default_factory=list)
    tradeoffs: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)


FAIRNESS_GOALS = {
    "opportunity_access": FairnessRecommendation(
        metric="demographic_parity",
        display_name="Demographic parity",
        reasoning=[
            "This goal focuses on whether groups receive favorable decisions "
            "at similar rates."
        ],
        assumptions=[
            "A favorable prediction represents access to an opportunity.",
            "Differences in underlying outcome prevalence do not justify "
            "different selection rates for this use case.",
        ],
        tradeoffs=[
            "Equal selection rates can conflict with calibration or equal "
            "error rates when base rates differ.",
            "It does not consider whether predictions are correct.",
        ],
        alternatives=[
            "Equal opportunity when missing qualified people is the main concern.",
            "Calibration when comparable risk interpretation is the main concern.",
        ],
    ),
    "avoid_missed_need": FairnessRecommendation(
        metric="equal_opportunity",
        display_name="Equal opportunity",
        reasoning=[
            "This goal prioritizes equal true-positive rates across groups, "
            "so people who genuinely need or qualify for an intervention are "
            "equally likely to be identified."
        ],
        assumptions=[
            "The positive ground-truth label is meaningful and measured comparably.",
            "False negatives are the primary harm.",
        ],
        tradeoffs=[
            "False-positive rates may still differ.",
            "The metric depends on the quality of the ground-truth label.",
        ],
        alternatives=[
            "Equalized odds when both false negatives and false positives matter.",
            "Predictive equality when false positives are the primary harm.",
        ],
    ),
    "avoid_wrong_flags": FairnessRecommendation(
        metric="predictive_equality",
        display_name="Predictive equality",
        reasoning=[
            "This goal prioritizes equal false-positive rates across groups, "
            "so people are not wrongly flagged at different rates."
        ],
        assumptions=[
            "False positives are the primary harm.",
            "The negative ground-truth label is meaningful and comparable.",
        ],
        tradeoffs=[
            "True-positive rates may still differ.",
            "Reducing false-positive disparities can affect sensitivity.",
        ],
        alternatives=[
            "Equal opportunity when false negatives are the primary concern.",
            "Equalized odds when both error types matter.",
        ],
    ),
    "comparable_risk_scores": FairnessRecommendation(
        metric="calibration",
        display_name="Calibration within groups",
        reasoning=[
            "This goal asks whether the same predicted risk has the same "
            "observed meaning across groups."
        ],
        assumptions=[
            "The model produces probabilities or risk scores.",
            "Observed outcomes are measured consistently across groups.",
        ],
        tradeoffs=[
            "Calibration can conflict with equalized error rates when outcome "
            "base rates differ and prediction is imperfect."
        ],
        alternatives=[
            "Equal opportunity when access for qualified individuals matters most.",
            "Predictive equality when wrongful flags matter most.",
        ],
    ),
}


def recommend_fairness_metric(goal: str) -> FairnessRecommendation:
    """Return the metric recommendation associated with a stated goal."""
    if goal not in FAIRNESS_GOALS:
        valid = ", ".join(sorted(FAIRNESS_GOALS))
        raise ValueError(f"Unknown fairness goal '{goal}'. Valid goals: {valid}.")

    return FAIRNESS_GOALS[goal]
