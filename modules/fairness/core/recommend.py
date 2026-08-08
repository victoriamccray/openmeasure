"""
Recommend fairness metrics based on the user's evaluation goal.

Recommendations are guides rather than declarations of the single correct
definition of fairness. The UI should show the metric, rationale,
assumptions, tradeoffs, and reasonable alternatives.

applicable_domains names illustrative, non-exhaustive contexts where a
goal can become relevant. It is not a domain-to-metric lookup table: the
same domain can raise several different fairness goals depending on the
specific decision being evaluated, and appearing here is not a claim that
a domain requires this goal's metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainContext:
    """
    One domain or context where a fairness goal can become relevant.

    Two fields only, by design: domain names the setting, relevance says
    why the goal can matter there. A domain's presence here is illustrative
    of where the concern can come up, not a claim that the domain requires
    this goal's metric specifically, still less that it requires no other.
    Kept as a dataclass rather than a bare string so it has somewhere to
    grow if domain-specific validation guidance is added later; no such
    guidance is attached yet, and none should be inferred from a domain
    simply appearing in this list.
    """

    domain: str
    relevance: str


@dataclass(frozen=True)
class FairnessRecommendation:
    metric: str
    display_name: str
    reasoning: list[str]
    assumptions: list[str] = field(default_factory=list)
    tradeoffs: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    applicable_domains: list[DomainContext] = field(default_factory=list)


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
        applicable_domains=[
            DomainContext(
                domain="Employment",
                relevance=(
                    "Hiring or promotion screening often treats equal access "
                    "to being considered as a policy goal in its own right, "
                    "separate from eventual hiring outcomes."
                ),
            ),
            DomainContext(
                domain="Public-resource allocation",
                relevance=(
                    "Eligibility screening for public benefits or services "
                    "can carry an explicit policy goal of equal rates of "
                    "being offered access."
                ),
            ),
            DomainContext(
                domain="Credit/economics",
                relevance=(
                    "Loan pre-qualification or marketing outreach can raise "
                    "questions about equal rates of being invited to apply, "
                    "independent of eventual approval rates."
                ),
            ),
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
        applicable_domains=[
            DomainContext(
                domain="Healthcare",
                relevance=(
                    "Risk-screening tools meant to flag patients needing "
                    "further care can raise this concern, since a missed "
                    "case can mean a missed or delayed diagnosis."
                ),
            ),
            DomainContext(
                domain="Education",
                relevance=(
                    "Identifying students who qualify for additional support "
                    "services raises this concern when under-identification "
                    "means support never reaches a student who needed it."
                ),
            ),
            DomainContext(
                domain="Public-resource allocation",
                relevance=(
                    "Needs-based benefit programs raise this concern when "
                    "failing to identify an eligible recipient withholds a "
                    "benefit they qualify for."
                ),
            ),
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
        applicable_domains=[
            DomainContext(
                domain="Law enforcement",
                relevance=(
                    "Risk-assessment tools used in pretrial or sentencing "
                    "contexts raise this concern, since a false flag there "
                    "can mean unwarranted scrutiny or restricted liberty."
                ),
            ),
            DomainContext(
                domain="Employment",
                relevance=(
                    "Background-screening or fraud-detection tools raise "
                    "this concern when a false flag can unfairly cost "
                    "someone an opportunity."
                ),
            ),
            DomainContext(
                domain="Healthcare",
                relevance=(
                    "Screening tools that trigger further testing raise "
                    "this concern when a false flag imposes unnecessary, "
                    "sometimes invasive or costly, follow-up."
                ),
            ),
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
        applicable_domains=[
            DomainContext(
                domain="Credit/economics",
                relevance=(
                    "Credit-scoring models raise this concern, since a given "
                    "score is expected to imply comparable default risk "
                    "regardless of group."
                ),
            ),
            DomainContext(
                domain="Healthcare",
                relevance=(
                    "Clinical risk scores used to prioritize treatment raise "
                    "this concern, since a given score is expected to imply "
                    "comparable clinical risk across patient groups."
                ),
            ),
            DomainContext(
                domain="Law enforcement",
                relevance=(
                    "Risk scores used to inform decisions raise this "
                    "concern, since the same score implying different "
                    "real-world risk across groups would undermine what "
                    "the score is supposed to mean."
                ),
            ),
        ],
    ),
}


def recommend_fairness_metric(goal: str) -> FairnessRecommendation:
    """Return the metric recommendation associated with a stated goal."""
    if goal not in FAIRNESS_GOALS:
        valid = ", ".join(sorted(FAIRNESS_GOALS))
        raise ValueError(f"Unknown fairness goal '{goal}'. Valid goals: {valid}.")

    return FAIRNESS_GOALS[goal]
