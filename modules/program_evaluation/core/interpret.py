"""
Plain-language readings of Impact Evaluation results.

Per CLAUDE.md, interpretive logic (thresholds, and the plain-language
reading of a statistic) belongs beside the statistic it explains rather
than written page-locally. This module currently covers the
difference-in-differences design only; the other designs' interpretation
text still lives in pages/2_Impact_Evaluation.py and has deliberately not
been moved here yet.

Nothing here decides anything for the reader. Each function returns what
a result does and does not support, together with the assumptions that
would have to hold for the causal reading to be the right one, so that
the assumptions arrive attached to the estimate instead of as a footnote
underneath it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .did import DiDResult

# The conventional significance threshold, named rather than hardcoded at
# each use, and treated throughout as a convention rather than a fact
# about the world (docs/design-standards.md section 5).
CONVENTIONAL_ALPHA = 0.05

# What the data at hand can say about an assumption. A closed set,
# because the three are genuinely different situations for a reader: one
# is settled by how the data is shaped, one could be examined with data
# this design does not include, and one is a question about the setting
# that no dataset answers on its own.
CHECK_BY_DESIGN = "Held by how these data are structured"
CHECK_NOT_TESTABLE_HERE = "Not testable with two time points"
CHECK_FROM_CONTEXT = "Answered by knowledge of the setting, not by these data"


@dataclass(frozen=True)
class Assumption:
    """One condition a causal reading of an estimate rests on."""

    name: str
    statement: str
    checkable: str
    citation: str

    def __post_init__(self) -> None:
        for field_name in ("name", "statement", "citation"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"{self.name or 'An assumption'} is missing a value for "
                    f"'{field_name}'."
                )

        if self.checkable not in (
            CHECK_BY_DESIGN,
            CHECK_NOT_TESTABLE_HERE,
            CHECK_FROM_CONTEXT,
        ):
            raise ValueError(
                f"{self.name} has checkable '{self.checkable}', which is not "
                "one of the declared states."
            )


@dataclass(frozen=True)
class DiDReading:
    """A plain-language reading of one difference-in-differences estimate."""

    estimand: str
    headline: str
    supports: tuple[str, ...]
    does_not_support: tuple[str, ...]
    observations: tuple[str, ...]


def did_assumptions() -> tuple[Assumption, ...]:
    """
    The conditions a causal reading of a 2x2 DiD estimate rests on.

    Fixed rather than derived from a result, because none of them are
    established or refuted by the numbers: the point of listing them is
    that the estimate is the same arithmetic whether they hold or not.

    Parallel trends is listed first because it is the one that does the
    causal work, and the one a two-period design cannot examine.
    """
    return (
        Assumption(
            name="Parallel trends",
            statement=(
                "Without the program, the treated group's outcome would "
                "have changed by the same amount the comparison group's "
                "did. This is what the comparison group is standing in "
                "for, and it is an assumption about something that did not "
                "happen."
            ),
            checkable=CHECK_NOT_TESTABLE_HERE,
            citation=(
                "Angrist, J. D., & Pischke, J.-S. (2009). Mostly Harmless "
                "Econometrics, ch. 5."
            ),
        ),
        Assumption(
            name="No anticipation",
            statement=(
                "Nobody changed behavior before the program started "
                "because they expected it. If they did, the baseline "
                "already contains part of the effect and the estimate is "
                "measuring what is left."
            ),
            checkable=CHECK_FROM_CONTEXT,
            citation=(
                "Abadie, A. (2005). Semiparametric difference-in-differences "
                "estimators. Review of Economic Studies, 72(1), 1-19."
            ),
        ),
        Assumption(
            name="No spillover between groups",
            statement=(
                "The program affected the treated group only. If it also "
                "reached the comparison group, or displaced activity into "
                "it, the comparison group's change is contaminated and the "
                "difference between the two understates or overstates the "
                "effect."
            ),
            checkable=CHECK_FROM_CONTEXT,
            citation=(
                "Rubin, D. B. (1980). Comment on Basu, D. Journal of the "
                "American Statistical Association, 75(371), 591-593 (the "
                "stable unit treatment value assumption)."
            ),
        ),
        Assumption(
            name="Stable composition",
            statement=(
                "The same units are measured before and after, so a change "
                "in the group average is a change in those units rather "
                "than a change in who is being averaged."
            ),
            checkable=CHECK_BY_DESIGN,
            citation=(
                "Shadish, W. R., Cook, T. D., & Campbell, D. T. (2002). "
                "Experimental and Quasi-Experimental Designs for "
                "Generalized Causal Inference (attrition and "
                "selection-maturation as threats to internal validity)."
            ),
        ),
        Assumption(
            name="Comparison group stayed untreated",
            statement=(
                "The comparison group did not receive this program, or "
                "another program aimed at the same outcome, between the two "
                "measurements."
            ),
            checkable=CHECK_FROM_CONTEXT,
            citation=(
                "Shadish, W. R., Cook, T. D., & Campbell, D. T. (2002). "
                "Experimental and Quasi-Experimental Designs for "
                "Generalized Causal Inference (history as a threat to "
                "internal validity)."
            ),
        ),
    )


def interpret_did(
    result: DiDResult,
    *,
    alpha: float = CONVENTIONAL_ALPHA,
) -> DiDReading:
    """
    Read one DiDResult in plain language, conservatively.

    The estimand is named explicitly. A 2x2 DiD identifies the average
    effect on the units that were treated, under the assumptions above,
    and says nothing about what the program would have done for the
    comparison group.

    ``alpha`` is a convention rather than a property of the data
    (docs/design-standards.md section 5), so the headline names the
    threshold it used instead of reporting significance as a bare fact.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")

    detected = result.p_value < alpha
    direction = "increase" if result.did_estimate > 0 else "decrease"

    estimand = (
        f"The average effect on the {result.treated_label} group, the units "
        "that were actually treated. It does not estimate what the program "
        f"would have done for the {result.comparison_label} group."
    )

    if detected:
        headline = (
            f"The {result.treated_label} group changed by "
            f"{result.change_treated:.2f} and the {result.comparison_label} "
            f"group by {result.change_comparison:.2f}, a difference of "
            f"{result.did_estimate:.2f} "
            f"(95% CI {result.ci_95_low:.2f} to {result.ci_95_high:.2f}). "
            f"That difference is larger than sampling variation alone would "
            f"usually produce at the conventional threshold of {alpha}."
        )
        supports = (
            f"An {direction} of about {result.did_estimate:.2f} in the "
            f"{result.treated_label} group beyond the change the "
            f"{result.comparison_label} group saw over the same period.",
            "Ruling out any explanation that would have moved both groups "
            "equally, such as a seasonal pattern, a policy change reaching "
            "both, or measurement drift.",
        )
    else:
        headline = (
            f"The {result.treated_label} group changed by "
            f"{result.change_treated:.2f} and the {result.comparison_label} "
            f"group by {result.change_comparison:.2f}, a difference of "
            f"{result.did_estimate:.2f} "
            f"(95% CI {result.ci_95_low:.2f} to {result.ci_95_high:.2f}). "
            f"At the conventional threshold of {alpha}, that difference is "
            "within what sampling variation alone would produce."
        )
        supports = (
            "A statement that no difference in change was detected at this "
            "sample size, which is a different claim from the two groups "
            "having changed by the same amount.",
            f"The interval, {result.ci_95_low:.2f} to "
            f"{result.ci_95_high:.2f}, as the range of effects these data "
            "are compatible with.",
        )

    does_not_support = (
        "A causal claim on its own. The estimate is causal only if parallel "
        "trends holds, and two time points give no way to examine whether "
        "the groups were moving together beforehand.",
        f"A claim about units outside the {result.treated_label} group. "
        "Whoever was treated here is who the estimate is about.",
        "A claim about a different period. The estimate covers the interval "
        "between these two measurements only.",
    )

    observations = _did_observations(result)

    return DiDReading(
        estimand=estimand,
        headline=headline,
        supports=supports,
        does_not_support=does_not_support,
        observations=observations,
    )


def _did_observations(result: DiDResult) -> tuple[str, ...]:
    """
    Things worth noticing about this particular estimate.

    Each is a fact about the numbers rather than a verdict on them: a
    large baseline gap does not invalidate a DiD estimate, and a
    comparison group that moved a lot does not either. Both change how
    much weight parallel trends is carrying, which is something the
    reader should decide with knowledge of the setting.
    """
    observations: list[str] = []

    naive_pre_post = result.change_treated
    if abs(naive_pre_post) > 0 and abs(result.change_comparison) > abs(naive_pre_post) / 2:
        observations.append(
            f"The comparison group moved {result.change_comparison:.2f} on "
            "its own, a substantial share of the treated group's "
            f"{naive_pre_post:.2f}. A pre/post comparison of the treated "
            f"group alone would have reported {naive_pre_post:.2f} as the "
            "effect."
        )

    if abs(result.baseline_difference) > 0:
        observations.append(
            f"The two groups differed by {result.baseline_difference:.2f} at "
            "baseline. DiD does not require them to start level, only to "
            "have been moving in parallel, though groups that start far "
            "apart are often ones with different trajectories."
        )

    if result.small_groups_flagged:
        observations.append(
            "Small group(s) flagged: "
            + "; ".join(result.small_groups_flagged)
            + ". The interval and p-value rest on little information."
        )

    if result.n_excluded_rows:
        observations.append(
            f"{result.n_excluded_rows} of {result.n_input_rows} rows were "
            f"excluded ({result.exclusion_reason}). If units dropped out in "
            "a way related to the program, the remaining units are no "
            "longer the group the estimate is meant to describe."
        )

    return tuple(observations)
