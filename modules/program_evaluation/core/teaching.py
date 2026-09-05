"""
A small worked scenario that shows what a comparison group does to an
impact estimate.

The scenario is fixed and the arithmetic is hand-checkable on purpose.
One number is adjustable, the comparison group's change over the same
period, because that single number is what separates a
difference-in-differences estimate from a before-and-after one: the
before-and-after estimate is what DiD reduces to when the comparison
group is assumed not to have moved at all.

Pure like the rest of core/. The page renders the sequence (scenario,
question, method, assumptions, result, what can and cannot be concluded);
everything it renders is defined here, so the teaching text and the
teaching arithmetic cannot drift apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TeachingScenario:
    """One fixed scenario, with the treated group's numbers already set."""

    outcome_label: str
    treated_label: str
    comparison_label: str
    scenario: str
    question: str
    method_fit: str
    assumptions: tuple[str, ...]
    can_conclude: tuple[str, ...]
    cannot_conclude: tuple[str, ...]

    pre_treated: float
    post_treated: float
    pre_comparison: float

    @property
    def change_treated(self) -> float:
        """How much the treated group moved. Fixed across the whole example."""
        return self.post_treated - self.pre_treated


@dataclass(frozen=True)
class TeachingOutcome:
    """The two estimates the scenario produces at one comparison-group change."""

    comparison_change: float
    post_comparison: float
    change_treated: float

    # What a pre/post comparison of the treated group alone would report.
    # Constant across every value of comparison_change, which is the
    # point of showing it beside the DiD estimate.
    before_after_estimate: float

    did_estimate: float
    reading: str


DID_TEACHING_SCENARIO = TeachingScenario(
    outcome_label="percentage of scheduled appointments kept",
    treated_label="Clinic A",
    comparison_label="Clinic B",
    scenario=(
        "Two community clinics in the same city serve similar patients. In "
        "January, Clinic A starts sending text-message appointment "
        "reminders. Clinic B does not. Both clinics record the percentage "
        "of scheduled appointments patients actually keep, in December "
        "before the change and in June after it. Clinic A goes from 60% to "
        "72%."
    ),
    question=(
        "Did the reminder texts raise the share of appointments kept at "
        "Clinic A, or would that share have risen anyway?"
    ),
    method_fit=(
        "Clinic A's 12-point rise is the whole story only if nothing else "
        "changed over those six months. Clinic B went through the same six "
        "months without the reminders, so whatever moved kept-appointment "
        "rates city-wide should show up in Clinic B's numbers too. "
        "Difference-in-differences subtracts Clinic B's change from Clinic "
        "A's, leaving the part of Clinic A's rise that Clinic B did not "
        "also experience."
    ),
    assumptions=(
        "Without the reminders, Clinic A's kept-appointment rate would have "
        "changed by the same amount Clinic B's did.",
        "Clinic B did not start its own reminder program, or anything else "
        "aimed at attendance, during the same six months.",
        "Clinic A's patients did not change their behavior in December in "
        "anticipation of the January launch.",
        "Patients did not move between the two clinics because of the "
        "reminders.",
    ),
    can_conclude=(
        "How much Clinic A's rate moved relative to Clinic B's over the "
        "same six months.",
        "That anything affecting both clinics equally, a winter illness "
        "season or a city-wide transit change, is not the explanation for "
        "the difference between them.",
    ),
    cannot_conclude=(
        "That the reminders caused the difference, unless Clinic B really "
        "was on the path Clinic A would have followed. Two measurements "
        "give no way to check whether the clinics were moving together "
        "before January.",
        "That the same program would work at Clinic B, or at a clinic "
        "serving different patients.",
        "That the effect would hold beyond June, or grow if the program "
        "continued.",
    ),
    pre_treated=60.0,
    post_treated=72.0,
    pre_comparison=58.0,
)


def teaching_did(
    comparison_change: float,
    *,
    scenario: TeachingScenario = DID_TEACHING_SCENARIO,
) -> TeachingOutcome:
    """
    Recompute the scenario's two estimates for one comparison-group change.

    ``comparison_change`` is how much the comparison group's outcome moved
    over the same period, in the outcome's own units.

    Raises
    ------
    TypeError
        If ``comparison_change`` is not a real number.
    ValueError
        If it is not finite, which would make both estimates undefined
        rather than merely large.
    """
    if isinstance(comparison_change, bool) or not isinstance(
        comparison_change, (int, float)
    ):
        raise TypeError(
            "comparison_change must be a real number, got "
            f"{type(comparison_change).__name__}."
        )

    if not math.isfinite(comparison_change):
        raise ValueError(
            f"comparison_change must be finite, got {comparison_change}."
        )

    comparison_change = float(comparison_change)
    change_treated = scenario.change_treated
    did_estimate = change_treated - comparison_change

    return TeachingOutcome(
        comparison_change=comparison_change,
        post_comparison=scenario.pre_comparison + comparison_change,
        change_treated=change_treated,
        before_after_estimate=change_treated,
        did_estimate=did_estimate,
        reading=_reading(scenario, comparison_change, did_estimate),
    )


def _reading(
    scenario: TeachingScenario,
    comparison_change: float,
    did_estimate: float,
) -> str:
    """
    State what this pair of numbers shows, in the scenario's own terms.

    Five cases, cut at the two values where the comparison changes
    character: zero (the comparison group held still, so the two
    estimates agree) and the treated group's own change (the comparison
    group moved just as much, so nothing is left over).
    """
    treated = scenario.treated_label
    comparison = scenario.comparison_label
    change_treated = scenario.change_treated

    if comparison_change == 0:
        return (
            f"{comparison} held still, so the two estimates agree at "
            f"{did_estimate:.1f} points. They agree because of an "
            f"assumption, that nothing moved {comparison} over these six "
            "months, rather than because a before-and-after comparison is "
            "sufficient here."
        )

    if comparison_change < 0:
        return (
            f"{comparison} fell {abs(comparison_change):.1f} points while "
            f"{treated} rose {change_treated:.1f}. Against a declining "
            f"backdrop, {treated}'s gain looks larger than its own "
            f"before-and-after number: {did_estimate:.1f} points rather "
            f"than {change_treated:.1f}."
        )

    if comparison_change < change_treated:
        share = comparison_change / change_treated
        return (
            f"{comparison} rose {comparison_change:.1f} points without the "
            f"reminders, which is {share:.0%} of {treated}'s "
            f"{change_treated:.1f}. Subtracting it leaves "
            f"{did_estimate:.1f} points. A before-and-after comparison of "
            f"{treated} alone would still report {change_treated:.1f}."
        )

    if comparison_change == change_treated:
        return (
            f"{comparison} rose by exactly as much as {treated} did, "
            f"{change_treated:.1f} points, with no reminder program at "
            "all. The estimate is 0.0. Every point of the rise is "
            "consistent with something that moved both clinics, and the "
            f"before-and-after figure of {change_treated:.1f} would have "
            "attributed all of it to the reminders."
        )

    return (
        f"{comparison} rose {comparison_change:.1f} points, more than "
        f"{treated}'s {change_treated:.1f}, without any reminder program. "
        f"The estimate turns negative at {did_estimate:.1f}: {treated} "
        "gained less than the clinic it is being compared against, even "
        "though its own numbers went up."
    )
