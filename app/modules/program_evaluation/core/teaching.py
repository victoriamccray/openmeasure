"""
A small worked scenario that shows what a comparison group does to an
impact estimate.

The scenario is fixed and the arithmetic is hand-checkable on purpose.
One number is adjustable, the comparison group's change over the same
period, because that single number is what separates a
difference-in-differences estimate from a before-and-after one: the
before-and-after estimate is what DiD reduces to when the comparison
group is assumed not to have moved at all.

The same lesson is told in several fields' terms. Every domain variant
carries identical numbers, deliberately: the clinic, the school, and the
workforce board all start at 60, the treated side reaches 72, and the
comparison side starts at 58. Only the story changes. A test asserts the
estimates come out identical across variants, which is the domain
separation rule (see domains.py) stated as arithmetic rather than as a
caption.

Domains without their own variant fall back to the canonical one rather
than getting a thin restatement of it. A researcher is better served by a
clearly-labeled example from another field than by a version of their own
field written to fill a slot.

Pure like the rest of core/. The page renders the sequence (scenario,
question, method, assumptions, result, what can and cannot be concluded);
everything it renders is defined here, so the teaching text and the
teaching arithmetic cannot drift apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .domains import DOMAIN_IDS, get_domain

# Shared across every domain variant. Named once here rather than typed
# into each scenario, so the variants cannot drift apart by a typo and
# the "same method, same numbers, different field" claim holds by
# construction as well as by test.
_PRE_TREATED = 60.0
_POST_TREATED = 72.0
_PRE_COMPARISON = 58.0


@dataclass(frozen=True)
class TeachingScenario:
    """
    One field's telling of the scenario, with the treated side already set.

    The label fields exist so the page renders a scenario without
    hardcoding any of its words. A page that spelled out "six months" or
    "December" itself would keep saying so under a variant where neither
    is true.
    """

    domain_id: str
    outcome_label: str
    treated_label: str
    comparison_label: str
    program_label: str
    period_label: str
    pre_period_label: str
    post_period_label: str
    unit_label: str

    scenario: str
    question: str
    method_fit: str
    assumptions: tuple[str, ...]
    can_conclude: tuple[str, ...]
    cannot_conclude: tuple[str, ...]

    pre_treated: float
    post_treated: float
    pre_comparison: float

    def __post_init__(self) -> None:
        if self.domain_id not in DOMAIN_IDS:
            raise ValueError(
                f"'{self.domain_id}' is not a known domain. Known domains: "
                f"{', '.join(DOMAIN_IDS)}."
            )

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


_PUBLIC_HEALTH_DID = TeachingScenario(
    domain_id="public_health",
    outcome_label="percentage of scheduled appointments kept",
    treated_label="Clinic A",
    comparison_label="Clinic B",
    program_label="reminder program",
    period_label="six months",
    pre_period_label="December",
    post_period_label="June",
    unit_label="percentage points",
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
    pre_treated=_PRE_TREATED,
    post_treated=_POST_TREATED,
    pre_comparison=_PRE_COMPARISON,
)


_EDUCATION_DID = TeachingScenario(
    domain_id="education",
    outcome_label="percentage of students reading at grade level",
    treated_label="Oak School",
    comparison_label="Pine School",
    program_label="tutoring program",
    period_label="school year",
    pre_period_label="September",
    post_period_label="May",
    unit_label="percentage points",
    scenario=(
        "Two elementary schools in the same district enroll similar "
        "students. In the fall, Oak School starts a small-group reading "
        "tutoring program. Pine School does not. Both schools record the "
        "percentage of third graders reading at grade level, in September "
        "before the program and in May after it. Oak School goes from 60% "
        "to 72%."
    ),
    question=(
        "Did the tutoring raise the share of Oak School's third graders "
        "reading at grade level, or would that share have risen anyway?"
    ),
    method_fit=(
        "Oak School's 12-point rise is the whole story only if nothing "
        "else changed over the school year. Third graders at Pine School "
        "spent the same year growing as readers without the tutoring, so "
        "ordinary progress over a school year, and anything the district "
        "changed for everyone, should show up in Pine School's numbers "
        "too. Difference-in-differences subtracts Pine School's change "
        "from Oak School's, leaving the part of Oak School's rise that "
        "Pine School did not also experience."
    ),
    assumptions=(
        "Without the tutoring, Oak School's reading rate would have changed "
        "by the same amount Pine School's did.",
        "Pine School did not start its own reading initiative, or change "
        "its literacy curriculum, during the same year.",
        "Oak School's teachers did not change instruction in the spring "
        "before the program in anticipation of it.",
        "Families did not move children between the two schools because of "
        "the tutoring.",
    ),
    can_conclude=(
        "How much Oak School's rate moved relative to Pine School's over "
        "the same school year.",
        "That anything affecting both schools equally, a district-wide "
        "curriculum change or a shift in the reading assessment, is not "
        "the explanation for the difference between them.",
    ),
    cannot_conclude=(
        "That the tutoring caused the difference, unless Pine School "
        "really was on the path Oak School would have followed. Two "
        "measurements give no way to check whether the schools were moving "
        "together in earlier years.",
        "That the same program would work at Pine School, or at a school "
        "serving different students.",
        "That the gain would still be there a year later, or grow if the "
        "tutoring continued.",
    ),
    pre_treated=_PRE_TREATED,
    post_treated=_POST_TREATED,
    pre_comparison=_PRE_COMPARISON,
)


_WORKFORCE_DID = TeachingScenario(
    domain_id="workforce",
    outcome_label="percentage of participants placed in a job within six months of exit",
    treated_label="North Region",
    comparison_label="South Region",
    program_label="training program",
    period_label="year",
    pre_period_label="Last year",
    post_period_label="This year",
    unit_label="percentage points",
    scenario=(
        "Two regions of the same state workforce agency serve similar "
        "jobseekers. This year, North Region adds a short technical "
        "training program to its standard services. South Region keeps its "
        "standard services unchanged. Both regions record the percentage "
        "of participants placed in a job within six months of exit, for "
        "last year's cohort and for this year's. North Region goes from "
        "60% to 72%."
    ),
    question=(
        "Did the training raise North Region's placement rate, or would "
        "that rate have risen anyway?"
    ),
    method_fit=(
        "North Region's 12-point rise is the whole story only if nothing "
        "else changed between the two cohorts. South Region served its "
        "cohort in the same labor market without the training, so a "
        "hiring recovery, a seasonal shift, or anything the agency changed "
        "statewide should show up in South Region's numbers too. "
        "Difference-in-differences subtracts South Region's change from "
        "North Region's, leaving the part of North Region's rise that "
        "South Region did not also experience."
    ),
    assumptions=(
        "Without the training, North Region's placement rate would have "
        "changed by the same amount South Region's did.",
        "South Region did not add its own training, or change its "
        "placement services, over the same year.",
        "North Region did not change who it enrolled in anticipation of "
        "the training starting.",
        "Jobseekers did not travel to North Region for the training, "
        "leaving South Region a different population than before.",
    ),
    can_conclude=(
        "How much North Region's placement rate moved relative to South "
        "Region's between the two cohorts.",
        "That anything affecting both regions equally, a statewide hiring "
        "recovery or a change to how placements are recorded, is not the "
        "explanation for the difference between them.",
    ),
    cannot_conclude=(
        "That the training caused the difference, unless South Region "
        "really was on the path North Region would have followed. Two "
        "cohorts give no way to check whether the regions were moving "
        "together before.",
        "That the same program would work in South Region, or for "
        "jobseekers with different backgrounds.",
        "That placements held beyond the six-month mark, which is not what "
        "this outcome measures.",
    ),
    pre_treated=_PRE_TREATED,
    post_treated=_POST_TREATED,
    pre_comparison=_PRE_COMPARISON,
)


# Domains with their own telling. The rest fall back to the canonical one
# below. Public health is both a variant and the fallback, because it is
# the scenario this example was first written as.
_DID_SCENARIOS: dict[str, TeachingScenario] = {
    scenario.domain_id: scenario
    for scenario in (_PUBLIC_HEALTH_DID, _EDUCATION_DID, _WORKFORCE_DID)
}

CANONICAL_DID_SCENARIO = _PUBLIC_HEALTH_DID

# Kept under its original name because the page and existing tests refer
# to it, and it is still exactly what it was: the scenario shown when no
# domain has been chosen.
DID_TEACHING_SCENARIO = CANONICAL_DID_SCENARIO


def did_scenario_for(domain_id: str | None) -> TeachingScenario:
    """
    The difference-in-differences scenario to show for a domain.

    Returns the canonical scenario for None (no domain chosen yet) and
    for any domain without its own telling. An unknown domain id raises,
    since that is a typo rather than a domain awaiting its own variant.

    Keyed by domain alone rather than by (concept, domain), because
    difference-in-differences is the only concept with an interactive
    example today. A second concept would add a sibling registry.
    """
    if domain_id is None:
        return CANONICAL_DID_SCENARIO

    # Raises on an unknown id, which is the check this function wants;
    # the returned domain itself is not needed.
    get_domain(domain_id)

    return _DID_SCENARIOS.get(domain_id, CANONICAL_DID_SCENARIO)


def has_own_did_scenario(domain_id: str) -> bool:
    """
    Whether a domain tells this example in its own terms.

    The page uses this to say so when it is showing another field's
    example, rather than presenting a clinic to someone who selected
    education and letting them wonder.
    """
    get_domain(domain_id)

    return domain_id in _DID_SCENARIOS


def teaching_did(
    comparison_change: float,
    *,
    scenario: TeachingScenario = CANONICAL_DID_SCENARIO,
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
    program = scenario.program_label
    period = scenario.period_label
    units = scenario.unit_label
    change_treated = scenario.change_treated

    if comparison_change == 0:
        return (
            f"{comparison} held still, so the two estimates agree at "
            f"{did_estimate:.1f} {units}. They agree because of an "
            f"assumption, that nothing moved {comparison} over this "
            f"{period}, rather than because a before-and-after comparison "
            "is sufficient here."
        )

    if comparison_change < 0:
        return (
            f"{comparison} fell {abs(comparison_change):.1f} {units} while "
            f"{treated} rose {change_treated:.1f}. Against a declining "
            f"backdrop, {treated}'s gain looks larger than its own "
            f"before-and-after number: {did_estimate:.1f} rather "
            f"than {change_treated:.1f}."
        )

    if comparison_change < change_treated:
        share = comparison_change / change_treated
        return (
            f"{comparison} rose {comparison_change:.1f} {units} without the "
            f"{program}, which is {share:.0%} of {treated}'s "
            f"{change_treated:.1f}. Subtracting it leaves "
            f"{did_estimate:.1f}. A before-and-after comparison of "
            f"{treated} alone would still report {change_treated:.1f}."
        )

    if comparison_change == change_treated:
        return (
            f"{comparison} rose by exactly as much as {treated} did, "
            f"{change_treated:.1f} {units}, with no {program} at all. The "
            "estimate is 0.0. Every point of the rise is consistent with "
            f"something that moved both, and the before-and-after figure "
            f"of {change_treated:.1f} would have attributed all of it to "
            f"the {program}."
        )

    return (
        f"{comparison} rose {comparison_change:.1f} {units}, more than "
        f"{treated}'s {change_treated:.1f}, without any {program}. The "
        f"estimate turns negative at {did_estimate:.1f}: {treated} gained "
        "less than the group it is being compared against, even though its "
        "own numbers went up."
    )
