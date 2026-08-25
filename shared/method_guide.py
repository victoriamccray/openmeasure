"""
Method Selection Decision Tree: one guided research question routing to a
workflow or research journey.

This is a discovery aid, not a workflow. It carries no module_key, is
never passed to shared/handoff.py, and its page is not part of
shared/catalog.py's lifecycle stages, for the same reason
shared/datasets.py's Explore Real Data catalog sits outside them: routing
a user to a destination is not itself an analysis.

The entry question is deliberately phrased as what a researcher is trying
to determine ("Did my program work?"), not as which validation category
or tool they need ("Impact Evaluation"). A researcher who already knows
they need Cronbach's alpha does not need this page; it exists for the
researcher who does not yet know that "several items measuring one thing
consistently" is what reliability means.

Each branch points to exactly one destination, of two kinds:

- workflow: one of shared/catalog.py's numbered analysis workflows, which
  has its own module_key, records to shared/handoff.py, and has its own,
  more specific method-selection logic once you're inside it (for
  example, Impact Evaluation's own recommendation of which statistical
  test fits your data's shape).
- journey: one of shared/research_journeys.py's guided worked examples,
  which has no module_key and records nothing, and instead walks a fixed
  sequence of stages rather than offering further method selection.

A branch carries exactly one of the two so the page always knows which
kind of destination it is pointing to and can describe it honestly,
rather than describing a journey as if it had a workflow's further
decision logic inside it.

This page deliberately stops at one level: duplicating any destination's
own deeper logic up here would be a second, driftable copy of it.

workflow must match a Workflow.workflow value in shared/catalog.py
exactly, and journey must match a ResearchJourney.title value in
shared/research_journeys.py exactly, so a rename in either place cannot
silently orphan a branch.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.catalog import WORKFLOWS
from shared.research_journeys import JOURNEYS

_WORKFLOW_NAMES: frozenset[str] = frozenset(item.workflow for item in WORKFLOWS)
_JOURNEY_TITLES: frozenset[str] = frozenset(item.title for item in JOURNEYS)


@dataclass(frozen=True)
class MethodBranch:
    """One answer to "what are you trying to determine?", and where it leads."""

    id: str
    question: str
    why: str
    youll_learn: str
    limitations: tuple[str, ...]
    workflow: str | None = None
    journey: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("id", "question", "why", "youll_learn"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"{self.id or 'A branch'} is missing a value for "
                    f"'{field_name}'."
                )

        if not self.limitations:
            raise ValueError(f"{self.id} lists no limitations.")

        if bool(self.workflow) == bool(self.journey):
            raise ValueError(
                f"{self.id} must name exactly one of 'workflow' or "
                "'journey', so it always points to exactly one "
                "destination and the page always knows which kind it is."
            )

        if self.workflow is not None and self.workflow not in _WORKFLOW_NAMES:
            raise ValueError(
                f"{self.id} names workflow '{self.workflow}', which does "
                f"not match any workflow in shared/catalog.py. Known "
                f"workflows: {sorted(_WORKFLOW_NAMES)}."
            )

        if self.journey is not None and self.journey not in _JOURNEY_TITLES:
            raise ValueError(
                f"{self.id} names journey '{self.journey}', which does "
                f"not match any journey in shared/research_journeys.py. "
                f"Known journeys: {sorted(_JOURNEY_TITLES)}."
            )

    @property
    def destination(self) -> str:
        """The workflow or journey name this branch points to."""

        return self.workflow if self.workflow is not None else self.journey

    @property
    def destination_kind(self) -> str:
        """"workflow" or "research journey", for describing it honestly."""

        return "workflow" if self.workflow is not None else "research journey"


BRANCHES: tuple[MethodBranch, ...] = (
    MethodBranch(
        id="reliability",
        question=(
            "Is my survey or scale actually measuring one thing "
            "consistently?"
        ),
        workflow="Reliability",
        why=(
            "Checks whether the items on a scale move together "
            "consistently, before you treat them as one score."
        ),
        youll_learn=(
            "Cronbach's alpha, item-level diagnostics, and split-half "
            "reliability."
        ),
        limitations=(
            "Says nothing about whether the scale measures the right "
            "construct, only whether the items behave consistently.",
            "Can't fix items that are miscoded or simply don't belong "
            "together.",
        ),
    ),
    MethodBranch(
        id="time_series_qa",
        question=(
            "Is my time-stamped data complete and regular enough to trust "
            "before I analyze it?"
        ),
        workflow="Time-Series QA",
        why=(
            "Checks the time axis and coverage before a trend, mean, or "
            "forecast becomes misleading."
        ),
        youll_learn=(
            "Gaps, duplicate timestamps, ordering, and coverage per period."
        ),
        limitations=(
            "Only checks the shape and completeness of the time series, "
            "not whether the measurements themselves are accurate.",
        ),
    ),
    MethodBranch(
        id="impact_evaluation",
        question="Did my program or intervention cause the change I'm seeing?",
        workflow="Impact Evaluation",
        why="Recommends and runs the comparison that fits your design.",
        youll_learn=(
            "Which statistical test fits your data, the result, and what "
            "your design can and cannot support."
        ),
        limitations=(
            "A weak design limits what can be concluded, no matter which "
            "test is run.",
        ),
    ),
    MethodBranch(
        id="fairness",
        question=(
            "Does an outcome or decision differ across groups in a way "
            "that raises fairness concerns?"
        ),
        workflow="Fairness",
        why=(
            "Helps match your specific concern to a fairness metric, and "
            "examines observed disparities."
        ),
        youll_learn=(
            "Favorable-outcome rate differences by group, and the "
            "tradeoffs between competing fairness definitions."
        ),
        limitations=(
            "A disparity does not by itself explain its cause.",
            "No single fairness metric can satisfy every goal at once.",
        ),
    ),
    MethodBranch(
        id="cross_analysis",
        question=(
            "I've already run more than one of these analyses. How do "
            "their results actually compare?"
        ),
        workflow="Cross-Analysis Implications",
        why=(
            "Different analyses can silently keep different rows from the "
            "same file."
        ),
        youll_learn=(
            "Retention counts and exclusion reasons for each analysis, "
            "side by side."
        ),
        limitations=(
            "Doesn't say why data was missing.",
            "Applies no threshold for how much exclusion is acceptable.",
        ),
    ),
    MethodBranch(
        id="portfolio_impact",
        question=(
            "Does the evidence I've already assembled about a program, "
            "grantee, or portfolio result actually support the claim I "
            "want to make?"
        ),
        journey="Portfolio Impact Analysis",
        why=(
            "Grades assembled evidence against the Nesta Standards of "
            "Evidence and flags where a claim expects more rigor than the "
            "evidence reaches, rather than running a new statistical test."
        ),
        youll_learn=(
            "How to spot evidence gaps and method bias, and how a "
            "grantee's result compares to the rest of a portfolio without "
            "a false apples-to-apples score."
        ),
        limitations=(
            "Reviews evidence already assembled; it does not run its own "
            "comparison test (see Impact Evaluation for that).",
            "A well-supported claim about one grantee does not generalize "
            "to the rest of the portfolio.",
        ),
    ),
)
