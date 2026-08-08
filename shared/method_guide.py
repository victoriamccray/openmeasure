"""
Method Selection Decision Tree: one guided question routing to a workflow.

This is a discovery aid, not a workflow. It carries no module_key, is
never passed to shared/handoff.py, and its page is not part of
shared/catalog.py's lifecycle stages, for the same reason
shared/datasets.py's Explore Real Data catalog sits outside them: routing
a user to a workflow is not itself an analysis.

Each branch answers one question ("what are you trying to validate?") at
one level. It deliberately stops there: every destination workflow has
its own, more specific method-selection logic once you're inside it (for
example, Impact Evaluation's own recommendation of which statistical test
fits your data's shape), and duplicating that logic up here would be a
second, driftable copy of it.

workflow must match a Workflow.workflow value in shared/catalog.py
exactly, so a rename there cannot silently orphan a branch.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.catalog import WORKFLOWS

_WORKFLOW_NAMES: frozenset[str] = frozenset(item.workflow for item in WORKFLOWS)


@dataclass(frozen=True)
class MethodBranch:
    """One answer to "what are you trying to validate?", and where it leads."""

    id: str
    situation: str
    workflow: str
    why: str
    youll_learn: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("id", "situation", "workflow", "why", "youll_learn"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"{self.id or 'A branch'} is missing a value for "
                    f"'{field_name}'."
                )

        if self.workflow not in _WORKFLOW_NAMES:
            raise ValueError(
                f"{self.id} names workflow '{self.workflow}', which does "
                f"not match any workflow in shared/catalog.py. Known "
                f"workflows: {sorted(_WORKFLOW_NAMES)}."
            )

        if not self.limitations:
            raise ValueError(f"{self.id} lists no limitations.")


BRANCHES: tuple[MethodBranch, ...] = (
    MethodBranch(
        id="reliability",
        situation=(
            "A survey or scale where several items should measure one thing"
        ),
        workflow="Reliability",
        why=(
            "Checks whether those items move together consistently, before "
            "you treat them as one score."
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
        situation="A time-stamped dataset you're about to analyze",
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
        situation=(
            "Whether a program or intervention caused a change in an outcome"
        ),
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
        situation=(
            "Whether an outcome or decision differs across groups in a way "
            "that raises fairness concerns"
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
        situation=(
            "You've already run one or more of the above and want to "
            "compare what each used"
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
)
