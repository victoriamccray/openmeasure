"""
What OpenMeasure offers, in one place.

The landing page used to describe the modules in a hand-written tree. That
tree drifted: it still advertised data validation as under development after
Time-Series QA had shipped. This module is the single source those
descriptions come from, and shared/tests/test_catalog.py fails when it and
the pages/ directory disagree.

Two different ideas are recorded separately because they answer different
questions:

- stage: where in a research workflow the question arises. This is the
  page's layout.
- category: which kind of validation the workflow performs. This is what
  the module READMEs and the sidebar use.

They are not always the same word. The fairness workflow currently examines
observed labels before any model exists, so its category is broader than
"Model Validation" alone would suggest, while its stage is still Analysis.

No streamlit import: this is data, so it can be tested directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# Stages of a research workflow, in the order a study moves through them.
# The landing page renders these in this order, including any with no
# workflow yet.
LIFECYCLE_STAGES: tuple[str, ...] = (
    "Research Question",
    "Data",
    "Measurement",
    "Analysis",
    "Interpretation",
)

# What each stage asks, shown under its heading.
STAGE_QUESTIONS: dict[str, str] = {
    "Research Question": "What is being asked, and what evidence would answer it?",
    "Data": "Are the observations there, and can the record be trusted?",
    "Measurement": "Does the instrument measure the intended construct consistently?",
    "Analysis": "Is the method appropriate, and who does the result hold for?",
    "Interpretation": "What do these results actually support?",
}

# Stages with no workflow yet. Named explicitly so the gap is a stated fact
# rather than something a reader has to notice, and so the test fails if a
# stage empties out unintentionally.
STAGES_WITHOUT_WORKFLOWS: tuple[str, ...] = ("Research Question",)

# Validation categories, in the order the sidebar groups them.
#
# Declared explicitly because the sidebar needs an order and there is no
# honest way to derive one. Alphabetical would be arbitrary, and taking it
# from stage order would break the moment a category held workflows at two
# different stages.
#
# Each category is a named slot in the navigation even when it holds a
# single workflow, so a second workflow has an obvious home.
CATEGORY_ORDER: tuple[str, ...] = (
    "Measurement Validation",
    "Data Validation",
    "Fairness & Model Validation",
    "Program Validation",
    "Cross-cutting validation",
)

# The keys workflow pages record their results under, via shared/handoff.py.
#
# These live here rather than in each page because two separate readers join
# on them: the Cross-Analysis Implications page and the progress status on the
# overview cards. When each page held its own string literal, renaming one
# would have made the status read "Not assessed" forever with nothing failing.
MODULE_RELIABILITY = "reliability"
MODULE_PROGRAM_EVALUATION = "program_evaluation"
MODULE_FAIRNESS = "fairness"
MODULE_TIME_SERIES_QA = "time_series_qa"
MODULE_EVIDENCE_TO_CLAIM = "evidence_to_claim"

MODULE_KEYS: tuple[str, ...] = (
    MODULE_RELIABILITY,
    MODULE_PROGRAM_EVALUATION,
    MODULE_FAIRNESS,
    MODULE_TIME_SERIES_QA,
    MODULE_EVIDENCE_TO_CLAIM,
)


@dataclass(frozen=True)
class Workflow:
    """
    One user-facing workflow, and where it belongs.

    taxonomy_key is the key used by shared/case_studies.py, which groups its
    research examples by validation category rather than by workflow. It is
    deliberately separate from category, whose wording is chosen to describe
    the workflow accurately on the landing page.

    module_key is the key this workflow records results under. None means the
    workflow records nothing, which is true of Cross-Analysis Implications:
    it reads what the others recorded, so it can never carry a status of its
    own.
    """

    workflow: str
    category: str
    stage: str
    version: str
    summary: str
    page: str
    taxonomy_key: str | None = None
    module_key: str | None = None

    def __post_init__(self) -> None:
        if self.stage not in LIFECYCLE_STAGES:
            raise ValueError(
                f"{self.workflow} has stage '{self.stage}', which is not a "
                f"known lifecycle stage. Valid stages: "
                f"{', '.join(LIFECYCLE_STAGES)}."
            )

        for name in ("workflow", "category", "version", "summary", "page"):
            if not getattr(self, name):
                raise ValueError(
                    f"{self.workflow or 'A workflow'} is missing a value for "
                    f"'{name}'."
                )

        if "\n" in self.summary:
            raise ValueError(
                f"{self.workflow}'s summary must be a single line, so it fits "
                "on a card."
            )

        if self.category not in CATEGORY_ORDER:
            raise ValueError(
                f"{self.workflow} has category '{self.category}', which is "
                f"not in CATEGORY_ORDER, so it would render in no sidebar "
                f"section. Known categories: {', '.join(CATEGORY_ORDER)}."
            )

        if self.module_key is not None and self.module_key not in MODULE_KEYS:
            raise ValueError(
                f"{self.workflow} has module_key '{self.module_key}', which "
                f"is not a declared key, so nothing it records would ever be "
                f"found. Known keys: {', '.join(MODULE_KEYS)}."
            )

    @property
    def url_path(self) -> str:
        """
        The URL segment for this workflow.

        Derived by stripping the numeric filename prefix, which is exactly
        what Streamlit's own file-based routing did before navigation became
        explicit. Deriving it rather than storing it keeps today's links
        working without a second field to keep in sync.
        """

        stem = Path(self.page).stem

        return re.sub(r"^\d+_", "", stem)


WORKFLOWS: tuple[Workflow, ...] = (
    Workflow(
        workflow="Time-Series QA",
        category="Data Validation",
        stage="Data",
        version="0.1",
        summary=(
            "Checks whether a time series is complete and regularly sampled "
            "enough to analyze: gaps, duplicate timestamps, ordering, and "
            "coverage per period."
        ),
        page="pages/4_Time_Series_QA.py",
        taxonomy_key="data_validation",
        module_key=MODULE_TIME_SERIES_QA,
    ),
    Workflow(
        workflow="Reliability",
        category="Measurement Validation",
        stage="Measurement",
        version="0.1",
        summary=(
            "Cronbach's alpha, corrected item-total correlations, alpha if "
            "item dropped, and split-half reliability for a scale or survey."
        ),
        page="pages/1_Reliability.py",
        taxonomy_key="measurement_validation",
        module_key=MODULE_RELIABILITY,
    ),
    Workflow(
        workflow="Impact Evaluation",
        category="Program Validation",
        stage="Analysis",
        version="0.1",
        summary=(
            "Recommends and runs the comparison that fits your design, then "
            "states what the design can and cannot support."
        ),
        page="pages/2_Impact_Evaluation.py",
        taxonomy_key="program_validation",
        module_key=MODULE_PROGRAM_EVALUATION,
    ),
    Workflow(
        workflow="Fairness",
        category="Fairness & Model Validation",
        stage="Analysis",
        version="0.05",
        summary=(
            "Compares favorable-outcome rates across groups and explains the "
            "assumptions and tradeoffs behind competing fairness definitions."
        ),
        page="pages/3_Fairness.py",
        taxonomy_key="model_validation",
        module_key=MODULE_FAIRNESS,
    ),
    Workflow(
        workflow="Cross-Analysis Implications",
        category="Cross-cutting validation",
        stage="Interpretation",
        version="0.1",
        summary=(
            "Shows how much of your data each analysis actually used, so you "
            "can see what each result describes."
        ),
        page="pages/5_Cross_Analysis_Implications.py",
        taxonomy_key="data_validation",
        # No module_key: this page reads the other workflows' records rather
        # than producing one of its own.
    ),
    Workflow(
        workflow="Portfolio Impact Analysis",
        category="Program Validation",
        stage="Interpretation",
        version="0.1",
        summary=(
            "Grades the strength of evidence behind a program, grantee, or "
            "portfolio claim and drafts reporting language, with "
            "limitations and comparability issues stated alongside it."
        ),
        page="pages/6_Portfolio_Impact_Analysis.py",
        taxonomy_key="portfolio_impact_analysis",
        module_key=MODULE_EVIDENCE_TO_CLAIM,
    ),
)


def workflows_by_stage() -> dict[str, tuple[Workflow, ...]]:
    """
    Group the workflows by lifecycle stage.

    Every stage in LIFECYCLE_STAGES is present, in order, including those
    with no workflow. The landing page renders an empty stage as a stated
    gap rather than omitting it, which would imply the lifecycle starts
    somewhere later than it does.
    """

    grouped: dict[str, list[Workflow]] = {
        stage: [] for stage in LIFECYCLE_STAGES
    }

    for item in WORKFLOWS:
        grouped[item.stage].append(item)

    return {stage: tuple(items) for stage, items in grouped.items()}


def workflows_by_category() -> dict[str, tuple[Workflow, ...]]:
    """
    Group the workflows by validation category, in CATEGORY_ORDER.

    This is what the sidebar is built from. Unlike workflows_by_stage, a
    category with no workflow is omitted rather than shown, because an empty
    navigation section is a dead heading rather than a stated gap. Every
    category currently holds one, and __post_init__ rejects a workflow whose
    category is unknown, so a workflow cannot silently vanish from the
    sidebar.
    """

    grouped: dict[str, list[Workflow]] = {
        category: [] for category in CATEGORY_ORDER
    }

    for item in WORKFLOWS:
        grouped[item.category].append(item)

    return {
        category: tuple(items)
        for category, items in grouped.items()
        if items
    }
