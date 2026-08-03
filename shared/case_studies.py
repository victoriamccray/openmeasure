"""Reusable research case studies for OpenMeasure modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseStudy:
    title: str
    category: str  # broad grouping used to organize display within a page
    principle: str  # specific principle this individual case illustrates
    summary: str
    takeaway: str
    citation: str
    modules: tuple[str, ...]


CASE_STUDIES = {
    "reverse_coding_artifact": CaseStudy(
        title="The Rosenberg Self-Esteem Scale's 'two factors'",
        category="Measurement & scale construction",
        principle="Reverse-coded items and method artifacts",
        summary=(
            "The widely used Rosenberg Self-Esteem Scale mixes positively "
            "and negatively worded items. Factor analyses have often found "
            "what appear to be separate positive- and negative-self-esteem "
            "dimensions. Follow-up work showed that much of this split can "
            "be explained by item-wording direction and response artifacts "
            "rather than two distinct underlying constructs."
        ),
        takeaway=(
            "Apparent multidimensionality in a scale can be a wording "
            "artifact rather than evidence of multiple substantive "
            "constructs. Confirm that reverse-worded items were correctly "
            "scored, and test for wording-method effects, before concluding "
            "that a scale measures more than one thing."
        ),
        citation=(
            "Marsh, H. W. (1996). Positive and negative global self-esteem: "
            "A substantively meaningful distinction or artifactors? Journal "
            "of Personality and Social Psychology, 70(4), 810-819."
        ),
        modules=("measurement_validation",),
    ),
    "head_start_impact_study": CaseStudy(
        title="Head Start Impact Study: rigorous design, and why timing matters",
        category="Study design & causal inference",
        principle="Strong design does not resolve the timing question",
        summary=(
            "The national Head Start Impact Study randomly assigned nearly "
            "5,000 eligible children either to a group offered access to "
            "Head Start or to a control group. Outcomes were measured at "
            "the end of the Head Start year and again after kindergarten "
            "and first grade. The study found several favorable impacts at "
            "the end of the program year, but most of the measured impacts "
            "were no longer statistically detectable by the end of first "
            "grade."
        ),
        takeaway=(
            "This is an affirmative example of a well-designed evaluation: "
            "random assignment greatly reduces the selection-bias problem "
            "illustrated by the LaLonde example below. It also demonstrates "
            "a separate issue. Even a rigorous design identifies effects "
            "only at the time points that are measured. Short-term and "
            "longer-term estimates can differ, and each can accurately "
            "describe its own measurement window."
        ),
        citation=(
            "Puma, M., Bell, S., Cook, R., & Heid, C. (2010). Head Start "
            "Impact Study: Final Report. Washington, DC: U.S. Department "
            "of Health and Human Services, Administration for Children and "
            "Families, Office of Planning, Research and Evaluation."
        ),
        modules=("program_validation",),
    ),
    "scared_straight": CaseStudy(
        title="'Scared Straight' juvenile deterrence programs",
        category="Study design & causal inference",
        principle="Comparison groups vs. pre/post-only evidence",
        summary=(
            "Prison-visit programs intended to deter at-risk youth were "
            "widely promoted using testimonials, anecdotal evidence, and "
            "favorable uncontrolled reports. A systematic review of "
            "randomized and quasi-randomized trials found that the programs "
            "were not effective and, on average, increased subsequent "
            "offending relative to no intervention. Effects varied across "
            "the included studies."
        ),
        takeaway=(
            "A program that appears effective in an uncontrolled pre/post "
            "design can turn out to be harmful when compared with a valid "
            "control group. Pre/post evidence alone cannot separate the "
            "program's effect from maturation, regression to the mean, "
            "outside events, or the outcomes that would have occurred "
            "without the program."
        ),
        citation=(
            "Petrosino, A., Turpin-Petrosino, C., Hollis-Peel, M. E., & "
            "Lavenberg, J. G. (2013). 'Scared Straight' and other juvenile "
            "awareness programs for preventing juvenile delinquency: A "
            "systematic review. Campbell Systematic Reviews, 9(1), 1-55."
        ),
        modules=("program_validation",),
    ),
    "lalonde_1986": CaseStudy(
        title="Job training programs: experimental vs. nonexperimental estimates",
        category="Study design & causal inference",
        principle="Selection bias in nonrandomized comparison groups",
        summary=(
            "LaLonde compared the benchmark estimate from a randomized "
            "job-training experiment with estimates produced by commonly "
            "used nonexperimental evaluation methods. Most of the "
            "nonexperimental estimators failed to recover the experimental "
            "benchmark, sometimes by a wide