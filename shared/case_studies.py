"""Reusable research case studies for OpenMeasure modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseStudy:
    title: str
    principle: str
    summary: str
    takeaway: str
    citation: str
    modules: tuple[str, ...]


CASE_STUDIES = {
    "reverse_coding_artifact": CaseStudy(
        title="The Rosenberg Self-Esteem Scale's 'two factors'",
        principle="Reverse-coded items and method artifacts",
        summary=(
            "The widely used Rosenberg Self-Esteem Scale mixes positively "
            "and negatively worded items. Factor analyses repeatedly found "
            "what looked like two separate self-esteem factors, but "
            "follow-up work showed the split tracked item wording "
            "direction, not two genuine underlying constructs."
        ),
        takeaway=(
            "Apparent multidimensionality in a scale can be a wording "
            "artifact rather than evidence of multiple real constructs. "
            "Check whether reverse-coded items were correctly recoded "
            "before concluding a scale measures more than one thing."
        ),
        citation=(
            "Marsh, H. W. (1996). Positive and negative self-esteem: A "
            "substantively meaningful distinction or artifactors? Journal "
            "of Personality and Social Psychology, 70, 810-819."
        ),
        modules=("measurement_validation",),
    ),
    "head_start_impact_study": CaseStudy(
        title="Head Start Impact Study: rigorous design, and why timing matters",
        principle="Strong design (randomization) does not fix the timing question",
        summary=(
            "The national Head Start evaluation randomly assigned nearly "
            "5,000 children to a Head Start group or a control group, then "
            "measured outcomes at multiple time points: end of preschool, "
            "kindergarten, and 1st grade. Positive effects on cognitive "
            "measures found at the end of preschool had faded by "
            "kindergarten and were no longer detectable by 1st grade."
        ),
        takeaway=(
            "This is an affirmative example of a well-designed evaluation, "
            "random assignment rules out the selection-bias problem seen "
            "in the LaLonde example above. But it also shows a separate "
            "issue: even a rigorous design only tells you about the time "
            "point(s) you measured. A short-term pre/post result and a "
            "longer-term follow-up can genuinely disagree, and both can be "
            "correct for the window they measured."
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
        principle="Comparison groups vs. pre/post-only evidence",
        summary=(
            "Prison-visit programs for at-risk youth were widely adopted "
            "based on positive anecdotal and pre/post reports. When "
            "randomized controlled trials with real comparison groups were "
            "run, a systematic review found the programs increased "
            "subsequent offending by 1-28% relative to doing nothing at all."
        ),
        takeaway=(
            "A program that looks effective in a pre/post-only design can "
            "turn out to be actively harmful once compared against a real "
            "control group. Pre/post evidence alone cannot rule this out, "
            "which is why this module flags that limitation on every "
            "single-group pre/post result."
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
        title="Job training programs: experimental vs. non-experimental estimates",
        principle="Selection bias in non-randomized comparison groups",
        summary=(
            "LaLonde compared results from a randomized job-training "
            "experiment to what standard non-experimental comparison-group "
            "methods, common in program evaluation at the time, would have "
            "estimated using the same data. Most non-experimental methods "
            "failed to replicate the true experimental result, some by a "
            "wide margin."
        ),
        takeaway=(
            "Comparing a treatment group to a non-randomized comparison "
            "group can produce a confident-looking, but wrong, estimate of "
            "a program's effect. When group membership wasn't randomly "
            "assigned, differences between groups may reflect who selected "
            "into the program, not the program's effect."
        ),
        citation=(
            "LaLonde, R. J. (1986). Evaluating the econometric evaluations "
            "of training programs with experimental data. American "
            "Economic Review, 76(4), 604-620."
        ),
        modules=("program_validation",),
    ),
    "confounding_video_games": CaseStudy(
        title="Spatial cognition and video-game experience",
        principle="Confounding and alternative explanations",
        summary=(
            "Action-video-game training substantially improved spatial "
            "attention and mental rotation, with women benefiting more than "
            "men and the initial gender difference being greatly reduced."
        ),
        takeaway=(
            "Observed group differences should not automatically be treated "
            "as fixed or biological; experience and exposure may explain part "
            "of the difference."
        ),
        citation=(
            "Feng, J., Spence, I., & Pratt, J. (2007). Playing an action "
            "video game reduces gender differences in spatial cognition. "
            "Psychological Science, 18(10), 850-855."
        ),
        modules=("program_validation", "model_validation"),
    ),
    "dead_salmon": CaseStudy(
        title="The dead-salmon fMRI demonstration",
        principle="Multiple testing and false positives",
        summary=(
            "Researchers demonstrated apparently significant fMRI activation "
            "in a dead salmon when many voxel-wise tests were performed "
            "without adequate multiple-comparison correction."
        ),
        takeaway=(
            "Large numbers of statistical tests increase false-positive risk. "
            "Multiplicity must be addressed before interpreting isolated "
            "significant findings."
        ),
        citation=(
            "Bennett, C. M., Wolford, G. L., & Miller, M. B. (2009). "
            "The principled control of false positives in neuroimaging. "
            "Social Cognitive and Affective Neuroscience, 4(4), 417-422."
        ),
        modules=("data_validation", "model_validation", "program_validation"),
    ),
    "narps": CaseStudy(
        title="Many analysts, one neuroimaging dataset",
        principle="Analytic variability",
        summary=(
            "Seventy independent teams analyzed the same dataset and tested "
            "the same hypotheses, but chose different workflows and reached "
            "substantially different conclusions."
        ),
        takeaway=(
            "Reasonable analytical choices can affect results. Sensitivity "
            "analyses and transparent reporting help show whether conclusions "
            "are robust to those choices."
        ),
        citation=(
            "Botvinik-Nezer, R., Holzmeister, F., Camerer, C. F., et al. "
            "(2020). Variability in the analysis of a single neuroimaging "
            "dataset by many teams. Nature, 582, 84-88."
        ),
        modules=("data_validation", "model_validation", "program_validation"),
    ),
}


def get_case_studies(module: str) -> list[CaseStudy]:
    """Return case studies relevant to an OpenMeasure module."""
    return [
        case_study
        for case_study in CASE_STUDIES.values()
        if module in case_study.modules
    ]
