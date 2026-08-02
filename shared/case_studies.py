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
