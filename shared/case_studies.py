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
            "benchmark, sometimes by a wide margin."
        ),
        takeaway=(
            "Comparing a treatment group with a nonrandomized comparison "
            "group can produce a precise-looking but misleading estimate of "
            "a program's effect. When treatment is not randomly assigned, "
            "observed differences may reflect who entered the program and "
            "how the comparison group was constructed rather than the "
            "program itself."
        ),
        citation=(
            "LaLonde, R. J. (1986). Evaluating the econometric evaluations "
            "of training programs with experimental data. American Economic "
            "Review, 76(4), 604-620."
        ),
        modules=("program_validation",),
    ),
    "many_labs_replication": CaseStudy(
        title="The Many Labs projects",
        category="Analytic robustness & sensitivity",
        principle="Confidence comes from replication",
        summary=(
            "Large collaborations replicated classic psychology findings "
            "across many laboratories and participant samples. Some effects "
            "were reproduced consistently, whereas others were smaller, "
            "more variable, or not reliably distinguishable from zero."
        ),
        takeaway=(
            "A single influential study rarely settles a question. "
            "Consistency across independent samples, investigators, and "
            "settings generally provides stronger evidence than any one "
            "result, regardless of how compelling the original study "
            "appeared."
        ),
        citation=(
            "Klein, R. A., Ratliff, K. A., Vianello, M., et al. (2014). "
            "Investigating variation in replicability: A 'Many Labs' "
            "replication project. Social Psychology, 45(3), 142-152."
        ),
        modules=("program_validation", "model_validation"),
    ),
    "fairness_impossibility": CaseStudy(
        title="Why you can't satisfy every fairness metric at once",
        category="Fairness metric tradeoffs",
        principle="Mathematical incompatibility between fairness definitions",
        summary=(
            "Related mathematical results showed that, when outcome base "
            "rates differ across groups and prediction is imperfect, a risk "
            "score generally cannot simultaneously satisfy calibration "
            "within groups and equalized error-rate criteria. Compatibility "
            "is possible only in restricted cases, such as perfect "
            "prediction or equal base rates."
        ),
        takeaway=(
            "Choosing a fairness criterion requires an explicit judgment "
            "about which guarantees are most important in a particular "
            "decision context. A calibrated model can still have unequal "
            "error rates, while equalizing error rates can require giving "
            "the same score a different meaning across groups. Reporting "
            "multiple metrics is therefore more informative than producing "
            "one mathematically universal 'fair' or 'unfair' verdict."
        ),
        citation=(
            "Kleinberg, J., Mullainathan, S., & Raghavan, M. (2017). "
            "Inherent trade-offs in the fair determination of risk scores. "
            "Proceedings of the 8th Innovations in Theoretical Computer "
            "Science Conference (ITCS). arXiv:1609.05807. See also: "
            "Chouldechova, A. (2017). Fair prediction with disparate impact: "
            "A study of bias in recidivism prediction instruments. Big Data, "
            "5(2), 153-163."
        ),
        modules=("model_validation",),
    ),
    "tamiflu_unpublished_data": CaseStudy(
        title="The Tamiflu Cochrane review",
        category="Publication bias & evidence completeness",
        principle=(
            "A systematic review depends on access to complete trial evidence"
        ),
        summary=(
            "For years, Cochrane reviewers were unable to obtain complete "
            "clinical study reports for many oseltamivir trials, while much "
            "of the detailed evidence remained unpublished. After a "
            "multi-year campaign, the reviewers obtained a much larger body "
            "of clinical study reports and based their 2014 review on that "
            "more complete evidence. The resulting assessment found more "
            "modest benefits and identified harms that were difficult to "
            "evaluate from journal publications alone."
        ),
        takeaway=(
            "Missing or unpublished evidence is not a minor gap; it can "
            "materially affect a systematic review's conclusions. Assessing "
            "whether the available evidence base is complete is a distinct "
            "validation task from calculating statistical metrics on the "
            "studies that happen to be accessible."
        ),
        citation=(
            "Jefferson, T., Jones, M., Doshi, P., Del Mar, C., et al. "
            "(2014). Oseltamivir for influenza in adults and children: "
            "Systematic review of clinical study reports and summary of "
            "regulatory comments. BMJ, 348, g2545."
        ),
        modules=("data_validation", "program_validation"),
    ),
    "reinhart_rogoff": CaseStudy(
        title="The Reinhart-Rogoff spreadsheet error",
        category="Data integrity & reproducibility",
        principle="Data-processing choices can materially affect conclusions",
        summary=(
            "An influential 2010 analysis reported that countries with "
            "public debt above 90% of GDP had markedly lower average "
            "economic growth and was frequently cited in debates over "
            "government austerity. A later replication identified an Excel "
            "range error that excluded several countries, along with "
            "consequential weighting and data-exclusion choices. Correcting "
            "and revisiting those choices materially altered the estimated "
            "debt-growth relationship, although the broader relationship "
            "between debt and growth remained a subject of debate."
        ),
        takeaway=(
            "The lesson is not that spreadsheets are uniquely incapable of "
            "supporting valid analysis. It is that analyses should be "
            "auditable and reproducible, because small processing errors and "
            "less-visible analytical choices can materially influence a "
            "headline result. Independent replication is a core validation "
            "practice rather than an optional afterthought."
        ),
        citation=(
            "Herndon, T., Ash, M., & Pollin, R. (2014). Does high public "
            "debt consistently stifle economic growth? A critique of "
            "Reinhart and Rogoff. Cambridge Journal of Economics, 38(2), "
            "257-279."
        ),
        modules=("data_validation", "program_validation"),
    ),
    "pulse_oximeter_bias": CaseStudy(
        title="Pulse oximeter racial bias",
        category="Measurement bias",
        principle="Bias can originate in measurement before any model exists",
        summary=(
            "Pulse oximeters estimate arterial oxygen saturation from light "
            "absorption through tissue. In paired pulse-oximeter and "
            "arterial-blood-gas measurements, researchers found that occult "
            "hypoxemia was detected substantially more often in Black "
            "patients than in White patients. The result indicates that "
            "measurement performance can differ across groups, with skin "
            "pigmentation among the plausible contributors to the optical "
            "measurement error."
        ),
        takeaway=(
            "Bias can arise during measurement itself, before a statistical "
            "or machine-learning model is applied. A downstream model cannot "
            "fully repair an input signal whose errors are systematic and "
            "unmeasured, so validation should examine the instruments and "
            "labels used to create the data."
        ),
        citation=(
            "Sjoding, M. W., Dickson, R. P., Iwashyna, T. J., Gay, S. E., "
            "& Valley, T. S. (2020). Racial bias in pulse oximetry "
            "measurement. New England Journal of Medicine, 383(25), "
            "2477-2478."
        ),
        modules=("measurement_validation", "model_validation"),
    ),
    "wearable_heart_rate_accuracy": CaseStudy(
        title="Consumer wearable heart-rate accuracy",
        category="Measurement bias",
        principle="Validation is an ongoing and sometimes contested process",
        summary=(
            "Researchers evaluated several commercial wearable devices and "
            "found that heart-rate accuracy varied by device and activity "
            "type. In their analysis, error was not significantly associated "
            "with skin tone. A subsequent critique questioned whether the "
            "skin-tone measurement, sample, and statistical power were "
            "sufficient to detect relevant differences, and the original "
            "authors published a reply defending their methods and "
            "interpretation."
        ),
        takeaway=(
            "Not every validation question has a single settled answer. "
            "Studies can reach different conclusions because of differences "
            "in sampling, measurement, design, and statistical power. "
            "Methodological criticism and reply are part of the scientific "
            "record and should be considered when judging the strength of a "
            "conclusion."
        ),
        citation=(
            "Bent, B., Goldstein, B. A., Kibbe, W. A., & Dunn, J. P. "
            "(2020). Investigating sources of inaccuracy in wearable "
            "optical heart rate sensors. npj Digital Medicine, 3, 18."
        ),
        modules=("measurement_validation", "model_validation"),
    ),
    "fisher_rothamsted": CaseStudy(
        title="Fisher and the Rothamsted experiments",
        category="Study design & causal inference",
        principle="Design helps separate treatment effects from background variation",
        summary=(
            "During his work at Rothamsted Experimental Station, Ronald "
            "Fisher developed and systematized principles that became "
            "foundational to modern experimental research, including "
            "randomization, blocking, factorial designs, and analysis of "
            "variance. Agricultural field trials provided an important "
            "setting for developing and demonstrating these methods."
        ),
        takeaway=(
            "Many statistical methods were developed to distinguish "
            "treatment effects from ordinary natural variability. "
            "Randomization, blocking, replication, and thoughtful design "
            "remain among the strongest tools for reducing bias and "
            "estimating uncertainty. This work forms part of the historical "
            "foundation for the ANOVA methods used elsewhere in the module."
        ),
        citation=(
            "Fisher, R. A. (1935). The Design of Experiments. Oliver & Boyd."
        ),
        modules=("program_validation",),
    ),
    "flint_water_sampling_bias": CaseStudy(
        title="Flint water crisis: sampling-site selection",
        category="Sampling bias & measurement validity",
        principle="How data are collected can matter as much as how they are analyzed",
        summary=(
            "During recovery from the Flint water crisis, results from an "
            "official sentinel-site monitoring program suggested improving "
            "water-lead levels, whereas voluntary residential samples "
            "showed a different temporal pattern. An analysis of the two "
            "sampling programs found that the sentinel sites were not "
            "representative of Flint's housing stock: sites with lead "
            "service lines tended to be older homes in less-poor areas, and "
            "the programs differed in their coverage of housing and "
            "infrastructure characteristics."
        ),
        takeaway=(
            "Validation must begin with trustworthy and appropriately "
            "representative measurement. Downstream statistical analysis "
            "cannot automatically repair a sample that systematically "
            "underrepresents the locations or people at greatest risk. "
            "Checking who or what was measured is therefore a substantive "
            "validation step rather than a preliminary formality."
        ),
        citation=(
            "Goovaerts, P. (2017). Monitoring the aftermath of Flint "
            "drinking water contamination crisis: Another case of sampling "
            "bias? Science of the Total Environment, 590-591, 139-153."
        ),
        modules=("data_validation",),
    ),
    "satellite_temperature_reconciliation": CaseStudy(
        title="Satellite temperature-record reconciliation",
        category="Analytic robustness & sensitivity",
        principle="Reasonable analytic choices can shift estimates",
        summary=(
            "Research groups analyzing satellite microwave observations "
            "produced differing estimates of lower-atmosphere temperature "
            "trends. Important sources of disagreement included methods for "
            "correcting orbital drift, changes in satellite instruments, "
            "and time-of-day sampling effects. As adjustment procedures "
            "were refined and independently scrutinized, some prominent "
            "differences between the resulting temperature records "
            "narrowed, although methodological differences did not vanish."
        ),
        takeaway=(
            "Reasonable analytical choices, rather than simple error or "
            "misconduct, can produce different estimates from the same "
            "underlying observations. Transparent methods, sensitivity "
            "analysis, and independent replication allow researchers to "
            "identify why estimates differ and determine which conclusions "
            "are robust."
        ),
        citation=(
            "Mears, C. A., & Wentz, F. J. (2017). A satellite-derived "
            "lower-tropospheric atmospheric temperature dataset using an "
            "optimized adjustment for diurnal effects. Journal of Climate, "
            "30(19), 7695-7718."
        ),
        modules=("model_validation", "program_validation"),
    ),
    "confounding_video_games": CaseStudy(
        title="Spatial cognition and video-game experience",
        category="Confounding variables",
        principle="Experience as an alternative explanation for group differences",
        summary=(
            "Feng and colleagues first observed gender differences in "
            "spatial attention and mental-rotation performance, alongside "
            "differences in video-game experience. In a training experiment, "
            "participants assigned to action-video-game training improved "
            "more on the studied spatial tasks than participants assigned "
            "to a control game. Women showed particularly large gains, and "
            "the measured gender gap was reduced."
        ),
        takeaway=(
            "Observed group differences should not automatically be treated "
            "as fixed, innate, or biologically determined. This study "
            "suggests that experience can contribute to measured "
            "differences and demonstrates that at least some of the observed "
            "performance gap was modifiable under the study's training "
            "conditions."
        ),
        citation=(
            "Feng, J., Spence, I., & Pratt, J. (2007). Playing an action "
            "video game reduces gender differences in spatial cognition. "
            "Psychological Science, 18(10), 850-855."
        ),
        modules=("model_validation",),
    ),
    "dead_salmon": CaseStudy(
        title="The dead-salmon fMRI demonstration",
        category="Multiple testing & false positives",
        principle="Multiple testing and false-positive control",
        summary=(
            "Researchers demonstrated that an fMRI analysis could produce "
            "apparently significant activation in a dead salmon when many "
            "voxel-wise tests were conducted without adequate correction "
            "for multiple comparisons. The apparent result disappeared "
            "when appropriate multiplicity controls were applied."
        ),
        takeaway=(
            "Conducting many statistical tests increases the probability of "
            "false-positive findings. Multiplicity must be addressed before "
            "isolated significant results are interpreted. The same "
            "principle explains why a set of pairwise post-hoc comparisons "
            "requires an adjustment method, such as Tukey HSD, rather than "
            "many uncorrected t-tests."
        ),
        citation=(
            "Bennett, C. M., Wolford, G. L., & Miller, M. B. (2009). "
            "The principled control of false positives in neuroimaging. "
            "Social Cognitive and Affective Neuroscience, 4(4), 417-422."
        ),
        modules=("model_validation", "program_validation"),
    ),
    "narps": CaseStudy(
        title="Many analysts, one neuroimaging dataset",
        category="Analytic robustness & sensitivity",
        principle="Analytic variability",
        summary=(
            "Seventy independent analysis teams examined the same "
            "neuroimaging dataset and tested the same nine hypotheses. "
            "Teams made different defensible workflow choices, and their "
            "estimated effects and hypothesis decisions varied "
            "substantially across several of the analyses."
        ),
        takeaway=(
            "Reasonable analytical choices can affect results. Sensitivity "
            "analyses, multiverse-style comparisons, and transparent "
            "reporting help reveal whether conclusions are robust to those "
            "choices. This study directly motivates the module's "
            "multi-coding sensitivity-analysis feature."
        ),
        citation=(
            "Botvinik-Nezer, R., Holzmeister, F., Camerer, C. F., et al. "
            "(2020). Variability in the analysis of a single neuroimaging "
            "dataset by many teams. Nature, 582, 84-88."
        ),
        modules=("model_validation", "program_validation"),
    ),
}


def get_case_studies(module: str) -> list[CaseStudy]:
    """Return case studies relevant to an OpenMeasure module."""
    return [
        case_study
        for case_study in CASE_STUDIES.values()
        if module in case_study.modules
    ]


def get_case_studies_grouped(module: str) -> dict[str, list[CaseStudy]]:
    """
    Return case studies relevant to a module, grouped by category.

    Categories retain the order in which they first appear. This supports
    rendering grouped tabs instead of one long, flat list when a module
    contains several case studies.
    """
    grouped: dict[str, list[CaseStudy]] = {}
    for case_study in get_case_studies(module):
        grouped.setdefault(case_study.category, []).append(case_study)
    return grouped