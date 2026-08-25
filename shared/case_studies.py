"""
Published research examples that ground each module's assumptions.

These are the educational core of the toolkit, so a wrong citation or an
overstated lesson matters more here than anywhere else in the codebase.
Every entry below was verified against publisher records: author names,
year, journal, volume, issue, and pages, and separately whether the summary
matches what the study actually found and whether the takeaway claims more
than the study supports.

Each study records the research stage where its lesson applies, using the
same vocabulary as shared/catalog.py. A study shown on two module pages
carries the same stage on both, so the reader sees one coherent idea rather
than two unrelated boxes.

Three principles for anything added here:

- The takeaway must follow from the cited study, not from adjacent
  knowledge that happens to be true.
- Never refer to another study by position ("the example below"). Grouping
  order changes, and a positional reference silently becomes false. Name
  the study.
- Where a finding is genuinely contested, say so rather than presenting one
  reading as settled.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.catalog import LIFECYCLE_STAGES


@dataclass(frozen=True)
class CaseStudy:
    """
    One published example, and where its lesson applies.

    stage is the research stage the lesson speaks to, which is how these are
    grouped for display. principle is the specific idea this case
    illustrates. modules lists the validation taxonomy keys whose pages show
    it.
    """

    title: str
    stage: str
    principle: str
    summary: str
    takeaway: str
    citation: str
    modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.stage not in LIFECYCLE_STAGES:
            raise ValueError(
                f"'{self.title}' has stage '{self.stage}', which is not a "
                f"known research stage. Valid stages: "
                f"{', '.join(LIFECYCLE_STAGES)}."
            )


CASE_STUDIES = {
    # ------------------------------------------------------------------
    # Research Question
    # ------------------------------------------------------------------
    "head_start_impact_study": CaseStudy(
        title="What the Head Start Impact Study could and could not settle",
        stage="Research Question",
        principle="A strong design still only answers what it measured",
        summary=(
            "The national Head Start Impact Study randomly assigned about "
            "4,400 eligible children either to a group offered access to "
            "Head Start or to a control group. Outcomes were measured at "
            "the end of the Head Start year and again after kindergarten "
            "and first grade. The study found several favorable impacts at "
            "the end of the program year, but most of the measured impacts "
            "were no longer statistically detectable by the end of first "
            "grade."
        ),
        takeaway=(
            "Random assignment greatly reduces the selection-bias problem "
            "that the LaLonde job training study illustrates, so the design "
            "here is strong. Two limits remain. A study estimates effects "
            "only at the points it measures, so short-term and longer-term "
            "results can differ. And many children in the control group "
            "attended other center-based programs, which makes this a "
            "comparison of Head Start against other available care rather "
            "than against none. Reanalyses that modeled that crossover "
            "reached more favorable conclusions, and the interpretation is "
            "still debated."
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
        stage="Research Question",
        principle="What a comparison group is for",
        summary=(
            "Prison-visit programs intended to deter at-risk youth were "
            "widely promoted using testimonials, anecdotal evidence, and "
            "favorable uncontrolled reports. A systematic review of "
            "randomized and quasi-randomized trials found that the programs "
            "were not effective and, on average, increased subsequent "
            "offending relative to no intervention. Effects varied across "
            "the included studies, which were mostly small and conducted "
            "some decades ago."
        ),
        takeaway=(
            "These programs were promoted on testimonials and uncontrolled "
            "before-and-after reports. When controlled trials were pooled, "
            "they looked harmful rather than helpful. Before-and-after "
            "evidence on its own cannot separate a program's effect from "
            "maturation, regression to the mean, outside events, or whatever "
            "would have happened anyway, so adding a valid comparison group "
            "can reverse the apparent conclusion."
        ),
        citation=(
            "Petrosino, A., Turpin-Petrosino, C., Hollis-Peel, M. E., & "
            "Lavenberg, J. G. (2013). 'Scared Straight' and other juvenile "
            "awareness programs for preventing juvenile delinquency: A "
            "systematic review. Campbell Systematic Reviews, 9(1), 1-55."
        ),
        modules=("program_validation",),
    ),
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    "flint_water_sampling_bias": CaseStudy(
        title="Sampling-site selection after the Flint water crisis",
        stage="Data",
        principle="How data are collected can matter as much as the analysis",
        summary=(
            "During recovery from the Flint water crisis, results from an "
            "official sentinel-site monitoring program suggested improving "
            "water-lead levels, while voluntary residential samples showed "
            "the opposite trend over the same period. An analysis of the two "
            "sampling programs found that the state-controlled sentinel "
            "sites were less representative of Flint's housing stock than "
            "the voluntary samples were. Sentinel sites with lead service "
            "lines tended to be older homes in less-poor areas, and some "
            "of the most impoverished wards had no sentinel site with a "
            "lead service line at all."
        ),
        takeaway=(
            "Validation has to begin with measurement that is trustworthy "
            "and appropriately representative. Downstream analysis cannot "
            "repair a sample that systematically underrepresents the places "
            "or people at greatest risk. Checking who or what was actually "
            "measured is a substantive step, not a formality."
        ),
        citation=(
            "Goovaerts, P. (2017). Monitoring the aftermath of Flint "
            "drinking water contamination crisis: Another case of sampling "
            "bias? Science of the Total Environment, 590-591, 139-153."
        ),
        modules=("data_validation",),
    ),
    "reinhart_rogoff": CaseStudy(
        title="The Reinhart-Rogoff spreadsheet error",
        stage="Data",
        principle="Data-processing choices can change a conclusion",
        summary=(
            "An influential 2010 analysis reported that countries with "
            "public debt above 90% of GDP had markedly lower average "
            "economic growth, and it was frequently cited in debates over "
            "government austerity. A later replication identified an Excel "
            "range error that excluded five countries from the averages, "
            "along with consequential weighting and data-exclusion choices. "
            "Correcting these raised average growth in the high-debt group "
            "from -0.1% to 2.2%, though the broader relationship between "
            "debt and growth remained a subject of debate."
        ),
        takeaway=(
            "The lesson is not about spreadsheets. It is that analyses "
            "should be auditable and reproducible, because small processing "
            "errors and less-visible analytical choices can materially "
            "influence a headline result. Independent replication is a core "
            "validation practice rather than an optional afterthought."
        ),
        citation=(
            "Herndon, T., Ash, M., & Pollin, R. (2014). Does high public "
            "debt consistently stifle economic growth? A critique of "
            "Reinhart and Rogoff. Cambridge Journal of Economics, 38(2), "
            "257-279."
        ),
        modules=("data_validation",),
    ),
    "tamiflu_unpublished_data": CaseStudy(
        title="The Tamiflu clinical study reports",
        stage="Data",
        principle="A review depends on access to the complete evidence",
        summary=(
            "For years, Cochrane reviewers were unable to obtain complete "
            "clinical study reports for many oseltamivir trials, while much "
            "of the detailed evidence remained unpublished. After a "
            "multi-year campaign the reviewers obtained a far larger body of "
            "clinical study reports and based their 2014 review on that more "
            "complete evidence. The resulting assessment found more modest "
            "benefits, no difference in hospital admissions, and harms "
            "including nausea and vomiting that were under-reported in the "
            "journal publications."
        ),
        takeaway=(
            "Missing or unpublished evidence is not a minor gap. It can "
            "materially affect a review's conclusions. Assessing whether the "
            "available evidence base is complete is a distinct validation "
            "task from calculating statistics on the studies that happen to "
            "be accessible."
        ),
        citation=(
            "Jefferson, T., Jones, M., Doshi, P., Spencer, E. A., "
            "Onakpoya, I., & Heneghan, C. J. (2014). Oseltamivir for "
            "influenza in adults and children: Systematic review of "
            "clinical study reports and summary of regulatory comments. "
            "BMJ, 348, g2545."
        ),
        modules=("data_validation",),
    ),
    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------
    "reverse_coding_artifact": CaseStudy(
        title="The Rosenberg Self-Esteem Scale's 'two factors'",
        stage="Measurement",
        principle="Item wording can look like a second construct",
        summary=(
            "The widely used Rosenberg Self-Esteem Scale mixes positively "
            "and negatively worded items. Factor analyses have often found "
            "what appear to be separate positive- and negative-self-esteem "
            "dimensions. Follow-up work showed that much of this split can "
            "be explained by item-wording direction and response artifacts "
            "rather than two distinct underlying constructs."
        ),
        takeaway=(
            "Apparent multidimensionality in a scale can reflect how items "
            "were worded rather than several substantive constructs. Before "
            "concluding that a scale measures more than one thing, test for "
            "wording-method effects. Checking that reverse-worded items were "
            "scored in the right direction is a separate and equally basic "
            "step."
        ),
        citation=(
            "Marsh, H. W. (1996). Positive and negative global self-esteem: "
            "A substantively meaningful distinction or artifactors? Journal "
            "of Personality and Social Psychology, 70(4), 810-819."
        ),
        modules=("measurement_validation",),
    ),
    "pulse_oximeter_bias": CaseStudy(
        title="Pulse oximeter racial bias",
        stage="Measurement",
        principle="Bias can enter before any model exists",
        summary=(
            "Pulse oximeters estimate arterial oxygen saturation from light "
            "absorption through tissue. In paired pulse-oximeter and "
            "arterial-blood-gas measurements, researchers found that occult "
            "hypoxemia occurred substantially more often in Black patients "
            "than in White patients. The result indicates that measurement "
            "performance can differ across groups, with skin pigmentation "
            "among the plausible contributors to the optical measurement "
            "error."
        ),
        takeaway=(
            "Bias can arise during measurement itself, before any "
            "statistical or machine-learning model is applied. A downstream "
            "model cannot fully repair an input whose errors are systematic "
            "and unmeasured, so validation should examine the instruments "
            "and labels used to create the data."
        ),
        citation=(
            "Sjoding, M. W., Dickson, R. P., Iwashyna, T. J., Gay, S. E., "
            "& Valley, T. S. (2020). Racial bias in pulse oximetry "
            "measurement [letter]. New England Journal of Medicine, "
            "383(25), 2477-2478."
        ),
        modules=("measurement_validation", "model_validation"),
    ),
    "wearable_heart_rate_accuracy": CaseStudy(
        title="Consumer wearable heart-rate accuracy",
        stage="Measurement",
        principle="Validation is ongoing and sometimes contested",
        summary=(
            "Researchers evaluated several commercial wearable devices and "
            "found that heart-rate accuracy varied by device and activity "
            "type, with error during activity averaging about 30% higher "
            "than at rest. In their analysis, error was not significantly "
            "associated with skin tone. A subsequent critique questioned "
            "whether the skin-tone measurement, sample, and statistical "
            "power were sufficient to detect relevant differences. The "
            "original authors replied defending their power calculation "
            "while acknowledging measurement limitations."
        ),
        takeaway=(
            "Not every validation question has a single settled answer. "
            "Studies can reach different conclusions because of differences "
            "in sampling, measurement, design, and statistical power. A "
            "finding of no significant difference is not the same as "
            "evidence of no difference, and methodological criticism and "
            "reply are part of the record to weigh."
        ),
        citation=(
            "Bent, B., Goldstein, B. A., Kibbe, W. A., & Dunn, J. P. "
            "(2020). Investigating sources of inaccuracy in wearable "
            "optical heart rate sensors. npj Digital Medicine, 3, 18. "
            "See also the critique by Colvonen, P. J. (2021), npj Digital "
            "Medicine, 4, 38, and the authors' reply, npj Digital "
            "Medicine, 4, 39."
        ),
        modules=("measurement_validation",),
    ),
    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    "lalonde_1986": CaseStudy(
        title="Job training estimates that did not match the experiment",
        stage="Analysis",
        principle="Selection bias in nonrandomized comparison groups",
        summary=(
            "LaLonde compared the benchmark estimate from a randomized "
            "job-training experiment with estimates produced by commonly "
            "used nonexperimental evaluation methods. Many of the "
            "nonexperimental estimators failed to recover the experimental "
            "benchmark, sometimes by a wide margin, and standard "
            "specification tests did not reliably flag the ones that failed."
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
    "fairness_impossibility": CaseStudy(
        title="Why one fairness metric cannot satisfy every goal",
        stage="Analysis",
        principle="Fairness definitions can be mathematically incompatible",
        summary=(
            "Related mathematical results showed that, when outcome base "
            "rates differ across groups and prediction is imperfect, a risk "
            "score generally cannot simultaneously satisfy calibration "
            "within groups and equal error rates across groups. "
            "Compatibility is possible only in two special cases, namely "
            "perfect prediction or equal base rates."
        ),
        takeaway=(
            "Choosing a fairness criterion requires an explicit judgment "
            "about which guarantees matter most in a particular decision. A "
            "calibrated model can still have unequal error rates, and "
            "equalizing error rates can require giving the same score a "
            "different meaning across groups. Reporting several metrics is "
            "therefore more informative than reducing fairness to a single "
            "pass or fail verdict."
        ),
        citation=(
            "Kleinberg, J., Mullainathan, S., & Raghavan, M. (2017). "
            "Inherent trade-offs in the fair determination of risk scores. "
            "8th Innovations in Theoretical Computer Science Conference "
            "(ITCS 2017), LIPIcs vol. 67, 43:1-43:23. arXiv:1609.05807. "
            "See also Chouldechova, A. (2017). Fair prediction with "
            "disparate impact: A study of bias in recidivism prediction "
            "instruments. Big Data, 5(2), 153-163."
        ),
        modules=("model_validation",),
    ),
    "dead_salmon": CaseStudy(
        title="The dead-salmon fMRI demonstration",
        stage="Analysis",
        principle="Multiple testing and false-positive control",
        summary=(
            "Researchers demonstrated that an fMRI analysis could produce "
            "apparently significant activation in a dead salmon when many "
            "voxel-wise tests were conducted without correction for "
            "multiple comparisons. The apparent result disappeared once "
            "familywise error rate or false discovery rate controls were "
            "applied, even at relaxed thresholds."
        ),
        takeaway=(
            "Conducting many statistical tests increases the probability of "
            "false-positive findings, so multiplicity has to be addressed "
            "before isolated significant results are interpreted. The same "
            "underlying principle is why a set of pairwise post-hoc "
            "comparisons needs an adjustment method such as Tukey HSD "
            "rather than many uncorrected t-tests."
        ),
        citation=(
            "Bennett, C. M., Wolford, G. L., & Miller, M. B. (2009). The "
            "principled control of false positives in neuroimaging. Social "
            "Cognitive and Affective Neuroscience, 4(4), 417-422, which "
            "describes the demonstration. Originally presented as Bennett, "
            "C. M., Baird, A. A., Miller, M. B., & Wolford, G. L., 15th "
            "Annual Meeting of the Organization for Human Brain Mapping "
            "(2009), and published in the Journal of Serendipitous and "
            "Unexpected Results, 1(1), 1-5 (2010)."
        ),
        modules=("program_validation",),
    ),
    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    "confounding_video_games": CaseStudy(
        title="Spatial cognition and video-game training",
        stage="Interpretation",
        principle="Experience as an alternative explanation for a group difference",
        summary=(
            "Feng and colleagues observed a gender difference in spatial "
            "attention and mental-rotation performance. In a training "
            "experiment, participants assigned to action-video-game "
            "training improved more on the studied spatial tasks than "
            "participants assigned to a non-action control game. Women "
            "showed particularly large gains, and the measured gender gap "
            "was reduced."
        ),
        takeaway=(
            "Observed group differences should not automatically be treated "
            "as fixed, innate, or biologically determined. This study "
            "provides evidence that experience can contribute to a measured "
            "difference, and that at least some of the observed performance "
            "gap was modifiable under the study's training conditions."
        ),
        citation=(
            "Feng, J., Spence, I., & Pratt, J. (2007). Playing an action "
            "video game reduces gender differences in spatial cognition. "
            "Psychological Science, 18(10), 850-855."
        ),
        modules=("model_validation",),
    ),
    "narps": CaseStudy(
        title="Many analysts, one neuroimaging dataset",
        stage="Interpretation",
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
            "reporting help reveal whether a conclusion is robust to those "
            "choices. This study directly motivates the multi-coding "
            "sensitivity analysis in the impact evaluation module."
        ),
        citation=(
            "Botvinik-Nezer, R., Holzmeister, F., Camerer, C. F., et al. "
            "(2020). Variability in the analysis of a single neuroimaging "
            "dataset by many teams. Nature, 582, 84-88."
        ),
        modules=("program_validation",),
    ),
}


# The order studies appear on each page, most relevant first.
#
# Ordering is explicit rather than derived from stage, because the two are
# different questions. Stage says where in a research workflow a lesson
# applies; this says which lesson a reader of that page should meet first.
# Deriving order from stage would put a measurement example ahead of a
# fairness example on the fairness page purely because measurement precedes
# analysis in a research workflow, which is not the right priority for
# someone reading about fairness.
#
# Membership is still owned by each study's `modules` field. A test asserts
# the two agree exactly, so a study cannot be tagged for a page and left out
# of its order, or vice versa.
PAGE_ORDER: dict[str, tuple[str, ...]] = {
    "measurement_validation": (
        "reverse_coding_artifact",
        "pulse_oximeter_bias",
        "wearable_heart_rate_accuracy",
    ),
    "data_validation": (
        "flint_water_sampling_bias",
        "reinhart_rogoff",
        "tamiflu_unpublished_data",
    ),
    "model_validation": (
        "fairness_impossibility",
        "confounding_video_games",
        "pulse_oximeter_bias",
    ),
    "program_validation": (
        "head_start_impact_study",
        "scared_straight",
        "lalonde_1986",
        "dead_salmon",
        "narps",
    ),
}


def get_case_studies(module: str) -> list[CaseStudy]:
    """
    Return case studies for a module, most relevant first.

    Falls back to definition order for a module with no declared order, so
    an untuned page still renders rather than silently showing nothing.
    """
    order = PAGE_ORDER.get(module)

    if order is None:
        return [
            case_study
            for case_study in CASE_STUDIES.values()
            if module in case_study.modules
        ]

    return [CASE_STUDIES[key] for key in order]
