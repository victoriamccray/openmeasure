"""
The eyes-closed and eyes-open resting study, explained for learners.

Become the Signal and the Multimodal Signal Convergence page both lean on
one finding: alpha power is higher with the eyes closed than open (Barry
et al., 2007). This module holds what a learner needs to read that
finding and the numbers built on it:

    STUDY          what Barry et al. (2007) recorded, from whom, and found
    MEASURES       what each unit and statistic on the pages means, how it
                   is computed here, how to read it, and what shifts it
    BANDS          the conventional EEG frequency bands, with their caveat
    DICHOTOMIES    resting versus task, and domain-general versus
                   domain-specific, across EEG, fMRI, and skin conductance
    reactivity()   the eyes-closed to eyes-open alpha power ratio
    chance()       whether an accuracy beats chance for its sample size

"Expected range" here means what the cited literature supports, and that
differs by measure. Some have a fixed reference by design (50% chance
for two balanced classes) or one that can be calculated (the binomial
threshold). Some have a well-replicated direction (alpha higher with the
eyes closed) without a universal absolute value, because absolute alpha
power and skin conductance level depend on the electrode reference,
device, age, and skin. Each Measure states which kind it is, so no page
presents a lab-specific number as a norm.

Prose follows the Become the Signal style rule: no dashes as punctuation,
and nothing described by what it lacks.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import binom

# How a measure's expected range is known.
RANGE_BY_DESIGN = "by design"
RANGE_CALCULATED = "calculated"
RANGE_DIRECTION = "direction"
RANGE_WITHIN_PERSON = "within person"
RANGE_KINDS = (RANGE_BY_DESIGN, RANGE_CALCULATED, RANGE_DIRECTION, RANGE_WITHIN_PERSON)

# Where a number on the page comes from. A learner should never mistake
# a published finding for something measured in this recording.
SOURCE_RECORDED = "recorded"
SOURCE_PUBLISHED = "published"
SOURCE_CONVENTION = "convention"
SOURCES = (SOURCE_RECORDED, SOURCE_PUBLISHED, SOURCE_CONVENTION)
SOURCE_LABEL = {
    SOURCE_RECORDED: "Computed from this recording",
    SOURCE_PUBLISHED: "Published finding",
    SOURCE_CONVENTION: "Field convention",
}

AXIS_STATE = "state"
AXIS_SCOPE = "scope"
AXES = (AXIS_STATE, AXIS_SCOPE)
AXIS_POLES = {
    AXIS_STATE: ("Resting", "Task"),
    AXIS_SCOPE: ("Domain-general", "Domain-specific"),
}

DEFAULT_SIGNIFICANCE = 0.05


class RestingStateError(ValueError):
    """Inputs these calculations cannot use."""


@dataclass(frozen=True)
class Reference:
    key: str
    text: str
    url: str


@dataclass(frozen=True)
class Study:
    """What one study recorded, from whom, and found."""

    key: str
    citation_key: str
    participants: str
    conditions: tuple[str, ...]
    recordings: tuple[str, ...]
    protocol: str
    findings: tuple[str, ...]
    interpretation: str
    refs: tuple[str, ...]


@dataclass(frozen=True)
class Measure:
    """One unit or statistic a learner meets on the pages."""

    key: str
    name: str
    unit: str
    what_it_is: str
    how_computed: str
    how_to_read: str
    range_kind: str
    expected: str
    what_shifts_it: str
    refs: tuple[str, ...]
    source: str = SOURCE_RECORDED

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"{self.key}: source must be one of {SOURCES}; got {self.source!r}.")
        if self.range_kind not in RANGE_KINDS:
            raise ValueError(f"{self.key}: range_kind must be one of {RANGE_KINDS}; got {self.range_kind!r}.")
        if not self.refs:
            raise ValueError(f"{self.key}: every measure needs at least one reference.")


@dataclass(frozen=True)
class Band:
    name: str
    low_hz: float
    high_hz: float
    plain: str


@dataclass(frozen=True)
class Dichotomy:
    """One modality on one axis, with an example at each pole."""

    modality: str
    axis: str
    first: str
    second: str
    refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ValueError(f"{self.modality}: axis must be one of {AXES}; got {self.axis!r}.")

    @property
    def poles(self) -> tuple[str, str]:
        return AXIS_POLES[self.axis]


@dataclass(frozen=True)
class Reactivity:
    """Eyes-closed against eyes-open alpha power, for one recording."""

    mean_closed: float
    mean_open: float
    ratio: float
    direction_as_expected: bool
    midpoint: float


@dataclass(frozen=True)
class ThresholdExplanation:
    """How the eyes-closed / eyes-open threshold was set, in plain words."""

    threshold: float
    text: str


@dataclass(frozen=True)
class ChanceAssessment:
    """Whether an accuracy beats chance, given how many windows it rests on."""

    n_correct: int
    n_trials: int
    accuracy: float
    chance_accuracy: float
    threshold_correct: int
    threshold_accuracy: float
    threshold_p_value: float
    p_value: float
    significance: float
    exceeds_threshold: bool


# ------------------------------------------------------------------
# References, each checked against PubMed
# ------------------------------------------------------------------

REFERENCES: tuple[Reference, ...] = (
    Reference(
        "barry",
        "Barry, R. J., Clarke, A. R., Johnstone, S. J., Magee, C. A., & Rushby, J. A. (2007). "
        "EEG differences between eyes-closed and eyes-open resting conditions. "
        "Clinical Neurophysiology, 118(12), 2765-2773.",
        "https://doi.org/10.1016/j.clinph.2007.07.028",
    ),
    Reference(
        "barry_2017",
        "Barry, R. J., & De Blasio, F. M. (2017). EEG differences between eyes-closed and eyes-open "
        "resting remain in healthy ageing. Biological Psychology, 129, 293-304.",
        "https://doi.org/10.1016/j.biopsycho.2017.09.010",
    ),
    Reference(
        "klimesch",
        "Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory "
        "performance: A review and analysis. Brain Research Reviews, 29(2-3), 169-195.",
        "https://doi.org/10.1016/s0165-0173(98)00056-3",
    ),
    Reference(
        "chiang",
        "Chiang, A. K. I., Rennie, C. J., Robinson, P. A., van Albada, S. J., & Kerr, C. C. (2011). "
        "Age trends and sex differences of alpha rhythms including split alpha peaks. "
        "Clinical Neurophysiology, 122(8), 1505-1517.",
        "https://doi.org/10.1016/j.clinph.2011.01.040",
    ),
    Reference(
        "buzsaki",
        "Buzsáki, G., Anastassiou, C. A., & Koch, C. (2012). The origin of extracellular fields "
        "and currents: EEG, ECoG, LFP and spikes. Nature Reviews Neuroscience, 13(6), 407-420.",
        "https://doi.org/10.1038/nrn3241",
    ),
    Reference(
        "welch",
        "Welch, P. D. (1967). The use of fast Fourier transform for the estimation of power spectra: "
        "A method based on time averaging over short, modified periodograms. IEEE Transactions on "
        "Audio and Electroacoustics, 15(2), 70-73.",
        "https://doi.org/10.1109/TAU.1967.1161901",
    ),
    Reference(
        "combrisson",
        "Combrisson, E., & Jerbi, K. (2015). Exceeding chance level by chance: The caveat of "
        "theoretical chance levels in brain signal classification and statistical assessment of "
        "decoding accuracy. Journal of Neuroscience Methods, 250, 126-136.",
        "https://doi.org/10.1016/j.jneumeth.2015.01.010",
    ),
    Reference(
        "boucsein",
        "Boucsein, W., Fowles, D. C., Grimnes, S., Ben-Shakhar, G., Roth, W. T., Dawson, M. E., "
        "Filion, D. L., & Society for Psychophysiological Research Ad Hoc Committee on Electrodermal "
        "Measures (2012). Publication recommendations for electrodermal measurements. "
        "Psychophysiology, 49(8), 1017-1034.",
        "https://doi.org/10.1111/j.1469-8986.2012.01384.x",
    ),
    Reference(
        "critchley",
        "Critchley, H. D. (2002). Electrodermal responses: What happens in the brain. "
        "The Neuroscientist, 8(2), 132-142.",
        "https://doi.org/10.1177/107385840200800209",
    ),
    Reference(
        "biswal",
        "Biswal, B., Yetkin, F. Z., Haughton, V. M., & Hyde, J. S. (1995). Functional connectivity in "
        "the motor cortex of resting human brain using echo-planar MRI. Magnetic Resonance in "
        "Medicine, 34(4), 537-541.",
        "https://doi.org/10.1002/mrm.1910340409",
    ),
    Reference(
        "raichle",
        "Raichle, M. E., MacLeod, A. M., Snyder, A. Z., Powers, W. J., Gusnard, D. A., & Shulman, G. L. "
        "(2001). A default mode of brain function. Proceedings of the National Academy of Sciences, "
        "98(2), 676-682.",
        "https://doi.org/10.1073/pnas.98.2.676",
    ),
    Reference(
        "fedorenko",
        "Fedorenko, E., Duncan, J., & Kanwisher, N. (2013). Broad domain generality in focal regions "
        "of frontal and parietal cortex. Proceedings of the National Academy of Sciences, 110(41), "
        "16616-16621.",
        "https://doi.org/10.1073/pnas.1315235110",
    ),
)

# ------------------------------------------------------------------
# The study
# ------------------------------------------------------------------

STUDY = Study(
    key="barry_2007",
    citation_key="barry",
    participants="28 university students.",
    conditions=("Eyes closed, resting", "Eyes open, resting"),
    recordings=(
        "EEG across the scalp, summarized as absolute power in the delta, theta, alpha, and beta "
        "bands in 9 scalp regions.",
        "Skin conductance level, as an index of arousal.",
    ),
    protocol="Participants sat at rest with no task, once with the eyes closed and once with them "
    "open. Both are common baselines for other studies, and they differ in visual input and in "
    "arousal.",
    findings=(
        "Skin conductance level rose from eyes closed to eyes open.",
        "With the eyes closed, people with higher skin conductance had lower mean alpha.",
        "Mean power across the scalp fell in all four bands when the eyes opened.",
        "Alpha fell evenly across the scalp. The other bands changed in specific places, such as "
        "lower posterior theta and lower posterior, higher frontal beta with the eyes open.",
    ),
    interpretation="The authors read the even drop in alpha as a change in overall arousal, and "
    "the local changes in other bands as the brain processing visual input. Eyes-closed and "
    "eyes-open baselines therefore differ in more than alpha, and a study should choose between "
    "them on purpose.",
    refs=("barry",),
)

AGEING_NOTE = Study(
    key="barry_2017",
    citation_key="barry_2017",
    participants="20 young adults (mean age 20.4) and 20 older adults (mean age 68.2).",
    conditions=("Eyes closed, resting, 3 minutes", "Eyes open, resting, 3 minutes"),
    recordings=("EEG across the scalp.",),
    protocol="Eyes closed first, then eyes open.",
    findings=(
        "Older adults showed some reduction in alpha.",
        "Power in all bands dropped when the eyes opened, by similar amounts at both ages.",
        "Older adults showed more local changes, especially in beta.",
    ),
    interpretation="The eyes-closed to eyes-open change holds in healthy ageing, which matters "
    "here: the recording in this journey comes from a woman aged 60 to 80, a different person "
    "and dataset from either study.",
    refs=("barry_2017",),
)

# ------------------------------------------------------------------
# Measures
# ------------------------------------------------------------------

MEASURES: tuple[Measure, ...] = (
    Measure(
        key="frequency",
        name="Frequency",
        unit="Hz (hertz), cycles per second",
        what_it_is="How many times per second a rhythm repeats.",
        how_computed="The recording is split into frequency components. Alpha here is everything "
        "from 8 to 13 Hz.",
        how_to_read="A higher number means a faster rhythm. Bands are named ranges of frequency.",
        range_kind=RANGE_BY_DESIGN,
        expected="The 8 to 13 Hz alpha band is a convention. Each person's alpha peak sits "
        "somewhere inside or near it.",
        what_shifts_it="Age and scalp position. Alpha frequency rises through childhood and slows "
        "in older adults, and it runs faster over the back of the head. Many people show more than "
        "one alpha peak.",
        refs=("chiang", "klimesch"),
        source=SOURCE_CONVENTION,
    ),
    Measure(
        key="voltage",
        name="Voltage",
        unit="µV (microvolts), millionths of a volt",
        what_it_is="The electrical potential at one electrode, measured against a reference "
        "electrode.",
        how_computed="The amplifier samples it 500 times per second. Each sample is one number.",
        how_to_read="Larger swings mean more synchronized activity near that electrode. Timing "
        "matters as much as numbers: neurons firing out of step cancel each other at the scalp. "
        "The sign depends on the reference.",
        range_kind=RANGE_WITHIN_PERSON,
        expected="Compare conditions within one recording. Absolute voltages change with the "
        "reference, the amplifier, and the person.",
        what_shifts_it="The reference electrode, skull and scalp thickness, electrode contact, "
        "filtering, and movement or blink artifacts.",
        refs=("buzsaki",),
    ),
    Measure(
        key="alpha_power",
        name="Alpha power",
        unit="µV² (microvolts squared)",
        what_it_is="How strong the 8 to 13 Hz rhythm is in one second of signal.",
        how_computed="Each 1-second window (500 samples) gets one Hann-windowed power spectrum, "
        "using Welch's method with a single segment, so there is no overlap or averaging. The "
        "spectrum has 1 Hz steps. Alpha power is the area under it from 8 to 13 Hz. Power comes "
        "from squared voltage, which is why the unit is µV².",
        how_to_read="Higher means a stronger alpha rhythm in that second. Each dot on the strip is "
        "one second.",
        range_kind=RANGE_WITHIN_PERSON,
        expected="Read it within a person, across conditions. Absolute alpha power has no single "
        "normal value.",
        what_shifts_it="Age, scalp position, the reference electrode, the device, and arousal. "
        "Alpha power is higher over the back of the head and differs between the sexes.",
        refs=("welch", "chiang", "klimesch"),
    ),
    Measure(
        key="reactivity",
        name="Eyes-closed to eyes-open ratio",
        unit="A ratio, with no unit",
        what_it_is="Mean alpha power with the eyes closed divided by mean alpha power with them open.",
        how_computed="Average the eyes-closed windows, average the eyes-open windows, and divide.",
        how_to_read="Above 1 means alpha was stronger with the eyes closed. A value of 1 means no "
        "difference.",
        range_kind=RANGE_DIRECTION,
        expected="Above 1 is the expected direction in relaxed, awake adults, young and older. "
        "How far above 1 varies from person to person.",
        what_shifts_it="Drowsiness, arousal, attention, age, and scalp position.",
        refs=("barry", "barry_2017"),
    ),
    Measure(
        key="accuracy",
        name="Accuracy against chance",
        unit="Percent of 1-second windows labeled correctly",
        what_it_is="How often the threshold rule labeled a second as eyes closed or eyes open "
        "correctly.",
        how_computed="Each second is labeled eyes closed when its alpha power is above a "
        "threshold, set at the midpoint of the two condition means. Accuracy is correct windows "
        "divided by all windows. It is in-sample: the threshold is set from the same windows it "
        "labels, so it describes this recording and overstates what to expect on new data.",
        how_to_read="With equal numbers of each condition, guessing gets 50% on average. A small "
        "sample can beat 50% by luck, so the bar to clear depends on how many windows there are.",
        range_kind=RANGE_CALCULATED,
        expected="Chance is 50% by design. For 30 windows, 20 correct (67%) is the smallest count "
        "that luck alone reaches less than 5% of the time.",
        what_shifts_it="The number of windows, how the threshold was set, and whether it was "
        "tested on separate data.",
        refs=("combrisson",),
    ),
    Measure(
        key="skin_conductance",
        name="Skin conductance level",
        unit="µS (microsiemens)",
        what_it_is="How easily a small current passes between two electrodes on the skin, which "
        "rises with sweat gland activity.",
        how_computed="Two electrodes, often on the fingers, with a small constant voltage between "
        "them. The recording in this journey has no skin conductance channel.",
        how_to_read="Higher means more sympathetic arousal. A slow level (tonic) and quick responses "
        "to events (phasic) are read separately.",
        range_kind=RANGE_WITHIN_PERSON,
        expected="Barry et al. (2007) found it rises from eyes closed to eyes open. Absolute "
        "levels vary widely between people and setups, so studies compare within a person.",
        what_shifts_it="Temperature, electrode site and gel, hydration, medication, and exercise.",
        refs=("barry", "boucsein", "critchley"),
        source=SOURCE_PUBLISHED,
    ),
)

# Conventional edges. Labs differ by about 1 Hz, and Klimesch (1999)
# argues for bands anchored on each person's alpha peak instead.
BANDS: tuple[Band, ...] = (
    Band("Delta", 0.5, 4.0, "The slowest band shown here."),
    Band("Theta", 4.0, 8.0, "Linked with encoding new information during memory tasks."),
    Band("Alpha", 8.0, 13.0, "Strongest in relaxed wakefulness with the eyes closed."),
    Band("Beta", 13.0, 30.0, "The fastest band shown here."),
)
BAND_CAVEAT = (
    "Band edges are conventions that vary by about 1 Hz between labs. Anchoring each band on a "
    "person's own alpha peak can be more accurate."
)
BAND_REFS = ("klimesch", "chiang", "barry")

# ------------------------------------------------------------------
# Dichotomies across modalities
# ------------------------------------------------------------------

DICHOTOMIES: tuple[Dichotomy, ...] = (
    Dichotomy(
        "EEG", AXIS_STATE,
        "Eyes-closed and eyes-open rest, with no task.",
        "A memory task, where alpha drops briefly (phasic) in response to each item.",
        ("barry", "klimesch"),
    ),
    Dichotomy(
        "EEG", AXIS_SCOPE,
        "Alpha falling evenly across the scalp, read as overall arousal.",
        "Theta and beta changing over specific regions, read as visual processing.",
        ("barry",),
    ),
    Dichotomy(
        "fMRI", AXIS_STATE,
        "Resting-state scans, where slow fluctuations below 0.1 Hz rise and fall together in "
        "connected regions.",
        "Task scans, where goal-directed work engages task regions and quiets the default mode "
        "network.",
        ("biswal", "raichle"),
    ),
    Dichotomy(
        "fMRI", AXIS_SCOPE,
        "Frontal and parietal regions that respond across many demanding tasks, from arithmetic "
        "to working memory.",
        "Neighboring regions that respond to one kind of content.",
        ("fedorenko",),
    ),
    Dichotomy(
        "Skin conductance", AXIS_STATE,
        "The slow tonic level across a resting condition.",
        "Quick phasic responses to individual events.",
        ("barry", "critchley"),
    ),
    Dichotomy(
        "Skin conductance", AXIS_SCOPE,
        "Sympathetic arousal shared by emotion, attention, and effort.",
        "A specific reading needs a task or another signal alongside it, because the same rise "
        "accompanies emotion, attention, and effort.",
        ("critchley",),
    ),
)

# ------------------------------------------------------------------
# Calculations
# ------------------------------------------------------------------


def measure(key: str) -> Measure:
    for entry in MEASURES:
        if entry.key == key:
            return entry
    raise RestingStateError(f"No measure {key!r}. Known: {', '.join(m.key for m in MEASURES)}.")


def reactivity(alpha_closed, alpha_open) -> Reactivity:
    """
    The eyes-closed to eyes-open alpha power ratio.

    Means of the per-window alpha powers in each condition, and their
    ratio. Above 1 is the direction Barry et al. (2007) and Barry and
    De Blasio (2017) report.
    """
    closed = [float(v) for v in alpha_closed]
    opened = [float(v) for v in alpha_open]
    if not closed or not opened:
        raise RestingStateError("Both conditions need at least one window of alpha power.")
    if any(v < 0 for v in closed + opened):
        raise RestingStateError("Alpha power cannot be negative.")
    mean_closed = sum(closed) / len(closed)
    mean_open = sum(opened) / len(opened)
    if mean_open == 0:
        raise RestingStateError("Mean eyes-open alpha power is zero, so the ratio is undefined.")
    ratio = mean_closed / mean_open
    return Reactivity(mean_closed, mean_open, ratio, ratio > 1, (mean_closed + mean_open) / 2)


def explain_threshold(result: Reactivity) -> ThresholdExplanation:
    """
    How the classification threshold was set, with this recording's numbers.

    The threshold is the midpoint of the two condition means, the rule
    intervention.classify_eyes_state applies. It is specific to the
    recording, and it was set from the same windows it then labels.
    """
    text = (
        f"The threshold is the midpoint of the two condition means in this recording: "
        f"({result.mean_closed:.1f} + {result.mean_open:.1f}) / 2 = {result.midpoint:.1f} µV². "
        "It fits this recording only, and another person or device would need its own. It was "
        "set from the same windows it labels, so the accuracy is in-sample, and accuracy on a "
        "new session or person would likely be lower."
    )
    return ThresholdExplanation(result.midpoint, text)


def chance(
    n_correct: int,
    n_trials: int,
    n_classes: int = 2,
    significance: float = DEFAULT_SIGNIFICANCE,
) -> ChanceAssessment:
    """
    Whether an accuracy beats chance, using the binomial distribution.

    The threshold is the smallest number of correct trials that guessing
    reaches with probability below the significance level (one-sided),
    the analytical approach Combrisson and Jerbi (2015) describe. The p
    value is the probability of at least n_correct by guessing. Assumes
    independent trials with equal class sizes.
    """
    if not isinstance(n_trials, int) or n_trials < 1:
        raise RestingStateError("n_trials must be a positive whole number.")
    if not isinstance(n_correct, int) or not 0 <= n_correct <= n_trials:
        raise RestingStateError("n_correct must be a whole number from 0 to n_trials.")
    if not isinstance(n_classes, int) or n_classes < 2:
        raise RestingStateError("n_classes must be a whole number of at least 2.")
    if not 0 < significance < 1:
        raise RestingStateError("significance must be between 0 and 1.")

    p_guess = 1 / n_classes
    threshold = next(
        (k for k in range(n_trials + 1) if binom.sf(k - 1, n_trials, p_guess) < significance),
        None,
    )
    if threshold is None:
        raise RestingStateError(
            f"With {n_trials} trials, even a perfect score is within chance at {significance}."
        )
    p_value = float(binom.sf(n_correct - 1, n_trials, p_guess))
    return ChanceAssessment(
        n_correct=n_correct,
        n_trials=n_trials,
        accuracy=n_correct / n_trials,
        chance_accuracy=p_guess,
        threshold_correct=threshold,
        threshold_accuracy=threshold / n_trials,
        threshold_p_value=float(binom.sf(threshold - 1, n_trials, p_guess)),
        p_value=p_value,
        significance=significance,
        exceeds_threshold=n_correct >= threshold,
    )


def cited_keys() -> tuple[str, ...]:
    """Every reference key this module cites, in first-use order."""
    keys: list[str] = []
    for group in (
        STUDY.refs, AGEING_NOTE.refs,
        *(m.refs for m in MEASURES), BAND_REFS,
        *(d.refs for d in DICHOTOMIES),
    ):
        for key in group:
            if key not in keys:
                keys.append(key)
    return tuple(keys)
