"""
Neuroanatomy for teaching: where each sensing modality gets a signal, and
what kind of signal it is.

The Become the Signal experience follows one scalp EEG recording. This
module places that recording in the brain: which electrode picked it up,
which systems contribute to the rhythm it carries, and how other sensing
modalities (fNIRS, fMRI, structural MRI, peripheral body sensors) compare.

Two questions stay separate, because answering one says nothing about
the other:

    coverage     Can this sensor, in the configuration described, obtain
                 a signal associated with this region?
    measurement  What does that signal physically measure: electrical
                 potentials, blood oxygenation, tissue anatomy, or a body
                 response? Only the first is neural electrical activity.

Coverage levels are teaching simplifications drawn from the cited
reviews, for the stated sensor configuration. Real coverage depends on
sensor count, placement, signal quality, and analysis, and the atlas
carries that caveat so every presentation layer can show it.

Every claim carries citation keys into REFERENCES. Prose follows the
experience's style rule (shared/tests/test_become_the_signal.py): no
dashes as punctuation, and nothing described by what it lacks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

COVERED = "covered"
PARTIAL = "partial"
OUT_OF_RANGE = "none"
COVERAGE_LEVELS = (COVERED, PARTIAL, OUT_OF_RANGE)
_RANK = {OUT_OF_RANGE: 0, PARTIAL: 1, COVERED: 2}

MEASURE_ELECTRICAL = "electrical"
MEASURE_HEMODYNAMIC = "hemodynamic"
MEASURE_ANATOMY = "anatomy"
MEASURE_PERIPHERAL = "peripheral"
MEASUREMENTS = (MEASURE_ELECTRICAL, MEASURE_HEMODYNAMIC, MEASURE_ANATOMY, MEASURE_PERIPHERAL)

DEPTH_SURFACE = "surface"
DEPTH_DEEP = "deep"
DEPTH_BODY = "body"
DEPTHS = (DEPTH_SURFACE, DEPTH_DEEP, DEPTH_BODY)

HEMISPHERE_LEFT = "left"
HEMISPHERE_RIGHT = "right"
HEMISPHERE_MIDLINE = "midline"


class NeuroanatomyLookupError(LookupError):
    """Base class for a lookup this module cannot answer."""


class UnknownModalityError(NeuroanatomyLookupError):
    """No sensing modality has this key."""


class UnknownSystemError(NeuroanatomyLookupError):
    """No brain system has this key."""


class UnknownElectrodeError(NeuroanatomyLookupError):
    """The label is outside the standard 10-20 electrode set."""


@dataclass(frozen=True)
class Reference:
    key: str
    text: str
    url: str


@dataclass(frozen=True)
class BrainSystem:
    """One brain or body system a learner can point to on the figure."""

    key: str
    name: str
    location: str
    role: str
    depth: str
    refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.depth not in DEPTHS:
            raise ValueError(f"{self.key}: depth must be one of {DEPTHS}; got {self.depth!r}.")
        if not self.refs:
            raise ValueError(f"{self.key}: every system needs at least one reference.")


@dataclass(frozen=True)
class SystemCoverage:
    """
    Whether one modality gets a signal from one system, in two senses.

    level is what this stated configuration covers. in_principle is what
    the method can reach with a different placement or setup. They differ
    for fNIRS: its light reaches outer cortex anywhere optodes are placed,
    while the forehead array here covers only the frontal cortex. An empty
    in_principle means the configuration already reaches all the method can.
    """

    modality: str
    system: str
    level: str
    note: str = ""
    in_principle: str = ""

    def __post_init__(self) -> None:
        if self.level not in COVERAGE_LEVELS:
            raise ValueError(
                f"{self.modality}/{self.system}: level must be one of {COVERAGE_LEVELS}; got {self.level!r}."
            )
        if not self.in_principle:
            object.__setattr__(self, "in_principle", self.level)
        if self.in_principle not in COVERAGE_LEVELS:
            raise ValueError(
                f"{self.modality}/{self.system}: in_principle must be one of {COVERAGE_LEVELS}; "
                f"got {self.in_principle!r}."
            )
        if _RANK[self.in_principle] < _RANK[self.level]:
            raise ValueError(
                f"{self.modality}/{self.system}: a configuration cannot cover more than the method "
                f"can reach ({self.level} configured, {self.in_principle} in principle)."
            )

    @property
    def reachable_elsewhere(self) -> bool:
        """True when this setup misses a system another placement would reach."""
        return _RANK[self.in_principle] > _RANK[self.level]


@dataclass(frozen=True)
class SensingModality:
    """One way of sensing the brain or body, in one stated configuration."""

    key: str
    name: str
    short_name: str
    configuration: str
    measurement: str
    measures: str
    neural_link: str
    timing: str
    coverage: tuple[SystemCoverage, ...]
    also_captures: str
    privacy_note: str
    refs: tuple[str, ...]
    in_journey: bool = False

    def __post_init__(self) -> None:
        if self.measurement not in MEASUREMENTS:
            raise ValueError(
                f"{self.key}: measurement must be one of {MEASUREMENTS}; got {self.measurement!r}."
            )
        if not self.refs:
            raise ValueError(f"{self.key}: every modality needs at least one reference.")
        systems = [c.system for c in self.coverage]
        if len(systems) != len(set(systems)):
            raise ValueError(f"{self.key}: a system appears twice in coverage.")

    def coverage_for(self, system_key: str) -> SystemCoverage:
        for entry in self.coverage:
            if entry.system == system_key:
                return entry
        raise UnknownSystemError(f"{self.key} has no coverage entry for system {system_key!r}.")


@dataclass(frozen=True)
class ElectrodeSite:
    """Where a 10-20 electrode sits, read from its label."""

    label: str
    region: str
    system: str
    hemisphere: str
    naming_note: str
    refs: tuple[str, ...]


@dataclass(frozen=True)
class NeuroanatomyAtlas:
    systems: tuple[BrainSystem, ...]
    modalities: tuple[SensingModality, ...]
    recording_electrode: ElectrodeSite
    electrodes: tuple[ElectrodeSite, ...]
    electrode_caveat: str
    electrode_caveat_refs: tuple[str, ...]
    caveat: str
    caveat_refs: tuple[str, ...]
    references: tuple[Reference, ...]


# ------------------------------------------------------------------
# References
# ------------------------------------------------------------------

REFERENCES: tuple[Reference, ...] = (
    Reference(
        "kandel",
        "Kandel, E. R., Koester, J. D., Mack, S. H., & Siegelbaum, S. A. (Eds.). (2021). "
        "Principles of Neural Science (6th ed.). McGraw Hill.",
        "",
    ),
    Reference(
        "barry",
        "Barry, R. J., Clarke, A. R., Johnstone, S. J., Magee, C. A., & Rushby, J. A. (2007). "
        "EEG differences between eyes-closed and eyes-open resting conditions. "
        "Clinical Neurophysiology, 118(12), 2765-2773.",
        "https://doi.org/10.1016/j.clinph.2007.07.028",
    ),
    Reference(
        "halgren",
        "Halgren, M., Ulbert, I., Bastuji, H., Fabó, D., Erőss, L., Rey, M., Devinsky, O., Doyle, W. K., "
        "Mak-McCully, R., Halgren, E., Wittner, L., Chauvel, P., Heit, G., Eskandar, E., Mandell, A., & "
        "Cash, S. S. (2019). The generation and propagation of the human alpha rhythm. Proceedings of "
        "the National Academy of Sciences, 116(47), 23772-23782.",
        "https://doi.org/10.1073/pnas.1913092116",
    ),
    Reference(
        "lopes_da_silva",
        "Lopes da Silva, F. (1991). Neural mechanisms underlying brain waves: From neural "
        "membranes to networks. Electroencephalography and Clinical Neurophysiology, 79(2), 81-93.",
        "https://doi.org/10.1016/0013-4694(91)90044-5",
    ),
    Reference(
        "critchley",
        "Critchley, H. D. (2002). Electrodermal responses: What happens in the brain. "
        "The Neuroscientist, 8(2), 132-142.",
        "https://doi.org/10.1177/107385840200800209",
    ),
    Reference(
        "buzsaki",
        "Buzsáki, G., Anastassiou, C. A., & Koch, C. (2012). The origin of extracellular fields "
        "and currents: EEG, ECoG, LFP and spikes. Nature Reviews Neuroscience, 13(6), 407-420.",
        "https://doi.org/10.1038/nrn3241",
    ),
    Reference(
        "seeber",
        "Seeber, M., Cantonas, L.-M., Hoevels, M., Sesia, T., Visser-Vandewalle, V., & Michel, C. M. "
        "(2019). Subcortical electrophysiological activity is detectable with high-density EEG "
        "source imaging. Nature Communications, 10, 753.",
        "https://doi.org/10.1038/s41467-019-08725-w",
    ),
    Reference(
        "bagley",
        "Bagley, B. A., Rose, N., Kilbourn, Q., & Canham, M. (2026). Threat vectors and the state "
        "of the art in defense methods for security in neurotechnology. arXiv:2607.10451.",
        "https://arxiv.org/abs/2607.10451",
    ),
    Reference(
        "pinti",
        "Pinti, P., Tachtsidis, I., Hamilton, A., Hirsch, J., Aichelburg, C., Gilbert, S., & "
        "Burgess, P. W. (2020). The present and future use of functional near-infrared "
        "spectroscopy (fNIRS) for cognitive neuroscience. Annals of the New York Academy of "
        "Sciences, 1464(1), 5-29.",
        "https://doi.org/10.1111/nyas.13948",
    ),
    Reference(
        "yucel",
        "Yücel, M. A., Lühmann, A. v., Scholkmann, F., Gervain, J., Dan, I., Ayaz, H., Boas, D., "
        "Cooper, R. J., Culver, J., Elwell, C. E., Eggebrecht, A., Franceschini, M. A., Grova, C., "
        "Homae, F., Lesage, F., Obrig, H., Tachtsidis, I., Tak, S., Tong, Y., Torricelli, A., Wabnitz, H., "
        "& Wolf, M. (2021). Best practices for fNIRS publications. Neurophotonics, 8(1), 012101.",
        "https://doi.org/10.1117/1.NPh.8.1.012101",
    ),
    Reference(
        "logothetis",
        "Logothetis, N. K. (2008). What we can do and what we cannot do with fMRI. "
        "Nature, 453(7197), 869-878.",
        "https://doi.org/10.1038/nature06976",
    ),
    Reference(
        "schwarz",
        "Schwarz, C. G., Kremers, W. K., Therneau, T. M., Sharp, R. R., Gunter, J. L., Vemuri, P., "
        "Arani, A., Spychalla, A. J., Kantarci, K., Knopman, D. S., Petersen, R. C., & Jack, C. R., Jr. "
        "(2019). Identification of anonymous MRI research participants with face-recognition "
        "software. New England Journal of Medicine, 381(17), 1684-1686. PMID: 31644852. "
        "PMCID: PMC7091256. Free full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC7091256/",
        "https://doi.org/10.1056/NEJMc1908881",
    ),
    Reference(
        "schwarz_2021",
        "Schwarz, C. G., Kremers, W. K., Wiste, H. J., Gunter, J. L., Vemuri, P., Spychalla, A. J., "
        "Kantarci, K., Schultz, A. P., Sperling, R. A., Knopman, D. S., Petersen, R. C., Jack, C. R., Jr., & "
        "Alzheimer's Disease Neuroimaging Initiative (2021). Changing the face of neuroimaging research: Comparing a new MRI de-facing technique "
        "with popular alternatives. NeuroImage, 231, 117845.",
        "https://doi.org/10.1016/j.neuroimage.2021.117845",
    ),
    Reference(
        "jasper",
        "Jasper, H. H. (1958). The ten-twenty electrode system of the International Federation. "
        "Electroencephalography and Clinical Neurophysiology, 10(2), 371-375.",
        "https://doi.org/10.1016/0013-4694(58)90053-1",
    ),
    Reference(
        "acharya",
        "Acharya, J. N., Hani, A. J., Cheek, J., Thirumala, P., & Tsuchida, T. N. (2016). American "
        "Clinical Neurophysiology Society Guideline 2: Guidelines for standard electrode position "
        "nomenclature. Journal of Clinical Neurophysiology, 33(4), 308-311.",
        "https://doi.org/10.1097/WNP.0000000000000316",
    ),
    Reference(
        "koessler",
        "Koessler, L., Maillard, L., Benhadid, A., Vignal, J. P., Felblinger, J., Vespignani, H., & "
        "Braun, M. (2009). Automated cortical projection of EEG sensors: Anatomical correlation via "
        "the international 10-10 system. NeuroImage, 46(1), 64-72.",
        "https://doi.org/10.1016/j.neuroimage.2009.02.006",
    ),
)

# ------------------------------------------------------------------
# Systems, front of the brain to the body
# ------------------------------------------------------------------

SYSTEMS: tuple[BrainSystem, ...] = (
    BrainSystem(
        "frontal", "Frontal lobe", "Front of the brain, behind the forehead",
        "Planning, attention, decision making, and control of behavior.",
        DEPTH_SURFACE, ("kandel",),
    ),
    BrainSystem(
        "sensorimotor", "Primary motor and sensory cortex",
        "A strip over the top of the brain, on both sides of the central sulcus",
        "The primary motor cortex, in front of the sulcus, sends movement commands. The primary "
        "sensory cortex, behind it, receives touch and body position. Planning a movement also "
        "draws on wider frontal and parietal areas. Motor imagery brain computer interfaces read "
        "rhythms over this strip.",
        DEPTH_SURFACE, ("kandel",),
    ),
    BrainSystem(
        "parietal", "Parietal lobe", "Top and back of the brain",
        "Combines touch, vision, and body position to guide attention and movement in space.",
        DEPTH_SURFACE, ("kandel",),
    ),
    BrainSystem(
        "temporal", "Temporal lobe", "Side of the brain, above the ear",
        "Hearing, language comprehension, and recognizing objects and faces.",
        DEPTH_SURFACE, ("kandel",),
    ),
    BrainSystem(
        "occipital", "Occipital lobe (visual cortex)", "Back of the brain",
        "Processes vision. In relaxed, awake adults, scalp alpha is usually strongest over "
        "occipital and parietal sites with the eyes closed. Its generators are distributed across "
        "cortical and thalamocortical networks, so no single patch of cortex produces it.",
        DEPTH_SURFACE, ("kandel", "barry", "halgren", "lopes_da_silva"),
    ),
    BrainSystem(
        "thalamus", "Thalamus", "Center of the brain, above the brainstem",
        "Relays sensory signals to the cortex and takes part in the thalamocortical loops that "
        "shape rhythms such as alpha.",
        DEPTH_DEEP, ("kandel", "lopes_da_silva"),
    ),
    BrainSystem(
        "hippocampus_amygdala", "Hippocampus and amygdala", "Deep inside the temporal lobe",
        "Forms new memories (hippocampus) and processes emotion and threat (amygdala).",
        DEPTH_DEEP, ("kandel",),
    ),
    BrainSystem(
        "cerebellum", "Cerebellum", "Below the occipital lobe, at the back of the head",
        "Coordinates and times movement and supports motor learning.",
        DEPTH_SURFACE, ("kandel",),
    ),
    BrainSystem(
        "brainstem", "Brainstem", "Base of the brain, where it joins the spinal cord",
        "Regulates heart rate, breathing, and arousal, and links the brain to the body.",
        DEPTH_DEEP, ("kandel",),
    ),
    BrainSystem(
        "autonomic", "Autonomic nervous system",
        "Nerves from the brainstem and spinal cord to the heart, skin, and organs",
        "Adjusts heart rate, sweating, and pupil size without conscious control.",
        DEPTH_BODY, ("kandel", "critchley"),
    ),
)

CORTICAL = ("frontal", "sensorimotor", "parietal", "temporal", "occipital")
BRAIN = CORTICAL + ("thalamus", "hippocampus_amygdala", "cerebellum", "brainstem")


def _coverage(
    modality: str,
    levels: dict[str, str],
    notes: dict[str, str] | None = None,
    in_principle: dict[str, str] | None = None,
) -> tuple[SystemCoverage, ...]:
    notes = notes or {}
    reach = in_principle or {}
    return tuple(
        SystemCoverage(
            modality, s.key, levels.get(s.key, OUT_OF_RANGE), notes.get(s.key, ""),
            reach.get(s.key, levels.get(s.key, OUT_OF_RANGE)),
        )
        for s in SYSTEMS
    )


COVERAGE_TEXT = {
    COVERED: "Covered by this setup",
    PARTIAL: "Partial or weak",
    OUT_OF_RANGE: "Out of range",
}
REACHABLE_TEXT = "Reachable with another setup"
ANATOMY_TEXT = "Anatomy visible"


def coverage_label(modality_key: str, system_key: str) -> str:
    """
    The plain label for one modality and system.

    Structural MRI images anatomy and records no activity, so its covered
    regions read "Anatomy visible". A region this setup misses but another
    placement would reach reads "Reachable with another setup".
    """
    m = modality(modality_key)
    entry = m.coverage_for(system_key)
    if entry.level == COVERED and m.measurement == MEASURE_ANATOMY:
        return ANATOMY_TEXT
    if entry.reachable_elsewhere and entry.level == OUT_OF_RANGE:
        return REACHABLE_TEXT
    return COVERAGE_TEXT[entry.level]


_EEG_CORTEX_NOTE = (
    "Electrodes over this lobe pick up its activity, blended with other sources by volume conduction. "
    "Orientation and depth matter as well as distance, so the lobe beneath an electrode does not "
    "always contribute most."
)
_HEMODYNAMIC_NOTE = "Blood oxygen changes that follow neural activity, measured here."

MODALITIES: tuple[SensingModality, ...] = (
    SensingModality(
        key="eeg",
        name="Scalp electroencephalography",
        short_name="EEG",
        configuration="A whole-head cap of scalp electrodes. This journey uses one of its channels, O1.",
        measurement=MEASURE_ELECTRICAL,
        measures="Voltage differences between each electrode and a reference.",
        neural_link="Direct electrical measure: mostly synchronized postsynaptic currents from "
        "large populations of neurons, particularly cortical pyramidal cells.",
        timing="Milliseconds. Fast enough to follow each cycle of the 8 to 13 Hz alpha rhythm.",
        coverage=_coverage(
            "eeg",
            {**{k: COVERED for k in CORTICAL},
             "thalamus": PARTIAL, "hippocampus_amygdala": PARTIAL, "cerebellum": PARTIAL},
            {
                **{k: _EEG_CORTEX_NOTE for k in CORTICAL},
                "thalamus": "Seen mostly through the cortical rhythms it helps shape.",
                "hippocampus_amygdala": "Detectable with many electrodes and source modeling, and weak at any single electrode.",
                "cerebellum": "Far from most scalp electrodes.",
            },
        ),
        also_captures="Research studies have identified people from their EEG. How reliably "
        "depends on recording conditions and methods.",
        privacy_note="A recording made for one task may also carry information about attention, "
        "drowsiness, or health. How well these can be inferred depends on signal quality, "
        "protocol, population, and validation.",
        refs=("buzsaki", "seeber", "bagley"),
        in_journey=True,
    ),
    SensingModality(
        key="fnirs",
        name="Functional near-infrared spectroscopy",
        short_name="fNIRS",
        configuration="A forehead array of light sources and detectors, a common wearable layout. "
        "Optodes placed elsewhere on the head cover the cortex beneath them instead.",
        measurement=MEASURE_HEMODYNAMIC,
        measures="Changes in oxygenated and deoxygenated hemoglobin, from how much near-infrared "
        "light returns to each detector.",
        neural_link="Indirect: blood oxygen changes that follow neural activity, mixed with scalp "
        "blood flow and changes across the whole body.",
        timing="Seconds. Blood flow follows neural activity with a delay.",
        coverage=_coverage(
            "fnirs",
            {"frontal": COVERED},
            {
                "frontal": "Outer cortex beneath the forehead optodes. " + _HEMODYNAMIC_NOTE,
                **{k: "Reachable with optodes placed over it. This forehead array leaves it uncovered."
                   for k in ("sensorimotor", "parietal", "temporal", "occipital")},
            },
            in_principle={k: COVERED for k in CORTICAL},
        ),
        also_captures="Heartbeat, breathing, and scalp blood flow, which ride on the hemoglobin signal.",
        privacy_note="Wearable and portable, so it can record outside the lab during everyday tasks.",
        refs=("pinti", "yucel"),
    ),
    SensingModality(
        key="fmri",
        name="Functional magnetic resonance imaging",
        short_name="fMRI",
        configuration="A whole-head MRI scanner.",
        measurement=MEASURE_HEMODYNAMIC,
        measures="The blood oxygen level dependent (BOLD) signal across the whole brain.",
        neural_link="Indirect: blood oxygen changes that follow neural activity.",
        timing="Seconds. The blood oxygen response peaks several seconds after neural activity.",
        coverage=_coverage("fmri", {k: COVERED for k in BRAIN}, {k: _HEMODYNAMIC_NOTE for k in BRAIN}),
        also_captures="A structural image of the head is usually acquired in the same session, "
        "and it includes the face.",
        privacy_note="Sharing the structural image alongside the functional data shares the face too.",
        refs=("logothetis", "schwarz"),
    ),
    SensingModality(
        key="smri",
        name="Structural magnetic resonance imaging",
        short_name="Structural MRI",
        configuration="A whole-head MRI scanner.",
        measurement=MEASURE_ANATOMY,
        measures="Shape and tissue of the brain and the whole head.",
        neural_link="Anatomy only. It images every region and records no moment to moment activity.",
        timing="One snapshot of anatomy.",
        coverage=_coverage(
            "smri", {k: COVERED for k in BRAIN}, {k: "Imaged as anatomy." for k in BRAIN}
        ),
        also_captures="The face. Face recognition software, choosing among the 84 participants' "
        "MRI face reconstructions, ranked the correct one first for 70 of 84 participants' "
        "photographs (83%) and within the top five for 80 of 84 (95%).",
        privacy_note="Removing the face before sharing, called defacing, lowers this risk. In a "
        "later study, face recognition matched 97% of 157 scans before defacing, and 28 to 38% "
        "still matched after popular defacing tools.",
        refs=("schwarz", "schwarz_2021"),
    ),
    SensingModality(
        key="peripheral",
        name="Peripheral body sensors (skin conductance and heart rate)",
        short_name="Body sensors",
        configuration="Sensors on the skin, wrist, or chest.",
        measurement=MEASURE_PERIPHERAL,
        measures="Sweat gland activity and the timing of each heartbeat.",
        neural_link="No brain region is measured. These signals reflect the autonomic nervous "
        "system's effects on skin and heart.",
        timing="Seconds for skin conductance. Every beat for heart rate.",
        coverage=_coverage(
            "peripheral",
            {"autonomic": COVERED},
            {
                "autonomic": "Measured through its effects on skin and heart.",
                "brainstem": "Body responses carry traces of brainstem control. Reading those "
                "traces is inference from another signal.",
            },
        ),
        also_captures="Signs of autonomic arousal. Exercise, temperature, medication, and illness "
        "change these signals too, so neither one measures stress by itself.",
        privacy_note="Common in consumer wearables, so these signals are often collected far "
        "outside research settings.",
        refs=("critchley",),
    ),
)

CAVEAT = (
    "Schematic drawing: region shapes and boundaries are approximate. Coverage levels are "
    "teaching simplifications for the configuration described, drawn from the cited reviews. "
    "Real coverage depends on how many sensors there are, where they sit, signal quality, and "
    "how the data are analyzed. Shading shows where a signal comes from, and its color shows what "
    "kind of signal it is."
)
CAVEAT_REFS = ("buzsaki", "seeber", "pinti", "yucel")

ELECTRODE_CAVEAT = (
    "An electrode records the voltage difference between its site and a reference. Currents "
    "spread through tissue (volume conduction), so each electrode picks up a blend of sources, "
    "with nearby cortex contributing most. The glow around O1 is schematic."
)
ELECTRODE_CAVEAT_REFS = ("buzsaki",)

# ------------------------------------------------------------------
# The 10-20 electrode system
# ------------------------------------------------------------------

# Jasper (1958), with the later names T7, T8, P7, P8 for T3, T4, T5, T6
# (Acharya et al., 2016).
STANDARD_10_20 = (
    "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "T7", "C3", "Cz", "C4", "T4", "T8",
    "T5", "P7", "P3", "Pz", "P4", "T6", "P8", "O1", "Oz", "O2",
)

CLASSIC_TO_MODERN = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
MODERN_TO_CLASSIC = {v: k for k, v in CLASSIC_TO_MODERN.items()}
# The two lower names sit at the junction of temporal and parietal cortex.
POSTERIOR_TEMPORAL = ("T5", "T6", "P7", "P8")

_PREFIX = {
    "Fp": ("frontal pole", "frontal"),
    "F": ("frontal", "frontal"),
    "C": ("central", "sensorimotor"),
    "T": ("temporal", "temporal"),
    "P": ("parietal", "parietal"),
    "O": ("occipital", "occipital"),
}
_LABEL = re.compile(r"^(Fp|F|C|T|P|O)(z|\d+)$")


def electrode_site(label: str) -> ElectrodeSite:
    """
    Read where a standard 10-20 electrode sits from its label.

    The letters name the region (Fp frontal pole, F frontal, C central,
    T temporal, P parietal, O occipital). Odd numbers are over the left
    hemisphere, even numbers over the right, and z over the midline
    (Jasper, 1958). Four positions have two names: the classic T3, T4,
    T5, T6 and the modern T7, T8, P7, P8 (Acharya et al., 2016).
    Koessler et al. (2009) confirm the regions by projecting electrode
    positions onto each person's cortex.
    """
    if not isinstance(label, str) or not label.strip():
        raise UnknownElectrodeError("An electrode label is required, for example O1.")

    raw = label.strip()
    normalized = raw[0].upper() + raw[1:].lower()
    if normalized not in STANDARD_10_20:
        raise UnknownElectrodeError(
            f"{raw!r} is outside the standard 10-20 set: {', '.join(STANDARD_10_20)}."
        )

    prefix, suffix = _LABEL.match(normalized).groups()
    region, system_key = _PREFIX[prefix]
    if suffix == "z":
        hemisphere = HEMISPHERE_MIDLINE
        side = "z means the midline"
    elif int(suffix) % 2:
        hemisphere = HEMISPHERE_LEFT
        side = f"{suffix} is odd, so the left hemisphere"
    else:
        hemisphere = HEMISPHERE_RIGHT
        side = f"{suffix} is even, so the right hemisphere"

    note = f"{prefix} stands for {region}, and {side}."
    if normalized in POSTERIOR_TEMPORAL:
        region, system_key = "posterior temporal", "temporal"
        note += " This position sits where the temporal and parietal lobes meet."
    if normalized in CLASSIC_TO_MODERN:
        note += f" Newer charts call this position {CLASSIC_TO_MODERN[normalized]}."
    elif normalized in MODERN_TO_CLASSIC:
        note += f" Classic 10-20 charts call this position {MODERN_TO_CLASSIC[normalized]}."

    return ElectrodeSite(normalized, region, system_key, hemisphere, note, ("jasper", "acharya", "koessler"))


# ------------------------------------------------------------------
# Lookups
# ------------------------------------------------------------------

RECORDING_ELECTRODE = "O1"


def system(key: str) -> BrainSystem:
    for entry in SYSTEMS:
        if entry.key == key:
            return entry
    raise UnknownSystemError(f"No brain system {key!r}. Known: {', '.join(s.key for s in SYSTEMS)}.")


def modality(key: str) -> SensingModality:
    for entry in MODALITIES:
        if entry.key == key:
            return entry
    raise UnknownModalityError(f"No modality {key!r}. Known: {', '.join(m.key for m in MODALITIES)}.")


def coverage(modality_key: str, system_key: str) -> SystemCoverage:
    """Whether one modality, as configured, gets a signal from one system."""
    system(system_key)
    return modality(modality_key).coverage_for(system_key)


def atlas() -> NeuroanatomyAtlas:
    """Everything the figure needs, with only the references it cites."""
    used: list[str] = []
    for key in (
        *(k for s in SYSTEMS for k in s.refs),
        *(k for m in MODALITIES for k in m.refs),
        *electrode_site(RECORDING_ELECTRODE).refs,
        *ELECTRODE_CAVEAT_REFS,
        *CAVEAT_REFS,
    ):
        if key not in used:
            used.append(key)
    by_key = {r.key: r for r in REFERENCES}
    missing = [k for k in used if k not in by_key]
    if missing:
        raise NeuroanatomyLookupError(f"Citation keys without a reference: {', '.join(missing)}.")

    return NeuroanatomyAtlas(
        systems=SYSTEMS,
        modalities=MODALITIES,
        recording_electrode=electrode_site(RECORDING_ELECTRODE),
        electrodes=tuple(electrode_site(label) for label in STANDARD_10_20),
        electrode_caveat=ELECTRODE_CAVEAT,
        electrode_caveat_refs=ELECTRODE_CAVEAT_REFS,
        caveat=CAVEAT,
        caveat_refs=CAVEAT_REFS,
        references=tuple(by_key[k] for k in used),
    )
