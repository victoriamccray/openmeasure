"""
Neuroanatomy for teaching: which brain systems each sensing modality reads.

The Become the Signal experience follows one scalp EEG recording. This
module places that recording in the brain: which system the alpha rhythm
comes from, which electrode picked it up, and which systems other
sensing modalities (fNIRS, fMRI, structural MRI, peripheral body
sensors) reach in comparison.

Reach levels are teaching simplifications drawn from the cited reviews.
Real coverage depends on sensor count, placement, and analysis, and the
atlas carries that caveat so every presentation layer can show it.

Every claim carries citation keys into REFERENCES. Prose follows the
experience's style rule (shared/tests/test_become_the_signal.py): no
dashes as punctuation, and nothing described by what it lacks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

REACH_DIRECT = "direct"
REACH_LIMITED = "limited"
REACH_NONE = "none"
REACH_LEVELS = (REACH_DIRECT, REACH_LIMITED, REACH_NONE)

READS_ACTIVITY = "activity"
READS_STRUCTURE = "structure"
READS_BODY_RESPONSE = "body response"
READS = (READS_ACTIVITY, READS_STRUCTURE, READS_BODY_RESPONSE)

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
class SystemReach:
    """How well one modality reads one system."""

    modality: str
    system: str
    level: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.level not in REACH_LEVELS:
            raise ValueError(
                f"{self.modality}/{self.system}: level must be one of {REACH_LEVELS}; got {self.level!r}."
            )


@dataclass(frozen=True)
class SensingModality:
    """One way of sensing the brain or body, and the systems it reaches."""

    key: str
    name: str
    short_name: str
    placement: str
    measures: str
    reads: str
    timing: str
    reach: tuple[SystemReach, ...]
    also_captures: str
    privacy_note: str
    refs: tuple[str, ...]
    in_journey: bool = False

    def __post_init__(self) -> None:
        if self.reads not in READS:
            raise ValueError(f"{self.key}: reads must be one of {READS}; got {self.reads!r}.")
        if not self.refs:
            raise ValueError(f"{self.key}: every modality needs at least one reference.")
        systems = [r.system for r in self.reach]
        if len(systems) != len(set(systems)):
            raise ValueError(f"{self.key}: a system appears twice in reach.")

    def level_for(self, system_key: str) -> SystemReach:
        for entry in self.reach:
            if entry.system == system_key:
                return entry
        raise UnknownSystemError(f"{self.key} has no reach entry for system {system_key!r}.")


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
        "software. New England Journal of Medicine, 381(17), 1684-1686.",
        "https://doi.org/10.1056/NEJMc1908881",
    ),
    Reference(
        "jasper",
        "Jasper, H. H. (1958). The ten-twenty electrode system of the International Federation. "
        "Electroencephalography and Clinical Neurophysiology, 10(2), 371-375.",
        "https://doi.org/10.1016/0013-4694(58)90053-1",
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
        "sensorimotor", "Sensorimotor cortex",
        "A strip over the top of the brain, on both sides of the central sulcus",
        "Sends movement commands and receives touch and body position. Motor imagery "
        "brain computer interfaces read this strip.",
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
        "Processes vision. The alpha rhythm in this recording is strongest here, and it grows "
        "when the eyes close.",
        DEPTH_SURFACE, ("kandel", "barry"),
    ),
    BrainSystem(
        "thalamus", "Thalamus", "Center of the brain, above the brainstem",
        "Relays sensory signals to the cortex and helps pace cortical rhythms such as alpha.",
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


def _reach(modality: str, levels: dict[str, str], notes: dict[str, str] | None = None) -> tuple[SystemReach, ...]:
    notes = notes or {}
    return tuple(
        SystemReach(modality, s.key, levels.get(s.key, REACH_NONE), notes.get(s.key, ""))
        for s in SYSTEMS
    )


MODALITIES: tuple[SensingModality, ...] = (
    SensingModality(
        key="eeg",
        name="Scalp electroencephalography",
        short_name="EEG",
        placement="Electrodes on the scalp",
        measures="Voltage from many cortical neurons firing in step, summed at the scalp.",
        reads=READS_ACTIVITY,
        timing="Milliseconds. Fast enough to follow each cycle of the 8 to 13 Hz alpha rhythm.",
        reach=_reach(
            "eeg",
            {**{k: REACH_DIRECT for k in CORTICAL},
             "thalamus": REACH_LIMITED, "hippocampus_amygdala": REACH_LIMITED, "cerebellum": REACH_LIMITED},
            {
                "thalamus": "Seen mostly through the cortical rhythms it helps pace.",
                "hippocampus_amygdala": "Detectable with many electrodes and source modeling, and weak at any single electrode.",
                "cerebellum": "Far from most scalp electrodes.",
            },
        ),
        also_captures="Patterns distinctive enough to identify a person from the signal itself.",
        privacy_note="A recording made for one task also carries information about attention, "
        "drowsiness, and health.",
        refs=("buzsaki", "seeber", "bagley"),
        in_journey=True,
    ),
    SensingModality(
        key="fnirs",
        name="Functional near-infrared spectroscopy",
        short_name="fNIRS",
        placement="Light sources and detectors on the scalp, often over the forehead",
        measures="Changes in blood oxygen in the outer cortex, from how much near-infrared light "
        "returns to the detector.",
        reads=READS_ACTIVITY,
        timing="Seconds. Blood flow follows neural activity with a delay.",
        reach=_reach(
            "fnirs",
            {k: REACH_DIRECT for k in CORTICAL},
            {k: "Outer cortex under the optodes. Many systems cover only the forehead." for k in CORTICAL},
        ),
        also_captures="Heartbeat and breathing, which ride on the blood oxygen signal.",
        privacy_note="Wearable and portable, so it can record outside the lab during everyday tasks.",
        refs=("pinti",),
    ),
    SensingModality(
        key="fmri",
        name="Functional magnetic resonance imaging",
        short_name="fMRI",
        placement="Inside an MRI scanner",
        measures="Changes in blood oxygen across the whole brain, called the BOLD signal.",
        reads=READS_ACTIVITY,
        timing="Seconds. The blood oxygen response peaks several seconds after neural activity.",
        reach=_reach("fmri", {k: REACH_DIRECT for k in BRAIN}),
        also_captures="A structural image of the head is usually acquired in the same session, "
        "and it includes the face.",
        privacy_note="Sharing the structural image alongside the functional data shares the face too.",
        refs=("logothetis", "schwarz"),
    ),
    SensingModality(
        key="smri",
        name="Structural magnetic resonance imaging",
        short_name="Structural MRI",
        placement="Inside an MRI scanner",
        measures="Shape and tissue of the brain and the whole head.",
        reads=READS_STRUCTURE,
        timing="One snapshot of anatomy. It records structure only, with no activity over time.",
        reach=_reach("smri", {k: REACH_DIRECT for k in BRAIN}),
        also_captures="The face. Face recognition software matched MRI face reconstructions to "
        "photographs for 70 of 84 research participants.",
        privacy_note="Removing the face from the image before sharing, called defacing, is the "
        "common safeguard.",
        refs=("schwarz",),
    ),
    SensingModality(
        key="peripheral",
        name="Peripheral body sensors (skin conductance and heart rate)",
        short_name="Body sensors",
        placement="Sensors on the skin, wrist, or chest",
        measures="Sweat gland activity and the timing of each heartbeat.",
        reads=READS_BODY_RESPONSE,
        timing="Seconds for skin conductance. Every beat for heart rate.",
        reach=_reach(
            "peripheral",
            {"autonomic": REACH_DIRECT, "brainstem": REACH_LIMITED},
            {"brainstem": "Read through the body responses it controls."},
        ),
        also_captures="Arousal and stress responses during everyday life.",
        privacy_note="Common in consumer wearables, so these signals are often collected far "
        "outside research settings.",
        refs=("critchley",),
    ),
)

CAVEAT = (
    "Reach levels are teaching simplifications drawn from the cited reviews. Real coverage "
    "depends on how many sensors there are, where they sit, and how the data are analyzed."
)
CAVEAT_REFS = ("buzsaki", "seeber", "pinti")

# ------------------------------------------------------------------
# The 10-20 electrode system
# ------------------------------------------------------------------

# Jasper (1958), with the later names T7, T8, P7, P8 for T3, T4, T5, T6.
STANDARD_10_20 = (
    "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "T7", "C3", "Cz", "C4", "T4", "T8",
    "T5", "P7", "P3", "Pz", "P4", "T6", "P8", "O1", "Oz", "O2",
)

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
    (Jasper, 1958). Koessler et al. (2009) confirm the regions by
    projecting electrode positions onto each person's cortex.
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
    region, system = _PREFIX[prefix]
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
    return ElectrodeSite(normalized, region, system, hemisphere, note, ("jasper", "koessler"))


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


def reach(modality_key: str, system_key: str) -> SystemReach:
    """How well one modality reads one system."""
    system(system_key)
    return modality(modality_key).level_for(system_key)


def atlas() -> NeuroanatomyAtlas:
    """Everything the figure needs, with only the references it cites."""
    used: list[str] = []
    for key in (
        *(k for s in SYSTEMS for k in s.refs),
        *(k for m in MODALITIES for k in m.refs),
        *electrode_site(RECORDING_ELECTRODE).refs,
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
        caveat=CAVEAT,
        caveat_refs=CAVEAT_REFS,
        references=tuple(by_key[k] for k in used),
    )
