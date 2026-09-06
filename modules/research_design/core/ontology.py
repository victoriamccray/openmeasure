"""
The vocabulary a study plan is assembled from, independent of field.

Method Selection used to hold one study's measures. A researcher asking
about mental models in implementation research was offered pain ratings,
body maps and electrodermal activity, because the measures were the
chronic-pain example's and the example was the planner.

The fix is not more domain templates. It is a small ontology that any
study can be expressed in:

    Concept        what has to be observed
    Concept kind   what sort of thing that is
    Measure        a documented way of observing that sort of thing
    Modality       what kind of evidence the measure produces

Domain sits outside this, and deliberately does not reach into it. A
neuroscience study can hold imaging, behavioral and self-report measures;
an implementation study can hold interviews, administrative records and
observation. Letting a domain decide which modalities are available is
what produced the failure above, so nothing here takes a domain at all.

The join between a concept and its candidate measures is the concept's
kind, not its subject. "Mental-model structure", "spatial pain pattern"
and "protein abundance" are three subjects; the first is how a person
organizes what they know, the second is what a person experiences, the
third is a molecular state, and those three kinds are what decide which
instruments are even applicable.

On sources
----------
Every measure names how it is documented and gives terms that find that
documentation. What this module does not do is assert a specific citation
for each method: naming a method is a claim OpenMeasure can stand behind,
and attaching a year and page number it has not checked is not. A
researcher picking a method is pointed at the literature to verify it in,
through the same OpenAlex search the rest of the toolkit uses.

Nothing here is generated from the wording of a research question. The
library is curated and finite, a researcher chooses from it, and a
measure that is not in it is a gap to be filled deliberately rather than
invented on demand.
"""

from __future__ import annotations

from dataclasses import dataclass

# What kind of evidence a measure produces. A closed set, because a study
# drawing on several of these is a multimodal study and that is worth
# being able to see; an open set of free-text labels would make two
# spellings of the same modality look like two.
MODALITY_SELF_REPORT = "Self-report"
MODALITY_BEHAVIORAL = "Behavioral"
MODALITY_PHYSIOLOGICAL = "Physiological"
MODALITY_IMAGING = "Imaging"
MODALITY_MOLECULAR = "Molecular"
MODALITY_CLINICAL = "Clinical"
MODALITY_ENVIRONMENTAL = "Environmental"
MODALITY_COMPUTATIONAL = "Computational"
MODALITY_ADMINISTRATIVE = "Administrative"
# Distinct from self-report: a survey returns scores on items chosen in
# advance, while an elicitation method returns structure the researcher
# did not specify. Both come from a person, and they are not the same
# kind of evidence.
MODALITY_QUALITATIVE = "Qualitative/elicitation"

MODALITIES: tuple[str, ...] = (
    MODALITY_SELF_REPORT,
    MODALITY_BEHAVIORAL,
    MODALITY_PHYSIOLOGICAL,
    MODALITY_IMAGING,
    MODALITY_MOLECULAR,
    MODALITY_CLINICAL,
    MODALITY_ENVIRONMENTAL,
    MODALITY_COMPUTATIONAL,
    MODALITY_ADMINISTRATIVE,
    MODALITY_QUALITATIVE,
)

# What sort of thing a concept is. This is the join: it decides which
# measures are applicable, and it is about the concept rather than about
# the field the concept belongs to.
KIND_EXPERIENCE = "Something a person experiences or believes"
KIND_KNOWLEDGE_STRUCTURE = "How a person organizes what they know"
KIND_AGREEMENT = "How far a group converges on the same view"
KIND_BEHAVIOR = "Something a person or system does"
KIND_BODILY_PROCESS = "A process in the body"
KIND_BIOLOGICAL_STATE = "A molecular or cellular state"
KIND_SYSTEM_OUTPUT = "Something a computational system produces"
KIND_SETTING = "A condition of the environment or setting"
KIND_RECORDED_EVENT = "Something an organization already recorded"

CONCEPT_KINDS: tuple[str, ...] = (
    KIND_EXPERIENCE,
    KIND_KNOWLEDGE_STRUCTURE,
    KIND_AGREEMENT,
    KIND_BEHAVIOR,
    KIND_BODILY_PROCESS,
    KIND_BIOLOGICAL_STATE,
    KIND_SYSTEM_OUTPUT,
    KIND_SETTING,
    KIND_RECORDED_EVENT,
)


@dataclass(frozen=True)
class Measure:
    """
    One documented way of observing something, and what it costs.

    ``limitation`` is required and is the field this record exists for.
    A catalog of methods with no limitations attached reads as a menu of
    equally good options, and choosing between measurement approaches is
    almost entirely a matter of which limitation a study can live with.
    """

    name: str
    observes: tuple[str, ...]
    modality: str
    captures: str
    produces: str
    burden: str
    limitation: str
    documented_as: str
    search_terms: str

    def __post_init__(self) -> None:
        required = (
            "name",
            "modality",
            "captures",
            "produces",
            "burden",
            "limitation",
            "documented_as",
            "search_terms",
        )
        for field_name in required:
            if not getattr(self, field_name):
                raise ValueError(
                    f"{self.name or 'A measure'} is missing a value for "
                    f"'{field_name}'."
                )

        if self.modality not in MODALITIES:
            raise ValueError(
                f"{self.name} has modality '{self.modality}', which is not "
                f"one of the declared modalities: {', '.join(MODALITIES)}."
            )

        if not self.observes:
            raise ValueError(f"{self.name} observes no kind of concept.")

        unknown = [kind for kind in self.observes if kind not in CONCEPT_KINDS]
        if unknown:
            raise ValueError(
                f"{self.name} observes {unknown}, which are not declared "
                "concept kinds."
            )


MEASURES: tuple[Measure, ...] = (
    Measure(
        name="Semi-structured interview",
        observes=(KIND_EXPERIENCE, KIND_KNOWLEDGE_STRUCTURE),
        modality=MODALITY_QUALITATIVE,
        captures="What a person says they think, in their own framing",
        produces="Transcript text",
        burden="30 to 90 minutes per participant, plus transcription",
        limitation=(
            "Reaches what a person can articulate on request, which is not "
            "the same as what they know or do"
        ),
        documented_as="An established qualitative research method",
        search_terms="semi-structured interview qualitative research method",
    ),
    Measure(
        name="Structured survey",
        observes=(KIND_EXPERIENCE, KIND_AGREEMENT),
        modality=MODALITY_SELF_REPORT,
        captures="Standing on items the researcher chose in advance",
        produces="Item scores, one row per respondent",
        burden="Low per respondent, high to design and validate",
        limitation=(
            "Only measures what the items ask about, so it cannot surface a "
            "construct nobody thought to write an item for"
        ),
        documented_as="An established survey-methodology instrument type",
        search_terms="survey instrument development validation measurement",
    ),
    Measure(
        name="Think-aloud protocol",
        observes=(KIND_KNOWLEDGE_STRUCTURE, KIND_BEHAVIOR),
        modality=MODALITY_QUALITATIVE,
        captures="Reasoning as it happens, rather than recalled afterwards",
        produces="Verbal protocol, time-aligned to the task",
        burden="One session per participant, intensive to code",
        limitation=(
            "Speaking while working changes the work, and some expertise is "
            "automatic enough not to be verbalized at all"
        ),
        documented_as="An established protocol-analysis method",
        search_terms="think aloud protocol analysis verbal report method",
    ),
    Measure(
        name="Card sorting",
        observes=(KIND_KNOWLEDGE_STRUCTURE,),
        modality=MODALITY_QUALITATIVE,
        captures="How a person groups concepts relative to each other",
        produces="Clusters or rankings per participant",
        burden="20 to 45 minutes per participant",
        limitation=(
            "The cards are the researcher's vocabulary, so a concept absent "
            "from the deck cannot appear in the result"
        ),
        documented_as="An established knowledge-elicitation technique",
        search_terms="card sorting knowledge elicitation technique",
    ),
    Measure(
        name="Causal or cognitive mapping",
        observes=(KIND_KNOWLEDGE_STRUCTURE,),
        modality=MODALITY_QUALITATIVE,
        captures="Which things a person believes affect which others",
        produces="A directed network per participant",
        burden="45 to 90 minutes, often facilitated",
        limitation=(
            "Records believed relationships, which is a claim about the "
            "person rather than about the system they are describing"
        ),
        documented_as="An established cognitive-mapping method",
        search_terms="cognitive mapping causal map elicitation method",
    ),
    Measure(
        name="Delphi or group elicitation",
        observes=(KIND_AGREEMENT,),
        modality=MODALITY_QUALITATIVE,
        captures="Where a group converges after structured iteration",
        produces="Round-by-round ratings and a convergence summary",
        burden="Several rounds over weeks, high coordination cost",
        limitation=(
            "Convergence after feedback can be agreement about the process "
            "rather than shared understanding"
        ),
        documented_as="An established consensus method",
        search_terms="Delphi method consensus expert elicitation",
    ),
    Measure(
        name="Structured observation",
        observes=(KIND_BEHAVIOR, KIND_AGREEMENT),
        modality=MODALITY_BEHAVIORAL,
        captures="What people actually do, against a coding scheme",
        produces="Coded event counts or durations",
        burden="Observer time in the setting, plus reliability coding",
        limitation=(
            "Being observed changes behaviour, and the coding scheme decides "
            "in advance what counts as an event"
        ),
        documented_as="An established observational research method",
        search_terms="structured observation coding scheme interrater",
    ),
    Measure(
        name="Document or artifact analysis",
        observes=(KIND_RECORDED_EVENT, KIND_KNOWLEDGE_STRUCTURE),
        modality=MODALITY_ADMINISTRATIVE,
        captures="What an organization wrote down at the time",
        produces="Coded documents or extracted fields",
        burden="No participant burden, substantial analyst time",
        limitation=(
            "Records what was worth recording to whoever kept them, which is "
            "a selection nobody documented"
        ),
        documented_as="An established documentary-analysis method",
        search_terms="document analysis archival research method",
    ),
    Measure(
        name="Administrative records extract",
        observes=(KIND_RECORDED_EVENT,),
        modality=MODALITY_ADMINISTRATIVE,
        captures="Events a system already logs for its own purposes",
        produces="Rows per event or per person",
        burden="No participant burden, access negotiation instead",
        limitation=(
            "Collected to run a service rather than to answer a research "
            "question, so definitions can change without notice"
        ),
        documented_as="An established secondary-data source type",
        search_terms="administrative data research secondary use validity",
    ),
    Measure(
        name="Rating scale, repeated in daily life",
        observes=(KIND_EXPERIENCE,),
        modality=MODALITY_SELF_REPORT,
        captures="How something a person feels changes within that person",
        produces="One score per prompt, many per person",
        burden="Several prompts a day, adherence falls over weeks",
        limitation=(
            "Prompting asks a person to notice something, which can change "
            "the thing being reported"
        ),
        documented_as="An established ecological momentary assessment design",
        search_terms="ecological momentary assessment experience sampling",
    ),
    Measure(
        name="Electrodermal activity",
        observes=(KIND_BODILY_PROCESS,),
        modality=MODALITY_PHYSIOLOGICAL,
        captures="Sympathetic arousal, continuously",
        produces="A conductance time series",
        burden="Wearable sensor worn continuously",
        limitation=(
            "Arousal is not specific to one emotion or state, and movement "
            "and temperature both move the signal"
        ),
        documented_as="An established psychophysiological measure",
        search_terms="electrodermal activity skin conductance measurement",
    ),
    Measure(
        name="Heart rate and heart-rate variability",
        observes=(KIND_BODILY_PROCESS,),
        modality=MODALITY_PHYSIOLOGICAL,
        captures="Cardiac and autonomic activity over time",
        produces="Beat intervals and derived variability indices",
        burden="Wearable or chest sensor worn continuously",
        limitation=(
            "Derived indices depend heavily on the preprocessing chosen, "
            "which is rarely reported in enough detail to reproduce"
        ),
        documented_as="An established cardiovascular measure",
        search_terms="heart rate variability measurement preprocessing",
    ),
    Measure(
        name="Functional neuroimaging",
        observes=(KIND_BODILY_PROCESS, KIND_BEHAVIOR),
        modality=MODALITY_IMAGING,
        captures="Where activity changes during a task or at rest",
        produces="Voxelwise time series",
        burden="Scanner session per participant, high cost",
        limitation=(
            "Measures a haemodynamic proxy for neural activity, on a slower "
            "timescale than the processes usually being inferred"
        ),
        documented_as="An established neuroimaging modality",
        search_terms="functional MRI BOLD measurement validity",
    ),
    Measure(
        name="Assay of a biological sample",
        observes=(KIND_BIOLOGICAL_STATE,),
        modality=MODALITY_MOLECULAR,
        captures="Concentration or abundance in a collected sample",
        produces="Quantified values per sample",
        burden="Collection, storage and laboratory processing",
        limitation=(
            "Reflects the moment and site of collection, and batch effects "
            "can exceed the differences under study"
        ),
        documented_as="An established laboratory measurement type",
        search_terms="assay batch effect biological sample measurement",
    ),
    Measure(
        name="Clinical assessment or chart review",
        observes=(KIND_RECORDED_EVENT, KIND_BODILY_PROCESS),
        modality=MODALITY_CLINICAL,
        captures="What a clinician assessed or recorded",
        produces="Coded diagnoses, measurements or events",
        burden="Clinician time, or analyst time for review",
        limitation=(
            "Reflects who reached care and what was coded for billing, so "
            "absence of a code is not absence of the condition"
        ),
        documented_as="An established clinical data source type",
        search_terms="chart review clinical coding validity ascertainment",
    ),
    Measure(
        name="Environmental sensor",
        observes=(KIND_SETTING,),
        modality=MODALITY_ENVIRONMENTAL,
        captures="A condition of the place, independent of any participant",
        produces="A time series per site",
        burden="Installation and maintenance per site",
        limitation=(
            "Measures the sensor's location, which may not be where the "
            "participants actually were"
        ),
        documented_as="An established environmental monitoring approach",
        search_terms="environmental sensor exposure measurement error",
    ),
    Measure(
        name="Model evaluation on held-out data",
        observes=(KIND_SYSTEM_OUTPUT,),
        modality=MODALITY_COMPUTATIONAL,
        captures="How a system performs on cases it was not fitted on",
        produces="Predictions, errors and rates by subgroup",
        burden="Compute, plus a split that has to be defensible",
        limitation=(
            "Performance is a property of the evaluation set as much as of "
            "the model, and a split leaking structure inflates it"
        ),
        documented_as="An established model-evaluation practice",
        search_terms="held out evaluation data leakage model validation",
    ),
)


def measures_for(kind: str) -> tuple[Measure, ...]:
    """
    Every measure that can observe this kind of concept.

    Raises on an unknown kind rather than returning nothing, so a typo
    reads as a mistake rather than as a concept nothing can measure.
    """
    if kind not in CONCEPT_KINDS:
        raise ValueError(
            f"'{kind}' is not a known concept kind. Known kinds: "
            f"{'; '.join(CONCEPT_KINDS)}."
        )

    return tuple(measure for measure in MEASURES if kind in measure.observes)


def get_measure(name: str) -> Measure:
    """Return one measure by name, raising on an unknown one."""
    for measure in MEASURES:
        if measure.name == name:
            return measure

    raise ValueError(
        f"'{name}' is not a known measure. This library is curated and "
        "finite; a measure it lacks is a gap to add deliberately."
    )


def modalities_of(names: tuple[str, ...]) -> tuple[str, ...]:
    """
    Which kinds of evidence a set of measures produces, in declared order.

    Declared order rather than selection order, so two studies choosing
    the same measures in a different sequence describe themselves the
    same way.
    """
    chosen = {get_measure(name).modality for name in names}

    return tuple(modality for modality in MODALITIES if modality in chosen)
