"""
What is known about the Right To Play Pakistan trial, and how each thing
is known.

This module exists because the journey it serves is about a boundary
rather than about a result. The trial's baseline microdata are public and
its 24-month participant microdata are not, so a reader can interrogate
how the study was built and cannot reproduce what it found. A page that
blurred those two would teach the opposite of what the study is useful
for.

So every fact here carries its provenance, as one of three states, and
nothing is recorded without one:

    REPORTED       stated in a publication, cited on the fact itself
    CALCULATED     arithmetic OpenMeasure performed, from stated inputs
    NOT_ESTABLISHED  cannot be settled from the artifacts that exist

The third is not a gap to be filled later. It is the finding. A reader
who leaves knowing which parts of a published evaluation can be checked,
and which cannot, has learned the thing this journey is for.

Nothing here is derived from the data file. OpenMeasure cannot open an
SPSS .sav today, and the facts below come from the publications and from
the catalog entry in shared/datasets.py, each cited where it is used.
"""

from __future__ import annotations

from dataclasses import dataclass

# How a fact is known. A closed set, because the whole journey turns on
# telling these apart, and a fourth informal state would be a way to
# record something without saying where it came from.
REPORTED = "Reported by the publication"
CALCULATED = "Calculated by OpenMeasure"
NOT_ESTABLISHED = "Not established from the available artifacts"

PROVENANCE_STATES: frozenset[str] = frozenset(
    {REPORTED, CALCULATED, NOT_ESTABLISHED}
)

# The same three states, short enough to sit beside a fact rather than
# under it. Printing the full state and citation on every row turned six
# facts into six paragraphs, and the reader's question at that point is
# only which of the three this is; the citation is what they open when
# the answer surprises them.
PROVENANCE_BADGES: dict[str, str] = {
    REPORTED: "Publication",
    CALCULATED: "OpenMeasure",
    NOT_ESTABLISHED: "Not established",
}


BASELINE_CITATION = (
    "Karmaliani, R., McFarlane, J., Somani, R., Khuwaja, H. M. A., "
    "Bhamani, S. S., Ali, T. S., Gulzar, S., Somani, Y., Chirwa, E. D., & "
    "Jewkes, R. (2017). Peer violence perpetration and victimization: "
    "Prevalence, associated factors and pathways among 1752 sixth grade "
    "boys and girls in schools in Pakistan. PLOS ONE, 12(8), e0180833."
)

TRIAL_CITATION = (
    "Karmaliani, R., et al. (2020). Global Health Action, 13(1), 1836604."
)

CATALOG_ENTRY = "shared/datasets.py, right_to_play_baseline"


@dataclass(frozen=True)
class StudyFact:
    """One thing about the trial, and how it is known."""

    label: str
    value: str
    provenance: str
    source: str

    def __post_init__(self) -> None:
        for field_name in ("label", "value", "provenance", "source"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"{self.label or 'A fact'} is missing a value for "
                    f"'{field_name}'. A fact without a source is not a "
                    "fact this module records."
                )

        if self.provenance not in PROVENANCE_STATES:
            raise ValueError(
                f"{self.label} has provenance '{self.provenance}', which is "
                f"not one of the declared states: "
                f"{', '.join(sorted(PROVENANCE_STATES))}."
            )

    @property
    def is_established(self) -> bool:
        """Whether the artifacts settle this."""
        return self.provenance != NOT_ESTABLISHED

    @property
    def badge(self) -> str:
        """The short form, for sitting beside the fact rather than under it."""
        return PROVENANCE_BADGES[self.provenance]


DESIGN: tuple[StudyFact, ...] = (
    StudyFact(
        label="Design",
        value="Two-arm cluster randomized controlled trial",
        provenance=REPORTED,
        source=TRIAL_CITATION,
    ),
    StudyFact(
        label="Unit of randomization",
        value="The school, not the student",
        provenance=REPORTED,
        source=TRIAL_CITATION,
    ),
    StudyFact(
        label="Clusters",
        value="40 single-sex public schools in Hyderabad, Pakistan",
        provenance=REPORTED,
        source=BASELINE_CITATION,
    ),
    StudyFact(
        label="Participants",
        value="1,752 grade 6 students",
        provenance=REPORTED,
        source=BASELINE_CITATION,
    ),
    StudyFact(
        label="Follow-up",
        value="Outcomes reported at 24 months",
        provenance=REPORTED,
        source=TRIAL_CITATION,
    ),
    StudyFact(
        label="Which schools went to which arm",
        value=(
            "The baseline file carries School and Group, so arm assignment "
            "and cluster membership are inspectable rather than only "
            "described"
        ),
        provenance=REPORTED,
        source=CATALOG_ENTRY,
    ),
)

MEASUREMENT: tuple[StudyFact, ...] = (
    StudyFact(
        label="Peer victimization",
        value="Peer Victimization Scale",
        provenance=REPORTED,
        source=BASELINE_CITATION,
    ),
    StudyFact(
        label="Peer perpetration",
        value="Peer Perpetration Scale",
        provenance=REPORTED,
        source=BASELINE_CITATION,
    ),
    StudyFact(
        label="Depressive symptoms",
        value="Children's Depression Inventory, second edition (CDI-2)",
        provenance=REPORTED,
        source=BASELINE_CITATION,
    ),
    StudyFact(
        label="Item-level responses",
        value=(
            "Present in the baseline file, across 350 variables, so the "
            "instruments can be examined as administered"
        ),
        provenance=REPORTED,
        source=CATALOG_ENTRY,
    ),
)

@dataclass(frozen=True)
class MeasuredFacet:
    """One aspect of a construct, and the instrument standing in for it."""

    name: str
    instrument: str

    def __post_init__(self) -> None:
        if not self.name or not self.instrument:
            raise ValueError(
                f"{self.name or 'A facet'} needs both a name and an "
                "instrument. A facet with no instrument was not measured."
            )


@dataclass(frozen=True)
class MeasuredConstruct:
    """One thing the study set out to measure, and how it was split."""

    name: str
    facets: tuple[MeasuredFacet, ...]

    def __post_init__(self) -> None:
        if not self.facets:
            raise ValueError(f"{self.name} names no facets.")


# What the instruments stand in for, as a structure rather than a list.
#
# Peer violence was measured as two separate things, being victimized and
# perpetrating, on two separate scales. A list of three instruments hides
# that; drawn, it is the first thing a reader sees, and it is what makes
# the trial's outcomes legible.
MEASUREMENT_MAP: tuple[MeasuredConstruct, ...] = (
    MeasuredConstruct(
        name="Peer violence",
        facets=(
            MeasuredFacet("Being victimized", "Peer Victimization Scale"),
            MeasuredFacet("Perpetrating", "Peer Perpetration Scale"),
        ),
    ),
    MeasuredConstruct(
        name="Depressive symptoms",
        facets=(MeasuredFacet("Depressive symptoms", "CDI-2"),),
    ),
)


ARTIFACTS: tuple[StudyFact, ...] = (
    StudyFact(
        label="Baseline participant microdata",
        value=(
            "Published as S1 File with the baseline article, 1,752 rows by "
            "350 variables, SPSS .sav, CC BY"
        ),
        provenance=REPORTED,
        source=BASELINE_CITATION,
    ),
    StudyFact(
        label="Design information",
        value="Described in both publications",
        provenance=REPORTED,
        source=TRIAL_CITATION,
    ),
    StudyFact(
        label="Reported 24-month findings",
        value="Published in the trial results article",
        provenance=REPORTED,
        source=TRIAL_CITATION,
    ),
    StudyFact(
        label="Follow-up participant microdata",
        value=(
            "Not among the published artifacts. The public supplementary "
            "file is the baseline wave only"
        ),
        provenance=NOT_ESTABLISHED,
        source=BASELINE_CITATION,
    ),
    StudyFact(
        label="The reported treatment effect, independently recomputed",
        value=(
            "Cannot be produced from the public artifacts. Recomputing it "
            "needs the 24-month outcomes for each participant, which are "
            "not published"
        ),
        provenance=NOT_ESTABLISHED,
        source=BASELINE_CITATION,
    ),
    StudyFact(
        label="The original a priori power calculation",
        value=(
            "Not independently reproducible from what is reported. The "
            "inputs it rests on, including the assumed intracluster "
            "correlation, are not stated in the artifacts catalogued here"
        ),
        provenance=NOT_ESTABLISHED,
        source=CATALOG_ENTRY,
    ),
)


@dataclass(frozen=True)
class ReplicationBoundary:
    """What the public artifacts settle, and what they leave open."""

    established: tuple[StudyFact, ...]
    unresolved: tuple[StudyFact, ...]

    @property
    def lesson(self) -> str:
        """The point of walking through this study."""
        return (
            "A published study can be methodologically interpretable "
            "without being reproducible from its public artifacts. This "
            "one is: the design, the instruments and the baseline "
            "measurements are all open to inspection, and the finding "
            "cannot be recomputed from them."
        )


def replication_boundary() -> ReplicationBoundary:
    """Split the artifact list on whether the artifacts settle it."""
    return ReplicationBoundary(
        established=tuple(fact for fact in ARTIFACTS if fact.is_established),
        unresolved=tuple(
            fact for fact in ARTIFACTS if not fact.is_established
        ),
    )


# What OpenMeasure can and cannot do with the baseline file today, kept
# beside the study facts because a reader hits it immediately and it is
# not a property of the study.
BASELINE_FILE_SUPPORT = (
    "The baseline file is SPSS .sav, which OpenMeasure cannot read today: "
    "no reader for that format is installed. The link below is the "
    "published file, and the design and measurement stages of this "
    "journey describe it rather than opening it."
)

# Stated wherever this journey approaches an estimate. Program Evaluation
# assumes independent observations, and 1,752 students inside 40 schools
# are not independent. Naming the direction matters: the mistake is not
# neutral, it makes a result look more certain than it is.
CLUSTERING_LIMIT = (
    "Students in one school are more alike than students drawn at random, "
    "and this trial randomized schools rather than students. OpenMeasure's "
    "comparisons currently assume independent observations, so running one "
    "on clustered data like this would report an interval narrower, and a "
    "p-value smaller, than the design supports. Cluster-robust inference "
    "is a stated next step, not something this toolkit does yet."
)
