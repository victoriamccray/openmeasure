"""
Real research datasets, and how each one reaches a workflow.

This began as a discovery catalog: entries described a dataset and left
the reader to go and get it. It is becoming the single place a dataset's
provenance and delivery are declared, so a workflow can offer real public
data without inventing its own loader and without losing the citation,
licence and access terms on the way. `delivery` is what carries that
change; entries that are still read-about-only simply declare
DELIVERY_UPLOAD_ONLY.

Reading about a dataset is still not running an analysis. This module
carries no module_key, is never passed to shared/handoff.py, and its page
is not part of shared/catalog.py's lifecycle stages. Nothing here should
make it look as though an analysis has been performed.

try_with names must match a Workflow.workflow value in shared/catalog.py
exactly, so a rename there cannot silently orphan a reference here.

Every entry is deliberately open-ended: a domain, a description, a
validation question worth asking, and where to get the data and its access
terms. No column names, steps, or expected results, because the point is
for the user to explore the dataset against the workflow themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.catalog import WORKFLOWS

# How a dataset can be obtained. A closed set because the access story
# changes what a reader can actually do next: "Open" means download it
# today, "Registration required" means an account but no approval step,
# "Controlled" means a data use agreement or comparable review gates access.
ACCESS_OPEN = "Open"
ACCESS_REGISTRATION_REQUIRED = "Registration required"
ACCESS_CONTROLLED = "Controlled"

ACCESS_LEVELS: frozenset[str] = frozenset(
    {ACCESS_OPEN, ACCESS_REGISTRATION_REQUIRED, ACCESS_CONTROLLED}
)

# How a dataset reaches a workflow, which is a different question from
# whether a reader may obtain it. A closed set rather than a handful of
# booleans, because booleans permit combinations that cannot exist: a
# dataset cannot be both bundled and upload-only, and "has a local copy"
# means nothing without knowing whether that copy may legally be there.
#
# DELIVERY_REMOTE_FETCH   the app retrieves it on a user's action
# DELIVERY_BUNDLED_PUBLIC a copy is checked into this repository
# DELIVERY_CACHED_PUBLIC  fetched once at runtime and held, not committed
# DELIVERY_UPLOAD_ONLY    the reader obtains their own copy and uploads it
DELIVERY_REMOTE_FETCH = "Fetched on request"
DELIVERY_BUNDLED_PUBLIC = "Bundled with OpenMeasure"
DELIVERY_CACHED_PUBLIC = "Fetched once and cached"
DELIVERY_UPLOAD_ONLY = "You supply the file"

DELIVERY_MODES: frozenset[str] = frozenset(
    {
        DELIVERY_REMOTE_FETCH,
        DELIVERY_BUNDLED_PUBLIC,
        DELIVERY_CACHED_PUBLIC,
        DELIVERY_UPLOAD_ONLY,
    }
)

# The two modes that put a copy somewhere other than the reader's own
# machine, and therefore require permission to redistribute. A runtime
# cache is included deliberately: on a hosted deployment one fetch serves
# every visitor, which is closer to distributing the file than to a
# reader keeping their own download. Conservative on purpose; relaxing it
# for a specific dataset should follow a specific reading of its terms,
# not a general assumption.
_DELIVERY_REQUIRING_REDISTRIBUTION: frozenset[str] = frozenset(
    {DELIVERY_BUNDLED_PUBLIC, DELIVERY_CACHED_PUBLIC}
)

_WORKFLOW_NAMES: frozenset[str] = frozenset(item.workflow for item in WORKFLOWS)


@dataclass(frozen=True)
class DataSource:
    """One place to obtain or read about a dataset."""

    label: str
    url: str

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("A DataSource must have a label.")

        if not self.url.startswith("https://"):
            raise ValueError(
                f"'{self.label}' has url '{self.url}', which is not an "
                "https link. Source links must be verifiable, not relative "
                "or unsecured."
            )


@dataclass(frozen=True)
class RealDataset:
    """
    One real dataset, and how it connects to OpenMeasure.

    sources is a tuple rather than a single link because some real datasets,
    the wastewater surveillance equity data among them, are not one
    download: the underlying data is a linkage of several public sources,
    and naming only one would misrepresent how it is actually obtained.
    """

    id: str
    name: str
    domain: str
    description: str
    try_with: tuple[str, ...]
    explore_question: str
    access: str
    sources: tuple[DataSource, ...]

    # How this dataset reaches a workflow, and whether OpenMeasure may
    # keep a copy of it. Both are required rather than defaulted: the
    # safe value differs per dataset, and a silent default is exactly the
    # provenance loss this catalog exists to prevent.
    delivery: str
    redistribution_permitted: bool

    citation: str = ""

    def __post_init__(self) -> None:
        for field_name in ("id", "name", "domain", "description", "explore_question"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"{self.id or 'A dataset'} is missing a value for "
                    f"'{field_name}'."
                )

        if not self.try_with:
            raise ValueError(f"{self.id} does not name any workflow to try it with.")

        unknown = [name for name in self.try_with if name not in _WORKFLOW_NAMES]
        if unknown:
            raise ValueError(
                f"{self.id} names workflow(s) {unknown} in try_with, which do "
                f"not match any workflow in shared/catalog.py. Known "
                f"workflows: {sorted(_WORKFLOW_NAMES)}."
            )

        if self.access not in ACCESS_LEVELS:
            raise ValueError(
                f"{self.id} has access '{self.access}', which is not one of "
                f"the declared access levels: {', '.join(sorted(ACCESS_LEVELS))}."
            )

        if self.delivery not in DELIVERY_MODES:
            raise ValueError(
                f"{self.id} has delivery '{self.delivery}', which is not one "
                f"of the declared delivery modes: "
                f"{', '.join(sorted(DELIVERY_MODES))}."
            )

        # Access and redistribution are independent, and conflating them
        # is the mistake this guard exists to catch. HealthRing is the
        # case in this repository: openly downloadable from Zenodo, and
        # deliberately not redistributed here (see
        # modules/healthring/sample_data/README.md). "Anyone may obtain
        # it" does not imply "OpenMeasure may hand it out".
        if (
            self.delivery in _DELIVERY_REQUIRING_REDISTRIBUTION
            and not self.redistribution_permitted
        ):
            raise ValueError(
                f"{self.id} is delivered as '{self.delivery}', which keeps a "
                "copy outside the reader's own machine, but its terms do not "
                "permit redistribution. Use "
                f"'{DELIVERY_REMOTE_FETCH}' or '{DELIVERY_UPLOAD_ONLY}', or "
                "record permission explicitly."
            )

        if not self.sources:
            raise ValueError(f"{self.id} lists no sources.")


DATASETS: tuple[RealDataset, ...] = (
    RealDataset(
        id="healthring",
        name="HealthRing (RingDatasetV2.1)",
        domain="Wearable sensing / digital health",
        description=(
            "Synchronized PPG and accelerometer signals from two "
            "smart-ring designs, time-aligned to reference heart rate, "
            "respiratory rate, SpO2, and blood pressure across 54 adults "
            "in stationary, motion, low-oxygen-simulation, and "
            "treadmill-running scenarios. Best suited to algorithm-"
            "development questions rather than population-level "
            "physiology; RingTool, published alongside the dataset, "
            "documents its preprocessing pipeline."
        ),
        try_with=("Time-Series QA",),
        explore_question=(
            "How much of the recorded signal is actually usable once "
            "sampling gaps and coverage are checked across different "
            "activity conditions and the two ring designs?"
        ),
        access=ACCESS_OPEN,
        # Openly downloadable, and deliberately not redistributed here:
        # modules/healthring/sample_data/README.md records that decision
        # and checks in no excerpt.
        delivery=DELIVERY_UPLOAD_ONLY,
        redistribution_permitted=False,
        sources=(
            DataSource(
                label="Zenodo record (RingDatasetV2.1)",
                url="https://doi.org/10.5281/zenodo.18426864",
            ),
        ),
        citation=(
            "Tang, J., Wang, K., Ding, Y., Ji, J., Wang, Y., Wang, Z., "
            "Zhang, X., Chen, P., Gao, N., Shi, Y., & Wang, Y. (2026). "
            "HealthRing: Physiology dataset for health sensing on rings. "
            "Scientific Data. https://doi.org/10.1038/s41597-026-07289-x"
        ),
    ),
    RealDataset(
        id="portable_mri_volumetrics",
        name="Ultra-low-field portable MRI volumetrics",
        domain="Neuroimaging / measurement validation",
        description=(
            "Repeated structural brain volume measurements from a 64 mT "
            "portable MRI scanner in neurologically typical adults, taken "
            "across two processing software versions, intended to examine "
            "test-retest reproducibility."
        ),
        try_with=("Reliability",),
        explore_question=(
            "How consistent are repeated regional volume measurements "
            "across sessions and across the two software versions?"
        ),
        access=ACCESS_CONTROLLED,
        # A data use agreement gates access, so nothing can be fetched or
        # held on a reader's behalf.
        delivery=DELIVERY_UPLOAD_ONLY,
        redistribution_permitted=False,
        sources=(
            DataSource(
                label="Vivli study record (data use agreement required)",
                url="https://doi.org/10.25934/PR00012002",
            ),
        ),
        citation=(
            "Stockbridge, M. D., Wang, R., Neal, V., Diaz-Carr, I., "
            "Hillis, A. E., & Faria, A. V. (2026). Reproducibility of "
            "volumetric analysis using ultra-low-field portable magnetic "
            "resonance imaging. Aperture Neuro, 6. "
            "https://doi.org/10.52294/001c.165475"
        ),
    ),
    RealDataset(
        id="wastewater_surveillance_equity",
        name="NY State wastewater surveillance, linked to social vulnerability",
        domain="Public health surveillance / environmental equity",
        description=(
            "Site-level SARS-CoV-2 wastewater surveillance data from New "
            "York State's statewide network, published at the sewershed "
            "and county level, pairable at the census-tract level with the "
            "CDC/ATSDR Social Vulnerability Index and Environmental "
            "Justice Index. The same kind of linkage was used to study "
            "equity in surveillance coverage and outbreak detection in "
            "New York, though that study also drew on the network's own "
            "internal sewered/unsewered coverage records, which are not "
            "part of this public feed."
        ),
        try_with=("Fairness", "Cross-Analysis Implications"),
        explore_question=(
            "Does surveillance coverage, or how quickly an outbreak would "
            "be detected, differ across communities at different levels of "
            "social vulnerability, and what would it take to show that "
            "convincingly rather than just suggestively?"
        ),
        access=ACCESS_OPEN,
        # Public government feeds, but this entry is a linkage of three
        # separate sources and their reuse terms have not been read
        # individually. Upload-only until they have been; loosening this
        # should follow a specific reading, not an assumption about
        # government data in general.
        delivery=DELIVERY_UPLOAD_ONLY,
        redistribution_permitted=False,
        sources=(
            DataSource(
                label="NY State statewide wastewater surveillance data (Health Data NY)",
                url="https://health.data.ny.gov/Health/New-York-State-Statewide-COVID-19-Wastewater-Surve/hdxs-icuh",
            ),
            DataSource(
                label="CDC NWSS public wastewater metric data (national context)",
                url="https://data.cdc.gov/Public-Health-Surveillance/NWSS-Public-SARS-CoV-2-Wastewater-Metric-Data/2ew6-ywp6",
            ),
            DataSource(
                label="CDC/ATSDR Social Vulnerability Index & Environmental Justice Index downloads",
                url="https://www.atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html",
            ),
        ),
        citation=(
            "Neyra Blatz, M., Pulido, N., Asiedu-Danso, M., Hill, D. T., "
            "Rose, M. G., Zhu, Y., Pollack Porter, K. M., & Larsen, D. A. "
            "(2026). Equities and inequities inherent in wastewater "
            "surveillance systems for public health: New York State, "
            "2020-2024. American Journal of Public Health. "
            "https://doi.org/10.2105/AJPH.2026.308472"
        ),
    ),
)
