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

# How the data was observed, and how it is arranged. Two facets rather
# than one, because they answer different questions and a reader looking
# for a dataset is usually holding one of them: "what does a wearable
# study look like" is a different search from "what does a repeated-
# measures file look like".
#
# Closed sets, so browsing by a facet cannot silently miss a dataset
# filed under a near-synonym. Both are classifications drawn from each
# dataset's own description, not measurements of the file.
MODALITY_SURVEY = "Survey or questionnaire"
MODALITY_WEARABLE = "Wearable sensor"
MODALITY_FIXED_SENSOR = "Fixed sensor"
MODALITY_ENVIRONMENTAL = "Environmental sampling"
MODALITY_CLINICAL_RECORDS = "Clinical records"
MODALITY_IMAGING = "Imaging"

MODALITIES: frozenset[str] = frozenset(
    {
        MODALITY_SURVEY,
        MODALITY_WEARABLE,
        MODALITY_FIXED_SENSOR,
        MODALITY_ENVIRONMENTAL,
        MODALITY_CLINICAL_RECORDS,
        MODALITY_IMAGING,
    }
)

STRUCTURE_CROSS_SECTIONAL = "Cross-sectional"
STRUCTURE_TIME_SERIES = "Time series"
STRUCTURE_REPEATED_MEASURES = "Repeated measures"
STRUCTURE_RANDOMIZED_TRIAL = "Randomized trial"
STRUCTURE_LINKED_SOURCES = "Linked sources"

STRUCTURES: frozenset[str] = frozenset(
    {
        STRUCTURE_CROSS_SECTIONAL,
        STRUCTURE_TIME_SERIES,
        STRUCTURE_REPEATED_MEASURES,
        STRUCTURE_RANDOMIZED_TRIAL,
        STRUCTURE_LINKED_SOURCES,
    }
)

# The longest a scale fact can be and still fit the strip that draws it.
# SVG text does not wrap, so a longer one is lost off the canvas rather
# than clipped.
SCALE_FACT_LIMIT = 30

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

    # How the data was observed and how it is arranged, so the catalog
    # can be browsed by something other than which workflow suits it.
    modality: str
    structure: str

    # A few short facts about this dataset's size and shape, drawn as a
    # strip above its question. Every one is already stated in the
    # description above it and was verified against the source when the
    # entry was written; this promotes them into a slot that can be
    # drawn, and adds nothing.
    #
    # Empty is a legitimate value. Several entries state no counts,
    # because none were verified, and an empty strip says that more
    # honestly than an estimate would.
    scale: tuple[str, ...] = ()

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

        if self.modality not in MODALITIES:
            raise ValueError(
                f"{self.id} has modality '{self.modality}', which is not "
                f"one of the declared modalities: "
                f"{', '.join(sorted(MODALITIES))}."
            )

        if self.structure not in STRUCTURES:
            raise ValueError(
                f"{self.id} has structure '{self.structure}', which is not "
                f"one of the declared structures: "
                f"{', '.join(sorted(STRUCTURES))}."
            )

        for fact in self.scale:
            if not fact.strip():
                raise ValueError(f"{self.id} has an empty scale fact.")
            if len(fact) > SCALE_FACT_LIMIT:
                raise ValueError(
                    f"{self.id}'s scale fact '{fact}' is longer than "
                    f"{SCALE_FACT_LIMIT} characters and would run off the "
                    "strip that draws it."
                )

        if not self.sources:
            raise ValueError(f"{self.id} lists no sources.")


DATASETS: tuple[RealDataset, ...] = (
    RealDataset(
        id="healthring",
        modality=MODALITY_WEARABLE,
        structure=STRUCTURE_TIME_SERIES,
        scale=(
            "54 adults",
            "PPG and accelerometer",
            "4 activity scenarios",
        ),
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
        modality=MODALITY_IMAGING,
        structure=STRUCTURE_REPEATED_MEASURES,
        scale=(
            "Repeated sessions",
            "2 software versions",
        ),
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
        modality=MODALITY_ENVIRONMENTAL,
        structure=STRUCTURE_LINKED_SOURCES,
        scale=(
            "Sewershed and county level",
            "Pairs with SVI and EJI",
        ),
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
    RealDataset(
        id="diabetes_130_hospitals",
        modality=MODALITY_CLINICAL_RECORDS,
        structure=STRUCTURE_CROSS_SECTIONAL,
        scale=(
            "101,766 encounters",
            "130 hospitals",
            "47 features",
        ),
        name="Diabetes 130-US hospitals, 1999-2008",
        domain="Clinical care / health services",
        description=(
            "101,766 hospital encounters for patients with diabetes across "
            "130 US hospitals, with 47 features covering demographics, "
            "admission and discharge details, diagnoses, medications and "
            "whether the patient was readmitted within 30 days. It records "
            "what happened to real patients, and contains no predictions "
            "from any deployed model, which is what decides the fairness "
            "questions it can answer. Race is recorded and has missing "
            "values, documented as such at source."
        ),
        try_with=("Fairness",),
        explore_question=(
            "Does 30-day readmission differ across racial groups, and what "
            "would it take to say that a difference reflects unequal care "
            "rather than unequal case mix?"
        ),
        access=ACCESS_OPEN,
        # CC BY 4.0, so redistribution is established rather than assumed:
        # the first entry in this catalog for which bundling or caching
        # would be lawful. Fetched on request all the same, because 101,766
        # rows is not a repository-sized file and nothing yet needs a local
        # copy. Revisit if performance, not licence, makes the case.
        delivery=DELIVERY_REMOTE_FETCH,
        redistribution_permitted=True,
        sources=(
            DataSource(
                label="UCI Machine Learning Repository (CC BY 4.0)",
                url="https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
            ),
            DataSource(
                label="Dataset DOI",
                url="https://doi.org/10.24432/C5230J",
            ),
        ),
        citation=(
            "Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). "
            "Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI "
            "Machine Learning Repository. "
            "https://doi.org/10.24432/C5230J"
        ),
    ),
    RealDataset(
        id="nhanes_dpq_phq9",
        modality=MODALITY_SURVEY,
        structure=STRUCTURE_CROSS_SECTIONAL,
        scale=(
            "6,337 participants",
            "9 items scored 0 to 3",
            "5,455 complete cases",
        ),
        name="NHANES depression screener (PHQ-9), 2021-2023",
        domain="Population health survey / measurement",
        description=(
            "Item-level responses from 6,337 NHANES participants to the "
            "nine-item PHQ-9 depression screener, each item scored 0 to 3. "
            "The instrument is freely available, so items can be shown by "
            "their actual wording rather than by variable name. Three "
            "things need deciding before an internal-consistency estimate "
            "means anything: codes 7 and 9 are Refused and Don't know "
            "rather than scores and affect 53 respondents; DPQ100 measures "
            "functional difficulty and is not one of the nine scored "
            "items; and pandas reads the file's zero category as a "
            "denormalized float near 5.4e-79 rather than 0. 5,455 "
            "respondents answered all nine items with a valid score."
        ),
        try_with=("Reliability",),
        explore_question=(
            "Do the nine items hold together well enough to be summed into "
            "one depression score, and which of them behaves least like "
            "the rest?"
        ),
        access=ACCESS_OPEN,
        # A US federal public-use file, not subject to domestic copyright,
        # so redistribution is established. Fetched on request anyway: the
        # file is small, and nothing yet needs a local copy. Read with
        # pandas.read_sas(format="xport"), which needs no dependency this
        # project does not already have.
        delivery=DELIVERY_REMOTE_FETCH,
        redistribution_permitted=True,
        sources=(
            DataSource(
                label="DPQ_L codebook and variable list (NCHS)",
                url="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DPQ_L.htm",
            ),
            DataSource(
                label="DPQ_L data file (SAS transport, .xpt)",
                url="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DPQ_L.xpt",
            ),
        ),
        citation=(
            "Centers for Disease Control and Prevention, National Center "
            "for Health Statistics. National Health and Nutrition "
            "Examination Survey, 2021-2023: Mental Health - Depression "
            "Screener (DPQ_L). Hyattsville, MD: U.S. Department of Health "
            "and Human Services."
        ),
    ),
    RealDataset(
        id="right_to_play_baseline",
        modality=MODALITY_SURVEY,
        structure=STRUCTURE_RANDOMIZED_TRIAL,
        scale=(
            "1,752 students",
            "40 schools",
            "350 variables",
        ),
        name="Right To Play Pakistan cluster RCT, baseline",
        domain="Violence prevention / school-based intervention",
        description=(
            "Baseline responses from all 1,752 grade 6 students in a "
            "two-arm cluster randomized trial across 40 single-sex public "
            "schools in Hyderabad, Pakistan, verified at 1,752 rows by 350 "
            "variables. It carries School, Group and Gender, so the "
            "clustering and the randomized arm assignment are inspectable "
            "rather than only described, alongside item-level responses to "
            "the Peer Victimization Scale, the Peer Perpetration Scale and "
            "the CDI-2. Distributed as an SPSS .sav file. Children were "
            "asked about being victimized, about perpetrating violence, and "
            "about depression; the responses are de-identified and were "
            "published by the authors for reuse, and a reader should know "
            "that before opening them."
        ),
        try_with=("Impact Evaluation",),
        explore_question=(
            "The trial reports large reductions in peer violence at 24 "
            "months. Which parts of that finding can be checked against "
            "the artifacts the study actually published?"
        ),
        access=ACCESS_OPEN,
        # Published as supporting information to a CC BY article, and PLOS
        # applies that licence to the works it publishes, so redistribution
        # is established. Fetched on request all the same: nothing needs a
        # local copy, and OpenMeasure cannot currently read .sav at all.
        # Whether to add that capability is an integration decision, kept
        # separate from cataloguing what the dataset is.
        delivery=DELIVERY_REMOTE_FETCH,
        redistribution_permitted=True,
        sources=(
            DataSource(
                label="S1 File, baseline dataset for all 1,752 participants (.sav)",
                url="https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0180833.s001&type=supplementary",
            ),
            DataSource(
                label="Baseline article, instruments and licence (PLOS ONE, CC BY)",
                url="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180833",
            ),
            DataSource(
                label="Trial results at 24 months (Global Health Action)",
                url="https://doi.org/10.1080/16549716.2020.1836604",
            ),
        ),
        citation=(
            "Karmaliani, R., McFarlane, J., Somani, R., Khuwaja, H. M. A., "
            "Bhamani, S. S., Ali, T. S., Gulzar, S., Somani, Y., Chirwa, "
            "E. D., & Jewkes, R. (2017). Peer violence perpetration and "
            "victimization: Prevalence, associated factors and pathways "
            "among 1752 sixth grade boys and girls in schools in Pakistan. "
            "PLOS ONE, 12(8), e0180833. Trial results: Karmaliani, R., et "
            "al. (2020). Global Health Action, 13(1), 1836604."
        ),
    ),
    RealDataset(
        id="nwss_wastewater_metrics",
        modality=MODALITY_ENVIRONMENTAL,
        structure=STRUCTURE_TIME_SERIES,
        scale=(
            "837,382 rows",
            "One site, one series",
        ),
        name="CDC NWSS public SARS-CoV-2 wastewater metrics",
        domain="Public health surveillance",
        description=(
            "Site-level SARS-CoV-2 wastewater measurements from the "
            "National Wastewater Surveillance System, 837,382 rows across "
            "all reporting sites, each row a site and a date with a "
            "percentile and a 15-day percent change. One site is one time "
            "series, selected through key_plot_id, and choosing which site "
            "to inspect is the reader's first decision. Values arrive as "
            "text and need parsing, and a -99 in ptc_15d is a sentinel "
            "rather than a measurement. Checked across 400 sites, every "
            "one had complete daily coverage: these series are placed on a "
            "regular grid upstream, so they show what an intact time axis "
            "looks like and cannot demonstrate outages, duplicate "
            "timestamps or out-of-order rows."
        ),
        try_with=("Time-Series QA",),
        explore_question=(
            "What does a healthy surveillance series look like on the "
            "checks this module runs, and what would have to be different "
            "about the data for any of them to fire?"
        ),
        access=ACCESS_OPEN,
        # CDC data, a US federal work not subject to domestic copyright,
        # so redistribution is established on the same footing as NHANES.
        # Fetched on request and necessarily so: the full extract is
        # 837,382 rows, and the Socrata API takes a $where filter, so a
        # single site's series is a request rather than a download.
        delivery=DELIVERY_REMOTE_FETCH,
        redistribution_permitted=True,
        sources=(
            DataSource(
                label="Dataset page (data.cdc.gov)",
                url="https://data.cdc.gov/Public-Health-Surveillance/NWSS-Public-SARS-CoV-2-Wastewater-Metric-Data/2ew6-ywp6",
            ),
            DataSource(
                label="Socrata API endpoint, filterable by site",
                url="https://data.cdc.gov/resource/2ew6-ywp6.json",
            ),
        ),
        citation=(
            "Centers for Disease Control and Prevention. NWSS Public "
            "SARS-CoV-2 Wastewater Metric Data. National Wastewater "
            "Surveillance System, data.cdc.gov dataset 2ew6-ywp6."
        ),
    ),
    RealDataset(
        id="noaa_lcd_hourly",
        modality=MODALITY_FIXED_SENSOR,
        structure=STRUCTURE_TIME_SERIES,
        scale=(
            "One station at a time",
            "4 report types interleaved",
        ),
        name="NOAA Local Climatological Data, hourly station observations",
        domain="Weather observation / sensor time series",
        description=(
            "Hourly surface weather observations from a single NOAA "
            "station, fetched for a chosen station and date range. The "
            "irregularity here is the measurement process rather than "
            "anything added: a quarter of observations from station "
            "70026027502 contains 43 gaps longer than 90 minutes, the "
            "longest 10 hours, alongside genuine jitter in observation "
            "times at 58 and 62 minutes rather than an exact hourly grid. "
            "Each file interleaves four report types, FM-15 hourly, FM-16 "
            "special, SOD daily summary and SOM monthly, so deciding which "
            "rows form the series is the reader's first step and the "
            "totals mean nothing until it is made."
        ),
        try_with=("Time-Series QA",),
        explore_question=(
            "Where did this station stop reporting, and can you tell an "
            "instrument outage from a period that was never scheduled to "
            "be observed?"
        ),
        access=ACCESS_OPEN,
        # NOAA/NCEI data, a US federal work not subject to domestic
        # copyright. Fetched on request per station and date range, which
        # is how the access service is designed; there is no single file
        # to bundle.
        delivery=DELIVERY_REMOTE_FETCH,
        redistribution_permitted=True,
        sources=(
            DataSource(
                label="Local Climatological Data product page (NCEI)",
                url="https://www.ncei.noaa.gov/products/land-based-station/local-climatological-data",
            ),
            DataSource(
                label="NCEI access service, filterable by station and date",
                url="https://www.ncei.noaa.gov/access/services/data/v1",
            ),
        ),
        citation=(
            "NOAA National Centers for Environmental Information. Local "
            "Climatological Data (LCD). Asheville, NC: U.S. Department of "
            "Commerce, National Oceanic and Atmospheric Administration."
        ),
    ),
    RealDataset(
        id="openmesh_nyc",
        modality=MODALITY_FIXED_SENSOR,
        structure=STRUCTURE_TIME_SERIES,
        scale=(
            "NetCDF",
            "Outages documented at source",
        ),
        name="OpenMesh urban weather sensing, New York City",
        domain="Urban sensing / wireless signal time series",
        description=(
            "Wireless signal measurements from a community mesh network "
            "across New York City, published in NetCDF with its collection "
            "failures documented by the authors rather than cleaned away. "
            "About 30% of sublinks began collecting on 7 November, leaving "
            "an initial 10-day gap after the nominal start; the collection "
            "system then suffered a hardware failure lasting a week in "
            "early March, producing missing or corrupted samples. Further "
            "outages come from hardware faults and weather-induced signal "
            "attenuation, encoded as NaN. Unusual among public datasets in "
            "treating its own gaps as part of what it reports."
        ),
        try_with=("Time-Series QA",),
        explore_question=(
            "Which of these gaps are equipment failure, which are weather "
            "attenuating the signal being measured, and can the record "
            "alone tell them apart?"
        ),
        access=ACCESS_OPEN,
        # CC BY 4.0 on Zenodo, stated in the article's data availability
        # statement, so redistribution is established. Fetched on request:
        # NetCDF is a format OpenMeasure cannot currently read, which the
        # description says, and adding that capability is an integration
        # decision rather than a cataloguing one.
        delivery=DELIVERY_REMOTE_FETCH,
        redistribution_permitted=True,
        sources=(
            DataSource(
                label="Dataset on Zenodo (CC BY 4.0, NetCDF)",
                url="https://doi.org/10.5281/zenodo.15287692",
            ),
            DataSource(
                label="Data descriptor paper (Earth System Science Data)",
                url="https://doi.org/10.5194/essd-18-5817-2026",
            ),
        ),
        citation=(
            "Jacoby, D., Yu, S., Hu, Q., Hine, Z., Johnson, R., "
            "Ostrometzky, J., Kadota, I., Zussman, G., & Messer, H. "
            "(2026). OpenMesh: wireless signal dataset for opportunistic "
            "urban weather sensing in New York City. Earth System Science "
            "Data, 18, 5817-5836. "
            "https://doi.org/10.5194/essd-18-5817-2026"
        ),
    ),
)


DATASET_IDS: tuple[str, ...] = tuple(dataset.id for dataset in DATASETS)

_DATASET_BY_ID: dict[str, RealDataset] = {d.id: d for d in DATASETS}


def get_dataset(dataset_id: str) -> RealDataset:
    """
    Return one catalogued dataset by id.

    Raises on an unknown id rather than returning None, so a typo in a
    loader registry or a page fails at the call site instead of quietly
    presenting nothing where a dataset was meant to appear.
    """
    if dataset_id not in _DATASET_BY_ID:
        raise ValueError(
            f"'{dataset_id}' is not a catalogued dataset. Known datasets: "
            f"{', '.join(DATASET_IDS)}."
        )

    return _DATASET_BY_ID[dataset_id]
