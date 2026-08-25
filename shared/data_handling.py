"""
Data-handling disclosure: what happens to data on each page, in one place.

Each page that touches a dataset states, in its own words, roughly the
same five facts in its own docstring or expander: where the data comes
from, whether it ever touches disk, whether anything persists, what
fields are actually read, and whether any third-party source is bundled.
That is accurate but decentralized - there is no shared vocabulary, no
single place to see every page at once, and nothing ties the claims to a
closed set of possible answers.

This module is the shared vocabulary. Each page's disclosure is a
DataHandlingDisclosure drawn from DISCLOSURES below, rendered consistently
via render_data_handling_summary(), with the full picture (every page at
once, plus known limitations) on pages/Privacy_and_Data_Access.py.

Deliberately not merged into shared/report.py: that module is analytical
reporting (verdicts, diagnostics, caveats on a computed result); this one
is about what happened to the data before any analysis ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

DATA_ACCESS_USER_UPLOAD = "User upload"
DATA_ACCESS_LOCAL = "Local"
DATA_ACCESS_PUBLIC_REMOTE = "Public remote"
DATA_ACCESS_BUNDLED_EXAMPLE = "Bundled example"

DATA_ACCESS_VALUES: tuple[str, ...] = (
    DATA_ACCESS_USER_UPLOAD,
    DATA_ACCESS_LOCAL,
    DATA_ACCESS_PUBLIC_REMOTE,
    DATA_ACCESS_BUNDLED_EXAMPLE,
)

PROCESSING_IN_MEMORY = "In memory"
PROCESSING_TEMPORARY_FILE = "Temporary file"

PROCESSING_VALUES: tuple[str, ...] = (PROCESSING_IN_MEMORY, PROCESSING_TEMPORARY_FILE)

PERSISTENT_STORAGE_NONE = "None"

# A closed set of one today. Kept as a set rather than a hardcoded string
# so that a future page reporting something other than "None" is a
# deliberate, validated addition here, not a silent one.
PERSISTENT_STORAGE_VALUES: tuple[str, ...] = (PERSISTENT_STORAGE_NONE,)

REDISTRIBUTION_BUNDLED = "Bundled"
REDISTRIBUTION_NOT_BUNDLED = "Not bundled"
# No third-party source dataset is involved at all - e.g. a page that only
# processes whatever CSV a user brings has nothing to redistribute.
REDISTRIBUTION_NOT_APPLICABLE = "Not applicable"

REDISTRIBUTION_VALUES: tuple[str, ...] = (
    REDISTRIBUTION_BUNDLED,
    REDISTRIBUTION_NOT_BUNDLED,
    REDISTRIBUTION_NOT_APPLICABLE,
)


@dataclass(frozen=True)
class DataHandlingDisclosure:
    """
    What happens to data on one page, in a closed, comparable vocabulary.

    data_access and processing are tuples because a page can genuinely
    have more than one path in (HealthRing accepts both an upload and a
    local path; fMRI QC both fetches public data and accepts an upload).
    notes carries any caveat that would otherwise be lost to a tidy
    summary - including caveats that make a page look less finished than
    it otherwise would, which is the point: this exists to be honest, not
    reassuring.
    """

    page: str
    data_access: tuple[str, ...]
    processing: tuple[str, ...]
    persistent_storage: str
    data_accessed: str
    redistribution: str
    notes: str = field(default="")

    def __post_init__(self) -> None:
        if not self.page:
            raise ValueError("page cannot be empty.")

        if not self.data_accessed:
            raise ValueError("data_accessed cannot be empty.")

        if not self.data_access:
            raise ValueError(f"{self.page}: data_access cannot be empty.")

        if not self.processing:
            raise ValueError(f"{self.page}: processing cannot be empty.")

        for value in self.data_access:
            if value not in DATA_ACCESS_VALUES:
                raise ValueError(
                    f"{self.page}: '{value}' is not a known data_access "
                    f"value. Valid values: {', '.join(DATA_ACCESS_VALUES)}."
                )

        for value in self.processing:
            if value not in PROCESSING_VALUES:
                raise ValueError(
                    f"{self.page}: '{value}' is not a known processing "
                    f"value. Valid values: {', '.join(PROCESSING_VALUES)}."
                )

        if self.persistent_storage not in PERSISTENT_STORAGE_VALUES:
            raise ValueError(
                f"{self.page}: '{self.persistent_storage}' is not a known "
                f"persistent_storage value. Valid values: "
                f"{', '.join(PERSISTENT_STORAGE_VALUES)}."
            )

        if self.redistribution not in REDISTRIBUTION_VALUES:
            raise ValueError(
                f"{self.page}: '{self.redistribution}' is not a known "
                f"redistribution value. Valid values: "
                f"{', '.join(REDISTRIBUTION_VALUES)}."
            )


_HANDOFF_NOTE = (
    "Also records a hash of the uploaded data (not the data itself) plus "
    "its filename to session state, for the Cross-Analysis Implications "
    "page (shared/handoff.py)."
)

DISCLOSURES: tuple[DataHandlingDisclosure, ...] = (
    DataHandlingDisclosure(
        page="pages/1_Reliability.py",
        data_access=(DATA_ACCESS_USER_UPLOAD, DATA_ACCESS_BUNDLED_EXAMPLE),
        processing=(PROCESSING_IN_MEMORY,),
        persistent_storage=PERSISTENT_STORAGE_NONE,
        data_accessed="Uploaded CSV's scale/item columns.",
        redistribution=REDISTRIBUTION_NOT_APPLICABLE,
        notes=_HANDOFF_NOTE,
    ),
    DataHandlingDisclosure(
        page="pages/2_Impact_Evaluation.py",
        data_access=(DATA_ACCESS_USER_UPLOAD, DATA_ACCESS_BUNDLED_EXAMPLE),
        processing=(PROCESSING_IN_MEMORY,),
        persistent_storage=PERSISTENT_STORAGE_NONE,
        data_accessed="Uploaded CSV's outcome/group or pre/post columns.",
        redistribution=REDISTRIBUTION_NOT_APPLICABLE,
        notes=_HANDOFF_NOTE,
    ),
    DataHandlingDisclosure(
        page="pages/3_Fairness.py",
        data_access=(DATA_ACCESS_USER_UPLOAD, DATA_ACCESS_BUNDLED_EXAMPLE),
        processing=(PROCESSING_IN_MEMORY,),
        persistent_storage=PERSISTENT_STORAGE_NONE,
        data_accessed="Uploaded CSV's outcome/group/protected-attribute columns.",
        redistribution=REDISTRIBUTION_NOT_APPLICABLE,
        notes=_HANDOFF_NOTE,
    ),
    DataHandlingDisclosure(
        page="pages/4_Time_Series_QA.py",
        data_access=(DATA_ACCESS_USER_UPLOAD, DATA_ACCESS_BUNDLED_EXAMPLE),
        processing=(PROCESSING_IN_MEMORY,),
        persistent_storage=PERSISTENT_STORAGE_NONE,
        data_accessed="Uploaded CSV's timestamp and value columns.",
        redistribution=REDISTRIBUTION_NOT_APPLICABLE,
        notes=_HANDOFF_NOTE,
    ),
    DataHandlingDisclosure(
        page="pages/Portfolio_Impact_Analysis.py",
        data_access=(DATA_ACCESS_USER_UPLOAD, DATA_ACCESS_BUNDLED_EXAMPLE),
        processing=(PROCESSING_IN_MEMORY,),
        persistent_storage=PERSISTENT_STORAGE_NONE,
        data_accessed=(
            "Uploaded evidence/portfolio CSV rows, or the bundled "
            "synthetic sample data."
        ),
        redistribution=REDISTRIBUTION_NOT_APPLICABLE,
        notes="Does not record to shared/handoff.py.",
    ),
    DataHandlingDisclosure(
        page="pages/HealthRing_Worked_Example.py",
        data_access=(DATA_ACCESS_USER_UPLOAD, DATA_ACCESS_LOCAL),
        processing=(PROCESSING_TEMPORARY_FILE, PROCESSING_IN_MEMORY),
        persistent_storage=PERSISTENT_STORAGE_NONE,
        data_accessed=(
            "Per-window summary columns (hr, bvp_hr, quality, Label) from "
            "the HealthRing archive; the raw signal is never loaded."
        ),
        redistribution=REDISTRIBUTION_NOT_BUNDLED,
        notes=(
            "An uploaded archive is written to a temporary file in the "
            "app environment, not actively deleted by OpenMeasure in the "
            "current implementation - it can outlive the browser session "
            "until the host/container reclaims it. A local-path option "
            "reads the archive directly with no temporary file."
        ),
    ),
    DataHandlingDisclosure(
        page="pages/FMRI_QC_Worked_Example.py",
        data_access=(DATA_ACCESS_PUBLIC_REMOTE, DATA_ACCESS_USER_UPLOAD),
        processing=(PROCESSING_IN_MEMORY,),
        persistent_storage=PERSISTENT_STORAGE_NONE,
        data_accessed=(
            "Pseudonymous subject IDs and derived QC metrics; never raw "
            "MRI data."
        ),
        redistribution=REDISTRIBUTION_NOT_BUNDLED,
        notes=(
            "cinnqc QC output is fetched live from GitHub by default; "
            "rater-decision tables can optionally be uploaded."
        ),
    ),
    DataHandlingDisclosure(
        page="pages/GAIA_Worked_Example.py",
        data_access=(DATA_ACCESS_BUNDLED_EXAMPLE,),
        processing=(PROCESSING_IN_MEMORY,),
        persistent_storage=PERSISTENT_STORAGE_NONE,
        data_accessed="Bundled sample_data/gaia_models.csv only; no user data.",
        redistribution=REDISTRIBUTION_NOT_APPLICABLE,
        notes=(
            "The source abstract (Jallais et al., 2026) is not committed "
            "to this repository; only figures extracted from it are used."
        ),
    ),
    DataHandlingDisclosure(
        page="pages/GRAND_Worked_Example.py",
        data_access=(DATA_ACCESS_BUNDLED_EXAMPLE,),
        processing=(PROCESSING_IN_MEMORY,),
        persistent_storage=PERSISTENT_STORAGE_NONE,
        data_accessed=(
            "Bundled sample_data/grand_modalities.csv only; no user data, "
            "and no raw imaging data of any kind - this page never fetches "
            "or loads GRAND's actual MRI files, only illustrative ratings "
            "about GRAND's four modalities plus dataset-level facts "
            "(participant count, age range, scanner) hardcoded from the "
            "dataset's own public files."
        ),
        redistribution=REDISTRIBUTION_NOT_APPLICABLE,
        notes=(
            "GRAND (Anderson et al., 2026; OpenNeuro ds007831) is CC0 "
            "licensed. This page reports participant-level counts already "
            "public in the dataset's own participants.tsv, never "
            "individual-level rows."
        ),
    ),
    DataHandlingDisclosure(
        page="pages/6_Evidence_Review.py",
        data_access=(DATA_ACCESS_PUBLIC_REMOTE,),
        processing=(PROCESSING_IN_MEMORY,),
        persistent_storage=PERSISTENT_STORAGE_NONE,
        data_accessed=(
            "Search terms typed by the user, and OpenAlex's public metadata "
            "for matching works (title, authors, year, venue, DOI, abstract, "
            "citation count)."
        ),
        redistribution=REDISTRIBUTION_NOT_BUNDLED,
        notes=(
            "OpenAlex results are fetched live at runtime and never bundled "
            "in this repository. Records a hash of the search-results table "
            "returned (not the reviewer's typed finding description) plus "
            "screening-decision counts to session state, for the "
            "Cross-Analysis Implications page (shared/handoff.py)."
        ),
    ),
    DataHandlingDisclosure(
        page="pages/Multimodal_Signal_Convergence.py",
        data_access=(DATA_ACCESS_BUNDLED_EXAMPLE,),
        processing=(PROCESSING_IN_MEMORY,),
        persistent_storage=PERSISTENT_STORAGE_NONE,
        data_accessed=(
            "Bundled sample_data/modality_profiles.csv only; no user data, "
            "and no real biosignal data of any kind."
        ),
        redistribution=REDISTRIBUTION_NOT_APPLICABLE,
        notes=(
            "Unlike this app's other worked examples, modality_profiles.csv "
            "is not one real study's reported data - every rating in it is "
            "an illustrative, author-assigned score informed by cited "
            "literature, not a value that literature reports directly."
        ),
    ),
)


def disclosure_for(page: str) -> DataHandlingDisclosure:
    """The registered disclosure for one page, or raise if none exists."""

    match = next((item for item in DISCLOSURES if item.page == page), None)

    if match is None:
        raise ValueError(f"No data-handling disclosure registered for '{page}'.")

    return match


PRIVACY_PAGE = "pages/Privacy_and_Data_Access.py"


def render_data_handling_summary(disclosure: DataHandlingDisclosure) -> None:
    """
    A compact, consistent per-page disclosure.

    The only streamlit-touching function in this module; every other page
    in the app renders its disclosure through this one function, so the
    format cannot drift page to page.
    """

    with st.expander("Data handling on this page"):
        st.write(f"**Data access:** {', '.join(disclosure.data_access)}")
        st.write(f"**Processing:** {', '.join(disclosure.processing)}")
        st.write(f"**Persistent storage:** {disclosure.persistent_storage}")
        st.write(f"**Data accessed:** {disclosure.data_accessed}")
        st.write(f"**Redistribution:** {disclosure.redistribution}")
        if disclosure.notes:
            st.caption(disclosure.notes)
        st.page_link(
            PRIVACY_PAGE,
            label="Full privacy and data access policy",
            icon=":material/privacy_tip:",
        )
