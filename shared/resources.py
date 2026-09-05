"""
Resources: existing toolkits, methods guides, and dataset directories that
sit outside OpenMeasure itself.

Explore_Real_Data.py already points to specific datasets chosen because
they pair with a specific OpenMeasure workflow (RealDataset.try_with
requires exactly that). This catalog is for the broader case: a tool, a
methods-selection guide, or a dataset directory worth knowing about that
does not fit that one-workflow contract, or is not itself a dataset at all.

Entries here are maintained by the OpenMeasure project rather than
verified against a published citation the way shared/case_studies.py's
entries are -- there is no paper to check a description against. Treat
this list as a pointer, not an endorsement of accuracy for content on the
linked site, which can change after the description below was written.

url is deliberately optional (RealDataset's DataSource requires one; a
Resource does not) because some entries are recorded by name only, with a
link to follow once confirmed, rather than left out of the list entirely.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Resource:
    """One external resource, and what it is for."""

    name: str
    kind: str
    description: str
    url: str = ""

    def __post_init__(self) -> None:
        for field_name in ("name", "kind", "description"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"{self.name or 'A resource'} is missing a value for "
                    f"'{field_name}'."
                )

        if self.kind not in KIND_ORDER:
            raise ValueError(
                f"{self.name} has kind '{self.kind}', which is not in "
                f"KIND_ORDER, so it would render in no section. Known "
                f"kinds: {', '.join(KIND_ORDER)}."
            )

        if self.url and not self.url.startswith("https://"):
            raise ValueError(
                f"'{self.name}' has url '{self.url}', which is not an "
                "https link. A given link must be verifiable, not relative "
                "or unsecured; leave url empty instead of using a "
                "non-https placeholder."
            )


# Display order for kinds. Declared explicitly, same reasoning as
# shared/catalog.py's CATEGORY_ORDER: there is no honest way to derive one.
KIND_ORDER: tuple[str, ...] = (
    "Method Selection Tool",
    "Statistical Toolkit",
    "Literature Search Tool",
    "Dataset Directory",
    "Measures Repository",
    "Research Platform",
    "Research Impact Resource",
)

RESOURCES: tuple[Resource, ...] = (
    Resource(
        name="Co-Creation Methods Navigator",
        kind="Method Selection Tool",
        description=(
            "A decision-support tool for exploring over 500 co-creation "
            "and participatory-research methods, to find an approach "
            "suited to a specific community-engagement or research "
            "context."
        ),
        url="https://ccmethodsselector.lovable.app/",
    ),
    Resource(
        name="OpenEpi",
        kind="Statistical Toolkit",
        description=(
            "Free, open-source web calculators for epidemiologic and "
            "biostatistical analysis: sample size and power, 2x2 and "
            "stratified tables, person-time rates, and related "
            "calculations."
        ),
        url="https://www.openepi.com/Menu/OE_Menu.htm",
    ),
    Resource(
        name="OpenAlex",
        kind="Literature Search Tool",
        description=(
            "A fully open, keyless index of scholarly works, authors, "
            "venues, and institutions, with a free public Works API. "
            "OpenMeasure's Evidence Review workflow searches it directly."
        ),
        url="https://openalex.org/",
    ),
    Resource(
        name="Neuro2.ai Neuroscience Datasets",
        kind="Dataset Directory",
        description=(
            "A searchable directory of neuroscience datasets, filterable "
            "by modality and source."
        ),
        url="https://datasets.neuro2.ai/",
    ),
    Resource(
        name="OpenNeuro",
        kind="Dataset Directory",
        description=(
            "A free, open platform for sharing and downloading BIDS-"
            "organized MRI, MEG, EEG, iEEG, ECoG, ASL, and PET datasets, "
            "with no data-access application required for its public "
            "datasets."
        ),
        url="https://openneuro.org/",
    ),
    Resource(
        name="HEALthy Brain and Child Development (HBCD) Study - Release 2.1",
        kind="Dataset Directory",
        description=(
            "A longitudinal dataset of more than 3,500 participants, with "
            "structural MRI, diffusion MRI, resting-state fMRI, and EEG "
            "alongside behavioral, cognitive, and health assessments of "
            "early child development. Access requires completing the "
            "study's own data-access process."
        ),
        url="https://www.nbdc-datahub.org/hbcd-release-2-1",
    ),
    Resource(
        name="Open Measurement Network Initiative for Alzheimer's Disease "
        "and Related Dementias (OMNI ADRD)",
        kind="Measures Repository",
        description=(
            "An open science repository of broadly applicable, precise, "
            "and sensitive measures for use in dementia prevention trials."
        ),
        url="https://omni-adrd.org/",
    ),
    Resource(
        name="OpenNerve",
        kind="Research Platform",
        description=(
            "An open-source implantable neuromodulation system intended "
            "to support diverse clinical research needs."
        ),
        url="https://sites.usc.edu/carss/",
    ),
    Resource(
        name="RuralSenses",
        kind="Research Platform",
        description=(
            "An AI-based platform for NGOs, foundations, and social "
            "enterprises to measure social and environmental impact by "
            "collecting qualitative voice data in local languages and "
            "analyzing it into impact reports."
        ),
        url="https://ruralsenses.com/",
    ),
    Resource(
        name="SPARC Portal",
        kind="Dataset Directory",
        description=(
            "The public data and tools portal for the NIH Common Fund's "
            "SPARC (Stimulating Peripheral Activity to Relieve "
            "Conditions) program, sharing FAIR peripheral-nervous-system "
            "and bioelectronic-medicine datasets."
        ),
        url="https://sparc.science/",
    ),
    Resource(
        name="Science With Impact (Anne Toomey)",
        kind="Research Impact Resource",
        description=(
            "Ways to make science more impactful for society, including "
            "community-based science, improved science communication, "
            "and science policy."
        ),
        url="https://digitalcommons.pace.edu/bookshelf/25/",
    ),
    Resource(
        name="Co-Creation Methods for Public Health Research (Agnello et al., 2025)",
        kind="Research Impact Resource",
        description=(
            "A Health CASCADE scoping review characterizing the "
            "co-creation methods used in public-health research, "
            "including their reported characteristics, benefits, and "
            "challenges. Published in BMC Medical Research Methodology."
        ),
        url="https://doi.org/10.1186/s12874-025-02514-4",
    ),
    Resource(
        # Name and description are the organization's own wording, supplied
        # by them when they approved being listed here (September 2026).
        # Change them only with their agreement.
        name="The Center for Implementation",
        kind="Research Impact Resource",
        description=(
            "Training, consulting, and free interactive online tools that "
            "help individuals and teams across diverse fields apply "
            "evidence-informed implementation methods to support change "
            "efforts."
        ),
        url="https://thecenterforimplementation.com/",
    ),
)


def resources_by_kind() -> dict[str, tuple[Resource, ...]]:
    """
    Group resources by kind, in KIND_ORDER.

    A kind with no resource is omitted, same reasoning as
    shared/catalog.py's workflows_by_category.
    """

    grouped: dict[str, list[Resource]] = {kind: [] for kind in KIND_ORDER}

    for resource in RESOURCES:
        grouped[resource.kind].append(resource)

    return {kind: tuple(items) for kind, items in grouped.items() if items}
