"""
Research Journeys: guided worked examples of what a real dataset supports.

The "Research Question" lifecycle stage in shared/catalog.py
(LIFECYCLE_STAGES) has no numbered workflow -- STAGES_WITHOUT_WORKFLOWS
names that as a stated gap rather than hiding it. Research Journeys are
what actually occupies that stage in practice: each one walks a real (or,
for Multimodal Signal Convergence, clearly-marked illustrative) dataset
through the question a Research Question stage asks -- what evidence would
answer this, and what does this data actually support -- without being a
workflow. A workflow has a validation category, a module_key, and a record
in shared/handoff.py; a journey has none of the three, so journeys are
kept out of shared/catalog.py's WORKFLOWS rather than forced to carry
fields that would never be true of them (see Home.py's docstring for the
fuller reasoning already given there).

domain groups journeys because a single undifferentiated "Research
Journeys" list stopped being legible once it held six entries spanning
three unrelated fields. Home.py renders one sidebar section per domain by
iterating journeys_by_domain(), so a new journey only needs a domain from
JOURNEY_DOMAINS (or a new domain added to that tuple) and an entry below --
no navigation restructuring.

title and page are unchanged from before this module existed, so every URL
already linked to (e.g. /HealthRing_Worked_Example) keeps working: url_path
is derived from the page filename the same way shared/catalog.py derives
it, not stored separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Domains, in the order Home.py renders their sidebar sections. Declared
# explicitly, same reasoning as shared/catalog.py's CATEGORY_ORDER: there is
# no honest way to derive an order, and alphabetical would be arbitrary.
JOURNEY_DOMAINS: tuple[str, ...] = (
    "Multi-Modal Health Imaging",
    "Social Impact Evaluation",
    "Responsible AI",
)


@dataclass(frozen=True)
class ResearchJourney:
    """
    One Research Journey, and where it belongs.

    subdomain is the more specific field named within domain (e.g.
    "Wearable signal processing" within "Multi-Modal Health Imaging"). It is
    not shown in the sidebar -- Streamlit's st.navigation only groups pages
    one level deep, the same limit Home.py's docstring already notes for
    the domain grouping itself -- but it is kept as data so a page rendering
    journeys (Overview.py, a future filter) can show it without re-deriving
    it from title text.
    """

    title: str
    domain: str
    subdomain: str
    summary: str
    page: str

    def __post_init__(self) -> None:
        if self.domain not in JOURNEY_DOMAINS:
            raise ValueError(
                f"{self.title} has domain '{self.domain}', which is not in "
                f"JOURNEY_DOMAINS, so it would render in no sidebar section. "
                f"Known domains: {', '.join(JOURNEY_DOMAINS)}."
            )

        for name in ("title", "subdomain", "summary", "page"):
            if not getattr(self, name):
                raise ValueError(
                    f"{self.title or 'A journey'} is missing a value for "
                    f"'{name}'."
                )

        if "\n" in self.summary:
            raise ValueError(
                f"{self.title}'s summary must be a single line, so it fits "
                "on a card."
            )

    @property
    def url_path(self) -> str:
        """Derived the same way as shared/catalog.py's Workflow.url_path."""

        stem = Path(self.page).stem

        return re.sub(r"^\d+_", "", stem)


JOURNEYS: tuple[ResearchJourney, ...] = (
    ResearchJourney(
        title="Wearables: HealthRing",
        domain="Multi-Modal Health Imaging",
        subdomain="Wearable signal processing",
        summary=(
            "A gated, step-by-step validation of consumer wearable "
            "heart-rate and SpO2 data, from signal quality through a "
            "leakage-aware train/test split to a closing validation record."
        ),
        page="pages/HealthRing_Worked_Example.py",
    ),
    ResearchJourney(
        title="Neuroimaging: GRAND",
        domain="Multi-Modal Health Imaging",
        subdomain="Neurocognitive",
        summary=(
            "How separate imaging and behavioral modalities in a real "
            "multimodal neuroimaging dataset are quality-checked on their "
            "own before being combined into evidence about a research "
            "question."
        ),
        page="pages/GRAND_Worked_Example.py",
    ),
    ResearchJourney(
        title="Medical Imaging: fMRI QC",
        domain="Multi-Modal Health Imaging",
        subdomain="fMRI QC",
        summary=(
            "Whether an existing fMRI quality-control tool's own metrics "
            "line up with trained human raters' decisions, and how much "
            "independent raters agree with each other in the first place."
        ),
        page="pages/FMRI_QC_Worked_Example.py",
    ),
    ResearchJourney(
        title="Grantmaking: Portfolio Impact Analysis",
        domain="Social Impact Evaluation",
        subdomain="Grantmaking portfolio analysis",
        summary=(
            "A review layer over evidence already assembled about a "
            "program, grantee, or portfolio result: what it supports, "
            "whether it can be compared to other results, and what is "
            "defensible to report."
        ),
        page="pages/Portfolio_Impact_Analysis.py",
    ),
    ResearchJourney(
        title="AI & Environmental Equity: GAIA",
        domain="Responsible AI",
        subdomain="Green AI",
        summary=(
            "When a more efficient model is good enough to replace a "
            "larger one, once energy use, CO2, and size are weighed "
            "alongside predictive performance, using a real "
            "knowledge-distillation study as the worked example."
        ),
        page="pages/GAIA_Worked_Example.py",
    ),
    ResearchJourney(
        title="Neurotechnology: Multimodal Signal Convergence",
        domain="Responsible AI",
        subdomain="Neurosecurity",
        summary=(
            "Whether the added interpretive value of combining more than "
            "one signal about a person justifies the added privacy, "
            "security, and agency cost of collecting each additional "
            "signal."
        ),
        page="pages/Multimodal_Signal_Convergence.py",
    ),
)


def journeys_by_domain() -> dict[str, tuple[ResearchJourney, ...]]:
    """
    Group journeys by domain, in JOURNEY_DOMAINS order.

    A domain with no journey is omitted rather than shown, same reasoning
    as shared/catalog.py's workflows_by_category: an empty navigation
    section is a dead heading. __post_init__ rejects a journey whose domain
    is unknown, so a journey cannot silently vanish from the sidebar.
    """

    grouped: dict[str, list[ResearchJourney]] = {
        domain: [] for domain in JOURNEY_DOMAINS
    }

    for journey in JOURNEYS:
        grouped[journey.domain].append(journey)

    return {
        domain: tuple(items) for domain, items in grouped.items() if items
    }
