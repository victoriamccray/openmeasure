"""
Describe evidence - core aggregation for the Evidence to Claim Journey.

All functions are pure: they accept plain dataclasses and return frozen
dataclasses. No I/O, no UI logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

COLLECTION_METHODS: tuple[str, ...] = (
    "survey",
    "administrative_record",
    "observation",
    "interview",
    "published_literature",
)

# Why an item was dropped, recorded on every result per
# docs/design-standards.md section 2.
MISSING_FINDING_OR_INDICATOR = "missing finding text or indicator id"


@dataclass(frozen=True)
class EvidenceItem:
    """
    One piece of evidence an analyst has assembled behind a claim.

    The four fields at the end are what separates another source about
    the same program from an independent evaluation that tested the same
    claim. They default to the conservative answer, so an item says
    nothing about replication unless an analyst states otherwise.

    They exist because counting sources is not counting replications. A
    grants-management export, a set of employer verification calls and a
    participant follow-up survey are three sources describing one
    program. They can triangulate a finding, and no number of them
    reproduces a causal result on a different sample.
    """

    source: str
    finding_text: str
    indicator_id: str | None
    sample_size: int | None
    has_comparison_group: bool
    collection_method: str
    time_lag_days: int | None

    # Which study or evaluation this item comes from. Empty means the
    # analyst has not identified one, which is the usual case for
    # administrative or monitoring data and is not a defect.
    study_identity: str = ""
    # Whether this evidence comes from a sample independent of the one
    # the original finding was made on.
    independent_sample: bool = False
    # Whether it concerns the same intervention, delivered in a
    # comparable context.
    same_intervention: bool = False
    # Whether that study used a design capable of supporting a causal
    # claim, rather than observing an outcome alongside the program.
    causal_design: bool = False

    @property
    def is_independent_evaluation(self) -> bool:
        """
        Whether this item is another evaluation that tested the claim.

        All four conditions, because dropping any one of them admits
        something that is not a replication: an unnamed study cannot be
        checked, the same sample re-analysed is not independent, a
        different intervention is not the same claim, and a study without
        a causal design did not reproduce a causal result.
        """
        return bool(
            self.study_identity.strip()
            and self.independent_sample
            and self.same_intervention
            and self.causal_design
        )


@dataclass(frozen=True)
class EvidenceBundle:
    """All usable evidence items assembled behind one claim."""

    claim_id: str
    items: tuple[EvidenceItem, ...]
    n_sources: int
    n_collection_methods: int
    has_any_comparison_group: bool

    # Distinct studies that independently tested the claim, and distinct
    # studies named at all. Both are needed: one study cannot replicate
    # itself, so a bundle naming a single study establishes no
    # replication however that study is described.
    n_replication_studies: int
    n_study_identities: int
    min_sample_size: int | None
    max_time_lag_days: int | None
    n_input_items: int
    n_usable_items: int
    n_excluded_items: int
    exclusion_reason: str


def summarize_evidence(items: Sequence[EvidenceItem], claim_id: str) -> EvidenceBundle:
    """
    Assemble the evidence items describing one claim into a bundle.

    An item missing finding_text or indicator_id cannot be tied to anything
    the claim asserts, so it is dropped and counted rather than silently
    included.
    """
    if not claim_id:
        raise ValueError("claim_id cannot be empty.")

    n_input_items = len(items)

    usable = tuple(
        item
        for item in items
        if item.finding_text.strip() and item.indicator_id
    )

    n_usable_items = len(usable)
    n_excluded_items = n_input_items - n_usable_items

    if n_usable_items == 0:
        raise ValueError(
            "No usable evidence items: every item is missing finding_text "
            "or indicator_id."
        )

    sample_sizes = [
        item.sample_size for item in usable if item.sample_size is not None
    ]
    time_lags = [
        item.time_lag_days for item in usable if item.time_lag_days is not None
    ]

    return EvidenceBundle(
        claim_id=claim_id,
        items=usable,
        n_sources=len({item.source for item in usable}),
        n_collection_methods=len({item.collection_method for item in usable}),
        has_any_comparison_group=any(item.has_comparison_group for item in usable),
        n_replication_studies=len(
            {
                item.study_identity.strip()
                for item in usable
                if item.is_independent_evaluation
            }
        ),
        n_study_identities=len(
            {
                item.study_identity.strip()
                for item in usable
                if item.study_identity.strip()
            }
        ),
        min_sample_size=min(sample_sizes) if sample_sizes else None,
        max_time_lag_days=max(time_lags) if time_lags else None,
        n_input_items=n_input_items,
        n_usable_items=n_usable_items,
        n_excluded_items=n_excluded_items,
        exclusion_reason=MISSING_FINDING_OR_INDICATOR,
    )
