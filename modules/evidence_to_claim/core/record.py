"""
Evidence record - assembles every prior stage's result into one auditable
Evidence to Claim record, and drafts a leadership-ready summary sentence.
"""

from __future__ import annotations

from dataclasses import dataclass

from .claim import ClaimDraft
from .evidence import EvidenceBundle
from .limitations import LimitationsResult
from .portfolio import PortfolioContextResult
from .strength import SupportedClaimResult
from .validate import ValidationResult


@dataclass(frozen=True)
class EvidenceRecord:
    """Every stage's result for one claim, kept together for the audit trail."""

    claim: ClaimDraft
    evidence: EvidenceBundle
    validation: ValidationResult
    supported_claim: SupportedClaimResult
    limitations: LimitationsResult
    portfolio_context: PortfolioContextResult | None = None

    def __post_init__(self) -> None:
        claim_ids = {
            self.claim.claim_id,
            self.evidence.claim_id,
            self.validation.claim_id,
            self.supported_claim.claim_id,
            self.limitations.claim_id,
        }
        if len(claim_ids) > 1:
            raise ValueError(
                f"All sub-results must share one claim_id; got {sorted(claim_ids)}."
            )


def build_leadership_summary(record: EvidenceRecord) -> str:
    """
    Draft a concise, leadership-ready sentence summarizing an evidence record.

    Deterministic and template-based, not generated: the technical detail
    behind it (checks, flags, portfolio context) is rendered separately,
    in full, alongside this summary rather than folded into it.
    """
    supported = record.supported_claim
    validation = record.validation

    sentence = (
        f"{record.claim.claim_text} "
        f"Based on {validation.corroboration_count} independent source(s), "
        f"the evidence reaches Nesta Standards of Evidence Level "
        f"{supported.nesta_level.level} ({supported.nesta_level.description})"
    )

    if supported.claim_type_alignment_warning:
        sentence += " " + supported.claim_type_alignment_warning
    else:
        sentence += (
            " This meets the rigor conventionally expected for a claim of "
            f"this type ('{record.claim.claim_type}')."
        )

    if record.limitations.n_blocking_flags > 0:
        sentence += (
            f" {record.limitations.n_blocking_flags} blocking limitation(s) "
            "should be addressed before this claim is published."
        )

    return sentence
