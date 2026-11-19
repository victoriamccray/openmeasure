"""
Examine limitations - deterministic flags derived from evidence and
validation, describing why an evidence base might be thinner than it looks.
"""

from __future__ import annotations

from dataclasses import dataclass

from .evidence import EvidenceBundle
from .validate import SEVERITY_ADVISORY, ValidationResult

CATEGORY_METHOD_BIAS = "method_bias"

LIMITATION_CATEGORIES: tuple[str, ...] = (
    "comparison_group",
    "sample_size",
    "corroboration",
    "time_lag",
    CATEGORY_METHOD_BIAS,
)


@dataclass(frozen=True)
class LimitationFlag:
    """One reason the evidence behind a claim may be weaker than it looks."""

    category: str
    message: str
    severity: str


@dataclass(frozen=True)
class LimitationsResult:
    """Every limitation flag raised for one claim's evidence."""

    claim_id: str
    flags: tuple[LimitationFlag, ...]
    n_flags: int
    n_blocking_flags: int


def examine_limitations(
    bundle: EvidenceBundle, validation: ValidationResult
) -> LimitationsResult:
    """
    Flag limitations in the evidence behind a claim.

    Every failed validation check becomes a flag, plus one bundle-only
    check validation does not cover: whether every evidence item shares a
    single collection method, which validation's corroboration count alone
    would not catch (three surveys are three sources, but one method).
    """
    if bundle.claim_id != validation.claim_id:
        raise ValueError(
            f"bundle.claim_id ({bundle.claim_id!r}) and validation.claim_id "
            f"({validation.claim_id!r}) must match."
        )

    flags = [
        LimitationFlag(
            category=check.criterion,
            message=check.detail,
            severity=check.severity,
        )
        for check in validation.checks
        if not check.passed
    ]

    if bundle.n_collection_methods == 1:
        flags.append(
            LimitationFlag(
                category=CATEGORY_METHOD_BIAS,
                message=(
                    f"All {bundle.n_usable_items} evidence item(s) share one "
                    f"collection method ('{bundle.items[0].collection_method}'), "
                    "so a bias shared by that method (e.g. self-report) "
                    "cannot be ruled out by triangulation."
                ),
                severity=SEVERITY_ADVISORY,
            )
        )

    n_blocking_flags = sum(
        1 for flag in flags if flag.severity == "blocking"
    )

    return LimitationsResult(
        claim_id=bundle.claim_id,
        flags=tuple(flags),
        n_flags=len(flags),
        n_blocking_flags=n_blocking_flags,
    )
