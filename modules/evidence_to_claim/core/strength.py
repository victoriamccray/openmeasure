"""
Determine supported claim - grades evidence against Nesta's Standards of
Evidence, Levels 1-4 of 5.

Puttick, R., & Ludlow, J. (2013). Standards of Evidence: An approach that
balances the need for evidence with innovation. Nesta. The five levels
described there concern causal rigor: (1) a stated logic model, (2) data
showing change without established causality, (3) causality demonstrated
via a comparison group, (4) independent replication of that finding, and
(5) manualized, systematized delivery that reliably reproduces the impact
at scale. Level 5 requires operational/delivery documentation this module
does not collect, so only Levels 1-4 are implemented (see LEVEL_5_NOTE).

This module never issues a go/no-go verdict on the claim itself: it states
which level of causal rigor the assembled evidence reaches, and flags a
tension when the claim's stated type expects more rigor than that.
"""

from __future__ import annotations

from dataclasses import dataclass

from .claim import ClaimDraft
from .evidence import EvidenceBundle
from .validate import ValidationResult

# "One or more independent replications" (Level 4): at least this many
# distinct sources reporting the same finding.
MIN_REPLICATION_SOURCES = 2

FRAMEWORK_CITATION = (
    "Puttick & Ludlow (2013), Nesta Standards of Evidence, Levels 1-4 of 5."
)

LEVEL_5_NOTE = (
    "Nesta Level 5 (manualized, systematized delivery that reliably "
    "reproduces impact at scale) requires operational and delivery "
    "documentation this module does not collect. Not assessed here."
)


@dataclass(frozen=True)
class NestaLevel:
    """One level of the Nesta Standards of Evidence ladder (1-4 of 5)."""

    level: int
    label: str
    description: str


NESTA_LEVELS: tuple[NestaLevel, ...] = (
    NestaLevel(
        1,
        "Logic model",
        "Describes what the program does and why it should work, without outcome data.",
    ),
    NestaLevel(
        2,
        "Data shows change",
        "Data shows a positive change, but causality is not yet established.",
    ),
    NestaLevel(
        3,
        "Causal comparison",
        "Causality is demonstrated using a comparison group.",
    ),
    NestaLevel(
        4,
        "Independent replication",
        "One or more independent replications confirm the finding.",
    ),
)

_LEVELS_BY_NUMBER: dict[int, NestaLevel] = {level.level: level for level in NESTA_LEVELS}

# OpenMeasure convention, not part of the cited framework: the minimum
# Nesta level conventionally expected for a claim of this type, since
# output claims describe delivery, outcome claims assert an observed
# change, and impact claims assert the program caused that change. Stated
# as an OpenMeasure convention wherever it is shown.
MIN_LEVEL_BY_CLAIM_TYPE: dict[str, int] = {"output": 1, "outcome": 2, "impact": 3}


@dataclass(frozen=True)
class SupportedClaimResult:
    """Which level of evidentiary rigor the assembled evidence reaches."""

    claim_id: str
    nesta_level: NestaLevel
    framework_citation: str
    level_5_note: str
    suggested_language: str
    claim_type_alignment_warning: str | None
    next_level_hint: str | None


def determine_supported_claim(
    claim: ClaimDraft, evidence: EvidenceBundle, validation: ValidationResult
) -> SupportedClaimResult:
    """
    Determine which Nesta level the assembled evidence reaches for a claim.

    Levels 1-2 are close to automatic once evidence is attached (a claim
    has been stated, and at least one usable evidence item exists); the
    module's real discrimination is at Levels 3-4, where a comparison
    group and independent replication are the deciding evidence.
    """
    if not (claim.claim_id == evidence.claim_id == validation.claim_id):
        raise ValueError(
            "claim, evidence, and validation must all describe the same "
            f"claim_id; got {claim.claim_id!r}, {evidence.claim_id!r}, "
            f"{validation.claim_id!r}."
        )

    level_number = 1
    if evidence.n_usable_items >= 1:
        level_number = 2
    if validation.has_comparison_group:
        level_number = 3
        if validation.corroboration_count >= MIN_REPLICATION_SOURCES:
            level_number = 4

    nesta_level = _LEVELS_BY_NUMBER[level_number]

    if level_number == 1:
        next_level_hint = (
            "To reach Level 2: add evidence data showing the claimed change "
            "(not just a description of the program)."
        )
    elif level_number == 2:
        next_level_hint = (
            "To reach Level 3: add evidence with a comparison group, so the "
            "change can be attributed to the program rather than other "
            "explanations."
        )
    elif level_number == 3:
        next_level_hint = (
            f"To reach Level 4: add independent source(s) confirming the "
            f"finding ({validation.corroboration_count} of "
            f"{MIN_REPLICATION_SOURCES} needed)."
        )
    else:
        next_level_hint = None

    min_expected = MIN_LEVEL_BY_CLAIM_TYPE[claim.claim_type]
    warning = None
    if level_number < min_expected:
        expected_level = _LEVELS_BY_NUMBER[min_expected]
        warning = (
            f"This is an '{claim.claim_type}' claim, which conventionally "
            f"expects at least Nesta Level {min_expected} "
            f"({expected_level.description}). The evidence reaches Level "
            f"{level_number}. Consider more tentative language, or "
            "reframing the claim to match the evidence available."
        )

    suggested_language = (
        f"Based on the evidence assembled, this claim reaches Nesta Level "
        f"{level_number} ({nesta_level.description})"
    )

    return SupportedClaimResult(
        claim_id=claim.claim_id,
        nesta_level=nesta_level,
        framework_citation=FRAMEWORK_CITATION,
        level_5_note=LEVEL_5_NOTE,
        suggested_language=suggested_language,
        claim_type_alignment_warning=warning,
        next_level_hint=next_level_hint,
    )
