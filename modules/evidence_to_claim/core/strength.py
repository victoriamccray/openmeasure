"""
Which level of causal rigor assembled evidence reaches, read against
Nesta's Standards of Evidence, Levels 1-4 of 5.

Puttick, R., & Ludlow, J. (2013). Standards of Evidence: An approach that
balances the need for evidence with innovation. Nesta. The five levels
described there concern causal rigor: (1) a stated logic model, (2) data
showing change without established causality, (3) causality demonstrated
via a comparison group, (4) independent replication of that finding, and
(5) manualized, systematized delivery that reliably reproduces the impact
at scale. Level 5 requires operational and delivery documentation this
module does not collect, so only Levels 1-4 are implemented.

Two things are separate here and are kept separate. The framework is
Nesta's and is cited as theirs. The rule deciding which level a
particular bundle of evidence reaches is OpenMeasure's, is not specified
by Nesta, and is disclosed as OpenMeasure's own convention wherever a
level is shown. Citing the framework for the rule would put Nesta's name
on an algorithm they did not write.

The rule previously promoted a bundle to Level 4 when two or more
distinct sources reported the same finding. That is not what independent
replication means. A grants-management export, a set of employer
verification calls and a participant follow-up survey are three sources
describing one program; they can triangulate a finding and none of them
reproduces a causal result on a different sample. Level 4 now requires an
evidence item that names another study, on an independent sample, of the
same intervention, using a causal design.

This module never issues a go/no-go verdict on the claim itself: it
states which level the assembled evidence reaches, and flags a tension
when the claim's stated type expects more rigor than that.
"""

from __future__ import annotations

from dataclasses import dataclass

from .claim import ClaimDraft
from .evidence import EvidenceBundle
from .validate import ValidationResult

# One study cannot replicate itself. A bundle naming a single study
# establishes no replication however that study is described, so Level 4
# needs a second identified study alongside the one that made the
# original finding.
MIN_STUDY_IDENTITIES_FOR_REPLICATION = 2

FRAMEWORK_CITATION = (
    "Puttick & Ludlow (2013), Nesta Standards of Evidence, Levels 1-4 of 5."
)

# Shown wherever a level is. The framework is Nesta's; the rule that maps
# a particular bundle onto it is not theirs and must not be presented as
# though it were.
OPENMEASURE_RULE_NOTE = (
    "The levels are Nesta's. How OpenMeasure decides which level a "
    "particular bundle of evidence reaches is OpenMeasure's own rule, "
    "stated below, and is not part of the cited framework."
)

OPENMEASURE_RULE_STATEMENT = (
    "Level 2 once at least one usable evidence item is attached. Level 3 "
    "once at least one item reports a comparison group. Level 4 once an "
    "item names another study, on an independent sample, of the same "
    "intervention, using a causal design, alongside a separately "
    "identified study."
)

# What triangulation is, so it is not mistaken for the thing above it.
# Several collection methods agreeing about one program is worth having
# and is a different claim from a finding reproduced elsewhere.
TRIANGULATION_NOTE = (
    "Several collection methods describing the same program agree. That "
    "is triangulation, which strengthens confidence that the change was "
    "recorded correctly, and it is a different claim from a causal "
    "finding reproduced on an independent sample."
)

REPLICATION_NOT_ESTABLISHED = (
    "Independent replication not established from the supplied evidence."
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

    # Named separately from the level, because they are separate claims.
    # Triangulation is about whether several ways of looking agree;
    # replication is about whether the causal finding held up elsewhere.
    triangulated: bool
    replication_established: bool
    replication_note: str
    openmeasure_rule_note: str = OPENMEASURE_RULE_NOTE
    openmeasure_rule_statement: str = OPENMEASURE_RULE_STATEMENT


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

    # Replication is a property of the evidence, not a count of the
    # sources describing it. Both conditions are required: an item that
    # names another study on an independent sample of the same
    # intervention using a causal design, and a second identified study
    # for it to be independent of.
    replication_established = (
        evidence.n_replication_studies >= 1
        and evidence.n_study_identities >= MIN_STUDY_IDENTITIES_FOR_REPLICATION
    )
    triangulated = evidence.n_collection_methods >= 2

    level_number = 1
    if evidence.n_usable_items >= 1:
        level_number = 2
    if validation.has_comparison_group:
        level_number = 3
        if replication_established:
            level_number = 4

    nesta_level = _LEVELS_BY_NUMBER[level_number]

    if replication_established:
        replication_note = (
            f"{evidence.n_replication_studies} identified study or studies "
            "tested this claim on an independent sample of the same "
            "intervention using a causal design."
        )
    else:
        replication_note = REPLICATION_NOT_ESTABLISHED
        if evidence.n_sources > 1:
            replication_note += (
                f" {evidence.n_sources} sources describe this program, "
                "which is a count of sources rather than of studies that "
                "reproduced the finding."
            )

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
            "To reach Level 4: add evidence from another study that tested "
            "this claim on an independent sample of the same intervention "
            "using a causal design, identified so it can be checked. More "
            "sources describing this program do not reach Level 4, however "
            "many there are."
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
        triangulated=triangulated,
        replication_established=replication_established,
        replication_note=replication_note,
    )
