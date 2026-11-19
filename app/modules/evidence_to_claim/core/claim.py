"""
Define claim - core data model for the Evidence to Claim Journey.

Pure construction and validation of a claim an analyst wants to make about
a program, grantee, or portfolio result. No statistic is computed here;
later stages (evidence.py, validate.py, strength.py) do that work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# W.K. Kellogg Foundation. (2004). Logic Model Development Guide. W.K.
# Kellogg Foundation. Output/outcome/impact are that guide's logic-model
# levels: an output claim describes what was delivered, an outcome claim
# asserts an observed change, and an impact claim asserts the program
# caused that change.
CLAIM_TYPES: tuple[str, ...] = ("output", "outcome", "impact")

CLAIM_LEVELS: tuple[str, ...] = ("program", "grantee", "portfolio")

_LEVEL_REQUIRED_FIELD: dict[str, str] = {
    "program": "program_id",
    "grantee": "grantee_id",
    "portfolio": "portfolio_id",
}


@dataclass(frozen=True)
class ClaimDraft:
    """One claim an analyst wants to responsibly report."""

    claim_id: str
    claim_text: str
    claim_type: str
    level: str
    program_id: str | None = None
    grantee_id: str | None = None
    portfolio_id: str | None = None
    related_indicator_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.claim_id:
            raise ValueError("claim_id cannot be empty.")

        if not self.claim_text.strip():
            raise ValueError("claim_text cannot be empty.")

        if self.claim_type not in CLAIM_TYPES:
            raise ValueError(
                f"claim_type '{self.claim_type}' is not one of {CLAIM_TYPES}."
            )

        if self.level not in CLAIM_LEVELS:
            raise ValueError(
                f"level '{self.level}' is not one of {CLAIM_LEVELS}."
            )

        required_field = _LEVEL_REQUIRED_FIELD[self.level]
        if getattr(self, required_field) is None:
            raise ValueError(
                f"level '{self.level}' requires '{required_field}' to be set."
            )
