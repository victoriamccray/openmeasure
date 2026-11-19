"""
How a statistic is explained: concept, then your numbers, then notation.

The rule this encodes is that nobody should have to decode notation
before understanding what a calculation is doing. A reader meets the
labeled concept first, then the same arithmetic with their own values
substituted in, then the result and what it means. Formal notation and
per-symbol anatomy exist, and are one click away, but they are never the
first thing on screen.

Pure data, no streamlit: a module's core/ imports this to describe its
own statistics, and core/ must stay framework-independent (CLAUDE.md,
and shared/tests/test_core_framework_independence.py). The rendering
half is render_formula() in shared/report.py.

Two invariants make an explanation trustworthy rather than decorative.

The substitution line is *generated* from the terms, never written out.
A number shown in it cannot disagree with the term it came from, because
it is that term's own value.

The result is *taken* from the statistic that was actually computed,
never recomputed for display. Each module's builder reads it off the
result object. A test then checks that the arithmetic shown really does
land on it, so a formula that does not produce its own answer fails
rather than renders.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Formatter


@dataclass(frozen=True)
class FormulaTerm:
    """One quantity in a formula, in every form a reader might need it."""

    key: str
    symbol: str
    plain_name: str
    meaning: str
    display_value: str

    # Plain-language provenance, for a reader: "the Treated row of the
    # table above". Deliberately prose and deliberately not machine
    # readable. Replication work will want structured provenance (a page,
    # a table number, a dataset column, whether a value is published or
    # computed here), and that belongs in its own fields rather than
    # overloading this one into a string nobody can parse.
    source: str = ""

    def __post_init__(self) -> None:
        for field_name in ("key", "symbol", "plain_name", "meaning", "display_value"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"{self.key or 'A term'} is missing a value for "
                    f"'{field_name}'."
                )


@dataclass(frozen=True)
class FormulaExplanation:
    """
    One statistic, explained at three levels of formality.

    blocks is the concept as labeled parts, in reading order, including
    its operators: ("Difference between group means", "/", "Typical
    within-group variation", "=", "Standardized difference"). This is
    what a reader sees first, and it carries no notation at all.

    substitution_template interpolates term keys, so the arithmetic a
    reader sees is assembled from the same values the anatomy lists.
    """

    name: str
    blocks: tuple[str, ...]
    substitution_template: str
    formal_latex: str
    terms: tuple[FormulaTerm, ...]
    result_display: str
    reading: str
    citation: str = ""

    def __post_init__(self) -> None:
        for field_name in ("name", "substitution_template", "formal_latex",
                           "result_display", "reading"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"{self.name or 'An explanation'} is missing a value for "
                    f"'{field_name}'."
                )

        if not self.blocks:
            raise ValueError(f"{self.name} lists no concept blocks.")

        if not self.terms:
            raise ValueError(f"{self.name} lists no terms.")

        keys = [term.key for term in self.terms]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{self.name} uses the same term key twice.")

        placeholders = {
            field
            for _, field, _, _ in Formatter().parse(self.substitution_template)
            if field
        }

        missing = placeholders - set(keys)
        if missing:
            raise ValueError(
                f"{self.name}'s substitution uses {sorted(missing)}, which "
                "no term defines. Every value shown has to be a term a "
                "reader can look up."
            )

        unused = set(keys) - placeholders
        if unused:
            raise ValueError(
                f"{self.name} defines {sorted(unused)} but never uses "
                "them. A term nobody can see in the arithmetic is anatomy "
                "for a formula this is not."
            )

    @property
    def substituted(self) -> str:
        """The arithmetic with this reader's own values filled in."""
        values = {term.key: term.display_value for term in self.terms}
        return f"{self.substitution_template.format(**values)} = {self.result_display}"

    @property
    def concept(self) -> str:
        """The labeled blocks as one line, for a compact rendering."""
        return " ".join(self.blocks)

    def term(self, key: str) -> FormulaTerm:
        """One term by key, raising on an unknown one."""
        for candidate in self.terms:
            if candidate.key == key:
                return candidate

        raise ValueError(
            f"'{key}' is not a term of {self.name}. Known terms: "
            f"{', '.join(t.key for t in self.terms)}."
        )
