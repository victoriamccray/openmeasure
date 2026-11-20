"""
The Decision Record, as Python objects.

A Decision Record documents one research decision: the question behind it,
the evidence and method it drew on, the conditions it rests on, the result,
what that result does and does not support, and where the record came
from. The field-by-field meaning is in docs/decision-record/specification.md
and the structure is pinned by docs/decision-record/decision_record.schema.json.

Modules populate this record; they do not invent their own documentation
format. Cross-module by definition, like shared/handoff.py, which is why it
lives here before a second module uses it.

This file does not import streamlit, so the record can be built and
checked from core/ and from tests.

The checks in __post_init__ repeat the schema's rules that a producer is
most likely to break while writing a mapping. They raise at construction,
where the mistake is, rather than when the record is later serialized and
validated. The schema remains the authority: shared/tests/test_decision_record.py
validates serialized records against it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Union

DECISION_RECORD_VERSION = "0.1"

ParameterValue = Union[str, int, float, bool, None, tuple]

# Who settled a choice. See "Origin Values" in the specification.
ORIGIN_RESEARCHER = "researcher"
ORIGIN_DEFAULT = "default"
ORIGIN_SUGGESTION_ACCEPTED = "suggestion_accepted"
ORIGIN_FIXED_BY_METHOD = "fixed_by_method"
ORIGINS = frozenset(
    {
        ORIGIN_RESEARCHER,
        ORIGIN_DEFAULT,
        ORIGIN_SUGGESTION_ACCEPTED,
        ORIGIN_FIXED_BY_METHOD,
    }
)

EVIDENCE_KINDS = frozenset({"dataset", "artifact", "literature"})
DATASET_STATUSES = frozenset({"real", "synthetic", "constructed", "unknown"})

# What is known about whether an assumption holds. See the specification's
# table under "assumptions".
STATUS_HELD_BY_DESIGN = "held_by_design"
STATUS_CHECKED_CONSISTENT = "checked_consistent"
STATUS_CHECKED_INCONSISTENT = "checked_inconsistent"
STATUS_CHECKED_INCONCLUSIVE = "checked_inconclusive"
STATUS_NOT_TESTABLE = "not_testable_with_these_data"
STATUS_REQUIRES_SETTING = "requires_knowledge_of_setting"
STATUS_NOT_CHECKED = "not_checked"
_CHECKED_STATUSES = frozenset(
    {
        STATUS_CHECKED_CONSISTENT,
        STATUS_CHECKED_INCONSISTENT,
        STATUS_CHECKED_INCONCLUSIVE,
    }
)
ASSUMPTION_STATUSES = _CHECKED_STATUSES | {
    STATUS_HELD_BY_DESIGN,
    STATUS_NOT_TESTABLE,
    STATUS_REQUIRES_SETTING,
    STATUS_NOT_CHECKED,
}

RETENTION_KINDS = frozenset({"rows_dropped", "cells_empty", "observations_absent"})

# Fields whose None is meaningful and serializes as JSON null. Every other
# None means "not given" and is left out of the serialized record.
_NULL_IS_MEANINGFUL = frozenset({"value", "n_retained_rows"})


def _require_text(owner: str, **values: str) -> None:
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{owner} is missing a value for '{name}'.")


def _require_member(owner: str, name: str, value: str, allowed: frozenset) -> None:
    if value not in allowed:
        raise ValueError(
            f"{owner} has {name} '{value}', which is not one of: "
            f"{', '.join(sorted(allowed))}."
        )


@dataclass(frozen=True)
class Question:
    text: str
    # Optional because a page may imply the question from the design the
    # researcher chose without the researcher stating it. Leaving it out
    # says that, rather than claiming an origin nobody can vouch for.
    origin: str | None = None

    def __post_init__(self) -> None:
        _require_text("The question", text=self.text)
        if self.origin is not None:
            _require_member("The question", "origin", self.origin, ORIGINS)


@dataclass(frozen=True)
class Fingerprint:
    algorithm: str
    digest: str
    n_rows: int | None = None
    n_columns: int | None = None
    columns: tuple[str, ...] | None = None


@dataclass(frozen=True)
class EvidenceItem:
    kind: str
    label: str
    description: str | None = None
    source: str | None = None
    status: str | None = None
    fingerprint: Fingerprint | None = None

    def __post_init__(self) -> None:
        _require_text("An evidence item", label=self.label)
        _require_member(self.label, "kind", self.kind, EVIDENCE_KINDS)
        if self.kind == "dataset":
            if self.status is None:
                raise ValueError(
                    f"Dataset '{self.label}' must state whether it is real, "
                    "synthetic, constructed, or unknown."
                )
            _require_member(self.label, "status", self.status, DATASET_STATUSES)


@dataclass(frozen=True)
class Alternative:
    option: str
    tradeoff: str | None = None

    def __post_init__(self) -> None:
        _require_text("An alternative", option=self.option)


@dataclass(frozen=True)
class Decision:
    statement: str
    origin: str
    alternatives: tuple[Alternative, ...] = ()

    def __post_init__(self) -> None:
        _require_text("The decision", statement=self.statement)
        _require_member("The decision", "origin", self.origin, ORIGINS)


@dataclass(frozen=True)
class Method:
    name: str
    parameters: Mapping[str, ParameterValue]
    identifier: str | None = None
    citations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text("The method", name=self.name)


@dataclass(frozen=True)
class RecordedAssumption:
    name: str
    statement: str
    status: str
    check: str | None = None
    citation: str | None = None

    def __post_init__(self) -> None:
        _require_text("An assumption", name=self.name, statement=self.statement)
        _require_member(self.name, "status", self.status, ASSUMPTION_STATUSES)
        if self.status in _CHECKED_STATUSES and not (self.check or "").strip():
            raise ValueError(
                f"Assumption '{self.name}' is marked '{self.status}' but does "
                "not say how it was checked."
            )


@dataclass(frozen=True)
class Interval:
    low: float
    high: float
    level: float
    method: str

    def __post_init__(self) -> None:
        if not 0 < self.level < 1:
            raise ValueError(
                f"An interval's level must be between 0 and 1, got {self.level}."
            )
        _require_text("An interval", method=self.method)


@dataclass(frozen=True)
class Quantity:
    name: str
    value: float | None
    value_note: str | None = None
    unit: str | None = None
    interval: Interval | None = None
    n: int | None = None

    def __post_init__(self) -> None:
        _require_text("A quantity", name=self.name)
        if self.value is None and not (self.value_note or "").strip():
            raise ValueError(
                f"Quantity '{self.name}' has no value and no value_note "
                "saying why it could not be computed."
            )


@dataclass(frozen=True)
class Convention:
    name: str
    citation: str
    threshold: ParameterValue = None

    def __post_init__(self) -> None:
        _require_text("A convention", name=self.name, citation=self.citation)


@dataclass(frozen=True)
class Reading:
    quantity: str
    text: str
    convention: Convention

    def __post_init__(self) -> None:
        _require_text("A reading", quantity=self.quantity, text=self.text)


@dataclass(frozen=True)
class Result:
    summary: str
    quantities: tuple[Quantity, ...]
    readings: tuple[Reading, ...] = ()
    observations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text("The result", summary=self.summary)
        names = {quantity.name for quantity in self.quantities}
        for reading in self.readings:
            if reading.quantity not in names:
                raise ValueError(
                    f"A reading refers to quantity '{reading.quantity}', "
                    "which is not among the result's quantities."
                )


@dataclass(frozen=True)
class Software:
    name: str
    version: str
    component: str | None = None

    def __post_init__(self) -> None:
        _require_text("The software", name=self.name, version=self.version)


@dataclass(frozen=True)
class RetentionEntry:
    label: str
    count: int
    kind: str
    mechanism: str

    def __post_init__(self) -> None:
        _require_text("A retention entry", label=self.label, mechanism=self.mechanism)
        _require_member(self.label, "kind", self.kind, RETENTION_KINDS)
        if self.count < 0:
            raise ValueError(f"Retention entry '{self.label}' has a negative count.")


@dataclass(frozen=True)
class DataHandling:
    evidence_label: str
    n_input_rows: int
    n_retained_rows: int | None
    retention: tuple[RetentionEntry, ...]
    columns_used: tuple[str, ...] = ()
    n_expanded_observations: int | None = None

    def __post_init__(self) -> None:
        _require_text("A data-handling entry", evidence_label=self.evidence_label)
        if self.n_retained_rows is None and self.n_expanded_observations is None:
            raise ValueError(
                f"Data handling for '{self.evidence_label}' reports neither "
                "retained rows nor expanded observations."
            )
        if self.n_retained_rows is not None and self.n_retained_rows > self.n_input_rows:
            raise ValueError(
                f"Data handling for '{self.evidence_label}' retains "
                f"{self.n_retained_rows} rows out of {self.n_input_rows}."
            )


@dataclass(frozen=True)
class ResearcherChoice:
    choice: str
    value: ParameterValue
    origin: str

    def __post_init__(self) -> None:
        _require_text("A researcher choice", choice=self.choice)
        _require_member(self.choice, "origin", self.origin, ORIGINS)


@dataclass(frozen=True)
class Provenance:
    software: Software
    generated_at: str
    data_handling: tuple[DataHandling, ...]
    researcher_choices: tuple[ResearcherChoice, ...]

    def __post_init__(self) -> None:
        _require_text("The provenance", generated_at=self.generated_at)


@dataclass(frozen=True)
class Unplaced:
    content: str
    why_no_field_fits: str

    def __post_init__(self) -> None:
        _require_text(
            "An unplaced entry",
            content=self.content,
            why_no_field_fits=self.why_no_field_fits,
        )


@dataclass(frozen=True)
class DecisionRecord:
    record_id: str
    question: Question
    evidence: tuple[EvidenceItem, ...]
    decision: Decision
    method: Method
    assumptions: tuple[RecordedAssumption, ...]
    result: Result
    supports: tuple[str, ...]
    does_not_support: tuple[str, ...]
    provenance: Provenance
    unplaced: tuple[Unplaced, ...] = ()
    record_version: str = DECISION_RECORD_VERSION

    def __post_init__(self) -> None:
        _require_text("The record", record_id=self.record_id)
        if not self.evidence:
            raise ValueError("A Decision Record needs at least one evidence item.")
        for name in ("supports", "does_not_support"):
            entries = getattr(self, name)
            if not entries:
                raise ValueError(
                    f"A Decision Record needs at least one '{name}' entry. "
                    "Every result has conclusions it does and does not reach."
                )
            for entry in entries:
                _require_text("The record", **{name: entry})

        dataset_labels = {item.label for item in self.evidence if item.kind == "dataset"}
        for handling in self.provenance.data_handling:
            if handling.evidence_label not in dataset_labels:
                raise ValueError(
                    f"Data handling refers to '{handling.evidence_label}', "
                    "which is not a dataset in the record's evidence."
                )

    def to_dict(self) -> dict[str, Any]:
        """The record as JSON-ready primitives, in the schema's shape."""

        return _serialize(self)


def _serialize(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        out: dict[str, Any] = {}
        for field_info in fields(value):
            item = getattr(value, field_info.name)
            if item is None and field_info.name not in _NULL_IS_MEANINGFUL:
                continue
            out[field_info.name] = _serialize(item)
        return out
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    # numpy scalars and similar: anything with item() collapses to a
    # Python primitive.
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize a {type(value).__name__} into a Decision Record.")


# -- the readable rendering ---------------------------------------------
#
# Built from the same DecisionRecord object as to_dict(), and from nothing
# else, so the two representations cannot disagree: a page shows this text
# and offers to_dict() as the download.

_STATUS_LABELS = {
    STATUS_HELD_BY_DESIGN: "Held by how the data are structured",
    STATUS_CHECKED_CONSISTENT: "Checked, data consistent with it",
    STATUS_CHECKED_INCONSISTENT: "Checked, data not consistent with it",
    STATUS_CHECKED_INCONCLUSIVE: "Checked, inconclusive",
    STATUS_NOT_TESTABLE: "Not testable with these data",
    STATUS_REQUIRES_SETTING: "Answered by knowledge of the setting, not by these data",
    STATUS_NOT_CHECKED: "Not checked",
}

_ORIGIN_LABELS = {
    ORIGIN_RESEARCHER: "set by the researcher",
    ORIGIN_DEFAULT: "software default, left in place",
    ORIGIN_SUGGESTION_ACCEPTED: "suggested by the software, accepted by the researcher",
    ORIGIN_FIXED_BY_METHOD: "fixed by the method",
}

NONE_RECORDED = "None recorded."

# Characters markdown would read as formatting. "$" matters most: Streamlit
# renders text between two of them as math, which would swallow a column
# name like "cost_$".
_MARKDOWN_SPECIALS = "\`*_$#|<>[]"


def _md(text: object) -> str:
    out = str(text)
    for character in _MARKDOWN_SPECIALS:
        out = out.replace(character, "\\" + character)
    return out


def _number(value: float | None) -> str:
    if value is None:
        return "not computed"
    return f"{value:.4g}"


def _bullets(lines: list[str]) -> list[str]:
    return [f"- {line}" for line in lines] if lines else [NONE_RECORDED]


def to_markdown(record: DecisionRecord) -> str:
    """The record as readable markdown, section by section in schema order."""

    sections: list[tuple[str, list[str]]] = []

    question = _md(record.question.text)
    if record.question.origin is not None:
        question += f" *({_ORIGIN_LABELS[record.question.origin]})*"
    else:
        question += " *(implied by the analysis chosen; not stated by the researcher)*"
    sections.append(("Question", [question]))

    evidence = []
    for item in record.evidence:
        line = f"**{_md(item.label)}** ({item.kind}"
        line += f", {item.status})" if item.status else ")"
        if item.description:
            line += f". {_md(item.description)}"
        if item.fingerprint is not None:
            line += f" Fingerprint `{item.fingerprint.digest}`."
        evidence.append(line)
    sections.append(("Evidence", _bullets(evidence)))

    decision = [
        f"{_md(record.decision.statement)} *({_ORIGIN_LABELS[record.decision.origin]})*",
        "",
        "Alternatives:",
        "",
    ] + _bullets(
        [
            f"**{_md(a.option)}**" + (f". {_md(a.tradeoff)}" if a.tradeoff else "")
            for a in record.decision.alternatives
        ]
    )
    sections.append(("Decision", decision))

    method = [f"**{_md(record.method.name)}**"]
    method += _bullets(
        [f"{_md(name)}: {_md(value)}" for name, value in record.method.parameters.items()]
    )
    if record.method.citations:
        method += [""] + [f"*{_md(citation)}*" for citation in record.method.citations]
    sections.append(("Method", method))

    assumptions = []
    for a in record.assumptions:
        line = f"**{_md(a.name)}** ({_STATUS_LABELS[a.status]}). {_md(a.statement)}"
        if a.check:
            line += f" Checked by: {_md(a.check)}"
        if a.citation:
            line += f" *{_md(a.citation)}*"
        assumptions.append(line)
    sections.append(("Assumptions", _bullets(assumptions)))

    result = [_md(record.result.summary), ""]
    for q in record.result.quantities:
        line = f"{_md(q.name)}: {_number(q.value)}"
        if q.unit:
            line += f" {_md(q.unit)}"
        if q.interval is not None:
            line += (
                f" ({q.interval.level:.0%} interval {_number(q.interval.low)} to "
                f"{_number(q.interval.high)}, {_md(q.interval.method)})"
            )
        if q.n is not None:
            line += f", n = {q.n}"
        if q.value_note:
            line += f". {_md(q.value_note)}"
        result.append(f"- {line}")
    for r in record.result.readings:
        result.append(
            f"- Reading of {_md(r.quantity)}: {_md(r.text)} "
            f"*Convention: {_md(r.convention.name)}, {_md(r.convention.citation)}*"
        )
    if record.result.observations:
        result += ["", "Observations:"] + _bullets([_md(o) for o in record.result.observations])
    sections.append(("Result", result))

    sections.append(("Supports", _bullets([_md(s) for s in record.supports])))
    sections.append(("Does Not Support", _bullets([_md(s) for s in record.does_not_support])))

    software = record.provenance.software
    provenance = [
        f"- Software: {_md(software.name)} {_md(software.version)}"
        + (f", {_md(software.component)}" if software.component else ""),
        f"- Generated: {_md(record.provenance.generated_at)}",
    ]
    for h in record.provenance.data_handling:
        if h.n_retained_rows is not None:
            used = f"{h.n_retained_rows} of {h.n_input_rows} rows used"
        else:
            used = f"{h.n_input_rows} rows expanded to {h.n_expanded_observations} observations"
        line = f"- {_md(h.evidence_label)}: {used}"
        if h.columns_used:
            line += f", columns {', '.join(_md(c) for c in h.columns_used)}"
        provenance.append(line)
        for entry in h.retention:
            provenance.append(
                f"    - {_md(entry.label)}: {entry.count} ({entry.kind.replace('_', ' ')}, "
                f"{_md(entry.mechanism)})"
            )
    choices = [
        f"{_md(c.choice)}: {_md(c.value)} ({_ORIGIN_LABELS[c.origin]})"
        for c in record.provenance.researcher_choices
    ]
    provenance += ["", "Researcher choices:", ""] + _bullets(choices)
    sections.append(("Provenance", provenance))

    if record.unplaced:
        sections.append(
            (
                "Recorded Outside the Schema",
                [
                    "Content that no v0.1 field fits, kept as evidence for the next revision.",
                    "",
                ]
                + _bullets(
                    [
                        f"{_md(u.content)} *Why no field fits: {_md(u.why_no_field_fits)}*"
                        for u in record.unplaced
                    ]
                ),
            )
        )

    lines = [f"*Decision Record v{record.record_version}, {_md(record.record_id)}*", ""]
    for heading, body in sections:
        lines += [f"#### {heading}", ""] + body + [""]
    return "\n".join(lines).rstrip() + "\n"
