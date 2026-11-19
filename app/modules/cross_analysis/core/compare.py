"""
Setting two recorded analyses beside each other.

The page this serves used to mean one thing by "cross-analysis": how many
observations each module retained from the same file. That is worth
knowing and it is not what the name promises. What a researcher running
two analyses on one dataset wants to know is what each found, what each
rests on, and what the two together do and do not support.

Nothing here scores agreement. A single number saying how much two
analyses agree would be the one output this module must not produce: it
would collapse a judgment that depends on knowing what the two analyses
are into a figure that looks like a measurement. These functions expose
the relationships and their limits, and the reading stays with the
reader.

The third comparison state does most of the work. Two analyses reporting
different kinds of quantity are not in agreement and not in conflict,
they are not comparable, and a view offering only converge or diverge
has to force them into one of those. "Not directly comparable" is a
first-class outcome here for the same reason a dashed line is elsewhere.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.handoff import HandoffEntry  # noqa: E402

# How two findings stand to each other.
CONVERGE = "Point the same way"
DIVERGE = "Point different ways"
NOT_COMPARABLE = "Not directly comparable"

COMPARISONS: frozenset[str] = frozenset({CONVERGE, DIVERGE, NOT_COMPARABLE})


@dataclass(frozen=True)
class FindingPair:
    """Two recorded findings, and what can be said about the pair."""

    module_a: str
    module_b: str
    label_a: str
    label_b: str
    reading_a: str
    reading_b: str
    comparison: str
    explanation: str
    # Both analyses ran on one upload and still may not have used the
    # same rows. Carried separately from the comparison itself, because
    # it qualifies a converging pair as much as a diverging one.
    same_retained_count: bool

    def __post_init__(self) -> None:
        if self.comparison not in COMPARISONS:
            raise ValueError(
                f"'{self.comparison}' is not a known comparison. Known: "
                f"{', '.join(sorted(COMPARISONS))}."
            )

        if not self.explanation:
            raise ValueError(
                f"The pair {self.module_a} and {self.module_b} has no "
                "explanation. A comparison that does not say why it reads "
                "that way is a verdict."
            )


@dataclass(frozen=True)
class AssumptionDifference:
    """One condition, and which of the recorded analyses depends on it."""

    name: str
    modules: tuple[str, ...]
    shared: bool


@dataclass(frozen=True)
class CrossReading:
    """Everything a comparison view has to show, for one dataset."""

    modules: tuple[str, ...]
    pairs: tuple[FindingPair, ...]
    assumptions: tuple[AssumptionDifference, ...]
    modules_without_findings: tuple[str, ...]

    @property
    def has_comparable_pair(self) -> bool:
        """Whether any two findings could be set beside each other."""
        return any(pair.comparison != NOT_COMPARABLE for pair in self.pairs)


# Said whenever two analyses of one upload retained different numbers of
# rows. The page's strongest existing idea, and it was buried in prose:
# equal counts do not establish the same subset either, so this is worded
# as a difference that is visible rather than as the only difference
# there is.
RETENTION_DIFFERS_NOTE = (
    "These two analyses kept different numbers of rows from the same "
    "upload, so they describe overlapping but different sets of "
    "observations."
)

RETENTION_MATCHES_NOTE = (
    "These two analyses kept the same number of rows. Equal counts do not "
    "establish that the same rows were kept, which nothing recorded here "
    "can show."
)


def _retained(entry: HandoffEntry) -> int | None:
    """How many rows an analysis kept, where it reports a count at all."""
    return entry.exclusion.n_retained_rows


def _compare(finding_a, finding_b) -> tuple[str, str]:
    """How two findings stand to each other, and why."""
    if finding_a.quantity != finding_b.quantity:
        return (
            NOT_COMPARABLE,
            f"One reports {finding_a.quantity.lower()} and the other "
            f"{finding_b.quantity.lower()}. Both are numbers about the same "
            "upload, and they answer different questions, so neither "
            "agreement nor disagreement between them is defined.",
        )

    if finding_a.reading == finding_b.reading:
        return (
            CONVERGE,
            f"Both read as {finding_a.reading.lower()}. Two analyses "
            "reaching the same reading is not independent confirmation "
            "when they run on the same data.",
        )

    return (
        DIVERGE,
        f"One reads as {finding_a.reading.lower()} and the other as "
        f"{finding_b.reading.lower()}, on the same quantity and the same "
        "upload. Which is right, or whether the difference is in what each "
        "analysis used, is not settled by the fact that they differ.",
    )


def compare_findings(entries: tuple[HandoffEntry, ...]) -> tuple[FindingPair, ...]:
    """
    Every pairing of recorded findings, with how the two stand.

    Every pairing rather than the comparable ones only. A pair that
    cannot be compared is a fact about the analyses worth showing, and
    dropping it would leave a reader to wonder whether the comparison was
    made and failed or never attempted.
    """
    with_findings = [entry for entry in entries if entry.findings]

    pairs: list[FindingPair] = []

    for entry_a, entry_b in combinations(with_findings, 2):
        retained_a, retained_b = _retained(entry_a), _retained(entry_b)

        for finding_a in entry_a.findings:
            for finding_b in entry_b.findings:
                comparison, explanation = _compare(finding_a, finding_b)

                pairs.append(
                    FindingPair(
                        module_a=entry_a.module,
                        module_b=entry_b.module,
                        label_a=finding_a.label,
                        label_b=finding_b.label,
                        reading_a=finding_a.reading,
                        reading_b=finding_b.reading,
                        comparison=comparison,
                        explanation=explanation,
                        same_retained_count=(
                            retained_a is not None
                            and retained_b is not None
                            and retained_a == retained_b
                        ),
                    )
                )

    return tuple(pairs)


def compare_assumptions(
    entries: tuple[HandoffEntry, ...],
) -> tuple[AssumptionDifference, ...]:
    """
    Which conditions the recorded analyses depend on, and which share
    them.

    Sorted with the shared ones first, because a condition two results
    both rest on is the one whose failure would move both of them, and
    that is the case a reader comparing two analyses most needs to see.
    """
    by_name: dict[str, list[str]] = {}

    for entry in entries:
        for assumption in entry.assumptions:
            modules = by_name.setdefault(assumption.name, [])
            if entry.module not in modules:
                modules.append(entry.module)

    differences = [
        AssumptionDifference(
            name=name,
            modules=tuple(modules),
            shared=len(modules) > 1,
        )
        for name, modules in by_name.items()
    ]

    return tuple(
        sorted(differences, key=lambda item: (not item.shared, item.name))
    )


def read_across(entries: tuple[HandoffEntry, ...]) -> CrossReading:
    """
    What one dataset's recorded analyses say next to each other.

    Modules that recorded no finding are named rather than omitted. A
    module wired to record retention and not yet wired to record what it
    found is absent from the comparison for a reason a reader should be
    able to see.
    """
    return CrossReading(
        modules=tuple(entry.module for entry in entries),
        pairs=compare_findings(entries),
        assumptions=compare_assumptions(entries),
        modules_without_findings=tuple(
            entry.module for entry in entries if not entry.findings
        ),
    )
