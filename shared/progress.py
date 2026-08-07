"""
Which workflows have recorded an analysis, and which have not.

Every module reports on the analysis you just ran. Nothing told a researcher
what they had not looked at. This joins the records in shared/handoff.py
against the catalog so the overview can say, per workflow, whether an
analysis has been recorded in this session.

The wording is a closed set. "Recorded" means an analysis ran and its result
was stored. It does not mean the corresponding validation stage has been
validated, which is why "Complete", "Incomplete", "Validated", and "Passed"
are not available words: running a workflow is not the same as having
validated anything, and a status implying otherwise would undercut every
caveat the module pages attach to their results. ALLOWED_STATES enforces
this as a closed set rather than as a list of forbidden words, so adding a
third state has to be a deliberate edit here.

No streamlit import. The join is the part that can be silently wrong, so it
is kept testable against a plain dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from shared.catalog import WORKFLOWS, Workflow
from shared.handoff import HandoffEntry

STATE_RECORDED = "Recorded"
STATE_NOT_ASSESSED = "Not assessed"

# The only states a workflow can be in. See the module docstring for why this
# is an allowlist.
ALLOWED_STATES: frozenset[str] = frozenset({STATE_RECORDED, STATE_NOT_ASSESSED})

# Shown instead of a status for a workflow that records nothing, which turns
# an unavoidable blank into an explanation of how the pages relate.
READS_OTHER_RECORDS = "Reads what the other workflows recorded."

# Shown once above the cards. Records live in the session, so they are gone on
# a reload, and a status that looks persistent would be a false promise.
SESSION_SCOPE_NOTE = (
    "Statuses below cover this browser session only. Reloading clears them."
)


@dataclass(frozen=True)
class WorkflowProgress:
    """
    One workflow's recording status.

    state is None for a workflow that records nothing, which is a different
    thing from having recorded nothing. Cross-Analysis Implications reads the
    other workflows' records, so "Not assessed" would be false for it and
    "Recorded" would be meaningless.

    dataset and recorded_at are populated only alongside STATE_RECORDED. The
    primary statistic is deliberately not carried here: a bare number on a
    card is severed from the interpretation band and caveats that make it
    mean anything, and those live on the module page.
    """

    workflow: Workflow
    state: str | None = None
    dataset: str | None = None
    recorded_at: str | None = None

    def __post_init__(self) -> None:
        if self.state is None:
            if self.workflow.module_key is not None:
                raise ValueError(
                    f"{self.workflow.workflow} records under "
                    f"'{self.workflow.module_key}', so it must have a state. "
                    "Only a workflow with no module_key has none."
                )
            return

        if self.state not in ALLOWED_STATES:
            raise ValueError(
                f"'{self.state}' is not an available status. OpenMeasure "
                f"records that an analysis was performed and must not imply "
                f"that a validation stage is finished. Use one of: "
                f"{', '.join(sorted(ALLOWED_STATES))}."
            )

    @property
    def is_recorded(self) -> bool:
        return self.state == STATE_RECORDED


def has_any_records(entries: tuple[HandoffEntry, ...]) -> bool:
    """
    Whether anything has been recorded at all.

    The overview shows no status until this is true, so a first visit is not
    a wall of "Not assessed" telling someone off for work they have not had
    the chance to do yet.
    """

    return bool(entries)


def workflow_progress(
    entries: tuple[HandoffEntry, ...],
    workflows: tuple[Workflow, ...] = WORKFLOWS,
) -> tuple[WorkflowProgress, ...]:
    """
    Status for every workflow, in catalog order.

    Records are keyed by module_key. A record whose key matches no workflow is
    ignored rather than reported, since there is no card to attach it to.
    """

    by_key = {entry.module: entry for entry in entries}

    progress: list[WorkflowProgress] = []

    for workflow in workflows:
        if workflow.module_key is None:
            progress.append(WorkflowProgress(workflow=workflow))
            continue

        entry = by_key.get(workflow.module_key)

        if entry is None:
            progress.append(
                WorkflowProgress(workflow=workflow, state=STATE_NOT_ASSESSED)
            )
            continue

        progress.append(
            WorkflowProgress(
                workflow=workflow,
                state=STATE_RECORDED,
                dataset=entry.fingerprint.filename,
                recorded_at=entry.recorded_at,
            )
        )

    return tuple(progress)


def _clock(recorded_at: str | None) -> str | None:
    """
    The time part of a stored timestamp, labelled UTC.

    Records store UTC. Rendering it as local time would be a lie on a
    deployed app, where the server's timezone is not the reader's, so the
    label stays explicit rather than converted.
    """

    if not recorded_at:
        return None

    try:
        moment = datetime.fromisoformat(recorded_at)
    except ValueError:
        return None

    return f"{moment.strftime('%H:%M')} UTC"


def status_caption(progress: WorkflowProgress) -> str:
    """
    The single line shown on a card.

    Always begins with the state, so the two words a reader scans for are the
    first thing they meet. Names the dataset because someone who has analyzed
    two files needs to know which result the card refers to.
    """

    if progress.state is None:
        return READS_OTHER_RECORDS

    if progress.state != STATE_RECORDED:
        return progress.state

    parts = [progress.state]

    if progress.dataset:
        parts.append(progress.dataset)

    clock = _clock(progress.recorded_at)

    if clock:
        parts.append(clock)

    return ", ".join(parts)
