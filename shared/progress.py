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

from shared.catalog import LIFECYCLE_STAGES, WORKFLOWS, Workflow
from shared.handoff import HandoffEntry

STATE_RECORDED = "Recorded"
STATE_NOT_ASSESSED = "Not assessed"

# The only states a workflow can be in. See the module docstring for why this
# is an allowlist.
ALLOWED_STATES: frozenset[str] = frozenset({STATE_RECORDED, STATE_NOT_ASSESSED})

# Stage states, for the strip that summarizes the lifecycle.
#
# A separate closed set from the workflow states above, because a stage can
# hold more than one workflow and therefore has cases a single workflow does
# not. Analysis holds both Impact Evaluation and Fairness today.
#
# STAGE_PARTLY_RECORDED exists so a stage with one of two workflows recorded
# does not read as "Recorded". Reporting the stage as recorded would let a
# glance at the strip suggest the stage was covered when half of it was not.
#
# STAGE_NO_MODULE is not a failing. The Research Question stage has no
# workflow at all, so "Not assessed" would blame the user for something the
# toolkit does not offer.
#
# STAGE_READS_RECORDS is a different case that looks the same from a distance.
# Interpretation does have a module, Cross-Analysis Implications, which reads
# the other workflows' records rather than producing one. Reporting it as
# having no module would be plainly false, and reporting it as unassessed
# would describe a status it can never reach.
STAGE_RECORDED = "Recorded"
STAGE_PARTLY_RECORDED = "Partly recorded"
STAGE_NOT_ASSESSED = "Not assessed"
STAGE_READS_RECORDS = "Reads records"
STAGE_NO_MODULE = "No module yet"

ALLOWED_STAGE_STATES: frozenset[str] = frozenset(
    {
        STAGE_RECORDED,
        STAGE_PARTLY_RECORDED,
        STAGE_NOT_ASSESSED,
        STAGE_READS_RECORDS,
        STAGE_NO_MODULE,
    }
)

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


@dataclass(frozen=True)
class StageProgress:
    """
    One lifecycle stage's recording status, for the strip.

    Three counts rather than two, because a stage can hold a workflow that
    records nothing. n_workflows is everything at the stage, n_recording is
    how many of those produce a record, and n_recorded is how many have. Only
    n_recording is a meaningful denominator.

    Deliberately not a score. The counts are carried so a caller could word
    things precisely, but the strip shows the state alone: a "1 of 2" counter
    would imply that 2 of 2 is the goal, which is the completeness claim this
    whole feature avoids making.
    """

    stage: str
    state: str
    n_workflows: int
    n_recording: int
    n_recorded: int

    def __post_init__(self) -> None:
        if self.stage not in LIFECYCLE_STAGES:
            raise ValueError(
                f"'{self.stage}' is not a known lifecycle stage. Valid "
                f"stages: {', '.join(LIFECYCLE_STAGES)}."
            )

        if self.state not in ALLOWED_STAGE_STATES:
            raise ValueError(
                f"'{self.state}' is not an available stage status. "
                f"OpenMeasure records that an analysis was performed and must "
                f"not imply that a validation stage is finished. Use one of: "
                f"{', '.join(sorted(ALLOWED_STAGE_STATES))}."
            )

        if self.n_recording > self.n_workflows:
            raise ValueError(
                f"{self.stage} reports {self.n_recording} recording workflows "
                f"out of {self.n_workflows} present, which is more than it "
                "has."
            )

        if self.n_recorded > self.n_recording:
            raise ValueError(
                f"{self.stage} reports {self.n_recorded} recorded analyses "
                f"across {self.n_recording} recording workflows, which is "
                "more than it has."
            )


def stage_progress(
    entries: tuple[HandoffEntry, ...],
    workflows: tuple[Workflow, ...] = WORKFLOWS,
) -> tuple[StageProgress, ...]:
    """
    Status for every lifecycle stage, in order.

    Every stage is present, including one with no workflow, because the strip
    shows the shape of a research lifecycle and omitting a stage would imply
    it does not exist.

    A workflow that records nothing is counted as present but not as a
    denominator. Treating Cross-Analysis Implications as an unrecorded
    workflow would leave Interpretation permanently unassessed, and dropping
    it entirely would claim the stage has no module when it has one.
    """

    by_stage: dict[str, list[WorkflowProgress]] = {
        stage: [] for stage in LIFECYCLE_STAGES
    }

    for item in workflow_progress(entries, workflows):
        by_stage[item.workflow.stage].append(item)

    stages: list[StageProgress] = []

    for stage in LIFECYCLE_STAGES:
        items = by_stage[stage]
        recording = [item for item in items if item.state is not None]
        n_recorded = sum(1 for item in recording if item.is_recorded)

        if not items:
            state = STAGE_NO_MODULE
        elif not recording:
            state = STAGE_READS_RECORDS
        elif n_recorded == 0:
            state = STAGE_NOT_ASSESSED
        elif n_recorded == len(recording):
            state = STAGE_RECORDED
        else:
            state = STAGE_PARTLY_RECORDED

        stages.append(
            StageProgress(
                stage=stage,
                state=state,
                n_workflows=len(items),
                n_recording=len(recording),
                n_recorded=n_recorded,
            )
        )

    return tuple(stages)


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
