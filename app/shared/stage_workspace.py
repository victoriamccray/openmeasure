"""
Moving through a research process, rather than scrolling past it.

OpenMeasure's staged pages grew by appending: each new stage went below
the last, so a reader reached stage seven by scrolling past six. That
makes stages read as sections of a report when they are decisions that
progressively construct an analysis, and it makes going back to change
one a matter of scrolling upward and hoping.

The rule this module implements:

    Stages move horizontally through the research process. Scrolling
    moves vertically through the evidence inside the current stage.

So one stage occupies the workspace, a rail across the top says where
you are, and the rail is how you move.

Not a wizard
------------
A wizard says step one is settled, now do step two. This is a research
notebook with stages: forward when ready, backward at any time,
selections preserved, and downstream decisions visibly flagged when an
earlier one changes.

That last part is the reason this is a module rather than a layout. A
stage that was complete before an upstream change is not still complete,
and it is not unreached either. It needs review, which is a third thing,
and a page that silently kept it green would be asserting that changing
a measure cannot affect the timing chosen for it.

Which downstream stages a change reaches is declared, not assumed.
Everything after the edit is the wrong answer: rewording a research
question does not invalidate a timing decision, and a page that said it
did would train a reader to dismiss the flag. So a recorded input names
the stages it actually feeds.

A flag also carries what caused it. "Something changed" is not enough to
act on once a study has several inputs; "Measures changed" tells a reader
what to look at.

Nothing here erases a downstream selection when an upstream one changes.
Erasing would be a decision about the reader's work; flagging leaves it
to them.

The state model is specified
----------------------------
Six states, six behaviours, written down in docs/stage-lifecycle.md and
tested against synthetic pages in shared/tests/test_stage_workspace.py.

That document exists because the first three migrations each discovered a
rule the previous one had not needed, and a shared component that learns
its own semantics one page at a time becomes fragile in exactly that way.
Two of those rules are enforced here rather than left to page code:

- A stage body may not assume its gate still holds. ``furthest`` is
  remembered independently of the gates, so a stage already reached stays
  on the rail after its prerequisite disappears. ``render_rail`` returns
  ``STAGE_UNAVAILABLE`` rather than the current index in that case, so no
  stage body runs and a page that forgets to check gets an explanation
  instead of a traceback.

- Raw data does not cross a stage boundary. ``keep`` refuses a DataFrame,
  so shared/upload.py's promise that nothing retains a reader's data is a
  refusal rather than a comment. A derived result may cross; the rows it
  came from may not.

Preserving state
----------------
Streamlit discards a widget's value when the widget is not rendered, and
in a workspace where only the current stage renders, that is every widget
in every other stage. keep() and kept() hold a stage's selections in
session state under their own names, so leaving a stage and coming back
finds it as it was.

Putting a held value back into its widget is the page's job, because this
module does not know a page's widget keys. docs/stage-lifecycle.md gives
the two-line pattern and says what it looks like when a page omits it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import streamlit as st

from shared import feedback

# What a stage is, from the reader's side. Six rather than two, because
# "was complete, and something it depended on has changed" and "was
# reached, and cannot be opened right now" are both real situations that
# reached/not-reached does not describe.
STATE_NOT_REACHED = "Not reached"
STATE_CURRENT = "Current"
STATE_COMPLETE = "Complete"
STATE_NEEDS_REVIEW = "Needs review"
STATE_REVIEWED = "Reviewed"
STATE_UNAVAILABLE = "Unavailable"

# The mark each state carries on the rail. Shape rather than colour, so
# the rail is legible without colour and a challenged stage is
# distinguishable from a settled one at a glance.
#
# The circle family means never challenged. The triangle family means
# challenged, hollow unanswered and filled answered. The slash means the
# stage cannot be entered right now whatever its history.
STATE_MARKS = {
    STATE_COMPLETE: "●",
    STATE_CURRENT: "◉",
    STATE_NOT_REACHED: "○",
    STATE_NEEDS_REVIEW: "△",
    STATE_REVIEWED: "▲",
    STATE_UNAVAILABLE: "⊘",
}

# What render_rail returns instead of a stage index when the current
# stage cannot render. No stage constant equals it, so every
# `if stage == STAGE_X:` fails and no body runs.
STAGE_UNAVAILABLE = -1

# How many stages the rail will put in one row.
#
# Streamlit's normal content width divided eleven ways is about sixty
# pixels, which truncates any label longer than a word to an ellipsis.
# Six keeps a cell wide enough to letter "Measurement" or "Conditions",
# and a page with more stages than this gets balanced rows rather than
# one unreadable one.
MAX_RAIL_COLUMNS = 6


def _digest(value) -> str:
    """A stable fingerprint of a recorded value."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _is_raw_data(value) -> bool:
    """
    Whether this is a table of a reader's rows.

    Checked by module rather than by importing pandas, so this module
    stays free of a dependency it would use for one guard. It catches the
    obvious mistake, not a frame hidden inside a result object, and not a
    page writing to st.session_state directly: no type check can promise
    either, which is why the policy is written down as well as enforced.
    """
    return (type(value).__module__ or "").split(".")[0] == "pandas"


def _refuse_raw_data(where: str, value) -> None:
    """Raise if a reader's rows are being put somewhere they persist."""
    if _is_raw_data(value):
        raise TypeError(
            f"{where} was given a {type(value).__name__}. Raw data does "
            "not cross a stage boundary: a derived result may, and the "
            "rows behind it may not. See docs/stage-lifecycle.md."
        )


@dataclass(frozen=True)
class Gate:
    """
    What has to be true before a stage can be entered.

    ``satisfied`` is re-evaluated on every run and may go false again. A
    stage already reached stays on the rail, so this is not a promise its
    body can rely on; see STAGE_UNAVAILABLE.

    ``optional`` separates information that would improve the analysis
    from information without which the next stage cannot run. A page that
    treated both the same would make a researcher invent a setting or a
    population to unlock a screen, which is the opposite of what this
    toolkit is for.
    """

    satisfied: bool
    requirement: str
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.requirement.strip():
            raise ValueError(
                "A gate needs a requirement, so a blocked stage can say "
                "what would unblock it rather than only that it is blocked."
            )

    @property
    def blocks(self) -> bool:
        """Whether this actually prevents entry."""
        return not self.satisfied and not self.optional


@dataclass(frozen=True)
class Stage:
    """One stage of a research process."""

    key: str
    label: str

    def __post_init__(self) -> None:
        for name in ("key", "label"):
            if not getattr(self, name).strip():
                raise ValueError(f"A stage is missing its {name}.")


@dataclass
class StageWorkspace:
    """
    One page's stages, and where the reader is among them.

    ``session_key`` must be unique per page. Every piece of state this
    holds is namespaced under it, so two workspaces on one page do not
    read each other's position.
    """

    session_key: str
    stages: tuple[Stage, ...]
    gates: dict[str, Gate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.stages) < 2:
            raise ValueError(
                "A workspace needs at least two stages; one stage is a page."
            )

        labels = [stage.key for stage in self.stages]
        if len(labels) != len(set(labels)):
            raise ValueError("Stage keys must be unique within a workspace.")

        unknown = set(self.gates) - set(labels)
        if unknown:
            raise ValueError(
                f"Gates name stages this workspace does not have: "
                f"{', '.join(sorted(unknown))}."
            )

        # A required gate on the opening stage is a deadlock: it blocks
        # the one screen from which it could be satisfied. An optional
        # one is fine, and is how a page names a gap without demanding it.
        first = self.stages[0].key
        if first in self.gates and self.gates[first].blocks:
            raise ValueError(
                f"'{first}' is the opening stage and carries a required "
                "gate, which would block the only screen able to satisfy "
                "it. Make the gate optional, or move it to the stage that "
                "needs it."
            )

    # -- position ----------------------------------------------------

    def _name(self, suffix: str) -> str:
        return f"{self.session_key}_{suffix}"

    @property
    def current(self) -> int:
        return int(st.session_state.get(self._name("current"), 0))

    @property
    def furthest(self) -> int:
        """The furthest stage reached, which is not where the reader is."""
        return int(st.session_state.get(self._name("furthest"), 0))

    def go_to(self, index: int) -> None:
        """Move to a stage, remembering how far the reader has been."""
        if not 0 <= index < len(self.stages):
            raise ValueError(
                f"Stage {index} is outside this workspace's "
                f"{len(self.stages)} stages."
            )

        st.session_state[self._name("current")] = index
        st.session_state[self._name("furthest")] = max(self.furthest, index)

    # -- what changed ------------------------------------------------

    def _review(self) -> dict:
        """Stage index -> the labels of the inputs that changed under it."""
        return dict(st.session_state.get(self._name("review"), {}))

    def _settled(self) -> set:
        """Stages that were flagged and then confirmed by the reader."""
        return set(st.session_state.get(self._name("settled"), set()))

    def _index_of(self, key: str) -> int:
        for index, stage in enumerate(self.stages):
            if stage.key == key:
                return index

        raise ValueError(
            f"'{key}' is not a stage in this workspace. Known: "
            f"{', '.join(stage.key for stage in self.stages)}."
        )

    def record_input(
        self, name: str, value, *, affects: tuple[str, ...], label: str
    ) -> None:
        """
        Record something later stages depend on, and which ones.

        ``affects`` is the specific list of stage keys this input feeds.
        Flagging everything downstream would be the wrong answer:
        rewording a research question does not invalidate a timing
        decision, and a page that said it did would train a reader to
        dismiss the flag.

        ``label`` is what the flag says it was, because "something
        changed" is not enough to act on once a study has several inputs.

        Only stages already visited are flagged. A stage nobody has
        reached is not stale, it is unreached.

        Reruns when it newly flags something, because a page records its
        inputs inside the stage that produced them, which is below the
        rail. Without the rerun the rail would show the previous run's
        marks.
        """
        for key in affects:
            self._index_of(key)

        digest = _digest(value)
        slot = self._name(f"input_{name}")
        previous = st.session_state.get(slot)
        st.session_state[slot] = digest

        if previous is None or previous == digest:
            return

        review = self._review()
        settled = self._settled()
        changed = False

        for key in affects:
            index = self._index_of(key)
            if index > self.furthest:
                continue

            causes = set(review.get(index, ()))
            if label not in causes:
                review[index] = sorted(causes | {label})
                # A stage vouched for under the previous conditions is
                # not vouched for under these.
                settled.discard(index)
                changed = True

        if not changed:
            return

        st.session_state[self._name("review")] = review
        st.session_state[self._name("settled")] = settled
        st.rerun()

    def record_gate_input(self, name: str, value) -> None:
        """
        Record a value the gates are computed from, rerunning if it moved.

        The gates are evaluated above the stage that produces their
        inputs, because a workspace has to know what is reachable before
        it can draw the rail. So a value written now was not available to
        the gate that has already been drawn, and without a rerun a
        reader who has just satisfied a requirement still sees it unmet
        until some unrelated interaction redraws the page.

        Separate from record_input because this changes what is
        *reachable* rather than what needs *reviewing*, and conflating
        them would flag a stage for having become available.
        """
        digest = _digest(value)
        slot = self._name(f"gate_{name}")
        previous = st.session_state.get(slot)
        st.session_state[slot] = digest

        if previous is not None and previous != digest:
            st.rerun()

    def publish(self, key: str, value) -> None:
        """
        Store an artifact a later stage's gate tests, redrawing if it is new.

        The write side of a gate. A stage that produces what the next
        stage requires does so below the rail, and the rail was drawn
        from the state as it stood before this run. So the first time an
        artifact appears, the stage it unlocks is still showing as
        blocked, and stays that way until some unrelated interaction
        redraws the page.

        record_gate_input does not cover this. It compares digests, and a
        first write has no previous digest to differ from.

        ``key`` is a plain session-state name rather than a namespaced
        one, because the gate that tests it is written by the page and
        has to be able to name the same thing.

        Call it last in its block. It reruns, so any statement after it
        is skipped on the pass that first creates the artifact. Portfolio
        Impact Analysis lost its evidence frame that way the first time
        this was wired up, because two plain writes sat underneath it.

        Impact Evaluation hand-rolled this with a first_result flag
        before it moved here. Portfolio Impact Analysis has six of these
        artifacts and would have repeated it six times.
        """
        _refuse_raw_data(f"publish('{key}')", value)

        first = key not in st.session_state
        st.session_state[key] = value

        if first:
            st.rerun()

    def clear_review(self, index: int) -> None:
        """
        Record that a reader confirmed a flagged stage still applies.

        Moves it to Reviewed rather than back to Complete. Complete means
        nothing has questioned this stage; Reviewed means something did
        and a person vouched for it anyway, which is a stronger statement
        and a different provenance.
        """
        review = self._review()
        review.pop(index, None)
        st.session_state[self._name("review")] = review
        st.session_state[self._name("settled")] = self._settled() | {index}

    def needs_review(self, index: int) -> bool:
        return index in self._review()

    def reviewed(self, index: int) -> bool:
        """Whether a reader has confirmed this stage after a change."""
        return index in self._settled()

    def review_causes(self, index: int) -> tuple[str, ...]:
        """What changed upstream of a flagged stage."""
        return tuple(self._review().get(index, ()))

    # -- state -------------------------------------------------------

    def state_of(self, index: int) -> str:
        """
        How this stage stands, in one word from STATE_MARKS.

        Unavailable outranks Current: a reader standing on a stage whose
        prerequisite has gone is not working in it, and a rail that said
        otherwise would be describing a screen that is not there.
        """
        if index <= self.furthest and self.blocked_reason(index):
            return STATE_UNAVAILABLE
        if index == self.current:
            return STATE_CURRENT
        if self.needs_review(index):
            return STATE_NEEDS_REVIEW
        if self.reviewed(index):
            return STATE_REVIEWED
        if index <= self.furthest:
            return STATE_COMPLETE

        return STATE_NOT_REACHED

    def gate_for(self, index: int) -> Gate | None:
        return self.gates.get(self.stages[index].key)

    def blocked_reason(self, index: int) -> str:
        """
        Why a stage cannot be entered, or an empty string.

        A stage is blocked by its own gate, and by every blocking gate
        before it, because reaching stage four through a stage three that
        was never satisfied would skip the decision three exists for.
        """
        for position in range(index + 1):
            gate = self.gate_for(position)
            if gate is not None and gate.blocks:
                return gate.requirement

        return ""

    # -- keeping selections ------------------------------------------

    def keep(self, key: str, value) -> None:
        """
        Hold a selection so leaving its stage does not discard it.

        Streamlit drops a widget's value when the widget is not rendered,
        and in a workspace where one stage renders at a time that is every
        widget in every other stage.

        Refuses a reader's rows. A derived result may cross a stage
        boundary and the data it came from may not, which is
        shared/upload.py's promise that nothing here retains a reader's
        data, enforced rather than described.
        """
        _refuse_raw_data(f"keep('{key}')", value)

        st.session_state[self._name(f"kept_{key}")] = value

    def kept(self, key: str, default=None):
        """A selection held from an earlier visit to its stage."""
        return st.session_state.get(self._name(f"kept_{key}"), default)

    # -- rendering ---------------------------------------------------

    def render_rail(self) -> int:
        """
        The rail, and the stage a page should render.

        Every reached stage is a button, so going back is a click rather
        than a scroll. An unreached stage is disabled and says what would
        unblock it, which is more use than a control that silently does
        nothing.

        Returns STAGE_UNAVAILABLE, not the current index, when the
        current stage is blocked. Its prerequisite may have disappeared
        since the stage was reached, and a body that assumed otherwise is
        how Impact Evaluation came to call support_boundary_claims("").
        """
        # Left for the feedback footer, which renders after the page and
        # would otherwise have to be passed the stage by every caller.
        st.session_state[feedback.ACTIVE_STAGE_KEY] = self.stages[
            self.current
        ].label

        # Where the reader is, in words, above the marks. Eleven marks
        # say the shape of the process; they do not say which one you are
        # in as directly as a sentence does.
        st.caption(
            f"Stage {self.current + 1} of {len(self.stages)} · "
            f"{self.stages[self.current].label}"
        )

        for row in self._rail_rows():
            columns = st.columns(len(row))

            for index, column in zip(row, columns):
                stage = self.stages[index]
                state = self.state_of(index)
                reason = self.blocked_reason(index)
                reachable = index <= self.furthest or not reason

                with column:
                    # The reason and the review cause go in the tooltip,
                    # not under the button. Rendered per stage they
                    # repeated the same sentence down the whole rail, in
                    # a column too narrow to wrap it on word boundaries.
                    if state == STATE_UNAVAILABLE or not reachable:
                        note = reason
                    elif state == STATE_NEEDS_REVIEW:
                        causes = self.review_causes(index)
                        note = (
                            f"{', '.join(causes)} changed"
                            if causes
                            else "Needs review"
                        )
                    else:
                        note = state

                    if st.button(
                        f"{STATE_MARKS[state]} {stage.label}",
                        key=self._name(f"rail_{index}"),
                        disabled=not reachable and index != self.current,
                        width="stretch",
                        type=(
                            "primary" if state == STATE_CURRENT else "secondary"
                        ),
                        help=note,
                    ):
                        self.go_to(index)
                        st.rerun()

        # One shared line rather than one per locked stage, and the
        # earliest unmet requirement rather than the nearest, because
        # reaching a later stage through an unsatisfied earlier one would
        # skip the decision that one exists for.
        earliest_unmet = self.blocked_reason(len(self.stages) - 1)

        if earliest_unmet:
            st.caption(earliest_unmet)

        if self.state_of(self.current) == STATE_UNAVAILABLE:
            st.warning(
                f"{self.stages[self.current].label} was reached earlier "
                f"and cannot open now. {self.blocked_reason(self.current)}."
            )
            return STAGE_UNAVAILABLE

        return self.current

    def _rail_rows(self) -> tuple[tuple[int, ...], ...]:
        """
        The stage indices, in balanced rows of at most MAX_RAIL_COLUMNS.

        Balanced rather than filled: eleven stages read better as six and
        five than as six, then five in a row half as wide, and seven read
        better as four and three than as six and one.
        """
        total = len(self.stages)

        if total <= MAX_RAIL_COLUMNS:
            return (tuple(range(total)),)

        rows = -(-total // MAX_RAIL_COLUMNS)
        per_row = -(-total // rows)

        return tuple(
            tuple(range(start, min(start + per_row, total)))
            for start in range(0, total, per_row)
        )

    def render_review_notice(self) -> None:
        """Say why the current stage is flagged, where it is."""
        if not self.needs_review(self.current):
            return

        causes = self.review_causes(self.current)
        named = " and ".join(causes) if causes else "Something upstream"

        st.warning(
            f"{named} changed since this stage was last reviewed. Check "
            "whether these settings still apply."
        )

        if st.button("These still apply", key=self._name("settle")):
            self.clear_review(self.current)
            st.rerun()

    def render_navigation(self) -> None:
        """
        Back and forward, at the foot of the stage.

        Both are shown wherever they exist, so a reader partway down a
        long stage does not have to return to the rail to move on. Drawn
        even where the stage itself could not render, so a prerequisite
        that disappeared leaves a way out rather than a dead end.
        """
        back_column, forward_column = st.columns(2)
        index = self.current

        with back_column:
            if index > 0:
                previous = self.stages[index - 1]
                if st.button(
                    f"← Back to {previous.label}",
                    key=self._name("back"),
                    width="stretch",
                ):
                    self.go_to(index - 1)
                    st.rerun()

        with forward_column:
            if index < len(self.stages) - 1:
                following = self.stages[index + 1]
                reason = self.blocked_reason(index + 1)

                if st.button(
                    f"Continue to {following.label} →",
                    key=self._name("forward"),
                    disabled=bool(reason),
                    type="primary",
                    width="stretch",
                ):
                    self.go_to(index + 1)
                    st.rerun()

                if reason:
                    st.caption(reason)

    def render_optional_gaps(self) -> None:
        """
        Information that would help and is not required.

        Named separately from what blocks, so nobody invents a value to
        unlock a screen.
        """
        unmet = [
            gate.requirement
            for gate in self.gates.values()
            if gate.optional and not gate.satisfied
        ]

        if not unmet:
            return

        with st.expander(f"Optional, not yet given ({len(unmet)})"):
            for requirement in unmet:
                st.caption(requirement)
