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

Preserving state
----------------
Streamlit discards a widget's value when the widget is not rendered, and
in a workspace where only the current stage renders, that is every widget
in every other stage. keep() and kept() hold a stage's selections in
session state under their own names, so leaving a stage and coming back
finds it as it was.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import streamlit as st

from shared import feedback

# What a stage is, from the reader's side. Four states rather than
# reached/not-reached, because "was complete, and something it depended
# on has changed" is a real situation that neither of the other two
# describes.
STATE_NOT_REACHED = "Not reached"
STATE_CURRENT = "Current"
STATE_COMPLETE = "Complete"
STATE_NEEDS_REVIEW = "Needs review"

# The mark each state carries on the rail. Shape rather than colour, so
# the rail is legible without colour and a needs-review stage is
# distinguishable from a complete one at a glance.
STATE_MARKS = {
    STATE_COMPLETE: "●",
    STATE_CURRENT: "◉",
    STATE_NOT_REACHED: "○",
    STATE_NEEDS_REVIEW: "△",
}


@dataclass(frozen=True)
class Gate:
    """
    What has to be true before a stage can be entered.

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

        digest = hashlib.sha256(
            json.dumps(value, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        slot = self._name(f"input_{name}")
        previous = st.session_state.get(slot)
        st.session_state[slot] = digest

        if previous is None or previous == digest:
            return

        review = self._review()
        changed = False

        for key in affects:
            index = self._index_of(key)
            if index > self.furthest:
                continue

            causes = set(review.get(index, ()))
            if label not in causes:
                review[index] = sorted(causes | {label})
                changed = True

        if not changed:
            return

        st.session_state[self._name("review")] = review
        st.rerun()

    def clear_review(self, index: int) -> None:
        """Mark a reviewed stage as settled again."""
        review = self._review()
        review.pop(index, None)
        st.session_state[self._name("review")] = review

    def needs_review(self, index: int) -> bool:
        return index in self._review()

    def review_causes(self, index: int) -> tuple[str, ...]:
        """What changed upstream of a flagged stage."""
        return tuple(self._review().get(index, ()))

    # -- state -------------------------------------------------------

    def state_of(self, index: int) -> str:
        if index == self.current:
            return STATE_CURRENT
        if self.needs_review(index):
            return STATE_NEEDS_REVIEW
        if index <= self.furthest:
            return STATE_COMPLETE

        return STATE_NOT_REACHED

    def gate_for(self, index: int) -> Gate | None:
        return self.gates.get(self.stages[index].key)

    def blocked_reason(self, index: int) -> str:
        """
        Why a stage cannot be entered yet, or an empty string.

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
        """
        st.session_state[self._name(f"kept_{key}")] = value

    def kept(self, key: str, default=None):
        """A selection held from an earlier visit to its stage."""
        return st.session_state.get(self._name(f"kept_{key}"), default)

    # -- rendering ---------------------------------------------------

    def render_rail(self) -> int:
        """
        The rail, and the current stage.

        Every reached stage is a button, so going back is a click rather
        than a scroll. An unreached stage is disabled and says what would
        unblock it, which is more use than a control that silently does
        nothing.
        """
        # Left for the feedback footer, which renders after the page and
        # would otherwise have to be passed the stage by every caller.
        st.session_state[feedback.ACTIVE_STAGE_KEY] = self.stages[
            self.current
        ].label

        columns = st.columns(len(self.stages))

        for index, (stage, column) in enumerate(zip(self.stages, columns)):
            state = self.state_of(index)
            reason = self.blocked_reason(index)
            reachable = index <= self.furthest or not reason

            with column:
                if st.button(
                    f"{STATE_MARKS[state]} {stage.label}",
                    key=self._name(f"rail_{index}"),
                    disabled=not reachable and index != self.current,
                    width="stretch",
                    type="primary" if state == STATE_CURRENT else "secondary",
                ):
                    self.go_to(index)
                    st.rerun()

                if state == STATE_NEEDS_REVIEW:
                    causes = self.review_causes(index)
                    st.caption(
                        f"{', '.join(causes)} changed" if causes else "Review"
                    )
                elif not reachable:
                    st.caption(reason)

        return self.current

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
        long stage does not have to return to the rail to move on.
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
