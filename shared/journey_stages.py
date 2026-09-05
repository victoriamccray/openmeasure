"""
Shared stage-gating UI for OpenMeasure's Research Journeys.

Every Research Journey page (see shared/research_journeys.py) reveals its
content one stage at a time: a stage's content renders once a reader has
made a decision or inspected a result in the stage before it, rather than
the whole journey being one long scroll visible at once. Before this
module existed, all six journeys reimplemented the same three pieces of
that mechanism independently: a session_state counter for the highest
unlocked stage, a rerun-on-unlock helper, and a breadcrumb-plus-restart
block rendered above the stages. That duplication is what StageTracker
replaces, and is also why it existed only as six near-identical, silently
divergent copies: pyfMRIqc's fourth stage was checked for but never
actually unlocked by any call, leaving it permanently unreachable.

This is UI code, not core logic, so it lives in shared/ rather than a
module's core/ for the same reason shared/report.py does (see that
module's docstring): Streamlit-dependent code used by more than one page
belongs here rather than duplicated across pages/.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class StageTracker:
    """
    Tracks the highest stage a reader has unlocked in one gated sequence.

    session_key must be unique among every tracker active on the same
    page (a page may use a second StageTracker for a nested sequence
    within one of its stages, as Multimodal Signal Convergence's
    cost-tradeoff stage does). stage_labels is the breadcrumb, in stage
    order; its length is the number of stages, and advance_to() rejects
    any index outside that range, so a typo in a call site fails loudly
    rather than silently unlocking nothing.
    """

    session_key: str
    stage_labels: tuple[str, ...]

    def current(self) -> int:
        return st.session_state.get(self.session_key, 0)

    def is_unlocked(self, stage: int) -> bool:
        return self.current() >= stage

    def advance_to(self, stage: int) -> None:
        """
        Unlock through `stage` and force an immediate rerun.

        A rerun is needed, not just the session_state write, because the
        button click that calls this is already mid-script: without
        rerunning, the rest of this same pass would still read the stale
        frontier, and the newly unlocked section would not appear until
        some later, unrelated interaction triggered a rerun on its own.
        """

        if not 0 <= stage < len(self.stage_labels):
            raise ValueError(
                f"Stage {stage} is out of range for '{self.session_key}', "
                f"which declares {len(self.stage_labels)} stage_labels."
            )

        st.session_state[self.session_key] = max(self.current(), stage)
        st.rerun()

    def mark_reached(self, stage: int) -> None:
        """
        Unlock through `stage` without rerunning.

        advance_to() is for a button: the reader asked to move on, and the
        rerun is what draws the newly unlocked section. This is for a stage
        the page has already decided to render in the current pass, where
        a rerun would discard the very thing that unlocked it. Impact
        Evaluation's interpretation stage is the case: it opens because an
        analysis just produced a result, and that result exists only in
        this pass.

        Same range check as advance_to(), so an out-of-range stage still
        fails loudly rather than recording a frontier no label matches.
        """

        if not 0 <= stage < len(self.stage_labels):
            raise ValueError(
                f"Stage {stage} is out of range for '{self.session_key}', "
                f"which declares {len(self.stage_labels)} stage_labels."
            )

        st.session_state[self.session_key] = max(self.current(), stage)

    def render_breadcrumb(self) -> int:
        """
        Render the stage breadcrumb and return the current stage.

        Plain wrapped text, not a fixed grid of columns: a journey can
        have up to eleven short labels, which do not all fit side by
        side at "centered" page width, and a rigid st.columns() split
        forces text to overflow its column instead of wrapping.
        """

        stage = self.current()
        parts = [
            f"**{label}**" if index == stage else label
            for index, label in enumerate(self.stage_labels)
        ]

        with st.container(border=True):
            st.markdown(" → ".join(parts))

        return stage

    def render_restart_button(
        self, extra_session_keys: tuple[str, ...] = ()
    ) -> None:
        """
        Show "Restart study" once any stage past the first is unlocked.

        Clears session_key plus extra_session_keys: whatever
        page-specific state (uploaded data, a revealed prediction, a
        fitted model) would otherwise leave a restarted journey looking
        like it remembers the previous run.
        """

        if self.current() == 0:
            return

        if st.button(
            "Restart study",
            icon=":material/restart_alt:",
            key=f"{self.session_key}_restart_button",
        ):
            for key in (self.session_key, *extra_session_keys):
                st.session_state.pop(key, None)
            st.rerun()
