"""
Cumulative reveal for a gated sequence inside one stage.

A session_state counter for the highest unlocked step, a
rerun-on-unlock helper, and a breadcrumb. Before this module existed all
six Research Journeys reimplemented those three pieces independently, in
six near-identical and silently divergent copies: pyfMRIqc's fourth
stage was checked for but never unlocked by any call, leaving it
permanently unreachable.

Its scope is now much narrower than that. Every journey moved to
shared/stage_workspace.py, which puts one stage on screen at a time with
a rail to move between them, because a cumulative reveal makes a
research process read as a report and going back to change a decision a
matter of scrolling upward and hoping.

What is left here is the case the workspace is not for. Multimodal
Signal Convergence's tradeoff stage unlocks its cost dimensions one at a
time, gain alone and then privacy, security, and agency. That is
progressive disclosure within a stage rather than a stage of a research
process, and the distinction is the rule the workspace states:

    Stages move horizontally through the research process. Scrolling
    moves vertically through the evidence inside the current stage.

A new page wanting stages wants StageWorkspace. A page wanting to reveal
evidence inside one stage, in an order that matters, wants this.

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
