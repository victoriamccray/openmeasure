"""
Unit tests for shared/journey_stages.py

Run with: pytest shared/tests/ -v

advance_to's range check happens before any Streamlit call and is tested
directly (same approach as shared/tests/test_report.py's
TestLifecycleTrackerValidation). The rest of StageTracker's behavior
(current(), render_breadcrumb(), render_restart_button()) is Streamlit UI
code, so it is exercised through AppTest.from_string against a minimal
script that uses StageTracker exactly as a Research Journey page does,
rather than through any one journey's own business logic (which the
journeys' own pages already carry, and which for pyfMRIqc's Signal
Inspection stage depends on a live network fetch this suite must not
make). shared/tests/test_pages_render.py's journey tests only check that
each real journey page loads on its first stage without raising.
"""

from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest

from shared.journey_stages import StageTracker

_SCRIPT = """
import streamlit as st
from shared.journey_stages import StageTracker

TRACKER = StageTracker(
    session_key="test_stage",
    stage_labels=("Start", "Middle", "End"),
)

stage = TRACKER.render_breadcrumb()
TRACKER.render_restart_button(extra_session_keys=("extra_state",))

if stage >= 1:
    st.session_state.setdefault("extra_state", "set-by-test")

if stage < 1:
    if st.button("Continue to middle"):
        TRACKER.advance_to(1)

if stage < 2:
    if st.button("Continue to end"):
        TRACKER.advance_to(2)

st.write(f"stage is {stage}")
"""


class TestAdvanceToValidation(unittest.TestCase):
    def test_negative_stage_is_rejected_before_any_streamlit_call(self):
        tracker = StageTracker(session_key="t_negative", stage_labels=("A", "B"))

        with self.assertRaises(ValueError) as context:
            tracker.advance_to(-1)

        self.assertIn("out of range", str(context.exception))

    def test_stage_past_the_declared_labels_is_rejected(self):
        tracker = StageTracker(session_key="t_overflow", stage_labels=("A", "B"))

        with self.assertRaises(ValueError) as context:
            tracker.advance_to(2)

        self.assertIn("out of range", str(context.exception))
        self.assertIn("t_overflow", str(context.exception))


class TestStageTrackerIsFrozen(unittest.TestCase):
    def test_entries_are_frozen(self):
        self.assertTrue(StageTracker.__dataclass_params__.frozen)


class TestStageTrackerBehavior(unittest.TestCase):
    """
    Runs _SCRIPT, a minimal page built the same way a Research Journey
    page uses StageTracker, through AppTest so the actual Streamlit
    behavior (not just the pre-Streamlit validation above) is exercised:
    the breadcrumb bolds the current stage, a "Continue" button unlocks
    and reruns to the next stage, and restarting clears both the stage
    and any extra session keys named.
    """

    def _run(self) -> AppTest:
        app = AppTest.from_string(_SCRIPT)
        app.run()

        self.assertFalse(app.exception)

        return app

    @staticmethod
    def _rendered(app: AppTest) -> str:
        # st.write of a plain string emits Markdown, same as the note in
        # shared/tests/test_pages_render.py's Method Selection tests.
        return " ".join(str(item.value) for item in app.markdown)

    def test_starts_on_the_first_stage_with_it_bolded(self):
        app = self._run()

        rendered = self._rendered(app)
        self.assertIn("**Start** → Middle → End", rendered)
        self.assertIn("stage is 0", rendered)

    def test_continue_button_unlocks_and_reruns_to_the_next_stage(self):
        app = self._run()

        app.button[0].click()
        app.run()

        self.assertFalse(app.exception)

        rendered = self._rendered(app)
        self.assertIn("Start → **Middle** → End", rendered)
        self.assertIn("stage is 1", rendered)

    def test_stages_stay_unlocked_after_a_later_stage_unlocks(self):
        app = self._run()

        app.button[0].click()
        app.run()
        app.button[1].click()
        app.run()

        self.assertFalse(app.exception)
        self.assertIn("stage is 2", self._rendered(app))

    def test_restart_clears_the_stage_and_extra_session_keys(self):
        app = self._run()

        app.button[0].click()
        app.run()

        self.assertEqual(app.session_state["extra_state"], "set-by-test")

        # The restart button only renders once a later stage is unlocked,
        # so it is app.button[0] again here: the "Continue to middle"
        # button from stage 0 no longer renders once stage 1 is current.
        restart_buttons = [b for b in app.button if b.label == "Restart study"]
        self.assertEqual(len(restart_buttons), 1)
        restart_buttons[0].click()
        app.run()

        self.assertFalse(app.exception)
        self.assertNotIn("test_stage", app.session_state)
        self.assertNotIn("extra_state", app.session_state)
        self.assertIn("stage is 0", self._rendered(app))


if __name__ == "__main__":
    unittest.main()
