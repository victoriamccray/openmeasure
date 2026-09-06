"""
Unit tests for shared/stage_workspace.py

Run with: pytest shared/tests/test_stage_workspace.py -v

Exercised through AppTest against small scripts, the same way
test_journey_stages.py exercises StageTracker, because the behaviour
under test is what happens across reruns: going back, coming back, and
what a page remembers in between.
"""

from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest

from shared.stage_workspace import (
    STATE_COMPLETE,
    STATE_CURRENT,
    STATE_NEEDS_REVIEW,
    STATE_NOT_REACHED,
    Gate,
    Stage,
)

_SCRIPT = """
import streamlit as st
from shared.stage_workspace import Gate, Stage, StageWorkspace

workspace = StageWorkspace(
    session_key="demo",
    stages=(
        Stage("question", "Question"),
        Stage("measures", "Measures"),
        Stage("timing", "Timing"),
    ),
    gates={"timing": Gate(
        satisfied=bool(st.session_state.get("picked")),
        requirement="Select at least one measure to continue",
    )},
)

stage = workspace.render_rail()
workspace.render_review_notice()

if stage == 0:
    st.write("QUESTION STAGE")
elif stage == 1:
    st.write("MEASURES STAGE")
    if st.button("Pick a measure"):
        st.session_state["picked"] = ["one"]
        st.rerun()
    if st.button("Pick another"):
        st.session_state["picked"] = ["one", "two"]
        st.rerun()
    workspace.record_input(
        "measures",
        st.session_state.get("picked", []),
        affects=("timing",),
        label="Measures",
    )
else:
    st.write("TIMING STAGE")

workspace.render_navigation()
st.write(f"STATES {[workspace.state_of(i) for i in range(3)]}")
"""


_SPECIFIC = """
import streamlit as st
from shared.stage_workspace import Stage, StageWorkspace

workspace = StageWorkspace(
    session_key="dep",
    stages=(
        Stage("question", "Question"),
        Stage("measures", "Measures"),
        Stage("timing", "Timing"),
        Stage("record", "Record"),
    ),
)

stage = workspace.render_rail()
workspace.render_review_notice()

if stage == 0:
    if st.button("Reword the question"):
        st.session_state["wording"] = "different"
        st.rerun()
    # Wording feeds the record and nothing else.
    workspace.record_input(
        "wording",
        st.session_state.get("wording", "original"),
        affects=("record",),
        label="Research question",
    )
elif stage == 1:
    if st.button("Drop a measure"):
        st.session_state["chosen"] = ["one"]
        st.rerun()
    workspace.record_input(
        "measures",
        st.session_state.get("chosen", ["one", "two"]),
        affects=("timing", "record"),
        label="Measures",
    )

workspace.render_navigation()
st.write(f"STATES {[workspace.state_of(i) for i in range(4)]}")
st.write(f"CAUSES {[list(workspace.review_causes(i)) for i in range(4)]}")
"""


def _app() -> AppTest:
    app = AppTest.from_string(_SCRIPT)
    app.run()
    return app


def _click(app: AppTest, label_starts: str) -> AppTest:
    for button in app.button:
        if str(button.label).startswith(label_starts):
            button.click()
            app.run()
            return app
    raise AssertionError(f"no button starting {label_starts!r}")


def _states(app: AppTest) -> list[str]:
    line = next(
        str(item.value) for item in app.markdown
        if str(item.value).startswith("STATES")
    )
    return eval(line[len("STATES "):])


class TestMovingForward(unittest.TestCase):
    def test_only_the_current_stage_renders(self):
        app = _app()
        rendered = " ".join(str(item.value) for item in app.markdown)

        self.assertIn("QUESTION STAGE", rendered)
        self.assertNotIn("MEASURES STAGE", rendered)

    def test_continue_moves_one_stage(self):
        app = _click(_app(), "Continue to Measures")
        rendered = " ".join(str(item.value) for item in app.markdown)

        self.assertIn("MEASURES STAGE", rendered)
        self.assertNotIn("QUESTION STAGE", rendered)


class TestGating(unittest.TestCase):
    """
    A blocked stage says what would unblock it, rather than being a
    control that silently does nothing.
    """

    def test_an_unsatisfied_gate_blocks_and_explains(self):
        app = _click(_app(), "Continue to Measures")
        captions = " ".join(str(item.value) for item in app.caption)

        self.assertIn("Select at least one measure", captions)

    def test_satisfying_the_gate_unblocks_it(self):
        app = _click(_click(_app(), "Continue to Measures"), "Pick a measure")
        app = _click(app, "Continue to Timing")

        self.assertIn(
            "TIMING STAGE", " ".join(str(i.value) for i in app.markdown)
        )


class TestGoingBackward(unittest.TestCase):
    """
    Backward is a first-class action, and what was chosen survives it.
    """

    def test_the_rail_moves_to_a_reached_stage(self):
        app = _click(_click(_app(), "Continue to Measures"), "Pick a measure")
        app = _click(app, "Continue to Timing")
        app = _click(app, "● Question")

        self.assertIn(
            "QUESTION STAGE", " ".join(str(i.value) for i in app.markdown)
        )

    def test_going_back_does_not_lose_the_furthest_point(self):
        app = _click(_click(_app(), "Continue to Measures"), "Pick a measure")
        app = _click(app, "Continue to Timing")
        app = _click(app, "● Question")

        self.assertEqual(_states(app)[2], STATE_COMPLETE)

    def test_a_selection_survives_leaving_and_returning(self):
        app = _click(_click(_app(), "Continue to Measures"), "Pick a measure")
        app = _click(app, "Continue to Timing")
        app = _click(app, "● Measures")

        self.assertEqual(app.session_state["picked"], ["one"])


class TestNeedsReview(unittest.TestCase):
    """
    A stage that was complete before an upstream change is not still
    complete, and is not unreached either.
    """

    def _changed_upstream(self) -> AppTest:
        app = _click(_click(_app(), "Continue to Measures"), "Pick a measure")
        app = _click(app, "Continue to Timing")
        app = _click(app, "● Measures")
        return _click(app, "Pick another")

    def test_a_downstream_stage_is_flagged(self):
        self.assertEqual(_states(self._changed_upstream())[2], STATE_NEEDS_REVIEW)

    def test_the_change_does_not_erase_the_downstream_selection(self):
        """
        Erasing would be a decision about the reader's work. Flagging
        leaves it to them.
        """
        app = self._changed_upstream()

        self.assertEqual(app.session_state["demo_furthest"], 2)

    def test_the_flag_can_be_settled(self):
        app = self._changed_upstream()
        app = _click(app, "△ Timing")
        app = _click(app, "These still apply")

        self.assertEqual(_states(app)[2], STATE_CURRENT)

    def test_an_unvisited_stage_is_not_flagged(self):
        """
        Only stages already reached can need review; the rest are simply
        not reached.
        """
        app = _click(_click(_app(), "Continue to Measures"), "Pick a measure")
        app = _click(app, "Pick another")

        self.assertEqual(_states(app)[2], STATE_NOT_REACHED)


class TestTheDependencyGraphIsSpecific(unittest.TestCase):
    """
    Flagging everything downstream is the wrong answer. Rewording a
    research question does not invalidate a timing decision, and a page
    that said it did would train a reader to dismiss the flag.
    """

    def _walked(self) -> AppTest:
        app = AppTest.from_string(_SPECIFIC)
        app.run()
        for _ in range(3):
            app = _click(app, "Continue to")
        return app

    def _causes(self, app: AppTest) -> list:
        line = next(
            str(item.value) for item in app.markdown
            if str(item.value).startswith("CAUSES")
        )
        return eval(line[len("CAUSES "):])

    def _states_of(self, app: AppTest) -> list:
        line = next(
            str(item.value) for item in app.markdown
            if str(item.value).startswith("STATES")
        )
        return eval(line[len("STATES "):])

    def test_rewording_the_question_leaves_timing_alone(self):
        app = self._walked()
        app = _click(app, "● Question")
        app = _click(app, "Reword the question")
        states = self._states_of(app)

        self.assertEqual(states[2], STATE_COMPLETE)
        self.assertEqual(states[3], STATE_NEEDS_REVIEW)

    def test_dropping_a_measure_reaches_both_stages_it_feeds(self):
        app = self._walked()
        app = _click(app, "● Measures")
        app = _click(app, "Drop a measure")
        states = self._states_of(app)

        self.assertEqual(states[2], STATE_NEEDS_REVIEW)
        self.assertEqual(states[3], STATE_NEEDS_REVIEW)

    def test_the_flag_names_what_changed(self):
        """
        "Something changed" is not enough to act on once a study has
        several inputs.
        """
        app = self._walked()
        app = _click(app, "● Measures")
        app = _click(app, "Drop a measure")

        self.assertEqual(self._causes(app)[2], ["Measures"])

    def test_a_different_input_names_itself(self):
        app = self._walked()
        app = _click(app, "● Question")
        app = _click(app, "Reword the question")

        self.assertEqual(self._causes(app)[3], ["Research question"])

    def test_an_unknown_stage_key_raises(self):
        from shared.stage_workspace import StageWorkspace

        workspace = StageWorkspace(
            session_key="x",
            stages=(Stage("a", "A"), Stage("b", "B")),
        )

        with self.assertRaises(ValueError) as raised:
            workspace._index_of("nowhere")

        self.assertIn("not a stage in this workspace", str(raised.exception))


class TestValidation(unittest.TestCase):
    def test_a_gate_without_a_requirement_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            Gate(satisfied=False, requirement="  ")

        self.assertIn("what would unblock it", str(raised.exception))

    def test_an_optional_gate_does_not_block(self):
        gate = Gate(satisfied=False, requirement="Setting", optional=True)

        self.assertFalse(gate.blocks)

    def test_a_required_gate_blocks(self):
        self.assertTrue(Gate(satisfied=False, requirement="Pick one").blocks)

    def test_a_stage_without_a_label_is_rejected(self):
        with self.assertRaises(ValueError):
            Stage("key", "  ")


if __name__ == "__main__":
    unittest.main()
