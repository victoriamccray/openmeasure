"""
Unit tests for shared/stage_workspace.py

Run with: pytest shared/tests/test_stage_workspace.py -v

Exercised through AppTest against small scripts, the same way
test_journey_stages.py exercises StageTracker, because the behaviour
under test is what happens across reruns: going back, coming back, and
what a page remembers in between.

Synthetic pages rather than real ones on purpose. This is where the state
model in docs/stage-lifecycle.md is established, and a real page's
data_editor and format_func selectboxes cannot be driven through AppTest,
which writes the formatted label where the page expects the key. The
per-page suites read their dependency declarations out of the source with
ast; that checks configuration, and this checks behaviour.

TestTheLifecycle below covers the six behaviours the specification names,
one class per behaviour, because the first three migrations each found a
rule the previous one had not needed.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]

from shared.stage_workspace import (
    STAGE_UNAVAILABLE,
    STATE_COMPLETE,
    STATE_CURRENT,
    STATE_NEEDS_REVIEW,
    STATE_NOT_REACHED,
    STATE_REVIEWED,
    STATE_UNAVAILABLE,
    Gate,
    Stage,
    StageWorkspace,
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


_VANISHING = """
import streamlit as st
from shared.stage_workspace import Gate, Stage, StageWorkspace, STAGE_UNAVAILABLE

workspace = StageWorkspace(
    session_key="van",
    stages=(
        Stage("data", "Data"),
        Stage("analyze", "Analyze"),
        Stage("interpret", "Interpret"),
    ),
    gates={
        "interpret": Gate(
            satisfied=st.session_state.get("result") is not None,
            requirement="Run an analysis to continue",
        ),
    },
)

stage = workspace.render_rail()
workspace.render_review_notice()

if stage == 0:
    if st.button("Load data"):
        st.session_state["loaded"] = True
        st.rerun()
    if st.button("Load other data"):
        # What Impact Evaluation does when a second dataset arrives.
        st.session_state["loaded"] = True
        st.session_state.pop("result", None)
        st.rerun()
    workspace.record_gate_input("loaded", st.session_state.get("loaded", False))
elif stage == 1:
    st.write("ANALYZE STAGE")
    if st.button("Run it"):
        st.session_state["result"] = {"estimate": 0.4}
        st.rerun()
    workspace.record_gate_input("result", st.session_state.get("result"))
elif stage == 2:
    # Would raise on None, the way support_boundary_claims("") did.
    st.write(f"INTERPRET {st.session_state['result']['estimate']}")
elif stage == STAGE_UNAVAILABLE:
    st.write("NOTHING RENDERED")

workspace.render_navigation()
st.write(f"STATES {[workspace.state_of(i) for i in range(3)]}")
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


def _vanishing() -> AppTest:
    app = AppTest.from_string(_VANISHING)
    app.run()
    return app


def _rendered(app: AppTest) -> str:
    return " ".join(str(item.value) for item in app.markdown)


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


class TestAPrerequisiteDisappears(unittest.TestCase):
    """
    Behaviour one. A gate makes a stage unreachable while unsatisfied,
    but furthest is remembered independently, so a stage already reached
    stays on the rail after its prerequisite goes.

    Impact Evaluation found this by calling support_boundary_claims("")
    on a session where a second dataset had discarded the estimate.
    """

    def _reached_then_lost(self) -> AppTest:
        app = _click(_vanishing(), "Load data")
        app = _click(app, "Continue to Analyze")
        app = _click(app, "Run it")
        app = _click(app, "Continue to Interpret")
        # Back to the data stage, and load something else.
        app = _click(app, "← Back to Analyze")
        app = _click(app, "← Back to Data")
        return _click(app, "Load other data")

    def test_the_stage_renders_while_its_prerequisite_holds(self):
        app = _click(_vanishing(), "Load data")
        app = _click(app, "Continue to Analyze")
        app = _click(app, "Run it")
        app = _click(app, "Continue to Interpret")

        self.assertFalse(app.exception)
        self.assertIn("INTERPRET 0.4", _rendered(app))

    def test_a_reached_stage_becomes_unavailable_rather_than_complete(self):
        app = self._reached_then_lost()

        self.assertEqual(_states(app)[2], STATE_UNAVAILABLE)

    def test_it_does_not_become_unreached(self):
        """
        Unreached would erase the history. The reader did get there.
        """
        app = self._reached_then_lost()

        self.assertNotEqual(_states(app)[2], STATE_NOT_REACHED)
        self.assertEqual(app.session_state["van_furthest"], 2)

    def test_no_stage_body_runs_when_the_current_stage_is_unavailable(self):
        """
        The whole point. A page that forgot to re-check gets a blank
        stage with an explanation, not a traceback.
        """
        app = self._reached_then_lost()
        app.session_state["van_current"] = 2
        app.run()

        self.assertFalse(app.exception)
        self.assertIn("NOTHING RENDERED", _rendered(app))
        self.assertNotIn("INTERPRET", _rendered(app))

    def test_it_says_what_would_open_it_again(self):
        app = self._reached_then_lost()
        app.session_state["van_current"] = 2
        app.run()

        warnings = " ".join(str(item.value) for item in app.warning)
        self.assertIn("was reached earlier and cannot open now", warnings)
        self.assertIn("Run an analysis to continue", warnings)

    def test_the_reader_is_not_moved_off_it(self):
        """
        Moving someone off a stage they chose would be a decision about
        their work. Navigation is drawn so they can leave themselves.
        """
        app = self._reached_then_lost()
        app.session_state["van_current"] = 2
        app.run()

        self.assertEqual(app.session_state["van_current"], 2)
        self.assertIsNotNone(app.button(key="van_back"))

    def test_the_prerequisite_returning_restores_the_stage(self):
        app = self._reached_then_lost()
        app = _click(app, "Continue to Analyze")
        app = _click(app, "Run it")
        app.session_state["van_current"] = 2
        app.run()

        self.assertFalse(app.exception)
        self.assertIn("INTERPRET 0.4", _rendered(app))


class TestAResultDisappears(unittest.TestCase):
    """
    Behaviour three, which is behaviour one where the prerequisite is a
    computed value rather than a reader's selection.

    Stated separately because it is the case that reads as guaranteed and
    is not.
    """

    def test_a_gate_on_a_computed_value_is_re_evaluated_every_run(self):
        app = _click(_vanishing(), "Load data")
        app = _click(app, "Continue to Analyze")
        app = _click(app, "Run it")

        self.assertFalse(app.button(key="van_rail_2").disabled)

        app.session_state["result"] = None
        app.run()

        self.assertTrue(app.button(key="van_rail_2").disabled)


class TestARerunHappens(unittest.TestCase):
    """
    Behaviour six. The gates are read above the stage that writes what
    they test, so a value written now was not available to the gate that
    has already been drawn.
    """

    def test_satisfying_a_requirement_does_not_lag_a_pass(self):
        """
        Impact Evaluation hand-rolled this before it moved into the
        workspace. Without it a reader who has just satisfied a
        requirement still sees it unmet.
        """
        app = _click(_vanishing(), "Load data")
        app = _click(app, "Continue to Analyze")
        app = _click(app, "Run it")

        # One pass, no further interaction: the rail already knows.
        self.assertFalse(app.button(key="van_rail_2").disabled)

    def test_the_position_and_the_flags_survive_a_rerun(self):
        app = _click(_vanishing(), "Load data")
        app = _click(app, "Continue to Analyze")
        app.run()

        self.assertEqual(app.session_state["van_current"], 1)
        self.assertEqual(app.session_state["van_furthest"], 1)

    def test_recording_the_same_gate_input_twice_is_a_no_op(self):
        """
        Rerunning on an unchanged value would loop.
        """
        app = _click(_vanishing(), "Load data")
        before = app.session_state["van_gate_loaded"]
        app.run()

        self.assertEqual(app.session_state["van_gate_loaded"], before)
        self.assertFalse(app.exception)


class TestRawDataDoesNotCrossAStage(unittest.TestCase):
    """
    Behaviour five. shared/upload.py promises that nothing retains a
    reader's data, and keep() is where that promise would quietly break.
    """

    def test_keep_refuses_a_dataframe(self):
        import pandas as pd

        script = """
import pandas as pd
import streamlit as st
from shared.stage_workspace import Stage, StageWorkspace

workspace = StageWorkspace(
    session_key="raw",
    stages=(Stage("a", "A"), Stage("b", "B")),
)
try:
    workspace.keep("frame", pd.DataFrame({"x": [1, 2]}))
    st.write("KEPT IT")
except TypeError as error:
    st.write(f"REFUSED {error}")
"""
        app = AppTest.from_string(script)
        app.run()

        rendered = _rendered(app)
        self.assertIn("REFUSED", rendered)
        self.assertIn("Raw data does not cross a stage boundary", rendered)
        self.assertNotIn("KEPT IT", rendered)

    def test_a_derived_result_is_allowed_through(self):
        script = """
import streamlit as st
from shared.stage_workspace import Stage, StageWorkspace
from dataclasses import dataclass

@dataclass(frozen=True)
class Estimate:
    difference: float

workspace = StageWorkspace(
    session_key="derived",
    stages=(Stage("a", "A"), Stage("b", "B")),
)
workspace.keep("estimate", Estimate(0.4))
st.write(f"KEPT {workspace.kept('estimate').difference}")
"""
        app = AppTest.from_string(script)
        app.run()

        self.assertFalse(app.exception)
        self.assertIn("KEPT 0.4", _rendered(app))


class TestReviewedIsNotComplete(unittest.TestCase):
    """
    Complete means nothing has questioned this stage. Reviewed means
    something did and a person vouched for it anyway, which is a stronger
    statement and a different provenance.
    """

    def _settled(self) -> AppTest:
        app = _click(_click(_app(), "Continue to Measures"), "Pick a measure")
        app = _click(app, "Continue to Timing")
        app = _click(app, "● Measures")
        app = _click(app, "Pick another")
        app = _click(app, "△ Timing")
        return _click(app, "These still apply")

    def test_a_settled_stage_reads_as_reviewed_once_it_is_left(self):
        app = _click(self._settled(), "← Back to Measures")

        self.assertEqual(_states(app)[2], STATE_REVIEWED)

    def test_it_is_distinguishable_from_a_stage_nothing_questioned(self):
        app = _click(self._settled(), "← Back to Measures")
        states = _states(app)

        self.assertEqual(states[0], STATE_COMPLETE)
        self.assertEqual(states[2], STATE_REVIEWED)

    def test_another_change_flags_it_again(self):
        """
        A stage vouched for under the previous conditions is not vouched
        for under these.
        """
        app = _click(self._settled(), "● Measures")
        app = _click(app, "Pick a measure")

        self.assertEqual(_states(app)[2], STATE_NEEDS_REVIEW)


class TestNoPageWritesAPresenceGate(unittest.TestCase):
    """
    A structural guard, not a behaviour test.

    `satisfied="x" in st.session_state` is satisfied by a None under that
    name, so the stage it guards runs on nothing and fails inside its own
    core call rather than being told it has nothing to work from.
    Portfolio Impact Analysis shipped six of these for about an hour and
    the first test written against them found the hole.

    Cheap to check across every page, and it fails when the weaker form
    comes back rather than when it next causes a crash.
    """

    @staticmethod
    def _gate_expressions() -> dict:
        """Every satisfied= expression in every page: file -> [source]."""
        import ast

        found = {}

        for page in sorted((ROOT / "pages").glob("*.py")):
            source = page.read_text(encoding="utf-8")

            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.Call):
                    continue

                name = node.func
                if not (isinstance(name, ast.Name) and name.id == "Gate"):
                    continue

                for keyword in node.keywords:
                    if keyword.arg == "satisfied":
                        found.setdefault(page.name, []).append(
                            ast.unparse(keyword.value)
                        )

        return found

    def test_some_pages_declare_gates(self):
        """So the check below cannot pass by finding nothing."""
        self.assertGreaterEqual(sum(
            len(v) for v in self._gate_expressions().values()
        ), 8)

    def test_no_gate_tests_key_presence(self):
        for page, expressions in self._gate_expressions().items():
            for expression in expressions:
                with self.subTest(page=page, gate=expression):
                    self.assertNotIn("in st.session_state", expression)


class TestTheOpeningStageCannotBeGatedShut(unittest.TestCase):
    def test_a_required_gate_on_the_first_stage_is_rejected(self):
        """
        It would block the one screen able to satisfy it.
        """
        with self.assertRaises(ValueError) as raised:
            StageWorkspace(
                session_key="deadlock",
                stages=(Stage("a", "A"), Stage("b", "B")),
                gates={"a": Gate(satisfied=False, requirement="Impossible")},
            )

        self.assertIn("opening stage", str(raised.exception))

    def test_an_optional_gate_on_the_first_stage_is_fine(self):
        """
        Which is how a page names a gap without demanding it. Method
        Selection's population gate is exactly this.
        """
        workspace = StageWorkspace(
            session_key="ok",
            stages=(Stage("a", "A"), Stage("b", "B")),
            gates={
                "a": Gate(
                    satisfied=False, requirement="Population", optional=True
                )
            },
        )

        self.assertEqual(workspace.blocked_reason(1), "")

    def test_a_gate_naming_an_absent_stage_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            StageWorkspace(
                session_key="typo",
                stages=(Stage("a", "A"), Stage("b", "B")),
                gates={"c": Gate(satisfied=True, requirement="Something")},
            )

        self.assertIn("does not have", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
