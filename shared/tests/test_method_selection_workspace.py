"""
Method Selection's state model, which is what this migration tests.

Run with: pytest shared/tests/test_method_selection_workspace.py -v

Right To Play proved the navigation. This page proves the state: going
back to change a measure has to flag what that measure feeds, leave what
it does not feed alone, preserve every selection, and delete nothing.

Driven by setting the workspace's position rather than by clicking
through. go_to() reruns mid-script and AppTest accumulates the widgets
from both passes, so clicking lands on stale instances; the navigation
itself is covered in test_stage_workspace.py.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from shared.stage_workspace import (
    STATE_COMPLETE,
    STATE_NEEDS_REVIEW,
)

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = str(ROOT / "Home.py")
LOAD_TIMEOUT_SECONDS = 90

QUESTION, MEASURES, TIMING, SIMULATE, RECORD = range(5)


def _design_mode(current: int = 0, furthest: int | None = None) -> AppTest:
    app = AppTest.from_file(ENTRYPOINT, default_timeout=LOAD_TIMEOUT_SECONDS)
    app.run()
    app.switch_page("pages/Method_Selection.py")
    app.run()
    app.radio[0].set_value("design")
    app.run()
    app.session_state["design_current"] = current
    app.session_state["design_furthest"] = (
        current if furthest is None else furthest
    )
    app.run()
    return app


def _states(app: AppTest) -> list[str]:
    from shared.stage_workspace import (
        STATE_CURRENT,
        STATE_NOT_REACHED,
    )

    marks = {
        "●": STATE_COMPLETE,
        "◉": STATE_CURRENT,
        "○": STATE_NOT_REACHED,
        "△": STATE_NEEDS_REVIEW,
    }

    return [
        marks[str(app.button(key=f"design_rail_{index}").label)[0]]
        for index in range(5)
    ]


class TestOneStageAtATime(unittest.TestCase):
    def test_the_design_mode_opens_on_its_question(self):
        app = _design_mode(QUESTION)

        self.assertFalse(app.exception)
        headers = " ".join(str(h.value) for h in app.subheader)
        self.assertIn("Research Question", headers)

    def test_the_measures_stage_does_not_also_render_the_question(self):
        app = _design_mode(MEASURES)
        headers = " ".join(str(h.value) for h in app.subheader)

        self.assertIn("Concepts And How To Observe Them", headers)
        self.assertNotIn("Research Question", headers)

    def test_every_stage_renders_without_error(self):
        for index in range(5):
            with self.subTest(stage=index):
                self.assertFalse(_design_mode(index, RECORD).exception)


class TestGating(unittest.TestCase):
    """
    Required to proceed against optional information. A page that treated
    them the same would make a researcher invent a population to reach
    the next screen.
    """

    def test_timing_is_blocked_until_a_measure_is_assembled(self):
        app = _design_mode(QUESTION)

        self.assertTrue(app.button(key="design_rail_2").disabled)

    def test_the_block_says_what_would_lift_it(self):
        app = _design_mode(QUESTION)
        captions = " ".join(str(item.value) for item in app.caption)

        self.assertIn("Assemble at least one measure", captions)

    def test_measures_is_never_blocked(self):
        """
        Looking at what could observe a concept is not something that has
        to be unlocked.
        """
        app = _design_mode(QUESTION)

        self.assertFalse(app.button(key="design_rail_1").disabled)

    def test_the_population_gate_is_declared_optional(self):
        """
        A page that treated an unstated population like an unassembled
        measure would make a researcher invent one to reach the next
        screen.
        """
        import re

        source = re.sub(
            r"\s+",
            " ",
            (ROOT / "pages" / "Method_Selection.py").read_text(encoding="utf-8"),
        )

        self.assertIn(
            'requirement="Population not stated", optional=True', source
        )


class TestTheDependencyWiring(unittest.TestCase):
    """
    Which inputs the page declares, and what each one feeds.

    Checked against the source rather than driven through the page. The
    planner rebuilds its state from a data editor and one multiselect per
    concept every run, and AppTest cannot set those before the widgets
    instantiate, so a click-through here would test the harness rather
    than the wiring.

    What the flags then do is covered directly in
    shared/tests/test_stage_workspace.py, which exercises record_input,
    the specific affects list, the named cause, and the guarantee that
    nothing downstream is erased.
    """

    @staticmethod
    def _recorded_inputs() -> dict:
        """Every record_input call on the page: name -> (affects, label)."""
        import ast

        source = (ROOT / "pages" / "Method_Selection.py").read_text(
            encoding="utf-8"
        )
        found = {}

        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute) and func.attr == "record_input"
            ):
                continue

            name = node.args[0].value
            keywords = {kw.arg: kw.value for kw in node.keywords}
            affects = tuple(
                element.value for element in keywords["affects"].elts
            )
            found[name] = (affects, keywords["label"].value)

        return found

    def test_measures_feed_timing_the_simulation_and_the_record(self):
        affects, label = self._recorded_inputs()["measures"]

        self.assertEqual(set(affects), {"timing", "simulate", "record"})
        self.assertEqual(label, "Measures")

    def test_the_research_question_feeds_the_record_alone(self):
        """
        Rewording a question does not invalidate a timing decision, and a
        page that said it did would train a reader to dismiss the flag.
        """
        affects, label = self._recorded_inputs()["question"]

        self.assertEqual(set(affects), {"record"})
        self.assertEqual(label, "Research question")

    def test_concepts_feed_the_record_alone(self):
        affects, _ = self._recorded_inputs()["concepts"]

        self.assertEqual(set(affects), {"record"})

    def test_timing_feeds_the_simulation_and_the_record(self):
        affects, label = self._recorded_inputs()["timing"]

        self.assertEqual(set(affects), {"simulate", "record"})
        self.assertEqual(label, "Timing")

    def test_every_declared_target_is_a_real_stage(self):
        stages = {"question", "measures", "timing", "simulate", "record"}

        for name, (affects, _) in self._recorded_inputs().items():
            for target in affects:
                with self.subTest(input=name, target=target):
                    self.assertIn(target, stages)

    def test_no_input_flags_a_stage_that_precedes_it(self):
        """
        An input cannot invalidate something decided before it.
        """
        order = ["question", "measures", "timing", "simulate", "record"]
        produced_in = {
            "question": "question",
            "concepts": "measures",
            "measures": "measures",
            "timing": "timing",
        }

        for name, (affects, _) in self._recorded_inputs().items():
            source_stage = order.index(produced_in[name])
            for target in affects:
                with self.subTest(input=name, target=target):
                    self.assertGreater(order.index(target), source_stage)


class TestValuesSurviveLeavingAStage(unittest.TestCase):
    """
    Streamlit drops a widget's value when the widget is not rendered, and
    in a workspace that is every widget in every other stage.
    """

    @staticmethod
    def _source() -> str:
        """The page, whitespace-normalised, since some calls wrap."""
        import re

        collapsed = re.sub(
            r"\s+",
            " ",
            (ROOT / "pages" / "Method_Selection.py").read_text(encoding="utf-8"),
        )

        # A wrapped call leaves a space after the paren once collapsed.
        return collapsed.replace("( ", "(")

    def test_the_page_holds_what_later_stages_need(self):
        source = self._source()

        for held in (
            "n_participants",
            "observations_per_day",
            "duration_days",
            "misalignment",
            "entered",
            "study",
            "estimate",
        ):
            with self.subTest(value=held):
                self.assertIn(f'design_workspace.keep("{held}"', source)

    def test_each_held_value_is_read_back(self):
        source = self._source()

        for held in ("n_participants", "misalignment", "entered", "study"):
            with self.subTest(value=held):
                self.assertIn(f'design_workspace.kept("{held}"', source)


if __name__ == "__main__":
    unittest.main()
