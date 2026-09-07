"""
The worked example must not appear in an unrelated study.

Run with: pytest shared/tests/test_planner_isolation.py -v

This is the cold test written down. A researcher entered "increase in
financial security" and the next thing on the page was "How could we
observe pain in everyday life?", followed by pain ratings, a body map
and electrodermal activity. The ontology underneath had already been
rebuilt; the page still showed one study's measures to everyone, so from
the researcher's side nothing had changed.

The invariant, in one line:

    Mentioning the example is allowed. Rendering its research content is
    not.

So the button offering it may name it, and nothing else may show it,
until a researcher asks. Loading it must then bring its concepts and
measures through the same planner path everyone else uses, rather than
switching the page into a second mode.

A second case runs a structurally different study, because a check that
only ever asks about one worked example stops catching leakage as soon
as the wording changes.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = str(ROOT / "Home.py")
LOAD_TIMEOUT_SECONDS = 60


def _rendered_text(app: AppTest) -> str:
    """Everything a reader would see, flattened."""
    parts: list[str] = []

    for collection in (
        app.markdown,
        app.caption,
        app.info,
        app.warning,
        app.error,
        app.text,
        app.subheader,
        app.title,
    ):
        parts.extend(str(item.value) for item in collection)

    parts.extend(str(item.label) for item in app.expander)
    parts.extend(str(item.label) for item in app.multiselect)
    parts.extend(str(item.label) for item in app.radio)

    # Button labels are excluded deliberately. The button offering the
    # worked example names it, which is the offer working rather than the
    # example leaking; what must not appear is its content.
    return " ".join(parts).lower()


def _plan_a_study() -> AppTest:
    app = AppTest.from_file(ENTRYPOINT, default_timeout=LOAD_TIMEOUT_SECONDS)
    app.run()
    app.switch_page("pages/Method_Selection.py")
    app.run()

    for radio in app.radio:
        if any("plan a study" in str(option).lower() for option in radio.options):
            radio.set_value("Plan a Study").run()
            break

    return app


# The planner is a staged workspace: the question is asked on the first
# stage and concepts are named on the second, so a test looking for the
# concepts screen has to be on it. Set by position rather than by
# clicking Continue, because go_to() reruns mid-script and AppTest
# accumulates the widgets from both passes; the navigation itself is
# covered in shared/tests/test_stage_workspace.py.
STAGE_MEASURES = 1


def _at_concepts(app: AppTest) -> AppTest:
    """The same session, moved to the stage that names concepts."""
    app.session_state["design_current"] = STAGE_MEASURES
    app.session_state["design_furthest"] = STAGE_MEASURES
    app.run()

    return app


class TestAnUnrelatedStudyStaysUnrelated(unittest.TestCase):
    def test_the_worked_example_is_absent_until_it_is_loaded(self):
        app = _plan_a_study()

        self.assertFalse(app.exception)
        self.assertNotIn("pain", _rendered_text(app))

    def test_none_of_the_examples_measures_are_offered(self):
        """
        The specific things a researcher asking about financial security
        was shown.
        """
        rendered = _rendered_text(_at_concepts(_plan_a_study()))

        for leaked in ("body map", "electrodermal", "heart-rate variability"):
            with self.subTest(measure=leaked):
                self.assertNotIn(leaked, rendered)

    def test_the_generalized_vocabulary_is_what_is_offered_instead(self):
        """
        Absence alone would also be satisfied by an empty page. The
        planner has to be the thing that is there.
        """
        rendered = _rendered_text(_at_concepts(_plan_a_study()))

        self.assertIn("concepts", rendered)
        self.assertIn("observe", rendered)

    def test_loading_the_example_is_offered_as_a_deliberate_choice(self):
        app = _plan_a_study()
        labels = " ".join(str(item.label).lower() for item in app.button)

        self.assertIn("worked example", labels)


class TestTheSimulationDisclosureSitsWhereItApplies(unittest.TestCase):
    def test_the_generalized_path_does_not_claim_input_is_ignored(self):
        """
        The intro used to say the simulation models one built-in scenario
        "regardless of what you enter below", at the top of the path a
        researcher plans their own study in.
        """
        app = _plan_a_study()

        self.assertNotIn("regardless of what you enter", _rendered_text(app))


class TestLoadingTheExampleUsesTheSamePath(unittest.TestCase):
    """
    Third invariant. The example is one instance of the planner, not a
    mode the page switches into.
    """

    def _loaded(self) -> AppTest:
        app = _plan_a_study()

        for button in app.button:
            if str(button.label).startswith("Load worked example"):
                button.click()
                app.run()
                break

        return app

    def test_its_concepts_appear_after_loading(self):
        app = self._loaded()

        self.assertFalse(app.exception)
        self.assertIn("pain", _rendered_text(app))

    def test_its_measures_arrive_through_the_shared_library(self):
        """
        The measures it selects are library entries, so they carry the
        same fields every other measure does rather than the example's
        own private descriptions.
        """
        from modules.research_design.core import examples, ontology

        for name in examples.CHRONIC_PAIN.study.selected_measures:
            with self.subTest(measure=name):
                measure = ontology.get_measure(name)
                self.assertTrue(measure.limitation)
                self.assertIn(measure.modality, ontology.MODALITIES)


class TestAStructurallyDifferentStudy(unittest.TestCase):
    """
    A knowledge-structure study, which shares no measure with the pain
    example. Run against the library rather than the page, because what
    is being checked is that the join is the concept's kind: if that ever
    reverts to being the field, this fails whatever the page looks like.
    """

    def test_a_knowledge_structure_concept_gets_elicitation_measures(self):
        from modules.research_design.core import ontology

        names = {
            measure.name
            for measure in ontology.measures_for(ontology.KIND_KNOWLEDGE_STRUCTURE)
        }

        self.assertIn("Card sorting", names)
        self.assertIn("Causal or cognitive mapping", names)

    def test_and_none_of_the_pain_examples_measures(self):
        from modules.research_design.core import examples, ontology

        names = {
            measure.name
            for measure in ontology.measures_for(ontology.KIND_KNOWLEDGE_STRUCTURE)
        }

        for leaked in examples.CHRONIC_PAIN.study.selected_measures:
            with self.subTest(measure=leaked):
                self.assertNotIn(leaked, names)


class TestTheQuestionDrivesWhatIsOffered(unittest.TestCase):
    """
    The acceptance test in one sentence: enter a question that is not
    about pain, load no example, and get plausible, question-relevant
    measures to assemble.

    Driven through the page rather than the library, because the library
    was already right when the page was still showing everyone one
    study's measures.
    """

    def _with_question(self, question: str) -> AppTest:
        """
        A session that has been asked the question, then moved on.

        The question is entered on the first stage and read on the
        second, which is the whole point of the workspace holding it:
        Streamlit drops a widget's value when the widget is not
        rendered, so a question that did not survive leaving its stage
        would be gone before anything could recognise it.
        """
        app = _plan_a_study()

        for area in app.text_area:
            if "question" in str(area.label).lower():
                area.set_value(question).run()
                break

        return _at_concepts(app)

    def test_a_financial_security_question_recognises_its_concept(self):
        app = self._with_question(
            "Does the program increase financial security among adults?"
        )
        rendered = _rendered_text(app)

        self.assertFalse(app.exception)
        self.assertIn("financial security", rendered)
        self.assertIn("economic or material state", rendered)

    def test_it_offers_the_concept_rather_than_adding_it(self):
        """
        The rule the lexicon exists to keep. A recognised concept is
        offered as a button; nothing enters the study until it is
        clicked.
        """
        app = self._with_question(
            "Does the program increase financial security among adults?"
        )
        labels = " ".join(str(item.label).lower() for item in app.button)

        self.assertIn("add financial security", labels)

    def test_confirming_it_surfaces_measures_that_fit(self):
        app = self._with_question(
            "Does the program increase financial security among adults?"
        )

        for button in app.button:
            if str(button.label).lower().startswith("add financial security"):
                button.click()
                app.run()
                break

        rendered = _rendered_text(app)

        self.assertFalse(app.exception)
        self.assertIn("material hardship indicators", rendered)
        self.assertIn("financial well-being scale", rendered)

    def test_and_still_none_of_the_worked_examples_measures(self):
        app = self._with_question(
            "Does the program increase financial security among adults?"
        )

        for button in app.button:
            if str(button.label).lower().startswith("add financial security"):
                button.click()
                app.run()
                break

        rendered = _rendered_text(app)

        for leaked in ("body map", "electrodermal", "heart-rate variability"):
            with self.subTest(measure=leaked):
                self.assertNotIn(leaked, rendered)

    def test_an_unrecognised_question_says_so_rather_than_guessing(self):
        app = self._with_question("Something entirely novel about widgets")
        rendered = _rendered_text(app)

        self.assertFalse(app.exception)
        self.assertIn("no concept in openmeasure's list was recognised", rendered)


if __name__ == "__main__":
    unittest.main()
