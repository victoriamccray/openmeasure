"""
Impact Evaluation's state model, which is what this migration tests.

Run with: pytest shared/tests/test_impact_workspace.py -v

The page moved from a cumulative reveal to a workspace, and the two fail
differently. A cumulative page loses nothing when a reader scrolls back,
because nothing was ever taken away; a workspace renders one stage and
Streamlit discards every widget in the others. So what is tested here is
what crosses a stage boundary: the question, the field, the query, and
the estimate. Also what deliberately does not, which is the data.

Driven by setting the workspace's position rather than by clicking
through. go_to() reruns mid-script and AppTest accumulates the widgets
from both passes, so clicking lands on stale instances; the navigation
itself is covered in test_stage_workspace.py.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = str(ROOT / "Home.py")
PAGE = "pages/2_Impact_Evaluation.py"
LOAD_TIMEOUT_SECONDS = 120

QUESTION, DOMAIN, RESEARCH, DESIGNS, EXAMPLE, ANALYZE, INTERPRET = range(7)

STAGE_KEYS = (
    "question",
    "domain",
    "research",
    "designs",
    "example",
    "analyze",
    "interpret",
)


def _open() -> AppTest:
    app = AppTest.from_file(ENTRYPOINT, default_timeout=LOAD_TIMEOUT_SECONDS)
    app.run()
    app.switch_page(PAGE).run()

    return app


def _answered() -> AppTest:
    """A session whose evaluation question names a program and an outcome."""
    app = _open()
    app.text_input(key="pe_q_program").set_value("text-message reminders")
    app.text_input(key="pe_q_outcome").set_value("appointments kept").run()

    return app


def _at(app: AppTest, stage: int, *, furthest: int | None = None) -> AppTest:
    app.session_state["pe_current"] = stage
    app.session_state["pe_furthest"] = stage if furthest is None else furthest
    app.run()

    return app


def _headings(app: AppTest) -> str:
    return " ".join(str(item.value) for item in app.subheader)


class TestOneStageAtATime(unittest.TestCase):
    def test_the_page_opens_on_its_question(self):
        app = _open()

        self.assertFalse(app.exception)
        self.assertIn("Evaluation Question", _headings(app))

    def test_a_later_stage_does_not_also_render_the_earlier_ones(self):
        """
        The reason the migration needed state at all. Under the old
        cumulative reveal the question was still on screen here.
        """
        app = _at(_answered(), DESIGNS, furthest=ANALYZE)
        headings = _headings(app)

        self.assertIn("Explore Designs", headings)
        self.assertNotIn("Evaluation Question", headings)
        self.assertNotIn("Find Research", headings)

    def test_every_stage_renders_without_error(self):
        for stage in range(7):
            with self.subTest(stage=stage):
                app = _at(_answered(), stage, furthest=INTERPRET)
                self.assertFalse(app.exception)


class TestGating(unittest.TestCase):
    def test_nothing_past_the_question_opens_without_one(self):
        app = _open()

        for stage in range(1, 7):
            with self.subTest(stage=stage):
                self.assertTrue(app.button(key=f"pe_rail_{stage}").disabled)

    def test_the_block_says_what_would_lift_it(self):
        app = _open()
        captions = " ".join(str(item.value) for item in app.caption)

        self.assertIn("Name the program and the outcome", captions)

    def test_answering_the_question_opens_everything_but_the_interpretation(self):
        app = _answered()

        for stage in range(DOMAIN, INTERPRET):
            with self.subTest(stage=stage):
                self.assertFalse(app.button(key=f"pe_rail_{stage}").disabled)

        self.assertTrue(app.button(key=f"pe_rail_{INTERPRET}").disabled)

    def test_the_interpretation_is_blocked_until_something_is_run(self):
        app = _at(_answered(), ANALYZE, furthest=ANALYZE)
        forward = app.button(key="pe_forward")

        self.assertTrue(forward.disabled)

        captions = " ".join(str(item.value) for item in app.caption)
        self.assertIn("Run an analysis to continue", captions)

    def test_the_research_stage_carries_no_gate_of_its_own(self):
        """
        Literature discovery enriches the workflow and never gates it, so
        the data stage is reachable without a search ever running.
        """
        app = _answered()

        self.assertFalse(app.button(key=f"pe_rail_{ANALYZE}").disabled)
        self.assertNotIn("pe_search_results", app.session_state)

    def test_the_comparison_gap_is_declared_optional(self):
        """
        A comparison sharpens the design stage and a page that required
        it would make someone invent one to see what the designs are.
        """
        import re

        source = re.sub(
            r"\s+", " ", (ROOT / PAGE).read_text(encoding="utf-8")
        )

        self.assertIn(
            'requirement="Comparison not named", optional=True', source
        )

    def test_the_optional_gap_is_named_rather_than_enforced(self):
        app = _answered()
        labels = [str(item.label) for item in app.expander]

        self.assertTrue(
            any("Optional, not yet given" in label for label in labels),
            labels,
        )


class TestSelectionsSurviveLeavingAStage(unittest.TestCase):
    """
    Streamlit drops a widget's value when the widget is not rendered, and
    in a workspace that is every widget in every other stage.
    """

    def test_the_question_is_still_there_on_returning_to_it(self):
        app = _at(_answered(), ANALYZE, furthest=ANALYZE)
        app = _at(app, QUESTION, furthest=ANALYZE)

        self.assertEqual(
            app.text_input(key="pe_q_program").value, "text-message reminders"
        )
        self.assertEqual(
            app.text_input(key="pe_q_outcome").value, "appointments kept"
        )

    def test_the_field_of_practice_is_still_there(self):
        app = _at(_answered(), DOMAIN)
        app.session_state["pe_domain"] = "education"
        app.run()

        app = _at(app, DESIGNS, furthest=DESIGNS)
        app = _at(app, DOMAIN, furthest=DESIGNS)

        self.assertEqual(app.selectbox(key="pe_domain").value, "education")

    def test_a_later_stage_reads_the_field_a_previous_one_chose(self):
        """
        The field decides the words on the design cards, and those are
        drawn on a screen where its selector is not present.
        """
        app = _at(_answered(), DOMAIN)
        app.session_state["pe_domain"] = "education"
        app.run()

        app = _at(app, DESIGNS, furthest=DESIGNS)
        held = app.session_state.filtered_state

        self.assertFalse(app.exception)
        self.assertEqual(held.get("pe_kept_domain_id"), "education")


class TestTheDependencyWiring(unittest.TestCase):
    """
    Which inputs the page declares, and what each one feeds.

    Read from the source rather than driven through the page, because
    AppTest cannot set a selectbox whose options are formatted: it writes
    the label where the page expects the key, and the widget falls back to
    its default. What the flags then do is covered in
    shared/tests/test_stage_workspace.py.
    """

    @staticmethod
    def _recorded_inputs() -> dict:
        """Every record_input call: name -> (affects, label)."""
        source = (ROOT / PAGE).read_text(encoding="utf-8")
        found = {}

        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue

            function = node.func
            if not (
                isinstance(function, ast.Attribute)
                and function.attr == "record_input"
            ):
                continue

            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            found[node.args[0].value] = (
                tuple(element.value for element in keywords["affects"].elts),
                keywords["label"].value,
            )

        return found

    def test_the_question_feeds_the_search_and_nothing_else(self):
        """
        Rewording a question does not invalidate a design comparison, and
        a page that said it did would train a reader to dismiss the flag.
        """
        affects, label = self._recorded_inputs()["question"]

        self.assertEqual(set(affects), {"research"})
        self.assertEqual(label, "Evaluation question")

    def test_the_field_feeds_everything_it_words(self):
        """
        Search terms, the design cards' vocabulary, and which telling of
        the worked example appears. Not the analysis: the field changes no
        statistic.
        """
        affects, label = self._recorded_inputs()["domain"]

        self.assertEqual(set(affects), {"research", "designs", "example"})
        self.assertEqual(label, "Field of practice")

    def test_the_analysis_plan_feeds_the_interpretation(self):
        affects, label = self._recorded_inputs()["analysis"]

        self.assertEqual(set(affects), {"interpret"})
        self.assertEqual(label, "Analysis plan")

    def test_every_declared_target_is_a_real_stage(self):
        for name, (affects, _) in self._recorded_inputs().items():
            for target in affects:
                with self.subTest(input=name, target=target):
                    self.assertIn(target, STAGE_KEYS)

    def test_no_input_flags_a_stage_that_precedes_it(self):
        """An input cannot invalidate something decided before it."""
        produced_in = {
            "question": "question",
            "domain": "domain",
            "analysis": "analyze",
        }

        for name, (affects, _) in self._recorded_inputs().items():
            source_stage = STAGE_KEYS.index(produced_in[name])
            for target in affects:
                with self.subTest(input=name, target=target):
                    self.assertGreater(STAGE_KEYS.index(target), source_stage)


class TestChangingAnInputFlagsWhatItFeeds(unittest.TestCase):
    def _after_changing_the_field(self) -> AppTest:
        app = _at(_answered(), DOMAIN, furthest=ANALYZE)
        app.session_state["pe_domain"] = "education"
        app.run()

        return app

    def test_the_stages_it_words_are_flagged(self):
        review = self._after_changing_the_field().session_state.filtered_state[
            "pe_review"
        ]

        self.assertEqual(set(review), {RESEARCH, DESIGNS, EXAMPLE})

    def test_the_flag_says_what_changed(self):
        review = self._after_changing_the_field().session_state.filtered_state[
            "pe_review"
        ]

        self.assertEqual(review[DESIGNS], ["Field of practice"])

    def test_the_flagged_stage_says_so_when_opened(self):
        app = _at(self._after_changing_the_field(), DESIGNS, furthest=ANALYZE)
        warnings = " ".join(str(item.value) for item in app.warning)

        self.assertIn("Field of practice changed", warnings)

    def test_nothing_downstream_is_erased(self):
        """
        Flagging leaves the decision to the reader. Erasing would make it
        for them.
        """
        app = self._after_changing_the_field()

        self.assertEqual(
            app.session_state.filtered_state.get("pe_kept_entered", {}).get(
                "program"
            ),
            "text-message reminders",
        )

    def test_rewording_the_question_leaves_the_designs_alone(self):
        app = _at(_answered(), QUESTION, furthest=ANALYZE)
        app.text_input(key="pe_q_program").set_value("SMS reminders").run()

        review = app.session_state.filtered_state["pe_review"]

        self.assertEqual(set(review), {RESEARCH})


class TestTheResultCrossesButTheDataDoesNot(unittest.TestCase):
    """
    The interpretation is its own screen and the analysis that produced
    the estimate is not running when it renders. What is held is the
    estimate and the method behind it. Holding the rows instead would be
    retaining a reader's data to save them a click.
    """

    def _analysed(self) -> AppTest:
        app = _at(_answered(), ANALYZE, furthest=ANALYZE)

        sample = next(
            b for b in app.button if "sample" in str(b.label).lower()
        )
        sample.click().run()

        next(
            b for b in app.button if b.label == "Get recommendation"
        ).click().run()
        next(b for b in app.button if b.label == "Run analysis").click().run()

        return app

    def test_the_estimate_is_held(self):
        held = self._analysed().session_state.filtered_state

        self.assertIsNotNone(held.get("pe_result"))
        self.assertTrue(held.get("pe_method"))

    def test_the_rows_are_not(self):
        held = self._analysed().session_state.filtered_state

        for slot in ("pe_frame", "pe_kept_frame", "pe_kept_data"):
            with self.subTest(slot=slot):
                self.assertIsNone(held.get(slot))

    def test_the_interpretation_opens_on_that_estimate(self):
        app = _at(self._analysed(), INTERPRET, furthest=INTERPRET)

        self.assertFalse(app.exception)
        self.assertIn("Interpret", _headings(app))

    def test_running_an_analysis_unblocks_the_interpretation(self):
        app = self._analysed()

        self.assertFalse(app.button(key=f"pe_rail_{INTERPRET}").disabled)

    def test_an_interpretation_with_nothing_behind_it_says_so(self):
        """
        The gate is a guard, not a guarantee. Loading a second dataset
        discards the stored estimate while the rail still remembers this
        stage was reached.

        The page used to answer this itself, with a message of its own.
        It is the workspace's job now: render_rail returns
        STAGE_UNAVAILABLE rather than the stage index, so no page can
        forget the check. See docs/stage-lifecycle.md.
        """
        app = _at(_answered(), INTERPRET, furthest=INTERPRET)

        self.assertFalse(app.exception)

        warnings = " ".join(str(item.value) for item in app.warning)
        self.assertIn("cannot open now", warnings)
        self.assertIn("Run an analysis to continue", warnings)

    def test_the_shared_layer_is_what_stops_it_rather_than_the_page(self):
        """
        A page-level re-check would work and would also be one more
        thing every migration has to remember.
        """
        source = (ROOT / PAGE).read_text(encoding="utf-8")

        self.assertNotIn("nothing here to interpret", source)

    def test_the_stage_says_the_file_is_not_carried(self):
        """
        The cost of not retaining the data is a reload, and saying so is
        better than a reader finding an empty uploader and assuming a bug.
        """
        app = _at(_answered(), ANALYZE, furthest=ANALYZE)
        captions = " ".join(str(item.value) for item in app.caption)

        self.assertIn("returning here later means loading it again", captions)


class TestTheDisclosureMatchesWhatIsHeld(unittest.TestCase):
    """
    The page's data-handling disclosure is what a reader relies on, so a
    migration that started holding a computed estimate between stages has
    to say so there rather than only in a comment.
    """

    @staticmethod
    def _notes() -> str:
        from shared.data_handling import disclosure_for

        return disclosure_for(PAGE).notes

    def test_it_names_the_estimate_it_now_holds(self):
        notes = self._notes()

        self.assertIn("estimate", notes)
        self.assertIn("session state", notes)

    def test_it_says_the_rows_are_not_held(self):
        self.assertIn("loaded rows are not held", self._notes())

    def test_it_still_names_the_handoff_hash(self):
        """
        The addition is an addition. The hash recorded for Cross-Analysis
        was already disclosed and still is.
        """
        self.assertIn("hash of the uploaded data", self._notes())


class TestWhatInvalidatesAStoredEstimate(unittest.TestCase):
    """
    An interpretation is only ever as current as the analysis under it.

    Three things can make it stale, and they are not the same. Changing
    the design changes which statistic was asked for. Clearing the data
    removes what it was computed from. Rewording the question changes
    neither, and a page that flagged it would train a reader to dismiss
    the flag.
    """

    DESIGNS = (
        "Two or more groups",
        "Pre/post (same participants)",
        "Two groups, each measured before and after",
    )

    def _analysed(self, design=None) -> AppTest:
        app = _at(_answered(), ANALYZE, furthest=ANALYZE)
        next(
            b for b in app.button if "sample" in str(b.label).lower()
        ).click()
        app.run()

        if design is not None:
            next(
                r for r in app.radio if "comparing" in str(r.label)
            ).set_value(design).run()

        next(b for b in app.button if b.label == "Get recommendation").click()
        app.run()
        next(b for b in app.button if b.label == "Run analysis").click()
        app.run()

        return app

    def test_all_three_designs_produce_their_own_method(self):
        methods = set()

        for design in self.DESIGNS:
            app = self._analysed(design)
            method = app.session_state.filtered_state.get("pe_method")

            with self.subTest(design=design):
                self.assertFalse(app.exception)
                self.assertTrue(method)

            methods.add(method)

        self.assertEqual(len(methods), 3, methods)

    def test_each_design_reaches_a_bounded_interpretation(self):
        for design in self.DESIGNS:
            app = _at(self._analysed(design), INTERPRET, furthest=INTERPRET)
            text = " ".join(str(item.value) for item in app.markdown)

            with self.subTest(design=design):
                self.assertFalse(app.exception)
                self.assertIn("Established here", text)

    def test_switching_the_design_discards_the_estimate(self):
        app = self._analysed(self.DESIGNS[0])
        self.assertIsNotNone(
            app.session_state.filtered_state.get("pe_result")
        )

        next(
            r for r in app.radio if "comparing" in str(r.label)
        ).set_value(self.DESIGNS[1]).run()
        held = app.session_state.filtered_state

        self.assertIsNone(held.get("pe_result"))
        self.assertTrue(app.button(key=f"pe_rail_{INTERPRET}").disabled)

    def test_clearing_the_data_discards_the_estimate(self):
        """
        The early exit for "nothing loaded" fires above the token check
        that used to be the only thing discarding a stale estimate, so
        the interpretation stage went on describing a result for data
        that was no longer there.
        """
        app = self._analysed()
        clear = [b for b in app.button if "Clear" in str(b.label)]

        if not clear:
            self.skipTest("no clear control on this stage")

        clear[0].click()
        app.run()
        held = app.session_state.filtered_state

        self.assertIsNone(held.get("pe_result"))
        self.assertIsNone(held.get("pe_recommendation"))
        self.assertTrue(app.button(key=f"pe_rail_{INTERPRET}").disabled)

    def test_rewording_the_question_does_not_discard_it(self):
        """
        The question feeds the literature search and nothing else. It is
        never paired with the estimate, so a reworded question cannot
        misattribute one, and flagging it would be crying wolf.
        """
        app = self._analysed()
        app = _at(app, QUESTION, furthest=INTERPRET)
        app.text_input(key="pe_q_program").set_value("SMS nudges").run()

        held = app.session_state.filtered_state

        self.assertIsNotNone(held.get("pe_result"))
        self.assertEqual(set(held.get("pe_review", {})), {RESEARCH})


if __name__ == "__main__":
    unittest.main()
