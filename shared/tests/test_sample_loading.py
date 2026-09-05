"""
End-to-end: the one-click sample loads on every numbered analysis page.

shared/tests/test_upload.py covers the registry and the dataclass.
shared/tests/test_pages_render.py covers that each page loads at all.
Neither reaches past the data-entry step, which is exactly where a
visitor with no CSV used to stop: the sample was offered as a download,
so seeing a worked result meant downloading a file and uploading it back.

These drive the actual button. Pages go through the entrypoint (Home.py)
via switch_page for the same reason test_pages_render.py does: st.page_link
in the lifecycle tracker resolves against the navigation Home.py declares.

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest

from shared.upload import SAMPLE_BUTTON_LABEL, SAMPLE_CLEAR_LABEL

# Every page that offers a bundled example, with the section heading that
# should be reachable once it is loaded.
IMPACT_EVALUATION = "pages/2_Impact_Evaluation.py"

ANALYSIS_PAGES = (
    "pages/1_Reliability.py",
    "pages/2_Impact_Evaluation.py",
    "pages/3_Fairness.py",
    "pages/4_Time_Series_QA.py",
)

TIMEOUT_SECONDS = 180


def _button(app: AppTest, label: str):
    """The button carrying an exact label, or None if it is not rendered."""
    return next((b for b in app.button if b.label == label), None)


def _open(page: str) -> AppTest:
    app = AppTest.from_file("Home.py", default_timeout=TIMEOUT_SECONDS)
    app.run()
    app.switch_page(page).run()
    if page == IMPACT_EVALUATION:
        _walk_to_analysis(app)
    return app


def _walk_to_analysis(app: AppTest) -> None:
    """
    Advance Impact Evaluation to its data step.

    That page is a gated sequence now, so its data entry sits behind five
    Continue clicks rather than being the first thing on the page. The
    first stage needs a program and an outcome before it will let anyone
    past, which is what the two set_value calls supply.
    """
    app.text_input(key="pe_q_program").set_value("reminder texts")
    app.text_input(key="pe_q_outcome").set_value("appointments kept").run()

    for label in (
        "Continue to domain",
        "Continue to find research",
        "Skip this step",
        "Continue to the worked example",
        "Continue to your own data",
    ):
        button = _button(app, label)
        assert button is not None, f"'{label}' was not offered."
        button.click().run()


class TestSampleLoadsWithoutUploading(unittest.TestCase):
    def test_every_analysis_page_offers_the_sample_before_any_upload(self):
        for page in ANALYSIS_PAGES:
            with self.subTest(page=page):
                app = _open(page)

                self.assertFalse(app.exception, f"{page} raised on load.")
                self.assertIsNotNone(
                    _button(app, SAMPLE_BUTTON_LABEL),
                    f"{page} does not offer '{SAMPLE_BUTTON_LABEL}'.",
                )

    def test_clicking_the_sample_button_loads_data(self):
        for page in ANALYSIS_PAGES:
            with self.subTest(page=page):
                app = _open(page)
                _button(app, SAMPLE_BUTTON_LABEL).click().run()

                self.assertFalse(
                    app.exception, f"{page} raised after loading the sample."
                )
                self.assertIsNotNone(
                    _button(app, SAMPLE_CLEAR_LABEL),
                    f"{page} did not enter its loaded state.",
                )

    def test_the_loaded_sample_reaches_the_data_profile(self):
        """
        Every page renders the shared data profile once data is in hand, so
        its expander appearing is the signal that a page got past data
        entry rather than merely redrawing.
        """
        for page in ANALYSIS_PAGES:
            with self.subTest(page=page):
                app = _open(page)
                _button(app, SAMPLE_BUTTON_LABEL).click().run()

                labels = [expander.label for expander in app.expander]
                self.assertTrue(
                    any("Data profile" in label for label in labels),
                    f"{page} did not render a data profile. Saw: {labels}",
                )

    def test_clearing_the_sample_returns_to_the_entry_step(self):
        app = _open("pages/1_Reliability.py")
        _button(app, SAMPLE_BUTTON_LABEL).click().run()
        _button(app, SAMPLE_CLEAR_LABEL).click().run()

        self.assertFalse(app.exception)
        self.assertIsNotNone(_button(app, SAMPLE_BUTTON_LABEL))
        self.assertIsNone(_button(app, SAMPLE_CLEAR_LABEL))


class TestTeachingContentIsReachableWithoutData(unittest.TestCase):
    """
    Impact Evaluation's difference-in-differences example, its field
    selector, and its glossary teach something that does not depend on the
    reader's own data, and all three come before the data step in the
    sequence. They used to sit below the upload gate, inside the DiD
    design branch, so a visitor could not reach any of it without
    downloading a CSV and uploading it back.
    """

    def test_the_did_example_is_reachable_before_any_data_is_loaded(self):
        """
        Asserts on the example's own content rather than on a wrapper.
        It used to sit behind an expander, and checking for that label
        broke the moment the example was made visible by default, which
        was an improvement rather than a regression.
        """
        app = _open(IMPACT_EVALUATION)

        self.assertFalse(app.exception)

        slider_labels = [slider.label for slider in app.slider]
        self.assertIn("How much did Clinic B change?", slider_labels)

        metric_labels = [metric.label for metric in app.metric]
        self.assertIn("Clinic A change", metric_labels)

    def test_the_example_leads_with_doing_not_explaining(self):
        """
        The subtraction comes before the justification. Everything that
        explains or qualifies the method is collapsed, so a reader works
        the comparison before reading about counterfactuals.
        """
        app = _open(IMPACT_EVALUATION)

        labels = [expander.label for expander in app.expander]
        for collapsed in (
            "Why use a comparison group at all?",
            "What this assumes",
            "What this example can and cannot support",
        ):
            with self.subTest(expander=collapsed):
                self.assertIn(collapsed, labels)

        rendered = " ".join(str(m.value) for m in app.markdown)
        self.assertLess(
            rendered.index("Compare the changes"),
            rendered.index("See it"),
            "the chart should come after the subtraction",
        )

    def test_the_field_selector_offers_every_domain(self):
        from modules.program_evaluation.core.domains import DOMAINS

        app = _open(IMPACT_EVALUATION)

        selectboxes = [s for s in app.selectbox if s.label == "Field of practice"]
        self.assertEqual(len(selectboxes), 1)
        self.assertEqual(len(selectboxes[0].options), len(DOMAINS))

    def test_the_first_stage_will_not_advance_without_a_question(self):
        """
        The evaluation question seeds the literature query, so an empty
        one would send a domain's terms alone to OpenAlex as if they were
        the researcher's question.
        """
        app = AppTest.from_file("Home.py", default_timeout=TIMEOUT_SECONDS)
        app.run()
        app.switch_page(IMPACT_EVALUATION).run()

        advance = _button(app, "Continue to domain")
        self.assertIsNotNone(advance)
        self.assertTrue(advance.disabled)


class TestStageThreeIsOptional(unittest.TestCase):
    """
    Literature discovery enriches the workflow and never gates it. Both
    routes past it have to reach the same place.
    """

    @staticmethod
    def _walk(app, third_stage_label):
        app.text_input(key="pe_q_program").set_value("reminder texts")
        app.text_input(key="pe_q_outcome").set_value("appointments kept").run()
        for label in (
            "Continue to domain",
            "Continue to find research",
            third_stage_label,
            "Continue to the worked example",
            "Continue to your own data",
        ):
            button = _button(app, label)
            assert button is not None, f"'{label}' was not offered."
            button.click().run()

    def test_skipping_the_search_still_reaches_the_data_step(self):
        app = AppTest.from_file("Home.py", default_timeout=TIMEOUT_SECONDS)
        app.run()
        app.switch_page(IMPACT_EVALUATION).run()

        self._walk(app, "Skip this step")

        self.assertFalse(app.exception)
        self.assertIsNotNone(_button(app, SAMPLE_BUTTON_LABEL))

    def test_continuing_without_searching_reaches_the_same_place(self):
        """
        A reader who opens the search stage, runs no search, and presses
        Continue must land where Skip lands, not somewhere else.
        """
        app = AppTest.from_file("Home.py", default_timeout=TIMEOUT_SECONDS)
        app.run()
        app.switch_page(IMPACT_EVALUATION).run()

        self._walk(app, "Continue to designs")

        self.assertFalse(app.exception)
        self.assertIsNotNone(_button(app, SAMPLE_BUTTON_LABEL))

    def test_no_search_runs_unless_the_button_is_pressed(self):
        """
        The query box is prefilled, which must not be mistaken for the
        search having been sent. Nothing leaves the machine on its own.
        """
        app = AppTest.from_file("Home.py", default_timeout=TIMEOUT_SECONDS)
        app.run()
        app.switch_page(IMPACT_EVALUATION).run()

        app.text_input(key="pe_q_program").set_value("reminder texts")
        app.text_input(key="pe_q_outcome").set_value("appointments kept").run()
        _button(app, "Continue to domain").click().run()
        _button(app, "Continue to find research").click().run()

        self.assertIsNotNone(_button(app, "Search OpenAlex"))
        self.assertNotIn("pe_search_results", app.session_state)
        self.assertNotIn("pe_search_provenance", app.session_state)


class TestInterpretationStagePreservesTheResult(unittest.TestCase):
    """
    Regression: stage 7 opens because an analysis just produced a result,
    and that result exists only in the pass that computed it.

    StageTracker.advance_to() reruns, which would discard it and leave the
    interpretation heading above an empty analysis. mark_reached() does
    not rerun, which is what lets both render together. This asserts the
    numbers and the interpretation are on screen at the same time.
    """

    def _run_analysis(self):
        app = _open(IMPACT_EVALUATION)
        _button(app, SAMPLE_BUTTON_LABEL).click().run()
        _button(app, "Get recommendation").click().run()
        _button(app, "Run analysis").click().run()
        return app

    def test_the_result_and_its_interpretation_render_together(self):
        app = self._run_analysis()

        self.assertFalse(app.exception)

        headings = [str(item.value) for item in app.subheader]
        self.assertIn("Result", headings)
        self.assertIn("7. Interpret", headings)

    def test_the_computed_numbers_survive_into_the_interpretation_stage(self):
        """
        The metrics are the result itself. If the tracker had rerun, this
        stage would be reached with the metrics gone.
        """
        app = self._run_analysis()

        metric_labels = [metric.label for metric in app.metric]
        self.assertIn("p-value", metric_labels)

    def test_the_analysis_is_recorded_for_cross_analysis(self):
        from shared.handoff import STORE_KEY

        app = self._run_analysis()

        self.assertIn(STORE_KEY, app.session_state)


class TestFormulaExplanationReachesTheResult(unittest.TestCase):
    """
    The formula component renders beside the statistic it explains, with
    notation collapsed. A reader meets the concept and their own numbers
    first; the symbols are one click away rather than the first thing on
    screen.
    """

    def _run_analysis(self):
        app = _open(IMPACT_EVALUATION)
        _button(app, SAMPLE_BUTTON_LABEL).click().run()
        _button(app, "Get recommendation").click().run()
        _button(app, "Run analysis").click().run()
        return app

    def test_the_concept_and_the_substituted_numbers_are_both_shown(self):
        app = self._run_analysis()

        self.assertFalse(app.exception)

        rendered = " ".join(str(m.value) for m in app.markdown)
        self.assertIn("Difference between group means", rendered)
        self.assertIn("Typical variation within a group", rendered)

    def test_the_notation_is_collapsed_rather_than_leading(self):
        app = self._run_analysis()

        labels = [expander.label for expander in app.expander]
        self.assertIn("Show the notation and what each symbol means", labels)

    def test_the_substituted_line_matches_the_computed_effect_size(self):
        """
        The page shows one Cohen's d in a metric and another inside the
        explanation. They have to be the same number.
        """
        app = self._run_analysis()

        metric = next(m for m in app.metric if m.label == "Cohen's d")
        rendered = " ".join(str(m.value) for m in app.markdown)

        self.assertIn(f"= {metric.value}", rendered)


class TestResultsSurviveInteraction(unittest.TestCase):
    """
    Regression: the result section holds interactive elements of its own,
    and st.button is True only on the pass that clicked it.

    While the section was gated on the button, touching the effect-size
    step-through reran the script, the gate found the button False, and
    the entire result disappeared. On screen that read as the page
    jumping back a step. Gating on stored state instead is what keeps a
    result on screen while a reader explores it.
    """

    def _run_analysis(self):
        app = _open(IMPACT_EVALUATION)
        _button(app, SAMPLE_BUTTON_LABEL).click().run()
        _button(app, "Get recommendation").click().run()
        _button(app, "Run analysis").click().run()
        return app

    def test_the_result_is_still_there_after_stepping_the_visual(self):
        app = self._run_analysis()

        stepper = next(
            s for s in app.select_slider if s.label == "Stage"
        )
        stepper.set_value("Take the difference").run()

        self.assertFalse(app.exception)

        headings = [str(item.value) for item in app.subheader]
        self.assertIn("Result", headings)
        self.assertIn("7. Interpret", headings)

    def test_the_stepper_actually_advances(self):
        app = self._run_analysis()

        stepper = next(s for s in app.select_slider if s.label == "Stage")
        stepper.set_value("Measure it against the spread").run()

        rendered = " ".join(str(m.value) for m in app.markdown)
        self.assertIn("Measure it against the spread", rendered)

    def test_a_rerun_that_touches_nothing_keeps_the_result(self):
        """
        Any widget anywhere on the page reruns the script, not only the
        ones inside the result.
        """
        app = self._run_analysis()
        app.run()

        self.assertFalse(app.exception)
        self.assertIn("Result", [str(i.value) for i in app.subheader])

    def test_a_new_recommendation_clears_the_previous_run(self):
        """
        Otherwise the last analysis would sit under a recommendation it
        was not produced from.
        """
        app = self._run_analysis()
        self.assertIn("Result", [str(i.value) for i in app.subheader])

        _button(app, "Get recommendation").click().run()

        self.assertNotIn("Result", [str(i.value) for i in app.subheader])


if __name__ == "__main__":
    unittest.main()
