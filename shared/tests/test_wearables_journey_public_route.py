"""
The Wearables journey, completed the way a web visitor completes it.

Run with: pytest shared/tests/test_wearables_journey_public_route.py -v

The definition this pins:

    A web user can load the bundled HealthRing example and complete the
    intended public-data journey without needing the original archive or
    encountering stale state.

Every part of that is checkable and none of it was covered. The
page-load test opens the page on its first stage. The migration's content
capture walked all eleven stages but asserted only that they render the
same things they used to. Neither says a visitor can get from the load
button to Finish Study.

The archive route is the specific hazard. It is 2.4 GiB against
Streamlit's 200 MB upload cap, so on a hosted copy it is a local-run
route and nothing more. A visitor who concluded they needed it would
abandon the journey at the second stage, so the tests below check both
that the public route completes and that the page says so.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = str(ROOT / "Home.py")
PAGE = "pages/HealthRing_Worked_Example.py"
LOAD_TIMEOUT_SECONDS = 300

LOAD_LABEL = "Load public HealthRing example"

QUESTION, MEASUREMENT, SIGNAL, DESIGN, BASELINE, MODEL = range(6)
EVALUATE, RETENTION, CONDITIONS, CONCLUSION, FINISH = range(6, 11)

STAGE_HEADINGS = {
    QUESTION: "Research Question",
    MEASUREMENT: "Understand Measurement",
    SIGNAL: "Signal Inspection",
    DESIGN: "Design the Evaluation",
    BASELINE: "Establish Baseline",
    MODEL: "Build Model",
    EVALUATE: "Evaluate",
    RETENTION: "Weigh the Retention Tradeoff",
    CONDITIONS: "Does It Hold Across Conditions?",
    CONCLUSION: "Defend Your Conclusion",
    FINISH: "Finish Study",
}


def _rendered(app: AppTest) -> set[str]:
    found = set()

    for name in ("markdown", "caption", "info", "subheader", "success", "warning"):
        for item in getattr(app, name, []):
            value = str(getattr(item, "value", "")).strip()
            if value:
                found.add(value)

    for item in app.metric:
        found.add(f"[metric] {item.label}={item.value}")

    return found


def _unloaded() -> AppTest:
    """The page as a visitor first meets it, with nothing loaded."""
    app = AppTest.from_file(ENTRYPOINT, default_timeout=LOAD_TIMEOUT_SECONDS)
    app.run()
    app.switch_page(PAGE).run()

    return app


def _at(app: AppTest, stage: int) -> AppTest:
    app.session_state["hr_current"] = stage
    app.session_state["hr_furthest"] = max(
        stage, int(app.session_state.filtered_state.get("hr_furthest", 0))
    )
    app.run()

    return app


def _loaded() -> AppTest:
    """A session that has pressed the public-example button and nothing else."""
    app = _at(_unloaded(), MEASUREMENT)
    load = [button for button in app.button if str(button.label) == LOAD_LABEL]

    if not load:
        raise AssertionError(
            f"{LOAD_LABEL!r} is not offered; found "
            f"{[str(b.label) for b in app.button]}"
        )

    load[0].click()
    app.run()

    return app


class TestThePublicExampleLoads(unittest.TestCase):
    def test_the_button_is_on_the_stage_before_the_data_is_needed(self):
        """
        Signal Inspection is the first stage that needs a recording, so
        the loader has to be reachable before it.
        """
        app = _at(_unloaded(), MEASUREMENT)
        labels = [str(button.label) for button in app.button]

        self.assertIn(LOAD_LABEL, labels)

    def test_pressing_it_loads_windows_from_the_bundled_artifact(self):
        held = _loaded().session_state.filtered_state

        self.assertEqual(held.get("healthring_source"), "public_subset")
        self.assertIsNotNone(held.get("healthring_windows"))
        self.assertEqual(held.get("healthring_n_subjects"), 28)

    def test_it_raises_nothing(self):
        self.assertFalse(_loaded().exception)


class TestEveryStageCompletesOnThePublicRoute(unittest.TestCase):
    """
    Eleven stages, none of them blocked and none of them raising, with
    only the bundled artifact loaded.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = _loaded()

    def test_each_stage_renders_its_own_heading(self):
        app = self.app

        for stage, heading in STAGE_HEADINGS.items():
            app = _at(app, stage)
            with self.subTest(stage=stage, heading=heading):
                self.assertFalse(app.exception)
                self.assertIn(
                    heading, [str(item.value) for item in app.subheader]
                )

    def test_no_stage_reports_itself_unavailable(self):
        """
        The workspace closes a stage whose prerequisite has gone. On this
        route nothing should be missing, so nothing should be closed.
        """
        app = self.app

        for stage in STAGE_HEADINGS:
            app = _at(app, stage)
            warnings = " ".join(str(item.value) for item in app.warning)

            with self.subTest(stage=stage):
                self.assertNotIn("cannot open now", warnings)

    def test_the_last_three_stages_report_measured_numbers(self):
        """
        Across conditions, then the conclusion, then the record. A
        placeholder in any of them means the chain stopped arriving.
        """
        app = self.app

        for stage in (CONDITIONS, CONCLUSION, FINISH):
            app = _at(app, stage)
            text = " ".join(_rendered(app))

            with self.subTest(stage=stage):
                self.assertRegex(text, r"\d+\.\d+ bpm")


class TestStatePersistsBothDirections(unittest.TestCase):
    def test_walking_forward_then_back_keeps_the_data(self):
        app = _loaded()

        for stage in STAGE_HEADINGS:
            app = _at(app, stage)

        for stage in reversed(list(STAGE_HEADINGS)):
            app = _at(app, stage)

        held = app.session_state.filtered_state

        self.assertIsNotNone(held.get("healthring_windows"))
        self.assertFalse(app.exception)

    def test_the_split_choice_survives_leaving_its_stage(self):
        app = _at(_loaded(), DESIGN)
        control = [item for item in app.radio if "split" in str(item.label).lower()]

        if not control:
            self.skipTest("the split control is not on this stage")

        options = list(control[0].options)
        app.radio(key="hr_split_choice").set_value(options[-1]).run()

        app = _at(app, FINISH)
        held = app.session_state.filtered_state

        self.assertEqual(held.get("hr_kept_split_choice"), "window")


class TestTheControlsMoveWhatComesAfterThem(unittest.TestCase):
    def test_the_retention_threshold_changes_its_own_stage(self):
        app = _at(_loaded(), RETENTION)
        sliders = [
            item for item in app.slider
            if "signal quality" in str(item.label).lower()
        ]

        if not sliders:
            self.skipTest("the retention slider is not on this stage")

        before = _rendered(app)
        sliders[0].set_value(0.95).run()

        self.assertNotEqual(before, _rendered(app))
        self.assertAlmostEqual(
            float(app.session_state.filtered_state["hr_kept_quality_threshold"]),
            0.95,
            places=3,
        )

    def test_the_finish_stage_reports_a_retention_figure(self):
        """
        The regression the migration's content diff caught: the record
        said there was no figure to report, because the filter was drawn
        two stages earlier and never reached this one.
        """
        text = " ".join(_rendered(_at(_loaded(), FINISH)))

        self.assertNotIn("no retention figure to report", text)

    def test_changing_the_split_changes_the_evaluation(self):
        before = " ".join(_rendered(_at(_loaded(), EVALUATE)))

        app = _at(_loaded(), DESIGN)
        control = [item for item in app.radio if "split" in str(item.label).lower()]

        if not control:
            self.skipTest("the split control is not on this stage")

        app.radio(key="hr_split_choice").set_value(list(control[0].options)[-1]).run()
        after = " ".join(_rendered(_at(app, EVALUATE)))

        self.assertNotEqual(before, after)


class TestTheArchiveIsNotPresentedAsRequired(unittest.TestCase):
    """
    2.4 GiB against a 200 MB upload cap makes this a local-run route. A
    visitor who took it for a requirement would stop at the second
    stage.
    """

    def test_the_archive_route_is_collapsed_and_labelled_advanced(self):
        app = _at(_unloaded(), MEASUREMENT)
        labels = [str(item.label) for item in app.expander]

        self.assertTrue(
            any("Advanced" in label and "archive" in label for label in labels),
            labels,
        )

    def test_the_artifact_description_says_the_journey_runs_on_it(self):
        from shared.datasets import get_dataset

        description = get_dataset("healthring").derived_artifact.description

        self.assertIn("Every stage of the journey runs on this", description)

    def test_the_one_stage_that_wants_waveforms_says_what_it_shows_instead(self):
        app = _at(_loaded(), SIGNAL)
        text = " ".join(_rendered(app))

        self.assertIn("Every later stage runs on these", text)
        self.assertIn("only the waveform walk-through needs the archive", text)

    def test_that_stage_still_shows_the_columns_it_does_have(self):
        app = _at(_loaded(), SIGNAL)

        self.assertFalse(app.exception)
        self.assertGreaterEqual(len(app.dataframe), 1)


if __name__ == "__main__":
    unittest.main()
