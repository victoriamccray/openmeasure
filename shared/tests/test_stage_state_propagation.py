"""
Controls in one stage still reaching the stages that read them.

Run with: pytest shared/tests/test_stage_state_propagation.py -v

The page-load tests establish that every page opens. The content
captures taken during the workspace migration established that every
stage renders the same things it used to. Neither covers the failure this
architecture invites:

    A stage renders correctly and stops updating.

Streamlit discards a widget's value when the widget is not rendered, and
in a workspace that is every widget in every other stage. A later stage
reading such a key gets the default and renders something plausible from
it. Nothing raises. GRAND shipped that way for an hour: choosing two
modalities and moving on left four stages describing the baseline scan
alone.

Deliberately small. Four interactions, chosen because each is a control
whose whole purpose is to change what a later stage says, plus one
structural check that finds the shape itself on any page. Not
exhaustive per-stage UI testing, which would cost more than it is worth
at this size.

Each test changes a control on the stage that owns it, moves to a stage
that reads it, and asserts the rendered content actually differs. A
weaker assertion, that a session key exists, would pass on exactly the
sessions where this bug bites.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = str(ROOT / "Home.py")
LOAD_TIMEOUT_SECONDS = 300


def _rendered(app: AppTest) -> set[str]:
    """Everything a reader could read, as a set for differencing."""
    found = set()

    for name in ("markdown", "caption", "info", "subheader", "success", "warning"):
        for item in getattr(app, name, []):
            value = str(getattr(item, "value", "")).strip()
            if value:
                found.add(value)

    for item in app.metric:
        found.add(f"[metric] {item.label}={item.value}")

    return found


def _at(app: AppTest, key: str, stage: int, *, furthest: int) -> AppTest:
    app.session_state[f"{key}_current"] = stage
    app.session_state[f"{key}_furthest"] = furthest
    app.run()

    return app


def _open(page: str) -> AppTest:
    app = AppTest.from_file(ENTRYPOINT, default_timeout=LOAD_TIMEOUT_SECONDS)
    app.run()
    app.switch_page(page).run()

    return app


class TestGrandModalitySelectionReachesLaterStages(unittest.TestCase):
    """
    The one that was broken. Four stages read this selection.
    """

    PAGE = "pages/GRAND_Worked_Example.py"

    def _at_stage_with(self, stage: int, n_modalities: int):
        app = _at(_open(self.PAGE), "grand", 1, furthest=8)
        control = app.multiselect[0]
        control.set_value(list(control.options)[:n_modalities]).run()

        return _at(app, "grand", stage, furthest=8)

    def test_the_selection_survives_leaving_the_stage_that_made_it(self):
        app = self._at_stage_with(3, 2)
        held = app.session_state.filtered_state

        self.assertEqual(len(held.get("grand_kept_modalities", [])), 2)

    def test_a_later_stage_reports_something_different_for_it(self):
        """
        Content, not a session key. Reading the key back would pass on
        the very sessions where this bug bites.
        """
        without = _rendered(self._at_stage_with(2, 0))
        with_two = _rendered(self._at_stage_with(2, 2))

        self.assertNotEqual(without, with_two)
        self.assertGreater(len(with_two - without), 0)

    def test_the_integration_stage_counts_what_was_added(self):
        app = self._at_stage_with(5, 2)
        counts = [
            item.value for item in app.metric
            if "Modalities integrated" in str(item.label)
        ]

        self.assertEqual(counts, ["3"])

    def test_no_stage_raises_with_a_selection_made(self):
        for stage in range(1, 9):
            with self.subTest(stage=stage):
                self.assertFalse(self._at_stage_with(stage, 2).exception)


class TestMultimodalSelectionAndWeightsReachTheDecision(unittest.TestCase):
    """
    Five controls here were read by later stages: the selection by three
    of them, the perspective by one, and each cost weight by the
    research decision.
    """

    PAGE = "pages/Multimodal_Signal_Convergence.py"

    def _decision_with(self, n_modalities: int):
        app = _at(_open(self.PAGE), "mmsc", 2, furthest=6)
        control = app.multiselect[0]
        control.set_value(list(control.options)[:n_modalities]).run()

        return _at(app, "mmsc", 6, furthest=6)

    def test_the_selection_reaches_the_research_decision(self):
        without = _rendered(self._decision_with(0))
        with_two = _rendered(self._decision_with(2))

        self.assertNotEqual(without, with_two)

    def test_it_is_held_under_a_name_that_outlives_the_widget(self):
        held = self._decision_with(2).session_state.filtered_state

        self.assertEqual(len(held.get("mmsc_kept_modalities", [])), 2)

    def test_the_cost_weights_are_held_for_the_decision(self):
        """
        The weights are drawn inside the tradeoff stage's own nested
        sequence and read two stages later.
        """
        app = _at(_open(self.PAGE), "mmsc", 2, furthest=6)
        control = app.multiselect[0]
        control.set_value(list(control.options)[:2]).run()

        app = _at(app, "mmsc", 4, furthest=6)
        sliders = [s for s in app.slider if "weight" in str(s.label).lower()]

        if not sliders:
            self.skipTest("the nested cost sequence has not reached a weight")

        sliders[0].set_value(0.9).run()
        held = app.session_state.filtered_state

        self.assertAlmostEqual(float(held["mmsc_kept_privacy"]), 0.9, places=3)


class TestGaiaPredictionReachesItsReveal(unittest.TestCase):
    """
    The prediction is committed before the result is revealed, and the
    reveal quotes it back. If the prediction stopped arriving, the page
    would report whatever the default was and look fine doing it.
    """

    PAGE = "pages/GAIA_Worked_Example.py"

    # The prediction sits inside the Understand-the-task stage, not the
    # performance comparison it predicts against.
    STAGE = 1

    def _predicted(self, answer: str) -> AppTest:
        app = _at(_open(self.PAGE), "gaia", self.STAGE, furthest=6)

        control = next(
            item for item in app.radio
            if "do you expect" in str(item.label)
        )
        control.set_value(answer).run()

        reveal = [b for b in app.button if str(b.label) == "Reveal result"]
        if reveal:
            reveal[0].click()
            app.run()

        return app

    def test_the_reveal_quotes_the_prediction_that_was_made(self):
        app = self._predicted("No, the Student should do better")
        rendered = " ".join(_rendered(app))

        self.assertIn("You predicted: No, the Student should do better", rendered)

    def test_a_different_prediction_is_quoted_differently(self):
        first = " ".join(_rendered(self._predicted("Yes, about the same")))
        second = " ".join(
            _rendered(self._predicted("No, the Student should do better"))
        )

        self.assertIn("Yes, about the same", first)
        self.assertNotIn("You predicted: Yes, about the same", second)

    def test_the_result_is_not_shown_before_it_is_revealed(self):
        """
        The prediction has to be committed first, or the exercise is
        pointless.
        """
        app = _at(_open(self.PAGE), "gaia", self.STAGE, furthest=6)
        rendered = " ".join(_rendered(app))

        self.assertNotIn("You predicted", rendered)


class TestHealthRingControlsReachTheConclusion(unittest.TestCase):
    """
    The longest chain of the six: the split choice, the seed and the
    quality threshold are drawn on two stages and read by five.
    """

    PAGE = "pages/HealthRing_Worked_Example.py"

    def _loaded(self, stage: int) -> AppTest:
        app = _at(_open(self.PAGE), "hr", 1, furthest=10)

        load = [
            b for b in app.button
            if str(b.label) == "Load public HealthRing example"
        ]
        if not load:
            self.skipTest("the public example is not offered")

        load[0].click()
        app.run()

        return _at(app, "hr", stage, furthest=10)

    def test_the_conclusion_reports_the_measured_error(self):
        """
        A number, not a placeholder. Before the derivation chain was
        hoisted this stage said the figure was unavailable.
        """
        rendered = " ".join(_rendered(self._loaded(9)))

        self.assertRegex(rendered, r"\d+\.\d+ bpm")

    def test_the_finish_stage_reports_a_retention_figure(self):
        """
        The regression a content diff caught: the retention filter is
        drawn two stages earlier and was not reaching this one, so the
        record said there was no figure to report.
        """
        rendered = " ".join(_rendered(self._loaded(10)))

        self.assertNotIn("no retention figure to report", rendered)

    def test_changing_the_quality_threshold_changes_what_is_retained(self):
        app = self._loaded(7)
        sliders = [
            s for s in app.slider if "signal quality" in str(s.label).lower()
        ]

        if not sliders:
            self.skipTest("the retention slider is not on this stage")

        before = _rendered(app)
        sliders[0].set_value(0.95).run()
        after = _rendered(app)

        self.assertNotEqual(before, after)
        self.assertAlmostEqual(
            float(app.session_state.filtered_state["hr_kept_quality_threshold"]),
            0.95,
            places=3,
        )

    def test_changing_the_split_changes_the_evaluation(self):
        app = self._loaded(3)
        control = [
            r for r in app.radio if "split" in str(r.label).lower()
        ]

        if not control:
            self.skipTest("the split control is not on this stage")

        before = _rendered(_at(app, "hr", 6, furthest=10))

        app = self._loaded(3)
        options = list(control[0].options)
        app.radio(key="hr_split_choice").set_value(options[-1]).run()
        after = _rendered(_at(app, "hr", 6, furthest=10))

        self.assertNotEqual(before, after)


class TestNoPageReadsAWidgetKeyFromAnotherStage(unittest.TestCase):
    """
    The structural version, which finds the shape rather than one
    instance of it.

    Six instances existed across three migrated pages and none raised
    anywhere. A test per interaction would not have found the ones
    nobody thought to write a test for; this does.
    """

    PAGES = (
        "2_Impact_Evaluation.py",
        "Portfolio_Impact_Analysis.py",
        "Method_Selection.py",
        "Right_To_Play_Journey.py",
        "Pulse_Oximeter_Worked_Example.py",
        "FMRI_QC_Worked_Example.py",
        "GAIA_Worked_Example.py",
        "GRAND_Worked_Example.py",
        "Multimodal_Signal_Convergence.py",
        "HealthRing_Worked_Example.py",
    )

    STAGE_START = re.compile(r"^\s*(?:el)?if\s+\w*stage\w*\s*==\s*(\w+)\s*[:\)]")
    WIDGET_KEY = re.compile(r'key=(?:"([^"]+)"|([A-Z_][A-Z_0-9]*))')
    READ = re.compile(
        r'st\.session_state(?:\.get\(\s*(?:"([^"]+)"|([A-Z_][A-Z_0-9]*))'
        r'|\[\s*(?:"([^"]+)"|([A-Z_][A-Z_0-9]*))\s*\])'
    )

    def _findings(self, name):
        text = (ROOT / "pages" / name).read_text(encoding="utf-8")
        lines = text.splitlines()

        resolve = {
            match.group(1): match.group(2)
            for match in re.finditer(
                r'^([A-Z_][A-Z_0-9]*)\s*=\s*"([^"]+)"$', text, re.MULTILINE
            )
        }

        owner = [None] * len(lines)
        current = None
        indent = 0

        for index, line in enumerate(lines):
            match = self.STAGE_START.match(line)

            if match:
                current = match.group(1)
                indent = len(line) - len(line.lstrip())
            elif current is not None:
                stripped = line.strip()
                if stripped and (len(line) - len(line.lstrip())) <= indent:
                    current = None

            owner[index] = current

        declared = {}
        read_in = {}

        for index, line in enumerate(lines):
            for match in self.WIDGET_KEY.finditer(line):
                key = match.group(1) or resolve.get(
                    match.group(2), match.group(2)
                )
                declared.setdefault(key, set()).add(owner[index])

            for match in self.READ.finditer(line):
                raw = next(group for group in match.groups() if group)
                key = raw if raw.islower() or '"' in line else resolve.get(
                    raw, raw
                )
                read_in.setdefault(key, set()).add(owner[index])

        findings = []

        for key, stages in declared.items():
            drawn_in = {stage for stage in stages if stage is not None}
            if not drawn_in:
                continue

            readers = read_in.get(key, set())
            elsewhere = {
                stage for stage in readers
                if stage is not None and stage not in drawn_in
            }

            if elsewhere or None in readers:
                findings.append((key, sorted(drawn_in), sorted(
                    stage or "<module level>"
                    for stage in elsewhere | ({None} if None in readers else set())
                )))

        return findings

    def test_the_audit_looks_at_pages_that_have_widgets(self):
        """So a pass cannot come from finding nothing to check."""
        with_widgets = [
            name for name in self.PAGES
            if 'key="' in (ROOT / "pages" / name).read_text(encoding="utf-8")
        ]

        self.assertGreaterEqual(len(with_widgets), 6)

    def test_no_widget_key_is_read_outside_the_stage_that_draws_it(self):
        for name in self.PAGES:
            for key, drawn, read in self._findings(name):
                with self.subTest(page=name, key=key):
                    self.fail(
                        f"{key!r} is drawn in {', '.join(drawn)} and read in "
                        f"{', '.join(read)}. Streamlit drops a widget's value "
                        "when the widget is not rendered, so that read gets "
                        "the default and the stage renders something "
                        "plausible from it. Mirror it into a name the "
                        "workspace keeps."
                    )


if __name__ == "__main__":
    unittest.main()
