"""
Portfolio Impact Analysis as a staged workspace.

Run with: pytest shared/tests/test_portfolio_workspace.py -v

This journey is a chain: a claim, then evidence for it, then whether that
evidence clears a bar, then what claim it supports, then what it leaves
open. Each stage already checked its own prerequisite inline before the
migration, so what these tests establish is what the inline checks could
not do.

Two things:

- An unreachable stage says what would unblock it, and says the first
  unmet requirement in the chain rather than the nearest one.
- A stage whose artifact is later discarded stops rendering rather than
  describing something that is gone. That is the Impact Evaluation crash
  in a page that never hand-wrote the guard, which is the point of it
  living in the workspace.

Driven by setting position rather than by clicking Continue: go_to()
reruns mid-script and AppTest accumulates the widgets from both passes.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = str(ROOT / "Home.py")
PAGE = "pages/Portfolio_Impact_Analysis.py"
LOAD_TIMEOUT_SECONDS = 180

CLAIM, EVIDENCE, VALIDATE, SUPPORTED, LIMITATIONS, PORTFOLIO, RECORD = range(7)

# Every artifact, and the stage each one unlocks.
ARTIFACT_FOR_STAGE = {
    EVIDENCE: "pia_claim",
    VALIDATE: "pia_bundle",
    SUPPORTED: "pia_validation",
    LIMITATIONS: "pia_supported",
    PORTFOLIO: "pia_limitations",
    RECORD: "pia_portfolio_context",
}


def _click(app: AppTest, label: str) -> AppTest:
    matches = [button for button in app.button if button.label == label]

    if not matches:
        raise AssertionError(
            f"no {label!r} button; found "
            f"{[str(b.label) for b in app.button]}"
        )

    matches[0].click()
    app.run()

    return app


def _open() -> AppTest:
    app = AppTest.from_file(ENTRYPOINT, default_timeout=LOAD_TIMEOUT_SECONDS)
    app.run()
    app.switch_page(PAGE).run()

    return app


def _at(app: AppTest, stage: int) -> AppTest:
    app.session_state["pia_current"] = stage
    app.session_state["pia_furthest"] = max(
        stage, int(app.session_state.filtered_state.get("pia_furthest", 0))
    )
    app.run()

    return app


def _through_validation() -> AppTest:
    """A session that has a claim, a bundle, and validation results."""
    app = _click(_open(), "Use this claim")
    app = _click(_at(app, EVIDENCE), "Summarize evidence")
    app = _click(_at(app, VALIDATE), "Run validation")

    return app


def _headings(app: AppTest) -> str:
    return " ".join(str(item.value) for item in app.subheader)


class TestTheChainOpensOneLinkAtATime(unittest.TestCase):
    def test_it_opens_on_the_claim(self):
        app = _open()

        self.assertFalse(app.exception)
        self.assertIn("Define Claim", _headings(app))

    def test_nothing_past_the_claim_is_reachable_at_first(self):
        app = _open()

        for stage in range(EVIDENCE, RECORD + 1):
            with self.subTest(stage=stage):
                self.assertTrue(app.button(key=f"pia_rail_{stage}").disabled)

    def test_defining_a_claim_opens_the_evidence_stage_and_no_further(self):
        """
        publish reruns, so the stage unlocks on the pass that created the
        artifact rather than one interaction later.
        """
        app = _click(_open(), "Use this claim")

        self.assertFalse(app.button(key=f"pia_rail_{EVIDENCE}").disabled)
        self.assertTrue(app.button(key=f"pia_rail_{VALIDATE}").disabled)

    def test_every_stage_renders_once_its_link_exists(self):
        app = _through_validation()

        for stage in (SUPPORTED, LIMITATIONS, PORTFOLIO, RECORD):
            app = _at(app, stage)
            with self.subTest(stage=stage):
                self.assertFalse(app.exception)
                self.assertTrue(_headings(app).strip())

    def test_the_whole_chain_produces_every_artifact(self):
        app = _through_validation()

        for stage in (SUPPORTED, LIMITATIONS, PORTFOLIO, RECORD):
            app = _at(app, stage)

        held = app.session_state.filtered_state

        for artifact in ARTIFACT_FOR_STAGE.values():
            with self.subTest(artifact=artifact):
                self.assertIn(artifact, held)


class TestAnUnreachableStageSaysWhatWouldOpenIt(unittest.TestCase):
    def test_it_names_the_first_unmet_requirement_not_the_nearest(self):
        """
        Reaching the evidence record through a validation nobody ran
        would skip the decision validation exists for, so the block
        reports the earliest gap rather than the last.
        """
        app = _click(_open(), "Use this claim")
        captions = " ".join(str(item.value) for item in app.caption)

        self.assertIn("Describe the evidence to continue", captions)
        self.assertNotIn("Open portfolio context to continue", captions)

    def test_forcing_a_position_past_the_chain_renders_nothing(self):
        """
        Not a crash and not a blank screen: the workspace refuses to
        return the stage index, so no body runs, and says why.
        """
        app = _at(_click(_open(), "Use this claim"), RECORD)

        self.assertFalse(app.exception)
        self.assertEqual(_headings(app).strip(), "")

        warnings = " ".join(str(item.value) for item in app.warning)
        self.assertIn("cannot open now", warnings)

    def test_the_way_back_is_still_offered(self):
        app = _at(_click(_open(), "Use this claim"), RECORD)

        self.assertIsNotNone(app.button(key="pia_back"))


class TestDiscardingAnArtifactClosesWhatDependedOnIt(unittest.TestCase):
    """
    The Impact Evaluation crash, in a page that never hand-wrote a guard
    against it.
    """

    def test_a_stage_stops_rendering_when_its_input_goes(self):
        app = _at(_through_validation(), SUPPORTED)
        self.assertIn("Supported Claim", _headings(app))

        app.session_state["pia_validation"] = None
        app.run()

        self.assertFalse(app.exception)
        self.assertNotIn("Supported Claim", _headings(app))

    def test_it_says_what_disappeared(self):
        app = _at(_through_validation(), SUPPORTED)
        app.session_state["pia_validation"] = None
        app.run()

        warnings = " ".join(str(item.value) for item in app.warning)
        self.assertIn("Run the validation checks to continue", warnings)

    def test_the_earlier_stages_are_untouched(self):
        """
        Nothing is erased. Only what depended on the missing thing
        closes.
        """
        app = _at(_through_validation(), SUPPORTED)
        app.session_state["pia_validation"] = None
        app.run()

        held = app.session_state.filtered_state

        self.assertIn("pia_claim", held)
        self.assertIn("pia_bundle", held)
        self.assertFalse(app.button(key=f"pia_rail_{EVIDENCE}").disabled)


class TestTheEvidenceRowsAreHeldAndDisclosed(unittest.TestCase):
    """
    This journey holds a reader's evidence rows across stages, because
    every later stage describes evidence loaded earlier. That is a real
    retention and the page's disclosure has to name it.
    """

    def test_the_rows_survive_the_stage_that_read_them(self):
        app = _at(_through_validation(), RECORD)
        held = app.session_state.filtered_state

        self.assertIn("pia_evidence_frame", held)

    def test_publishing_the_bundle_does_not_skip_them(self):
        """
        publish reruns, so a plain write underneath it never executed.
        The frame went missing exactly once, this way.
        """
        app = _click(_open(), "Use this claim")
        app = _click(_at(app, EVIDENCE), "Summarize evidence")
        held = app.session_state.filtered_state

        self.assertIn("pia_bundle", held)
        self.assertIn("pia_evidence_frame", held)
        self.assertTrue(held.get("pia_evidence_filename"))

    def test_the_disclosure_names_the_retention(self):
        from shared.data_handling import disclosure_for

        notes = disclosure_for(PAGE).notes

        self.assertIn("held in session state", notes)
        self.assertIn("clears them", notes)


if __name__ == "__main__":
    unittest.main()
