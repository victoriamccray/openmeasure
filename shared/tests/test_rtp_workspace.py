"""
Right To Play, as a stage workspace rather than a scroll.

Run with: pytest shared/tests/test_rtp_workspace.py -v

Driven by setting the workspace's position and asserting what renders,
rather than by clicking through. go_to() reruns mid-script, which is the
established pattern in this repository and correct in a browser, and
AppTest accumulates the widgets from both passes; clicking through it
lands on stale instances. The navigation logic itself is covered at the
unit level in test_stage_workspace.py, so what is worth checking here is
the page's side of the contract: one stage in the workspace, the right
one, and the rail present to move by.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = str(ROOT / "Home.py")
LOAD_TIMEOUT_SECONDS = 60

STAGE_LABELS = ("Question", "Design", "Measurement", "Artifacts", "Boundary")


def _journey(current: int = 0, furthest: int | None = None) -> AppTest:
    app = AppTest.from_file(ENTRYPOINT, default_timeout=LOAD_TIMEOUT_SECONDS)
    app.run()
    app.switch_page("pages/Right_To_Play_Journey.py")
    app.session_state["rtp_current"] = current
    app.session_state["rtp_furthest"] = (
        current if furthest is None else furthest
    )
    app.run()
    return app


def _headers(app: AppTest) -> str:
    return " ".join(str(item.value) for item in app.subheader)


class TestOneStageAtATime(unittest.TestCase):
    """
    The whole point: five decisions were five sections to scroll past.
    """

    def test_each_stage_renders_its_own_heading(self):
        expected = (
            "Study Question",
            "Design",
            "Measurement",
            "Available Artifacts",
            "Replication Boundary",
        )

        for index, heading in enumerate(expected):
            with self.subTest(stage=index):
                app = _journey(index)
                self.assertFalse(app.exception)
                self.assertIn(heading, _headers(app))

    def test_no_other_stage_renders_alongside_it(self):
        app = _journey(0)
        headers = _headers(app)

        for heading in ("Design", "Measurement", "Available Artifacts"):
            with self.subTest(heading=heading):
                self.assertNotIn(heading, headers)

    def test_the_last_stage_does_not_render_the_first(self):
        self.assertNotIn("Study Question", _headers(_journey(4)))


class TestTheRail(unittest.TestCase):
    def test_every_stage_has_a_rail_button(self):
        app = _journey(0)
        keys = {str(button.key) for button in app.button}

        for index in range(len(STAGE_LABELS)):
            with self.subTest(stage=index):
                self.assertIn(f"rtp_rail_{index}", keys)

    def test_the_rail_marks_the_current_stage(self):
        app = _journey(2)
        label = str(app.button(key="rtp_rail_2").label)

        self.assertIn("◉", label)

    def test_a_visited_stage_is_marked_complete(self):
        app = _journey(current=0, furthest=3)

        self.assertIn("●", str(app.button(key="rtp_rail_2").label))

    def test_an_unvisited_stage_is_marked_unreached(self):
        app = _journey(current=0, furthest=0)

        self.assertIn("○", str(app.button(key="rtp_rail_4").label))


class TestNavigationAtTheFoot(unittest.TestCase):
    """
    Both directions wherever they exist, so a reader partway down a long
    stage does not return to the rail to move on.
    """

    def test_the_first_stage_offers_forward_only(self):
        app = _journey(0)
        keys = {str(button.key) for button in app.button}

        self.assertIn("rtp_forward", keys)
        self.assertNotIn("rtp_back", keys)

    def test_a_middle_stage_offers_both(self):
        app = _journey(2)
        keys = {str(button.key) for button in app.button}

        self.assertIn("rtp_forward", keys)
        self.assertIn("rtp_back", keys)

    def test_the_last_stage_offers_back_only(self):
        app = _journey(4)
        keys = {str(button.key) for button in app.button}

        self.assertIn("rtp_back", keys)
        self.assertNotIn("rtp_forward", keys)

    def test_the_buttons_name_the_stage_they_go_to(self):
        app = _journey(2)

        self.assertIn("Design", str(app.button(key="rtp_back").label))
        self.assertIn("Artifacts", str(app.button(key="rtp_forward").label))


class TestNothingIsGated(unittest.TestCase):
    """
    Every stage here is something to read. A gate would be a wizard's
    manners on a page with no use for them.
    """

    def test_no_stage_reports_a_blocking_requirement(self):
        for index in range(len(STAGE_LABELS)):
            app = _journey(current=0, furthest=index)
            with self.subTest(stage=index):
                self.assertFalse(app.button(key=f"rtp_rail_{index}").disabled)

    def test_forward_is_never_disabled(self):
        for index in range(len(STAGE_LABELS) - 1):
            app = _journey(index)
            with self.subTest(stage=index):
                self.assertFalse(app.button(key="rtp_forward").disabled)


if __name__ == "__main__":
    unittest.main()
