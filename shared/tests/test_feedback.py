"""
The feedback control, which is the one place a reader's words leave the app.

Run with: pytest shared/tests/test_feedback.py -v

The tests worth having here are about what travels and where it goes:
that a report carries the page and nothing from the study, that the
reviewed content is the content that is sent, that the public route is
genuinely public, and that the private route refuses rather than accepts
a message it cannot deliver.
"""

from __future__ import annotations

import json
import unittest
import urllib.parse
from pathlib import Path
from unittest import mock

from shared import feedback

ROOT = Path(__file__).resolve().parents[2]


def _query(url: str) -> dict:
    return dict(urllib.parse.parse_qsl(url.split("?", 1)[1]))


class TestWhatTravels(unittest.TestCase):
    def test_a_report_carries_the_page(self):
        context = feedback.PageContext(page="Fairness")

        self.assertEqual(context.as_lines(), ("Page: Fairness",))

    def test_stage_and_version_are_named_when_known(self):
        context = feedback.PageContext(
            page="Method Selection", stage="Timing", version="5245f44a"
        )

        self.assertEqual(
            context.as_lines(),
            ("Page: Method Selection", "Stage: Timing", "Version: 5245f44a"),
        )

    def test_an_unknown_stage_is_omitted_rather_than_guessed(self):
        context = feedback.PageContext(page="Reliability", stage="")

        self.assertNotIn("Stage: ", " ".join(context.as_lines()))

    def test_the_body_holds_the_note_and_the_metadata_only(self):
        """
        A control that quietly attached the study would be a disclosure
        wearing a friendly icon, so the body is checked line by line.
        """
        context = feedback.PageContext(page="Fairness", stage="Choose a goal")
        fields = feedback.report_fields(
            context, feedback.CATEGORY_WRONG, "Ratio looks off."
        )

        self.assertEqual(
            fields["body"].splitlines(),
            ["Ratio looks off.", "", "---", "Page: Fairness", "Stage: Choose a goal"],
        )

    def test_an_empty_note_says_so_rather_than_looking_truncated(self):
        fields = feedback.report_fields(
            feedback.PageContext(page="Reliability"), feedback.CATEGORY_OTHER, "   "
        )

        self.assertIn("_(no note given)_", fields["body"])

    def test_an_unknown_category_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            feedback.report_fields(
                feedback.PageContext(page="Fairness"), "Five stars", ""
            )

        self.assertIn("is not a feedback category", str(raised.exception))


class TestTheReviewedContentIsWhatIsSent(unittest.TestCase):
    """
    The review section shows report_fields()' output, and both routes
    build from the same call, so what a reader approves is what leaves.
    """

    def _both_routes(self):
        context = feedback.PageContext(page="Fairness", stage="Choose a goal")
        reviewed = feedback.report_fields(
            context, feedback.CATEGORY_WRONG, "Ratio looks off."
        )

        public = _query(
            feedback.issue_url(context, feedback.CATEGORY_WRONG, "Ratio looks off.")
        )

        sent = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def capture(request, timeout=None):
            sent["url"] = request.full_url
            sent["payload"] = json.loads(request.data.decode("utf-8"))
            return Response()

        with mock.patch.object(
            feedback, "private_endpoint", return_value="https://example.org/f/abc"
        ), mock.patch.object(feedback.urllib.request, "urlopen", capture):
            feedback.submit_privately(
                context, feedback.CATEGORY_WRONG, "Ratio looks off."
            )

        return reviewed, public, sent

    def test_the_public_issue_is_the_reviewed_report(self):
        reviewed, public, _ = self._both_routes()

        self.assertEqual(public["title"], reviewed["title"])
        self.assertEqual(public["body"], reviewed["body"])

    def test_the_private_submission_is_the_reviewed_report(self):
        reviewed, _, sent = self._both_routes()

        self.assertEqual(sent["payload"]["subject"], reviewed["title"])
        self.assertEqual(sent["payload"]["message"], reviewed["body"])

    def test_the_private_submission_carries_no_more_than_the_review_shows(self):
        reviewed, _, sent = self._both_routes()

        self.assertEqual(
            set(sent["payload"]),
            {"subject", "message", "category", "page", "stage", "version"},
        )
        self.assertIn(sent["payload"]["page"], reviewed["body"])


class TestThePublicRoute(unittest.TestCase):
    def test_the_issue_opens_against_this_repository(self):
        url = feedback.issue_url(
            feedback.PageContext(page="Reliability"), feedback.CATEGORY_UNCLEAR, ""
        )

        self.assertTrue(url.startswith(f"{feedback.REPOSITORY_URL}/issues/new?"))

    def test_the_title_names_the_category_and_the_page(self):
        url = feedback.issue_url(
            feedback.PageContext(page="Time-Series QA"), feedback.CATEGORY_WRONG, ""
        )

        self.assertEqual(
            _query(url)["title"], "Something seems wrong: Time-Series QA"
        )

    def test_the_issue_is_labelled_so_it_lands_in_one_place(self):
        url = feedback.issue_url(
            feedback.PageContext(page="Fairness"), feedback.CATEGORY_SUGGESTION, ""
        )

        self.assertEqual(_query(url)["labels"], feedback.ISSUE_LABEL)


class TestThePrivateRoute(unittest.TestCase):
    def test_it_refuses_when_there_is_nowhere_to_send(self):
        """
        Accepting a private message with no endpoint would look like the
        message was received, which is worse than not offering it.
        """
        with mock.patch.object(feedback, "private_endpoint", return_value=""):
            with self.assertRaises(ValueError) as raised:
                feedback.submit_privately(
                    feedback.PageContext(page="Fairness"),
                    feedback.CATEGORY_WRONG,
                    "A note.",
                )

        self.assertIn("nowhere for this to go", str(raised.exception))

    def test_it_posts_to_the_configured_endpoint(self):
        sent = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def capture(request, timeout=None):
            sent["url"] = request.full_url
            sent["method"] = request.method
            return Response()

        with mock.patch.object(
            feedback, "private_endpoint", return_value="https://example.org/f/abc"
        ), mock.patch.object(feedback.urllib.request, "urlopen", capture):
            feedback.submit_privately(
                feedback.PageContext(page="Fairness"),
                feedback.CATEGORY_WRONG,
                "A note.",
            )

        self.assertEqual(sent["url"], "https://example.org/f/abc")
        self.assertEqual(sent["method"], "POST")

    def test_a_refusal_by_the_endpoint_is_raised_not_swallowed(self):
        """
        A success message over a failed POST is the one lie this module
        exists to avoid.
        """
        def refuse(request, timeout=None):
            raise OSError("502 Bad Gateway")

        with mock.patch.object(
            feedback, "private_endpoint", return_value="https://example.org/f/abc"
        ), mock.patch.object(feedback.urllib.request, "urlopen", refuse):
            with self.assertRaises(OSError):
                feedback.submit_privately(
                    feedback.PageContext(page="Fairness"),
                    feedback.CATEGORY_WRONG,
                    "A note.",
                )

    def test_no_destination_is_written_into_the_source(self):
        """
        A fork or a local run must not post to this project's endpoint,
        and an address written into public source is an address in a
        scraper's list.
        """
        source = (ROOT / "shared" / "feedback.py").read_text(encoding="utf-8")

        self.assertNotIn("@", source.replace("@dataclass", ""))
        self.assertIn("_secret(FORM_ENDPOINT_SETTING)", source)

    def test_the_public_route_does_not_depend_on_the_private_one(self):
        """
        With no endpoint, the button that could not deliver is the only
        thing that goes; Review and the public issue stay.
        """
        source = (ROOT / "shared" / "feedback.py").read_text(encoding="utf-8")

        self.assertIn("if endpoint and st.button(", source)
        self.assertIn("st.link_button(", source)


class TestWhatTheReaderIsTold(unittest.TestCase):
    def test_one_warning_covers_both_routes(self):
        self.assertEqual(
            feedback.PRIVACY_WARNING,
            "Please do not include participant data, private research "
            "data, or personal information.",
        )

    def test_the_report_is_shown_before_either_button(self):
        source = (ROOT / "shared" / "feedback.py").read_text(encoding="utf-8")

        review = source.index("REVIEW_HEADING, expanded=True")
        buttons = source.index('"Submit privately"')

        self.assertLess(review, buttons)

    def test_the_public_route_says_the_words_become_public(self):
        self.assertIn("public", feedback.PUBLIC_NOTICE)
        self.assertIn("edit it before posting", feedback.PUBLIC_NOTICE)

    def test_the_private_route_says_who_receives_it(self):
        self.assertIn("maintainer alone", feedback.PRIVATE_NOTICE)


class TestTheVersion(unittest.TestCase):
    def test_it_reads_the_checked_out_commit(self):
        version = feedback.app_version()

        self.assertRegex(version, r"^[0-9a-f]{8}$")

    def test_it_returns_nothing_rather_than_guessing(self):
        with mock.patch.object(
            feedback, "REPOSITORY_ROOT", Path("/does/not/exist")
        ):
            self.assertEqual(feedback.app_version(), "")


if __name__ == "__main__":
    unittest.main()
