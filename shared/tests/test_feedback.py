"""
The feedback control, which is the one place a reader's words leave the app.

Run with: pytest shared/tests/test_feedback.py -v

The tests worth having here are about what travels and where it goes:
that a report carries the page and nothing from the study, that the
public route is genuinely public and says so, and that the private route
refuses rather than accepts a message it cannot deliver.
"""

from __future__ import annotations

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
        url = feedback.issue_url(context, feedback.CATEGORY_WRONG, "Ratio looks off.")

        body = _query(url)["body"]

        self.assertEqual(
            body.splitlines(),
            ["Ratio looks off.", "", "---", "Page: Fairness", "Stage: Choose a goal"],
        )

    def test_an_empty_note_says_so_rather_than_looking_truncated(self):
        url = feedback.issue_url(
            feedback.PageContext(page="Reliability"), feedback.CATEGORY_OTHER, "   "
        )

        self.assertIn("_(no note given)_", _query(url)["body"])


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

    def test_an_unknown_category_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            feedback.issue_url(
                feedback.PageContext(page="Fairness"), "Five stars", ""
            )

        self.assertIn("is not a feedback category", str(raised.exception))


class TestThePrivateRoute(unittest.TestCase):
    def test_it_refuses_when_there_is_nowhere_to_send(self):
        """
        Accepting a private message with no address would look like the
        message was received, which is worse than no form at all.
        """
        with mock.patch.object(feedback, "private_destination", return_value=""):
            with self.assertRaises(ValueError) as raised:
                feedback.mailto_url(
                    feedback.PageContext(page="Fairness"),
                    feedback.CATEGORY_WRONG,
                    "A note.",
                )

        self.assertIn("nowhere for this to go", str(raised.exception))

    def test_it_addresses_the_configured_address(self):
        with mock.patch.object(
            feedback, "_secret", return_value="someone@example.org"
        ):
            url = feedback.mailto_url(
                feedback.PageContext(page="Fairness"),
                feedback.CATEGORY_WRONG,
                "A note.",
            )

        self.assertTrue(url.startswith("mailto:someone@example.org?"))

    def test_it_carries_the_same_fields_as_the_public_route(self):
        """
        Both routes use one field builder, so a report does not depend on
        which one a reader chose.
        """
        context = feedback.PageContext(page="Fairness", stage="Choose a goal")

        with mock.patch.object(
            feedback, "_secret", return_value="someone@example.org"
        ):
            private = _query(
                feedback.mailto_url(
                    context, feedback.CATEGORY_WRONG, "Ratio looks off."
                )
            )

        public = _query(
            feedback.issue_url(context, feedback.CATEGORY_WRONG, "Ratio looks off.")
        )

        self.assertEqual(private["body"], public["body"])
        self.assertIn(public["title"], private["subject"])

    def test_no_address_is_written_into_the_source(self):
        """
        A fork or a local run must not mail this project's maintainer,
        and an address in a public repository is an address in a
        scraper's list.
        """
        source = (ROOT / "shared" / "feedback.py").read_text(encoding="utf-8")

        self.assertNotIn("@", source.replace("@dataclass", ""))
        self.assertIn("_secret(PRIVATE_ADDRESS_SETTING)", source)

    def test_nothing_is_sent_on_the_reader_s_behalf(self):
        """
        Both routes hand over something pre-filled. A module that posted
        or mailed for the reader would need a credential, a backend, and
        a reason to trust both.
        """
        source = (ROOT / "shared" / "feedback.py").read_text(encoding="utf-8")

        for outbound in ("urllib.request", "requests", "smtplib"):
            with self.subTest(mechanism=outbound):
                self.assertNotIn(outbound, source)


class TestWhatTheReaderIsTold(unittest.TestCase):
    def test_one_warning_covers_both_routes(self):
        self.assertEqual(
            feedback.PRIVACY_WARNING,
            "Please do not include participant data, private research "
            "data, or personal information.",
        )

    def test_the_public_route_says_the_words_become_public(self):
        self.assertIn("public", feedback.PUBLIC_NOTICE)
        self.assertIn("edit it before posting", feedback.PUBLIC_NOTICE)

    def test_the_private_route_says_it_is_not_posted_publicly(self):
        self.assertEqual(
            feedback.PRIVATE_NOTICE, "Private feedback is not posted publicly."
        )

    def test_both_choices_are_offered_by_name(self):
        self.assertEqual(feedback.ROUTE_PUBLIC, "Open a public GitHub issue")
        self.assertEqual(feedback.ROUTE_PRIVATE, "Send private feedback")


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
