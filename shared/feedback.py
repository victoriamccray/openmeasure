"""
A small, everywhere-available way to say something went wrong.

Deliberately not a rating. Stars, thumbs and NPS ask how someone feels
about a page; what is worth hearing about a research tool is which claim
looked wrong, which control was unclear, and what a reader expected
instead. So the categories are those, and "Something seems wrong" is
first among them, because a methodological error reported once is worth
more than a hundred satisfaction scores.

Review, then choose where it goes
---------------------------------
The report is shown in full before either button: the note, and the page,
stage, version and category attached to it. A form that revealed what it
had collected only after sending would be asking for trust it had not
earned, and one of the two routes here publishes.

    Feedback -> Review -> [ Submit privately ] [ Open a public issue ]

Public opens a pre-filled GitHub issue for the reader to post themselves.
Private posts the same reviewed content to a form endpoint, without
leaving OpenMeasure and without anyone opening a mail client.

What travels
------------
The note, and the page, stage, version and category. Never the research
question, the data, the selections or the results: a feedback control
that quietly shipped a study's contents somewhere would be a data
disclosure wearing a friendly icon. The review section is the proof of
that rather than a promise about it, since it shows the whole payload.

Where private feedback goes
---------------------------
To a form endpoint configured in st.secrets, which forwards it to the
maintainer. The address behind that endpoint is not in this repository
and does not need to be: an address written into public source is an
address in a scraper's list.

A form endpoint rather than a private issue tracker, because the tracker
would mean a deployed web app holding a write credential, which is a lot
of standing infrastructure for a button that may be pressed twice a
month. Automating it into a tracker later is a small change from here.

With no endpoint configured, Submit privately is not offered. Review and
the public route stay, so what disappears is the one button that could
not deliver rather than the whole form apologising.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Where a staged page leaves the stage a reader is looking at, so the
# footer can name it without every page having to pass it down.
ACTIVE_STAGE_KEY = "openmeasure_active_stage"

REPOSITORY_URL = "https://github.com/victoriamccray/openmeasure"

# Read from st.secrets rather than written here, so a fork or a local run
# does not post to this project's endpoint.
FORM_ENDPOINT_SETTING = "feedback_form_endpoint"

REQUEST_TIMEOUT_SECONDS = 15

# The label a submitted issue carries, so feedback lands in one place in
# the existing queue rather than mixing with planned work.
ISSUE_LABEL = "user-feedback"

# What someone might be telling us. Ordered by how much it is worth
# hearing rather than by how often it is chosen.
CATEGORY_WRONG = "Something seems wrong"
CATEGORY_UNCLEAR = "Something was unclear"
CATEGORY_SUGGESTION = "I have a suggestion"
CATEGORY_OTHER = "Other"

CATEGORIES: tuple[str, ...] = (
    CATEGORY_WRONG,
    CATEGORY_UNCLEAR,
    CATEGORY_SUGGESTION,
    CATEGORY_OTHER,
)

# Shown above the box, before anything is typed. Above rather than below,
# because a warning under a filled-in field arrives too late.
PRIVACY_WARNING = (
    "Please do not include participant data, private research data, or "
    "personal information."
)

REVIEW_HEADING = "This is what will be sent"

PUBLIC_NOTICE = (
    "Opens a pre-filled issue on OpenMeasure's public tracker. What you "
    "write there is public, and you can edit it before posting."
)

PRIVATE_NOTICE = "Submitting privately sends this to the maintainer alone."

SENT_CONFIRMATION = "Sent. Thank you."


@dataclass(frozen=True)
class PageContext:
    """Where the reader was when they had something to say."""

    page: str
    stage: str = ""
    version: str = ""

    def as_lines(self) -> tuple[str, ...]:
        """The metadata attached to a report, and only this."""
        lines = [f"Page: {self.page}"]

        if self.stage:
            lines.append(f"Stage: {self.stage}")
        if self.version:
            lines.append(f"Version: {self.version}")

        return tuple(lines)


def app_version() -> str:
    """
    The deployed commit, short, or an empty string.

    Read out of .git rather than shelled out to git, because the deployed
    process may have no git binary and a report that silently lost its
    version would be harder to place than one that says nothing.
    """
    head = REPOSITORY_ROOT / ".git" / "HEAD"

    try:
        pointer = head.read_text(encoding="utf-8").strip()
    except OSError:
        return ""

    if pointer.startswith("ref: "):
        reference = REPOSITORY_ROOT / ".git" / pointer[len("ref: ") :]
        try:
            pointer = reference.read_text(encoding="utf-8").strip()
        except OSError:
            # A packed ref, which is not worth parsing for a footer.
            return ""

    return pointer[:8]


def _secret(name: str) -> str:
    """One configured value, or an empty string."""
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        # No secrets file at all, which is the ordinary local case.
        return ""


def private_endpoint() -> str:
    """Where private feedback posts to, or an empty string."""
    return _secret(FORM_ENDPOINT_SETTING)


def report_fields(context: PageContext, category: str, note: str) -> dict:
    """
    The whole report: what the reader wrote, and what is attached.

    One builder for both routes, so what is sent does not depend on which
    button was pressed, and so the review section can show the thing that
    actually travels rather than a description of it.
    """
    if category not in CATEGORIES:
        raise ValueError(
            f"'{category}' is not a feedback category. Known: "
            f"{', '.join(CATEGORIES)}."
        )

    return {
        "title": f"{category}: {context.page}",
        "body": "\n".join(
            [note.strip() or "_(no note given)_", "", "---", *context.as_lines()]
        ),
        "labels": ISSUE_LABEL,
    }


def issue_url(context: PageContext, category: str, note: str) -> str:
    """
    A pre-filled GitHub issue, for the reader to review and post.

    Pre-filled rather than posted. Submitting on someone's behalf would
    publish their words before they had seen how they read in public, and
    the title alone is often the part they want to change.
    """
    query = urllib.parse.urlencode(report_fields(context, category, note))

    return f"{REPOSITORY_URL}/issues/new?{query}"


def submit_privately(context: PageContext, category: str, note: str) -> None:
    """
    Post the report to the configured form endpoint.

    Raises on failure rather than returning quietly: telling someone
    their feedback was sent when the endpoint refused it would be a lie
    in the one place this module exists to be honest about.
    """
    endpoint = private_endpoint()

    if not endpoint:
        raise ValueError(
            "No private feedback endpoint is configured, so there is "
            "nowhere for this to go."
        )

    fields = report_fields(context, category, note)
    payload = json.dumps(
        {
            "subject": fields["title"],
            "message": fields["body"],
            "category": category,
            "page": context.page,
            "stage": context.stage,
            "version": context.version,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "OpenMeasure-feedback",
        },
    )

    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS):
        return


def render_feedback_control(
    page: str, *, stage: str = "", version: str | None = None
) -> None:
    """
    The control itself: one small trigger, opening one small form.

    A popover rather than a section, so a page ends on its own content
    and this is available without competing with it.
    """
    context = PageContext(
        page=page,
        stage=stage or str(st.session_state.get(ACTIVE_STAGE_KEY, "")),
        version=app_version() if version is None else version,
    )

    with st.popover("Feedback", icon=":material/feedback:", type="tertiary"):
        st.caption(PRIVACY_WARNING)

        category = st.radio(
            "What happened?",
            options=CATEGORIES,
            key=f"feedback_category_{page}",
        )

        note = st.text_area(
            "Your note",
            key=f"feedback_note_{page}",
            placeholder="What did you expect, and what happened instead?",
        )

        fields = report_fields(context, category, note)

        with st.expander(REVIEW_HEADING, expanded=True):
            st.caption(fields["title"])
            st.code(fields["body"], language=None)

        endpoint = private_endpoint()
        private_column, public_column = st.columns(2)

        with private_column:
            # Offered only where it can deliver. The public route stays
            # either way, so what disappears is the one button that would
            # not have worked.
            if endpoint and st.button(
                "Submit privately",
                key=f"feedback_submit_{page}",
                type="primary",
                width="stretch",
            ):
                try:
                    submit_privately(context, category, note)
                except Exception as error:
                    st.error(
                        "That did not send, so it has not reached anyone: "
                        f"{error}"
                    )
                else:
                    st.success(SENT_CONFIRMATION)

        with public_column:
            st.link_button(
                "Open a public issue",
                issue_url(context, category, note),
                width="stretch",
            )

        st.caption(f"{PUBLIC_NOTICE} {PRIVATE_NOTICE}" if endpoint else PUBLIC_NOTICE)
