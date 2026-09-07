"""
A small, everywhere-available way to say something went wrong.

Deliberately not a rating. Stars, thumbs and NPS ask how someone feels
about a page; what is worth hearing about a research tool is which claim
looked wrong, which control was unclear, and what a reader expected
instead. So the categories are those, and "Something seems wrong" is
first among them, because a methodological error reported once is worth
more than a hundred satisfaction scores.

Two routes, and the reader chooses
----------------------------------
A public GitHub issue suits a reproducible bug or a feature request.
Private feedback suits anything that might touch unpublished research
context. Posting the second to a public tracker because the first was
easier to build would be a decision about someone else's confidentiality,
so both are offered and the difference is stated before either is used.

What is attached
----------------
The page, the stage, the version and the category. Never the research
question, the data, the selections or the results: a feedback control
that quietly shipped a study's contents somewhere would be a data
disclosure wearing a friendly icon. Whatever the reader types is the only
free text that travels, and the form says so above the box.

Where private feedback goes
---------------------------
To a second, private repository, through the same issue API as the public
route. An email inbox would need less setup, but feedback that has to be
copied out of a mailbox before it can be worked on tends to stay in the
mailbox, and the labels, the queue and the history all already live in
issues. A form service would add a third party to the path of exactly the
submissions most likely to be sensitive, which is the wrong direction.

Nothing is sent unless both the repository and a token are configured.
There is no default destination, and a form that accepted a private
message with nothing behind it would be worse than no form: it would look
like the message was received. When it is not configured the control says
so and offers the public route instead.

The public route hands the reader a pre-filled URL and lets them post it.
The private route posts on their behalf, because there is no private
equivalent of that URL, so it says what will happen before the button and
reports the issue number after it.
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
GITHUB_API = "https://api.github.com"

# Both read from st.secrets rather than hardcoded, so a fork or a local
# run sends nothing to this project's maintainer.
PRIVATE_REPOSITORY_SETTING = "private_feedback_repo"
GITHUB_TOKEN_SETTING = "github_feedback_token"

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

ROUTE_PUBLIC = "Open a public GitHub issue"
ROUTE_PRIVATE = "Send private feedback"

# Shown above both routes, before anything is typed. Above rather than
# below, because a warning under a filled-in box arrives too late.
PRIVACY_WARNING = (
    "Please do not include participant data, private research data, or "
    "personal information."
)

PUBLIC_NOTICE = (
    "This opens a pre-filled issue on OpenMeasure's public tracker. What "
    "you write there is public, and you can edit it before posting."
)

PRIVATE_NOTICE = "Private feedback is not posted publicly."

# Said when no private destination is configured, instead of accepting a
# message that would go nowhere.
PRIVATE_UNAVAILABLE = (
    "Private feedback has no destination configured in this deployment, "
    "so nothing would reach anyone. Use the public route, or contact the "
    "maintainer directly."
)

# Said before the button, since this route sends rather than pre-fills.
PRIVATE_NOTICE_DETAIL = (
    "It opens an issue in OpenMeasure's private feedback tracker, which "
    "only the maintainer can read."
)


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


def _secret(name: str) -> str:
    """One configured value, or an empty string."""
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        # No secrets file at all, which is the ordinary local case.
        return ""


def private_destination() -> str:
    """
    The private repository, as owner/name, or an empty string.

    Empty unless both halves are configured. A repository without a token
    cannot be written to, and reporting the route as available would
    promise a delivery that the submit call would then fail to make.
    """
    repository = _secret(PRIVATE_REPOSITORY_SETTING)

    return repository if repository and _secret(GITHUB_TOKEN_SETTING) else ""


def _issue_fields(context: PageContext, category: str, note: str) -> dict:
    """The title, body and label a report carries, on either route."""
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
    query = urllib.parse.urlencode(_issue_fields(context, category, note))

    return f"{REPOSITORY_URL}/issues/new?{query}"


def submit_private_feedback(
    context: PageContext, category: str, note: str
) -> int:
    """
    Open an issue in the private repository, and return its number.

    Returned rather than swallowed so the control can show the reader
    that the report arrived somewhere. A failure raises: telling someone
    their feedback was sent when the API refused it would be a lie in the
    one place this module exists to be honest about.
    """
    destination = private_destination()

    if not destination:
        raise ValueError(
            "No private feedback destination is configured, so there is "
            "nowhere for this to go."
        )

    fields = _issue_fields(context, category, note)
    payload = json.dumps(
        {
            "title": fields["title"],
            "body": fields["body"],
            "labels": [fields["labels"]],
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{GITHUB_API}/repos/{destination}/issues",
        data=payload,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {_secret(GITHUB_TOKEN_SETTING)}",
            "Content-Type": "application/json",
            "User-Agent": "OpenMeasure-feedback",
        },
    )

    with urllib.request.urlopen(
        request, timeout=REQUEST_TIMEOUT_SECONDS
    ) as response:
        created = json.loads(response.read().decode("utf-8"))

    return int(created["number"])


def render_feedback_control(page: str, *, stage: str = "", version: str | None = None) -> None:
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

    with st.popover(
        "Feedback", icon=":material/feedback:", type="tertiary"
    ):
        st.caption(PRIVACY_WARNING)

        category = st.radio(
            "What happened?",
            options=CATEGORIES,
            key=f"feedback_category_{page}",
        )

        note = st.text_area(
            "Optional note",
            key=f"feedback_note_{page}",
            placeholder="What did you expect, and what happened instead?",
        )

        route = st.radio(
            "How would you like to send this?",
            options=(ROUTE_PUBLIC, ROUTE_PRIVATE),
            key=f"feedback_route_{page}",
        )

        if route == ROUTE_PUBLIC:
            st.caption(PUBLIC_NOTICE)
            st.link_button(
                "Review and open the issue",
                issue_url(context, category, note),
            )
        elif private_destination():
            st.caption(f"{PRIVATE_NOTICE} {PRIVATE_NOTICE_DETAIL}")

            if st.button("Send privately", key=f"feedback_send_{page}"):
                try:
                    number = submit_private_feedback(context, category, note)
                except Exception as error:
                    st.error(
                        "That did not send, so it has not reached anyone: "
                        f"{error}"
                    )
                else:
                    st.success(f"Sent. It is logged as #{number}.")
        else:
            st.caption(PRIVATE_UNAVAILABLE)

        st.caption("Attached: " + "; ".join(context.as_lines()))
