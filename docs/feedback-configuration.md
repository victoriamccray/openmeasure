# Feedback Configuration

The feedback control at the foot of every page offers two routes. One
needs no setup; the other does nothing until it is configured, and says
so rather than accepting a message it cannot deliver.

## Public Route

Always available. It builds a pre-filled URL for a new issue on
`victoriamccray/openmeasure` and hands it to the reader, who edits and
posts it themselves. Nothing is submitted on their behalf, so nothing
becomes public without them seeing it first.

## Private Route

Opens an issue in a separate private repository through the GitHub API.
Two secrets, both read from `st.secrets`:

```toml
# .streamlit/secrets.toml, or the Secrets box in Streamlit Cloud
private_feedback_repo = "victoriamccray/openmeasure-feedback"
github_feedback_token = "github_pat_..."
```

Neither has a default. With either missing, `private_destination()`
returns an empty string, the route is not offered, and the control tells
the reader that private feedback has no destination in this deployment.
That is deliberate: a half-configured deployment would show a working
form whose submissions went nowhere.

The token should be a fine-grained personal access token scoped to the
private feedback repository alone, with `Issues: Read and write` and
nothing else. It is a write credential held by a deployed web app, so the
blast radius of a leak is whatever the token can reach.

Why a private repository rather than an email inbox or a form service:
feedback that has to be copied out of a mailbox before it can be worked
on tends to stay in the mailbox, and the labels, the queue and the
history already live in issues. A form service would put a third party in
the path of exactly the submissions most likely to be sensitive.

## What Is Attached

Page, stage, commit, and category, on both routes. Not the research
question, the uploaded data, the selections, or the results. The reader's
own note is the only free text that travels, and the form says so above
the box:

> Please do not include participant data, private research data, or
> personal information.

The commit comes from reading `.git/HEAD` rather than shelling out to
`git`, since a deployed process may have no git binary. Where it cannot
be read, the version line is omitted rather than guessed.
