# Feedback Configuration

The feedback control at the foot of every page offers two routes. One
needs no setup; the other does nothing until an address is configured,
and says so rather than accepting a message it cannot deliver.

Neither route sends anything on the reader's behalf. Both hand over
something pre-filled and let the reader send it, so nothing leaves
without them seeing it first.

## Public Route

Always available. It builds a pre-filled URL for a new issue on
`victoriamccray/openmeasure`, which the reader edits and posts.

## Private Route

A pre-filled email, opened in the reader's own mail client. One secret:

```toml
# .streamlit/secrets.toml, or the Secrets box in Streamlit Cloud
feedback_email = "someone@example.org"
```

No default. With it missing, `private_destination()` returns an empty
string, the route is not offered, and the control tells the reader that
private feedback has no address in this deployment. That is deliberate: a
half-configured deployment would otherwise show a working form whose
submissions went nowhere.

The address lives in secrets rather than in the source because a fork or
a local run should not mail this project's maintainer, and an address
written into a public repository is an address in a scraper's list.

Why email rather than a private issue tracker: filing straight into a
private repository would organize the feedback better, and it would also
mean a deployed web app holding an issue-write credential plus a backend
to keep working. That is a lot of standing infrastructure for a button
that may be pressed twice a month. If private feedback ever arrives in
volume, automating it into a tracker is a small change from here.

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
