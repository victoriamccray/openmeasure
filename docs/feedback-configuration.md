# Feedback Configuration

The feedback control sits at the foot of every page. It shows the report
in full before either route is offered:

```text
Feedback -> Review -> [ Submit privately ]  [ Open a public issue ]
```

The review section renders `report_fields()`, which is the same call both
routes build from, so what a reader approves is what leaves.

## Public Route

Always available, and needs no configuration. It builds a pre-filled URL
for a new issue on `victoriamccray/openmeasure`, which the reader edits
and posts themselves. Nothing is published on anyone's behalf.

## Private Route

Posts the reviewed report to a form endpoint, from inside OpenMeasure.
The reader does not leave the app or open a mail client. One secret:

```toml
# .streamlit/secrets.toml, or the Secrets box in Streamlit Cloud
feedback_form_endpoint = "https://formspree.io/f/xxxxxxxx"
```

Any endpoint that accepts a JSON POST works. The payload is:

```json
{
  "subject": "Something seems wrong: Fairness",
  "message": "<the note, then the metadata>",
  "category": "Something seems wrong",
  "page": "Fairness",
  "stage": "Choose a goal",
  "version": "80642bf1"
}
```

The recipient address is configured at the form service, not here. It is
deliberately not in this repository: an address written into public source
is an address in a scraper's list.

With no endpoint configured, **Submit privately** is not rendered. Review
and the public route stay, so what disappears is the one button that could
not have delivered rather than the whole form.

A form endpoint rather than issue-creation in a private repository,
because the latter would mean a deployed web app holding a write
credential. That is a lot of standing infrastructure for a button that
may be pressed twice a month. Automating it into a tracker later is a
small change from here.

A failed POST raises and the control says the feedback did not send. It
never reports success over a refused request.

## What Is Attached

Page, stage, commit, and category, on both routes. Not the research
question, the uploaded data, the selections, or the results. The reader's
own note is the only free text that travels, and the form says so above
the box:

> Please do not include participant data, private research data, or
> personal information.

The review section is the proof of this rather than a promise about it,
since it displays the entire payload.

The commit comes from reading `.git/HEAD` rather than shelling out to
`git`, since a deployed process may have no git binary. Where it cannot
be read, the version line is omitted rather than guessed.
