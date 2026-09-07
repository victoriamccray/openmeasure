# Stage Lifecycle

The state model behind `shared/stage_workspace.py`. Written down because
three migrations each discovered a rule the previous one had not needed,
and a shared component that learns its own semantics one page at a time
becomes fragile in exactly that way.

This document is the specification. `shared/tests/test_stage_workspace.py`
tests it against a synthetic page, so the behaviour is checked at the
shared layer rather than rediscovered per migration.

## States

Six, each with a shape mark rather than a colour, so the rail is legible
without colour and a challenged stage is distinguishable from a settled
one at a glance.

| Mark | State | Means |
|---|---|---|
| `○` | Not reached | Never visited. Not stale, just unvisited. |
| `◉` | Current | On screen now. |
| `●` | Complete | Visited, and nothing has challenged it. |
| `△` | Needs review | Was complete, and a declared upstream input changed. |
| `▲` | Reviewed | Was flagged, and the reader confirmed it still applies. |
| `⊘` | Unavailable | Reached before; a prerequisite has since disappeared. |

The circle family means never challenged. The triangle family means
challenged: hollow is unanswered, filled is answered. The slash means the
stage cannot be entered right now whatever its history.

`Reviewed` is deliberately distinct from `Complete`. Complete means
nothing has questioned this stage. Reviewed means something did and a
person vouched for it anyway, which is a stronger statement and a
different provenance.

`Unavailable` is a fact about the world rather than about the reader's
progress, which is why it does not erase the review flag underneath it.
When the prerequisite returns, the stage returns to whatever it was.

## Transitions

```
Not reached --(gates ahead satisfied, reader navigates)--> Current
Current     --(reader leaves)--------------------------->  Complete
Complete    --(declared upstream input changes)---------->  Needs review
Needs review--(reader confirms "These still apply")----->  Reviewed
Reviewed    --(another declared input changes)---------->  Needs review
any reached --(a gate before or on it stops holding)---->  Unavailable
Unavailable --(the prerequisite returns)---------------->  its prior state
```

Nothing in this model erases a reader's selection. Erasing would be a
decision about their work; flagging leaves the decision to them.

## The Six Behaviours

### A prerequisite disappears

A gate makes a stage unreachable while unsatisfied, but `furthest` is
remembered independently, so a stage already reached stays on the rail
after its prerequisite goes. Impact Evaluation found this the hard way: a
reader who loaded a second dataset discarded the stored estimate while
the rail still remembered the interpretation stage, which then called
`support_boundary_claims("")` and raised into the page.

The rule: **a stage body may not assume its gate still holds.** Enforced
rather than documented. `render_rail()` returns `STAGE_UNAVAILABLE`
instead of the current index when the current stage is blocked, having
already drawn the requirement, so every `if stage == STAGE_X:` fails and
no body runs. A page that forgets to check gets a blank stage with an
explanation, not a traceback.

Navigation still renders, so the reader is never stranded. Position is
not changed for them: moving someone off a stage they chose would be a
decision about their work.

### A prerequisite appears

The mirror of the first behaviour, and the one every migration hits.

A stage produces what the next stage requires, and it does so below the
rail. The rail has already been drawn from the state as it stood before
this run, so the first time an artifact appears the stage it unlocks is
still showing as blocked, and stays that way until some unrelated
interaction redraws the page. On screen that reads as the gate not
working.

`publish(key, value)` stores the artifact and reruns when it is new.
`record_gate_input(name, value)` covers the other shape, where the value
already existed and moved: it compares digests, and a first write has no
previous digest to differ from.

Impact Evaluation hand-rolled the first with a `first_result` flag and
the second with a `gate_moved` comparison. Both are here now, because
Portfolio Impact Analysis has six such artifacts and one page's
workaround repeated six times is how shared infrastructure becomes
fragile.

### An input changes

`record_input(name, value, affects=..., label=...)` flags the stages it
names, and only those. Flagging everything downstream is the wrong
answer: rewording a research question does not invalidate a timing
decision, and a page that said it did would train a reader to dismiss the
flag.

- Only stages at or before `furthest` are flagged. An unreached stage is
  not stale.
- The flag carries the label, because "something changed" is not enough
  to act on once a study has several inputs.
- Re-recording the same value is a no-op.
- An input that already caused a flag does not add a duplicate cause.
- It reruns when it newly flags, because the rail is drawn above the
  stage that records its inputs, and without the rerun the rail would
  show the previous run's marks.

### A result disappears

Not a separate rule. This is a prerequisite disappearing, where the
prerequisite is a computed value rather than a reader's selection.

Stating it separately matters because it is the case that reads as
guaranteed and is not: a gate that tests `result is not None` is
re-evaluated on every run, and the stage it guards can be on screen from
a previous run when the answer changes.

### A reader returns to a stage

Position and `furthest` survive. Review flags survive. Kept values
survive.

Widget values do not. Streamlit discards a widget's value when the widget
is not rendered, and in a workspace that is every widget in every stage
but the current one. `keep()` and `kept()` hold the value; **restoring it
into the widget is the page's job**, because the workspace does not know a
page's widget keys:

```python
for field, remembered in entered.items():
    slot = f"pe_q_{field}"
    if remembered and slot not in st.session_state:
        st.session_state[slot] = remembered
```

Seeded into session state rather than passed as `value=`, which is the one
combination Streamlit warns about once the key exists.

A page that omits this looks like it lost the reader's work: the value is
alive in the record and gone from the form. Method Selection shipped that
way until Impact Evaluation's migration found the same shape.

### Raw data disappears

By design. An uploaded frame belongs to the stage that loaded it and is
not held between stages, because `shared/upload.py`'s promise is that
nothing retains a reader's data.

The rule: **derived results may cross a stage boundary; raw data may
not.** A summary statistic is more than a hash and less than the rows it
came from, and holding one is disclosed in `shared/data_handling.py`.

`keep()` and `publish()` both refuse a DataFrame or Series, so the policy
is a refusal rather than a comment. Neither can catch a frame hidden
inside a result object, nor a page writing to `st.session_state`
directly, which is why the rule is written here as well as enforced
there. A stage that needs the rows must obtain them again, and
should say so rather than presenting an empty uploader.

### A rerun happens

Everything the workspace holds is in session state and survives: current,
furthest, review flags, settled flags, input digests, kept values.

Two reruns are the workspace's own:

- `record_input` reruns when it newly flags, so the rail is not a run
  behind.
- `record_gate_input` reruns when a value the gates are computed from
  moves. The gates are evaluated above the stage that produces their
  inputs, so without this a reader who has just satisfied a requirement
  still sees it unmet until some unrelated interaction redraws the page.
  Impact Evaluation hand-rolled this before it moved here.

## Testing

`shared/tests/test_stage_workspace.py` drives synthetic pages through
AppTest, the same way `test_journey_stages.py` drives `StageTracker`. That
is where lifecycle behaviour is established.

Per-page tests that read `record_input` calls out of the source with `ast`
are checking configuration, not behaviour: that a page declares the
dependency graph it means to. They are cheap and they do not establish
that anything renders correctly. Where a real page can be driven, drive
it; the synthetic pages exist because a real page's `data_editor` and
`format_func` selectboxes cannot be set through AppTest, which writes the
formatted label where the page expects the key.
