# OpenMeasure Decision Record, Version 0.1

**Status:** draft for comment. Nothing in the app produces this record yet.
**Schema:** [`decision_record.schema.json`](decision_record.schema.json) (JSON Schema, draft 2020-12)

## What a Decision Record Is

A Decision Record documents one research decision: the question behind it,
the evidence and method it drew on, the conditions it rests on, what the
result was, what that result supports, what it does not support, and where
the record came from.

It is written for two readers. A person reads the human-readable rendering
to see why a decision was made and how far its conclusion reaches. A program
reads the JSON to compare, archive, or check records. Both renderings come
from the same record, and neither carries information the other lacks.

The question this draft puts to researchers is: what information would you
need to evaluate a decision that this record fails to capture?

## Design Rules

1. **Modules populate the record. They do not invent their own documentation
   format.** A module that needs a field this schema lacks puts the content
   in `unplaced` and the schema is revised, rather than the module adding a
   private field.
2. **One record, one decision.** A workflow that makes three decisions
   produces three records. A record that needs two methods to explain one
   decision is a signal for the open questions below.
3. **The result comes before its reading.** `result.quantities` holds what
   was computed. `result.readings` holds threshold-based interpretations, and
   each names the convention it applies. A reading never replaces the number.
4. **`does_not_support` is required and cannot be empty.** Every result has
   conclusions it does not reach. A record that cannot name one has not been
   written carefully enough to publish.
5. **No composite verdict.** The schema has no pass/fail or overall-quality
   field, and producers do not encode one in free text. Contested questions
   stay decomposed into their parts.
6. **Defaults are choices.** A setting left at the software's default is
   recorded with `origin: "default"`, not omitted. A reader cannot tell an
   unrecorded choice from one that was never available.
7. **Absence is stated, not implied.** An empty `alternatives` list means
   none were recorded, not that none existed. A `null` quantity requires a
   `value_note` saying why it could not be computed.

## Fields

### `record_version`

Always `"0.1"` for records written against this draft. See Versioning.

### `record_id`

An identifier unique within the system that produced the record. Its format
is not specified.

### `question`

What the researcher was trying to determine, phrased as it would have been
before the result was known. "Did the two groups differ on the measured
outcome?" fits. "Why was the program effective?" does not, because it assumes
the answer.

`origin` records whether the researcher wrote the question or accepted one
the software proposed. Most module pages today imply the question from the
method chosen, so `suggestion_accepted` will be common at first, and that is
worth seeing.

### `evidence`

The data, artifacts, and literature the decision drew on. Each item has a
`kind`:

| Kind | Covers |
|---|---|
| `dataset` | Tabular or signal data that was analyzed |
| `artifact` | Anything else examined: a codebook, protocol, instrument, preregistration, figure |
| `literature` | Published work the decision relied on, beyond the method's own citation |

A `dataset` must state its `status` (`real`, `synthetic`, `constructed`, or
`unknown`), following the real-data-first principle: a reader should never
have to guess whether the numbers came from the world. A `fingerprint` lets
two records show they used the same data. OpenMeasure's existing
`DatasetFingerprint` in `shared/handoff.py` maps onto it directly.

### `decision`

The methodological or interpretive decision that was made, as a statement:
"Compare two independent groups with Welch's t-test" rather than "Welch's
t-test".

`alternatives` lists options that were available and not taken, each with
an optional `tradeoff`. This is where a record shows that the decision was a
choice between reasonable options, as the project's interpretation principle
requires of every recommendation.

`origin` records who settled it (see Origin Values).

### `method`

The analysis, check, or rule that produced the result. `parameters` holds
every setting the method ran with, defaults included. Values are JSON
primitives or flat arrays of them, so the record stays readable without the
software that produced it. `identifier` is the method's machine name inside
the producing software, for example `compare_two_groups`.

### `assumptions`

Conditions that must hold for the result to be read as intended. Each has a
`name`, a full `statement`, a `status`, and optionally a `citation` and the
`check` that examined it.

| Status | Meaning |
|---|---|
| `held_by_design` | The structure of the data guarantees it, for example the same units measured at both time points |
| `checked_consistent` | A diagnostic was run and the data are consistent with the assumption |
| `checked_inconsistent` | A diagnostic was run and the data are not consistent with it |
| `checked_inconclusive` | A diagnostic was run and could not distinguish the two |
| `not_testable_with_these_data` | It could be examined in principle, but not with this design, for example parallel trends with two time points |
| `requires_knowledge_of_setting` | No dataset answers it on its own, for example whether the comparison group received another program |
| `not_checked` | It could have been examined here and was not |

The three `checked_*` statuses require a `check` describing the diagnostic.
`checked_consistent` is deliberately not called "holds": a diagnostic that
fails to reject an assumption does not establish it.

These statuses extend the three `CHECK_*` states in
`modules/program_evaluation/core/interpret.py`, which describe what the data
*can* say about an assumption, with four that describe what was actually
done.

An empty list is valid only for purely descriptive methods. A producer that
records no assumptions says so in `result.summary`.

### `result`

What the analysis or check found, before any interpretation.

- `summary`: one or two sentences stating the finding in plain language.
- `quantities`: each computed number with its `name`, `value`, and
  optionally `unit`, `n`, and an `interval` that names its confidence
  `level` and the `method` used to compute it.
- `readings`: threshold-based interpretations. Each names the `quantity` it
  reads, the `text` of the reading, and the `convention` it applies, with a
  `citation`. "Good reliability" without a named convention is not a valid
  reading.
- `observations`: facts about this particular result worth noticing that are
  not verdicts, such as "the comparison group changed more than the treated
  group".

### `supports`

Conclusions this evidence supports, one per entry. At least one is required.
Each should be specific to the quantity computed: "The two groups differed on
this outcome at this time point" rather than "The analysis was significant".

### `does_not_support`

Conclusions a reader might plausibly draw that this evidence does not
establish. At least one is required. The most useful entries are the ones a
reader is most likely to draw by mistake, such as a causal claim from a group
comparison.

### `provenance`

Where the record came from.

- `software`: `name`, `version`, and optionally the `component` (module)
  that produced the record. When there is no release, `version` is a commit
  hash.
- `generated_at`: ISO 8601 timestamp with a UTC offset.
- `data_handling`: for each dataset in `evidence`, the columns used, rows
  received, rows retained, and every reason data was unavailable. The
  `retention` kinds (`rows_dropped`, `cells_empty`, `observations_absent`)
  are not interchangeable and are never summed. An analysis that expands rows
  sets `n_retained_rows` to `null` and reports `n_expanded_observations`.
  This mirrors `ExclusionAccount` in `shared/handoff.py`.
- `researcher_choices`: settings outside the method's own parameters that the
  result depends on, such as which column was treated as the outcome or which
  rows were filtered before analysis. Parameters already listed in
  `method.parameters` are not repeated.

### `unplaced`

Information the producer judges scientifically important that no field fits,
with a sentence on why. This field exists for v0.1 specifically: every entry
is evidence for the next revision. A schema that forces a producer to drop
something important, or to put it in the wrong field, has failed at its job,
and this is where that failure becomes visible.

## Origin Values

Used by `question`, `decision`, and `researcher_choices`.

| Value | Meaning |
|---|---|
| `researcher` | The researcher set it |
| `default` | The software's default was left in place |
| `suggestion_accepted` | The software proposed it and the researcher accepted |
| `fixed_by_method` | The method allows no other value |

## Conformance Beyond the Schema

The JSON Schema checks structure. A conforming producer also:

1. Names in each `readings[].quantity` a quantity that appears in
   `result.quantities`.
2. Names in each `data_handling[].evidence_label` a `dataset` in `evidence`.
3. Writes `supports` and `does_not_support` entries specific to this result,
   not boilerplate shared across every record a module produces.
4. Never places a verdict, ranking, or score in `supports` that the
   quantities do not compute.

Rules 1 and 2 can be checked by code and will be once a producer exists.
Rules 3 and 4 need a reader.

## Example

An illustrative record for an Impact Evaluation two-group comparison. The
numbers are invented to show the shape and do not come from a real analysis.

```json
{
  "record_version": "0.1",
  "record_id": "impact-evaluation-0001",
  "question": {
    "text": "Did participants and non-participants differ on the post-program wellbeing score?",
    "origin": "researcher"
  },
  "evidence": [
    {
      "kind": "dataset",
      "label": "program_eval_example.csv",
      "status": "constructed",
      "description": "Example dataset bundled with OpenMeasure; its origin is undocumented.",
      "fingerprint": {
        "algorithm": "sha256 of pandas row hashes, column names, and dtypes, truncated to 12 characters",
        "digest": "3f9a1c07b2e4",
        "n_rows": 200,
        "n_columns": 6
      }
    }
  ],
  "decision": {
    "statement": "Compare the two independent groups' mean outcomes with Welch's t-test.",
    "origin": "suggestion_accepted",
    "alternatives": [
      {
        "option": "Student's t-test",
        "tradeoff": "Slightly more power when group variances are equal; misleading Type I error rates when they are not and group sizes differ."
      },
      {
        "option": "Mann-Whitney U test",
        "tradeoff": "Does not assume normality, but tests a difference in distributions rather than in means."
      }
    ]
  },
  "method": {
    "name": "Welch's t-test",
    "identifier": "compare_two_groups",
    "parameters": { "alpha": 0.05, "alternative": "two-sided", "confidence_level": 0.95 },
    "citations": ["Welch, B. L. (1947). The generalization of 'Student's' problem when several different population variances are involved. Biometrika, 34(1-2), 28-35."]
  },
  "assumptions": [
    {
      "name": "Independent observations",
      "statement": "Each participant contributes one observation, and no participant's score depends on another's.",
      "status": "requires_knowledge_of_setting"
    },
    {
      "name": "Approximately normal sampling distribution of the mean",
      "statement": "With these group sizes, the difference in means is close enough to normally distributed for the interval to have its stated coverage.",
      "status": "checked_consistent",
      "check": "Shapiro-Wilk test per group, p > 0.05 in both, with group sizes above 30."
    }
  ],
  "result": {
    "summary": "Participants scored higher than non-participants on the post-program wellbeing score.",
    "quantities": [
      {
        "name": "Difference in means (participants minus non-participants)",
        "value": 4.2,
        "unit": "scale points",
        "n": 180,
        "interval": { "low": 1.1, "high": 7.3, "level": 0.95, "method": "Welch-Satterthwaite degrees of freedom" }
      },
      { "name": "p-value", "value": 0.008 }
    ],
    "readings": [
      {
        "quantity": "p-value",
        "text": "Smaller than the conventional threshold of 0.05.",
        "convention": {
          "name": "Conventional significance threshold",
          "threshold": 0.05,
          "citation": "Fisher, R. A. (1925). Statistical Methods for Research Workers."
        }
      }
    ]
  },
  "supports": [
    "The two groups differed on this outcome at this time point, by roughly 1 to 7 scale points."
  ],
  "does_not_support": [
    "That the program caused the difference. The groups were not randomly assigned and baseline scores were not compared.",
    "A claim about how the groups differed before the program."
  ],
  "provenance": {
    "software": { "name": "OpenMeasure", "version": "26440d4", "component": "Impact Evaluation" },
    "generated_at": "2026-11-19T15:04:00+00:00",
    "data_handling": [
      {
        "evidence_label": "program_eval_example.csv",
        "columns_used": ["group", "post_score"],
        "n_input_rows": 200,
        "n_retained_rows": 180,
        "retention": [
          { "label": "Missing post_score", "count": 20, "kind": "rows_dropped", "mechanism": "Listwise deletion on the outcome column" }
        ]
      }
    ],
    "researcher_choices": [
      { "choice": "Outcome column", "value": "post_score", "origin": "researcher" },
      { "choice": "Group column", "value": "group", "origin": "researcher" }
    ]
  }
}
```

## What v0.1 Leaves Out

Deliberately not in this draft: extension fields, links between records,
an interoperability API, signatures or integrity hashes over the record
itself, and a governance process for changes. Each is easier to design once
three real workflows have produced records.

## Open Questions for the Next Step

Mapping Reliability and Time-Series QA onto this schema is the test of v0.1.
These are the places it is most likely to strain:

1. **Records with no single decision.** Time-Series QA runs several checks
   over one signal. Is that one record per check, or one record whose
   `result` holds several quantities? Rule 2 says the former, which may
   fragment what a researcher thinks of as one assessment.
2. **Questions that are not about a difference.** Reliability asks how
   consistently an instrument measures. `supports` and `does_not_support`
   fit, but the wording guidance above leans on group comparisons.
3. **Decisions made before OpenMeasure.** Item selection, rater training, or
   sampling rate are decisions that shape the result and were made upstream.
   `evidence` of kind `artifact` may hold them, or they may need their own
   place.
4. **Assumptions checked by another record.** A normality check that is
   itself a decision may deserve its own record, which needs record-to-record
   links that v0.1 leaves out.

## Versioning

Until 1.0, any change to the schema raises the minor version (0.1 to 0.2)
and records state the version they were written against. A reader that does
not recognize a record's version rejects it rather than guessing, matching
how `shared/handoff.py` discards records from an older schema. Full
versioning rules come with publication.
