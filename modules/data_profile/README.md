# Data Profile

**Type:** Shared pre-analysis step, not a numbered validation workflow
**Status:** Available (v0.1)

## Purpose

Before a reader picks which column is an outcome, a group, or a timestamp, what does their upload actually contain? This module answers that question automatically, immediately after a file loads and before any workflow-specific setup, so a reader sees their data's shape before making choices about it rather than after.

**Use cases:**

- Summarize every column's type, missingness, and cardinality at a glance.
- Guess a column's likely role (identifier, datetime, categorical, continuous, free text) as a hint a page's own column picker can default to.
- Flag structural issues (a constant column, an entirely missing column, high missingness, duplicate rows) before they surface as a confusing result three steps later.
- On `pages/Method_Selection.py`, suggest which workflow(s) an uploaded file's column shape fits, as a second, upload-first way to reach a destination alongside the page's research-question radio.

## Scope for v0.1

- `core/profile.py` - `profile_dataframe()` profiles every column: dtype, missing count and percentage, unique-value count, and a role guess.
- `core/quality.py` - `quality_flags()` flags constant columns, all-missing columns, columns above a missingness threshold, and duplicate rows.
- `core/suggest.py` - `suggest_workflows()` matches a profile's column-role shape against four patterns (a datetime column for Time-Series QA; three or more low-cardinality numeric columns for Reliability; a categorical column next to a continuous one for Impact Evaluation; a categorical column next to a binary one for Fairness) and returns every match together, never a single best guess.

## Non-goals

- No automatic decision. Every role guess, quality flag, and workflow suggestion is a hint surfaced to the reader, never applied silently: it never drops a column, never blocks an upload, and never picks a workflow or a column for the reader without their confirmation.
- No statistical validation of the kind the numbered workflows perform (this does not check reliability, fairness, or time-series regularity; it only describes shape). Time-Series QA, Reliability, Impact Evaluation, and Fairness remain the place those specific checks happen.
- No column-role certainty claim: a role guess is a heuristic (e.g. "unique values with an ID-like name" for identifier-like), not a type system. A column can coincidentally match a heuristic without actually playing that role, and vice versa.
- No suggestion for Cross-Analysis Implications (reads already-recorded analyses, not a fresh upload) or Portfolio Impact Analysis (expects a specific, named-column evidence/claim shape this generic profiler cannot recognize from dtypes alone).
- No ranking or confidence score across multiple matching workflows: when a shape fits more than one workflow, that ambiguity is real (column dtypes cannot distinguish "was this a program effect" from "is this a fairness concern"), so every match is shown together rather than picking one.
- No cross-column relationship analysis (e.g. correlated columns, redundant columns) in v0.1.
- No claim of methodological appropriateness. This module can infer data structure; it cannot infer research intent or methodological validity from structure alone. A categorical column next to a continuous one does not establish that a group difference is causally meaningful, and three or more low-cardinality numeric columns does not establish that they form a coherent scale. Both require judgment this module has no access to.

## Role heuristics

- **identifier-like**: every non-missing value is unique, and either the column name suggests an ID (`id`, `index`, `key`, `uuid`) or the values are a sequential run of integers. Reimplements the same heuristic already used, in a narrower context, by `modules/program_evaluation/core/recommend.py`.
- **datetime-like**: the column's dtype is already a datetime type, or a small sample of its values parses as dates at a high success rate.
- **categorical-like**: a numeric column with few distinct values, or a non-numeric column whose distinct-value count is a small share of its row count.
- **continuous-like**: a numeric column with many distinct values.
- **free text**: a non-numeric column whose values are close to unique (e.g. open-ended responses), or a column with no non-missing values to judge.

These thresholds (`CATEGORICAL_MAX_UNIQUE`, `CATEGORICAL_MAX_UNIQUE_RATIO`, the datetime-parse success threshold) are OpenMeasure conventions chosen for a reasonable default, not drawn from a cited source, and are documented as constants in `core/profile.py` for that reason.

```
modules/
└── data_profile/
    ├── README.md
    ├── core/
    │   ├── __init__.py
    │   ├── profile.py
    │   ├── quality.py
    │   └── suggest.py
    └── tests/
        ├── __init__.py
        ├── test_profile.py
        ├── test_quality.py
        └── test_suggest.py
```
