# Data Profile

**Type:** Shared pre-analysis step, not a numbered validation workflow
**Status:** Available (v0.1)

## Purpose

Before a reader picks which column is an outcome, a group, or a timestamp, what does their upload actually contain? This module answers that question automatically, immediately after a file loads and before any workflow-specific setup, so a reader sees their data's shape before making choices about it rather than after.

**Use cases:**

- Summarize every column's type, missingness, and cardinality at a glance.
- Guess a column's likely role (identifier, datetime, categorical, continuous, free text) as a hint a page's own column picker can default to.
- Flag structural issues (a constant column, an entirely missing column, high missingness, duplicate rows) before they surface as a confusing result three steps later.

## Scope for v0.1

- `core/profile.py` - `profile_dataframe()` profiles every column: dtype, missing count and percentage, unique-value count, and a role guess.
- `core/quality.py` - `quality_flags()` flags constant columns, all-missing columns, columns above a missingness threshold, and duplicate rows.

## Non-goals

- No automatic decision. Every role guess and quality flag is a hint surfaced to the reader, never applied silently: it never drops a column, never blocks an upload, and never picks a workflow's column for the reader without their confirmation.
- No statistical validation of the kind the numbered workflows perform (this does not check reliability, fairness, or time-series regularity; it only describes shape). Time-Series QA, Reliability, Impact Evaluation, and Fairness remain the place those specific checks happen.
- No column-role certainty claim: a role guess is a heuristic (e.g. "unique values with an ID-like name" for identifier-like), not a type system. A column can coincidentally match a heuristic without actually playing that role, and vice versa.
- No cross-column relationship analysis (e.g. correlated columns, redundant columns) in v0.1.

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
    │   └── quality.py
    └── tests/
        ├── __init__.py
        ├── test_profile.py
        └── test_quality.py
```
