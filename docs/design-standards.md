# OpenMeasure Design Standards

This document defines the conventions every module follows. The goal is
that switching from the Reliability module to the Fairness module to the
Program Evaluation module feels like the same tool, not three different
projects stapled together.

## 1. Code structure

Every module follows the same layout:

```
modules/<module_name>/
├── README.md          # scope, formulas, references, explicit non-goals
├── core/               # pure statistical functions, no UI code, fully unit tested
│   ├── __init__.py
│   └── <module_name>.py
├── tests/
│   ├── __init__.py
│   └── test_<module_name>.py
└── sample_data/
    └── example.csv
```

The Streamlit page for a module lives in `pages/N_ModuleName.py` at the
repo root (Streamlit's multipage convention), and imports only from
`modules/<module_name>/core` and `shared/`. UI files never contain
statistical logic, and core files never import `streamlit`.

## 2. Result objects

Every module's core `analyze()` function (or equivalent) returns a frozen
dataclass, never a raw dict or tuple. This keeps results self-documenting
and type-checkable. Every result dataclass includes, at minimum:

- Sample size(s) used
- Any cases excluded, and why
- The primary statistic(s)
- A list of per-item or per-group diagnostics, where applicable

## 3. Reporting layer

Every module's Streamlit page uses `shared/report.py` for:

- `section_header()` — consistent section dividers, never ad hoc `st.subheader()` calls
- `Band` / `classify()` / `render_verdict()` — consistent plain-language
  verdict banners, built from an explicit, documented threshold table
  (never a bare if/elif chain duplicated per module)
- `flagged_item_note()` — consistent styling for per-item/per-group flags
- `caveat()` — consistent styling for "what this does NOT tell you" statements

## 4. Report structure (the order every module follows)

1. **Dataset** — what was loaded, what was excluded, and why
2. **Headline result(s)** — the primary statistic(s), always with a
   plain-language verdict, never a bare number with no interpretation
3. **Diagnostics** — item-level, group-level, or case-level detail,
   with explicit flags where something needs review
4. **Caveats** — an explicit, un-skippable statement of what this result
   does not establish (a template, not an afterthought)

## 5. Interpretation language

- Verdicts are always framed as conventions, not laws ("Excellent
  internal consistency" is a convention from George & Mallery, 2003, not
  a fact about the world).
- Every module states its citation for any threshold it uses.
- No module produces a single composite "pass/fail" score when the
  underlying literature treats the question as genuinely contested (see
  the fairness module's explicit refusal to produce one "is this fair"
  verdict).

## 6. Testing standard

Every `core/` function has:

- At least one test against a hand-calculable or literature-cited known value
- At least one test for a degenerate/edge case (zero variance, too few
  cases, duplicate columns, etc.) that raises a clear, typed exception
- No test that merely asserts "runs without crashing" as its only check

## Methodological transparency

OpenMeasure recommends analytical methods based on the reported study design and data structure, but it does not assume there is always a single correct approach.

Where multiple established methods are reasonable, each module should:

- Recommend a default method and explain why it was selected.
- Document the assumptions required for that method.
- Explain important tradeoffs relative to common alternatives.
- Allow users to override the recommendation when appropriate.
- State how alternative methodological choices may affect interpretation.

The goal is not to prescribe a single workflow, but to support transparent, reproducible, and well-documented analyses.
