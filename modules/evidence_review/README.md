# Evidence Review

**Type:** Numbered validation workflow (Cross-Cutting Validation)
**Status:** Available (v0.1, MVP) -- a transparent literature discovery and screening aid, not yet a systematic-review tool. See Non-goals.

## Purpose

Given a finding from any OpenMeasure workflow (or typed in directly), what does existing published literature say? This module searches a real, public bibliographic API, supports transparent screening with human-recorded decisions, and produces a structured evidence record with full source provenance, on purpose without any generative-AI call: every result is a real, dereferenceable published work, and every relevance judgment names the exact keywords that matched rather than asserting a semantic verdict.

**Use cases:**

- Compare a finding (e.g. a Reliability alpha, an Impact Evaluation effect size, a GAIA efficiency tradeoff) against what published literature reports.
- Screen search results with Include / Exclude / Uncertain decisions, structured for compatibility with PRISMA 2020 reporting (Page et al., 2021) -- PRISMA is a reporting guideline, not the source of this specific three-way label set; see Non-goals for what it actually requires and what v0.1 does not yet do.
- Produce an exportable, auditable record of the exact query run, every result returned, and every screening decision made, including the reviewer's stated reason for an exclusion.

## Scope for v0.1 (MVP)

- `core/record.py` - `LiteratureRecord`, and `from_openalex_work()` to parse a raw OpenAlex API response into that shape.
- `core/relevance.py` - `score_relevance()`, a deterministic keyword-overlap score between a finding's description and a result's title/abstract. Classic information retrieval, not a semantic or generative model; the matched keywords are always shown, never a bare confidence number.
- `core/screening.py` - `ScreeningDecision`, `ScreeningSummary`, and `summarize_screening()`, implementing the Include/Exclude/Uncertain screening convention as real typed values for the first time in this codebase (previously only an illustrative docstring example in `modules/reliability/core/interrater.py`), with an optional reason the page prompts for on Exclude.
- The actual network call to OpenAlex's public Works API lives in `pages/6_Evidence_Review.py`, not in `core/`, matching `pages/FMRI_QC_Worked_Example.py`'s existing pattern for a public-remote fetch (`urllib.request`, a custom User-Agent, a timeout, `st.cache_data`).

## Non-goals

- No generative AI. Nothing here calls an LLM to search, summarize, or rank; every step is a documented, deterministic keyword/API operation a reader could reproduce by hand.
- No claim that a keyword-overlap score measures actual relevance, only that specific words co-occur. A high overlap can still be irrelevant, and a low one can still be the single most important match; the score is a sorting aid, not a verdict, which is why the matched words are always shown.
- No causal or evidentiary grading of the literature found (no Nesta-level scoring, unlike `modules/evidence_to_claim`): this module finds and screens candidate literature, it does not grade how strong that literature's own evidence is. A later version may connect a screened-in record into `modules/evidence_to_claim`'s existing grading, but v0.1 does not.
- No persistence of the user's typed finding text: the record kept for Cross-Analysis Implications fingerprints the search *results* returned (public OpenAlex data), never the reviewer's own finding description.
- No predefined eligibility criteria (population, intervention/exposure, outcome, study design, date range, setting, etc.) before screening starts, which is what PRISMA 2020 actually calls for -- it requires a review to define and report its own eligibility criteria and selection process; it does not supply a generic Include/Exclude/Uncertain criterion itself. v0.1 lets a reviewer screen directly against the finding description with no criteria stated up front. A later version may add a criteria-definition step (Finding -> Search question -> Eligibility criteria -> Search -> Relevance sorting -> Human screening -> Exclusion reasons -> Evidence record) before this is closer to a systematic-review tool rather than a discovery/screening aid.
- No requirement that an Exclude decision states a reason (PRISMA recommends this for records that might otherwise look eligible, not as an absolute rule for every exclusion): `pages/6_Evidence_Review.py` prompts for one, and `core/screening.py`'s `ScreeningDecision.reason` carries it, but leaving it blank is accepted rather than blocked.

## Data source

[OpenAlex](https://openalex.org) (Works API), chosen for this MVP because it is fully open, requires no API key, and covers health, social science, and AI/ML research alike, matching OpenMeasure's own cross-domain scope. OpenAlex does not return plain abstract text (publisher licensing); `core/record.py` reconstructs it from OpenAlex's documented `abstract_inverted_index` format.

```
modules/
└── evidence_review/
    ├── README.md
    ├── core/
    │   ├── __init__.py
    │   ├── record.py
    │   ├── relevance.py
    │   └── screening.py
    └── tests/
        ├── __init__.py
        ├── test_record.py
        ├── test_relevance.py
        └── test_screening.py
```

## References

- Page, M. J., et al. (2021). The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. *BMJ, 372*, n71. (A reporting guideline: requires a review to define and report its own eligibility criteria and selection process, and to explain exclusions that might otherwise look eligible. Does not itself define an Include/Exclude/Uncertain label set; this module adopts that as a common, PRISMA-compatible screening convention, not a PRISMA requirement.)
- Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. *arXiv:2205.01833.*
