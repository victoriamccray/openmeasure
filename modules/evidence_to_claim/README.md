# Portfolio Impact Analysis

**Type:** Research Journey (see `pages/Portfolio_Impact_Analysis.py`), not a numbered validation workflow
**Status:** Available (v0.1)

> This module's folder is named `evidence_to_claim` for historical reasons.
> The current user-facing page is titled "Portfolio Impact Analysis" and
> lives under the app's "Research Journeys" section rather than a
> validation category: it records nothing to `shared/handoff.py` and
> carries no catalog entry, matching the pattern already established by
> the other Research Journeys.

## Purpose

Given a program, grantee, or portfolio result, what can be responsibly claimed about it, and how should that be communicated?

Program validation (Impact Evaluation) asks whether an observed change is attributable to the program, by running a comparison test. This module asks a later, different question on evidence an analyst has already assembled: what can be responsibly claimed about it, in what language, and with what limitations and comparability issues stated alongside it. It is a review layer, not a statistical test.

**Use cases:**

- Review whether reported results support a proposed claim.
- Identify limitations or evidence gaps.
- Determine which grantee results can reasonably be compared or synthesized.
- Produce concise, defensible language for leadership, board, donor, or public reporting.

## Scope for v0.1 (MVP)

- **Define claim** - state a claim (output, outcome, or impact), at a program, grantee, or portfolio level.
- **Describe evidence** - assemble evidence items (source, finding, sample size, comparison group, collection method, recency) behind that claim.
- **Validate** - check the evidence against configurable minimum-bar thresholds: comparison group present, sample size, corroboration (independent sources), and recency.
- **Determine supported claim** - grade the evidence's causal rigor against Levels 1-4 of Puttick & Ludlow's (2013) Nesta Standards of Evidence, and flag a tension when the claim's stated type expects more rigor than the evidence reaches.
- **Examine limitations** - flag every failed validation check, plus a method-bias check (all evidence sharing one collection method) that validation alone does not cover.
- **Portfolio context** - compare one grantee's indicator value against the rest of a portfolio (median, quartiles, Tukey fences), and flag indicators reported in incompatible units across grantees.
- **Evidence record** - a leadership-ready summary sentence plus a full technical-detail record.
- Accepts evidence and indicators in a row-per-observation shape close to what a grant-management or MEL system export looks like, so real data can be substituted using the same columns. v0.1 ships with a small synthetic portfolio; see `sample_data/`.

## Bringing your own data

Two CSVs can be uploaded in place of the bundled sample data (see `sample_data/` for worked examples):

- **Evidence file** - one row per piece of evidence behind a claim: `source, finding_text, indicator_id, sample_size, has_comparison_group, collection_method, time_lag_days`.
- **Portfolio file** - one row per grantee's reported result for an indicator: `grantee_id, grantee_name, indicator_id, indicator_name, result_value, unit` (extra columns are ignored). Units must match across grantees to be compared; a grantee reporting a different unit is set aside automatically rather than plotted.

## Non-goals for v0.1

- No automated go/no-go decision on the claim itself; the module states what the evidence supports, never whether to publish it.
- No causal inference beyond what is already present in the assembled evidence (this module does not run its own comparison test; see Impact Evaluation for that).
- No cross-portfolio benchmarking against other foundations' portfolios.
- Does not assess Nesta Level 5 (manualized, systematized delivery at scale), which requires operational and delivery documentation this module does not collect.
- No integration with any specific external system.

## Output, outcome, and impact

These are escalating evidentiary expectations, not synonyms (W.K. Kellogg Foundation, 2004):

- **Output** - what was delivered (e.g., number of participants served).
- **Outcome** - a measured change in participants (e.g., confidence increased), with no claim about what caused it.
- **Impact** - that change attributed to the program specifically, which conventionally requires a comparison group (Nesta Level 3+).

The claim type an analyst selects sets the level of evidence conventionally expected of it (see `core/strength.py:MIN_LEVEL_BY_CLAIM_TYPE`); a mismatch is flagged, not blocked.

## Interpretations

This module implements Levels 1-4 of Puttick & Ludlow's (2013) Nesta Standards of Evidence; Level 5 is out of scope (see Non-goals). A level describes the evidence's causal rigor - whether it demonstrates a change, and whether that change is tied to the program via a comparison group and independent replication - not whether the underlying claim is true. The mapping from claim type to a minimum expected level is an OpenMeasure convention, not part of the cited framework, and is labeled as such wherever it appears. The sample-size and corroboration-count thresholds used in the Validate stage are OpenMeasure defaults informed by the general principles in the cited sources (adequate sample size, triangulation across independent sources); the specific numbers are not drawn from those sources and are configurable per analysis.

## Planned output

1. Claim and evidence summary - what was claimed, and what evidence was assembled behind it, including any evidence items excluded and why.
2. Headline result - the Nesta level the evidence reaches, with a plain-language explanation, never a bare number.
3. Diagnostics - every validation check and limitation flag, itemized.
4. Caveats - an explicit statement of what a Nesta level does and does not establish, and (when applicable) portfolio comparability flags.

```
modules/
└── evidence_to_claim/
    ├── README.md
    ├── core/
    │   ├── __init__.py
    │   ├── claim.py
    │   ├── evidence.py
    │   ├── validate.py
    │   ├── strength.py
    │   ├── limitations.py
    │   ├── portfolio.py
    │   └── record.py
    ├── tests/
    │   ├── __init__.py
    │   ├── test_claim.py
    │   ├── test_evidence.py
    │   ├── test_validate.py
    │   ├── test_strength.py
    │   ├── test_limitations.py
    │   ├── test_portfolio.py
    │   └── test_record.py
    └── sample_data/
        ├── portfolio_indicators.csv
        ├── evidence_items.csv
        └── claims.csv
```

## References

- Puttick, R., & Ludlow, J. (2013). *Standards of Evidence: An approach that balances the need for evidence with innovation.* Nesta. (Levels 1-4 of 5 implemented; Level 5 is out of scope for v0.1.)
- Guyatt, G. H., Oxman, A. D., Vist, G. E., Kunz, R., Falck-Ytter, Y., Alonso-Coello, P., & Schünemann, H. J. (2008). GRADE: an emerging consensus on rating quality of evidence and strength of recommendations. *BMJ, 336*(7650), 924-926. (Cited as an alternative evidence-grading convention from clinical medicine, included for context; not implemented here.)
- Gertler, P. J., Martinez, S., Premand, P., Rawlings, L. B., & Vermeersch, C. M. J. (2016). *Impact Evaluation in Practice* (2nd ed.). World Bank. (General discussion of sample size and statistical power; the module's specific default threshold is an OpenMeasure convention, not drawn from this source.)
- Patton, M. Q. (1999). Enhancing the quality and credibility of qualitative analysis. *Health Services Research, 34*(5 Pt 2), 1189-1208. (General triangulation principle; the module's specific corroboration-count default is an OpenMeasure convention, not drawn from this source.)
- Tukey, J. W. (1977). *Exploratory Data Analysis.* Addison-Wesley. (Portfolio-context fences: Q1 - 1.5×IQR and Q3 + 1.5×IQR.)
- W.K. Kellogg Foundation. (2004). *Logic Model Development Guide.* W.K. Kellogg Foundation. (Output/outcome/impact claim-type vocabulary.)
