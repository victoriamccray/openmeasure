# Program Validation - Impact Evaluation Module

**Category:** Program Validation
**Workflow:** Impact Evaluation
**Status:** Available (v0.1)

> This module's folder is named `program_evaluation` for historical reasons.
> The current user-facing workflow is titled "Impact Evaluation" under the
> "Program Validation" category.

## Purpose

Given data from a program or intervention, what conclusions are supported by the available evidence?

This is a **program validation** question. Reliability asks whether a measure is consistent; program validation asks whether an observed change is likely attributable to the program rather than chance, bias, or limitations of the study design.

## Scope for v0.1 (MVP)

- **Design 1: Pre/post, single group**
  - Paired t-test, shown alongside a Wilcoxon signed-rank test. The
    recommender does not check normality, so both are always computed
    and shown together, with a note on whether they agree at α=0.05.
  - Effect size (Cohen's d and matched-pairs rank-biserial correlation)
  - Design-specific limitations

- **Design 2: Treatment vs. comparison group**
  - Independent-samples (Welch's) t-test, shown alongside a
    Mann-Whitney U test. The recommender does not check normality, so
    both are always computed and shown together, with a note on
    whether they agree at α=0.05.
  - Effect size (Cohen's d and rank-biserial correlation)
  - Baseline comparison

- **Design 3: Difference-in-differences (2x2)**
  - Two groups, each measured before and after, one row per unit.
  - The estimate is the difference between the two groups' mean change
    scores, which is identical to the interaction coefficient of an OLS
    regression of the outcome on treated, post, and their interaction.
  - Inference is a Welch two-sample comparison of the change scores.
    Differencing removes each unit's own level, so the standard error is
    robust both to unequal variance between the groups and to
    correlation between a unit's two observations. It is identical to
    the HC2 robust standard error of a regression of the change score
    on the treated indicator; both equivalences are pinned in
    `tests/test_did.py` against statsmodels.
  - The assumptions a causal reading rests on are listed with the
    estimate rather than under it, each labeled with what these data can
    say about it. Parallel trends cannot be tested with two time points,
    and the module says so instead of treating a well-shaped dataset as
    evidence the assumption holds.
  - A collapsed teaching example holds the treated group's numbers fixed
    and lets the reader move the comparison group's change, showing the
    before-and-after estimate staying put while the
    difference-in-differences estimate moves with it.

## Domain context

`core/domains.py` carries seven fields of practice (public health &
healthcare, social programs, education, workforce, criminal justice &
reentry, digital interventions, and Other). A domain supplies literature
search terms, the words that field uses for each design concept, the
outcomes it commonly measures, and the measurement caveats those
outcomes carry. Several outcomes carry a caveat that travels with the
suggestion rather than sitting in page prose, so an outcome cannot be
offered anywhere its known problems are not stated: rearrest counts
police contact as well as behavior, state wage records miss
self-employment and out-of-state work, standardized scores respond to
instructional emphasis, and in-app surveys reach only the users a change
did not drive away.

**A domain never reaches the analysis.** It changes what a researcher
reads and searches for, never which method is recommended or what any
statistic comes out as. This is enforced, not just documented.
`tests/test_domains.py` asserts that `Domain` carries no method or design
field, that `recommend.py`, `comparison.py`, and `did.py` never import
`domains`, that no analysis function accepts a domain argument, and that
every domain variant of the teaching example returns an identical
estimate at every slider value.

The difference-in-differences teaching example is told in public health,
education, and workforce terms; the remaining domains fall back to the
canonical telling and the page says so. Every variant carries identical
numbers (60 to 72 treated, 58 comparison baseline) by construction, so
the field changes the story and not the arithmetic.

Search terms name fields and populations, never study designs. Seeding a
search with "randomized" or "quasi-experimental" would steer a
researcher toward literature using one design before they have decided
what their own question supports.

The selected domain is not recorded to `shared/handoff.py`.
`HandoffEntry` has no free-form metadata field, and `primary_statistics`
is typed to floats, so recording it would require a shared-schema
change.

- Plain-language interpretation
- Assumptions and limitations
- Standardized reporting

## Non-goals for v0.1

- Interrupted time series
- Staggered treatment adoption and two-way fixed effects
- Pre-trend or event-study plots with more than one baseline period
- Covariate-adjusted (conditional) parallel trends
- Cluster-robust standard errors for group-level treatment assignment
- Synthetic control, regression discontinuity, instrumental variables
- Propensity score methods
- Automated qualitative coding
- Automated causal conclusions
- Multi-arm study designs

## Interpretations

A statistically significant result does not necessarily demonstrate that a program caused an observed effect. Every analysis will include explicit statements about what the selected design can and cannot support.

## Planned output

1. Study design summary
2. Selected analysis and rationale
3. Effect estimate with confidence interval and effect size
4. Plain-language interpretation
5. Assumptions and design-specific limitations

```
modules/
└── program_evaluation/
    ├── README.md
    ├── core/
    │   ├── __init__.py
    │   ├── comparison.py
    │   ├── did.py
    │   ├── domains.py
    │   ├── interpret.py
    │   ├── recommend.py
    │   └── teaching.py
    ├── tests/
    │   ├── __init__.py
    │   ├── test_comparison.py
    │   ├── test_did.py
    │   ├── test_domains.py
    │   ├── test_interpret.py
    │   ├── test_recommend.py
    │   └── test_teaching.py
    └── sample_data/
```

## References

- McCray, V. P., Dukes, A. J., & Pittman, N. (in press, 2026). *Community-driven programming to strengthen scientific conference experiences: A first look at Black In Neuro and Black In Micro event outcomes.* *Oxford Open Neuroscience.*
- Shadish, W. R., Cook, T. D., & Campbell, D. T. (2002). *Experimental and Quasi-Experimental Designs for Generalized Causal Inference.*
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.).
- Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics, 18*(1), 50-60.
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin, 1*(6), 80-83.
- Card, D., & Krueger, A. B. (1994). Minimum wages and employment: A case study of the fast-food industry in New Jersey and Pennsylvania. *American Economic Review, 84*(4), 772-793.
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*, ch. 5.
- Abadie, A. (2005). Semiparametric difference-in-differences estimators. *Review of Economic Studies, 72*(1), 1-19.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics, 29*(3), 305-325.
