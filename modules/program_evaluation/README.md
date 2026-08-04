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
  - Paired t-test (or Wilcoxon signed-rank test)
  - Effect size
  - Design-specific limitations

- **Design 2: Treatment vs. comparison group**
  - Independent-samples t-test (or Mann-Whitney U test)
  - Effect size
  - Baseline comparison

- Plain-language interpretation
- Assumptions and limitations
- Standardized reporting

## Non-goals for v0.1

- Regression-based causal inference
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
    │   └── recommend.py
    ├── tests/
    │   ├── __init__.py
    │   ├── test_comparison.py
    │   └── test_recommend.py
    └── sample_data/
```

## References

- McCray, V. P., Dukes, A. J., & Pittman, N. (in press, 2026). *Community-driven programming to strengthen scientific conference experiences: A first look at Black In Neuro and Black In Micro event outcomes.* *Oxford Open Neuroscience.*
- Shadish, W. R., Cook, T. D., & Campbell, D. T. (2002). *Experimental and Quasi-Experimental Designs for Generalized Causal Inference.*
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.).
