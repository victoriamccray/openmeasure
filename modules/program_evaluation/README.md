# Program Evaluation Module (planned)

**Status: not yet built.** This document defines the scope before any code is written.

## What this module answers

Given outcome data from a program (before/after a single group, or a
treatment group vs. a control/comparison group), is there evidence the
program had an effect, and how confident should you be in that evidence?

This is a **program validation** question: reliability asks if a measure
is consistent, this asks if an observed change is real and attributable
to the program rather than noise, regression to the mean, or a confound.

## Scope for v0.1 (deliberately narrow)

- **Design 1: Pre/post, single group**
  - Paired t-test (or Wilcoxon signed-rank if normality assumption fails)
  - Effect size: Cohen's d for paired samples
  - Explicit warning that pre/post-only designs cannot rule out maturation,
    regression to the mean, or concurrent external events as alternative
    explanations
- **Design 2: Treatment vs. control/comparison group**
  - Independent-samples t-test (or Mann-Whitney U if normality fails)
  - Effect size: Cohen's d for independent samples
  - Baseline balance check between groups (are they comparable *before*
    treatment, since imbalance undermines the comparison)
- **Explicit non-goal for v0.1**: no difference-in-differences, no
  regression adjustment/covariates, no propensity score matching, no
  multi-arm designs. These are natural v0.2 extensions once the two basic
  designs are solid and well-explained.

## Why this needs to be handled carefully

A statistically significant result from a weak design (no control group,
small sample, self-selected participants) can produce false confidence
that a program works, which is arguably worse than no evaluation, since it
can justify continued funding or scaling of something that isn't actually
effective. This module will pair every result with an explicit statement
of what the chosen design can and cannot establish, not just a p-value.

## Planned output shape (matches shared/report.py conventions)

1. Design summary (which design was detected/selected, sample sizes per group)
2. Effect estimate with confidence interval and effect size, not just p-value
3. Design-specific caveats (stated automatically based on which design was run)
4. Plain-language verdict banded by effect size and precision, not just
   "significant/not significant"

## References (for when this is built)

- Shadish, W. R., Cook, T. D., & Campbell, D. T. (2002). *Experimental and
  Quasi-Experimental Designs for Generalized Causal Inference.* Houghton Mifflin.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
  Sciences* (2nd ed.). Lawrence Erlbaum Associates.
