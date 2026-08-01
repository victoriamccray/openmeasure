# Fairness Module (planned)

**Status: not yet built.** This document defines the scope before any code is written.

## What this module answers

Given a classifier's predictions, the true outcomes, and a group membership
column (e.g. race, sex, insurance type), does the model perform differently
across groups in ways that matter?

This is a **model validation** question, distinct from the reliability
module: reliability asks whether a *scale* is internally consistent,
fairness auditing asks whether a *model's predictions or errors* differ
systematically by group.

## Scope for v0.1 (deliberately narrow)

- **Input**: CSV with columns for predicted label (or predicted
  probability), true label, and one group column. Binary classification
  only for v0.1.
- **Metrics**:
  - Demographic parity difference (difference in positive prediction rate across groups)
  - Equalized odds difference (difference in true positive rate and false positive rate across groups)
  - Calibration by group (does a given predicted probability mean the same thing in every group)
- **Explicit non-goal for v0.1**: no multi-class, no continuous outcomes,
  no intersectional subgroups (e.g. race × sex combined), no causal claims
  about *why* a disparity exists.
- **Dataset**: a synthetic biomedical classification dataset modeled loosely
  on the structure of the Obermeyer et al. (2019) healthcare risk-score
  case, where a proxy outcome (cost) correlated with, but was not
  equivalent to, the outcome that mattered (need). This lets the module be
  validated against a known, documented real-world bias pattern instead of
  an arbitrary synthetic setup.

## Why fairness metrics are hard to package responsibly

Different fairness metrics can mathematically contradict each other
(satisfying demographic parity can violate equalized odds, and vice versa,
except in specific edge cases). This module will not recommend a single
"the model is fair" verdict. It will report multiple metrics side by side
and require the user to state which metric matters most for their specific
decision, with plain-language explanation of the tradeoff.

## Planned output shape (matches shared/report.py conventions)

1. Dataset summary (group sizes, base rates)
2. Metric-by-metric results, each with its own verdict, not one composite score
3. Explicit tradeoff explanation when metrics disagree
4. A caveat section stating what this audit does NOT establish (causal
   mechanism, legal compliance, ground-truth label quality)

## References (for when this is built)

- Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019).
  Dissecting racial bias in an algorithm used to manage the health of
  populations. *Science, 366*(6464), 447-453.
- Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine
  Learning: Limitations and Opportunities.* fairmlbook.org
