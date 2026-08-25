# Fairness

Evaluate group fairness and understand the assumptions, tradeoffs, and limitations of common fairness metrics.

**Status:** Active development (v0.1)

## Current features

- Goal-based fairness metric recommendation
- Demographic parity guidance
- Pre-model fairness analysis: disparate impact, statistical parity difference, group-level outcome summaries
- Post-model fairness analysis: equal opportunity, predictive equality, equalized odds (reported as the equal-opportunity and predictive-equality gaps together, never combined into one number), and calibration within groups (a positive-predictive-value proxy, not a full calibration curve)
- A metric-tension view that shows, using the uploaded dataset's own numbers, how the three post-model gaps can differ, so that a smaller gap on one metric is not implied to mean a smaller gap on another
- Plain-language interpretation
- Research case studies

## Current scope

Version 0.1 covers **pre-model fairness** (whether observed outcome rates differ across groups before model training) and **post-model fairness** (whether a model's predictions, given the observed outcome, differ across groups).

These analyses describe group differences but do **not** determine whether a dataset or model is fair, explain why disparities exist, establish discrimination, or rank the available metrics by importance for a given decision.

Calibration within groups is approximated with positive predictive value, a binary proxy. A full calibration analysis, comparing predicted probabilities to observed outcome frequency across score bins, needs a probability column this module does not yet collect.

## Planned features

- Fairness mitigation methods
- Threshold analysis
- Full calibration curves over predicted probabilities
- Publication-ready reporting

## Design principles

The Fairness module:

- recommends metrics based on evaluation goals;
- explains assumptions and tradeoffs;
- avoids a single "fairness score";
- encourages transparent reporting; and
- connects concepts to published research examples.

## Repository structure

```
modules/fairness/
├── README.md
├── core/
│   ├── recommend.py
│   ├── pre_model_metrics.py
│   ├── post_model_metrics.py
│   └── mitigation.py           # planned
├── sample_data/
└── tests/
```

## References

- Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning: Limitations and Opportunities.*
- Hardt, M., Price, E., & Srebro, N. (2016). *Equality of Opportunity in Supervised Learning.* Advances in Neural Information Processing Systems (NeurIPS) 29.
- Chouldechova, A. (2017). *Fair Prediction with Disparate Impact.*
- Kleinberg, J., Mullainathan, S., & Raghavan, M. (2017). *Inherent Trade-Offs in the Fair Determination of Risk Scores.*
- Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). *Dissecting Racial Bias in an Algorithm Used to Manage the Health of Populations.*