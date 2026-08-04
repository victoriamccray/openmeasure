# Fairness

Evaluate group fairness and understand the assumptions, tradeoffs, and limitations of common fairness metrics.

**Status:** Active development (v0.05)

## Current features

- Goal-based fairness metric recommendation
- Demographic parity guidance
- Pre-model fairness analysis
- Disparate impact
- Statistical parity difference
- Group-level outcome summaries
- Plain-language interpretation
- Research case studies

## Current scope

Version 0.1 focuses on **pre-model fairness**, examining whether observed outcome rates differ across groups before model training.

These analyses describe group differences but do **not** determine whether a dataset or model is fair, explain why disparities exist, or establish discrimination.

## Planned features

- Equal opportunity
- Predictive equality
- Equalized odds
- Calibration within groups
- Fairness mitigation methods
- Threshold analysis
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
│   ├── post_model_metrics.py   # planned
│   └── mitigation.py           # planned
├── sample_data/
└── tests/
```

## References

- Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning: Limitations and Opportunities.*
- Chouldechova, A. (2017). *Fair Prediction with Disparate Impact.*
- Kleinberg, J., Mullainathan, S., & Raghavan, M. (2017). *Inherent Trade-Offs in the Fair Determination of Risk Scores.*
- Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). *Dissecting Racial Bias in an Algorithm Used to Manage the Health of Populations.*