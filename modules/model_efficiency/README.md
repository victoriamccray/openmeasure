# Model Efficiency Tradeoff

**Type:** Research Journey core module (see `pages/GAIA_Worked_Example.py`), not a numbered validation workflow
**Status:** Available (v0.1)

## Purpose

When is a more efficient model appropriate to replace a larger one? Most model validation looks at predictive performance alone. This module makes resource use (parameter count, energy, CO2, or any other lower-is-better cost) a first-class dimension alongside performance, so a comparison can state a model's standing on both at once rather than treating efficiency as an afterthought.

**This core is domain-agnostic.** It has no notion of medical imaging, diffusion MRI, or knowledge distillation anywhere in it. GAIA (see below) is the worked example this module ships with, not the framework itself: the same `ModelProfile`/`compute_frontier`/`rank_by_preference`/`project_deployment_savings` functions apply equally to comparing models in wearables, multimodal AI, or any other setting where a set of models can each be scored on one performance metric and one resource metric.

## Scope for v0.1 (MVP)

- **`ModelProfile`** - one model's performance value and resource value (both assumed lower-is-better), with an explicit approximate/exact flag on each so a page never presents an estimated number as if it were reported exactly.
- **`compute_frontier`** - which models are Pareto-efficient (not beaten on both dimensions at once by another model in the set) versus dominated.
- **`rank_by_preference`** - a user-supplied performance-vs-resource weight, applied to min-max-normalized values within the given set, naming the result the model "favored by these weights" - never "preferred" or "recommended," since the weight is the user's choice, not this module's judgment.
- **`project_deployment_savings`** - scales an already-known per-unit resource difference to a chosen deployment count, linearly.

## Non-goals for v0.1

- No single composite "best model" score; `rank_by_preference`'s output is explicitly attributed to the weight that produced it, and a page showing it must also show what the weight was.
- No automated recommendation on which model to deploy; see the "Research decision" stage of the worked example, which leaves that judgment with the reader.
- Does not derive a per-unit resource difference from raw data - `project_deployment_savings` takes one as an input on purpose, since the worked example's own per-unit figure is a source's directly reported number, not something recomputed from unavailable absolute values.

## Interpretations

A weighted score is one way to combine two incommensurable quantities into a single number; it is only as meaningful as the weight behind it, and different reasonable weights can favor different models from the exact same evidence. This mirrors the Fairness module's refusal to reduce a contested tradeoff to one verdict (see `docs/design-standards.md`).

## The GAIA worked example

`pages/GAIA_Worked_Example.py` uses this module with real (and, in two places, real-but-approximated) data from:

> Jallais, M., Mancini, M., & Palombo, M. (2026). *GAIA - Green Artificial Intelligence for Accelerated medical imaging: Sustainable and Efficient Diffusion MRI Analysis.* Research output, Cardiff University Brain Research Imaging Centre (CUBRIC). https://doi.org/10.13140/RG.2.2.15336.12807 - an early-stage research output (e.g. a conference abstract), not confirmed as peer-reviewed.

GAIA compares three 3D U-Nets on predicting high-b-value diffusion MRI signals from lower-b-value inputs: a Teacher (depth=5, 4,118,219 parameters), a Light Model (depth=2, 52,848 parameters, trained normally with no teacher guidance - the control), and a Student (identical architecture to the Light Model, trained with a combined loss that is 75% standard loss and 25% knowledge-distillation loss matching the Teacher's latent representation).

**What is exact, and what is approximate, matters here:**

- **Exact, quoted directly from the abstract**: a 78% model-size reduction (Student vs. Teacher); a 35% decrease in total energy/CO2 (Student vs. Teacher, per subject); ~20% less CO2 per deployment (Student vs. Teacher); 0.44 kg of CO2 saved per 1,000,000 deployments (Student vs. Teacher, stated by the paper as a linear scaling); the exact parameter counts and training setup (161 WAND participants - out of WAND's 170 total - 80/10/10 split, 80 epochs, Adam lr=1e-4).
- **A discrepancy that is stated, not resolved**: the paper reports a 78% model-size reduction, but the meaning of "model size" is not specified by the provided parameter counts, which would themselves imply roughly a 98.7% reduction (52,848 vs. 4,118,219 parameters). This is presented as-is in the worked example; it is not reconciled or guessed at, pending clarification from the authors.
- **Approximate, visually read from a figure**: the abstract states no exact absolute MSE or CO2 value anywhere in its text - only in an unlabeled scatter plot (its Figure 5) and unlabeled box plots (its Figure 3). `sample_data/gaia_models.csv`'s `performance_value` and `resource_value` for all three models are estimated by reading Figure 5's plotted points against its axis gridlines, flagged `performance_is_approximate=True` / `resource_is_approximate=True`, and labeled "Approximate values read from Figure 5" everywhere they are rendered. They are used only because they are the sole source giving both a performance and a resource value for all three models at once, which any frontier chart needs; they are kept visually and narratively secondary to the exact quoted findings above, and are pending confirmation from the authors.

## Planned output

1. Which models are Pareto-efficient (`compute_frontier`), and which are strictly dominated.
2. Which model a chosen performance/resource weighting favors (`rank_by_preference`), with the weight itself always shown alongside.
3. A deployment-scale projection of an already-known per-unit resource difference.
4. A worked example (GAIA) demonstrating all of the above with a real study, explicit about which of its numbers are exact and which are estimated.

```
modules/
└── model_efficiency/
    ├── README.md
    ├── core/
    │   ├── __init__.py
    │   ├── models.py
    │   ├── frontier.py
    │   ├── preference.py
    │   └── deployment.py
    ├── tests/
    │   ├── __init__.py
    │   ├── test_models.py
    │   ├── test_frontier.py
    │   ├── test_preference.py
    │   └── test_deployment.py
    └── sample_data/
        └── gaia_models.csv
```

## References

- Jallais, M., Mancini, M., & Palombo, M. (2026). *GAIA - Green Artificial Intelligence for Accelerated medical imaging: Sustainable and Efficient Diffusion MRI Analysis.* Research output, Cardiff University Brain Research Imaging Centre (CUBRIC). https://doi.org/10.13140/RG.2.2.15336.12807 - an early-stage research output (e.g. a conference abstract), not confirmed as peer-reviewed.
- McNabb, C. B., Driver, I. D., Hyde, V. et al. (2025). WAND: A multi-modal dataset integrating advanced MRI, MEG, and TMS for multi-scale brain analysis. *Sci Data, 12*, 220. https://doi.org/10.1038/s41597-024-04154-7 (170 participants; GAIA used a 161-participant subset.)
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *NeurIPS Deep Learning Workshop.* https://doi.org/10.48550/arXiv.1503.02531
- Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2015). FitNets: Hints for thin deep nets. *International Conference on Learning Representations.* https://doi.org/10.48550/arXiv.1412.6550
- Kaack, L. H., Donti, P. L., Strubell, E., Kamiya, G., Creutzig, F., & Rolnick, D. (2022). Aligning artificial intelligence with climate change mitigation. *Nature Climate Change, 12*, 518-527. https://doi.org/10.1038/s41558-022-01377-7
- Dhar, P. (2020). The carbon impact of artificial intelligence. *Nature Machine Intelligence, 2*, 423-425. https://doi.org/10.1038/s42256-020-0219-9
