# HealthRing Worked Example

A guided, single-page walkthrough of wearable/biosignal model validation: can a ring heart-rate estimate be trusted, and under what conditions?

**Status:** v0.1 prototype of OpenMeasure's immersive worked-example architecture, not a production Wearables module.

## Current features

- Explore -> Baseline -> Split -> Model -> Evaluate -> Inspect retention -> Stress-test -> Interpret -> Document, as one continuous page
- Subject-level train/test split, with the leakage risk of a window-level split stated explicitly
- One interpretable model: a single-feature linear recalibration of reference heart rate from the ring's own heart-rate estimate
- Signal-quality retention accounting that always reports how much data a quality threshold kept, alongside how performance changed
- Per-condition and per-quality-bin breakdowns, so a context-specific failure cannot hide behind an aggregate MAE
- A closing validation record (checked / learned / limitations / unresolved questions / next checks) and an explicit WIGOR mapping

## Current scope

Version 0.1 uses one ring hardware design (`ring1`) and one predictor (`bvp_hr`, the ring's own heart-rate estimate) to recalibrate against reference `hr`. It does not compare hardware designs, does not use the dataset's `Experiment` column, and does not fit more than one model.

This module does not determine whether the resulting agreement is "good enough": that depends on the intended measurement purpose, which the Interpret step asks about rather than answers.

## Data

This module never bundles HealthRing data. It reads a local copy of `RingDatasetV2.1_submission.zip` (Zenodo record 18426864) by path, at run time, and only from the page (see `pages/HealthRing_Worked_Example.py`) -- `core/` never touches a file. See `shared/datasets.py`'s `healthring` entry for the dataset's citation and access terms, and `modules/healthring/sample_data/README.md` for why no sample rows are bundled here.

## Repository structure

```
modules/healthring/
├── README.md
├── core/
│   └── acquisition_robustness.py
├── sample_data/
│   └── README.md          # no bundled data; explains why
└── tests/
    └── test_acquisition_robustness.py
```

## Planned / explicitly out of scope for v0.1

- `ring2` (the dataset's second ring hardware design)
- The `Experiment` column
- A second or competing model
- A production "Wearables" module with its own workflow-catalog entry

## References

- Bland, J. M., & Altman, D. G. (1986). *Statistical methods for assessing agreement between two methods of clinical measurement.* The Lancet, 327(8476), 307-310.
- Tang, J., Wang, K., Ding, Y., Ji, J., Wang, Y., Wang, Z., Zhang, X., Chen, P., Gao, N., Shi, Y., & Wang, Y. (2026). *HealthRing: Physiology dataset for health sensing on rings.* Scientific Data. https://doi.org/10.1038/s41597-2026-07289-x
- Puszkarski, B. (2026). *Automated analysis of wearable ECG: machine learning methods, preprocessing pipelines, and benchmark datasets.* Physiological Measurement, 47, 08TR02. https://doi.org/10.1088/1361-6579/ae8b71 (source of the WIGOR reporting checklist)
