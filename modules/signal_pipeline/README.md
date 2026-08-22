# Multimodal Signal Pipeline

**Type:** Research Journey core module (see `pages/Multimodal_Signal_Convergence.py` and `pages/GRAND_Worked_Example.py`), not a numbered validation workflow
**Status:** Available (v0.1)

## Purpose

A multimodal sensing pipeline (a neurotech BCI, a wearable, a clinical monitoring system, a neuroimaging protocol) is usually described informally: "we combine EEG with a wearable and self-report," or "we scan structural, functional, and diffusion MRI." This module makes that description a structure - `Signals -> Sensors -> Processing -> Inference -> Decision -> Action -> Feedback` - so that adding a modality is adding a node to an existing shape rather than redrawing the pipeline from scratch, and so that the question both worked examples ask is answerable in the same shape every time: *does combining modalities improve interpretation enough to justify the added privacy, security, and agency cost of collecting them?*

**This core is domain-agnostic.** It has no notion of EEG, neurotechnology, MRI, or any specific study anywhere in it. `Modality`, `build_pipeline`, `compute_convergence`, `combine_costs`, `compute_gain_cost_frontier`, and `select_necessary_feature_set` apply to any set of signal categories that can each be rated on one interpretive-gain dimension and three cost dimensions - a wearables company adding a new sensor, a hospital adding a new monitoring stream, or either worked example this module ships with. `pages/GRAND_Worked_Example.py` in particular exists to demonstrate this directly: going from illustrative EEG/ECG/EMG-style modalities to real structural/functional/diffusion MRI, a derived connectome, and behavioral measures required exactly one backward-compatible addition to this core - a `convergence_stage` parameter on `build_pipeline` (see below) - not a rewrite.

## Important distinction between this module's two worked examples

GAIA, fMRI QC, and HealthRing are each anchored to one real, cited study or dataset, with `is_approximate` flags marking exactly which numbers are estimates. `pages/GRAND_Worked_Example.py` follows that same convention - it is anchored to one real, cited, CC0-licensed dataset (GRAND; Anderson et al., 2026; OpenNeuro ds007831) - but `pages/Multimodal_Signal_Convergence.py` does not: **`sample_data/modality_profiles.csv` contains no value any single cited source reports directly.** Every `interpretive_gain`, `privacy_cost`, `security_cost`, and `agency_cost` rating in that file is an illustrative, author-assigned score *informed by* the cited literature's directional findings (e.g. "location data is highly re-identifying" per de Montjoye et al., 2013), not a number *measured by* it. `sample_data/grand_modalities.csv` is different again: its five rows' ratings are illustrative too, but informed by GRAND's own real acquisition parameters (scan duration, direction counts, data volume) and by a real re-identification finding (Schwarz et al., 2019) specific to structural MRI, not by generic domain literature. GRAND's worked example also carries a second, separate citation-integrity flag its sibling does not: a handful of quality-control figures (motion thresholds, tSNR, behavioral reliability) that this page's author could not independently verify against the published manuscript before shipping - see the page's own docstring for exactly which figures and why. Each worked example page states its own citation-integrity standard once, prominently, rather than repeating a per-field approximate flag on every row.

## Scope for v0.1 (MVP)

- **`Modality`** - one signal category's interpretive-gain rating (higher is better) and three separate cost ratings: privacy, security, agency (each higher is worse), plus its supporting citation.
- **`build_pipeline`** - assembles a `Signals -> ... -> Feedback` DAG for a set of modalities: one node per modality at every stage before `convergence_stage` (default `"Processing"`, reproducing this module's original single-shape behavior), converging into a single shared chain from `convergence_stage` onward. `pages/GRAND_Worked_Example.py` passes `convergence_stage="Inference"` so each modality keeps its own Processing node - representing separate, modality-specific preprocessing - and only converges once that processing is done.
- **`compute_convergence`** - which modalities and categories feed a given downstream stage, validated against whichever stage the pipeline actually converges at rather than a hardcoded assumption.
- **`combine_costs`** - a user-supplied privacy/security/agency weighting (normalized to sum to 1), applied to each modality's three cost ratings, following `modules/model_efficiency/core/preference.py`'s "favored by these weights" convention: the result is named after the weights, never presented as this module's own judgment. Used by `Multimodal_Signal_Convergence.py`; `GRAND_Worked_Example.py` does not use this function (see `feature_selection` below for the question it asks instead).
- **`compute_gain_cost_frontier`** - which modalities are Pareto-efficient on gain versus weighted cost, using the same dominance rule as `model_efficiency`'s performance/resource frontier (`shared/pareto.py`).
- **`feature_selection.select_necessary_feature_set`** - given a set of named feature sets each with a performance estimate and its standard error, which is best (highest performance) and which is necessary (fewest features within one standard error of best, per Hastie, Tibshirani, & Friedman's one-standard-error rule). Used by `GRAND_Worked_Example.py` to ask "does adding this modality's evidence justify collecting and integrating it," a better fit for that page's manuscript-style framing than a privacy/security/agency weighting.

## Non-goals for v0.1

- No single composite "worth it" score for any modality or combination; `compute_gain_cost_frontier` and `select_necessary_feature_set` flag dominance/necessity only, and `combine_costs`'s output is explicitly attributed to the weights that produced it.
- No automated recommendation on which modalities to collect; see each worked example's final stage, which leaves that judgment with the reader.
- `build_pipeline`'s per-modality processing is uniform once `convergence_stage` is chosen - it does not model a stage-by-stage QC gate that could stop one modality's chain early while others continue; a worked example wanting that would need to filter its own modality set stage by stage, not something this core does for it.
- Does not claim the illustrative gain/cost ratings generalize beyond either worked example; a real deployment's ratings would need to be re-derived for its own signals, population, and context.

## Interpretations

A weighted cost combination is one way to reduce three costs that aren't directly comparable (loss of privacy is not the same kind of harm as a security breach or a loss of agency) into a single number for charting purposes; it is only as meaningful as the weights behind it. This mirrors `model_efficiency`'s and the Fairness module's refusal to collapse a contested tradeoff into one verdict (see `docs/design-standards.md`).

## The worked examples

`pages/Multimodal_Signal_Convergence.py` starts the pipeline with one modality, EEG (Neural), and adds Autonomic, Muscular, Behavioral, Subjective, Environmental, and Institutional modalities one at a time, following the seven-category taxonomy the pipeline is designed to generalize to. Each addition is illustrated with the same `SignalPipeline` structure and the same gain/cost frontier, so the worked example doubles as a demonstration that the underlying engine did not need to change to add a modality.

`pages/GRAND_Worked_Example.py` runs the same engine against one real, cited neuroimaging dataset: GRAND (Anderson et al., 2026), 110 healthy older adults scanned with structural, functional, and diffusion MRI, a derived connectome, and behavioral reading/language measures collected outside the scanner, studying what each modality contributes to understanding reading and language. It starts from Structural MRI (every GRAND participant has one) and adds the other four, walking through acquisition, per-modality QC, separate processing (`build_pipeline(..., convergence_stage="Inference")`, so each modality keeps its own Processing node), alignment into derived features, integration (`compute_convergence`), evaluating added value (`feature_selection.select_necessary_feature_set`), and interpretation.

```
modules/
└── signal_pipeline/
    ├── README.md
    ├── core/
    │   ├── __init__.py
    │   ├── modality.py
    │   ├── pipeline.py
    │   ├── tradeoff.py
    │   └── feature_selection.py
    ├── tests/
    │   ├── __init__.py
    │   ├── test_modality.py
    │   ├── test_pipeline.py
    │   ├── test_tradeoff.py
    │   └── test_feature_selection.py
    └── sample_data/
        ├── modality_profiles.csv
        └── grand_modalities.csv
```

## References

Cited by `sample_data/modality_profiles.csv`, per row (see that file's `citation` column for the exact pairing):

- Ienca, M., & Andorno, R. (2017). Towards new human rights in the age of neuroscience and neurotechnology. *Life Sciences, Society and Policy, 13*, 5. https://doi.org/10.1186/s40504-017-0050-1
- Koelstra, S., Muhl, C., Soleymani, M., Lee, J.-S., Yazdani, A., Ebrahimi, T., Pun, T., Nijholt, A., & Patras, I. (2012). DEAP: A database for emotion analysis using physiological signals. *IEEE Transactions on Affective Computing, 3*(1), 18-31. https://doi.org/10.1109/T-AFFC.2011.15
- Poria, S., Cambria, E., Bajpai, R., & Hussain, A. (2017). A review of affective computing: From unimodal analysis to multimodal fusion. *Information Fusion, 37*, 98-125. https://doi.org/10.1016/j.inffus.2017.02.003
- Onnela, J.-P., & Rauch, S. L. (2016). Harnessing smartphone-based digital phenotyping to enhance behavioral and mental health. *Neuropsychopharmacology, 41*(7), 1691-1696. https://doi.org/10.1038/npp.2016.7
- Nautsch, A., Jiménez, A., Treiber, A., et al. (2019). Preserving privacy in speaker and speech characterisation. *Computer Speech & Language, 58*, 441-480. https://doi.org/10.1016/j.csl.2019.06.001
- Shiffman, S., Stone, A. A., & Hufford, M. R. (2008). Ecological momentary assessment. *Annual Review of Clinical Psychology, 4*, 1-32. https://doi.org/10.1146/annurev.clinpsy.3.022806.091415
- de Montjoye, Y.-A., Hidalgo, C. A., Verleysen, M., & Blondel, V. D. (2013). Unique in the crowd: The privacy bounds of human mobility. *Scientific Reports, 3*, 1376. https://doi.org/10.1038/srep01376
- Dey, A. K. (2001). Understanding and using context. *Personal and Ubiquitous Computing, 5*(1), 4-7. https://doi.org/10.1007/s007790170019
- Rothstein, M. A. (2010). Is deidentification sufficient to protect health privacy in research? *American Journal of Bioethics, 10*(9), 3-11. https://doi.org/10.1080/15265161.2010.494215

Cited by `sample_data/grand_modalities.csv`, per row:

- Anderson, E. J., Staples, R., Dyslin, S. M., Chang, E. H. T., Laks, A. B., Dickens, J. V., Mathur, D., Paul, S., Dvorak, E., & Turkeltaub, P. E. (2026). The Georgetown Reading in Aging Neuroimaging Dataset (GRAND) [Data set]. OpenNeuro. https://doi.org/10.18112/openneuro.ds007831.v1.0.1
- Anderson, E. J. et al. (2026). The Georgetown Reading in Aging Neuroimaging Dataset (GRAND): Reading and multimodal MRI data in older adults. *bioRxiv*. https://doi.org/10.64898/2026.05.18.725986
- Wilson, S. M., Yen, M., & Eriksson, D. K. (2018). An adaptive semantic matching paradigm for reliable and valid language mapping in individuals with aphasia. *Human Brain Mapping, 39*, 3285-3307. https://doi.org/10.1002/hbm.24077
- Schwarz, C. G., Kremers, W. K., Therneau, T. M., et al. (2019). Identification of anonymous MRI research participants with face-recognition software. *New England Journal of Medicine, 381*(17), 1684-1686. https://doi.org/10.1056/NEJMc1908881
- Nautsch, A., Jimenez, A., Treiber, A., et al. (2019). Preserving privacy in speaker and speech characterisation. *Computer Speech & Language, 58*, 441-480. https://doi.org/10.1016/j.csl.2019.06.001

Cited by `core/feature_selection.py`'s one-standard-error rule:

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. Section 7.10.
