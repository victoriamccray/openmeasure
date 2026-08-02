# OpenMeasure

An open-source validation toolkit for research data, measures, models, and programs.

## Live Application

https://openmeasure.streamlit.app

OpenMeasure brings together statistical methods, transparent reporting, and plain-language interpretation to help researchers evaluate measurements, datasets, analytical models, and program evaluations.

OpenMeasure is designed for researchers and practitioners working in community health, social services, education, public policy, and applied research. The toolkit emphasizes validation methods that are accessible, reproducible, transparent, and adaptable across disciplines.

## Modules

### Measurement Validation

Evaluates the consistency and quality of research instruments and supports future assessment of additional measurement properties.

**Available:** Reliability v0.1

### Data Validation

Evaluates data quality, completeness, consistency, and integrity before analysis.

### Model Validation

Evaluates predictive performance, robustness, calibration, subgroup behavior, and fairness using transparent, documented metrics.

OpenMeasure does not prescribe a single definition of fairness. Future modules will present multiple established metrics alongside their assumptions, tradeoffs, and ethical considerations.

### Program Validation

Supports evaluation of interventions using research designs and statistical methods appropriate to the program, population, and evaluation goals.

## Design principles

Every module follows the same principles:

- Transparent statistical methods
- Reproducible analyses
- Explicit assumptions and limitations
- Plain-language interpretation
- Documented references
- Responsible use aligned with research and professional ethics

## Quickstart

```bash
git clone https://github.com/victoriamccray/openmeasure.git
cd openmeasure
pip install -r requirements.txt
streamlit run Home.py

## Repository structure

```
openmeasure/
├── Home.py
├── pages/
│   └── 1_Reliability.py
├── modules/
│   ├── reliability/
│   │   ├── README.md
│   │   ├── core/
│   │   ├── tests/
│   │   └── sample_data/
│   ├── fairness/
│   │   └── README.md
│   └── program_evaluation/
│       └── README.md
├── shared/
│   └── report.py
├── docs/
│   └── design-standards.md
└── requirements.txt
```

Each module's `README.md` documents its formulas, thresholds, references,
and explicit non-goals before any code is written, and `docs/design-standards.md`
defines the conventions every module follows so the tool feels consistent
as it grows.

## Current status

**Reliability (v0.1)**: complete. Cronbach's alpha, corrected item-total
correlations, alpha-if-item-dropped, split-half/Spearman-Brown, missing
data handling via listwise deletion, and plain-language interpretation.
24 unit tests passing. See `modules/reliability/README.md`.

**Fairness auditing**: See `modules/fairness/README.md`.

OpenMeasure does not prescribe a single definition of fairness.

Where appropriate, modules will present multiple established evaluation metrics alongside their assumptions, tradeoffs, and ethical considerations. The goal is to support transparent evaluation rather than prescribe a single framework.

## Current release
### Reliability v0.1

The first measurement-validation module includes:

- Cronbach's alpha
- Corrected item-total correlations
- Alpha if item dropped
- Odd-even split-half reliability
- Spearman-Brown correction
- Listwise missing-data handling
- Plain-language interpretation
- Assumptions and limitations
- 24 passing unit tests

See modules/reliability/README.md for methodology and references.

Future releases will expand OpenMeasure with data validation, model validation, and program evaluation modules.

## Running tests

```bash
pip install pytest
pytest modules/reliability/tests/ -v
```

## Contributing

OpenMeasure is an early-stage project. Researchers, clinicians, technologists, and evaluators are welcome to contribute.

Please open an issue before submitting a pull request so proposed changes can be discussed and aligned with the project's scope and design principles.

## License

See `LICENSE.md`.
