# OpenMeasure

An open-source validation toolkit for research data, measures, models, and programs.

## Live Application Demo 

https://openmeasure.streamlit.app

OpenMeasure is an open-source validation toolkit for research data, measures, models, and programs.

It brings together statistical methods, transparent reporting, and plain-language interpretation to help researchers evaluate the quality of measurements, datasets, analytical models, and program evaluations.

```
OpenMeasure

Measurement Validation
    Reliability (v0.1)

Data Validation
    Data quality, completeness, and integrity

Model Validation
    Performance, subgroup evaluation, and fairness analyses

Program Validation
    Study design and program evaluation
```

- **Measurement validation** evaluates whether instruments consistently measure the intended construct and supports future assessment of additional measurement properties.

- **Data validation** evaluates the quality, completeness, consistency, and integrity of datasets before analysis.

- **Model validation** evaluates predictive performance, robustness, calibration, subgroup behavior, and fairness using transparent, documented metrics.

- **Program validation** supports evaluation of interventions using established research designs and statistical analyses appropriate to the study context.

OpenMeasure is designed for researchers and practitioners working in community health, social services, education, public policy, and applied research. The toolkit emphasizes transparent validation methods that are accessible, reproducible, and adaptable across disciplines.

### Design principles

Every module follows the same principles:

- transparent statistical methods
- reproducible analyses
- explicit assumptions and limitations
- plain-language interpretation
- documented references
- responsible use aligned with established research and professional ethics

## Quickstart

```bash
git clone https://github.com/victoriamccray/openmeasure.git
cd openmeasure
pip install -r requirements.txt
streamlit run Home.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).
Use the sidebar to open the Reliability module, currently the only module
with a working UI.

## Repository structure

```
openmeasure/
├── Home.py                     # landing page (Streamlit multipage entry point)
├── pages/
│   └── 1_Reliability.py        # Reliability module UI
├── modules/
│   ├── reliability/
│   │   ├── README.md
│   │   ├── core/                # pure statistics, fully unit tested
│   │   ├── tests/
│   │   └── sample_data/
│   ├── fairness/
│   │   └── README.md            # scope defined, not yet built
│   └── program_evaluation/
│       └── README.md            # scope defined, not yet built
├── shared/
│   └── report.py                # standardized verdicts, sections, layout
├── docs/
│   └── design-standards.md      # conventions every module must follow
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

**Program evaluation**: scoped, not yet built. See `modules/program_evaluation/README.md`.

**Data validation**: not yet scoped.

## Running tests

```bash
pip install pytest
pytest modules/reliability/tests/ -v
```

## Contributing

OpenMeasure is early-stage. If you're a researcher, clinician, or
technologist interested in contributing to a module, open an issue to
discuss scope before submitting a pull request, especially for the
fairness and program evaluation modules, where getting the framing right
matters more than getting code merged quickly.

## License

See `LICENSE.md`.
