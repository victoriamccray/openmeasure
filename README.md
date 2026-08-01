# OpenMeasure

An open-source validation toolkit for research data, measures, models, and programs.

OpenMeasure answers one question, applied to four different objects:

**Can you trust this?**

```
OpenMeasure
├── Measurement Validation
│   └── Reliability (v0.1)          ✅ available
├── Data Validation                  planned
├── Model Validation
│   └── Fairness auditing            🚧 in progress (synthetic biomedical classification data)
└── Program Validation
    └── Pre/post & treatment/control 🚧 in progress
```

- **Measurement validation**: can you trust a scale or survey to measure consistently? (Reliability)
- **Data validation**: can you trust your dataset to be clean and well-understood before analyzing it?
- **Model validation**: can you trust a model to behave fairly across groups? (Fairness auditing)
- **Program validation**: can you trust that a program caused the effect you observed? (Program evaluation)

Built for people running community health, social service, and applied
research programs who need trustworthy statistical validation without a
dedicated data science team or an expensive statistics package.

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

**Fairness auditing**: scoped, not yet built. See `modules/fairness/README.md`.

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
