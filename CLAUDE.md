# OpenMeasure

OpenMeasure is a Streamlit toolkit for validating research measurements, data, models, and programs. Guiding philosophy: validation over automation. The toolkit computes statistics and states assumptions, tradeoffs, and limitations; it does not make the analytical decision for the user.

## Architecture

- `modules/<name>/core/`: pure functions only. No `streamlit` import, no I/O. Every public function returns a frozen `@dataclass`, never a raw dict or tuple.
- `pages/N_Name.py`: presentation only. Imports only from `modules/<name>/core` and `shared/`. Never contains statistical logic.
- `shared/`: cross-cutting helpers used by more than one module (`report.py` for UI reporting primitives, `case_studies.py` for case-study content, `validation.py` for structural checks literally duplicated across modules, `handoff.py` for recording analysis results so they can be compared across modules). Only extract code here once it is actually duplicated; do not build ahead of need. The one exception is infrastructure that is cross-module by definition, such as `handoff.py`, which every module page writes to and which has no single-module version.
- Full layout, result-object, and reporting conventions are documented in `docs/design-standards.md`.

## Testing

- Tests live in `modules/<name>/tests/` and run per module: `pytest modules/<name>/tests/ -v`. A single root-level `pytest` invocation does not work today, because multiple modules' tests import a same-named top-level `core` package and collide with each other. See `.github/workflows/tests.yml` for how CI handles this.
- Every `core/` function needs at least one test against a hand-calculable or literature-cited value, and at least one test for a degenerate or edge case that raises a clear, typed exception. No test should only check that the code runs without crashing.

## Interpretation principle

Recommendations, such as which statistical test to use, which fairness metric fits a goal, or how to read a verdict, are guidance rather than definitive decisions. Every recommendation states its assumptions, tradeoffs, and reasonable alternatives. Every threshold-based verdict cites the convention it is drawn from rather than presenting it as settled fact. No module collapses a contested question into a single composite pass/fail score.
