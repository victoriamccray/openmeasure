# Data Validation - Time-Series QA Module

**Category:** Data Validation
**Workflow:** Time-Series QA
**Status:** Available (v0.1)

## Purpose

Given a time series, is its time axis trustworthy enough, and is enough of it
populated, to support the analysis you intend to run?

This is a **data validation** question, and it comes before the questions the
other modules ask. Reliability asks whether a measure is consistent; impact
evaluation asks whether a change is attributable to a program. This module
asks something more basic: are the observations actually there, at the times
they claim to be?

## Scope for v0.1

Input is **long format, a single series**: one row per observation, one
timestamp column, one value column.

### Temporal integrity

- **Gap detection** against an expected or inferred sampling frequency
- **Duplicate timestamps**, including whether the duplicated rows conflict
- **Chronological order** of the rows as supplied
- **Interval regularity**, the share of intervals matching the dominant one

### Completeness and coverage

- **Value completeness**, rows present but empty
- **Longest unavailable run**, counting both absent and empty observations
- **Coverage per period**, measured against expected observations

### Guidance

- A recommender stating **which checks are defensible** for the series, with
  reasoning, assumptions, tradeoffs, alternatives, and limitations

## Non-goals for v0.1

These are deliberate exclusions, not omissions:

- **No cleaning of any kind.** Nothing is imputed, filled, dropped,
  deduplicated, or corrected. The module reports; it never modifies.
- **No judgment about whether a finding is an error.** A gap may be a
  closure, a planned outage, a reporting change, or nothing having happened.
  The recommender guides which checks are defensible; it does not decide
  whether a flagged observation is wrong.
- **No value anomalies**: outliers, spikes, and flatline or stuck-sensor
  detection are out of scope.
- **No level shifts or changepoints.**
- **No multi-series or panel input.** Check each series separately.
- **No sentinel-value detection.** Values such as `-999` or `NA` are counted
  as **present**, because deciding that a value stands for "missing" is a
  value-quality judgment. Recode them before uploading if they should be
  treated as missing.
- **No business-day, trading, or holiday calendars.** A detected business-day
  frequency is reported, and gap detection is declined, rather than flagging
  every weekend as a gap.
- **No localizing of naive timestamps.** Naive timestamps stay naive, which
  avoids an entire class of ambiguous and nonexistent local-time errors.
- **Mixed UTC offsets are rejected**, not converted. Converting can merge two
  distinct input values into the same instant and manufacture a duplicate
  that was not in the data.

## How gaps are detected

Expected frequency is represented as a pandas `DateOffset`, and gaps are found
by set-differencing the observed timestamps against a generated
`pd.date_range` grid. It is deliberately **not** done by comparing
consecutive differences to a fixed `Timedelta`, because:

- A `Timedelta` cannot express "one month". Months are 28 to 31 days long, so
  a fixed interval either rejects valid monthly data or miscounts it.
- Difference comparison reports a false gap at **every** daylight saving
  transition. A tz-aware daily series spans 23 or 25 absolute hours across a
  transition.

The grid approach handles monthly, quarterly, annual, leap years, and DST with
one mechanism.

Frequency inference is two-tier, because `pandas.infer_freq` returns `None` in
exactly the cases that matter most: it needs a gapless series to succeed, and
it does not recognize regular mid-month monthly data.

1. `pandas.infer_freq` on the distinct timestamps.
2. Otherwise the modal strictly-positive interval, snapped to a clean unit so
   sub-interval logging jitter does not make every difference unique, then
   promoted to a calendar offset where the spacing implies one.

Matching is tolerant: an observation logged a few seconds off its expected
time is that observation, not a missing one. Interval regularity is measured
on local wall-clock time, and is not reported at all for calendar
frequencies, whose spacing varies by design.

## Two different kinds of missing

The distinction is load-bearing and easy to conflate:

| | What it looks like | Where it is reported |
|---|---|---|
| **Absent** | No row exists for an expected time | Gaps, coverage per period |
| **Empty** | A row exists but its value is blank | Value completeness |

Coverage per period is therefore computed as non-missing values divided by
**expected** observations, never by rows present. A month where 28 of 30 days
never arrived, and the two that did have values, is 7% covered, not 100%
complete.

## Thresholds

`docs/design-standards.md` requires every threshold to state its source. For
data completeness there is an honest problem: **no canonical cross-domain
numeric standard exists.** The defaults are therefore labeled as what they
are.

Every value below is an **OpenMeasure project convention, not a published
standard.**

| Constant | Where | Default | Purpose |
|---|---|---|---|
| `DEFAULT_PERIOD_COVERAGE_THRESHOLD` | `completeness.py` | 0.90 | Share of expected observations a period must contain. User-editable in the UI. |
| `DEFAULT_MAX_MISSING_PER_MONTH` | `completeness.py` | 10 | Total-missing flag for daily data grouped by month. |
| `DEFAULT_MAX_CONSECUTIVE_MISSING` | `completeness.py` | 5 | Consecutive-run flag for daily data grouped by month. |
| `DEFAULT_JITTER_TOLERANCE_FRACTION` | `grid.py` | 0.05 | How far off schedule an observation may fall and still count as present. Operational; surfaced in the report for reproducibility. |
| `MIN_MODAL_SHARE_FOR_GRID` | `frequency.py` | 0.50 | Minimum share the dominant interval must describe before it is used to build an expected grid. Operational guard: a grid built from a minority interval produces a meaningless expected count. |
| `INTERVAL_SNAP_FRACTION` | `frequency.py` | 0.05 | Interval-snapping resolution, so logging jitter does not make every difference unique. |

Interval-regularity verdict bands live in `pages/4_Time_Series_QA.py` alongside
the other bands, not as a core constant, because they are presentation
thresholds rather than inputs to a calculation. No published standard for
interval regularity exists.

**On the 10/5 run flags.** Every threshold in this module is an OpenMeasure
project convention. **No threshold here is presented as a citable standard**,
because no verified published source has been established for any of them.

The two run flags are a starting point for reviewing whether a monthly
summary can be computed from daily observations. They are applied only to
daily data grouped by month, are not extrapolated to other frequencies, and
report only that a threshold was crossed. They do not assert that a month is
unusable, because whether it is depends entirely on what will be computed
from it.

If a domain-specific published standard applies to your work, use its
numbers rather than these defaults.

## Structure

```
modules/
└── time_series_qa/
    ├── README.md
    ├── core/
    │   ├── __init__.py
    │   ├── prepare.py        # parse, validate, sort, ingest accounting
    │   ├── frequency.py      # two-tier frequency inference
    │   ├── grid.py           # expected-grid construction and matching
    │   ├── temporal.py       # gaps, duplicates, order, regularity
    │   ├── completeness.py   # missing values, runs, coverage per period
    │   ├── recommend.py      # which checks are defensible
    │   └── qa.py             # pipeline entry point
    ├── tests/
    │   ├── __init__.py
    │   ├── test_prepare.py
    │   ├── test_frequency.py
    │   ├── test_temporal.py
    │   ├── test_completeness.py
    │   ├── test_recommend.py
    │   └── test_qa.py
    └── sample_data/
        └── time_series_example.csv
```

`core/grid.py` exists so the temporal and completeness checks cannot diverge
on what counts as present. When completeness used exact matching while
temporal matching allowed jitter, the two reports contradicted each other.

`core/qa.py` exists so the Streamlit page owns no sequencing or skip logic,
per the standard that UI files contain no statistical logic.

## References

- Kahn, M. G., Callahan, T. J., Barnard, J., Bauck, A. E., et al. (2016). A harmonized data quality assessment terminology and framework for the secondary use of electronic health record data. *eGEMs*, 4(1), 1244. Conformance, completeness, and plausibility as distinct dimensions, which is why the two check families here are reported separately rather than combined into one score.
- Wang, E., Cook, D., & Hyndman, R. J. (2020). A new tidy data structure to support exploration and modeling of time series. *Journal of Computational and Graphical Statistics*, 29(3), 466-478. Formalizes implicit versus explicit gaps and the regular versus irregular index distinction. Cited for those concepts, not for any numeric threshold.
- Madley-Dowd, P., Hughes, R., Tilling, K., & Heron, J. (2019). The proportion of missing data should not be used to guide decisions on multiple imputation. *Journal of Clinical Epidemiology*, 110, 63-73. Argues directly against fixed missingness cut-offs, and is the reason this module labels its thresholds as conventions and reports per-check rather than producing a composite score.
- Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley. The mechanism behind missingness matters more than the rate; in a time series, gaps are frequently not missing at random.
- Moritz, S., & Bartz-Beielstein, T. (2017). imputeTS: Time series missing value imputation in R. *The R Journal*, 9(1), 207-218. Established vocabulary for describing distributions of missing-value run lengths.
