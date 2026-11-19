"""
Validate - checks evidence against configurable minimum-bar thresholds.

All functions are pure: they accept plain dataclasses and return frozen
dataclasses. No I/O, no UI logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from .evidence import EvidenceBundle

# An OpenMeasure default, not a number drawn from a specific source: larger
# samples support more stable group-level estimates, a general point
# discussed in Gertler, P. J., Martinez, S., Premand, P., Rawlings, L. B.,
# & Vermeersch, C. M. J. (2016). Impact Evaluation in Practice (2nd ed.).
# World Bank (see its discussion of statistical power and sample size).
# That source does not specify 30 as a threshold; this default is
# configurable per analysis.
MIN_SAMPLE_SIZE_DEFAULT = 30

# An OpenMeasure default, not a number drawn from a specific source:
# requiring a second independent source reflects the general triangulation
# principle in Patton, M. Q. (1999). Enhancing the quality and credibility
# of qualitative analysis. Health Services Research, 34(5 Pt 2), 1189-1208
# (using multiple sources or methods to strengthen credibility). That
# source does not specify 2 as a threshold; this default is configurable
# per analysis.
MIN_CORROBORATION_DEFAULT = 2

# No literature convention cited; an analyst-set default, configurable on
# the page and stated as such in the module README.
MAX_TIME_LAG_DAYS_DEFAULT = 365

CRITERION_COMPARISON_GROUP = "comparison_group"
CRITERION_SAMPLE_SIZE = "sample_size"
CRITERION_CORROBORATION = "corroboration"
CRITERION_TIME_LAG = "time_lag"

CRITERIA: tuple[str, ...] = (
    CRITERION_COMPARISON_GROUP,
    CRITERION_SAMPLE_SIZE,
    CRITERION_CORROBORATION,
    CRITERION_TIME_LAG,
)

SEVERITY_BLOCKING = "blocking"
SEVERITY_ADVISORY = "advisory"


@dataclass(frozen=True)
class ValidationCheck:
    """Whether the evidence meets one criterion, and why."""

    criterion: str
    passed: bool
    detail: str
    severity: str


@dataclass(frozen=True)
class ValidationResult:
    """
    The evidence's standing against every criterion.

    meets_minimum_bar describes the evidence, not the claim: it is true
    when every blocking check passed, regardless of the advisory checks.
    """

    claim_id: str
    checks: tuple[ValidationCheck, ...]
    n_checks_passed: int
    n_checks_total: int
    has_comparison_group: bool
    sample_size: int | None
    corroboration_count: int
    max_time_lag_days: int | None
    meets_minimum_bar: bool


def validate_evidence(
    bundle: EvidenceBundle,
    *,
    min_sample_size: int = MIN_SAMPLE_SIZE_DEFAULT,
    min_corroboration: int = MIN_CORROBORATION_DEFAULT,
    max_time_lag_days: int = MAX_TIME_LAG_DAYS_DEFAULT,
) -> ValidationResult:
    """Check an evidence bundle against configurable minimum-bar thresholds."""

    if min_sample_size <= 0:
        raise ValueError(f"min_sample_size must be positive; got {min_sample_size}.")
    if min_corroboration <= 0:
        raise ValueError(f"min_corroboration must be positive; got {min_corroboration}.")
    if max_time_lag_days <= 0:
        raise ValueError(f"max_time_lag_days must be positive; got {max_time_lag_days}.")

    sample_size = bundle.min_sample_size
    corroboration_count = bundle.n_sources
    time_lag = bundle.max_time_lag_days

    checks = (
        ValidationCheck(
            criterion=CRITERION_COMPARISON_GROUP,
            passed=bundle.has_any_comparison_group,
            detail=(
                "At least one evidence item reports a comparison group."
                if bundle.has_any_comparison_group
                else "No evidence item reports a comparison group."
            ),
            severity=SEVERITY_BLOCKING,
        ),
        ValidationCheck(
            criterion=CRITERION_SAMPLE_SIZE,
            passed=sample_size is not None and sample_size >= min_sample_size,
            detail=(
                f"Smallest reported sample size is {sample_size} "
                f"(minimum {min_sample_size})."
                if sample_size is not None
                else "No evidence item reports a sample size."
            ),
            severity=SEVERITY_ADVISORY,
        ),
        ValidationCheck(
            criterion=CRITERION_CORROBORATION,
            passed=corroboration_count >= min_corroboration,
            detail=(
                f"{corroboration_count} independent source(s) reported "
                f"(minimum {min_corroboration})."
            ),
            severity=SEVERITY_ADVISORY,
        ),
        ValidationCheck(
            criterion=CRITERION_TIME_LAG,
            passed=time_lag is not None and time_lag <= max_time_lag_days,
            detail=(
                f"Most recent evidence is {time_lag} day(s) old "
                f"(maximum {max_time_lag_days})."
                if time_lag is not None
                else "No evidence item reports a time lag."
            ),
            severity=SEVERITY_ADVISORY,
        ),
    )

    n_checks_passed = sum(1 for check in checks if check.passed)
    meets_minimum_bar = all(
        check.passed for check in checks if check.severity == SEVERITY_BLOCKING
    )

    return ValidationResult(
        claim_id=bundle.claim_id,
        checks=checks,
        n_checks_passed=n_checks_passed,
        n_checks_total=len(checks),
        has_comparison_group=bundle.has_any_comparison_group,
        sample_size=sample_size,
        corroboration_count=corroboration_count,
        max_time_lag_days=time_lag,
        meets_minimum_bar=meets_minimum_bar,
    )
