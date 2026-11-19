"""
Best vs. necessary feature set - when adding a feature/modality stops
being worth it.

"Best" (highest observed performance) and "necessary" (the fewest
features whose performance cannot be statistically distinguished from
best) are different questions. Conflating them treats a small, possibly
noise-driven improvement as justification for the added features that
produced it. This module keeps that distinction explicit rather than
reporting only the best-performing feature set.

The "not statistically distinguishable from best" rule used here is the
one-standard-error rule (Hastie, Tibshirani, & Friedman, 2009, The
Elements of Statistical Learning, 2nd ed., Springer, section 7.10):
among feature sets whose performance is within one standard error of the
best-observed performance, prefer the one with the fewest features. It
is a convention for trading a small, uncertain performance gain against
model simplicity, not a significance test.
"""

from __future__ import annotations

from dataclasses import dataclass

ONE_STANDARD_ERROR_RULE_CITATION = (
    "Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of "
    "Statistical Learning: Data Mining, Inference, and Prediction (2nd "
    "ed.). Springer. Section 7.10 (the one-standard-error rule)."
)


@dataclass(frozen=True)
class FeatureSetResult:
    """
    One feature set's observed predictive performance.

    performance is assumed higher-is-better (e.g. a coefficient of
    determination or a Pearson correlation) and standard_error is that
    performance estimate's own standard error - both required, since the
    one-standard-error rule needs the uncertainty, not just the point
    estimate, to decide whether a smaller feature set is distinguishable
    from the best one.
    """

    name: str
    n_features: int
    performance: float
    standard_error: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name cannot be empty.")

        if self.n_features <= 0:
            raise ValueError(
                f"{self.name}: n_features must be positive; got {self.n_features}."
            )

        if self.standard_error < 0:
            raise ValueError(
                f"{self.name}: standard_error cannot be negative; got "
                f"{self.standard_error}."
            )


@dataclass(frozen=True)
class FeatureSetSelectionResult:
    """Which feature set is best, which is necessary, and why."""

    best: str
    necessary: str
    within_one_se_of_best: dict[str, bool]


def select_necessary_feature_set(
    feature_sets: tuple[FeatureSetResult, ...],
) -> FeatureSetSelectionResult:
    """
    best is the feature set with the highest performance. necessary is
    the feature set with the fewest features among those within one
    standard error of best's performance - by construction, best is
    always within one standard error of itself, so necessary always
    exists and may equal best.
    """

    if not feature_sets:
        raise ValueError("feature_sets cannot be empty.")

    names = [fs.name for fs in feature_sets]
    if len(names) != len(set(names)):
        raise ValueError("feature_sets must have unique names.")

    best_result = max(feature_sets, key=lambda fs: fs.performance)
    threshold = best_result.performance - best_result.standard_error

    within_one_se_of_best = {
        fs.name: fs.performance >= threshold for fs in feature_sets
    }
    candidates = [fs for fs in feature_sets if within_one_se_of_best[fs.name]]
    necessary_result = min(candidates, key=lambda fs: fs.n_features)

    return FeatureSetSelectionResult(
        best=best_result.name,
        necessary=necessary_result.name,
        within_one_se_of_best=within_one_se_of_best,
    )
