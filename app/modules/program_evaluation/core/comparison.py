"""
Core program evaluation statistics.

All functions are pure: they accept pandas data structures and return
frozen dataclasses. No I/O, no UI logic. Follows the same design pattern
as modules/reliability/core/reliability.py.

Test selection follows McCray, Dukes, & Pittman (in press), Oxford Open
Neuroscience: Welch's t-test for two groups, Welch's one-way ANOVA with
Games-Howell comparisons for three or more groups, chi-square for
categorical outcomes, and a sensitivity-analysis pattern (using standard
ANOVA + Tukey HSD per coding) for multi-select categorical predictors.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.validation import validate_is_dataframe


MIN_GROUP_SIZE = 5  # below this, warn rather than compute an unstable estimate

# Why rows were dropped, recorded on every result per
# docs/design-standards.md section 2.
MISSING_GROUP_OR_OUTCOME = "missing group or outcome value"
INCOMPLETE_PAIR = "incomplete pre/post pair"


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TwoGroupResult:
    """Welch's t-test comparing two groups on a continuous outcome."""

    group_a_label: str
    group_b_label: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    sd_a: float
    sd_b: float
    t_statistic: float
    degrees_of_freedom: float
    p_value: float
    cohens_d: float
    mean_difference: float
    ci_95_low: float
    ci_95_high: float

    # Exclusion accounting, per docs/design-standards.md section 2. Rows
    # missing either the group or the outcome cannot be assigned to a group,
    # so they are dropped before any statistic is computed.
    n_input_rows: int
    n_rows_used: int
    n_excluded_rows: int
    exclusion_reason: str


@dataclass(frozen=True)
class TwoGroupNonparametricResult:
    """Mann-Whitney U test comparing two groups on an ordinal or non-normal outcome."""

    group_a_label: str
    group_b_label: str
    n_a: int
    n_b: int
    median_a: float
    median_b: float
    u_statistic: float
    p_value: float
    rank_biserial_correlation: float

    n_input_rows: int
    n_rows_used: int
    n_excluded_rows: int
    exclusion_reason: str


@dataclass(frozen=True)
class PairwiseComparison:
    """One pairwise post-hoc comparison from Tukey HSD."""

    group_a: str
    group_b: str
    mean_difference: float
    p_value: float
    significant: bool


@dataclass(frozen=True)
class MultiGroupResult:
    """One-way ANOVA with Tukey HSD post-hoc, for 3+ groups."""

    group_labels: list[str]
    group_ns: dict[str, int]
    group_means: dict[str, float]
    f_statistic: float
    df_between: int
    df_within: int
    p_value: float
    pairwise_comparisons: list[PairwiseComparison]
    small_groups_flagged: list[str]

    n_input_rows: int
    n_rows_used: int
    n_excluded_rows: int
    exclusion_reason: str


@dataclass(frozen=True)
class MultiGroupWelchResult:
    """
    Welch's one-way ANOVA with Games-Howell post-hoc, for 3+ groups.

    Unlike standard ANOVA, neither the omnibus test nor the post-hoc
    comparisons assume equal variances across groups, matching the same
    philosophy as using Welch's t-test (rather than Student's) for the
    2-group case.
    """

    group_labels: list[str]
    group_ns: dict[str, int]
    group_means: dict[str, float]
    f_statistic: float
    df_between: float
    df_within: float  # fractional (Welch-Satterthwaite), unlike standard ANOVA's integer df
    p_value: float
    pairwise_comparisons: list[PairwiseComparison]
    small_groups_flagged: list[str]

    n_input_rows: int
    n_rows_used: int
    n_excluded_rows: int
    exclusion_reason: str


@dataclass(frozen=True)
class ChiSquareResult:
    """Chi-square test of independence for a categorical outcome."""

    chi2_statistic: float
    degrees_of_freedom: int
    p_value: float
    contingency_table: pd.DataFrame
    expected_frequencies: pd.DataFrame
    low_expected_frequency_warning: bool

    n_input_rows: int
    n_rows_used: int
    n_excluded_rows: int
    exclusion_reason: str


@dataclass(frozen=True)
class PairedResult:
    """Paired t-test for a single-group pre/post comparison."""

    n: int
    mean_pre: float
    mean_post: float
    mean_difference: float
    t_statistic: float
    degrees_of_freedom: int
    p_value: float
    cohens_d: float

    n_input_rows: int
    n_rows_used: int
    n_excluded_rows: int
    exclusion_reason: str


@dataclass(frozen=True)
class PairedNonparametricResult:
    """Wilcoxon signed-rank test for a single-group pre/post comparison."""

    n: int
    median_pre: float
    median_post: float
    median_difference: float
    w_statistic: float
    p_value: float
    matched_pairs_rank_biserial_correlation: float

    n_input_rows: int
    n_rows_used: int
    n_excluded_rows: int
    exclusion_reason: str

    # Real, observed pairs with no change. Reported separately rather than
    # folded into n_excluded_rows: a zero difference is complete data, not
    # a missing value, and scipy's Wilcoxon convention drops it from
    # ranking (Wilcoxon, 1945), so it must still be accounted for somewhere.
    n_zero_differences_dropped: int


@dataclass(frozen=True)
class SensitivityResult:
    """
    Result of re-running a group comparison under multiple codings of a
    multi-select categorical predictor (e.g. participants who could select
    more than one demographic category).
    """

    coding_results: dict[str, "MultiGroupResult | TwoGroupResult"]
    p_values_by_coding: dict[str, float]
    consistent_conclusion: bool
    alpha: float

    # Participants received, before any coding is applied.
    n_input_rows: int

    # Observations under the expanded coding, which emits one row per
    # selection. A participant who selected two categories contributes two
    # observations, so this legitimately exceeds n_input_rows and is not a
    # count of retained participants. It must be reported as expanded
    # observations rather than compared against other analyses' retention.
    n_expanded_rows: int

    # Each coding's own result carries its own row accounting.
    rows_can_exceed_participants: bool = True


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_two_columns(data: pd.DataFrame, group_col: str, outcome_col: str) -> None:
    validate_is_dataframe(data)
    if group_col not in data.columns:
        raise ValueError(f"Group column '{group_col}' not found in data.")
    if outcome_col not in data.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in data.")


def _cohens_d_independent(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for independent samples, using pooled standard deviation."""
    n_a, n_b = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_sd == 0:
        raise ValueError("Pooled standard deviation is zero; Cohen's d is undefined.")
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def _cohens_d_paired(differences: np.ndarray) -> float:
    """Cohen's d for paired samples: mean difference divided by SD of differences."""
    sd_diff = np.std(differences, ddof=1)
    if sd_diff == 0:
        raise ValueError("Standard deviation of differences is zero; Cohen's d is undefined.")
    return float(np.mean(differences) / sd_diff)


# ---------------------------------------------------------------------------
# Two-group comparison: Welch's t-test
# ---------------------------------------------------------------------------

def compare_two_groups(
    data: pd.DataFrame,
    group_col: str,
    outcome_col: str,
) -> TwoGroupResult:
    """
    Compare a continuous outcome between exactly two groups using Welch's
    t-test (does not assume equal variances between groups).

    Reference: Welch, B. L. (1947). The generalization of "Student's"
    problem when several different population variances are involved.
    Biometrika, 34(1-2), 28-35.
    """
    _validate_two_columns(data, group_col, outcome_col)

    clean = data[[group_col, outcome_col]].dropna()
    labels = clean[group_col].unique().tolist()

    if len(labels) != 2:
        raise ValueError(
            f"compare_two_groups requires exactly 2 groups, found {len(labels)}: {labels}. "
            "Use compare_multiple_groups for 3 or more groups."
        )

    group_a_label, group_b_label = sorted(labels, key=str)
    a = clean.loc[clean[group_col] == group_a_label, outcome_col].to_numpy(dtype=float)
    b = clean.loc[clean[group_col] == group_b_label, outcome_col].to_numpy(dtype=float)

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group needs at least 2 observations.")

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    # Welch-Satterthwaite degrees of freedom (what scipy uses internally for equal_var=False)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    n_a, n_b = len(a), len(b)
    df = (var_a / n_a + var_b / n_b) ** 2 / (
        (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    )

    mean_diff = float(np.mean(a) - np.mean(b))
    se_diff = float(np.sqrt(var_a / n_a + var_b / n_b))
    t_crit = stats.t.ppf(0.975, df)
    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff

    d = _cohens_d_independent(a, b)

    return TwoGroupResult(
        group_a_label=str(group_a_label),
        group_b_label=str(group_b_label),
        n_a=n_a,
        n_b=n_b,
        mean_a=float(np.mean(a)),
        mean_b=float(np.mean(b)),
        sd_a=float(np.std(a, ddof=1)),
        sd_b=float(np.std(b, ddof=1)),
        t_statistic=float(t_stat),
        degrees_of_freedom=float(df),
        p_value=float(p_value),
        cohens_d=d,
        mean_difference=mean_diff,
        ci_95_low=float(ci_low),
        ci_95_high=float(ci_high),
        n_input_rows=int(len(data)),
        n_rows_used=int(len(clean)),
        n_excluded_rows=int(len(data) - len(clean)),
        exclusion_reason=MISSING_GROUP_OR_OUTCOME,
    )


# ---------------------------------------------------------------------------
# Two-group comparison, nonparametric: Mann-Whitney U
# ---------------------------------------------------------------------------

def compare_two_groups_nonparametric(
    data: pd.DataFrame,
    group_col: str,
    outcome_col: str,
) -> TwoGroupNonparametricResult:
    """
    Compare an outcome between exactly two groups using the Mann-Whitney
    U test, a rank-based alternative to compare_two_groups' Welch's
    t-test that does not assume the outcome is normally distributed
    within each group.

    Effect size is a rank-biserial correlation, r = 2U/(n_a * n_b) - 1,
    where U is scipy's mannwhitneyu statistic computed for group A
    against group B. U ranges from 0 (every group-A value below every
    group-B value) to n_a*n_b (every group-A value above every group-B
    value), so this rescales U onto [-1, 1] with the same sign
    convention as Cohen's d: positive means group A tends higher. This
    specific formula and sign convention were verified by hand against
    scipy's own U statistic (see modules/program_evaluation/tests/
    test_comparison.py), since different sources define "U" for
    opposite samples and a borrowed formula can silently flip sign.

    References:
        Mann, H. B., & Whitney, D. R. (1947). On a test of whether one
        of two random variables is stochastically larger than the
        other. Annals of Mathematical Statistics, 18(1), 50-60.
    """
    _validate_two_columns(data, group_col, outcome_col)

    clean = data[[group_col, outcome_col]].dropna()
    labels = clean[group_col].unique().tolist()

    if len(labels) != 2:
        raise ValueError(
            f"compare_two_groups_nonparametric requires exactly 2 groups, "
            f"found {len(labels)}: {labels}. Use compare_multiple_groups "
            "for 3 or more groups."
        )

    group_a_label, group_b_label = sorted(labels, key=str)
    a = clean.loc[clean[group_col] == group_a_label, outcome_col].to_numpy(dtype=float)
    b = clean.loc[clean[group_col] == group_b_label, outcome_col].to_numpy(dtype=float)

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group needs at least 2 observations.")

    u_stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    n_a, n_b = len(a), len(b)
    r = (2.0 * float(u_stat)) / (n_a * n_b) - 1.0

    return TwoGroupNonparametricResult(
        group_a_label=str(group_a_label),
        group_b_label=str(group_b_label),
        n_a=n_a,
        n_b=n_b,
        median_a=float(np.median(a)),
        median_b=float(np.median(b)),
        u_statistic=float(u_stat),
        p_value=float(p_value),
        rank_biserial_correlation=r,
        n_input_rows=int(len(data)),
        n_rows_used=int(len(clean)),
        n_excluded_rows=int(len(data) - len(clean)),
        exclusion_reason=MISSING_GROUP_OR_OUTCOME,
    )


# ---------------------------------------------------------------------------
# Multi-group comparison: one-way ANOVA + Tukey HSD
# ---------------------------------------------------------------------------

def compare_multiple_groups(
    data: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    *,
    alpha: float = 0.05,
) -> MultiGroupResult:
    """
    Compare a continuous outcome across 3 or more groups using one-way
    ANOVA, followed by Tukey HSD post-hoc pairwise comparisons.

    References:
        Tukey, J. W. (1949). Comparing individual means in the analysis
        of variance. Biometrics, 5(2), 99-114.
    """
    _validate_two_columns(data, group_col, outcome_col)

    clean = data[[group_col, outcome_col]].dropna()
    labels = sorted(clean[group_col].unique().tolist(), key=str)

    if len(labels) < 3:
        raise ValueError(
            f"compare_multiple_groups requires 3 or more groups, found {len(labels)}. "
            "Use compare_two_groups for exactly 2 groups."
        )

    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
    except ImportError as exc:
        raise ImportError(
            "compare_multiple_groups requires statsmodels. "
            "Install it with: pip install statsmodels"
        ) from exc

    groups = [clean.loc[clean[group_col] == label, outcome_col].to_numpy(dtype=float) for label in labels]

    small_groups = [str(label) for label, g in zip(labels, groups) if len(g) < MIN_GROUP_SIZE]

    f_stat, p_value = stats.f_oneway(*groups)

    k = len(groups)
    n_total = sum(len(g) for g in groups)
    df_between = k - 1
    df_within = n_total - k

    tukey = pairwise_tukeyhsd(
        endog=clean[outcome_col].to_numpy(dtype=float),
        groups=clean[group_col].astype(str).to_numpy(),
        alpha=alpha,
    )

    pairwise = []
    for row in tukey.summary().data[1:]:
        group_a, group_b, mean_diff, p_adj, lower, upper, reject = row
        pairwise.append(
            PairwiseComparison(
                group_a=str(group_a),
                group_b=str(group_b),
                mean_difference=float(mean_diff),
                p_value=float(p_adj),
                significant=bool(reject),
            )
        )

    return MultiGroupResult(
        group_labels=[str(l) for l in labels],
        group_ns={str(label): len(g) for label, g in zip(labels, groups)},
        group_means={str(label): float(np.mean(g)) for label, g in zip(labels, groups)},
        f_statistic=float(f_stat),
        df_between=df_between,
        df_within=df_within,
        p_value=float(p_value),
        pairwise_comparisons=pairwise,
        small_groups_flagged=small_groups,
        n_input_rows=int(len(data)),
        n_rows_used=int(len(clean)),
        n_excluded_rows=int(len(data) - len(clean)),
        exclusion_reason=MISSING_GROUP_OR_OUTCOME,
    )


def compare_multiple_groups_welch(
    data: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    *,
    alpha: float = 0.05,
) -> MultiGroupWelchResult:
    """
    Compare a continuous outcome across 3 or more groups using Welch's
    one-way ANOVA, followed by Games-Howell post-hoc pairwise comparisons.

    Unlike compare_multiple_groups (standard ANOVA + Tukey HSD), neither
    step here assumes equal variances across groups, matching the same
    philosophy as compare_two_groups' use of Welch's t-test rather than
    Student's.

    References:
        Welch, B. L. (1951). On the comparison of several mean values:
        An alternative approach. Biometrika, 38(3/4), 330-336.

        Games, P. A., & Howell, J. F. (1976). Pairwise multiple
        comparison procedures with unequal n's and/or variances: A
        Monte Carlo study. Journal of Educational Statistics, 1(2),
        113-125.

    Implemented directly with scipy and statsmodels' studentized range
    distribution functions (statsmodels.stats.libqsturng), rather than a
    separate dependency, since statsmodels is already required for
    compare_multiple_groups' Tukey HSD step.
    """
    _validate_two_columns(data, group_col, outcome_col)

    clean = data[[group_col, outcome_col]].dropna()
    labels = sorted(clean[group_col].unique().tolist(), key=str)

    if len(labels) < 3:
        raise ValueError(
            f"compare_multiple_groups_welch requires 3 or more groups, found "
            f"{len(labels)}. Use compare_two_groups for exactly 2 groups."
        )

    groups = [clean.loc[clean[group_col] == label, outcome_col].to_numpy(dtype=float) for label in labels]
    small_groups = [str(label) for label, g in zip(labels, groups) if len(g) < MIN_GROUP_SIZE]

    k = len(groups)
    ns = np.array([len(g) for g in groups], dtype=float)
    means = np.array([np.mean(g) for g in groups])
    variances = np.array([np.var(g, ddof=1) for g in groups])

    if (variances == 0).any():
        raise ValueError(
            "One or more groups have zero variance; Welch's ANOVA is undefined."
        )

    weights = ns / variances
    sum_w = weights.sum()
    grand_mean = float((weights * means).sum() / sum_w)

    numerator = float((weights * (means - grand_mean) ** 2).sum() / (k - 1))
    term = float(((1 - weights / sum_w) ** 2 / (ns - 1)).sum())
    denominator = 1 + (2 * (k - 2) / (k ** 2 - 1)) * term

    f_stat = numerator / denominator
    df_between = float(k - 1)
    df_within = float((k ** 2 - 1) / (3 * term))
    p_value = float(stats.f.sf(f_stat, df_between, df_within))

    # Only the Games-Howell post-hoc step needs statsmodels' studentized
    # range distribution functions, everything above is scipy-only.
    try:
        from statsmodels.stats.libqsturng import psturng, qsturng
    except ImportError as exc:
        raise ImportError(
            "compare_multiple_groups_welch requires statsmodels for the "
            "Games-Howell post-hoc step. Install it with: "
            "pip install statsmodels"
        ) from exc

    # Games-Howell pairwise comparisons: each pair uses its own
    # Welch-Satterthwaite degrees of freedom (same logic as compare_two_groups),
    # with critical values and p-values from the studentized range
    # distribution instead of the t-distribution, correcting for the
    # number of groups being compared.
    pairwise: list[PairwiseComparison] = []
    for (i, label_a), (j, label_b) in combinations(enumerate(labels), 2):
        mean_a, mean_b = means[i], means[j]
        var_a, var_b = variances[i], variances[j]
        n_a, n_b = ns[i], ns[j]

        se = float(np.sqrt(var_a / n_a + var_b / n_b))
        mean_diff = float(mean_a - mean_b)

        pair_df = (var_a / n_a + var_b / n_b) ** 2 / (
            (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        )

        t_stat = mean_diff / se
        q_stat = abs(t_stat) * np.sqrt(2)
        # psturng returns a numpy array rather than a Python scalar, even
        # for scalar input. float() on a non-0-dimensional array raises
        # TypeError on numpy >= 2.0, so extract the single value explicitly.
        p_adj = float(np.asarray(psturng(q_stat, k, pair_df)).item())

        pairwise.append(
            PairwiseComparison(
                group_a=str(label_a),
                group_b=str(label_b),
                mean_difference=mean_diff,
                p_value=p_adj,
                significant=bool(p_adj < alpha),
            )
        )

    return MultiGroupWelchResult(
        group_labels=[str(l) for l in labels],
        group_ns={str(label): len(g) for label, g in zip(labels, groups)},
        group_means={str(label): float(np.mean(g)) for label, g in zip(labels, groups)},
        f_statistic=float(f_stat),
        df_between=df_between,
        df_within=df_within,
        p_value=p_value,
        pairwise_comparisons=pairwise,
        small_groups_flagged=small_groups,
        n_input_rows=int(len(data)),
        n_rows_used=int(len(clean)),
        n_excluded_rows=int(len(data) - len(clean)),
        exclusion_reason=MISSING_GROUP_OR_OUTCOME,
    )


# ---------------------------------------------------------------------------
# Categorical outcome: chi-square test of independence
# ---------------------------------------------------------------------------

def compare_categorical(
    data: pd.DataFrame,
    group_col: str,
    outcome_col: str,
) -> ChiSquareResult:
    """
    Test whether a categorical outcome is independent of group membership,
    using a chi-square test of independence.

    Reference: Pearson, K. (1900). On the criterion that a given system
    of deviations from the probable in the case of a correlated system of
    variables is such that it can be reasonably supposed to have arisen
    from random sampling. Philosophical Magazine, 50(302), 157-175.
    """
    _validate_two_columns(data, group_col, outcome_col)

    clean = data[[group_col, outcome_col]].dropna()
    contingency = pd.crosstab(clean[group_col], clean[outcome_col])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        raise ValueError(
            "Chi-square test requires at least 2 groups and 2 outcome categories."
        )

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    expected_df = pd.DataFrame(
        expected, index=contingency.index, columns=contingency.columns
    )
    low_expected = bool((expected_df < 5).any().any())

    return ChiSquareResult(
        chi2_statistic=float(chi2),
        degrees_of_freedom=int(dof),
        p_value=float(p_value),
        contingency_table=contingency,
        expected_frequencies=expected_df,
        low_expected_frequency_warning=low_expected,
        n_input_rows=int(len(data)),
        n_rows_used=int(len(clean)),
        n_excluded_rows=int(len(data) - len(clean)),
        exclusion_reason=MISSING_GROUP_OR_OUTCOME,
    )


# ---------------------------------------------------------------------------
# Pre/post, single group: paired t-test
# ---------------------------------------------------------------------------

def compare_pre_post(pre: pd.Series, post: pd.Series) -> PairedResult:
    """
    Paired t-test comparing pre- and post-program scores for the same
    participants.

    pre and post must be the same length and aligned by participant
    (e.g. by index or row order); rows with a missing value in either
    are excluded via listwise deletion.
    """
    if len(pre) != len(post):
        raise ValueError("pre and post must have the same number of observations.")

    combined = pd.DataFrame({"pre": pre.to_numpy(), "post": post.to_numpy()}).dropna()

    if combined.shape[0] < 2:
        raise ValueError("At least 2 complete paired observations are required.")

    pre_clean = combined["pre"].to_numpy(dtype=float)
    post_clean = combined["post"].to_numpy(dtype=float)
    differences = post_clean - pre_clean

    if np.var(differences, ddof=1) == 0:
        raise ValueError("All differences are identical; the paired t-test is undefined.")

    t_stat, p_value = stats.ttest_rel(post_clean, pre_clean)
    n = len(differences)
    d = _cohens_d_paired(differences)

    return PairedResult(
        n=n,
        mean_pre=float(np.mean(pre_clean)),
        mean_post=float(np.mean(post_clean)),
        mean_difference=float(np.mean(differences)),
        t_statistic=float(t_stat),
        degrees_of_freedom=n - 1,
        p_value=float(p_value),
        cohens_d=d,
        n_input_rows=int(len(pre)),
        n_rows_used=int(len(combined)),
        n_excluded_rows=int(len(pre) - len(combined)),
        exclusion_reason=INCOMPLETE_PAIR,
    )


# ---------------------------------------------------------------------------
# Paired comparison, nonparametric: Wilcoxon signed-rank
# ---------------------------------------------------------------------------

def compare_pre_post_nonparametric(pre: pd.Series, post: pd.Series) -> PairedNonparametricResult:
    """
    Wilcoxon signed-rank test comparing pre- and post-program scores for
    the same participants, a rank-based alternative to compare_pre_post's
    paired t-test that does not assume the differences are normally
    distributed.

    Ties at exactly zero (no change) are dropped before ranking, scipy's
    default convention (Wilcoxon's own original treatment); the count
    dropped is reported on n_zero_differences_dropped rather than folded
    into n_excluded_rows, since a zero difference is real, observed data,
    not a missing value.

    Effect size is the matched-pairs rank-biserial correlation,
    r = (W+ - W-) / (W+ + W-), where W+ and W- are the summed ranks of
    the positive and negative differences (ranked by absolute value).
    Ranges from -1 to 1, positive meaning post tended to exceed pre. This
    formula was verified by hand against scipy's own W statistic (see
    modules/program_evaluation/tests/test_comparison.py): scipy reports
    min(W+, W-) as its statistic, so this is computed independently
    rather than derived from it, to keep the sign information scipy's
    own statistic discards.

    References:
        Wilcoxon, F. (1945). Individual comparisons by ranking methods.
        Biometrics Bulletin, 1(6), 80-83.

        King, B. M., & Minium, E. W. (2003). Statistical Reasoning in
        Psychology and Education (4th ed.). Wiley. (Matched-pairs
        rank-biserial correlation.)
    """
    if len(pre) != len(post):
        raise ValueError("pre and post must have the same number of observations.")

    combined = pd.DataFrame({"pre": pre.to_numpy(), "post": post.to_numpy()}).dropna()

    if combined.shape[0] < 2:
        raise ValueError("At least 2 complete paired observations are required.")

    pre_clean = combined["pre"].to_numpy(dtype=float)
    post_clean = combined["post"].to_numpy(dtype=float)
    differences = post_clean - pre_clean

    nonzero = differences[differences != 0]
    n_zero = int(len(differences) - len(nonzero))

    if len(nonzero) < 1:
        raise ValueError(
            "Every paired difference is zero; the Wilcoxon signed-rank "
            "test is undefined."
        )

    w_stat, p_value = stats.wilcoxon(nonzero, alternative="two-sided")

    ranks = stats.rankdata(np.abs(nonzero))
    w_pos = float(ranks[nonzero > 0].sum())
    w_neg = float(ranks[nonzero < 0].sum())
    r = (w_pos - w_neg) / (w_pos + w_neg)

    return PairedNonparametricResult(
        n=len(differences),
        median_pre=float(np.median(pre_clean)),
        median_post=float(np.median(post_clean)),
        median_difference=float(np.median(differences)),
        w_statistic=float(w_stat),
        p_value=float(p_value),
        matched_pairs_rank_biserial_correlation=r,
        n_input_rows=int(len(pre)),
        n_rows_used=int(len(combined)),
        n_excluded_rows=int(len(pre) - len(combined)),
        exclusion_reason=INCOMPLETE_PAIR,
        n_zero_differences_dropped=n_zero,
    )


# ---------------------------------------------------------------------------
# Multi-select sensitivity analysis
# ---------------------------------------------------------------------------

def expand_multiselect(
    data: pd.DataFrame,
    multiselect_col: str,
    delimiter: str = ",",
) -> pd.DataFrame:
    """
    'Expanded' coding: a row with multiple selections (e.g. "Black/African,
    Black/Caribbean") becomes one row per selection, each keeping the same
    outcome value. A participant who selected 2 categories contributes to
    both group comparisons.
    """
    rows = []
    for _, row in data.iterrows():
        raw = row[multiselect_col]
        if pd.isna(raw):
            continue
        categories = [c.strip() for c in str(raw).split(delimiter) if c.strip()]
        for category in categories:
            new_row = row.copy()
            new_row[multiselect_col] = category
            rows.append(new_row)
    return pd.DataFrame(rows).reset_index(drop=True)


def single_select_multiselect(
    data: pd.DataFrame,
    multiselect_col: str,
    delimiter: str = ",",
) -> pd.DataFrame:
    """'Single-selection' coding: keep only the first listed category per row."""
    result = data.copy()
    result[multiselect_col] = result[multiselect_col].apply(
        lambda raw: str(raw).split(delimiter)[0].strip() if pd.notna(raw) else raw
    )
    return result


def combined_category_multiselect(
    data: pd.DataFrame,
    multiselect_col: str,
    delimiter: str = ",",
    combined_label: str = "Multiple categories",
) -> pd.DataFrame:
    """'Combined-category' coding: anyone who selected more than one
    category is relabeled under a single combined group."""
    result = data.copy()

    def recode(raw):
        if pd.isna(raw):
            return raw
        categories = [c.strip() for c in str(raw).split(delimiter) if c.strip()]
        if len(categories) > 1:
            return combined_label
        return categories[0] if categories else raw

    result[multiselect_col] = result[multiselect_col].apply(recode)
    return result


def sensitivity_analysis(
    data: pd.DataFrame,
    multiselect_col: str,
    outcome_col: str,
    *,
    delimiter: str = ",",
    alpha: float = 0.05,
) -> SensitivityResult:
    """
    Re-run a group comparison under three codings of a multi-select
    categorical predictor, and report whether the conclusion (significant
    vs. not, at the given alpha) is consistent across all three.

    This generalizes the sensitivity-analysis approach used in McCray,
    Dukes, & Pittman (in press), where race/ethnicity was a multi-select
    field and results were checked against expanded, single-selection, and
    combined-category codings before being trusted.
    """
    codings = {
        "expanded": expand_multiselect(data, multiselect_col, delimiter),
        "single_selection": single_select_multiselect(data, multiselect_col, delimiter),
        "combined_category": combined_category_multiselect(data, multiselect_col, delimiter),
    }

    coding_results: dict[str, object] = {}
    p_values: dict[str, float] = {}

    for name, coded_data in codings.items():
        n_groups = coded_data[multiselect_col].dropna().nunique()
        if n_groups == 2:
            result = compare_two_groups(coded_data, multiselect_col, outcome_col)
        elif n_groups >= 3:
            result = compare_multiple_groups(coded_data, multiselect_col, outcome_col, alpha=alpha)
        else:
            raise ValueError(
                f"Coding '{name}' produced fewer than 2 groups; cannot compare."
            )
        coding_results[name] = result
        p_values[name] = result.p_value

    significances = {name: (p < alpha) for name, p in p_values.items()}
    consistent = len(set(significances.values())) == 1

    return SensitivityResult(
        coding_results=coding_results,
        p_values_by_coding=p_values,
        consistent_conclusion=consistent,
        alpha=alpha,
        n_input_rows=int(len(data)),
        n_expanded_rows=int(len(codings["expanded"])),
    )
