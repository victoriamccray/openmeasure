"""
Difference-in-differences (2x2) for the Impact Evaluation module.

This is the third design the module supports, and it is the combination
of the two that already existed: a group column (as in the two-group
comparison) plus a pre and a post column for the same units (as in the
paired pre/post comparison). One row per unit, observed in both periods.

The estimate is the difference between how much the treated group
changed and how much the comparison group changed:

    DiD = (post_treated - pre_treated) - (post_comparison - pre_comparison)

Equivalently, and this is how it is computed here, it is the difference
in mean change scores between the two groups. Working on change scores
rather than on the four raw means is what makes the standard error
correct for repeated observations of the same unit: differencing removes
each unit's own level entirely, so the two sets of change scores are
independent of each other and the usual two-sample machinery applies.

Standard error, confidence interval, and p-value therefore come from a
Welch two-sample comparison of the change scores, which does not assume
the two groups' changes have equal variance. This matches the module's
existing preference for Welch's t-test over Student's, and it is robust
in the two senses that matter for this design: to unequal variance
between the groups, and to arbitrary correlation between a unit's own two
observations. It is not robust to correlation *between* units (spillover,
shared shocks within a cluster); that is a design assumption stated in
interpret.py, not something this standard error can absorb.

The point estimate is identical to the interaction coefficient of an OLS
regression of the outcome on treated, post, and their interaction, and
the standard error is identical to the HC2 robust standard error of a
regression of the change score on the treated indicator. Both
equivalences are pinned in tests/test_did.py against statsmodels.

References:
    Card, D., & Krueger, A. B. (1994). Minimum wages and employment: A
    case study of the fast-food industry in New Jersey and Pennsylvania.
    American Economic Review, 84(4), 772-793.

    Angrist, J. D., & Pischke, J.-S. (2009). Mostly Harmless
    Econometrics, ch. 5 (differences-in-differences).

    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-
    consistent covariance matrix estimators with improved finite sample
    properties. Journal of Econometrics, 29(3), 305-325 (the HC2
    estimator whose equivalence is pinned in tests/test_did.py).

    Welch, B. L. (1947). The generalization of "Student's" problem when
    several different population variances are involved. Biometrika,
    34(1-2), 28-35.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.validation import validate_is_dataframe  # noqa: E402

from .comparison import MIN_GROUP_SIZE  # noqa: E402

# Why rows were dropped, recorded on the result per
# docs/design-standards.md section 2. A unit missing any one of the three
# fields has no change score, so it cannot enter either difference.
MISSING_GROUP_PRE_OR_POST = "missing group, pre, or post value"

# The standard error's own description, carried on the result rather than
# written into the page, so a reader is never shown an interval without
# being told what produced it.
SE_METHOD_WELCH_ON_CHANGES = (
    "Welch two-sample standard error on within-unit change scores"
)

# Below this many units in a group, the group's variance estimate (and so
# the interval and p-value) rests on very little information. Same
# threshold and same warn-rather-than-refuse posture as comparison.py.
_SMALL_GROUP_NOTE = (
    "fewer than {threshold} units, so this group's variance estimate is unstable"
)


@dataclass(frozen=True)
class DiDResult:
    """
    A 2x2 difference-in-differences estimate and the four means behind it.

    The four cell means and the two changes are carried alongside the
    estimate deliberately: the estimate is a difference of differences,
    and a reader cannot tell a large treated change from a large
    comparison change without seeing both.
    """

    treated_label: str
    comparison_label: str
    n_treated: int
    n_comparison: int

    mean_pre_treated: float
    mean_post_treated: float
    mean_pre_comparison: float
    mean_post_comparison: float

    change_treated: float
    change_comparison: float

    did_estimate: float
    standard_error: float
    standard_error_method: str
    t_statistic: float
    degrees_of_freedom: float
    p_value: float
    ci_95_low: float
    ci_95_high: float

    # The baseline gap between the groups before either changed. Not an
    # assumption test (parallel trends is about slopes, not levels), but
    # a large gap tells a reader the two groups were not alike to begin
    # with, which is what makes the parallel-trends assumption harder to
    # take on faith.
    baseline_difference: float

    small_groups_flagged: tuple[str, ...]

    n_input_rows: int
    n_rows_used: int
    n_excluded_rows: int
    exclusion_reason: str


def _validate_columns(
    data: pd.DataFrame,
    group_col: str,
    pre_col: str,
    post_col: str,
) -> None:
    """Validate the three column inputs before any row is touched."""
    validate_is_dataframe(data)

    for column, role in ((group_col, "Group"), (pre_col, "Pre"), (post_col, "Post")):
        if column not in data.columns:
            raise ValueError(f"{role} column '{column}' not found in data.")

    if len({group_col, pre_col, post_col}) != 3:
        raise ValueError(
            "The group, pre, and post columns must be three different "
            f"columns, got group='{group_col}', pre='{pre_col}', "
            f"post='{post_col}'."
        )


def estimate_did(
    data: pd.DataFrame,
    group_col: str,
    pre_col: str,
    post_col: str,
    *,
    treated_label: object,
) -> DiDResult:
    """
    Estimate a 2x2 difference-in-differences effect.

    Parameters
    ----------
    data:
        One row per unit, observed in both periods.
    group_col:
        Column identifying exactly two groups, one treated and one
        comparison.
    pre_col:
        The outcome before the treated group was treated, for every unit.
    post_col:
        The same outcome after, for every unit.
    treated_label:
        Which value of ``group_col`` received the treatment. Required
        rather than inferred: nothing in the data marks which group was
        treated, and guessing (by sort order, or by which group changed
        more) would silently decide the sign of the estimate.

    Raises
    ------
    TypeError
        If ``data`` is not a DataFrame.
    ValueError
        If a column is missing or repeated, if the group column does not
        hold exactly two nonmissing groups, if ``treated_label`` is not
        one of them, if either group has fewer than two complete units,
        or if neither group's change scores vary at all (which leaves the
        standard error at zero and the p-value undefined).
    """
    _validate_columns(data, group_col, pre_col, post_col)

    clean = data[[group_col, pre_col, post_col]].dropna()
    labels = sorted(clean[group_col].unique().tolist(), key=str)

    if len(labels) != 2:
        raise ValueError(
            f"Difference-in-differences requires exactly 2 groups in "
            f"'{group_col}', found {len(labels)}: {labels}. One group is "
            "treated and one is the comparison."
        )

    if treated_label not in labels:
        raise ValueError(
            f"treated_label '{treated_label}' is not one of the groups "
            f"found in '{group_col}': {labels}."
        )

    comparison_label = next(label for label in labels if label != treated_label)

    treated = clean.loc[clean[group_col] == treated_label]
    comparison = clean.loc[clean[group_col] == comparison_label]

    if len(treated) < 2 or len(comparison) < 2:
        raise ValueError(
            "Each group needs at least 2 units with a complete pre and "
            f"post value, found {len(treated)} treated and "
            f"{len(comparison)} comparison."
        )

    change_treated = (
        treated[post_col].to_numpy(dtype=float) - treated[pre_col].to_numpy(dtype=float)
    )
    change_comparison = (
        comparison[post_col].to_numpy(dtype=float)
        - comparison[pre_col].to_numpy(dtype=float)
    )

    n_treated, n_comparison = len(change_treated), len(change_comparison)
    var_treated = float(np.var(change_treated, ddof=1))
    var_comparison = float(np.var(change_comparison, ddof=1))

    if var_treated == 0 and var_comparison == 0:
        raise ValueError(
            "Every unit in both groups changed by exactly the same amount, "
            "so the standard error is zero and no interval or p-value can "
            "be computed. Check whether the pre and post columns are the "
            "ones you intended."
        )

    did_estimate = float(np.mean(change_treated) - np.mean(change_comparison))
    standard_error = float(
        np.sqrt(var_treated / n_treated + var_comparison / n_comparison)
    )

    # Welch-Satterthwaite degrees of freedom, the same formula
    # comparison.py uses for the two-group test.
    degrees_of_freedom = (var_treated / n_treated + var_comparison / n_comparison) ** 2 / (
        (var_treated / n_treated) ** 2 / (n_treated - 1)
        + (var_comparison / n_comparison) ** 2 / (n_comparison - 1)
    )

    # Computed from the estimate and its standard error rather than by
    # handing the two change vectors to scipy.stats.ttest_ind. The two
    # are the same test and agree to floating-point precision (pinned in
    # tests/test_did.py), but ttest_ind recomputes variances that are
    # already in hand, and warns about catastrophic cancellation when one
    # group's units all changed by an identical amount, which is a case
    # this estimator handles correctly.
    t_statistic = did_estimate / standard_error
    p_value = 2 * stats.t.sf(abs(t_statistic), degrees_of_freedom)

    t_critical = stats.t.ppf(0.975, degrees_of_freedom)

    small_groups_flagged = tuple(
        f"{label} ({_SMALL_GROUP_NOTE.format(threshold=MIN_GROUP_SIZE)})"
        for label, count in (
            (str(treated_label), n_treated),
            (str(comparison_label), n_comparison),
        )
        if count < MIN_GROUP_SIZE
    )

    return DiDResult(
        treated_label=str(treated_label),
        comparison_label=str(comparison_label),
        n_treated=n_treated,
        n_comparison=n_comparison,
        mean_pre_treated=float(treated[pre_col].mean()),
        mean_post_treated=float(treated[post_col].mean()),
        mean_pre_comparison=float(comparison[pre_col].mean()),
        mean_post_comparison=float(comparison[post_col].mean()),
        change_treated=float(np.mean(change_treated)),
        change_comparison=float(np.mean(change_comparison)),
        did_estimate=did_estimate,
        standard_error=standard_error,
        standard_error_method=SE_METHOD_WELCH_ON_CHANGES,
        t_statistic=float(t_statistic),
        degrees_of_freedom=float(degrees_of_freedom),
        p_value=float(p_value),
        ci_95_low=did_estimate - t_critical * standard_error,
        ci_95_high=did_estimate + t_critical * standard_error,
        baseline_difference=float(
            treated[pre_col].mean() - comparison[pre_col].mean()
        ),
        small_groups_flagged=small_groups_flagged,
        n_input_rows=int(len(data)),
        n_rows_used=int(len(clean)),
        n_excluded_rows=int(len(data) - len(clean)),
        exclusion_reason=MISSING_GROUP_PRE_OR_POST,
    )
