"""
Recommend statistical methods for the Program Validation module.

Recommendations are based on study design, outcome type, and group
structure. The UI should display the rationale and allow users to confirm
or override the recommendation because column types alone cannot always
determine a variable's measurement level or the most appropriate analysis.

The initial design is informed by the mixed-methods program-evaluation
workflow described by McCray, Dukes, and Pittman (in press).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class MethodRecommendation:
    """A suggested statistical method and the reasoning behind it."""

    method: str
    display_name: str
    reasoning: list[str]
    warnings: list[str] = field(default_factory=list)
    supported: bool = True


def _validate_column_exists(
    data: pd.DataFrame,
    column: str,
    *,
    role: str,
) -> None:
    """Raise a clear error when a required column is missing."""
    if column not in data.columns:
        raise ValueError(
            f"{role.capitalize()} column '{column}' was not found in the data."
        )


def _outcome_looks_categorical(series: pd.Series) -> bool:
    """
    Estimate whether a variable should be treated as categorical.

    A variable is treated as categorical when:

    - it is nonnumeric; or
    - it is numeric with exactly two distinct nonmissing values.

    Numeric variables with three or more values are treated as
    continuous-like by default. This includes common 1-to-5 Likert
    outcomes, but users should be able to override the recommendation.
    """
    clean = series.dropna()

    if clean.empty:
        raise ValueError(
            f"Column '{series.name}' contains no nonmissing observations."
        )

    if not pd.api.types.is_numeric_dtype(clean):
        return True

    return clean.nunique() == 2


def _numeric_binary_warning(series: pd.Series) -> str | None:
    """Return a warning when a numeric variable has exactly two values."""
    clean = series.dropna()

    if (
        pd.api.types.is_numeric_dtype(clean)
        and clean.nunique() == 2
    ):
        return (
            f"'{series.name}' has exactly two numeric values and is being "
            "treated as a binary categorical variable. Confirm that these "
            "values represent categories rather than a continuous quantity."
        )

    return None


def _validate_pre_post_columns(
    data: pd.DataFrame,
    pre_col: str,
    post_col: str,
) -> None:
    """Validate paired pre/post column inputs."""
    _validate_column_exists(
        data,
        pre_col,
        role="pre",
    )
    _validate_column_exists(
        data,
        post_col,
        role="post",
    )

    if pre_col == post_col:
        raise ValueError(
            "The pre and post columns must be different columns."
        )

    complete_pairs = data[[pre_col, post_col]].dropna()

    if complete_pairs.shape[0] < 2:
        raise ValueError(
            "At least two complete pre/post pairs are required."
        )


def _validate_group_column(
    data: pd.DataFrame,
    group_col: str,
) -> int:
    """Validate a group column and return its nonmissing group count."""
    _validate_column_exists(
        data,
        group_col,
        role="group",
    )

    number_of_groups = data[group_col].dropna().nunique()

    if number_of_groups < 2:
        raise ValueError(
            f"Group column '{group_col}' contains fewer than two "
            "nonmissing groups."
        )

    return int(number_of_groups)


def recommend_method(
    data: pd.DataFrame,
    *,
    outcome_col: str | None = None,
    group_col: str | None = None,
    pre_col: str | None = None,
    post_col: str | None = None,
    is_multiselect_group: bool = False,
) -> MethodRecommendation:
    """
    Recommend an analysis based on the supplied study design.

    Provide exactly one of these designs:

    1. Independent-group comparison:
       ``outcome_col`` and ``group_col``

    2. Paired pre/post comparison:
       ``pre_col`` and ``post_col``

    Parameters
    ----------
    data:
        Input dataset.

    outcome_col:
        Outcome variable for independent-group comparisons.

    group_col:
        Variable identifying independent groups.

    pre_col:
        Baseline outcome for paired pre/post analysis.

    post_col:
        Follow-up outcome for paired pre/post analysis.

    is_multiselect_group:
        Whether each participant can belong to multiple categories in
        ``group_col``. For example, a participant may select multiple
        race or ethnicity categories.

    Returns
    -------
    MethodRecommendation
        Recommended method, display label, rationale, warnings, and
        whether the method is currently supported.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Data must be provided as a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "The dataset contains no rows."
        )

    has_group_design = (
        outcome_col is not None
        or group_col is not None
    )
    has_pre_post_design = (
        pre_col is not None
        or post_col is not None
    )

    if has_group_design and has_pre_post_design:
        raise ValueError(
            "Provide either an outcome and group column or a paired "
            "pre/post column pair, not both."
        )

    if not has_group_design and not has_pre_post_design:
        raise ValueError(
            "Provide either outcome_col and group_col, or pre_col and "
            "post_col."
        )

    if has_pre_post_design:
        if pre_col is None or post_col is None:
            raise ValueError(
                "Both pre_col and post_col are required for a paired "
                "pre/post analysis."
            )

        return _recommend_pre_post(
            data=data,
            pre_col=pre_col,
            post_col=post_col,
        )

    if outcome_col is None or group_col is None:
        raise ValueError(
            "Both outcome_col and group_col are required for an "
            "independent-group comparison."
        )

    return _recommend_group_comparison(
        data=data,
        outcome_col=outcome_col,
        group_col=group_col,
        is_multiselect_group=is_multiselect_group,
    )


def _recommend_pre_post(
    data: pd.DataFrame,
    *,
    pre_col: str,
    post_col: str,
) -> MethodRecommendation:
    """Recommend a method for paired pre/post observations."""
    _validate_pre_post_columns(
        data,
        pre_col,
        post_col,
    )

    pre_series = data[pre_col]
    post_series = data[post_col]

    pre_is_categorical = _outcome_looks_categorical(
        pre_series
    )
    post_is_categorical = _outcome_looks_categorical(
        post_series
    )

    reasoning = [
        "Pre and post columns were provided for the same participants, "
        "so this is a within-participant paired comparison."
    ]
    warnings: list[str] = []

    for series in (pre_series, post_series):
        warning = _numeric_binary_warning(series)

        if warning is not None:
            warnings.append(warning)

    if pre_is_categorical != post_is_categorical:
        warnings.append(
            "The pre and post columns appear to use different measurement "
            "types. Confirm that both columns use the same coding and scale."
        )

        return MethodRecommendation(
            method="review_pre_post_coding",
            display_name="Review pre/post coding",
            reasoning=reasoning,
            warnings=warnings,
            supported=False,
        )

    if pre_is_categorical and post_is_categorical:
        reasoning.append(
            "Both measurements appear categorical. A paired categorical "
            "method is required because observations from the same "
            "participant are not independent."
        )
        reasoning.append(
            "For binary paired outcomes, McNemar's test evaluates whether "
            "the proportion of responses changed between time points."
        )

        warnings.append(
            "Paired categorical analysis is planned but not yet implemented "
            "in Program Validation v0.1."
        )

        return MethodRecommendation(
            method="compare_paired_categorical",
            display_name="McNemar's test",
            reasoning=reasoning,
            warnings=warnings,
            supported=False,
        )

    reasoning.append(
        "Both measurements appear continuous-like, so the analysis should "
        "evaluate the participant-level change between post and pre scores."
    )
    reasoning.append(
        "A paired t-test is recommended because it accounts for the "
        "dependence between repeated observations from the same participant."
    )

    warnings.append(
        "A single-group pre/post design can estimate change over time but "
        "cannot rule out maturation, regression to the mean, external "
        "events, or other alternative explanations."
    )

    return MethodRecommendation(
        method="compare_pre_post",
        display_name="Paired t-test",
        reasoning=reasoning,
        warnings=warnings,
    )


def _recommend_group_comparison(
    data: pd.DataFrame,
    *,
    outcome_col: str,
    group_col: str,
    is_multiselect_group: bool,
) -> MethodRecommendation:
    """Recommend a method for independent-group comparisons."""
    _validate_column_exists(
        data,
        outcome_col,
        role="outcome",
    )

    number_of_groups = _validate_group_column(
        data,
        group_col,
    )

    outcome_series = data[outcome_col]
    outcome_is_categorical = _outcome_looks_categorical(
        outcome_series
    )

    reasoning: list[str] = []
    warnings: list[str] = []

    binary_warning = _numeric_binary_warning(
        outcome_series
    )

    if binary_warning is not None:
        warnings.append(binary_warning)

    if is_multiselect_group:
        reasoning.append(
            "The group variable allows participants to select multiple "
            "categories."
        )
        reasoning.append(
            "A single coding approach could change group membership and "
            "therefore affect the conclusion. A sensitivity analysis across "
            "multiple defensible coding strategies is recommended."
        )
        reasoning.append(
            "The planned workflow compares expanded coding, "
            "single-selection coding, and a combined multiselect category."
        )

        warnings.append(
            "Participants may contribute to more than one group under "
            "expanded coding, so observations are not fully independent. "
            "Results should be interpreted as a sensitivity analysis rather "
            "than as three equivalent primary analyses."
        )

        return MethodRecommendation(
            method="sensitivity_analysis",
            display_name="Multiselect-group sensitivity analysis",
            reasoning=reasoning,
            warnings=warnings,
        )

    if outcome_is_categorical:
        reasoning.append(
            f"Outcome column '{outcome_col}' appears categorical, and "
            f"'{group_col}' identifies {number_of_groups} independent groups."
        )
        reasoning.append(
            "A chi-square test of independence evaluates whether the "
            "distribution of the categorical outcome differs across groups."
        )

        warnings.append(
            "The analysis should inspect expected cell counts. Fisher's "
            "exact test or another exact method may be more appropriate "
            "when expected counts are small."
        )

        return MethodRecommendation(
            method="compare_categorical",
            display_name="Chi-square test of independence",
            reasoning=reasoning,
            warnings=warnings,
        )

    if number_of_groups == 2:
        reasoning.append(
            f"Exactly two independent groups were found in '{group_col}', "
            f"and '{outcome_col}' appears continuous-like."
        )
        reasoning.append(
            "Welch's t-test is recommended because it compares group means "
            "without assuming equal variances or equal sample sizes."
        )

        warnings.append(
            "If group assignment was not randomized, the observed "
            "difference may reflect baseline imbalance, selection, or "
            "unmeasured confounding rather than the program alone."
        )

        return MethodRecommendation(
            method="compare_two_groups",
            display_name="Welch's independent-samples t-test",
            reasoning=reasoning,
            warnings=warnings,
        )

    reasoning.append(
        f"{number_of_groups} independent groups were found in "
        f"'{group_col}', and '{outcome_col}' appears continuous-like."
    )
    reasoning.append(
        "Welch's one-way ANOVA is recommended because it does not require "
        "equal variances or equal sample sizes across groups."
    )
    reasoning.append(
        "If the overall test indicates group differences, Games-Howell "
        "comparisons can identify which pairs differ while accommodating "
        "unequal variances and sample sizes."
    )

    if number_of_groups > 6:
        pair_count = (
            number_of_groups
            * (number_of_groups - 1)
            // 2
        )

        warnings.append(
            f"{number_of_groups} groups produce {pair_count} pairwise "
            "comparisons. Consider whether all comparisons are "
            "scientifically meaningful and whether any categories can be "
            "combined based on a defensible rationale."
        )

    warnings.append(
        "If group membership was not randomized, group differences may "
        "reflect selection, baseline imbalance, or unmeasured confounding."
    )

    return MethodRecommendation(
        method="compare_multiple_groups_welch",
        display_name="Welch's one-way ANOVA with Games-Howell comparisons",
        reasoning=reasoning,
        warnings=warnings,
    )
