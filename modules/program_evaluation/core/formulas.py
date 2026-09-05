"""
This module's statistics, explained as concept then numbers then notation.

The shape is shared/formula.py's; the content is specific to each
statistic, so it lives beside the statistic rather than in shared/, the
same rule interpret.py follows.

Every builder here takes a result that has already been computed and
reads its values off it. None of them recompute the headline statistic
for display: the number a reader sees is the number the analysis
produced. Where a builder needs an intermediate the result does not carry
(Cohen's d's pooled standard deviation, most notably), it derives it with
the same formula comparison.py used, and tests/test_formulas.py checks
that the arithmetic shown really does land on the stored statistic.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.formula import FormulaExplanation, FormulaTerm  # noqa: E402

from .comparison import TwoGroupResult  # noqa: E402
from .did import DiDResult  # noqa: E402

COHENS_D_CITATION = (
    "Cohen, J. (1988). Statistical Power Analysis for the Behavioral "
    "Sciences (2nd ed.)."
)


def pooled_standard_deviation(result: TwoGroupResult) -> float:
    """
    The denominator of Cohen's d, rebuilt from the result's own fields.

    comparison.py computes Cohen's d from the raw samples and keeps each
    group's standard deviation and size, but not the pooled value itself.
    This reconstructs it with the same formula, so the explanation can
    show the number it divided by rather than presenting a denominator a
    reader cannot locate.
    """
    n_a, n_b = result.n_a, result.n_b
    numerator = (n_a - 1) * result.sd_a**2 + (n_b - 1) * result.sd_b**2

    return math.sqrt(numerator / (n_a + n_b - 2))


def cohens_d_explanation(result: TwoGroupResult) -> FormulaExplanation:
    """
    Cohen's d for a two-group comparison, as concept then numbers.

    The concept line says "typical variation within a group" rather than
    "pooled standard deviation". What the denominator does is describe
    how much scores usually vary within a group, and naming that is more
    use to a reader than naming the statistic before they know what it is
    for. The statistic's own name is still there, in the term anatomy and
    in the notation.
    """
    pooled_sd = pooled_standard_deviation(result)

    return FormulaExplanation(
        name="Cohen's d",
        blocks=(
            "Difference between group means",
            "/",
            "Typical variation within a group",
            "=",
            "Standardized difference",
        ),
        substitution_template="({mean_a} - {mean_b}) / {pooled_sd}",
        formal_latex=r"d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}",
        terms=(
            FormulaTerm(
                key="mean_a",
                symbol="x̄₁",
                plain_name=f"{result.group_a_label} mean",
                meaning=(
                    f"The average outcome across the {result.n_a} "
                    f"{result.group_a_label} observations used."
                ),
                display_value=f"{result.mean_a:.2f}",
                source=f"the {result.group_a_label} mean shown above",
            ),
            FormulaTerm(
                key="mean_b",
                symbol="x̄₂",
                plain_name=f"{result.group_b_label} mean",
                meaning=(
                    f"The average outcome across the {result.n_b} "
                    f"{result.group_b_label} observations used."
                ),
                display_value=f"{result.mean_b:.2f}",
                source=f"the {result.group_b_label} mean shown above",
            ),
            FormulaTerm(
                key="pooled_sd",
                symbol="s_p",
                plain_name="pooled standard deviation",
                meaning=(
                    "How much scores typically vary within a group, "
                    "combining both groups' spread and weighting each by "
                    "its size. It is what turns a difference measured in "
                    "the outcome's own units into one comparable across "
                    "studies."
                ),
                display_value=f"{pooled_sd:.2f}",
                source=(
                    f"combined from both groups' spread, SD "
                    f"{result.sd_a:.2f} and {result.sd_b:.2f}"
                ),
            ),
        ),
        result_display=f"{result.cohens_d:.2f}",
        reading=(
            f"The two group means differ by {abs(result.cohens_d):.2f} "
            "within-group standard deviations. Conventional labels put 0.2 "
            "at small, 0.5 at medium and 0.8 at large (Cohen, 1988), which "
            "describe a magnitude rather than settle whether one matters "
            "here."
        ),
        citation=COHENS_D_CITATION,
    )


def did_explanation(result: DiDResult) -> FormulaExplanation:
    """
    The difference-in-differences estimate, as concept then numbers.

    Two subtractions, shown as two: each group's own change, then the
    difference between those changes. Writing it as one expression over
    four cell means is what makes the estimate look like a formula rather
    than like the thing it actually is.
    """
    return FormulaExplanation(
        name="Difference-in-differences",
        blocks=(
            f"{result.treated_label}'s change",
            "-",
            f"{result.comparison_label}'s change",
            "=",
            "Difference-in-differences",
        ),
        substitution_template="{change_treated} - {change_comparison}",
        formal_latex=(
            r"\text{DiD} = (\bar{y}_{T,\text{post}} - \bar{y}_{T,\text{pre}})"
            r" - (\bar{y}_{C,\text{post}} - \bar{y}_{C,\text{pre}})"
        ),
        terms=(
            FormulaTerm(
                key="change_treated",
                symbol="Δ_T",
                plain_name=f"{result.treated_label}'s change",
                meaning=(
                    f"How much {result.treated_label} moved between the two "
                    f"measurements: {result.mean_post_treated:.2f} minus "
                    f"{result.mean_pre_treated:.2f}."
                ),
                display_value=f"{result.change_treated:.2f}",
                source=f"the {result.treated_label} row's Change column",
            ),
            FormulaTerm(
                key="change_comparison",
                symbol="Δ_C",
                plain_name=f"{result.comparison_label}'s change",
                meaning=(
                    f"How much {result.comparison_label} moved over the same "
                    f"period without the program: "
                    f"{result.mean_post_comparison:.2f} minus "
                    f"{result.mean_pre_comparison:.2f}. This is what stands "
                    "in for what would have happened anyway."
                ),
                display_value=f"{result.change_comparison:.2f}",
                source=f"the {result.comparison_label} row's Change column",
            ),
        ),
        result_display=f"{result.did_estimate:.2f}",
        reading=(
            f"{result.treated_label} moved {result.did_estimate:.2f} beyond "
            f"what {result.comparison_label} moved over the same period. "
            "That subtraction removes anything affecting both groups "
            "equally. It does not establish that the program caused the "
            "remainder."
        ),
    )
