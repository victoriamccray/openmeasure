"""
Impact Evaluation results as Decision Records.

Maps what this module already computes and already says onto the shared
record in shared/decision_record.py. Nothing here changes how an analysis
runs or what the page shows. Where the module has no content for a field,
the field is left as the specification allows (an empty list, an omitted
optional value) rather than filled with new claims, and content the schema
has no field for goes in ``unplaced``, which is the evidence the v0.1
schema exists to collect.

Mapped so far: difference-in-differences (estimate_did) and Welch's
two-group comparison (compare_two_groups). They sit at the two ends of this
module: DiD has a full plain-language reading in interpret.py, and the
two-group comparison's interpretation still lives mostly in the page. Every
other method raises, so an unmapped method never produces a record with
silently missing parts.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.decision_record import (  # noqa: E402
    ORIGIN_RESEARCHER,
    ORIGIN_SUGGESTION_ACCEPTED,
    STATUS_HELD_BY_DESIGN,
    STATUS_NOT_CHECKED,
    STATUS_NOT_TESTABLE,
    STATUS_REQUIRES_SETTING,
    Alternative,
    Convention,
    DataHandling,
    Decision,
    DecisionRecord,
    EvidenceItem,
    Interval,
    Method,
    Provenance,
    Quantity,
    Question,
    Reading,
    RecordedAssumption,
    ResearcherChoice,
    Result,
    RetentionEntry,
    Software,
    Unplaced,
)

from . import interpret  # noqa: E402
from .comparison import TwoGroupResult  # noqa: E402
from .did import DiDResult  # noqa: E402
from .recommend import MethodRecommendation  # noqa: E402

MAPPED_METHODS = ("compare_two_groups", "estimate_did")

COMPONENT = "Impact Evaluation"

# interpret.py treats 0.05 as a convention and names it, but cites no
# source for it. A record's reading must, so the citation is added here.
ALPHA_CONVENTION = Convention(
    name="Conventional significance threshold",
    threshold=interpret.CONVENTIONAL_ALPHA,
    citation=(
        "Fisher, R. A. (1925). Statistical Methods for Research Workers. "
        "Edinburgh: Oliver and Boyd."
    ),
)

# DiD assumptions say what the data *can* tell about them; a record says
# what is known. With nothing checked on the page, each maps directly.
_CHECKABLE_TO_STATUS = {
    interpret.CHECK_BY_DESIGN: STATUS_HELD_BY_DESIGN,
    interpret.CHECK_NOT_TESTABLE_HERE: STATUS_NOT_TESTABLE,
    interpret.CHECK_FROM_CONTEXT: STATUS_REQUIRES_SETTING,
}

# The conditions a causal reading of a two-group difference rests on, named
# as support_boundary_conditions("compare_two_groups") names them so the
# record and the page's boundary diagram cannot drift apart (pinned by a
# test). interpret.py carries only the short names for these; the
# statements are written here because a record needs them and the
# recommender's warning states all three in one sentence.
_GROUP_COMPARISON_CAUSAL_ASSUMPTIONS = (
    RecordedAssumption(
        name="Baseline imbalance",
        statement=(
            "The groups would have had the same average outcome without "
            "the program. If they already differed at the start, the "
            "difference measured afterwards includes that starting gap."
        ),
        status=STATUS_NOT_TESTABLE,
        citation=(
            "Shadish, W. R., Cook, T. D., & Campbell, D. T. (2002). "
            "Experimental and Quasi-Experimental Designs for Generalized "
            "Causal Inference (selection as a threat to internal validity)."
        ),
    ),
    RecordedAssumption(
        name="Selection",
        statement=(
            "Who ended up in each group did not depend on anything that "
            "also affects the outcome. When people choose the program, or "
            "are chosen for it, the groups can differ for that reason alone."
        ),
        status=STATUS_REQUIRES_SETTING,
        citation=(
            "Shadish, W. R., Cook, T. D., & Campbell, D. T. (2002). "
            "Experimental and Quasi-Experimental Designs for Generalized "
            "Causal Inference (selection as a threat to internal validity)."
        ),
    ),
    RecordedAssumption(
        name="Unmeasured confounding",
        statement=(
            "No factor missing from these data influences both group "
            "membership and the outcome. A factor like that produces a "
            "difference between the groups whether or not the program had "
            "any effect."
        ),
        status=STATUS_REQUIRES_SETTING,
        citation="Rosenbaum, P. R. (2002). Observational Studies (2nd ed.). Springer.",
    ),
)

_WELCH_CITATION = (
    "Welch, B. L. (1947). The generalization of \"Student's\" problem when "
    "several different population variances are involved. Biometrika, "
    "34(1-2), 28-35."
)

_WARNING_WHY = (
    "The recommender's warnings mix caveats about the design, which this "
    "record also states as assumptions, with checks on the selected "
    "columns. No field holds a warning about the inputs themselves."
)


def build_decision_record(
    *,
    record_id: str,
    result: object,
    recommendation: MethodRecommendation,
    columns: Mapping[str, str],
    dataset: EvidenceItem,
    software_version: str,
    generated_at: str,
    researcher_choices: tuple[ResearcherChoice, ...] = (),
    question: Question | None = None,
) -> DecisionRecord:
    """
    Build the Decision Record for one Impact Evaluation analysis.

    ``columns`` is the page's analysis context: the keys ending in
    ``_col`` name the columns the analysis used. ``researcher_choices``
    comes from the caller because only the page knows whether a column
    was picked by the researcher or left at the page's suggestion.

    ``question`` is the researcher's own, from evaluation_question(). When
    it is None, the record states the question the chosen analysis
    answers, with no origin, since nobody stated it.

    Raises ValueError for a method that is not mapped yet, and TypeError
    when the result does not belong to the recommended method.
    """
    method = recommendation.method

    if method == "compare_two_groups":
        if not isinstance(result, TwoGroupResult):
            raise TypeError(
                "compare_two_groups needs a TwoGroupResult, got "
                f"{type(result).__name__}."
            )
        parts = _two_group_parts(result, columns)
    elif method == "estimate_did":
        if not isinstance(result, DiDResult):
            raise TypeError(
                f"estimate_did needs a DiDResult, got {type(result).__name__}."
            )
        parts = _did_parts(result, columns)
    else:
        raise ValueError(
            f"'{method}' is not mapped onto the Decision Record yet. Mapped "
            f"methods: {', '.join(MAPPED_METHODS)}."
        )

    implied_question, method_record, assumptions, result_record, supports, does_not, alternatives, extra = parts

    handling = DataHandling(
        evidence_label=dataset.label,
        columns_used=_columns_used(columns),
        n_input_rows=result.n_input_rows,
        n_retained_rows=result.n_rows_used,
        retention=(
            RetentionEntry(
                label="Rows excluded",
                count=result.n_excluded_rows,
                kind="rows_dropped",
                mechanism=result.exclusion_reason,
            ),
        ),
    )

    unplaced = tuple(
        Unplaced(content=warning, why_no_field_fits=_WARNING_WHY)
        for warning in recommendation.warnings
    ) + extra

    return DecisionRecord(
        record_id=record_id,
        question=question if question is not None else implied_question,
        evidence=(dataset,),
        decision=Decision(
            statement=f"Analyze with {recommendation.display_name}.",
            # The page offers only the recommended method, and running it
            # is the researcher accepting that suggestion.
            origin=ORIGIN_SUGGESTION_ACCEPTED,
            alternatives=alternatives,
        ),
        method=method_record,
        assumptions=assumptions,
        result=result_record,
        supports=supports,
        does_not_support=does_not,
        provenance=Provenance(
            software=Software(
                name="OpenMeasure",
                version=software_version,
                component=COMPONENT,
            ),
            generated_at=generated_at,
            data_handling=(handling,),
            researcher_choices=researcher_choices,
        ),
        unplaced=unplaced,
    )


# The Evaluation Question stage's fields, in the order the stage asks
# for them and with its own labels.
_QUESTION_FIELDS = (
    ("program", "Program or intervention"),
    ("population", "Population"),
    ("outcome", "Outcome"),
    ("comparison", "Comparison"),
    ("timing", "Timing"),
)


def evaluation_question(entered: Mapping[str, str]) -> Question | None:
    """
    The researcher's evaluation question, as they entered it.

    Joined field by field with the stage's own labels rather than
    rewritten into a sentence, so the record never words the question
    differently from how the researcher did. Returns None when nothing
    was entered.
    """
    values = {key: str(entered.get(key) or "").strip().rstrip(".") for key, _ in _QUESTION_FIELDS}
    parts = [f"{label}: {values[key]}" for key, label in _QUESTION_FIELDS if values[key]]

    if not parts:
        return None

    return Question(text=". ".join(parts) + ".", origin=ORIGIN_RESEARCHER)


def _columns_used(columns: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        str(value) for key, value in columns.items() if key.endswith("_col") and value
    )


def _quantity(name: str, value: float, **extra) -> Quantity:
    """A quantity, with a non-finite value recorded as null and explained."""

    if value is None or not math.isfinite(value):
        return Quantity(
            name=name,
            value=None,
            value_note=f"The computation returned {value}, which is not a finite number.",
            **extra,
        )
    return Quantity(name=name, value=float(value), **extra)


def _p_value_reading(p_value: float) -> tuple[Reading, ...]:
    if not math.isfinite(p_value):
        return ()

    alpha = interpret.CONVENTIONAL_ALPHA
    text = (
        f"Below the conventional threshold of {alpha}."
        if p_value < alpha
        else f"At or above the conventional threshold of {alpha}."
    )
    return (Reading(quantity="p-value", text=text, convention=ALPHA_CONVENTION),)


def _two_group_parts(result: TwoGroupResult, columns: Mapping[str, str]):
    outcome = columns.get("outcome_col", "the outcome")
    a, b = result.group_a_label, result.group_b_label
    detected = result.p_value < interpret.CONVENTIONAL_ALPHA

    question = Question(text=f"Did the {a} and {b} groups differ on mean {outcome}?")

    method = Method(
        name="Welch's independent-samples t-test",
        identifier="compare_two_groups",
        parameters={
            "equal_variances_assumed": False,
            "alternative": "two-sided",
            "confidence_level": 0.95,
        },
        citations=(_WELCH_CITATION,),
    )

    # The statistical conditions come first, then the causal ones. The
    # normality wording is the page's own, from its Mann-Whitney comparison.
    assumptions = (
        RecordedAssumption(
            name="Independent observations",
            statement=(
                "Each row is a different unit, and no unit's outcome "
                "depends on another's."
            ),
            status=STATUS_REQUIRES_SETTING,
            citation=_WELCH_CITATION,
        ),
        RecordedAssumption(
            name="Approximately normal outcome within each group",
            statement=(
                "The outcome is roughly normally distributed within each "
                "group."
            ),
            status=STATUS_NOT_CHECKED,
            citation=_WELCH_CITATION,
        ),
    ) + _GROUP_COMPARISON_CAUSAL_ASSUMPTIONS

    interval = Interval(
        low=result.ci_95_low,
        high=result.ci_95_high,
        level=0.95,
        method="Welch-Satterthwaite degrees of freedom",
    )
    quantities = (
        _quantity(f"Mean {outcome}, {a}", result.mean_a, n=result.n_a),
        _quantity(f"Mean {outcome}, {b}", result.mean_b, n=result.n_b),
        _quantity(
            f"Difference in means ({a} minus {b})",
            result.mean_difference,
            interval=interval,
            n=result.n_a + result.n_b,
        ),
        _quantity("t-statistic", result.t_statistic),
        _quantity("Degrees of freedom", result.degrees_of_freedom),
        _quantity("p-value", result.p_value),
        _quantity("Cohen's d", result.cohens_d),
    )

    summary = (
        f"The {a} group's mean {outcome} was {result.mean_a:.2f} (n = "
        f"{result.n_a}) and the {b} group's was {result.mean_b:.2f} (n = "
        f"{result.n_b}), a difference of {result.mean_difference:.2f} (95% "
        f"CI {result.ci_95_low:.2f} to {result.ci_95_high:.2f})."
    )

    if detected:
        supports = (
            f"A difference in mean {outcome} between the {a} and {b} groups "
            f"of about {result.mean_difference:.2f}, under Welch's t-test "
            "assumptions.",
        )
    else:
        supports = (
            "That no difference was detected at this sample size, which is "
            "a different claim from the groups having the same mean.",
            f"The interval, {result.ci_95_low:.2f} to "
            f"{result.ci_95_high:.2f}, as the range of mean differences "
            "these data are compatible with.",
        )

    does_not_support = (
        "A causal claim about the program. Whether a difference, or the "
        "lack of one, can be attributed to the program depends on the "
        "study design, not on the p-value.",
        f"A claim about the groups at any other time. The outcome, "
        f"{outcome}, was measured once.",
    )

    alternatives = (
        Alternative(
            option="Mann-Whitney U test",
            tradeoff=(
                "Compares ranks instead of means and makes no "
                "distributional assumption. It tests a different "
                "hypothesis and can reach a different conclusion. The page "
                "shows its result beside this one."
            ),
        ),
    )

    result_record = Result(
        summary=summary,
        quantities=quantities,
        readings=_p_value_reading(result.p_value),
    )

    return (
        question,
        method,
        assumptions,
        result_record,
        supports,
        does_not_support,
        alternatives,
        (),
    )


def _did_parts(result: DiDResult, columns: Mapping[str, str]):
    pre = columns.get("pre_col", "the baseline")
    post = columns.get("post_col", "the follow-up")
    treated, comparison = result.treated_label, result.comparison_label
    reading = interpret.interpret_did(result)

    question = Question(
        text=(
            f"Did the {treated} group change more between {pre} and {post} "
            f"than the {comparison} group did?"
        )
    )

    method = Method(
        name="Difference-in-differences (2x2)",
        identifier="estimate_did",
        parameters={
            "standard_error_method": result.standard_error_method,
            "confidence_level": 0.95,
        },
        citations=(
            "Angrist, J. D., & Pischke, J.-S. (2009). Mostly Harmless "
            "Econometrics, ch. 5.",
        ),
    )

    assumptions = tuple(
        RecordedAssumption(
            name=assumption.name,
            statement=assumption.statement,
            status=_CHECKABLE_TO_STATUS[assumption.checkable],
            citation=assumption.citation,
        )
        for assumption in interpret.did_assumptions()
    )

    quantities = (
        _quantity(f"Mean {pre}, {treated}", result.mean_pre_treated, n=result.n_treated),
        _quantity(f"Mean {post}, {treated}", result.mean_post_treated, n=result.n_treated),
        _quantity(f"Mean {pre}, {comparison}", result.mean_pre_comparison, n=result.n_comparison),
        _quantity(f"Mean {post}, {comparison}", result.mean_post_comparison, n=result.n_comparison),
        _quantity(f"Change, {treated}", result.change_treated, n=result.n_treated),
        _quantity(f"Change, {comparison}", result.change_comparison, n=result.n_comparison),
        _quantity(
            "Difference-in-differences estimate",
            result.did_estimate,
            interval=Interval(
                low=result.ci_95_low,
                high=result.ci_95_high,
                level=0.95,
                method=result.standard_error_method,
            ),
            n=result.n_treated + result.n_comparison,
        ),
        _quantity("Standard error", result.standard_error),
        _quantity("t-statistic", result.t_statistic),
        _quantity("Degrees of freedom", result.degrees_of_freedom),
        _quantity("p-value", result.p_value),
        _quantity(f"Baseline difference ({treated} minus {comparison})", result.baseline_difference),
    )

    result_record = Result(
        summary=reading.headline,
        quantities=quantities,
        readings=_p_value_reading(result.p_value),
        observations=reading.observations,
    )

    # The estimand says which population and quantity the estimate refers
    # to. It is neither the result nor a claim the result supports, and the
    # schema has nowhere else to put it.
    estimand = (
        Unplaced(
            content=reading.estimand,
            why_no_field_fits=(
                "The estimand, the population and quantity the estimate "
                "refers to, is not the result and not a conclusion it "
                "supports. v0.1 has no field for it."
            ),
        ),
    )

    return (
        question,
        method,
        assumptions,
        result_record,
        reading.supports,
        reading.does_not_support,
        (),
        estimand,
    )
