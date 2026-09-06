"""
Unit tests for core/interpret.py

Run with: pytest modules/program_evaluation/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import did, interpret  # noqa: E402


def _result_with(
    *,
    treated_change: float,
    comparison_change: float,
    n: int = 40,
    noise: float = 3.0,
    seed: int = 3,
) -> did.DiDResult:
    """Build a DiDResult with a known direction of effect."""
    rng = np.random.RandomState(seed)
    pre_treated = rng.normal(50, 5, n)
    pre_comparison = rng.normal(50, 5, n)

    frame = pd.DataFrame(
        {
            "arm": ["treated"] * n + ["comparison"] * n,
            "pre": np.r_[pre_treated, pre_comparison],
            "post": np.r_[
                pre_treated + treated_change + rng.normal(0, noise, n),
                pre_comparison + comparison_change + rng.normal(0, noise, n),
            ],
        }
    )
    return did.estimate_did(frame, "arm", "pre", "post", treated_label="treated")


class TestAssumptionValidation(unittest.TestCase):
    def test_rejects_an_unknown_checkable_state(self):
        with self.assertRaises(ValueError) as raised:
            interpret.Assumption(
                name="Parallel trends",
                statement="Something.",
                checkable="Probably fine",
                citation="Someone (2020).",
            )

        self.assertIn("not", str(raised.exception))

    def test_rejects_a_missing_citation(self):
        with self.assertRaises(ValueError) as raised:
            interpret.Assumption(
                name="Parallel trends",
                statement="Something.",
                checkable=interpret.CHECK_NOT_TESTABLE_HERE,
                citation="",
            )

        self.assertIn("citation", str(raised.exception))


class TestDidAssumptions(unittest.TestCase):
    def test_parallel_trends_comes_first_and_is_not_testable(self):
        assumptions = interpret.did_assumptions()

        self.assertEqual(assumptions[0].name, "Parallel trends")
        self.assertEqual(
            assumptions[0].checkable, interpret.CHECK_NOT_TESTABLE_HERE
        )

    def test_every_assumption_carries_a_citation(self):
        for assumption in interpret.did_assumptions():
            with self.subTest(assumption=assumption.name):
                self.assertTrue(assumption.citation)

    def test_spillover_and_stable_composition_are_both_stated(self):
        names = {a.name for a in interpret.did_assumptions()}

        self.assertIn("No spillover between groups", names)
        self.assertIn("Stable composition", names)

    def test_stable_composition_is_held_by_the_data_shape(self):
        """
        One row per unit with both measurements settles this one, and
        saying so is what keeps the other four from reading as boilerplate.
        """
        composition = next(
            a for a in interpret.did_assumptions() if a.name == "Stable composition"
        )

        self.assertEqual(composition.checkable, interpret.CHECK_BY_DESIGN)


class TestInterpretDid(unittest.TestCase):
    def test_names_the_treated_group_as_the_estimand(self):
        reading = interpret.interpret_did(
            _result_with(treated_change=8.0, comparison_change=1.0)
        )

        self.assertIn("treated", reading.estimand)
        self.assertIn("comparison", reading.estimand)

    def test_headline_reflects_a_detected_difference(self):
        reading = interpret.interpret_did(
            _result_with(treated_change=10.0, comparison_change=0.0)
        )

        self.assertIn("larger than sampling variation", reading.headline)

    def test_headline_reflects_no_detected_difference(self):
        reading = interpret.interpret_did(
            _result_with(treated_change=2.0, comparison_change=2.0)
        )

        self.assertIn("within what sampling variation", reading.headline)

    def test_alpha_is_named_in_the_headline_rather_than_assumed(self):
        result = _result_with(treated_change=10.0, comparison_change=0.0)

        self.assertIn("0.01", interpret.interpret_did(result, alpha=0.01).headline)
        self.assertIn("0.05", interpret.interpret_did(result).headline)

    def test_rejects_an_alpha_outside_zero_and_one(self):
        result = _result_with(treated_change=5.0, comparison_change=0.0)

        for bad_alpha in (0.0, 1.0, -0.5, 2.0):
            with self.subTest(alpha=bad_alpha):
                with self.assertRaises(ValueError):
                    interpret.interpret_did(result, alpha=bad_alpha)

    def test_the_parallel_trends_caveat_is_always_present(self):
        """
        A detected difference is exactly when a reader is most likely to
        drop the assumption, so it must appear in both branches.
        """
        for treated_change, comparison_change in ((10.0, 0.0), (2.0, 2.0)):
            with self.subTest(treated_change=treated_change):
                reading = interpret.interpret_did(
                    _result_with(
                        treated_change=treated_change,
                        comparison_change=comparison_change,
                    )
                )
                self.assertTrue(
                    any(
                        "parallel trends" in item.lower()
                        for item in reading.does_not_support
                    )
                )

    def test_a_moving_comparison_group_is_reported_with_the_naive_number(self):
        reading = interpret.interpret_did(
            _result_with(treated_change=10.0, comparison_change=6.0)
        )

        self.assertTrue(
            any("pre/post comparison" in item for item in reading.observations)
        )

    def test_excluded_rows_are_surfaced_as_an_observation(self):
        frame = pd.DataFrame(
            {
                "arm": ["treated"] * 6 + ["comparison"] * 6,
                "pre": [10.0, 12.0, 14.0, 11.0, 13.0, 15.0] * 2,
                "post": [
                    18.0, 21.0, 23.0, 19.0, 22.0, 25.0,
                    12.0, 13.0, 16.0, 12.0, 15.0, np.nan,
                ],
            }
        )
        result = did.estimate_did(
            frame, "arm", "pre", "post", treated_label="treated"
        )
        reading = interpret.interpret_did(result)

        self.assertEqual(result.n_excluded_rows, 1)
        self.assertTrue(
            any("were excluded" in item for item in reading.observations)
        )


class TestPValueNote(unittest.TestCase):
    """
    The page shows a p-value in every method branch and never said what
    one was. This is the one definition every analysis reaches, so its
    wording matters more than most.
    """

    def test_it_is_phrased_as_a_frequency_of_results(self):
        """
        The correct reading is how often data like this would turn up
        under a scenario, not a probability attached to a hypothesis.
        """
        self.assertIn("how often", interpret.P_VALUE_NOTE)

    def test_it_rules_out_the_three_common_misreadings(self):
        note = interpret.P_VALUE_NOTE.lower()

        self.assertIn("not the chance the finding is wrong", note)
        self.assertIn("how large", note)
        self.assertIn("caused it", note)

    def test_it_never_calls_a_p_value_a_probability_about_a_hypothesis(self):
        """
        Guards the phrasings that make a p-value sound like the
        probability the null is true, which is the misreading this note
        exists to head off.
        """
        note = interpret.P_VALUE_NOTE.lower()

        for wrong in (
            "probability that the null",
            "probability the null",
            "chance that there is no difference",
            "probability the groups are the same",
        ):
            with self.subTest(phrase=wrong):
                self.assertNotIn(wrong, note)


class TestSupportBoundaryConditions(unittest.TestCase):
    """
    The names drawn on the stage 7 boundary diagram.

    These were the first sentence of each recommender warning, which is
    150 to 170 characters. SVG text does not wrap, so those did not
    truncate, they ran off the canvas: the conditions were invisible on
    every design except difference-in-differences.
    """

    # Every method the recommender can return with supported=True, and so
    # every method that can reach the interpretation stage. Listed
    # literally so adding one to the recommender without a boundary entry
    # fails here rather than in front of a reader.
    ANALYSABLE_METHODS = (
        "compare_two_groups",
        "compare_multiple_groups_welch",
        "compare_categorical",
        "compare_pre_post",
        "sensitivity_analysis",
        "estimate_did",
    )

    def test_every_analysable_method_has_conditions(self):
        for method in self.ANALYSABLE_METHODS:
            with self.subTest(method=method):
                self.assertTrue(interpret.support_boundary_conditions(method))

    def test_every_name_fits_the_figure_that_draws_it(self):
        for method in self.ANALYSABLE_METHODS:
            for name in interpret.support_boundary_conditions(method):
                with self.subTest(method=method, name=name):
                    self.assertLessEqual(
                        len(name), interpret.SUPPORT_BOUNDARY_LABEL_LIMIT
                    )

    def test_a_name_is_a_short_label_not_a_sentence(self):
        for method in self.ANALYSABLE_METHODS:
            for name in interpret.support_boundary_conditions(method):
                with self.subTest(method=method, name=name):
                    self.assertNotIn(".", name)

    def test_did_takes_its_conditions_from_its_assumptions(self):
        """
        One list, so the branches on the diagram and the statements
        printed under it cannot drift apart.
        """
        self.assertEqual(
            interpret.support_boundary_conditions("estimate_did"),
            tuple(a.name for a in interpret.did_assumptions()),
        )

    def test_an_unknown_method_raises_and_names_the_known_ones(self):
        with self.assertRaises(ValueError) as raised:
            interpret.support_boundary_conditions("regression_discontinuity")

        message = str(raised.exception)
        self.assertIn("regression_discontinuity", message)
        self.assertIn("estimate_did", message)

    def test_the_names_use_the_words_the_warnings_already_use(self):
        """
        Nothing in the mapping is a new claim about a design. Each name
        is a word the recommender's own warning for that design already
        uses, so the branch label and the sentence it points at agree.
        """
        import pandas as pd

        from core import recommend

        frame = pd.DataFrame(
            {
                "pre": [1.0, 2, 3, 4, 5, 6, 7, 9],
                "post": [2.0, 3, 4, 5, 6, 7, 8, 11],
                "y": [1.0, 2, 3, 4, 5, 6, 7, 9],
                "g": list("aaaabbbb"),
            }
        )
        cases = (
            (dict(pre_col="pre", post_col="post"), "compare_pre_post"),
            (dict(outcome_col="y", group_col="g"), "compare_two_groups"),
        )

        for columns, expected_method in cases:
            recommendation = recommend.recommend_method(frame, **columns)
            self.assertEqual(recommendation.method, expected_method)
            warning_text = " ".join(recommendation.warnings).lower()

            for name in interpret.support_boundary_conditions(expected_method):
                with self.subTest(method=expected_method, name=name):
                    self.assertIn(name.lower(), warning_text)


if __name__ == "__main__":
    unittest.main()
