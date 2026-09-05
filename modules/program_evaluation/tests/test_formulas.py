"""
Unit tests for core/formulas.py

The load-bearing checks here are the drift ones: that the arithmetic an
explanation shows really does produce the statistic the analysis
computed. An explanation that renders a formula which does not give its
own answer is worse than no explanation, because it looks authoritative.

Run with: pytest modules/program_evaluation/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import comparison, did, formulas  # noqa: E402


def _two_group_frame(seed=11, n=40):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "arm": ["treated"] * n + ["control"] * n,
            "score": np.r_[rng.normal(70, 12, n), rng.normal(60, 9, n)],
        }
    )


def _did_frame(seed=5, n=30):
    rng = np.random.RandomState(seed)
    pre_t = rng.normal(50, 8, n)
    pre_c = rng.normal(48, 8, n)
    return pd.DataFrame(
        {
            "arm": ["treated"] * n + ["control"] * n,
            "pre": np.r_[pre_t, pre_c],
            "post": np.r_[
                pre_t + rng.normal(6, 4, n), pre_c + rng.normal(2, 4, n)
            ],
        }
    )


class TestPooledStandardDeviation(unittest.TestCase):
    def test_matches_a_hand_calculation(self):
        """
        Two groups of 3 with SD 2 and 4: pooled = sqrt((2*4 + 2*16)/4) =
        sqrt(10).
        """
        result = comparison.compare_two_groups(
            pd.DataFrame(
                {
                    "arm": ["a", "a", "a", "b", "b", "b"],
                    "score": [8.0, 10.0, 12.0, 16.0, 20.0, 24.0],
                }
            ),
            "arm",
            "score",
        )

        self.assertAlmostEqual(result.sd_a, 2.0, places=12)
        self.assertAlmostEqual(result.sd_b, 4.0, places=12)
        self.assertAlmostEqual(
            formulas.pooled_standard_deviation(result), np.sqrt(10.0), places=12
        )

    def test_reproduces_the_denominator_cohens_d_actually_used(self):
        """
        comparison.py computes d from the raw samples and never stores the
        pooled value. If this reconstruction drifted, the explanation
        would show a denominator the statistic was not divided by.
        """
        result = comparison.compare_two_groups(_two_group_frame(), "arm", "score")
        pooled = formulas.pooled_standard_deviation(result)

        self.assertAlmostEqual(
            (result.mean_a - result.mean_b) / pooled,
            result.cohens_d,
            places=12,
        )


class TestCohensDExplanation(unittest.TestCase):
    def setUp(self):
        self.result = comparison.compare_two_groups(
            _two_group_frame(), "arm", "score"
        )
        self.explanation = formulas.cohens_d_explanation(self.result)

    def test_the_shown_result_is_the_computed_statistic(self):
        self.assertEqual(
            self.explanation.result_display, f"{self.result.cohens_d:.2f}"
        )

    def test_the_shown_arithmetic_lands_on_the_shown_result(self):
        """
        Evaluates the displayed numbers, not the underlying ones. This is
        what catches an explanation whose rounded values visibly fail to
        produce its own answer.
        """
        mean_a = float(self.explanation.term("mean_a").display_value)
        mean_b = float(self.explanation.term("mean_b").display_value)
        pooled = float(self.explanation.term("pooled_sd").display_value)

        self.assertAlmostEqual(
            (mean_a - mean_b) / pooled,
            float(self.explanation.result_display),
            places=2,
        )

    def test_the_concept_line_carries_no_notation(self):
        for symbol in ("x̄", "s_p", "/frac", "\\"):
            with self.subTest(symbol=symbol):
                self.assertNotIn(symbol, self.explanation.concept)

    def test_the_group_labels_appear_in_the_terms(self):
        names = [term.plain_name for term in self.explanation.terms]

        self.assertIn(f"{self.result.group_a_label} mean", names)
        self.assertIn(f"{self.result.group_b_label} mean", names)

    def test_every_term_says_where_its_value_came_from(self):
        for term in self.explanation.terms:
            with self.subTest(term=term.key):
                self.assertTrue(term.source.strip())

    def test_the_reading_cites_the_convention_it_uses(self):
        self.assertIn("Cohen, 1988", self.explanation.reading)
        self.assertIn("Cohen, J. (1988)", self.explanation.citation)

    def test_a_negative_effect_still_reads_as_a_magnitude(self):
        """
        compare_two_groups sorts its labels, so "control" is group A here
        and d comes out negative even though the treated group scored
        higher. The reading describes the size of the gap, so it quotes
        the absolute value while result_display keeps the sign.
        """
        self.assertLess(self.result.cohens_d, 0)
        self.assertIn(
            f"{abs(self.result.cohens_d):.2f}", self.explanation.reading
        )
        self.assertTrue(self.explanation.result_display.startswith("-"))


class TestDidExplanation(unittest.TestCase):
    def setUp(self):
        self.result = did.estimate_did(
            _did_frame(), "arm", "pre", "post", treated_label="treated"
        )
        self.explanation = formulas.did_explanation(self.result)

    def test_the_shown_result_is_the_computed_estimate(self):
        self.assertEqual(
            self.explanation.result_display, f"{self.result.did_estimate:.2f}"
        )

    def test_the_shown_arithmetic_lands_on_the_shown_result(self):
        treated = float(self.explanation.term("change_treated").display_value)
        comparison_change = float(
            self.explanation.term("change_comparison").display_value
        )

        self.assertAlmostEqual(
            treated - comparison_change,
            float(self.explanation.result_display),
            places=2,
        )

    def test_it_is_shown_as_two_subtractions_not_four_means(self):
        """
        Each group's own change first, then the difference between them.
        Writing it over four cell means makes the estimate look like a
        formula rather than like what it is.
        """
        self.assertEqual(len(self.explanation.terms), 2)
        self.assertIn("change", self.explanation.substitution_template)

    def test_the_group_labels_are_used_rather_than_treated_and_control(self):
        self.assertIn(self.result.treated_label, self.explanation.concept)
        self.assertIn(self.result.comparison_label, self.explanation.concept)

    def test_the_reading_declines_to_claim_causation(self):
        self.assertIn("does not establish", self.explanation.reading)

    def test_the_comparison_term_names_what_it_stands_in_for(self):
        meaning = self.explanation.term("change_comparison").meaning

        self.assertIn("would have happened anyway", meaning)


class TestExplanationsAreValidForRealResults(unittest.TestCase):
    """
    Both builders construct a FormulaExplanation, whose __post_init__
    rejects an unusable one. Building from real results is what confirms
    the templates and terms agree in practice, not only in a fixture.
    """

    def test_cohens_d_builds_across_several_datasets(self):
        for seed in (0, 3, 11, 42):
            with self.subTest(seed=seed):
                result = comparison.compare_two_groups(
                    _two_group_frame(seed=seed), "arm", "score"
                )
                explanation = formulas.cohens_d_explanation(result)

                self.assertTrue(explanation.substituted)
                self.assertNotIn("{", explanation.substituted)

    def test_did_builds_across_several_datasets(self):
        for seed in (0, 3, 11, 42):
            with self.subTest(seed=seed):
                result = did.estimate_did(
                    _did_frame(seed=seed),
                    "arm",
                    "pre",
                    "post",
                    treated_label="treated",
                )
                explanation = formulas.did_explanation(result)

                self.assertTrue(explanation.substituted)
                self.assertNotIn("{", explanation.substituted)


if __name__ == "__main__":
    unittest.main()
