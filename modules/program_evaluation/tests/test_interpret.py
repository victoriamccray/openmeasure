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


if __name__ == "__main__":
    unittest.main()
