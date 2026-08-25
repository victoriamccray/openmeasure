"""
Unit tests for core/pulse_oximeter_demo.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import pulse_oximeter_demo as pod  # noqa: E402


class TestGenerateSyntheticCohort(unittest.TestCase):
    def test_reproduces_published_rate_within_rounding(self):
        # Hand-calculable against Sjoding et al. (2020): the deterministic
        # hypoxemic count is round(n * rate), so the generated rate must
        # match the cited rate to within one patient's worth of rounding.
        for cohort_id in ("michigan", "multicenter"):
            cohort = pod.generate_synthetic_cohort(cohort_id, seed=0)

            for group_rate in cohort.calibration_target:
                group_rows = cohort.data.loc[cohort.data["group"] == group_rate.group]
                self.assertEqual(len(group_rows), group_rate.n)

                actual_rate = group_rows["hypoxemic"].mean()
                self.assertAlmostEqual(actual_rate, group_rate.occult_hypoxemia_rate, delta=0.01)

    def test_readings_stay_within_band(self):
        cohort = pod.generate_synthetic_cohort("michigan", seed=1)

        self.assertTrue((cohort.data["device_reading"] >= pod.BAND_LOWER).all())
        self.assertTrue((cohort.data["device_reading"] < pod.BAND_UPPER).all())

    def test_unknown_cohort_raises(self):
        with self.assertRaises(ValueError):
            pod.generate_synthetic_cohort("not_a_real_cohort")

    def test_assumptions_are_disclosed(self):
        cohort = pod.generate_synthetic_cohort("michigan")
        self.assertGreaterEqual(len(cohort.assumptions), 2)


class TestEvaluateActionThreshold(unittest.TestCase):
    def test_higher_prevalence_group_has_lower_detection_rate_at_mid_threshold(self):
        # Black patients carry the higher published occult-hypoxemia rate
        # in both cohorts, and are modeled with hypoxemic readings
        # concentrated nearer the top of the band (module docstring), so
        # at a threshold midway through the band their detection rate
        # (true-positive rate) should be no better than White patients'.
        cohort = pod.generate_synthetic_cohort("michigan", seed=0)

        result = pod.evaluate_action_threshold(
            cohort, 94.0, privileged_group="White", unprivileged_group="Black"
        )

        self.assertLessEqual(
            result.unprivileged_true_positive_rate, result.privileged_true_positive_rate
        )

    def test_threshold_at_or_below_band_lower_raises(self):
        cohort = pod.generate_synthetic_cohort("michigan")
        with self.assertRaises(ValueError):
            pod.evaluate_action_threshold(
                cohort, pod.BAND_LOWER, privileged_group="White", unprivileged_group="Black"
            )

    def test_threshold_at_or_above_band_upper_raises(self):
        cohort = pod.generate_synthetic_cohort("michigan")
        with self.assertRaises(ValueError):
            pod.evaluate_action_threshold(
                cohort, pod.BAND_UPPER, privileged_group="White", unprivileged_group="Black"
            )

    def test_result_covers_the_whole_cohort(self):
        cohort = pod.generate_synthetic_cohort("multicenter", seed=2)
        result = pod.evaluate_action_threshold(
            cohort, 94.0, privileged_group="White", unprivileged_group="Black"
        )

        expected_white_n = next(
            gr.n for gr in cohort.calibration_target if gr.group == "White"
        )
        expected_black_n = next(
            gr.n for gr in cohort.calibration_target if gr.group == "Black"
        )
        self.assertEqual(result.privileged_n, expected_white_n)
        self.assertEqual(result.unprivileged_n, expected_black_n)


if __name__ == "__main__":
    unittest.main()
