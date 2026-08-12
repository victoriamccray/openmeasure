"""
Unit tests for core/acquisition_robustness.py

Run with: pytest modules/healthring/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core import acquisition_robustness as ar  # noqa: E402


def _make_df(rows):
    """
    rows is a list of dicts with subject_id, Label, hr, bvp_hr, ir-quality,
    red-quality.
    """
    return pd.DataFrame(rows)


# Eight hand-built windows across three subjects and two conditions.
# "rest" windows have a constant +2 bpm ring bias; "walk" windows have a
# constant +8 bpm ring bias, so per-condition MAE/bias are hand-calculable
# and a pooled statistic (mean bias 5.0) visibly hides the per-condition
# spread (2.0 vs 8.0).
_ROWS = [
    {"subject_id": 0, "Label": "rest", "hr": 60, "bvp_hr": 62, "ir-quality": 0.9, "red-quality": 0.9},
    {"subject_id": 0, "Label": "rest", "hr": 62, "bvp_hr": 64, "ir-quality": 0.8, "red-quality": 0.8},
    {"subject_id": 0, "Label": "walk", "hr": 100, "bvp_hr": 108, "ir-quality": 0.4, "red-quality": 0.4},
    {"subject_id": 0, "Label": "walk", "hr": 102, "bvp_hr": 110, "ir-quality": 0.3, "red-quality": 0.3},
    {"subject_id": 1, "Label": "rest", "hr": 61, "bvp_hr": 63, "ir-quality": 0.85, "red-quality": 0.85},
    {"subject_id": 1, "Label": "walk", "hr": 101, "bvp_hr": 109, "ir-quality": 0.35, "red-quality": 0.35},
    {"subject_id": 2, "Label": "rest", "hr": 59, "bvp_hr": 61, "ir-quality": 0.95, "red-quality": 0.95},
    {"subject_id": 2, "Label": "walk", "hr": 99, "bvp_hr": 107, "ir-quality": 0.45, "red-quality": 0.45},
]


class TestPrepareWindows(unittest.TestCase):
    def test_derives_error_and_quality_columns(self):
        windows = ar.prepare_windows(_make_df(_ROWS))

        self.assertEqual(windows.n_input_windows, 8)
        self.assertEqual(windows.n_usable_windows, 8)
        self.assertEqual(windows.n_excluded_windows, 0)
        self.assertEqual(windows.condition_order, ("rest", "walk"))

        first = windows.data.iloc[0]
        self.assertAlmostEqual(first["abs_error"], 2.0)
        self.assertAlmostEqual(first["signed_error"], 2.0)
        self.assertAlmostEqual(first["mean_hr"], 61.0)
        self.assertAlmostEqual(first["quality"], 0.9)

    def test_rows_missing_hr_bvp_hr_or_label_are_dropped(self):
        rows = list(_ROWS) + [
            {"subject_id": 3, "Label": "rest", "hr": None, "bvp_hr": 61, "ir-quality": 0.9, "red-quality": 0.9}
        ]

        windows = ar.prepare_windows(_make_df(rows))

        self.assertEqual(windows.n_input_windows, 9)
        self.assertEqual(windows.n_usable_windows, 8)
        self.assertEqual(windows.n_excluded_windows, 1)

    def test_missing_required_column_raises(self):
        rows = [{k: v for k, v in row.items() if k != "bvp_hr"} for row in _ROWS]

        with self.assertRaises(ValueError) as ctx:
            ar.prepare_windows(_make_df(rows))

        self.assertIn("Expected columns missing", str(ctx.exception))

    def test_empty_dataframe_raises(self):
        empty = pd.DataFrame(columns=list(ar.REQUIRED_COLUMNS))

        with self.assertRaises(ValueError) as ctx:
            ar.prepare_windows(empty)

        self.assertIn("no rows", str(ctx.exception))

    def test_all_rows_unusable_raises(self):
        rows = [
            {"subject_id": 0, "Label": None, "hr": None, "bvp_hr": None, "ir-quality": 0.5, "red-quality": 0.5}
        ]

        with self.assertRaises(ValueError) as ctx:
            ar.prepare_windows(_make_df(rows))

        self.assertIn("No usable windows remain", str(ctx.exception))


class TestAgreementSummary(unittest.TestCase):
    def test_known_bias_mae_and_limits_of_agreement(self):
        windows = ar.prepare_windows(_make_df(_ROWS))

        result = ar.agreement_summary(
            windows.data["bvp_hr"], windows.data["hr"]
        )

        self.assertEqual(result.n, 8)
        self.assertAlmostEqual(result.mae, 5.0)
        self.assertAlmostEqual(result.bias, 5.0)
        self.assertAlmostEqual(result.sd, float(np.std([2, 2, 8, 8, 2, 8, 2, 8], ddof=1)))
        self.assertAlmostEqual(result.upper_loa, result.bias + 1.96 * result.sd)
        self.assertAlmostEqual(result.lower_loa, result.bias - 1.96 * result.sd)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError) as ctx:
            ar.agreement_summary(pd.Series([1, 2, 3]), pd.Series([1, 2]))

        self.assertIn("same windows", str(ctx.exception))

    def test_single_window_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ar.agreement_summary(pd.Series([1.0]), pd.Series([2.0]))

        self.assertIn("at least two windows", str(ctx.exception).lower())

    def test_empty_series_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ar.agreement_summary(pd.Series([], dtype=float), pd.Series([], dtype=float))

        self.assertIn("zero windows", str(ctx.exception))


class TestSplitBySubject(unittest.TestCase):
    def test_split_has_no_subject_overlap(self):
        windows = ar.prepare_windows(_make_df(_ROWS))

        split = ar.split_by_subject(windows, test_fraction=0.3, seed=0)

        self.assertEqual(set(split.train_subjects) & set(split.test_subjects), set())
        self.assertEqual(
            set(split.train_subjects) | set(split.test_subjects), {0, 1, 2}
        )
        self.assertEqual(split.n_train_windows + split.n_test_windows, 8)

    def test_deterministic_for_a_fixed_seed(self):
        windows = ar.prepare_windows(_make_df(_ROWS))

        first = ar.split_by_subject(windows, test_fraction=0.3, seed=7)
        second = ar.split_by_subject(windows, test_fraction=0.3, seed=7)

        self.assertEqual(first.test_subjects, second.test_subjects)

    def test_single_subject_raises(self):
        rows = [row for row in _ROWS if row["subject_id"] == 0]
        windows = ar.prepare_windows(_make_df(rows))

        with self.assertRaises(ValueError) as ctx:
            ar.split_by_subject(windows)

        self.assertIn("at least 2 distinct subjects", str(ctx.exception))

    def test_invalid_test_fraction_raises(self):
        windows = ar.prepare_windows(_make_df(_ROWS))

        with self.assertRaises(ValueError):
            ar.split_by_subject(windows, test_fraction=1.5)

        with self.assertRaises(ValueError):
            ar.split_by_subject(windows, test_fraction=0.0)


class TestSplitByWindow(unittest.TestCase):
    def test_split_covers_every_window_exactly_once(self):
        windows = ar.prepare_windows(_make_df(_ROWS))

        split = ar.split_by_window(windows, test_fraction=0.25, seed=0)

        self.assertEqual(split.n_train_windows + split.n_test_windows, 8)

    def test_a_subject_can_appear_on_both_sides(self):
        # The whole point of this split: unlike split_by_subject, nothing
        # here prevents the same subject's windows from landing in both
        # train and test. Subject 0 has 4 of the 8 windows, so a 50/50
        # window-level split reliably puts some of them on each side.
        windows = ar.prepare_windows(_make_df(_ROWS))

        split = ar.split_by_window(windows, test_fraction=0.5, seed=0)

        self.assertTrue(set(split.train_subjects) & set(split.test_subjects))

    def test_deterministic_for_a_fixed_seed(self):
        windows = ar.prepare_windows(_make_df(_ROWS))

        first = ar.split_by_window(windows, test_fraction=0.3, seed=7)
        second = ar.split_by_window(windows, test_fraction=0.3, seed=7)

        self.assertEqual(first.n_test_windows, second.n_test_windows)
        pd.testing.assert_frame_equal(first.test_data, second.test_data)

    def test_too_few_windows_raises(self):
        rows = _ROWS[:1]
        windows = ar.prepare_windows(_make_df(rows))

        with self.assertRaises(ValueError) as ctx:
            ar.split_by_window(windows)

        self.assertIn("at least 2 windows", str(ctx.exception))

    def test_invalid_test_fraction_raises(self):
        windows = ar.prepare_windows(_make_df(_ROWS))

        with self.assertRaises(ValueError):
            ar.split_by_window(windows, test_fraction=1.5)

        with self.assertRaises(ValueError):
            ar.split_by_window(windows, test_fraction=0.0)


class TestFitAndApplyRecalibration(unittest.TestCase):
    def setUp(self):
        # bvp_hr is exactly hr + 2, so the exact recovering model is
        # hr = -2 + 1 * bvp_hr.
        hr = [60.0, 70.0, 80.0, 90.0]
        bvp_hr = [value + 2.0 for value in hr]
        self.train_data = pd.DataFrame({"hr": hr, "bvp_hr": bvp_hr})

    def test_known_intercept_slope_and_r_squared(self):
        model = ar.fit_recalibration(self.train_data)

        self.assertAlmostEqual(model.intercept, -2.0, places=6)
        self.assertAlmostEqual(model.slope, 1.0, places=6)
        self.assertAlmostEqual(model.r_squared, 1.0, places=6)
        self.assertEqual(model.n_train, 4)

    def test_apply_recalibration_recovers_hr(self):
        model = ar.fit_recalibration(self.train_data)

        predicted = ar.apply_recalibration(model, self.train_data["bvp_hr"])

        pd.testing.assert_series_equal(
            predicted.reset_index(drop=True),
            self.train_data["hr"].reset_index(drop=True),
            check_names=False,
        )

    def test_single_training_window_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ar.fit_recalibration(self.train_data.iloc[:1])

        self.assertIn("At least 2 training windows", str(ctx.exception))


class TestFilterByQuality(unittest.TestCase):
    def setUp(self):
        windows = ar.prepare_windows(_make_df(_ROWS))
        self.data = windows.data

    def test_retention_and_agreement_reported_together(self):
        # Only the four "rest" windows have quality >= 0.8.
        result = ar.filter_by_quality(
            self.data,
            predicted_col="bvp_hr",
            target_col="hr",
            threshold=0.8,
        )

        self.assertEqual(result.n_input_windows, 8)
        self.assertEqual(result.n_retained_windows, 4)
        self.assertEqual(result.n_excluded_windows, 4)
        self.assertAlmostEqual(result.retention_rate, 0.5)
        self.assertAlmostEqual(result.agreement.bias, 2.0)
        self.assertAlmostEqual(result.agreement.mae, 2.0)

    def test_threshold_excluding_everything_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ar.filter_by_quality(
                self.data,
                predicted_col="bvp_hr",
                target_col="hr",
                threshold=2.0,
            )

        self.assertIn("No windows remain", str(ctx.exception))

    def test_empty_dataframe_raises(self):
        empty = self.data.iloc[:0]

        with self.assertRaises(ValueError) as ctx:
            ar.filter_by_quality(
                empty,
                predicted_col="bvp_hr",
                target_col="hr",
                threshold=0.5,
            )

        self.assertIn("zero windows", str(ctx.exception))


class TestBreakdownByCondition(unittest.TestCase):
    def setUp(self):
        windows = ar.prepare_windows(_make_df(_ROWS))
        self.data = windows.data

    def test_known_per_condition_mae_and_bias(self):
        results = {
            r.group: r
            for r in ar.breakdown_by_condition(
                self.data, predicted_col="bvp_hr", target_col="hr"
            )
        }

        self.assertEqual(results["rest"].n, 4)
        self.assertAlmostEqual(results["rest"].mae, 2.0)
        self.assertAlmostEqual(results["rest"].bias, 2.0)

        self.assertEqual(results["walk"].n, 4)
        self.assertAlmostEqual(results["walk"].mae, 8.0)
        self.assertAlmostEqual(results["walk"].bias, 8.0)

    def test_alphabetical_group_ordering(self):
        results = ar.breakdown_by_condition(
            self.data, predicted_col="bvp_hr", target_col="hr"
        )

        self.assertEqual([r.group for r in results], ["rest", "walk"])

    def test_single_window_group_does_not_raise(self):
        # Regression guard: a thin group must still be reported, unlike
        # agreement_summary which requires at least two windows for a
        # standard deviation of agreement.
        rows = _ROWS[:1] + [
            {"subject_id": 9, "Label": "thin", "hr": 70, "bvp_hr": 71, "ir-quality": 0.9, "red-quality": 0.9}
        ]
        data = ar.prepare_windows(_make_df(rows)).data

        results = {
            r.group: r
            for r in ar.breakdown_by_condition(data, predicted_col="bvp_hr", target_col="hr")
        }

        self.assertEqual(results["thin"].n, 1)
        self.assertAlmostEqual(results["thin"].mae, 1.0)

    def test_missing_group_column_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ar.breakdown_by_condition(
                self.data,
                predicted_col="bvp_hr",
                target_col="hr",
                group_col="not_a_column",
            )

        self.assertIn("was not found", str(ctx.exception))

    def test_empty_dataframe_raises(self):
        empty = pd.DataFrame(columns=["Label", "hr", "bvp_hr"])

        with self.assertRaises(ValueError) as ctx:
            ar.breakdown_by_condition(empty, predicted_col="bvp_hr", target_col="hr")

        self.assertIn("zero windows", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
