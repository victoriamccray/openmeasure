"""
Unit tests for core/transformation.py

Run with: pytest modules/program_evaluation/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import comparison, transformation  # noqa: E402


def _frame(seed=7, n=25):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "arm": ["treated"] * n + ["control"] * n,
            "score": np.r_[rng.normal(70, 10, n), rng.normal(62, 8, n)],
        }
    )


def _view(frame=None):
    frame = _frame() if frame is None else frame
    result = comparison.compare_two_groups(frame, "arm", "score")
    return transformation.effect_size_transformation(
        frame, "arm", "score", result
    ), result


class TestStageSequence(unittest.TestCase):
    def test_five_stages_in_the_documented_order(self):
        view, _ = _view()

        self.assertEqual(
            tuple(stage.key for stage in view.stages),
            transformation.STAGE_ORDER,
        )

    def test_stage_index_locates_a_stage(self):
        view, _ = _view()

        self.assertEqual(view.stage_index(transformation.STAGE_OBSERVATIONS), 0)
        self.assertEqual(view.stage_index(transformation.STAGE_EFFECT_SIZE), 4)

    def test_an_unknown_stage_raises(self):
        view, _ = _view()

        with self.assertRaises(ValueError) as raised:
            view.stage_index("standard_error")

        self.assertIn("not a known stage", str(raised.exception))

    def test_every_stage_explains_itself(self):
        view, _ = _view()

        for stage in view.stages:
            with self.subTest(stage=stage.key):
                self.assertTrue(stage.label.strip())
                self.assertTrue(stage.explanation.strip())
                self.assertTrue(stage.headline.strip())

    def test_a_stage_with_an_unknown_key_is_rejected(self):
        with self.assertRaises(ValueError):
            transformation.TransformationStage(
                key="variance", label="L", explanation="E", headline="H"
            )

    def test_a_stage_missing_text_is_rejected(self):
        for field_name in ("label", "explanation", "headline"):
            fields = {
                "key": transformation.STAGE_MEANS,
                "label": "L",
                "explanation": "E",
                "headline": "H",
            }
            fields[field_name] = ""
            with self.subTest(field=field_name):
                with self.assertRaises(ValueError):
                    transformation.TransformationStage(**fields)


class TestHeadlinesUseTheComputedNumbers(unittest.TestCase):
    """
    Each headline restates a number the analysis produced. If any were
    recomputed here, the chart could disagree with the result above it.
    """

    def test_the_group_stage_reports_the_result_counts(self):
        view, result = _view()
        headline = view.stages[1].headline

        self.assertIn(str(result.n_a), headline)
        self.assertIn(str(result.n_b), headline)

    def test_the_means_stage_reports_the_result_means(self):
        view, result = _view()
        headline = view.stages[2].headline

        self.assertIn(f"{result.mean_a:.2f}", headline)
        self.assertIn(f"{result.mean_b:.2f}", headline)

    def test_the_effect_size_stage_shows_the_division_that_produced_d(self):
        view, result = _view()
        headline = view.stages[4].headline

        self.assertIn(f"{result.cohens_d:.2f}", headline)
        self.assertIn(f"{view.pooled_sd:.2f}", headline)

    def test_the_shown_division_lands_on_the_shown_effect_size(self):
        view, result = _view()

        self.assertAlmostEqual(
            result.mean_difference / view.pooled_sd, result.cohens_d, places=12
        )


class TestObservations(unittest.TestCase):
    def test_the_values_drawn_are_the_ones_analyzed(self):
        view, result = _view()

        self.assertEqual(len(view.values_a), result.n_a)
        self.assertEqual(len(view.values_b), result.n_b)

    def test_rows_missing_a_value_are_left_out_of_both(self):
        frame = _frame()
        frame.loc[0, "score"] = np.nan
        frame.loc[30, "arm"] = np.nan

        view, result = _view(frame)

        self.assertEqual(len(view.all_values), result.n_rows_used)

    def test_all_values_covers_both_groups(self):
        view, _ = _view()

        self.assertEqual(
            len(view.all_values), len(view.values_a) + len(view.values_b)
        )


class TestJitter(unittest.TestCase):
    def test_the_same_count_always_gives_the_same_offsets(self):
        """
        Random offsets would reshuffle the cloud on every rerun, and a
        reader stepping through the stages would read that as the data
        changing underneath them.
        """
        first = transformation._deterministic_jitter(40)
        second = transformation._deterministic_jitter(40)

        self.assertEqual(first, second)

    def test_offsets_stay_within_the_row(self):
        for count in (1, 5, 40, 200):
            with self.subTest(count=count):
                for offset in transformation._deterministic_jitter(count):
                    self.assertLessEqual(abs(offset), 1.0)

    def test_there_is_one_offset_per_observation(self):
        view, _ = _view()

        self.assertEqual(len(view.jitter_a), len(view.values_a))
        self.assertEqual(len(view.jitter_b), len(view.values_b))

    def test_offsets_are_not_all_the_same(self):
        """A constant offset would stack every point on one line."""
        offsets = transformation._deterministic_jitter(30)

        self.assertGreater(len(set(offsets)), 20)


class TestDegenerateInput(unittest.TestCase):
    def test_rejects_non_dataframe(self):
        _, result = _view()

        with self.assertRaises(TypeError):
            transformation.effect_size_transformation(
                [1, 2, 3], "arm", "score", result
            )

    def test_rejects_a_missing_column(self):
        frame = _frame()
        _, result = _view(frame)

        with self.assertRaises(ValueError) as raised:
            transformation.effect_size_transformation(
                frame, "arm", "outcome", result
            )

        self.assertIn("outcome", str(raised.exception))

    def test_rejects_a_result_computed_from_different_data(self):
        """
        Drawing observations the result did not analyze would put a chart
        and a statistic side by side that describe different samples.
        """
        frame = _frame()
        _, result = _view(frame)
        larger = _frame(n=40)

        with self.assertRaises(ValueError) as raised:
            transformation.effect_size_transformation(
                larger, "arm", "score", result
            )

        self.assertIn("different set of observations", str(raised.exception))

    def test_rejects_zero_spread_within_both_groups(self):
        """
        With no within-group variation the final stage has nothing to
        measure the difference against.

        compare_two_groups already refuses this case, so a result like
        this cannot arrive through the normal path. The guard covers the
        boundary instead: effect_size_transformation accepts whatever
        result it is handed, and a caller constructing one directly
        should not get a chart dividing by zero.
        """
        frame = pd.DataFrame(
            {
                "arm": ["treated"] * 3 + ["control"] * 3,
                "score": [5.0, 5.0, 5.0, 8.0, 8.0, 8.0],
            }
        )

        with self.assertRaises(ValueError):
            comparison.compare_two_groups(frame, "arm", "score")

        flat = comparison.TwoGroupResult(
            group_a_label="control",
            group_b_label="treated",
            n_a=3,
            n_b=3,
            mean_a=8.0,
            mean_b=5.0,
            sd_a=0.0,
            sd_b=0.0,
            t_statistic=float("inf"),
            degrees_of_freedom=4.0,
            p_value=0.0,
            cohens_d=float("inf"),
            mean_difference=3.0,
            ci_95_low=3.0,
            ci_95_high=3.0,
            n_input_rows=6,
            n_rows_used=6,
            n_excluded_rows=0,
            exclusion_reason=comparison.MISSING_GROUP_OR_OUTCOME,
        )

        with self.assertRaises(ValueError) as raised:
            transformation.effect_size_transformation(
                frame, "arm", "score", flat
            )

        self.assertIn("nothing to measure", str(raised.exception))


class TestFrozen(unittest.TestCase):
    def test_both_models_are_frozen(self):
        self.assertTrue(transformation.TransformationStage.__dataclass_params__.frozen)
        self.assertTrue(transformation.TransformationView.__dataclass_params__.frozen)


if __name__ == "__main__":
    unittest.main()
