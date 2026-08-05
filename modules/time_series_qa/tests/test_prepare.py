"""
Unit tests for core/prepare.py

Run with: pytest modules/time_series_qa/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from core.prepare import prepare_series  # noqa: E402


def frame(timestamps, values=None):
    if values is None:
        values = list(range(len(timestamps)))
    return pd.DataFrame({"ts": timestamps, "val": values})


class TestIngestAccounting(unittest.TestCase):
    def test_separates_null_from_unparseable_timestamps(self):
        data = frame(
            ["2020-01-01", None, "", "not a date", "2020-01-02", "2020-01-03"]
        )

        result = prepare_series(data, "ts", "val")

        self.assertEqual(result.n_input_rows, 6)
        self.assertEqual(result.n_null_timestamps, 1)
        # An empty string is present but unusable, so it is unparseable
        # rather than null.
        self.assertEqual(result.n_unparseable_timestamps, 2)
        self.assertEqual(result.n_rows_used, 3)

    def test_retains_example_unparseable_values(self):
        data = frame(["2020-01-01", "not a date", "2020-01-02"])

        result = prepare_series(data, "ts", "val")

        self.assertIn("not a date", result.example_unparseable)

    def test_inconsistent_fractional_seconds_are_still_parsed(self):
        # pandas infers one format from the first value, so a column mixing
        # "00:00:00" with "00:05:00.4" would otherwise reject every value
        # whose fractional-second precision differs. They are valid
        # timestamps and must be kept.
        data = frame(
            [
                "2024-01-01 00:00:00",
                "2024-01-01 00:05:00.4",
                "2024-01-01 00:10:00",
                "2024-01-01 00:15:00.2",
            ]
        )

        result = prepare_series(data, "ts", "val")

        self.assertEqual(result.n_rows_used, 4)
        self.assertEqual(result.n_unparseable_timestamps, 0)

    def test_genuinely_unparseable_values_are_still_rejected(self):
        # The per-element retry must not turn into blanket permissiveness.
        data = frame(
            ["2024-01-01", "nonsense", "2024-01-03", "2024-01-04"]
        )

        result = prepare_series(data, "ts", "val")

        self.assertEqual(result.n_rows_used, 3)
        self.assertEqual(result.n_unparseable_timestamps, 1)
        self.assertIn("nonsense", result.example_unparseable)

    def test_caps_example_unparseable_values(self):
        data = frame(["2020-01-01", "2020-01-02"] + [f"bad{i}" for i in range(20)])

        result = prepare_series(data, "ts", "val")

        self.assertEqual(result.n_unparseable_timestamps, 20)
        self.assertLessEqual(len(result.example_unparseable), 5)


class TestOrdering(unittest.TestCase):
    def test_detects_out_of_order_input(self):
        data = frame(["2020-01-03", "2020-01-01", "2020-01-02"])

        result = prepare_series(data, "ts", "val")

        self.assertTrue(result.was_out_of_order)
        self.assertEqual(result.n_out_of_order_steps, 1)

    def test_sorted_input_is_not_flagged(self):
        data = frame(["2020-01-01", "2020-01-02", "2020-01-03"])

        result = prepare_series(data, "ts", "val")

        self.assertFalse(result.was_out_of_order)
        self.assertEqual(result.n_out_of_order_steps, 0)

    def test_output_is_chronologically_sorted(self):
        data = frame(["2020-01-03", "2020-01-01", "2020-01-02"])

        result = prepare_series(data, "ts", "val")

        self.assertTrue(result.timestamps.is_monotonic_increasing)

    def test_sort_is_stable_for_duplicate_timestamps(self):
        # Rows sharing a timestamp must keep their input order, or which
        # value is reported "first" changes between runs.
        data = frame(
            ["2020-01-02", "2020-01-01", "2020-01-02"],
            [10, 20, 30],
        )

        first = prepare_series(data, "ts", "val").values.tolist()
        second = prepare_series(data, "ts", "val").values.tolist()

        self.assertEqual(first, [20, 10, 30])
        self.assertEqual(first, second)


class TestTimezoneHandling(unittest.TestCase):
    def test_timezone_aware_input_keeps_its_timezone(self):
        data = frame(
            pd.date_range("2020-01-01", periods=3, tz="America/New_York")
        )

        result = prepare_series(data, "ts", "val")

        self.assertEqual(result.timezone, "America/New_York")

    def test_naive_input_is_not_localized(self):
        data = frame(["2020-01-01", "2020-01-02"])

        result = prepare_series(data, "ts", "val")

        self.assertIsNone(result.timezone)

    def test_mixed_utc_offsets_are_rejected(self):
        # pandas 3 raises here while pandas 2 returns object dtype; both
        # must surface the same clear error rather than a pandas internal.
        data = frame(
            ["2020-01-01 00:00:00+00:00", "2020-01-01 00:00:00+05:00"]
        )

        with self.assertRaises(ValueError) as context:
            prepare_series(data, "ts", "val")

        self.assertIn("mixed UTC offsets", str(context.exception))


class TestRejectedInput(unittest.TestCase):
    def test_numeric_timestamp_column_is_rejected(self):
        # pd.to_datetime([1577836800]) silently yields 1970-01-01, so
        # guessing the unit would corrupt every downstream result.
        data = frame([1577836800, 1577923200])

        with self.assertRaises(ValueError) as context:
            prepare_series(data, "ts", "val")

        self.assertIn("numeric", str(context.exception))

    def test_no_usable_timestamps_raises(self):
        data = frame(["not a date", "also not a date"])

        with self.assertRaises(ValueError) as context:
            prepare_series(data, "ts", "val")

        self.assertIn("No rows remain", str(context.exception))

    def test_empty_dataframe_raises(self):
        data = pd.DataFrame({"ts": [], "val": []})

        with self.assertRaises(ValueError):
            prepare_series(data, "ts", "val")

    def test_non_dataframe_raises_typeerror(self):
        with self.assertRaises(TypeError):
            prepare_series([("2020-01-01", 1)], "ts", "val")

    def test_missing_timestamp_column_raises(self):
        data = frame(["2020-01-01", "2020-01-02"])

        with self.assertRaises(ValueError) as context:
            prepare_series(data, "nope", "val")

        self.assertIn("not found", str(context.exception))

    def test_missing_value_column_raises(self):
        data = frame(["2020-01-01", "2020-01-02"])

        with self.assertRaises(ValueError) as context:
            prepare_series(data, "ts", "nope")

        self.assertIn("not found", str(context.exception))

    def test_identical_columns_raise(self):
        data = frame(["2020-01-01", "2020-01-02"])

        with self.assertRaises(ValueError) as context:
            prepare_series(data, "ts", "ts")

        self.assertIn("must be different", str(context.exception))


class TestSingleObservation(unittest.TestCase):
    def test_one_usable_row_is_accepted_here(self):
        # prepare_series succeeds; requiring two observations is the
        # pipeline's decision, not an ingest concern.
        data = frame(["2020-01-01"])

        result = prepare_series(data, "ts", "val")

        self.assertEqual(result.n_rows_used, 1)


if __name__ == "__main__":
    unittest.main()
