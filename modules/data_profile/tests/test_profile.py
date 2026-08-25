"""
Unit tests for core/profile.py

Run with: pytest modules/data_profile/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from core import profile  # noqa: E402


class TestGuessRole(unittest.TestCase):
    def test_sequential_integer_column_is_identifier(self):
        series = pd.Series([1, 2, 3, 4, 5])
        result = profile.profile_column(series, name="row_id")
        self.assertEqual(result.role, profile.ROLE_IDENTIFIER)

    def test_id_suggesting_name_with_unique_values_is_identifier(self):
        series = pd.Series([104, 88, 233, 71])
        result = profile.profile_column(series, name="participant_id")
        self.assertEqual(result.role, profile.ROLE_IDENTIFIER)

    def test_datetime_dtype_column_is_datetime(self):
        series = pd.to_datetime(pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"]))
        result = profile.profile_column(series, name="timestamp")
        self.assertEqual(result.role, profile.ROLE_DATETIME)

    def test_date_like_strings_are_datetime(self):
        series = pd.Series(["2024-01-01", "2024-02-15", "2024-03-20", "2024-04-30"])
        result = profile.profile_column(series, name="visit_date")
        self.assertEqual(result.role, profile.ROLE_DATETIME)

    def test_numeric_low_cardinality_is_categorical(self):
        series = pd.Series([1, 2, 1, 2, 3, 1, 2, 3])
        result = profile.profile_column(series, name="condition_code")
        self.assertEqual(result.role, profile.ROLE_CATEGORICAL)

    def test_numeric_high_cardinality_is_continuous(self):
        series = pd.Series([float(i) + 0.37 for i in range(50)])
        result = profile.profile_column(series, name="score")
        self.assertEqual(result.role, profile.ROLE_CONTINUOUS)

    def test_nonnumeric_low_cardinality_is_categorical(self):
        series = pd.Series(["red", "blue", "red", "green", "blue", "red"])
        result = profile.profile_column(series, name="color")
        self.assertEqual(result.role, profile.ROLE_CATEGORICAL)

    def test_nonnumeric_high_cardinality_is_free_text(self):
        series = pd.Series([f"unique comment {i}" for i in range(20)])
        result = profile.profile_column(series, name="notes")
        self.assertEqual(result.role, profile.ROLE_TEXT)

    def test_all_missing_column_defaults_to_free_text(self):
        series = pd.Series([None, None, None])
        result = profile.profile_column(series, name="empty_col")
        self.assertEqual(result.role, profile.ROLE_TEXT)


class TestColumnProfileMissingness(unittest.TestCase):
    def test_missing_count_and_percentage(self):
        series = pd.Series([1.0, None, 3.0, None])
        result = profile.profile_column(series, name="x")
        self.assertEqual(result.n_missing, 2)
        self.assertAlmostEqual(result.pct_missing, 50.0)

    def test_invalid_role_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            profile.ColumnProfile(
                name="x", dtype="float64", n_missing=0, pct_missing=0.0,
                n_unique=1, role="not a real role",
            )


class TestProfileDataframe(unittest.TestCase):
    def test_profiles_every_column(self):
        df = pd.DataFrame({"id": [1, 2, 3], "score": [4.5, 5.5, 6.5]})
        result = profile.profile_dataframe(df)

        self.assertEqual(result.n_rows, 3)
        self.assertEqual(result.n_columns, 2)
        self.assertEqual({c.name for c in result.columns}, {"id", "score"})

    def test_column_lookup_by_name(self):
        df = pd.DataFrame({"id": [1, 2, 3]})
        result = profile.profile_dataframe(df)

        self.assertEqual(result.column("id").name, "id")

    def test_unknown_column_lookup_raises_keyerror(self):
        df = pd.DataFrame({"id": [1, 2, 3]})
        result = profile.profile_dataframe(df)

        with self.assertRaises(KeyError):
            result.column("does_not_exist")

    def test_columns_with_role_filters_correctly(self):
        df = pd.DataFrame({"id": [1, 2, 3], "notes": ["a b c", "d e f", "g h i"]})
        result = profile.profile_dataframe(df)

        self.assertEqual(result.columns_with_role(profile.ROLE_IDENTIFIER), ("id",))

    def test_empty_dataframe_raises(self):
        with self.assertRaises(ValueError) as ctx:
            profile.profile_dataframe(pd.DataFrame({"a": []}))

        self.assertIn("no rows", str(ctx.exception))

    def test_non_dataframe_input_raises_typeerror(self):
        with self.assertRaises(TypeError):
            profile.profile_dataframe([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
