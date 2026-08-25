"""
Unit tests for core/quality.py

Run with: pytest modules/data_profile/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from core import profile, quality  # noqa: E402


class TestColumnFlags(unittest.TestCase):
    def test_all_missing_column_is_flagged(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
        flags = quality.quality_flags(df, profile.profile_dataframe(df))

        b_flags = [f for f in flags if f.column == "b"]
        self.assertEqual(len(b_flags), 1)
        self.assertIn("Every value is missing", b_flags[0].message)

    def test_constant_column_is_flagged_but_not_as_all_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5]})
        flags = quality.quality_flags(df, profile.profile_dataframe(df))

        b_flags = [f for f in flags if f.column == "b"]
        self.assertEqual(len(b_flags), 1)
        self.assertIn("same", b_flags[0].message)

    def test_high_missingness_column_is_flagged(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, None, None, None]})
        flags = quality.quality_flags(df, profile.profile_dataframe(df))

        a_flags = [f for f in flags if f.column == "a"]
        self.assertTrue(any("missing" in f.message for f in a_flags))

    def test_clean_column_is_not_flagged(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        flags = quality.quality_flags(df, profile.profile_dataframe(df))

        self.assertEqual([f for f in flags if f.column == "a"], [])


class TestDuplicateRowFlag(unittest.TestCase):
    def test_duplicate_rows_are_flagged_once(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        flags = quality.quality_flags(df, profile.profile_dataframe(df))

        dataset_flags = [f for f in flags if f.column is None]
        self.assertEqual(len(dataset_flags), 1)
        self.assertIn("1 duplicate row", dataset_flags[0].message)

    def test_no_duplicates_means_no_dataset_level_flag(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        flags = quality.quality_flags(df, profile.profile_dataframe(df))

        self.assertEqual([f for f in flags if f.column is None], [])


class TestQualityFlagValidation(unittest.TestCase):
    def test_unknown_severity_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            quality.QualityFlag(column="a", message="something", severity="urgent")

    def test_empty_message_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            quality.QualityFlag(column="a", message="", severity=quality.SEVERITY_INFO)


class TestShapeMismatch(unittest.TestCase):
    def test_stale_profile_from_a_different_shape_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        stale_profile = profile.profile_dataframe(pd.DataFrame({"a": [1, 2]}))

        with self.assertRaises(ValueError) as ctx:
            quality.quality_flags(df, stale_profile)

        self.assertIn("rows", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
