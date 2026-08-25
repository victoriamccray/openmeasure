"""
Unit tests for shared/validation.py

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest

import pandas as pd

from shared.validation import format_value_sample, validate_is_dataframe


class TestValidateIsDataframe(unittest.TestCase):
    def test_dataframe_passes_silently(self):
        validate_is_dataframe(pd.DataFrame({"a": [1, 2, 3]}))

    def test_non_dataframe_raises_typeerror(self):
        with self.assertRaises(TypeError) as ctx:
            validate_is_dataframe([1, 2, 3])

        self.assertEqual(
            str(ctx.exception),
            "Data must be provided as a pandas DataFrame.",
        )


class TestFormatValueSample(unittest.TestCase):
    def test_short_list_shown_in_full(self):
        self.assertEqual(format_value_sample([0, 1]), "[0, 1]")

    def test_long_list_is_truncated_with_a_count(self):
        values = [round(i * 0.01, 2) for i in range(130)]
        result = format_value_sample(values, limit=8)

        self.assertEqual(result, f"{values[:8]} and 122 more")

    def test_limit_is_respected(self):
        values = list(range(20))
        result = format_value_sample(values, limit=3)

        self.assertEqual(result, "[0, 1, 2] and 17 more")

    def test_empty_list(self):
        self.assertEqual(format_value_sample([]), "[]")


if __name__ == "__main__":
    unittest.main()
