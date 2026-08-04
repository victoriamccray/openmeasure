"""
Unit tests for shared/validation.py

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest

import pandas as pd

from shared.validation import validate_is_dataframe


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


if __name__ == "__main__":
    unittest.main()
