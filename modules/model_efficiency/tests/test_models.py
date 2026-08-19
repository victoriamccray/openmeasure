"""
Unit tests for core/models.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import models as m  # noqa: E402


def _profile(**overrides):
    defaults = dict(
        name="Model A",
        performance_metric_name="MSE",
        performance_value=0.1,
        performance_is_approximate=False,
        n_parameters=1000,
        resource_metric_name="CO2 (kg)",
        resource_value=1.0,
        resource_is_approximate=False,
    )
    defaults.update(overrides)
    return m.ModelProfile(**defaults)


class TestModelProfile(unittest.TestCase):
    def test_valid_profile_constructs(self):
        profile = _profile()
        self.assertEqual(profile.name, "Model A")
        self.assertEqual(profile.n_parameters, 1000)

    def test_raises_on_empty_name(self):
        with self.assertRaises(ValueError):
            _profile(name="")

    def test_raises_on_negative_performance_value(self):
        with self.assertRaises(ValueError):
            _profile(performance_value=-0.1)

    def test_raises_on_negative_resource_value(self):
        with self.assertRaises(ValueError):
            _profile(resource_value=-1.0)

    def test_raises_on_non_positive_n_parameters(self):
        with self.assertRaises(ValueError):
            _profile(n_parameters=0)


if __name__ == "__main__":
    unittest.main()
