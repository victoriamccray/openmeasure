"""
Unit tests for core/deployment.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import deployment as d  # noqa: E402


class TestProjectDeploymentSavings(unittest.TestCase):
    def test_hand_calculable_projection(self):
        result = d.project_deployment_savings(
            per_unit_difference=2.0, n_deployments=10, resource_metric_name="CO2 (kg)"
        )
        self.assertAlmostEqual(result.projected_difference, 20.0)

    def test_matches_gaia_own_reported_figure(self):
        # GAIA reports the student model saves 0.44 kg of CO2 per 1,000,000
        # deployments versus the teacher. Cross-checking the per-unit rate
        # implied by that figure against the same 1,000,000 deployments
        # should reproduce the paper's own number exactly.
        per_unit = 0.44 / 1_000_000
        result = d.project_deployment_savings(
            per_unit_difference=per_unit,
            n_deployments=1_000_000,
            resource_metric_name="CO2 (kg)",
        )
        self.assertAlmostEqual(result.projected_difference, 0.44)

    def test_raises_on_non_positive_deployments(self):
        with self.assertRaises(ValueError):
            d.project_deployment_savings(
                per_unit_difference=1.0, n_deployments=0, resource_metric_name="CO2 (kg)"
            )

    def test_raises_on_empty_resource_metric_name(self):
        with self.assertRaises(ValueError):
            d.project_deployment_savings(
                per_unit_difference=1.0, n_deployments=10, resource_metric_name=""
            )


if __name__ == "__main__":
    unittest.main()
