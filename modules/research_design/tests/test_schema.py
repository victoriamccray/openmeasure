"""
Unit tests for core/schema.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.design import DesignAssumptions  # noqa: E402
from core.schema import measurement_plan_profile  # noqa: E402
from modules.data_profile.core.suggest import (  # noqa: E402
    WORKFLOW_FAIRNESS,
    WORKFLOW_IMPACT_EVALUATION,
    WORKFLOW_RELIABILITY,
    WORKFLOW_TIME_SERIES_QA,
    suggest_workflows,
)


def _assumptions(**overrides):
    kwargs = dict(
        n_participants=20,
        observations_per_day=4,
        duration_days=7,
        adherence_rate=0.8,
        sensor_noise_sd=0.5,
        within_person_sd=0.3,
        between_person_sd=0.2,
        effect_magnitude=0.4,
        pain_state_prevalence=0.3,
        temporal_misalignment_minutes=5.0,
        seed=42,
    )
    kwargs.update(overrides)
    return DesignAssumptions(**kwargs)


class TestMeasurementPlanProfile(unittest.TestCase):
    def test_row_count_matches_planned_observations(self):
        assumptions = _assumptions(n_participants=20, observations_per_day=4, duration_days=7)
        profile = measurement_plan_profile(assumptions)
        self.assertEqual(profile.n_rows, 20 * 4 * 7)

    def test_expected_columns_present(self):
        profile = measurement_plan_profile(_assumptions())
        names = {column.name for column in profile.columns}
        self.assertEqual(
            names,
            {"participant_id", "timestamp", "pain_state", "pain_rating", "physio_signal"},
        )

    def test_pain_state_is_binary_categorical(self):
        profile = measurement_plan_profile(_assumptions())
        pain_state = profile.column("pain_state")
        self.assertEqual(pain_state.n_unique, 2)

    def test_suggests_time_series_qa_and_impact_evaluation_only(self):
        # This measurement plan has a timestamp (Time-Series QA's shape)
        # and a binary group column next to a continuous outcome (Impact
        # Evaluation's shape), but only one categorical column and no
        # multi-item scale, so Fairness and Reliability should not match.
        profile = measurement_plan_profile(_assumptions())
        suggestions = suggest_workflows(profile)
        matched = {suggestion.workflow for suggestion in suggestions}

        self.assertIn(WORKFLOW_TIME_SERIES_QA, matched)
        self.assertIn(WORKFLOW_IMPACT_EVALUATION, matched)
        self.assertNotIn(WORKFLOW_FAIRNESS, matched)
        self.assertNotIn(WORKFLOW_RELIABILITY, matched)


if __name__ == "__main__":
    unittest.main()
