"""
Unit tests for core/simulate.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from core.design import DesignAssumptions  # noqa: E402
from core.simulate import generate_naturalistic_pain_study  # noqa: E402


def _assumptions(**overrides):
    kwargs = dict(
        n_participants=15,
        observations_per_day=3,
        duration_days=7,
        adherence_rate=0.85,
        sensor_noise_sd=0.4,
        within_person_sd=0.3,
        between_person_sd=0.2,
        effect_magnitude=0.5,
        pain_state_prevalence=0.35,
        temporal_misalignment_minutes=4.0,
        seed=7,
    )
    kwargs.update(overrides)
    return DesignAssumptions(**kwargs)


class TestGenerateNaturalisticPainStudy(unittest.TestCase):
    def test_identical_assumptions_produce_identical_data(self):
        assumptions = _assumptions()

        first = generate_naturalistic_pain_study(assumptions)
        second = generate_naturalistic_pain_study(assumptions)

        pd.testing.assert_frame_equal(first.data, second.data)
        self.assertEqual(first.n_observations_retained, second.n_observations_retained)
        self.assertEqual(first.n_localized_observations, second.n_localized_observations)
        self.assertEqual(first.n_distributed_observations, second.n_distributed_observations)

    def test_different_seed_produces_different_data(self):
        first = generate_naturalistic_pain_study(_assumptions(seed=1))
        second = generate_naturalistic_pain_study(_assumptions(seed=2))

        self.assertFalse(first.data["physio_signal"].equals(second.data["physio_signal"]))

    def test_full_adherence_retains_every_planned_observation(self):
        assumptions = _assumptions(adherence_rate=1.0)
        study = generate_naturalistic_pain_study(assumptions)

        self.assertEqual(study.n_observations_planned, assumptions.n_observations_planned)
        self.assertEqual(study.n_observations_retained, assumptions.n_observations_planned)
        self.assertEqual(study.pct_missing, 0.0)

    def test_observation_counts_hand_calculable(self):
        assumptions = _assumptions(n_participants=10, observations_per_day=2, duration_days=5)
        study = generate_naturalistic_pain_study(assumptions)

        self.assertEqual(study.n_observations_planned, 10 * 2 * 5)
        self.assertEqual(
            study.n_localized_observations + study.n_distributed_observations,
            study.n_observations_retained,
        )

    def test_returned_columns(self):
        study = generate_naturalistic_pain_study(_assumptions())
        expected_columns = {
            "participant_id",
            "day",
            "observation_index",
            "pain_state",
            "pain_rating",
            "physio_signal",
        }
        self.assertEqual(set(study.data.columns), expected_columns)

    def test_pain_rating_stays_within_scale_bounds(self):
        study = generate_naturalistic_pain_study(
            _assumptions(temporal_misalignment_minutes=60.0)
        )
        self.assertTrue((study.data["pain_rating"] >= 0.0).all())
        self.assertTrue((study.data["pain_rating"] <= 10.0).all())

    def test_pct_missing_hand_calculable_with_deterministic_adherence(self):
        assumptions = _assumptions(adherence_rate=1.0)
        study = generate_naturalistic_pain_study(assumptions)
        self.assertEqual(study.pct_missing, 0.0)


if __name__ == "__main__":
    unittest.main()
