"""
Unit tests for core/design.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.design import DesignAssumptions  # noqa: E402


def _valid_kwargs(**overrides):
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
    return kwargs


class TestDesignAssumptions(unittest.TestCase):
    def test_valid_construction(self):
        assumptions = DesignAssumptions(**_valid_kwargs())
        self.assertEqual(assumptions.n_participants, 20)

    def test_n_observations_planned_is_hand_calculable(self):
        assumptions = DesignAssumptions(
            **_valid_kwargs(n_participants=20, observations_per_day=4, duration_days=7)
        )
        self.assertEqual(assumptions.n_observations_planned, 20 * 4 * 7)

    def test_non_positive_n_participants_raises(self):
        with self.assertRaises(ValueError):
            DesignAssumptions(**_valid_kwargs(n_participants=0))

    def test_non_positive_observations_per_day_raises(self):
        with self.assertRaises(ValueError):
            DesignAssumptions(**_valid_kwargs(observations_per_day=0))

    def test_non_positive_duration_days_raises(self):
        with self.assertRaises(ValueError):
            DesignAssumptions(**_valid_kwargs(duration_days=0))

    def test_zero_adherence_rate_raises(self):
        with self.assertRaises(ValueError):
            DesignAssumptions(**_valid_kwargs(adherence_rate=0.0))

    def test_adherence_rate_above_one_raises(self):
        with self.assertRaises(ValueError):
            DesignAssumptions(**_valid_kwargs(adherence_rate=1.1))

    def test_negative_sensor_noise_sd_raises(self):
        with self.assertRaises(ValueError):
            DesignAssumptions(**_valid_kwargs(sensor_noise_sd=-0.1))

    def test_negative_within_person_sd_raises(self):
        with self.assertRaises(ValueError):
            DesignAssumptions(**_valid_kwargs(within_person_sd=-0.1))

    def test_negative_between_person_sd_raises(self):
        with self.assertRaises(ValueError):
            DesignAssumptions(**_valid_kwargs(between_person_sd=-0.1))

    def test_pain_state_prevalence_of_zero_raises(self):
        with self.assertRaises(ValueError):
            DesignAssumptions(**_valid_kwargs(pain_state_prevalence=0.0))

    def test_pain_state_prevalence_of_one_raises(self):
        with self.assertRaises(ValueError):
            DesignAssumptions(**_valid_kwargs(pain_state_prevalence=1.0))

    def test_negative_temporal_misalignment_raises(self):
        with self.assertRaises(ValueError):
            DesignAssumptions(**_valid_kwargs(temporal_misalignment_minutes=-1.0))


if __name__ == "__main__":
    unittest.main()
