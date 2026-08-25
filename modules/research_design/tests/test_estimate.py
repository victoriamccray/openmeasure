"""
Unit tests for core/estimate.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from core.design import DesignAssumptions  # noqa: E402
from core.estimate import estimate_coupling_difference  # noqa: E402
from core.simulate import SimulatedStudy  # noqa: E402


def _assumptions(**overrides):
    kwargs = dict(
        n_participants=2,
        observations_per_day=3,
        duration_days=1,
        adherence_rate=1.0,
        sensor_noise_sd=0.0,
        within_person_sd=0.0,
        between_person_sd=0.0,
        effect_magnitude=0.0,
        pain_state_prevalence=0.5,
        temporal_misalignment_minutes=0.0,
        seed=1,
    )
    kwargs.update(overrides)
    return DesignAssumptions(**kwargs)


def _study(rows: list[dict]) -> SimulatedStudy:
    data = pd.DataFrame(rows)
    return SimulatedStudy(
        assumptions=_assumptions(),
        data=data,
        n_observations_planned=len(rows),
        n_observations_retained=len(rows),
        n_localized_observations=int((data["pain_state"] == "localized").sum()),
        n_distributed_observations=int((data["pain_state"] == "distributed").sum()),
    )


def _rows(participant_id, pain_state, ratings, signals):
    return [
        {
            "participant_id": participant_id,
            "day": 0,
            "observation_index": i,
            "pain_state": pain_state,
            "pain_rating": r,
            "physio_signal": s,
        }
        for i, (r, s) in enumerate(zip(ratings, signals))
    ]


class TestEstimateCouplingDifference(unittest.TestCase):
    def test_hand_calculable_two_participants(self):
        # Participant 0: localized r=+1 (physio == rating), distributed
        # r=-1 (physio is the mirror image of rating) -> difference -2.
        # Participant 1: localized r=+1, distributed r=+1 -> difference 0.
        # Mean difference = -1.0; sample SD of [-2, 0] is sqrt(2), so the
        # standard error (SD / sqrt(2)) is exactly 1.0.
        rows = (
            _rows(0, "localized", [1, 2, 3], [1, 2, 3])
            + _rows(0, "distributed", [1, 2, 3], [3, 2, 1])
            + _rows(1, "localized", [1, 2, 3], [1, 2, 3])
            + _rows(1, "distributed", [1, 2, 3], [1, 2, 3])
        )
        study = _study(rows)

        result = estimate_coupling_difference(study, min_observations_per_state=3)

        self.assertAlmostEqual(result.estimated_difference, -1.0, places=6)
        self.assertAlmostEqual(result.standard_error, 1.0, places=6)
        self.assertEqual(result.n_participants_used, 2)
        self.assertEqual(result.n_participants_excluded_insufficient_data, 0)
        self.assertEqual(result.per_participant_differences, (-2.0, 0.0))

    def test_all_participants_insufficient_data_returns_none(self):
        # Only 2 observations per state, below the default minimum of 3.
        rows = _rows(0, "localized", [1, 2], [1, 2]) + _rows(0, "distributed", [1, 2], [2, 1])
        study = _study(rows)

        result = estimate_coupling_difference(study)

        self.assertIsNone(result.estimated_difference)
        self.assertIsNone(result.standard_error)
        self.assertEqual(result.n_participants_used, 0)
        self.assertEqual(result.n_participants_excluded_insufficient_data, 1)

    def test_single_usable_participant_has_no_standard_error(self):
        rows = (
            _rows(0, "localized", [1, 2, 3], [1, 2, 3])
            + _rows(0, "distributed", [1, 2, 3], [1, 2, 3])
            # Participant 1 has too few distributed observations.
            + _rows(1, "localized", [1, 2, 3], [1, 2, 3])
            + _rows(1, "distributed", [1, 2], [1, 2])
        )
        study = _study(rows)

        result = estimate_coupling_difference(study, min_observations_per_state=3)

        self.assertEqual(result.n_participants_used, 1)
        self.assertIsNone(result.standard_error)
        self.assertAlmostEqual(result.estimated_difference, 0.0, places=6)

    def test_constant_pain_rating_within_state_excludes_participant(self):
        # Constant pain_rating makes the within-state correlation undefined.
        rows = _rows(0, "localized", [5, 5, 5], [1, 2, 3]) + _rows(
            0, "distributed", [1, 2, 3], [1, 2, 3]
        )
        study = _study(rows)

        result = estimate_coupling_difference(study, min_observations_per_state=3)

        self.assertEqual(result.n_participants_used, 0)
        self.assertEqual(result.n_participants_excluded_insufficient_data, 1)

    def test_min_observations_per_state_below_two_raises(self):
        study = _study(_rows(0, "localized", [1, 2, 3], [1, 2, 3]))
        with self.assertRaises(ValueError):
            estimate_coupling_difference(study, min_observations_per_state=1)


if __name__ == "__main__":
    unittest.main()
