"""
Turn the naturalistic pain study's measurement plan into a DataProfile.

This is the "simulated data structure" step in the
Research question -> Study design -> Measurement plan -> Simulated data
structure -> Method selection -> Analysis considerations flow: before
any data is collected (or, here, before it is even simulated),
declaring what columns the study's measurement plan will produce is
enough to ask modules/data_profile/core/suggest.py's suggest_workflows
which existing OpenMeasure workflow(s) that shape fits.

profile_dataframe() (modules/data_profile/core/profile.py) infers a
ColumnProfile by inspecting real data. There is no real data yet, so
measurement_plan_profile() builds the same ColumnProfile shape by hand
from what the measurement plan specifies. n_unique values below are
therefore planning estimates, not measured counts, and are documented as
such rather than left looking like real data would produce them.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.data_profile.core.profile import (
    ROLE_CATEGORICAL,
    ROLE_CONTINUOUS,
    ROLE_DATETIME,
    ROLE_IDENTIFIER,
    ColumnProfile,
    DataProfile,
)

from .design import DesignAssumptions

N_PAIN_STATES = 2


def measurement_plan_profile(assumptions: DesignAssumptions) -> DataProfile:
    """
    The DataProfile a real version of this measurement plan would
    produce: one participant identifier, one timestamp, the pain-state
    label, the pain rating, and the wearable physiological reading.

    Row count and each continuous/datetime column's n_unique scale with
    n_observations_planned, since a real study run under these
    assumptions would produce that many rows; whether all of them are
    actually captured is a question for the Simulation stage, not this
    schema.
    """

    n_planned = assumptions.n_observations_planned

    columns = (
        ColumnProfile(
            name="participant_id",
            dtype="int64",
            n_missing=0,
            pct_missing=0.0,
            n_unique=assumptions.n_participants,
            role=ROLE_IDENTIFIER,
        ),
        ColumnProfile(
            name="timestamp",
            dtype="datetime64[ns]",
            n_missing=0,
            pct_missing=0.0,
            n_unique=n_planned,
            role=ROLE_DATETIME,
        ),
        ColumnProfile(
            name="pain_state",
            dtype="object",
            n_missing=0,
            pct_missing=0.0,
            n_unique=N_PAIN_STATES,
            role=ROLE_CATEGORICAL,
        ),
        ColumnProfile(
            name="pain_rating",
            dtype="float64",
            n_missing=0,
            pct_missing=0.0,
            n_unique=n_planned,
            role=ROLE_CONTINUOUS,
        ),
        ColumnProfile(
            name="physio_signal",
            dtype="float64",
            n_missing=0,
            pct_missing=0.0,
            n_unique=n_planned,
            role=ROLE_CONTINUOUS,
        ),
    )

    return DataProfile(n_rows=n_planned, n_columns=len(columns), columns=columns)
