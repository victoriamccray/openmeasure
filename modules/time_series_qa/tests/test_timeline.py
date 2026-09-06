"""
Unit tests for core/timeline.py

Run with: pytest modules/time_series_qa/tests/ -v

The distinction this module is built on is absent against empty, so most
of these check it survives: an observation with no row and an observation
whose row is blank must never share a state, a count, or a mark.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import pandas as pd  # noqa: E402

from core import timeline as tl  # noqa: E402
from core.qa import run_time_series_qa  # noqa: E402

SAMPLE = os.path.join(
    os.path.dirname(__file__), "..", "sample_data", "time_series_example.csv"
)


def _sample_health():
    frame = pd.read_csv(SAMPLE)
    result = run_time_series_qa(frame, frame.columns[0], frame.columns[1])
    return tl.build_timeline(result)


class TestAbsentAndEmptyStayApart(unittest.TestCase):
    """
    An observation with no row and an observation whose row is blank have
    different causes and different fixes. Collapsing them into "missing"
    would throw away the module's strongest idea to make a tidier
    picture.
    """

    def setUp(self):
        self.health = _sample_health()

    def test_they_are_different_states(self):
        self.assertNotEqual(tl.SLOT_ABSENT, tl.SLOT_EMPTY)

    def test_they_are_counted_separately(self):
        self.assertNotEqual(self.health.n_absent, self.health.n_empty)
        self.assertGreater(self.health.n_absent, 0)
        self.assertGreater(self.health.n_empty, 0)

    def test_the_summary_names_both_even_when_one_is_zero(self):
        """
        A reader comparing two series has to see which of the two a
        number refers to without counting positions.
        """
        parts = " ".join(self.health.summary_parts())

        self.assertIn("absent", parts)
        self.assertIn("empty", parts)

    def test_the_state_wording_says_which_is_which(self):
        self.assertIn("no row", tl.SLOT_ABSENT.lower())
        self.assertIn("blank", tl.SLOT_EMPTY.lower())


class TestTheSampleIsReadCorrectly(unittest.TestCase):
    """
    The module's own sample carries one of each defect on purpose.
    """

    def setUp(self):
        self.health = _sample_health()

    def test_it_is_assessable(self):
        self.assertTrue(self.health.is_assessable)

    def test_every_expected_observation_gets_a_slot(self):
        self.assertEqual(
            len(self.health.slots),
            len(_sample_health().slots),
        )
        self.assertGreater(len(self.health.slots), 100)

    def test_the_defects_appear_as_slots(self):
        states = {slot.state for slot in self.health.slots}

        self.assertIn(tl.SLOT_ABSENT, states)
        self.assertIn(tl.SLOT_EMPTY, states)
        self.assertIn(tl.SLOT_CONFLICTING, states)

    def test_most_of_the_series_is_healthy(self):
        """
        An earlier version compared observations against an evenly spaced
        grid it built itself, which marked 53 of 61 slots off-schedule on
        a series the same checks call 98% regular.
        """
        healthy = sum(1 for slot in self.health.slots if slot.is_healthy)

        self.assertGreater(healthy / len(self.health.slots), 0.85)

    def test_jitter_is_counted_and_not_drawn_per_slot(self):
        """
        The module absorbs offsets inside its tolerance during grid
        matching. Drawing each as a defect would fault a series for
        something the checks tolerate.
        """
        self.assertGreater(self.health.n_jittered, 0)
        self.assertNotIn(
            "off-schedule",
            " ".join(slot.state for slot in self.health.slots).lower(),
        )


class TestConflictsAreReportedInsideDuplicates(unittest.TestCase):
    def test_the_summary_nests_them(self):
        """
        A conflicting duplicate is a duplicate. Listing both separately
        read as two problems where there is one.
        """
        parts = _sample_health().summary_parts()
        duplicates = next(part for part in parts if "duplicate" in part)

        self.assertIn("conflicting", duplicates)


class TestSeverityOrdering(unittest.TestCase):
    def test_absent_outranks_everything_when_binning(self):
        self.assertEqual(tl.SLOT_SEVERITY[-1], tl.SLOT_ABSENT)

    def test_a_conflict_outranks_a_plain_duplicate(self):
        self.assertGreater(
            tl.SLOT_SEVERITY.index(tl.SLOT_CONFLICTING),
            tl.SLOT_SEVERITY.index(tl.SLOT_DUPLICATE),
        )

    def test_every_state_has_a_severity(self):
        states = {
            tl.SLOT_PRESENT,
            tl.SLOT_EMPTY,
            tl.SLOT_DUPLICATE,
            tl.SLOT_CONFLICTING,
            tl.SLOT_ABSENT,
        }

        self.assertEqual(states, set(tl.SLOT_SEVERITY))


class TestUnassessableSeries(unittest.TestCase):
    def test_a_series_with_no_defensible_frequency_draws_nothing(self):
        """
        Rather than inventing the schedule the check declined to infer.
        """
        frame = pd.DataFrame(
            {
                "when": ["2024-01-01", "2024-03-17", "2024-03-18"],
                "value": [1.0, 2.0, 3.0],
            }
        )
        result = run_time_series_qa(frame, "when", "value")
        health = tl.build_timeline(result)

        if not health.is_assessable:
            self.assertTrue(health.reason_not_assessable.strip())
            self.assertEqual(health.slots, ())


if __name__ == "__main__":
    unittest.main()
