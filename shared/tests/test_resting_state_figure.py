"""
Tests for shared/resting_state_figure.py.

Each drawing must show what core/resting_state.py computes, and stay
well-formed SVG.

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.signal_pipeline.core import resting_state as rs  # noqa: E402
from shared import resting_state_figure as fig  # noqa: E402


class TestDrawings(unittest.TestCase):
    def test_every_drawing_is_well_formed(self):
        for svg in (
            fig.participant_svg(True),
            fig.participant_svg(False),
            fig.bands_svg(),
            fig.reactivity_svg(rs.reactivity([80], [10])),
            fig.accuracy_svg(rs.chance(27, 30)),
            fig.skin_conductance_svg(),
        ):
            ET.fromstring(svg)

    def test_participant_shows_the_condition(self):
        self.assertIn("Eyes closed, at rest", fig.participant_svg(True))
        self.assertIn("Eyes open, at rest", fig.participant_svg(False))
        # Open eyes are drawn with a pupil; closed eyes are a lid line.
        self.assertIn('fill="#2a1a10"', fig.participant_svg(False))
        self.assertNotIn('fill="#2a1a10"', fig.participant_svg(True))

    def test_accuracy_marks_the_binomial_threshold(self):
        svg = fig.accuracy_svg(rs.chance(27, 30))

        self.assertIn("beats chance from 20/30", svg)
        self.assertIn(">27/30<", svg)

    def test_reactivity_labels_the_ratio(self):
        self.assertIn(">8.0×<", fig.reactivity_svg(rs.reactivity([80], [10])))

    def test_extreme_ratios_stay_on_the_scale(self):
        for ratio_inputs in (([1000], [1]), ([1], [1000])):
            svg = fig.reactivity_svg(rs.reactivity(*ratio_inputs))
            marker = ET.fromstring(svg).find("{http://www.w3.org/2000/svg}circle")
            self.assertTrue(10 <= float(marker.get("cx")) <= 350)

    def test_bands_label_every_band(self):
        svg = fig.bands_svg()
        for band in rs.BANDS:
            self.assertIn(f">{band.name}<", svg)


if __name__ == "__main__":
    unittest.main()
