"""
Tests for shared/neuroanatomy_figure.py.

The drawing must show what core/neuroanatomy.py says, and nothing more:
shading only where a sensor has coverage, in the colour of its signal type.

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import re
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.signal_pipeline.core import neuroanatomy as na  # noqa: E402
from shared import neuroanatomy_figure as fig  # noqa: E402

SVG = "{http://www.w3.org/2000/svg}"


def _shaded(svg: str, color: str) -> set[str]:
    """Names of the regions filled with this colour, read from their titles."""
    root = ET.fromstring(svg)
    names = set()
    for el in root.iter():
        if el.get("fill") == color:
            title = el.find(f"{SVG}title")
            if title is not None:
                names.add(title.text)
    return names


class TestSideView(unittest.TestCase):
    def test_every_modality_draws_well_formed_svg(self):
        for m in na.MODALITIES:
            with self.subTest(modality=m.key):
                ET.fromstring(fig.side_view_svg(m.key))

    def test_forehead_fnirs_shades_only_the_frontal_lobe(self):
        svg = fig.side_view_svg("fnirs")

        self.assertEqual(_shaded(svg, fig.MEASURE_COLOR[na.MEASURE_HEMODYNAMIC]), {"Frontal lobe"})

    def test_eeg_shades_the_cortex_in_the_electrical_colour(self):
        svg = fig.side_view_svg("eeg")
        expected = {na.system(k).name for k in na.CORTICAL}

        self.assertEqual(_shaded(svg, fig.MEASURE_COLOR[na.MEASURE_ELECTRICAL]), expected)

    def test_hemodynamic_and_electrical_colours_differ(self):
        self.assertNotEqual(
            fig.MEASURE_COLOR[na.MEASURE_ELECTRICAL], fig.MEASURE_COLOR[na.MEASURE_HEMODYNAMIC]
        )

    def test_face_is_marked_only_for_mri(self):
        for key in ("eeg", "fnirs", "fmri", "smri", "peripheral"):
            with self.subTest(modality=key):
                self.assertEqual(">Face<" in fig.side_view_svg(key), key in ("fmri", "smri"))

    def test_electrode_view_shades_no_lobe_and_shows_the_glow(self):
        svg = fig.side_view_svg("eeg", electrode_only=True)

        self.assertIn("o1-glow", svg)
        self.assertEqual(_shaded(svg, fig.MEASURE_COLOR[na.MEASURE_ELECTRICAL]), set())

    def test_unknown_keys_raise(self):
        with self.assertRaises(na.UnknownModalityError):
            fig.side_view_svg("meg")
        with self.assertRaises(na.UnknownSystemError):
            fig.side_view_svg("eeg", selected_system="spleen")


class TestElectrodeMap(unittest.TestCase):
    def test_map_draws_one_dot_per_modern_label(self):
        svg = fig.electrode_map_svg()
        drawn = re.findall(r">([A-Z][a-z]?[z0-9]+)</text>", svg)

        modern = [label for label in na.STANDARD_10_20 if label not in na.CLASSIC_TO_MODERN]
        self.assertEqual(sorted(drawn), sorted(modern))

    def test_classic_name_outlines_its_modern_position(self):
        self.assertEqual(fig.electrode_map_svg("T3"), fig.electrode_map_svg("T7").replace("T7 is outlined", "T3 is outlined"))

    def test_o1_sits_back_left(self):
        site = {label: (x, y) for label, x, y in fig.MAP_SITES}["O1"]

        self.assertLess(site[0], 0)  # left of the midline
        self.assertGreater(site[1], 0)  # behind the ears, nose is up

    def test_unknown_electrode_raises(self):
        with self.assertRaises(na.UnknownElectrodeError):
            fig.electrode_map_svg("X9")


if __name__ == "__main__":
    unittest.main()
