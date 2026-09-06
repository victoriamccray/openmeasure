"""
Unit tests for shared/visuals.py

The primitives return SVG strings, so they are testable without a running
app. What is worth pinning is the grammar they encode, not the geometry:
a convention that drifts silently is worse than no convention.

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from shared.visuals import (
    ACCENT,
    DASH_UNESTABLISHED,
    INK_MUTED,
    arrow,
    figure,
    unit_cluster,
)


class TestUnitCluster(unittest.TestCase):
    def test_one_circle_per_unit(self):
        for count in (1, 3, 5, 10):
            with self.subTest(count=count):
                self.assertEqual(unit_cluster(50, 50, ACCENT, count=count).count("<circle"), count)

    def test_the_same_group_drawn_twice_looks_the_same(self):
        """
        Random offsets would reshuffle on every rerun, and a reader
        moving a control would read that as the data changing.
        """
        self.assertEqual(
            unit_cluster(40, 40, ACCENT, count=6),
            unit_cluster(40, 40, ACCENT, count=6),
        )

    def test_units_sit_near_the_point_they_are_drawn_at(self):
        svg = unit_cluster(100, 60, ACCENT, count=10)
        xs = [float(v) for v in re.findall(r'cx="([-\d.]+)"', svg)]
        ys = [float(v) for v in re.findall(r'cy="([-\d.]+)"', svg)]

        self.assertTrue(all(abs(x - 100) <= 15 for x in xs))
        self.assertTrue(all(abs(y - 60) <= 15 for y in ys))

    def test_a_count_too_large_to_be_countable_is_rejected(self):
        """
        A pictograph of forty dots is a texture, not a count, and being
        countable is the only reason to draw one.
        """
        with self.assertRaises(ValueError) as raised:
            unit_cluster(0, 0, ACCENT, count=40)

        self.assertIn("countable", str(raised.exception))

    def test_zero_units_is_rejected(self):
        with self.assertRaises(ValueError):
            unit_cluster(0, 0, ACCENT, count=0)


class TestArrowCarriesEstablishedState(unittest.TestCase):
    """
    Solid means established, dashed means not, everywhere. A reader
    learns the mark once, so it has to mean the same thing each time.
    """

    def test_an_established_arrow_is_solid(self):
        self.assertNotIn("stroke-dasharray", arrow(0, 10, 50, 10))

    def test_an_unestablished_arrow_is_dashed(self):
        self.assertIn(
            DASH_UNESTABLISHED, arrow(0, 10, 50, 10, established=False)
        )

    def test_both_forms_still_carry_a_head(self):
        for established in (True, False):
            with self.subTest(established=established):
                self.assertIn(
                    "<polygon", arrow(0, 10, 50, 10, established=established)
                )

    def test_colour_does_not_change_with_state(self):
        """
        Colour is not the signal for doubt: something unresolved is not a
        failure, and red would assert that it was.
        """
        solid = arrow(0, 10, 50, 10)
        dashed = arrow(0, 10, 50, 10, established=False)

        self.assertIn(INK_MUTED, solid)
        self.assertIn(INK_MUTED, dashed)


class TestFigure(unittest.TestCase):
    def test_it_sizes_and_labels_the_svg(self):
        svg = figure("<circle/>", width=100, height=50, label="A description.")

        self.assertIn('viewBox="0 0 100 50"', svg)
        self.assertIn('aria-label="A description."', svg)
        self.assertIn('role="img"', svg)

    def test_an_unlabelled_figure_is_rejected(self):
        """A figure nobody can read without seeing it is not finished."""
        for blank in ("", "   "):
            with self.subTest(label=repr(blank)):
                with self.assertRaises(ValueError):
                    figure("<circle/>", width=10, height=10, label=blank)


class TestNoFrameworkDependency(unittest.TestCase):
    def test_the_primitives_do_not_import_streamlit(self):
        """
        They return strings for a caller to render, the same reason
        shared/charts.py returns a spec rather than drawing one.
        """
        source = (Path(__file__).resolve().parents[1] / "visuals.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("import streamlit", source)


if __name__ == "__main__":
    unittest.main()
