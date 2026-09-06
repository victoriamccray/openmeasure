"""
Unit tests for shared/measure_visuals.py

Run with: pytest shared/tests/test_measure_visuals.py -v

The vocabulary only works if it stays a vocabulary. These check that
every declared primitive can be drawn, that every measure uses one, and
that primitives are shared rather than accumulating one per measure.
"""

from __future__ import annotations

import unittest

from modules.research_design.core import ontology
from shared.measure_visuals import (
    builders_cover_the_vocabulary,
    measure_visual_svg,
)


class TestEveryMeasureCanBeDrawn(unittest.TestCase):
    def test_the_builders_cover_the_declared_vocabulary(self):
        """
        A primitive in the vocabulary with no builder renders as a gap in
        a row of measures, which is the one failure a reader cannot
        interpret.
        """
        self.assertTrue(builders_cover_the_vocabulary())

    def test_no_measure_raises(self):
        for measure in ontology.MEASURES:
            with self.subTest(measure=measure.name):
                self.assertTrue(measure_visual_svg(measure))

    def test_an_unbuildable_visual_type_raises_rather_than_drawing_nothing(self):
        from dataclasses import replace

        measure = replace(ontology.MEASURES[0])
        object.__setattr__(measure, "visual_type", "hologram")

        with self.assertRaises(ValueError) as raised:
            measure_visual_svg(measure)

        self.assertIn("has no builder", str(raised.exception))


class TestTheVocabularyStaysAVocabulary(unittest.TestCase):
    def test_every_declared_primitive_is_actually_used(self):
        """
        A primitive nothing uses is dead weight, and one measure per
        primitive would mean it had stopped being shared.
        """
        used = {measure.visual_type for measure in ontology.MEASURES}

        self.assertEqual(used, set(ontology.VISUAL_TYPES))

    def test_primitives_are_shared_across_measures(self):
        used = [measure.visual_type for measure in ontology.MEASURES]

        self.assertLess(len(set(used)), len(used))

    def test_measures_of_the_same_shape_share_a_primitive(self):
        """
        Electrodermal activity and heart-rate variability both produce a
        continuous stream, so they draw the same shape.
        """
        eda = ontology.get_measure("Electrodermal activity")
        hrv = ontology.get_measure("Heart rate and heart-rate variability")

        self.assertEqual(eda.visual_type, hrv.visual_type)


class TestTheFigureDescribesItself(unittest.TestCase):
    def test_the_label_names_what_the_data_is(self):
        measure = ontology.get_measure("Card sorting")
        label = measure_visual_svg(measure).split('aria-label="')[1].split('"')[0]

        self.assertIn("Card sorting", label)
        self.assertIn("card cluster", label)

    def test_every_primitive_is_drawn_at_one_size(self):
        """
        So a row of candidate measures reads as a row of alternatives
        rather than as differently sized pictures.
        """
        boxes = {
            measure_visual_svg(measure).split('viewBox="')[1].split('"')[0]
            for measure in ontology.MEASURES
        }

        self.assertEqual(len(boxes), 1)


if __name__ == "__main__":
    unittest.main()
