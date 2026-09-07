"""
Unit tests for shared/measure_visuals.py

Run with: pytest shared/tests/test_measure_visuals.py -v

Two things have to hold at once. Every measure has to be drawable, and
measures that produce genuinely different artifacts have to be drawn
differently: the earlier abstract vocabulary satisfied the first and
failed the second, which is what these tests are here to stop happening
again. The rest check that the drawings are structurally sound SVG and
that the motion on the sampled traces is optional.
"""

from __future__ import annotations

import re
import unittest
import xml.etree.ElementTree as ElementTree

from modules.research_design.core import ontology
from shared.measure_visuals import (
    ANIMATED_TYPES,
    CARD_H,
    CARD_W,
    builders_cover_the_vocabulary,
    measure_visual_svg,
)


def _svg(measure) -> str:
    return measure_visual_svg(measure)


class TestEveryMeasureCanBeDrawn(unittest.TestCase):
    def test_the_builders_cover_the_declared_vocabulary(self):
        """
        A type in the vocabulary with no builder renders as a gap in a
        row of measures, which is the one failure a reader cannot
        interpret.
        """
        self.assertTrue(builders_cover_the_vocabulary())

    def test_no_measure_raises(self):
        for measure in ontology.MEASURES:
            with self.subTest(measure=measure.name):
                self.assertTrue(_svg(measure))

    def test_an_unbuildable_visual_type_raises_rather_than_drawing_nothing(self):
        from dataclasses import replace

        measure = replace(ontology.MEASURES[0])
        object.__setattr__(measure, "visual_type", "hologram")

        with self.assertRaises(ValueError) as raised:
            measure_visual_svg(measure)

        self.assertIn("has no builder", str(raised.exception))

    def test_every_drawing_is_well_formed_svg(self):
        """
        These go out through unsafe_allow_html, so a malformed path or an
        unclosed group is a broken page rather than a broken figure.
        """
        for measure in ontology.MEASURES:
            with self.subTest(measure=measure.name):
                ElementTree.fromstring(_svg(measure))


class TestDifferentArtifactsAreDrawnDifferently(unittest.TestCase):
    """
    The failure this replaces: one line stood for EDA, ECG and an
    ambient sensor, so the explorer showed the same picture for three
    different decisions.
    """

    def test_the_physiological_measures_do_not_share_a_drawing(self):
        eda = ontology.get_measure("Electrodermal activity")
        ecg = ontology.get_measure("Heart rate and heart-rate variability")
        sensor = ontology.get_measure("Environmental sensor")

        types = {eda.visual_type, ecg.visual_type, sensor.visual_type}

        self.assertEqual(len(types), 3)
        self.assertEqual(len({_svg(m) for m in (eda, ecg, sensor)}), 3)

    def test_a_repeated_daily_rating_is_not_drawn_as_a_single_item(self):
        """
        Compliance is what separates the two, and a scale drawn once
        cannot show a missed prompt.
        """
        survey = ontology.get_measure("Structured survey")
        diary = ontology.get_measure("Rating scale, repeated in daily life")

        self.assertNotEqual(survey.visual_type, diary.visual_type)

    def test_an_assay_and_a_held_out_evaluation_are_not_the_same_picture(self):
        assay = ontology.get_measure("Assay of a biological sample")
        evaluation = ontology.get_measure("Model evaluation on held-out data")

        self.assertNotEqual(assay.visual_type, evaluation.visual_type)

    def test_a_debt_burden_is_drawn_against_a_whole_not_as_a_checklist(self):
        debt = ontology.get_measure("Debt burden measure")
        hardship = ontology.get_measure("Material hardship indicators")

        self.assertEqual(debt.visual_type, ontology.VISUAL_RATIO_BAR)
        self.assertNotEqual(debt.visual_type, hardship.visual_type)


class TestTypesAreStillSharedWhereTheArtifactIsTheSame(unittest.TestCase):
    """
    Precision is not one drawing per measure. Three measures that all
    return rows a system already wrote down are one record table.
    """

    def test_the_record_measures_share_one_table(self):
        shared_type = {
            ontology.get_measure(name).visual_type
            for name in (
                "Administrative records extract",
                "Clinical assessment or chart review",
                "Income or earnings record",
            )
        }

        self.assertEqual(shared_type, {ontology.VISUAL_RECORD})

    def test_the_language_measures_share_one_transcript(self):
        shared_type = {
            ontology.get_measure(name).visual_type
            for name in (
                "Semi-structured interview",
                "Think-aloud protocol",
                "Document or artifact analysis",
            )
        }

        self.assertEqual(shared_type, {ontology.VISUAL_TEXT})

    def test_there_are_fewer_types_than_measures(self):
        used = [measure.visual_type for measure in ontology.MEASURES]

        self.assertLess(len(set(used)), len(used))

    def test_every_declared_type_is_actually_used(self):
        """A type nothing uses is dead weight."""
        used = {measure.visual_type for measure in ontology.MEASURES}

        self.assertEqual(used, set(ontology.VISUAL_TYPES))


class TestTheMotionIsOptional(unittest.TestCase):
    def test_only_the_sampled_traces_move(self):
        """
        Motion here means "this arrives as a stream". On anything else it
        would be decoration.
        """
        moving = {
            measure.visual_type
            for measure in ontology.MEASURES
            if "om-measure-sweep" in _svg(measure)
        }

        self.assertEqual(moving, set(ANIMATED_TYPES))

    def test_reduced_motion_turns_it_off(self):
        ecg = ontology.get_measure("Heart rate and heart-rate variability")

        self.assertIn("prefers-reduced-motion:reduce", _svg(ecg))

    def test_the_trace_is_drawn_without_the_animation_too(self):
        """
        The highlight is a second copy of the geometry, so switching it
        off leaves the whole trace rather than a gap.
        """
        ecg = ontology.get_measure("Heart rate and heart-rate variability")
        traces = re.findall(r'<polyline points="([^"]+)"', _svg(ecg))

        self.assertEqual(len(traces), 2)
        self.assertEqual(traces[0], traces[1])


class TestTheFigureDescribesItself(unittest.TestCase):
    def test_the_label_names_the_measure_and_the_drawing(self):
        measure = ontology.get_measure("Functional neuroimaging")
        label = _svg(measure).split('aria-label="')[1].split('"')[0]

        self.assertIn("Functional neuroimaging", label)
        self.assertIn("brain slice", label)

    def test_every_drawing_is_made_at_one_size(self):
        """
        So a row of candidate measures reads as a row of alternatives
        rather than as differently sized pictures.
        """
        boxes = {
            _svg(measure).split('viewBox="')[1].split('"')[0]
            for measure in ontology.MEASURES
        }

        self.assertEqual(boxes, {f"0 0 {CARD_W:.0f} {CARD_H:.0f}"})

    def test_the_drawings_stay_inside_the_box(self):
        """
        A coordinate outside the viewBox is clipped, which reads as a
        drawing mistake rather than as a cropped one.
        """
        for measure in ontology.MEASURES:
            numbers = [
                float(value)
                for value in re.findall(
                    r'(?:cx|cy|x|y|x1|y1|x2|y2)="(-?[\d.]+)"', _svg(measure)
                )
            ]

            with self.subTest(measure=measure.name):
                self.assertGreaterEqual(min(numbers), 0)
                self.assertLessEqual(max(numbers), max(CARD_W, CARD_H))


class TestTheBodyOutlineIsSymmetric(unittest.TestCase):
    def test_the_left_and_right_halves_mirror(self):
        """
        Built by mirroring one half rather than typed twice, so an
        anatomical figure cannot drift lopsided.
        """
        from shared.measure_visuals import _BODY_HALF, _body_map

        drawing = _body_map()
        offsets = [
            round(float(x) - 75.0, 1)
            for x, _ in re.findall(r"([\d.]+) ([\d.]+)", drawing.split('d="M ')[1])
        ]

        self.assertEqual(
            sorted(round(-dx, 1) for dx in offsets if dx > 0),
            sorted(dx for dx in offsets if dx < 0),
        )
        self.assertEqual(len(_BODY_HALF) * 2 - 1, len(offsets))


if __name__ == "__main__":
    unittest.main()
