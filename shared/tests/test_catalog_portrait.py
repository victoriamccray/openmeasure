"""
Unit tests for the catalog portrait in shared/portraits.py

Run with: pytest shared/tests/ -v

Kept separate from test_portraits.py because this builder takes a
RealDataset rather than a DataProfile, and the point of it being a
separate builder is that those describe different things: one is
measured from a file someone has, the other describes a dataset most
readers have not obtained.
"""

from __future__ import annotations

import unittest

from shared.datasets import DATASETS, SCALE_FACT_LIMIT, get_dataset
from shared.portraits import (
    _STRIP_FONT,
    _STRIP_GAP,
    _STRIP_MAX_FACTS,
    _STRIP_W,
    _text_width,
    catalog_portrait_svg,
)


class TestEveryEntryDraws(unittest.TestCase):
    def test_no_entry_raises(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset.id):
                catalog_portrait_svg(dataset)

    def test_every_drawn_fact_fits_the_canvas(self):
        """
        SVG text does not wrap, so a strip wider than its viewBox loses
        its last fact off the edge rather than showing it clipped.
        """
        for dataset in DATASETS:
            facts = dataset.scale[:_STRIP_MAX_FACTS]
            if not facts:
                continue

            width = (
                58.0
                + sum(_text_width(fact, _STRIP_FONT) for fact in facts)
                + (len(facts) - 1) * _STRIP_GAP
            )
            with self.subTest(dataset=dataset.id):
                self.assertLess(width, _STRIP_W)

    def test_the_facts_appear_in_the_figure(self):
        dataset = get_dataset("nhanes_dpq_phq9")
        svg = catalog_portrait_svg(dataset)

        for fact in dataset.scale:
            with self.subTest(fact=fact):
                self.assertIn(fact, svg)


class TestAnEntryWithoutVerifiedCountsDrawsNothing(unittest.TestCase):
    def test_an_empty_scale_returns_an_empty_string(self):
        """
        Several entries state no counts, because none were verified. An
        estimate drawn where a verified figure goes would be
        indistinguishable from one.
        """
        from dataclasses import replace

        dataset = replace(get_dataset("nhanes_dpq_phq9"), scale=())

        self.assertEqual(catalog_portrait_svg(dataset), "")


class TestScaleFactsComeFromTheDescription(unittest.TestCase):
    """
    The strip promotes facts the entry already states. It is not a place
    to add a number that was never verified against the source.
    """

    NUMERIC_ENTRIES = (
        "diabetes_130_hospitals",
        "nhanes_dpq_phq9",
        "right_to_play_baseline",
        "nwss_wastewater_metrics",
        "healthring",
    )

    def test_every_count_in_a_fact_also_appears_in_the_description(self):
        import re

        for dataset_id in self.NUMERIC_ENTRIES:
            dataset = get_dataset(dataset_id)
            for fact in dataset.scale:
                for number in re.findall(r"[\d,]*\d", fact):
                    with self.subTest(dataset=dataset_id, number=number):
                        self.assertIn(number, dataset.description)


class TestTextWidth(unittest.TestCase):
    def test_a_wider_string_estimates_wider(self):
        self.assertGreater(
            _text_width("MMMM", 15.0), _text_width("iiii", 15.0)
        )

    def test_it_scales_with_font_size(self):
        self.assertAlmostEqual(
            _text_width("abc", 30.0), 2 * _text_width("abc", 15.0)
        )

    def test_an_empty_string_has_no_width(self):
        self.assertEqual(_text_width("", 15.0), 0.0)


class TestFacetsAreClosedSets(unittest.TestCase):
    """
    Browsing by a facet must not silently miss a dataset filed under a
    near-synonym.
    """

    def test_every_entry_declares_a_modality_and_a_structure(self):
        from shared.datasets import MODALITIES, STRUCTURES

        for dataset in DATASETS:
            with self.subTest(dataset=dataset.id):
                self.assertIn(dataset.modality, MODALITIES)
                self.assertIn(dataset.structure, STRUCTURES)

    def test_a_fact_too_long_for_the_strip_is_rejected(self):
        from dataclasses import replace

        with self.assertRaises(ValueError) as raised:
            replace(get_dataset("nhanes_dpq_phq9"), scale=("x" * 80,))

        self.assertIn(str(SCALE_FACT_LIMIT), str(raised.exception))
