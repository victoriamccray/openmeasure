"""
Unit tests for shared/dataset_loaders.py

No network. These cover the registry, which is the part that decides
what a page may offer; the fetch itself is exercised by using it.

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest

from shared.dataset_loaders import LOADERS, can_load, loadable_dataset_ids
from shared.datasets import DATASETS, DATASET_IDS, get_dataset


class TestGetDataset(unittest.TestCase):
    def test_returns_the_requested_dataset(self):
        self.assertEqual(get_dataset("nhanes_dpq_phq9").id, "nhanes_dpq_phq9")

    def test_unknown_id_raises_and_names_the_known_ones(self):
        with self.assertRaises(ValueError) as raised:
            get_dataset("nhanes")

        message = str(raised.exception)
        self.assertIn("nhanes", message)
        self.assertIn("nhanes_dpq_phq9", message)

    def test_every_catalogued_id_is_retrievable(self):
        for dataset_id in DATASET_IDS:
            with self.subTest(dataset=dataset_id):
                self.assertEqual(get_dataset(dataset_id).id, dataset_id)


class TestLoaderRegistry(unittest.TestCase):
    """
    The catalog says what a dataset is; this says what OpenMeasure can
    open. Membership is the whole point, so absence has to be meaningful.
    """

    def test_every_loader_names_a_catalogued_dataset(self):
        for dataset_id in LOADERS:
            with self.subTest(dataset=dataset_id):
                self.assertIn(dataset_id, DATASET_IDS)

    def test_nhanes_is_loadable(self):
        self.assertTrue(can_load("nhanes_dpq_phq9"))

    def test_the_formats_openmeasure_cannot_read_are_not_claimed(self):
        """
        Right To Play is .sav and OpenMesh is NetCDF. Both are catalogued
        honestly and neither has a loader, which is what lets a page say
        so rather than offer a button that fails.
        """
        for dataset_id in ("right_to_play_baseline", "openmesh_nyc"):
            with self.subTest(dataset=dataset_id):
                self.assertFalse(can_load(dataset_id))

    def test_an_unknown_id_raises_rather_than_reporting_unloadable(self):
        """A typo is a mistake, not a dataset OpenMeasure cannot open."""
        with self.assertRaises(ValueError):
            can_load("nhanes_dpq")

    def test_loadable_ids_follow_catalog_order(self):
        catalogued = [d.id for d in DATASETS if d.id in LOADERS]

        self.assertEqual(list(loadable_dataset_ids()), catalogued)

    def test_a_dataset_without_a_loader_is_the_normal_case(self):
        """
        Most catalogued datasets are not loadable, and that is expected
        rather than a gap to close: a loader is written when a workflow
        needs one.
        """
        self.assertLess(len(LOADERS), len(DATASET_IDS))


if __name__ == "__main__":
    unittest.main()
