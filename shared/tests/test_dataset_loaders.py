"""
Unit tests for shared/dataset_loaders.py

No network. These cover the registry, which is the part that decides
what a page may offer; the fetch itself is exercised by using it.

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest

from dataclasses import replace

from shared.dataset_loaders import (
    ARTIFACT_DERIVED,
    LOADERS,
    ArtifactChecksumError,
    can_load,
    load_public_artifact,
    loadable_dataset_ids,
)
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


class TestLoadPublicArtifact(unittest.TestCase):
    """
    One path for every catalogued dataset, and no silent substitution of
    one source for another.
    """

    def test_a_committed_subset_opens_and_says_it_is_derived(self):
        loaded = load_public_artifact(get_dataset("healthring"))

        self.assertEqual(loaded.kind, ARTIFACT_DERIVED)
        self.assertTrue(loaded.is_derived)
        self.assertFalse(loaded.frame.empty)

    def test_it_carries_provenance_alongside_the_data(self):
        """
        Together, so a page showing a derived subset can say what was
        done to it without a caller having to go and find the record.
        """
        loaded = load_public_artifact(get_dataset("healthring"))

        self.assertFalse(loaded.provenance["is_the_original_dataset"])
        self.assertEqual(loaded.provenance["source"]["license"], "CC BY 4.0")
        self.assertIn("columns_excluded", loaded.provenance["transformation"])

    def test_the_subset_holds_only_the_summary_columns(self):
        """
        The waveform columns are the whole size reduction, and their
        absence is what the Signal Inspection stage has to adapt to.
        """
        columns = set(load_public_artifact(get_dataset("healthring")).frame.columns)

        self.assertIn("bvp_hr", columns)
        self.assertIn("quality", columns)
        for waveform in ("ir-filtered", "red-filtered", "ax-filtered", "fs"):
            with self.subTest(column=waveform):
                self.assertNotIn(waveform, columns)

    def test_a_dataset_with_no_route_raises_and_says_what_to_do(self):
        """
        Rather than returning something from somewhere else. A page
        silently substituting one source for another is the failure this
        catalog exists to prevent.
        """
        with self.assertRaises(ValueError) as raised:
            load_public_artifact(get_dataset("noaa_lcd_hourly"))

        message = str(raised.exception)
        self.assertIn("cannot open", message)
        self.assertIn("supply it yourself", message)

    def test_a_mismatched_checksum_refuses_to_load(self):
        """
        Verified against the catalog's declared digest, not against one
        stored beside the file: a file hashed to whatever it happens to
        contain verifies nothing.
        """
        dataset = get_dataset("healthring")
        tampered = replace(
            dataset,
            derived_artifact=replace(dataset.derived_artifact, sha256="0" * 64),
        )

        with self.assertRaises(ArtifactChecksumError) as raised:
            load_public_artifact(tampered)

        self.assertIn("come apart", str(raised.exception))


class TestDerivedArtifactsAreDeclaredHonestly(unittest.TestCase):
    def test_a_derived_artifact_requires_redistribution_permission(self):
        """
        A subset committed here is a redistribution of the work it came
        from, whatever its size.
        """
        dataset = get_dataset("healthring")

        with self.assertRaises(ValueError) as raised:
            replace(dataset, redistribution_permitted=False)

        self.assertIn("redistribution", str(raised.exception))

    def test_a_derived_artifact_does_not_change_how_the_dataset_arrives(self):
        """
        HealthRing's archive is still 2.4 GiB and still supplied by the
        reader. Recording the subset as a delivery mode would say
        OpenMeasure distributes the dataset, which it does not.
        """
        dataset = get_dataset("healthring")

        self.assertEqual(dataset.delivery, "You supply the file")
        self.assertIsNotNone(dataset.derived_artifact)

    def test_every_declared_artifact_is_present_and_matches(self):
        for dataset in DATASETS:
            if dataset.derived_artifact is None:
                continue
            with self.subTest(dataset=dataset.id):
                load_public_artifact(dataset)

