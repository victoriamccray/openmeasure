"""
Unit tests for shared/data_handling.py

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

from shared.data_handling import (
    DATA_ACCESS_BUNDLED_EXAMPLE,
    DATA_ACCESS_USER_UPLOAD,
    DISCLOSURES,
    PERSISTENT_STORAGE_NONE,
    PROCESSING_IN_MEMORY,
    REDISTRIBUTION_NOT_APPLICABLE,
    DataHandlingDisclosure,
    disclosure_for,
)

ROOT = Path(__file__).resolve().parents[2]


def _disclosure(**overrides) -> DataHandlingDisclosure:
    defaults = dict(
        page="pages/Example.py",
        data_access=(DATA_ACCESS_USER_UPLOAD,),
        processing=(PROCESSING_IN_MEMORY,),
        persistent_storage=PERSISTENT_STORAGE_NONE,
        data_accessed="Example columns.",
        redistribution=REDISTRIBUTION_NOT_APPLICABLE,
    )
    defaults.update(overrides)
    return DataHandlingDisclosure(**defaults)


class TestDataHandlingDisclosure(unittest.TestCase):
    def test_valid_disclosure_constructs(self):
        disclosure = _disclosure()
        self.assertEqual(disclosure.page, "pages/Example.py")

    def test_raises_on_empty_page(self):
        with self.assertRaises(ValueError):
            _disclosure(page="")

    def test_raises_on_empty_data_accessed(self):
        with self.assertRaises(ValueError):
            _disclosure(data_accessed="")

    def test_raises_on_empty_data_access_tuple(self):
        with self.assertRaises(ValueError):
            _disclosure(data_access=())

    def test_raises_on_empty_processing_tuple(self):
        with self.assertRaises(ValueError):
            _disclosure(processing=())

    def test_raises_on_unknown_data_access_value(self):
        with self.assertRaises(ValueError):
            _disclosure(data_access=("Telepathy",))

    def test_raises_on_unknown_processing_value(self):
        with self.assertRaises(ValueError):
            _disclosure(processing=("Osmosis",))

    def test_raises_on_unknown_persistent_storage_value(self):
        with self.assertRaises(ValueError):
            _disclosure(persistent_storage="Forever")

    def test_raises_on_unknown_redistribution_value(self):
        with self.assertRaises(ValueError):
            _disclosure(redistribution="Sold to advertisers")

    def test_multiple_data_access_values_are_allowed(self):
        disclosure = _disclosure(
            data_access=(DATA_ACCESS_USER_UPLOAD, DATA_ACCESS_BUNDLED_EXAMPLE)
        )
        self.assertEqual(
            disclosure.data_access, (DATA_ACCESS_USER_UPLOAD, DATA_ACCESS_BUNDLED_EXAMPLE)
        )


class TestDisclosureFor(unittest.TestCase):
    def test_returns_the_matching_entry(self):
        result = disclosure_for("pages/GAIA_Worked_Example.py")
        self.assertEqual(result.page, "pages/GAIA_Worked_Example.py")

    def test_raises_on_unregistered_page(self):
        with self.assertRaises(ValueError):
            disclosure_for("pages/Does_Not_Exist.py")


class TestDisclosuresRegistry(unittest.TestCase):
    def test_no_duplicate_pages(self):
        pages = [item.page for item in DISCLOSURES]
        self.assertEqual(len(pages), len(set(pages)))

    def test_every_registered_page_exists_on_disk(self):
        for item in DISCLOSURES:
            with self.subTest(page=item.page):
                self.assertTrue(
                    (ROOT / item.page).exists(),
                    f"{item.page} is registered but not found on disk.",
                )


if __name__ == "__main__":
    unittest.main()
