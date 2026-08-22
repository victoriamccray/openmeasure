"""
Unit tests for shared/resources.py

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest

from shared.resources import KIND_ORDER, RESOURCES, Resource, resources_by_kind


class TestResourceFields(unittest.TestCase):
    def test_required_fields_are_populated(self):
        for resource in RESOURCES:
            with self.subTest(resource=resource.name):
                self.assertTrue(resource.name)
                self.assertTrue(resource.kind)
                self.assertTrue(resource.description)

    def test_url_is_optional(self):
        # Some entries are recorded by name only, with no confirmed link
        # yet -- see shared/resources.py's docstring. This must not raise.
        Resource(
            name="Placeholder",
            kind=KIND_ORDER[0],
            description="A description with no link yet.",
        )

    def test_empty_field_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            Resource(name="Nameless", kind="", description="A description.")

    def test_unknown_kind_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            Resource(
                name="Orphan",
                kind="Not A Kind",
                description="A description.",
            )

        self.assertIn("not in KIND_ORDER", str(context.exception))

    def test_non_https_url_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            Resource(
                name="Insecure",
                kind=KIND_ORDER[0],
                description="A description.",
                url="http://example.com",
            )

        self.assertIn("not an https link", str(context.exception))

    def test_entries_are_frozen(self):
        self.assertTrue(Resource.__dataclass_params__.frozen)

    def test_names_are_unique(self):
        names = [resource.name for resource in RESOURCES]

        self.assertEqual(len(names), len(set(names)))


class TestKindGrouping(unittest.TestCase):
    def test_grouping_covers_every_resource_exactly_once(self):
        grouped = resources_by_kind()
        seen = [
            resource.name
            for resources in grouped.values()
            for resource in resources
        ]

        self.assertEqual(sorted(seen), sorted(r.name for r in RESOURCES))
        self.assertEqual(len(seen), len(set(seen)))

    def test_kinds_appear_in_declared_order(self):
        grouped = resources_by_kind()
        expected = [kind for kind in KIND_ORDER if kind in grouped]

        self.assertEqual(list(grouped.keys()), expected)

    def test_no_kind_group_is_empty(self):
        for kind, resources in resources_by_kind().items():
            with self.subTest(kind=kind):
                self.assertTrue(resources)


if __name__ == "__main__":
    unittest.main()
