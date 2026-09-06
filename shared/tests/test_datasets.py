"""
Unit tests for shared/datasets.py

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest

from shared.catalog import WORKFLOWS
from shared.datasets import (
    ACCESS_LEVELS,
    ACCESS_OPEN,
    DATASETS,
    DELIVERY_BUNDLED_PUBLIC,
    DELIVERY_CACHED_PUBLIC,
    DELIVERY_MODES,
    DELIVERY_REMOTE_FETCH,
    DELIVERY_UPLOAD_ONLY,
    DataSource,
    RealDataset,
)

_WORKFLOW_NAMES = {item.workflow for item in WORKFLOWS}


class TestDatasetFields(unittest.TestCase):
    def test_ids_are_unique(self):
        ids = [dataset.id for dataset in DATASETS]

        self.assertEqual(len(ids), len(set(ids)))

    def test_required_fields_are_populated(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset.id):
                self.assertTrue(dataset.name)
                self.assertTrue(dataset.domain)
                self.assertTrue(dataset.description)
                self.assertTrue(dataset.explore_question)
                self.assertTrue(dataset.try_with)
                self.assertTrue(dataset.sources)

    def test_empty_field_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            RealDataset(
                id="",
                name="Nameless",
                domain="Some domain",
                description="A description.",
                try_with=("Reliability",),
                explore_question="A question?",
                access="Open",
                delivery=DELIVERY_UPLOAD_ONLY,
                redistribution_permitted=False,
                sources=(DataSource(label="Source", url="https://example.org"),),
            )

    def test_entries_are_frozen(self):
        self.assertTrue(RealDataset.__dataclass_params__.frozen)
        self.assertTrue(DataSource.__dataclass_params__.frozen)


class TestAccessLevels(unittest.TestCase):
    def test_every_dataset_has_a_declared_access_level(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset.id):
                self.assertIn(dataset.access, ACCESS_LEVELS)

    def test_unknown_access_level_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            RealDataset(
                id="bogus",
                name="Bogus",
                domain="Some domain",
                description="A description.",
                try_with=("Reliability",),
                explore_question="A question?",
                access="Free for all",
                delivery=DELIVERY_UPLOAD_ONLY,
                redistribution_permitted=False,
                sources=(DataSource(label="Source", url="https://example.org"),),
            )

        self.assertIn("not one of the declared access levels", str(context.exception))


class TestWorkflowReferences(unittest.TestCase):
    def test_every_try_with_entry_matches_a_known_workflow(self):
        for dataset in DATASETS:
            for workflow_name in dataset.try_with:
                with self.subTest(dataset=dataset.id, workflow=workflow_name):
                    self.assertIn(workflow_name, _WORKFLOW_NAMES)

    def test_unknown_workflow_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            RealDataset(
                id="orphan",
                name="Orphan",
                domain="Some domain",
                description="A description.",
                try_with=("Not A Real Workflow",),
                explore_question="A question?",
                access="Open",
                delivery=DELIVERY_UPLOAD_ONLY,
                redistribution_permitted=False,
                sources=(DataSource(label="Source", url="https://example.org"),),
            )

        self.assertIn("do not match any workflow", str(context.exception))

    def test_no_try_with_entries_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            RealDataset(
                id="aimless",
                name="Aimless",
                domain="Some domain",
                description="A description.",
                try_with=(),
                explore_question="A question?",
                access="Open",
                delivery=DELIVERY_UPLOAD_ONLY,
                redistribution_permitted=False,
                sources=(DataSource(label="Source", url="https://example.org"),),
            )

        self.assertIn("does not name any workflow", str(context.exception))


class TestSources(unittest.TestCase):
    def test_every_source_is_an_https_link(self):
        for dataset in DATASETS:
            for source in dataset.sources:
                with self.subTest(dataset=dataset.id, source=source.label):
                    self.assertTrue(source.url.startswith("https://"))

    def test_non_https_source_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            DataSource(label="Insecure", url="http://example.org")

        self.assertIn("not an https link", str(context.exception))

    def test_no_sources_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as context:
            RealDataset(
                id="sourceless",
                name="Sourceless",
                domain="Some domain",
                description="A description.",
                try_with=("Reliability",),
                explore_question="A question?",
                access="Open",
                delivery=DELIVERY_UPLOAD_ONLY,
                redistribution_permitted=False,
                sources=(),
            )

        self.assertIn("lists no sources", str(context.exception))


class TestWastewaterSourceIsTheActualNetwork(unittest.TestCase):
    """
    v0.2C audit fix: the wastewater entry originally pointed only at CDC's
    national NWSS dashboard, but the equity study it's paired with analyzed
    New York State's own statewide network. Pins the correction: the more
    specific NY source must stay listed, so a future edit can't quietly
    drop back to citing only the generic national aggregator.
    """

    def test_wastewater_entry_lists_the_ny_state_source(self):
        dataset = next(
            d for d in DATASETS if d.id == "wastewater_surveillance_equity"
        )
        source_urls = [source.url for source in dataset.sources]

        self.assertTrue(
            any("health.data.ny.gov" in url for url in source_urls),
            f"Expected a health.data.ny.gov source; got: {source_urls}",
        )


class TestDoesNotLeakIntoValidationLifecycle(unittest.TestCase):
    """
    Explore Real Data must never look like a validation workflow.

    A dataset gaining a module_key or a lifecycle stage would let it appear
    on the overview's progress cards or stage strip, which would misstate
    what this page does: point at data, not record an analysis of it.
    """

    def test_real_dataset_has_no_module_key_field(self):
        field_names = {
            field.name for field in RealDataset.__dataclass_fields__.values()
        }

        self.assertNotIn("module_key", field_names)
        self.assertNotIn("stage", field_names)
        self.assertNotIn("category", field_names)


if __name__ == "__main__":
    unittest.main()


class TestDeliveryMode(unittest.TestCase):
    """
    How a dataset reaches a workflow, which is a different question from
    whether a reader is allowed to obtain it.
    """

    @staticmethod
    def _dataset(**overrides):
        fields = {
            "id": "probe",
            "name": "Probe",
            "domain": "Some domain",
            "description": "A description.",
            "try_with": ("Reliability",),
            "explore_question": "A question?",
            "access": ACCESS_OPEN,
            "delivery": DELIVERY_UPLOAD_ONLY,
            "redistribution_permitted": False,
            "sources": (DataSource(label="Source", url="https://example.org"),),
        }
        fields.update(overrides)
        return RealDataset(**fields)

    def test_every_declared_dataset_names_a_known_delivery_mode(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset.id):
                self.assertIn(dataset.delivery, DELIVERY_MODES)

    def test_an_unknown_delivery_mode_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            self._dataset(delivery="Magic")

        self.assertIn("not one of the declared delivery modes", str(raised.exception))

    def test_delivery_and_redistribution_are_required_not_defaulted(self):
        """
        A silent default would be exactly the provenance loss this
        catalog exists to prevent, and the safe value differs per
        dataset.
        """
        fields = {
            "id": "probe",
            "name": "Probe",
            "domain": "Some domain",
            "description": "A description.",
            "try_with": ("Reliability",),
            "explore_question": "A question?",
            "access": ACCESS_OPEN,
            "sources": (DataSource(label="Source", url="https://example.org"),),
        }

        with self.assertRaises(TypeError):
            RealDataset(**fields)


class TestRedistributionIsSeparateFromAccess(unittest.TestCase):
    """
    The distinction this pair of fields exists for. HealthRing is the
    case in this repository that proves one field cannot carry both:
    openly downloadable from Zenodo, and deliberately not redistributed
    here.
    """

    def test_an_open_dataset_can_still_be_non_redistributable(self):
        healthring = next(d for d in DATASETS if d.id == "healthring")

        self.assertEqual(healthring.access, ACCESS_OPEN)
        self.assertFalse(healthring.redistribution_permitted)

    def test_bundling_without_permission_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            TestDeliveryMode._dataset(
                delivery=DELIVERY_BUNDLED_PUBLIC,
                redistribution_permitted=False,
            )

        self.assertIn("do not permit redistribution", str(raised.exception))

    def test_caching_without_permission_is_rejected(self):
        """
        A runtime cache counts. On a hosted deployment one fetch serves
        every visitor, which is closer to distributing the file than to a
        reader keeping their own download.
        """
        with self.assertRaises(ValueError) as raised:
            TestDeliveryMode._dataset(
                delivery=DELIVERY_CACHED_PUBLIC,
                redistribution_permitted=False,
            )

        self.assertIn("do not permit redistribution", str(raised.exception))

    def test_fetching_on_request_needs_no_redistribution_permission(self):
        """
        Retrieving a file to the reader's own session is not OpenMeasure
        handing out a copy, so an open but non-redistributable dataset
        can still be fetched.
        """
        dataset = TestDeliveryMode._dataset(
            delivery=DELIVERY_REMOTE_FETCH, redistribution_permitted=False
        )

        self.assertEqual(dataset.delivery, DELIVERY_REMOTE_FETCH)

    def test_bundling_with_permission_is_allowed(self):
        dataset = TestDeliveryMode._dataset(
            delivery=DELIVERY_BUNDLED_PUBLIC, redistribution_permitted=True
        )

        self.assertEqual(dataset.delivery, DELIVERY_BUNDLED_PUBLIC)

    # Datasets whose terms have actually been read, with the licence that
    # establishes the permission. Adding a name here is the deliberate act
    # the test below exists to force: a redistribution claim cannot drift
    # in, it has to be argued for in this list.
    REDISTRIBUTION_ESTABLISHED = {
        "diabetes_130_hospitals": "CC BY 4.0, stated on its UCI record",
        "nhanes_dpq_phq9": "US federal public-use file, not subject to domestic copyright",
        "right_to_play_baseline": "CC BY 4.0, PLOS applies it to the works it publishes",
    }

    def test_redistribution_is_claimed_only_where_it_was_established(self):
        """
        Every other entry stays conservative. Loosening one should be a
        deliberate edit that trips this test and gets justified here,
        rather than something that arrives unnoticed.
        """
        for dataset in DATASETS:
            with self.subTest(dataset=dataset.id):
                self.assertEqual(
                    dataset.redistribution_permitted,
                    dataset.id in self.REDISTRIBUTION_ESTABLISHED,
                    f"{dataset.id} claims redistribution without a recorded "
                    "licence, or records one it does not claim.",
                )

    def test_an_established_licence_names_what_established_it(self):
        for dataset_id, licence in self.REDISTRIBUTION_ESTABLISHED.items():
            with self.subTest(dataset=dataset_id):
                self.assertTrue(licence.strip())
                self.assertIn(dataset_id, {d.id for d in DATASETS})

    def test_a_permitted_dataset_may_still_be_fetched_rather_than_bundled(self):
        """
        Permission to bundle is not an instruction to. Diabetes 130 is
        CC BY 4.0 and could legally carry a local copy; it is fetched on
        request because 101,766 rows is not a repository-sized file.
        """
        diabetes = next(d for d in DATASETS if d.id == "diabetes_130_hospitals")

        self.assertTrue(diabetes.redistribution_permitted)
        self.assertEqual(diabetes.delivery, DELIVERY_REMOTE_FETCH)


class TestNhanesProvenanceContract(unittest.TestCase):
    """
    The NHANES entry exists because three things about the file have to be
    decided before an internal-consistency estimate means anything. All
    three were found by reading the file rather than the codebook, and an
    entry that quietly lost them would send a reader to real data with
    none of the warnings that made it worth featuring.
    """

    @staticmethod
    def _entry():
        return next(d for d in DATASETS if d.id == "nhanes_dpq_phq9")

    def test_it_is_fetched_rather_than_bundled(self):
        entry = self._entry()

        self.assertEqual(entry.delivery, DELIVERY_REMOTE_FETCH)
        self.assertTrue(entry.redistribution_permitted)

    def test_it_is_offered_to_the_reliability_workflow(self):
        self.assertEqual(self._entry().try_with, ("Reliability",))

    def test_the_refused_and_dont_know_codes_are_named(self):
        """
        7 and 9 are not scores on a 0-to-3 item. Read as scores they
        become extreme values and corrupt alpha silently, which is the
        failure this description exists to pre-empt.
        """
        description = self._entry().description

        self.assertIn("7 and 9", description)
        self.assertIn("Refused", description)

    def test_the_item_that_is_not_part_of_the_scale_is_named(self):
        """DPQ100 measures functional difficulty, not a symptom."""
        self.assertIn("DPQ100", self._entry().description)

    def test_the_zero_decoding_artifact_is_named(self):
        """
        pandas' xport reader returns the zero category as a denormalized
        float rather than 0, so any equality test against 0 fails. Not a
        property of the data, but a reader hits it either way.
        """
        self.assertIn("5.4e-79", self._entry().description)

    def test_both_the_codebook_and_the_data_file_are_linked(self):
        """
        A reader needs the codebook to interpret the codes and the file to
        read them; one without the other is not enough to act on.
        """
        labels = " ".join(source.label for source in self._entry().sources)

        self.assertIn("codebook", labels.lower())
        self.assertIn("data file", labels.lower())

    def test_the_data_file_link_points_at_the_transport_file(self):
        urls = [source.url for source in self._entry().sources]

        self.assertTrue(any(url.endswith(".xpt") for url in urls))

    def test_the_instrument_is_named_so_items_can_be_shown_by_wording(self):
        """
        The reason this beat Right To Play for Reliability: PHQ-9 is
        freely available, so item diagnostics can carry actual question
        text rather than variable names.
        """
        entry = self._entry()

        self.assertIn("PHQ-9", entry.name)
        self.assertIn("freely available", entry.description)


class TestRightToPlayBoundaries(unittest.TestCase):
    """
    This entry is worth featuring because of what it cannot support, not
    in spite of it. Two different kinds of limit meet in one study, and
    an entry that lost either would offer a reader a real trial with the
    impression that all of it is reproducible.
    """

    @staticmethod
    def _entry():
        return next(d for d in DATASETS if d.id == "right_to_play_baseline")

    def test_it_is_offered_to_impact_evaluation_only(self):
        """
        The scales in it would suit Reliability too, but CDI-2 is
        commercially published, so item wording could not be shown.
        NHANES carries that workflow instead.
        """
        self.assertEqual(self._entry().try_with, ("Impact Evaluation",))

    def test_the_entry_says_it_is_baseline_only(self):
        entry = self._entry()

        self.assertIn("Baseline", entry.description)
        self.assertIn("baseline", entry.name.lower())

    def test_the_randomisation_structure_is_described_as_present(self):
        """
        School, Group and Gender are in the public file, which is what
        makes the design inspectable rather than only described.
        """
        description = self._entry().description

        for variable in ("School", "Group", "Gender"):
            with self.subTest(variable=variable):
                self.assertIn(variable, description)

    def test_the_verified_shape_is_recorded(self):
        """1,752 by 350, confirmed by reading the file, not the paper."""
        description = self._entry().description

        self.assertIn("1,752", description)
        self.assertIn("350", description)

    def test_the_sensitive_content_is_disclosed_in_the_entry_itself(self):
        """
        Not left to documentation. A reader should know what they are
        opening before they open it.
        """
        description = self._entry().description

        self.assertIn("victimized", description)
        self.assertIn("depression", description)
        self.assertIn("de-identified", description)

    def test_both_the_baseline_and_the_trial_results_are_linked(self):
        """
        The published effects live in a different paper from the data, and
        a reader needs both to see where reproduction stops.
        """
        labels = " ".join(source.label for source in self._entry().sources)

        self.assertIn("baseline dataset", labels.lower())
        self.assertIn("24 months", labels.lower())

    def test_the_file_format_is_stated(self):
        """
        OpenMeasure cannot read .sav today. The catalog says what the
        dataset is; whether to add that capability is a separate call.
        """
        self.assertIn(".sav", self._entry().description)

