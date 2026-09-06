"""
Unit tests for core/ontology.py and core/assembly.py

Run with: pytest modules/research_design/tests/ -v

The property under test is mostly independence: that no field decides
which measures exist, that a study can be expressed in this vocabulary
whatever it is about, and that nothing a researcher did not choose
reaches their record.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import assembly, ontology  # noqa: E402


class TestTheLibraryIsFieldIndependent(unittest.TestCase):
    """
    The failure this replaces: a researcher asking about mental models in
    implementation research was offered pain ratings and body maps,
    because the measures belonged to one worked example.
    """

    def test_nothing_in_the_module_takes_a_domain(self):
        for name in dir(ontology):
            if name.startswith("_"):
                continue
            with self.subTest(name=name):
                self.assertNotIn("domain", name.lower())

    def test_a_mental_model_study_finds_its_measures(self):
        structure = ontology.measures_for(ontology.KIND_KNOWLEDGE_STRUCTURE)
        names = {measure.name for measure in structure}

        self.assertIn("Card sorting", names)
        self.assertIn("Causal or cognitive mapping", names)
        self.assertNotIn("Electrodermal activity", names)

    def test_a_physiological_study_finds_its_measures(self):
        bodily = ontology.measures_for(ontology.KIND_BODILY_PROCESS)
        names = {measure.name for measure in bodily}

        self.assertIn("Electrodermal activity", names)
        self.assertNotIn("Card sorting", names)

    def test_every_declared_kind_has_at_least_one_measure(self):
        """
        A kind nothing can observe is a kind a researcher can select and
        then be told nothing about.
        """
        for kind in ontology.CONCEPT_KINDS:
            with self.subTest(kind=kind):
                self.assertTrue(ontology.measures_for(kind))

    def test_the_library_spans_more_than_one_modality_per_kind_somewhere(self):
        """
        If every concept kind mapped to exactly one modality, the
        modality registry would be decoration.
        """
        spans = [
            len({measure.modality for measure in ontology.measures_for(kind)})
            for kind in ontology.CONCEPT_KINDS
        ]

        self.assertGreater(max(spans), 1)


class TestEveryMeasureIsTraceable(unittest.TestCase):
    def test_each_names_how_it_is_documented(self):
        for measure in ontology.MEASURES:
            with self.subTest(measure=measure.name):
                self.assertTrue(measure.documented_as.strip())
                self.assertTrue(measure.search_terms.strip())

    def test_each_states_a_limitation(self):
        """
        A catalog of methods with no limitations reads as a menu of
        equally good options, and the choice between measurement
        approaches is mostly about which limitation a study can live
        with.
        """
        for measure in ontology.MEASURES:
            with self.subTest(measure=measure.name):
                self.assertTrue(measure.limitation.strip())

    def test_a_measure_missing_a_limitation_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            ontology.Measure(
                name="Something",
                observes=(ontology.KIND_BEHAVIOR,),
                modality=ontology.MODALITY_BEHAVIORAL,
                captures="Things",
                produces="Data",
                burden="Some",
                limitation="",
                documented_as="Somewhere",
                search_terms="terms",
                visual_type=ontology.VISUAL_TEXT,
            )

        self.assertIn("limitation", str(raised.exception))

    def test_a_measure_without_a_visual_type_is_rejected(self):
        """
        A measure with no shape cannot appear in the explorer beside ones
        that have one.
        """
        with self.assertRaises(TypeError):
            ontology.Measure(
                name="Something",
                observes=(ontology.KIND_BEHAVIOR,),
                modality=ontology.MODALITY_BEHAVIORAL,
                captures="Things",
                produces="Data",
                burden="Some",
                limitation="A limit",
                documented_as="Somewhere",
                search_terms="terms",
            )

    def test_a_visual_type_outside_the_vocabulary_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            ontology.Measure(
                name="Something",
                observes=(ontology.KIND_BEHAVIOR,),
                modality=ontology.MODALITY_BEHAVIORAL,
                captures="Things",
                produces="Data",
                burden="Some",
                limitation="A limit",
                documented_as="Somewhere",
                search_terms="terms",
                visual_type="hologram",
            )

        self.assertIn("Reuse a primitive", str(raised.exception))

    def test_an_unknown_modality_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            ontology.Measure(
                name="Something",
                observes=(ontology.KIND_BEHAVIOR,),
                modality="Vibes",
                captures="Things",
                produces="Data",
                burden="Some",
                limitation="A limit",
                documented_as="Somewhere",
                search_terms="terms",
                visual_type=ontology.VISUAL_TEXT,
            )

        self.assertIn("not one of the declared modalities", str(raised.exception))

    def test_an_unknown_measure_name_raises(self):
        with self.assertRaises(ValueError) as raised:
            ontology.get_measure("Vibes-based inference")

        self.assertIn("curated and finite", str(raised.exception))


class TestCoverage(unittest.TestCase):
    """
    What the assembled measures reach, and what they do not.
    """

    def setUp(self):
        self.concepts = (
            assembly.Concept("Mental-model content", ontology.KIND_EXPERIENCE),
            assembly.Concept(
                "Mental-model structure", ontology.KIND_KNOWLEDGE_STRUCTURE
            ),
            assembly.Concept("Observed behaviour", ontology.KIND_BEHAVIOR),
        )

    def test_a_concept_nothing_reaches_is_reported_as_unobserved(self):
        study = assembly.AssembledStudy(
            concepts=self.concepts,
            selected_measures=("Semi-structured interview", "Card sorting"),
        )

        unobserved = [concept.name for concept in study.unobserved]

        self.assertEqual(unobserved, ["Observed behaviour"])

    def test_an_unobserved_concept_is_not_called_an_error(self):
        study = assembly.AssembledStudy(
            concepts=self.concepts, selected_measures=("Card sorting",)
        )
        statuses = {row.status for row in study.coverage}

        for status in statuses:
            with self.subTest(status=status):
                for blame in ("error", "missing", "invalid", "fix"):
                    self.assertNotIn(blame, status.lower())

    def test_coverage_names_which_measure_reaches_each_concept(self):
        study = assembly.AssembledStudy(
            concepts=self.concepts,
            selected_measures=("Card sorting", "Structured observation"),
        )
        by_name = {row.concept.name: row for row in study.coverage}

        self.assertIn("Card sorting", by_name["Mental-model structure"].measures)
        self.assertIn(
            "Structured observation", by_name["Observed behaviour"].measures
        )

    def test_coverage_matches_what_the_explorer_would_have_offered(self):
        """
        Kind alone counts a body map as observing pain experience,
        because both are things a person experiences. The narrowed list
        knows it is for the spatial pattern, and a measure the explorer
        would not have offered for a concept must not count as observing
        it.
        """
        study = assembly.AssembledStudy(
            concepts=(
                assembly.Concept("Pain experience", ontology.KIND_EXPERIENCE),
                assembly.Concept("Spatial pain pattern", ontology.KIND_EXPERIENCE),
            ),
            selected_measures=(
                "Rating scale, repeated in daily life",
                "Body map or pain drawing",
            ),
        )
        by_name = {row.concept.name: row.measures for row in study.coverage}

        self.assertEqual(
            by_name["Pain experience"], ("Rating scale, repeated in daily life",)
        )
        self.assertEqual(
            by_name["Spatial pain pattern"], ("Body map or pain drawing",)
        )

    def test_coverage_keeps_the_order_the_concepts_were_named_in(self):
        study = assembly.AssembledStudy(
            concepts=self.concepts, selected_measures=("Card sorting",)
        )

        self.assertEqual(
            [row.concept.name for row in study.coverage],
            [concept.name for concept in self.concepts],
        )


class TestShapeFollowsTheAssembly(unittest.TestCase):
    def test_one_modality_once_is_a_single_shape(self):
        study = assembly.AssembledStudy(
            concepts=(), selected_measures=("Card sorting",)
        )

        self.assertEqual(study.shape, assembly.SHAPE_SINGLE)

    def test_several_modalities_once_is_multimodal(self):
        study = assembly.AssembledStudy(
            concepts=(),
            selected_measures=("Card sorting", "Electrodermal activity"),
        )

        self.assertEqual(study.shape, assembly.SHAPE_MULTIMODAL)

    def test_one_modality_repeated_is_repeated(self):
        study = assembly.AssembledStudy(
            concepts=(), selected_measures=("Card sorting",), occasions=4
        )

        self.assertEqual(study.shape, assembly.SHAPE_REPEATED)

    def test_several_modalities_repeated_is_both(self):
        study = assembly.AssembledStudy(
            concepts=(),
            selected_measures=("Card sorting", "Electrodermal activity"),
            occasions=8,
        )

        self.assertEqual(study.shape, assembly.SHAPE_MULTIMODAL_REPEATED)

    def test_modalities_are_listed_in_declared_order(self):
        """
        So two studies choosing the same measures in a different sequence
        describe themselves the same way.
        """
        one = assembly.AssembledStudy(
            concepts=(),
            selected_measures=("Electrodermal activity", "Card sorting"),
        )
        other = assembly.AssembledStudy(
            concepts=(),
            selected_measures=("Card sorting", "Electrodermal activity"),
        )

        self.assertEqual(one.modalities, other.modalities)


class TestNothingUnchosenReachesTheRecord(unittest.TestCase):
    """
    The failure this was written after: a worked example's population
    printed under a researcher's own question, with nothing marking the
    join.
    """

    def test_a_blank_field_prints_as_unset(self):
        lines = assembly.design_record_lines(
            assembly.AssembledStudy(concepts=(), selected_measures=()),
            entered={"question": "Do mental models affect implementation?"},
        )
        text = "\n".join(lines)

        self.assertIn("Do mental models affect implementation?", text)
        self.assertIn(f"Population: {assembly.UNSET}", text)
        self.assertIn(f"Setting: {assembly.UNSET}", text)

    def test_no_concepts_prints_as_unset_rather_than_an_example(self):
        lines = assembly.design_record_lines(
            assembly.AssembledStudy(concepts=(), selected_measures=()),
            entered={},
        )
        text = "\n".join(lines)

        self.assertIn(assembly.UNSET, text)
        for leaked in ("pain", "EDA", "body map", "chronic"):
            with self.subTest(token=leaked):
                self.assertNotIn(leaked.lower(), text.lower())

    def test_the_record_states_what_the_design_would_not_observe(self):
        study = assembly.AssembledStudy(
            concepts=(
                assembly.Concept("Behaviour", ontology.KIND_BEHAVIOR),
                assembly.Concept("Arousal", ontology.KIND_BODILY_PROCESS),
            ),
            selected_measures=("Structured observation",),
        )
        text = "\n".join(assembly.design_record_lines(study, entered={}))

        self.assertIn("What this design would not observe", text)
        self.assertIn("- Arousal", text)

    def test_full_coverage_is_not_reported_as_completeness(self):
        """
        Reaching every concept a researcher named is not the same as
        having named the right ones.
        """
        study = assembly.AssembledStudy(
            concepts=(assembly.Concept("Behaviour", ontology.KIND_BEHAVIOR),),
            selected_measures=("Structured observation",),
        )
        text = "\n".join(assembly.design_record_lines(study, entered={}))

        self.assertIn("not of the ones that were never named", text)


class TestValidation(unittest.TestCase):
    def test_an_unknown_concept_kind_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            assembly.Concept("Something", "A vibe")

        self.assertIn("not a declared concept kind", str(raised.exception))

    def test_a_study_measured_fewer_than_once_is_rejected(self):
        with self.assertRaises(ValueError):
            assembly.AssembledStudy(
                concepts=(), selected_measures=(), occasions=0
            )

    def test_selecting_an_unknown_measure_raises_at_construction(self):
        with self.assertRaises(ValueError):
            assembly.AssembledStudy(
                concepts=(), selected_measures=("Something invented",)
            )


if __name__ == "__main__":
    unittest.main()
