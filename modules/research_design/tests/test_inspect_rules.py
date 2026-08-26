"""
Unit tests for core/inspect_rules.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.inspect_rules import (  # noqa: E402
    COMPARISON_BETWEEN_PERSON,
    COMPARISON_WITHIN_PERSON,
    DESIGN_TYPE_EXPERIMENTAL,
    DESIGN_TYPE_OBSERVATIONAL,
    MEASUREMENT_TYPE_SURVEY_SCALE,
    MEASUREMENT_TYPE_WEARABLE_SENSOR,
    TIME_STRUCTURE_CROSS_SECTIONAL,
    TIME_STRUCTURE_LONGITUDINAL,
    StudyStructure,
    inspect_study_structure,
)


def _structure(**overrides):
    kwargs = dict(
        design_type=DESIGN_TYPE_OBSERVATIONAL,
        comparison_structure=COMPARISON_WITHIN_PERSON,
        time_structure=TIME_STRUCTURE_CROSS_SECTIONAL,
        measurement_types=(),
        compares_subgroups=False,
    )
    kwargs.update(overrides)
    return StudyStructure(**kwargs)


class TestStudyStructureValidation(unittest.TestCase):
    def test_valid_construction(self):
        structure = _structure()
        self.assertEqual(structure.design_type, DESIGN_TYPE_OBSERVATIONAL)

    def test_unknown_design_type_raises(self):
        with self.assertRaises(ValueError):
            _structure(design_type="Quasi-experimental")

    def test_unknown_comparison_structure_raises(self):
        with self.assertRaises(ValueError):
            _structure(comparison_structure="Mixed")

    def test_unknown_time_structure_raises(self):
        with self.assertRaises(ValueError):
            _structure(time_structure="Panel")

    def test_unknown_measurement_type_raises(self):
        with self.assertRaises(ValueError):
            _structure(measurement_types=("Telepathy",))

    def test_empty_measurement_types_is_allowed(self):
        structure = _structure(measurement_types=())
        self.assertEqual(structure.measurement_types, ())


class TestInspectStudyStructure(unittest.TestCase):
    def test_no_rules_fire_for_minimal_structure(self):
        structure = _structure()
        self.assertEqual(inspect_study_structure(structure), ())

    def test_experimental_between_person_fires_assignment_rule(self):
        structure = _structure(
            design_type=DESIGN_TYPE_EXPERIMENTAL,
            comparison_structure=COMPARISON_BETWEEN_PERSON,
        )
        triggers = {i.trigger for i in inspect_study_structure(structure)}
        self.assertIn("Experimental, between-person comparison", triggers)

    def test_experimental_within_person_does_not_fire_assignment_rule(self):
        structure = _structure(
            design_type=DESIGN_TYPE_EXPERIMENTAL,
            comparison_structure=COMPARISON_WITHIN_PERSON,
        )
        triggers = {i.trigger for i in inspect_study_structure(structure)}
        self.assertNotIn("Experimental, between-person comparison", triggers)

    def test_longitudinal_fires_two_rules(self):
        structure = _structure(time_structure=TIME_STRUCTURE_LONGITUDINAL)
        inspections = inspect_study_structure(structure)
        self.assertEqual(len(inspections), 2)
        modules = {i.suggested_module for i in inspections}
        self.assertIn("Time-Series QA", modules)

    def test_cross_sectional_does_not_fire_longitudinal_rules(self):
        structure = _structure(time_structure=TIME_STRUCTURE_CROSS_SECTIONAL)
        modules = {i.suggested_module for i in inspect_study_structure(structure)}
        self.assertNotIn("Time-Series QA", modules)

    def test_survey_scale_fires_reliability_rule(self):
        structure = _structure(measurement_types=(MEASUREMENT_TYPE_SURVEY_SCALE,))
        modules = {i.suggested_module for i in inspect_study_structure(structure)}
        self.assertIn("Reliability", modules)

    def test_wearable_alone_does_not_fire_reliability_rule(self):
        structure = _structure(measurement_types=(MEASUREMENT_TYPE_WEARABLE_SENSOR,))
        modules = {i.suggested_module for i in inspect_study_structure(structure)}
        self.assertNotIn("Reliability", modules)

    def test_subgroup_comparison_fires_fairness_rule(self):
        structure = _structure(compares_subgroups=True)
        modules = {i.suggested_module for i in inspect_study_structure(structure)}
        self.assertIn("Fairness", modules)

    def test_no_subgroup_comparison_does_not_fire_fairness_rule(self):
        structure = _structure(compares_subgroups=False)
        modules = {i.suggested_module for i in inspect_study_structure(structure)}
        self.assertNotIn("Fairness", modules)

    def test_all_rules_can_fire_together(self):
        structure = _structure(
            design_type=DESIGN_TYPE_EXPERIMENTAL,
            comparison_structure=COMPARISON_BETWEEN_PERSON,
            time_structure=TIME_STRUCTURE_LONGITUDINAL,
            measurement_types=(MEASUREMENT_TYPE_SURVEY_SCALE,),
            compares_subgroups=True,
        )
        self.assertEqual(len(inspect_study_structure(structure)), 5)


if __name__ == "__main__":
    unittest.main()
