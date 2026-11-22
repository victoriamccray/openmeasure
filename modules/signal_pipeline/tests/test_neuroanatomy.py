"""
Unit tests for core/neuroanatomy.py

Run with: pytest modules/signal_pipeline/tests/ -v
"""

from __future__ import annotations

import dataclasses
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import neuroanatomy as na  # noqa: E402


class TestElectrodeSite(unittest.TestCase):
    """The 10-20 naming convention, Jasper (1958)."""

    def test_o1_is_over_the_left_occipital_lobe(self):
        site = na.electrode_site("O1")

        self.assertEqual(site.system, "occipital")
        self.assertEqual(site.hemisphere, na.HEMISPHERE_LEFT)

    def test_even_numbers_are_right_and_z_is_midline(self):
        self.assertEqual(na.electrode_site("O2").hemisphere, na.HEMISPHERE_RIGHT)
        self.assertEqual(na.electrode_site("Cz").hemisphere, na.HEMISPHERE_MIDLINE)
        self.assertEqual(na.electrode_site("C3").system, "sensorimotor")

    def test_frontal_pole_prefix_is_read_before_frontal(self):
        site = na.electrode_site("Fp2")

        self.assertEqual((site.region, site.system, site.hemisphere), ("frontal pole", "frontal", "right"))

    def test_case_is_normalized(self):
        self.assertEqual(na.electrode_site("oz").label, "Oz")
        self.assertEqual(na.electrode_site("FP1").label, "Fp1")

    def test_naming_note_explains_the_label(self):
        self.assertEqual(
            na.electrode_site("O1").naming_note,
            "O stands for occipital, and 1 is odd, so the left hemisphere.",
        )

    def test_labels_outside_the_10_20_set_raise(self):
        for bad in ("", "   ", "O3", "FC1", "X1", None):
            with self.subTest(label=bad), self.assertRaises(na.UnknownElectrodeError):
                na.electrode_site(bad)

    def test_lookup_errors_share_a_base_class(self):
        self.assertTrue(issubclass(na.UnknownElectrodeError, LookupError))


class TestReach(unittest.TestCase):
    def test_eeg_reads_the_five_cortical_systems_directly(self):
        # Scalp EEG is dominated by cortical pyramidal neurons (Buzsaki et al., 2012).
        eeg = na.modality("eeg")
        direct = [r.system for r in eeg.reach if r.level == na.REACH_DIRECT]

        self.assertEqual(direct, list(na.CORTICAL))

    def test_eeg_reach_into_deep_structures_is_limited(self):
        # Seeber et al. (2019): detectable with high-density source imaging.
        self.assertEqual(na.reach("eeg", "hippocampus_amygdala").level, na.REACH_LIMITED)

    def test_fnirs_reaches_no_deep_structure(self):
        # Near-infrared light reaches only the outer cortex (Pinti et al., 2020).
        for key in ("thalamus", "hippocampus_amygdala", "brainstem"):
            with self.subTest(system=key):
                self.assertEqual(na.reach("fnirs", key).level, na.REACH_NONE)

    def test_fmri_reaches_every_brain_system_and_no_body_system(self):
        fmri = na.modality("fmri")

        for key in na.BRAIN:
            self.assertEqual(fmri.level_for(key).level, na.REACH_DIRECT)
        self.assertEqual(fmri.level_for("autonomic").level, na.REACH_NONE)

    def test_structural_mri_reads_structure_and_captures_the_face(self):
        # Schwarz et al. (2019): 70 of 84 participants matched.
        smri = na.modality("smri")

        self.assertEqual(smri.reads, na.READS_STRUCTURE)
        self.assertIn("70 of 84", smri.also_captures)

    def test_body_sensors_read_only_the_autonomic_system_directly(self):
        direct = [r.system for r in na.modality("peripheral").reach if r.level == na.REACH_DIRECT]

        self.assertEqual(direct, ["autonomic"])

    def test_every_modality_has_an_entry_for_every_system(self):
        keys = [s.key for s in na.SYSTEMS]
        for m in na.MODALITIES:
            with self.subTest(modality=m.key):
                self.assertEqual([r.system for r in m.reach], keys)

    def test_only_eeg_is_the_journey_modality(self):
        self.assertEqual([m.key for m in na.MODALITIES if m.in_journey], ["eeg"])

    def test_unknown_keys_raise(self):
        with self.assertRaises(na.UnknownModalityError):
            na.modality("meg")
        with self.assertRaises(na.UnknownSystemError):
            na.reach("eeg", "spleen")

    def test_invalid_levels_and_depths_are_rejected(self):
        with self.assertRaises(ValueError):
            na.SystemReach("eeg", "frontal", "strong")
        with self.assertRaises(ValueError):
            na.BrainSystem("x", "X", "here", "does things", "middle", ("kandel",))

    def test_results_are_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            na.modality("eeg").short_name = "MEG"


class TestAtlas(unittest.TestCase):
    def test_recording_electrode_is_o1(self):
        self.assertEqual(na.atlas().recording_electrode.label, "O1")

    def test_every_standard_electrode_is_exported(self):
        self.assertEqual([e.label for e in na.atlas().electrodes], list(na.STANDARD_10_20))

    def test_references_are_exactly_the_cited_keys(self):
        atlas = na.atlas()
        cited = set(atlas.caveat_refs) | set(atlas.recording_electrode.refs)
        for item in (*atlas.systems, *atlas.modalities):
            cited.update(item.refs)

        self.assertEqual({r.key for r in atlas.references}, cited)

    def test_prose_follows_the_experience_style_rule(self):
        # Same rule as shared/tests/test_become_the_signal.py.
        pattern = re.compile(r" - |–|—|, not |not an? |rather than", re.I)
        atlas = na.atlas()
        texts = [atlas.caveat, atlas.recording_electrode.naming_note]
        for s in atlas.systems:
            texts += [s.name, s.location, s.role]
        for m in atlas.modalities:
            texts += [m.name, m.placement, m.measures, m.timing, m.also_captures, m.privacy_note]
            texts += [r.note for r in m.reach]

        self.assertEqual([t for t in texts if pattern.search(t)], [])


if __name__ == "__main__":
    unittest.main()
