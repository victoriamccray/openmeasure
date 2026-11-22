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

    def test_classic_and_modern_temporal_names_point_to_each_other(self):
        # Acharya et al. (2016): T3, T4, T5, T6 renamed T7, T8, P7, P8.
        self.assertIn("Newer charts call this position T7.", na.electrode_site("T3").naming_note)
        self.assertIn("Classic 10-20 charts call this position T5.", na.electrode_site("P7").naming_note)

    def test_both_names_of_a_position_agree_on_the_lobe(self):
        for classic, modern in na.CLASSIC_TO_MODERN.items():
            with self.subTest(position=classic):
                a, b = na.electrode_site(classic), na.electrode_site(modern)
                self.assertEqual((a.system, a.hemisphere), (b.system, b.hemisphere))

    def test_labels_outside_the_10_20_set_raise(self):
        for bad in ("", "   ", "O3", "FC1", "X1", None):
            with self.subTest(label=bad), self.assertRaises(na.UnknownElectrodeError):
                na.electrode_site(bad)

    def test_lookup_errors_share_a_base_class(self):
        self.assertTrue(issubclass(na.UnknownElectrodeError, LookupError))


class TestCoverage(unittest.TestCase):
    def test_eeg_covers_the_five_cortical_systems(self):
        # Scalp EEG is dominated by cortical pyramidal neurons (Buzsaki et al., 2012).
        eeg = na.modality("eeg")
        covered = [c.system for c in eeg.coverage if c.level == na.COVERED]

        self.assertEqual(covered, list(na.CORTICAL))

    def test_eeg_coverage_of_deep_structures_is_partial(self):
        # Seeber et al. (2019): detectable with high-density source imaging.
        self.assertEqual(na.coverage("eeg", "hippocampus_amygdala").level, na.PARTIAL)

    def test_forehead_fnirs_covers_only_the_frontal_lobe(self):
        # fNIRS samples the outer cortex beneath its optodes (Pinti et al., 2020).
        fnirs = na.modality("fnirs")
        covered = [c.system for c in fnirs.coverage if c.level == na.COVERED]

        self.assertEqual(covered, ["frontal"])
        self.assertIn("forehead", fnirs.configuration)
        self.assertIn("optodes placed over it", fnirs.coverage_for("occipital").note)

    def test_fnirs_reaches_no_deep_structure(self):
        for key in ("thalamus", "hippocampus_amygdala", "brainstem"):
            with self.subTest(system=key):
                self.assertEqual(na.coverage("fnirs", key).level, na.OUT_OF_RANGE)

    def test_fmri_covers_every_brain_system_and_no_body_system(self):
        fmri = na.modality("fmri")

        for key in na.BRAIN:
            self.assertEqual(fmri.coverage_for(key).level, na.COVERED)
        self.assertEqual(fmri.coverage_for("autonomic").level, na.OUT_OF_RANGE)

    def test_only_eeg_measures_electrical_activity(self):
        # Coverage and measurement are separate: fMRI and fNIRS cover regions
        # through blood oxygen, structural MRI through anatomy.
        kinds = {m.key: m.measurement for m in na.MODALITIES}

        self.assertEqual(
            kinds,
            {"eeg": "electrical", "fnirs": "hemodynamic", "fmri": "hemodynamic",
             "smri": "anatomy", "peripheral": "peripheral"},
        )

    def test_structural_mri_face_matching_figures(self):
        # Schwarz et al. (2019): 70 of 84 top match, 80 of 84 within the top five.
        text = na.modality("smri").also_captures

        self.assertIn("70 of 84", text)
        self.assertIn(f"({round(70 / 84 * 100)}%)", text)
        self.assertIn("80 of 84", text)
        self.assertIn(f"({round(80 / 84 * 100)}%)", text)

    def test_defacing_is_described_as_reducing_risk(self):
        smri = na.modality("smri")

        self.assertIn("lowers this risk", smri.privacy_note)
        self.assertIn("schwarz_2021", smri.refs)

    def test_body_sensors_cover_only_the_autonomic_system(self):
        covered = [c.system for c in na.modality("peripheral").coverage if c.level == na.COVERED]

        self.assertEqual(covered, ["autonomic"])
        self.assertEqual(na.coverage("peripheral", "brainstem").level, na.OUT_OF_RANGE)

    def test_every_modality_has_an_entry_for_every_system(self):
        keys = [s.key for s in na.SYSTEMS]
        for m in na.MODALITIES:
            with self.subTest(modality=m.key):
                self.assertEqual([c.system for c in m.coverage], keys)

    def test_only_eeg_is_the_journey_modality(self):
        self.assertEqual([m.key for m in na.MODALITIES if m.in_journey], ["eeg"])

    def test_unknown_keys_raise(self):
        with self.assertRaises(na.UnknownModalityError):
            na.modality("meg")
        with self.assertRaises(na.UnknownSystemError):
            na.coverage("eeg", "spleen")

    def test_invalid_levels_measurements_and_depths_are_rejected(self):
        with self.assertRaises(ValueError):
            na.SystemCoverage("eeg", "frontal", "strong")
        with self.assertRaises(ValueError):
            na.BrainSystem("x", "X", "here", "does things", "middle", ("kandel",))
        eeg = na.modality("eeg")
        with self.assertRaises(ValueError):
            dataclasses.replace(eeg, measurement="magnetic")

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
        cited = set(atlas.caveat_refs) | set(atlas.recording_electrode.refs) | set(atlas.electrode_caveat_refs)
        for item in (*atlas.systems, *atlas.modalities):
            cited.update(item.refs)

        self.assertEqual({r.key for r in atlas.references}, cited)

    def test_prose_follows_the_experience_style_rule(self):
        # Same rule as shared/tests/test_become_the_signal.py.
        pattern = re.compile(r" - |–|—|, not |not an? |rather than", re.I)
        atlas = na.atlas()
        texts = [atlas.caveat, atlas.electrode_caveat]
        texts += [e.naming_note for e in atlas.electrodes]
        for s in atlas.systems:
            texts += [s.name, s.location, s.role]
        for m in atlas.modalities:
            texts += [m.name, m.configuration, m.measures, m.neural_link, m.timing, m.also_captures, m.privacy_note]
            texts += [c.note for c in m.coverage]

        self.assertEqual([t for t in texts if pattern.search(t)], [])


if __name__ == "__main__":
    unittest.main()
