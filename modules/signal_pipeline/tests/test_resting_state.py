"""
Unit tests for core/resting_state.py

Run with: pytest modules/signal_pipeline/tests/ -v
"""

from __future__ import annotations

import dataclasses
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import resting_state as rs  # noqa: E402


class TestChance(unittest.TestCase):
    def test_thirty_windows_need_twenty_correct(self):
        # Hand-calculable: P(X >= 20 | n=30, p=0.5) = 0.0494 < 0.05,
        # and P(X >= 19) = 0.1002 > 0.05 (Combrisson & Jerbi, 2015).
        result = rs.chance(27, 30)

        self.assertEqual(result.threshold_correct, 20)
        self.assertAlmostEqual(result.threshold_p_value, 0.0494, places=4)
        self.assertAlmostEqual(result.threshold_accuracy, 20 / 30)

    def test_twenty_seven_of_thirty_is_far_above_chance(self):
        # P(X >= 27 | n=30, p=0.5) = (C(30,27)+C(30,28)+C(30,29)+C(30,30)) / 2^30
        #                          = (4060 + 435 + 30 + 1) / 1073741824
        result = rs.chance(27, 30)

        self.assertAlmostEqual(result.p_value, 4526 / 2**30, places=12)
        self.assertTrue(result.exceeds_threshold)
        self.assertEqual(result.chance_accuracy, 0.5)

    def test_nineteen_of_thirty_does_not_clear_the_bar(self):
        self.assertFalse(rs.chance(19, 30).exceeds_threshold)

    def test_four_classes_lower_the_chance_level(self):
        self.assertEqual(rs.chance(10, 20, n_classes=4).chance_accuracy, 0.25)

    def test_too_few_trials_to_ever_beat_chance_raises(self):
        # With 4 trials a perfect score has P = 1/16 = 0.0625 > 0.05.
        with self.assertRaises(rs.RestingStateError):
            rs.chance(4, 4)

    def test_invalid_inputs_raise(self):
        for args in ((31, 30), (-1, 30), (5, 0), (5, 10, 1)):
            with self.subTest(args=args), self.assertRaises(rs.RestingStateError):
                rs.chance(*args)
        with self.assertRaises(rs.RestingStateError):
            rs.chance(5, 10, significance=1.5)


class TestReactivity(unittest.TestCase):
    def test_ratio_of_means(self):
        result = rs.reactivity([10, 30], [5, 5])

        self.assertEqual((result.mean_closed, result.mean_open, result.ratio), (20.0, 5.0, 4.0))
        self.assertTrue(result.direction_as_expected)

    def test_midpoint_is_the_classifier_threshold(self):
        # Hand-calculable: (77.8408 + 9.2448) / 2 = 43.5428, the threshold
        # intervention.classify_eyes_state reports for the bundled recording.
        result = rs.reactivity([77.8408], [9.2448])

        self.assertAlmostEqual(result.midpoint, 43.5428, places=4)
        explanation = rs.explain_threshold(result)
        self.assertIn("(77.8 + 9.2) / 2 = 43.5 µV²", explanation.text)
        self.assertIn("same windows it labels", explanation.text)

    def test_midpoint_matches_the_pipeline_threshold(self):
        import numpy as np
        import pandas as pd
        from core import intervention

        frame = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "..", "sample_data", "ds005420_sub-5_O1_eyes_snippet.csv")
        )
        signal = {c: frame.loc[frame["condition"] == c, "voltage_uv"].to_numpy() for c in ("eyes_closed", "eyes_open")}
        classified = intervention.classify_eyes_state(signal["eyes_closed"], signal["eyes_open"], intervention.SAMPLING_RATE_HZ)
        result = rs.reactivity(classified.alpha_power_eyes_closed, classified.alpha_power_eyes_open)

        self.assertTrue(np.isclose(result.midpoint, classified.threshold))

    def test_reversed_direction_is_reported(self):
        self.assertFalse(rs.reactivity([1], [2]).direction_as_expected)

    def test_degenerate_inputs_raise(self):
        for closed, opened in (([], [1]), ([1], []), ([1], [0]), ([-1], [1])):
            with self.subTest(closed=closed, opened=opened), self.assertRaises(rs.RestingStateError):
                rs.reactivity(closed, opened)


class TestContent(unittest.TestCase):
    def test_study_matches_barry_2007(self):
        # PubMed 17911042: 28 university students, EEG and skin conductance.
        self.assertEqual(rs.STUDY.participants, "28 university students.")
        self.assertEqual(len(rs.STUDY.recordings), 2)

    def test_ageing_study_matches_barry_and_de_blasio_2017(self):
        # PubMed 28943465: 20 young (20.4) and 20 older (68.2) adults.
        self.assertIn("mean age 68.2", rs.AGEING_NOTE.participants)

    def test_bands_are_contiguous_and_alpha_matches_the_pipeline(self):
        from core import intervention

        for lower, upper in zip(rs.BANDS, rs.BANDS[1:]):
            self.assertEqual(lower.high_hz, upper.low_hz)
        alpha = next(b for b in rs.BANDS if b.name == "Alpha")
        self.assertEqual((alpha.low_hz, alpha.high_hz), intervention.ALPHA_BAND_HZ)

    def test_every_measure_names_how_its_range_is_known(self):
        kinds = {m.key: m.range_kind for m in rs.MEASURES}

        self.assertEqual(kinds["accuracy"], rs.RANGE_CALCULATED)
        self.assertEqual(kinds["reactivity"], rs.RANGE_DIRECTION)
        self.assertEqual(kinds["alpha_power"], rs.RANGE_WITHIN_PERSON)

    def test_every_modality_has_both_axes(self):
        by_modality: dict[str, set[str]] = {}
        for d in rs.DICHOTOMIES:
            by_modality.setdefault(d.modality, set()).add(d.axis)
        for modality, axes in by_modality.items():
            with self.subTest(modality=modality):
                self.assertEqual(axes, set(rs.AXES))

    def test_skin_conductance_is_labeled_a_published_finding(self):
        # The recording has no skin conductance channel.
        sources = {m.key: m.source for m in rs.MEASURES}

        self.assertEqual(sources.pop("skin_conductance"), rs.SOURCE_PUBLISHED)
        self.assertEqual(sources.pop("frequency"), rs.SOURCE_CONVENTION)
        self.assertEqual(set(sources.values()), {rs.SOURCE_RECORDED})

    def test_voltage_reading_credits_synchrony(self):
        self.assertIn("synchronized", rs.measure("voltage").how_to_read)
        self.assertNotIn("more neurons", rs.measure("voltage").how_to_read)

    def test_every_cited_key_has_a_reference(self):
        self.assertEqual(set(rs.cited_keys()), {r.key for r in rs.REFERENCES})

    def test_prose_follows_the_style_rule(self):
        pattern = re.compile(r" - |–|—|, not |not an? |rather than", re.I)
        texts = [rs.BAND_CAVEAT, *(b.plain for b in rs.BANDS)]
        for study in (rs.STUDY, rs.AGEING_NOTE):
            texts += [study.participants, study.protocol, study.interpretation, *study.findings, *study.recordings]
        for m in rs.MEASURES:
            texts += [m.what_it_is, m.how_computed, m.how_to_read, m.expected, m.what_shifts_it]
        texts.append(rs.explain_threshold(rs.reactivity([77.8], [9.2])).text)
        for d in rs.DICHOTOMIES:
            texts += [d.first, d.second]

        self.assertEqual([t for t in texts if pattern.search(t)], [])

    def test_invalid_records_are_rejected_and_results_frozen(self):
        with self.assertRaises(ValueError):
            rs.Dichotomy("EEG", "speed", "a", "b", ("barry",))
        with self.assertRaises(ValueError):
            dataclasses.replace(rs.measure("accuracy"), range_kind="vibes")
        with self.assertRaises(rs.RestingStateError):
            rs.measure("loudness")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            rs.STUDY.participants = "x"


if __name__ == "__main__":
    unittest.main()
