"""
Tests for shared/pipeline_figures.py.

The drawings show the caller's data and nothing more: one lane per
modality, each modality's facts in its tap text, recorded versus
illustrative labels, and the recorded EEG numbers where given.

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import re
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from modules.signal_pipeline.core import agency as ag, intervention as iv, modality as mc, pipeline as pc  # noqa: E402
from shared import pipeline_figures as pf  # noqa: E402

DATA = ROOT / "modules" / "signal_pipeline" / "sample_data"
COLORS = {"Neural": "#2a78d6", "Autonomic": "#eb6834", "Muscular": "#1baf7a", "Behavioral": "#eda100",
          "Subjective": "#e87ba4", "Environmental": "#008300", "Institutional": "#4a3aa7"}
NOTES = {s: f"what {s} does" for s in pc.STAGES}
HANDOFFS = {s: f"output of {s}" for s in pc.STAGES[:-1]}


def _modalities():
    frame = pd.read_csv(DATA / "modality_profiles.csv")
    return tuple(
        mc.Modality(
            name=r["name"], category=r["category"], signal_examples=r["signal_examples"],
            interpretive_gain=r["interpretive_gain"], privacy_cost=r["privacy_cost"],
            security_cost=r["security_cost"], agency_cost=r["agency_cost"], citation=r["citation"],
            is_body_sensed=bool(r["is_body_sensed"]),
            body_region="" if pd.isna(r["body_region"]) else r["body_region"],
            privacy_exit_point=r["privacy_exit_point"], agency_control_point=r["agency_control_point"],
        )
        for _, r in frame.iterrows()
    )


def _eeg():
    frame = pd.read_csv(DATA / "ds005420_sub-5_O1_eyes_snippet.csv")
    closed = frame.loc[frame["condition"] == "eyes_closed", "voltage_uv"].to_numpy()
    opened = frame.loc[frame["condition"] == "eyes_open", "voltage_uv"].to_numpy()
    w = iv.labelled_windows(iv.classify_eyes_state(closed, opened, 500.0))
    return {"raw": list(closed[:500:4]), "filtered": list(closed[:500:4]), "powers": list(w.powers),
            "labels": list(w.labelled_closed), "threshold": w.threshold, "decision_value": w.powers[0],
            "decision_label": "eyes closed"}


def _agency(mods):
    return {m.category: ag.agency_profile(m.category) for m in mods}


def _svg(html: str) -> ET.Element:
    return ET.fromstring(re.search(r"<svg.*?</svg>", html, re.S).group(0))


class TestPipeline(unittest.TestCase):
    def test_well_formed_with_one_and_with_all_modalities(self):
        mods = _modalities()
        for subset in (mods[:1], mods):
            with self.subTest(n=len(subset)):
                _svg(pf.pipeline_html(subset, pc.STAGES, "Processing", NOTES, HANDOFFS, "loop", COLORS, eeg=_eeg()))

    def test_every_stage_and_handoff_is_shown(self):
        html = pf.pipeline_html(_modalities()[:2], pc.STAGES, "Processing", NOTES, HANDOFFS, "loop", COLORS, eeg=_eeg())
        for stage in pc.STAGES:
            self.assertIn(f">{stage}<", html)
        for text in HANDOFFS.values():
            self.assertIn(text, html)

    def test_recorded_and_illustrative_are_labelled(self):
        mods = _modalities()
        html = pf.pipeline_html(mods, pc.STAGES, "Processing", NOTES, HANDOFFS, "loop", COLORS, eeg=_eeg())
        self.assertIn(">recorded<", html)
        self.assertIn(">illustrative<", html)

    def test_convergence_is_marked_only_with_several_modalities(self):
        one = pf.pipeline_html(_modalities()[:1], pc.STAGES, "Processing", NOTES, HANDOFFS, "loop", COLORS, eeg=_eeg())
        many = pf.pipeline_html(_modalities()[:3], pc.STAGES, "Processing", NOTES, HANDOFFS, "loop", COLORS, eeg=_eeg())
        self.assertNotIn("Signals meet here", one)
        self.assertIn("Signals meet here", many)

    def test_decision_shows_the_recorded_numbers(self):
        eeg = _eeg()
        html = pf.pipeline_html(_modalities()[:1], pc.STAGES, "Processing", NOTES, HANDOFFS, "loop", COLORS, eeg=eeg)
        self.assertIn(f"{eeg['decision_value']:.1f}", html)
        self.assertIn(f"{eeg['threshold']:.1f}", html)

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            pf.pipeline_html((), pc.STAGES, "Processing", NOTES, HANDOFFS, "loop", COLORS)
        with self.assertRaises(ValueError):
            pf.pipeline_html(_modalities(), pc.STAGES, "Nowhere", NOTES, HANDOFFS, "loop", COLORS)


class TestBody(unittest.TestCase):
    def test_every_perspective_is_well_formed(self):
        for view in ("Build", "Privacy", "Agency"):
            with self.subTest(view=view):
                _svg(pf.body_html(_modalities(), view, COLORS, agency=_agency(_modalities())))

    def test_tap_text_carries_each_modalitys_own_facts(self):
        mods = _modalities()
        privacy = pf.body_html(mods, "Privacy", COLORS)
        for m in mods:
            with self.subTest(modality=m.name):
                self.assertIn(m.privacy_exit_point[:40].replace("&", "&amp;"), privacy.replace("\\u2019", "'"))

    def test_agency_tap_text_quotes_each_answers_evidence(self):
        mods = _modalities()
        html = pf.body_html(mods, "Agency", COLORS, agency=_agency(mods))
        for m in mods:
            for a in ag.agency_profile(m.category).answers:
                if a.evidence:
                    with self.subTest(modality=m.name, question=a.key):
                        self.assertIn(a.evidence.replace(",", ","), html)

    def test_agency_view_draws_no_rating(self):
        mods = _modalities()
        html = pf.body_html(mods, "Agency", COLORS, agency=_agency(mods))
        for m in mods:
            self.assertNotIn(f"{m.agency_cost:.2f}", html)
        self.assertNotIn("agency cost", html.lower().replace("illustrative agency cost rating", ""))

    def test_agency_view_needs_the_answers(self):
        with self.assertRaises(ValueError):
            pf.body_html(_modalities(), "Agency", COLORS)

    def test_never_body_sensed_modalities_are_drawn_outside_the_body(self):
        html = pf.body_html(_modalities(), "Build", COLORS)
        self.assertIn("Never sensed from the body.", html)

    def test_bad_perspective_raises(self):
        with self.assertRaises(ValueError):
            pf.body_html(_modalities(), "Vibes", COLORS)


if __name__ == "__main__":
    unittest.main()
