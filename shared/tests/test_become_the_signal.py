"""
Tests for experiences/become-the-signal.

The experience is a static page, so its honesty rests on two committed
files: data.json (numbers) and content.json (words). These tests hold
both to their sources.

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import ast
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIENCE = ROOT / "experiences" / "become-the-signal"
CONTENT = json.loads((EXPERIENCE / "content.json").read_text(encoding="utf-8"))
DATA = json.loads((EXPERIENCE / "data.json").read_text(encoding="utf-8"))


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_become_the_signal_data", ROOT / "scripts" / "build_become_the_signal_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _page_constant(name: str):
    """A literal constant from the Multimodal Signal Convergence page, read without running it."""
    source = (ROOT / "pages" / "Multimodal_Signal_Convergence.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found on the page.")


class TestDataIsCurrent(unittest.TestCase):
    def test_committed_data_matches_the_core_functions(self):
        # A change to the core functions that moves a number fails here
        # until scripts/build_become_the_signal_data.py is rerun.
        rebuilt = json.loads(json.dumps(_load_builder().build()))

        self.assertEqual(rebuilt, DATA)

    def test_baseline_accuracy_is_the_known_value(self):
        # 27 of 30 one-second windows, the figure the Multimodal Signal
        # Convergence page reports for the same recording.
        self.assertEqual(DATA["baseline"]["accuracy"], 0.9)

    def test_too_low_a_rate_fails_rather_than_guessing(self):
        lowest = DATA["downsample"][-1]

        self.assertIsNone(lowest["accuracy"])
        self.assertIn("cannot resolve", lowest["failure"])


class TestScientificReading(unittest.TestCase):
    """The labels and accuracy the experience shows, rederived from the alpha powers."""

    def test_accuracy_follows_from_the_alpha_powers_and_threshold(self):
        # The page labels a second eyes closed when its alpha power is
        # above the threshold, the same rule classify_eyes_state applies.
        base = DATA["baseline"]
        closed, opened, t = base["alpha_power_eyes_closed"], base["alpha_power_eyes_open"], base["threshold"]
        correct = sum(p > t for p in closed) + sum(p <= t for p in opened)

        self.assertEqual((correct, len(closed) + len(opened)), (27, 30))
        self.assertAlmostEqual(correct / 30, base["accuracy"], places=4)

    def test_threshold_is_the_midpoint_of_the_class_means(self):
        base = DATA["baseline"]
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        midpoint = (mean(base["alpha_power_eyes_closed"]) + mean(base["alpha_power_eyes_open"])) / 2

        self.assertAlmostEqual(base["threshold"], midpoint, places=2)

    def test_alpha_is_higher_with_eyes_closed(self):
        # The direction Barry et al. (2007) report, which the Signals stage states.
        base = DATA["baseline"]

        self.assertGreater(
            sum(base["alpha_power_eyes_closed"]) / 15, sum(base["alpha_power_eyes_open"]) / 15
        )

    def test_every_offered_level_has_a_measured_result(self):
        decision = next(s for s in CONTENT["stages"] if s["name"] == "Decision")
        noise = {n["sd_uv"] for n in DATA["noise"]}
        factors = {d["factor"] for d in DATA["downsample"]}

        for level in decision["safeguards"]["noise"]["levels"]:
            self.assertIn(level["value"], noise)
        for level in decision["safeguards"]["downsample"]["levels"]:
            self.assertIn(level["value"], factors)

    def test_25_hz_keeps_the_inference_and_12_5_hz_loses_it(self):
        # What the downsampling safeguard's text claims, checked against the data.
        by_factor = {d["factor"]: d for d in DATA["downsample"]}

        self.assertEqual(by_factor[20]["effective_rate_hz"], 25.0)
        self.assertEqual(by_factor[20]["accuracy"], DATA["baseline"]["accuracy"])
        self.assertIsNone(by_factor[40]["accuracy"])

    def test_heavy_noise_reaches_chance(self):
        # What the noise safeguard's text claims.
        heavy = next(n for n in DATA["noise"] if n["sd_uv"] == 250.0)

        self.assertEqual(heavy["accuracy"], 0.5)


class TestContentMatchesItsSources(unittest.TestCase):
    def test_stages_are_the_pipeline_stages(self):
        from modules.signal_pipeline.core import pipeline

        names = [stage["name"] for stage in CONTENT["stages"]]

        self.assertEqual(names, list(pipeline.STAGES))
        self.assertEqual(names, DATA["stages"])

    def test_handoffs_match_the_streamlit_page(self):
        handoffs = [stage["handoff"] for stage in CONTENT["stages"]]
        page = list(_page_constant("PIPELINE_HANDOFFS")) + [_page_constant("PIPELINE_RETURN_HANDOFF")]

        self.assertEqual(handoffs[:-1], page[:-1])
        self.assertEqual(handoffs[-1], page[-1])

    def test_every_citation_key_has_a_reference(self):
        references = CONTENT["references"]
        used = set()
        for stage in CONTENT["stages"]:
            for block in (stage, stage.get("threat") or {}, stage.get("second_inference") or {}):
                used.update(block.get("refs", []))

        self.assertEqual(sorted(used - set(references)), [])
        self.assertEqual(sorted(set(references) - used), [])

    def test_every_safeguard_states_what_it_does_not_protect(self):
        decision = next(s for s in CONTENT["stages"] if s["name"] == "Decision")

        for key, safeguard in decision["safeguards"].items():
            with self.subTest(safeguard=key):
                self.assertTrue(safeguard["protects"].strip())
                self.assertTrue(safeguard["does_not"].strip())
                self.assertTrue(safeguard["acts_on"].strip())

    def test_every_stage_says_what_changed(self):
        for stage in CONTENT["stages"]:
            with self.subTest(stage=stage["name"]):
                self.assertTrue(stage["what_changed"].strip())

    def test_every_stage_has_all_three_perspectives(self):
        for stage in CONTENT["stages"]:
            with self.subTest(stage=stage["name"]):
                self.assertEqual(set(stage["perspectives"]), set(CONTENT["perspectives"]))


if __name__ == "__main__":
    unittest.main()
