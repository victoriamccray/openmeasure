"""
Unit tests for core/pipeline.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import modality as mod  # noqa: E402
from core import pipeline as pl  # noqa: E402


def _modality(name, category="Neural"):
    return mod.Modality(
        name=name,
        category=category,
        signal_examples="example signal",
        interpretive_gain=0.5,
        privacy_cost=0.5,
        security_cost=0.5,
        agency_cost=0.5,
        citation="Example citation (2026).",
        is_body_sensed=True,
        body_region="Example region",
        privacy_exit_point="Example privacy exit point.",
        agency_control_point="Example agency control point.",
    )


class TestBuildPipeline(unittest.TestCase):
    def test_raises_on_empty_modalities(self):
        with self.assertRaises(ValueError):
            pl.build_pipeline(())

    def test_raises_on_duplicate_modality_names(self):
        with self.assertRaises(ValueError):
            pl.build_pipeline((_modality("EEG"), _modality("EEG")))

    def test_single_modality_produces_one_node_per_stage(self):
        pipeline = pl.build_pipeline((_modality("EEG"),))
        self.assertEqual(len(pipeline.nodes), len(pl.STAGES))

    def test_two_modalities_share_every_stage_from_processing_onward(self):
        pipeline = pl.build_pipeline((_modality("EEG"), _modality("ECG")))
        # 2 Signals + 2 Sensors + 5 shared stages (Processing..Feedback) = 9
        self.assertEqual(len(pipeline.nodes), 9)

    def test_edges_connect_every_modality_into_processing(self):
        pipeline = pl.build_pipeline((_modality("EEG"), _modality("ECG")))
        processing_node = next(n for n in pipeline.nodes if n.stage == "Processing")
        incoming_to_processing = [
            edge for edge in pipeline.edges if edge.target is processing_node
        ]
        self.assertEqual(len(incoming_to_processing), 2)

    def test_shared_chain_is_fully_linear_after_processing(self):
        pipeline = pl.build_pipeline((_modality("EEG"),))
        shared_stage_names = pl.STAGES[2:]  # Processing .. Feedback
        for upstream, downstream in zip(shared_stage_names, shared_stage_names[1:]):
            with self.subTest(upstream=upstream, downstream=downstream):
                match = [
                    edge
                    for edge in pipeline.edges
                    if edge.source.stage == upstream and edge.target.stage == downstream
                ]
                self.assertEqual(len(match), 1)


class TestBuildPipelineConvergenceStage(unittest.TestCase):
    def test_default_convergence_stage_is_processing(self):
        pipeline = pl.build_pipeline((_modality("EEG"),))
        self.assertEqual(pipeline.convergence_stage, "Processing")

    def test_raises_on_unknown_convergence_stage(self):
        with self.assertRaises(ValueError):
            pl.build_pipeline((_modality("EEG"),), convergence_stage="Telepathy")

    def test_raises_when_convergence_stage_is_signals(self):
        with self.assertRaises(ValueError):
            pl.build_pipeline((_modality("EEG"),), convergence_stage="Signals")

    def test_converging_at_inference_gives_each_modality_its_own_processing_node(self):
        pipeline = pl.build_pipeline(
            (_modality("Structural"), _modality("Functional")),
            convergence_stage="Inference",
        )
        # 2 Signals + 2 Sensors + 2 Processing + 4 shared (Inference..Feedback) = 10
        self.assertEqual(len(pipeline.nodes), 10)
        processing_nodes = [n for n in pipeline.nodes if n.stage == "Processing"]
        self.assertEqual(len(processing_nodes), 2)
        self.assertTrue(all(n.modality is not None for n in processing_nodes))

    def test_converging_at_inference_shares_inference_onward(self):
        pipeline = pl.build_pipeline(
            (_modality("Structural"), _modality("Functional")),
            convergence_stage="Inference",
        )
        inference_node = next(n for n in pipeline.nodes if n.stage == "Inference")
        incoming = [edge for edge in pipeline.edges if edge.target is inference_node]
        self.assertEqual(len(incoming), 2)
        self.assertIsNone(inference_node.modality)


class TestPipelineNode(unittest.TestCase):
    def test_raises_on_unknown_stage(self):
        with self.assertRaises(ValueError):
            pl.PipelineNode(stage="Telepathy")


class TestComputeConvergence(unittest.TestCase):
    def test_raises_on_unknown_stage(self):
        pipeline = pl.build_pipeline((_modality("EEG"),))
        with self.assertRaises(ValueError):
            pl.compute_convergence(pipeline, stage="Telepathy")

    def test_raises_for_signals_stage(self):
        pipeline = pl.build_pipeline((_modality("EEG"),))
        with self.assertRaises(ValueError):
            pl.compute_convergence(pipeline, stage="Signals")

    def test_raises_for_sensors_stage(self):
        pipeline = pl.build_pipeline((_modality("EEG"),))
        with self.assertRaises(ValueError):
            pl.compute_convergence(pipeline, stage="Sensors")

    def test_raises_for_processing_when_pipeline_converges_at_inference(self):
        pipeline = pl.build_pipeline(
            (_modality("Structural"), _modality("Functional")),
            convergence_stage="Inference",
        )
        with self.assertRaises(ValueError):
            pl.compute_convergence(pipeline, stage="Processing")

    def test_succeeds_at_inference_when_pipeline_converges_at_inference(self):
        pipeline = pl.build_pipeline(
            (_modality("Structural"), _modality("Functional")),
            convergence_stage="Inference",
        )
        result = pl.compute_convergence(pipeline, stage="Inference")
        self.assertEqual(result.n_modalities, 2)

    def test_lists_every_modality_and_unique_category_at_inference(self):
        pipeline = pl.build_pipeline(
            (
                _modality("EEG", category="Neural"),
                _modality("ECG", category="Autonomic"),
                _modality("EDA", category="Autonomic"),
            )
        )
        result = pl.compute_convergence(pipeline, stage="Inference")
        self.assertEqual(result.stage, "Inference")
        self.assertEqual(result.modality_names, ("EEG", "ECG", "EDA"))
        self.assertEqual(result.categories, ("Neural", "Autonomic"))
        self.assertEqual(result.n_modalities, 3)


if __name__ == "__main__":
    unittest.main()
