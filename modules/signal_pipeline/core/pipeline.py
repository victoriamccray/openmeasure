"""
Signal-flow pipeline - the general DAG engine behind this journey.

Neurotech (EEG) is the first populated instance of this pipeline, not a
hard-coded shape: the stages below (Signals -> Sensors -> Processing ->
Inference -> Decision -> Action -> Feedback) are domain-agnostic, and a
multimodal pipeline is built by adding Modality nodes at the leading
stages rather than by rebuilding the pipeline shape itself. This is what
lets the worked examples add modalities (Autonomic, Muscular, ... for the
illustrative journey; Functional MRI, Diffusion MRI, ... for GRAND) one
at a time without any change to this module.

Where each modality's nodes stop being separate and start sharing the
rest of the chain is itself a parameter (convergence_stage), not a fixed
assumption: the illustrative journey's modalities converge immediately
after Sensors (the default), but GRAND's modalities are each acquired,
quality-checked, and preprocessed separately, converging only at
Inference - a different, equally valid instance of the same DAG shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .modality import Modality

STAGES: tuple[str, ...] = (
    "Signals",
    "Sensors",
    "Processing",
    "Inference",
    "Decision",
    "Action",
    "Feedback",
)

# Historically the only convergence point this module supported; kept as
# the default so every existing caller's behavior is unchanged.
DEFAULT_CONVERGENCE_STAGE = "Processing"


@dataclass(frozen=True)
class PipelineNode:
    """One node in the signal-flow DAG: a stage, optionally carrying one modality."""

    stage: str
    modality: Modality | None = None

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(
                f"'{self.stage}' is not a known stage. Valid stages: "
                f"{', '.join(STAGES)}."
            )


@dataclass(frozen=True)
class PipelineEdge:
    """One directed edge between two nodes of the same pipeline."""

    source: PipelineNode
    target: PipelineNode


@dataclass(frozen=True)
class SignalPipeline:
    """
    A full Signals -> ... -> Feedback pipeline for a set of modalities.

    convergence_stage records where this particular pipeline's modalities
    stop being separate nodes and start sharing the rest of the chain -
    every stage before it has one node per modality; every stage from it
    onward is a single shared node.
    """

    modalities: tuple[Modality, ...]
    nodes: tuple[PipelineNode, ...]
    edges: tuple[PipelineEdge, ...]
    convergence_stage: str = field(default=DEFAULT_CONVERGENCE_STAGE)


def build_pipeline(
    modalities: tuple[Modality, ...],
    convergence_stage: str = DEFAULT_CONVERGENCE_STAGE,
) -> SignalPipeline:
    """
    Build one Signals -> ... -> Feedback pipeline: one node per modality at
    every stage before convergence_stage, converging into a single shared
    chain from convergence_stage onward.

    Defaulting to "Processing" reproduces this module's original,
    single-shape behavior (every modality's Sensors node feeds directly
    into one shared Processing node). Passing a later stage - e.g.
    "Inference" - gives each modality its own Processing node too, for a
    worked example where each modality is preprocessed separately before
    any evidence is combined.
    """

    if not modalities:
        raise ValueError("modalities cannot be empty.")

    names = [modality.name for modality in modalities]
    if len(names) != len(set(names)):
        raise ValueError("modalities must have unique names.")

    if convergence_stage not in STAGES:
        raise ValueError(
            f"'{convergence_stage}' is not a known stage. Valid stages: "
            f"{', '.join(STAGES)}."
        )

    convergence_index = STAGES.index(convergence_stage)
    if convergence_index < 1:
        raise ValueError(
            f"convergence_stage must be '{STAGES[1]}' or later: every "
            "modality needs at least its own Signals node before any "
            "stage can be shared."
        )

    per_modality_stages = STAGES[:convergence_index]
    shared_stages = STAGES[convergence_index:]

    per_modality_nodes = {
        stage: tuple(
            PipelineNode(stage=stage, modality=modality) for modality in modalities
        )
        for stage in per_modality_stages
    }
    shared_nodes = tuple(PipelineNode(stage=stage) for stage in shared_stages)

    nodes = (
        tuple(node for stage in per_modality_stages for node in per_modality_nodes[stage])
        + shared_nodes
    )

    edges = []
    for index in range(len(modalities)):
        chain = [per_modality_nodes[stage][index] for stage in per_modality_stages]
        edges += [
            PipelineEdge(source=upstream, target=downstream)
            for upstream, downstream in zip(chain, chain[1:])
        ]
        edges.append(PipelineEdge(source=chain[-1], target=shared_nodes[0]))
    edges += [
        PipelineEdge(source=upstream, target=downstream)
        for upstream, downstream in zip(shared_nodes, shared_nodes[1:])
    ]

    return SignalPipeline(
        modalities=modalities,
        nodes=nodes,
        edges=tuple(edges),
        convergence_stage=convergence_stage,
    )


@dataclass(frozen=True)
class ConvergenceResult:
    """Which modalities and categories converge at one downstream stage."""

    stage: str
    modality_names: tuple[str, ...]
    categories: tuple[str, ...]
    n_modalities: int


def compute_convergence(
    pipeline: SignalPipeline, stage: str = "Inference"
) -> ConvergenceResult:
    """
    Describe convergence at one stage at or after this pipeline's own
    convergence_stage.

    Every modality passes through the same downstream nodes from
    convergence_stage onward - the DAG never branches again there - so
    convergence is simply "how many modalities, and which categories,
    feed this pipeline," not a per-node computation.
    """

    if stage not in STAGES:
        raise ValueError(
            f"'{stage}' is not a known stage. Valid stages: {', '.join(STAGES)}."
        )

    if STAGES.index(stage) < STAGES.index(pipeline.convergence_stage):
        raise ValueError(
            f"'{stage}' has one node per modality in this pipeline, not a "
            f"convergence point; pass a stage at or after "
            f"'{pipeline.convergence_stage}'."
        )

    modality_names = tuple(modality.name for modality in pipeline.modalities)

    categories: list[str] = []
    for modality in pipeline.modalities:
        if modality.category not in categories:
            categories.append(modality.category)

    return ConvergenceResult(
        stage=stage,
        modality_names=modality_names,
        categories=tuple(categories),
        n_modalities=len(modality_names),
    )
