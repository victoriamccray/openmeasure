"""
Which catalogued datasets OpenMeasure can currently read, and how.

shared/datasets.py records what a dataset is, where it comes from and
what it may be used for. It deliberately says nothing about whether this
program can open it: Right To Play is an SPSS .sav and OpenMesh is
NetCDF, and both are catalogued honestly as things OpenMeasure cannot
read today. This module is the other half of that separation. A dataset
appears here once someone has written a loader for it, and until then
`can_load` returns False and a page can say so plainly rather than
offering a button that fails.

Loaders return the file **as published**. No renaming, no recoding, no
dropping of sentinel values. NHANES codes 7 as Refused and 9 as Don't
know on items scored 0 to 3, and a loader that quietly turned those into
missing would have made the reader's first real decision for them,
invisibly. Deciding what to do with them belongs to the page, in front of
the reader, which is the entire reason for featuring real data.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

from shared.datasets import (
    DELIVERY_BUNDLED_PUBLIC,
    DELIVERY_CACHED_PUBLIC,
    DELIVERY_REMOTE_FETCH,
    RealDataset,
    get_dataset,
)

# Long enough for a large file over a slow link, short enough that a dead
# endpoint surfaces as an error rather than a hang.
FETCH_TIMEOUT_SECONDS = 120

USER_AGENT = "OpenMeasure/0.1 (+https://github.com/victoriamccray/openmeasure)"

# What a caller should catch around load_dataset, so a page does not have
# to know these are built on urllib.
LOAD_ERRORS = (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError)


# Where this repository is checked out, so a committed artifact resolves
# from a repo-relative path in the catalog rather than from the working
# directory a page happens to be run from.
ROOT = Path(__file__).resolve().parents[1]

# What a reader is actually holding. Named rather than implied, because
# the difference matters and is invisible in a DataFrame: a derived
# subset is not the dataset it came from, and a page showing one must be
# able to say so without knowing how it was produced.
ARTIFACT_ORIGINAL = "The dataset as published"
ARTIFACT_DERIVED = "A subset derived by OpenMeasure"


@dataclass(frozen=True)
class LoadedArtifact:
    """Data, and what it is."""

    dataset_id: str
    frame: "pd.DataFrame"
    kind: str
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def is_derived(self) -> bool:
        """Whether this is a subset rather than the published dataset."""
        return self.kind == ARTIFACT_DERIVED


class ArtifactChecksumError(ValueError):
    """A committed artifact does not match the checksum the catalog declares."""


def _verify(path: Path, expected_sha256: str) -> None:
    """
    Check a committed artifact against the catalog's declared digest.

    Against the catalog rather than against a digest stored beside the
    file, because a file hashed to whatever it happens to contain
    verifies nothing. A mismatch means the artifact and the entry
    describing it have come apart, and the honest response is to stop.
    """
    observed = hashlib.sha256(path.read_bytes()).hexdigest()

    if observed != expected_sha256:
        raise ArtifactChecksumError(
            f"{path.name} hashes to {observed}, and the catalog declares "
            f"{expected_sha256}. The file and the entry describing it have "
            "come apart. Rebuild the artifact, or update the entry, rather "
            "than loading a file nothing vouches for."
        )


def _load_derived(dataset: RealDataset) -> LoadedArtifact:
    """Read a committed subset, with the record of what was done to it."""
    artifact = dataset.derived_artifact
    path = ROOT / artifact.path

    if not path.exists():
        raise FileNotFoundError(
            f"{dataset.name} declares a derived artifact at "
            f"{artifact.path}, which is not present. It is built by a "
            "script in scripts/ and committed; it is not fetched."
        )

    _verify(path, artifact.sha256)

    provenance_path = ROOT / artifact.provenance_path
    provenance: dict[str, Any] = {}
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    return LoadedArtifact(
        dataset_id=dataset.id,
        frame=pd.read_parquet(path),
        kind=ARTIFACT_DERIVED,
        provenance=provenance,
    )


def load_public_artifact(dataset: RealDataset) -> LoadedArtifact:
    """
    Open whatever a page can open for this dataset, and say what it is.

    One path for every catalogued dataset: the entry names how it is
    delivered, the artifact is read or fetched accordingly, a committed
    artifact is verified against the digest the entry declares, and the
    result carries the provenance alongside the data rather than leaving
    a caller to find it.

    Precedence is fixed and does not fall back. A committed subset wins
    where one exists, because its existence is a deliberate decision that
    this is the immediate path for a reader. Otherwise a fetch runs, if
    the dataset is delivered that way and a loader has been written.
    Where neither applies this raises and names what a reader has to do
    instead, rather than quietly returning something from somewhere else:
    a page silently substituting one source for another is the failure
    this catalog exists to prevent.
    """
    if dataset.derived_artifact is not None:
        return _load_derived(dataset)

    fetchable = (
        DELIVERY_REMOTE_FETCH,
        DELIVERY_CACHED_PUBLIC,
        DELIVERY_BUNDLED_PUBLIC,
    )

    if dataset.delivery in fetchable and dataset.id in LOADERS:
        return LoadedArtifact(
            dataset_id=dataset.id,
            frame=load_dataset(dataset.id),
            kind=ARTIFACT_ORIGINAL,
        )

    raise ValueError(
        f"OpenMeasure cannot open '{dataset.name}' on its own. It is "
        f"delivered as '{dataset.delivery}' and has no committed subset, "
        "so obtain it from the source listed in the catalog and supply it "
        "yourself."
    )


def _fetch_bytes(url: str) -> bytes:
    """Retrieve a file into memory, without writing it to disk."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
        return response.read()


def _load_nhanes_dpq(dataset) -> pd.DataFrame:
    """
    The NHANES depression screener, exactly as NCHS publishes it.

    A SAS transport file, read with pandas' own xport support, so this
    needs no dependency the project does not already have.

    Three things are deliberately left alone. Codes 7 and 9 stay as 7 and
    9, because Refused and Don't know are the reader's decision. DPQ100
    stays in the frame, because whether it belongs in the scale is also
    theirs, and removing it here would hide the question. And the zero
    category is left as pandas decodes it, a denormalised float rather
    than an exact 0, since rounding it would be a silent edit to the
    values a reader is about to inspect.
    """
    url = next(
        source.url for source in dataset.sources if source.url.endswith(".xpt")
    )

    return pd.read_sas(io.BytesIO(_fetch_bytes(url)), format="xport")


# Dataset id -> loader. Membership is the whole point: a catalogued
# dataset absent from this mapping is one OpenMeasure knows about and
# cannot yet open.
LOADERS: dict[str, Callable[[object], pd.DataFrame]] = {
    "nhanes_dpq_phq9": _load_nhanes_dpq,
}


def can_load(dataset_id: str) -> bool:
    """
    Whether a loader exists for this dataset.

    Raises on an unknown id rather than returning False, so a typo reads
    as a mistake rather than as a dataset OpenMeasure cannot open.
    """
    get_dataset(dataset_id)

    return dataset_id in LOADERS


def loadable_dataset_ids() -> tuple[str, ...]:
    """Every catalogued dataset a loader exists for, in catalog order."""
    from shared.datasets import DATASETS

    return tuple(d.id for d in DATASETS if d.id in LOADERS)


@st.cache_data(ttl=3600, show_spinner="Fetching the dataset...")
def load_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Fetch and read one catalogued dataset, as published.

    Cached for an hour, so moving through a page does not re-download the
    same file on every rerun. Returns a DataFrame rather than anything
    OpenMeasure-specific, because the caller's next step is to show a
    reader what actually arrived.
    """
    dataset = get_dataset(dataset_id)

    if dataset_id not in LOADERS:
        raise ValueError(
            f"'{dataset.name}' is catalogued but OpenMeasure has no loader "
            f"for it. Known loaders: {', '.join(sorted(LOADERS))}."
        )

    return LOADERS[dataset_id](dataset)
