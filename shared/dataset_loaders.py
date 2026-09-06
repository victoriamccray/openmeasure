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

import io
import urllib.request
from typing import Callable

import pandas as pd
import streamlit as st

from shared.datasets import get_dataset

# Long enough for a large file over a slow link, short enough that a dead
# endpoint surfaces as an error rather than a hang.
FETCH_TIMEOUT_SECONDS = 120

USER_AGENT = "OpenMeasure/0.1 (+https://github.com/victoriamccray/openmeasure)"

# What a caller should catch around load_dataset, so a page does not have
# to know these are built on urllib.
LOAD_ERRORS = (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError)


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
