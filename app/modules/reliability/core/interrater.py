"""
Multi-rater categorical agreement.

Cronbach's alpha and split-half reliability (reliability.py) ask whether
continuous scale items measure one underlying construct consistently.
These functions ask a different question: do independent raters apply
the same categorical judgment consistently to the same items (for
example, an Include / Uncertain / Exclude quality-control decision)?

Cohen's kappa (two raters) and Fleiss' kappa (three or more raters, every
item rated by the same fixed set of raters) are computed via
statsmodels.stats.inter_rater, an independently maintained implementation,
rather than reimplemented here. Krippendorff's alpha, which tolerates a
variable number of raters per item and missing ratings (unlike Fleiss'
kappa, whose chance-agreement term assumes a fixed rater panel), is
computed via the krippendorff package for the same reason: this module
wraps established implementations in frozen, documented result objects
rather than reimplementing agreement statistics from scratch.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import krippendorff as _krippendorff
import numpy as np
import pandas as pd
from statsmodels.stats import inter_rater as _sm_inter_rater

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.validation import validate_is_dataframe


@dataclass(frozen=True)
class CohensKappaResult:
    """Pairwise agreement between exactly two raters."""

    rater_a: str
    rater_b: str
    n_items: int
    kappa: float


def cohens_kappa(data: pd.DataFrame, rater_a: str, rater_b: str) -> CohensKappaResult:
    """
    Pairwise agreement between two raters' categorical judgments.

    data holds one row per rated item and one column per rater. Rows
    missing either named rater's judgment are excluded first: only items
    both raters actually rated can contribute to a pairwise comparison.
    """
    validate_is_dataframe(data)

    if rater_a not in data.columns:
        raise ValueError(f"Rater column '{rater_a}' was not found in the data.")

    if rater_b not in data.columns:
        raise ValueError(f"Rater column '{rater_b}' was not found in the data.")

    if rater_a == rater_b:
        raise ValueError("rater_a and rater_b must be different columns.")

    paired = data[[rater_a, rater_b]].dropna()

    if len(paired) < 2:
        raise ValueError(
            f"At least 2 items rated by both '{rater_a}' and '{rater_b}' "
            f"are required; found {len(paired)}."
        )

    categories = sorted(set(paired[rater_a]) | set(paired[rater_b]), key=str)

    table = pd.crosstab(
        pd.Categorical(paired[rater_a], categories=categories),
        pd.Categorical(paired[rater_b], categories=categories),
        dropna=False,
    ).to_numpy()

    kappa = float(_sm_inter_rater.cohens_kappa(table, return_results=False))

    if np.isnan(kappa):
        raise ValueError(
            "Cohen's kappa is undefined for this pair (a rater used only "
            "one category across every shared item, leaving no variance "
            "to measure agreement against)."
        )

    return CohensKappaResult(
        rater_a=rater_a,
        rater_b=rater_b,
        n_items=int(len(paired)),
        kappa=kappa,
    )


@dataclass(frozen=True)
class FleissKappaResult:
    """Agreement among three or more raters who all rated every item."""

    n_items: int
    n_raters: int
    categories: tuple[str, ...]
    kappa: float


def fleiss_kappa(data: pd.DataFrame) -> FleissKappaResult:
    """
    Agreement among raters, every one of whom rated every item.

    data holds one row per rated item, one column per rater, with no
    missing values. Fleiss' kappa's chance-agreement term assumes a fixed
    number of raters per item, so a partially-rated item cannot be
    included without violating that assumption; use krippendorff_alpha
    when raters did not all rate every item.
    """
    validate_is_dataframe(data)

    if data.shape[1] < 3:
        raise ValueError(
            f"Fleiss' kappa requires at least 3 raters (columns); found "
            f"{data.shape[1]}."
        )

    if data.shape[0] < 2:
        raise ValueError("At least 2 rated items are required.")

    if data.isna().any().any():
        raise ValueError(
            "Fleiss' kappa requires every item to be rated by every rater "
            "in this data, with no missing values. Restrict to fully-"
            "rated items first, or use krippendorff_alpha for partial "
            "rater coverage."
        )

    categories = sorted({str(value) for value in data.to_numpy().ravel()}, key=str)
    category_index = {category: i for i, category in enumerate(categories)}

    counts = np.zeros((data.shape[0], len(categories)), dtype=int)

    for row_index, row in enumerate(data.itertuples(index=False)):
        for value in row:
            counts[row_index, category_index[str(value)]] += 1

    kappa = float(_sm_inter_rater.fleiss_kappa(counts, method="fleiss"))

    return FleissKappaResult(
        n_items=int(data.shape[0]),
        n_raters=int(data.shape[1]),
        categories=tuple(categories),
        kappa=kappa,
    )


@dataclass(frozen=True)
class KrippendorffAlphaResult:
    """Agreement among any number of raters, tolerating partial coverage."""

    n_items: int
    n_raters: int
    categories: tuple[str, ...]
    alpha: float


def krippendorff_alpha(data: pd.DataFrame) -> KrippendorffAlphaResult:
    """
    Agreement among raters, tolerating items that not every rater rated.

    data holds one row per rated item, one column per rater; a missing
    value (None/NaN) means that rater did not rate that item. Unlike
    Fleiss' kappa, this does not require a fixed rater panel per item:
    an item with only 2 of 4 raters' judgments still contributes, which
    is why it is the better-suited default whenever rater coverage is
    uneven, per Krippendorff's own stated rationale for the statistic.
    """
    validate_is_dataframe(data)

    if data.shape[1] < 2:
        raise ValueError(
            f"Krippendorff's alpha requires at least 2 raters (columns); "
            f"found {data.shape[1]}."
        )

    n_pairable = int((data.notna().sum(axis=1) >= 2).sum())

    if n_pairable < 2:
        raise ValueError(
            "At least 2 items with 2 or more ratings each are required."
        )

    raw = data.to_numpy(dtype=object)
    missing = pd.isna(raw)

    categories = sorted({str(value) for value in raw[~missing]}, key=str)

    # krippendorff.alpha expects (raters x units); this DataFrame is
    # (units x raters), so transpose first. The result must be a plain
    # nested list, not a pre-cast dtype=object ndarray: the package's own
    # np.asarray() call is what coerces a mix of strings and np.nan into
    # a string array where missing values become the literal string
    # "nan" (its own value-domain detection only recognizes numeric or
    # string array dtypes, not "object"). Building the object array
    # ourselves and handing it over skips that coercion, and every value
    # domain lookup then silently fails to match.
    reliability_data = [
        [str(value) if pd.notna(value) else np.nan for value in rater_row]
        for rater_row in raw.T
    ]

    alpha = float(
        _krippendorff.alpha(
            reliability_data=reliability_data,
            value_domain=categories,
            level_of_measurement="nominal",
        )
    )

    return KrippendorffAlphaResult(
        n_items=int(data.shape[0]),
        n_raters=int(data.shape[1]),
        categories=tuple(categories),
        alpha=alpha,
    )
