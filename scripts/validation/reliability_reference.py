"""
One-time generator for the Reliability reference-validation numbers used in
docs/validation/reference-validation.md.

This script is a dev-only tool, not part of the OpenMeasure application or
its test suite. It computes three independent legs for each statistic:

  1. "Hand" leg: the textbook formula applied directly with numpy, with no
     call into OpenMeasure's core/reliability.py and no external package.
  2. OpenMeasure leg: core/reliability.py, the code actually shipped.
  3. Reference-software leg: printed separately by
     reliability_reference.R (R's psych::alpha()), run independently.

Run from the repo root:
    python scripts/validation/reliability_reference.py

Requires numpy and pandas only (both are already OpenMeasure runtime
dependencies). It does NOT require pingouin or R - those are exercised
separately and are not installed as part of this repo's dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modules" / "reliability"))

from core import reliability as rel  # noqa: E402


def hand_cronbach_alpha(data: pd.DataFrame) -> float:
    """Direct textbook formula, independent of core/reliability.py."""
    item_variances = data.var(axis=0, ddof=1)
    total_variance = data.sum(axis=1).var(ddof=1)
    k = data.shape[1]
    return float(k / (k - 1) * (1 - item_variances.sum() / total_variance))


def hand_item_total_correlations(data: pd.DataFrame) -> dict[str, float]:
    out = {}
    for column in data.columns:
        rest = data.drop(columns=[column]).sum(axis=1)
        out[column] = float(data[column].corr(rest))
    return out


def hand_alpha_if_dropped(data: pd.DataFrame) -> dict[str, float]:
    out = {}
    for column in data.columns:
        out[column] = hand_cronbach_alpha(data.drop(columns=[column]))
    return out


def hand_split_half(data: pd.DataFrame) -> tuple[float, float]:
    odd = data.iloc[:, ::2].sum(axis=1)
    even = data.iloc[:, 1::2].sum(axis=1)
    r = float(odd.corr(even))
    sb = (2 * r) / (1 + r)
    return r, sb


DATASET_A = pd.DataFrame(
    {
        "item1": [2, 4, 6, 4],
        "item2": [3, 5, 5, 3],
        "item3": [1, 3, 7, 5],
    },
    dtype=float,
)

DATASET_B = pd.DataFrame(
    {
        "q1": [4, 3, 5, 2, 4, 5, 3, 4],
        "q2": [5, 2, 5, 2, 4, 5, 3, 4],
        "q3": [4, 3, 4, 1, 5, 4, 2, 5],
        "q4": [5, 3, 5, 2, 4, 5, 3, 4],
    },
    dtype=float,
)


def report(name: str, data: pd.DataFrame, *, include_split_half: bool) -> None:
    print(f"\n=== {name} ===")
    print(data.to_string(index=False))

    hand_alpha = hand_cronbach_alpha(data)
    om_alpha = rel.cronbach_alpha(data)
    print(f"\nCronbach's alpha: hand={hand_alpha!r} openmeasure={om_alpha!r}")

    hand_corr = hand_item_total_correlations(data)
    om_corr = rel.item_total_correlations(data).to_dict()
    print("Item-total correlations:")
    for col in data.columns:
        print(f"  {col}: hand={hand_corr[col]!r} openmeasure={om_corr[col]!r}")

    hand_drop = hand_alpha_if_dropped(data)
    om_drop = rel.alpha_if_item_dropped(data).to_dict()
    print("Alpha if item dropped:")
    for col in data.columns:
        print(f"  {col}: hand={hand_drop[col]!r} openmeasure={om_drop[col]!r}")

    if include_split_half:
        hand_r, hand_sb = hand_split_half(data)
        om_r, om_sb = rel.split_half_reliability(data)
        print(f"\nSplit-half correlation: hand={hand_r!r} openmeasure={om_r!r}")
        print(f"Spearman-Brown: hand={hand_sb!r} openmeasure={om_sb!r}")


if __name__ == "__main__":
    report("Dataset A (3 items x 4 participants)", DATASET_A, include_split_half=False)
    report("Dataset B (4 items x 8 participants)", DATASET_B, include_split_half=True)

    print("\n--- CSVs for R (reliability_reference.R reads these) ---")
    a_path = Path(__file__).parent / "dataset_a.csv"
    b_path = Path(__file__).parent / "dataset_b.csv"
    DATASET_A.to_csv(a_path, index=False)
    DATASET_B.to_csv(b_path, index=False)
    print(f"Wrote {a_path}")
    print(f"Wrote {b_path}")
