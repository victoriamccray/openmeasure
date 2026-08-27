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
        remaining = data.drop(columns=[column])
        if remaining.shape[1] < 2:
            out[column] = float("nan")
        else:
            out[column] = hand_cronbach_alpha(remaining)
    return out


def hand_split_half(data: pd.DataFrame) -> tuple[float, float]:
    odd = data.iloc[:, ::2].sum(axis=1)
    even = data.iloc[:, 1::2].sum(axis=1)
    r = float(odd.corr(even))
    sb = (2 * r) / (1 + r)
    return r, sb


# Dataset A: 3 items x 4 participants. Small integers, every statistic
# reduces to hand-checkable fractions.
DATASET_A = pd.DataFrame(
    {
        "item1": [2, 4, 6, 4],
        "item2": [3, 5, 5, 3],
        "item3": [1, 3, 7, 5],
    },
    dtype=float,
)

# Dataset B: 4 items x 8 participants. Also the fixed dataset already
# pinned in test_reliability.py::TestCronbachAlpha::test_known_value_regression.
DATASET_B = pd.DataFrame(
    {
        "q1": [4, 3, 5, 2, 4, 5, 3, 4],
        "q2": [5, 2, 5, 2, 4, 5, 3, 4],
        "q3": [4, 3, 4, 1, 5, 4, 2, 5],
        "q4": [5, 3, 5, 2, 4, 5, 3, 4],
    },
    dtype=float,
)

# Dataset C: high reliability. 5 items that move together strongly.
DATASET_C = pd.DataFrame(
    {
        "item1": [2, 3, 4, 5, 6, 7, 8, 9],
        "item2": [2, 3, 5, 5, 7, 7, 9, 9],
        "item3": [3, 4, 4, 6, 6, 8, 8, 10],
        "item4": [2, 4, 4, 5, 7, 7, 8, 10],
        "item5": [3, 3, 5, 6, 6, 8, 9, 9],
    },
    dtype=float,
)

# Dataset D: low/negative reliability. 4 items, each independently
# shuffled, so items do not move together (and several move oppositely).
DATASET_D = pd.DataFrame(
    {
        "item1": [5, 3, 6, 2, 7, 1, 4, 8],
        "item2": [1, 7, 2, 8, 3, 6, 5, 4],
        "item3": [8, 2, 5, 3, 6, 1, 7, 4],
        "item4": [4, 8, 1, 7, 2, 5, 3, 6],
    },
    dtype=float,
)

# Dataset E: problematic/reverse item. item1-4 move together; item5 is
# the reverse pattern of item1, not recoded before analysis.
DATASET_E = pd.DataFrame(
    {
        "item1": [2, 3, 4, 5, 6, 7, 8, 9],
        "item2": [2, 4, 4, 6, 6, 8, 8, 10],
        "item3": [3, 3, 5, 5, 7, 7, 9, 9],
        "item4": [2, 4, 5, 5, 7, 8, 8, 10],
        "item5": [9, 8, 7, 6, 5, 4, 3, 2],
    },
    dtype=float,
)

# Dataset F: missing data. 4 items x 10 participants, with one item
# missing for participant 8 and a different item missing for participant
# 9 (2 of 10 rows affected -> OpenMeasure's analyze() listwise-deletes
# both, leaving 8 complete cases).
DATASET_F = pd.DataFrame(
    {
        "q1": [4, 3, 5, 2, 4, 5, 3, 4, 6, 2],
        "q2": [5, 2, 5, 2, 4, 5, 3, 4, 6, 3],
        "q3": [4, 3, 4, 1, 5, 4, 2, np.nan, 6, 2],
        "q4": [5, 3, 5, 2, 4, 5, 3, 4, np.nan, 3],
    },
    dtype=float,
)

# Dataset G: small/edge case. The minimum allowed item count (2) with a
# small sample (5 participants).
DATASET_G = pd.DataFrame(
    {
        "item1": [1, 2, 3, 4, 5],
        "item2": [2, 3, 3, 5, 4],
    },
    dtype=float,
)


def report(
    name: str,
    data: pd.DataFrame,
    *,
    include_split_half: bool,
) -> None:
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


def report_missing_data(name: str, data: pd.DataFrame) -> None:
    print(f"\n=== {name} ===")
    print(data.to_string(index=False))

    # OpenMeasure's documented missing-data handling is analyze()'s
    # listwise deletion (see core/reliability.py's analyze() docstring).
    # Calling cronbach_alpha() directly on data containing NaNs does NOT
    # perform listwise deletion (pandas' default skipna behavior instead
    # silently sums fewer items for affected rows) - shown here as a
    # documented pitfall, not used as the comparison point.
    raw_alpha_no_listwise = rel.cronbach_alpha(data)
    result = rel.analyze(data)
    complete = data.dropna()
    hand_alpha = hand_cronbach_alpha(complete)

    print(
        f"\ncronbach_alpha() called directly on data WITH NaNs "
        f"(bypasses listwise deletion): {raw_alpha_no_listwise!r} "
        f"<- documented pitfall, not the comparison point"
    )
    print(
        f"analyze() alpha (listwise deletion, {result.n_excluded_cases} "
        f"case(s) excluded): {result.cronbach_alpha!r}"
    )
    print(f"hand alpha on the same listwise-deleted rows: {hand_alpha!r}")


if __name__ == "__main__":
    report("Dataset A (3 items x 4 participants)", DATASET_A, include_split_half=False)
    report("Dataset B (4 items x 8 participants)", DATASET_B, include_split_half=True)
    report("Dataset C - high reliability (5 items x 8 participants)", DATASET_C, include_split_half=True)
    report("Dataset D - low/negative reliability (4 items x 8 participants)", DATASET_D, include_split_half=True)
    report("Dataset E - problematic reverse item (5 items x 8 participants)", DATASET_E, include_split_half=True)
    report_missing_data("Dataset F - missing data (4 items x 10 participants)", DATASET_F)
    report("Dataset G - small edge case (2 items x 5 participants)", DATASET_G, include_split_half=False)

    print("\n--- CSVs for R (reliability_reference.R reads these) ---")
    datasets = {
        "dataset_a": DATASET_A,
        "dataset_b": DATASET_B,
        "dataset_c": DATASET_C,
        "dataset_d": DATASET_D,
        "dataset_e": DATASET_E,
        "dataset_f": DATASET_F,
        "dataset_g": DATASET_G,
    }
    for stem, df in datasets.items():
        path = Path(__file__).parent / f"{stem}.csv"
        df.to_csv(path, index=False)
        print(f"Wrote {path}")
