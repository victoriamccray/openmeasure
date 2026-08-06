"""
Reporting what each analysis used, across modules.

Every module records how much data it received and how much it kept. This
module gathers those records and describes them together, so a user can see
that one analysis kept 180 of 200 rows while another kept 165.

It computes no combined figure. Rows dropped by listwise deletion, empty
cells inside rows that were kept, and observations that never arrived have
no common denominator, so summing them is a category error and an overall
retention percentage would be a composite score in disguise. Each account is
reported on its own terms.

Nothing here judges whether an exclusion rate is acceptable. There is no
threshold, because the share of data missing does not tell you why it is
missing, and the reason usually matters more than the rate.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.handoff import DatasetFingerprint, ExclusionAccount


@dataclass(frozen=True)
class DatasetRetention:
    """Every analysis recorded against one uploaded dataset."""

    fingerprint: DatasetFingerprint
    accounts: tuple[ExclusionAccount, ...]

    # The analysis that kept the fewest rows, as a raw count rather than a
    # grade. Row-expanding analyses are excluded from this comparison
    # because their retained counts are not comparable.
    smallest_retained_n: int | None
    smallest_retained_analysis: str | None

    # Analyses whose retention figure cannot be compared to the others.
    incomparable_analyses: tuple[str, ...]


@dataclass(frozen=True)
class RetentionSummary:
    """
    Retention across every recorded analysis, grouped by dataset.

    Deliberately has no field describing overall retention. Analyses of
    different uploads describe different data, and the kinds of exclusion
    within one analysis are not additive.
    """

    datasets: tuple[DatasetRetention, ...]
    n_analyses: int
    n_datasets: int
    shared_implication: str
    limitations: tuple[str, ...]


SHARED_IMPLICATION = (
    "Each result above describes only the observations that analysis "
    "retained, not everything that was uploaded. Where analyses kept "
    "different amounts, they are describing different subsets of the data."
)

LIMITATIONS = (
    "The share of data excluded does not establish why it is missing, and "
    "the reason usually matters more than the rate. Observations are often "
    "missing precisely because circumstances were unusual, which no "
    "retention figure can detect.",
    "Analyses of the same file do not necessarily use the same rows. Each "
    "analysis drops rows based on the columns it needs, so two analyses can "
    "retain the same number of different observations. The overlap between "
    "retained subsets is not reported.",
    "Counts of dropped rows, empty cells, and absent observations are "
    "reported separately because they are different things. They are not "
    "added together, and no overall retention figure is produced.",
    "Values that stand in for missing data, such as -999 or NA, are counted "
    "as present throughout, because deciding that a particular value means "
    "missing is a judgment this toolkit does not make.",
)


def summarize_retention(
    accounts_by_dataset: dict[DatasetFingerprint, tuple[ExclusionAccount, ...]],
) -> RetentionSummary:
    """
    Describe retention across every recorded analysis.

    Parameters
    ----------
    accounts_by_dataset:
        Exclusion accounts grouped by the dataset they were computed from.

    Returns
    -------
    RetentionSummary

    Raises
    ------
    ValueError
        If no accounts are supplied, or a dataset has no accounts.
    """

    if not accounts_by_dataset:
        raise ValueError(
            "At least one recorded analysis is required to summarize "
            "retention."
        )

    datasets: list[DatasetRetention] = []
    n_analyses = 0

    for fingerprint, accounts in accounts_by_dataset.items():
        if not accounts:
            raise ValueError(
                f"Dataset '{fingerprint.label}' has no recorded analyses."
            )

        n_analyses += len(accounts)

        # Only analyses with a retained-participant count can be compared.
        # A row-expanding analysis reports expanded observations instead,
        # which is a different quantity and is not ranked against these.
        comparable = [
            account
            for account in accounts
            if not account.rows_can_exceed_participants
            and account.n_retained_rows is not None
        ]

        smallest = (
            min(comparable, key=lambda account: account.n_retained_rows)
            if comparable
            else None
        )

        datasets.append(
            DatasetRetention(
                fingerprint=fingerprint,
                accounts=tuple(accounts),
                smallest_retained_n=(
                    smallest.n_retained_rows if smallest is not None else None
                ),
                smallest_retained_analysis=(
                    smallest.analysis_label if smallest is not None else None
                ),
                incomparable_analyses=tuple(
                    account.analysis_label
                    for account in accounts
                    if account.rows_can_exceed_participants
                ),
            )
        )

    return RetentionSummary(
        datasets=tuple(datasets),
        n_analyses=n_analyses,
        n_datasets=len(datasets),
        shared_implication=SHARED_IMPLICATION,
        limitations=LIMITATIONS,
    )
