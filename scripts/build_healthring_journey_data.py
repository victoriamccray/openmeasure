"""
Build the small, journey-ready HealthRing artifact from the full archive.

Run once, locally, against your own copy of RingDatasetV2.1_submission.zip.
The output is committed; the 2.4 GiB archive is not.

    python scripts/build_healthring_journey_data.py --archive /path/to/RingDatasetV2.1_submission.zip

Why this exists
---------------
The worked example needed a 2.4 GiB local file before a visitor reached
any research interaction, and Streamlit caps browser uploads at 200 MB,
so a hosted copy of the app could not start the journey at all. The
journey's analysis reads six per-window summary columns. Those columns
are a few megabytes across every subject, so shipping them removes the
wall without shipping the archive.

What is dropped, and what that costs
-------------------------------------
HealthRing's per-window pickle carries roughly sixty columns, most of
them raw, standardized, filtered, difference and Welch variants of the
PPG and accelerometer waveforms. All of those are dropped. That is the
entire size reduction, and it has one real consequence: the Signal
Inspection stage draws the filtered waveforms and cannot run on this
artifact. That stage stays available to anyone pointing the page at the
full archive, and the provenance file records the exclusion so the page
can say so rather than failing quietly.

Redistribution
--------------
The Zenodo record for RingDatasetV2.1 is licensed CC BY 4.0, which
permits redistribution of the dataset and of derived works with
attribution. This script writes the attribution, the source DOI, the
licence, and a description of the transformation alongside the data, so
the artifact travels with the terms it is published under. It is a
derived subset and is labelled as one; it is not the HealthRing dataset.

Security
--------
HealthRing ships per-subject data as pickles, and unpickling executes
whatever the file says to execute. This script therefore refuses to
unpickle anything until the archive's MD5 matches the checksum Zenodo
publishes. --allow-unverified exists for a deliberately modified local
copy and should not be used on a file obtained from anywhere else.

Known upstream issue
--------------------
The published archive has no central directory and its final entry is
truncated, which is a property of the file as published rather than a
bad download (the MD5 matches). Entries are therefore located by walking
local file headers, the same way pages/HealthRing_Worked_Example.py
does.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import struct
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# What Zenodo publishes for RingDatasetV2.1_submission.zip. Checked
# against the record's own API response when this script was written.
SOURCE_DOI = "10.5281/zenodo.18426864"
SOURCE_URL = "https://doi.org/10.5281/zenodo.18426864"
SOURCE_TITLE = "HealthRing RingDatasetV2.1"
SOURCE_LICENSE = "CC BY 4.0"
PUBLISHED_MD5 = "6d96016e45a311b1cfc9108d3df23195"
PUBLISHED_SIZE_BYTES = 2582075992

# The columns the journey's analysis reads, plus the two that identify a
# window's origin. Everything else in the source pickle is dropped.
SUMMARY_COLUMNS: tuple[str, ...] = (
    "Label",
    "hr",
    "bvp_hr",
    "ir-quality",
    "red-quality",
)

# Present in some subjects' frames and not others. Kept where it exists,
# recorded in provenance either way, never fabricated.
OPTIONAL_COLUMNS: tuple[str, ...] = ("Experiment",)

# Dropped, and why it matters. Named individually rather than as "the
# waveforms" so the provenance file says what a reader is not getting.
EXCLUDED_COLUMN_NOTE = (
    "Raw, standardized, filtered, difference and Welch variants of the "
    "PPG and accelerometer channels, and the fs sampling rate. These are "
    "the whole size difference. The Signal Inspection stage of the worked "
    "example draws the filtered waveforms and cannot run on this artifact."
)

DEFAULT_OUTPUT = ROOT / "data" / "public" / "healthring_journey.parquet"
DEFAULT_PROVENANCE = ROOT / "data" / "public" / "healthring_journey_provenance.json"

# One ring design, matching the worked example's own v0.1 scope. Mixing
# both would confound "does this generalize across people" with "do the
# two rings measure the same thing".
DEFAULT_RING = "ring1"

_READ_CHUNK = 1024 * 1024


def archive_md5(archive: Path) -> str:
    """The archive's MD5, read in chunks so a 2.4 GiB file fits in memory."""
    hasher = hashlib.md5()

    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def index_entries(archive: Path) -> dict[str, tuple[int, int, int]]:
    """
    Map entry name to (data offset, compressed size, compression method).

    Walks local file headers because the published archive has no central
    directory. Stops at the first entry with no recoverable size, which
    is the one entry known to be truncated upstream.
    """
    index: dict[str, tuple[int, int, int]] = {}

    with archive.open("rb") as handle:
        handle.seek(0, 2)
        filesize = handle.tell()
        position = 0

        while position < filesize - 4:
            handle.seek(position)

            if handle.read(4) != b"PK\x03\x04":
                break

            (
                _version,
                flags,
                method,
                _mtime,
                _mdate,
                _crc32,
                compressed_size,
                _uncompressed_size,
                name_length,
                extra_length,
            ) = struct.unpack("<HHHHHIIIHH", handle.read(26))

            name = handle.read(name_length).decode("utf-8", errors="replace")
            handle.seek(extra_length, 1)
            data_offset = handle.tell()

            if flags & 0x8 or compressed_size == 0:
                break

            index[name] = (data_offset, compressed_size, method)
            position = data_offset + compressed_size

    return index


def read_entry(archive: Path, entry: tuple[int, int, int]) -> bytes:
    """Decompress one archive entry into memory."""
    data_offset, compressed_size, method = entry

    with archive.open("rb") as handle:
        handle.seek(data_offset)
        payload = handle.read(compressed_size)

    if method == 0:
        return payload

    return zlib.decompress(payload, -15)


def subject_frame(raw: pd.DataFrame, subject_id: int, ring: str) -> pd.DataFrame:
    """
    One subject's per-window summary rows.

    Columns are selected, never renamed or recomputed, apart from
    `quality`, which averages the two per-channel quality scores exactly
    as modules/healthring/core/acquisition_robustness.py does. Deriving
    it here keeps the artifact self-describing without the page having to
    know it was derived rather than published.
    """
    missing = [column for column in SUMMARY_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(
            f"Subject {subject_id} ({ring}) is missing {missing}. The "
            "archive layout differs from what this script was written "
            "against; check the source before editing this list."
        )

    keep = list(SUMMARY_COLUMNS) + [
        column for column in OPTIONAL_COLUMNS if column in raw.columns
    ]

    frame = raw[keep].copy()
    frame["quality"] = frame[["ir-quality", "red-quality"]].mean(axis=1)
    frame["subject_id"] = subject_id
    frame["ring"] = ring

    return frame


def build(archive: Path, ring: str) -> tuple[pd.DataFrame, dict]:
    """Read every intact subject entry for one ring design."""
    index = index_entries(archive)
    suffix = f"_{ring}_processed.pkl"

    names = sorted(name for name in index if name.endswith(suffix))
    if not names:
        raise ValueError(
            f"No entries ending in '{suffix}' were found. Known entry "
            f"count: {len(index)}."
        )

    frames: list[pd.DataFrame] = []
    subjects: list[int] = []
    optional_present: set[str] = set()

    for name in names:
        subject_id = int(Path(name).name.split("_")[0])
        raw = pickle.loads(read_entry(archive, index[name]))

        if not isinstance(raw, pd.DataFrame):
            raise TypeError(
                f"Entry {name} unpickled to {type(raw).__name__}, not a "
                "DataFrame."
            )

        optional_present.update(
            column for column in OPTIONAL_COLUMNS if column in raw.columns
        )
        frames.append(subject_frame(raw, subject_id, ring))
        subjects.append(subject_id)
        print(f"  {name}: {len(frames[-1]):,} windows", flush=True)

    data = pd.concat(frames, ignore_index=True)

    return data, {
        "n_entries_indexed": len(index),
        "n_subject_entries_read": len(names),
        "subject_ids": sorted(subjects),
        "optional_columns_found": sorted(optional_present),
        "optional_columns_absent": sorted(
            set(OPTIONAL_COLUMNS) - optional_present
        ),
    }


def provenance(
    data: pd.DataFrame, archive: Path, observed_md5: str, ring: str, detail: dict
) -> dict:
    """
    What this file is, where it came from, and what was done to it.

    Written beside the data rather than into it, so a reader can see the
    terms and the transformation without loading the parquet, and so the
    page can show them without restating them.
    """
    return {
        "artifact": "OpenMeasure-derived HealthRing journey subset",
        "is_the_original_dataset": False,
        "source": {
            "title": SOURCE_TITLE,
            "doi": SOURCE_DOI,
            "url": SOURCE_URL,
            "license": SOURCE_LICENSE,
            "published_md5": PUBLISHED_MD5,
            "published_size_bytes": PUBLISHED_SIZE_BYTES,
            "observed_md5": observed_md5,
            "checksum_matches_published": observed_md5 == PUBLISHED_MD5,
        },
        "attribution": (
            f"{SOURCE_TITLE}, {SOURCE_URL}, licensed {SOURCE_LICENSE}. "
            "This file is a derived subset produced by OpenMeasure and is "
            "not the original dataset."
        ),
        "transformation": {
            "ring_design": ring,
            "columns_kept": sorted(data.columns.tolist()),
            "columns_derived": {
                "quality": (
                    "mean of ir-quality and red-quality, the same rule "
                    "modules/healthring/core/acquisition_robustness.py "
                    "applies"
                )
            },
            "columns_excluded": EXCLUDED_COLUMN_NOTE,
            "rows_dropped": "none; every intact window for this ring is kept",
        },
        "contents": {
            "n_rows": int(len(data)),
            "n_subjects": int(data["subject_id"].nunique()),
            **detail,
        },
        "generated_by": "scripts/build_healthring_journey_data.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "RingDatasetV2.1_submission.zip",
        help="Path to your local RingDatasetV2.1_submission.zip",
    )
    parser.add_argument("--ring", default=DEFAULT_RING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help=(
            "Unpickle even if the archive's MD5 does not match Zenodo's "
            "published checksum. Only for a deliberately modified local "
            "copy."
        ),
    )
    args = parser.parse_args()

    if not args.archive.exists():
        print(f"Archive not found: {args.archive}", file=sys.stderr)
        return 1

    print(f"Hashing {args.archive} ({args.archive.stat().st_size:,} bytes)...")
    observed = archive_md5(args.archive)
    matches = observed == PUBLISHED_MD5

    print(f"  observed  {observed}")
    print(f"  published {PUBLISHED_MD5}")

    if not matches and not args.allow_unverified:
        print(
            "\nChecksum does not match. This script unpickles data from the "
            "archive, and unpickling an unverified file executes whatever "
            "it says to execute. Refusing. Re-download, or pass "
            "--allow-unverified if you modified this copy yourself.",
            file=sys.stderr,
        )
        return 2

    if not matches:
        print("\nProceeding on an unverified archive at your request.\n")

    data, detail = build(args.archive, args.ring)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(args.output, index=False)

    record = provenance(data, args.archive, observed, args.ring, detail)
    args.provenance.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    size = args.output.stat().st_size
    print(
        f"\nWrote {args.output} "
        f"({len(data):,} rows, {data['subject_id'].nunique()} subjects, "
        f"{size:,} bytes, "
        f"{size / PUBLISHED_SIZE_BYTES:.5%} of the archive)"
    )
    print(f"Wrote {args.provenance}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
