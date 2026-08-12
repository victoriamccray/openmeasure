"""
HealthRing Acquisition Robustness Prototype (v0.1)

Standalone, throwaway exploration script. This is NOT an OpenMeasure module:
it is not wired into the Streamlit app, has no core/pages split, and has no
test suite. Its only job is to answer one question well enough to decide
whether a "Measurement Agreement / Acquisition Robustness" capability is
worth building into OpenMeasure for real.

Question: does ring heart-rate measurement error change across
acquisition/activity conditions?

The dataset is HealthRing's RingDatasetV2.1_submission.zip (Zenodo record
18426864). Keep that file local and gitignored -- do not commit, modify,
or redistribute it. Load it by path; nothing here assumes a fixed location.

Known upstream issue: the published archive is missing its central
directory and its final entry is truncated (verified by MD5 match against
Zenodo's published checksum -- this is not a bad download). Standard
zipfile.ZipFile cannot open it, so entries are recovered by walking local
file headers sequentially instead.

Security note: this script unpickles data with the standard `pickle`
module, which can execute arbitrary code for a malicious file. Only point
it at a HealthRing archive whose checksum you've verified.

Usage:
    python scripts/healthring_prototype.py \
        --zip /path/to/RingDatasetV2.1_submission.zip \
        --entry 00020_ring1_processed.pkl \
        --out private_data/healthring_prototype
"""

from __future__ import annotations

import argparse
import pickle
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NEEDED_COLUMNS = ["id", "Label", "Experiment", "hr", "bvp_hr", "ir-quality", "red-quality"]

CATEGORICAL_PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


class ArchiveTruncatedError(Exception):
    """Raised when the requested entry has no recoverable size in the local header."""


def find_and_extract_entry(zip_path: Path, entry_name: str) -> bytes:
    """
    Walk local file headers sequentially and decompress the named entry.

    General-purpose flag bit 3 is unset for every intact entry in this
    archive, so local headers carry real compressed/uncompressed sizes and
    entries can be located without the (missing) central directory.
    """
    with open(zip_path, "rb") as f:
        f.seek(0, 2)
        filesize = f.tell()
        pos = 0
        while pos < filesize - 4:
            f.seek(pos)
            sig = f.read(4)
            if sig != b"PK\x03\x04":
                break
            hdr = f.read(26)
            (_ver, flags, method, _mtime, _mdate, _crc32, csize, usize, nlen, elen) = (
                struct.unpack("<HHHHHIIIHH", hdr)
            )
            name = f.read(nlen).decode("utf-8", errors="replace")
            f.seek(elen, 1)
            data_offset = f.tell()
            if name == entry_name:
                if flags & 0x8 or csize == 0:
                    raise ArchiveTruncatedError(
                        f"entry {entry_name!r} has no recoverable size -- it is "
                        "truncated in the published archive"
                    )
                f.seek(data_offset)
                comp = f.read(csize)
                if method == 0:
                    return comp
                if method == 8:
                    return zlib.decompress(comp, -15)
                raise ValueError(f"unsupported compression method {method} for {entry_name!r}")
            pos = data_offset + csize
    raise KeyError(f"entry {entry_name!r} not found before the archive was truncated")


def load_subject_frame(zip_path: Path, entry_name: str) -> pd.DataFrame:
    raw = find_and_extract_entry(zip_path, entry_name)
    df = pickle.loads(raw)
    missing = [c for c in NEEDED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"expected columns missing from {entry_name}: {missing}")
    return df[NEEDED_COLUMNS].copy()


@dataclass(frozen=True)
class ErrorFrame:
    data: pd.DataFrame
    n_total_windows: int
    n_usable_windows: int
    condition_order: list[str]
    condition_colors: dict[str, str]


def prepare_error_frame(df: pd.DataFrame) -> ErrorFrame:
    n_total = len(df)
    usable = df.dropna(subset=["hr", "bvp_hr", "Label"]).copy()
    usable["abs_error"] = (usable["bvp_hr"] - usable["hr"]).abs()
    usable["signed_error"] = usable["bvp_hr"] - usable["hr"]
    usable["mean_hr"] = (usable["bvp_hr"] + usable["hr"]) / 2
    usable["quality"] = usable[["ir-quality", "red-quality"]].mean(axis=1)

    conditions = sorted(usable["Label"].unique())
    if len(conditions) > len(CATEGORICAL_PALETTE):
        raise ValueError(
            f"{len(conditions)} conditions exceed the {len(CATEGORICAL_PALETTE)}-slot "
            "categorical palette; extend the palette or facet before adding more."
        )
    colors = dict(zip(conditions, CATEGORICAL_PALETTE))

    return ErrorFrame(
        data=usable,
        n_total_windows=n_total,
        n_usable_windows=len(usable),
        condition_order=conditions,
        condition_colors=colors,
    )


def mae_by_condition(ef: ErrorFrame) -> pd.DataFrame:
    g = ef.data.groupby("Label")["abs_error"]
    table = pd.DataFrame({"n": g.size(), "mae": g.mean(), "sd": g.std()})
    return table.loc[ef.condition_order]


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.title.set_color(INK_PRIMARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def plot_mae_by_condition(ef: ErrorFrame, mae_table: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    conditions = ef.condition_order
    values = mae_table.loc[conditions, "mae"].to_numpy()
    colors = [ef.condition_colors[c] for c in conditions]
    bars = ax.bar(conditions, values, color=colors, width=0.62)
    for bar, n in zip(bars, mae_table.loc[conditions, "n"]):
        ax.annotate(
            f"n={n}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
            color=INK_MUTED,
        )
    ax.set_ylabel("MAE (bpm)")
    ax.set_title("Ring heart-rate MAE by acquisition condition")
    _style_axes(ax)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_error_distribution(ef: ErrorFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    conditions = ef.condition_order
    groups = [ef.data.loc[ef.data["Label"] == c, "abs_error"].to_numpy() for c in conditions]
    bp = ax.boxplot(
        groups,
        labels=conditions,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": INK_PRIMARY, "linewidth": 1.5},
        whiskerprops={"color": INK_MUTED},
        capprops={"color": INK_MUTED},
        flierprops={"markeredgecolor": INK_MUTED, "markersize": 4},
    )
    for patch, cond in zip(bp["boxes"], conditions):
        patch.set_facecolor(ef.condition_colors[cond])
        patch.set_alpha(0.75)
        patch.set_edgecolor(INK_SECONDARY)
    ax.set_ylabel("Absolute error |bvp_hr - hr| (bpm)")
    ax.set_title("Error distribution by acquisition condition")
    _style_axes(ax)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_quality_vs_error(ef: ErrorFrame, out_path: Path) -> None:
    conditions = ef.condition_order
    n = len(conditions)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), dpi=150, sharex=True, sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes).flatten()
    for ax, cond in zip(axes, conditions):
        sub = ef.data.loc[ef.data["Label"] == cond]
        ax.scatter(
            sub["quality"], sub["abs_error"], s=22, color=CATEGORICAL_PALETTE[0], alpha=0.75,
            edgecolors="none",
        )
        ax.set_title(cond, fontsize=9, color=INK_PRIMARY)
        _style_axes(ax)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.supxlabel("Mean signal quality (ir-quality, red-quality)", color=INK_SECONDARY)
    fig.supylabel("Absolute error (bpm)", color=INK_SECONDARY)
    fig.suptitle("Signal quality vs. heart-rate error, by condition", color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_bland_altman(ef: ErrorFrame, out_path: Path) -> tuple[float, float, float]:
    diffs = ef.data["signed_error"].to_numpy()
    means = ef.data["mean_hr"].to_numpy()
    bias = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    upper_loa = bias + 1.96 * sd
    lower_loa = bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.scatter(means, diffs, s=22, color=CATEGORICAL_PALETTE[0], alpha=0.65, edgecolors="none")
    ax.axhline(bias, color=INK_PRIMARY, linewidth=1.5, label=f"bias = {bias:.2f} bpm")
    ax.axhline(
        upper_loa, color=INK_SECONDARY, linewidth=1, linestyle="--",
        label=f"+1.96 SD = {upper_loa:.2f} bpm",
    )
    ax.axhline(
        lower_loa, color=INK_SECONDARY, linewidth=1, linestyle="--",
        label=f"-1.96 SD = {lower_loa:.2f} bpm",
    )
    ax.set_xlabel("Mean of ring and reference HR (bpm)")
    ax.set_ylabel("Ring - reference HR (bpm)")
    ax.set_title("Bland-Altman agreement: ring vs. reference heart rate")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return bias, sd, (upper_loa - lower_loa)


@dataclass(frozen=True)
class Section:
    question: str
    comparison: str
    visualization: str
    takeaway: str
    limitation: str


def render_report(sections: list[Section], entry_name: str, ef: ErrorFrame) -> str:
    lines = [
        "# HealthRing Acquisition Robustness Prototype (v0.1)",
        "",
        f"Source entry: `{entry_name}`",
        f"Usable windows: {ef.n_usable_windows} of {ef.n_total_windows} "
        f"(dropped {ef.n_total_windows - ef.n_usable_windows} missing hr/bvp_hr)",
        f"Conditions observed: {', '.join(ef.condition_order)}",
        "",
    ]
    for s in sections:
        lines += [
            f"## {s.question}",
            "",
            f"**Comparison:** {s.comparison}",
            "",
            f"**Visualization:** {s.visualization}",
            "",
            f"**Takeaway:** {s.takeaway}",
            "",
            f"**Limitation:** {s.limitation}",
            "",
        ]
    return "\n".join(lines)


def build_sections(ef: ErrorFrame, mae_table: pd.DataFrame, ba_stats: tuple[float, float, float]) -> list[Section]:
    worst = mae_table["mae"].idxmax()
    best = mae_table["mae"].idxmin()
    spread = mae_table["mae"].max() - mae_table["mae"].min()
    bias, sd, loa_width = ba_stats
    q_corr = ef.data[["quality", "abs_error"]].corr().iloc[0, 1]

    return [
        Section(
            question="Does mean absolute HR error differ across acquisition conditions?",
            comparison=(
                "MAE of |bvp_hr - hr| computed separately per `Label` "
                f"({', '.join(ef.condition_order)})."
            ),
            visualization="healthring_mae_by_condition.png",
            takeaway=(
                f"MAE ranges from {mae_table.loc[best, 'mae']:.2f} bpm ({best}) to "
                f"{mae_table.loc[worst, 'mae']:.2f} bpm ({worst}), a spread of "
                f"{spread:.2f} bpm across conditions for this one subject/ring. "
                "Heart-rate error differs across acquisition conditions, suggesting "
                "performance is context-sensitive."
            ),
            limitation=(
                "Single subject, single ring, non-randomized condition order -- this "
                "does not establish why the difference occurs (motion artifact vs. "
                "physiological change vs. order/fatigue effects are all confounded), "
                "and does not generalize past this one person's session."
            ),
        ),
        Section(
            question="Is the full error distribution (not just the mean) different by condition?",
            comparison="Per-window absolute error distribution, grouped by `Label`.",
            visualization="healthring_error_distribution.png",
            takeaway=(
                "The boxplots show whether wider spread, not just a higher mean, "
                "accompanies harder conditions -- check whether high-MAE conditions "
                "also show fatter tails/more outliers rather than a uniformly shifted "
                "distribution."
            ),
            limitation=(
                "Small per-condition sample sizes (tens of 30-second windows) make "
                "distributional shape read as noisy; this is descriptive only, no "
                "hypothesis test is applied."
            ),
        ),
        Section(
            question="Is error related to the ring's own signal-quality estimate?",
            comparison=(
                "Per-window absolute error vs. mean of `ir-quality`/`red-quality`, "
                "faceted by condition."
            ),
            visualization="healthring_quality_vs_error.png",
            takeaway=(
                f"Correlation between mean signal quality and absolute error across "
                f"all usable windows: r = {q_corr:.2f}. "
                + (
                    "A negative correlation would support using the ring's own "
                    "quality signal as a leading indicator of unreliable readings; "
                    "check whether that relationship holds within each condition "
                    "panel, not just pooled."
                )
            ),
            limitation=(
                "Quality is averaged across the two PPG channels for simplicity; the "
                "dataset doesn't document which channel(s) actually feed `bvp_hr`, so "
                "this averaging is an assumption, not a documented fact."
            ),
        ),
        Section(
            question="Do ring and reference HR agree overall, and is the disagreement systematic?",
            comparison="Bland-Altman: ring-minus-reference difference vs. their mean, pooled across conditions.",
            visualization="healthring_bland_altman.png",
            takeaway=(
                f"Bias = {bias:.2f} bpm, limits of agreement span {loa_width:.2f} bpm "
                f"(±1.96 SD = {sd * 1.96:.2f} bpm around the bias). "
                "A non-zero bias indicates systematic over/under-estimation rather "
                "than pure noise."
            ),
            limitation=(
                "Bland-Altman here pools all conditions together, so it describes "
                "average agreement, not condition-specific agreement -- pair it with "
                "the MAE-by-condition result above rather than reading it alone."
            ),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", required=True, type=Path, help="Path to RingDatasetV2.1_submission.zip")
    parser.add_argument(
        "--entry", default="00020_ring1_processed.pkl",
        help="Entry name inside the zip (default: %(default)s)",
    )
    parser.add_argument(
        "--out", default=Path("private_data/healthring_prototype"), type=Path,
        help="Output directory for plots/report (default: %(default)s). Must stay gitignored.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df = load_subject_frame(args.zip, args.entry)
    ef = prepare_error_frame(df)
    mae_table = mae_by_condition(ef)

    plot_mae_by_condition(ef, mae_table, args.out / "healthring_mae_by_condition.png")
    plot_error_distribution(ef, args.out / "healthring_error_distribution.png")
    plot_quality_vs_error(ef, args.out / "healthring_quality_vs_error.png")
    ba_stats = plot_bland_altman(ef, args.out / "healthring_bland_altman.png")

    mae_table.to_csv(args.out / "healthring_mae_by_condition.csv")

    sections = build_sections(ef, mae_table, ba_stats)
    report = render_report(sections, args.entry, ef)
    (args.out / "healthring_report.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"\nPlots, CSV, and report written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
