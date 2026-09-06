"""
The recurring objects OpenMeasure draws, as opposed to the marks it draws
them with.

shared/visuals.py holds the grammar's primitives: a palette, a unit
pictograph, an arrow, a sized wrapper. This module holds the things built
out of them that more than one page needs, and it exists because the
dataset portrait was the first of those. It was written page-locally for
Impact Evaluation's analysis stage, and then Fairness, Time-Series QA,
Reliability and Method Selection all wanted the same object in front of
the same decision: which column is which, before anyone is asked to say.

A builder earns a place here by being needed twice, not by looking
general. A support boundary or a difference-in-differences chart still
belongs to the page that owns it.

No streamlit import. These return SVG strings, so they are testable
without a running app; shared/upload.py has the thin renderer that puts
one on a page next to the caveat that belongs with it.
"""

from __future__ import annotations

from modules.data_profile.core.profile import ROLES, DataProfile
from shared import visuals
from shared.datasets import RealDataset


# The dataset portrait. A reader about to map columns onto roles is
# deciding which column is the outcome, which is the group, which two are
# the baseline and follow-up, or which is the model's decision. What
# decides that is each column's role and whether it is complete, and
# neither is legible in five rows of raw values.
#
# So the portrait counts what is in the dataset, then groups columns by
# the role the profiler guessed, with a completeness bar for each.
# Entities as a pictograph, structure as the grouping, quantity as the
# bars, following shared/visuals.py's grammar. The raw rows stay on the
# page underneath wherever this is used, because a portrait is not a
# substitute for looking at the actual values.
#
# The counts come first, and they are facts: how many columns hold
# numbers, how many of those have few enough distinct values to read
# either way, how many hold text, how many values are absent. The role
# grouping comes second and is labelled as a guess, because it is one. A
# nine-point confidence scale is under the profiler's categorical
# threshold, so Impact Evaluation's own sample lands its pre and post
# scores in categorical-like; whether that is the right reading is a
# modelling question, and a portrait that summarised itself as having no
# continuous columns would be settling it on the reader's behalf.
_PORTRAIT_W = 640.0
_PORTRAIT_HEADER_H = 132.0

# The four counts across the top, and where each sits.
_PORTRAIT_TALLY_X = (0.0, 168.0, 348.0, 508.0)
_PORTRAIT_ROW_STEP = 24.0
_PORTRAIT_GROUP_STEP = 30.0
_PORTRAIT_BAR_X, _PORTRAIT_BAR_W = 412.0, 200.0
_PORTRAIT_COUNT_X = 396.0

# Past this the portrait stops being a glance. The profile table
# underneath carries every column, so the rest are counted rather than
# drawn.
_PORTRAIT_MAX_COLUMNS = 14

# A name longer than this is shortened here and read in full in the table
# below. Unlike a claim on the support boundary, a shortened column name
# is still recognisable, so this truncates rather than refusing.
_PORTRAIT_MAX_NAME = 46


def _portrait_bar(y: float, pct_missing: float) -> str:
    """One column's completeness, as a bar with its gap drawn."""
    present_w = _PORTRAIT_BAR_W * (100.0 - pct_missing) / 100.0

    bar = (
        f'<rect x="{_PORTRAIT_BAR_X:.0f}" y="{y - 8:.0f}" '
        f'width="{_PORTRAIT_BAR_W:.0f}" height="9" rx="2" '
        f'fill="{visuals.MISSING}"/>'
    )

    if present_w > 0:
        bar += (
            f'<rect x="{_PORTRAIT_BAR_X:.0f}" y="{y - 8:.0f}" '
            f'width="{present_w:.1f}" height="9" rx="2" fill="{visuals.ACCENT}" '
            f'fill-opacity="0.55"/>'
        )

    return bar


def dataset_portrait_svg(
    profile: DataProfile, *, source_name: str, is_sample: bool
) -> str:
    """
    What this dataset is, before anyone picks a column out of it.

    Columns are grouped by the role the profiler guessed, in the order
    the roles are declared, so the grouping is stable across datasets
    rather than reordering itself with the data. The guess is a heuristic
    and the caption underneath says so; the portrait shows what it
    guessed, not a determination.
    """
    rows = f"{profile.n_rows:,} rows, {profile.n_columns:,} columns"

    shown = profile.columns[:_PORTRAIT_MAX_COLUMNS]
    by_role = {
        role: [column for column in shown if column.role == role]
        for role in ROLES
    }

    n_groups = sum(1 for columns in by_role.values() if columns)
    drawn_groups = 0

    body = ""
    y = _PORTRAIT_HEADER_H + 24

    for role, columns in by_role.items():
        if not columns:
            continue

        body += (
            f'<text x="0" y="{y:.0f}" font-size="12" fill="{visuals.INK_MUTED}" '
            f'font-weight="600">Likely role: {role}</text>'
        )
        y += _PORTRAIT_ROW_STEP

        for column in columns:
            name = column.name
            if len(name) > _PORTRAIT_MAX_NAME:
                name = name[: _PORTRAIT_MAX_NAME - 1] + "…"

            detail = f"{column.n_unique:,} unique"
            if column.n_missing:
                detail += f", {column.n_missing:,} missing"

            body += (
                f'<text x="14" y="{y:.0f}" font-size="14" fill="{visuals.INK_MUTED}">'
                f"{name}</text>"
                f'<text x="{_PORTRAIT_COUNT_X:.0f}" y="{y:.0f}" font-size="12" '
                f'fill="{visuals.INK_MUTED}" text-anchor="end">{detail}</text>'
                + _portrait_bar(y, column.pct_missing)
            )
            y += _PORTRAIT_ROW_STEP

        drawn_groups += 1
        if drawn_groups < n_groups:
            y += _PORTRAIT_GROUP_STEP - _PORTRAIT_ROW_STEP

    remaining = profile.n_columns - len(shown)
    if remaining:
        body += (
            f'<text x="0" y="{y:.0f}" font-size="12" fill="{visuals.INK_MUTED}">'
            f"and {remaining:,} more columns, listed in the profile below"
            "</text>"
        )
        y += _PORTRAIT_ROW_STEP

    provenance = (
        "Bundled sample dataset" if is_sample else f"Uploaded: {source_name}"
    )

    # Four counts of how the data is stored, before any reading of it.
    # "of them low-cardinality" rather than a fourth category, because
    # those columns are a subset of the numeric ones and printing the two
    # side by side would read as seven columns where there are five.
    tally = (
        (f"{profile.n_numeric:,}", "numeric columns"),
        (f"{profile.n_low_cardinality_numeric:,}", "of them low-cardinality"),
        (f"{profile.n_non_numeric:,}", "text, date or category"),
        (f"{profile.n_missing_cells:,}", "values missing"),
    )

    counts = "".join(
        f'<text x="{x:.0f}" y="102" font-size="26" fill="{visuals.INK_MUTED}">'
        f"{value}</text>"
        f'<text x="{x:.0f}" y="120" font-size="12" fill="{visuals.INK_MUTED}">'
        f"{caption}</text>"
        for x, (value, caption) in zip(_PORTRAIT_TALLY_X, tally)
    )

    header = (
        visuals.unit_cluster(24, 30, visuals.ACCENT, count=5, radius=6.5)
        + f'<text x="58" y="36" font-size="17" fill="{visuals.INK_MUTED}">{rows}</text>'
        + f'<text x="{_PORTRAIT_W:.0f}" y="36" font-size="12" '
        f'fill="{visuals.INK_MUTED}" text-anchor="end">{provenance}</text>'
        + f'<line x1="0" y1="60" x2="{_PORTRAIT_W:.0f}" y2="60" '
        f'stroke="{visuals.GRIDLINE}" stroke-width="1"/>'
        + counts
        + f'<line x1="0" y1="{_PORTRAIT_HEADER_H:.0f}" x2="{_PORTRAIT_W:.0f}" '
        f'y2="{_PORTRAIT_HEADER_H:.0f}" stroke="{visuals.GRIDLINE}" '
        f'stroke-width="1"/>'
    )

    return visuals.figure(
        header + body,
        width=_PORTRAIT_W,
        height=y,
        label=(
            f"{provenance}. {rows}. "
            + ", ".join(f"{value} {caption}" for value, caption in tally)
            + ". Grouped by a guess at each column's likely role: "
            + "; ".join(
                f"{role}, {', '.join(c.name for c in columns)}"
                for role, columns in by_role.items()
                if columns
            )
            + ". Each bar shows how much of a column is present."
        ),
    )


# The catalog portrait. Deliberately a different builder from the one
# above, taking a RealDataset rather than a DataProfile.
#
# The two describe different things and conflating them would erase the
# distinction the catalog exists to hold: a DataProfile is measured from
# a file someone has, and a catalog entry describes a dataset most
# readers have not obtained and several cannot be handed. Six of the nine
# entries here cannot be opened by OpenMeasure at all. Drawing both from
# one builder would mean either inventing a profile for a file nobody
# has, or dropping the entries that are only ever descriptions.
#
# So this draws only what the catalog knows: a few verified facts about
# the dataset's size and shape, as a strip. Every fact comes from the
# entry's own description.
_STRIP_W = 640.0
_STRIP_H = 46.0
_STRIP_TEXT_Y = 30.0

_STRIP_FONT = 15.0
_STRIP_GAP = 34.0
_STRIP_MAX_FACTS = 3

# SVG cannot measure text, so a strip laid out left to right has to
# estimate how wide each fact is before placing the arrow after it. One
# flat width per character is not good enough: "837,382 rows" and "One
# station at a time" are the same length in characters and visibly
# different in pixels, so the arrows landed against one fact and adrift
# from another.
#
# Three buckets, as fractions of the font size. Approximate, but wrong by
# a character or two rather than by a word, which is the difference
# between arrows that look evenly spaced and arrows that do not.
_NARROW_CHARACTERS = "ijl.,'!|:;() "
_WIDE_CHARACTERS = "mwMW@ABCDEFGHIJKLNOPQRSTUVXYZ"
# Digits are tabular in the faces this renders in, so they are all one
# width and a little wider than lowercase. Counts are most of what a
# scale fact contains, so getting them wrong showed up immediately as
# arrows pressed against the text after them.
_DIGITS = "0123456789"
_NARROW_RATIO, _NORMAL_RATIO, _WIDE_RATIO = 0.28, 0.53, 0.72
_DIGIT_RATIO = 0.56


def _text_width(text: str, font_size: float) -> float:
    """Roughly how wide a string renders, for laying figures out."""
    total = 0.0

    for character in text:
        if character in _NARROW_CHARACTERS:
            total += _NARROW_RATIO
        elif character in _DIGITS:
            total += _DIGIT_RATIO
        elif character in _WIDE_CHARACTERS:
            total += _WIDE_RATIO
        else:
            total += _NORMAL_RATIO

    return total * font_size


def catalog_portrait_svg(dataset: RealDataset) -> str:
    """
    A catalogued dataset's shape, as a strip of facts.

    Returns an empty string when the entry states no verified counts,
    which several do not. An empty strip is the honest rendering of that:
    an estimate drawn in the same place as a verified figure would be
    indistinguishable from one.

    Facts are drawn left to right with arrows between them, since they
    read as a narrowing (people, then items, then the ones with complete
    answers) rather than as an unordered set.
    """
    facts = tuple(dataset.scale[:_STRIP_MAX_FACTS])

    if not facts:
        return ""

    x = 58.0
    parts = [visuals.unit_cluster(24, 23, visuals.ACCENT, count=5, radius=5.5)]

    for index, fact in enumerate(facts):
        if index:
            parts.append(
                visuals.arrow(x, _STRIP_TEXT_Y - 5, x + _STRIP_GAP - 10, _STRIP_TEXT_Y - 5)
            )
            x += _STRIP_GAP

        parts.append(
            f'<text x="{x:.0f}" y="{_STRIP_TEXT_Y:.0f}" '
            f'font-size="{_STRIP_FONT:.0f}" '
            f'fill="{visuals.INK_MUTED}">{fact}</text>'
        )
        x += _text_width(fact, _STRIP_FONT)

    return visuals.figure(
        "".join(parts),
        width=_STRIP_W,
        height=_STRIP_H,
        label=f"{dataset.name}: " + ", then ".join(facts) + ".",
    )
