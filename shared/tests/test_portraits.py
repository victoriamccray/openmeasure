"""
Unit tests for shared/portraits.py

Run with: pytest shared/tests/ -v

The portrait is drawn as SVG, and SVG text does not wrap. Most of what
can go wrong here is a value running off the canvas rather than an
exception, so these check what the figure actually contains.
"""

from __future__ import annotations

import unittest

import pandas as pd

from modules.data_profile.core.profile import profile_dataframe
from shared.portraits import dataset_portrait_svg


def _portrait(frame, *, source_name="study.csv", is_sample=False) -> str:
    return dataset_portrait_svg(
        profile_dataframe(frame), source_name=source_name, is_sample=is_sample
    )


class TestCountsAreFacts(unittest.TestCase):
    """
    The four counts across the top say how the data is stored, which is
    true whatever anyone decides the roles are.
    """

    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "participant_id": range(1, 21),
                "rating": [(i % 5) + 1 for i in range(20)],
                "arm": ["a", "b"] * 10,
            }
        )
        self.frame.loc[self.frame.index[:4], "rating"] = None

    def test_rows_and_columns_are_stated(self):
        self.assertIn("20 rows, 3 columns", _portrait(self.frame))

    def test_the_tally_labels_are_present(self):
        svg = _portrait(self.frame)

        for label in (
            "numeric columns",
            "of them low-cardinality",
            "text, date or category",
            "values missing",
        ):
            with self.subTest(label=label):
                self.assertIn(label, svg)

    def test_it_never_summarises_the_role_guess_as_a_count(self):
        """
        A dataset of rating scales has no continuous-like column, and a
        headline saying so would state the heuristic more confidently
        than the heuristic supports.
        """
        svg = _portrait(self.frame).lower()

        self.assertNotIn("no continuous", svg)
        self.assertNotIn("continuous columns", svg)


class TestRolesAreShownAsGuesses(unittest.TestCase):
    def test_each_group_heading_says_likely(self):
        frame = pd.DataFrame({"arm": ["a", "b"] * 10, "score": range(20)})

        self.assertIn("Likely role:", _portrait(frame))

    def test_a_rating_scale_appears_without_being_called_continuous(self):
        frame = pd.DataFrame({"confidence": [(i % 9) + 1 for i in range(40)]})
        svg = _portrait(frame)

        self.assertIn("confidence", svg)
        self.assertIn("categorical-like", svg)


class TestProvenance(unittest.TestCase):
    def test_a_bundled_sample_says_so(self):
        frame = pd.DataFrame({"a": [1, 2, 3]})

        self.assertIn("Bundled sample dataset", _portrait(frame, is_sample=True))

    def test_an_upload_is_named(self):
        frame = pd.DataFrame({"a": [1, 2, 3]})

        self.assertIn("Uploaded: my_study.csv", _portrait(frame, source_name="my_study.csv"))


class TestItFitsTheCanvas(unittest.TestCase):
    """
    SVG text does not wrap, so anything too long is lost off the edge
    rather than clipped visibly.
    """

    def test_a_long_column_name_is_shortened_where_it_is_drawn(self):
        name = "an_extremely_long_column_name_that_would_run_past_the_bars_entirely"
        frame = pd.DataFrame({name: [1, 2, 3]})
        svg = _portrait(frame)
        drawn = svg.split("aria-label=")[1].split(">", 1)[1]

        self.assertNotIn(name, drawn)
        self.assertIn("…", drawn)

    def test_the_full_name_survives_in_the_accessible_label(self):
        """
        Shortening is a drawing constraint, not an edit to the data. A
        reader who cannot see the figure should get the column's actual
        name, which is also the one they pick from the dropdown.
        """
        name = "an_extremely_long_column_name_that_would_run_past_the_bars_entirely"
        frame = pd.DataFrame({name: [1, 2, 3]})
        label = _portrait(frame).split('aria-label="')[1].split('"')[0]

        self.assertIn(name, label)


    def test_many_columns_are_capped_and_the_rest_counted(self):
        frame = pd.DataFrame({f"item_{i:02d}": range(5) for i in range(20)})
        svg = _portrait(frame)

        self.assertIn("more columns, listed in the profile below", svg)
        self.assertNotIn("item_19", svg)

    def test_the_figure_scales_rather_than_taking_a_pixel_height(self):
        """
        A pixel height alongside width=100% pins the drawing to native
        size inside a much wider empty box.
        """
        svg = _portrait(pd.DataFrame({"a": [1, 2, 3]}))

        self.assertIn("width:100%;height:auto", svg)
        self.assertNotIn('height="', svg.split("viewBox")[0])


class TestAccessibleLabel(unittest.TestCase):
    def test_the_label_names_the_counts_and_the_columns(self):
        frame = pd.DataFrame({"arm": ["a", "b"] * 10, "score": range(20)})
        label = _portrait(frame).split('aria-label="')[1].split('"')[0]

        self.assertIn("20 rows, 2 columns", label)
        self.assertIn("arm", label)
        self.assertIn("guess", label)
