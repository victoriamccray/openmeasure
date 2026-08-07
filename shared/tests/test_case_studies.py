"""
Unit tests for shared/case_studies.py

These protect correctness and architecture, not writing style. Every
assertion here catches something that can be false or broken. Formatting
preferences such as em dashes or colons in titles are handled in review,
not enforced here, because a future title may legitimately need them.

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import re
import unittest

from shared.case_studies import (
    CASE_STUDIES,
    PAGE_ORDER,
    CaseStudy,
    get_case_studies,
)
from shared.catalog import LIFECYCLE_STAGES, WORKFLOWS

# Generous ceiling, as a guardrail against silently re-cluttering the pages.
# Before this was addressed one page carried ten studies and roughly 960
# words of prose. The point is to catch a slide back, not to police curation.
MAX_STUDIES_PER_PAGE = 6

# Phrases that refer to another study by its position on the page. These
# become false when grouping order changes, which is exactly what happened
# when "the LaLonde example below" survived a regrouping.
#
# Deliberately phrase-based rather than word-based: a bare "above" is
# legitimate ("public debt above 90% of GDP"), and "later" is legitimate
# when it means temporally later ("later reanalyses").
POSITIONAL_REFERENCE = re.compile(
    r"\b(?:"
    r"(?:example|study|case|entry|entries|one)s?\s+(?:below|above)"
    r"|see\s+(?:below|above)"
    r"|(?:listed|shown|described|discussed)\s+(?:below|above)"
    r"|the\s+(?:following|preceding|next|previous)\s+(?:example|study|case|entry)"
    r")\b",
    re.IGNORECASE,
)


def taxonomy_keys() -> set[str]:
    """Taxonomy keys the module catalog actually uses."""

    return {
        workflow.taxonomy_key
        for workflow in WORKFLOWS
        if workflow.taxonomy_key is not None
    }


class TestStages(unittest.TestCase):
    def test_every_study_has_a_known_stage(self):
        for key, study in CASE_STUDIES.items():
            with self.subTest(study=key):
                self.assertIn(study.stage, LIFECYCLE_STAGES)

    def test_unknown_stage_is_rejected_at_construction(self):
        # An unknown stage would render in no group at all, so the study
        # would silently disappear from every page.
        with self.assertRaises(ValueError) as context:
            CaseStudy(
                title="Bogus",
                stage="Not A Stage",
                principle="p",
                summary="s",
                takeaway="t",
                citation="c",
                modules=("data_validation",),
            )

        self.assertIn("not a known research stage", str(context.exception))


class TestPageOrder(unittest.TestCase):
    """
    Order and membership are declared in two places, so they must agree.

    A study tagged for a page but missing from that page's order would
    silently disappear from the page. One listed in the order but not tagged
    would appear on a page it was never meant for.
    """

    def test_order_and_tagging_agree_exactly(self):
        for key, order in PAGE_ORDER.items():
            tagged = {
                study_key
                for study_key, study in CASE_STUDIES.items()
                if key in study.modules
            }
            with self.subTest(page=key):
                self.assertEqual(
                    set(order),
                    tagged,
                    f"'{key}' order and module tagging disagree. Only in "
                    f"order: {sorted(set(order) - tagged)}. Only tagged: "
                    f"{sorted(tagged - set(order))}.",
                )

    def test_every_page_declares_an_order(self):
        for key in sorted(taxonomy_keys()):
            with self.subTest(page=key):
                self.assertIn(key, PAGE_ORDER)

    def test_no_page_lists_a_study_twice(self):
        for key, order in PAGE_ORDER.items():
            with self.subTest(page=key):
                self.assertEqual(len(order), len(set(order)))

    def test_every_ordered_key_exists(self):
        for key, order in PAGE_ORDER.items():
            for study_key in order:
                with self.subTest(page=key, study=study_key):
                    self.assertIn(study_key, CASE_STUDIES)

    def test_studies_are_returned_in_declared_order(self):
        for key, order in PAGE_ORDER.items():
            with self.subTest(page=key):
                titles = [study.title for study in get_case_studies(key)]
                expected = [CASE_STUDIES[k].title for k in order]
                self.assertEqual(titles, expected)

    def test_fairness_leads_with_its_two_most_relevant_examples(self):
        # Requested explicitly: the fairness tradeoff result and the video
        # game training study should be the first two a reader meets.
        leading = [study.title for study in get_case_studies("model_validation")][:2]

        self.assertEqual(
            leading,
            [
                CASE_STUDIES["fairness_impossibility"].title,
                CASE_STUDIES["confounding_video_games"].title,
            ],
        )


class TestTaxonomyKeys(unittest.TestCase):
    def test_every_module_key_is_one_the_catalog_uses(self):
        valid = taxonomy_keys()

        for key, study in CASE_STUDIES.items():
            for module in study.modules:
                with self.subTest(study=key, module=module):
                    self.assertIn(
                        module,
                        valid,
                        f"'{module}' is not a taxonomy key any workflow uses, "
                        "so this study would render on no page.",
                    )

    def test_every_page_has_at_least_one_study(self):
        # An unused key means a page shows an empty research examples
        # section, which reads as a bug to a user.
        for key in sorted(taxonomy_keys()):
            with self.subTest(page=key):
                self.assertTrue(
                    get_case_studies(key),
                    f"No case study is tagged for '{key}'.",
                )

    def test_no_study_is_orphaned(self):
        for key, study in CASE_STUDIES.items():
            with self.subTest(study=key):
                self.assertTrue(
                    study.modules,
                    f"'{key}' is tagged to no page, so it is dead content.",
                )


class TestNoPositionalReferences(unittest.TestCase):
    """
    A study must not point at another study by position.

    This is a correctness test, not a style test. "The LaLonde example
    below" was true only because two studies happened to sit adjacent under
    the same grouping, and regrouping would have made the sentence false
    without anything failing.
    """

    def test_no_summary_or_takeaway_refers_to_a_position(self):
        for key, study in CASE_STUDIES.items():
            for field in ("summary", "takeaway", "principle", "title"):
                text = getattr(study, field)
                match = POSITIONAL_REFERENCE.search(text)
                with self.subTest(study=key, field=field):
                    self.assertIsNone(
                        match,
                        f"'{key}.{field}' refers to another study by "
                        f"position ({match.group(0) if match else ''}). Name "
                        "the study instead, because grouping order changes.",
                    )

    def test_the_pattern_catches_the_phrasing_that_caused_this(self):
        # Guards the guard. If the pattern stops matching the original bug,
        # the test above is worthless.
        original = (
            "random assignment greatly reduces the selection-bias problem "
            "illustrated by the LaLonde example below"
        )

        self.assertIsNotNone(POSITIONAL_REFERENCE.search(original))

    def test_the_pattern_does_not_fire_on_legitimate_wording(self):
        # A bare "above" is a numeric comparison, and "later" can be
        # temporal. Neither is a positional reference.
        for legitimate in (
            "countries with public debt above 90% of GDP",
            "Later reanalyses that modeled that crossover",
            "error during activity was higher than at rest",
            "the study found several favorable impacts",
        ):
            with self.subTest(text=legitimate):
                self.assertIsNone(POSITIONAL_REFERENCE.search(legitimate))


class TestPageVolume(unittest.TestCase):
    def test_no_page_exceeds_the_ceiling(self):
        for key in sorted(taxonomy_keys()):
            studies = get_case_studies(key)
            with self.subTest(page=key, count=len(studies)):
                self.assertLessEqual(
                    len(studies),
                    MAX_STUDIES_PER_PAGE,
                    f"'{key}' would show {len(studies)} studies. Prune for "
                    "relevance rather than raising the ceiling.",
                )


class TestRequiredContent(unittest.TestCase):
    def test_every_field_is_populated(self):
        for key, study in CASE_STUDIES.items():
            for field in (
                "title",
                "stage",
                "principle",
                "summary",
                "takeaway",
                "citation",
            ):
                with self.subTest(study=key, field=field):
                    self.assertTrue(getattr(study, field).strip())

    def test_titles_are_unique(self):
        titles = [study.title for study in CASE_STUDIES.values()]

        self.assertEqual(len(titles), len(set(titles)))

    def test_entries_are_frozen(self):
        self.assertTrue(CaseStudy.__dataclass_params__.frozen)


if __name__ == "__main__":
    unittest.main()
