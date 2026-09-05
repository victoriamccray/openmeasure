"""
Unit tests for core/domains.py

The separation rule has its own class below. A domain tailors what a
researcher reads and searches for and never what the toolkit computes,
and that is enforced here rather than left to review.

Run with: pytest modules/program_evaluation/tests/ -v
"""

from __future__ import annotations

import inspect
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import comparison, did, domains, recommend, teaching  # noqa: E402

CORE_DIR = Path(__file__).resolve().parents[1] / "core"

# The six named fields, plus Other. Pinned as a list rather than derived
# from DOMAINS, so silently dropping one from the selector fails here.
EXPECTED_DOMAIN_IDS = (
    "public_health",
    "social_programs",
    "education",
    "workforce",
    "criminal_justice",
    "digital",
    "other",
)


class TestDomainCoverage(unittest.TestCase):
    def test_all_seven_domains_are_present_in_order(self):
        self.assertEqual(domains.DOMAIN_IDS, EXPECTED_DOMAIN_IDS)

    def test_every_named_domain_seeds_search_terms(self):
        for domain in domains.DOMAINS:
            if domain.id == domains.DOMAIN_OTHER:
                continue
            with self.subTest(domain=domain.id):
                self.assertTrue(domain.search_terms)

    def test_every_named_domain_suggests_outcomes(self):
        for domain in domains.DOMAINS:
            if domain.id == domains.DOMAIN_OTHER:
                continue
            with self.subTest(domain=domain.id):
                self.assertTrue(domain.outcomes)

    def test_every_domain_glosses_every_concept(self):
        for domain in domains.DOMAINS:
            for concept in domains.CONCEPTS:
                with self.subTest(domain=domain.id, concept=concept):
                    self.assertTrue(domain.term_for(concept))

    def test_every_named_domain_renames_most_concepts(self):
        """
        A domain that called every concept by its statistical name would
        be Other wearing a label. Named domains must differ on at least
        half the concepts; identity on one or two is legitimate, since
        social program evaluation really does say "comparison group".
        """
        for domain in domains.DOMAINS:
            if domain.id == domains.DOMAIN_OTHER:
                continue
            renamed = sum(
                1
                for concept in domains.CONCEPTS
                if domain.term_for(concept).lower() != concept.lower()
            )
            with self.subTest(domain=domain.id):
                self.assertGreaterEqual(renamed, len(domains.CONCEPTS) // 2)

    def test_search_terms_never_name_a_study_design(self):
        """
        Search terms describe a field, not a method. Seeding a search with
        design vocabulary would steer a researcher toward literature using
        one design before they have chosen their own.
        """
        design_words = (
            "randomi",
            "quasi-experiment",
            "difference-in-differences",
            "regression discontinuity",
            "propensity",
            "controlled trial",
            "interrupted time series",
        )
        for domain in domains.DOMAINS:
            for term in domain.search_terms:
                for word in design_words:
                    with self.subTest(domain=domain.id, term=term, word=word):
                        self.assertNotIn(word, term.lower())


class TestCaveats(unittest.TestCase):
    def test_reentry_recidivism_outcomes_carry_a_caveat(self):
        """
        Offering recidivism as a tidy outcome chip with no note would be
        the toolkit doing what it exists to prevent.
        """
        reentry = domains.get_domain("criminal_justice")
        recidivism = [
            outcome
            for outcome in reentry.outcomes
            if "arrest" in outcome.label.lower()
            or "incarcerat" in outcome.label.lower()
        ]

        self.assertTrue(recidivism)
        for outcome in recidivism:
            with self.subTest(outcome=outcome.label):
                self.assertTrue(outcome.caveat.strip())

    def test_test_score_and_retention_outcomes_carry_a_caveat(self):
        for domain_id, fragment in (
            ("education", "standardized test"),
            ("digital", "retention"),
            ("workforce", "earnings"),
        ):
            domain = domains.get_domain(domain_id)
            matching = [
                outcome
                for outcome in domain.outcomes
                if fragment in outcome.label.lower()
            ]
            with self.subTest(domain=domain_id):
                self.assertTrue(matching)
                self.assertTrue(all(o.caveat.strip() for o in matching))

    def test_every_caveat_is_a_sentence_not_a_fragment(self):
        for domain in domains.DOMAINS:
            for outcome in domain.outcomes:
                if not outcome.caveat:
                    continue
                with self.subTest(domain=domain.id, outcome=outcome.label):
                    self.assertTrue(outcome.caveat.endswith("."))


class TestOtherDomain(unittest.TestCase):
    """
    Other has to be a real answer. Selecting it must leave a researcher
    exactly where they would be with no domain stage at all, never worse.
    """

    def test_other_seeds_no_search_terms(self):
        self.assertEqual(domains.get_domain(domains.DOMAIN_OTHER).search_terms, ())

    def test_other_leaves_the_query_untouched(self):
        query = domains.build_search_query(
            "after-school program attendance", domains.DOMAIN_OTHER
        )

        self.assertEqual(query, "after-school program attendance")

    def test_other_glosses_every_concept_with_the_concept_itself(self):
        other = domains.get_domain(domains.DOMAIN_OTHER)

        for concept in domains.CONCEPTS:
            with self.subTest(concept=concept):
                self.assertEqual(other.term_for(concept), concept)


class TestSearchQuery(unittest.TestCase):
    def test_domain_terms_are_appended_after_the_researchers_words(self):
        query = domains.build_search_query("appointment reminders", "public_health")

        self.assertTrue(query.startswith("appointment reminders"))
        self.assertIn("public health", query)

    def test_the_researchers_words_are_never_dropped(self):
        for domain in domains.DOMAINS:
            with self.subTest(domain=domain.id):
                query = domains.build_search_query("my own phrasing", domain.id)
                self.assertIn("my own phrasing", query)

    def test_empty_question_terms_raise(self):
        for blank in ("", "   ", "\n"):
            with self.subTest(question=repr(blank)):
                with self.assertRaises(ValueError):
                    domains.build_search_query(blank, "education")

    def test_unknown_domain_raises(self):
        with self.assertRaises(ValueError):
            domains.build_search_query("something", "astrophysics")


class TestValidation(unittest.TestCase):
    def test_unknown_domain_id_raises_and_names_the_known_ids(self):
        with self.assertRaises(ValueError) as raised:
            domains.get_domain("healthcare")

        message = str(raised.exception)
        self.assertIn("healthcare", message)
        self.assertIn("public_health", message)

    def test_unknown_concept_raises_rather_than_falling_back(self):
        with self.assertRaises(ValueError):
            domains.get_domain("education").term_for("Sample size")

    def test_a_gloss_for_an_unknown_concept_is_rejected(self):
        with self.assertRaises(ValueError):
            domains.TermGloss(concept="Effect size", domain_term="impact")

    def test_a_domain_missing_a_concept_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            domains.Domain(
                id="partial",
                label="Partial",
                search_terms=(),
                outcomes=(),
                terminology=(
                    domains.TermGloss(domains.CONCEPT_UNIT, "widget"),
                ),
            )

        self.assertIn("does not gloss", str(raised.exception))

    def test_a_domain_glossing_a_concept_twice_is_rejected(self):
        doubled = (
            *domains._identity_terminology(),
            domains.TermGloss(domains.CONCEPT_UNIT, "again"),
        )

        with self.assertRaises(ValueError) as raised:
            domains.Domain(
                id="doubled",
                label="Doubled",
                search_terms=(),
                outcomes=(),
                terminology=doubled,
            )

        self.assertIn("more than once", str(raised.exception))

    def test_an_outcome_without_a_label_is_rejected(self):
        with self.assertRaises(ValueError):
            domains.OutcomeSuggestion(label="")


class TestDomainCannotReachTheAnalysis(unittest.TestCase):
    """
    The separation rule, enforced three ways.

    A domain informs what a researcher reads and searches for. It must not
    change which method is recommended or what any statistic comes out as.
    Wording in a docstring would not survive a future edit; these will
    fail on one.
    """

    ANALYSIS_MODULES = ("recommend.py", "comparison.py", "did.py")

    def test_domain_carries_no_method_or_design_field(self):
        forbidden = ("method", "design", "test", "statistic", "estimator", "model")
        field_names = set(domains.Domain.__dataclass_fields__)

        for name in field_names:
            for word in forbidden:
                with self.subTest(field=name, word=word):
                    self.assertNotIn(word, name.lower())

    def test_the_analysis_modules_do_not_import_domains(self):
        for filename in self.ANALYSIS_MODULES:
            source = (CORE_DIR / filename).read_text(encoding="utf-8")
            for forbidden in ("import domains", "from .domains", "domains."):
                with self.subTest(module=filename, forbidden=forbidden):
                    self.assertNotIn(
                        forbidden,
                        source,
                        f"core/{filename} references domains. A domain must "
                        "not reach the method recommendation or the "
                        "statistical computation.",
                    )

    def test_no_analysis_function_accepts_a_domain_argument(self):
        functions = [
            recommend.recommend_method,
            did.estimate_did,
            comparison.compare_two_groups,
            comparison.compare_pre_post,
            comparison.compare_multiple_groups_welch,
            comparison.compare_categorical,
        ]

        for function in functions:
            parameters = inspect.signature(function).parameters
            for name in parameters:
                with self.subTest(function=function.__name__, parameter=name):
                    self.assertNotIn("domain", name.lower())

    def test_every_domain_variant_produces_the_same_estimate(self):
        """
        The separation rule as arithmetic. The clinic, the school, and the
        workforce board tell different stories about the same numbers, so
        the estimate must not move when the domain does.
        """
        for change in (-10.0, -4.0, 0.0, 5.0, 12.0, 20.0):
            estimates = {
                teaching.teaching_did(
                    change, scenario=teaching.did_scenario_for(domain.id)
                ).did_estimate
                for domain in domains.DOMAINS
            }
            with self.subTest(comparison_change=change):
                self.assertEqual(len(estimates), 1)

    def test_every_domain_variant_shares_the_same_fixed_numbers(self):
        constants = {
            (
                teaching.did_scenario_for(domain.id).pre_treated,
                teaching.did_scenario_for(domain.id).post_treated,
                teaching.did_scenario_for(domain.id).pre_comparison,
            )
            for domain in domains.DOMAINS
        }

        self.assertEqual(constants, {(60.0, 72.0, 58.0)})


if __name__ == "__main__":
    unittest.main()
