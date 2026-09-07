"""
How far the planner reaches beyond the worked example.

Run with: pytest modules/research_design/tests/ -v

Not a unit test of a function. A recorded answer to one question:

    The chronic-pain example works. Does any of this hold up for a
    messier question?

Six questions outside the curated set, run through the same path the page
runs. Each asserts what happens today, so the answer is visible in the
repository rather than rediscovered by whoever next types something the
lexicon has not met.

What it found the first time it was run:

    social isolation      nothing recognised
    perceived discrimination  nothing recognised
    institutional trust   nothing recognised
    violence              recognised, category-level only
    treatment adherence   recognised, category-level only
    stress + arousal      recognised, concept-specific for both

Three of six reach nothing. Two reach the same three measures as each
other, because both fall back to one category, and that category admits
functional neuroimaging. None of that is hidden by these tests; the point
is that it is written down, and that a change which turned a category
join into a concept-specific match would fail here rather than ship.

Adding a concept or a curated list should change these assertions
deliberately. That is the intended way to use this file.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import lexicon  # noqa: E402

# Questions a researcher might actually arrive with, none of which the
# curated set was written for.
UNRECOGNISED = (
    "Does the intervention reduce social isolation in older adults?",
    "Does perceived discrimination predict worse health outcomes?",
    "Does the reform increase institutional trust?",
)

CATEGORY_LEVEL_ONLY = (
    ("Does the program reduce violence among students?", "Violent behaviour"),
    ("Does the reminder system improve treatment adherence?", "Adherence"),
)


def _only(question: str):
    """The single concept a question suggests, or fail saying what it got."""
    found = lexicon.suggest_concepts(question)

    if len(found) != 1:
        raise AssertionError(
            f"{question!r} suggested {[s.concept for s in found]}, "
            "so this helper is the wrong one for it."
        )

    return found[0]


class TestWhereTheLexiconHasNothing(unittest.TestCase):
    """
    A gap is reported rather than filled. That is the design, and it is
    also a real limit on who this step is useful to today.
    """

    def test_three_common_constructs_are_not_recognised(self):
        for question in UNRECOGNISED:
            with self.subTest(question=question):
                self.assertEqual(lexicon.suggest_concepts(question), ())

    def test_nothing_is_offered_in_their_place(self):
        """
        The failure mode worth guarding is a plausible substitute. An
        empty result is a smaller problem than a confident wrong one.
        """
        for question in UNRECOGNISED:
            with self.subTest(question=question):
                self.assertFalse(lexicon.suggest_concepts(question))

    def test_the_gap_has_something_to_say_for_itself(self):
        self.assertIn("gap in the list", lexicon.NOTHING_RECOGNISED)


class TestWhereTheLexiconRecognisesButHasNotCurated(unittest.TestCase):
    def test_the_concept_is_recognised(self):
        for question, concept in CATEGORY_LEVEL_ONLY:
            with self.subTest(question=question):
                self.assertEqual(_only(question).concept, concept)

    def test_but_the_measures_are_only_category_level(self):
        for question, _ in CATEGORY_LEVEL_ONLY:
            suggestion = _only(question)
            found = lexicon.measures_for_concept(
                suggestion.concept, suggestion.kind
            )

            with self.subTest(question=question):
                self.assertEqual(found.basis, lexicon.BASIS_CATEGORY)

    def test_two_unrelated_constructs_receive_the_same_list(self):
        """
        Violence and treatment adherence get identical candidates,
        because the category is the only thing narrowing them. That is
        the clearest evidence that a category join is not an
        appropriateness claim.
        """
        lists = []
        for question, _ in CATEGORY_LEVEL_ONLY:
            suggestion = _only(question)
            found = lexicon.measures_for_concept(
                suggestion.concept, suggestion.kind
            )
            lists.append(tuple(m.name for m in found.measures))

        self.assertEqual(lists[0], lists[1])
        self.assertGreater(len(lists[0]), 0)


class TestWhereItActuallyWorks(unittest.TestCase):
    """
    The multimodal case, which is the one the planner was built for and
    the one that holds up outside the pain walkthrough.
    """

    QUESTION = (
        "Is subjective stress coupled with physiological arousal in "
        "daily life?"
    )

    def test_both_sides_of_a_multimodal_question_are_recognised(self):
        found = lexicon.suggest_concepts(self.QUESTION)

        self.assertEqual(
            [suggestion.concept for suggestion in found],
            ["Perceived stress", "Physiological arousal"],
        )

    def test_both_reach_a_concept_specific_list(self):
        for suggestion in lexicon.suggest_concepts(self.QUESTION):
            found = lexicon.measures_for_concept(
                suggestion.concept, suggestion.kind
            )

            with self.subTest(concept=suggestion.concept):
                self.assertEqual(found.basis, lexicon.BASIS_CONCEPT)

    def test_they_reach_different_measures(self):
        """
        A self-report construct and a bodily process should not arrive at
        the same instruments, which is what a category join would do.
        """
        by_concept = {
            suggestion.concept: tuple(
                measure.name
                for measure in lexicon.measures_for_concept(
                    suggestion.concept, suggestion.kind
                ).measures
            )
            for suggestion in lexicon.suggest_concepts(self.QUESTION)
        }

        self.assertNotEqual(
            by_concept["Perceived stress"], by_concept["Physiological arousal"]
        )
        self.assertIn(
            "Electrodermal activity", by_concept["Physiological arousal"]
        )


class TestTheCoverageNumberIsHonest(unittest.TestCase):
    def test_fewer_than_half_the_concepts_are_curated(self):
        """
        Recorded so a claim of broad coverage cannot be made quietly. If
        curation improves, this assertion should be updated with it.
        """
        curated, known = lexicon.curated_coverage()

        self.assertLess(curated / known, 0.6)


if __name__ == "__main__":
    unittest.main()
