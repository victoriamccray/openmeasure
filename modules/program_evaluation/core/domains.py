"""
Domain context for the Impact Evaluation workflow.

A domain tailors what a researcher reads and searches for: the words
their field uses for a design concept, the outcomes their field measures,
the caveats those outcomes carry, and the terms that seed a literature
search. It does not tailor the analysis.

That separation is the point of this module, and it is enforced rather
than described. A Domain carries no method, design, or test field, and
tests/test_domains.py asserts that neither recommend.py, comparison.py,
nor did.py imports this module or accepts a domain argument. Two
evaluations with the same data shape get the same recommendation and the
same numbers whether the researcher called their field public health or
left the domain unset.

What a domain does carry is vocabulary and measurement judgment, which
are genuinely field-specific. "Comparison group" is a control group in a
clinic, a business-as-usual condition in a school, and a holdout in a
product experiment. A rearrest count and a test score are both plausible
outcomes and both carry known measurement problems, which travel with
the suggestion here rather than being left for the researcher to
remember.

Search terms deliberately name fields and populations, never study
designs. Seeding a search with "randomized" or "quasi-experimental"
would push a researcher toward literature using one design before they
have decided which design their own question supports.

Case-study keys are not carried here. Anchoring published examples by
domain is a reasonable extension, but nothing needs it yet, and adding
it now would put the same association in two places.
"""

from __future__ import annotations

from dataclasses import dataclass

# The design concepts every domain names in its own words. Fixed and
# shared, so the glossary reads as one idea per row across domains
# rather than a different list per field.
CONCEPT_TREATMENT_GROUP = "Treatment group"
CONCEPT_COMPARISON_GROUP = "Comparison group"
CONCEPT_UNIT = "Unit of analysis"
CONCEPT_BASELINE = "Baseline measurement"
CONCEPT_FOLLOW_UP = "Follow-up measurement"
CONCEPT_OUTCOME = "Outcome"

CONCEPTS: tuple[str, ...] = (
    CONCEPT_TREATMENT_GROUP,
    CONCEPT_COMPARISON_GROUP,
    CONCEPT_UNIT,
    CONCEPT_BASELINE,
    CONCEPT_FOLLOW_UP,
    CONCEPT_OUTCOME,
)

# The id for "my field is not listed, or I would rather not say". Named
# rather than written as a bare string at each use, because several rules
# below and in teaching.py turn on it.
DOMAIN_OTHER = "other"


@dataclass(frozen=True)
class TermGloss:
    """What one field calls one design concept."""

    concept: str
    domain_term: str

    def __post_init__(self) -> None:
        if self.concept not in CONCEPTS:
            raise ValueError(
                f"'{self.concept}' is not a known design concept. Known "
                f"concepts: {', '.join(CONCEPTS)}."
            )

        if not self.domain_term:
            raise ValueError(f"The gloss for '{self.concept}' has no term.")


@dataclass(frozen=True)
class OutcomeSuggestion:
    """
    One outcome a field commonly measures, with what is known about it.

    The caveat travels with the suggestion rather than sitting in page
    prose, so an outcome cannot be offered somewhere the warning about it
    is not. Empty when the outcome carries no measurement problem worth
    raising at the point of choosing it; several here do.
    """

    label: str
    caveat: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("An outcome suggestion must have a label.")


@dataclass(frozen=True)
class Domain:
    """
    One field of practice, and the vocabulary and measures it uses.

    Deliberately has no method, design, or test field. See the module
    docstring: a domain informs what a researcher reads and searches,
    and never what the toolkit computes or recommends.
    """

    id: str
    label: str
    search_terms: tuple[str, ...]
    outcomes: tuple[OutcomeSuggestion, ...]
    terminology: tuple[TermGloss, ...]

    def __post_init__(self) -> None:
        for field_name in ("id", "label"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"{self.id or 'A domain'} is missing a value for "
                    f"'{field_name}'."
                )

        glossed = [gloss.concept for gloss in self.terminology]

        if len(glossed) != len(set(glossed)):
            raise ValueError(
                f"{self.id} glosses the same concept more than once."
            )

        missing = [concept for concept in CONCEPTS if concept not in glossed]

        if missing:
            raise ValueError(
                f"{self.id} does not gloss {missing}. Every domain names "
                "every concept, so the glossary has one row per concept "
                "whichever domain is selected."
            )

    def term_for(self, concept: str) -> str:
        """
        This domain's word for a concept.

        Raises on an unknown concept rather than falling back to the
        concept's own name: a silent fallback would make a typo look like
        a field that happens to use the statistical term.
        """
        for gloss in self.terminology:
            if gloss.concept == concept:
                return gloss.domain_term

        raise ValueError(
            f"'{concept}' is not a known design concept. Known concepts: "
            f"{', '.join(CONCEPTS)}."
        )


def _identity_terminology() -> tuple[TermGloss, ...]:
    """
    The neutral glossary, where each concept is called what it is called.

    Used by the "Other" domain. This is a real answer rather than a
    placeholder: a researcher who has not picked a field should see the
    general vocabulary, which is exactly the concept names.
    """
    return tuple(TermGloss(concept=concept, domain_term=concept) for concept in CONCEPTS)


DOMAINS: tuple[Domain, ...] = (
    Domain(
        id="public_health",
        label="Public health & healthcare",
        search_terms=(
            "public health",
            "health intervention",
            "patient outcomes",
            "health services",
        ),
        outcomes=(
            OutcomeSuggestion(label="Appointments kept"),
            OutcomeSuggestion(label="Screening or vaccination uptake"),
            OutcomeSuggestion(
                label="Hospital readmission within 30 days",
                caveat=(
                    "Readmission responds to discharge practice, bed "
                    "availability, and how a system codes an observation "
                    "stay, so it reflects the care system alongside the "
                    "patient's health."
                ),
            ),
            OutcomeSuggestion(
                label="Medication adherence",
                caveat=(
                    "Pharmacy refill records measure what was dispensed, "
                    "which is a proxy for what was taken."
                ),
            ),
        ),
        terminology=(
            TermGloss(CONCEPT_TREATMENT_GROUP, "intervention arm"),
            TermGloss(CONCEPT_COMPARISON_GROUP, "control group"),
            TermGloss(CONCEPT_UNIT, "patient"),
            TermGloss(CONCEPT_BASELINE, "baseline visit"),
            TermGloss(CONCEPT_FOLLOW_UP, "follow-up visit"),
            TermGloss(CONCEPT_OUTCOME, "clinical or utilization outcome"),
        ),
    ),
    Domain(
        id="social_programs",
        label="Social programs",
        search_terms=(
            "social program evaluation",
            "human services",
            "benefit take-up",
            "antipoverty program",
        ),
        outcomes=(
            OutcomeSuggestion(label="Housing stability at 12 months"),
            OutcomeSuggestion(label="Food security score"),
            OutcomeSuggestion(
                label="Benefit enrollment among eligible households",
                caveat=(
                    "Outreach can change who is found to be eligible, so "
                    "the denominator of this rate may move along with the "
                    "numerator."
                ),
            ),
            OutcomeSuggestion(
                label="Caseworker-recorded service receipt",
                caveat=(
                    "Administrative records capture services a caseworker "
                    "logged, and logging practice can change when a program "
                    "is introduced."
                ),
            ),
        ),
        terminology=(
            TermGloss(CONCEPT_TREATMENT_GROUP, "program participants"),
            TermGloss(CONCEPT_COMPARISON_GROUP, "comparison group"),
            TermGloss(CONCEPT_UNIT, "household"),
            TermGloss(CONCEPT_BASELINE, "intake"),
            TermGloss(CONCEPT_FOLLOW_UP, "follow-up survey"),
            TermGloss(CONCEPT_OUTCOME, "participant outcome"),
        ),
    ),
    Domain(
        id="education",
        label="Education",
        search_terms=(
            "education intervention",
            "school program",
            "student achievement",
            "classroom practice",
        ),
        outcomes=(
            OutcomeSuggestion(label="Attendance rate"),
            OutcomeSuggestion(label="On-time grade progression"),
            OutcomeSuggestion(
                label="Standardized test score",
                caveat=(
                    "Scores respond to instructional emphasis on tested "
                    "content, so a gain can reflect a shift in what was "
                    "taught alongside a change in what was learned."
                ),
            ),
            OutcomeSuggestion(
                label="Disciplinary referrals",
                caveat=(
                    "Referrals record staff decisions as well as student "
                    "behavior, and referral rates differ across schools for "
                    "students doing the same things."
                ),
            ),
        ),
        terminology=(
            TermGloss(CONCEPT_TREATMENT_GROUP, "treatment schools or classrooms"),
            TermGloss(CONCEPT_COMPARISON_GROUP, "business-as-usual condition"),
            TermGloss(CONCEPT_UNIT, "student"),
            TermGloss(CONCEPT_BASELINE, "pretest"),
            TermGloss(CONCEPT_FOLLOW_UP, "posttest"),
            TermGloss(CONCEPT_OUTCOME, "student outcome"),
        ),
    ),
    Domain(
        id="workforce",
        label="Workforce",
        search_terms=(
            "workforce development",
            "job training program",
            "employment program",
            "labor market outcomes",
        ),
        outcomes=(
            OutcomeSuggestion(label="Credential or certificate completion"),
            OutcomeSuggestion(label="Job placement within 6 months of exit"),
            OutcomeSuggestion(
                label="Quarterly earnings after exit",
                caveat=(
                    "State wage records miss self-employment, cash work, "
                    "federal and military employment, and jobs taken in "
                    "another state."
                ),
            ),
            OutcomeSuggestion(
                label="Job retention at 12 months",
                caveat=(
                    "Retention is measured only for those who were placed, "
                    "so it describes a group the program itself selected."
                ),
            ),
        ),
        terminology=(
            TermGloss(CONCEPT_TREATMENT_GROUP, "program enrollees"),
            TermGloss(CONCEPT_COMPARISON_GROUP, "comparison cohort"),
            TermGloss(CONCEPT_UNIT, "jobseeker"),
            TermGloss(CONCEPT_BASELINE, "intake"),
            TermGloss(CONCEPT_FOLLOW_UP, "post-program follow-up"),
            TermGloss(CONCEPT_OUTCOME, "employment or earnings outcome"),
        ),
    ),
    Domain(
        id="criminal_justice",
        label="Criminal justice & reentry",
        search_terms=(
            "reentry program",
            "community supervision",
            "criminal justice intervention",
            "post-release outcomes",
        ),
        outcomes=(
            OutcomeSuggestion(label="Stable housing at 6 months post-release"),
            OutcomeSuggestion(label="Employment after release"),
            OutcomeSuggestion(
                label="Rearrest within 12 months",
                caveat=(
                    "Rearrest counts police contact, which varies with "
                    "patrol and supervision intensity where a person lives, "
                    "so it measures system activity alongside behavior. "
                    "Rearrest, reconviction, and reincarceration are three "
                    "different outcomes and often point different ways."
                ),
            ),
            OutcomeSuggestion(
                label="Reincarceration within 12 months",
                caveat=(
                    "Returns to custody include technical violations of "
                    "supervision conditions as well as new offenses, and "
                    "the two respond to very different things."
                ),
            ),
        ),
        terminology=(
            TermGloss(CONCEPT_TREATMENT_GROUP, "program participants"),
            TermGloss(CONCEPT_COMPARISON_GROUP, "matched comparison group"),
            TermGloss(CONCEPT_UNIT, "participant"),
            TermGloss(CONCEPT_BASELINE, "pre-release or intake"),
            TermGloss(CONCEPT_FOLLOW_UP, "post-release follow-up window"),
            TermGloss(CONCEPT_OUTCOME, "reentry outcome"),
        ),
    ),
    Domain(
        id="digital",
        label="Digital interventions",
        search_terms=(
            "digital intervention",
            "mobile health intervention",
            "online platform",
            "user engagement",
        ),
        outcomes=(
            OutcomeSuggestion(label="Feature activation rate"),
            OutcomeSuggestion(label="Task completion rate"),
            OutcomeSuggestion(
                label="7-day retention",
                caveat=(
                    "The length of the retention window is an analyst's "
                    "choice, and different windows can support different "
                    "conclusions from the same data."
                ),
            ),
            OutcomeSuggestion(
                label="In-app satisfaction rating",
                caveat=(
                    "In-app prompts reach the users who are still using the "
                    "product, so responses come from those a change did not "
                    "drive away."
                ),
            ),
        ),
        terminology=(
            TermGloss(CONCEPT_TREATMENT_GROUP, "treatment variant"),
            TermGloss(CONCEPT_COMPARISON_GROUP, "control condition or holdout"),
            TermGloss(CONCEPT_UNIT, "user"),
            TermGloss(CONCEPT_BASELINE, "pre-launch period"),
            TermGloss(CONCEPT_FOLLOW_UP, "post-launch period"),
            TermGloss(CONCEPT_OUTCOME, "engagement or conversion outcome"),
        ),
    ),
    Domain(
        id=DOMAIN_OTHER,
        label="Other",
        # No seed terms: a search from here is built from the
        # researcher's own question alone, which is what they would get
        # if this stage did not exist. Selecting Other should never leave
        # someone worse off than not choosing.
        search_terms=(),
        outcomes=(),
        terminology=_identity_terminology(),
    ),
)

DOMAIN_IDS: tuple[str, ...] = tuple(domain.id for domain in DOMAINS)

_DOMAIN_BY_ID: dict[str, Domain] = {domain.id: domain for domain in DOMAINS}


def get_domain(domain_id: str) -> Domain:
    """
    Return one domain by id.

    Raises on an unknown id rather than falling back to Other, so a typo
    surfaces instead of silently producing the least tailored result.
    """
    if domain_id not in _DOMAIN_BY_ID:
        raise ValueError(
            f"'{domain_id}' is not a known domain. Known domains: "
            f"{', '.join(DOMAIN_IDS)}."
        )

    return _DOMAIN_BY_ID[domain_id]


def build_search_query(question_terms: str, domain_id: str) -> str:
    """
    Compose the literature-search query a domain suggests.

    The researcher's own words come first and are never dropped; the
    domain's terms are appended. The result is meant to be shown in an
    editable field rather than sent directly, so that what the domain
    added is visible and removable rather than applied behind the box.

    Returns the question alone when the domain seeds no terms, which is
    what Other does.
    """
    question = question_terms.strip()

    if not question:
        raise ValueError(
            "A search query needs the researcher's own terms; the domain "
            "only adds to them."
        )

    domain = get_domain(domain_id)

    if not domain.search_terms:
        return question

    return " ".join((question, *domain.search_terms))
