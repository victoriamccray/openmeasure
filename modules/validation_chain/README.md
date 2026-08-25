# Cross-Analysis Implications Module

**Category:** Cross-Cutting Validation

> The folder is named `validation_chain` for historical reasons. The
> user-facing workflow is titled "Cross-Analysis Implications". "Chain" was
> deliberately dropped: a chain is intact or broken, which invites exactly
> the composite verdict this module refuses to produce.

## Purpose

The Cross-Analysis Implications module connects findings from separate OpenMeasure analyses to show how validation decisions in one part of a research workflow may affect interpretation elsewhere.

It does not produce an overall validity score or pass/fail judgment. Instead, it surfaces relationships, limitations, and differences in the data retained across analyses.

## Initial scope

Version 0.1 focuses on **data retention and exclusions across analyses**.

For each recorded analysis, OpenMeasure reports:

- Input observations
- Retained observations
- Excluded observations
- Reasons for exclusion
- Dataset and analysis provenance

Results are grouped by dataset and displayed separately rather than combined into a single retention percentage.

Each module page records its result through `shared/handoff.py`, which holds the data model and store. Records last for one browser session.

### Participants versus observations

Most analyses report retained participants. Row-expanding analyses cannot: the multi-select sensitivity analysis emits one observation per selection, so a participant who selected two categories becomes two observations, and no single "retained participants" figure exists.

Those analyses report **expanded observations** instead. They show the participants received, the expanded observation count, and `not applicable` where a retained count would go. They are left out of the retained-participant comparison and the retention chart, because an observation count and a participant count are different quantities.

## Validation framework

Cross-analysis implications contribute to the broader OpenMeasure workflow:

**Research Question → Data → Measurement → Analysis → Interpretation**

Beyond retention, this page also surfaces whatever a module already recorded in `HandoffEntry.primary_statistics` (e.g. Fairness's disparate impact and statistical parity difference), grouped alongside retention per dataset. These are shown as each analysis's own raw recorded numbers, never combined, ranked, or checked against a threshold this module does not itself define: a Reliability alpha and a Fairness disparity measure different things and have no shared scale, and a signal co-occurring with heavy exclusion on the same dataset does not establish that one caused the other.

Future versions may go further -- e.g. surfacing measurement-quality or evidence-synthesis findings the same way, once a module records them.

## Non-goals

This module does not:

- Produce an overall validation score
- Declare a study valid or invalid
- Assume analyses retained the same participants
- Define a universal acceptable exclusion threshold
- Automatically combine results from different datasets
