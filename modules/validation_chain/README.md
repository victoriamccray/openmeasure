# Cross-Analysis Implications Module

**Category:** Cross-cutting validation

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

Future versions may connect additional findings across these stages, including measurement quality, statistical assumptions, fairness, robustness, and evidence synthesis.

## Non-goals

This module does not:

- Produce an overall validation score
- Declare a study valid or invalid
- Assume analyses retained the same participants
- Define a universal acceptable exclusion threshold
- Automatically combine results from different datasets
