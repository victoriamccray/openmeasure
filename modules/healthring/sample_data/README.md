# No bundled sample data here

Every other OpenMeasure module ships a small synthetic or de-identified CSV in
its own `sample_data/` so a visitor can try the workflow without their own
data. This module does not, and the reason has changed.

## What this file used to say, and why it was wrong

It said HealthRing was "published under terms that this repository is not
redistributing under", and that even a small derived excerpt would be a
redistribution that could not be checked in.

That was a mistake. The Zenodo record for `RingDatasetV2.1` (record 18426864)
carries the license id `cc-by-4.0`, verified against the record's own API
response. CC BY 4.0 permits redistribution of the dataset and of derived works,
with attribution. Nothing about the licence prevented a derived subset from
being committed.

## What is true

The archive stays out of the repository because of its size, not its terms.
`RingDatasetV2.1_submission.zip` is 2,582,075,992 bytes against Streamlit's
200 MB browser-upload cap, so a hosted copy of the app cannot receive it at all
and `.gitignore` excludes it at the repository root.

A derived subset is a different question and is permitted. See
`scripts/build_healthring_journey_data.py`, which reads a local archive once and
writes the per-window summary columns the worked example's analysis actually
uses, alongside a provenance file recording the source DOI, the licence, the
attribution, and exactly which columns were kept, derived and dropped. That
subset is not the HealthRing dataset and is labelled as a derived subset
wherever it appears.

`modules/healthring/tests/` needs neither: its tests build small,
hand-calculable synthetic frames directly in the test file.
