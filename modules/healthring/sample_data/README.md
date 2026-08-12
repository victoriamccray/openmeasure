# No bundled sample data

Every other OpenMeasure module ships a small synthetic or de-identified CSV here so a visitor can try the workflow without their own data.

This module does not, on purpose. HealthRing (`RingDatasetV2.1_submission.zip`, Zenodo record 18426864) is published under terms that this repository is not redistributing under -- see `shared/datasets.py`'s `healthring` entry for the citation and access terms. Even a small derived excerpt would still be a redistribution of that dataset, so none is checked in here.

To run the worked example, obtain your own local copy of the archive (see `pages/HealthRing_Worked_Example.py`, which reads it by path) and keep it out of version control, the same way `.gitignore` already excludes it at the repository root.

`modules/healthring/tests/` does not need real HealthRing data either: its tests build small, hand-calculable synthetic frames directly in the test file.
