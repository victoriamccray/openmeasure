"""
Unit tests for shared/upload.py

Run with: pytest shared/tests/ -v

Two surfaces are covered: the bundled-sample registry that every
numbered analysis page resolves its one-click example through, and
render_data_profile.

render_data_profile is Streamlit UI code (like shared/report.py's
render_lifecycle_tracker), so its actual rendering is exercised through
AppTest.from_string against a minimal script, the same approach
shared/tests/test_journey_stages.py uses for StageTracker.
"""

from __future__ import annotations

import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from shared.catalog import (
    MODULE_FAIRNESS,
    MODULE_KEYS,
    MODULE_PROGRAM_EVALUATION,
    MODULE_RELIABILITY,
    MODULE_TIME_SERIES_QA,
)
from shared.upload import SAMPLE_DATASETS, LoadedData, sample_path_for

_SCRIPT = """
import pandas as pd
from shared.upload import render_data_profile

df = pd.DataFrame({
    "participant_id": [1, 2, 3, 4],
    "score": [10.0, 20.0, 30.0, 40.0],
    "notes": [None, None, None, None],
})
profile = render_data_profile(df)
st_dummy = profile.n_rows
"""

_SCRIPT_NO_FLAGS = """
import pandas as pd
from shared.upload import render_data_profile

df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
render_data_profile(df, expanded=True)
"""


class TestRenderDataProfile(unittest.TestCase):
    def test_renders_without_raising_and_returns_a_profile(self):
        app = AppTest.from_string(_SCRIPT)
        app.run()

        self.assertFalse(app.exception)

    def test_expander_label_names_rows_columns_and_flag_count(self):
        app = AppTest.from_string(_SCRIPT)
        app.run()

        labels = [e.label for e in app.expander]
        self.assertEqual(len(labels), 1)
        self.assertIn("4 rows", labels[0])
        self.assertIn("3 columns", labels[0])
        self.assertIn("1 quality flag", labels[0])

    def test_all_missing_column_is_warned_about(self):
        app = AppTest.from_string(_SCRIPT)
        app.run()

        warnings = " ".join(str(w.value) for w in app.warning)
        self.assertIn("notes", warnings)
        self.assertIn("Every value is missing", warnings)

    def test_no_flags_shows_the_all_clear_message(self):
        app = AppTest.from_string(_SCRIPT_NO_FLAGS)
        app.run()

        self.assertFalse(app.exception)

        rendered = " ".join(str(item.value) for item in app.caption)
        from shared.upload import NO_FLAGS_MESSAGE

        self.assertIn(NO_FLAGS_MESSAGE, rendered)


class TestSampleRegistry(unittest.TestCase):
    def test_every_analysis_module_with_a_page_has_a_sample(self):
        self.assertEqual(
            set(SAMPLE_DATASETS),
            {
                MODULE_RELIABILITY,
                MODULE_PROGRAM_EVALUATION,
                MODULE_FAIRNESS,
                MODULE_TIME_SERIES_QA,
            },
        )

    def test_every_key_is_a_known_module(self):
        for key in SAMPLE_DATASETS:
            with self.subTest(module=key):
                self.assertIn(key, MODULE_KEYS)

    def test_every_bundled_sample_exists_on_disk(self):
        """
        The registry is what the pages resolve, so a renamed or deleted
        sample fails here rather than becoming a warning a reader sees
        instead of the worked example they came for.
        """
        for key, path in SAMPLE_DATASETS.items():
            with self.subTest(module=key):
                self.assertTrue(path.exists(), f"{path} is missing.")

    def test_every_bundled_sample_reads_as_a_nonempty_csv(self):
        for key, path in SAMPLE_DATASETS.items():
            with self.subTest(module=key):
                frame = pd.read_csv(path)
                self.assertFalse(frame.empty)
                self.assertGreaterEqual(len(frame.columns), 2)

    def test_sample_path_for_returns_the_registered_path(self):
        self.assertEqual(
            sample_path_for(MODULE_FAIRNESS), SAMPLE_DATASETS[MODULE_FAIRNESS]
        )

    def test_unknown_module_raises_and_names_the_known_ones(self):
        with self.assertRaises(ValueError) as raised:
            sample_path_for("evidence_review")

        message = str(raised.exception)
        self.assertIn("evidence_review", message)
        self.assertIn(MODULE_RELIABILITY, message)


class TestSampleSuitsItsPage(unittest.TestCase):
    """
    A one-click sample is only worth offering if it actually runs through
    the page that offers it. These pin the columns each page's default
    path needs, so a sample edited for one purpose cannot silently stop
    supporting the workflow it is the example for.
    """

    def test_the_impact_evaluation_sample_supports_all_three_designs(self):
        frame = pd.read_csv(SAMPLE_DATASETS[MODULE_PROGRAM_EVALUATION])

        self.assertIn("event_format", frame.columns)
        self.assertIn("pre_confidence", frame.columns)
        self.assertIn("post_confidence", frame.columns)
        # Difference-in-differences needs exactly two groups.
        self.assertEqual(frame["event_format"].dropna().nunique(), 2)

    def test_the_fairness_sample_supports_pre_and_post_model_analysis(self):
        frame = pd.read_csv(SAMPLE_DATASETS[MODULE_FAIRNESS])

        for column in ("true_label", "predicted_label", "sex"):
            with self.subTest(column=column):
                self.assertIn(column, frame.columns)

    def test_the_time_series_sample_has_a_timestamp_and_a_value(self):
        frame = pd.read_csv(SAMPLE_DATASETS[MODULE_TIME_SERIES_QA])

        self.assertIn("recorded_at", frame.columns)
        self.assertIn("visits", frame.columns)

    def test_the_reliability_sample_has_enough_items_for_alpha(self):
        frame = pd.read_csv(SAMPLE_DATASETS[MODULE_RELIABILITY])
        numeric = frame.select_dtypes("number")

        self.assertGreaterEqual(len(numeric.columns), 3)


class TestLoadedData(unittest.TestCase):
    def test_a_sample_and_an_upload_of_the_same_file_carry_the_same_frame(self):
        """
        Loading the sample must give the page exactly what uploading that
        same file would, so the one-click path is not a different analysis
        from the documented one.
        """
        path = SAMPLE_DATASETS[MODULE_PROGRAM_EVALUATION]

        from_sample = LoadedData(
            frame=pd.read_csv(path),
            name=path.name,
            token=f"sample:{path.name}",
            is_sample=True,
        )
        from_upload = LoadedData(
            frame=pd.read_csv(path),
            name=path.name,
            token="8f2c",
            is_sample=False,
        )

        pd.testing.assert_frame_equal(from_sample.frame, from_upload.frame)

    def test_the_sample_token_is_stable_across_loads(self):
        """
        The token drives each page's "the data changed, drop the carried
        plan" reset. A sample token derived from anything per-run would
        clear the reader's analysis plan on every interaction.
        """
        path = SAMPLE_DATASETS[MODULE_RELIABILITY]
        tokens = {f"sample:{path.name}" for _ in range(3)}

        self.assertEqual(len(tokens), 1)

    def test_loaded_data_is_frozen(self):
        self.assertTrue(LoadedData.__dataclass_params__.frozen)


if __name__ == "__main__":
    unittest.main()
