"""
Unit tests for shared/upload.py

Run with: pytest shared/tests/ -v

render_data_profile is Streamlit UI code (like shared/report.py's
render_lifecycle_tracker), so its actual rendering is exercised through
AppTest.from_string against a minimal script, the same approach
shared/tests/test_journey_stages.py uses for StageTracker.
"""

from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest

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


if __name__ == "__main__":
    unittest.main()
