"""
Regression guard: modules/<name>/core/ stays framework-independent.

CLAUDE.md states the rule directly: "modules/<name>/core/: pure functions
only. No streamlit import, no I/O." An audit across all nine modules
found zero violations today, but nothing enforced it -- a future core/
file could import streamlit to render something quick, and nothing would
fail until a reviewer noticed. This is what protects a later migration
away from Streamlit: if core/ never depends on it, replacing the
presentation layer never touches the analytical logic underneath it.

Same style as shared/tests/test_no_decorative_icons.py: read every file's
source text and check for the forbidden import, rather than trying to
exercise the code.

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULES_DIR = ROOT / "modules"

FORBIDDEN_IMPORTS = ("import streamlit", "from streamlit")


def _core_files() -> list[Path]:
    return sorted(MODULES_DIR.glob("*/core/*.py"))


class TestCoreHasNoStreamlitImport(unittest.TestCase):
    def test_at_least_one_core_file_is_discovered(self):
        # Pins the discovery glob itself: if modules/ is ever restructured
        # so */core/*.py matches nothing, this test must fail loudly
        # rather than the suite below silently passing on zero files.
        self.assertTrue(_core_files(), "No modules/*/core/*.py files were discovered.")

    def test_no_core_file_imports_streamlit(self):
        for path in _core_files():
            if path.name == "__init__.py":
                continue

            source = path.read_text(encoding="utf-8")

            for forbidden in FORBIDDEN_IMPORTS:
                with self.subTest(file=str(path.relative_to(ROOT)), forbidden=forbidden):
                    self.assertNotIn(
                        forbidden,
                        source,
                        f"{path.relative_to(ROOT)} imports streamlit. "
                        "modules/<name>/core/ must stay pure Python with "
                        "no framework dependency (see CLAUDE.md).",
                    )


if __name__ == "__main__":
    unittest.main()
