"""
Regression guard: decorative emoji were replaced with a muted, consistent
Material-icon treatment (page icons via page_icon=":material/...:" and
expander icons via icon=":material/...:", rather than emoji embedded in
labels or table cells).

This does not ban every emoji character outright, since that would also
flag legitimate typographic characters used throughout the app, such as
the arrows in Impact Evaluation's method list or the math symbols in
Reliability's conventional-interpretation table. It names the specific
markers that were removed and fails if a page reintroduces one of them.

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "pages"

REMOVED_DECORATIVE_MARKERS = ("📖", "⚖️", "⚠️", "✅", "∑", ":bar_chart:")


class TestNoDecorativeEmojiRegression(unittest.TestCase):
    def test_no_page_reintroduces_a_removed_decorative_marker(self):
        pages = sorted(PAGES.glob("*.py"))
        self.assertTrue(pages, "No page scripts were discovered.")

        for path in pages:
            source = path.read_text(encoding="utf-8")

            for marker in REMOVED_DECORATIVE_MARKERS:
                with self.subTest(page=path.name, marker=marker):
                    self.assertNotIn(
                        marker,
                        source,
                        f"{path.name} contains '{marker}', a decorative "
                        "marker that was deliberately replaced with a "
                        "muted, consistent Material icon.",
                    )


if __name__ == "__main__":
    unittest.main()
