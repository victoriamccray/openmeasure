"""
No module uses a NumPy name that NumPy 2.0 removed.

Run with: pytest shared/tests/test_numpy_compatibility.py -v

requirements.txt asks for numpy>=1.24, so a deployment resolves to
whatever 2.x is current while a development machine may sit on 1.26.
Every name below still exists on 1.x and raises AttributeError on 2.x,
which makes this the one class of fault that cannot be reproduced
locally and cannot be caught by any test that runs on the older version.

`np.trapz` reached production this way: one line, in the stage that runs
a real EEG recording, on a page that had passed every test.

Checked as text rather than by importing, because the point is to fail
on the version that has the name rather than only on the version that
does not.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Removed in NumPy 2.0, with what replaced each one. Aliases only: a
# name that merely moved is not a hazard, because the old spelling still
# resolves.
REMOVED_NAMES = {
    "trapz": "trapezoid",
    "in1d": "isin",
    "alltrue": "all",
    "sometrue": "any",
    "product": "prod",
    "cumproduct": "cumprod",
    "round_": "round",
    "float_": "float64",
    "complex_": "complex128",
    "unicode_": "str_",
    "string_": "bytes_",
    "bool8": "bool_",
    "NaN": "nan",
    "Inf": "inf",
    "Infinity": "inf",
    "NINF": "-inf",
    "PINF": "inf",
    "NZERO": "-0.0",
    "PZERO": "0.0",
    "issctype": "issubdtype",
    "set_string_function": "no replacement",
    "get_array_wrap": "no replacement",
}

SEARCHED = ("modules", "shared", "pages", "scripts")


def _python_files():
    for directory in SEARCHED:
        root = ROOT / directory

        if not root.is_dir():
            continue

        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue

            # This file names every removed alias in order to look for
            # them, so it would report itself.
            if path.name == Path(__file__).name:
                continue

            yield path


class TestNoRemovedNumpyNames(unittest.TestCase):
    def test_the_search_actually_reaches_the_code(self):
        """So a pass cannot come from looking in the wrong place."""
        files = list(_python_files())

        self.assertGreater(len(files), 50)
        self.assertTrue(
            any("intervention.py" == path.name for path in files),
            "the file that hit this in production is not being checked",
        )

    def test_no_module_calls_a_name_numpy_2_removed(self):
        pattern = re.compile(
            r"\bnp\.(" + "|".join(re.escape(name) for name in REMOVED_NAMES) + r")\b"
        )

        for path in _python_files():
            source = path.read_text(encoding="utf-8")

            for match in pattern.finditer(source):
                name = match.group(1)
                line = source[: match.start()].count("\n") + 1

                # A getattr fallback is how the one real case is handled,
                # and reads as a use of the old name without being one.
                context = source[max(0, match.start() - 90):match.start()]

                if 'getattr(np, "' in context:
                    continue

                with self.subTest(file=path.name, name=name):
                    self.fail(
                        f"{path.relative_to(ROOT)}:{line} uses np.{name}, "
                        f"removed in NumPy 2.0. Use np.{REMOVED_NAMES[name]}, "
                        "or getattr(np, new, old) where both versions must "
                        "work. requirements.txt allows 2.x, so this raises "
                        "in deployment and not on a machine pinned lower."
                    )

    def test_the_one_real_case_is_handled_for_both_versions(self):
        from modules.signal_pipeline.core import intervention

        self.assertTrue(callable(intervention._trapezoid))

    def test_the_shim_computes_what_it_replaced(self):
        """
        Same answer either way, so the fallback is a rename and not a
        different integral.
        """
        import numpy as np

        from modules.signal_pipeline.core import intervention

        y = np.array([0.0, 1.0, 2.0, 3.0])
        x = np.array([0.0, 1.0, 2.0, 3.0])

        self.assertAlmostEqual(
            float(intervention._trapezoid(y, x)), 4.5, places=9
        )


if __name__ == "__main__":
    unittest.main()
