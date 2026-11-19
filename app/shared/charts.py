"""
Chart-spec builders shared across module pages.

Streamlit has no built-in multi-line time-series chart with an explicit
color scale, so this is a small Vega-Lite spec builder rather than a
statistical helper: it renders series a page's core module already
computed, and computes nothing itself. Extracted here once a second
page (fMRI QC, alongside HealthRing) needed the exact same shape, per
the project's own rule of extracting only what is actually duplicated.

No streamlit import: this returns a plain dict spec for the caller to
pass to st.vega_lite_chart, so it stays testable without a running app.
"""

from __future__ import annotations

import pandas as pd


def multiline_time_series_chart(
    time_values,
    series: dict[str, object],
    colors: dict[str, str],
    y_title: str,
    *,
    config: dict | None = None,
    x_title: str = "Time (seconds)",
    height: int = 220,
) -> dict:
    """
    A multi-line chart against a shared time (or index) axis.

    series maps a series name to an array-like of values sharing
    time_values as its x-axis; colors maps the same series names to a
    hex color, so the two dicts must share the same keys. A legend is
    always drawn (Vega-Lite's default for a color-encoded field), since
    two or more series should never rely on color alone to distinguish
    them.
    """

    frames = [
        pd.DataFrame({"t": time_values, "value": values, "series": name})
        for name, values in series.items()
    ]
    rows = pd.concat(frames, ignore_index=True).to_dict("records")

    domain = list(series.keys())
    range_ = [colors[name] for name in domain]

    spec: dict = {
        "data": {"values": rows},
        "mark": {"type": "line", "strokeWidth": 1.5},
        "encoding": {
            "x": {"field": "t", "type": "quantitative", "title": x_title},
            "y": {"field": "value", "type": "quantitative", "title": y_title},
            "color": {
                "field": "series",
                "type": "nominal",
                "scale": {"domain": domain, "range": range_},
                "legend": {"title": None, "orient": "top"},
            },
        },
        "width": "container",
        "height": height,
    }

    if config is not None:
        spec["config"] = config

    return spec
