"""
Build the measured data behind the Become the Signal experience.

    python scripts/build_become_the_signal_data.py

Writes experiences/become-the-signal/data.json. The output is committed,
so the experience runs as a static page with no Python behind it.

Every number the experience shows comes from this file, and every number
in this file is computed by modules/signal_pipeline/core from the bundled
recorded EEG excerpt (OpenNeuro ds005420, sub-5, channel O1, CC0). Nothing
here is an illustrative rating. The stage names are taken from
pipeline.STAGES, so the experience's levels are the module's stages.

The noise and downsampling grids are the ones the Multimodal Signal
Convergence page offers, with the same seeds, so a learner who meets a
number here can reproduce it on that page.

shared/tests/test_become_the_signal.py regenerates this data and compares
it with the committed file, so a change to the core functions that moves
a number fails a test until this script is rerun.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "modules" / "signal_pipeline"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The full package path, not a bare "core": several modules each have a
# top-level package of that name, and the shared tests import this file.
from modules.signal_pipeline.core import intervention, pipeline  # noqa: E402

SNIPPET = MODULE_DIR / "sample_data" / "ds005420_sub-5_O1_eyes_snippet.csv"
OUTPUT = ROOT / "experiences" / "become-the-signal" / "data.json"

# The Multimodal Signal Convergence page's own controls.
NOISE_SD_UV = tuple(float(sd) for sd in range(0, 301, 10))
NOISE_SEEDS = {"eyes_closed": 0, "eyes_open": 1}
DOWNSAMPLE_FACTORS = (1, 2, 5, 10, 20, 40)

# How much of each condition the experience draws. Every point is a
# recorded sample; only the count drawn is reduced, by keeping every
# DISPLAY_STRIDE-th one, and the file says so.
DISPLAY_SECONDS = 2.0
DISPLAY_STRIDE = 2

# Noise levels whose waveform is drawn. Accuracy is exported for every
# level in NOISE_SD_UV; the experience offers these steps.
NOISE_WAVEFORM_SD_UV = (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0)

SOURCE = {
    "dataset": "OpenNeuro ds005420, version 1.0.0",
    "doi": "https://doi.org/10.18112/openneuro.ds005420.v1.0.0",
    "licence": "CC0",
    "subject": "sub-5",
    "channel": "O1",
    "seconds_per_condition": 15,
    "sampling_rate_hz": intervention.SAMPLING_RATE_HZ,
}


def _round(values, digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in values]


def _classification(closed: np.ndarray, opened: np.ndarray, rate: float) -> dict:
    try:
        result = intervention.classify_eyes_state(closed, opened, rate)
    except ValueError as error:
        return {"accuracy": None, "failure": str(error)}

    return {
        "accuracy": round(result.accuracy, 4),
        "threshold": round(result.threshold, 4),
        "alpha_power_eyes_closed": _round(result.alpha_power_eyes_closed),
        "alpha_power_eyes_open": _round(result.alpha_power_eyes_open),
    }


def build() -> dict:
    frame = pd.read_csv(SNIPPET)
    signal = {
        condition: frame.loc[frame["condition"] == condition, "voltage_uv"].to_numpy()
        for condition in ("eyes_closed", "eyes_open")
    }
    rate = intervention.SAMPLING_RATE_HZ

    n_shown = int(rate * DISPLAY_SECONDS)
    waveform = {
        condition: _round(values[:n_shown:DISPLAY_STRIDE], 2)
        for condition, values in signal.items()
    }

    def shown(values: np.ndarray, sample_rate: float) -> list[float]:
        # The same span of time at any rate, so a degraded trace lines up
        # with the original. Only the full-rate recording is thinned for
        # drawing; a degraded one keeps every sample it has left.
        count = int(sample_rate * DISPLAY_SECONDS)
        stride = DISPLAY_STRIDE if sample_rate >= rate else 1
        return _round(values[:count:stride], 2)

    noise = []
    for sd in NOISE_SD_UV:
        closed = intervention.apply_noise(signal["eyes_closed"], sd, seed=NOISE_SEEDS["eyes_closed"])
        opened = intervention.apply_noise(signal["eyes_open"], sd, seed=NOISE_SEEDS["eyes_open"])
        entry = {"sd_uv": sd, **_classification(closed, opened, rate)}
        if sd in NOISE_WAVEFORM_SD_UV:
            entry["waveform"] = {"eyes_closed": shown(closed, rate), "eyes_open": shown(opened, rate)}
        noise.append(entry)

    downsample = []
    for factor in DOWNSAMPLE_FACTORS:
        closed, new_rate = intervention.apply_temporal_degradation(signal["eyes_closed"], rate, factor)
        opened, _ = intervention.apply_temporal_degradation(signal["eyes_open"], rate, factor)
        downsample.append(
            {
                "factor": factor,
                "effective_rate_hz": round(new_rate, 4),
                **_classification(closed, opened, new_rate),
                "waveform": {
                    "eyes_closed": shown(closed, new_rate),
                    "eyes_open": shown(opened, new_rate),
                },
            }
        )

    example_metadata = {
        "device_serial_number": "SN-48213",
        "recording_timestamp": "2024-03-11T09:42:00Z",
        "subject_session_code": "sub-5_ses-01",
        "task": "eyes_closed",
    }

    return {
        "generated_by": "scripts/build_become_the_signal_data.py",
        "stages": list(pipeline.STAGES),
        "source": SOURCE,
        "alpha_band_hz": list(intervention.ALPHA_BAND_HZ),
        "window_seconds": intervention.WINDOW_SECONDS,
        "waveform": {
            "seconds": DISPLAY_SECONDS,
            "stride": DISPLAY_STRIDE,
            "note": (
                f"The first {DISPLAY_SECONDS:g} seconds of each condition, "
                f"keeping one sample in {DISPLAY_STRIDE}. Values are recorded "
                "microvolt readings."
            ),
            **waveform,
        },
        "baseline": _classification(signal["eyes_closed"], signal["eyes_open"], rate),
        "noise": noise,
        "noise_seeds": NOISE_SEEDS,
        "downsample": downsample,
        "metadata": {
            "illustrative_fields": example_metadata,
            "after_removal": dict(intervention.strip_metadata(example_metadata)),
            "note": (
                "Illustrative fields of the kind a recording and sharing "
                "pipeline attaches. The dataset's own metadata files are "
                "mostly blank."
            ),
        },
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
