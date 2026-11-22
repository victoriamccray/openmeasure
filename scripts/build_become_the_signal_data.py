"""
Build the measured data behind the Become the Signal experience.

    python scripts/build_become_the_signal_data.py

Writes experiences/become-the-signal/data.json. The output is committed,
so the experience runs as a static page with no Python behind it.

The neuroanatomy figure's facts (systems, electrode site, and what each
modality reaches) come from modules/signal_pipeline/core/neuroanatomy.py.

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

import dataclasses
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
from modules.signal_pipeline.core import filtering, intervention, neuroanatomy, pipeline, power_explainer, resting_state, source_lesson  # noqa: E402

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


# Slider steps for the "Why the Math Works" explainer. Each step's
# numbers are computed by core/power_explainer.py, so the page only draws.
EXPLAINER_AMPLITUDES_UV = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
EXPLAINER_ALPHA_UV = (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0)


def _spectrum_dict(spectrum) -> dict:
    keep = [i for i, f in enumerate(spectrum.freqs_hz) if f <= power_explainer.DISPLAY_MAX_HZ]
    return {
        "freqs_hz": [spectrum.freqs_hz[i] for i in keep],
        "density": _round([spectrum.density[i] for i in keep], 4),
        "alpha_power": round(spectrum.alpha_power, 4),
        "total_power": round(spectrum.total_power, 4),
        "tapered_mean_square": round(spectrum.tapered_mean_square, 4),
        "mean_square": round(spectrum.mean_square, 4),
    }


def _explainer(signal: dict) -> dict:
    square = []
    for amplitude in EXPLAINER_AMPLITUDES_UV:
        r = power_explainer.mean_and_mean_square(amplitude)
        square.append({
            "amplitude": amplitude,
            "mean": round(r.mean, 6),
            "mean_square": round(r.mean_square, 4),
            "values": _round(r.values[::5], 3),
        })
    mixture = []
    for amplitude in EXPLAINER_ALPHA_UV:
        r = power_explainer.mixture_spectrum(amplitude)
        mixture.append({"alpha_amplitude": amplitude, "values": _round(r.values[::2], 2), "spectrum": _spectrum_dict(r.spectrum)})
    rate = intervention.SAMPLING_RATE_HZ
    recorded = {}
    for condition, values in signal.items():
        window = values[: int(rate)]
        recorded[condition] = {
            "values": _round(window[::2], 2),
            "spectrum": _spectrum_dict(power_explainer.window_spectrum(window, rate)),
        }
    first = signal["eyes_closed"][: int(rate)]
    w = power_explainer.welch_steps(first, rate)
    keep = [i for i, f in enumerate(w.freqs_hz) if f <= power_explainer.DISPLAY_MAX_HZ]
    welch = {
        "steps": [dataclasses.asdict(s) for s in power_explainer.WELCH_STEPS],
        "centered": _round(w.centered[::2], 2),
        "taper": _round(w.taper[::2], 4),
        "tapered": _round(w.tapered[::2], 2),
        "freqs_hz": [w.freqs_hz[i] for i in keep],
        "fft_magnitude": _round([w.fft_magnitude[i] for i in keep], 3),
        "density": _round([w.density[i] for i in keep], 4),
        "in_band": [w.in_band[i] for i in keep],
        "alpha_power": round(w.alpha_power, 4),
    }
    return {
        "steps": [dataclasses.asdict(s) for s in power_explainer.STEPS],
        "welch": welch,
        "symbols": dict(power_explainer.SYMBOLS),
        "square": {"frequency_hz": 2.0, "seconds": 1.0, "levels": square},
        "mixture": {
            "alpha_hz": power_explainer.MIXTURE_ALPHA_HZ,
            "fixed_components": [list(c) for c in power_explainer.MIXTURE_COMPONENTS_HZ_UV],
            "noise_sd_uv": power_explainer.MIXTURE_NOISE_SD_UV,
            "levels": mixture,
        },
        "recorded": recorded,
        "alpha_band_hz": list(intervention.ALPHA_BAND_HZ),
    }


# Two seconds from the middle of the eyes-closed excerpt, away from the
# recording's edges, for the raw versus filtered comparison.
FILTER_SHOWN_SECONDS = (4.0, 6.0)


def _filter_compare(signal: dict) -> dict:
    rate = intervention.SAMPLING_RATE_HZ
    result = filtering.band_pass(signal["eyes_closed"], rate)
    a, b = (int(s * rate) for s in FILTER_SHOWN_SECONDS)
    return {
        "shown_seconds": list(FILTER_SHOWN_SECONDS),
        "raw": _round(result.raw[a:b:2], 2),
        "filtered": _round(result.filtered[a:b:2], 2),
        "band_hz": [result.low_hz, result.high_hz],
        "order": result.order,
        "kept_variance_share": round(result.kept_variance_share, 4),
        "gain": {str(int(f)): round(filtering.amplitude_gain(f, rate), 4) for f in (4.0, 8.0, 10.0, 13.0, 20.0)},
        "causal_delay_s": round(filtering.causal_burst_delay(rate), 3),
        "lesson": {k: (list(v) if isinstance(v, tuple) else v) for k, v in filtering.LESSON.items()},
    }


def _sources() -> dict:
    result = source_lesson.ambiguity()
    return {
        **dataclasses.asdict(result),
        "sources": list(source_lesson.SOURCES),
        "electrodes": list(source_lesson.ELECTRODES),
        "lesson": {k: (list(v) if isinstance(v, tuple) else v) for k, v in source_lesson.LESSON.items()},
        "symbols": dict(source_lesson.SYMBOLS),
    }


def _lesson_references() -> list[dict]:
    refs = {**filtering.REFERENCES, **source_lesson.REFERENCES}
    return [{"key": k, "text": t, "url": u} for k, (t, u) in refs.items()]


def _resting_state(baseline: dict) -> dict:
    """The study, measures, ranges, and dichotomies, with this recording's own numbers."""
    closed = baseline["alpha_power_eyes_closed"]
    opened = baseline["alpha_power_eyes_open"]
    n_trials = len(closed) + len(opened)
    n_correct = round(baseline["accuracy"] * n_trials)
    react = resting_state.reactivity(closed, opened)
    odds = resting_state.chance(n_correct, n_trials)
    return {
        "study": dataclasses.asdict(resting_state.STUDY),
        "ageing": dataclasses.asdict(resting_state.AGEING_NOTE),
        "measures": [dataclasses.asdict(m) for m in resting_state.MEASURES],
        "bands": [dataclasses.asdict(b) for b in resting_state.BANDS],
        "band_caveat": resting_state.BAND_CAVEAT,
        "band_refs": list(resting_state.BAND_REFS),
        "axis_poles": {k: list(v) for k, v in resting_state.AXIS_POLES.items()},
        "dichotomies": [dataclasses.asdict(d) for d in resting_state.DICHOTOMIES],
        "references": [dataclasses.asdict(r) for r in resting_state.REFERENCES],
        "reactivity": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in dataclasses.asdict(react).items()},
        "chance": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in dataclasses.asdict(odds).items()},
        "threshold": {"value": round(react.midpoint, 4), "text": resting_state.explain_threshold(react).text},
        "source_labels": dict(resting_state.SOURCE_LABEL),
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

    baseline = _classification(signal["eyes_closed"], signal["eyes_open"], rate)

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
        "baseline": baseline,
        # Barry et al. (2007) and what the numbers mean, for the study room
        # and the measures panel at the Signals stage.
        "resting_state": _resting_state(baseline),
        # "Why the Math Works": mean versus mean square, a mixture's
        # spectrum, and the first recorded second's spectrum.
        "explainer": _explainer(signal),
        # Raw versus filtered at the Processing stage, and the source
        # localization teaching model at the Sensors stage.
        "filter_compare": _filter_compare(signal),
        "sources": _sources(),
        "lesson_references": _lesson_references(),
        "noise": noise,
        "noise_seeds": NOISE_SEEDS,
        "downsample": downsample,
        # Which brain system the recording comes from, which electrode
        # picked it up, and what other modalities reach, for the anatomy
        # figure at the Signals and Sensors stages.
        "neuroanatomy": dataclasses.asdict(neuroanatomy.atlas()),
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
