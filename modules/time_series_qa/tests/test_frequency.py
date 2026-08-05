"""
Unit tests for core/frequency.py

Run with: pytest modules/time_series_qa/tests/ -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from core import frequency as fq  # noqa: E402


def index(values):
    return pd.DatetimeIndex(pd.to_datetime(pd.Series(values)))


class TestPandasRecognizedFrequency(unittest.TestCase):
    def test_clean_daily_series(self):
        result = fq.infer_frequency(pd.date_range("2020-01-01", periods=5))

        self.assertEqual(result.source, fq.SOURCE_PANDAS)
        self.assertEqual(result.modal_interval, pd.Timedelta("1D"))
        self.assertEqual(result.modal_interval_share, 1.0)
        self.assertFalse(result.is_calendar_anchored)

    def test_month_end_series_is_calendar_anchored(self):
        result = fq.infer_frequency(
            pd.date_range("2020-01-31", periods=5, freq=pd.tseries.offsets.MonthEnd())
        )

        self.assertEqual(result.source, fq.SOURCE_PANDAS)
        self.assertTrue(result.is_calendar_anchored)

    def test_business_day_series_is_flagged(self):
        result = fq.infer_frequency(pd.bdate_range("2020-01-06", periods=10))

        self.assertTrue(result.is_business_day)


class TestModalFallback(unittest.TestCase):
    def test_gappy_daily_recovers_daily_interval(self):
        # Regression: pd.infer_freq returns None for a gappy series, which
        # is exactly when gap detection matters, so the modal fallback must
        # carry it. Three intervals, two of them one day: share is 2/3.
        result = fq.infer_frequency(
            index(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"])
        )

        self.assertEqual(result.source, fq.SOURCE_MODAL)
        self.assertEqual(result.modal_interval, pd.Timedelta("1D"))
        self.assertAlmostEqual(result.modal_interval_share, 2 / 3)

    def test_sub_interval_jitter_still_yields_a_clean_interval(self):
        # Regression: without snapping, every difference is unique and a
        # perfectly regular series looks event-driven.
        result = fq.infer_frequency(
            index(
                [
                    "2020-01-01 09:00:01",
                    "2020-01-02 09:00:03",
                    "2020-01-03 09:00:00",
                    "2020-01-04 09:00:02",
                ]
            )
        )

        self.assertEqual(result.modal_interval, pd.Timedelta("1D"))
        self.assertIsNotNone(result.offset)

    def test_snapping_works_at_every_granularity(self):
        # The snap ladder needs sub-multiple rungs. One-second data with a
        # couple of milliseconds of jitter has a 50 ms budget, which falls
        # between "s" and "ms"; without a 10 ms rung the jitter is not
        # absorbed and regular data is misread as event-driven.
        cases = {
            pd.Timedelta("1h"): [
                "2024-01-01 00:00:00",
                "2024-01-01 01:00:02",
                "2024-01-01 02:00:01",
                "2024-01-01 03:00:00",
            ],
            pd.Timedelta("5min"): [
                "2024-01-01 00:00:00.000",
                "2024-01-01 00:05:00.400",
                "2024-01-01 00:10:00.000",
                "2024-01-01 00:15:00.200",
            ],
            pd.Timedelta("1s"): [
                "2024-01-01 00:00:00.000",
                "2024-01-01 00:00:01.002",
                "2024-01-01 00:00:02.000",
                "2024-01-01 00:00:03.001",
            ],
        }

        for expected, values in cases.items():
            with self.subTest(interval=str(expected)):
                result = fq.infer_frequency(index(values))

                self.assertEqual(result.modal_interval, expected)
                self.assertIsNotNone(result.offset)

    def test_every_snap_unit_is_a_valid_pandas_alias(self):
        series = pd.Series([pd.Timedelta("1D") + pd.Timedelta("3s")])

        for alias, _ in fq._SNAP_UNITS:
            with self.subTest(alias=alias):
                series.dt.round(alias)

    def test_timezone_aware_daily_is_regular_across_dst(self):
        # A daily series spans 23 absolute hours across spring-forward.
        # Measured on the wall clock it is perfectly regular.
        result = fq.infer_frequency(
            pd.date_range(
                "2021-03-12", periods=5, freq="D", tz="America/New_York"
            )
        )

        self.assertEqual(result.modal_interval_share, 1.0)


class TestCalendarPromotion(unittest.TestCase):
    def test_mid_month_monthly_is_promoted(self):
        # Regression: pd.infer_freq returns None for regular mid-month
        # monthly data, and its 28-31 day spacing must not be treated as
        # an irregular fixed interval.
        result = fq.infer_frequency(
            index(
                [
                    "2020-01-15",
                    "2020-02-15",
                    "2020-03-15",
                    "2020-04-15",
                    "2020-05-15",
                ]
            )
        )

        self.assertEqual(result.source, fq.SOURCE_CALENDAR)
        self.assertTrue(result.is_calendar_anchored)
        self.assertIsNotNone(result.offset)

    def test_monthly_is_accepted_despite_low_modal_share(self):
        # Monthly spacing varies by construction, so the regularity gate
        # must not be applied before calendar promotion.
        result = fq.infer_frequency(
            index(["2020-01-15", "2020-02-15", "2020-03-15", "2020-04-15"])
        )

        self.assertLess(result.modal_interval_share, 0.75)
        self.assertIsNotNone(result.offset)

    def test_quarterly_is_promoted(self):
        result = fq.infer_frequency(
            pd.date_range(
                "2020-01-15", periods=6, freq=pd.DateOffset(months=3)
            )
        )

        self.assertEqual(result.source, fq.SOURCE_CALENDAR)
        self.assertTrue(result.is_calendar_anchored)


class TestDegenerateInput(unittest.TestCase):
    def test_all_timestamps_identical(self):
        result = fq.infer_frequency(index(["2020-01-01"] * 4))

        self.assertEqual(result.source, fq.SOURCE_ALL_IDENTICAL)
        self.assertIsNone(result.offset)
        self.assertIsNone(result.modal_interval)

    def test_modal_interval_is_never_zero(self):
        # A zero-length grid step would make date_range unbounded.
        result = fq.infer_frequency(
            index(["2020-01-01"] * 3 + ["2020-01-02", "2020-01-03"])
        )

        self.assertNotEqual(result.modal_interval, pd.Timedelta(0))
        self.assertEqual(result.modal_interval, pd.Timedelta("1D"))

    def test_two_distinct_timestamps(self):
        # pd.infer_freq raises "Need at least 3 dates" here; that must be
        # caught rather than escaping to the caller.
        result = fq.infer_frequency(index(["2020-01-01", "2020-01-02"]))

        self.assertEqual(result.source, fq.SOURCE_SINGLE_INTERVAL)
        self.assertIsNone(result.offset)
        self.assertIsNone(result.modal_interval_share)

    def test_single_timestamp(self):
        result = fq.infer_frequency(index(["2020-01-01"]))

        self.assertEqual(result.source, fq.SOURCE_ALL_IDENTICAL)
        self.assertIsNone(result.offset)

    def test_empty_index(self):
        result = fq.infer_frequency(pd.DatetimeIndex([]))

        self.assertEqual(result.source, fq.SOURCE_INSUFFICIENT)
        self.assertIsNone(result.offset)

    def test_irregular_series_is_not_defensible(self):
        # Every interval differs, so the modal share is 1/n_intervals and
        # no expected schedule can be justified.
        result = fq.infer_frequency(
            index(
                [
                    "2020-01-01",
                    "2020-01-05",
                    "2020-01-11",
                    "2020-01-20",
                    "2020-02-03",
                ]
            )
        )

        self.assertEqual(result.source, fq.SOURCE_NOT_DEFENSIBLE)
        self.assertIsNone(result.offset)
        self.assertAlmostEqual(result.modal_interval_share, 0.25)

    def test_every_outcome_explains_itself(self):
        for timestamps in [
            index(["2020-01-01"]),
            index(["2020-01-01", "2020-01-02"]),
            pd.date_range("2020-01-01", periods=5),
            index(["2020-01-01", "2020-01-05", "2020-01-11", "2020-01-20"]),
        ]:
            with self.subTest(n=len(timestamps)):
                self.assertTrue(fq.infer_frequency(timestamps).reason)


if __name__ == "__main__":
    unittest.main()
