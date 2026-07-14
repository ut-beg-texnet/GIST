"""Unit tests for per-well disposal monthly BPD aggregation in gist_graphs."""

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gist_graphs import _aggregate_disposal_to_monthly_bpd


def _make_disposal_df(rows):
    """Build a minimal disposal dataframe accepted by the monthly aggregator."""
    return pd.DataFrame(rows)


class AggregateDisposalToMonthlyBpdTests(unittest.TestCase):
    """Verify monthly disposal bars use interval-weighted average daily rates."""

    def test_constant_rate_stays_constant_across_month_lengths(self):
        """A flat daily rate should not vary with 28-, 30-, or 31-day months."""
        rate = 40000.0
        df = _make_disposal_df(
            [
                {"subgraph": "Well A", "Days": 31, "Date": "2021-01-31", "BPD": rate},
                {"subgraph": "Well A", "Days": 59, "Date": "2021-02-28", "BPD": rate},
                {"subgraph": "Well A", "Days": 89, "Date": "2021-03-31", "BPD": rate},
            ]
        )

        monthly = _aggregate_disposal_to_monthly_bpd(df)

        self.assertEqual(len(monthly), 3)
        for _, row in monthly.iterrows():
            self.assertAlmostEqual(row["BPD"], rate, places=6)

    def test_boundary_crossing_interval_splits_volume_and_days(self):
        """An interval spanning two months should preserve the input daily rate in each month."""
        rate = 10000.0
        df = _make_disposal_df(
            [
                {"subgraph": "Well B", "Days": 100, "Date": "2021-01-05", "BPD": rate},
                {"subgraph": "Well B", "Days": 145, "Date": "2021-02-15", "BPD": rate},
            ]
        )

        monthly = _aggregate_disposal_to_monthly_bpd(df)

        jan = monthly.loc[monthly["Date"] == pd.Timestamp("2021-01-01")].iloc[0]
        feb = monthly.loc[monthly["Date"] == pd.Timestamp("2021-02-01")].iloc[0]
        self.assertAlmostEqual(jan["BPD"], rate, places=6)
        self.assertAlmostEqual(feb["BPD"], rate, places=6)

    def test_irregular_intervals_use_duration_weighted_average(self):
        """Two intervals in one month combine as sum(volume) / sum(days)."""
        df = _make_disposal_df(
            [
                {"subgraph": "Well C", "Days": 500, "Date": "2021-06-15", "BPD": 1000.0},
                {"subgraph": "Well C", "Days": 510, "Date": "2021-06-30", "BPD": 4000.0},
            ]
        )

        monthly = _aggregate_disposal_to_monthly_bpd(df)
        expected = ((1000.0 * 10.0) + (4000.0 * 10.0)) / (10.0 + 10.0)
        june = monthly.loc[monthly["Date"] == pd.Timestamp("2021-06-01")].iloc[0]

        self.assertAlmostEqual(june["BPD"], expected, places=6)

    def test_monthly_totals_are_not_returned(self):
        """Regression guard: output values should be daily rates, not monthly barrel sums."""
        rate = 30000.0
        days = 31.0
        df = _make_disposal_df(
            [{"subgraph": "Well D", "Days": days, "Date": "2021-01-31", "BPD": rate}]
        )

        monthly = _aggregate_disposal_to_monthly_bpd(df)

        self.assertAlmostEqual(monthly.iloc[0]["BPD"], rate, places=6)
        self.assertLess(monthly.iloc[0]["BPD"], rate * days)


if __name__ == "__main__":
    unittest.main()
