"""Unit tests for Pressure Ranges HTML axis/gridline alignment."""

import re
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gist_graphs import save_pressure_ranges_graph_artifact


class _StubHelper:
    """Minimal helper that writes artifacts into a temp scratch directory."""

    def __init__(self, scratch_path):
        self.scratchPath = str(scratch_path)

    def saveGraphArtifact(self, **kwargs):
        return None


def _make_disaggregation_df():
    """Tiny strip-plot frame with a known max pressure of 200 PSI."""
    return pd.DataFrame(
        [
            {"Name": "WELL A", "WellNo": 2, "Order": 1, "Pressures": 50.0},
            {"Name": "WELL A", "WellNo": 2, "Order": 2, "Pressures": 100.0},
            {"Name": "WELL B", "WellNo": 1, "Order": 1, "Pressures": 200.0},
        ]
    )


class PressureRangesAxisAlignmentTests(unittest.TestCase):
    """Guard the shared CSS-grid layout that keeps ticks on the gridlines."""

    def test_gridline_and_tick_share_plot_x(self):
        """Each PSI tick label must use the same viewBox x as its gridline."""
        df = _make_disaggregation_df()
        with tempfile.TemporaryDirectory() as scratch:
            helper = _StubHelper(scratch)
            save_pressure_ranges_graph_artifact(
                helper,
                df,
                artifact_key="gist-test-pressure-ranges",
                display_order=1,
            )
            html_path = Path(scratch) / "graphs" / "gist-test-pressure-ranges.html"
            self.assertTrue(html_path.is_file())
            html = html_path.read_text(encoding="utf-8")

            self.assertNotIn("margin-left: 240px", html)
            self.assertEqual(html.count('class="chart-grid'), 2)
            self.assertIn("plot-grid", html)
            self.assertIn("axis-row", html)

            grid_xs = {
                tick: x
                for tick, x in re.findall(
                    r'<line class="gridline" data-tick="([^"]+)" x1="([^"]+)"',
                    html,
                )
            }
            tick_xs = {
                tick: x
                for tick, x in re.findall(
                    r'<text class="x-tick" data-tick="([^"]+)" x="([^"]+)"',
                    html,
                )
            }
            self.assertTrue(grid_xs)
            self.assertEqual(set(grid_xs), set(tick_xs))
            self.assertEqual(grid_xs, tick_xs)
            self.assertEqual(grid_xs["0"], "0.0")


if __name__ == "__main__":
    unittest.main()
