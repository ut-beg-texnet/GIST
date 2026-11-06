"""Unit tests for the R-t well-name tooltip and the zoom controls on every GIST HTML graph."""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gist_graphs
from gist_graphs import (
    save_pressure_ranges_graph_artifact,
    save_rt_plot_graph_artifact,
    save_time_series_quantiles_graph_artifact,
    save_time_series_quantiles_per_well_graph_artifact,
    save_time_series_spaghetti_graph_artifact,
    save_time_series_spaghetti_per_well_graph_artifact,
)


class _RecordingHelper:
    """Writes artifacts into a temp scratch directory and records saveGraphArtifact kwargs."""

    def __init__(self, scratch_path):
        self.scratchPath = str(scratch_path)
        self.saved = []

    def saveGraphArtifact(self, **kwargs):
        self.saved.append(kwargs)


def _payload(html, script_id):
    match = re.search(rf'<script type="application/json" id="{script_id}">(.*?)</script>', html, re.S)
    return json.loads(match.group(1))


def _small_pp_df():
    rows = []
    for label, scale in (("Minimum", 1.0), ("Maximum", 3.0)):
        for t in np.linspace(-10.0, 0.0, 5):
            rows.append({"Diffusivity": label, "Years Before Earthquake": t, "Distance": scale * (-t) ** 0.5})
    return pd.DataFrame(rows)


def _rt_well_df():
    return pd.DataFrame(
        [
            {"UICNumber": 100361, "WellName": "EAST PENWELL /SAN ANDRES/ UNIT 1D", "Selection": "Must Include",
             "YearsInjectingToEarthquake": -16.772073921971252, "Distances": 13.69888552517907, "MMBBL": 11.953415991299858},
            {"UICNumber": 100370, "WellName": np.nan, "Selection": "Exclude",
             "YearsInjectingToEarthquake": -4.0, "Distances": 5.0, "MMBBL": 0.5},
            {"UICNumber": 100397, "WellName": "   ", "Selection": "May Include",
             "YearsInjectingToEarthquake": -2.0, "Distances": 2.0, "MMBBL": 1.0},
        ]
    )


def _time_series_df(group_column, groups, n_points, extra=None):
    dates = pd.date_range("2015-01-01", periods=n_points, freq="D")
    frames = []
    for i, g in enumerate(groups):
        frame = pd.DataFrame(
            {
                "Date": dates,
                "DeltaPressure": np.linspace(0.0, 10.0 + i, n_points),
                group_column: g,
            }
        )
        if extra:
            for column, value_fn in extra.items():
                frame[column] = value_fn(i)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class RtWellNameTests(unittest.TestCase):
    def test_payload_has_well_name_only_when_known_and_rounded_numbers(self):
        with tempfile.TemporaryDirectory() as scratch:
            helper = _RecordingHelper(scratch)
            save_rt_plot_graph_artifact(helper, _small_pp_df(), _rt_well_df(), "rt", 10)
            html = (Path(scratch) / "graphs" / "rt.html").read_text(encoding="utf-8")

        points = _payload(html, "rt-payload")["points"]
        by_uic = {p["uic"]: p for p in points}
        self.assertEqual(by_uic["100361"]["wellName"], "EAST PENWELL /SAN ANDRES/ UNIT 1D")
        self.assertEqual(by_uic["100370"]["wellName"], "")
        self.assertEqual(by_uic["100397"]["wellName"], "")
        self.assertEqual(by_uic["100361"]["x"], -16.7721)
        self.assertEqual(by_uic["100361"]["y"], 13.6989)
        self.assertEqual(by_uic["100361"]["mmbbl"], 11.9534)
        self.assertIn("Well Name", html)
        self.assertIn("well name", helper.saved[0]["caption"])

    def test_well_name_cannot_close_the_payload_script(self):
        well_df = _rt_well_df()
        well_df.loc[0, "WellName"] = "BAD </script><b>NAME"
        with tempfile.TemporaryDirectory() as scratch:
            save_rt_plot_graph_artifact(_RecordingHelper(scratch), _small_pp_df(), well_df, "rt", 10)
            html = (Path(scratch) / "graphs" / "rt.html").read_text(encoding="utf-8")
        self.assertNotIn("BAD </script>", html)
        names = {p["wellName"] for p in _payload(html, "rt-payload")["points"]}
        self.assertIn("BAD </script><b>NAME", names)


class ZoomOnAllGraphsTests(unittest.TestCase):
    def _assert_zoomable(self, html):
        self.assertIn("GistZoom.attach(", html)
        self.assertIn("GistZoom.addButtons(", html)
        self.assertNotIn("___ZOOM_JS___", html)
        self.assertNotIn("___ZOOM_CSS___", html)
        self.assertNotIn("addEventListener('wheel'", html)

    def test_every_graph_is_html_with_zoom_controls(self):
        quantiles = _time_series_df("Percentile", [0.0, 49.0, 100.0], 30)
        spaghetti = _time_series_df("Realization", [0, 1, 2], 30, extra={"Diffusivity": lambda i: 0.5 + i})
        per_well_quantiles = _time_series_df("Percentile", [0.0, 49.0, 100.0], 30, extra={"subgraph": lambda i: "WELL A"})
        per_well_spaghetti = _time_series_df(
            "Realization", [0, 1], 30, extra={"subgraph": lambda i: "WELL A", "Diffusivity": lambda i: 1.0 + i}
        )
        disposal = pd.DataFrame(
            {"Date": pd.date_range("2015-01-01", periods=60, freq="7D"), "BPD": 1000.0, "subgraph": "WELL A"}
        )
        disaggregation = pd.DataFrame(
            [
                {"Name": "WELL A", "WellNo": 2, "Order": 1, "Pressures": 50.0},
                {"Name": "WELL B", "WellNo": 1, "Order": 2, "Pressures": 200.0},
            ]
        )

        with tempfile.TemporaryDirectory() as scratch:
            helper = _RecordingHelper(scratch)
            save_rt_plot_graph_artifact(helper, _small_pp_df(), _rt_well_df(), "rt", 10)
            save_pressure_ranges_graph_artifact(helper, disaggregation, "pr", 20)
            save_time_series_quantiles_graph_artifact(helper, quantiles, "tq", 30)
            save_time_series_spaghetti_graph_artifact(helper, spaghetti, "ts", 40)
            save_time_series_quantiles_per_well_graph_artifact(helper, per_well_quantiles, disposal, "wq", 50)
            save_time_series_spaghetti_per_well_graph_artifact(helper, per_well_spaghetti, disposal, "ws", 60)

            self.assertEqual([s["key"] for s in helper.saved], ["rt", "pr", "tq", "ts", "wq", "ws"])
            for saved in helper.saved:
                self.assertEqual(saved["renderer"], "html", saved["key"])
                self.assertEqual(saved["contentType"], "text/html", saved["key"])
                path = Path(saved["path"])
                self.assertEqual(path.suffix, ".html")
                self._assert_zoomable(path.read_text(encoding="utf-8"))

            pr_html = (Path(scratch) / "graphs" / "pr.html").read_text(encoding="utf-8")
            self.assertIn("axes: 'x'", pr_html)
            self.assertEqual(_payload(pr_html, "pr-zoom-config")["ticks"][0], 0)

            tq = _payload((Path(scratch) / "graphs" / "tq.html").read_text(encoding="utf-8"), "payload")
            self.assertFalse(tq["showSelector"])
            self.assertFalse(tq["showDisposal"])
            self.assertEqual(tq["colorLabel"], "Percentile")
            emphasized = [s["colorValue"] for s in tq["wells"][0]["series"] if s.get("emphasis")]
            self.assertEqual(emphasized, [49.0])

            ts = _payload((Path(scratch) / "graphs" / "ts.html").read_text(encoding="utf-8"), "payload")
            self.assertEqual(ts["colorLabel"], "Diffusivity")
            self.assertEqual(ts["colorDomain"], [0.5, 2.5])

            wq = _payload((Path(scratch) / "graphs" / "wq.html").read_text(encoding="utf-8"), "payload")
            self.assertNotIn("showSelector", wq)
            self.assertTrue(wq["wells"][0]["disposal"])

    def test_total_spaghetti_respects_realization_and_point_caps(self):
        n_real = gist_graphs._TOTAL_SPAGHETTI_MAX_REALIZATIONS + 20
        n_points = gist_graphs._TOTAL_SPAGHETTI_MAX_POINTS_PER_GROUP * 3
        spaghetti = _time_series_df("Realization", list(range(n_real)), n_points, extra={"Diffusivity": lambda i: float(i)})
        with tempfile.TemporaryDirectory() as scratch:
            save_time_series_spaghetti_graph_artifact(_RecordingHelper(scratch), spaghetti, "ts", 40)
            payload = _payload((Path(scratch) / "graphs" / "ts.html").read_text(encoding="utf-8"), "payload")
        series = payload["wells"][0]["series"]
        self.assertLessEqual(len(series), gist_graphs._TOTAL_SPAGHETTI_MAX_REALIZATIONS)
        self.assertTrue(all(len(s["points"]) <= gist_graphs._TOTAL_SPAGHETTI_MAX_POINTS_PER_GROUP for s in series))


if __name__ == "__main__":
    unittest.main()
