"""Regression tests for Disposal Correction identifiers and optional metadata."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gistMC import SMALL_WELLS_ID, getWinWells, gistMC, summarizePPResults


class DisposalCorrectionIdTests(unittest.TestCase):
    """Ensure corrected uploads preserve their string well identifiers."""

    def test_add_wells_accepts_long_string_id_without_api_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            well_file = root / "wells.csv"
            injection_file = root / "injection.csv"
            well_id = "10142010000130975"

            pd.DataFrame([{
                "ID": well_id,
                "WellName": "String ID well",
                "StartDate": "2020-01-01",
                "SurfaceHoleLatitude": 31.0,
                "SurfaceHoleLongitude": -102.0,
            }]).to_csv(well_file, index=False)
            pd.DataFrame([
                {"ID": well_id, "Days": 18000 + day * 10, "BPD": 10.0,
                 "Date": "2019-04-14"}
                for day in range(100)
            ]).to_csv(injection_file, index=False)

            gist = gistMC()
            gist.addWells(well_file, injection_file)

            self.assertEqual(gist.wellDF.loc[0, "ID"], well_id)
            self.assertEqual(gist.injAllDF.loc[0, "ID"], well_id)

    def test_string_well_ids_exclude_only_the_synthetic_summary_row(self):
        pressure_rows = pd.DataFrame([{
            "ID": "00123", "Name": "A", "Pressures": 1.0,
            "Percentages": 100.0, "Realization": 0, "TotalPressure": 1.0,
            "EventID": "event", "EventLatitude": 1.0, "EventLongitude": 1.0,
            "LagInDays": 0.0,
        }])
        summary, _ = summarizePPResults(pressure_rows, pd.DataFrame(), threshold=0.1)
        wells = pd.DataFrame({"ID": ["00123"]})
        injection = pd.DataFrame({"ID": ["00123"]})

        selected_wells, selected_injection = getWinWells(summary, wells, injection)

        self.assertNotIn(SMALL_WELLS_ID, selected_wells["ID"].tolist())
        self.assertEqual(selected_wells["ID"].tolist(), ["00123"])
        self.assertEqual(selected_injection["ID"].tolist(), ["00123"])


if __name__ == "__main__":
    unittest.main()