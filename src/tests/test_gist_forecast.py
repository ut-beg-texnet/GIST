"""Regression tests for the Forecast step: proposed-rate extension and pressure engine."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.special as sc

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gistMC import extendDisposal, gistMC, prepInj
from gistStepCore import PROPOSED_RATE_COLUMN, load_proposed_rates

EPOCH = pd.to_datetime('1970-01-01')


def reference_extend_disposal(injDF, startDate, endDate, rateDict, dDays=10, epoch=EPOCH):
    """Original per-well loop implementation of extendDisposal, kept as the reference."""
    startDay = int((startDate - epoch).days)
    endDay = int((endDate - epoch).days)
    pastInjDF = injDF[injDF['Days'] < startDay].copy()
    lastDay = int(pastInjDF['Days'].max())
    lastInjDF = injDF[injDF['Days'] == lastDay]
    futureDays = np.arange(lastDay + dDays, endDay + dDays, float(dDays))
    futureDates = [epoch + pd.to_timedelta(d, unit='day') for d in futureDays]
    futureInjDF = pd.DataFrame(columns=['ID', 'Days', 'BPD', 'Date', 'Type'])
    for wellID in injDF['ID'].unique():
        IDs = [wellID] * len(futureDays)
        if wellID in rateDict.keys():
            BPD = rateDict[wellID]
            rateType = ['Set'] * len(futureDays)
        elif len(lastInjDF[lastInjDF['ID'] == wellID]) > 0:
            BPD = lastInjDF[lastInjDF['ID'] == wellID]['BPD'].to_list()[0]
            rateType = ['Extrapolated'] * len(futureDays)
        else:
            BPD = 0.
            rateType = ['No Data'] * len(futureDays)
        BPDs = np.ones(len(futureDays)) * BPD
        futureWellInjDF = pd.DataFrame({'ID': IDs, 'Days': futureDays, 'BPD': BPDs, 'Date': futureDates, 'Type': rateType})
        futureInjDF = pd.concat([futureInjDF, futureWellInjDF])
    pastInjDF['Type'] = 'Original'
    return pd.concat([pastInjDF, futureInjDF])


def make_injection():
    """Three wells on a 10-day grid; well C stops reporting before the last day."""
    rows = []
    for well_id, base, n in (("A", 100.0, 50), ("B", 250.0, 50), ("C", 75.0, 40)):
        for i in range(n):
            day = 18000.0 + 10.0 * i
            rows.append({"ID": well_id, "Days": day, "BPD": base + i})
    inj = pd.DataFrame(rows)
    inj['ID'] = inj['ID'].astype('string')
    inj['Date'] = EPOCH + pd.to_timedelta(inj['Days'], unit='d')
    return inj


class ExtendDisposalTests(unittest.TestCase):
    """The vectorized extendDisposal must reproduce the original loop exactly."""

    def assert_matches_reference(self, rate_dict, start_day):
        inj = make_injection()
        start = EPOCH + pd.Timedelta(days=start_day)
        end = EPOCH + pd.Timedelta(days=18900)
        expected = reference_extend_disposal(inj, start, end, rate_dict, dDays=10.0)
        actual = extendDisposal(inj, start, end, rate_dict, dDays=10.0)
        pd.testing.assert_frame_equal(actual, expected)
        return actual

    def test_set_extrapolated_and_no_data_rates(self):
        actual = self.assert_matches_reference({"A": 10000.0, "B": 0.0}, start_day=18500)
        future = actual[actual['Type'] != 'Original']
        self.assertEqual(set(future.loc[future['ID'] == "A", 'BPD']), {10000.0})
        self.assertEqual(set(future.loc[future['ID'] == "B", 'BPD']), {0.0})
        # Well C has no sample on the last day, so it is not extrapolated.
        self.assertEqual(set(future.loc[future['ID'] == "C", 'Type']), {'No Data'})

    def test_empty_rate_dict_holds_last_rates(self):
        actual = self.assert_matches_reference({}, start_day=18500)
        future = actual[actual['ID'] == "B"]
        self.assertEqual(set(future.loc[future['Type'] == 'Extrapolated', 'BPD']), {299.0})

    def test_switch_date_inside_history_truncates_history(self):
        actual = self.assert_matches_reference({"C": 5.0}, start_day=18205)
        self.assertEqual(actual.loc[actual['Type'] == 'Original', 'Days'].max(), 18200.0)


class ProposedRatesTests(unittest.TestCase):
    """Proposed Future Rates table from Updated Analysis becomes a well-ID keyed dict."""

    def write_rates(self, root, rows):
        path = Path(root) / "rates.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    def test_ids_are_normalized_and_blank_rates_are_dropped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.write_rates(temp_dir, [
                {"ID": " 00123 ", "WellName": "A", PROPOSED_RATE_COLUMN: 10000.0},
                {"ID": "456", "WellName": "B", PROPOSED_RATE_COLUMN: 0.0},
                {"ID": "789", "WellName": "C", PROPOSED_RATE_COLUMN: None},
            ])
            self.assertEqual(load_proposed_rates(path), {"00123": 10000.0, "456": 0.0})

    def test_negative_rate_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.write_rates(temp_dir, [{"ID": "1", "WellName": "A", PROPOSED_RATE_COLUMN: -5.0}])
            with self.assertRaises(ValueError):
                load_proposed_rates(path)


class PressureScenariosVecTests(unittest.TestCase):
    """Evaluating the well function only where it is used must not change results."""

    def test_matches_full_length_well_function(self):
        gist = gistMC(nReal=25)
        gist.initPP()
        gist.injDT = 10.0
        inj = make_injection()
        wells = pd.DataFrame({
            "ID": pd.array(["A", "B", "C"], dtype="string"), "WellName": ["A", "B", "C"],
            "APINumber": [1, 2, 3], "SurfaceHoleLatitude": [31.0, 31.1, 31.2],
            "SurfaceHoleLongitude": [-102.0, -102.1, -102.2], "StartDate": ["2019-04-14"] * 3,
            "Distances": [2.0, 5.0, 9.0], "DXs": [1.0, 3.0, 6.0], "DYs": [1.7, 4.0, 6.7],
        })
        # Event in the middle of the data, so many trailing samples go unused.
        eq = {"Latitude": 31.0, "Longitude": -102.0, "LatitudeError": 0.0, "LongitudeError": 0.0,
              "Origin Date": (EPOCH + pd.Timedelta(days=18203)).strftime("%Y-%m-%d"), "EventID": "test"}

        actual = gist.runPressureScenariosVec(eq, wells, inj)

        # Reference: previous implementation that evaluated exp1 on all nt durations.
        eqDay = (pd.to_datetime(eq['Origin Date']) - gist.epoch).days
        (_, _, _, nt, _, bpdArray, secArray, _, _, wellDistances, ieq, f) = prepInj(
            wells, inj, gist.injDT, eqDay=eqDay, epoch=gist.epoch)
        dQdtArray = np.diff(1.84013e-6 * bpdArray, axis=1)
        ppp = np.outer(wellDistances * wellDistances, gist.TVec * gist.SVec / (4. * gist.TVec * gist.TVec))
        durations = np.max(secArray) - secArray + gist.injDT * 24 * 60 * 60
        epp = sc.exp1(ppp[:, :, np.newaxis] / durations[np.newaxis, np.newaxis, :nt])
        sum1 = np.einsum('ijk,ik->ij', epp[:, :, -ieq:], dQdtArray[:, :ieq], optimize=True)
        sum2 = np.einsum('ijk,ik->ij', epp[:, :, -(ieq + 1):], dQdtArray[:, :ieq + 1], optimize=True)
        scale = gist.rhoVec * gist.g / (4. * np.pi * gist.TVec) / 6894.76
        dPatEQ = (1. - f) * (sum1 * scale) + f * (sum2 * scale)
        expected = dPatEQ.T.flatten()  # scenario rows are realization-major

        self.assertTrue(np.array_equal(actual['Pressures'].to_numpy(dtype=float), expected))


class DisposalPayloadCacheTests(unittest.TestCase):
    """The per-well graphs reuse one disposal payload without changing its content."""

    def test_cached_payload_matches_and_changes_with_input(self):
        from gist_graphs import _build_disposal_payload, _build_disposal_payload_uncached

        disposal = make_injection().rename(columns={"ID": "subgraph"})
        first = _build_disposal_payload(disposal)
        self.assertEqual(first, _build_disposal_payload_uncached(disposal))
        self.assertEqual(_build_disposal_payload(disposal), first)

        edited = disposal.copy()
        edited.loc[0, "BPD"] = 9999.0
        self.assertEqual(_build_disposal_payload(edited), _build_disposal_payload_uncached(edited))
        self.assertNotEqual(_build_disposal_payload(edited), first)


if __name__ == "__main__":
    unittest.main()
