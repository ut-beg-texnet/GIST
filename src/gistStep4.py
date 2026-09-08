import urllib3
import requests
import json
import numpy as np
from pandas import Timestamp
import pandas as pd
import matplotlib.colors as mcolors
# from pandas import DataFrame
import sys
import os
import time
import pathlib

from datetime import datetime
from math import ceil

from TexNetWebToolGPWrappers import TexNetWebToolLaunchHelper

from gistStepCore import (
    DEFAULT_EARTHQUAKE_LOCATION_ERROR_KM,
    eq_location_error_km,
    runGistCore,
)
from gistMC import normalizeGistIds
from progress import report_progress
from gist_graphs import (
    filter_rt_plot_wells_future_start_date,
    save_pressure_ranges_graph_artifact,
    save_rt_plot_graph_artifact,
    save_time_series_quantiles_graph_artifact,
    save_time_series_quantiles_per_well_graph_artifact,
    save_time_series_spaghetti_graph_artifact,
    save_time_series_spaghetti_per_well_graph_artifact,
)


def get_corrected_gist_data_paths(helper, well_type):
    wellcsv = helper.getDatasetFilePathWithStepIndexAndParamName(2, "GISTWells")
    injectioncsv = helper.getDatasetFilePathWithStepIndexAndParamName(2, "GISTInjection")

    if wellcsv is None or injectioncsv is None:
        raise ValueError("Upload both corrected GIST well and injection datasets before running Updated Analysis.")

    if wellcsv is not None and injectioncsv is not None:
        return wellcsv, injectioncsv

    if well_type == 'Shallow':
        return (
            'C:/texnetwebtools/tools/GIST/src/data/gist_well_shallow.csv',
            'C:/texnetwebtools/tools/GIST/src/data/gist_injection_shallow.csv',
        )

    return (
        'C:/texnetwebtools/tools/GIST/src/data/gist_well_deep.csv',
        'C:/texnetwebtools/tools/GIST/src/data/gist_injection_deep.csv',
    )


scratchPath = sys.argv[1]

# #instantiate the helper
helper = TexNetWebToolLaunchHelper(scratchPath)

#Get the args data out of it.
argsData = helper.argsData

report_progress("Preparing updated analysis inputs")

#getParameterValueWithStepIndexAndParamName
eventType = helper.getParameterValueWithStepIndexAndParamName(0, "eventType")
selectedEvent = helper.getParameterValueWithStepIndexAndParamName(0, "Earthquake")
if eventType == "Scenario":
    scenarioLoc = helper.getParameterValueWithStepIndexAndParamName(0, "scenarioLoc")
    scenarioDate = helper.getParameterValueWithStepIndexAndParamName(0, "scenarioDate")
    selectedEvent = {"selectedRow": {"attributes": {
        "Origin Date": scenarioDate,
        "Latitude (WGS84)": scenarioLoc.get("y"), "Latitude Error (km)": 0,
        "Longitude (WGS84)": scenarioLoc.get("x"), "Longitude Error (km)": 0,
        "EventID": "AAAAAA"
    }}}
Earthquake = (selectedEvent or {}).get("selectedRow", {}).get("attributes")
if Earthquake is None:
    raise ValueError("Select an earthquake or scenario before running Updated Analysis.")

date = Timestamp(Earthquake.get("Origin Date"), unit="ms")
formatted_date = date.strftime("%Y-%m-%d")

# Catalog earthquakes may omit location error; scenarios already supply 0.
location_error_default = DEFAULT_EARTHQUAKE_LOCATION_ERROR_KM if eventType == "Earthquake" else 0
formattedEarthquake = {
    "Latitude": Earthquake.get("Latitude (WGS84)"),
    "LatitudeError": eq_location_error_km(Earthquake.get("Latitude Error (km)"), location_error_default),
    "Longitude": Earthquake.get("Longitude (WGS84)"),
    "LongitudeError": eq_location_error_km(Earthquake.get("Longitude Error (km)"), location_error_default),
    "Origin Date": formatted_date,
    "EventID": Earthquake.get("EventID")
}

# Forecast end date is collected with the shared event inputs.
forecastDate = helper.getParameterValueWithStepIndexAndParamName(0,"forecastEndDate")

eq_date = datetime.strptime(formatted_date, "%Y-%m-%d")
future_date = datetime.strptime(forecastDate, "%Y-%m-%dT%H:%M:%S.%fZ")

days_diff = (future_date - eq_date).days
years_diff = days_diff / 365

realizationCount = helper.getParameterValueWithStepIndexAndParamName(3,"realizationCount")
wellType = helper.getParameterValueWithStepIndexAndParamName(3,"wellType")
rho0 = helper.getParameterValueWithStepIndexAndParamName(3,"rho0")
phi = helper.getParameterValueWithStepIndexAndParamName(3,"phi")
nta = helper.getParameterValueWithStepIndexAndParamName(3,"nta")
kMD = helper.getParameterValueWithStepIndexAndParamName(3,"kMD")
h = helper.getParameterValueWithStepIndexAndParamName(3,"h")
cppMS = helper.getParameterValueWithStepIndexAndParamName(3,"cppMS")
betaMS = helper.getParameterValueWithStepIndexAndParamName(3,"betaMS")

input = {
    "years_diff": years_diff,
    "realizationCount": realizationCount,
    "porePressureParams" : {
        "rho0_min": float(rho0.get("min")),
        "rho0_max": float(rho0.get("max")),
        "nta_min": float(nta.get("min")),
        "nta_max": float(nta.get("max")),
        "phi_min": float(phi.get("min")),
        "phi_max": float(phi.get("max")),
        "kMD_min": float(kMD.get("min")),
        "kMD_max": float(kMD.get("max")),
        "h_min": float(h.get("min")),
        "h_max": float(h.get("max")),
        "cppMS_min": float(cppMS.get("min")),
        "cppMS_max": float(cppMS.get("max")),
        "betaMS_min": float(betaMS.get("min")),
        "betaMS_max": float(betaMS.get("max"))
    },
    "eq": formattedEarthquake
}

wellcsv, injectioncsv = get_corrected_gist_data_paths(helper, wellType)

report_progress("Running updated analysis workflow")
try:
    smallPPDF, smallWellList, disaggregationDF, orderedWellList, totalPPQuantilesDF, totalPPSpaghettiDF, allPerWellPPQuantilesDF, allPerWellPPSpaghettiDF, allPerWellDisposalDF = runGistCore(input, wellcsv, injectioncsv)
except ValueError as e:
    helper.addMessageWithStepIndex(3, str(e), 2)
    helper.setSuccessForStepIndex(3, False)
    helper.writeResultsFile()
    sys.exit(1)

report_progress("Preparing graph data")

# Calculate cutoff for R-T Plot
max_dist_max_diff = smallPPDF[smallPPDF['Diffusivity'] == 'Maximum']['Distance'].max()
rt_plot_cutoff = max_dist_max_diff * 3
smallWellList_r_t_plot_updated = smallWellList[smallWellList['Distances'] <= rt_plot_cutoff].copy()
smallWellList_r_t_plot_updated = smallWellList_r_t_plot_updated.dropna(subset=['YearsInjectingToEarthquake', 'Distances'])
smallWellList_r_t_plot_updated = filter_rt_plot_wells_future_start_date(smallWellList_r_t_plot_updated)

if disaggregationDF.empty:
    helper.addMessageWithStepIndex(3, "No Wells Found.", 2)
    helper.setSuccessForStepIndex(3, False)
else:
    # orderedWellList with proposed Future Rate initalize at 10000
    originalWellDF = pd.read_csv(wellcsv, dtype={'ID': 'string'})
    originalWellDF['ID'] = normalizeGistIds(originalWellDF['ID'])
    if 'PermittedMaxLiquidBPD' not in originalWellDF.columns:
        originalWellDF['PermittedMaxLiquidBPD'] = np.nan
    orderedWellList = pd.DataFrame(orderedWellList, columns=['ID'])
    orderedWellList = orderedWellList.merge(
        originalWellDF[['ID', 'WellName', 'PermittedMaxLiquidBPD']],
        left_on='ID',
        right_on='ID',
        how='left'
    )
    # Keep missing permit maxima blank; the 10,000 BPD cap applies only to proposed rate.
    orderedWellList["PermittedMaxLiquidBPD"] = pd.to_numeric(
        orderedWellList["PermittedMaxLiquidBPD"], errors='coerce'
    )
    permitted_rate = orderedWellList["PermittedMaxLiquidBPD"]
    orderedWellList['Proposed Future Rate (BPD)'] = np.where(
        permitted_rate.fillna(10000.0) < 10000,
        permitted_rate,  # Use permit max if it is below the standard cap
        10000
    )
    orderedWellListWithFutureRates = orderedWellList.drop(orderedWellList.index[-1])

    report_progress("Saving results")
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "smallPPDF_updated", smallPPDF)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "smallWellList_updated", smallWellList)
    # D3 graph datasets temporarily disabled for matplotlib-only portal performance testing.
    # helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "smallWellList_r_t_plot_updated", smallWellList_r_t_plot_updated)
    # helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "disaggregationDF_updated", disaggregationDF)
    # helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "totalPPQuantilesDF_updated", totalPPQuantilesDF)
    # helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "totalPPSpaghettiDF_updated", totalPPSpaghettiDF)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "orderedWellListWithFutureRates", orderedWellListWithFutureRates)
    # helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "allPerWellPPQuantilesDF_updated", allPerWellPPQuantilesDF)
    # helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "allPerWellPPSpaghettiDF_updated", allPerWellPPSpaghettiDF)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "allPerWellDisposalDF_updated", allPerWellDisposalDF)

    report_progress("Generating result graphs")
    save_rt_plot_graph_artifact(
        helper,
        smallPPDF,
        smallWellList_r_t_plot_updated,
        artifact_key="gist-updated-r-t-plot",
        display_order=10,
    )
    save_pressure_ranges_graph_artifact(
        helper,
        disaggregationDF,
        artifact_key="gist-updated-pressure-ranges",
        display_order=20,
    )
    save_time_series_quantiles_graph_artifact(
        helper,
        totalPPQuantilesDF,
        artifact_key="gist-updated-time-series-quantiles",
        display_order=30,
    )
    save_time_series_spaghetti_graph_artifact(
        helper,
        totalPPSpaghettiDF,
        artifact_key="gist-updated-time-series-spaghetti",
        display_order=40,
    )
    save_time_series_quantiles_per_well_graph_artifact(
        helper,
        allPerWellPPQuantilesDF,
        allPerWellDisposalDF,
        artifact_key="gist-updated-time-series-quantiles-per-well",
        display_order=50,
    )
    save_time_series_spaghetti_per_well_graph_artifact(
        helper,
        allPerWellPPSpaghettiDF,
        allPerWellDisposalDF,
        artifact_key="gist-updated-time-series-spaghetti-per-well",
        display_order=60,
    )

    helper.setSuccessForStepIndex(2, True)
    helper.setSuccessForStepIndex(3, True)

helper.writeResultsFile()
