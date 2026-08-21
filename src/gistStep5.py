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

from gistStepCore import runGistCore, get_corrected_gist_data_paths
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


scratchPath = sys.argv[1]

# #instantiate the helper
helper = TexNetWebToolLaunchHelper(scratchPath)

#Get the args data out of it.
argsData = helper.argsData

report_progress("Preparing forecast inputs")

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
    raise ValueError("Select an earthquake or scenario before running Forecast.")

date = Timestamp(Earthquake.get("Origin Date"), unit="ms")
formatted_date = date.strftime("%Y-%m-%d")

formattedEarthquake = {
    "Latitude": Earthquake.get("Latitude (WGS84)"),
    "LatitudeError": Earthquake.get("Latitude Error (km)"),
    "Longitude": Earthquake.get("Longitude (WGS84)"),
    "LongitudeError": Earthquake.get("Longitude Error (km)"),
    "Origin Date": formatted_date,
    "EventID": Earthquake.get("EventID")
}

forecastDate = helper.getParameterValueWithStepIndexAndParamName(3,"forecastEndDate")

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

wellcsv, injectioncsv = get_corrected_gist_data_paths(helper)

report_progress("Running forecast workflow")
smallPPDF, smallWellList, disaggregationDF, orderedWellList, totalPPQuantilesDF, totalPPSpaghettiDF, allPerWellPPQuantilesDF, allPerWellPPSpaghettiDF, allPerWellDisposalDF = runGistCore(input, wellcsv, injectioncsv)

report_progress("Preparing graph data")

if disaggregationDF.empty:
    helper.addMessageWithStepIndex(4, "No Wells Found.", 2)
    helper.setSuccessForStepIndex(4, False)
else:
    # Calculate cutoff for R-T Plot
    max_dist_max_diff = smallPPDF[smallPPDF['Diffusivity'] == 'Maximum']['Distance'].max()
    rt_plot_cutoff = max_dist_max_diff * 3
    smallWellList_r_t_plot_forecast = smallWellList[smallWellList['Distances'] <= rt_plot_cutoff].copy()
    smallWellList_r_t_plot_forecast = smallWellList_r_t_plot_forecast.dropna(subset=['YearsInjectingToEarthquake', 'Distances'])
    smallWellList_r_t_plot_forecast = filter_rt_plot_wells_future_start_date(smallWellList_r_t_plot_forecast)
    # D3 graph datasets temporarily disabled for matplotlib-only portal performance testing.
    # helper.saveDataFrameAsParameterWithStepIndexAndParamName(4, "smallWellList_r_t_plot_forecast", smallWellList_r_t_plot_forecast)

    # helper.saveDataFrameAsParameterWithStepIndexAndParamName(4, "disaggregationDF_forecast", disaggregationDF)
    # helper.saveDataFrameAsParameterWithStepIndexAndParamName(4, "totalPPQuantilesDF_forecast", totalPPQuantilesDF)
    # helper.saveDataFrameAsParameterWithStepIndexAndParamName(4, "totalPPSpaghettiDF_forecast", totalPPSpaghettiDF)
    # helper.saveDataFrameAsParameterWithStepIndexAndParamName(4, "allPerWellPPQuantilesDF_forecast", allPerWellPPQuantilesDF)
    # helper.saveDataFrameAsParameterWithStepIndexAndParamName(4, "allPerWellPPSpaghettiDF_forecast", allPerWellPPSpaghettiDF)
    report_progress("Saving result datasets")
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(4, "smallPPDF_forecast", smallPPDF)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(4, "smallWellList_forecast", smallWellList)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(4, "allPerWellDisposalDF_forecast", allPerWellDisposalDF)

    report_progress("Generating result graphs")
    save_rt_plot_graph_artifact(
        helper,
        smallPPDF,
        smallWellList_r_t_plot_forecast,
        artifact_key="gist-forecast-r-t-plot",
        display_order=10,
    )
    save_pressure_ranges_graph_artifact(
        helper,
        disaggregationDF,
        artifact_key="gist-forecast-pressure-ranges",
        display_order=20,
    )
    save_time_series_quantiles_graph_artifact(
        helper,
        totalPPQuantilesDF,
        artifact_key="gist-forecast-time-series-quantiles",
        display_order=30,
    )
    save_time_series_spaghetti_graph_artifact(
        helper,
        totalPPSpaghettiDF,
        artifact_key="gist-forecast-time-series-spaghetti",
        display_order=40,
    )
    save_time_series_quantiles_per_well_graph_artifact(
        helper,
        allPerWellPPQuantilesDF,
        allPerWellDisposalDF,
        artifact_key="gist-forecast-time-series-quantiles-per-well",
        display_order=50,
    )
    save_time_series_spaghetti_per_well_graph_artifact(
        helper,
        allPerWellPPSpaghettiDF,
        allPerWellDisposalDF,
        artifact_key="gist-forecast-time-series-spaghetti-per-well",
        display_order=60,
    )

    helper.setSuccessForStepIndex(4, True)

helper.writeResultsFile()
