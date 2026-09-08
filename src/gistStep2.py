import urllib3
import requests
from pandas import Timestamp
import pandas as pd
# from pandas import DataFrame
import sys
import os
import time

from datetime import datetime
from math import ceil

from TexNetWebToolGPWrappers import TexNetWebToolLaunchHelper

from gistStepCore import (
    DEFAULT_EARTHQUAKE_LOCATION_ERROR_KM,
    eq_location_error_km,
    runGistCore,
)
from progress import report_progress
from gist_graphs import (
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

#getParameterValueWithStepIndexAndParamName
eventType = helper.getParameterValueWithStepIndexAndParamName(0,"eventType")

formattedEarthquake = {}

report_progress("Preparing analysis inputs")

if eventType == 'Earthquake':

    Earthquake = helper.getParameterValueWithStepIndexAndParamName(0,"Earthquake").get("selectedRow").get("attributes")

    date = Timestamp(Earthquake.get("Origin Date"), unit="ms")
    formatted_date = date.strftime("%Y-%m-%d")

    formattedEarthquake = {
        "Latitude": Earthquake.get("Latitude (WGS84)"),
        "LatitudeError": eq_location_error_km(
            Earthquake.get("Latitude Error (km)"), DEFAULT_EARTHQUAKE_LOCATION_ERROR_KM
        ),
        "Longitude": Earthquake.get("Longitude (WGS84)"),
        "LongitudeError": eq_location_error_km(
            Earthquake.get("Longitude Error (km)"), DEFAULT_EARTHQUAKE_LOCATION_ERROR_KM
        ),
        "Origin Date": formatted_date,
        "EventID": Earthquake.get("EventID")
    }



if eventType == 'Scenario':

    scenarioLoc = helper.getParameterValueWithStepIndexAndParamName(0,"scenarioLoc")
    scenarioDate = helper.getParameterValueWithStepIndexAndParamName(0,"scenarioDate")
    
    date = Timestamp(scenarioDate, unit="ms")
    formatted_date = date.strftime("%Y-%m-%d")

    #this is a hypothecical earthquake
    formattedEarthquake = {
        "Latitude": scenarioLoc.get("y"),
        "LatitudeError": 0,
        "Longitude": scenarioLoc.get("x"),
        "LongitudeError": 0,
        "Origin Date": formatted_date,
        "EventID": "AAAAAA"
    }


# Forecast end date is collected with the shared event inputs.
forecastDate = helper.getParameterValueWithStepIndexAndParamName(0,"forecastEndDate")

eq_date = datetime.strptime(formatted_date, "%Y-%m-%d")
future_date = datetime.strptime(forecastDate, "%Y-%m-%dT%H:%M:%S.%fZ")

days_diff = (future_date - eq_date).days
years_diff = days_diff / 365

realizationCount = helper.getParameterValueWithStepIndexAndParamName(1,"realizationCount")
wellType = helper.getParameterValueWithStepIndexAndParamName(1,"wellType")
rho0 = helper.getParameterValueWithStepIndexAndParamName(1,"rho0")
phi = helper.getParameterValueWithStepIndexAndParamName(1,"phi")
nta = helper.getParameterValueWithStepIndexAndParamName(1,"nta")
kMD = helper.getParameterValueWithStepIndexAndParamName(1,"kMD")
h = helper.getParameterValueWithStepIndexAndParamName(1,"h")
cppMS = helper.getParameterValueWithStepIndexAndParamName(1,"cppMS")
betaMS = helper.getParameterValueWithStepIndexAndParamName(1,"betaMS")

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

if wellType == 'Shallow':
    wellcsv = 'C:/texnetwebtools/tools/GIST/src/data/gist_well_shallow.csv'
    injectioncsv = 'C:/texnetwebtools/tools/GIST/src/data/gist_injection_shallow.csv'
else:
    wellcsv = 'C:/texnetwebtools/tools/GIST/src/data/gist_well_deep.csv'
    injectioncsv = 'C:/texnetwebtools/tools/GIST/src/data/gist_injection_deep.csv'

try:
    report_progress("Running analysis workflow")
    smallPPDF, smallWellList, disaggregationDF, orderedWellList, totalPPQuantilesDF, totalPPSpaghettiDF, allPerWellPPQuantilesDF, allPerWellPPSpaghettiDF, allPerWellDisposalDF = runGistCore(input, wellcsv, injectioncsv)
except ValueError as e:
    helper.addMessageWithStepIndex(1, str(e), 2)
    helper.setSuccessForStepIndex(1, False)
    helper.writeResultsFile()
    sys.exit(1)

report_progress("Preparing graph data")

# Calculate cutoff for R-T Plot
max_dist_max_diff = smallPPDF[smallPPDF['Diffusivity'] == 'Maximum']['Distance'].max()

rt_plot_cutoff = max_dist_max_diff * 3

# Filter the dataset
smallWellList_r_t_plot = smallWellList[smallWellList['Distances'] <= rt_plot_cutoff].copy()
smallWellList_r_t_plot = smallWellList_r_t_plot.dropna(subset=['YearsInjectingToEarthquake', 'Distances'])




if disaggregationDF.empty:
    helper.addMessageWithStepIndex(1, "No Wells Found.", 2)
    helper.setSuccessForStepIndex(1, False)
else:
    report_progress("Saving result datasets")
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "smallPPDF", smallPPDF)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "smallWellList", smallWellList)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "smallWellList_r_t_plot", smallWellList_r_t_plot)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "disaggregationDF", disaggregationDF)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "totalPPQuantilesDF", totalPPQuantilesDF)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "totalPPSpaghettiDF", totalPPSpaghettiDF)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "allPerWellPPQuantilesDF", allPerWellPPQuantilesDF)
    # helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "allPerWellPPSpaghettiDF", allPerWellPPSpaghettiDF)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "allPerWellDisposalDF", allPerWellDisposalDF)

    # Passed through unmodified for step 3 (download/correct/reupload), so copy the files
    # directly instead of round-tripping them through pandas read_csv/to_csv - avoids a second
    # full parse plus a full CSV re-serialization of the (very large) injection dataset.
    helper.saveFileAsParameterWithStepIndexAndParamName(1, "GISTWells-corrections", wellcsv)
    helper.saveFileAsParameterWithStepIndexAndParamName(1, "GISTInjection-corrections", injectioncsv)
    report_progress("Generating result graphs")
    save_rt_plot_graph_artifact(
        helper,
        smallPPDF,
        smallWellList_r_t_plot,
        artifact_key="gist-analysis-r-t-plot",
        display_order=10,
    )
    save_pressure_ranges_graph_artifact(
        helper,
        disaggregationDF,
        artifact_key="gist-analysis-pressure-ranges",
        display_order=20,
    )
    save_time_series_quantiles_graph_artifact(
        helper,
        totalPPQuantilesDF,
        artifact_key="gist-analysis-time-series-quantiles",
        display_order=30,
    )
    save_time_series_spaghetti_graph_artifact(
        helper,
        totalPPSpaghettiDF,
        artifact_key="gist-analysis-time-series-spaghetti",
        display_order=40,
    )
    save_time_series_quantiles_per_well_graph_artifact(
        helper,
        allPerWellPPQuantilesDF,
        allPerWellDisposalDF,
        artifact_key="gist-analysis-time-series-quantiles-per-well",
        display_order=50,
    )
    save_time_series_spaghetti_per_well_graph_artifact(
        helper,
        allPerWellPPSpaghettiDF,
        allPerWellDisposalDF,
        artifact_key="gist-analysis-time-series-spaghetti-per-well",
        display_order=60,
    )

    #Since step 0 doesnt have business logic set its sucess to true.
    helper.setSuccessForStepIndex(0, True)
    helper.setSuccessForStepIndex(1, True)

helper.writeResultsFile()
