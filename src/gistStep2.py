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

from gistMC import gistMC
from gistMC import prepRTPlot
from gistMC import prepDisaggregationPlot
from gistMC import getWinWells
from gistMC import summarizePPResults
from gistMC import prepTotalPressureTimeSeriesQuantilesPlot

from TexNetWebToolGPWrappers import TexNetWebToolLaunchHelper


def interpolate_colors(vals, min_val, max_val, color_list):
    """Interpolates colors for an array of values across multiple hex colors."""
    # Convert hex colors to RGB
    rgb_colors = np.array([mcolors.hex2color(c) for c in color_list])

    # Define breakpoints for interpolation (evenly spaced)
    num_colors = len(color_list)
    breakpoints = np.linspace(min_val, max_val, num_colors)

    # Normalize values within the min-max range
    ratios = (vals - min_val) / (max_val - min_val) * (num_colors - 1)

    # Find lower and upper indices for interpolation
    low_idx = np.floor(ratios).astype(int)
    high_idx = np.ceil(ratios).astype(int)

    # Clamp indices within valid range
    low_idx = np.clip(low_idx, 0, num_colors - 2)
    high_idx = np.clip(high_idx, 1, num_colors - 1)

    # Compute interpolation weights
    weight = ratios - low_idx

    # Interpolate RGB colors
    interp_rgbs = (1 - weight[:, None]) * rgb_colors[low_idx] + weight[:, None] * rgb_colors[high_idx]

    # Convert RGB values back to hex
    return np.array([mcolors.to_hex(rgb) for rgb in interp_rgbs])


def step2(input):
    # Initialize gistMC class
    gistMC_instance = gistMC()
    gistMC_instance.initPP()
    eq = input.get("eq")
    wellcsv = 'C:/Users/peter/Documents/workspace/GIST/src/data/gist_well_data.csv'
    injectioncsv = 'C:/Users/peter/Documents/workspace/GIST/src/data/gist_injection_data.csv'
    gistMC_instance.addWells(wellcsv, injectioncsv, verbose=2)
    forecastYears = 5 #change to use forcast date
    considered_wells_df, excluded_wells_df, inj_df = gistMC_instance.findWells(eq,PE=False, responseYears=forecastYears)

    # r-t plot combination of considered well and excluded wells df reference plots.py
    smallPPDF,smallWellList = prepRTPlot(considered_wells_df, excluded_wells_df, 1980, [0.1, 2], eq, True)

    # disaggregationPlot plot
    currentWellsDF=considered_wells_df[considered_wells_df['EncompassingDay']<0.].reset_index(drop=True)
    scenarioDF = gistMC_instance.runPressureScenarios(eq,currentWellsDF,inj_df)
    dPCutoff=0.5
    nWells=50
    filteredDF,orderedWellList = summarizePPResults(scenarioDF,currentWellsDF,threshold=dPCutoff,nOrder=nWells)
    disaggregationDF = prepDisaggregationPlot(filteredDF,orderedWellList,jitter=0.1, verbose=1)
    # update order column to force type int prevent upload failure
    disaggregationDF['Order'] = disaggregationDF['Order'].astype(int)
    min_order = disaggregationDF['Order'].min()
    max_order = disaggregationDF['Order'].max()
    color_list = ["#453179", "#485F8A", "#54868D", "#97CE62", "#E5E44E"]
    if min_order == max_order:
        disaggregationDF['Color'] = color_list[0]
    else:
        disaggregationDF['Color'] = interpolate_colors(disaggregationDF['Order'].values, min_order, max_order, color_list)

    # time series plot
    # winWellsDF,winInjDF = getWinWells(filteredDF,currentWellsDF,inj_df)
    # scenarioTSRDF,dPTimeSeriesR,wellIDsR,dayVecR = gistMC_instance.runPressureScenariosTimeSeries(eq,winWellsDF,winInjDF, verbose=2)
    # totalPPQuantilesDF = prepTotalPressureTimeSeriesQuantilesPlot(dPTimeSeriesR,dayVecR,nQuantiles=11,epoch=pd.to_datetime('1970-01-01'), )

    return smallPPDF, smallWellList, disaggregationDF


scratchPath = sys.argv[1]

# #instantiate the helper
helper = TexNetWebToolLaunchHelper(scratchPath)

#Get the args data out of it.
argsData = helper.argsData

#getParameterValueWithStepIndexAndParamName
Earthquake = helper.getParameterValueWithStepIndexAndParamName(0,"Earthquake").get("selectedRow").get("attributes")

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

realizationCount = helper.getParameterValueWithStepIndexAndParamName(1,"realizationCount")
rho0 = helper.getParameterValueWithStepIndexAndParamName(1,"rho0")
phi = helper.getParameterValueWithStepIndexAndParamName(1,"phi")
nta = helper.getParameterValueWithStepIndexAndParamName(1,"nta")
kMD = helper.getParameterValueWithStepIndexAndParamName(1,"kMD")
h = helper.getParameterValueWithStepIndexAndParamName(1,"h")
alphav = helper.getParameterValueWithStepIndexAndParamName(1,"alphav")
beta = helper.getParameterValueWithStepIndexAndParamName(1,"beta")

input = {
    "realizationCount": realizationCount,
    "porePressureParams" : {
        "rho0_min": rho0.get("min"),
        "rho0_max": rho0.get("max"),
        "nta_min": nta.get("min"),
        "nta_max": nta.get("max"),
        "phi_min": phi.get("min"),
        "phi_max": phi.get("max"),
        "kMD_min": kMD.get("min"),
        "kMD_max": kMD.get("max"),
        "h_min": h.get("min"),
        "h_max": h.get("max"),
        "alphav_min": alphav.get("min"),
        "alphav_max": alphav.get("max"),
        "beta_min": beta.get("min"),
        "beta_max": beta.get("max")
    },
    "eq": formattedEarthquake
}

smallPPDF, smallWellList, disaggregationDF = step2(input)


helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "smallPPDF", smallPPDF)
helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "smallWellList", smallWellList)
helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "disaggregationDF", disaggregationDF)
# helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "totalPPQuantilesDF", totalPPQuantilesDF)

helper.writeResultsFile()