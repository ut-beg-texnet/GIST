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
from gistMC import prepTotalPressureTimeSeriesPlot

from TexNetWebToolGPWrappers import TexNetWebToolLaunchHelper

from gistStepCore import runGistCore


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

realizationCount = helper.getParameterValueWithStepIndexAndParamName(3,"realizationCount")
rho0 = helper.getParameterValueWithStepIndexAndParamName(3,"rho0")
phi = helper.getParameterValueWithStepIndexAndParamName(3,"phi")
nta = helper.getParameterValueWithStepIndexAndParamName(3,"nta")
kMD = helper.getParameterValueWithStepIndexAndParamName(3,"kMD")
h = helper.getParameterValueWithStepIndexAndParamName(3,"h")
alphav = helper.getParameterValueWithStepIndexAndParamName(3,"alphav")
beta = helper.getParameterValueWithStepIndexAndParamName(3,"beta")

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

smallPPDF, smallWellList, disaggregationDF = runGistCore(input)


wellcsv = 'C:/texnetwebtools/tools/GIST/src/data/gist_well_data.csv'
# orderedWellList with proposed Future Rate initalize at 10000
originalWellDF = pd.read_csv(wellcsv)
orderedWellList = pd.DataFrame(orderedWellList, columns=['ID', 'Name'])
orderedWellList = orderedWellList.merge(
    originalWellDF[['ID', 'WellName', 'PermittedMaxLiquidBPD']],
    left_on='Name',
    right_on='WellName',
    how='left'
).drop(columns=['WellName', 'ID_x']).rename(columns={'ID_y': 'ID'})
orderedWellList['Proposed Future Rate (BPD)'] = np.where(
orderedWellList['PermittedMaxLiquidBPD'] < 10000,
orderedWellList['PermittedMaxLiquidBPD'],  # Use PermittedMaxLiquidBPD if it's less
10000  # Otherwise, use 10000
)
orderedWellListWithFutureRates = orderedWellList.drop(orderedWellList.index[-1])


helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "smallPPDF", smallPPDF)
helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "smallWellList", smallWellList)
helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "disaggregationDF", disaggregationDF)
# helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "totalPPQuantilesDF", totalPPQuantilesDF)
helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "orderedWellListWithFutureRates", orderedWellListWithFutureRates)

helper.writeResultsFile()