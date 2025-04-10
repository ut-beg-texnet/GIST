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

forecastDate = helper.getParameterValueWithStepIndexAndParamName(1,"forecastEndDate")

eq_date = datetime.strptime(formatted_date, "%Y-%m-%d")
future_date = datetime.strptime(forecastDate, "%Y-%m-%dT%H:%M:%S.%fZ")

days_diff = (future_date - eq_date).days
years_diff = days_diff / 365

realizationCount = helper.getParameterValueWithStepIndexAndParamName(3,"realizationCount")
rho0 = helper.getParameterValueWithStepIndexAndParamName(3,"rho0")
phi = helper.getParameterValueWithStepIndexAndParamName(3,"phi")
nta = helper.getParameterValueWithStepIndexAndParamName(3,"nta")
kMD = helper.getParameterValueWithStepIndexAndParamName(3,"kMD")
h = helper.getParameterValueWithStepIndexAndParamName(3,"h")
alphav = helper.getParameterValueWithStepIndexAndParamName(3,"alphav")
beta = helper.getParameterValueWithStepIndexAndParamName(3,"beta")

input = {
    "years_diff": years_diff,
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

smallPPDF, smallWellList, disaggregationDF, orderedWellList = runGistCore(input)

wellcsv = 'C:/texnetwebtools/tools/GIST/src/data/gist_well_data.csv'
# orderedWellList with proposed Future Rate initalize at 10000
originalWellDF = pd.read_csv(wellcsv)
orderedWellList = pd.DataFrame(orderedWellList, columns=['ID'])
orderedWellList = orderedWellList.merge(
    originalWellDF[['ID', 'WellName', 'PermittedMaxLiquidBPD']],
    left_on='ID',
    right_on='ID',
    how='right'
)
orderedWellList['Proposed Future Rate (BPD)'] = np.where(
orderedWellList['PermittedMaxLiquidBPD'] < 10000,
orderedWellList['PermittedMaxLiquidBPD'],  # Use PermittedMaxLiquidBPD if it's less
10000  # Otherwise, use 10000
)
orderedWellListWithFutureRates = orderedWellList.drop(orderedWellList.index[-1])

helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "smallPPDF_updated", smallPPDF)
helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "smallWellList_updated", smallWellList)
helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "disaggregationDF_updated", disaggregationDF)
# helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "totalPPQuantilesDF", totalPPQuantilesDF)
helper.saveDataFrameAsParameterWithStepIndexAndParamName(3, "orderedWellListWithFutureRates", orderedWellListWithFutureRates)

helper.setSuccessForStepIndex(2, True)
helper.setSuccessForStepIndex(3, True)

helper.writeResultsFile()