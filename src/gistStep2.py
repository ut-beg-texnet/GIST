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

realizationCount = helper.getParameterValueWithStepIndexAndParamName(1,"realizationCount")
wellType = helper.getParameterValueWithStepIndexAndParamName(1,"wellType")
rho0 = helper.getParameterValueWithStepIndexAndParamName(1,"rho0")
phi = helper.getParameterValueWithStepIndexAndParamName(1,"phi")
nta = helper.getParameterValueWithStepIndexAndParamName(1,"nta")
kMD = helper.getParameterValueWithStepIndexAndParamName(1,"kMD")
h = helper.getParameterValueWithStepIndexAndParamName(1,"h")
alphav = helper.getParameterValueWithStepIndexAndParamName(1,"alphav")
beta = helper.getParameterValueWithStepIndexAndParamName(1,"beta")

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
        "alphav_min": float(alphav.get("min")),
        "alphav_max": float(alphav.get("max")),
        "beta_min": float(beta.get("min")),
        "beta_max": float(beta.get("max"))
    },
    "eq": formattedEarthquake
}

if wellType == 'Shallow':
    wellcsv = 'C:/texnetwebtools/tools/GIST/src/data/gist_well_shallow.csv'
    injectioncsv = 'C:/texnetwebtools/tools/GIST/src/data/gist_injection_shallow.csv'
else:
    wellcsv = 'C:/texnetwebtools/tools/GIST/src/data/gist_well_deep.csv'
    injectioncsv = 'C:/texnetwebtools/tools/GIST/src/data/gist_injection_deep.csv'

smallPPDF, smallWellList, disaggregationDF, orderedWellList, totalPPQuantilesDF, totalPPSpaghettiDF = runGistCore(input, wellcsv, injectioncsv)

if disaggregationDF.empty:
    helper.addMessageWithStepIndex(1, "No Wells Found.", 2)
    helper.setSuccessForStepIndex(1, False)
else:
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "smallPPDF", smallPPDF)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "smallWellList", smallWellList)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "disaggregationDF", disaggregationDF)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "totalPPQuantilesDF", totalPPQuantilesDF)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "totalPPSpaghettiDF", totalPPSpaghettiDF)

    GISTWells = pd.read_csv(wellcsv)
    GISTInjection = pd.read_csv(injectioncsv)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "GISTWells-corrections", GISTWells)
    helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "GISTInjection-corrections", GISTInjection)

    #Since step 0 doesnt have business logic set its sucess to true.
    helper.setSuccessForStepIndex(0, True)
    helper.setSuccessForStepIndex(1, True)

helper.writeResultsFile()