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

smallPPDF, smallWellList, disaggregationDF, orderedWellList = runGistCore(input)


helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "smallPPDF", smallPPDF)
helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "smallWellList", smallWellList)
helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "disaggregationDF", disaggregationDF)
# helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "totalPPQuantilesDF", totalPPQuantilesDF)


wellcsv = 'C:/texnetwebtools/tools/GIST/src/data/gist_well_data.csv'
injectioncsv = 'C:/texnetwebtools/tools/GIST/src/data/gist_injection_data.csv'

GISTWells = pd.read_csv(wellcsv)
GISTInjection = pd.read_csv(injectioncsv)
helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "GISTWells-corrections", GISTWells)
helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "GISTInjection-corrections", GISTInjection)

#Since step 0 doesnt have business logic set its sucess to true.
helper.setSuccessForStepIndex(0, True)
helper.setSuccessForStepIndex(1, True)

helper.writeResultsFile()