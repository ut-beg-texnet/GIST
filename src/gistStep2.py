import urllib3
import requests
import json
import numpy as np
from pandas import Timestamp
from pandas import DataFrame
import sys
import os
import time
import pathlib

from gistMC import gistMC
from gistMC import prepRTPlot
from gistMC import prepDisaggregationPlot

from TexNetWebToolGPWrappers import TexNetWebToolLaunchHelper

def step2(input):
    # Initialize gistMC class
    gistMC_instance = gistMC()
    gistMC_instance.initPP()
    eq = input.get("eq")
    wellcsv = 'C:/Users/peter/Documents/workspace/GIST/src/gist_well_data.csv'
    injectioncsv = 'C:/Users/peter/Documents/workspace/GIST/src/gist_injection_data.csv'
    gistMC_instance.addWells(wellcsv, injectioncsv)
    considered_wells_df, excluded_wells_df, inj_df = gistMC_instance.findWells(eq,PE=False)

    # r-t plot combination of considered well and excluded wells df reference plots.py
    smallPPDF,smallWellList = prepRTPlot(considered_wells_df, excluded_wells_df, 1980, [0.1, 2], eq, True)

    # disaggregationPlotDF = prepDisaggregationPlot(smallPPDF, smallWellList)

    return smallPPDF, smallWellList


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

print(f"test: {input}")

smallPPDF, smallWellList = step2(input)

print(f"test: {smallPPDF} {smallWellList}")

helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "smallPPDF", smallPPDF)
helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "smallWellList", smallWellList)

helper.writeResultsFile()