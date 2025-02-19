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
    wellcsv = 'C:/Users/peter/Documents/workspace/GIST/src/gist_well_data.csv'
    injectioncsv = 'C:/Users/peter/Documents/workspace/GIST/src/gist_injection_data.csv'
    gistMC_instance.addWells(wellcsv, injectioncsv)
    eq = {
            "Latitude": 32.14416504,
            "LatitudeError": 0.17663684,
            "Longitude": -101.80521151,
            "LongitudeError": 0.18082184,
            "Origin Date": "2024-07-27",
            "EventID": "texnet2024oqfb"
    }
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
realizationCount = helper.getParameterValueWithStepIndexAndParamName(1,"realizationCount")
# rho0_min = helper.getParameterValueWithStepIndexAndParamName(1,"rho0_min")
# rho0_max = helper.getParameterValueWithStepIndexAndParamName(1,"rho0_max")
# print(f"test: {realizationCount} {rho0_min} {rho0_max}")

# input = {
#     "realizationCount": realizationCount,
#     "porePressureParams" : {
#         "rho0_min": rho0_min,
#         "rho0_max": rho0_max
#     }
# }
input = {
    "realizationCount": realizationCount,
    "porePressureParams" : {}
}

smallPPDF, smallWellList = step2(input)

print(f"test: {smallPPDF} {smallWellList}")

helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "smallPPDF", smallPPDF)
helper.saveDataFrameAsParameterWithStepIndexAndParamName(1, "smallWellList", smallWellList)

helper.writeResultsFile()