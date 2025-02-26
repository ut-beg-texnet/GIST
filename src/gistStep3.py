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



def step3(input):
    # Validate Injection and Well Data
    GISTWells = {}
    GISTInjection = {}
    return GISTWells, GISTInjection


scratchPath = sys.argv[1]

# #instantiate the helper
helper = TexNetWebToolLaunchHelper(scratchPath)

#Get the args data out of it.
argsData = helper.argsData

#getParameterValueWithStepIndexAndParamName


GISTWells, GISTInjection = step3(input)

helper.saveDataFrameAsParameterWithStepIndexAndParamName(2, "smallPPDF", GISTWells)
helper.saveDataFrameAsParameterWithStepIndexAndParamName(2, "smallWellList", GISTInjection)

helper.writeResultsFile()