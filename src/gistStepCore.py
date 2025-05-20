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
# from gistMC import prepTotalPressureTimeSeriesPlot

wellcsv = 'C:/texnetwebtools/tools/GIST/src/data/gist_well_data.csv'
injectioncsv = 'C:/texnetwebtools/tools/GIST/src/data/gist_injection_data.csv'

def runGistCore(input):
    # Initialize gistMC class
    gistMC_instance = gistMC()
    gistMC_instance.initPP()
    eq = input.get("eq")
    gistMC_instance.addWells(wellcsv, injectioncsv)
    forecastYears = input.get("years_diff")
    considered_wells_df, excluded_wells_df, inj_df = gistMC_instance.findWells(eq,PE=False, responseYears=forecastYears)

    # r-t plot combination of considered well and excluded wells df reference plots.py
    smallPPDF,smallWellList = prepRTPlot(considered_wells_df, excluded_wells_df, 1980, [0.1, 2], eq, True)

    # disaggregationPlot plot
    currentWellsDF=considered_wells_df[considered_wells_df['EncompassingDay']<0.].reset_index(drop=True)
    scenarioDF = gistMC_instance.runPressureScenarios(eq,currentWellsDF,inj_df)
    nWells=50

    # if scenarioDF is empty then we need to abort
    if scenarioDF.empty:
        return smallPPDF, smallWellList, scenarioDF, []

    dPCutoff=0.5
    filteredDF,orderedWellList = summarizePPResults(scenarioDF,currentWellsDF,dPCutoff,nOrder=nWells)
    if len(orderedWellList) > 20:
        dPCutoff=1
        filteredDF,orderedWellList = summarizePPResults(scenarioDF,currentWellsDF,dPCutoff,nOrder=nWells)
        if len(orderedWellList) > 20:
            dPCutoff=5
            filteredDF,orderedWellList = summarizePPResults(scenarioDF,currentWellsDF,dPCutoff,nOrder=nWells)
            if len(orderedWellList) > 20:
                dPCutoff=10
                filteredDF,orderedWellList = summarizePPResults(scenarioDF,currentWellsDF,dPCutoff,nOrder=nWells)
  
    disaggregationDF = prepDisaggregationPlot(filteredDF,orderedWellList,jitter=0.1)

    # time series plot
    # winWellsDF,winInjDF = getWinWells(filteredDF,currentWellsDF,inj_df)
    # scenarioTSRDF,dPTimeSeriesR,wellIDsR,dayVecR = gistMC_instance.runPressureScenariosTimeSeries(eq,winWellsDF,winInjDF, verbose=2)
    # totalPPQuantilesDF = prepTotalPressureTimeSeriesPlot(dPTimeSeriesR,dayVecR,nQuantiles=11,epoch=pd.to_datetime('1970-01-01'), )

    return smallPPDF, smallWellList, disaggregationDF, orderedWellList