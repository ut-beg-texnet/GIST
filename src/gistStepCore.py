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
from gistMC import prepTotalPressureTimeSeriesSpaghettiPlot
from gistMC import getPerWellPressureTimeSeriesSpaghettiAndQuantiles

def runGistCore(input, wellcsv, injectioncsv):
    # Initialize gistMC class
    gistMC_instance = gistMC()
    porePressureParams = input.get("porePressureParams")
    gistMC_instance.initPP(**porePressureParams)
    eq = input.get("eq")
    gistMC_instance.addWells(wellcsv, injectioncsv)
    forecastYears = input.get("years_diff")
    considered_wells_df, excluded_wells_df, inj_df = gistMC_instance.findWells(eq,PE=False, responseYears=forecastYears)

    # r-t plot combination of considered well and excluded wells df reference plots.py
    smallPPDF,smallWellList = prepRTPlot(considered_wells_df, excluded_wells_df, 1980, [gistMC_instance.diffPPMin, gistMC_instance.diffPPMax], eq, True)

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
    winWellsDF,winInjDF = getWinWells(filteredDF,currentWellsDF,inj_df)
    scenarioTSRDF,dPTimeSeriesR,wellIDsR,dayVecR = gistMC_instance.runPressureScenariosTimeSeries(eq,winWellsDF,winInjDF, verbose=2)
    totalPPQuantilesDF = prepTotalPressureTimeSeriesQuantilesPlot(dPTimeSeriesR,dayVecR,nQuantiles=11,epoch=pd.to_datetime('1970-01-01'))
    totalPPSpaghettiDF = prepTotalPressureTimeSeriesSpaghettiPlot(dPTimeSeriesR,dayVecR,gistMC_instance.diffPPVec,epoch=pd.to_datetime('1970-01-01'))
    # add unix timestamp
    totalPPQuantilesDF['timestamp'] = pd.to_datetime(totalPPQuantilesDF['Date'], format='%m/%d/%Y').view('int64') // 10**6
    totalPPQuantilesDF = totalPPQuantilesDF.sort_values('timestamp').reset_index(drop=True)

    totalPPSpaghettiDF['timestamp'] = pd.to_datetime(totalPPSpaghettiDF['Date'], format='%m/%d/%Y').view('int64') // 10**6
    totalPPSpaghettiDF = totalPPSpaghettiDF.sort_values('timestamp').reset_index(drop=True)

    allPerWellPPQuantilesDF,allPerWellPPSpaghettiDF = getPerWellPressureTimeSeriesSpaghettiAndQuantiles(dPTimeSeriesR,dayVecR,gistMC_instance.diffPPVec,wellIDsR,nQuantiles=11,epoch=pd.to_datetime('01-01-1970'))

    # combine well name and well id to make the subgraph column needed for graph filtering. drop the unused columns.
    allPerWellPPQuantilesDF['subgraph'] = allPerWellPPQuantilesDF['WellID'].map(
        smallWellList.set_index('ID').apply(lambda row: f"{row['WellName']} ({row.name})", axis=1)
    )

    allPerWellPPQuantilesDF['timestamp'] = pd.to_datetime(allPerWellPPQuantilesDF['Date'], format='%m/%d/%Y').view('int64') // 10**6
    allPerWellPPQuantilesDF = allPerWellPPQuantilesDF.sort_values('timestamp').reset_index(drop=True)

    allPerWellPPQuantilesDF = allPerWellPPQuantilesDF.drop(columns=['Days', 'Realization', 'Order', 'Date', 'WellID'])

    allPerWellPPSpaghettiDF['subgraph'] = allPerWellPPSpaghettiDF['WellID'].map(
        smallWellList.set_index('ID').apply(lambda row: f"{row['WellName']} ({row.name})", axis=1)
    )

    allPerWellPPSpaghettiDF['timestamp'] = pd.to_datetime(allPerWellPPSpaghettiDF['Date'], format='%m/%d/%Y').view('int64') // 10**6
    allPerWellPPSpaghettiDF = allPerWellPPSpaghettiDF.sort_values('timestamp').reset_index(drop=True)

    allPerWellPPSpaghettiDF = allPerWellPPSpaghettiDF.drop(columns=['Days', 'Realization', 'Date', 'WellID'])
     

    return smallPPDF, smallWellList, disaggregationDF, orderedWellList, totalPPQuantilesDF, totalPPSpaghettiDF, allPerWellPPQuantilesDF, allPerWellPPSpaghettiDF