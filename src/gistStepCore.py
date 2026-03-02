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
from gistMC import prepPressureAndDisposalTimeSeriesPlots

def runGistCore(input, wellcsv, injectioncsv):
    # Initialize gistMC class
    gistMC_instance = gistMC()
    porePressureParams = input.get("porePressureParams")
    gistMC_instance.initPP(**porePressureParams)
    eq = input.get("eq")
    gistMC_instance.addWells(wellcsv, injectioncsv)
    forecastYears = input.get("years_diff")

    print("Info: Finding Wells")

    considered_wells_df, excluded_wells_df, inj_df = gistMC_instance.findWellsVec(eq,PE=False, responseYears=forecastYears)
    if 'Date' not in inj_df.columns:
        if 'Days' not in inj_df.columns:
            raise KeyError("inj_df is missing required columns: 'Date' or 'Days'")
        inj_df['Date'] = pd.to_datetime('1970-01-01') + pd.to_timedelta(inj_df['Days'], unit='d')
    else:
        inj_df['Date'] = pd.to_datetime(inj_df['Date'])

    print("Info: Generating r-t Plot")

    # r-t plot combination of considered well and excluded wells df reference plots.py
    smallPPDF,smallWellList = prepRTPlot(considered_wells_df, excluded_wells_df, 1980, [gistMC_instance.diffPPMin, gistMC_instance.diffPPMax], eq, True)

    print("Info: Generating Disaggregation Plot")

    # disaggregationPlot plot
    currentWellsDF=considered_wells_df[considered_wells_df['EncompassingDay']<0.].reset_index(drop=True)
    scenarioDF = gistMC_instance.runPressureScenariosVec(eq,currentWellsDF,inj_df)
    nWells=50

    # if scenarioDF is empty then we need to abort
    if scenarioDF.empty:
        return smallPPDF, smallWellList, scenarioDF, [], [], [] ,[], []

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

    print("Info: Generating Time Series Plots")

    # time series plot
    winWellsDF,winInjDF = getWinWells(filteredDF,currentWellsDF,inj_df)
    scenarioTSRDF,dPTimeSeriesR,wellIDsR,dayVecR = gistMC_instance.runPressureScenariosTimeSeriesConv(eq,winWellsDF,winInjDF, verbose=2)
    totalPPQuantilesDF = prepTotalPressureTimeSeriesQuantilesPlot(dPTimeSeriesR,dayVecR,nQuantiles=11,epoch=pd.to_datetime('1970-01-01'))
    totalPPSpaghettiDF = prepTotalPressureTimeSeriesSpaghettiPlot(dPTimeSeriesR,dayVecR,gistMC_instance.diffPPVec,epoch=pd.to_datetime('1970-01-01'))
    # add unix timestamp
    totalPPQuantilesDF['timestamp'] = pd.to_datetime(totalPPQuantilesDF['Date'], format='%m/%d/%Y').view('int64') // 10**6
    totalPPQuantilesDF = totalPPQuantilesDF.sort_values('timestamp').reset_index(drop=True)

    totalPPSpaghettiDF['timestamp'] = pd.to_datetime(totalPPSpaghettiDF['Date'], format='%m/%d/%Y').view('int64') // 10**6
    totalPPSpaghettiDF = totalPPSpaghettiDF.sort_values('timestamp').reset_index(drop=True)

    allPerWellPPQuantilesDF,allPerWellPPSpaghettiDF = getPerWellPressureTimeSeriesSpaghettiAndQuantiles(dPTimeSeriesR,dayVecR,gistMC_instance.diffPPVec,wellIDsR,nQuantiles=11,epoch=pd.to_datetime('01-01-1970'))

    wellPressureDict = prepPressureAndDisposalTimeSeriesPlots(
        allPerWellPPQuantilesDF,
        allPerWellPPSpaghettiDF,
        winWellsDF,
        winInjDF,
        orderedWellList,
        verbose=0
    )

    # Build per-well dataframes with a subgraph key from the dict keys.
    perWellQuantiles = []
    perWellSpaghetti = []
    perWellDisposal = []
    for wellKey, wellDict in wellPressureDict.items():
        quantilesDF = wellDict.get('PPQuantiles')
        if isinstance(quantilesDF, pd.DataFrame) and not quantilesDF.empty:
            quantilesDF = quantilesDF.copy()
            quantilesDF['subgraph'] = wellKey
            perWellQuantiles.append(quantilesDF)

        spaghettiDF = wellDict.get('Spaghetti')
        if isinstance(spaghettiDF, pd.DataFrame) and not spaghettiDF.empty:
            spaghettiDF = spaghettiDF.copy()
            spaghettiDF['subgraph'] = wellKey
            perWellSpaghetti.append(spaghettiDF)

        disposalDF = wellDict.get('Disposal')
        if isinstance(disposalDF, pd.DataFrame) and not disposalDF.empty:
            disposalDF = disposalDF.copy()
            disposalDF['subgraph'] = wellKey
            perWellDisposal.append(disposalDF)

    if len(perWellQuantiles) > 0:
        allPerWellPPQuantilesDF = pd.concat(perWellQuantiles, ignore_index=True)
    if len(perWellSpaghetti) > 0:
        allPerWellPPSpaghettiDF = pd.concat(perWellSpaghetti, ignore_index=True)
    if len(perWellDisposal) > 0:
        allPerWellDisposalDF = pd.concat(perWellDisposal, ignore_index=True)
    else:
        allPerWellDisposalDF = pd.DataFrame(columns=['ID', 'Days', 'BPD', 'Date', 'subgraph'])
    
    allPerWellPPQuantilesDF['timestamp'] = pd.to_datetime(allPerWellPPQuantilesDF['Date'], format='%m/%d/%Y').view('int64') // 10**6
    allPerWellPPQuantilesDF = allPerWellPPQuantilesDF.sort_values('timestamp').reset_index(drop=True)

    allPerWellPPSpaghettiDF['timestamp'] = pd.to_datetime(allPerWellPPSpaghettiDF['Date'], format='%m/%d/%Y').view('int64') // 10**6
    allPerWellPPSpaghettiDF = allPerWellPPSpaghettiDF.sort_values('timestamp').reset_index(drop=True)

    allPerWellDisposalDF['timestamp'] = pd.to_datetime(allPerWellDisposalDF['Date'], format='%m/%d/%Y').view('int64') // 10**6
    allPerWellDisposalDF = allPerWellDisposalDF.sort_values('timestamp').reset_index(drop=True)
     

    return smallPPDF, smallWellList, disaggregationDF, orderedWellList, totalPPQuantilesDF, totalPPSpaghettiDF, allPerWellPPQuantilesDF, allPerWellPPSpaghettiDF, allPerWellDisposalDF