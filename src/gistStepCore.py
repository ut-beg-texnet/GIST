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
from progress import report_progress

def runGistCore(input, wellcsv, injectioncsv):
    # Initialize gistMC class
    gistMC_instance = gistMC()
    porePressureParams = input.get("porePressureParams")
    gistMC_instance.initPP(**porePressureParams)
    eq = input.get("eq")
    report_progress("Loading well and injection data")
    gistMC_instance.addWells(wellcsv, injectioncsv)
    forecastYears = input.get("years_diff")

    report_progress("Finding nearby wells")

    considered_wells_df, excluded_wells_df, inj_df = gistMC_instance.findWellsVec(eq,PE=False, responseYears=forecastYears)
    if 'Date' not in inj_df.columns:
        if 'Days' not in inj_df.columns:
            raise KeyError("inj_df is missing required columns: 'Date' or 'Days'")
        inj_df['Date'] = pd.to_datetime('1970-01-01') + pd.to_timedelta(inj_df['Days'], unit='d')
    else:
        inj_df['Date'] = pd.to_datetime(inj_df['Date'])

    report_progress("Preparing data for R-t plot")

    # r-t plot combination of considered well and excluded wells df reference plots.py
    smallPPDF,smallWellList = prepRTPlot(considered_wells_df, excluded_wells_df, 1980, [gistMC_instance.diffPPMin, gistMC_instance.diffPPMax], eq, True)

    report_progress("Running pressure scenarios")

    # disaggregationPlot plot
    currentWellsDF=considered_wells_df[considered_wells_df['EncompassingDay']<0.].reset_index(drop=True)
    scenarioDF = gistMC_instance.runPressureScenariosVec(eq,currentWellsDF,inj_df)
    nWells=50

    # if scenarioDF is empty then we need to abort
    if scenarioDF.empty:
        return smallPPDF, smallWellList, scenarioDF, [], [], [] ,[], [], pd.DataFrame()

    for dPCutoff in [0.5, 1, 5, 10]:
        filteredDF, orderedWellList = summarizePPResults(scenarioDF, currentWellsDF, dPCutoff, nOrder=nWells)
        if len(orderedWellList) <= 20:
            break
  
    disaggregationDF = prepDisaggregationPlot(filteredDF,orderedWellList,jitter=0.1)

    report_progress("Building pressure time series")

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

    # Preserve the existing "last duplicate well name wins" behavior from
    # prepPressureAndDisposalTimeSeriesPlots without repeatedly slicing
    # and copying the full spaghetti dataframe one well at a time.
    selected_wells = []
    seen_subgraphs = set()
    winWellsIndexedDF = winWellsDF.set_index('ID', drop=False)
    for wellID in reversed(orderedWellList):
        if wellID not in winWellsIndexedDF.index:
            continue
        wellRow = winWellsIndexedDF.loc[wellID]
        if isinstance(wellRow, pd.DataFrame):
            wellRow = wellRow.iloc[0]
        wellName = str(wellRow.get('WellName', wellID))
        if wellName in seen_subgraphs:
            continue
        seen_subgraphs.add(wellName)
        selected_wells.append({"WellID": wellID, "subgraph": wellName})
    selected_wells.reverse()

    if selected_wells:
        selectedWellsDF = pd.DataFrame(selected_wells)
        injectionStartDF = (
            winInjDF.loc[winInjDF['Date'].notnull(), ['ID', 'Date']]
            .groupby('ID', as_index=False)['Date']
            .min()
            .rename(columns={'ID': 'WellID', 'Date': 'InjectionStartDate'})
        )
        selectedWellsDF = selectedWellsDF.merge(injectionStartDF, on='WellID', how='left')

        allPerWellPPQuantilesDF = allPerWellPPQuantilesDF.merge(selectedWellsDF, on='WellID', how='inner')
        allPerWellPPQuantilesDF = allPerWellPPQuantilesDF[
            allPerWellPPQuantilesDF['InjectionStartDate'].isna()
            | (allPerWellPPQuantilesDF['Date'] > allPerWellPPQuantilesDF['InjectionStartDate'])
        ].drop(columns=['InjectionStartDate'])

        allPerWellPPSpaghettiDF = allPerWellPPSpaghettiDF.merge(selectedWellsDF, on='WellID', how='inner')
        allPerWellPPSpaghettiDF = allPerWellPPSpaghettiDF[
            allPerWellPPSpaghettiDF['InjectionStartDate'].isna()
            | (allPerWellPPSpaghettiDF['Date'] > allPerWellPPSpaghettiDF['InjectionStartDate'])
        ].drop(columns=['InjectionStartDate'])

        allPerWellDisposalDF = winInjDF.merge(
            selectedWellsDF[['WellID', 'subgraph']],
            left_on='ID',
            right_on='WellID',
            how='inner',
        ).drop(columns=['WellID'])
    else:
        allPerWellPPQuantilesDF = pd.DataFrame(columns=['DeltaPressure', 'Days', 'Realization', 'Order', 'WellID', 'Percentile', 'Date', 'subgraph'])
        allPerWellPPSpaghettiDF = pd.DataFrame(columns=['DeltaPressure', 'Days', 'Realization', 'WellID', 'Diffusivity', 'Date', 'subgraph'])
        allPerWellDisposalDF = pd.DataFrame(columns=['ID', 'Days', 'BPD', 'Date', 'subgraph'])
    
    allPerWellPPQuantilesDF['timestamp'] = pd.to_datetime(allPerWellPPQuantilesDF['Date'], format='%m/%d/%Y').view('int64') // 10**6
    allPerWellPPQuantilesDF = allPerWellPPQuantilesDF.sort_values('timestamp').reset_index(drop=True)

    allPerWellPPSpaghettiDF['timestamp'] = pd.to_datetime(allPerWellPPSpaghettiDF['Date'], format='%m/%d/%Y').view('int64') // 10**6
    allPerWellPPSpaghettiDF = allPerWellPPSpaghettiDF.sort_values('timestamp').reset_index(drop=True)

    allPerWellDisposalDF['timestamp'] = pd.to_datetime(allPerWellDisposalDF['Date'], format='%m/%d/%Y').view('int64') // 10**6
    allPerWellDisposalDF = allPerWellDisposalDF.sort_values('timestamp').reset_index(drop=True)
     

    return smallPPDF, smallWellList, disaggregationDF, orderedWellList, totalPPQuantilesDF, totalPPSpaghettiDF, allPerWellPPQuantilesDF, allPerWellPPSpaghettiDF, allPerWellDisposalDF
