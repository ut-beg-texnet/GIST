import urllib3
import requests
import json
import numpy as np
from pandas import Timestamp
from datetime import datetime
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

DEFAULT_REALIZATION_COUNT = 50
# Catalog earthquakes sometimes omit location error; use 1 km so checkEQ/findWellsVec get a number. For eventType 'Earthquake' we default to 1 km, 'Scenario' we default to 0 km.
DEFAULT_EARTHQUAKE_LOCATION_ERROR_KM = 1


def eq_location_error_km(value, missing_default=0):
    """Return a numeric location error in km; use missing_default when the catalog value is absent."""
    if value is None or value == "":
        return missing_default
    return value


def get_selected_event(helper):
    """Return the event selected in the portal as a GIST event dictionary."""
    event_type = helper.getParameterValueWithStepIndexAndParamName(0, "eventType")
    if event_type == "Earthquake":
        selected_event = helper.getParameterValueWithStepIndexAndParamName(0, "Earthquake")
        attributes = (selected_event or {}).get("selectedRow", {}).get("attributes")
        if attributes is None:
            raise ValueError("Select an earthquake before running GIST.")
        event_date = Timestamp(attributes.get("Origin Date"), unit="ms")
        return {
            "Latitude": attributes.get("Latitude (WGS84)"),
            "LatitudeError": eq_location_error_km(
                attributes.get("Latitude Error (km)"), DEFAULT_EARTHQUAKE_LOCATION_ERROR_KM
            ),
            "Longitude": attributes.get("Longitude (WGS84)"),
            "LongitudeError": eq_location_error_km(
                attributes.get("Longitude Error (km)"), DEFAULT_EARTHQUAKE_LOCATION_ERROR_KM
            ),
            "Origin Date": event_date.strftime("%Y-%m-%d"),
            "EventID": attributes.get("EventID"),
        }
    if event_type == "Scenario":
        scenario_loc = helper.getParameterValueWithStepIndexAndParamName(0, "scenarioLoc")
        scenario_date = helper.getParameterValueWithStepIndexAndParamName(0, "scenarioDate")
        if scenario_loc is None or scenario_date is None:
            raise ValueError("Select a scenario location and date before running GIST.")
        event_date = Timestamp(scenario_date, unit="ms")
        return {
            "Latitude": scenario_loc.get("y"), "LatitudeError": 0,
            "Longitude": scenario_loc.get("x"), "LongitudeError": 0,
            "Origin Date": event_date.strftime("%Y-%m-%d"), "EventID": "AAAAAA",
        }
    raise ValueError("Select either an earthquake or a scenario before running GIST.")


def get_portal_analysis_input(helper, step_index):
    """Build independent GIST model input from one portal step."""
    event = get_selected_event(helper)
    # Forecast end date is shared by every analysis step from Select Event.
    forecast_date = helper.getParameterValueWithStepIndexAndParamName(0, "forecastEndDate")
    if forecast_date is None:
        raise ValueError("Provide a forecast end date before running GIST.")
    try:
        future_date = datetime.strptime(forecast_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        future_date = datetime.fromisoformat(str(forecast_date).replace("Z", "+00:00").replace("+00:00", ""))
    event_date = datetime.strptime(event["Origin Date"], "%Y-%m-%d")
    parameter_names = ("realizationCount", "wellType", "rho0", "phi", "nta", "kMD", "h", "cppMS", "betaMS")
    values = {name: helper.getParameterValueWithStepIndexAndParamName(step_index, name) for name in parameter_names}
    if any(values[name] is None for name in parameter_names):
        raise ValueError("Complete all Updated Analysis inputs before running GIST.")
    return {
        "years_diff": (future_date - event_date).days / 365,
        "realizationCount": values["realizationCount"], "wellType": values["wellType"],
        "porePressureParams": {
            "rho0_min": float(values["rho0"].get("min")), "rho0_max": float(values["rho0"].get("max")),
            "nta_min": float(values["nta"].get("min")), "nta_max": float(values["nta"].get("max")),
            "phi_min": float(values["phi"].get("min")), "phi_max": float(values["phi"].get("max")),
            "kMD_min": float(values["kMD"].get("min")), "kMD_max": float(values["kMD"].get("max")),
            "h_min": float(values["h"].get("min")), "h_max": float(values["h"].get("max")),
            "cppMS_min": float(values["cppMS"].get("min")), "cppMS_max": float(values["cppMS"].get("max")),
            "betaMS_min": float(values["betaMS"].get("min")), "betaMS_max": float(values["betaMS"].get("max")),
        }, "eq": event,
    }


def get_corrected_gist_data_paths(helper):
    """Return the corrected disposal datasets required by later GIST steps."""
    well_csv = helper.getDatasetFilePathWithStepIndexAndParamName(2, "GISTWells")
    injection_csv = helper.getDatasetFilePathWithStepIndexAndParamName(2, "GISTInjection")
    if well_csv is None or injection_csv is None:
        raise ValueError("Upload both corrected GIST well and injection datasets before running Updated Analysis.")
    return well_csv, injection_csv


def _resolve_realization_count(raw_count):
    """Return a non-negative int realization count, or the portal default of 50.

    Missing, unparseable, and negative values fall back to DEFAULT_REALIZATION_COUNT.
    Values 0 and 1 are passed through; gistMC still rejects nReal < 2.
    """
    if raw_count is None:
        return DEFAULT_REALIZATION_COUNT
    try:
        n_real = int(raw_count)
    except (TypeError, ValueError):
        return DEFAULT_REALIZATION_COUNT
    if n_real < 0:
        return DEFAULT_REALIZATION_COUNT
    return n_real


def runGistCore(input, wellcsv, injectioncsv):
    """Run well filtering and pore-pressure Monte Carlo for a portal GIST step.
    """
    # Initialize gistMC class
    n_real = _resolve_realization_count(input.get("realizationCount"))
    gistMC_instance = gistMC(nReal=n_real)
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
