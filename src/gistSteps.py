import urllib3
import requests
import json
import numpy as np
from pandas import Timestamp
from pandas import DataFrame

from gistMC import gistMC

urllib3.disable_warnings()


class jsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, DataFrame):
            return obj.to_dict()
        if isinstance(obj, Timestamp):
            return str(obj)
        return super().default(obj)


"""
################################################### STEP 1 ###################################################
#
# User selects an earthquake by either providing a coordinate and date or through an existing texnet event ID.
# updated the session earthquake details with the information retrieved from the arcgis service.
#
##############################################################################################################
"""

def step1(input, session):

    api_url = "https://maps.texnet.beg.utexas.edu/arcgis/rest/services/catalog/catalog_all/MapServer/0/query?where=EventId='{}'&outfields=*&f=pjson"
    if 'eventId' in input:
        response = requests.get(api_url.format(input.get('eventId')), verify=False)
        # Check if the request was successful
        if response.status_code == 200:
            earthquake_data =  json.loads(response.text).get('features')[0].get('attributes')
            earthquakeDetails = {
                'Latitude': earthquake_data.get('Latitude'),
                'LatitudeError': earthquake_data.get('LatitudeError'),
                'Longitude': earthquake_data.get('Longitude'),
                'LongitudeError': earthquake_data.get('LongitudeError'),
                'Origin Date': earthquake_data.get('Event_Date'),
                'EventID': earthquake_data.get('EventId')
            }
        else:
            return False
    elif 'earthquakeDetails' in input:
        earthquakeDetails = input.get('earthquakeDetails')
    else:
        return False
    
    session["sessionValues"]["earthquakeDetails"] = earthquakeDetails

    return session


"""
################################################### STEP 2 ###################################################
#
# Initialize GIST instance using user input parameters
# Parameters:
# Well classifcation (deep vs shallow), Number of realizations and subsurface parameters (see gistMC.py initPP for details)
# Save GIST instance within the session json object
#
##############################################################################################################
"""


def step2(input, session):
    # Initialize gistMC class
    gistMC_instance = gistMC(nReal=input.get('realizationCount'))
    gistMC_instance.initPP(**input.get('porePressureParams'))
    wellcsv = './gist_well_data.csv'
    injectioncsv = './gist_injection_data.csv'
    gistMC_instance.addWells(wellcsv, injectioncsv, verbose=1)
    eq = session["sessionValues"]["earthquakeDetails"]
    print('Here is the earthquake details: ' + str(eq))
    considered_wells_df, excluded_wells_df, inj_df = gistMC_instance.findWells(eq, verbose=1)
    # r-t plot combination of considered well and excluded wells df reference plots.py
    gistMC_instance.runPressureScenarios(eq, considered_wells_df, inj_df, verbose=1)

    obj = gistMC_instance.to_dict()
    json_str = json.dumps(obj, cls=jsonEncoder)

    session["gistInstance"] = json_str
    session["sessionValues"]["depthType"] = input.get('depthType')
    return session


"""
################################################### STEP 3 ###################################################
#
# Initialize GIST instance using user input parameters
# Parameters:
# Well classifcation (deep vs shallow), Number of realizations and subsurface parameters (see gistMC.py initPP for details)
# Save GIST instance within the session json object
#
##############################################################################################################
"""


# addWells, findWells, runPressureScenarios
def step3(input, session):
    # wellcsv = './gist_well_data.csv'
    # injectioncsv = './gist_injection_data.csv'
    # prevSession = json.loads(session["gistInstance"])
    # gistMC_instance = gistMC(pSession=prevSession)
    # gistMC_instance.addWells(wellcsv, injectioncsv, verbose=1)

    # eq = session["sessionValues"]["earthquakeDetails"]
    # print('Here is the earthquake details: ' + str(eq))
    # considered_wells_df, excluded_wells_df, inj_df = gistMC_instance.findWells(eq, verbose=1)
    # gistMC_instance.runPressureScenarios(eq, considered_wells_df, inj_df, verbose=1)

    #save gist instance after add and find wells
    # obj = gistMC_instance.to_dict()
    # json_str = json.dumps(obj, cls=jsonEncoder)

    # session["gistInstance"] = json_str
    return session

#Determine List of Contributing Wells (Auto Filtering)
def step4(input, session):
    return input

#QC Well Data (manual editing)
def step5(input, session):
    return input

#Rerun
def step6(input, session):
    return input

#Review Rerun
def step7(input, session):
    return input

#Planning
def step8(input, session):
    return input

#Review Planning Rerun
def step9(input, session):
    return input