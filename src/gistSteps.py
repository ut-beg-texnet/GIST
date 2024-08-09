import requests
import json

from gistMC import initPP

# Step 1: User Selects Earthquake
# INPUT:
# {
#   eventId: '',
#   earthquakeDetails: {
#   ...
#   } 
# }

def step1(input):

    api_url = "https://maps.texnet.beg.utexas.edu/arcgis/rest/services/catalog/catalog_all/MapServer/0/query?where=EventId='{}'&outfields=*&f=pjson"
    if 'eventId' in input:
        response = requests.get(api_url.format(input.get('eventId')), verify=False)
        # Check if the request was successful
        if response.status_code == 200:
            earthquake_data =  json.loads(response.text).get('features')[0].get('attributes')
            return {
                'Magnitude': earthquake_data.get('Magnitude'),
                'Latitude': earthquake_data.get('Latitude'),
                'Longitude': earthquake_data.get('Longitude')
            }
        else:
            return false
    elif 'earthquakeDetails' in input:
        return input.get('earthquakeDetails')


# Step 2: Parameterize Subsurface
# INPUT:
# {
#   depthType: '',
#   realizationCount: '',
#   porePressureParams: {
#   ...
#   } 
# }

def step2(input):
    initPP()