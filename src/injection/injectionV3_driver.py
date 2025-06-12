from io import StringIO
import scipy.special as sc
import numpy as np
import pandas as pd
import math
import injectionV3 as inj3
import requests
requests.packages.urllib3.disable_warnings(
    requests.packages.urllib3.exceptions.InsecureRequestWarning)
import json
import csv
from datetime import datetime
import urllib3
import credentials
# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

now = datetime.now()

# ====================================================================================
# ============================== Driver for injectionV3 ==============================
# ====================================================================================


# ============================== STEP 1: FETCH LATEST CSV FROM DISPOSAL SERVICE ==============================

def authenticate(username, password, auth_url):
    try:
        response = requests.post(auth_url, json={"username": username, "password": password}, verify=False)
        response.raise_for_status()  # Raise an exception for HTTP errors
        token = response.json()["Token"]
        return token
    except requests.exceptions.RequestException as e:
        print(f"Error authenticating: {e}")
        return None

def fetch_data_and_save_csv(api_url, output_file, token, method='GET', data=None, json=None, params=None):
    """
    Fetch data from API and save as CSV file.
    
    Parameters:
    - api_url: str, the API endpoint URL
    - output_file: str, path to save the CSV file
    - token: str, authorization token
    - method: str, HTTP method ('GET' or 'POST'), default 'GET'
    - data: dict or str, data to send in POST request body (form data)
    - json: dict, JSON data to send in POST request body
    - params: dict, URL parameters for GET requests
    
    Returns:
    - str: The response data, or None if error occurred
    """
    try:
        headers = {"Authorization": f"Bearer {token}"}
        
        # Add Content-Type header for POST requests with JSON data
        if method.upper() == 'POST' and json is not None:
            headers["Content-Type"] = "application/json"
        
        # Make the appropriate HTTP request
        if method.upper() == 'GET':
            response = requests.get(api_url, headers=headers, params=params, verify=False)
        elif method.upper() == 'POST':
            response = requests.post(api_url, headers=headers, data=data, json=json, params=params, verify=False)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}. Use 'GET' or 'POST'.")
        
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.text
        
        # Save the CSV data to file
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvfile.write(data)
        
        print(f"CSV data successfully saved to {output_file}")
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


# Authenticate and get token
auth_url = "https://injection.texnet.beg.utexas.edu/api/Users/Authenticate"
username = credentials.USERNAME
password = credentials.PASSWORD
token = authenticate(username, password, auth_url)

if token:
    well_url = "https://injection.texnet.beg.utexas.edu/api/well/wellswithinjectioncsv"
    well_file =  "./src/data/disposal_well.csv"
    well_data = fetch_data_and_save_csv(well_url, well_file, token)

    #get well ids from well_data
    well_df = pd.read_csv(StringIO(well_data))
    filtered_df = well_df[(well_df['SurfaceLatitude'] != 0) & (well_df['SurfaceLongitude'] != 0)]
    filtered_df.to_csv(well_file, index=False)
    id_array = filtered_df['Id'].to_numpy()
    payload = {
        'BeginMonth': 1,
        'BeginYear': 2016,
        'EndMonth': now.month,
        'EndYear': now.year,
        'Format': 'excel',
        'IncludeWellIds': True,
        'WellIds': id_array.tolist()
    }
    inj_url = "https://injection.texnet.beg.utexas.edu/api/Export"
    inj_file =  "./src/data/disposal_inj.csv"
    inj_data = fetch_data_and_save_csv(inj_url, inj_file, token, 'POST', None, payload)




# # ============================== STEP 3: DATA TRANSFORMATION ==============================

def well_to_b3_format(input_file, output_file, header_map):
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(input_file)

    #combine LeaseName and WellNumber to create WellName
    df['WellName'] = df['LeaseName'] + ' ' + df['WellNumber']
    df.drop(columns=['LeaseName', 'WellNumber'], inplace=True)
    df.rename(columns=header_map, inplace=True)
    df.to_csv(output_file, index=False)

def inj_to_b3_format(input_file, output_file, header_map):
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(input_file)
    df.rename(columns=header_map, inplace=True)
    df.to_csv(output_file, index=False)



input_file = "./src/data/disposal_well.csv"
output_file = "./src/data/disposal_well_b3_format.csv"
header_map = {
    'Id': 'InjectionWellId',
    'Apinumber': 'APINumber',
    'Uicnumber': 'UICNumber',
    'SurfaceLatitude': 'SurfaceHoleLatitude',
    'SurfaceLongitude': 'SurfaceHoleLongitude',
    'OriginalPermitDate': 'WellActivatedDate',
    'TotalBpdmax': 'PermittedMaxLiquidBPD',
    'InjectionBottomInterval': 'PermittedIntervalBottomFt',
    'InjectionTopInterval': 'PermittedIntervalTopFt',
    'WellClassification' : 'CompletedWellDepthClassification'
}

well_to_b3_format(input_file, output_file, header_map)


input_file = "./src/data/disposal_inj.csv"
output_file = "./src/data/disposal_inj_b3_format.csv"
header_map = {
    'Id': 'InjectionWellId',
    'Date of Injection': 'Date',
    'Volume Injected (BBLs)' : 'InjectedLiquidBBL'
}

inj_to_b3_format(input_file, output_file, header_map)

# ============================== STEP 4: HISTORICAL WELLS ==============================

## Function to detect encoding
#def detect_encoding(file_path):
#    with open(file_path, 'rb') as f:
#        result = chardet.detect(f.read())
#    return result['encoding']
#
## Path to your .txt file
#file_path = './uif700a.txt'
#
## Detect encoding
#encoding = detect_encoding(file_path)
#
## Read .txt file with detected encoding
#df = pd.read_csv(file_path, sep='\t', encoding=encoding)
#
## Display the DataFrame
#
#input_file = "../data/disposalWellsWithType.csv"
#output_file = "../data/disposalWellsB3Format.csv"
#header_map = {
#    'Id': 'InjectionWellId',
#    'Apinumber': 'APINumber',
#    'Uicnumber': 'UICNumber',
#    'SurfaceLatitude': 'SurfaceHoleLatitude',
#    'SurfaceLongitude': 'SurfaceHoleLongitude',
#    'OriginalPermitDate': 'WellActivatedDate',
#    'TotalBpdmax': 'PermittedMaxLiquidBPD',
#    'InjectionBottomInterval': 'PermittedIntervalBottomFt',
#    'InjectionTopInterval': 'PermittedIntervalTopFt'
#    }

# # =========================================================================================

def reformat(file, header_map):
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(file)
    df.rename(columns=header_map, inplace=True)
    df.to_csv(file, index=False)

disposal_well_file = './src/data/disposal_well_b3_format.csv'
disposal_inj_file = './src/data/disposal_inj_b3_format.csv'

well_map = {
    'InjectionWellId': 'ID'
}

# Shallow
ShallowWellFile = './src/data/gist_well_shallow.csv'
ShallowInjFile = './src/data/gist_injection_shallow.csv'

TXDInj=inj3.injTX(disposal_well_file,'Shallow',7000.)
TXDInj.addDaily(disposal_inj_file,100000)

ShallowWells=inj3.inj(None,TXDInj,'01-01-1970',ShallowWellFile)
reformat(ShallowWellFile, well_map)

ShallowWells.processRates(200000.,10,now.strftime('%m-%d-%Y'),False)
ShallowWells.outputReg(ShallowInjFile)


# Deep
DeepWellFile = './src/data/gist_well_deep.csv'
DeepInjFile = './src/data/gist_injection_deep.csv'

TXDInj=inj3.injTX(disposal_well_file,'Deep',7000.)
TXDInj.addDaily(disposal_inj_file,100000)

DeepWells=inj3.inj(None,TXDInj,'01-01-1970',DeepWellFile)
reformat(DeepWellFile, well_map)

DeepWells.processRates(200000.,10,now.strftime('%m-%d-%Y'),False)
DeepWells.outputReg(DeepInjFile)