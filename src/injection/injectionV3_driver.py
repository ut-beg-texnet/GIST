"""
Driver for injectionV3: fetch TexNet disposal CSVs, map to B3 format, run injTX/inj pipeline.

Usage (from GIST/src/injection):
  python injectionV3_driver.py
  python injectionV3_driver.py --dev

Use --dev to append '_dev' before each CSV extension so production files under ./src/data are not overwritten
(same naming as injection_updater_V4.resolve_path).
"""
import argparse
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import urllib3

import credentials
import injectionV3 as inj3
from injection_updater_V4 import resolve_path

requests.packages.urllib3.disable_warnings(
    requests.packages.urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

def fetch_data_and_save_csv(
    api_url, output_file, token, method='GET', data=None, json_payload=None, params=None
):
    """
    Fetch data from API and save as CSV file.

    Parameters:
    - api_url: str, the API endpoint URL
    - output_file: str or Path, path to save the CSV file
    - token: str, authorization token
    - method: str, HTTP method ('GET' or 'POST'), default 'GET'
    - data: dict or str, data to send in POST request body (form data)
    - json_payload: dict, JSON body for POST (passed to requests.post json=)
    - params: dict, URL parameters for GET requests

    Returns:
    - str: The response data, or None if error occurred
    """
    try:
        headers = {"Authorization": f"Bearer {token}"}

        # Add Content-Type header for POST requests with JSON data
        if method.upper() == 'POST' and json_payload is not None:
            headers["Content-Type"] = "application/json"

        # Make the appropriate HTTP request
        if method.upper() == 'GET':
            response = requests.get(api_url, headers=headers, params=params, verify=False)
        elif method.upper() == 'POST':
            response = requests.post(
                api_url, headers=headers, data=data, json=json_payload, params=params, verify=False
            )
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


# # ============================== STEP 3b (historical notes) ==============================
#
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
#
# # =========================================================================================

# ============================== STEP 4: HISTORICAL WELLS ==============================

def reformat(file, header_map):
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(file)
    df.rename(columns=header_map, inplace=True)
    df.to_csv(file, index=False)


def parse_args():
    """Parse CLI flags for the injectionV3 full refresh driver."""
    parser = argparse.ArgumentParser(
        description="Fetch TexNet disposal data and regenerate GIST injection CSVs via injectionV3.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Append '_dev' to all CSV basenames under ./src/data (safe testing; avoids overwriting production files).",
    )
    return parser.parse_args()


def main():
    """Authenticate, fetch API CSVs, transform to B3 layout, run shallow/deep inj pipeline."""
    args = parse_args()
    dev = args.dev
    tdir = Path("./src/data")
    now = datetime.now()

    well_raw = resolve_path(tdir, "disposal_well.csv", dev)
    inj_raw = resolve_path(tdir, "disposal_inj.csv", dev)
    well_b3 = resolve_path(tdir, "disposal_well_b3_format.csv", dev)
    inj_b3 = resolve_path(tdir, "disposal_inj_b3_format.csv", dev)
    shallow_well = resolve_path(tdir, "gist_well_shallow.csv", dev)
    shallow_inj = resolve_path(tdir, "gist_injection_shallow.csv", dev)
    deep_well = resolve_path(tdir, "gist_well_deep.csv", dev)
    deep_inj = resolve_path(tdir, "gist_injection_deep.csv", dev)

    # Authenticate and get token
    auth_url = "https://injection.texnet.beg.utexas.edu/api/Users/Authenticate"
    username = credentials.USERNAME
    password = credentials.PASSWORD
    token = authenticate(username, password, auth_url)

    if token:
        well_url = "https://injection.texnet.beg.utexas.edu/api/well/wellswithinjectioncsv"
        well_data = fetch_data_and_save_csv(well_url, well_raw, token)

        # get well ids from well_data
        well_df = pd.read_csv(StringIO(well_data))
        filtered_df = well_df[(well_df['SurfaceLatitude'] != 0) & (well_df['SurfaceLongitude'] != 0)]
        filtered_df.to_csv(well_raw, index=False)
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
        fetch_data_and_save_csv(inj_url, inj_raw, token, 'POST', None, json_payload=payload)

    well_header_map = {
        'Id': 'InjectionWellId',
        'Apinumber': 'APINumber',
        'Uicnumber': 'UICNumber',
        'SurfaceLatitude': 'SurfaceHoleLatitude',
        'SurfaceLongitude': 'SurfaceHoleLongitude',
        'OriginalPermitDate': 'WellActivatedDate',
        'TotalBpdmax': 'PermittedMaxLiquidBPD',
        'InjectionBottomInterval': 'PermittedIntervalBottomFt',
        'InjectionTopInterval': 'PermittedIntervalTopFt',
        'WellClassification': 'CompletedWellDepthClassification'
    }

    well_to_b3_format(well_raw, well_b3, well_header_map)

    inj_header_map = {
        'Id': 'InjectionWellId',
        'Date of Injection': 'Date',
        'Volume Injected (BBLs)': 'InjectedLiquidBBL'
    }

    inj_to_b3_format(inj_raw, inj_b3, inj_header_map)

    well_map = {
        'InjectionWellId': 'ID'
    }

    # Shallow
    TXDInj = inj3.injTX(well_b3, 'Shallow', 7000.)
    TXDInj.addDaily(inj_b3, 100000)

    ShallowWells = inj3.inj(None, TXDInj, '01-01-1970', shallow_well)
    reformat(shallow_well, well_map)

    ShallowWells.processRates(200000., 10, now.strftime('%m-%d-%Y'), False)
    ShallowWells.outputReg(shallow_inj)

    # Deep
    TXDInj = inj3.injTX(well_b3, 'Deep', 7000.)
    TXDInj.addDaily(inj_b3, 100000)

    DeepWells = inj3.inj(None, TXDInj, '01-01-1970', deep_well)
    reformat(deep_well, well_map)

    DeepWells.processRates(200000., 10, now.strftime('%m-%d-%Y'), False)
    DeepWells.outputReg(deep_inj)


if __name__ == "__main__":
    main()
