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
import datetime
import urllib3
# Disable SSL warnings
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

def fetch_data_and_save_csv(api_url, output_file, token):
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(api_url, headers=headers, verify=False)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.text
        
        # Assuming the response is in CSV format
        with open(output_file, 'w', newline='') as csvfile:
            csvfile.write(data)
        
        print(f"CSV data successfully saved to {output_file}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")


auth_url = "https://localhost:44353/api/Users/Authenticate"
api_url = "https://localhost:44353/api/well/wellswithinjectioncsv"
output_file = "../data/disposalWells.csv"
username = "petersarkis@gmail.com"
password = "12345678"

# Authenticate and get token
token = authenticate(username, password, auth_url)

if token:
    fetch_data_and_save_csv(api_url, output_file, token)



# ============================== STEP 2: CALCULATE WELL TYPE ==============================
# Referenced from Constantinos's earthquake analyis tool: https://github.com/Costaki33/earthquake-analysis/blob/a68e575cc8225854496f57a19f71940742a15a1f/inj_pandas.py#L75 


def classify_well_type(well_lat, well_lon, well_depth):
    """
    Function classifys well type between either Shallow or Deep based on the Z-depth of the well in comparison to the
    Z-depth of the closest position of the Strawn Formation
    :param well_lat:
    :param well_lon:
    :param well_depth:
    :return: 1 or 0 for Deep or Shallow
    """
    df = pd.read_csv("../data/TopStrawn_RD_GCSNAD27.csv", delimiter=',')
    df['lat_nad27'] = pd.to_numeric(df['lat_nad27'], errors='coerce')
    df['lon_nad27'] = pd.to_numeric(df['lon_nad27'], errors='coerce')

    # Extract latitude and longitude columns from the DataFrame
    dataset_latitudes = df['lat_nad27'].values
    dataset_longitudes = df['lon_nad27'].values

    # Convert well position to numpy array for vectorized operations
    well_position = np.array([well_lat, well_lon])

    # Convert dataset positions to numpy array for vectorized operations
    dataset_positions = np.column_stack((dataset_latitudes, dataset_longitudes))
    # Calculate the Euclidean distance between the well's position and each position in the dataset
    distances = np.linalg.norm(dataset_positions - well_position, axis=1)
    # Find the index of the position with the minimum distance
    closest_index = np.argmin(distances)

    # Get Straw Formation Depth
    closest_strawn_depth = df['Zft_sstvd'].values[closest_index]
    if abs(well_depth) + closest_strawn_depth > 0:  # IE it's deeper
        return 'Deep'
    elif abs(well_depth) + closest_strawn_depth < 0:  # IE it's above the S.F.
        return 'Shallow'


def find_header_index(headers, target_header):
    for idx, header in enumerate(headers):
        if header == target_header:
            return idx

    # If the target header is not found, return None
    return None

def add_well_type_to_csv(input_file, output_file):
    with open(input_file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        header.append('CompletedWellDepthClassification')

        well_lat_index = find_header_index(header, 'SurfaceLatitude')
        well_long_index = find_header_index(header, 'SurfaceLongitude')
        well_depth_index = find_header_index(header, 'WellTDFt')

        rows = []
        for row in reader:
            well_lat = row[well_lat_index]
            well_long = row[well_long_index]
            well_depth = row[well_depth_index]
            if well_lat and well_long and well_depth:
                well_type = classify_well_type(float(well_lat), float(well_long), float(well_depth))
            else:
                well_type = ''

            # Add well type to the current row
            row.append(well_type)

            # Update rows list
            rows.append(row)

    # Write the processed data to the output file
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Processed data saved to {output_file}")


input_file = "../data/disposalWells.csv"
output_file = "../data/disposalWellsWithType.csv"

add_well_type_to_csv(input_file, output_file)


# ============================== STEP 3: DATA TRANSFORMATION ==============================

def transform_disposal_to_b3_format(input_file, output_file, header_map):
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(input_file)

    #combine LeaseName and WellNumber to create WellName
    df['WellName'] = df['LeaseName'] + ' ' + df['WellNumber']
    df.drop(columns=['LeaseName', 'WellNumber'], inplace=True)
    df.rename(columns=header_map, inplace=True)
    df.to_csv(output_file, index=False)



input_file = "../data/disposalWellsWithType.csv"
output_file = "../data/disposalWellsB3Format.csv"
header_map = {
    'Id': 'InjectionWellId',
    'Apinumber': 'APINumber',
    'Uicnumber': 'UICNumber',
    'SurfaceLatitude': 'SurfaceHoleLatitude',
    'SurfaceLongitude': 'SurfaceHoleLongitude',
    'OriginalPermitDate': 'WellActivatedDate',
    'TotalBpdmax': 'PermittedMaxLiquidBPD',
    'InjectionBottomInterval': 'PermittedIntervalBottomFt',
    'InjectionTopInterval': 'PermittedIntervalTopFt'
    }

transform_disposal_to_b3_format(input_file, output_file, header_map)


# ============================== STEP 4: HISTORICAL WELLS ==============================

# Function to detect encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# Path to your .txt file
file_path = './uif700a.txt'

# Detect encoding
encoding = detect_encoding(file_path)

# Read .txt file with detected encoding
df = pd.read_csv(file_path, sep='\t', encoding=encoding)

# Display the DataFrame

input_file = "../data/disposalWellsWithType.csv"
output_file = "../data/disposalWellsB3Format.csv"
header_map = {
    'Id': 'InjectionWellId',
    'Apinumber': 'APINumber',
    'Uicnumber': 'UICNumber',
    'SurfaceLatitude': 'SurfaceHoleLatitude',
    'SurfaceLongitude': 'SurfaceHoleLongitude',
    'OriginalPermitDate': 'WellActivatedDate',
    'TotalBpdmax': 'PermittedMaxLiquidBPD',
    'InjectionBottomInterval': 'PermittedIntervalBottomFt',
    'InjectionTopInterval': 'PermittedIntervalTopFt'
    }

# =========================================================================================

# Step 1 - point to files - will replace this with REST API calls at some point
# Points to current B3 v3 datasets as of 10/31/22 - NM column names will change

TXWellFile='/data/xom/seismicity/shared/B3/TX/InjectionWell.csv'
TXMonthlyFile='/data/xom/seismicity/shared/B3/TX/MonthlyInjection.csv'
TXDailyFile='/data/xom/seismicity/shared/B3/TX/DailyInjection.csv'

TXDTempWFile='/data/xom/seismicity/bcurry/GIST/TestTXWellsDeep.csv'
TXDTempMFile='/data/xom/seismicity/bcurry/GIST/TestTXMonthlyInjDeep.csv'
TXDTempDFile='/data/xom/seismicity/bcurry/GIST/TestTXDailyInjDeep.csv'
TXSTempWFile='/data/xom/seismicity/bcurry/GIST/TestTXWellsShallow.csv'
TXSTempMFile='/data/xom/seismicity/bcurry/GIST/TestTXMonthlyInjShallow.csv'
TXSTempDFile='/data/xom/seismicity/bcurry/GIST/TestTXDailyInjShallow.csv'

NMWellFile='/data/xom/seismicity/shared/B3/NM/InjectionWell.csv'
NMMonthlyFile='/data/xom/seismicity/shared/B3/NM/MonthlyInjection.csv'
NMDailyFile='/data/xom/seismicity/shared/B3/NM/DailyInjection.csv'

NMDTempWFile='/data/xom/seismicity/bcurry/GIST/TestNMWellsDeep.csv'
NMDTempMFile='/data/xom/seismicity/bcurry/GIST/TestNMMonthlyInjDeep.csv'
NMDTempDFile='/data/xom/seismicity/bcurry/GIST/TestNMDailyInjDeep.csv'
NMSTempWFile='/data/xom/seismicity/bcurry/GIST/TestNMWellsShallow.csv'
NMSTempMFile='/data/xom/seismicity/bcurry/GIST/TestNMMonthlyInjShallow.csv'
NMSTempDFile='/data/xom/seismicity/bcurry/GIST/TestNMDailyInjShallow.csv'

DeepWellFile='/data/xom/seismicity/bcurry/GIST/testDeep.csv'
DeepInjFile='/data/xom/seismicity/bcurry/GIST/testDeepInj.csv'
DeepPrefix='/data/xom/seismicity/bcurry/GIST/testDeep'

ShallowWellFile='/data/xom/seismicity/bcurry/GIST/testShallow.csv'
ShallowInjFile='/data/xom/seismicity/bcurry/GIST/testShallowInj.csv'
ShallowPrefix='/data/xom/seismicity/bcurry/GIST/testShallow'
# First deal with Deep
# Texas
TXDInj=inj3.injTX(TXWellFile,'Deep',7000.,TXDTempWFile)
TXDInj.addMonthly(TXMonthlyFile,1000000,TXDTempMFile)
TXDInj.addDaily(TXDailyFile,100000,TXDTempDFile)

# New Mexico
NMDInj=inj3.injNM(NMWellFile,'Deep',7000.,NMDTempWFile)
NMDInj.addMonthly(NMMonthlyFile,1000000,NMDTempMFile)
NMDInj.addDaily(NMDailyFile,100000,NMDTempDFile)

# Merge and process results

DeepWells=inj3.inj(NMDInj,TXDInj,'01-01-1980',DeepWellFile,DeepInjFile)
DeepWells.processRates(200000.,1,'12-16-2022',True)
DeepWells.output(DeepPrefix)

# Shallow
TXSInj=inj3.injTX(TXWellFile,'Shallow',7000.,TXSTempWFile)
TXSInj.addMonthly(TXMonthlyFile,1000000,TXSTempMFile)
TXSInj.addDaily(TXDailyFile,100000,TXSTempDFile)

NMSInj=inj3.injNM(NMWellFile,'Shallow',7000.,NMSTempWFile)
NMSInj.addMonthly(NMMonthlyFile,1000000,NMSTempMFile)
NMSInj.addDaily(NMDailyFile,100000,NMSTempDFile)


ShallowWells=inj3.inj(NMSInj,TXSInj,'01-01-1980',ShallowWellFile,ShallowInjFile)
ShallowWells.processRates(200000.,1,'12-16-2022',True)
ShallowWells.output(ShallowPrefix)

# New Mexico should look the same