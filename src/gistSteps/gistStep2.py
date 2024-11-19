import pandas as pd
from numba import jit

# @jit(nopython=True)
def transform_well_csv_headers(well_file, header_map):
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(well_file)

    # Combine LeaseName and WellNumber to create WellName
    df['WellName'] = df['LeaseName'] + ' ' + df['WellNumber']
    df.drop(columns=['LeaseName', 'WellNumber'], inplace=True)
    df.rename(columns=header_map, inplace=True)
    df.to_csv(well_file, index=False)

input_file = "../data/wells.csv"
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

transform_well_csv_headers(input_file, header_map)