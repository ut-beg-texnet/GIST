import pandas as pd
import numpy as np
import re

def normalize_uic_string(value):
    """
    Normalizes a UIC/ID string to a canonical numeric form if possible.
    
    1. Treats None, NaN, empty, or "nan" as ""
    2. Strips whitespace
    3. If numeric (including float-style "123.0"), converts to int then string
       to remove leading zeros and ".0" artifacts.
    4. Otherwise returns stripped string.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    
    s = str(value).strip()
    if s.lower() == "nan" or s == "":
        return ""
    
    # Check for numeric pattern (optional sign, then digits, optional .0)
    # We want to catch "01234", "1234", "1234.0"
    # But avoid mangling non-pure-numeric IDs if they exist
    if re.match(r"^-?\d+(\.0+)?$", s):
        try:
            # Convert to float then int to handle "1234.0"
            return str(int(float(s)))
        except (ValueError, TypeError):
            return s
            
    return s

def apply_uic_normalization(df, columns):
    """
    Applies normalize_uic_string to specified columns in a DataFrame.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(normalize_uic_string)
    return df
