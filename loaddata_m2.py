catfold='catalogs/'
import numpy as np
print('loading GSW')


import pandas as pd 



def read_data(catfold, filename, columns):
    filepath = catfold + filename
    return pd.read_csv(filepath, delim_whitespace=True, header=None, names=columns)

def load_catalogs(catfold):
    # Define column names based on the provided information
    columns = [
        'ObjID', 'GLXID', 'plate', 'MJD', 'fiber_ID',
        'RA', 'Decl', 'z', '2r', 'mass', 'mass_Error',
        'sfr', 'sfr_error', 'afuv', 'afuv_error',
        'ab', 'ab_error', 'av_gsw', 'av_gsw_error', 'flag_sed',
        'uv_survey', 'flag_uv', 'flag_midir', 'flag_mgs'
    ]
    
    # Load data into Pandas DataFrames
    m2_df = read_data(catfold, "GSWLC-M2.dat", columns)
    a2_df = read_data(catfold, "GSWLC-A2.dat", columns)
    x2_df = read_data(catfold, "GSWLC-X2.dat", columns)
    m2_df.reset_index(drop=True, inplace=True)

    # Additional data to be concatenated
    additional_data_files = [
        ('sigma1_mis.dat', [2], 'sigma1_m'),
        ('envir_nyu_mis.dat', [0], 'env_nyu_m'),
        ('baldry_mis.dat', [4], 'env_bald_m'),
        ('irexcess_mis.dat', [0], 'irx_m'),
        ('simard_ellip_mis.dat', [1], 'axisrat')
    ]

    for file, cols, new_col_name in additional_data_files:
        additional_df = read_data(catfold, file, cols)
        additional_df.reset_index(drop=True, inplace=True)
        m2_df[new_col_name] = additional_df.iloc[:, 0]

    return m2_df, a2_df, x2_df

m2_df, a2_df, x2_df = load_catalogs(catfold)

print('GSW loaded')