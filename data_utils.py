import pandas as pd
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import os

#class 
class DBConnector:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = sqlite3.connect(f"{self.db_name}")
        self.cursor = self.conn.cursor()
    def query(self, query_text, load_results = True):
        if load_results:
            return pd.read_sql_query(query_text, self.conn)
        else:
            self.cursor.execute(query_text)    
    def create_table(self, table_name, columns):
        # Create table as per requirement
        column_definitions = ', '.join([f"{col} {dtype}" for col, dtype in columns.items()])
        self.query(f"CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions})", load_results=False)
    def delete_table(self, table_name):
        self.query(f"DROP TABLE IF EXISTS {table_name}", load_results=False)
    def list_tables(self):
        return self.query("SELECT name FROM sqlite_master WHERE type='table';")
    def add_column_to_table(self, table_name,column ):
        self.query(f"ALTER TABLE {table_name} ADD COLUMN {column.name} {column.dtype}", load_results=False)

'''
from astroquery.sdss import SDSS

class SDSS_DB(SDSS):
    def query(self, query_text):
        return SDSS.query_sql(query_text)
'''        

def coordinate_matching(left_df, 
                        right_df, 
                        ra_dec_1, 
                        ra_dec_2, 
                        dist_threshold=7, 
                        matches_filename='coord_matches.csv'):
    '''
    Match two catalogs based on their coordinates.
    
    Parameters:
    left_df (DataFrame): DataFrame with the first set of coordinates (e.g., X-ray sources).
    right_df (DataFrame): DataFrame with the second set of coordinates (e.g., optical sources).
    ra_dec_1 (tuple): Tuple of strings for RA and Dec column names in left_df.
    ra_dec_2 (tuple): Tuple of strings for RA and Dec column names in right_df.
    dist_threshold (float): Matching distance threshold in arcseconds.
    matches_filename (str): Path to save the matches CSV file.
    
    Returns:
    DataFrame: A DataFrame with matched coordinates and additional data.
    '''
    fullpath = 'catalogs/'+matches_filename
    if os.path.exists(fullpath):
        matched_df = pd.read_csv(fullpath)
    else:

        # Convert RA and Dec to SkyCoord objects
        def validate_dec(df, dec_col):
            # Filter out any declination values that are not within the valid range
            return df[(df[dec_col] >= -90) & (df[dec_col] <= 90)]

        # Validate declination values
        left_df_valid = validate_dec(left_df, ra_dec_1[1])
        right_df_valid = validate_dec(right_df, ra_dec_2[1])

        # Create SkyCoord objects for valid entries
        left_coords = SkyCoord(ra=left_df_valid[ra_dec_1[0]].values*u.degree, dec=left_df_valid[ra_dec_1[1]].values*u.degree)
        right_coords = SkyCoord(ra=right_df_valid[ra_dec_2[0]].values*u.degree, dec=right_df_valid[ra_dec_2[1]].values*u.degree)

        # Find the nearest neighbors for each source in the left catalog within the right catalog
        idx, d2d, _ = left_coords.match_to_catalog_sky(right_coords)
        
        # Filter matches within the threshold distance
        matches_within_threshold = d2d.arcsec < dist_threshold
        matched_indices = idx[matches_within_threshold]
        separations = d2d.arcsec[matches_within_threshold]
        
        # Create a DataFrame for the matches
        matched_df = left_df.iloc[matches_within_threshold].copy()
        matched_df['r_matched_index'] = matched_indices
        matched_df['l_matched_index'] = matched_df.index

        matched_df['separation_arcsec'] = separations    
        
        # If matches_filename exists, load it, otherwise save the new matches
        matched_df.to_csv(fullpath, index=False)
    
    return matched_df


def coordinate_matching_and_join(left_df, 
                                 right_df, 
                                 ra_dec_1, 
                                 ra_dec_2, 
                                 dist_threshold=7, 
                                 matches_filename='coord_matches.csv', full_output_name = 'combined_output.csv'):
    '''
    Perform an outer join on the left and right DataFrames using the matched indices from coordinate_matching.
    left_df: DataFrame with RA and Dec of the first catalog.
    right_df: DataFrame with RA and Dec of the second catalog.
    ra_dec_1: Tuple of column names for RA and Dec in the first catalog.
    ra_dec_2: Tuple of column names for RA and Dec in the second catalog.
    dist_threshold: Maximum distance in arcseconds for a match.
    matches_filename: Filename to save the matches to.
    
    Returns a merged DataFrame from an outer join of left_df and right_df.
    '''
    fullpath = 'catalogs/'+full_output_name
    print(fullpath)
    # Get the matches indices DataFrame
    matches_df = coordinate_matching(left_df, right_df, ra_dec_1, ra_dec_2, dist_threshold, matches_filename)

    # Perform an outer join on the indices
    
    merged_df = pd.merge(left_df, matches_df[['l_matched_index', 'r_matched_index', 'separation_arcsec']], how='left', left_index=True, right_on='l_matched_index')
    merged_df = pd.merge(merged_df, right_df, how='outer', left_on='r_matched_index', right_index=True, suffixes=('', '_right'), indicator=True)

    # Save the merged DataFrame to disk
    merged_df.to_csv(fullpath, index=False)
    print(f"Saved merged DataFrame to '{fullpath}'.")
    return merged_df

# Usage example
# d = coordinate_matching_and_join(x3.data, ea, ra_dec_1=['RAJ2000','DEJ2000'], ra_dec_2=['RA', 'DEC'], matches_filename='catalogs/test_xr_sdss_10000.csv')





def match_and_merge(left_df, right_df, left_on, right_on, how='outer', left_suffix='', right_suffix=''):
    """
    Match two DataFrames based on a set of columns and perform an outer merge.

    Parameters:
    left_df (pd.DataFrame): The left DataFrame.
    right_df (pd.DataFrame): The right DataFrame.
    left_on (list of str): The column names in the left DataFrame to match on.
    right_on (list of str): The column names in the right DataFrame to match on.
    how (str): Type of merge to perform. Defaults to 'outer'.

    Returns:
    pd.DataFrame: A DataFrame with the matched and merged results.
    """
    left_df = left_df.reset_index().rename({'index':'left_index'})

    right_df = right_df.reset_index().rename({'index':'right_index'})

    merged_df = pd.merge(left_df, right_df, left_on=left_on, right_on=right_on, how=how, indicator=True, suffixes=(left_suffix, right_suffix))
    return merged_df

# Example usage:
# Assuming left_df and right_df are your dataframes and 'left_id_columns' and 'right_id_columns' are the lists of column names.
#left_id_columns = ['plate', 'MJD', 'fiber_ID']  # replace with actual column names in left_dfplate', 'MJD', 'fiber_ID'
#right_id_columns = ['PLATEID', 'MJD', 'FIBERID']  # replace with actual column names in right_df
#merged_data = match_and_merge(left_df, right_df, left_on=left_id_columns, right_on=right_id_columns)



def iterative_merge(dataframes):
    """
    Iteratively merges a list of DataFrames, keeping the left DataFrame's columns when overlaps occur.

    Parameters:
    dataframes (list of pd.DataFrame): List of DataFrames to merge. All DataFrames must be of the same length.

    Returns:
    pd.DataFrame: The merged DataFrame with all unique columns from the list of DataFrames.
    
    """
    # Start with the first DataFrame as the base
    merged_df = dataframes[0]

    # Iterate over the remaining DataFrames and merge them one by one
    for right_df in dataframes[1:]:
        # Find the overlapping columns between the current merged DataFrame and the next DataFrame
        overlapping_columns = merged_df.columns.intersection(right_df.columns)
        # Non-overlapping columns are those in the right DataFrame that are not in the overlapping_columns
        non_overlapping_columns = right_df.columns.difference(overlapping_columns)
        # Merge while selecting only the non-overlapping columns from the right DataFrame
        merged_df = pd.merge(merged_df, right_df[non_overlapping_columns], left_index=True, right_index=True, how='outer')

    return merged_df

# Example usage:
# Assuming dataframes is a list of your DataFrames [df1, df2, df3, ..., df8]
#merged_sdss = AstroTablePD(dataframe = iterative_merge([galinfo.data, galline.data, galindx.data, galfiboh.data, galmass.data, fibsfr.data, fibssfr.data]))



import sqlite3
def create_database(db_name):
    # Connect to the database (if it doesn't exist, it will be created)
    conn = sqlite3.connect(f"{db_name}.db")
    
    # Create a cursor object using the cursor() method
    cursor = conn.cursor()

    # Define a function to create tables
    def create_table(table_name, columns):
        # Create table as per requirement
        column_definitions = ', '.join([f"{col} {dtype}" for col, dtype in columns.items()])
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions})")

    # Example: Define the table structure
    # Let's say you want a table for storing user information
    create_table('Users', {
        'id': 'INTEGER PRIMARY KEY',
        'name': 'TEXT',
        'age': 'INTEGER',
        'email': 'TEXT'
    })

    # Commit your changes in the database
    conn.commit()

    # Close the connection
    conn.close()
def insert_dataframe_to_table(df, table_name, db_name, chunksize=10000,write_mode='replace'):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)

    # Create a copy of the dataframe to avoid changing the original one

    # Normalize column names to lowercase to identify duplicates
    normalized_cols = df.columns.str.lower()
    duplicates = normalized_cols.duplicated(keep=False)

    # Create a dictionary to count the occurrences of the normalized column names
    col_counts = {} 

    # Rename duplicate columns
    new_columns = []
    for col in df.columns:
        normalized_col = col.lower()
        if duplicates[normalized_cols.tolist().index(normalized_col)]:
            # Increment the count in the dictionary
            count = col_counts.get(normalized_col, 0)
            new_col = f"{col}_dup{count}" if count > 0 else col
            col_counts[normalized_col] = count + 1
        else:
            new_col = col
        new_columns.append(new_col)
    df.columns = new_columns

    # Check for any remaining duplicates
    if df.columns.duplicated().any():
        raise Exception("Duplicate column names detected after renaming.")

    # Insert the DataFrame data to the table in chunks
    if write_mode=='replace':
        df.iloc[0:0].to_sql(table_name, conn, if_exists='replace', index=False)
        
    for i in range(0, df.shape[0], chunksize):
        df.iloc[i:i+chunksize].to_sql(table_name, conn, if_exists='append', index=False)
        print(f"Inserted chunk {i//chunksize + 1}")

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()