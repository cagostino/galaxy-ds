"""
Made for loading in Fits files by name and to make it easy to grab column data.
"""
import astropy.io.fits as pf
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='serif')
plt.rc('text',usetex=True)

from scipy.constants import c
from astropy.table import Table
import pandas as pd
class Fits_set:
    '''
    Made for loading in binary fits tables
    '''
    def __init__(self,fname):
        self.fname = fname
        self.fitsimg = pf.open(fname)
        self.header = self.fitsimg[1].header
        self.data = Table(self.fitsimg[1].data)


        # Initialize an empty list to store exploded columns
        exploded_columns = []
        columns_to_remove = []

        # Iterate through the columns
        for colname in self.data.colnames:
            column = self.data[colname]
            # Check if the column is multidimensional
            if column.ndim > 1:
                # Create a new column for each sub-element of the multidimensional column
                for i in range(column.shape[1]):
                    new_colname = f"{colname}_{i}"
                    new_column = column[:, i]                    
                    exploded_columns.append((new_colname, new_column))
                columns_to_remove.append(colname)

        # Add the exploded columns to the original Astropy table
        for colname, coldata in exploded_columns:
            self.data[colname] = coldata
        for colname in columns_to_remove:
            self.data.remove_column(colname)
            
        self.data = self.data.to_pandas()

