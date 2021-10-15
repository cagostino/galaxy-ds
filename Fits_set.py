"""
Made for loading in Fits files by name and to make it easy to grab column data.
"""
import astropy.io.fits as pf
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='serif')
plt.rc('text',usetex=True)

from scipy.constants import c

class Fits_set:
    '''
    Made for loading in binary fits tables
    '''
    def __init__(self,fname):
        self.fname = fname
        self.fitsimg = pf.open(fname)
        self.header = self.fitsimg[1].header
        self.data = self.fitsimg[1].data
    def getcol(self,colnum):
        if type(colnum)==list:
            return np.array([self.data.field(col) for col in colnum])
        else:
            return self.data.field(colnum)

