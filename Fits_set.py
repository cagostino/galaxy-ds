#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Made for loading in Fits files by name and to make it easy to grab column data.
"""
import astropy.io.fits as pf
import numpy as np
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