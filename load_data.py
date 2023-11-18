#!/usr/bin/env python3
# -*- coding: utf-8 -*-

catfold='catalogs/'
import numpy as np
import pandas as pd
from ast_utils import getfluxfromlum, getlumfromflux, dustcorrect, extinction
from data_models import *


'''
Load info from X-ray Surveys
'''
#chand =CSC(catfold+'Chandra_multiwavelength.fits')
xmm = AstroTablePD(catfold+'XMM_multiwavelength_cat.fits')
#xmmfu = fits_set(catfold+'XMM_followup.fits')
xmmfu= AstroTablePD(catfold+'XMM_followup_all.fits')




xmm3obs = XMM3obs(catfold+'3xmmobs.fits')
x3 = XMM(catfold+'3xmm.fits', xmm3obs)

xmm4obs = XMM4obs(catfold+'4xmmobs.tsv')
x4 = XMM(catfold+'4xmm.fits', xmm4obs)

#csc = Fits_set(catfold+'csc.fits')
#csc_sdss = Fits_set(catfold+'csc_sdss.fits')





galinfo  = Gal_Info(catfold+'gal_info_dr7_v5_2.fit')
galline = Gal_Line(catfold+'gal_line_dr7_v5_2.fit')
galindx = Gal_Indx(catfold+'gal_indx_dr7_v5_2.fit')

galfiboh = Gal_Fib(catfold+'gal_fiboh_dr7_v5_2.fits', 'fiboh')
galmass = Gal_Fib(catfold+'totlgm_dr7_v5_2b.fit', 'mass')
fibmass = Gal_Fib(catfold+'fiblgm_dr7_v5_2.fit', 'fibmass')
fibsfr = Gal_Fib(catfold+'gal_fibsfr_dr7_v5_2.fits','fibsfr')
fibssfr = Gal_Fib(catfold+'gal_fibspecsfr_dr7_v5_2.fits', 'fibssfr')


m2_gsw = AstroTablePD(catfold+'GSWLC-M2.dat')

a2_gsw = AstroTablePD(catfold+'GSWLC-A2.dat')

x2_gsw = AstroTablePD(catfold+'GSWLC-X2.dat')

d2_gsw = AstroTablePD(catfold+'GSWLC-D2.dat')

left_id_columns = ['plate', 'MJD', 'fiber_ID']  # replace with actual column names in left_dfplate', 'MJD', 'fiber_ID'
right_id_columns = ['PLATEID', 'MJD', 'FIBERID']  # replace with actual column names in right_df

from data_utils import create_database, insert_dataframe_to_table, match_and_merge, iterative_merge


merged_sdss = AstroTablePD(dataframe = iterative_merge([galinfo.data, galline.data, galindx.data, galfiboh.data, galmass.data, fibsfr.data, fibssfr.data]))


m2_sdss = match_and_merge(m2_gsw.data, merged_sdss.data, left_on=left_id_columns, right_on=right_id_columns)

import sqlite3

#initial use 
# insert_dataframe_to_table(m2_sdss, 'gsw_sdss', 'catalogs/catalog_database.db')

'''
csc_cat = CSC(merged_csc)
comm_csc_gsw,  gsw_csc, csc_gsw= np.intersect1d(  m2[0], merged_csc.SDSSDR15, return_indices=True)
csc_cat.gswmatch_inds = gsw_csc
'''
