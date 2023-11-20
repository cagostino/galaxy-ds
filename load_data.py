#!/usr/bin/env python3
# -*- coding: utf-8 -*-

catfold='catalogs/'
import numpy as np
import pandas as pd
from ast_utils import getfluxfromlum, getlumfromflux, dustcorrect, extinction
from data_models import *
from data_utils import create_database, insert_dataframe_to_table, match_and_merge, iterative_merge, coordinate_matching_and_join, DBConnector, coordinate_matching


import sqlite3
#initial use for merging sdss and gsw
db_conn = DBConnector('catalogs/catalog_database.db')

'''

###Load info from X-ray Surveys
#chand =CSC(catfold+'Chandra_multiwavelength.fits')
xmm = AstroTablePD(catfold+'XMM_multiwavelength_cat.fits')
#xmmfu = fits_set(catfold+'XMM_followup.fits')
xmmfu= AstroTablePD(catfold+'XMM_followup_all.fits')



xmm3obs = XMM3obs(catfold+'3xmmobs.fits')
x3 = XMM(catfold+'3xmm.fits', xmm3obs)

xmm4obs = XMM4obs(catfold+'4xmmobs.tsv')
x4 = XMM(catfold+'4xmm.fits', xmm4obs)

#insert_dataframe_to_table(x4.data, '4xmm', 'catalogs/catalog_database.db')


#csc = Fits_set(catfold+'csc.fits')
#csc_sdss = Fits_set(catfold+'csc_sdss.fits')



### Load info from SDSS
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
merged_sdss = AstroTablePD(dataframe = iterative_merge([galinfo.data, galline.data, galindx.data, galfiboh.data, galmass.data, fibsfr.data, fibssfr.data]))

merged_sdss.data = get_line_filters(merged_sdss.data)
merged_sdss.data = add_dust_corrected_fluxes_by_model(merged_sdss.data,
                        galline.lines,
                        modelfn = get_extinction, 
                        model='a19', 
                        dec_rat=3.1)
merged_sdss.data = get_dust_correction_quantities(merged_sdss.data, model='a19')
merged_sdss.data = merged_sdss.data.reset_index()
merged_sdss.data = merged_sdss.data.rename(columns={'index': 'sdss_index'})


#insert_dataframe_to_table(merged_sdss.data, 'sdss_combined', 'catalogs/catalog_database.db')



### match and merge sdss and gsw

m2_sdss = match_and_merge(m2_gsw.data, merged_sdss.data, left_on=left_id_columns, right_on=right_id_columns)
m2_sdss['RA_span'] =m2_sdss['RA_y'].combine_first(m2_sdss['RA_x'])

m2_sdss['dec_span'] =m2_sdss['DEC'].combine_first(m2_sdss['Decl'])

m2_sdss['plate_id_span'] =m2_sdss['PLATEID'].combine_first(m2_sdss['plate'])

m2_sdss['fiberid_span'] =m2_sdss['FIBERID'].combine_first(m2_sdss['fiber_ID'])





m2_sdss = m2_sdss.reset_index()
m2_sdss = m2_sdss.rename(columns={'index': 'sdss_gsw_index'})
m2_sdss = add_dust_corrected_fluxes_by_model(m2_sdss,  galline.lines, model='a21')
m2_sdss = get_dust_correction_quantities(m2_sdss, model='a21')

#insert_dataframe_to_table(m2_sdss, 'gsw_m2_sdss', 'catalogs/catalog_database.db')


m2_sdss_inds = db_conn.query("""SELECT sdss_gsw_index, RA_span, dec_span from gsw_m2_sdss""")
d = coordinate_matching_and_join(x4.data, m2_sdss_inds, ra_dec_1=['RA_ICRS','DE_ICRS'], ra_dec_2=['RA_span', 'dec_span'], matches_filename='4xmm_sdss_gsw_upd.csv')
d = d.reset_index()
d = d.rename(columns={'index': 'xmm_sdss_gsw_index'})

#insert_dataframe_to_table(d, 'xmm4_gsw_sdss', 'catalogs/catalog_database.db')




xr_m2_sdss = db_conn.query("""SELECT xr.*, m2.* from gsw_m2_sdss as m2 join xmm4_gsw_m2_sdss as xr on m2.sdss_gsw_index = xr.sdss_gsw_index where xr._merge =="both" """)
insert_dataframe_to_table(xr_m2_sdss, 'xmm4_gsw_sdss_combined', 'catalogs/catalog_database.db')
    

xr_m2_sdss_texp_p1_ = db_conn.query("""SELECT xr.*, m2.* from gsw_m2_sdss as m2 join xmm4_gsw_m2_sdss as xr on m2.sdss_gsw_index = xr.sdss_gsw_index where xr._merge =="both" and texp<31622 and texp>12589 and flag_sed==0""")
insert_dataframe_to_table(xr_m2_sdss_texp_p1_, 'xmm4_gsw_sdss_combined_texp_limit', 'catalogs/catalog_database.db')


xr_m2_sdss_sed0 = db_conn.query("""SELECT xr.*, m2.* from gsw_m2_sdss as m2 join xmm4_gsw_m2_sdss as xr on m2.sdss_gsw_index = xr.sdss_gsw_index where xr._merge =="both" and flag_sed==0""")
insert_dataframe_to_table(xr_m2_sdss_sed0, 'xmm4_gsw_sdss_sed0_combined', 'catalogs/catalog_database.db')


xr_m2_sdss_sed1 = db_conn.query("""SELECT xr.*, m2.* from gsw_m2_sdss as m2 join xmm4_gsw_m2_sdss as xr on m2.sdss_gsw_index = xr.sdss_gsw_index where xr._merge =="both" and flag_sed ==1 
    """)
insert_dataframe_to_table(xr_m2_sdss_sed1, 'xmm4_gsw_sdss_sed1_combined', 'catalogs/catalog_database.db')
    

'''


