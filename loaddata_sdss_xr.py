#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Fits_set import *
catfold='catalogs/'
import numpy as np
import pandas as pd
m2 = np.loadtxt(catfold+"GSWLC-M2.dat", unpack = True, usecols=(0,1), dtype=np.int64)
galinfo  = Fits_set(catfold+'gal_info_dr7_v5_2.fit')
galline = Fits_set(catfold+'gal_line_dr7_v5_2.fit')
galfiboh = Fits_set(catfold+'gal_fiboh_dr7_v5_2.fits')
galindx = Fits_set(catfold+'gal_indx_dr7_v5_2.fit')
galmass = Fits_set(catfold+'totlgm_dr7_v5_2b.fit')
fibmass = Fits_set(catfold+'fiblgm_dr7_v5_2.fit')
fibsfr = Fits_set(catfold+'gal_fibsfr_dr7_v5_2.fits')
fibssfr = Fits_set(catfold+'gal_fibspecsfr_dr7_v5_2.fits')
'''
Load info from X-ray Surveys
'''
chand =Fits_set(catfold+'Chandra_multiwavelength.fits')
xmm = Fits_set(catfold+'XMM_multiwavelength_cat.fits')
#xmmfu = fits_set(catfold+'XMM_followup.fits')
xmmfu= Fits_set(catfold+'XMM_followup_all.fits')

data_dict = {
    'fiboh': galfiboh.data['AVG'],
    'allplateids': galinfo.data['PLATEID'],
    'allmjds': galinfo.data['MJD'],
    'allfiberids': galinfo.data['FIBERID'],
    'all_fibmass': fibmass.data['AVG'],
    #'all_fibmags': galinfo.data['PLUG_MAG'],
    'all_fibsfr_mpa': fibsfr.data['AVG'],
    'all_fibssfr_mpa': fibssfr.data['AVG'],
    #'all_gmags': np.transpose(galinfo.data['PLUG_MAG'])[1],
    'all_spectype': galinfo.data['SPECTROTYPE'],
    'allhdelta': galline.data['H_DELTA_FLUX'],
    'allhdelta_err': galline.data['H_DELTA_FLUX_ERR'],
    'allhgamma': galline.data['H_GAMMA_FLUX'],
    'allhgamma_err': galline.data['H_GAMMA_FLUX_ERR'],
    'alloIII4363': galline.data['OIII_4363_FLUX'],
    'alloIII4363_err': galline.data['OIII_4363_FLUX_ERR'],
    'alloIII4959': galline.data['OIII_4959_FLUX'],
    'alloIII4959_err': galline.data['OIII_4959_FLUX_ERR'],
    'allheI': galline.data['HEI_5876_FLUX'],
    'allheI_err': galline.data['HEI_5876_FLUX_ERR'],
    'alld4000': galindx.data['D4000_N'],
    'hdelta_lick': galindx.data['LICK_HD_A'],
    'tauv_cont': galindx.data['TAUV_CONT'],
    'alloII3726': galline.data['OII_3726_FLUX'],
    'alloII3726err': galline.data['OII_3726_FLUX_ERR'],
    'alloII3729': galline.data['OII_3729_FLUX'],
    'alloII3729err': galline.data['OII_3729_FLUX_ERR'],
    'allneIII': galline.data['NEIII_3869_FLUX_ERR'],
    'allneIIIerr': galline.data['NEIII_3869_FLUX'],
    'alloI': galline.data['OI_6300_FLUX'],
    'alloI_err': galline.data['OI_6300_FLUX_ERR'],
    'allSII_6717': galline.data['SII_6717_FLUX'],
    'allSII_6717_err': galline.data['SII_6717_FLUX_ERR'],
    'allSII_6731': galline.data['SII_6731_FLUX'],
    'allSII_6731_err': galline.data['SII_6731_FLUX_ERR'],
    'alloIII': galline.data['OIII_5007_FLUX'],
    'alloIII_err': galline.data['OIII_5007_FLUX_ERR'],
    'allhbeta': galline.data['H_BETA_FLUX'],
    'allhbeta_err': galline.data['H_BETA_FLUX_ERR'],
    'alloIII_eqw': galline.data['OIII_5007_EQW'],
    'alloIII_eqw_err': galline.data['OIII_5007_EQW_ERR'],
    'allnII': galline.data['NII_6584_FLUX'],
    'allnII_err': galline.data['NII_6584_FLUX_ERR'],
    'allhalpha': galline.data['H_ALPHA_FLUX'],
    'allhalpha_err': galline.data['H_ALPHA_FLUX_ERR'],
    'allha_eqw': galline.data['H_ALPHA_EQW'],
    'allha_eqw_err': galline.data['H_ALPHA_EQW_ERR'],
    'allnII_6548': galline.data['NII_6548_FLUX'],
    'allnII_6548_err': galline.data['NII_6548_FLUX_ERR'],
    'all_sdss_avgmasses': galmass.data['AVG'],
    'allvdisp': galinfo.data['V_DISP'],
    'allbalmerdisperr': galline.data['SIGMA_BALMER_ERR'],
    'allbalmerdisp': galline.data['SIGMA_BALMER'],
    'allforbiddendisp': galline.data['SIGMA_FORBIDDEN'],
    'allforbiddendisperr': galline.data['SIGMA_FORBIDDEN_ERR']
}

# Create a DataFrame
sdssobj = pd.DataFrame(data_dict)



xmm3 = Fits_set(catfold+'3xmm.fits')
xmm3obs = Fits_set(catfold+'3xmmobs.fits')


xmm4 = Fits_set(catfold+'4xmm.fits')
xmm4obs = pd.read_csv(catfold+'4xmmobs.tsv', delimiter='\t')



#csc = Fits_set(catfold+'csc.fits')
#csc_sdss = Fits_set(catfold+'csc_sdss.fits')


sterntab1 = Fits_set(catfold+'sterntab.fits')
match_to_gsw = np.load(catfold+'match_gsw_to_stern.txt.npy', allow_pickle=True)
match_stern = np.load(catfold+'match_stern_to_gsw.txt.npy',  allow_pickle=True)
match_to_xmm = np.load(catfold+'match_xmm_to_stern.npy', allow_pickle=True)
match_x_stern = np.load(catfold+'match_stern_to_xmm.npy',  allow_pickle=True)
match_liu_to_xmm = np.load(catfold+'match_xmm_to_liu.npy', allow_pickle=True)
match_x_liu = np.load(catfold+'match_x_liu.npy', allow_pickle=True)

sternspecz = np.loadtxt(catfold+'stern_z_caug37.csv', skiprows=1, delimiter=',', unpack=True)
sternspecz_ids = np.loadtxt(catfold+'stern_z_caug37.csv', skiprows=1,usecols=(6), dtype=np.int64, delimiter=',', unpack=True)

sternspec_pass = np.load(catfold+'sternspec_pass.npy', allow_pickle=True)
'''
sternspec_pass = []
for i in range(len(sterntab1.getcol('_RA'))):
    radiff = np.abs(sterntab1.getcol('_RA')[i] - sternspecz[3])
    decdiff = np.abs(sterntab1.getcol('_DE')[i] - sternspecz[4])
    if np.min(radiff) <arcsec and np.min(decdiff) <arcsec:
        sternspec_pass.append(i)
    
'''

sternobj = {}

sternobj['lbha'], sternobj['logM'], sternobj['luv'], sternobj['alpha'] = sterntab1.getcol(['logLbHa','logM_', 'logLUV', 'alpha']) 

sternobj['lha'], sternobj['e_lha'], sternobj['lhb'], sternobj['e_lhb']  = sterntab1.getcol(['logLHa','l_logLHa', 'logLHb', 'l_logLHb']) 

sternobj['e_lo3'], sternobj['lo3'], sternobj['ln2'], sternobj['e_ln2']  = sterntab1.getcol(['l_logL_OIII_','logL_OIII_', 'logL_NII_', 'l_logL_NII_']) 

sternobj['robust'], sternobj['abs'] = sterntab1.getcol(['fs', 'fa'])
#sternobj['sdssids'] = m2[0][match_to_gsw]
sternobj_df = pd.DataFrame(sternobj)
sternobj_df_spec = sternobj_df.iloc[sternspec_pass].copy()
sternobj_df_spec['ids'] =sternspecz_ids
sternobj_df_spec['z'] = sternspecz[0]

sternobj_df_spec_xr = sternobj_df_spec.iloc[match_x_stern].copy()

sternobj_m2_df = sternobj_df.iloc[match_stern]
sternobj_m2_df.loc[:, 'ids'] = m2[0][match_to_gsw]



liu_basic = Fits_set(catfold+'liu_basic.fits')

liu_spec = Fits_set(catfold+'liu_qsos_spec.fits')
liu_mw = Fits_set(catfold+'liu_qsos_multiwavelength.fits')



merged_csc= pd.read_csv(catfold+'merged_csc.csv')

first = Fits_set(catfold+'FIRST.fits')
first_data_dict = {
    'allplateids': first.data['SPEC_PLATE'],
    'allmjds': first.data['SPEC_MJD'],
    'allfiberids': first.data['SPEC_FIBERID'],
    'nvss_flux': first.data['NVSS_FLUX'],
    'first_flux': first.data['FIRST_FINT'],
    'wenss_flux': first.data['WENSS_FLUX'],
    'vlss_flux': first.data['VLSS_FLUX']
}

firstobj_df = pd.DataFrame(first_data_dict)