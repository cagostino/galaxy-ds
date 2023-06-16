#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Fits_set import *
catfold='catalogs/'
import numpy as np
import pandas as pd
class SDSSObj:
    def __init__(self):
        '''
        made for storing various SDSS quantities that are needed throughout
        '''
        pass
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



sdssobj = SDSSObj()

#from dr7
print('loading DR7')
sdssobj.fiboh = galfiboh.getcol('AVG')
sdssobj.allplateids,sdssobj.allmjds,sdssobj.allfiberids =galinfo.getcol(['PLATEID','MJD','FIBERID'])
sdssobj.all_fibmass = fibmass.getcol('AVG')
sdssobj.all_fibmags = galinfo.getcol('PLUG_MAG')
sdssobj.all_fibsfr_mpa = fibsfr.getcol('AVG')
sdssobj.all_fibssfr_mpa = fibssfr.getcol('AVG')
sdssobj.all_gmags = np.transpose(sdssobj.all_fibmags)[1]
sdssobj.all_spectype = galinfo.getcol('SPECTROTYPE')

sdssobj.allhdelta, sdssobj.allhdelta_err = galline.getcol(['H_DELTA_FLUX','H_DELTA_FLUX_ERR'])
sdssobj.allhgamma, sdssobj.allhgamma_err = galline.getcol(['H_GAMMA_FLUX','H_GAMMA_FLUX_ERR'])
sdssobj.alloIII4363, sdssobj.alloIII4363_err = galline.getcol(['OIII_4363_FLUX','OIII_4363_FLUX_ERR'])
sdssobj.alloIII4959, sdssobj.alloIII4959_err = galline.getcol(['OIII_4959_FLUX','OIII_4959_FLUX_ERR'])
sdssobj.allheI, sdssobj.allheI_err = galline.getcol(['HEI_5876_FLUX','HEI_5876_FLUX_ERR'])
sdssobj.alld4000 = galindx.getcol('D4000_n')
sdssobj.hdelta_lick = galindx.getcol('Lick_Hd_A')
sdssobj.tauv_cont = galindx.getcol('tauv_cont')

sdssobj.alloII3726, sdssobj.alloII3726err = galline.getcol(['OII_3726_FLUX','OII_3726_FLUX_ERR'])
sdssobj.alloII3729, sdssobj.alloII3729err = galline.getcol(['OII_3729_FLUX','OII_3729_FLUX_ERR'])
sdssobj.alloII = sdssobj.alloII3726 + sdssobj.alloII3729
sdssobj.alloII_err = np.sqrt(sdssobj.alloII3726err**2 + sdssobj.alloII3729err**2)

sdssobj.allneIII, sdssobj.allneIIIerr = galline.getcol(['NEIII_3869_FLUX_ERR', 'NEIII_3869_FLUX'])

sdssobj.alloI, sdssobj.alloI_err, sdssobj.allSII_6717, sdssobj.allSII_6717_err = galline.getcol(['OI_6300_FLUX', 'OI_6300_FLUX_ERR', 'SII_6717_FLUX', 'SII_6717_FLUX_ERR'])
sdssobj.allSII_6731,sdssobj.allSII_6731_err,sdssobj.alloIII,sdssobj.alloIII_err,sdssobj.allhbeta,sdssobj.allhbeta_err = galline.getcol(['SII_6731_FLUX','SII_6731_FLUX_ERR','OIII_5007_FLUX','OIII_5007_FLUX_ERR','H_BETA_FLUX','H_BETA_FLUX_ERR'])
sdssobj.alloIII_eqw, sdssobj.alloIII_eqw_err = galline.getcol(['OIII_5007_EQW', 'OIII_5007_EQW_ERR'])
sdssobj.allSII = sdssobj.allSII_6731+sdssobj.allSII_6717
sdssobj.allSII_err = np.sqrt(sdssobj.allSII_6731_err**2 + sdssobj.allSII_6717_err**2)
sdssobj.allnII, sdssobj.allnII_err, sdssobj.allhalpha, sdssobj.allhalpha_err = galline.getcol(['NII_6584_FLUX', 'NII_6584_FLUX_ERR', 'H_ALPHA_FLUX', 'H_ALPHA_FLUX_ERR'])
sdssobj.allha_eqw, sdssobj.allha_eqw_err = galline.getcol(['H_ALPHA_EQW','H_ALPHA_EQW_ERR'])
sdssobj.allnII_6548, sdssobj.allnII_6548_err = galline.getcol(['NII_6548_FLUX', 'NII_6548_FLUX_ERR'])

sdssobj.all_sdss_avgmasses = galmass.getcol('AVG')
sdssobj.allvdisp = galinfo.getcol('V_DISP')
sdssobj.allbalmerdisperr = galline.getcol('SIGMA_BALMER_ERR')
sdssobj.allbalmerdisp = galline.getcol('SIGMA_BALMER')
sdssobj.allforbiddendisp = galline.getcol('SIGMA_FORBIDDEN')
sdssobj.allforbiddendisperr = galline.getcol('SIGMA_FORBIDDEN_ERR')

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


'''

match_liu_to_xmm = []
match_x_liu = []
for i in range(len(liu_basic.getcol('DEC'))):
    radiff = (liu_basic.getcol('RA')[i] - x4.ra)
    decdiff = (liu_basic.getcol('DEC')[i] - x4.dec)
    dist = np.sqrt((radiff*np.cos(np.mean(np.radians(liu_basic.getcol('DEC')[i]))))**2+decdiff**2)
    mind = np.where(dist<7*arcsec)[0]
    if len(mind) >1:
        print(mind, dist[mind]/arcsec)
        maxflx = np.where(x4.fullflux[mind] == np.max(x4.fullflux[mind]))[0]
        mind = mind[maxflx]
    if len(mind)==1:            
        print(i)
        match_liu_to_xmm.append(mind[0])
        match_x_liu.append(i)

match_to_gsw = []
match_stern = []
for i in range(len(sterntab1.getcol('_DE'))):
    radiff = (sterntab1.getcol('_RA')[i] - m2Cat_GSW.allra)
    decdiff = (sterntab1.getcol('_DE')[i] - m2Cat_GSW.alldec)
    dist = np.sqrt((radiff*np.cos(np.radians(sterntab1.getcol('_DE')[i])))**2+decdiff**2)
    mind = np.where(dist<3*arcsec)[0]
    if len(mind) >1:
        print(mind, dist[mind]/arcsec)
    elif len(mind) == 1:
        print(i, mind)
        match_to_gsw.append(mind[0])
        match_stern.append(i)
        
match_to_xmm = []
match_x_stern = []
for i in range(len(sterntab1.getcol('_DE')[sternspec_pass])):
    radiff = (sterntab1.getcol('_RA')[sternspec_pass][i] - x4.ra)
    decdiff = (sterntab1.getcol('_DE')[sternspec_pass][i] - x4.dec)
    dist = np.sqrt((radiff*np.cos(np.radians(sterntab1.getcol('_DE')[sternspec_pass][i])))**2+decdiff**2)
    mind = np.where(dist<7*arcsec)[0]
    if len(mind) >1:
        print(mind, dist[mind]/arcsec)
        maxflx = np.where(x4.fullflux[mind] == np.max(x4.fullflux[mind]))[0]
        mind = mind[maxflx]
    elif len(mind) == 1:
        match_to_xmm.append(mind[0])
        match_x_stern.append(i)
'''


'''
csc2_names = []
for i in range(len(csc_sdss_pd.CSC2)):
    splitted = (str(csc_sdss_pd.CSC2.iloc[i]).split('_'))
    csc2_names.append(splitted[1])
    
cscx_names = []
for i in range(len(csc_pd._2CXO)):
    splitted =str(csc_pd._2CXO.iloc[i])[1:-1]
    cscx_names.append(splitted)
'''
merged_csc= pd.read_csv(catfold+'merged_csc.csv')

first = Fits_set(catfold+'FIRST.fits')
firstobj = SDSSObj()

firstobj.allplateids,firstobj.allmjds,firstobj.allfiberids =first.getcol(['SPEC_PLATE','SPEC_MJD','SPEC_FIBERID'])
firstobj.nvss_flux, firstobj.first_flux, firstobj.wenss_flux, firstobj.vlss_flux = first.getcol(['NVSS_FLUX', 'FIRST_FINT', 'WENSS_FLUX', 'VLSS_FLUX'])