#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Fits_set import *
catfold='catalogs/'
import numpy as np
class SDSSObj:
    def __init__(self):
        '''
        made for storing various SDSS quantities that are needed throughout
        '''
        pass

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

sdssobj.allhdelta, sdssobj.allhdelta_err = galline.getcol(['H_DELTA_FLUX','H_DELTA_FLUX_ERR'])
sdssobj.allhgamma, sdssobj.allhgamma_err = galline.getcol(['H_GAMMA_FLUX','H_GAMMA_FLUX_ERR'])
sdssobj.alloIII4363, sdssobj.alloIII4363_err = galline.getcol(['OIII_4363_FLUX','OIII_4363_FLUX_ERR'])
sdssobj.alloIII4959, sdssobj.alloIII4959_err = galline.getcol(['OIII_4959_FLUX','OIII_4959_FLUX_ERR'])
sdssobj.allheI, sdssobj.allheI_err = galline.getcol(['HEI_5876_FLUX','HEI_5876_FLUX_ERR'])
sdssobj.alld4000 = galindx.getcol('D4000_n')
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