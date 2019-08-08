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
galmass = Fits_set(catfold+'totlgm_dr7_v5_2b.fit')
fibmass = Fits_set(catfold+'fiblgm_dr7_v5_2.fit')

'''
Load info from Stripe 82X
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
sdssobj.all_gmags = np.transpose(sdssobj.all_fibmags)[1]

sdssobj.alloII3726, sdssobj.alloII3726err = galline.getcol(['OII_3726_FLUX','OII_3726_FLUX_ERR'])
sdssobj.alloII3729, sdssobj.alloII3729err = galline.getcol(['OII_3729_FLUX','OII_3729_FLUX_ERR'])
sdssobj.alloII = sdssobj.alloII3726 + sdssobj.alloII3729
sdssobj.alloIIerr = np.sqrt(sdssobj.alloII3726err**2 + sdssobj.alloII3729err**2)

sdssobj.allneIII, sdssobj.allneIIIerr = galline.getcol(['NEIII_3869_FLUX_ERR', 'NEIII_3869_FLUX'])

sdssobj.alloI, sdssobj.alloI_err, sdssobj.allSII_6717, sdssobj.allSII_6717_err = galline.getcol(['OI_6300_FLUX', 'OI_6300_FLUX_ERR', 'SII_6717_FLUX', 'SII_6717_FLUX_ERR'])
sdssobj.allSII_6731,sdssobj.allSII_6731_err,sdssobj.alloIII,sdssobj.alloIII_err,sdssobj.allhbeta,sdssobj.allhbeta_err = galline.getcol(['SII_6731_FLUX','SII_6731_FLUX_ERR','OIII_5007_FLUX','OIII_5007_FLUX_ERR','H_BETA_FLUX','H_BETA_FLUX_ERR'])
sdssobj.allSII = sdssobj.allSII_6731+sdssobj.allSII_6717
sdssobj.allSII_err = np.sqrt(sdssobj.allSII_6731_err**2 + sdssobj.allSII_6717_err**2)
sdssobj.allnII, sdssobj.allnII_err, sdssobj.allhalpha, sdssobj.allhalpha_err = galline.getcol(['NII_6584_FLUX', 'NII_6584_FLUX_ERR', 'H_ALPHA_FLUX', 'H_ALPHA_FLUX_ERR'])
sdssobj.all_sdss_avgmasses = galmass.getcol('AVG')
sdssobj.allvdisp = galinfo.getcol('V_DISP')
sdssobj.allbalmerdisperr = galline.getcol('SIGMA_BALMER_ERR')
sdssobj.allbalmerdisp = galline.getcol('SIGMA_BALMER')
sdssobj.allforbiddendisp = galline.getcol('SIGMA_FORBIDDEN')
sdssobj.allforbiddendisperr = galline.getcol('SIGMA_FORBIDDEN_ERR')

print('loading GSW')
m2 = np.loadtxt(catfold+"GSWLC-M2.dat", unpack = True, usecols=(0,1), dtype=np.int64)

redshift_m2 = np.loadtxt(catfold+"GSWLC-M2.dat", unpack = True, usecols=(7,))

allm2 = np.loadtxt(catfold+"GSWLC-M2.dat", unpack = True, usecols=(5, 6, 11, 9, 2, 4, 3, 19))

#to make it easier to switch indicies in case more things get added, FOR GSWLC
raind = 0
decind = 1
sfrind = 2
massind = 3
plateind = 4
fiberind = 5
mjdind = 6
sedind = 7

xmm3 = Fits_set(catfold+'3xmm.fits')
xmm3obs = Fits_set(catfold+'3xmmobs.fits')



#for getting r mags for doing x-ray duplicate removal
m1_photcatids = np.loadtxt(catfold+'gs_mis_sdss_phot.dat', unpack=True,usecols=(6,),dtype=np.int64)
m1_modelrflux = np.loadtxt(catfold+'gs_mis_sdss_phot.dat', unpack=True, usecols=(41,))

ind2_m1phot = np.loadtxt(catfold+'photmatchinginds.txt', dtype=np.int64)