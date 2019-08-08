 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import astropy.cosmology as apc
cosmo = apc.Planck15
import astropy.coordinates as coords
from astropy.coordinates import SkyCoord
from scipy.interpolate import griddata

from astropy import units as u
#from sklearn import linear_model
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from demarcations import *
from Gdiffs import *
from Fits_set import *
from setops import *
from matchspec import *
from loaddata import *
catfold='catalogs/'
plt.rc('font', family='serif')

arcsec = 1/3600 #degrees


def getlumfromflux(flux, z):
    distances=np.array(cosmo.luminosity_distance(z))* (3.086e+24) #Mpc to cm
    lum = 4*np.pi*(distances**2)*flux
    return lum
'''
Load info from SDSS DR7
'''

#xxln = Fits_set(catfold+'xll_n_spec.fits')

'''
done separately for the different datatypes
truncation caused integers to produce false matches
load in redshift, star formation rates for the different catalogs
'''

'''
indices for specific gsw properties as they're loaded in above
'''

#commid, ind1_m1, ind2_m1phot = commonpts1d(m1[0], m1_photcatids)
m1_photrfluxmatched = m1_modelrflux[ind2_m1phot]
posflux = np.where(m1_photrfluxmatched >0 )[0]
#np.savetxt(catfold+'photmatchinginds.txt', ind2_m1phot,fmt='%s')
def get_massfrac(fibmass, totmass):
    '''
    get the mass fraction of the fibers
    '''
    val_masses = np.where((fibmass > 6) & ( fibmass < 13)  &(totmass > 6) &(totmass < 13))[0]
    inval_masses =[(fibmass < 6) | (fibmass > 13) | (totmass < 6) | (totmass > 13)]
    mass_frac = np.ones_like(fibmass)
    mass_frac[val_masses] = 10**(fibmass[val_masses])/(10**(totmass[val_masses]))
    mass_frac[inval_masses] *= -99
    return mass_frac, val_masses
all_sdss_massfrac, val_massinds  = get_massfrac(sdssobj.all_fibmass, sdssobj.all_sdss_avgmasses)
def conv_ra(ra):
    '''
    convert ra for plotting spherical aitoff projection.
    '''
    copra = np.copy(ra)
    for i in range(len(ra)):
        if copra[i] > 270:
            copra[i] =- 360 + copra[i]

    return (copra)*(-1)
'''
matching the xray catalog to the gswlc
'''

class GSWCatmatch3xmm:
    def __init__(self, goodinds, gswlcids, redshift, sfrplus, fullflux_filt=[], hardflux_filt=[], hardflux2_filt=[], softflux_filt=[], sedflag=0):
        self.sedflags = sfrplus[sedind]
        self.sedflag = sedflag
        self.inds = goodinds
        self.sedfilt = np.where(self.sedflags[self.inds] == sedflag)[0]
        self.allids = gswlcids[0]
        self.ids = gswlcids[0][self.inds][self.sedfilt]
        self.matchsedflags = sfrplus[sedind][self.inds][self.sedfilt]
        self.gids = gswlcids[1][self.inds][self.sedfilt]
        self.z = redshift[self.inds][self.sedfilt]
        self.allz = redshift
        self.sfr= np.transpose(sfrplus[2][self.inds][self.sedfilt])[0]
        self.allra = sfrplus[raind]
        self.alldec = sfrplus[decind]
        self.matchra =sfrplus[raind][self.inds][self.sedfilt]
        self.matchdec = sfrplus[decind][self.inds][self.sedfilt]
        self.allsfr = sfrplus[sfrind]
        self.matchsfr=sfrplus[sfrind][self.inds][self.sedfilt]
        self.allmass = sfrplus[massind]
        self.matchmass = sfrplus[massind][self.inds][self.sedfilt]
        self.allplate = sfrplus[plateind]
        self.matchplate = sfrplus[plateind][self.inds][self.sedfilt]
        self.allfiber = sfrplus[fiberind]
        self.matchfiber = sfrplus[fiberind][self.inds][self.sedfilt]
        self.allmjds = sfrplus[mjdind]
        self.matchmjd = sfrplus[mjdind][self.inds][self.sedfilt]
        self.allids = gswlcids[0]
        if len(fullflux_filt) != 0:
            self.softlums =getlumfromflux(softflux_filt[self.sedfilt], self.z)
            self.hardlums =getlumfromflux(hardflux_filt[self.sedfilt], self.z)
            self.hardlums2 =getlumfromflux(hardflux2_filt[self.sedfilt], self.z)
    
            self.fulllums =getlumfromflux(fullflux_filt[self.sedfilt], self.z)
    
            self.softlumsrf = np.log10(self.softlums*(1+self.z)**(1.7-2))
            self.hardlumsrf = np.log10(self.hardlums*(1+self.z)**(1.7-2))
            self.hardlums2rf = np.log10(self.hardlums2*(1+self.z)**(1.7-2))
    
            self.fulllumsrf = np.log10(self.fulllums*(1+self.z)**(1.7-2))

class XMM3:
    def __init__(self, xmmcat, xmmcatobs):
        self.sourceids = xmmcat.getcol(0)
        self.ra = xmmcat.getcol(2)
        self.radeg = self.ra*u.degree
        self.dec = xmmcat.getcol(3)
        self.decdeg = self.dec*u.degree
        self.obsids = xmmcatobs.getcol(1)
        self.skyco = SkyCoord(ra=self.radeg, dec=self.decdeg)
        self.tpn = xmmcatobs.getcol(7)
        self.tmos1 = xmmcatobs.getcol(9)
        self.tmos2 = xmmcatobs.getcol(11)
        self.flux1 = xmmcat.getcol(5) #0.2-0.5 keV
        self.flux2 = xmmcat.getcol(7) #0.5-1 keV
        self.flux3 = xmmcat.getcol(9) #1-2 keV
        self.flux4  = xmmcat.getcol(11) #2-4.5
        self.flux5 = xmmcat.getcol(13) #4.5-12
        self.flux8 = xmmcat.getcol(15) #0.2-12
        self.hardflux2 = (self.flux4 + self.flux5)*0.87 #2.547/2.92097
        self.hardflux = (self.flux8 - (self.flux1 + self.flux2 + self.flux3))*0.87 #2.547/2.92097
        self.softflux = self.flux2 + self.flux3
        self.fullflux = (self.flux8-self.flux1)*0.91    #3.94337/4.31728
        #self.flux8 = xmmcat.getcol
    def obsmatching(self, filtinds):
        matchedsetinds = []
        matchedsetids = []
        for i, obsid in enumerate(self.obsids):
            #print(i)
            currentsetinds = []
            currentsetids= []
            for j, sourceid in enumerate(self.sourceids[filtinds]):
                if str(obsid) == str(sourceid)[1:11]:
                    currentsetinds.append(j)
                    currentsetids.append(sourceid)
                else:
                    currentsetids.append(-999)

            matchedsetinds.append(len(currentsetinds))
            matchedsrcids = np.where(currentsetids != -999)[0]
            matchedsetids.append(currentsetids)
        return np.array(matchedsetinds), np.array(matchedsetids)
    def singtimearr(self, matchedsetinds):
        exptimes = []
        exptimesbyobs =[]
        val = np.where(matchedsetinds != 0)[0]
        for ind in val:
            exp = np.array([self.tpn[ind], self.tmos1[ind], self.tmos2[ind]])
            for k in range(matchedsetinds[ind]):
                exptimes.append(np.max(exp))
            exptimesbyobs.append(np.max(exp))
        return np.array(exptimes), np.array(exptimesbyobs)

print('matching catalogs')


x3 = XMM3(xmm3,xmm3obs)

coordgswm = SkyCoord(allm1[raind]*u.degree, allm1[decind]*u.degree)

#coordchand = SkyCoord(ra95*u.degree, dec95*u.degree)

#actmatch_3xmm_gsw = catmatch_act(x3.ra,x3.dec,m1Cat_GSW.allra,m1Cat_GSW.alldec, m1_photrfluxmatched, x3.fullflux)


idm1 = np.loadtxt(catfold+'xmm3gswmatch1indnew.txt')
goodm1 = np.loadtxt(catfold+'xmm3gswmatch1goodnew.txt')
d2d1m = np.loadtxt(catfold+'xmm3gswmmatchd2d1new.txt')
decdiff = np.loadtxt(catfold+'xmm3gswmatchdeltadecnew.txt')
radiff = np.loadtxt(catfold+'xmm3gswmatchdeltaranew.txt')
x3.idm1 = np.int64(idm1)
x3.good_med = np.int64(goodm1)
x3.medmatchinds, x3.medmatchsetids = x3.obsmatching(x3.good_med)
#x3.medmatchindsall, x3.medmatchsetidsall = x3.obsmatching(np.int64(np.ones_like(x3.ra)))

x3.goodobsids = np.array(x3.obsids[np.where(x3.medmatchinds !=0)[0]])

x3.medtimes_all,x3.medtimes_allt = x3.singtimearr(np.int64(np.ones_like(x3.tpn)))
x3.logmedtimes_all = np.log10(x3.medtimes_all)
x3.medtimes_allfilt = np.where((x3.logmedtimes_all>4.1)&(x3.logmedtimes_all <4.5))
x3.medcovgobs = x3.obsids[x3.medtimes_allfilt]
np.savetxt(catfold+'goodobsalltimes.txt',np.array(x3.medcovgobs), fmt='%s')

x3.goodra = np.array(x3.ra[x3.good_med])
x3.gooddec = np.array(x3.dec[x3.good_med])
#x3.allxmm3times, x3.allxmm3timesobs = x3.obsmatching(np.int64(np.ones(x3.ra.size)))
x3.medtimes, x3.medtimesobs = x3.singtimearr(x3.medmatchinds)
x3.logmedtimes = np.log10(x3.medtimes)
x3.logmedtimesobs = np.log10(x3.medtimesobs)
x3.medtimefilt = np.where((x3.logmedtimes > 4.1) & (x3.logmedtimes < 4.5))[0]
x3.medtimeobsfilt = np.where((x3.logmedtimesobs >4.1) & (x3.logmedtimesobs < 4.5))[0]
x3.goodratimefilt = x3.goodra[x3.medtimefilt]
x3.gooddectimefilt = x3.gooddec[x3.medtimefilt]
np.savetxt(catfold+'goodobstimes.txt',np.array(x3.goodobsids[x3.medtimeobsfilt]), fmt='%s')
#x3.medtimefilt = np.where((x3.logmedtimes > 3.5) & (x3.logmedtimes <3.9))[0] #shorter, s82x-esque exposures
x3.medtimefilt_all = np.where(x3.logmedtimes >0 )[0] #all times

x3.softflux_filt = x3.softflux[x3.good_med][x3.medtimefilt]
x3.hardflux_filt = x3.hardflux[x3.good_med][x3.medtimefilt]
x3.hardflux2_filt = x3.hardflux2[x3.good_med][x3.medtimefilt]
x3.fullflux_filt = x3.fullflux[x3.good_med][x3.medtimefilt]

x3.softflux_all = x3.softflux[x3.good_med][x3.medtimefilt_all]
x3.hardflux_all = x3.hardflux[x3.good_med][x3.medtimefilt_all]
x3.hardflux2_all = x3.hardflux2[x3.good_med][x3.medtimefilt_all]
x3.fullflux_all = x3.fullflux[x3.good_med][x3.medtimefilt_all]



#a1Cat_GSW_3xmm = GSWCatmatch3xmm(x3,x3.ida[x3.good_all][x3.alltimefilt], a1, redshift_a1, alla1)
m1Cat_GSW_qsos = GSWCatmatch3xmm( np.arange(len(m1[0])), m1, redshift_m1, allm1, sedflag=1)

m1Cat_GSW_3xmm = GSWCatmatch3xmm(x3.idm1[x3.medtimefilt], m1, redshift_m1, allm1, x3.fullflux_filt, x3.hardflux_filt, x3.hardflux2_filt, x3.softflux_filt)
m1Cat_GSW_3xmm_all = GSWCatmatch3xmm( x3.idm1[x3.medtimefilt_all], m1, redshift_m1, allm1, x3.fullflux_all, x3.hardflux_all, x3.hardflux2_all, x3.softflux_all)
m1Cat_GSW_3xmm.exptimes = x3.logmedtimes[x3.medtimefilt]
m1Cat_GSW_3xmm_all.exptimes = x3.logmedtimes[x3.medtimefilt_all][m1Cat_GSW_3xmm_all.sedfilt]
m1Cat_GSW_3xmm.matchxrayra = x3.goodra[x3.medtimefilt][m1Cat_GSW_3xmm.sedfilt]
m1Cat_GSW_3xmm.matchxraydec = x3.gooddec[x3.medtimefilt][m1Cat_GSW_3xmm.sedfilt]
m1Cat_GSW_3xmm_all.matchxrayra = x3.goodra[x3.medtimefilt_all][m1Cat_GSW_3xmm_all.sedfilt]
m1Cat_GSW_3xmm_all.matchxraydec = x3.gooddec[x3.medtimefilt_all][m1Cat_GSW_3xmm_all.sedfilt]

'''
np.savetxt(catfold+'xmm3gswamatch1ind.txt',ida1)
np.savetxt(catfold+'xmm3gswamatch1good.txt',gooda1)
np.savetxt(catfold+'xmm3gswamatchd2d1.txt',d2d1a)

np.savetxt(catfold+'xmm3gswmmatch1ind.txt',idm1)
np.savetxt(catfold+'xmm3gswmmatch1good.txt',goodm1)
np.savetxt(catfold+'xmm3gswmmatchd2d1.txt',d2d1m)

np.savetxt(catfold+'xmm3gswdmatch1ind.txt',idd1)
np.savetxt(catfold+'xmm3gswdmatch1good.txt',goodd1)
np.savetxt(catfold+'xmm3gswdmatchd2d1.txt',d2d1d)

'''
mpa_spec_m1_3xmm = MPAJHU_Spec(m1Cat_GSW_3xmm, sdssobj)
mpa_spec_qsos = MPAJHU_Spec(m1Cat_GSW_qsos, sdssobj, sedtyp=1)

mpa_spec_m1_3xmm_all = MPAJHU_Spec(m1Cat_GSW_3xmm_all, sdssobj)
mpa_spec_allm1 = MPAJHU_Spec(m1Cat_GSW_3xmm, sdssobj, find=False)

mpa_spec_allm1.spec_inds_prac, mpa_spec_allm1.spec_plates_prac, mpa_spec_allm1.spec_fibers_prac, mpa_spec_allm1.spec_mass_prac, mpa_spec_allm1.make_prac = np.loadtxt(catfold+'gsw_dr7_matching_info.txt')


#spec_inds_m1_3xmm, spec_plates_m1_3xmm, spec_fibers_m1_3xmm, specmass_m1_3xmm, make_m1_3xmm, miss_m1_3xmm, ids_sp_m1_3xmm  = matchspec_full(m1Cat_GSW_3xmm, sdssobj)
#spec_inds_m1_3xmm, spec_plates_m1_3xmm, spec_fibers_m1_3xmm, specmass_m1_3xmm, make_m1_3xmm, miss_m1_3xmm, ids_sp_m1_3xmm  = matchspec_prac(m1Cat_GSW_3xmm, sdssobj)
#spec_inds_m1_3xmm_all, spec_plates_m1_3xmm_all, spec_fibers_m1_3xmm_all, specmass_m1_3xmm_all, make_m1_3xmm_all, miss_m1_3xmm_all, ids_sp_m1_3xmm_all  = matchspec_prac(m1Cat_GSW_3xmm_all, sdssobj)

#spec_inds_allm1, spec_plates_allm1, spec_fibers_allm1, mass_allm1,make_allm1 = np.loadtxt(catfold+'gsw_dr7_matching_info.txt')
mpa_spec_allm1.spec_inds_prac = np.int64(mpa_spec_allm1.spec_inds_prac).reshape(-1)
mpa_spec_allm1.make_prac = np.int64(mpa_spec_allm1.make_prac).reshape(-1)
mpa_spec_qsos.spec_inds_prac = np.int64(mpa_spec_qsos.spec_inds_prac).reshape(-1)
mpa_spec_qsos.make_prac = np.int64(mpa_spec_qsos.make_prac).reshape(-1)

mpa_spec_m1_3xmm.spec_inds_prac = np.int64(mpa_spec_m1_3xmm.spec_inds_prac ).reshape(-1)
mpa_spec_m1_3xmm_all.spec_inds_prac  = np.int64(mpa_spec_m1_3xmm_all.spec_inds_prac ).reshape(-1)

def latextable(table):
    for row in table:
        out= '%s & '*(len(row)-1) +'%s \\\\'
        print(out % tuple(row))
'''
actual analysis begins below
'''

class ELObj:
    def __init__(self, sdssinds, sdss, make_spec, gswcat, gsw = False, xr=False):
        self.halp_sn = np.reshape(sdss.allhalpha[sdssinds]/sdss.allhalpha_err[sdssinds],-1)
        self.nii_sn = np.reshape(sdss.allnII[sdssinds]/sdss.allnII_err[sdssinds],-1)
        self.oi_sn = np.reshape(sdss.alloI[sdssinds]/sdss.alloI_err[sdssinds],-1)
        self.oiii_sn = np.reshape(sdss.alloIII[sdssinds]/sdss.alloIII_err[sdssinds],-1)
        self.hbeta_sn = np.reshape(sdss.allhbeta[sdssinds]/sdss.allhbeta_err[sdssinds],-1)
        self.sii6731_sn = np.reshape(sdss.allSII_6731[sdssinds]/sdss.allSII_6731_err[sdssinds],-1)
        self.sii6717_sn = np.reshape(sdss.allSII_6717[sdssinds]/sdss.allSII_6717_err[sdssinds],-1)
        self.neIII_sn = np.reshape(sdss.allneIII[sdssinds]/sdss.allneIIIerr[sdssinds],-1)
        self.oII_sn = np.reshape(sdss.alloII[sdssinds]/sdss.alloIIerr[sdssinds],-1)

        self.halp_filt = np.where((self.halp_sn > 2) & (self.hbeta_sn > 2) & (self.oiii_sn > 2) & (self.nii_sn > 2))[0]

        self.sdss_filt = sdssinds[self.halp_filt]
        self.halp_filt_bool = (self.halp_sn>2) & (self.hbeta_sn > 2) & (self.oiii_sn > 2) & (self.nii_sn > 2)
        self.not_halp_filt_bool  = np.logical_not(self.halp_filt_bool)
        self.not_halp_filt  = np.where(self.not_halp_filt_bool)
        self.sdss_filt_weak = sdssinds[self.not_halp_filt]

        if gsw:
            self.allmass = gswcat.allmass[make_spec]
            self.allsfr =gswcat.allsfr[make_spec]
            self.allz =  gswcat.allz[make_spec]
            self.mass = gswcat.allmass[make_spec][self.halp_filt]
            self.weakmass = gswcat.allmass[make_spec][self.not_halp_filt]
            self.sfr =gswcat.allsfr[make_spec][self.halp_filt]
            self.weaksfr = gswcat.allsfr[make_spec][self.not_halp_filt]
            self.ssfr= self.sfr-self.mass
            self.weakssfr = self.weaksfr-self.weakmass
            self.z =  gswcat.allz[make_spec][self.halp_filt]
            self.tbt_filtall = np.where((self.neIII_sn>5)&(self.oII_sn >5  ) &
                                 ( gswcat.allz[make_spec]>0.02))[0]   
            self.ssfrtbt = self.allsfr-self.allmass 
            
        else:
            self.allmass = gswcat.matchmass[make_spec]
            self.allsfr =gswcat.matchsfr[make_spec]
            self.allz =  gswcat.allz[make_spec]
            self.mass = gswcat.matchmass[make_spec][self.halp_filt]
            self.weakmass = gswcat.matchmass[make_spec][self.not_halp_filt]
            self.sfr = gswcat.matchsfr[make_spec][self.halp_filt]
            self.weaksfr = gswcat.matchsfr[make_spec][self.not_halp_filt]
            self.ssfr = self.sfr-self.mass
            self.weakssfr = self.weaksfr-self.weakmass
            self.z =  gswcat.z[make_spec][self.halp_filt]
            self.weakz = gswcat.z[make_spec][self.not_halp_filt]
            self.distances=np.array(cosmo.luminosity_distance(self.z))* (3.086e+24) #Mpc to cm
            if len(self.weakz) !=0:
                self.weakdistances = np.array(cosmo.luminosity_distance(self.weakz))*(3.086e+24)
            self.ra = gswcat.matchra[make_spec][self.halp_filt]
            self.dec = gswcat.matchdec[make_spec][self.halp_filt]
            self.xrayra = gswcat.matchxrayra[make_spec][self.halp_filt]
            self.xraydec = gswcat.matchxraydec[make_spec][self.halp_filt]
            self.xrayfulllum = gswcat.fulllumsrf[make_spec][self.halp_filt]
            self.ids = gswcat.ids[make_spec][self.halp_filt]
            self.tbt_filtall = np.where((self.neIII_sn>5)&(self.oII_sn >5  ) &
                                 ( gswcat.z[make_spec]>0.02))[0]   
            self.ssfrtbt = self.allsfr-self.allmass
            
        if xr:
            self.exptimes = gswcat.exptimes[make_spec][self.halp_filt]
        self.yvals_bpt = sdss.alloIII[sdssinds]/sdss.allhbeta[sdssinds]
        self.xvals1_bpt = sdss.allnII[sdssinds]/sdss.allhalpha[sdssinds]
        self.xvals2_bpt = sdss.allSII[sdssinds]/sdss.allhalpha[sdssinds]
        self.xvals3_bpt = sdss.alloI[sdssinds]/sdss.allhalpha[sdssinds]
        self.tbtx = sdss.allneIII[sdssinds]/sdss.alloII[sdssinds]
        self.tbt_filt = np.where((self.neIII_sn[self.halp_filt]>5)&(self.oII_sn[self.halp_filt] >5  ) &
                                 (self.z>0.02))[0]      
   
        
        self.neiiioii = np.log10(np.copy(self.tbtx))[self.halp_filt]
        
        self.niiha = np.log10(np.copy(self.xvals1_bpt))[self.halp_filt]
        self.siiha = np.log10(np.copy(self.xvals2_bpt))[self.halp_filt]
        self.oiha = np.log10(np.copy(self.xvals3_bpt))[self.halp_filt]
        self.oiiihb = np.log10(np.copy(self.yvals_bpt))[self.halp_filt]

        self.massfrac = np.copy(10**(sdss.all_fibmass[self.sdss_filt]))/np.copy(10**(sdss.all_sdss_avgmasses[self.sdss_filt]))

        self.massfracgsw = np.copy(10**(sdss.all_fibmass[self.sdss_filt]))/np.copy(10**(self.mass))
        self.weakmassfracgsw = np.copy(10**(sdss.all_fibmass[self.sdss_filt_weak]))/np.copy(10**(self.weakmass))

        self.weakmassfrac = np.copy(10**(sdss.all_fibmass[self.sdss_filt_weak])/np.copy(10**sdss.all_sdss_avgmasses[self.sdss_filt_weak]))
        self.oiiiflux = np.copy(sdss.alloIII[self.sdss_filt])/1e17
        self.ohabund = np.reshape(sdss.fiboh[self.sdss_filt],-1)

        self.hbetaflux = np.copy(sdss.allhbeta[self.sdss_filt])/1e17
        self.halpflux = np.copy(sdss.allhalpha[self.sdss_filt])/1e17
        self.av = self.extinction()
        self.halpflux_corr = self.dustcorrect(self.halpflux)
        self.oiiiflux_corr = self.dustcorrect(self.oiiiflux)
        self.vdisp = np.copy(sdss.allvdisp[self.sdss_filt])
        self.balmerfwhm = np.copy(sdss.allbalmerdisp[self.sdss_filt]*2 * np.sqrt(2*np.log(2)))
        self.forbiddenfwhm = np.copy(sdss.allforbiddendisp[self.sdss_filt]*2 * np.sqrt(2*np.log(2)))
        self.forbiddenfwhmerr = np.copy(sdss.allforbiddendisperr[self.sdss_filt]*2*np.sqrt(2*np.log(2)))
        self.balmerfwhmerr = np.copy(sdss.allbalmerdisperr[self.sdss_filt]*2*np.sqrt(2*np.log(2)))

        self.gswfilt = self.halp_filt
        self.oiiilum = getlumfromflux(self.oiiiflux_corr,self.z)
        self.halplum = getlumfromflux(self.halpflux_corr, self.z)
        self.oiiilum_uncorr = getlumfromflux(self.oiiiflux,self.z)
        self.halplum_uncorr = getlumfromflux(self.halpflux, self.z) #units from mpa/jhu

        self.fibmass = sdss.all_fibmass[self.sdss_filt]
        self.fibsfr = np.copy(self.sfr + np.log10(self.massfrac))
        self.weakfibsfr = np.copy(self.weaksfr+np.log10(self.weakmassfrac))
        self.fibsfrgsw = np.copy(self.sfr + np.log10(self.massfracgsw))
        self.weakfibsfrgsw = np.copy(self.weaksfr+np.log10(self.weakmassfracgsw))
        self.fibssfr = np.copy(self.fibsfr-self.fibmass)
        
        #self.fibsfr = np.copy(self.sfr*self.massfrac)
        #self.fibsfr_corr = self.halptofibsfr_corr()
        #self.fibsfr_uncorr = self.halptofibsfr_uncorr()
        #self.oiiilumsfr = np.log10(np.copy(self.oiiilum))- np.copy(self.fibsfr_uncorr)
        #self.oiiilumfiboh = np.log10(np.copy(self.oiiilum))-self.ohabund
        #self.fibssfr1 = self.fibsfr_corr - self.fibmass
        #self.fibssfr = self.fibsfr_uncorr - self.fibmass
    def get_bpt1_groups(self, filt=[]):
        groups =[]
        if len(filt) != 0:
            xvals = np.copy(self.niiha[filt])
            yvals = np.copy(self.oiiihb[filt])
        else:
            xvals = np.copy(self.niiha)
            yvals = np.copy(self.oiiihb)
        for i in range(len(xvals)):
            if xvals[i] < 0:
                if yvals[i] <np.log10(y1_kauffman(xvals[i])):
                    groups.append('HII')
                else:
                    groups.append('AGN')
            else:
                groups.append('AGN')
        groups=np.array(groups)
        agn = np.where(groups == 'AGN')[0]
        nonagn = np.where(groups == 'HII')[0]
        return groups,nonagn, agn
    def get_bpt2_groups(self):
        groups=[]
        xvals = np.copy(self.siiha)
        yvals = np.copy(self.oiiihb)
        for i in range(len(xvals)):
            if xvals[i] < 0.32:
                if yvals[i] < np.log10(y2_agn(xvals[i])):
                    groups.append('HII')
                    continue
            if yvals[i] > np.log10(y2_linersy2(xvals[i])):
                groups.append('Seyfert')
            else:
                groups.append('LINER')
        groups = np.array(groups)
        hii = np.where(groups == 'HII')[0]
        seyf = np.where(groups == 'Seyfert')[0]
        liner = np.where(groups == 'LINER')[0]
        return groups,hii, seyf, liner
    def get_bpt3_groups(self):
        groups=[]
        xvals= np.copy(self.oiha)
        yvals= np.copy(self.oiiihb)
        for i in range(len(xvals)):
            if np.isnan(xvals[i]):
                groups.append('NO')
                continue
            if xvals[i] < -0.59:
                if yvals[i] < np.log10(y3_agn(xvals[i])):
                    groups.append('HII')
                    continue
            if  yvals[i] > np.log10(y3_linersy2(xvals[i])):
                groups.append('Seyfert')
            else:
                groups.append('LINER')
        groups = np.array(groups)
        hii = np.where(groups == 'HII')[0]
        seyf = np.where(groups=='Seyfert')[0]
        liner = np.where(groups=='LINER')[0]
        return groups,hii,seyf,liner#hii,seyf,liner
    def extinction(self):
        av = 7.23*np.log10((self.halpflux/self.hbetaflux) / 2.86) # A(V) mag
        return av
    def dustcorrect(self,flux):
        return flux*10**(0.4*self.av*1.120)
    def halptofibsfr_corr(self):
        logsfrfib = np.log10(7.9e-42)+np.log10(self.halplum)-0.24#+0.4*A(H_alpha)
        return logsfrfib
    def halptofibsfr_uncorr(self):
        logsfrfib = np.log10(7.9e-42)+np.log10(self.halplum_uncorr)-0.24+0.4*self.av
        return logsfrfib

EL_qsos = ELObj(mpa_spec_qsos.spec_inds_prac , sdssobj, mpa_spec_qsos.make_prac,m1Cat_GSW_qsos,gsw=True)

EL_m1 = ELObj(mpa_spec_allm1.spec_inds_prac , sdssobj, mpa_spec_allm1.make_prac,m1Cat_GSW_3xmm,gsw=True)
EL_3xmm  = ELObj(mpa_spec_m1_3xmm.spec_inds_prac , sdssobj, mpa_spec_m1_3xmm.make_prac,m1Cat_GSW_3xmm, xr=True)
EL_3xmm_all = ELObj(mpa_spec_m1_3xmm_all.spec_inds_prac , sdssobj, mpa_spec_m1_3xmm_all.make_prac, m1Cat_GSW_3xmm_all, xr=True)

groups1_3xmmm,nonagn_3xmmm, agn_3xmmm = EL_3xmm.get_bpt1_groups()
groups1_3xmmm_all,nonagn_3xmmm_all, agn_3xmmm_all = EL_3xmm_all.get_bpt1_groups()

groups1_gsw, nonagn_gsw, agn_gsw = EL_m1.get_bpt1_groups()
groups1_gswtbt, nonagn_gswtbt, agn_gswtbt = EL_m1.get_bpt1_groups(filt=EL_m1.tbt_filt)


groups1_qsos, nonagn_qsos, agn_qsos = EL_qsos.get_bpt1_groups()

groups1_gsw_no, nonagn_gsw_no, agn_gsw_no = EL_m1.get_bpt1_groups()
groups1_gsw_oiii, nonagn_gsw_oiii, agn_gsw_oiii = EL_m1.get_bpt1_groups()

'''
XRAY LUM FILT PLOTS
'''
lum_arr = np.logspace(34,46,1000)
loglum_arr = np.log10(lum_arr)

sfrsoft = 2.2e-40*lum_arr
logsfrsoft = np.log10(sfrsoft)#from salpeter IMF
logsfrsoft = logsfrsoft - 0.2 #converted to Chabrier IMF

sfrhard = 2.0e-40*lum_arr
logsfrhard = np.log10(sfrhard)#from salpeter IMF
logsfrhard =logsfrhard - 0.2 #converted to Chabrier IMF
#combine the supposed sfr form soft and hard to get the full
sfrfull= 1.05e-40*lum_arr
logsfrfull = np.log10(sfrfull) -0.2
xrayranallidict = {'soft':1/(1.39e-40),'hard':1/(1.26e-40),'full':1/(0.66e-40)}


class Xraysfr:
    def __init__(self, xraylums, gswcat, filt, agn, nonagn, typ):
        self.typ = typ
        self.mass = gswcat.matchmass
        self.z = gswcat.z
        self.z_filt = self.z[filt]
        self.lum_mass =  xraylums-gswcat.matchmass #mass_m1_match
        self.lum = xraylums
        self.sfr_mass = gswcat.matchsfr-gswcat.matchmass# sfr_m1_match-mass_m1_match
        self.sfr = gswcat.matchsfr
        self.lxsfr = np.log10(xrayranallidict[typ]*10**(self.sfr))

        self.notagn = np.where(self.lum[filt]==0)
        self.mass_filt = self.mass[filt]
        self.lum_mass_filt = self.lum_mass[filt]
        self.sfr_filt = self.sfr[filt]
        self.sfr_mass_filt = self.sfr_mass[filt]
        self.lum_filt = self.lum[filt]
        #self.mess = np.where((self.lum_mass_val >29) & (self.lum_mass_val<30) & (self.softsfr_mass_val >-11))[0]
        self.valid = np.where(self.lum[filt] >0)[0]
        self.validagn = np.where(self.lum[filt][agn]>0)[0]
        self.validnoagn = np.where(self.lum[filt][nonagn]>0)[0]
        #valid overall
        self.lum_mass_val_filt =  self.lum_mass[filt][self.valid] #mass_m1_match
        self.lum_val_filt = self.lum[filt][self.valid]
        self.sfr_mass_val_filt = self.sfr_mass[filt][self.valid]
        self.sfr_val_filt = self.sfr[filt][self.valid]
        self.mass_val_filt = self.mass[filt][self.valid]
        #valid bpt agn
        self.lum_mass_val_filtagn =  self.lum_mass[filt][agn][self.validagn]
        self.lum_val_filtagn = self.lum[filt][agn][self.validagn]
        self.sfr_mass_val_filtagn = self.sfr_mass[filt][agn][self.validagn]
        self.sfr_val_filtagn = self.sfr[filt][agn][self.validagn]
        self.mass_filtagn = self.mass[filt][agn][self.validagn]

        #valid bpt hii
        self.lum_mass_val_filtnoagn =  self.lum_mass[filt][nonagn][self.validnoagn]
        self.lum_val_filtnoagn = self.lum[filt][nonagn][self.validnoagn]
        self.sfr_mass_val_filtnoagn = self.sfr_mass[filt][nonagn][self.validnoagn]
        self.sfr_val_filtnoagn = self.sfr[filt][nonagn][self.validnoagn]
        self.mass_filtnoagn = self.mass[filt][nonagn][self.validnoagn]

        self.likelyagn_xr = np.where((self.lxsfr[filt][self.valid] < self.lum[filt][self.valid] - 0.6) &
                                     (self.lum[filt][self.valid] > 0))[0]
        self.likelyagnbpthii = np.where((self.lxsfr[filt][nonagn][self.validnoagn] < self.lum[filt][nonagn][self.validnoagn]-0.6) &
                                        (self.lum[filt][nonagn][self.validnoagn] >0))[0]
        self.likelyagnbptagn = np.where((self.lxsfr[filt][agn][self.validagn] < self.lum[filt][agn][self.validagn]-0.6) &
                                        (self.lum[filt][agn][self.validagn] >0))[0]
        self.likelysf = np.where((abs(self.lxsfr[filt][self.valid] - self.lum[filt][self.valid]) < 0.3) &
                                     (self.lum[filt][self.valid] > 0))[0]
        self.likelysfbpthii = np.where((abs(self.lxsfr[filt][nonagn][self.validnoagn] - self.lum[filt][nonagn][self.validnoagn])<0.3) &
                                        (self.lum[filt][nonagn][self.validnoagn] >0))[0]
        self.likelysfbptagn = np.where((abs(self.lxsfr[filt][agn][self.validagn] - self.lum[filt][agn][self.validagn])<0.3) &
                                        (self.lum[filt][agn][self.validagn] >0))[0]
        #valid xray agn
        self.lum_mass_val_filt_xrayagn =  self.lum_mass[filt][self.valid][self.likelyagn_xr]
        self.lum_val_filt_xrayagn = self.lum[filt][self.valid][self.likelyagn_xr ]
        self.sfr_mass_val_filt_xrayagn = self.sfr_mass[filt][self.valid][self.likelyagn_xr]
        self.sfr_val_filt_xrayagn = self.sfr[filt][self.valid][self.likelyagn_xr ]
        self.mass_filt_xrayagn = self.mass[filt][self.valid][self.likelyagn_xr ]
        self.z_filt_xrayagn = self.z[filt][self.valid][self.likelyagn_xr]
        
        #valid bpt agn

        self.lum_mass_val_filtagn_xrayagn =  self.lum_mass[filt][agn][self.validagn][self.likelyagnbptagn]
        self.lum_val_filtagn_xrayagn = self.lum[filt][agn][self.validagn][self.likelyagnbptagn]
        self.sfr_mass_val_filtagn_xrayagn = self.sfr_mass[filt][agn][self.validagn][self.likelyagnbptagn]
        self.sfr_val_filtagn_xrayagn = self.sfr[filt][agn][self.validagn][self.likelyagnbptagn]
        self.mass_filtagn_xrayagn = self.mass[filt][agn][self.validagn][self.likelyagnbptagn]

        #valid bpt hii
        self.lum_mass_val_filtnoagn_xrayagn =  self.lum_mass[filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.lum_val_filtnoagn_xrayagn = self.lum[filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.sfr_mass_val_filtnoagn_xrayagn = self.sfr_mass[filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.sfr_val_filtnoagn_xrayagn = self.sfr[filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.mass_filtnoagn_xrayagn = self.mass[filt][nonagn][self.validnoagn][self.likelyagnbpthii]

softxray_xmm = Xraysfr(m1Cat_GSW_3xmm.softlumsrf, m1Cat_GSW_3xmm, mpa_spec_m1_3xmm.make_prac[EL_3xmm.halp_filt], agn_3xmmm,nonagn_3xmmm, 'soft')
hardxray_xmm = Xraysfr(m1Cat_GSW_3xmm.hardlumsrf, m1Cat_GSW_3xmm, mpa_spec_m1_3xmm.make_prac[EL_3xmm.halp_filt], agn_3xmmm,nonagn_3xmmm,'hard')
fullxray_xmm = Xraysfr(m1Cat_GSW_3xmm.fulllumsrf, m1Cat_GSW_3xmm, mpa_spec_m1_3xmm.make_prac[EL_3xmm.halp_filt], agn_3xmmm, nonagn_3xmmm, 'full')

fullxray_xmm_dr7 = Xraysfr(m1Cat_GSW_3xmm.fulllumsrf, m1Cat_GSW_3xmm, mpa_spec_m1_3xmm.make_prac, agn_3xmmm, nonagn_3xmmm, 'full')

softxray_xmm_all = Xraysfr(m1Cat_GSW_3xmm_all.softlumsrf, m1Cat_GSW_3xmm_all, mpa_spec_m1_3xmm_all.make_prac[EL_3xmm_all.halp_filt], agn_3xmmm_all,nonagn_3xmmm_all, 'soft')
hardxray_xmm_all = Xraysfr(m1Cat_GSW_3xmm_all.hardlumsrf, m1Cat_GSW_3xmm_all, mpa_spec_m1_3xmm_all.make_prac[EL_3xmm_all.halp_filt], agn_3xmmm_all,nonagn_3xmmm_all, 'hard')
fullxray_xmm_all = Xraysfr(m1Cat_GSW_3xmm_all.fulllumsrf, m1Cat_GSW_3xmm_all, mpa_spec_m1_3xmm_all.make_prac[EL_3xmm_all.halp_filt], agn_3xmmm_all, nonagn_3xmmm_all, 'full')


softxray_xmm_no = Xraysfr(m1Cat_GSW_3xmm.softlumsrf, m1Cat_GSW_3xmm, mpa_spec_m1_3xmm.make_prac[EL_3xmm.not_halp_filt_bool], [], [], 'soft')
hardxray_xmm_no = Xraysfr(m1Cat_GSW_3xmm.hardlumsrf, m1Cat_GSW_3xmm, mpa_spec_m1_3xmm.make_prac[EL_3xmm.not_halp_filt_bool], [], [], 'hard')
fullxray_xmm_no = Xraysfr(m1Cat_GSW_3xmm.fulllumsrf, m1Cat_GSW_3xmm, mpa_spec_m1_3xmm.make_prac[EL_3xmm.not_halp_filt_bool], [], [], 'full')

xmm3eldiagmed_xrfilt = ELObj(mpa_spec_m1_3xmm.spec_inds_prac[EL_3xmm.halp_filt][fullxray_xmm.valid][fullxray_xmm.likelyagn_xr], sdssobj,
                             mpa_spec_m1_3xmm.make_prac[EL_3xmm.halp_filt][fullxray_xmm.valid][fullxray_xmm.likelyagn_xr], m1Cat_GSW_3xmm,xr=True)

xmm3eldiagmed_xrsffilt = ELObj(mpa_spec_m1_3xmm.spec_inds_prac[EL_3xmm.halp_filt][fullxray_xmm.valid][fullxray_xmm.likelysf], sdssobj,
                             mpa_spec_m1_3xmm.make_prac[EL_3xmm.halp_filt][fullxray_xmm.valid][fullxray_xmm.likelysf], m1Cat_GSW_3xmm,xr=True)



groups1_3xmmm, nonagn_3xmmm_xrfilt, agn_3xmmm_xrfilt = xmm3eldiagmed_xrfilt.get_bpt1_groups()
groups1_3xmmmtbt, nonagn_3xmmm_xrfilttbt, agn_3xmmm_xrfilttbt = xmm3eldiagmed_xrfilt.get_bpt1_groups(filt=xmm3eldiagmed_xrfilt.tbt_filt)
groups1_3xmmmtbtonly, nonagn_3xmmm_xrfilttbtonly, agn_3xmmm_xrfilttbtonly = xmm3eldiagmed_xrfilt.get_bpt1_groups(filt=xmm3eldiagmed_xrfilt.tbt_filtall)

lowestmassgals = np.where(xmm3eldiagmed_xrfilt.mass[nonagn_3xmmm_xrfilt] <9.2)

lowssfr = np.where(xmm3eldiagmed_xrfilt.ssfr[nonagn_3xmmm_xrfilt] < -11)[0]

highssfrbptagn = np.where(xmm3eldiagmed_xrfilt.ssfr[agn_3xmmm_xrfilt]>-10)[0]

xmm3eldiagmed_xrfilt_all =  ELObj(mpa_spec_m1_3xmm_all.spec_inds_prac[EL_3xmm_all.halp_filt][fullxray_xmm_all.valid][fullxray_xmm_all.likelyagn_xr], sdssobj,
                                  mpa_spec_m1_3xmm_all.make_prac[EL_3xmm_all.halp_filt][fullxray_xmm_all.valid][fullxray_xmm_all.likelyagn_xr], m1Cat_GSW_3xmm_all, xr=True)

groups1_3xmmm_all, nonagn_3xmmm_all_xrfilt, agn_3xmmm_all_xrfilt = xmm3eldiagmed_xrfilt_all.get_bpt1_groups()

groups1_3xmmm_alltbt, nonagn_3xmmm_all_xrfilttbt, agn_3xmmm_all_xrfilttbt = xmm3eldiagmed_xrfilt_all.get_bpt1_groups(filt=xmm3eldiagmed_xrfilt_all.tbt_filt)

groups1_3xmmm_xrsffilt, nonagn_3xmmm_xrsffilt, agn_3xmmm_xrsffilt = xmm3eldiagmed_xrsffilt.get_bpt1_groups()



'''
Distance analysis begins here
'''
#GDIFFS
xmm3inds = m1Cat_GSW_3xmm.inds[mpa_spec_m1_3xmm.make_prac][EL_3xmm.halp_filt][fullxray_xmm.valid][fullxray_xmm.likelyagn_xr]
xmm3ids = m1Cat_GSW_3xmm.ids[mpa_spec_m1_3xmm.make_prac][EL_3xmm.halp_filt][fullxray_xmm.valid][fullxray_xmm.likelyagn_xr]
gswids = m1[0][mpa_spec_allm1.make_prac][EL_m1.halp_filt]
covered_gsw = np.int64(np.loadtxt('catalogs/matched_gals_s82_fields.txt'))
covered_gsw_x3 = np.int64(np.loadtxt('catalogs/xraycov/matched_gals_xmm3_xrcovg_fields_set.txt'))

xmm3gdiff = Gdiffs(xmm3ids, gswids, xmm3eldiagmed_xrfilt, EL_m1)
xmm3gdiff.get_filt(covered_gsw_x3)
commdiffidsxmm3, xmm3diffcomm, xmm3gswdiffcomm = commonpts1d(xmm3ids, gswids[covered_gsw_x3])
xmm3gdiff.nearbyx(xmm3gswdiffcomm)
xmm3gdiff.getdist_by_thresh(3.0)
binnum = 60
contaminations_xmm3 = xmm3gdiff.xrgswfracs[:,binnum]
contaminations_xmm3_2 = xmm3gdiff.xrgswfracs[:, 40]
contaminations_xmm3_25 = xmm3gdiff.xrgswfracs[:, 50]
contaminations_xmm3_3 = xmm3gdiff.xrgswfracs[:, 60]
contaminations_xmm3_35 = xmm3gdiff.xrgswfracs[:, 70]
contaminations_xmm3_4 = xmm3gdiff.xrgswfracs[:, 80]
xmm3gdiff.interpdistgrid(11,11,50,method='linear')
#a,e = xmm3gdiff.maskgrid()
#xmm3gdiff.grid[e,a] = np.nan
#cont = xmm3gdiff

#xmm3gdiff.interpdistgrid(10,10,50,method='linear')
#cont2 = xmm3gdiff
#gsweldiagcov = ELDiag(gsw_xvals1_bpt, gsw_xvals2_bpt, gsw_xvals3_bpt,
#                      m1Cat_GSW.allmass[make_allm1], gsw_yvals_bpt,
#                      m1Cat_GSW.allz[make_allm1],
#                      m1Cat_GSW.allsfr[make_allm1],
#                      sdssobj,halp_filt[valid_bpt][stripe82[covered_gsw]],      make_allm1[halp_filt][valid_bpt][stripe82[covered_gsw]])


'''sf
match sfr
'''

class SFRMatch:
    def __init__(self, eldiag):
        self.eldiag=eldiag
    def getsfrmatch(self, agn_inds, sf_inds, lim=0.2):
        '''
        Matches BPT AGN to BPT SF of same SFR/M*
        '''
        agns = []
        sfs = []
        for agn in agn_inds:
            diffsfr = abs(self.eldiag.sfr[agn] - self.eldiag.sfr[sf_inds])
            diffmass = abs(self.eldiag.mass[agn] - self.eldiag.mass[sf_inds])
            mindiff_ind = np.where(( diffsfr == np.min(diffsfr))&(diffmass ==np.min(diffmass) ) &
                                   (np.min(diffsfr) +np.min(diffmass) < lim))[0]
            
            if mindiff_ind.size > 0:

                agns.append(agn)
                sfs.append(sf_inds[mindiff_ind][0])
        self.agns = np.array(agns)
        self.sfs = np.array(sfs)
def write_to_fil_obs(filnam):
    f = open(filnam,'w+')
    #f.write("#Name, OBJID, RA, DEC, BPT CAT\n")
    f.write("#Name, OBJID, RA, DEC, Exp time, SN Min, SN Min Line, SN Max, SN Max Line, z\n")

    sn =  np.vstack([xmm3eldiagmed_xrfilt.halp_sn,xmm3eldiagmed_xrfilt.nii_sn,                   xmm3eldiagmed_xrfilt.oiii_sn,xmm3eldiagmed_xrfilt.hbeta_sn])
    snmin = np.min(sn, axis= 0)
    snmax = np.max(sn, axis= 0)
    sncodemin = matchcode(sn, snmin)
    sncodemax = matchcode(sn, snmax)

    filtxmm = mpa_spec_m1_3xmm.make_prac[EL_3xmm.halp_filt][fullxray_xmm.valid][fullxray_xmm.likelyagn_xr][nonagn_3xmmm_xrfilt]

    for i in range(len(filtxmm)):
        #f.write('3XMMHII'+str(i)+','+str(m1Cat_GSW_3xmm.ids[filtxmm][i])+','+str(m1Cat_GSW_3xmm.matchra[filtxmm][i])+','+str(m1Cat_GSW_3xmm.matchdec[filtxmm][i])+',HII\n')
        f.write('3XMMHII'+'-'+str(i)+','+str(m1Cat_GSW_3xmm.ids[filtxmm][i])+','+                    str(m1Cat_GSW_3xmm.matchra[filtxmm][i])+','+str(m1Cat_GSW_3xmm.matchdec[filtxmm][i])+
                             ','+str(m1Cat_GSW_3xmm.exptimes[filtxmm][i])+',' +               str(snmin[nonagn_3xmmm_xrfilt][i])+','+sncodemin[nonagn_3xmmm_xrfilt][i]+','+str(snmax[nonagn_3xmmm_xrfilt][i])+','+ sncodemax[nonagn_3xmmm_xrfilt][i]+','+str(xmm3eldiagmed_xrfilt.z[nonagn_3xmmm_xrfilt][i])+'\n')

    f.close()
#write_to_fil_obs('sdssobjlookup.txt')

codes = {0:'halp', 1:'nii', 2:'oiii', 3:'hbeta'}
def matchcode(sn,sndes):
    code = []
    for i in range(len(sndes)):
        whermin = np.where(sn[:,i] == sndes[i])[0]
        code.append(codes[whermin[0]])
    return np.array(code)

def write_to_fil_obs_supp(filnam):
    f = open(filnam,'w+')
    f.write("#Name, OBJID, RA, DEC, Exp time, SN Min, SN Min Line, SN Max, SN Max Line, z\n")
    filtxmm = make_m1_3xmm_all[EL_3xmm_all.halp_filt][fullxray_xmm_all.valid][fullxray_xmm_all.likelyagn_xr][nonagn_3xmmm_all_xrfilt]
    sn =  np.vstack([xmm3eldiagmed_xrfilt_all.halp_sn,xmm3eldiagmed_xrfilt_all.nii_sn,xmm3eldiagmed_xrfilt_all.oiii_sn,xmm3eldiagmed_xrfilt_all.hbeta_sn])
    snmin = np.min(sn, axis= 0)
    snmax = np.max(sn, axis= 0)
    sncodemin = matchcode(sn, snmin)
    sncodemax = matchcode(sn, snmax)

    for i in range(len(filtxmm)):
        print(m1Cat_GSW_3xmm_all.matchra[filtxmm][i])
        f.write('3XMMHII'+'-'+str(i)+','+str(m1Cat_GSW_3xmm_all.ids[filtxmm][i])+','+str(m1Cat_GSW_3xmm_all.matchra[filtxmm][i])+','+str(m1Cat_GSW_3xmm_all.matchdec[filtxmm][i])+
                ','+str(m1Cat_GSW_3xmm_all.exptimes[filtxmm][i])+','+str(snmin[nonagn_3xmmm_all_xrfilt][i])+','+
                sncodemin[nonagn_3xmmm_all_xrfilt][i]+','+str(snmax[nonagn_3xmmm_all_xrfilt][i])+','+sncodemax[nonagn_3xmmm_all_xrfilt][i]+','+str(xmm3eldiagmed_xrfilt_all.z[nonagn_3xmmm_all_xrfilt][i])+'\n')
    #+str(round(m1Cat_GSW_3xmm_all.exptimes[filtxmm][i],4))
    f.close()
#write_to_fil_obs_supp('observing_sample_supp.txt')
