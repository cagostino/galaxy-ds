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
from loaddata_m1 import *
from loaddata_sdss_xr import *
from ELObj import *
from XMM3_obj import *
from xraysfr_obj import *
from gsw_3xmm_match import *
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
x3.medmatchinds, x3.medmatchsetids, x3.medmatchobsinds, x3.medmatchobsids, x3.medexptimes = x3.obsmatching(x3.good_med)
#x3.medmatchindsall, x3.medmatchsetidsall = x3.obsmatching(np.int64(np.ones_like(x3.ra)))

x3.goodobsids = np.array(x3.obsids[np.where(x3.medmatchinds !=0)[0]])

x3.medtimes_all,x3.medtimes_allt, x3.obsids_allt = x3.singtimearr(np.int64(np.ones_like(x3.tpn)))
x3.logmedtimes_all = np.log10(x3.medtimes_all)
x3.medtimes_allfilt = np.where((x3.logmedtimes_all>4.1)&(x3.logmedtimes_all <4.5))
x3.medcovgobs = x3.obsids[x3.medtimes_allfilt]
np.savetxt(catfold+'goodobsalltimes.txt',np.array(x3.medcovgobs), fmt='%s')

x3.goodra = np.array(x3.ra[x3.good_med])
x3.gooddec = np.array(x3.dec[x3.good_med])
#x3.allxmm3times, x3.allxmm3timesobs = x3.obsmatching(np.in:t64(np.ones(x3.ra.size)))
x3.medtimes, x3.medtimesobs, x3.obsids_matches = x3.singtimearr(x3.medmatchinds)
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
x3.efullflux_filt = x3.efullflux[x3.good_med][x3.medtimefilt]
x3.qualflag_filt = x3.qualflag[x3.good_med][x3.medtimefilt]
x3.ext_filt = x3.ext[x3.good_med][x3.medtimefilt_all]

x3.softflux_all = x3.softflux[x3.good_med][x3.medtimefilt_all]
x3.hardflux_all = x3.hardflux[x3.good_med][x3.medtimefilt_all]
x3.hardflux2_all = x3.hardflux2[x3.good_med][x3.medtimefilt_all]
x3.fullflux_all = x3.fullflux[x3.good_med][x3.medtimefilt_all]
x3.efullflux_all = x3.efullflux[x3.good_med][x3.medtimefilt_all]
x3.qualflag_all = x3.qualflag[x3.good_med][x3.medtimefilt_all]
x3.ext_all = x3.ext[x3.good_med][x3.medtimefilt_all]


#a1Cat_GSW_3xmm = GSWCatmatch3xmm(x3,x3.ida[x3.good_all][x3.alltimefilt], a1, redshift_a1, alla1)m1Cat_GSW_qsos = GSWCat( np.arange(len(m1[0])), m1, redshift_m1, allm1, sedflag=1)

m1Cat_GSW = GSWCat( np.arange(len(m1[0])), m1, redshift_m1, allm1)


m1Cat_GSW_3xmm = GSWCatmatch3xmm(x3.idm1[x3.medtimefilt], m1, redshift_m1, allm1, x3.qualflag_filt,
                                 x3.fullflux_filt, x3.efullflux_filt, 
                                 x3.hardflux_filt, x3.hardflux2_filt, x3.softflux_filt, x3.ext_filt)
m1Cat_GSW_3xmm_all = GSWCatmatch3xmm( x3.idm1[x3.medtimefilt_all], m1, 
                                     redshift_m1, allm1, x3.qualflag_all, x3.fullflux_all, 
                                     x3.efullflux_all,x3.hardflux_all, 
                                     x3.hardflux2_all, x3.softflux_all, x3.ext_all)
m1Cat_GSW_3xmm.exptimes = np.log10(x3.medexptimes)[x3.medtimefilt][m1Cat_GSW_3xmm.sedfilt]
m1Cat_GSW_3xmm_all.exptimes = x3.logmedtimes[x3.medtimefilt_all][m1Cat_GSW_3xmm_all.sedfilt]
m1Cat_GSW_3xmm.matchxrayra = x3.goodra[x3.medtimefilt][m1Cat_GSW_3xmm.sedfilt]
m1Cat_GSW_3xmm.matchxraydec = x3.gooddec[x3.medtimefilt][m1Cat_GSW_3xmm.sedfilt]
#m1Cat_GSW_3xmm.xraysourceids = x3.sourceids[x3.good_med][x3.medtimefilt][m1Cat_GSW_3xmm.sedfilt]
m1Cat_GSW_3xmm.xrayobsids =  x3.obsids_matches[x3.medtimefilt][m1Cat_GSW_3xmm.sedfilt]
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

#%% mpa matching

mpa_spec_m1_3xmm = MPAJHU_Spec(m1Cat_GSW_3xmm, sdssobj)
#mpa_spec_qsos = MPAJHU_Spec(m1Cat_GSW_qsos, sdssobj, sedtyp=1)
mpa_spec_m1_3xmm_all = MPAJHU_Spec(m1Cat_GSW_3xmm_all, sdssobj)
mpa_spec_allm1 = MPAJHU_Spec(m1Cat_GSW, sdssobj, find=False, gsw=True)
#gsw_2matching_info = np.vstack([mpa_spec_allm1.spec_inds_prac, mpa_spec_allm1.spec_plates_prac, mpa_spec_allm1.spec_fibers_prac, mpa_spec_allm1.spec_mass_prac, mpa_spec_allm1.make_prac ])
mpa_spec_allm1.spec_inds_prac, mpa_spec_allm1.spec_plates_prac, mpa_spec_allm1.spec_fibers_prac, mpa_spec_allm1.spec_mass_prac, mpa_spec_allm1.make_prac = np.loadtxt(catfold+'gsw_dr7_matching_info.txt')

inds_comm, gsw_sedfilt_mpamake,mpamake_gsw_sedfilt=np.intersect1d(m1Cat_GSW.sedfilt, mpa_spec_allm1.make_prac, return_indices=True)
#new way of setting up gsw_3xmm_match filters sed flag beforehand so some of 
#the previous objects are now eing leftout (all QSOs/bad SEDfits so not a problem)
#need to combine inds from m1Cat_GSW.sedfilt ana mpa_spec_allm1.make_prac




#spec_inds_m1_3xmm, spec_plates_m1_3xmm, spec_fibers_m1_3xmm, specmass_m1_3xmm, make_m1_3xmm, miss_m1_3xmm, ids_sp_m1_3xmm  = matchspec_full(m1Cat_GSW_3xmm, sdssobj)
#spec_inds_m1_3xmm, spec_plates_m1_3xmm, spec_fibers_m1_3xmm, specmass_m1_3xmm, make_m1_3xmm, miss_m1_3xmm, ids_sp_m1_3xmm  = matchspec_prac(m1Cat_GSW_3xmm, sdssobj)
#spec_inds_m1_3xmm_all, spec_plates_m1_3xmm_all, spec_fibers_m1_3xmm_all, specmass_m1_3xmm_all, make_m1_3xmm_all, miss_m1_3xmm_all, ids_sp_m1_3xmm_all  = matchspec_prac(m1Cat_GSW_3xmm_all, sdssobj)

#spec_inds_allm1, spec_plates_allm1, spec_fibers_allm1, mass_allm1,make_allm1 = np.loadtxt(catfold+'gsw_dr7_matching_info.txt')
mpa_spec_allm1.spec_inds_prac = np.int64(mpa_spec_allm1.spec_inds_prac).reshape(-1)
mpa_spec_allm1.make_prac = np.int64(mpa_spec_allm1.make_prac).reshape(-1)
#mpa_spec_qsos.spec_inds_prac = np.int64(mpa_spec_qsos.spec_inds_prac).reshape(-1)
#mpa_spec_qsos.make_prac = np.int64(mpa_spec_qsos.make_prac).reshape(-1)
mpa_spec_m1_3xmm.spec_inds_prac = np.int64(mpa_spec_m1_3xmm.spec_inds_prac ).reshape(-1)
mpa_spec_m1_3xmm_all.spec_inds_prac  = np.int64(mpa_spec_m1_3xmm_all.spec_inds_prac ).reshape(-1)

def latextable(table):
    for row in table:
        out= '%s & '*(len(row)-1) +'%s \\\\'
        print(out % tuple(row))


'''
actual analysis begins below
'''

#%% Emission line objects
#EL_qsos = ELObj(mpa_spec_qsos.spec_inds_prac , sdssobj, mpa_spec_qsos.make_prac,m1Cat_GSW_qsos,gsw=True)
EL_m1 = ELObj(mpa_spec_allm1.spec_inds_prac , sdssobj, gsw_sedfilt_mpamake,m1Cat_GSW,gsw=True, dustbinning=True)

EL_3xmm  = ELObj(mpa_spec_m1_3xmm.spec_inds_prac , sdssobj, mpa_spec_m1_3xmm.make_prac,m1Cat_GSW_3xmm, xr=True)
EL_3xmm_all = ELObj(mpa_spec_m1_3xmm_all.spec_inds_prac , sdssobj, mpa_spec_m1_3xmm_all.make_prac, m1Cat_GSW_3xmm_all, xr=True)


#%% X-ray lum sfr filtering
'''
XRAY LUM -SFR
'''


softxray_xmm = Xraysfr(m1Cat_GSW_3xmm.softlumsrf, m1Cat_GSW_3xmm, 
                       mpa_spec_m1_3xmm.make_prac[EL_3xmm.bpt_sn_filt], 
                       EL_3xmm.bptagn,EL_3xmm.bptsf, 'soft')
hardxray_xmm = Xraysfr(m1Cat_GSW_3xmm.hardlumsrf, m1Cat_GSW_3xmm, 
                       mpa_spec_m1_3xmm.make_prac[EL_3xmm.bpt_sn_filt], 
                       EL_3xmm.bptagn,EL_3xmm.bptsf,'hard')
fullxray_xmm = Xraysfr(m1Cat_GSW_3xmm.fulllumsrf, m1Cat_GSW_3xmm,
                       mpa_spec_m1_3xmm.make_prac[EL_3xmm.bpt_sn_filt], 
                       EL_3xmm.bptagn, EL_3xmm.bptsf, 'full')
fullxray_xmm_bptplus = Xraysfr(m1Cat_GSW_3xmm.fulllumsrf, m1Cat_GSW_3xmm,
                       mpa_spec_m1_3xmm.make_prac[EL_3xmm.bpt_sn_filt], 
                       EL_3xmm.bptplsagn, EL_3xmm.bptplssf, 'full')


fullxray_xmm_dr7 = Xraysfr(m1Cat_GSW_3xmm.fulllumsrf, m1Cat_GSW_3xmm, mpa_spec_m1_3xmm.make_prac,  EL_3xmm.bptagn, EL_3xmm.bptsf, 'full')

softxray_xmm_all = Xraysfr(m1Cat_GSW_3xmm_all.softlumsrf, m1Cat_GSW_3xmm_all, 
                           mpa_spec_m1_3xmm_all.make_prac[EL_3xmm_all.bpt_sn_filt], 
                           EL_3xmm_all.bptagn,EL_3xmm_all.bptsf, 'soft')
hardxray_xmm_all = Xraysfr(m1Cat_GSW_3xmm_all.hardlumsrf, m1Cat_GSW_3xmm_all,
                           mpa_spec_m1_3xmm_all.make_prac[EL_3xmm_all.bpt_sn_filt],
                           EL_3xmm_all.bptagn,EL_3xmm_all.bptsf, 'hard')
fullxray_xmm_all = Xraysfr(m1Cat_GSW_3xmm_all.fulllumsrf, m1Cat_GSW_3xmm_all, 
                           mpa_spec_m1_3xmm_all.make_prac[EL_3xmm_all.bpt_sn_filt],
                           EL_3xmm_all.bptagn, EL_3xmm_all.bptsf, 'full')

fullxray_xmm_all_bptplus = Xraysfr(m1Cat_GSW_3xmm_all.fulllumsrf, m1Cat_GSW_3xmm_all, 
                                   mpa_spec_m1_3xmm_all.make_prac[EL_3xmm_all.bpt_sn_filt], 
                                   EL_3xmm_all.bptplsagn, EL_3xmm_all.bptplssf, 'full')


softxray_xmm_no = Xraysfr(m1Cat_GSW_3xmm.softlumsrf, m1Cat_GSW_3xmm, mpa_spec_m1_3xmm.make_prac[EL_3xmm.not_bpt_sn_filt_bool], [], [], 'soft')
hardxray_xmm_no = Xraysfr(m1Cat_GSW_3xmm.hardlumsrf, m1Cat_GSW_3xmm, mpa_spec_m1_3xmm.make_prac[EL_3xmm.not_bpt_sn_filt_bool], [], [], 'hard')
fullxray_xmm_no = Xraysfr(m1Cat_GSW_3xmm.fulllumsrf, m1Cat_GSW_3xmm, mpa_spec_m1_3xmm.make_prac[EL_3xmm.not_bpt_sn_filt_bool], [], [], 'full')
fullxray_xmm_all_no = Xraysfr(m1Cat_GSW_3xmm_all.fulllumsrf, m1Cat_GSW_3xmm_all, mpa_spec_m1_3xmm_all.make_prac[EL_3xmm_all.not_bpt_sn_filt_bool], [], [], 'full')


#%% refiltering emission line objects by x-ray properties
xmm3eldiagmed_xrfilt = ELObj(mpa_spec_m1_3xmm.spec_inds_prac[EL_3xmm.bpt_sn_filt][fullxray_xmm.valid][fullxray_xmm.likelyagn_xr], 
                             sdssobj,
                             mpa_spec_m1_3xmm.make_prac[EL_3xmm.bpt_sn_filt][fullxray_xmm.valid][fullxray_xmm.likelyagn_xr], 
                             m1Cat_GSW_3xmm,xr=True)

xmm3eldiagmed_xrfiltbptplus = ELObj(mpa_spec_m1_3xmm.spec_inds_prac[EL_3xmm.bpt_sn_filt][fullxray_xmm_bptplus.valid][fullxray_xmm_bptplus.likelyagn_xr], sdssobj,
                             mpa_spec_m1_3xmm.make_prac[EL_3xmm.bpt_sn_filt][fullxray_xmm_bptplus.valid][fullxray_xmm_bptplus.likelyagn_xr], m1Cat_GSW_3xmm,xr=True)

xmm3eldiagmed_xrsffilt = ELObj(mpa_spec_m1_3xmm.spec_inds_prac[EL_3xmm.bpt_sn_filt][fullxray_xmm.valid][fullxray_xmm.likelysf], sdssobj,
                             mpa_spec_m1_3xmm.make_prac[EL_3xmm.bpt_sn_filt][fullxray_xmm.valid][fullxray_xmm.likelysf], m1Cat_GSW_3xmm,xr=True)

xmm3eldiagmed_xrfilt_all =  ELObj(mpa_spec_m1_3xmm_all.spec_inds_prac[EL_3xmm_all.bpt_sn_filt][fullxray_xmm_all.valid][fullxray_xmm_all.likelyagn_xr], sdssobj,
                                  mpa_spec_m1_3xmm_all.make_prac[EL_3xmm_all.bpt_sn_filt][fullxray_xmm_all.valid][fullxray_xmm_all.likelyagn_xr], m1Cat_GSW_3xmm_all, xr=True)
xmm3eldiagmed_xrfiltbptplus_all =  ELObj(mpa_spec_m1_3xmm_all.spec_inds_prac[EL_3xmm_all.bpt_sn_filt][fullxray_xmm_all_bptplus.valid][fullxray_xmm_all_bptplus.likelyagn_xr], 
                                         sdssobj,
                                  mpa_spec_m1_3xmm_all.make_prac[EL_3xmm_all.bpt_sn_filt][fullxray_xmm_all_bptplus.valid][fullxray_xmm_all_bptplus.likelyagn_xr], 
                                  m1Cat_GSW_3xmm_all, xr=True)





#groups1_3xmmm, nonagn_3xmmm_xrfilt, agn_3xmmm_xrfilt = xmm3eldiagmed_xrfilt.get_bpt1_groups()
#groups1_3xmmmplus, nonagn_3xmmm_xrfilt_bptplus, agn_3xmmm_xrfilt_bptplus = xmm3eldiagmed_xrfilt.get_bptplus_groups()
#groups1_3xmmmplus, nonagn_3xmmm_xrfilt_bptplus, agn_3xmmm_xrfilt_bptplus = xmm3eldiagmed_xrfiltbptplus.get_bptplus_groups()#

#groups1_3xmmmplusnii, nonagn_3xmmm_xrfilt_bptplusnii, agn_3xmmm_xrfilt_bptplusnii = xmm3eldiagmed_xrfilt.get_bptplus_niigroups()
#groups1_3xmmmtbt, nonagn_3xmmm_xrfilttbt, agn_3xmmm_xrfilttbt = xmm3eldiagmed_xrfilt.get_bpt1_groups(filt=xmm3eldiagmed_xrfilt.bpt_sn_filt[xmm3eldiagmed_xrfilt.tbt_filt])
#groups1_3xmmmvo87_1, nonagn_3xmmm_xrfiltvo87_1, agn_3xmmm_xrfiltvo87_1 = xmm3eldiagmed_xrfilt.get_bpt1_groups(filt=xmm3eldiagmed_xrfilt.bpt_sn_filt[xmm3eldiagmed_xrfilt.vo87_1_filt])
#groups1_3xmmmvo87_2, nonagn_3xmmm_xrfiltvo87_2, agn_3xmmm_xrfiltvo87_2 = xmm3eldiagmed_xrfilt.get_bpt1_groups(filt=xmm3eldiagmed_xrfilt.bpt_sn_filt[xmm3eldiagmed_xrfilt.vo87_2_filt])

#groups1_3xmmmtbtonly, nonagn_3xmmm_xrfilttbtonly, agn_3xmmm_xrfilttbtonly = xmm3eldiagmed_xrfilt.get_bpt1_groups(filt=xmm3eldiagmed_xrfilt.tbt_filtall)

#lowestmassgals = np.where(xmm3eldiagmed_xrfilt.mass[nonagn_3xmmm_xrfilt] <9.2)
#lowssfr = np.where(xmm3eldiagmed_xrfilt.ssfr[nonagn_3xmmm_xrfilt] < -11)[0]
#highssfrbptagn = np.where(xmm3eldiagmed_xrfilt.ssfr[agn_3xmmm_xrfilt]>-10)[0]


#groups1_3xmmm_all, nonagn_3xmmm_all_xrfilt, agn_3xmmm_all_xrfilt = xmm3eldiagmed_xrfilt_all.get_bpt1_groups()
#groups1_3xmmm_alltbt, nonagn_3xmmm_all_xrfilttbt, agn_3xmmm_all_xrfilttbt = xmm3eldiagmed_xrfilt_all.get_bpt1_groups(filt=xmm3eldiagmed_xrfilt_all.bpt_sn_filt[xmm3eldiagmed_xrfilt_all.tbt_filt])
#groups1_3xmmm_xrsffilt, nonagn_3xmmm_xrsffilt, agn_3xmmm_xrsffilt = xmm3eldiagmed_xrsffilt.get_bpt1_groups()

#groups1_3xmmmbptplus_all, nonagn_3xmmmbptplus_all_xrfilt, agn_3xmmmbptplus_all_xrfilt = xmm3eldiagmed_xrfiltbptplus_all.get_bpt1_groups()
#groups1_3xmmmbptplus_all_bptplus, nonagn_3xmmmbptplus_all_xrfilt_bptplus, agn_3xmmmbptplus_all_xrfilt_bptplus = xmm3eldiagmed_xrfiltbptplus_all.get_bptplus_groups()


#obs_gals_inds = [1,2,4,5,6,8,9,10,12,13,14,16,21,52,60,76,78,81,82,83,84,85,86,87]


#unclass = mpa_spec_m1_3xmm_all.make_prac[EL_3xmm_all.not_bpt_sn_filt][fullxray_xmm_all_no.likelyagn_xr]
#unclass_EL_3xmm_all_xragn = EL_3xmm_all.not_bpt_sn_filt[fullxray_xmm_all_no.likelyagn_xr]
#muse_samp_inds = [3,6, 28, 57] #7,27, 46
#unclass_muse_samp_inds = [22,23,57,104,210]

#het_samp_inds = [11, 13, 16,  21, 22,  79, 82, 85, 87] #9,10, 14, 20, 77
#unclass_het_samp_inds = [68, 87, 91, 102, 110, 483, 487, 497, 510, 523, 525, 526, 534]


#%% Gdiffs

'''
Gdiffs
'''
xmm3inds = m1Cat_GSW_3xmm.inds[m1Cat_GSW_3xmm.sedfilt][mpa_spec_m1_3xmm.make_prac][EL_3xmm.bpt_sn_filt][fullxray_xmm.valid][fullxray_xmm.likelyagn_xr]
xmm3ids = m1Cat_GSW_3xmm.ids[mpa_spec_m1_3xmm.make_prac][EL_3xmm.bpt_sn_filt][fullxray_xmm.valid][fullxray_xmm.likelyagn_xr]
gswids = m1[0][mpa_spec_allm1.make_prac][EL_m1.bpt_sn_filt]
covered_gsw = np.int64(np.loadtxt('catalogs/matched_gals_s82_fields.txt'))
covered_gsw_x3 = np.int64(np.loadtxt('catalogs/xraycov/matched_gals_xmm3_xrcovg_fields_set.txt'))

xmm3gdiff = Gdiffs(xmm3ids, gswids, xmm3eldiagmed_xrfilt.bpt_EL_gsw_df, EL_m1.bpt_EL_gsw_df)
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

xmm3eldiagmed_xrfilt.bpt_EL_gsw_df['xrfracs']=contaminations_xmm3_25


top_sf = np.where(xmm3eldiagmed_xrfilt.bpt_sf_df.oiiihb>0)[0]
mid_sf = np.where((xmm3eldiagmed_xrfilt.bpt_sf_df.oiiihb<0) &(xmm3eldiagmed_xrfilt.bpt_sf_df.oiiihb>-0.5))[0]
bot_sf = np.where((xmm3eldiagmed_xrfilt.bpt_sf_df.oiiihb<-0.5) &(xmm3eldiagmed_xrfilt.bpt_sf_df.oiiihb>-1))[0]

np.mean(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df.xrfracs.iloc[xmm3eldiagmed_xrfilt.bptsf[bot_sf]])
np.mean(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df.xrfracs.iloc[xmm3eldiagmed_xrfilt.bptsf[mid_sf]])
np.mean(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df.xrfracs.iloc[xmm3eldiagmed_xrfilt.bptsf[top_sf]])
np.mean(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df.xrfracs.iloc[xmm3eldiagmed_xrfilt.bptsf])

top_agn = np.where(xmm3eldiagmed_xrfilt.bpt_agn_df.oiiihb>1)[0]
agn2 = np.where((xmm3eldiagmed_xrfilt.bpt_agn_df.oiiihb<1)&(xmm3eldiagmed_xrfilt.bpt_agn_df.oiiihb>0.5))[0]
agn3 = np.where((xmm3eldiagmed_xrfilt.bpt_agn_df.oiiihb<0.5)&(xmm3eldiagmed_xrfilt.bpt_agn_df.oiiihb>0.))[0]
bot_agn = np.where(xmm3eldiagmed_xrfilt.bpt_agn_df.oiiihb<0)[0]

np.mean(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df.xrfracs.iloc[xmm3eldiagmed_xrfilt.bptagn])
np.mean(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df.xrfracs.iloc[xmm3eldiagmed_xrfilt.bptagn[top_agn]])
np.mean(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df.xrfracs.iloc[xmm3eldiagmed_xrfilt.bptagn[agn2]])
np.mean(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df.xrfracs.iloc[xmm3eldiagmed_xrfilt.bptagn[agn3]])
np.mean(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df.xrfracs.iloc[xmm3eldiagmed_xrfilt.bptagn[bot_agn]])
