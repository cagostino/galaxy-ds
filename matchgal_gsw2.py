import matplotlib
import matplotlib.pyplot as plt
import astropy.cosmology as apc
cosmo = apc.Planck15
from loaddata_sdss_xr import *
import pickle

import astropy.coordinates as coords
from astropy.coordinates import SkyCoord
from scipy.interpolate import griddata
from astropy import units as u
#from sklearn import linear_model
import numpy as np

from scipy.ndimage.filters import gaussian_filter


arcsec = 1/3600. #degrees



from demarcations import *
from Gdiffs import *
from Fits_set import *
from setops import *
from matchspec import *
from loaddata_m2 import *
from ast_func import *
from catmatching_xr import *
from ELObj import *
from xraysfr_obj import *
from gsw_3xmm_match import *
catfold='catalogs/'
plt.rc('font', family='serif')





#commid, ind1_m2, ind2_m2phot = commonpts1d(m2[0], m1_photcatids)
#m2_photrfluxmatched = m1_modelrflux[ind2_m1phot]
#posflux = np.where(m2_photrfluxmatched >0 )[0]
#np.savetxt(catfold+'photmatchinginds.txt', ind2_m2phot,fmt='%s')
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
#all_sdss_massfrac, val_massinds  = get_massfrac(sdssobj.all_fibmass, sdssobj.all_sdss_avgmasses)

def get_stat_quant(quant, quantmin=0, quantmax=999):
    quant_val = quant[(quant>quantmin)&(quant<quantmax)]
    mn = np.nanmean(quant_val)
    med = np.nanmedian(quant_val)
    q16 = np.quantile(quant_val,.16)
    q84 = np.quantile(quant_val, .84)
    return mn, med,q16, q84
    

'''
matching the xray catalog to the gswlc
'''

#%% GSW loading

#a1Cat_GSW_3xmm = GSWCatmatch3xmm(x3,x3.ida[x3.good_all][x3.alltimefilt], a1, redshift_a1, alla1)
m2Cat_GSW_qsos = GSWCat( np.arange(len(m2[0])),  m2_df, sedflag=1)

m2Cat_GSW = GSWCat( np.arange(len(m2[0])), m2_df)


m2Cat_GSW_4xmm = GSWCatmatch3xmm(x4.idm2[x4.medtimefilt], m2_df, x4.qualflag, #softflux here should be qualflag
                                 x4.fullflux_filt, x4.efullflux_filt, 
                                 x4.hardflux_filt, x4.ehardflux_filt,x4.hardflux2_filt, x4.softflux_filt,x4.esoftflux_filt, x4.ext_filt, #second softflux should be ext
                                 x4.HR1_filt, x4.HR2_filt, x4.HR3_filt, x4.HR4_filt)



m2Cat_GSW_4xmm_all = GSWCatmatch3xmm( x4.idm2[x4.medtimefilt_all], m2_df, 
                                     x4.qualflag_all, x4.fullflux_all, 
                                     x4.efullflux_all,x4.hardflux_all, x4.ehardflux_all,
                                     x4.hardflux2_all, x4.softflux_all,x4.esoftflux_all, x4.ext_all,
                                     x4.HR1_all, x4.HR2_all, x4.HR3_all, x4.HR4_all)

m2Cat_GSW_4xmm_all_qsos = GSWCatmatch3xmm( x4.idm2[x4.medtimefilt_all], m2_df, 
                                     x4.qualflag_all, x4.fullflux_all, 
                                     x4.efullflux_all,x4.hardflux_all, x4.ehardflux_all,
                                     x4.hardflux2_all, x4.softflux_all, x4.esoftflux_all,x4.ext_all,
                                     x4.HR1_all, x4.HR2_all, x4.HR3_all, x4.HR4_all, sedflag=1)




#%% mpa matching


def latextable(table):
    for row in table:
        out= '%s & '*(len(row)-1) +'%s \\\\'
        print(out % tuple(row))


'''
actual analysis begins below
'''

#%% Emission line objects
EL_qsos = ELObj(mpa_spec_qsos.spec_inds_prac , sdssobj, mpa_spec_qsos.make_prac,m2Cat_GSW_qsos,gsw=True)
EL_m2 = ELObj(mpa_spec_allm2.spec_inds_prac , sdssobj, gsw_sedfilt_mpamake, m2Cat_GSW,gsw=True, dustbinning=True, empirdust=True)
EL_first = ELObj(mpa_spec_allm2_first.spec_inds_prac , sdssobj, mpa_spec_allm2_first.make_prac,m2Cat_GSW_first,gsw=True, dustbinning=False, empirdust=False, radio = True )

EL_m2.EL_gsw_df.to_csv('EL_m2_df.csv')
EL_m2.bpt_EL_gsw_df.to_csv('EL_m2_bpt_EL_gsw_df.csv')
EL_4xmm  = ELObj(mpa_spec_m2_4xmm.spec_inds_prac , sdssobj, mpa_spec_m2_4xmm.make_prac,m2Cat_GSW_4xmm, xr=True, xmm=True, empirdust=False)
EL_4xmm.EL_gsw_df.to_csv('EL_4xmm_df.csv')

EL_4xmm_all = ELObj(mpa_spec_m2_4xmm_all.spec_inds_prac , sdssobj, mpa_spec_m2_4xmm_all.make_prac, m2Cat_GSW_4xmm_all, xr=True, xmm=True, empirdust=False)
EL_4xmm_all.EL_gsw_df.to_csv('EL_4xmm_all_df.csv')
EL_4xmm_all.not_bpt_EL_gsw_df.to_csv('EL_4xmm_not_bpt_EL_gsw_df.csv')
EL_4xmm_all.bpt_sf_df.to_csv('EL_4xmm_bpt_sf_df.csv')
EL_4xmm_all_qsos = ELObj(mpa_spec_m2_4xmm_all_qsos.spec_inds_prac , sdssobj, mpa_spec_m2_4xmm_all_qsos.make_prac, m2Cat_GSW_4xmm_all_qsos, xr=True, xmm=True, empirdust=False)


#%% X-ray lum sfr filtering
'''
XRAY LUM -SFR
'''



fullxray_xmm4 = Xraysfr(np.array(m2Cat_GSW_4xmm.gsw_df.fulllumsrf), m2Cat_GSW_4xmm,
                       mpa_spec_m2_4xmm.make_prac[EL_4xmm.bpt_sn_filt], 
                       EL_4xmm.bptagn, EL_4xmm.bptsf, 'full')
fullxray_xmm4_bptplus = Xraysfr(np.array(m2Cat_GSW_4xmm.gsw_df.fulllumsrf), m2Cat_GSW_4xmm,
                       mpa_spec_m2_4xmm.make_prac[EL_4xmm.bpt_sn_filt], 
                       EL_4xmm.bptplsagn, EL_4xmm.bptplssf, 'full')

fullxray_xmm4_dr7 = Xraysfr(np.array(m2Cat_GSW_4xmm.gsw_df.fulllumsrf), m2Cat_GSW_4xmm, mpa_spec_m2_4xmm.make_prac,  EL_4xmm.bptagn, EL_4xmm.bptsf, 'full')

fullxray_xmm4_all = Xraysfr(np.array(m2Cat_GSW_4xmm_all.gsw_df.fulllumsrf), m2Cat_GSW_4xmm_all, 
                           mpa_spec_m2_4xmm_all.make_prac[EL_4xmm_all.bpt_sn_filt],
                           EL_4xmm_all.bptagn, EL_4xmm_all.bptsf, 'full')
fullxray_xmm4_all_bptplus = Xraysfr(np.array(m2Cat_GSW_4xmm_all.gsw_df.fulllumsrf), m2Cat_GSW_4xmm_all, 
                           mpa_spec_m2_4xmm_all.make_prac[EL_4xmm_all.bpt_sn_filt],
                           EL_4xmm_all.bptplsagn, EL_4xmm_all.bptplssf, 'full')



fullxray_xmm4_all_high_sn_o3 = Xraysfr(np.array(m2Cat_GSW_4xmm_all.gsw_df.fulllumsrf), m2Cat_GSW_4xmm_all, 
                           mpa_spec_m2_4xmm_all.make_prac[EL_4xmm_all.high_sn_o3],
                           np.arange(len(EL_4xmm_all.high_sn_o3)), [], 'full')
fullxray_xmm4_all_xragn = Xraysfr(np.array(m2Cat_GSW_4xmm_all.gsw_df.fulllumsrf), m2Cat_GSW_4xmm_all, 
                           mpa_spec_m2_4xmm_all.make_prac,
                           np.arange(len(EL_4xmm_all.EL_gsw_df)), [], 'full')
fullxray_xmm4_all_bptplus = Xraysfr(np.array(m2Cat_GSW_4xmm_all.gsw_df.fulllumsrf), m2Cat_GSW_4xmm_all, 
                                   mpa_spec_m2_4xmm_all.make_prac[EL_4xmm_all.bpt_sn_filt], 
                                   EL_4xmm_all.bptplsagn, EL_4xmm_all.bptplssf, 'full')
fullxray_xmm4_all_unclass_p2 = Xraysfr(np.array(m2Cat_GSW_4xmm_all.gsw_df.fulllumsrf), m2Cat_GSW_4xmm_all,
                              mpa_spec_m2_4xmm_all.make_prac[EL_4xmm_all.neither_filt], [], [], 'full')

fullxray_xmm4_all_no = Xraysfr(np.array(m2Cat_GSW_4xmm_all.gsw_df.fulllumsrf), m2Cat_GSW_4xmm_all,
                              mpa_spec_m2_4xmm_all.make_prac[EL_4xmm_all.not_bpt_sn_filt_bool], [], [], 'full')
#%% refiltering emission line objects by x-ray properties
xmm4eldiagmed_xrfilt = ELObj(mpa_spec_m2_4xmm.spec_inds_prac[EL_4xmm.bpt_sn_filt][fullxray_xmm4.valid][fullxray_xmm4.likelyagn_xr], 
                             sdssobj,
                             mpa_spec_m2_4xmm.make_prac[EL_4xmm.bpt_sn_filt][fullxray_xmm4.valid][fullxray_xmm4.likelyagn_xr], 
                             m2Cat_GSW_4xmm,xr=True, xmm=True)

xmm4eldiagmed_xrfilt_all =  ELObj(mpa_spec_m2_4xmm_all.spec_inds_prac[EL_4xmm_all.bpt_sn_filt][fullxray_xmm4_all.valid][fullxray_xmm4_all.likelyagn_xr], sdssobj,
                                  mpa_spec_m2_4xmm_all.make_prac[EL_4xmm_all.bpt_sn_filt][fullxray_xmm4_all.valid][fullxray_xmm4_all.likelyagn_xr], 
                                  m2Cat_GSW_4xmm_all,xr=True, xmm=True)
xmm4eldiagmed_sffilt_all =  ELObj(mpa_spec_m2_4xmm_all.spec_inds_prac[EL_4xmm_all.bpt_sn_filt][fullxray_xmm4_all.valid][fullxray_xmm4_all.likelysf], sdssobj,
                                  mpa_spec_m2_4xmm_all.make_prac[EL_4xmm_all.bpt_sn_filt][fullxray_xmm4_all.valid][fullxray_xmm4_all.likelysf], 
                                  m2Cat_GSW_4xmm_all,xr=True, xmm=True)

xmm4eldiagmed_xrfilt_xragn =  ELObj(mpa_spec_m2_4xmm_all.spec_inds_prac[fullxray_xmm4_all_xragn.valid][fullxray_xmm4_all_xragn.likelyagn_xr], 
                                    sdssobj,
                                  mpa_spec_m2_4xmm_all.make_prac[fullxray_xmm4_all_xragn.valid][fullxray_xmm4_all_xragn.likelyagn_xr], 
                                  m2Cat_GSW_4xmm_all,xr=True, xmm=True)


xmm4eldiagmed_xrsffilt = ELObj(mpa_spec_m2_4xmm_all.spec_inds_prac[EL_4xmm_all.bpt_sn_filt][fullxray_xmm4_all.valid][fullxray_xmm4_all.likelysf], sdssobj,
                             mpa_spec_m2_4xmm_all.make_prac[EL_4xmm_all.bpt_sn_filt][fullxray_xmm4_all.valid][fullxray_xmm4_all.likelysf], m2Cat_GSW_4xmm_all,xr=True, xmm=True)


xmm4eldiagmed_xrfilt_unclass_p2 =  ELObj(mpa_spec_m2_4xmm_all.spec_inds_prac[EL_4xmm_all.neither_filt][fullxray_xmm4_all_unclass_p2.valid][fullxray_xmm4_all_unclass_p2.likelyagn_xr], 
                                    sdssobj,
                                  mpa_spec_m2_4xmm_all.make_prac[EL_4xmm_all.neither_filt][fullxray_xmm4_all_unclass_p2.valid][fullxray_xmm4_all_unclass_p2.likelyagn_xr], 
                                  m2Cat_GSW_4xmm_all,xr=True, xmm=True, weak_lines=True)
xmm4eldiagmed_xrfilt_unclass_p1 =  ELObj(mpa_spec_m2_4xmm_all.spec_inds_prac[EL_4xmm_all.not_bpt_sn_filt_bool][fullxray_xmm4_all_no.valid][fullxray_xmm4_all_no.likelyagn_xr], 
                                    sdssobj,
                                  mpa_spec_m2_4xmm_all.make_prac[EL_4xmm_all.not_bpt_sn_filt_bool][fullxray_xmm4_all_no.valid][fullxray_xmm4_all_no.likelyagn_xr], 
                                  m2Cat_GSW_4xmm_all,xr=True, xmm=True)




#obs_gals_inds = [1,2,4,5,6,8,9,10,12,13,14,16,21,52,60,76,78,81,82,83,84,85,86,87]


#unclass = mpa_spec_m2_3xmm_all.make_prac[EL_3xmm_all.not_bpt_sn_filt][fullxray_xmm_all_no.likelyagn_xr]
#unclass_EL_3xmm_all_xragn = EL_3xmm_all.not_bpt_sn_filt[fullxray_xmm_all_no.likelyagn_xr]
#muse_samp_inds = [3,6, 28, 57] #7,27, 46
#unclass_muse_samp_inds = [22,23,57,104,210]

#het_samp_inds = [11, 13, 16,  21, 22,  79, 82, 85, 87] #9,10, 14, 20, 77
#unclass_het_samp_inds = [68, 87, 91, 102, 110, 483, 487, 497, 510, 523, 525, 526, 534]


#%% Gdiffs

'''
Gdiffs
'''
gswids = m2[0][mpa_spec_allm2.make_prac][EL_m2.bpt_sn_filt]
covered_gsw_x3 = np.int64(np.loadtxt('catalogs/xraycov/matched_gals_gsw2xmm3_xrcovg_fields_set.txt'))

'''
xmm3gdiff = Gdiffs(xmm3ids, gswids, xmm3eldiagmed_xrfilt.bpt_EL_gsw_df, EL_m2.bpt_EL_gsw_df, x_xr=xmm3eldiagmed_xrfilt.bpt_EL_gsw_df['niiha'],
y_xr=xmm3eldiagmed_xrfilt.bpt_EL_gsw_df['oiiihb'], x_gsw=EL_m2.bpt_EL_gsw_df['niiha'], y_gsw=EL_m2.bpt_EL_gsw_df['oiiihb'])
    
xmm3gdiff.get_filt(covered_gsw_x3)
commdiffidsxmm3, xmm3diffcomm, xmm3gswdiffcomm = commonpts1d(xmm3ids, gswids[covered_gsw_x3])
xmm3gdiff.nearbyx(xmm3gswdiffcomm)
xmm3gdiff.getdist_by_thresh(3.0)
'''

def get_gdiffs_set(xr_ids, gswids, covered_inds, xr_df, gsw_df, quantities):
    
    gdiff_set = Gdiffs(xr_ids, gswids, xr_df, gsw_df, quantities)
    commdiffids, x3diffcomm, x3gswdiffcomm = commonpts1d(xr_ids, gswids[covered_inds])
    gdiff_set.get_filt(covered_inds)    

    gdiff_set.nearbyx(x3gswdiffcomm)
    gdiff_set.getdist_by_thresh(3.0)
    return gdiff_set
#%% sfrm

'''

match sfr
'''


#kmnclust = np.loadtxt('catalogs/kmeans2.dat', unpack=True)



#kmnclust2 =  kmnclust[1]
'''
val2 = np.where(kmnclust2 != 0 )[0]
sy2_1 = np.where(kmnclust2==1)[0]
hliner_1 = np.where(kmnclust2==2)[0]
sliner_1 = np.where(kmnclust2==3)[0]
'''
from sfrmatch import SFRMatch

sfrm_gsw2 = SFRMatch(EL_m2,EL_m2.bpt_EL_gsw_df,
                     EL_m2.plus_EL_gsw_df, EL_m2.neither_EL_gsw_df)

sfrm_gsw2.get_highsn_match_only(EL_m2.bptplsagn, EL_m2.bptplssf, 
                                EL_m2.bptplsnii_sf, EL_m2.bptplsnii_agn, second=True,
                                load=True,  sncut_forb=2,sncut_balm=2, with_av=True, fname='_second_1117')

sfrm_gsw2.subtract_elflux(sncut=2,  halphbeta_sncut=2, second=False)

'''



sfrm_gsw2_old = SFRMatch(EL_m2,EL_m2.bpt_EL_gsw_df,
                     EL_m2.plus_EL_gsw_df, EL_m2.neither_EL_gsw_df )
sfrm_gsw2_old.get_highsn_match_only(EL_m2.bptplsagn, EL_m2.bptplssf, 
                                EL_m2.bptplsnii_sf, EL_m2.bptplsnii_agn, 
                                load=True,   sncut_forb=-9999,sncut_balm=-9999, with_av=True, fname='', old=True, subfold='old_matching/')

sfrm_gsw2_old.subtract_elflux(sncut=2,  halphbeta_sncut=2, second=False)


'''

'''
sfrm_gsw2_best = SFRMatch(EL_m2,EL_m2.bpt_EL_gsw_df,
                     EL_m2.plus_EL_gsw_df, EL_m2.neither_EL_gsw_df )
sfrm_gsw2_best.get_highsn_match_only(EL_m2.bptplsagn, EL_m2.bptplssf, 
                                EL_m2.bptplsnii_sf, EL_m2.bptplsnii_agn, 
                                load=True,  sncut_forb=-9999,sncut_balm=-9999, with_av=True, fname='_no_pos_sn_req', old=False)
sfrm_gsw2_best.subtract_elflux(sncut=2,  halphbeta_sncut=2, second=False)

sfrm_gsw2_half = SFRMatch(EL_m2,EL_m2.bpt_EL_gsw_df,
                     EL_m2.plus_EL_gsw_df, EL_m2.neither_EL_gsw_df )
sfrm_gsw2_half.get_highsn_match_only(EL_m2.bptplsagn, EL_m2.bptplssf, 
                                EL_m2.bptplsnii_sf, EL_m2.bptplsnii_agn, 
                                load=True, sncut_forb=2, sncut_balm= -99, with_av=True, fname='_no_pos_balm_req', old=False)
sfrm_gsw2_half.subtract_elflux(sncut=2,  halphbeta_sncut=2, second=False)



sfrm_gsw2_best0 = SFRMatch(EL_m2,EL_m2.bpt_EL_gsw_df,
                     EL_m2.plus_EL_gsw_df, EL_m2.neither_EL_gsw_df )
sfrm_gsw2_best0.get_highsn_match_only(EL_m2.bptplsagn, EL_m2.bptplssf, 
                                EL_m2.bptplsnii_sf, EL_m2.bptplsnii_agn, 
                                load=True,  sncut=0, with_av=True, fname='_pos_sn_req', old=False)

67071

sfrm_gsw2_best0.agn_dist_inds = np.delete(sfrm_gsw2_best0.agn_dist_inds,22309)
sfrm_gsw2_best0.agn_plus_dist_inds = np.insert(sfrm_gsw2_best0.agn_plus_dist_inds, 10702, 67071)

sfrm_gsw2_best0.subtract_elflux(sncut=2,  halphbeta_sncut=2, second=False)


'''

full_agn_filt= np.where(sfrm_gsw2.allagn_df.full_agn)[0]
#sfrm_gsw2.fullagn_df = sfrm_gsw2.allagn_df.iloc[full_agn_filt].copy()


kmnclust = np.loadtxt('catalogs/kmeans_sfrm.dat', unpack=True)

kmnclust1 =  kmnclust[full_agn_filt]
val1 = np.where(kmnclust1 != 0 )[0]
sy2_1 = np.where(kmnclust1==1)[0]
hliner_1 = np.where(kmnclust1==2)[0]
sliner_1 = np.where(kmnclust1==3)[0]
allliners = np.append(sliner_1, hliner_1)



kmnclust_woo1 = np.loadtxt('catalogs/kmeans_sfrm_woo1.dat', unpack=True)

kmnclust2=  kmnclust_woo1[full_agn_filt]
val2 = np.where(kmnclust2 != 0 )[0]
sy2_2 = np.where(kmnclust2==1)[0]
hliner_2 = np.where(kmnclust2==2)[0]
sliner_2 = np.where(kmnclust2==3)[0]
allliners_2 = np.append(sliner_2, hliner_2)

combo_classes =[]
for i in range(len(kmnclust)):
    if kmnclust[i] !=0:
        combo_classes.append(kmnclust[i])
    else:
        combo_classes.append(kmnclust_woo1[i])
combo_classes=np.array(combo_classes) 

combo_classes_full = combo_classes[full_agn_filt]

combo_val = np.where(combo_classes_full !=0)[0]
combo_sy2 = np.where(combo_classes_full ==1)[0]
combo_hliner = np.where(combo_classes_full ==2)[0]
combo_sliner = np.where(combo_classes_full ==3)[0]
combo_allliners = np.where((combo_classes_full==2)|(combo_classes_full==3))

kmnclust1[kmnclust1==0] = 9
combo_classes_full[combo_classes_full==0] = 9

sfrm_gsw2.fullagn_df['sevenline_classes'] = kmnclust1
sfrm_gsw2.fullagn_df['sixline_classes'] = combo_classes_full

fulltab = pd.merge(EL_m2.EL_gsw_df, sfrm_gsw2.fullagn_df, on ='ids', how='left')

fulltab['sevenline_classes'] = fulltab['sevenline_classes'].fillna(0).copy()

fulltab['sixline_classes'] = fulltab['sixline_classes'].fillna(0).copy()
fulltab['genclass'] = np.ones_like(fulltab['sixline_classes'])*9

fulltab.loc[EL_m2.neither_filt,'genclass']=0

fulltab.loc[EL_m2.bpt_sn_filt[EL_m2.bptplssf],'genclass']=1
fulltab.loc[EL_m2.halp_nii_filt[EL_m2.bptplsnii_sf],'genclass']=2

fulltab.loc[EL_m2.halp_nii_filt[EL_m2.bptplsnii_agn],'genclass']=3

fulltab.loc[np.where(fulltab['sixline_classes']!=0)[0],'genclass']=5
fulltab.loc[np.where(fulltab['genclass']==9)[0], 'genclass']=4

low_z = np.where(sfrm_gsw2.fullagn_df.z<=0.07)[0]
low_z_val1 =np.where(kmnclust1[low_z] !=0 )[0]
low_z_sy2_1=np.where(kmnclust1[low_z] ==1 )[0]
low_z_hliner_1 =np.where(kmnclust1[low_z] ==2 )[0]
low_z_sliner_1 =np.where(kmnclust1[low_z] ==3 )[0]
low_z_allliners = np.append(low_z_sliner_1, low_z_hliner_1)

allnon_sy2 = pd.concat([sfrm_gsw2.low_z_df.iloc[low_z_allliners].copy(), EL_m2.allnonagn_df.copy()])
allnon_sy2_highsn = pd.concat([sfrm_gsw2.low_z_df.iloc[low_z_allliners].copy(), EL_m2.alllines_bpt_sf_df.copy()])

allnon_sy2_lowz = pd.concat([sfrm_gsw2.low_z_df.iloc[low_z_allliners].copy(), EL_m2.allnonagn_df.iloc[np.where(EL_m2.allnonagn_df.z<=0.07)].copy()])
allnon_sy2_highsn_lowz = pd.concat([sfrm_gsw2.low_z_df.iloc[low_z_allliners].copy(), EL_m2.alllines_bpt_sf_df.iloc[np.where(EL_m2.alllines_bpt_sf_df.z<=0.07)].copy()])

allnonsy2_lowz = pd.concat([sfrm_gsw2.low_z_df.iloc[low_z_allliners].copy(), EL_m2.allnonagn_df.iloc[np.where(EL_m2.allnonagn_df.z<=0.07)].copy()])
allnonagn_lowz = EL_m2.allnonagn_df.iloc[np.where(EL_m2.allnonagn_df.z<=0.07)].copy()

sf_lowz = EL_m2.bptplus_sf_df.iloc[np.where(EL_m2.bptplus_sf_df.z<=0.07)].copy()

sf_highsn_lowz = EL_m2.alllines_bptplus_sf_df.iloc[np.where(EL_m2.alllines_bptplus_sf_df.z<=0.07)].copy()
sf_highsn_lowz = pd.concat([sfrm_gsw2.low_z_df.iloc[low_z_allliners].copy(), EL_m2.alllines_bptplus_sf_df.iloc[np.where(EL_m2.alllines_bptplus_sf_df.z<=0.07)].copy()])

sfplus_lowz = EL_m2.bptplusnii_sf_df.iloc[np.where(EL_m2.bptplusnii_sf_df.z<=0.07)].copy()

'''

val1_n2o2_match = np.where((sfrm_gsw2.fullmatch_df.niiflux_sn.iloc[val1]>2)&(sfrm_gsw2.fullmatch_df.oiiflux_sn.iloc[val1]>2))[0]
sy2_n2o2_match = np.where((sfrm_gsw2.fullmatch_df.niiflux_sn.iloc[sy2_1]>2)&(sfrm_gsw2.fullmatch_df.oiiflux_sn.iloc[sy2_1]>2))[0]
sliner_n2o2_match = np.where((sfrm_gsw2.fullmatch_df.niiflux_sn.iloc[sliner_1]>2)&(sfrm_gsw2.fullmatch_df.oiiflux_sn.iloc[sliner_1]>2))[0]
hliner_n2o2_match = np.where((sfrm_gsw2.fullmatch_df.niiflux_sn.iloc[hliner_1]>2)&(sfrm_gsw2.fullmatch_df.oiiflux_sn.iloc[hliner_1]>2))[0]


val1_oh = np.where(sfrm_gsw2.fullmatch_df.ohabund.iloc[val1]>0)[0]
sy2_oh = np.where(sfrm_gsw2.fullmatch_df.ohabund.iloc[sy2_1]>0)[0]
sliner_oh = np.where(sfrm_gsw2.fullmatch_df.ohabund.iloc[sliner_1]>0)[0]
hliner_oh = np.where(sfrm_gsw2.fullmatch_df.ohabund.iloc[hliner_1]>0)[0]



kmnclust_hd = np.loadtxt('catalogs/kmeans_d4000_hdelta.dat', unpack=True)

full_agn_filt_hd= np.where(d4000mhd_gsw2.allagn_df.full_agn)[0]

kmnclust1_hd =  kmnclust_hd[full_agn_filt_hd]
val1_hd = np.where(kmnclust1_hd != 0 )[0]
sy2_1_hd = np.where(kmnclust1_hd==1)[0]
hliner_1_hd = np.where(kmnclust1_hd==2)[0]
sliner_1_hd = np.where(kmnclust1_hd==3)[0]


full_agn = np.where(sfrm_gsw2.allagn_df.full_agn==1)[0]
val_full_agn = np.where(sfrm_gsw2.fullagn_df.iloc[val1]==1)
sy2_full_agn = np.where(sfrm_gsw2.allagn_df.iloc[sy2_1]==1)
sf_full_agn = np.where(sfrm_gsw2.allagn_df.iloc[sliner_1]==1)
liner2_full_agn = np.where(sfrm_gsw2.allagn_df.iloc[hliner_1]==1)


sfrm_gsw2_sec = SFRMatch(EL_m2,EL_m2.bpt_EL_gsw_df,
                     EL_m2.plus_EL_gsw_df, EL_m2.neither_EL_gsw_df)

sfrm_gsw2_sec.get_highsn_match_only(EL_m2.bptplsagn, EL_m2.bptplssf, 
                                EL_m2.bptplsnii_sf, EL_m2.bptplsnii_agn, 
                                load=True,  sncut=2, with_av=True, fname='_second_')
sfrm_gsw2_sec.subtract_elflux(sncut=2,  halphbeta_sncut=2)
'''

val2_sii_doub= np.where((kmnclust1 != 0) &(sfrm_gsw2.fullagn_df.sii6717flux_sub_sn>2) &(sfrm_gsw2.fullagn_df.sii6731flux_sub_sn>2))[0]
sy2_1_sii_doub = np.where((kmnclust1==1) &(sfrm_gsw2.fullagn_df.sii6717flux_sub_sn>2) &(sfrm_gsw2.fullagn_df.sii6731flux_sub_sn>2))[0]
sliner_1_sii_doub = np.where((kmnclust1==3) &(sfrm_gsw2.fullagn_df.sii6717flux_sub_sn>2) &(sfrm_gsw2.fullagn_df.sii6731flux_sub_sn>2))[0]
hliner_1_sii_doub = np.where((kmnclust1==2) &(sfrm_gsw2.fullagn_df.sii6717flux_sub_sn>2) &(sfrm_gsw2.fullagn_df.sii6731flux_sub_sn>2))[0]


val2_oi_3 =  np.where((kmnclust1 != 0) &(sfrm_gsw2.fullagn_df.oiflux_sub_sn>3) )[0]
sy2_1_oi_3 = np.where((kmnclust1==1) &(sfrm_gsw2.fullagn_df.oiflux_sub_sn>3) )[0]
sliner_1_oi_3 = np.where((kmnclust1==3) &(sfrm_gsw2.fullagn_df.oiflux_sub_sn>3) )[0]
hliner_1_oi_3 = np.where((kmnclust1==2) &(sfrm_gsw2.fullagn_df.oiflux_sub_sn>3))[0]




sncut=2
combo_sncut=2, 
delta_ssfr_cut=-0.7
minmass=10.2
maxmass=10.4, 
d4000_cut=1.6


upperoiiilum=40.14
loweroiiilum=40.06

upperU_cut=-0.2
lowerU_cut=-0.3,


filts = {'delta_ssfr': {'cut':[delta_ssfr_cut]},
             #'mass':{'cut': [minmass,maxmass]},
             #'d4000':{'cut':[d4000_cut]},
             #'oiiilum': {'cut':[loweroiiilum,upperoiiilum]},
             #'oiiilum_sub_dered': {'cut':[loweroiiilum,upperoiiilum]},
             #'U_sub': {'cut':[lowerU_cut,upperU_cut]},
             #'U': {'cut':[lowerU_cut,upperU_cut]},
             #'qc_sub':{'cut':[lowerqc_cut, upperqc_cut]},
             #'q_sub':{'cut':[lowerq_cut, upperq_cut]}
             'sy2_liner_bool':{'cut':[0.5]}
              }
gen_filts = {}
match_filts = {'delta_ssfr': {'cut':[delta_ssfr_cut]},
             #'mass':{'cut': [minmass,maxmass]},
             #'d4000':{'cut':[d4000_cut]},
             #'oiiilum': {'cut':[loweroiiilum,upperoiiilum]},
             #'oiiilum_sub_dered': {'cut':[loweroiiilum,upperoiiilum]},
             #'U_sub': {'cut':[lowerU_cut,upperU_cut]},
             #'U': {'cut':[lowerU_cut,upperU_cut]},
             'sy2_liner_bool':{'cut':[0.5]}
             
             #'qc_sub':{'cut':[lowerq_cut, upperq_cut]},
             #'q_sub':{'cut':[lowerq_cut, upperq_cut]}

              }
line_filts = [['oiflux',sncut],
              ['oiiflux',sncut],
              ['siiflux',sncut],
              ['oiflux_sub',sncut],
              ['oiiflux_sub',sncut],
              ['siiflux_sub',sncut]                      
              ]
line_filts_comb = {'sii_oii_sub':[['siiflux_sub', 'oiiflux_sub'],combo_sncut],
                   'sii_oi_sub':[['siiflux_sub','oiflux_sub'],3],
                   'oi_oii_sub':[['oiflux_sub', 'oiiflux_sub'],combo_sncut],
                   'sii_oii_oi_sub':[['siiflux_sub', 'oiiflux_sub', 'oiflux_sub'], combo_sncut],
                   'sii_oii_ha_hb_sub':[['siiflux_sub','oiiflux_sub','halpflux_sub','hbetaflux_sub'],combo_sncut],
                   #'o32_sub':[['oiii4959flux_sub','oiiflux_sub'], combo_sncut],
                   'siidoub_sub':[['sii6717flux_sub', 'sii6731flux_sub'],combo_sncut ],
                   
                   #'sii_oii':[['siiflux', 'oiiflux'],combo_sncut],
                   #'sii_oi':[['siiflux','oiflux'],combo_sncut],
                   #'oi_oii':[['oiflux', 'oiiflux'],combo_sncut],
                   #'sii_oii_ha_hb':[['siiflux','oiiflux','halpflux','hbetaflux'],combo_sncut],
                   #'siidoub':[['sii6717flux', 'sii6731flux'],combo_sncut ]
                   
                   }
#sfrm_gsw2.get_filt_dfs(filts, gen_filts, match_filts, line_filts, line_filts_comb, loweroiiilum=40.0, upperoiiilum=40.2)
#get_filt_dfs(self, sncut=2, combo_sncut=2, 
#                     delta_ssfr_cut=-0.7, 
#                     minmass=10.2, maxmass=10.4, 
#                     d4000_cut=1.6, 
#                     loweroiiilum=40.2, upperoiiilum=40.3, 
#                     upperU_cut=-0.2, lowerU_cut=-0.3,):



#sfrm_gsw2.bin_by_bpt( val1, sy2_1, sliner_1, hliner_1,binsize=0.1)


cols_to_use = sfrm_gsw2.fullagn_df.columns.difference(xmm4eldiagmed_xrfilt.bpt_EL_gsw_df.columns)
cols_to_use  = cols_to_use.tolist()
cols_to_use.append('ids')




merged_xr = pd.merge(xmm4eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[ cols_to_use], on='ids')
merged_xr_all = pd.merge(xmm4eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[ cols_to_use], on='ids')

merged_xr_val = pd.merge(xmm4eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[val1], on='ids')
merged_xr_val_all = pd.merge(xmm4eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[val1], on='ids')

merged_xr_val_combo = pd.merge(xmm4eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_val], on='ids')
merged_xr_val_all_combo = pd.merge(xmm4eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_val], on='ids')
merged_xr_val_all_combo_allxr = pd.merge(EL_4xmm_all.EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_val], on='ids')

merged_xr_sy2 = pd.merge(xmm4eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[sy2_1], on='ids')
merged_xr_sf = pd.merge(xmm4eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[sliner_1], on='ids')
merged_xr_liner2 = pd.merge(xmm4eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[hliner_1], on='ids')

merged_xr_sy2_all = pd.merge(xmm4eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[sy2_1], on='ids')
merged_xr_sf_all = pd.merge(xmm4eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[sliner_1], on='ids')
merged_xr_liner2_all = pd.merge(xmm4eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[hliner_1], on='ids')


merged_xr_sy2_combo = pd.merge(xmm4eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_sy2], on='ids')
merged_xr_sf_combo = pd.merge(xmm4eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_sliner], on='ids')
merged_xr_liner2_combo = pd.merge(xmm4eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_hliner], on='ids')

merged_xr_sy2_all_combo = pd.merge(xmm4eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_sy2], on='ids')
merged_xr_sf_all_combo = pd.merge(xmm4eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_sliner], on='ids')
merged_xr_liner2_all_combo = pd.merge(xmm4eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_hliner], on='ids')


merged_xr_sy2_all_combo_allxr = pd.merge(EL_4xmm_all.EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_sy2], on='ids')
merged_xr_sf_all_combo_allxr = pd.merge(EL_4xmm_all.EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_sliner], on='ids')
merged_xr_liner2_all_combo_allxr = pd.merge(EL_4xmm_all.EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_hliner], on='ids')

merged_xr_sy2_all.to_csv(catfold+'merged_xr_sy2_all.csv')
merged_xr_liner2_all.to_csv(catfold+'merged_xr_hliner_all.csv')
merged_xr_sf_all.to_csv(catfold+'merged_xr_sliner_all.csv')
merged_xr_val_all.to_csv(catfold+'merged_xr_val_all.csv')
merged_xr_all.to_csv(catfold+'merged_xr_all.csv')

merged_xr_sy2_all_combo.to_csv(catfold+'merged_xr_sy2_all_combo.csv')
merged_xr_liner2_all_combo.to_csv(catfold+'merged_xr_hliner_all_combo.csv')
merged_xr_sf_all_combo.to_csv(catfold+'merged_xr_sliner_all_combo.csv')
merged_xr_val_all_combo.to_csv(catfold+'merged_xr_val_all_combo.csv')


merged_xr_sy2_all_combo_allxr.to_csv(catfold+'merged_xr_sy2_all_combo_allxr.csv')
merged_xr_liner2_all_combo_allxr.to_csv(catfold+'merged_xr_hliner_all_combo_allxr.csv')
merged_xr_sf_all_combo_allxr.to_csv(catfold+'merged_xr_sliner_all_combo_allxr.csv')
merged_xr_val_all_combo_allxr.to_csv(catfold+'merged_xr_val_all_combo_allxr.csv')

EL_4xmm_all.EL_gsw_df.to_csv(catfold+'x4_xray_all_sample.csv')
xmm4eldiagmed_xrfilt_xragn.EL_gsw_df.to_csv(catfold+'x4_xragn_all_sample.csv')

xmm4eldiagmed_xrfilt_xragn.bptplus_sf_df.to_csv(catfold+'xragn_bptplussf.csv')
xmm4eldiagmed_xrfilt_xragn.bptplusnii_agn_df.to_csv(catfold+'xragn_bptplusnii.csv')
xmm4eldiagmed_xrfilt_xragn.bptplusnii_sf_df.to_csv(catfold+'xragn_bptplusniisf.csv')

EL_4xmm_all.bpt_sf_df.to_csv(catfold+'xr_bptsf.csv')
EL_4xmm_all.bptplus_sf_df.to_csv(catfold+'xr_bptplussf.csv')

EL_4xmm_all.bptplusnii_agn_df.to_csv(catfold+'xr_bptniiagn.csv')

xmm4eldiagmed_xrfilt_unclass_p1.EL_gsw_df.to_csv(catfold+'xragn_sample_unclass_p1_cuts.csv')
xmm4eldiagmed_xrfilt_unclass_p2.EL_gsw_df.to_csv(catfold+'xragn_sample_unclass_p2_cuts.csv')

bptplus_xr_o3 = EL_4xmm_all.bptplusnii_agn_df.iloc[np.where((EL_4xmm_all.bptplusnii_agn_df.oiiiflux_sn>1)&
                                                            (EL_4xmm_all.bptplusnii_agn_df.hardflux_sn>2))].copy()
neither_xr_o3 = EL_4xmm_all.neither_EL_gsw_df.iloc[np.where((EL_4xmm_all.neither_EL_gsw_df.oiiiflux_sn>1)&
                                                            (EL_4xmm_all.neither_EL_gsw_df.hardflux_sn>2))].copy()

comm_covered_bptplsagn, comm_bptpls, comm_covered = np.intersect1d(EL_m2.bptplsagn, covered_gsw_x3, return_indices=True)



#%% 
def match_xrgal_to_sfrm_obj(xrids, sfrmids):
    matching_ind_bpt = []
    xrind_bpt = []
    matching_ind_plus = []
    xrind_plus = []
    matching_ind_neither = []
    xrind_neither = []
    for i, xrid in enumerate(xrids):
        match = np.int64(np.where(sfrmids == xrid)[0][0])
        if match <sfrm_gsw2.agns.size:
            matching_ind_bpt.append(match)
            xrind_bpt.append(i)
        elif match <sfrm_gsw2.agns.size+sfrm_gsw2.agns_plus.size:
            matching_ind_plus.append(match-sfrm_gsw2.agns.size)
            xrind_plus.append(i)
        else:
            matching_ind_neither.append(match-sfrm_gsw2.agns.size+sfrm_gsw2.agns_plus.size)
            xrind_neither.append(i)
    return np.array(matching_ind_bpt), np.array(xrind_bpt), np.array(matching_ind_plus), np.array(xrind_plus), np.array(matching_ind_neither), np.array(xrind_neither)

#xmm3all_xr_to_sfrm_bpt, xmm3all_xrind_bpt, xmm3all_xr_to_sfrm_plus,xmm3all_xrind_plus, xmm3all_xr_to_sfrm_neither, xmm3all_xrind_neither = match_xrgal_to_sfrm_obj(xmm3eldiagmed_xrfilt_all.ids[agn_3xmmmbptplus_all_xrfilt_bptplus], sfrm_gsw2.sdssids_sing)
#xmm3_xr_to_sfrm_bpt,xmm3_xrind_bpt, xmm3_xr_to_sfrm_plus,xmm3_xrind_plus, xmm3_xr_to_sfrm_neither, xmm3_xrind_neither = match_xrgal_to_sfrm_obj(xmm3eldiagmed_xrfiltbptplus.ids[agn_3xmmm_xrfilt_bptplus], sfrm_gsw2.sdssids_sing)

#sfrm_gsw2.bin_by_bpt(binsize=0.1)


'''
for creating target files
'''
def write_to_fil_obs(filnam):
    f = open(filnam,'w+')
    #f.write("#Name, OBJID, RA, DEC, BPT CAT\n")
    f.write("#Name, OBJID, RA, DEC, Exp time, SN Min, SN Min Line, SN Max, SN Max Line, z\n")

    sn =  np.vstack([xmm3eldiagmed_xrfilt.halp_sn,xmm3eldiagmed_xrfilt.nii_sn,                   xmm3eldiagmed_xrfilt.oiii_sn,xmm3eldiagmed_xrfilt.hbeta_sn])
    snmin = np.min(sn, axis= 0)
    snmax = np.max(sn, axis= 0)
    sncodemin = matchcode(sn, snmin)
    sncodemax = matchcode(sn, snmax)

    filtxmm = mpa_spec_m2_3xmm.make_prac[EL_3xmm.bpt_sn_filt][fullxray_xmm.valid][fullxray_xmm.likelyagn_xr][nonagn_3xmmm_xrfilt]

    for i in range(len(filtxmm)):
        f.write(str(m2Cat_GSW_3xmm.matchra[filtxmm][i])+','+str(m2Cat_GSW_3xmm.matchdec[filtxmm][i])+','+str(m2Cat_GSW_3xmm.matchmjd[filtxmm][i])+ ','+str(m2Cat_GSW_3xmm.matchfiber[filtxmm][i])+','+str(m2Cat_GSW_3xmm.matchplate[filtxmm][i])+'\n')
        #f.write('3XMMHII'+'-'+str(i)+','+str(m2Cat_GSW_3xmm.ids[filtxmm][i])+','+                    str(m2Cat_GSW_3xmm.matchra[filtxmm][i])+','+str(m2Cat_GSW_3xmm.matchdec[filtxmm][i])+
        #                      ','+str(m2Cat_GSW_3xmm.exptimes[filtxmm][i])+',' +               str(snmin[nonagn_3xmmm_xrfilt][i])+','+sncodemin[nonagn_3xmmm_xrfilt][i]+','+str(snmax[nonagn_3xmmm_xrfilt][i])+','+ sncodemax[nonagn_3xmmm_xrfilt][i]+','+str(xmm3eldiagmed_xrfilt.z[nonagn_3xmmm_xrfilt][i])+'\n')

    f.close()
#write_to_fil_obs('sdssquerytable.txt')

codes = {0:'halp', 1:'nii', 2:'oiii', 3:'hbeta'}
def matchcode(sn,sndes):
    code = []
    for i in range(len(sndes)):
        whermin = np.where(sn[:,i] == sndes[i])[0]
        code.append(codes[whermin[0]])
    return np.array(code)

def write_to_fil_obs_supp(filnam):
    f = open(filnam,'w+')
    f.write("#Name, OBJID, RA, DEC, Exp time, SN Min, SN Min Line, SN Max, SN Max Line, z, ssfr \n")
    filtxmm = mpa_spec_m2_3xmm_all.make_prac[EL_3xmm_all.bpt_sn_filt][fullxray_xmm_all.valid][fullxray_xmm_all.likelyagn_xr][nonagn_3xmmm_all_xrfilt]
    sn =  np.vstack([xmm3eldiagmed_xrfilt_all.halp_sn,xmm3eldiagmed_xrfilt_all.nii_sn,xmm3eldiagmed_xrfilt_all.oiii_sn,xmm3eldiagmed_xrfilt_all.hbeta_sn])
    snmin = np.min(sn, axis= 0)
    snmax = np.max(sn, axis= 0)
    sncodemin = matchcode(sn, snmin)
    sncodemax = matchcode(sn, snmax)

    for i in range(len(filtxmm)):
        print(m2Cat_GSW_3xmm_all.matchra[filtxmm][i])
        f.write('3XMMHII'+'-'+str(i)+','+str(m2Cat_GSW_3xmm_all.ids[filtxmm][i])+','+str(m2Cat_GSW_3xmm_all.matchra[filtxmm][i])+','+str(m2Cat_GSW_3xmm_all.matchdec[filtxmm][i])+
                ','+str(m2Cat_GSW_3xmm_all.exptimes[filtxmm][i])+','+str(snmin[nonagn_3xmmm_all_xrfilt][i])+','+
                sncodemin[nonagn_3xmmm_all_xrfilt][i]+','+str(snmax[nonagn_3xmmm_all_xrfilt][i])+','+
                sncodemax[nonagn_3xmmm_all_xrfilt][i]+','+str(xmm3eldiagmed_xrfilt_all.z[nonagn_3xmmm_all_xrfilt][i])+
                ','+str(xmm3eldiagmed_xrfilt_all.ssfr[nonagn_3xmmm_all_xrfilt][i])+'\n')

    f.close()
#write_to_fil_obs_supp('observing_sample_supp_SP19.txt')
