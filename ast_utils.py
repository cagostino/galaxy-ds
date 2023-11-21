import astropy.cosmology as apc
cosmo = apc.Planck15
samircosmo = apc.FlatLambdaCDM(H0=70, Om0=0.3)
import numpy as np
from astropy.coordinates import SkyCoord 
from astropy import units as  u

def getlumfromflux(flux, z):
    '''
    getting the luminosity for a given flux, based on L= 4pi*d^2 *flux
    -uses luminosity distance here so additional factors not necessary
    -in cgs so need to convert lum dist from Mpc to cm 
    '''
    distances=np.array(cosmo.luminosity_distance(z))* (3.086e+24) #Mpc to cm
    lum = 4*np.pi*(distances**2)*flux
    return lum
def getfluxfromlum(lum,z):
    distances=np.array(cosmo.luminosity_distance(z))* (3.086e+24) #Mpc to cm
    flux = lum/(4*np.pi*(distances**2))
    return flux
    

def comp_skydist(ra1,dec1,ra2,dec2):
    '''
    computing distance between two objects on the sky
    '''
    c1 = SkyCoord(ra1,dec1, frame='icrs')
    c2 = SkyCoord(ra2,dec2, frame='icrs')
    return c1.separation(c2)

def conv_ra(ra):
    '''
    convert ra for plotting spherical aitoff projection.
    '''
    copra = np.copy(ra)
    for i in range(len(ra)):
        if copra[i] > 270:
            copra[i] =- 360 + copra[i]

    return (copra)*(-1)

import numpy as np
def ra_hr_arr(ra):
    racop = np.empty_like(ra,dtype='U32')
    for i in range(len(ra)):
        rahr = str(ra[i] // 15)
        ram = str((ra[i]%15 *60)//15)
        ras = str((np.round((ra[i]*60)%60)))
        rastr =rahr+' '+ram+' '+ras+' '
        racop[i]= rastr
    return racop
def ra_hr(ra):
    rahr = str(ra // 15)
    ram = str((ra%15 *60)//15)
    ras = str((np.round((ra*60)%60)))
    rastr =rahr+' '+ram+' '+ras+' '
    return rastr



import numpy as np
from ast_utils import *
from demarcations import *
import extinction
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from chroptiks.plotting_utils import plot2dhist
import extinction as ext

ext.ccm89(np.array([3727.0]), 1.0, 3.1)/ext.ccm89(np.array([6563.0]), 1.0, 3.1)
xrayranallidict = {'soft':1/(1.39e-40),'hard':1/(1.26e-40),'full':1/(0.66e-40)}


def get_extinction(ha, hb, dec_rat=2.86, dustlaw='cardelli', zeroed=False, ha_hb_err =[]):
    ha=np.copy(np.array(ha))ipyt
    hb=np.copy(np.array(hb))
    
    if dustlaw=='cardelli':
        av = 7.23*np.log10((ha/hb) / dec_rat) # A(V) mag
    if zeroed:
        zeroav = np.where(av<0)[0]
        av[zeroav] = 0
        maxed = np.where(av>3)[0]
        av[maxed]=3
    if len(ha_hb_err) != 0:
        sigma = 7.23*ha_hb_err/(np.log(10)*ha/hb )
        return np.array(av), np.array(sigma)
        
    return np.array(av)


def get_av_bd_a21( data, av_balm_col, hb_sn, av_gal_col, sn_cut=10):

    def fix_av_balm(row, reg):
        i = row.name
        if row['hb_sn'] < 10:
            x_samp = np.array([row[av_gal_col]]).reshape(1,-1)
            av_fix = np.float64(reg.predict(x_samp))
            if av_fix < 0:
                av_fix = 0
        elif row[av_balm_col] < 0:
            av_fix = 0
        else:
            av_fix = row[av_balm_col]
        return av_fix

    condition = (hb_sn > sn_cut) & (np.isfinite(hb_sn)) & (~hb_sn.isna()) & (~data[av_gal_col].isna() &(np.isfinite(data[av_gal_col]))) & (~data[av_balm_col].isna() & (np.isfinite(data[av_balm_col])))    
    y_reg = np.array(data.loc[condition,av_balm_col])
    X_reg_av = np.vstack([data.loc[condition,av_gal_col]]).transpose()
    reg_av = LinearRegression().fit(X_reg_av,y_reg)
            
    df = pd.DataFrame({'hb_sn': hb_sn, av_balm_col: data[av_balm_col], av_gal_col: data[av_gal_col]})
    output = df.apply(lambda row: fix_av_balm(row, reg_av) if np.isfinite(row[av_gal_col]) else 0, axis=1)

    return output           

def dustcorrect(flux, av, wl):
    fact = ext.ccm89(np.array([wl]), 1.0, 3.1)
    return np.array(flux*10**(0.4*av*fact))

def redden(flux, av, wl):
    '''
    Need to adjust so that the factor is dependent on the different lines
    '''
    fact = ext.ccm89(np.array([wl]), 1.0, 3.1)
    return np.array(flux/(10**(0.4*av*fact)))

def halptofibsfr_corr( halplum):
    logsfrfib = np.log10(7.9e-42) + np.log10(halplum) - 0.24 #+0.4*A(H_alpha)
    return np.array(logsfrfib)
def halptofibsfr_uncorr( halplum,av):
    logsfrfib = np.log10(7.9e-42) + np.log10(halplum) - 0.24 + 0.4*av
    return np.array(logsfrfib)
        
def get_deltassfr( mass, ssfr):
    m_ssfr = -0.4597
    b_ssfr = -5.2976
    delta_ssfr = ssfr-(mass*m_ssfr+b_ssfr)
    return np.array(delta_ssfr)

def nii_oii_to_oh( nii, oii):
    '''
    Uses castro et al 2017 method for relating NII/OII to metallicity
    for Sy2.
    '''
    z = 1.08*(np.log10(nii/oii)**2) + 1.78*(np.log10(nii/oii)) + 1.24
    logoh = 12+np.log10(z*10**(-3.31))
    return np.array(logoh)

def nii_oii_to_oh_ke02(nii,oii):
    logoh = np.log10(1.54020+1.26602*nii/oii+0.167977*(nii/oii)**2)+8.93
    return logoh

def nii_logoh_o32_to_q(logoh, o32):
    return (32.81-1.153*o32**2+(logoh)*(-3.396-0.025*o32+0.1444*o32**2))* ((4.603-0.3119*o32-0.163*o32**2)+(logoh)*(-0.48+0.0271*o32+0.02037*o32**2))**(-1)
def oiii_oii_to_U(oiii_oii):
    return 0.57*(oiii_oii**2)+1.38*(oiii_oii)-3.14
def sii_doub_to_ne(sii_doub, te=10000):
    ne = 0.0543*np.tan(-3.0553*sii_doub+2.8506)+6.98-10.6905*sii_doub+9.9186*sii_doub**2-3.5442*sii_doub**3
    ne_te_scaled = ne*(10000/te)**(-1./2)
    return ne_te_scaled

def correct_av(reg, x_test, av_balm, hb_sn, empirdust=False, sub = False):
    #x_test = av_gsw.transpose()
    av_balm_fixed = []
    for i in range(len(hb_sn)):

        if hb_sn[i] <10:
            x_samp = x_test[i].reshape(1,-1)
            if empirdust:
                av_fix = np.float64(reg.predict(x_samp))
            else:
                if sub:
                    av_fix = np.float64(1.43431072*x_samp+0.34834144657321453)
                else:
                    av_fix = np.float64(1.78281143*x_samp+0.2085823945997035)
            if av_fix<0:
                av_fix=0
            if av_fix>3:
                av_fix=3
        elif av_balm[i] <0:
            av_fix = 0
        elif av_balm[i]>3:
            av_fix = 3
        else:
            av_fix = av_balm[i]
        av_balm_fixed.append(av_fix)
    return np.array(av_balm_fixed)



sflocus = (-0.5, -0.5)

#agn_branch_vec = [1.1643, 0.1543]
agn_branch_vec = [-1.1643, -0.1543]

#perp_agn_branch_vec = [-0.86,-1]
#line def would by Ax+By+C=0
perp_agn_branch_vec =[0.86,1.5]

def get_perpdist( x, y):
    top = np.abs(perp_agn_branch_vec[0]*x+y+perp_agn_branch_vec[1])
    bot= np.sqrt(perp_agn_branch_vec[0]**2+1)
    return np.array(top/bot)
def get_pardist( x, y):
    top = agn_branch_vec[0]*x+y+agn_branch_vec[1]
    bot = np.sqrt(agn_branch_vec[0]**2+1)
    return np.array(top/bot)

def get_thom_dist(x,y):
    slope = (-0.408-0.979)/(-0.466-0.003)
    #xmix = np.arange(-0.566, 0.003, 0.001)
    #ymix = slope*(xmix)+0.97013
    ke_x = -0.22321
    ke_y= 0.310
    top = -slope*(x-ke_x)-ke_y+y
    bot= np.sqrt(slope**2+1)
    perp = np.abs(top)/bot
    d_obj = np.sqrt( (x-ke_x)**2 + (y-ke_y)**2)
    theta = np.arcsin(perp/ d_obj)
    
    slope_inv = -0.327
    top_inv = -slope_inv*(x-ke_x)-ke_y+y    
    
    print(theta)
    proj_d = d_obj * np.cos(theta)
    #sign = (np.sign(x-ke_int_x)*(x-ke_int_x)**2 +np.sign(y-ke_int_y)*(y-ke_int_y)**2)
    return np.sign(top_inv)*proj_d #sign*np.array(top/bot)
def get_classifiability(n2_sn, ha_sn, o3_sn, hb_sn, sncut=2):
    bpt_sn_filt_bool = (ha_sn>sncut) & (hb_sn > sncut) & (o3_sn > sncut) & (n2_sn> sncut)
    halp_nii_filt_bool = ( (ha_sn > sncut) & (n2_sn > sncut) &( (o3_sn<=sncut) | (hb_sn <=sncut) ) )
    neither_filt_bool = np.logical_not(((bpt_sn_filt_bool ) | (halp_nii_filt_bool)) )
    options = []
    for i in range(len(bpt_sn_filt_bool)):
        if bpt_sn_filt_bool.iloc[i]:
            options.append('bpt')
        elif halp_nii_filt_bool.iloc[i]:
            options.append('nii')
        else:
            options.append('uncl')
    return np.array(options)
    
    

import numpy as np
import pandas as pd

def get_bpt1_groups_ke01(x, y):
    # Convert to numpy arrays
    xvals = np.asarray(x)
    yvals = np.asarray(y)

    # Vectorized condition checks
    condition_hii = (xvals < 0) & (yvals < np.log10(y1_kauffmann(xvals)))
    condition_comp = (xvals < 0.43) & (yvals >= np.log10(y1_kauffmann(xvals))) & (yvals < np.log10(y1_kewley(xvals)))
    condition_sy2 = (yvals > np.log10(y1_schawinski(xvals)))
    condition_liner = ~condition_hii & ~condition_comp & ~condition_sy2

    # Assign groups
    groups = np.full_like(xvals, 'LINER', dtype=object)
    groups[condition_hii] = 'HII'
    groups[condition_comp] = 'COMP'
    groups[condition_sy2] = 'Sy2'

    # Index arrays for each group
    sy2_indices = np.where(groups == 'Sy2')[0]
    liner_indices = np.where(groups == 'LINER')[0]
    comp_indices = np.where(groups == 'COMP')[0]
    nonagn_indices = np.where(groups == 'HII')[0]

    return groups, nonagn_indices, comp_indices, sy2_indices, liner_indices


def get_bpt1_groups(x, y):
    # Convert to numpy arrays
    xvals = np.asarray(x)
    yvals = np.asarray(y)

    # Vectorized condition checks
    condition_hii = (xvals < 0) & (yvals < np.log10(y1_kauffmann(xvals)))
    condition_agn = ~np.isnan(xvals) & ~np.isnan(yvals) & ~condition_hii

    # Assign groups
    groups = np.full_like(xvals, 'NO', dtype=object)
    groups[condition_hii] = 'HII'
    groups[condition_agn] = 'AGN'

    # Index arrays for each group
    agn_indices = np.where(groups == 'AGN')[0]
    nonagn_indices = np.where(groups == 'HII')[0]

    return groups, nonagn_indices, agn_indices

def get_bpt2_groups(x, y):
    # Convert to numpy arrays
    xvals = np.asarray(x)
    yvals = np.asarray(y)

    # Vectorized condition checks
    condition_hii = (xvals < 0.32) & (yvals < np.log10(y2_agn(xvals)))
    condition_seyfert = yvals > np.log10(y2_linersy2(xvals))
    condition_liner = ~condition_hii & ~condition_seyfert

    # Assign groups
    groups = np.full_like(xvals, 'LINER', dtype=object)
    groups[condition_hii] = 'HII'
    groups[condition_seyfert] = 'Seyfert'

    # Index arrays for each group
    hii_indices = np.where(groups == 'HII')[0]
    seyf_indices = np.where(groups == 'Seyfert')[0]
    liner_indices = np.where(groups == 'LINER')[0]

    return groups, hii_indices, seyf_indices, liner_indices

    return groups,hii, seyf, liner
def get_bpt3_groups(x, y):
    # Convert to numpy arrays
    xvals = np.asarray(x)
    yvals = np.asarray(y)

    # Vectorized condition checks
    condition_no = np.isnan(xvals)
    condition_hii = (xvals < -0.59) & (yvals < np.log10(y3_agn(xvals)))
    condition_seyfert = yvals > np.log10(y3_linersy2(xvals))
    condition_liner = ~condition_no & ~condition_hii & ~condition_seyfert

    # Assign groups
    groups = np.full_like(xvals, 'NO', dtype=object)
    groups[condition_hii] = 'HII'
    groups[condition_seyfert] = 'Seyfert'
    groups[condition_liner] = 'LINER'

    # Index arrays for each group
    hii_indices = np.where(groups == 'HII')[0]
    seyf_indices = np.where(groups == 'Seyfert')[0]
    liner_indices = np.where(groups == 'LINER')[0]

    return groups, hii_indices, seyf_indices, liner_indices

def get_whan_groups(x, y):
    # Convert to numpy arrays
    xvals = np.asarray(x)
    yvals = np.asarray(y)

    # Vectorized condition checks
    condition_no = np.isnan(xvals)
    condition_pg = ~condition_no & (yvals < 0.5)
    condition_rg = ~condition_no & (yvals >= 0.5) & (yvals < 3)
    condition_wagn = ~condition_no & (yvals >= 3) & (yvals < 6) & (xvals > -0.4)
    condition_sagn = ~condition_no & (yvals >= 6) & (xvals > -0.4)
    condition_sf = ~condition_no & (yvals >= 3) & (xvals < -0.4)

    # Assign groups
    groups = np.full_like(xvals, 'NO', dtype=object)
    groups[condition_pg] = 'PG'
    groups[condition_rg] = 'RG'
    groups[condition_wagn] = 'wAGN'
    groups[condition_sagn] = 'sAGN'
    groups[condition_sf] = 'SF'

    # Index arrays for each group
    sf_indices = np.where(groups == 'SF')[0]
    sagn_indices = np.where(groups == 'sAGN')[0]
    wagn_indices = np.where(groups == 'wAGN')[0]
    pg_indices = np.where(groups == 'PG')[0]
    rg_indices = np.where(groups == 'RG')[0]

    return groups, sf_indices, sagn_indices, wagn_indices, rg_indices, pg_indices



def get_ooo_groups(x, y):
    # Convert to numpy arrays
    xvals = np.asarray(x)
    yvals = np.asarray(y)

    # Vectorized condition checks
    condition_no = np.isnan(xvals) | np.isnan(yvals)
    condition_hii = ~condition_no & (yvals < np.log10(ooo_agn(xvals)))
    condition_seyfert = ~condition_no & (yvals > np.log10(ooo_linersy2(xvals)))
    condition_liner = ~condition_no & ~condition_hii & ~condition_seyfert

    # Assign groups
    groups = np.full_like(xvals, 'NO', dtype=object)
    groups[condition_hii] = 'HII'
    groups[condition_seyfert] = 'Seyfert'
    groups[condition_liner] = 'LINER'

    # Index arrays for each group
    hii_indices = np.where(groups == 'HII')[0]
    seyf_indices = np.where(groups == 'Seyfert')[0]
    liner_indices = np.where(groups == 'LINER')[0]

    return groups, hii_indices, seyf_indices, liner_indices
def get_bptplus_groups(x, y):
    # Convert to numpy arrays
    xvals = np.asarray(x)
    yvals = np.asarray(y)

    # Vectorized condition checks
    condition_no = np.isnan(xvals) | np.isnan(yvals)
    condition_agn = ~condition_no & (xvals >= -0.35)
    condition_hii = ~condition_no & (xvals < -0.35) & (yvals < np.log10(y1_kauffmann(xvals)))
    condition_mix = ~condition_no & ~condition_agn & ~condition_hii

    # Assign groups
    groups = np.full_like(xvals, 'NO', dtype=object)
    groups[condition_agn] = 'AGN'
    groups[condition_hii] = 'HII'
    groups[condition_mix] = 'MIX'

    # Index arrays for each group
    agn_indices = np.where(groups == 'AGN')[0]
    nonagn_indices = np.where(groups == 'HII')[0]

    return groups, nonagn_indices, agn_indices


def get_bptplus_niigroups(x):
    # Convert to numpy array if not already
    xvals = pd.Series(x).to_numpy()

    # Define conditions
    hii_condition = xvals < -0.4
    agn_condition = xvals > -0.35
    mix_condition = ~hii_condition & ~agn_condition

    # Apply conditions
    groups = np.full(xvals.shape, 'NO', dtype=object)  # Default to 'NO'
    groups[hii_condition] = 'HII'
    groups[agn_condition] = 'AGN'
    groups[mix_condition] = 'MIX'

    # Get indices
    agn_indices = np.where(groups == 'AGN')[0]
    nonagn_indices = np.where(groups == 'HII')[0]

    return groups, nonagn_indices, agn_indices



import numpy as np

from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import Distance
from scipy.optimize import curve_fit
from astropy.stats import bootstrap
from scipy.odr import ODR, Model, Data, RealData
'''

import cosmolopy as cpy

from cosmolopy import fidcosmo
from cosmolopy import magnitudes as mag
unit_flx = u.erg/u.s/u.cm/u.cm

app_oiii_sub, abs_oiii_sub = mag.magnitude_AB(sfrm_gsw2.allagn_df.z, sfrm_gsw2.allagn_df.oiiiflux_corr_sub, 5007*(sfrm_gsw2.allagn_df.z) +5007, **fidcosmo)
app_oiii_presub, abs_oiii_presub = mag.magnitude_AB(sfrm_gsw2.allagn_df.z, sfrm_gsw2.allagn_df.oiiiflux_corr, 5007*(sfrm_gsw2.allagn_df.z) +5007, **fidcosmo)

def schechter_mag(mags, alpha, phi_star, m_star):
    first = 0.4*np.log(10)*phi_star
    second = (10**(0.4*(m_star-mags)))**(alpha+1)
    third = np.exp(-10**(0.4*(m_star-mags)))
    return np.log10(first*second*third)
def pow_mag(mags, alpha, phi_star, m_star):
    first = 0.4*np.log(10)*phi_star
    second = (10**(0.4*(m_star-mags)))**(alpha+1)
    third = np.exp(-10**(0.4*(m_star-mags)))
    return np.log10(first*second*third)


def schechter_lum (lums, alpha, phi_star, lum_star):
    first = phi_star/lum_star
    second = (lums/lum_star)**(alpha)
    third = np.exp(-lums/lum_star)
    return first*second*third
def comp_lum_func(lum, distances, bins =[], zmin=0.01, zmax=0.3, nbins=-1, maglim=17.77):
    lum = np.copy(np.array(lum))
    nums = []
    lum_sorted = np.sort(lum)
    if nbins ==-1:
        nbins=15
    if len(bins) ==0:
        bins = np.round(np.linspace(lum_sorted[np.int(0.02*len(lum))],lum_sorted[np.int(0.98*len(lum))], nbins),1)
    bincenters = []
    densities = []
    completeness_vals = []
    compl_factors = []
    distmax_z = 10*10**(np.float64(Distance(z=zmax).distmod/5))/1e6 #Mpc
    distmin = 10*10**(np.float64(Distance(z=zmin).distmod)/5)/1e6 # in Mpc
    
    for i in range(len(bins)-1):            
        binmin = bins[i]
        binmax = bins[i+1]
        bincenter= (binmin+binmax)/2.
        bincenters.append(bincenter)
        distmodmax = maglim-bincenter
        distmax = 10*10**(distmodmax/5)/1e6 #in Mpc
        #print('distmax', distmax)
        if distmax > distmax_z:
            #if the maximum distance is beyond the redshift cut, correct for that
            distmax = distmax_z

        #print('distmin', distmin)
        binned_gals = np.where((lum>binmin) & (lum<binmax ))[0]
        count = binned_gals.size
        distmod_avg = np.array(distances)[binned_gals]
        dist_avg = np.mean(10*10**(distmod_avg/5)/1e6) #in Mpc
        #print('distavg', dist_avg)
        Vmax = 4*np.pi*(distmax-distmin)**3/3
        V = 4*np.pi*(dist_avg-distmin)**3/3
        nums.append(count)
        densities.append(count/(Vmax))
        
        v_over_vmax = V/Vmax
        compl_factor = 0.5/v_over_vmax
        completeness_vals.append(v_over_vmax)
        compl_factors.append(compl_factor)
    return bins,np.array(bincenters), np.array(nums), np.array(densities), np.array(completeness_vals), np.array(compl_factors)

def fit_schecht(mags, dens, alpha_0=-1.2, weights=[]):
    m_star_0 = np.mean(mags)
    phi_star_0 = 10**np.mean(dens)
    nans = np.where((np.isinf(dens)) | (np.isnan(dens)))[0]
    print(phi_star_0)
    if len(nans) !=0:
        mags = np.copy(np.delete(mags,nans))
        dens = np.copy(np.delete(dens, nans))
    #pdb.set_trace()
    if len(weights)!=0:
        if len(nans) !=0:
            weights = np.copy(np.delete(weights, nans))
        popt, pcov = curve_fit(schechter_mag, mags, dens,sigma=weights, p0 = (alpha_0, phi_star_0, m_star_0), maxfev=20000)

    else:
        popt, pcov = curve_fit(schechter_mag, mags, dens, p0 = (alpha_0, phi_star_0, m_star_0))
    alpha, phi_star, m_star = popt
    mags_fit = np.linspace(np.min(mags)-0.1,np.max(mags)+0.1, 100)
    schecht_fit = schechter_mag(mags_fit, alpha, phi_star, m_star)
    print('alpha: ',alpha)
    print('phi*: ',phi_star)
    print('m*: ',m_star)
    
    return alpha, phi_star, m_star, schecht_fit, mags_fit
lum_fn_sub = comp_lum_func(np.array(abs_oiii_sub), np.array(sfrm_gsw2.allagn_df.z), maglim=20.4)
lum_fn_presub = comp_lum_func(np.array(abs_oiii_presub), np.array(sfrm_gsw2.allagn_df.z), maglim=20.4)
fit_sch_sub = fit_schecht(lum_fn_sub[1], np.log10(lum_fn_sub[3]))
fit_sch_presub = fit_schecht(lum_fn_presub[1], np.log10(lum_fn_presub[3]))
def bootstrap_lum_fn( lum, dist, n_resamp = 10000, zmin=0.01, zmax = 0.3, bins = [], nbins=15, maglim=20.32):
    inds_lum = np.arange(len(lum))
    bootstr_samps = np.int64(bootstrap(inds_lum, n_resamp))
    
    nums_fns = np.zeros(shape=(n_resamp, nbins-1))
    dens_fns = np.zeros(shape=(n_resamp, nbins-1))
    completeness_fns =  np.zeros(shape=(n_resamp, nbins-1))
    compl_factors_fns = np.zeros(shape=(n_resamp, nbins-1))
    
    for i, samp in enumerate(bootstr_samps):
        lumfn = comp_lum_func(lum[samp], dist[samp], zmin = zmin, zmax = zmax, bins = bins, nbins= nbins, maglim=maglim)
   
        bincenters_abs_r = lumfn[1]
        nums_abs_r = lumfn[2]
        nums_fns[i, :] = nums_abs_r
        dens_abs_r = lumfn[3]
        dens_fns[i, :] = dens_abs_r
        
        completeness_abs_r = lumfn[4] 
        completeness_fns[i,:] = completeness_abs_r
        
        compl_factor = lumfn[5]
        compl_factors_fns[i, :] = compl_factor
    
    dens_fns_means = np.mean(dens_fns, axis=0)
    dens_fns_stds = np.std(dens_fns, axis=0)
    dens_fns_errs = dens_fns_stds/np.sqrt(n_resamp)
    completeness_fns_means = np.mean(completeness_fns, axis=0)
    compl_factors_means = np.mean(compl_factors_fns, axis=0)
    try:
        schecht = fit_schecht(bincenters_abs_r, np.log10(dens_fns_means), weights = compl_factors_means*0.434*dens_fns_stds/dens_fns_means )
        alpha, phi_star, m_star, schecht_fit, mags_fit = schecht
        
        return bincenters_abs_r, dens_fns_means, dens_fns_stds, dens_fns_errs, compl_factors_means, completeness_fns_means,  alpha, phi_star, m_star, schecht_fit, mags_fit
    except :
        return bincenters_abs_r, dens_fns_means, dens_fns_stds, dens_fns_errs, compl_factors_means, completeness_fns_means, -99, -99, -99, -99, -99
    #return schecht
bootstr_sub_fn = bootstrap_lum_fn(np.array(abs_oiii_sub), np.array(sfrm_gsw2.allagn_df.z), maglim=20.4)
bootstr_presub_fn = bootstrap_lum_fn(np.array(abs_oiii_presub), np.array(sfrm_gsw2.allagn_df.z), maglim=20.4)

def get_z_bins(self, nbins=-1): 
    zbins = np.array([0.01, 0.10666, 0.20333, 0.3])
    bincenters = []
    dens = []
    dens_std = []
    dens_errs = []        
    compl = []
    compl_factors = []
    alphas = []
    phi_stars = []
    m_stars = []
    schecht_fits = []
    mags_fits = []
    if nbins==-1:
        nbins=self.nbins_lf
    for i in range(len(zbins)-1):
        zmn = zbins[i]
        zmx = zbins[i+1]
        print('zmn, zmax: ', zmn, zmx)
        zfilt = np.where( (self.z_filt_kcorr < zmx) &(self.z_filt_kcorr>=zmn))[0]
        magsort = np.sort(self.abs_petromag_r_dered_kcorr[zfilt])
        print(nbins)
        bins_in = np.linspace(np.min(magsort[2:]), np.max(magsort[:-2]), nbins)
        print(bins_in)
        lumfn = self.bootstrap_lum_fn(self.abs_petromag_r_dered_kcorr[zfilt], self.distmod[zfilt], zmin= zmn, zmax = zmx, bins=bins_in, nbins=nbins)
        

        bincenters_abs_r = lumfn[0]
        bincenters.append(bincenters_abs_r)

        dens_abs_r = lumfn[1]
        dens.append(dens_abs_r)

        dens_std_r = lumfn[2]
        dens_std.append(dens_std_r)
        
        dens_errs_r = lumfn[3]
        dens_errs.append(dens_errs_r)
       
        compl_factor = lumfn[4]
        compl_factors.append(compl_factor)

        
        completeness_abs_r = lumfn[5] 
        compl.append(completeness_abs_r)
        
        alpha_r = lumfn[6]
        phi_star_r = lumfn[7]
        m_star_r = lumfn[8]
        schecht_fit_r = lumfn[9]
        mags_fit_r = lumfn[10]
            
        alphas.append(alpha_r)
        phi_stars.append(phi_star_r)
        m_stars.append(m_star_r)
        schecht_fits.append(schecht_fit_r)
        mags_fits.append(mags_fit_r)
        
    return  bincenters, dens, dens_std, compl,compl_factors, alphas, phi_stars, m_stars, schecht_fits, mags_fits

def plotlumfn(x, y, xlab, ylab, title, sym, save=False, mags=False, magfit=[],fit=[], alpha=0, phi_star=0,m_star=0, lab='', errsy=[]):
    
    plt.scatter(x,y, label=lab, marker=sym, edgecolor='k', facecolor='none')
    plt.xlabel(xlab, fontsize=20)
    plt.ylabel(ylab, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    if mags:
        plt.gca().invert_xaxis()
        
    if type(fit) != int and len(fit) !=0 and alpha !=0:
        plt.plot(magfit, fit,'k--', label=r'Fit: $\alpha = $'+str(round(alpha,2))+r', $M^{*}$ = '+str(round(m_star,2))+r', log($\Phi^{*}$) =' + str(round(phi_star,2)))    
    plt.legend(frameon=False, fontsize=14)
    if len(errsy) !=0:
        plt.errorbar(x, y, yerr=errsy, fmt='none', capsize=4, ecolor='k', elinewidth=1, capthick=1)
    plt.tight_layout()
    if save:
        plt.savefig('figures/'+title+'.png', dpi=250)
        plt.close()
        
        
plotlumfn(lum_fn_sub[1], np.log10(lum_fn_sub[3]), r'Abs. $[OIII]$', 
  r'log(N/V$_{\mathrm{max}}$)','./oiiilum_fn', 'o',
  alpha=fit_sch_sub[0], m_star=fit_sch_sub[2], phi_star = np.log10(fit_sch_sub[1]),
  lab='AGN', mags=True, fit=fit_sch_sub[-2], magfit=fit_sch_sub[-1],save=False)
plotlumfn(lum_fn_presub[1], np.log10(lum_fn_presub[3]), r'Abs. $[OIII]$', 
  r'log(N/V$_{\mathrm{max}}$)','./oiiilum_fn', 'o',
  alpha=fit_sch_presub[0], m_star=fit_sch_presub[2], phi_star = np.log10(fit_sch_presub[1]),
  lab='AGN', mags=True, fit=fit_sch_presub[-2], magfit=fit_sch_presub[-1],save=False)



        
plotlumfn(bootstr_sub_fn[0], np.log10(bootstr_sub_fn[1]), r'Abs. $[OIII]$', 
  r'log(N/V$_{\mathrm{max}}$)','./oiiilum_fn', 'o',
  alpha=bootstr_sub_fn[6], m_star=bootstr_sub_fn[8], phi_star = np.log10(bootstr_sub_fn[7]),
  lab='AGN', mags=True, fit=bootstr_sub_fn[9], magfit=bootstr_sub_fn[10],save=False)
plotlumfn(lum_fn_presub[1], np.log10(lum_fn_presub[3]), r'Abs. $[OIII]$', 
  r'log(N/V$_{\mathrm{max}}$)','./oiiilum_fn', 'o',
  alpha=fit_sch_presub[0], m_star=fit_sch_presub[2], phi_star = np.log10(fit_sch_presub[1]),
  lab='AGN', mags=True, fit=fit_sch_presub[-2], magfit=fit_sch_presub[-1],save=False)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:59:02 2020

@author: caug
"""

#lupton
def gr_to_V_lupton(g,r):
    return g-0.5784*(g-r)-0.0038
def gr_to_R_lupton(g,r):
    return r-0.1837*(g-r)-0.0971
def gr_to_V_lester(g,r):
    return g-0.59*(g-r)-0.01


tts8_V_lup = gr_to_V_lupton(16.34, 16.12)
tts8_R_lup = gr_to_R_lupton(16.34, 16.12)
tts8_V_les = gr_to_V_lester(16.34, 16.12)

tts4_V_lup = gr_to_V_lupton(16.04, 13.71)
tts4_R_lup = gr_to_R_lupton(16.04, 13.71)
tts4_V_les = gr_to_V_lester(16.04, 13.71)

tts31_V_lup = gr_to_V_lupton(18.08, 16.9)
tts31_R_lup = gr_to_R_lupton(18.08, 16.9)
tts31_V_les = gr_to_V_lester(18.08, 16.9)

tts59_V_lup = gr_to_V_lupton(14.54, 14.08)
tts59_R_lup = gr_to_R_lupton(14.54, 14.08)
tts59_V_les = gr_to_V_lester(14.54, 14.08)

tts22_V_lup = gr_to_V_lupton(15.72, 12.79)
tts22_R_lup = gr_to_R_lupton(15.72, 12.79)
tts22_V_les = gr_to_V_lester(15.72, 12.79)

tts23_V_lup = gr_to_V_lupton(16.79, 16.18)
tts23_R_lup = gr_to_R_lupton(16.79, 16.18)
tts23_V_les = gr_to_V_lester(16.79, 16.18)

tts57_V_lup = gr_to_V_lupton(17.27, 16.22)
tts57_R_lup = gr_to_R_lupton(17.27, 16.22)
tts57_V_les = gr_to_V_lester(17.27, 16.22)


tts104_V_lup = gr_to_V_lupton(14.92, 13.94)
tts104_R_lup = gr_to_R_lupton(14.92, 13.94)
tts104_V_les = gr_to_V_lester(14.92, 13.94)

tts210_V_lup = gr_to_V_lupton(17.19, 16.14)
tts210_R_lup = gr_to_R_lupton(17.19, 16.14)
tts210_V_les = gr_to_V_lester(17.19, 16.14)


'''