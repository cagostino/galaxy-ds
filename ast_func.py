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
from ast_func import *
from demarcations import *
import extinction
import pandas as pd
from setops import *
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from plotresults_sfrm import plot2dhist
import extinction as ext

ext.ccm89(np.array([3727.0]), 1.0, 3.1)/ext.ccm89(np.array([6563.0]), 1.0, 3.1)
xrayranallidict = {'soft':1/(1.39e-40),'hard':1/(1.26e-40),'full':1/(0.66e-40)}


def extinction(ha, hb, agn=False, dustlaw='cardelli', zeroed=False, ha_hb_err =[]):
    ha=np.copy(np.array(ha))
    hb=np.copy(np.array(hb))
    
    if agn:
        dec_rat = 3.1
    else:
        dec_rat = 2.86
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
def dustcorrect(flux, av, wl):
    '''
    Need to adjust so that the factor is dependent on the different lines
    '''
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
    
    

def get_bpt1_groups_ke01(x,y):
    groups =[]
    if type(x)!=np.array:
        xvals = np.copy(np.array(x))
    if type(y)!=np.array:
        yvals = np.copy(np.array(y))
    for i in range(len(xvals)):
        if xvals[i] < 0:
            if yvals[i] < np.log10(y1_kauffmann(xvals[i])):
                groups.append('HII')
            elif yvals[i] < np.log10(y1_kewley(xvals[i])):
                groups.append('COMP')
            else:
                if yvals[i] > np.log10(y1_schawinski(xvals[i])):
                    groups.append('Sy2')
                else:
                    groups.append('LINER')
        else:
            if xvals[i] < 0.43:
                if yvals[i] < np.log10(y1_kewley(xvals[i])):
                    groups.append('COMP')
                else:
                    if yvals[i] > np.log10(y1_schawinski(xvals[i])):
                        groups.append('Sy2')
                    else:
                        groups.append('LINER')
            else:
                if yvals[i] > np.log10(y1_schawinski(xvals[i])):
                    groups.append('Sy2')
                else:
                    groups.append('LINER')
            
    groups=np.array(groups)    
    sy2 = np.where(groups == 'Sy2')[0]
    liner = np.where(groups=='LINER')[0]
    comp = np.where(groups== 'COMP')[0]
    nonagn = np.where(groups == 'HII')[0]
    return groups,nonagn, comp, sy2, liner   
    
def get_bpt1_groups(x,y):
    groups =[]
    if type(x)!=np.array:
        xvals = np.copy(np.array(x))
    if type(y)!=np.array:
        yvals = np.copy(np.array(y))
    for i in range(len(xvals)):
        if np.isnan(xvals[i]) or np.isnan(yvals[i]):
            groups.append('NO')
        else:
            if xvals[i] < 0:
                if yvals[i] <np.log10(y1_kauffmann(xvals[i])):
                    groups.append('HII')
                else:
                    groups.append('AGN')
            else:
                groups.append('AGN')
    groups=np.array(groups)    
    agn = np.where(groups == 'AGN')[0]
    nonagn = np.where(groups == 'HII')[0]
    return groups,nonagn, agn
def get_bpt2_groups( x, y, filt=[]):
    groups=[]
    if type(x)!=np.array:
        xvals = np.copy(np.array(x))
    if type(y)!=np.array:
        yvals = np.copy(np.array(y))
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
def get_bpt3_groups( x, y, filt=[]):
    groups=[]
    if type(x)!=np.array:
        xvals = np.copy(np.array(x))
    if type(y)!=np.array:
        yvals = np.copy(np.array(y))
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
def get_whan_groups(x,y):
    groups = []
    if type(x)!=np.array:
        xvals = np.copy(np.array(x))
    if type(y)!=np.array:
        yvals = np.copy(np.array(y))
    for i in range(len(xvals)):
        if np.isnan(xvals[i]):
            groups.append('NO')
        elif yvals[i] < 0.5:
            groups.append('PG')
        elif  yvals[i] < 3:
            groups.append('RG')
        elif yvals[i] <6 and xvals[i] >-0.4:
            groups.append('wAGN')
        elif yvals[i]>6 and xvals[i]>-0.4:
            groups.append('sAGN')
        elif yvals[i]>3 and xvals[i]<-0.4:
            groups.append('SF')
    groups = np.array(groups)
    sf = np.where(groups == 'SF')[0]
    sagn = np.where(groups=='sAGN')[0]
    wagn = np.where(groups=='wAGN')[0]
    pg = np.where(groups=='PG')[0]
    rg = np.where(groups=='RG')[0]
    
    return groups,sf,sagn,wagn, rg, pg#hii,seyf,liner


def get_ooo_groups( x, y, filt=[]):
    groups=[]
    if type(x)!=np.array:
        xvals = np.copy(np.array(x))
    if type(y)!=np.array:
        yvals = np.copy(np.array(y))
    for i in range(len(xvals)):
        if np.isnan(xvals[i]) or np.isnan(yvals[i]):
            groups.append('NO')
            continue
        elif yvals[i] < np.log10(ooo_agn(xvals[i])):
            groups.append('HII')
            continue
        elif  yvals[i] > np.log10(ooo_linersy2(xvals[i])):
            groups.append('Seyfert')
        else:
            groups.append('LINER')
    groups = np.array(groups)
    hii = np.where(groups == 'HII')[0]
    seyf = np.where(groups=='Seyfert')[0]
    liner = np.where(groups=='LINER')[0]
    return groups,hii,seyf,liner#hii,seyf,liner
def get_bptplus_groups(x,y, filt=[]):
    groups =[]
    if type(x)!=np.array:
        xvals = np.copy(np.array(x))
    if type(y)!=np.array:
        yvals = np.copy(np.array(y))
    for i in range(len(xvals)):
        if np.isnan(xvals[i]) or np.isnan(yvals[i]):
            groups.append('NO')
        else:
            if xvals[i] < -0.35:
                if yvals[i] > np.log10(y1_kauffmann(xvals[i])):
                    groups.append('AGN')
                elif xvals[i]<-0.4:
                    groups.append('HII')
                else:
                    groups.append('MIX')
            else:
                groups.append('AGN')
    groups=np.array(groups)
    agn = np.where(groups == 'AGN')[0]
    nonagn = np.where(groups == 'HII')[0]
    return groups,nonagn, agn
def get_bptplus_niigroups( x, filt=[]):
    groups =[]
    if type(x)!=np.array:
        xvals = np.copy(np.array(x))
    for i in range(len(xvals)):
        if np.isnan(xvals[i]):
            groups.append('NO')
        else:
            if xvals[i] < -0.4:
                groups.append('HII')
            elif xvals[i] >-0.35:
                groups.append('AGN')
            else:
                groups.append('MIX')
    groups=np.array(groups)
    agn = np.where(groups == 'AGN')[0]
    nonagn = np.where(groups == 'HII')[0]
    return groups,nonagn, agn