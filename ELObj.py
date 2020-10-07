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
def halptofibsfr_corr( halplum):
    logsfrfib = np.log10(7.9e-42) + np.log10(halplum) - 0.24 #+0.4*A(H_alpha)
    return np.array(logsfrfib)
def halptofibsfr_uncorr( halplum,av):
    logsfrfib = np.log10(7.9e-42) + np.log10(halum) - 0.24 + 0.4*av
    return np.array(logsfrfib)
        
def get_deltassfr( mass, ssfr):
    m_ssfr = -0.4597
    b_ssfr = -5.2976
    delta_ssfr = ssfr-(mass*m_ssfr+b_ssfr)
    return np.array(delta_ssfr)

def nii_oii_to_oh( nii, oii):
    '''
    Uses Sanchez et al 2017 method for relating NII/OII to metallicity
    for Sy2.
    '''
    z = 1.08*(np.log10(nii/oii)**2) + 1.78*(np.log10(nii/oii)) + 1.24
    logoh = 12+np.log10(z*10**(-3.31))
    return np.array(logoh)


def nii_logoh_o32_to_q(logoh, o32):
    return (32.81-1.153*o32**2+(logoh)*(-3.396-0.025*o32+0.1444*o32**2))* ((4.603-0.3119*o32-0.163*o32**2)+(logoh)*(-0.48+0.0271*o32+0.02037*o32**2))**(-1)
def oiii_oii_to_U(oiii_oii):
    return 0.57*(oiii_oii**2)+1.38*(oiii_oii)-3.14


def correct_av(reg, x_test, av_balm, hb_sn):
    #x_test = av_gsw.transpose()
    av_balm_fixed = []
    for i in range(len(hb_sn)):

        if hb_sn[i] <10:
            x_samp = x_test[i].reshape(1,-1)
            av_fix = np.float64(reg.predict(x_samp))
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

def get_bpt1_groups(x,y):
    groups =[]
    if type(x)!=np.array:
        xvals = np.copy(np.array(x))
    if type(y)!=np.array:
        yvals = np.copy(np.array(y))
    for i in range(len(xvals)):
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
def get_bptplus_groups(x,y, filt=[]):
    groups =[]
    if type(x)!=np.array:
        xvals = np.copy(np.array(x))
    if type(y)!=np.array:
        yvals = np.copy(np.array(y))
    for i in range(len(xvals)):
        if xvals[i] < -0.4:
            if yvals[i] < np.log10(y1_kauffmann(xvals[i])):
                groups.append('HII')
            else:
                groups.append('AGN')
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
        if xvals[i] < -0.4:
            groups.append('HII')
        else:
            groups.append('AGN')
    groups=np.array(groups)
    agn = np.where(groups == 'AGN')[0]
    nonagn = np.where(groups == 'HII')[0]
    return groups,nonagn, agn
class ELObj:
    def __init__(self, sdssinds, sdss, make_spec, gswcat, gsw = False, xr=False, 
                 sncut=2, hb_sn_cut=2, dustbinning=False):
        self.sdss = sdss
        self.make_spec = make_spec
        self.gswcat = gswcat
        self.gsw = gsw
        self.xr = xr
        self.sncut= sncut
        
        
        self.halp_sn = np.reshape(sdss.allhalpha[sdssinds]/sdss.allhalpha_err[sdssinds],-1)
        self.nii_sn = np.reshape(sdss.allnII[sdssinds]/sdss.allnII_err[sdssinds],-1)
        self.halpnii_sn = 1./np.sqrt( (1./self.halp_sn)**2 +(1/self.nii_sn)**2)
        self.oi_sn = np.reshape(sdss.alloI[sdssinds]/sdss.alloI_err[sdssinds],-1)
        self.oiii_sn = np.reshape(sdss.alloIII[sdssinds]/sdss.alloIII_err[sdssinds],-1)
        self.hbeta_sn = np.reshape(sdss.allhbeta[sdssinds]/sdss.allhbeta_err[sdssinds],-1)
        self.hbetaoiii_sn = 1./np.sqrt( (1./self.hbeta_sn)**2 + (1/self.oiii_sn)**2 )

        self.sii6731_sn = np.reshape(sdss.allSII_6731[sdssinds]/sdss.allSII_6731_err[sdssinds],-1)
        self.sii6717_sn = np.reshape(sdss.allSII_6717[sdssinds]/sdss.allSII_6717_err[sdssinds],-1)
        self.sii_sn = np.reshape(sdss.allSII[sdssinds]/sdss.allSII_err[sdssinds],-1)
        
        self.neIII_sn = np.reshape(sdss.allneIII[sdssinds]/sdss.allneIIIerr[sdssinds],-1)
        self.oII_sn = np.reshape(sdss.alloII[sdssinds]/sdss.alloII_err[sdssinds],-1)
        #BPT filt used for everything
        self.bpt_sn_filt_bool = (self.halp_sn>sncut) & (self.hbeta_sn > sncut) & (self.oiii_sn > sncut) & (self.nii_sn > sncut)
        self.bpt_sn_filt = np.where(self.bpt_sn_filt_bool)[0]
        
        self.not_bpt_sn_filt_bool  = np.logical_not(self.bpt_sn_filt_bool)

        self.halp_nii_filt_bool = ( (self.halp_sn > sncut) & (self.nii_sn > sncut) &( (self.oiii_sn<=sncut) | (self.hbeta_sn<=sncut) ) )
        self.halp_nii_filt = np.where(self.halp_nii_filt_bool)[0]
        
        self.not_bpt_sn_filt_bool  = np.logical_not(self.bpt_sn_filt_bool)
        self.not_bpt_sn_filt  = np.where(self.not_bpt_sn_filt_bool)[0]
        
        self.neither_filt_bool = np.logical_not( ( (self.bpt_sn_filt_bool) | (self.halp_nii_filt_bool) ))#neither classifiable by BPT, or just by NII
        self.neither_filt = np.where(self.neither_filt_bool)[0]
        #testing S/N each line >3
        self.bpt_sn3_filt_bool = ((self.halp_sn > 3) & (self.hbeta_sn > 3) & (self.oiii_sn > 3) & (self.nii_sn > 3))
        self.bpt_sn3_filt = np.where(self.bpt_sn3_filt_bool)[0]
        
        self.halp_nii3_filt_bool = ( (self.halp_sn > 3) & (self.nii_sn > 3) &( (self.oiii_sn<=3) | (self.hbeta_sn<=3) ))
        self.halp_nii3_filt = np.where(self.halp_nii3_filt_bool)[0]
        
        self.neither3_filt_bool = np.logical_not( ((self.bpt_sn3_filt_bool ) | (self.halp_nii3_filt_bool)))#neither classifiable by BPT, or just by NII
        self.neither3_filt = np.where(self.neither3_filt_bool)[0]

        #testing S/N each line >1
        self.bpt_sn1_filt_bool = ((self.halp_sn > 1) & (self.hbeta_sn > 1) & (self.oiii_sn > 1) & (self.nii_sn > 1))
        self.bpt_sn1_filt = np.where(self.bpt_sn1_filt_bool)[0]
        
        self.halp_nii1_filt_bool = ( (self.halp_sn > 1) & (self.nii_sn > 1) &( (self.oiii_sn<=1) | (self.hbeta_sn<=1) ))
        self.halp_nii1_filt = np.where(self.halp_nii1_filt_bool)[0]
        
        self.neither1_filt_bool = np.logical_not( ((self.bpt_sn1_filt_bool ) | (self.halp_nii1_filt_bool)))#neither classifiable by BPT, or just by NII
        self.neither1_filt = np.where(self.neither1_filt_bool)[0]


        #testing S/N line ratio >3
        self.bpt_sn_lr3_filt_bool = ((self.halpnii_sn>3) &(self.hbetaoiii_sn >3) & (self.halp_sn > 0) &
                                        (self.hbeta_sn > 0) & (self.oiii_sn > 0) & (self.nii_sn > 0) &
                                        (np.isfinite(self.halpnii_sn)) & (np.isfinite(self.hbetaoiii_sn)))
        self.bpt_sn_lr3_filt = np.where( self.bpt_sn_lr3_filt_bool )[0]

        self.halp_nii_lr3_filt_bool =((self.halpnii_sn>3) &(self.hbetaoiii_sn <=3) & (self.halp_sn > 0) &
                                        (self.nii_sn > 0) & (np.isfinite(self.halpnii_sn)))
        self.halp_nii_lr3_filt = np.where(self.halp_nii_lr3_filt_bool)[0]
        
        self.neither_lr3_filt_bool = np.logical_not(((self.bpt_sn_lr3_filt_bool ) | (self.halp_nii_lr3_filt_bool)) )
        self.neither_lr3_filt = np.where(self.neither_lr3_filt_bool)[0]
    
        #testing S/N line ratio >3/sqrt(2)
        self.bpt_sn_lr2_filt_bool = ((self.halpnii_sn>2.12) &(self.hbetaoiii_sn >2.12) & (self.halp_sn > 0) &
                                        (self.hbeta_sn > 0) & (self.oiii_sn > 0) & (self.nii_sn > 0) &
                                        (np.isfinite(self.halpnii_sn)) & (np.isfinite(self.hbetaoiii_sn)))
        self.bpt_sn_lr2_filt = np.where( self.bpt_sn_lr2_filt_bool)[0]
        
        self.halp_nii_lr2_filt_bool = ((self.halpnii_sn > 2.12) & (self.hbetaoiii_sn <=2.12) & (self.halp_sn > 0) &
                                        (self.nii_sn > 0) & (np.isfinite(self.halpnii_sn)))
        self.halp_nii_lr2_filt = np.where( self.halp_nii_lr2_filt_bool)[0]
   
        self.neither_lr2_filt_bool=  np.logical_not(((self.halp_nii_lr2_filt_bool) | self.bpt_sn_lr2_filt_bool))
        self.neither_lr2_filt = np.where(self.neither_lr2_filt_bool)[0]
        #testing S/N line ratio > 2/sqrt(2)
        self.bpt_sn_minus_bpt_sn_lr2_filt_bool = ( (self.bpt_sn_filt_bool) & np.logical_not(self.bpt_sn_lr2_filt_bool))
        self.bpt_sn_minus_bpt_sn_lr2_filt = np.where(self.bpt_sn_minus_bpt_sn_lr2_filt_bool)[0]        
        self.halp_nii_minus_halp_nii_lr2_filt_bool = ( (self.halp_nii_filt_bool) & np.logical_not(self.halp_nii_lr2_filt_bool))
        self.halp_nii_minus_halp_nii_lr2_filt = np.where(self.halp_nii_minus_halp_nii_lr2_filt_bool)[0]
        
        
        self.bpt_sn_lr1_filt_bool = ((self.halpnii_sn > 1.41) &(self.hbetaoiii_sn > 1.41)& (self.halp_sn > 0) &
                                        (self.hbeta_sn > 0) & (self.oiii_sn > 0) & (self.nii_sn > 0) &
                                        (np.isfinite(self.halpnii_sn)) & (np.isfinite(self.hbetaoiii_sn)))        
        self.bpt_sn_lr1_filt = np.where(self.bpt_sn_lr1_filt_bool)[0]        
        self.halp_nii_lr1_filt_bool = ((self.halpnii_sn > 1.41) &(self.hbetaoiii_sn <= 1.41) & (self.halp_sn > 0) &
                                        (self.nii_sn > 0) & (np.isfinite(self.halpnii_sn)))
        self.halp_nii_lr1_filt = np.where(self.halp_nii_lr1_filt_bool )[0]
        self.neither_lr1_filt_bool = np.logical_not(( (self.bpt_sn_lr1_filt_bool) | (self.halp_nii_lr1_filt_bool)  ))
        self.neither_lr1_filt = np.where(self.neither_lr1_filt_bool)[0]
        
        self.vo87_1_filt = np.where(self.sii_sn[self.bpt_sn_filt]>2)[0]
        self.vo87_2_filt = np.where(self.oi_sn[self.bpt_sn_filt]>2)[0]
        
        self.sdss_filt = sdssinds[self.bpt_sn_filt]
        self.sdss_filt_plus = sdssinds[self.halp_nii_filt]
        self.sdss_filt_neither = sdssinds[self.neither_filt]
        self.sdss_filt_weak = sdssinds[self.not_bpt_sn_filt]

        self.tbtx = sdss.allneIII[sdssinds]/sdss.alloII[sdssinds]
             
  

        self.EL_dict = {'oiiiflux': np.copy(sdss.alloIII)/1e17,
                        'hbetaflux': np.copy(sdss.allhbeta)/1e17,
                        'halpflux': np.copy(sdss.allhalpha)/1e17,
                        'niiflux': np.copy(sdss.allnII)/1e17,
                        'siiflux': np.copy(sdss.allSII)/1e17,
                        'sii6731flux': np.copy(sdss.allSII_6731)/1e17,
                        'sii6717flux': np.copy(sdss.allSII_6717)/1e17,
                        'oiflux': np.copy(sdss.alloI)/1e17,
                        'hdeltaflux': np.copy(sdss.allhdelta)/1e17,
                        'hgammaflux': np.copy(sdss.allhgamma)/1e17,
                        'oiii4363flux': np.copy(sdss.alloIII4363)/1e17,
                        'oiii4959flux': np.copy(sdss.alloIII4959)/1e17,
                        'heIflux': np.copy(sdss.allheI)/1e17,
                        'nII6548flux': np.copy(sdss.allnII_6548)/1e17,
                        'oiiflux': np.copy(sdss.alloII)/1e17,
                        
                        'd4000': np.copy(sdss.alld4000),
                        'halp_eqw':np.copy(sdss.allha_eqw),
                        'oiii_eqw':np.copy(sdss.alloIII_eqw),
                        'vdisp':np.copy(sdss.allvdisp),
                        'mbh': np.log10(np.copy(3*(sdss.allvdisp/200)**4)*1e8),
                        'forbiddenfwhm':  np.copy(sdss.allforbiddendisp*2 * np.sqrt(2*np.log(2))),
                        'balmerfwhm': np.copy(sdss.allbalmerdisp*2 * np.sqrt(2*np.log(2))),
                        'fibmass':np.copy(sdss.all_fibmass),
                        'fibsfr_mpa' : np.copy(sdss.all_fibsfr_mpa),
                        'fibssfr_mpa' : np.copy(sdss.all_fibssfr_mpa),                 
                        'ohabund':  np.copy(sdss.fiboh),

                        'oiii_err': np.copy(sdss.alloIII_err)/1e17,
                        'hbeta_err': np.copy(sdss.allhbeta_err)/1e17,
                        'halp_err': np.copy(sdss.allhalpha_err)/1e17,
                        'nii_err': np.copy(sdss.allnII_err)/1e17,
                        'sii_err': np.copy(sdss.allSII_err)/1e17,
                        'sii6731_err': np.copy(sdss.allSII_6731_err)/1e17,
                        'sii6717_err': np.copy(sdss.allSII_6717_err)/1e17,
                        'oi_err': np.copy(sdss.alloI_err)/1e17,
                        'oiii4363_err': np.copy(sdss.alloIII4363_err)/1e17,
                        'oiii4959_err': np.copy(sdss.alloIII4959_err)/1e17,
                        'oii_err': np.copy(sdss.alloII_err)/1e17,
                        
                        'halpflux_sn':  np.copy(sdss.allhalpha/sdss.allhalpha_err),
                        'niiflux_sn': np.copy(sdss.allnII/sdss.allnII_err),
                        'oiflux_sn': np.copy(sdss.alloI/sdss.alloI_err),
                        'oiiiflux_sn': np.copy(sdss.alloIII/sdss.alloIII_err),
                        'hbetaflux_sn':  np.copy(sdss.allhbeta/sdss.allhbeta_err),
                        'sii6731flux_sn':  np.copy(sdss.allSII_6731/sdss.allSII_6731_err),
                        'sii6717flux_sn': np.copy(sdss.allSII_6717/sdss.allSII_6717_err),
                        'siiflux_sn':np.copy(sdss.allSII/sdss.allSII_err),
                        'oiiflux_sn': np.copy(sdss.alloII/sdss.alloII_err),
                        'forbiddenfwhmerr': np.copy(sdss.allforbiddendisperr*2*np.sqrt(2*np.log(2))),
                        'balmerfwhmerr': np.copy(sdss.allbalmerdisperr*2*np.sqrt(2*np.log(2)))
                        }
 

        self.EL_dict['massfrac'] = np.copy(10**(sdss.all_fibmass))/np.copy(10**(sdss.all_sdss_avgmasses))
        self.EL_dict['yvals_bpt'] =self.EL_dict['oiiiflux']/self.EL_dict['hbetaflux']
        self.EL_dict['xvals1_bpt'] =self.EL_dict['niiflux']/self.EL_dict['halpflux']
        self.EL_dict['xvals2_bpt'] =self.EL_dict['siiflux']/self.EL_dict['halpflux']
        self.EL_dict['xvals3_bpt'] =self.EL_dict['niiflux']/self.EL_dict['halpflux']
            
            
        self.EL_dict['av'] = extinction(self.EL_dict['halpflux'], self.EL_dict['hbetaflux'])
        self.EL_dict['av_agn'] = extinction(self.EL_dict['halpflux'], self.EL_dict['hbetaflux'],agn=True)
        
        self.EL_dict['niiha'] = np.log10(np.copy(self.EL_dict['xvals1_bpt']))
        self.EL_dict['siiha'] = np.log10(np.copy(self.EL_dict['xvals2_bpt']))
        self.EL_dict['oiha'] = np.log10(np.copy(self.EL_dict['xvals3_bpt']))
        self.EL_dict['oiiihb'] = np.log10(np.copy(self.EL_dict['yvals_bpt']))

        self.EL_dict_gsw = {}
        for key in self.EL_dict.keys():
            self.EL_dict_gsw[key] = self.EL_dict[key][sdssinds]
        self.EL_dict_gsw['mass'] = np.copy(self.gswcat.mass[self.make_spec])
        self.EL_dict_gsw['massfracgsw'] = np.copy(10**( self.EL_dict_gsw['fibmass']))/np.copy(10**(self.EL_dict_gsw['mass']))

        self.EL_dict_gsw['ids'] = np.copy(self.gswcat.ids[self.make_spec])
        self.EL_dict_gsw['sfr'] = np.copy(self.gswcat.sfr[self.make_spec])

        self.EL_dict_gsw['ssfr'] = np.copy(self.EL_dict_gsw['sfr'] - self.EL_dict_gsw['mass'])
        self.EL_dict_gsw['delta_ssfr'] = get_deltassfr(self.EL_dict_gsw['mass'], self.EL_dict_gsw['ssfr'])

        
        self.EL_dict_gsw['z'] =  np.copy(self.gswcat.z[self.make_spec])
        self.EL_dict_gsw['ra'] = np.copy(self.gswcat.ra[self.make_spec])
        self.EL_dict_gsw['dec'] = np.copy(self.gswcat.dec[self.make_spec])
        self.EL_dict_gsw['av_gsw'] = np.copy(self.gswcat.av[self.make_spec])
        self.EL_dict_gsw['fibsfr']= np.copy(self.EL_dict_gsw['sfr']+np.log10(self.EL_dict_gsw['massfrac']))
        self.EL_dict_gsw['fibsfrgsw']= np.copy(self.EL_dict_gsw['sfr']+np.log10(self.EL_dict_gsw['massfracgsw']))


        high_sn10_hb = np.where((self.EL_dict_gsw['hbetaflux_sn']>10)&(self.EL_dict_gsw['halpflux_sn']>0))           
        self.y_reg = np.array(self.EL_dict_gsw['av_agn'][high_sn10_hb])

        self.X_reg_av = np.vstack([self.EL_dict_gsw['av_gsw'][high_sn10_hb]]).transpose()
        self.reg_av = LinearRegression().fit(self.X_reg_av,self.y_reg)
        self.x_pred_av = np.vstack([self.EL_dict_gsw['av_gsw']]).transpose()
        self.EL_dict_gsw['corrected_presub_av'] = correct_av(self.reg_av, self.x_pred_av, 
                                                         np.array(self.EL_dict_gsw['av_agn']),
                                                         np.array(self.EL_dict_gsw['hbetaflux_sn']))
        '''
        self.X_reg_av_sfr = np.vstack([self.EL_dict_gsw['av_gsw'][high_sn10_hb], self.EL_dict_gsw['sfr'][high_sn10_hb]]).transpose()
        self.reg_av_sfr = LinearRegression().fit(self.X_reg_av_sfr,self.y_reg)
        self.x_pred_av_sfr = np.vstack([self.EL_dict_gsw['av_gsw'], self.EL_dict_gsw['sfr']]).transpose()
        self.EL_dict_gsw['corrected_presub_av_sfr'] = correct_av(self.reg_av_sfr, self.x_pred_av_sfr, 
                                                         np.array(self.EL_dict_gsw['av_agn']),
                                                         np.array(self.EL_dict_gsw['hbetaflux_sn']))
        self.X_reg_av_mass = np.vstack([self.EL_dict_gsw['av_gsw'][high_sn10_hb], self.EL_dict_gsw['mass'][high_sn10_hb]]).transpose()
        self.reg_av_mass = LinearRegression().fit(self.X_reg_av_mass,self.y_reg)
        self.x_pred_av_mass = np.vstack([self.EL_dict_gsw['av_gsw'], self.EL_dict_gsw['mass']]).transpose()
        self.EL_dict_gsw['corrected_presub_av_mass'] = correct_av(self.reg_av_mass, self.x_pred_av_mass, 
                                                         np.array(self.EL_dict_gsw['av_agn']),
                                                         np.array(self.EL_dict_gsw['hbetaflux_sn']))

        self.X_reg_av_mass_sfr = np.vstack([self.EL_dict_gsw['av_gsw'][high_sn10_hb], self.EL_dict_gsw['mass'][high_sn10_hb],self.EL_dict_gsw['sfr'][high_sn10_hb]]).transpose()
        self.reg_av_mass_sfr = LinearRegression().fit(self.X_reg_av_mass_sfr,self.y_reg)
        self.x_pred_av_mass_sfr = np.vstack([self.EL_dict_gsw['av_gsw'], self.EL_dict_gsw['mass'],self.EL_dict_gsw['sfr']]).transpose()
        self.EL_dict_gsw['corrected_presub_av_mass_sfr'] = correct_av(self.reg_av_mass_sfr, self.x_pred_av_mass_sfr, 
                                                         np.array(self.EL_dict_gsw['av_agn']),
                                                         np.array(self.EL_dict_gsw['hbetaflux_sn']))
        '''
        self.EL_dict_gsw['hbetaflux_corr'] = dustcorrect(self.EL_dict_gsw['hbetaflux'], self.EL_dict_gsw['corrected_presub_av'], 4861.0)
        self.EL_dict_gsw['halpflux_corr'] = dustcorrect(self.EL_dict_gsw['halpflux'], self.EL_dict_gsw['corrected_presub_av'], 6563.0)
        
        self.EL_dict_gsw['oiiiflux_corr'] = dustcorrect(self.EL_dict_gsw['oiiiflux'], self.EL_dict_gsw['corrected_presub_av'], 5007.0)
        
        self.EL_dict_gsw['oiiflux_corr'] = dustcorrect(self.EL_dict_gsw['oiiflux'], self.EL_dict_gsw['corrected_presub_av'], 3727.0)
        self.EL_dict_gsw['niiflux_corr'] = dustcorrect(self.EL_dict_gsw['niiflux'], self.EL_dict_gsw['corrected_presub_av'], 6583.0)
        self.EL_dict_gsw['oiflux_corr'] = dustcorrect(self.EL_dict_gsw['oiflux'], self.EL_dict_gsw['corrected_presub_av'], 6300.0)
        self.EL_dict_gsw['siiflux_corr'] = dustcorrect(self.EL_dict_gsw['siiflux'], self.EL_dict_gsw['corrected_presub_av'], 6724.0)
        self.EL_dict_gsw['sii6717flux_corr'] = dustcorrect(self.EL_dict_gsw['sii6717flux'], self.EL_dict_gsw['corrected_presub_av'], 6717.0)
        self.EL_dict_gsw['sii6731flux_corr'] = dustcorrect(self.EL_dict_gsw['sii6731flux'], self.EL_dict_gsw['corrected_presub_av'], 6731.0)


        self.EL_dict_gsw['oiii_oii'] = np.log10(self.EL_dict_gsw['oiiiflux_corr']/self.EL_dict_gsw['oiiflux_corr'])
        self.EL_dict_gsw['U'] =oiii_oii_to_U(self.EL_dict_gsw['oiii_oii'] )
        self.EL_dict_gsw['nii_oii'] =np.log10(self.EL_dict_gsw['niiflux']/self.EL_dict_gsw['oiiflux_corr'])
        self.EL_dict_gsw['log_oh'] = nii_oii_to_oh(self.EL_dict_gsw['niiflux_corr'], self.EL_dict_gsw['oiiflux_corr'])             
        self.EL_dict_gsw['sii_ratio'] =(self.EL_dict_gsw['sii6717flux_corr']/self.EL_dict_gsw['sii6731flux_corr'])
        self.EL_dict_gsw['sii_oii'] =np.log10(self.EL_dict_gsw['siiflux']/self.EL_dict_gsw['oiiflux'])
        self.EL_dict_gsw['oi_sii'] =np.log10(self.EL_dict_gsw['oiflux']/self.EL_dict_gsw['siiflux'])
        
        self.EL_dict_gsw['oiiilum'] = np.log10(getlumfromflux(self.EL_dict_gsw['oiiiflux_corr'],self.EL_dict_gsw['z']))
        self.EL_dict_gsw['halplum'] = np.log10(getlumfromflux(self.EL_dict_gsw['halpflux_corr'], self.EL_dict_gsw['z']))
        self.EL_dict_gsw['oiiilum_uncorr'] = getlumfromflux(self.EL_dict_gsw['oiiiflux'],self.EL_dict_gsw['z'])
        self.EL_dict_gsw['halplum_uncorr'] = getlumfromflux(self.EL_dict_gsw['halpflux'], self.EL_dict_gsw['z'])
        self.EL_dict_gsw['halpfibsfr'] = halptofibsfr_corr(10**self.EL_dict_gsw['halplum'])

        if self.xr:
            self.EL_dict_gsw['exptimes'] = np.copy(self.gswcat.exptimes[self.make_spec])
            self.EL_dict_gsw['xrayflag'] = np.copy(self.gswcat.xrayflag[self.make_spec])
            self.EL_dict_gsw['ext'] = np.copy(self.gswcat.ext[self.make_spec])


        
        self.EL_df = pd.DataFrame.from_dict(self.EL_dict)
        self.EL_gsw_df = pd.DataFrame.from_dict(self.EL_dict_gsw)

   
        self.bpt_EL_gsw_df = self.EL_gsw_df.iloc[self.bpt_sn_filt].copy()
        self.vo87_1_EL_gsw_df = self.bpt_EL_gsw_df.iloc[self.vo87_1_filt].copy()
        self.vo87_2_EL_gsw_df = self.bpt_EL_gsw_df.iloc[self.vo87_2_filt].copy()
        
        bptgroups, bptsf, bptagn = get_bpt1_groups( np.log10(self.bpt_EL_gsw_df['xvals1_bpt']), np.log10(self.bpt_EL_gsw_df['yvals_bpt'] ) )
        
        bptplsugroups, bptplssf, bptplsagn = get_bptplus_groups(np.log10(self.bpt_EL_gsw_df['xvals1_bpt']), np.log10(self.bpt_EL_gsw_df['yvals_bpt']))
        

        #self.bpt_EL_gsw_df['bptgroups'] = bptgroups
        #self.bpt_EL_gsw_df['bptplusgroups'] = bptplsugroups
        
        self.bpt_sf_df = self.bpt_EL_gsw_df.iloc[bptsf].copy()
        self.bpt_agn_df = self.bpt_EL_gsw_df.iloc[bptagn].copy()
        
        self.bptplus_sf_df = self.bpt_EL_gsw_df.iloc[bptplssf].copy()
        self.bptplus_agn_df = self.bpt_EL_gsw_df.iloc[bptplsagn].copy()
                                                                                                                                                                                           
        self.plus_EL_gsw_df = self.EL_gsw_df.iloc[self.halp_nii_filt].copy()

        groups_bptplusnii, bptplsnii_sf, bptplsnii_agn= get_bptplus_niigroups(np.log10(self.plus_EL_gsw_df['xvals1_bpt']))

        self.bptsf = bptsf
        self.bptagn = bptagn
        self.bptplssf = bptplssf
        self.bptplsagn = bptplsagn
        self.bptplsnii_sf = bptplsnii_sf
        self.bptplsnii_agn = bptplsnii_agn
        #self.plus_EL_gsw_df['bptplusniigroups'] = groups_bptplusnii
        
        self.bptplusnii_sf_df = self.plus_EL_gsw_df.iloc[bptplsnii_sf].copy()
        self.bptplusnii_agn_df = self.plus_EL_gsw_df.iloc[bptplsnii_agn].copy()
        
        self.neither_EL_gsw_df = self.EL_gsw_df.iloc[self.neither_filt].copy()
        self.allnonagn_df=pd.concat([self.bptplus_sf_df,self.bptplusnii_sf_df, self.neither_EL_gsw_df],join='outer')
        self.lines = ['oiiiflux','hbetaflux','halpflux','niiflux', 'siiflux',
                  'sii6731flux', 'sii6717flux','oiflux','oiii4363flux',
                  'oiii4959flux','oiiflux']
        
        self.line_errs = ['oiii_err', 'hbeta_err', 'halp_err', 'nii_err', 
                    'sii_err','sii6731_err', 'sii6717_err', 'oi_err',
                    'oiii4363_err', 'oiii4959_err', 'oii_err']
        '''
        self.allnonagn_hb_sn_filts_df = []
        self.hb_sn_filts = []

        self.agn_hb_sn_filts_df = []
        self.hb_agn_sn_filts = []


        self.allnonagn_hb_ha_sn_filts_df = []
        self.hb_ha_sn_filts = []


        sns = [2,5,10,20]
        for i in range(len(sns)):
            hb_sn_filter = np.where((self.allnonagn_df.hbeta_sn>sns[i])    &(self.allnonagn_df.halp_sn>0))
            allnonagn_hb_filt_df = self.allnonagn_df.iloc[hb_sn_filter].copy()            
            self.hb_sn_filts.append(hb_sn_filter)
            self.allnonagn_hb_sn_filts_df.append(allnonagn_hb_filt_df)

            hb_agn_sn_filter = np.where((self.bptplus_agn_df.hbeta_sn>sns[i]) &(self.bptplus_agn_df.halp_sn>0))    
            agn_hb_filt_df = self.bptplus_agn_df.iloc[hb_agn_sn_filter].copy()            
            self.hb_agn_sn_filts.append(hb_agn_sn_filter)
            self.agn_hb_sn_filts_df.append(agn_hb_filt_df)


            hb_ha_sn_filter = np.where((self.allnonagn_df.hbeta_sn>sns[i])&(self.allnonagn_df.halp_sn>sns[i]))    
            allnonagn_hb_ha_filt_df = self.allnonagn_df.iloc[hb_ha_sn_filter].copy()            
            self.hb_sn_filts.append(hb_ha_sn_filter)
            self.allnonagn_hb_ha_sn_filts_df.append(allnonagn_hb_ha_filt_df)



        self.allnonagn_ha_sn_filts_df = []
        self.ha_sn_filts = []
        sns = [2,5,10,20]
        for i in range(len(sns)):
            ha_sn_filter = np.where(self.allnonagn_df.halp_sn>sns[i])    
            allnonagn_ha_filt_df = self.allnonagn_df.iloc[ha_sn_filter].copy()            
            self.ha_sn_filts.append(ha_sn_filter)
            self.allnonagn_ha_sn_filts_df.append(allnonagn_ha_filt_df)

            
            
        bin_by_av_gsw_filts = np.arange(0,1.2, 0.2)
        self.avgsw_filts = []
        self.allnonagn_avgsw_filts_df = []

        self.agn_avgsw_filts = []
        self.agn_avgsw_filts_df = []
        
        for i in range(len(bin_by_av_gsw_filts)-1):            
            avgsw_filter = np.where((self.allnonagn_hb_sn_filts_df[2].av_gsw>bin_by_av_gsw_filts[i]) &(self.allnonagn_hb_sn_filts_df[2].av_gsw<bin_by_av_gsw_filts[i+1]))    
            allnonagn_avgsw_filt_df = self.allnonagn_hb_sn_filts_df[2].iloc[avgsw_filter].copy()
            self.avgsw_filts.append(avgsw_filter)
            self.allnonagn_avgsw_filts_df.append(allnonagn_avgsw_filt_df)

            agn_avgsw_filt = np.where((self.agn_hb_sn_filts_df[2].av_gsw>bin_by_av_gsw_filts[i]) &(self.agn_hb_sn_filts_df[2].av_gsw<bin_by_av_gsw_filts[i+1]))    
            agn_avgsw_filt_df = self.agn_hb_sn_filts_df[2].iloc[agn_avgsw_filt].copy()
            self.agn_avgsw_filts.append(agn_avgsw_filt)
            self.agn_avgsw_filts_df.append(agn_avgsw_filt_df)



        if dustbinning:            
            self.bin_by_dust(self.allnonagn_df.copy(),'sfr',-3.5, 3.5)
            self.get_pred_av()
            self.fixed_nonagn_av = self.correct_av(self.sfr_av_reg, np.array(self.allnonagn_df.sfr),
                                            np.array(self.allnonagn_df.av_gsw),
                                            np.array(self.allnonagn_df.av), 
                                            np.array(self.allnonagn_df.hbeta_sn) )
            self.fixed_agn_av_sfcorr = self.correct_av(self.sfr_av_reg, np.array(self.bptplus_agn_df.sfr),
                                            np.array(self.bptplus_agn_df.av_gsw),
                                            np.array(self.bptplus_agn_df.av_agn), 
                                            np.array(self.bptplus_agn_df.hbeta_sn) )

            self.fixed_agn_av_agncorr = self.correct_av(self.sfr_av_reg_agn, np.array(self.bptplus_agn_df.sfr),
                                            np.array(self.bptplus_agn_df.av_gsw),
                                            np.array(self.bptplus_agn_df.av_agn), 
                                            np.array(self.bptplus_agn_df.hbeta_sn) )
            self.fixed_agn2_av_sfcorr = self.correct_av(self.sfr_av_reg, np.array(self.bptplus_agn_df.sfr),
                                            np.array(self.bptplus_agn_df.av_gsw),
                                            np.array(self.bptplus_agn_df.av_agn), 
                                            np.array(self.bptplus_agn_df.hbeta_sn) )
            self.fixed_agn2_av_agncorr = self.correct_av(self.sfr_av_reg_agn2, np.array(self.bptplus_agn_df.sfr),
                                            np.array(self.bptplus_agn_df.av_gsw),
                                            np.array(self.bptplus_agn_df.av_agn), 
                                            np.array(self.bptplus_agn_df.hbeta_sn) )


        '''
    def bin_by_dust(self,binningdf, quantity, minq, maxq, step=0.1):            
            
        self.halp_binfluxes = []
        self.hbeta_binfluxes = []
        self.balmdec_binned = []
        #self.sfr_bin_centers = (sfr_rang[1:]+sfr_rang[:-1])/2

        self.bn_edges, self.bncenters, self.bns, self.bn_inds, self.valbns = bin_quantity(np.array(binningdf[quantity]), step, minq, maxq, threshold=20)

        self.hbeta_flux_binned = bin_by_ind(np.array(binningdf.hbetaflux), 
                                                        self.bn_inds,
                                                        self.bncenters[self.valbns])
        self.halpha_flux_binned = bin_by_ind(np.array(binningdf.halpflux),
                                                         self.bn_inds, 
                                                         self.bncenters[self.valbns])
        self.halp_binfluxes_sums = []
        self.hbeta_binfluxes_sums = []
        self.balmdec_binned_sums = []

        self.halp_binfluxes_medians = []
        self.hbeta_binfluxes_medians = []
        self.balmdec_binned_medians = []

        for i in range(len(self.bn_edges)-1):
            filt = (binningdf[quantity]>self.bn_edges[i]) & (binningdf[quantity]<self.bn_edges[i+1])
            if sum(filt) <20:
                self.halp_binfluxes_sums.append(np.nan)
                self.hbeta_binfluxes_sums.append(np.nan)
                self.balmdec_binned_sums.append(np.nan)

                self.halp_binfluxes_medians.append(np.nan)
                self.hbeta_binfluxes_medians.append(np.nan)
                self.balmdec_binned_medians.append(np.nan)

                continue
            halpflux = np.array(binningdf['halpflux'][filt].copy())
            hbetaflux = np.array(binningdf['hbetaflux'][filt].copy())
    
            halpflux_sum = np.nansum(halpflux)
            hbetaflux_sum = np.nansum(hbetaflux)
            
            halpflux_med = np.nanmedian(halpflux)
            hbetaflux_med = np.nanmedian(hbetaflux)
            
            self.halp_binfluxes_sums.append(halpflux_sum)
            self.hbeta_binfluxes_sums.append(hbetaflux_sum)
            self.balmdec_binned_sums.append(halpflux_sum/hbetaflux_sum)

            self.halp_binfluxes_medians.append(halpflux_med)
            self.hbeta_binfluxes_medians.append(hbetaflux_med)
            self.balmdec_binned_medians.append(halpflux_med/hbetaflux_med)

        self.halp_binfluxes_sums = np.array(self.halp_binfluxes_sums)
        self.hbeta_binfluxes_sums = np.array(self.hbeta_binfluxes_sums)
        
        self.halp_binfluxes_medians = np.array(self.halp_binfluxes_medians)
        self.hbeta_binfluxes_medians = np.array(self.hbeta_binfluxes_medians)
        
        self.ext_binned_sums = extinction(self.halp_binfluxes_sums, self.hbeta_binfluxes_sums)
        self.ext_binned_medians = extinction(self.halp_binfluxes_medians, self.hbeta_binfluxes_medians)
        
    
        self.bootstrapped_inds = [bootstrap(np.array(self.bn_inds[i]), 1000, data_only=True) for i in range(len(self.bn_inds))]

        self.bootstrapped_halpha_mean_sums = []
        self.bootstrapped_halpha_median = []
        self.bootstrapped_halpha_std_sums = []
        self.bootstrapped_halpha_std_median = []

        self.bootstrapped_hbeta_mean_sums = []
        self.bootstrapped_hbeta_median = []
        self.bootstrapped_hbeta_std_sums = []
        self.bootstrapped_hbeta_std_median = []


        
        for i in range(len(self.bootstrapped_inds)): 
            halp_flux_sum = []
            halp_flux_medians= [] 
            hbeta_flux_sum = []
            hbeta_flux_medians= [] 


            for j in range(len(self.bootstrapped_inds[i])):            
                halp_flux_sum.append( np.sum(binningdf.halpflux.iloc[np.int64(self.bootstrapped_inds[i][j])]))
                halp_flux_medians.append( np.median(binningdf.halpflux.iloc[np.int64(self.bootstrapped_inds[i][j])]))
                hbeta_flux_sum.append( np.sum(binningdf.hbetaflux.iloc[np.int64(self.bootstrapped_inds[i][j])]))
                hbeta_flux_medians.append( np.median(binningdf.hbetaflux.iloc[np.int64(self.bootstrapped_inds[i][j])]))
                
            mean_halp_flux_sums = np.mean(halp_flux_sum)
            std_halp_flux_sums = np.std(halp_flux_sum)
            mdn_halp_flux = np.median(halp_flux_medians)
            std_halp_flux_meds = np.std(halp_flux_medians)

            self.bootstrapped_halpha_mean_sums.append(mean_halp_flux_sums)
            self.bootstrapped_halpha_median.append(mdn_halp_flux)
            self.bootstrapped_halpha_std_sums.append(std_halp_flux_sums)                       
            self.bootstrapped_halpha_std_median.append(std_halp_flux_meds)
            mean_hbeta_flux_sums = np.mean(hbeta_flux_sum)

            std_hbeta_flux_sums = np.std(hbeta_flux_sum)
            mdn_hbeta_flux = np.median(hbeta_flux_medians)
            std_hbeta_flux_meds = np.std(hbeta_flux_medians)

            self.bootstrapped_hbeta_mean_sums.append(mean_hbeta_flux_sums)
            self.bootstrapped_hbeta_median.append(mdn_hbeta_flux)
            self.bootstrapped_hbeta_std_sums.append(std_hbeta_flux_sums)
            self.bootstrapped_hbeta_std_median.append(std_hbeta_flux_meds)

            
        self.bootstrapped_halpha_mean_sums = np.array(self.bootstrapped_halpha_mean_sums)
        self.bootstrapped_halpha_median = np.array(self.bootstrapped_halpha_median)
        self.bootstrapped_halpha_std_sums = np.array(self.bootstrapped_halpha_std_sums)
        self.bootstrapped_halpha_std_median = np.array(self.bootstrapped_halpha_std_median)
        
        self.bootstrapped_hbeta_mean_sums = np.array(self.bootstrapped_hbeta_mean_sums)
        self.bootstrapped_hbeta_median = np.array(self.bootstrapped_hbeta_median)
        self.bootstrapped_hbeta_std_sums = np.array(self.bootstrapped_hbeta_std_sums)
        self.bootstrapped_hbeta_std_median = np.array(self.bootstrapped_hbeta_std_median)

        
        self.bootstrapped_balmdecs_sums = self.bootstrapped_halpha_mean_sums/self.bootstrapped_hbeta_mean_sums
        self.bootstrapped_balmdecs_medians = self.bootstrapped_halpha_median/self.bootstrapped_hbeta_median
        self.bootstrapped_halphbeta_sums = self.bootstrapped_halpha_mean_sums/self.bootstrapped_hbeta_mean_sums
        self.bootstrapped_halphbeta_err_sums = self.bootstrapped_halphbeta_sums*np.sqrt( (self.bootstrapped_halpha_std_sums/self.bootstrapped_halpha_mean_sums)**2+
                                                                              (self.bootstrapped_hbeta_std_sums/self.bootstrapped_hbeta_mean_sums)**2)
        self.bootstrapped_halphbeta_meds = self.bootstrapped_halpha_median/self.bootstrapped_hbeta_median
        self.bootstrapped_halphbeta_err_meds = self.bootstrapped_halphbeta_meds*np.sqrt( (self.bootstrapped_halpha_std_median/self.bootstrapped_halpha_median)**2+
                                                                              (self.bootstrapped_hbeta_std_median /self.bootstrapped_hbeta_median)**2)

        self.bootstr_ext_sums, self.bootstr_ext_sums_err =  extinction(self.bootstrapped_halpha_mean_sums, self.bootstrapped_hbeta_mean_sums, ha_hb_err =self.bootstrapped_halphbeta_err_sums )
        self.bootstr_ext_meds, self.bootstr_ext_meds_err =  extinction(self.bootstrapped_halpha_median, self.bootstrapped_hbeta_median, ha_hb_err= self.bootstrapped_halphbeta_err_meds )
        
    def get_pred_av(self):
        xmids_gsw = []
        yavs_gsw = []
        av_avgs = []
        xmids_agn = []
        yavs_agn = []
        av_avgs_agn = []
        xmids_agn2 = []
        av_avgs_agn2 = []
        yavs_agn2 = []
        
        
        for i in range(len(self.allnonagn_avgsw_filts_df)):
            plt.figure()
            xmid, yav, av_av = plot2dhist(self.allnonagn_avgsw_filts_df[i]['sfr'], 
                                          self.allnonagn_avgsw_filts_df[i]['av'],
                                          bin_quantity = self.allnonagn_avgsw_filts_df[i]['av_gsw'],
                                          nan=True, maxx=3.5, minx=-3.5, miny=-5, maxy=5.,
                                          lim=True, xlabel='log(SFR)', ylabel=r'A$_{\mathrm{V}}$',
                                          setplotlims=True, bin_y=True)
            av_avgs.append(av_av)  
            xmids_gsw.append(xmid)
            yavs_gsw.append(yav)
            plt.close()
            xmid, yav, av_agn2 = plot2dhist(self.agn_avgsw_filts_df[i]['sfr'], self.agn_avgsw_filts_df[i]['av_agn'],
                                   bin_quantity = self.agn_avgsw_filts_df[i]['av_gsw'],
                                   nan=True, maxx=3.5, minx=-3.5, miny=-5, maxy=5., 
                                   lim=True, xlabel='log(SFR)', ylabel=r'A$_{\mathrm{V}}$', 
                                   setplotlims=True, bin_y=True)
        
            xmids_agn2.append(xmid)
            yavs_agn2.append(yav)
            av_avgs_agn2.append(av_agn2)
            plt.close()
            
            xmid, yav, av_agn = plot2dhist(self.agn_avgsw_filts_df[i]['sfr'], self.agn_avgsw_filts_df[i]['av'], 
                                   bin_quantity = self.agn_avgsw_filts_df[i]['av_gsw'],
                                   nan=True, maxx=3.5, minx=-3.5, miny=-5, maxy=5., lim=True, xlabel='log(SFR)',
                                   ylabel=r'A$_{\mathrm{V}}$', setplotlims=True, bin_y=True)
        
            xmids_agn.append(xmid)
            yavs_agn.append(yav)
            av_avgs_agn.append(av_agn)
            plt.close()            
        self.xmids_gsw = np.array(xmids_gsw)
        self.yavs_gsw = np.array(yavs_gsw)
        self.av_avgs = np.array(av_avgs)
        self.xmids_agn = np.array(xmids_agn)
        self.xmids_agn2 = np.array(xmids_agn2)
        self.yavs_agn = np.array(yavs_agn)
        self.yavs_agn2 = np.array(yavs_agn2)
        self.av_avgs_agn = np.array(av_avgs_agn)
        self.av_avgs_agn2 = np.array(av_avgs_agn)
        
        X = np.vstack([self.allnonagn_hb_sn_filts_df[2].sfr, self.allnonagn_hb_sn_filts_df[2].av_gsw]).transpose()
        X_agn = np.vstack([self.agn_avgsw_filts_df[2].sfr, self.agn_avgsw_filts_df[2].av_gsw]).transpose()
        
        self.X_reg = X
        self.X_reg_agn = X_agn
        y = np.array(np.transpose(self.allnonagn_hb_sn_filts_df[2].av))
        y_agn = np.array(np.transpose(self.agn_avgsw_filts_df[2].av))
        y_agn2 = np.array(np.transpose(self.agn_avgsw_filts_df[2].av_agn))

        reg = LinearRegression().fit(X,y)
        reg_agn = LinearRegression().fit(X_agn,y_agn)
        reg_agn2 = LinearRegression().fit(X_agn,y_agn2)

        ypreds = []
        for i in range(len(xmids_gsw)):
            X_test = np.vstack([xmids_gsw[i],av_avgs[i]]).transpose()
            ypred = reg.predict(X_test)
            ypreds.append(ypred)
        self.sfr_av_reg = reg
        self.sfr_av_ypreds = ypreds
        
        ypreds_agn = []
        for i in range(len(xmids_agn)):
            X_test = np.vstack([xmids_agn[i],av_avgs_agn[i]]).transpose()
            ypred = reg_agn.predict(X_test)
            ypreds_agn.append(ypred)
        self.sfr_av_reg_agn = reg_agn
        self.sfr_av_ypreds_agn = ypreds_agn
        
        ypreds_agn2 = []
        for i in range(len(xmids_agn2)):
            X_test = np.vstack([xmids_agn2[i],av_avgs_agn2[i]]).transpose()
            ypred = reg_agn2.predict(X_test)
            ypreds_agn2.append(ypred)
        self.sfr_av_reg_agn2 = reg_agn2
        self.sfr_av_ypreds_agn2 = ypreds_agn2




    def correct_av(self,reg, sfr, av_gsw, av_balm, hb_sn):
        x_test = np.vstack([sfr, av_gsw])
        av_balm_fixed = []
        for i in range(len(hb_sn)):

            if hb_sn[i] <5:
                x_samp = np.array([x_test[:,i]])
                av_fix = np.float64(reg.predict(x_samp))
                if av_fix<0:
                    av_fix=0
            elif av_balm[i] <0:
                av_fix = 0
            else:
                av_fix = av_balm[i]
            av_balm_fixed.append(av_fix)
        return np.array(av_balm_fixed)
'''



plt.plot(xmids_gsw[0], yavs_gsw[0], 'k',linestyle='--', label='0$<$A$_{\mathrm{V,GSW}}<$0.2, non-AGN')
plt.plot(xmids_gsw[1], yavs_gsw[1], 'b',linestyle='--', label='0.2$<$A$_{\mathrm{V,GSW}}<$0.4, non-AGN')
plt.plot(xmids_gsw[2], yavs_gsw[2], 'r',linestyle='--', label='0.4$<$A$_{\mathrm{V,GSW}}<$0.6, non-AGN')
plt.plot(xmids_gsw[3], yavs_gsw[3], 'g',linestyle='--', label=r'0.6$<$A$_{\mathrm{V,GSW}}<$0.8, non-AGN')
plt.plot(xmids_gsw[4], yavs_gsw[4], 'c',linestyle='--', label='0.8$<$A$_{\mathrm{V,GSW}}<$1.0, non-AGN')

plt.plot(xmids_gsw[0], ypreds[0], 'k', label=r'0$<$A$_{\mathrm{V,GSW}}<$0.2,pred')
plt.plot(xmids_gsw[1], ypreds[1], 'b', label=r'0.2$<$A$_{\mathrm{V,GSW}}<$0.4,pred')
plt.plot(xmids_gsw[2], ypreds[2], 'r', label=r'0.4$<$A$_{\mathrm{V,GSW}}<$0.6,pred')
plt.plot(xmids_gsw[3], ypreds[3], 'g', label=r'0.6$<$A$_{\mathrm{V,GSW}}<$0.8,pred')
plt.plot(xmids_gsw[4], ypreds[4], 'c', label=r'0.8$<$A$_{\mathrm{V,GSW}}<$1,pred')
plt.legend()
plt.xlabel('log(SFR)')
plt.ylabel(r'A$_{\mathrm{V, Balmer}}$')

plt.ylim([-1,2.5])
plt.xlim([-2,2])




lims =[0,0.1,0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for i in range(len(lims)-1):
    EL_m2.bin_by_dust(EL_m2.allnonagn_avgsw_filts_df[i],'sfr',-3.5, 3.5)
    plot2dhist(EL_m2.allnonagn_avgsw_filts_df[i]['sfr'], EL_m2.allnonagn_avgsw_filts_df[i]['av'], nan=True, maxx=3.5, minx=-3.5, miny=-5, maxy=5., lim=True, xlabel='log(SFR)', ylabel=r'A$_{\mathrm{V}}$', setplotlims=True)
    plt.tight_layout()
    plt.savefig('./plots/sfrmatch/pdf/diagnostic/Av_sfr_allnonagn_av_bin_'+str(lims[i])+'_'+str(lims[i+1])+'.pdf')
    plt.close()
    
    plot2dhist(EL_m2.allnonagn_avgsw_filts_df[i]['sfr'], EL_m2.allnonagn_avgsw_filts_df[i]['av'], nan=True, maxx=3.5, minx=-3.5, miny=-5, maxy=5., lim=True, xlabel='log(SFR)', ylabel=r'A$_{\mathrm{V}}$', setplotlims=True)
    plt.plot(EL_m2.bncenters,EL_m2.ext_binned_sums, label='Sums')
    plt.plot(EL_m2.bncenters,EL_m2.ext_binned_medians,  label='Medians')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/sfrmatch/pdf/diagnostic/Av_sfr_allnonagn_av_bin_'+str(lims[i])+'_'+str(lims[i+1])+'_binned.pdf')
    plt.close()
    
    plot2dhist(EL_m2.allnonagn_avgsw_filts_df[i]['sfr'], EL_m2.allnonagn_avgsw_filts_df[i]['av'], nan=True, maxx=3.5, minx=-3.5, miny=-5, maxy=5., lim=True, xlabel='log(SFR)', ylabel=r'A$_{\mathrm{V}}$', setplotlims=True)
    plt.errorbar(EL_m2.bncenters[EL_m2.valbns],EL_m2.bootstr_ext_sums, yerr = EL_m2.bootstr_ext_sums_err, label='Bootstrapped Sums',capsize=2)
    plt.errorbar(EL_m2.bncenters[EL_m2.valbns],EL_m2.bootstr_ext_meds, yerr = EL_m2.bootstr_ext_meds_err, label='Bootstrapped Medians', capsize=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/sfrmatch/pdf/diagnostic/Av_sfr_allnonagn_av_bin_'+str(lims[i])+'_'+str(lims[i+1])+'_binned_bootstrapped.pdf')
    plt.close()
    
xmids = []
yavs = []
for i in range(len(EL_m2.allnonagn_avgsw_filts_df)):
    xmid, yav = plot2dhist(EL_m2.allnonagn_avgsw_filts_df[i]['sfr'], EL_m2.allnonagn_avgsw_filts_df[i]['av'], nan=True, maxx=3.5, minx=-3.5, miny=-5, maxy=5., lim=True, xlabel='log(SFR)', ylabel=r'A$_{\mathrm{V}}$', setplotlims=True, bin_y=True)
    plt.close()
    


plt.tight_layout()
plt.savefig('./plots/sfrmatch/pdf/diagnostic/Av_sfr_allnonagn_av_bin_'+str(lims[i])+'_'+str(lims[i+1])+'.pdf')
plt.close()
        
    xmid, avg_y
    
sns =[2,5,10,20]
for i in range(len(sns)-1):
    EL_m2.bin_by_dust(EL_m2.allnonagn_hb_ha_sn_filts_df[i],'sfr',-3.5, 3.5)
    plot2dhist(EL_m2.allnonagn_hb_ha_sn_filts_df[i]['sfr'], EL_m2.allnonagn_hb_ha_sn_filts_df[i]['av'], nan=True, maxx=3.5, minx=-3.5, miny=-5, maxy=5., lim=True, xlabel='log(SFR)', ylabel=r'A$_{\mathrm{V}}$', setplotlims=True)
    plt.tight_layout()
    plt.savefig('./plots/sfrmatch/pdf/diagnostic/Av_sfr_allnonagn_hbeta_halp_sn_'+str(lims[i])+'.pdf')
    plt.close()
    
    plot2dhist(EL_m2.allnonagn_hb_ha_sn_filts_df[i]['sfr'], EL_m2.allnonagn_hb_ha_sn_filts_df[i]['av'], nan=True, maxx=3.5, minx=-3.5, miny=-5, maxy=5., lim=True, xlabel='log(SFR)', ylabel=r'A$_{\mathrm{V}}$', setplotlims=True)
    plt.plot(EL_m2.bncenters,EL_m2.ext_binned_sums, label='Sums')
    plt.plot(EL_m2.bncenters,EL_m2.ext_binned_medians,  label='Medians')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/sfrmatch/pdf/diagnostic/Av_sfr_allnonagn_hbeta_halp_sn_'+str(lims[i])+'_binned.pdf')
    plt.close()
    
    plot2dhist(EL_m2.allnonagn_hb_ha_sn_filts_df[i]['sfr'], EL_m2.allnonagn_hb_ha_sn_filts_df[i]['av'], nan=True, maxx=3.5, minx=-3.5, miny=-5, maxy=5., lim=True, xlabel='log(SFR)', ylabel=r'A$_{\mathrm{V}}$', setplotlims=True)
    plt.errorbar(EL_m2.bncenters[EL_m2.valbns],EL_m2.bootstr_ext_sums, yerr = EL_m2.bootstr_ext_sums_err, label='Bootstrapped Sums',capsize=2)
    plt.errorbar(EL_m2.bncenters[EL_m2.valbns],EL_m2.bootstr_ext_meds, yerr = EL_m2.bootstr_ext_meds_err, label='Bootstrapped Medians', capsize=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/sfrmatch/pdf/diagnostic/Av_sfr_allnonagn_hbeta_halp_sn_'+str(lims[i])+'_binned_bootstrapped.pdf')
    plt.close()
    
        

'''