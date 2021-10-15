import numpy as np
from ast_func import *
catfold='catalogs/'
import astropy.cosmology as apc
from loaddata_m2 import redshift_m2
from loaddata_sdss_xr import sdssobj
cosmo = apc.Planck15
from setops import *
import time
from demarcations import *
import pandas as pd
from ELObj import *
from sklearn.linear_model import LinearRegression

#kmnclust = np.loadtxt('catalogs/kmeans_sfrm.dat', unpack=True)
#kmnclust1 =  kmnclust

#val1 = np.where(kmnclust1 != 0 )[0]
#sy2_1 = np.where(kmnclust1==1)[0]
#liner2_1 = np.where(kmnclust1==2)[0]
#sf_1 = np.where(kmnclust1==3)[0]

#kmnclust2 =  kmnclust[1]
'''
val2 = np.where(kmnclust2 != 0 )[0]
sy2_1 = np.where(kmnclust2==1)[0]
liner2_1 = np.where(kmnclust2==2)[0]
sf_1 = np.where(kmnclust2==3)[0]
    '''    
class SFRMatch:
    def __init__(self, eldiag, bpt_eldiag, plus_eldiag, neither_eldiag):
        self.eldiag=eldiag
        self.bpt_eldiag=bpt_eldiag
        self.plus_eldiag=plus_eldiag
        self.neither_eldiag=neither_eldiag
    def get_dists(self, quant1, quant1_match):
        return (quant1-quant1_match)**2
    
    def get_highsn_match_only(self, agn_inds, sf_inds, sf_plus_inds, 
                              agnplus_inds, sncut_balm=2, sncut_forb=2, load=False, fname='', with_av=True,
                              balmdecmin =-99, subfold='', second=True, old=False):
        '''
        Matches BPT AGN to BPT SF, BPT+ SF, unclassified of similar SFR, M*, Mfib, z
        Ensures that the match has a high S/N in subtracted fluxes
        '''
        self.bptsf_inds = sf_inds
        self.bptagn_inds = agn_inds
        self.bptplus_sf_inds = sf_plus_inds
        self.bptplus_agn_inds = agnplus_inds 
        self.fname=fname
        sncut = np.min([sncut_balm, sncut_forb])
        if subfold=='':
            if with_av:
                subfold = 'matching_with_av/'
            else:
                subfold = 'matching_without_av/'        

        if not load:
            #setting up lists/arrays for storing info
            agns_selfmatch = []
            agns_selfmatch_other = []
            
            agnsplus_selfmatch = []
            agnsplus_selfmatch_other = []
            
            #lists for storing agn indices
            agns = []
            agns_plus = []
            neither_agn = []
            
            second_agns = []
            second_agns_plus = []
            second_neither_agn = []
            #lists for storing match indices         
            sfs = []
            sfs_plus = []
            neither_matches = []
            
            second_sfs = []
            second_sfs_plus = []
            second_neither_matches = []
            
            numpassed = np.zeros((3, len(agn_inds)))
            numpassed_best = []
            mininds = np.zeros((3, len(agn_inds)))
            second_mininds = np.zeros((3, len(agn_inds)))

            mindists = np.zeros((3, len(agn_inds)))
            second_mindists = np.zeros((3, len(agn_inds)))

            minids = np.zeros((3, len(agn_inds)), dtype=np.int64)
            mininds_agn = np.zeros((2, len(agn_inds)))

            mindists_agn = np.zeros((2, len(agn_inds)))
            minids_agn = np.zeros((2, len(agn_inds)), dtype=np.int64)
            
            for i, agn_ind in enumerate(agn_inds):
                if i%100 == 0:
                    print(i)
                    t = time.localtime()
                    current_time = time.strftime("%H:%M:%S", t)
                    print(current_time)
                #for self-matching, don't want to compare to itself
                otheragns = np.where(agn_inds != agn_ind)[0]
                '''
                BPT AGN self-matching
                '''
                #computing differences for self matching agn
                diffsfr_agn = (self.bpt_eldiag.sfr.iloc[agn_ind]-self.bpt_eldiag.sfr.iloc[agn_inds[otheragns]])**2
                diffmass_agn = (self.bpt_eldiag.mass.iloc[agn_ind]-self.bpt_eldiag.mass.iloc[agn_inds[otheragns]])**2
                difffibmass_agn = (self.bpt_eldiag.fibmass.iloc[agn_ind]-self.bpt_eldiag.fibmass.iloc[agn_inds[otheragns]])**2
                diffz_agn = (self.bpt_eldiag.z.iloc[agn_ind]-self.bpt_eldiag.z.iloc[agn_inds[otheragns]])**2/np.var(redshift_m2)
                diff_av_agn = (self.bpt_eldiag.av_gsw.iloc[agn_ind]-self.bpt_eldiag.av_gsw.iloc[agn_inds[otheragns]])**2
                diffs_agn = np.array(np.sqrt(diffsfr_agn+diffmass_agn+difffibmass_agn+diffz_agn+diff_av_agn))
                
                mindiff_ind_agn = np.where(diffs_agn == np.min(diffs_agn) )[0]
                '''
                BPT Plus self-agn matching
                '''
                #computing differences for self matching agn to bpt+ agn
                diffsfr_agnplus = (self.bpt_eldiag.sfr.iloc[agn_ind]-self.plus_eldiag.sfr.iloc[agnplus_inds])**2
                diffmass_agnplus = (self.bpt_eldiag.mass.iloc[agn_ind]-self.plus_eldiag.mass.iloc[agnplus_inds])**2
                difffibmass_agnplus = (self.bpt_eldiag.fibmass.iloc[agn_ind]-self.plus_eldiag.fibmass.iloc[agnplus_inds])**2
                diffz_agnplus = (self.bpt_eldiag.z.iloc[agn_ind]-self.plus_eldiag.z.iloc[agnplus_inds])**2/np.var(redshift_m2)
                diff_av_agnplus = (self.bpt_eldiag.av_gsw.iloc[agn_ind]-self.plus_eldiag.av_gsw.iloc[agnplus_inds])**2
                diffs_agnplus = np.array(np.sqrt(diffsfr_agnplus+diffmass_agnplus+difffibmass_agnplus+diffz_agnplus+diff_av_agnplus))
                mindiff_ind_agnplus = np.where(diffs_agnplus == np.min(diffs_agnplus) )[0]
                
                
                
                '''
                BPT AGN-BPT SF matching
                '''
                #computing differences for bpt SF 
                diffsfr_bpt = (self.bpt_eldiag.sfr.iloc[agn_ind] - self.bpt_eldiag.sfr.iloc[sf_inds])**2
                diffmass_bpt = (self.bpt_eldiag.mass.iloc[agn_ind] - self.bpt_eldiag.mass.iloc[sf_inds])**2
                difffibmass_bpt = (self.bpt_eldiag.fibmass.iloc[agn_ind] - self.bpt_eldiag.fibmass.iloc[sf_inds])**2
                diffz_bpt = (self.bpt_eldiag.z.iloc[agn_ind]-self.bpt_eldiag.z.iloc[sf_inds])**2/np.var(redshift_m2)
                diff_av_bpt = (self.bpt_eldiag.av_gsw.iloc[agn_ind]-self.bpt_eldiag.av_gsw.iloc[sf_inds])**2
                diffs_bpt = np.array(np.sqrt(diffsfr_bpt+diffmass_bpt+difffibmass_bpt+diffz_bpt+diff_av_bpt))
                #mindiff_ind_bpt = np.where(diffs_bpt == np.min(diffs_bpt) )[0]
                bptdistrat = (np.array(cosmo.luminosity_distance(self.bpt_eldiag.z.iloc[sf_inds]))/np.array(cosmo.luminosity_distance(self.bpt_eldiag.z.iloc[agn_ind])))**2 

                oiiiflux_sub_bpt = self.bpt_eldiag.oiiiflux.iloc[agn_ind]-self.bpt_eldiag.oiiiflux.iloc[sf_inds]*bptdistrat
                niiflux_sub_bpt = self.bpt_eldiag.niiflux.iloc[agn_ind]-self.bpt_eldiag.niiflux.iloc[sf_inds]*bptdistrat
                hbetaflux_sub_bpt =self.bpt_eldiag.hbetaflux.iloc[agn_ind]-self.bpt_eldiag.hbetaflux.iloc[sf_inds]*bptdistrat
                halpflux_sub_bpt  = self.bpt_eldiag.halpflux.iloc[agn_ind]- self.bpt_eldiag.halpflux.iloc[sf_inds]*bptdistrat
                
                oiiiflux_err_sub_bpt = np.sqrt(self.bpt_eldiag.oiii_err.iloc[agn_ind]**2 + (self.bpt_eldiag.oiii_err.iloc[sf_inds]*bptdistrat)**2)
                niiflux_err_sub_bpt =  np.sqrt(self.bpt_eldiag.nii_err.iloc[agn_ind]**2 + (self.bpt_eldiag.nii_err.iloc[sf_inds]*bptdistrat)**2)
                hbetaflux_err_sub_bpt = np.sqrt(self.bpt_eldiag.hbeta_err.iloc[agn_ind]**2 + (self.bpt_eldiag.hbeta_err.iloc[sf_inds]*bptdistrat)**2)
                halpflux_err_sub_bpt  = np.sqrt(self.bpt_eldiag.halp_err.iloc[agn_ind]**2 + (self.bpt_eldiag.halp_err.iloc[sf_inds]*bptdistrat)**2)
                
                #postsub_av_bpt = extinction(ha=halpflux_sub_bpt, hb=hbetaflux_sub_bpt, agn=True)
                
                oiiiflux_sn_sub_bpt = oiiiflux_sub_bpt/ oiiiflux_err_sub_bpt
                niiflux_sn_sub_bpt = niiflux_sub_bpt/niiflux_err_sub_bpt
                hbetaflux_sn_sub_bpt = hbetaflux_sub_bpt/hbetaflux_err_sub_bpt
                halpflux_sn_sub_bpt = halpflux_sub_bpt/halpflux_err_sub_bpt
                diffs_bpt_sort = np.argsort(diffs_bpt)    
                
                
                inds_high_sn_bpt = np.where((oiiiflux_sn_sub_bpt.iloc[diffs_bpt_sort]>sncut_forb) & 
                                            (niiflux_sn_sub_bpt.iloc[diffs_bpt_sort]>sncut_forb) &
                                            (hbetaflux_sn_sub_bpt.iloc[diffs_bpt_sort]>sncut_balm) &
                                            (halpflux_sn_sub_bpt.iloc[diffs_bpt_sort] >sncut_balm))[0]# &
                                            #(postsub_av_bpt[diffs_bpt_sort]>balmdecmin)
                
                #print(inds_high_sn_bpt)
                #print(inds_high_sn_bpt)
                #print(diffs_bpt_sort)
                if len(inds_high_sn_bpt) >0:    
                    mindiff_ind_bpt = diffs_bpt_sort[inds_high_sn_bpt[0]]
                    if len(inds_high_sn_bpt)>1:
                        second_best_ind_bpt = diffs_bpt_sort[inds_high_sn_bpt[1]]
                    else:
                        second_best_ind_bpt=mindiff_ind_bpt+1
                    n_pass_bpt = len(inds_high_sn_bpt)
                else:
                    mindiff_ind_bpt = -1
                    second_best_ind_bpt = -2
                    n_pass_bpt = len(diffs_bpt_sort)
                #computing differences for bpt+ SF 
                '''
                BPT AGN-BPT PLUS SF matching
                '''
                diffsfr_bptplus = (self.bpt_eldiag.sfr.iloc[agn_ind] - self.plus_eldiag.sfr.iloc[sf_plus_inds])**2
                diffmass_bptplus = (self.bpt_eldiag.mass.iloc[agn_ind] - self.plus_eldiag.mass.iloc[sf_plus_inds])**2
                difffibmass_bptplus = (self.bpt_eldiag.fibmass.iloc[agn_ind] - self.plus_eldiag.fibmass.iloc[sf_plus_inds])**2
                diffz_bptplus = (self.bpt_eldiag.z.iloc[agn_ind]-self.plus_eldiag.z.iloc[sf_plus_inds])**2/np.var(redshift_m2)
                diff_av_bptplus = (self.bpt_eldiag.av_gsw.iloc[agn_ind]-self.plus_eldiag.av_gsw.iloc[sf_plus_inds])**2
                diffs_bptplus = np.array(np.sqrt(diffsfr_bptplus+diffmass_bptplus+difffibmass_bptplus+diffz_bptplus+diff_av_bptplus))

                bptdistrat_plus = (np.array(cosmo.luminosity_distance(self.plus_eldiag.z.iloc[sf_plus_inds]))/np.array(cosmo.luminosity_distance(self.bpt_eldiag.z.iloc[agn_ind])))**2 

                oiiiflux_sub_plus = self.bpt_eldiag.oiiiflux.iloc[agn_ind]-self.plus_eldiag.oiiiflux.iloc[sf_plus_inds]*bptdistrat_plus
                niiflux_sub_plus = self.bpt_eldiag.niiflux.iloc[agn_ind]-self.plus_eldiag.niiflux.iloc[sf_plus_inds]*bptdistrat_plus
                hbetaflux_sub_plus = self.bpt_eldiag.hbetaflux.iloc[agn_ind]-self.plus_eldiag.hbetaflux.iloc[sf_plus_inds]*bptdistrat_plus
                halpflux_sub_plus  = self.bpt_eldiag.halpflux.iloc[agn_ind]- self.plus_eldiag.halpflux.iloc[sf_plus_inds]*bptdistrat_plus
                
                oiiiflux_err_sub_plus = np.sqrt(self.bpt_eldiag.oiii_err.iloc[agn_ind]**2 + (self.plus_eldiag.oiii_err.iloc[sf_plus_inds]*bptdistrat_plus)**2)
                niiflux_err_sub_plus =  np.sqrt(self.bpt_eldiag.nii_err.iloc[agn_ind]**2 + (self.plus_eldiag.nii_err.iloc[sf_plus_inds]*bptdistrat_plus)**2)
                hbetaflux_err_sub_plus = np.sqrt(self.bpt_eldiag.hbeta_err.iloc[agn_ind]**2 + (self.plus_eldiag.hbeta_err.iloc[sf_plus_inds]*bptdistrat_plus)**2)
                halpflux_err_sub_plus  = np.sqrt(self.bpt_eldiag.halp_err.iloc[agn_ind]**2 + (self.plus_eldiag.halp_err.iloc[sf_plus_inds]*bptdistrat_plus)**2)
                
                #postsub_av_plus = extinction(ha=halpflux_sub_plus, hb=hbetaflux_sub_plus, agn=True)
                
                oiiiflux_sn_sub_plus = oiiiflux_sub_plus/ oiiiflux_err_sub_plus
                niiflux_sn_sub_plus = niiflux_sub_plus/niiflux_err_sub_plus
                hbetaflux_sn_sub_plus = hbetaflux_sub_plus/hbetaflux_err_sub_plus
                halpflux_sn_sub_plus = halpflux_sub_plus/halpflux_err_sub_plus
                
                
  
                diffs_bptplus_sort = np.argsort(diffs_bptplus)    
                
                inds_high_sn_bptplus = np.where((oiiiflux_sn_sub_plus.iloc[diffs_bptplus_sort]>sncut_forb) & 
                                                (niiflux_sn_sub_plus.iloc[diffs_bptplus_sort]>sncut_forb) &
                                                (hbetaflux_sn_sub_plus.iloc[diffs_bptplus_sort]>sncut_balm) &
                                                (halpflux_sn_sub_plus.iloc[diffs_bptplus_sort] >sncut_balm)&
                                                (self.plus_eldiag.oiii_err.iloc[sf_plus_inds[diffs_bptplus_sort]]*bptdistrat_plus[diffs_bptplus_sort] != 0 ) &
                                                (self.plus_eldiag.hbeta_err.iloc[sf_plus_inds[diffs_bptplus_sort]]*bptdistrat_plus[diffs_bptplus_sort] != 0 ) )[0]
                                                #(postsub_av_plus[diffs_bptplus_sort]>balmdecmin)
                #print(inds_high_sn_bptplus)

                if len(inds_high_sn_bptplus) >0:    
                    mindiff_ind_bptplus = diffs_bptplus_sort[inds_high_sn_bptplus[0]]
                    if len(inds_high_sn_bptplus)>1:
                        
                        second_best_ind_bptplus = diffs_bptplus_sort[inds_high_sn_bptplus[1]]
                    else:
                        second_best_ind_bptplus = mindiff_ind_bptplus+1
                    n_pass_plus = len(inds_high_sn_bptplus)
                else:
                    mindiff_ind_bptplus = -1
                    second_best_ind_bptplus = -2
                    
                    n_pass_plus = len(diffs_bptplus_sort)
                '''
                BPT AGN-Neither match
                '''
                #computing differences for unclassifiable 
                diffsfr_neither = (self.bpt_eldiag.sfr.iloc[agn_ind] - self.neither_eldiag.sfr)**2
                diffmass_neither = (self.bpt_eldiag.mass.iloc[agn_ind] - self.neither_eldiag.mass)**2
                difffibmass_neither = (self.bpt_eldiag.fibmass.iloc[agn_ind] - self.neither_eldiag.fibmass)**2
                diffz_neither = (self.bpt_eldiag.z.iloc[agn_ind]-self.neither_eldiag.z)**2/np.var(redshift_m2)
                diff_av_agn_neither = (self.bpt_eldiag.av_gsw.iloc[agn_ind]-self.neither_eldiag.av_gsw)**2
                diffs_neither = np.array(np.sqrt(diffsfr_neither+diffmass_neither+difffibmass_neither+diffz_neither+diff_av_agn_neither))

                bptdistrat_neither = (np.array(cosmo.luminosity_distance(self.neither_eldiag.z))/np.array(cosmo.luminosity_distance(self.bpt_eldiag.z.iloc[agn_ind])))**2 
                        

                oiiiflux_sub_neither = self.bpt_eldiag.oiiiflux.iloc[agn_ind]-self.neither_eldiag.oiiiflux*bptdistrat_neither
                niiflux_sub_neither = self.bpt_eldiag.niiflux.iloc[agn_ind]-self.neither_eldiag.niiflux*bptdistrat_neither
                hbetaflux_sub_neither =self.bpt_eldiag.hbetaflux.iloc[agn_ind]-self.neither_eldiag.hbetaflux*bptdistrat_neither
                halpflux_sub_neither  = self.bpt_eldiag.halpflux.iloc[agn_ind]- self.neither_eldiag.halpflux*bptdistrat_neither
                
                #postsub_av_neither = extinction(ha=halpflux_sub_neither, hb=hbetaflux_sub_neither, agn=True)

                oiiiflux_err_sub_neither = np.sqrt(self.bpt_eldiag.oiii_err.iloc[agn_ind]**2 + (self.neither_eldiag.oiii_err*bptdistrat_neither)**2)
                niiflux_err_sub_neither =  np.sqrt(self.bpt_eldiag.nii_err.iloc[agn_ind]**2 + (self.neither_eldiag.nii_err*bptdistrat_neither)**2)
                hbetaflux_err_sub_neither = np.sqrt(self.bpt_eldiag.hbeta_err.iloc[agn_ind]**2 + (self.neither_eldiag.hbeta_err*bptdistrat_neither)**2)
                halpflux_err_sub_neither  = np.sqrt(self.bpt_eldiag.halp_err.iloc[agn_ind]**2 + (self.neither_eldiag.halp_err*bptdistrat_neither)**2)
                
                oiiiflux_sn_sub_neither = oiiiflux_sub_neither/ oiiiflux_err_sub_neither
                niiflux_sn_sub_neither = niiflux_sub_neither/niiflux_err_sub_neither
                hbetaflux_sn_sub_neither = hbetaflux_sub_neither/hbetaflux_err_sub_neither
                halpflux_sn_sub_neither = halpflux_sub_neither/halpflux_err_sub_neither
  
                diffs_neither_sort = np.argsort(diffs_neither)    
  
                inds_high_sn_neither = np.where((oiiiflux_sn_sub_neither.iloc[diffs_neither_sort]>sncut_forb) & 
                                                (niiflux_sn_sub_neither.iloc[diffs_neither_sort]>sncut_forb) &
                                                (hbetaflux_sn_sub_neither.iloc[diffs_neither_sort]>sncut_balm) &
                                                (halpflux_sn_sub_neither.iloc[diffs_neither_sort] >sncut_balm) &
                                                (self.neither_eldiag.oiii_err.iloc[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) &
                                                (self.neither_eldiag.hbeta_err.iloc[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) &
                                                (self.neither_eldiag.halp_err.iloc[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) &
                                                (self.neither_eldiag.nii_err.iloc[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ))[0]# &
                                                #(postsub_av_neither[diffs_neither_sort]>balmdecmin)
                if len(inds_high_sn_neither)>0:
                    mindiff_ind_neither = diffs_neither_sort[inds_high_sn_neither[0]]
                    if len(inds_high_sn_neither)>1:
                        second_best_ind_neither = diffs_neither_sort[inds_high_sn_neither[1]]
                    else:
                        second_best_ind_neither = mindiff_ind_neither + 1
                    n_pass_neither = len(inds_high_sn_neither)
                else:
                    mindiff_ind_neither = -1
                    second_best_ind_neither = -2
                    n_pass_neither = len(diffs_neither_sort)
                #print(inds_high_sn_neitherni)                            
                mindiffinds = [mindiff_ind_bpt, mindiff_ind_bptplus, mindiff_ind_neither]
                second_mindiffinds = [second_best_ind_bpt, second_best_ind_bptplus, second_best_ind_neither]
                
                
                mindist_out = [diffs_bpt[mindiff_ind_bpt], diffs_bptplus[mindiff_ind_bptplus],  
                               diffs_neither[mindiff_ind_neither]]
                secondmindist_out = [diffs_bpt[second_best_ind_bpt], diffs_bptplus[second_best_ind_bptplus],  
                               diffs_neither[second_best_ind_neither]]

                bad_dists = np.where(mindiffinds == -1)[0]
                bad_dists2 = np.where(mindiffinds == -2)[0]

                if bad_dists.size >0:
                    mindist_out[bad_dists]=99999
                if bad_dists2.size >0:
                    secondmindist_out[bad_dists2]=99999
                                        
                #print(mindist_out)
                n_pass_out = [n_pass_bpt, n_pass_plus, n_pass_neither]
                 
                #assigning the ids, inds, dists to be saved
                minid_out =[ self.bpt_eldiag.ids.iloc[sf_inds[mindiff_ind_bpt]],  
                            self.plus_eldiag.ids.iloc[sf_plus_inds[mindiff_ind_bptplus]],
                            self.neither_eldiag.ids.iloc[mindiff_ind_neither]]
                minind_out =[ sf_inds[mindiff_ind_bpt],  
                            sf_plus_inds[mindiff_ind_bptplus],
                            mindiff_ind_neither]
                secondminind_out =[ sf_inds[second_best_ind_bpt],  
                            sf_plus_inds[second_best_ind_bptplus],
                            second_best_ind_neither]

                #assigning the ids, inds, dists to be saved for self-matching

                minid_outagn = [self.bpt_eldiag.ids.iloc[agn_inds[mindiff_ind_agn]],  
                            self.plus_eldiag.ids.iloc[agnplus_inds[mindiff_ind_agnplus]]]
                minind_outagn =[ agn_inds[mindiff_ind_agn],  
                            agnplus_inds[mindiff_ind_bptplus]]
                mindist_outagn = [diffs_agn[mindiff_ind_agn], diffs_agnplus[mindiff_ind_agnplus]]
                numpassed[:, i] = n_pass_out
                
                #saving the relevant info 
                mindists[:, i] = mindist_out
                minids[:, i] = minid_out
                mininds[:, i] = minind_out
                
                second_mininds[:,i] = secondminind_out
                second_mindists[:, i] = secondmindist_out
    
                mindists_agn[:, i] = mindist_outagn
                minids_agn[:, i] = minid_outagn
                mininds_agn[:, i] = minind_outagn

                mindist_ind = np.int(np.where(mindist_out == np.min(mindist_out))[0])

                #getting the best one, 0 = BPT SF, 1 = BPT SF+, 2 = Unclassifiable
                if mindist_ind ==0:
                    sfs.append(sf_inds[mindiff_ind_bpt])
                    agns.append(agn_ind)
                    numpassed_best.append(n_pass_bpt)
                    
                elif mindist_ind==1:
                    sfs_plus.append(sf_plus_inds[mindiff_ind_bptplus])
                    agns_plus.append(agn_ind)
                    numpassed_best.append(n_pass_plus)
                    
                else:
                    neither_matches.append(mindiff_ind_neither)
                    neither_agn.append(agn_ind)
                    numpassed_best.append(n_pass_neither)

                comb_first_sec = np.hstack([mindist_out, secondmindist_out])
                sort_comb_first_sec = np.argsort(comb_first_sec)
                sec_best_ind = np.where(comb_first_sec == comb_first_sec[sort_comb_first_sec[1]])[0]
                
                if sec_best_ind ==0:
                    second_sfs.append(sf_inds[mindiff_ind_bpt])
                    second_agns.append(agn_ind)
                elif sec_best_ind==1:
                    second_sfs_plus.append(sf_plus_inds[mindiff_ind_bptplus])
                    second_agns_plus.append(agn_ind)        
                elif sec_best_ind==2:
                    second_neither_matches.append(mindiff_ind_neither)
                    second_neither_agn.append(agn_ind)
                elif sec_best_ind ==3:
                    second_sfs.append(sf_inds[second_best_ind_bpt])
                    second_agns.append(agn_ind)
                elif sec_best_ind==4:
                    second_sfs_plus.append(sf_plus_inds[second_best_ind_bptplus])
                    second_agns_plus.append(agn_ind)        
                elif sec_best_ind==5:
                    second_neither_matches.append(second_best_ind_neither)
                    second_neither_agn.append(agn_ind)


                                        
                mindistagn_ind = np.int(np.where(mindist_outagn==np.min(mindist_outagn))[0])
                if mindistagn_ind ==0:
                    agns_selfmatch.append(agn_ind)
                    agns_selfmatch_other.append(mindiff_ind_agn[0])
                else:
                    agnsplus_selfmatch.append(agn_ind)
                    agnsplus_selfmatch_other.append(agnplus_inds[mindiff_ind_agnplus[0]])  
                            

            #converting lists to arrays and saving to class attributes
            self.numpassed = numpassed
            self.numpassed_best = np.array(numpassed_best)
            self.agns = np.array(agns)
            self.sfs = np.array(sfs)
            self.agns_plus = np.array(agns_plus)
            self.sfs_plus = np.array(sfs_plus)
            self.neither_matches = np.array(neither_matches)
            self.neither_agn = np.array(neither_agn)
            
            self.second_agns=np.array(second_agns)
            self.second_sfs = np.array(second_sfs)
            self.second_agns_plus = np.array(second_agns_plus)
            self.second_sfs_plus = np.array(second_sfs_plus)
            self.second_neither_matches = np.array(second_neither_matches)
            self.second_neither_agn = np.array(second_neither_agn)


            self.mindists = mindists
            self.mininds= mininds
            
            self.second_mininds = second_mininds
            
            self.minids = minids
            self.mindists_best = np.min(self.mindists, axis=0)
            self.second_mindists = np.array(second_mindists)
            self.second_mindists_best = np.min(self.second_mindists, axis=0)
    
            self.agns_selfmatch = np.array(agns_selfmatch)
            self.agns_selfmatch_other = np.array(agns_selfmatch_other)
            self.agnsplus_selfmatch = np.array(agnsplus_selfmatch)
            self.agnsplus_selfmatch_other = np.array(agnsplus_selfmatch_other)
            self.mindists_agn = mindists_agn
            self.minids_agn = minids_agn
            self.mininds_agn = mininds_agn
            self.mindistsagn_best = np.min(self.mindists_agn, axis=0)
            
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_n_pass_best_highsn'+str(sncut)+fname+'.txt',self.numpassed_best, fmt='%6.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_n_pass_highsn'+str(sncut)+fname+'.txt',self.numpassed.transpose(), fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_agns_selfmatch_highsn'+str(sncut)+fname+'.txt',self.agns_selfmatch, fmt='%6.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_agns_selfmatch_other_highsn'+str(sncut)+fname+'.txt',self.agns_selfmatch_other, fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_agnsplus_selfmatch_highsn'+str(sncut)+fname+'.txt',self.agnsplus_selfmatch, fmt='%6.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_agnsplus_selfmatch_other_highsn'+str(sncut)+fname+'.txt',self.agnsplus_selfmatch_other, fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_agns_highsn'+str(sncut)+fname+'.txt',self.agns, fmt='%6.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_sfs_highsn'+str(sncut)+fname+'.txt',self.sfs, fmt='%6.d')


            
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_agns_plus_highsn'+str(sncut)+fname+'.txt',self.agns_plus, fmt='%6.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_sfs_plus_highsn'+str(sncut)+fname+'.txt',self.sfs_plus, fmt='%6.d')

            
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_neither_matches_highsn'+str(sncut)+fname+'.txt',self.neither_matches, fmt='%6.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_neither_agn_highsn'+str(sncut)+fname+'.txt',self.neither_agn, fmt='%6.d')


            
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_mindists_best_highsn'+str(sncut)+fname+'.txt',self.mindists_best, fmt='%8.6f')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_second_mindists_best_highsn'+str(sncut)+fname+'.txt',self.second_mindists_best, fmt='%8.6f')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_second_mindists_highsn'+str(sncut)+fname+'.txt',self.second_mindists.transpose(), fmt='%8.6f')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_second_mininds_highsn'+str(sncut)+fname+'.txt',self.second_mininds.transpose(), fmt='%6.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_second_neither_matches_highsn'+str(sncut)+fname+'.txt',self.second_neither_matches, fmt='%6.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_second_neither_agn_highsn'+str(sncut)+fname+'.txt',self.second_neither_agn, fmt='%6.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_second_agns_plus_highsn'+str(sncut)+fname+'.txt',self.second_agns_plus, fmt='%6.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_second_sfs_plus_highsn'+str(sncut)+fname+'.txt',self.second_sfs_plus, fmt='%6.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_second_agns_highsn'+str(sncut)+fname+'.txt',self.second_agns, fmt='%6.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_second_sfs_highsn'+str(sncut)+fname+'.txt',self.second_sfs, fmt='%6.d')


            
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_mindists_highsn'+str(sncut)+fname+'.txt',self.mindists.transpose(), fmt='%8.6f')
            
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_mininds_highsn'+str(sncut)+fname+'.txt',self.mininds.transpose(), fmt='%6.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_minids_highsn'+str(sncut)+fname+'.txt',self.minids.transpose(), fmt='%18.d')
            
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_mindists_agn_highsn'+str(sncut)+fname+'.txt',self.mindists_agn.transpose(), fmt='%8.6f')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_mininds_agn_highsn'+str(sncut)+fname+'.txt',self.mininds_agn.transpose(), fmt='%6.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_minids_agn_highsn'+str(sncut)+fname+'.txt',self.minids_agn.transpose(), fmt='%18.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_mindistsagn_best_highsn'+str(sncut)+fname+'.txt',self.mindistsagn_best, fmt='%8.6f')
                
        else:
            #once the matching is already done just need to load items in
            if not old:
                self.numpassed =  np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_n_pass_highsn'+str(sncut)+fname+'.txt',dtype=np.int64, unpack=True)
                
                self.numpassed_best = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_n_pass_best_highsn'+str(sncut)+fname+'.txt',dtype=np.int64)
                
                self.agns_selfmatch = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_agns_selfmatch_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
                self.agns_selfmatch_other = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_agns_selfmatch_other_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
                
                self.agnsplus_selfmatch = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_agnsplus_selfmatch_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
                self.agnsplus_selfmatch_other = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_agnsplus_selfmatch_other_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
                
                self.agns = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_agns_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
                self.sfs = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_sfs_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
                
                self.agns_plus = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_agns_plus_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
                self.sfs_plus = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_sfs_plus_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
    
                self.neither_agn = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_neither_agn_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)            
                self.neither_matches = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_neither_matches_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
                
    
                
                self.mindists = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_mindists_highsn'+str(sncut)+fname+'.txt', unpack=True)
                self.mininds = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_mininds_highsn'+str(sncut)+fname+'.txt', dtype=np.int64, unpack=True)
    
                self.minids = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_minids_highsn'+str(sncut)+fname+'.txt', dtype=np.int64, unpack=True)
                self.mindists_best = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_mindists_best_highsn'+str(sncut)+fname+'.txt')
                if second:
                    self.second_agns = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_second_agns_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
                    self.second_sfs = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_second_sfs_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
                    
                    self.second_agns_plus = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_second_agns_plus_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
                    self.second_sfs_plus = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_second_sfs_plus_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
        
                    self.second_neither_agn = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_second_neither_agn_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)            
                    self.second_neither_matches = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_second_neither_matches_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
        
                    self.second_mindists = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_second_mindists_highsn'+str(sncut)+fname+'.txt', unpack=True)
                    self.second_mininds = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_second_mininds_highsn'+str(sncut)+fname+'.txt', dtype=np.int64, unpack=True)
        
                    self.second_mindists_best = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_second_mindists_best_highsn'+str(sncut)+fname+'.txt')
                
                self.mindists_agn = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_mindists_agn_highsn'+str(sncut)+fname+'.txt', unpack=True)
                self.minids_agn = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_minids_agn_highsn'+str(sncut)+fname+'.txt', dtype=np.int64, unpack=True)
                self.mininds_agn = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_mininds_agn_highsn'+str(sncut)+fname+'.txt', dtype=np.int64, unpack=True)
                self.mindistsagn_best = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_mindistsagn_best_highsn'+str(sncut)+fname+'.txt')
            else:
    
                self.agnsplus_selfmatch = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_agnsplus_selfmatch.txt', dtype=np.int64)
                self.agnsplus_selfmatch_other = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_agnsplus_selfmatch_other.txt', dtype=np.int64)
                
                self.agns = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_agns.txt', dtype=np.int64)
                self.sfs = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_sfs.txt', dtype=np.int64)
                
                self.agns_plus = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_agns_plus.txt', dtype=np.int64)
                self.sfs_plus = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_sfs_plus.txt', dtype=np.int64)
    
                self.neither_agn = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_neither_agn.txt', dtype=np.int64)            
                self.neither_matches = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_neither_matches.txt', dtype=np.int64)
                
                
                self.mindists = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_mindists.txt', unpack=True)
                self.mininds = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_mininds.txt', dtype=np.int64, unpack=True)
                self.minids = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_minids.txt', dtype=np.int64, unpack=True)
                self.mindists_best = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_mindists_best.txt')
                
                self.mindists_agn = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_mindists_agn.txt', unpack=True)
                self.minids_agn = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_mininds_agn.txt', dtype=np.int64, unpack=True)
                self.mininds_agn = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_minids_agn.txt', dtype=np.int64, unpack=True)
                self.mindistsagn_best = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_mindistsagn_best.txt')
        agn_dist_inds = []
        agn_plus_dist_inds = [] 
        agn_neither_dist_inds = []
        agn_selfdist_inds = []
        agn_selfplus_dist_inds = [] 

        for i in range(len(self.mindists_best)):
            mn = np.where(self.mindists[:,i] == self.mindists_best[i])[0]
            if len(mn) ==1:
                    
                if mn == 0:
                    agn_dist_inds.append(i)
                elif mn==1: 
                    agn_plus_dist_inds.append(i)
                else:
                    agn_neither_dist_inds.append(i)
            else:
                if 0 in mn:
                    agn_dist_inds.append(i)
                elif 1 in mn:
                    agn_plus_dist_inds.append(i)
                else:
                    agn_neither_dist_inds.append(i)
            mnself = np.where(self.mindists_agn[:,i] == self.mindistsagn_best[i])[0]
            if len(mnself)==1:
                if mnself==0:
                    agn_selfdist_inds.append(i)
                else:
                    agn_selfplus_dist_inds.append(i)
            else:
                if 0 in mnself:
                    agn_selfdist_inds.append(i)
                else:
                    agn_selfplus_dist_inds.append(i)
            
        self.agn_dist_inds = np.array(agn_dist_inds)
        self.agn_plus_dist_inds = np.array(agn_plus_dist_inds)
        self.agn_neither_dist_inds = np.array(agn_neither_dist_inds)
        self.agn_ind_mapping = combine_arrs([self.agn_dist_inds, self.agn_plus_dist_inds, self.agn_neither_dist_inds])

        self.agn_selfdist_inds = np.array(agn_selfdist_inds)
        self.agn_selfplus_dist_inds = np.array(agn_selfplus_dist_inds)
        self.agn_selfind_mapping = combine_arrs([self.agn_selfdist_inds, self.agn_selfplus_dist_inds])
        
        self.mindists_argmin = np.argmin(self.mindists, axis=0)
        self.mininds_best = np.array([self.mininds[:,j][self.mindists_argmin[j]] for j in range(len(self.mindists_argmin))])
        
        self.mindists_best_sing_ord = combine_arrs([self.mindists_best[self.agn_dist_inds], self.mindists_best[self.agn_plus_dist_inds], self.mindists_best[self.agn_neither_dist_inds] ])
    def subtract_elflux(self, sncut=2, halphbeta_sncut=2, second=True):
    
        '''
        Subtracting the SF component and setting a number of attributes that relate.    
        '''
        #######################
        #BPT fluxes
        #######################
        '''
        sfrm_gsw2.fullagn_df[['corrected_av','niiha_sub', 'oiiihb_sub', 'siiha_sub','oiha_sub','oi_sii_sub','U_sub','vdisp','mbh','sfr',
                     'mass','ssfr','delta_ssfr','massfracgsw',
                     'oiiiflux_sub','oiiiflux_corr_sub', 'oiii_err_sub',
                     'hbetaflux_sub','hbetaflux_corr_sub', 'hbeta_err_sub',
                     'niiflux_sub','niiflux_corr_sub', 'nii_err_sub',
                     'halpflux_sub','halpflux_corr_sub', 'halp_err_sub',
                     'oiiflux_sub','oiiflux_corr_sub', 'oii_err_sub',
                     'oiflux_sub','oiflux_corr_sub', 'oi_err_sub',
                     'siiflux_sub','siiflux_corr_sub', 'sii_err_sub',
                     'sii6717flux_sub','sii6717flux_corr_sub', 'sii6717_err_sub',
                     'sii6731flux_sub','sii6731flux_corr_sub', 'sii6731_err_sub',
                     'oiii4959flux_sub','oiii4959flux_corr_sub', 'oiii4959_err_sub',
                     'oiii4363flux_sub','oiii4363flux_corr_sub', 'oiii4363_err_sub']]
        '''
        lines = ['oiiiflux','hbetaflux','halpflux','niiflux', 'siiflux',
                  'sii6731flux', 'sii6717flux','oiflux','oiii4363flux',
                  'oiii4959flux','oiiflux']
        
        line_errs = ['oiii_err', 'hbeta_err', 'halp_err', 'nii_err', 
                    'sii_err','sii6731_err', 'sii6717_err', 'oi_err',
                    'oiii4363_err', 'oiii4959_err', 'oii_err']
        '''
        otherprops =- 

        '''

        agn_inds = np.hstack([self.agns, self.agns_plus, self.neither_agn])
        agn_ord = np.argsort(agn_inds)

        self.agn_df = self.bpt_eldiag.iloc[self.agns].copy()
        self.agnplus_df = self.bpt_eldiag.iloc[self.agns_plus].copy()
        self.neither_agn_df = self.bpt_eldiag.iloc[self.neither_agn].copy()
    
        self.sf_df = self.bpt_eldiag.iloc[self.sfs].copy()
        self.sfplus_df = self.plus_eldiag.iloc[self.sfs_plus].copy()
        self.neither_match_df = self.neither_eldiag.iloc[self.neither_matches].copy()
        if second:
            second_agn_inds = np.hstack([self.second_agns, self.second_agns_plus, self.second_neither_agn])
            second_agn_ord = np.argsort(second_agn_inds)

            self.second_agn_df = self.bpt_eldiag.iloc[self.second_agns].copy()
            self.second_agnplus_df = self.bpt_eldiag.iloc[self.second_agns_plus].copy()
            self.second_neither_agn_df = self.bpt_eldiag.iloc[self.second_neither_agn].copy()
        
            self.second_sf_df = self.bpt_eldiag.iloc[self.second_sfs].copy()
            self.second_sfplus_df = self.plus_eldiag.iloc[self.second_sfs_plus].copy()
            self.second_neither_match_df = self.neither_eldiag.iloc[self.second_neither_matches].copy()
        
            agn_eldiags = [self.agn_df, self.agnplus_df, self.neither_agn_df,
                           self.second_agn_df, self.second_agnplus_df, self.second_neither_agn_df]
            match_eldiags = [self.sf_df, self.sfplus_df, self.neither_match_df,
                             self.second_sf_df, self.second_sfplus_df, self.second_neither_match_df]

        else:
            agn_eldiags = [self.agn_df, self.agnplus_df, self.neither_agn_df]
            match_eldiags = [self.sf_df, self.sfplus_df, self.neither_match_df]

            agn_dist_mapping = [self.agn_dist_inds, self.agn_plus_dist_inds, self.agn_neither_dist_inds]

        for num in range(len(match_eldiags)):
            distrat =  (np.array(cosmo.luminosity_distance(match_eldiags[num].loc[:,'z']))/np.array(cosmo.luminosity_distance(agn_eldiags[num].loc[:,'z'])))**2
            agn_eldiags[num]['dist_ratio'] = distrat
            #print(len(agn_dist_mapping[num]))
            #agn_eldiags[num]['match_dist'] = self.mindists_best[np.int64(agn_dist_mapping[num])]

            for i in range(len(self.eldiag.lines)):
                print(lines[i])
                flux_sub = np.copy(np.array(agn_eldiags[num].loc[:,lines[i]])-np.array(match_eldiags[num].loc[:,lines[i]])*distrat)
                flux_sub_err = np.copy(np.sqrt(np.array(agn_eldiags[num].loc[:,line_errs[i]])**2+(np.array(match_eldiags[num].loc[:,line_errs[i]]*distrat))**2))
                flux_sub_sn = flux_sub/flux_sub_err
                flux_sfratio = np.copy(np.array(match_eldiags[num].loc[:,lines[i]])*distrat / (np.array(agn_eldiags[num].loc[:,lines[i]])))
                flux_agnratio = np.copy(flux_sub /(np.array(agn_eldiags[num].loc[:,lines[i]]) ))
                
                agn_eldiags[num][lines[i]+'_sub']= np.array(flux_sub)
                agn_eldiags[num][line_errs[i]+'_sub']= np.array(flux_sub_err)
                agn_eldiags[num][lines[i]+'_sub_sn']= np.array(flux_sub_sn)
                agn_eldiags[num][lines[i]+'_sfratio']= np.array(flux_sfratio)
                agn_eldiags[num][lines[i]+'_agnratio']= np.array(flux_agnratio)
                

            agn_eldiags[num]['yvals_bpt_sub'] = agn_eldiags[num]['oiiiflux_sub']/agn_eldiags[num]['hbetaflux_sub']
            agn_eldiags[num]['xvals1_bpt_sub'] = agn_eldiags[num]['niiflux_sub']/agn_eldiags[num]['halpflux_sub']
            agn_eldiags[num]['xvals2_bpt_sub'] = agn_eldiags[num]['siiflux_sub']/agn_eldiags[num]['halpflux_sub']
            agn_eldiags[num]['xvals3_bpt_sub'] = agn_eldiags[num]['oiflux_sub']/agn_eldiags[num]['halpflux_sub']
            
            
            agn_eldiags[num]['niiha_sub'] = np.log10(np.copy(agn_eldiags[num]['xvals1_bpt_sub']))
            agn_eldiags[num]['siiha_sub'] = np.log10(np.copy(agn_eldiags[num]['xvals2_bpt_sub']))          
            agn_eldiags[num]['oiha_sub'] = np.log10(np.copy(agn_eldiags[num]['xvals3_bpt_sub']))
            agn_eldiags[num]['oiiihb_sub'] = np.log10(np.copy(agn_eldiags[num]['yvals_bpt_sub']))
            agn_eldiags[num]['ji_p1_sub'] = (np.copy(agn_eldiags[num]['niiha_sub']*0.63 + 0.51*agn_eldiags[num]['siiha_sub']*0.51 + 0.59*agn_eldiags[num]['oiiihb_sub']) )
            agn_eldiags[num]['ji_p2_sub'] =  (np.copy(agn_eldiags[num]['niiha_sub']*(-0.63) + agn_eldiags[num]['siiha_sub']*0.78) )
            agn_eldiags[num]['ji_p3_sub'] = (np.copy(agn_eldiags[num]['niiha_sub']*(-0.46) -0.37*agn_eldiags[num]['siiha_sub'] + 0.81*agn_eldiags[num]['oiiihb_sub']) )
            
            agn_eldiags[num]['oi_sii_sub'] = np.log10(agn_eldiags[num]['oiflux_sub']/agn_eldiags[num]['siiflux_sub'])
            agn_eldiags[num]['av_sub'] = extinction(agn_eldiags[num]['halpflux_sub'], agn_eldiags[num]['hbetaflux_sub'])
            agn_eldiags[num]['av_sub_agn'] = extinction(agn_eldiags[num]['halpflux_sub'], agn_eldiags[num]['hbetaflux_sub'], agn=True)
        
            
            agn_eldiags[num]['offset_oiiihb'] = np.copy(agn_eldiags[num]['oiiihb_sub'] - agn_eldiags[num]['oiiihb'] ) 
            agn_eldiags[num]['offset_niiha'] = np.copy(agn_eldiags[num]['niiha_sub'] - agn_eldiags[num]['niiha'] ) 
            agn_eldiags[num]['offset_tot'] = np.sqrt(agn_eldiags[num]['offset_oiiihb']**2+agn_eldiags[num]['offset_niiha']**2)
            
            agn_eldiags[num]['offset_oiii'] = np.copy(np.log10(agn_eldiags[num]['oiiiflux_sub']) - np.log10(agn_eldiags[num]['oiiiflux'] ) )
            agn_eldiags[num]['offset_nii'] = np.copy(np.log10(agn_eldiags[num]['niiflux_sub']) - np.log10(agn_eldiags[num]['niiflux'] ) )
            agn_eldiags[num]['offset_hb'] = np.copy(np.log10(agn_eldiags[num]['hbetaflux_sub']) - np.log10(agn_eldiags[num]['hbetaflux'] )) 
            agn_eldiags[num]['offset_ha'] = np.copy(np.log10(agn_eldiags[num]['halpflux_sub']) - np.log10(agn_eldiags[num]['halpflux']) ) 
        #derived params: metallicity, (nii/oii), ionization parameter (oiii/oii), electron density (sii)
            
        self.allagn_df=pd.concat([self.agn_df,self.agnplus_df, self.neither_agn_df],join='outer')
        self.allmatch_df=pd.concat([self.sf_df,self.sfplus_df, self.neither_match_df],join='outer')
        self.sorted_allagn_df = self.allagn_df.iloc[agn_ord].copy()

        agn_dist_mapping = np.concatenate((self.agn_dist_inds, self.agn_plus_dist_inds, self.agn_neither_dist_inds))
        agn_dists = self.mindists_best[agn_dist_mapping]
        self.allagn_df['match_dist'] = agn_dists        
        if second:
            self.second_allagn_df = pd.concat([self.second_agn_df, self.second_agnplus_df, self.second_neither_agn_df], join='outer')
            self.second_allmatch_df = pd.concat([self.second_sf_df, self.second_sfplus_df, self.second_neither_match_df], join='outer')
            self.sorted_second_allagn_df = self.second_allagn_df.iloc[second_agn_ord].copy()

        full_agns = []
        for i in range(len(self.allagn_df['niiha'])):
            if self.allagn_df['niiha'].iloc[i] <-0.35:
                if np.float64(self.allagn_df['oiiihb'].iloc[i]) > np.log10(y1_kauffmann(np.float64(self.allagn_df['niiha'].iloc[i]))):
                    full_agns.append(1)
                else:
                    full_agns.append(0)
            else:
                full_agns.append(1)
        
        self.allagn_df['full_agn'] = full_agns
        full_ = np.where(self.allagn_df['full_agn']==1)[0]
        self.fullagn_df = self.allagn_df.iloc[full_].copy()
        self.fullmatch_df = self.allmatch_df.iloc[full_].copy()
        self.sn2_filt_bool = ((self.fullagn_df['hbetaflux_sub_sn']>2)&
                                 (self.fullagn_df['halpflux_sub_sn']>2)&
                                 (self.fullagn_df['niiflux_sub_sn']>2)&
                                 (self.fullagn_df['oiiiflux_sub_sn']>2)
                                 )
        self.sn2_filt = np.where(self.sn2_filt_bool)[0]
        self.sn2_filt_bool_all7 = ((self.fullagn_df['hbetaflux_sub_sn']>2)&
                                 (self.fullagn_df['halpflux_sub_sn']>2)&
                                 (self.fullagn_df['niiflux_sub_sn']>2)&
                                 (self.fullagn_df['oiiiflux_sub_sn']>2)&
                                 (self.fullagn_df['oiiflux_sub_sn']>2)&
                                 (self.fullagn_df['oiflux_sub_sn']>2)&
                                 (self.fullagn_df['siiflux_sub_sn']>2)                                 
                                 )

        self.sn2_filt_all7 = np.where(self.sn2_filt_bool_all7)[0]
        
        self.high_sn10_hb = np.where((self.fullagn_df['hbetaflux_sub_sn'] > 10))[0]
        self.high_hb10_allagn_df = self.fullagn_df.iloc[self.high_sn10_hb].copy()
        #X_reg = np.array(self.high_hb10_allagn_df.av_gsw).reshape(-1,1)

        self.X_reg_av_tau = np.vstack([self.high_hb10_allagn_df.tauv_cont]).transpose()
        self.X_reg_av = np.vstack([self.high_hb10_allagn_df.av_gsw]).transpose()
        self.X_reg_av_sfr = np.vstack([self.high_hb10_allagn_df.av_gsw, self.high_hb10_allagn_df.sfr]).transpose()
        self.X_reg_av_mass = np.vstack([self.high_hb10_allagn_df.av_gsw, self.high_hb10_allagn_df.mass]).transpose()
        self.X_reg_av_mass_sfr = np.vstack([self.high_hb10_allagn_df.av_gsw, self.high_hb10_allagn_df.mass,self.high_hb10_allagn_df.sfr]).transpose()

        self.y_reg = np.array(self.high_hb10_allagn_df.av_sub_agn)
        
        finite_y = np.where(np.isfinite(self.y_reg))
        self.reg_av = LinearRegression().fit(self.X_reg_av[finite_y],self.y_reg[finite_y])
        self.reg_av_tau = LinearRegression().fit(self.X_reg_av_tau[finite_y], self.y_reg[finite_y])
        
        self.reg_av_sfr = LinearRegression().fit(self.X_reg_av_sfr[finite_y],self.y_reg[finite_y])
        self.reg_av_mass = LinearRegression().fit(self.X_reg_av_mass[finite_y],self.y_reg[finite_y])
        self.reg_av_mass_sfr = LinearRegression().fit(self.X_reg_av_mass_sfr[finite_y],self.y_reg[finite_y])

        self.x_pred_av = np.vstack([self.fullagn_df.av_gsw]).transpose()
        self.x_pred_av_sfr = np.vstack([self.fullagn_df.av_gsw, self.fullagn_df.sfr]).transpose()
        self.x_pred_av_mass = np.vstack([self.fullagn_df.av_gsw, self.fullagn_df.mass]).transpose()
        self.x_pred_av_mass_sfr = np.vstack([self.fullagn_df.av_gsw, self.fullagn_df.mass, self.fullagn_df.sfr]).transpose()

        self.fullagn_df['corrected_av'] = correct_av(self.reg_av, self.x_pred_av, 
                                                         np.array(self.fullagn_df.av_sub_agn),
                                                         np.array(self.fullagn_df.hbetaflux_sub_sn), empirdust=True)

        self.fullagn_df['corrected_av_sfr'] = correct_av(self.reg_av_sfr, self.x_pred_av_sfr, 
                                                         np.array(self.fullagn_df.av_sub_agn),
                                                         np.array(self.fullagn_df.hbetaflux_sub_sn), empirdust=True)

        self.fullagn_df['corrected_av_mass'] = correct_av(self.reg_av_mass, self.x_pred_av_mass, 
                                                         np.array(self.fullagn_df.av_sub_agn),
                                                         np.array(self.fullagn_df.hbetaflux_sub_sn), empirdust=True)

        self.fullagn_df['corrected_av_mass_sfr'] = correct_av(self.reg_av_mass_sfr, self.x_pred_av_mass_sfr, 
                                                         np.array(self.fullagn_df.av_sub_agn),
                                                         np.array(self.fullagn_df.hbetaflux_sub_sn), empirdust=True)
        self.fullagn_df['halp_eqw_sub'] = np.array(self.fullagn_df['halp_eqw'])*np.array(self.fullagn_df['halpflux_sub'])/np.array(self.fullagn_df['halpflux'] )
        self.fullagn_df['halpflux_corr_sub'] = dustcorrect(self.fullagn_df['halpflux_sub'], self.fullagn_df['corrected_av'],6563.0)
        self.fullagn_df['hbetaflux_corr_sub'] = dustcorrect(self.fullagn_df['hbetaflux_sub'], self.fullagn_df['corrected_av'],4861.0)

        self.fullagn_df['halpflux_corrbd_sub'] = dustcorrect(self.fullagn_df['halpflux_sub'], self.fullagn_df['av_sub_agn'],6563.0)
        self.fullagn_df['hbetaflux_corrbd_sub'] = dustcorrect(self.fullagn_df['hbetaflux_sub'], self.fullagn_df['av_sub_agn'],4861.0)


        self.fullagn_df['oiiiflux_corr_sub'] = dustcorrect(self.fullagn_df['oiiiflux_sub'], self.fullagn_df['corrected_av'], 5007.0)
        self.fullagn_df['oiii4959flux_corr_sub'] = dustcorrect(self.fullagn_df['oiii4959flux_sub'], self.fullagn_df['corrected_av'], 4959.0)
        self.fullagn_df['oiii4363flux_corr_sub'] = dustcorrect(self.fullagn_df['oiii4363flux_sub'], self.fullagn_df['corrected_av'], 4363.0)
        self.fullagn_df['oiii_err_corr_sub'] = dustcorrect(self.fullagn_df['oiii_err_sub'], self.fullagn_df['corrected_av'], 5007.0)



        self.fullagn_df['niiflux_corr_sub'] = dustcorrect(self.fullagn_df['niiflux_sub'], self.fullagn_df['corrected_av'], 6583.0)
        
        self.fullagn_df['siiflux_corr_sub'] = dustcorrect(self.fullagn_df['siiflux_sub'], self.fullagn_df['corrected_av'], 6724.0)
        self.fullagn_df['sii6717flux_corr_sub'] = dustcorrect(self.fullagn_df['sii6717flux_sub'], self.fullagn_df['corrected_av'], 6717.0)
        self.fullagn_df['sii6731flux_corr_sub'] = dustcorrect(self.fullagn_df['sii6731flux_sub'], self.fullagn_df['corrected_av'], 6731.0)
        self.fullagn_df['oiiflux_corr_sub'] = dustcorrect(self.fullagn_df['oiiflux_sub'], self.fullagn_df['corrected_av'], 3727.0)
        self.fullagn_df['oiiflux_match_corr_sub'] = dustcorrect(self.fullagn_df['oiiflux_sub']*self.fullagn_df['oiiflux_sfratio'], self.fullagn_df['corrected_av'], 3727.0)

        self.fullagn_df['oiflux_corr_sub'] = dustcorrect(self.fullagn_df['oiflux_sub'], self.fullagn_df['corrected_av'], 6300.0)
        
        self.fullagn_df['oiiilum_sub_dered'] = np.log10(getlumfromflux(self.fullagn_df['oiiiflux_corr_sub'], self.fullagn_df['z']))
        self.fullagn_df['oiiilum_sub'] = np.log10(getlumfromflux(self.fullagn_df['oiiiflux_corr'], self.fullagn_df['z']))
        self.fullagn_df['oiiilum_up_sub_dered'] = np.log10(getlumfromflux(self.fullagn_df['oiiiflux_corr_sub']+self.fullagn_df['oiii_err_corr_sub'],self.fullagn_df['z']))
        self.fullagn_df['oiiilum_down_sub_dered'] = np.log10(getlumfromflux(self.fullagn_df['oiiiflux_corr_sub']-self.fullagn_df['oiii_err_corr_sub'],self.fullagn_df['z']))
        self.fullagn_df['e_oiiilum_down_sub_dered'] = self.fullagn_df['oiiilum_sub_dered']-self.fullagn_df['oiiilum_down_sub_dered']
        self.fullagn_df['e_oiiilum_up_sub_dered'] = self.fullagn_df['oiiilum_up_sub_dered']- self.fullagn_df['oiiilum_sub_dered']
        
        self.fullagn_df['oiiilum_up_sub'] = np.log10(getlumfromflux(self.fullagn_df['oiiiflux_sub']+self.fullagn_df['oiii_err_sub'],self.fullagn_df['z']))
        self.fullagn_df['oiiilum_down_sub'] = np.log10(getlumfromflux(self.fullagn_df['oiiiflux_sub']-self.fullagn_df['oiii_err_sub'],self.fullagn_df['z']))
        self.fullagn_df['e_oiiilum_down_sub'] = self.fullagn_df['oiiilum_sub']-self.fullagn_df['oiiilum_down_sub']
        self.fullagn_df['e_oiiilum_up_sub'] = self.fullagn_df['oiiilum_up_sub']- self.fullagn_df['oiiilum_sub']
        
        self.fullagn_df['oiilum_sub_dered'] = np.log10(getlumfromflux(self.fullagn_df['oiiflux_corr_sub'], self.fullagn_df['z']))
        self.fullmatch_df['oii_matchlum_dered'] = np.log10(getlumfromflux(self.fullmatch_df['oiiflux_corr'], self.fullmatch_df['z']))
        self.fullagn_df['oiilum_sub_dered_host'] = np.log10(np.array(10**self.fullagn_df['oiilum_sub_dered'])-np.array(10**self.fullmatch_df['oii_matchlum_dered']))
        
        
        
        
        self.fullagn_df['halplum_sub_dered'] = np.log10(getlumfromflux(self.fullagn_df['halpflux_corr_sub'], self.fullagn_df['z']))
        
        self.fullagn_df['edd_par_sub'] = self.fullagn_df['oiiilum_sub_dered']-self.fullagn_df['mbh']
        self.fullagn_df['edd_ratio'] = self.fullagn_df['oiiilum_sub_dered']+np.log10(600)-self.fullagn_df['edd_lum']

        self.fullagn_df['oiiilum_uncorr_sub'] = getlumfromflux(self.fullagn_df['halpflux_sub'], self.fullagn_df['z'])
        self.fullagn_df['halplum_uncorr_sub'] = getlumfromflux(self.fullagn_df['oiiiflux_sub'], self.fullagn_df['z'])

        self.fullagn_df['halpfibsfr_sub'] = halptofibsfr_corr(10**self.fullagn_df['halplum_sub_dered'])
        self.fullagn_df['halphbeta_sub_ratio']= np.copy(np.array(self.fullagn_df['halpflux_sub']/self.fullagn_df['hbetaflux_sub']))
        self.fullagn_df['halphbeta_sub_ratio_err']= np.copy(self.fullagn_df['halphbeta_sub_ratio']*np.sqrt((self.fullagn_df['halp_err_sub']/(self.fullagn_df['halpflux_sub']))**2 + 
                                                            (self.fullagn_df['hbeta_err_sub']/(self.fullagn_df['hbetaflux_sub']))**2) )
        self.fullagn_df['halphbeta_sub_ratio_sn']= np.copy(np.array(self.fullagn_df['halphbeta_sub_ratio']/self.fullagn_df['halphbeta_sub_ratio_err']))
        

        self.fullagn_df['nii_oii_sub'] =np.log10(self.fullagn_df['niiflux_corr_sub']/self.fullagn_df['oiiflux_corr_sub'])
        self.fullagn_df['log_oh_sub'] = nii_oii_to_oh(self.fullagn_df['niiflux_corr_sub'],self.fullagn_df['oiiflux_corr_sub'])         
        self.fullagn_df['log_oh_ke02_sub'] = nii_oii_to_oh_ke02(self.fullagn_df['niiflux_corr_sub'],self.fullagn_df['oiiflux_corr_sub'])         

        self.fullagn_df['sii_ratio_sub'] =self.fullagn_df['sii6717flux_sub']/self.fullagn_df['sii6731flux_sub']
        self.fullagn_df['sii_oii_sub'] =np.log10(self.fullagn_df['siiflux_sub']/self.fullagn_df['oiiflux_sub'])
        self.fullagn_df['n_e14000'] = sii_doub_to_ne(self.fullagn_df['sii_ratio_sub'], te=14000)
        self.fullagn_df['n_e12000'] = sii_doub_to_ne(self.fullagn_df['sii_ratio_sub'], te=12000)
        self.fullagn_df['n_e11000'] = sii_doub_to_ne(self.fullagn_df['sii_ratio_sub'], te=11000)

        '''
        self.fullagn_df['O32_sub'] = np.log10( (self.fullagn_df['oiii4959flux_corr_sub'] + self.fullagn_df['oiiiflux_corr_sub'])/self.fullagn_df['oiiflux_corr_sub'])
        self.fullagn_df['q_sub'] = nii_logoh_o32_to_q(self.fullagn_df['log_oh_sub'], self.fullagn_df['O32_sub'])
        self.fullagn_df['qc_sub'] = self.fullagn_df['q_sub']-np.log10(3e10)
        '''
        self.fullagn_df['oiii_oii_sub'] =np.log10(self.fullagn_df['oiiiflux_corr_sub']/self.fullagn_df['oiiflux_corr_sub'])
        self.sn2_filt_sy2_bool = ((self.sn2_filt_bool)&
                                     (self.fullagn_df['oiiihb_sub']> np.log10(y2_linersy2(self.fullagn_df['siiha_sub'])))&
                                     (self.fullagn_df['oiiihb_sub']> np.log10(y2_agn(self.fullagn_df['siiha_sub'])))&
                                     (self.fullagn_df['siiflux_sub_sn']>2))
        self.sn2_filt_sy2= np.where(self.sn2_filt_sy2_bool)
        self.sn2_filt_liner= np.where(np.logical_not(self.sn2_filt_sy2_bool))

        self.sn2_filt_sy2_bool_oi = ((self.sn2_filt_bool)&
                                     (self.fullagn_df['oiiihb_sub']> np.log10(y3_linersy2(self.fullagn_df['oiha_sub'])))&
                                     (self.fullagn_df['oiiihb_sub']> np.log10(y3_agn(self.fullagn_df['oiha_sub'])))&
                                     (self.fullagn_df['oiflux_sub_sn']>2))
        self.sn2_filt_sy2_oi= np.where(self.sn2_filt_sy2_bool_oi)
        self.sn2_filt_liner_oi= np.where(np.logical_not(self.sn2_filt_sy2_bool_oi))
        

        self.fullagn_df['U_sub'] = oiii_oii_to_U(self.fullagn_df['oiii_oii_sub'])
        
        sii_diag_oiiihb_exp =  np.array(np.log10(y2_linersy2(self.fullagn_df['siiha_sub'])))
        sy2_liner = (np.array(self.fullagn_df['oiiihb_sub'])>sii_diag_oiiihb_exp)
        sy2_liner[np.where(sy2_liner == True)[0]] = 1
        sy2_liner[np.where(sy2_liner == False)[0]] = 0
        self.fullagn_df['sy2_liner_bool'] = sy2_liner  
        self.low_z_df = self.fullagn_df.iloc[np.where(self.fullagn_df.z<=0.07)[0]].copy()#to focus on sig1
    def get_filt_dfs(self, filts, gen_filts, 
                     match_filts, line_filts, line_filts_comb, 
                     sncut=2, combo_sncut=2, 
                     delta_ssfr_cut=-0.7, 
                     minmass=10.2, maxmass=10.4, 
                     d4000_cut=1.6, 
                     loweroiiilum=40.2, upperoiiilum=40.3, 
                     upperU_cut=-0.2, lowerU_cut=-0.3):
        self.filts = filts
        self.gen_filts = gen_filts
        self.match_filts = match_filts
        for i in range(len(line_filts)):
            line_sn_filt = np.where(self.fullagn_df[line_filts[i][0]+'_sn']> line_filts[i][1])        
            self.gen_filts[line_filts[i][0]+'_filt'] = line_sn_filt
            self.gen_filts[line_filts[i][0]+'_df'] = self.fullagn_df.iloc[line_sn_filt].copy()
            self.match_filts[line_filts[i][0]+'_df'] = self.fullmatch_df.iloc[line_sn_filt].copy()
            
            
        for filt in self.filts.keys():
            if len(self.filts[filt]['cut'])==2:
                cent_filt = np.where((self.fullagn_df[filt]>self.filts[filt]['cut'][0])  &
                                     (self.fullagn_df[filt]<self.filts[filt]['cut'][1]))[0]
                self.gen_filts['mid_'+filt+'_filt'] = cent_filt
                self.gen_filts['mid_'+filt+'_df'] = self.fullagn_df.iloc[cent_filt].copy()
                self.match_filts['mid_'+filt+'_df'] = self.fullmatch_df.iloc[cent_filt].copy()
                 
        for comb in line_filts_comb.keys():
            n_comb = len(line_filts_comb[comb][0])
            init_ = [True]*len(self.fullagn_df)
            for i in range(n_comb):
               
                mask = (self.fullagn_df[line_filts_comb[comb][0][i]+'_sn']>line_filts_comb[comb][1])
                pass_mask = (mask & init_)
                init_ = pass_mask
            
            comb_sn_filt  = np.where(pass_mask)
            self.gen_filts['filt_comb_'+comb] = comb_sn_filt
            self.gen_filts['df_comb_'+comb] = self.fullagn_df.iloc[comb_sn_filt].copy()            
            self.match_filts['df_comb_'+comb] = self.fullmatch_df.iloc[comb_sn_filt].copy()            
                
        for filt in self.filts.keys():
            if len(self.filts[filt]['cut']) ==1:                

                up = np.where(self.fullagn_df[filt] > self.filts[filt]['cut'][0])[0]
                down = np.where(self.fullagn_df[filt] <= self.filts[filt]['cut'][0])[0]
                
                self.filts[filt]['up_filt'] = up
                self.filts[filt]['up_df'] = self.fullagn_df.iloc[up].copy()
                self.match_filts[filt]['up_df'] = self.fullmatch_df.iloc[up].copy()
                
                self.filts[filt]['down_filt'] = down
                self.filts[filt]['down_df'] = self.fullagn_df.iloc[down].copy()
                self.match_filts[filt]['down_df'] = self.fullmatch_df.iloc[down].copy()

                uphamatch = np.where((np.array(self.fullagn_df[filt]) > self.filts[filt]['cut'][0])&
                                     (np.array(self.fullmatch_df['halpflux_sn'])>sncut))[0]
                downhamatch = np.where((np.array(self.fullagn_df[filt]) <= self.filts[filt]['cut'][0])&
                                       (np.array(self.fullmatch_df['halpflux_sn'])>sncut))[0]
                self.filts[filt]['up_hamatch_filt'] = uphamatch
                self.filts[filt]['up_hamatch_df'] = self.fullagn_df.iloc[uphamatch].copy()
                self.match_filts[filt]['up_hamatch_df'] = self.fullmatch_df.iloc[uphamatch].copy()
                
                self.filts[filt]['down_hamatch_filt'] = downhamatch
                self.filts[filt]['down_hamatch_df'] = self.fullagn_df.iloc[downhamatch].copy()
                self.match_filts[filt]['down_hamatch_df'] = self.fullmatch_df.iloc[downhamatch].copy()
                for sub_filt in self.filts.keys():
                    if sub_filt == filt:
                        continue
                    if len(self.filts[sub_filt]['cut']) == 2:
                        
                        sub_mid_up = np.where((self.fullagn_df[filt] > self.filts[filt]['cut'][0]) &
                                              (self.fullagn_df[sub_filt] > self.filts[sub_filt]['cut'][0])&
                                              (self.fullagn_df[sub_filt] <= self.filts[sub_filt]['cut'][1])
                                              )[0]
                        sub_mid_down = np.where((self.fullagn_df[filt] <= self.filts[filt]['cut'][0])&
                                                (self.fullagn_df[sub_filt] > self.filts[sub_filt]['cut'][0])&
                                                (self.fullagn_df[sub_filt] <= self.filts[sub_filt]['cut'][1]))[0]

                        self.filts[filt]['up_mid_'+sub_filt+'_filt'] = sub_mid_up
                        self.filts[filt]['down_mid_'+sub_filt+'_filt'] = sub_mid_down
                        
                        self.filts[filt]['up_mid_'+sub_filt+'_df'] = self.fullagn_df.iloc[sub_mid_up].copy()
                        self.filts[filt]['down_mid_'+sub_filt+'_df'] = self.fullagn_df.iloc[sub_mid_down].copy()
                        
                        for i in range(len(line_filts)):
                            up_filt = np.where(self.filts[filt]['up_mid_'+sub_filt+'_df'][line_filts[i][0]+'_sn'] > line_filts[i][1])[0]
        
                            down_filt = np.where(self.filts[filt]['down_mid_'+sub_filt+'_df'][line_filts[i][0]+'_sn'] > line_filts[i][1])[0]
        
                            self.filts[filt]['down_mid_'+sub_filt+'_'+line_filts[i][0]+'_filt'] = down_filt
                            self.filts[filt]['down_mid_'+sub_filt+'_'+line_filts[i][0]+'_df'] = self.filts[filt]['down_mid_'+sub_filt+'_df'].iloc[down_filt]
        
                            self.filts[filt]['up_mid_'+sub_filt+'_'+line_filts[i][0]+'_filt'] = up_filt
                            self.filts[filt]['up_mid_'+sub_filt+'_'+line_filts[i][0]+'_df'] = self.filts[filt]['up_mid_'+sub_filt+'_df'].iloc[up_filt].copy()
        
                        for comb in line_filts_comb:
                            n_comb = len(line_filts_comb[comb][0])
                            init_mask = [True]*len(self.filts[filt]['up_mid_'+sub_filt+'_df'])                    
                            for i in range(n_comb):
                               
                                mask = (self.filts[filt]['up_mid_'+sub_filt+'_df'][line_filts_comb[comb][0][i]+'_sn']>line_filts_comb[comb][1])
                                pass_mask= (mask & init_mask)
                                init_mask = pass_mask
        
                            up_filt_combo = np.where(pass_mask)[0]
                            self.filts[filt]['up_mid_'+sub_filt+ '_'+ comb +'_filt'] = up_filt_combo
                            self.filts[filt]['up_mid_'+sub_filt+ '_'+ comb +'_df'] = self.filts[filt]['up_mid_'+sub_filt+'_df'].iloc[up_filt_combo].copy()
        
                            mask_filts_sub = []
                            init_mask = [True]*len(self.filts[filt]['down_mid_'+sub_filt+'_df'])
                            for i in range(n_comb):
                                mask = (self.filts[filt]['down_mid_'+sub_filt+'_df'][line_filts_comb[comb][0][i]+'_sn']>line_filts_comb[comb][1])
                                pass_mask = (mask & init_mask)
                                init_mask = pass_mask
        
                                              
                            down_filt_combo = np.where(pass_mask)[0]
                            
                            self.filts[filt]['down_mid_'+sub_filt+ '_'+ comb +'_filt' ] = down_filt_combo
                            self.filts[filt]['down_mid_'+sub_filt+ '_'+ comb +'_df'] = self.filts[filt]['down_mid_'+sub_filt+'_df'].iloc[down_filt_combo].copy()
                    if sub_filt == 'sy2_liner_bool':
                        
                        sub_up_up = np.where((self.fullagn_df[filt] > self.filts[filt]['cut'][0]) &
                                              (self.fullagn_df[sub_filt] > self.filts[sub_filt]['cut'][0]))[0]
                        sub_up_down = np.where((self.fullagn_df[filt] > self.filts[filt]['cut'][0])&
                                                (self.fullagn_df[sub_filt] <= self.filts[sub_filt]['cut'][0]))[0]
                        
                        sub_down_up = np.where((self.fullagn_df[filt] <= self.filts[filt]['cut'][0]) &
                                              (self.fullagn_df[sub_filt] > self.filts[sub_filt]['cut'][0]))[0]
                        sub_down_down = np.where((self.fullagn_df[filt] <= self.filts[filt]['cut'][0])&
                                                (self.fullagn_df[sub_filt] <= self.filts[sub_filt]['cut'][0]))[0]


                        self.filts[filt]['up_up_'+sub_filt+'_filt'] = sub_up_up
                        self.filts[filt]['up_down_'+sub_filt+'_filt'] = sub_up_down
                        
                        self.filts[filt]['up_up_'+sub_filt+'_df'] = self.fullagn_df.iloc[sub_up_up].copy()
                        self.filts[filt]['up_down_'+sub_filt+'_df'] = self.fullagn_df.iloc[sub_up_down].copy()

                        self.filts[filt]['down_up_'+sub_filt+'_filt'] = sub_down_up
                        self.filts[filt]['down_down_'+sub_filt+'_filt'] = sub_down_down
                        
                        self.filts[filt]['down_up_'+sub_filt+'_df'] = self.fullagn_df.iloc[sub_down_up].copy()
                        self.filts[filt]['down_down_'+sub_filt+'_df'] = self.fullagn_df.iloc[sub_down_down].copy()

                        
                        for i in range(len(line_filts)):
                            up_up_filt = np.where(self.filts[filt]['up_up_'+sub_filt+'_df'][line_filts[i][0]+'_sn'] > line_filts[i][1])[0]
        
                            down_up_filt = np.where(self.filts[filt]['down_up_'+sub_filt+'_df'][line_filts[i][0]+'_sn'] > line_filts[i][1])[0]
        
                            self.filts[filt]['up_up_'+sub_filt+'_'+line_filts[i][0]+'_filt'] = up_up_filt
                            self.filts[filt]['up_up_'+sub_filt+'_'+line_filts[i][0]+'_df'] = self.filts[filt]['up_up_'+sub_filt+'_df'].iloc[up_up_filt]
        
                            self.filts[filt]['down_up_'+sub_filt+'_'+line_filts[i][0]+'_filt'] = down_up_filt
                            self.filts[filt]['down_up'+sub_filt+'_'+line_filts[i][0]+'_df'] = self.filts[filt]['down_up_'+sub_filt+'_df'].iloc[down_up_filt].copy()

                            up_down_filt = np.where(self.filts[filt]['up_down_'+sub_filt+'_df'][line_filts[i][0]+'_sn'] > line_filts[i][1])[0]
        
                            down_down_filt = np.where(self.filts[filt]['down_down_'+sub_filt+'_df'][line_filts[i][0]+'_sn'] > line_filts[i][1])[0]
        
                            self.filts[filt]['up_down_'+sub_filt+'_'+line_filts[i][0]+'_filt'] = up_down_filt
                            self.filts[filt]['up_down_'+sub_filt+'_'+line_filts[i][0]+'_df'] = self.filts[filt]['up_down_'+sub_filt+'_df'].iloc[up_down_filt]
        
                            self.filts[filt]['down_down_'+sub_filt+'_'+line_filts[i][0]+'_filt'] = down_down_filt
                            self.filts[filt]['down_down_'+sub_filt+'_'+line_filts[i][0]+'_df'] = self.filts[filt]['down_down_'+sub_filt+'_df'].iloc[down_down_filt].copy()

        
                        for comb in line_filts_comb:
                            n_comb = len(line_filts_comb[comb][0])
                            init_mask = [True]*len(self.filts[filt]['up_up_'+sub_filt+'_df'])                    
                            for i in range(n_comb):
                               
                                mask = (self.filts[filt]['up_up_'+sub_filt+'_df'][line_filts_comb[comb][0][i]+'_sn']>line_filts_comb[comb][1])
                                pass_mask= (mask & init_mask)
                                init_mask = pass_mask
        
                            up_up_filt_combo = np.where(pass_mask)[0]
                            self.filts[filt]['up_up_'+sub_filt+ '_'+ comb +'_filt'] = up_up_filt_combo
                            self.filts[filt]['up_up_'+sub_filt+ '_'+ comb +'_df'] = self.filts[filt]['up_up_'+sub_filt+'_df'].iloc[up_up_filt_combo].copy()
        
                            mask_filts_sub = []
                            init_mask = [True]*len(self.filts[filt]['down_up_'+sub_filt+'_df'])
                            for i in range(n_comb):
                                mask = (self.filts[filt]['down_up_'+sub_filt+'_df'][line_filts_comb[comb][0][i]+'_sn']>line_filts_comb[comb][1])
                                pass_mask = (mask & init_mask)
                                init_mask = pass_mask
        
                                              
                            down_up_filt_combo = np.where(pass_mask)[0]
                            
                            self.filts[filt]['down_up_'+sub_filt+ '_'+ comb +'_filt' ] = down_up_filt_combo
                            self.filts[filt]['down_up_'+sub_filt+ '_'+ comb +'_df'] = self.filts[filt]['down_up_'+sub_filt+'_df'].iloc[down_up_filt_combo].copy()

                            init_mask = [True]*len(self.filts[filt]['up_down_'+sub_filt+'_df'])                    
                            for i in range(n_comb):
                               
                                mask = (self.filts[filt]['up_down_'+sub_filt+'_df'][line_filts_comb[comb][0][i]+'_sn']>line_filts_comb[comb][1])
                                pass_mask= (mask & init_mask)
                                init_mask = pass_mask
        
                            up_down_filt_combo = np.where(pass_mask)[0]
                            self.filts[filt]['up_down_'+sub_filt+ '_'+ comb +'_filt'] = up_down_filt_combo
                            self.filts[filt]['up_down_'+sub_filt+ '_'+ comb +'_df'] = self.filts[filt]['up_down_'+sub_filt+'_df'].iloc[up_down_filt_combo].copy()
        
                            mask_filts_sub = []
                            init_mask = [True]*len(self.filts[filt]['down_down_'+sub_filt+'_df'])
                            for i in range(n_comb):
                                mask = (self.filts[filt]['down_down_'+sub_filt+'_df'][line_filts_comb[comb][0][i]+'_sn']>line_filts_comb[comb][1])
                                pass_mask = (mask & init_mask)
                                init_mask = pass_mask
        
                                              
                            down_down_filt_combo = np.where(pass_mask)[0]
                            
                            self.filts[filt]['down_down_'+sub_filt+ '_'+ comb +'_filt' ] = down_down_filt_combo
                            self.filts[filt]['down_down_'+sub_filt+ '_'+ comb +'_df'] = self.filts[filt]['down_down_'+sub_filt+'_df'].iloc[down_down_filt_combo].copy()
             

                for i in range(len(line_filts)):
                    up_filt = np.where(self.filts[filt]['up_df'][line_filts[i][0]+'_sn'] > line_filts[i][1])[0]

                    down_filt = np.where(self.filts[filt]['down_df'][line_filts[i][0]+'_sn'] > line_filts[i][1])[0]

                    self.filts[filt]['down_filt_'+line_filts[i][0]] = down_filt
                    self.filts[filt]['down_df_'+line_filts[i][0]] = self.filts[filt]['down_df'].iloc[down_filt]
                    self.match_filts[filt]['down_df_'+line_filts[i][0]] = self.match_filts[filt]['down_df'].iloc[down_filt]

                    self.filts[filt]['up_filt_'+line_filts[i][0]] = up_filt
                    self.filts[filt]['up_df_'+line_filts[i][0]] = self.filts[filt]['up_df'].iloc[up_filt].copy()
                    self.match_filts[filt]['up_df_'+line_filts[i][0]] = self.match_filts[filt]['up_df'].iloc[up_filt].copy()

                for comb in line_filts_comb:
                    n_comb = len(line_filts_comb[comb][0])
                    init_mask = [True]*len(self.filts[filt]['up_df'])                    
                    for i in range(n_comb):
                       
                        mask = (self.filts[filt]['up_df'][line_filts_comb[comb][0][i]+'_sn']>line_filts_comb[comb][1])
                        pass_mask= (mask & init_mask)
                        init_mask = pass_mask

                    up_filt_combo = np.where(pass_mask)[0]
                    self.filts[filt]['up_filt_' + comb ] = up_filt_combo
                    self.filts[filt]['up_df_' + comb ] = self.filts[filt]['up_df'].iloc[up_filt_combo].copy()
                    self.match_filts[filt]['up_df_' + comb] = self.match_filts[filt]['up_df'].iloc[up_filt_combo].copy()

                    mask_filts_sub = []
                    init_mask = [True]*len(self.filts[filt]['down_df'])
                    for i in range(n_comb):
                        mask = (self.filts[filt]['down_df'][line_filts_comb[comb][0][i]+'_sn']>line_filts_comb[comb][1])
                        pass_mask = (mask & init_mask)
                        init_mask = pass_mask

                                      
                    down_filt_combo = np.where(pass_mask)[0]
                    
                    self.filts[filt]['down_filt_' + comb ] = down_filt_combo
                    self.filts[filt]['down_df_' + comb ] = self.filts[filt]['down_df'].iloc[down_filt_combo].copy()
                    self.match_filts[filt]['down_df_' + comb ] = self.match_filts[filt]['down_df'].iloc[down_filt_combo].copy()
        
                    
            elif len(self.filts[filt]['cut']) == 2:
                mid = np.where((self.fullagn_df[filt] > self.filts[filt]['cut'][0])&
                               (self.fullagn_df[filt] <= self.filts[filt]['cut'][1]) )[0]                                    
                self.filts[filt]['mid'] = mid
                self.filts[filt]['mid_df'] = self.fullagn_df.iloc[mid].copy()
                self.match_filts[filt]['mid_df'] = self.fullmatch_df.iloc[mid].copy()
                for sub_filt in self.filts.keys():
                    if sub_filt == filt:
                        continue
                    if len(self.filts[sub_filt]['cut']) == 1:
                        
                        sub_up_mid = np.where((self.fullagn_df[sub_filt] > self.filts[sub_filt]['cut'][0]) &
                                              (self.fullagn_df[filt]>self.filts[filt]['cut'][0])&
                                              (self.fullagn_df[filt]<=self.filts[filt]['cut'][1])
                                              )[0]
                        sub_down_mid = np.where((self.fullagn_df[sub_filt] <= self.filts[sub_filt]['cut'][0])&
                                                (self.fullagn_df[filt]>self.filts[filt]['cut'][0])&
                                                (self.fullagn_df[filt]<=self.filts[filt]['cut'][1]))[0]

                        self.filts[filt]['mid_up'+sub_filt+'_filt'] = sub_up_mid
                        self.filts[filt]['mid_down'+sub_filt+'_filt'] = sub_down_mid
                        
                        self.filts[filt]['mid_up'+sub_filt+'_df'] = self.fullagn_df.iloc[sub_up_mid].copy()
                        self.filts[filt]['mid_down'+sub_filt+'_df'] = self.fullagn_df.iloc[sub_down_mid].copy()
                        

                for i in range(len(line_filts)):
                    mid_filt = np.where(self.filts[filt]['mid_df'][line_filts[i][0]+'_sn'] > line_filts[i][1])[0]
                    self.filts[filt]['mid_filt_'+line_filts[i][0]] = mid_filt
                    self.filts[filt]['mid_df_'+line_filts[i][0]] = self.filts[filt]['mid_df'].iloc[mid_filt].copy()
                    self.match_filts[filt]['mid_df_'+line_filts[i][0]] = self.match_filts[filt]['mid_df'].iloc[mid_filt].copy()


    

                for comb in line_filts_comb.keys():
                    n_comb = len(line_filts_comb[comb][0])
                    init_mask = [True]*len(self.filts[filt]['mid_df']) 
                    for i in range(n_comb):
                       
                        mask = (self.filts[filt]['mid_df'][line_filts_comb[comb][0][i]+'_sn']>line_filts_comb[comb][1])
                        pass_mask = (mask & init_mask)
                        init_mask = pass_mask
                        
                    mid_filt_combo = np.where(pass_mask)[0]                    
 
  
                    self.filts[filt]['mid_filt_' + comb ] = mid_filt_combo
                    self.filts[filt]['mid_df_' + comb ] = self.filts[filt]['mid_df'].iloc[mid_filt_combo].copy()
                    self.match_filts[filt]['mid_df_' + comb ] = self.match_filts[filt]['mid_df'].iloc[mid_filt_combo].copy()
                    
        hamatch_filt = np.where(self.fullmatch_df['halpflux_sn']>2)[0]
        self.match_filts['df_hamatch'] = self.fullmatch_df.iloc[hamatch_filt].copy()            
        self.filts['df_hamatch'] = self.fullagn_df.iloc[hamatch_filt].copy()            



    def bin_by_bpt(self, val1, sy2, sf, liner2, binsize=0.1):
        '''
        bin galaxies in the BPT diagram to find average offset
        '''
        #nx = 3/0.1
        #ny = 2.4/0.1
        #minx=-2, maxx=1, miny=-1.2, maxy=1.2
        xvals = np.arange(-2, 1.5, binsize)
        yvals = np.arange(-1.2, 2, binsize)
        mid_x = (xvals[:-1]+xvals[1:])/2
        mid_y = (yvals[:-1]+yvals[1:])/2
        
        bpt_set = []
        plus_set = []
        neither_set = []
        clustering_set = []
        clust1_set = []
        clust2_set = []
        clust3_set = []
        sing_set = []
        low_set = []
        high_set = []
        sy2_set = []
        liner_low_set = []
        liner_high_set = []
        mesh_x, mesh_y = np.meshgrid((xvals[:-1] +xvals[1:])/2, (yvals[:-1]+yvals[1:])/2) 
        coordpairs = {}
        for i in range(len(xvals)-1):
            for j in range(len(yvals)-1):

                full_agn = np.where(self.fullagn_df.full_agn==1)[0]

                valid_sing = np.where((self.fullagn_df.oiiihb.iloc[full_agn] >= yvals[j]) &
                                      (self.fullagn_df.oiiihb.iloc[full_agn] < yvals[j+1]) &
                                      (self.fullagn_df.niiha.iloc[full_agn] >= xvals[i]) &
                                      (self.fullagn_df.niiha.iloc[full_agn] < xvals[i+1] )
                                      )[0]
                
                valid_bpt = np.where((self.agn_df.oiiihb >= yvals[j] ) &
                                     (self.agn_df.oiiihb < yvals[j+1]) &
                                     (self.agn_df.niiha >= xvals[i] ) &
                                     (self.agn_df.niiha < xvals[i+1])
                                     )[0]
                valid_plus = np.where((self.agnplus_df.oiiihb >= yvals[j] ) &
                                     (self.agnplus_df.oiiihb < yvals[j+1]) &
                                     (self.agnplus_df.niiha >= xvals[i] ) &
                                     (self.agnplus_df.niiha < xvals[i+1])
                                     )[0]
                valid_neither = np.where((self.neither_agn_df.oiiihb  >= yvals[j] ) &
                                     (self.neither_agn_df.oiiihb  < yvals[j+1]) &
                                     (self.neither_agn_df.niiha  >= xvals[i] ) &
                                     (self.neither_agn_df.niiha  < xvals[i+1])
                                     )[0]
                valid_high = np.where((self.filts['delta_ssfr']['up_df'].oiiihb >= yvals[j] ) &
                                     (self.filts['delta_ssfr']['up_df'].oiiihb < yvals[j+1]) &
                                     (self.filts['delta_ssfr']['up_df'].niiha >= xvals[i] ) &
                                     (self.filts['delta_ssfr']['up_df'].niiha < xvals[i+1])
                                     )[0]
                valid_low = np.where((self.filts['delta_ssfr']['down_df'].oiiihb >= yvals[j] ) &
                                     (self.filts['delta_ssfr']['down_df'].oiiihb < yvals[j+1]) &
                                     (self.filts['delta_ssfr']['down_df'].niiha >= xvals[i] ) &
                                     (self.filts['delta_ssfr']['down_df'].niiha < xvals[i+1])
                                     )[0]
                valid_sy2 = np.where((self.filts['sy2_liner_bool']['up_df_siiflux_sub'].oiiihb >= yvals[j] ) &
                                     (self.filts['sy2_liner_bool']['up_df_siiflux_sub'].oiiihb < yvals[j+1]) &
                                     (self.filts['sy2_liner_bool']['up_df_siiflux_sub'].niiha >= xvals[i] ) &
                                     (self.filts['sy2_liner_bool']['up_df_siiflux_sub'].niiha < xvals[i+1])
                                     )[0]
                valid_liner_high_ssfr = np.where((self.filts['delta_ssfr']['up_down_sy2_liner_bool_siiflux_sub_df'].oiiihb >= yvals[j] ) &
                                                 (self.filts['delta_ssfr']['up_down_sy2_liner_bool_siiflux_sub_df'].oiiihb < yvals[j+1]) &
                                                 (self.filts['delta_ssfr']['up_down_sy2_liner_bool_siiflux_sub_df'].niiha >= xvals[i] ) &
                                                 (self.filts['delta_ssfr']['up_down_sy2_liner_bool_siiflux_sub_df'].niiha < xvals[i+1])
                                                 )[0]
                valid_liner_low_ssfr = np.where((self.filts['delta_ssfr']['down_down_sy2_liner_bool_siiflux_sub_df'].oiiihb >= yvals[j] ) &
                                                (self.filts['delta_ssfr']['down_down_sy2_liner_bool_siiflux_sub_df'].oiiihb < yvals[j+1]) &
                                                (self.filts['delta_ssfr']['down_down_sy2_liner_bool_siiflux_sub_df'].niiha >= xvals[i] ) &
                                                (self.filts['delta_ssfr']['down_down_sy2_liner_bool_siiflux_sub_df'].niiha < xvals[i+1])
                                                )[0]



                valid_clustering  = np.where((self.fullagn_df.oiiihb.iloc[val1] >= yvals[j]) &
                                      (self.fullagn_df.oiiihb.iloc[val1] < yvals[j+1]) &
                                      (self.fullagn_df.niiha.iloc[val1] >= xvals[i]) &
                                      (self.fullagn_df.niiha.iloc[val1] < xvals[i+1] )
                                      )[0]

                valid_clust1  = np.where((self.fullagn_df.oiiihb.iloc[sy2] >= yvals[j]) &
                                      (self.fullagn_df.oiiihb.iloc[sy2]  < yvals[j+1]) &
                                      (self.fullagn_df.niiha.iloc[sy2]  >= xvals[i]) &
                                      (self.fullagn_df.niiha.iloc[sy2]  < xvals[i+1] )
                                      )[0]


                valid_clust2 = np.where((self.fullagn_df.oiiihb.iloc[sf]  >= yvals[j]) &
                                      (self.fullagn_df.oiiihb.iloc[sf]  < yvals[j+1]) &
                                      (self.fullagn_df.niiha.iloc[sf]  >= xvals[i]) &
                                      (self.fullagn_df.niiha.iloc[sf]  < xvals[i+1] )
                                      )[0]

                valid_clust3 = np.where((self.fullagn_df.oiiihb.iloc[liner2]  >= yvals[j]) &
                                      (self.fullagn_df.oiiihb.iloc[liner2] < yvals[j+1]) &
                                      (self.fullagn_df.niiha.iloc[liner2] >= xvals[i]) &
                                      (self.fullagn_df.niiha.iloc[liner2] < xvals[i+1] )
                                      )[0]

                match_dist_bpt = np.copy(self.mindists_best[self.agn_dist_inds])[valid_bpt]
                match_dist_plus = np.copy(self.mindists_best[self.agn_plus_dist_inds])[valid_plus]
                match_dist_neither = np.copy(self.mindists_best[self.agn_neither_dist_inds])[valid_neither]
                match_dist_sing = np.copy(self.mindists_best_sing_ord)[full_agn[valid_sing]]
                match_dist_high = np.copy(self.mindists_best_sing_ord)[valid_high]
                match_dist_low = np.copy(self.mindists_best_sing_ord)[valid_low]
                
                match_dist_clustering = np.copy(self.mindists_best_sing_ord)[val1]

                match_dist_clust1 = np.copy(self.mindists_best_sing_ord)[sy2[valid_clust1]]
                match_dist_clust2 = np.copy(self.mindists_best_sing_ord)[sf[valid_clust2]]
                match_dist_clust3 = np.copy(self.mindists_best_sing_ord)[liner2[valid_clust3]]
                
                match_dist_sy2 = np.copy(self.mindists_best_sing_ord)[valid_sy2]
                match_dist_liner_high = np.copy(self.mindists_best_sing_ord)[valid_liner_high_ssfr]
                match_dist_liner_low = np.copy(self.mindists_best_sing_ord)[valid_liner_low_ssfr]
                if valid_clustering.size > 10:
                    distro_clustering_x = np.copy(self.fullagn_df.offset_niiha.iloc[val1[valid_clustering]])
                    distro_clustering_y = np.copy(self.fullagn_df.offset_oiiihb.iloc[val1[valid_clustering]])

                    mn_x_clustering = np.mean(distro_clustering_x)
                    mn_y_clustering = np.mean(distro_clustering_y)
                    med_x_clustering = np.median(distro_clustering_x)
                    med_y_clustering = np.median(distro_clustering_y)
                    clustering_set.append([i,j, mid_x[i], mid_y[j], mn_x_clustering, 
                                     mn_y_clustering, med_x_clustering, med_y_clustering, 
                                     distro_clustering_x, distro_clustering_y, match_dist_clustering, valid_clustering])
                else:
                    clustering_set.append([i,j,mid_x[i], mid_y[j],0, 0, 0, 0, [], [],[], []])

                if valid_clust1.size > 10:
                    distro_clust1_x = np.copy(self.fullagn_df.offset_niiha.iloc[sy2[valid_clust1]])
                    distro_clust1_y = np.copy(self.fullagn_df.offset_oiiihb.iloc[sy2[valid_clust1]])

                    mn_x_clust1 = np.mean(distro_clust1_x)
                    mn_y_clust1 = np.mean(distro_clust1_y)
                    med_x_clust1 = np.median(distro_clust1_x)
                    med_y_clust1 = np.median(distro_clust1_y)
                    clust1_set.append([i,j, mid_x[i], mid_y[j], mn_x_clust1, 
                                     mn_y_clust1, med_x_clust1, med_y_clust1, 
                                     distro_clust1_x, distro_clust1_y, match_dist_clust1, valid_clust1])
                else:
                    clust1_set.append([i,j,mid_x[i], mid_y[j],0, 0, 0, 0, [], [],[], []])

                if valid_clust2.size > 10:
                    distro_clust2_x = np.copy(self.fullagn_df.offset_niiha.iloc[sf[valid_clust2]])
                    distro_clust2_y = np.copy(self.fullagn_df.offset_oiiihb.iloc[sf[valid_clust2]])

                    mn_x_clust2 = np.mean(distro_clust2_x)
                    mn_y_clust2 = np.mean(distro_clust2_y)
                    med_x_clust2 = np.median(distro_clust2_x)
                    med_y_clust2 = np.median(distro_clust2_y)
                    clust2_set.append([i,j, mid_x[i], mid_y[j], mn_x_clust2, 
                                     mn_y_clust2, med_x_clust2, med_y_clust2, 
                                     distro_clust2_x, distro_clust2_y, match_dist_clust2, valid_clust2])
                else:
                    clust2_set.append([i,j,mid_x[i], mid_y[j],0, 0, 0, 0, [], [],[], []])

                if valid_clust3.size > 10:
                    distro_clust3_x = np.copy(self.fullagn_df.offset_niiha.iloc[liner2[valid_clust3]])
                    distro_clust3_y = np.copy(self.fullagn_df.offset_oiiihb.iloc[liner2[valid_clust3]])

                    mn_x_clust3 = np.mean(distro_clust3_x)
                    mn_y_clust3 = np.mean(distro_clust3_y)
                    med_x_clust3 = np.median(distro_clust3_x)
                    med_y_clust3 = np.median(distro_clust3_y)
                    clust3_set.append([i,j, mid_x[i], mid_y[j], mn_x_clust3, 
                                     mn_y_clust3, med_x_clust3, med_y_clust3, 
                                     distro_clust3_x, distro_clust3_y, match_dist_clust3, valid_clust3])
                else:
                    clust3_set.append([i,j,mid_x[i], mid_y[j],0, 0, 0, 0, [], [],[], []])

                if valid_sy2.size > 10:
                    distro_sy2_x = np.copy(self.filts['sy2_liner_bool']['up_df_siiflux_sub'].offset_niiha.iloc[valid_sy2])
                    distro_sy2_y = np.copy(self.filts['sy2_liner_bool']['up_df_siiflux_sub'].offset_oiiihb.iloc[valid_sy2])

                    mn_x_sy2 = np.mean(distro_sy2_x)
                    mn_y_sy2 = np.mean(distro_sy2_y)
                    med_x_sy2 = np.median(distro_sy2_x)
                    med_y_sy2 = np.median(distro_sy2_y)
                    sy2_set.append([i,j, mid_x[i], mid_y[j], mn_x_sy2, 
                                     mn_y_sy2, med_x_sy2, med_y_sy2, 
                                     distro_sy2_x, distro_sy2_y, match_dist_sy2, valid_sy2])
                else:
                    sy2_set.append([i,j,mid_x[i], mid_y[j],0, 0, 0, 0, [], [],[], []])
                if valid_liner_high_ssfr.size > 10:
                    distro_liner_high_x = np.copy(self.filts['delta_ssfr']['up_down_sy2_liner_bool_siiflux_sub_df'].offset_niiha.iloc[valid_liner_high_ssfr])
                    distro_liner_high_y = np.copy(self.filts['delta_ssfr']['up_down_sy2_liner_bool_siiflux_sub_df'].offset_oiiihb.iloc[valid_liner_high_ssfr])

                    mn_x_liner_high = np.mean(distro_liner_high_x)
                    mn_y_liner_high = np.mean(distro_liner_high_y)
                    med_x_liner_high = np.median(distro_liner_high_x)
                    med_y_liner_high = np.median(distro_liner_high_y)
                    liner_high_set.append([i,j, mid_x[i], mid_y[j], mn_x_liner_high, 
                                     mn_y_liner_high, med_x_liner_high, med_y_liner_high, 
                                     distro_liner_high_x, distro_liner_high_y, match_dist_liner_high, valid_liner_high_ssfr])
                else:
                    liner_high_set.append([i,j,mid_x[i], mid_y[j],0, 0, 0, 0, [], [],[], []])
                if valid_liner_low_ssfr.size > 10:
                    distro_liner_low_x = np.copy(self.filts['delta_ssfr']['down_down_sy2_liner_bool_siiflux_sub_df'].offset_niiha.iloc[valid_liner_low_ssfr])
                    distro_liner_low_y = np.copy(self.filts['delta_ssfr']['down_down_sy2_liner_bool_siiflux_sub_df'].offset_oiiihb.iloc[valid_liner_low_ssfr])

                    mn_x_liner_low = np.mean(distro_liner_low_x)
                    mn_y_liner_low = np.mean(distro_liner_low_y)
                    med_x_liner_low = np.median(distro_liner_low_x)
                    med_y_liner_low = np.median(distro_liner_low_y)
                    liner_low_set.append([i,j, mid_x[i], mid_y[j], mn_x_liner_low, 
                                     mn_y_liner_low, med_x_liner_low, med_y_liner_low, 
                                     distro_liner_low_x, distro_liner_low_y, match_dist_liner_low, valid_liner_low_ssfr])
                else:
                    liner_low_set.append([i,j,mid_x[i], mid_y[j],0, 0, 0, 0, [], [],[], []])
                                             
                
                if valid_sing.size > 10:
                    distro_full_x = np.copy(self.fullagn_df.offset_niiha.iloc[full_agn[valid_sing]])
                    distro_full_y = np.copy(self.fullagn_df.offset_oiiihb.iloc[full_agn[valid_sing]])

                    mn_x_full = np.mean(distro_full_x)
                    mn_y_full = np.mean(distro_full_y)
                    med_x_full = np.median(distro_full_x)
                    med_y_full = np.median(distro_full_y)
                    sing_set.append([i,j, mid_x[i], mid_y[j], mn_x_full, 
                                     mn_y_full, med_x_full, med_y_full, 
                                     distro_full_x, distro_full_y, match_dist_sing, valid_sing])

                else:
                    sing_set.append([i,j,mid_x[i], mid_y[j],0, 0, 0, 0, [], [],[], []])
                
                if valid_high.size > 10:
                    distro_high_x = np.copy(self.filts['delta_ssfr']['up_df'].offset_niiha.iloc[valid_high])
                    distro_high_y = np.copy(self.filts['delta_ssfr']['up_df'].offset_oiiihb.iloc[valid_high])

                    mn_x_high = np.mean(distro_high_x)
                    mn_y_high = np.mean(distro_high_y)
                    med_x_high = np.median(distro_high_x)
                    med_y_high = np.median(distro_high_y)
                    high_set.append([i,j, mid_x[i], mid_y[j], mn_x_high, 
                                     mn_y_high, med_x_high, med_y_high, 
                                     distro_high_x, distro_high_y, match_dist_sing, valid_high])
                else:
                    high_set.append([i,j,mid_x[i], mid_y[j],0, 0, 0, 0, [], [],[], []])
               
                if valid_low.size > 10:
                    distro_low_x = np.copy(self.filts['delta_ssfr']['down_df'].offset_niiha.iloc[valid_low])
                    distro_low_y = np.copy(self.filts['delta_ssfr']['down_df'].offset_oiiihb.iloc[valid_low])             
                    mn_x_low = np.mean(distro_low_x)
                    mn_y_low = np.mean(distro_low_y)
                    med_x_low = np.median(distro_low_x)
                    med_y_low = np.median(distro_low_y)
                    low_set.append([i,j, mid_x[i], mid_y[j], mn_x_low, mn_y_low, 
                                    med_x_low, med_y_low, distro_low_x, distro_low_y, 
                                    match_dist_low, valid_low])
                else:
                    low_set.append([i,j,mid_x[i], mid_y[j],0, 0, 0, 0, [], [],[], []])
                
                

                if valid_bpt.size >10:
                    distro_bpt_x = np.copy(self.agn_df.offset_niiha.iloc[valid_bpt])
                    distro_bpt_y = np.copy(self.agn_df.offset_oiiihb.iloc[valid_bpt])
                    mn_x_bpt = np.mean(distro_bpt_x)
                    mn_y_bpt = np.mean(distro_bpt_y)
                    med_x_bpt = np.median(distro_bpt_x)
                    med_y_bpt = np.median(distro_bpt_y)
                    bpt_set.append([i,j, mid_x[i], mid_y[j], mn_x_bpt, mn_y_bpt, med_x_bpt, med_y_bpt, distro_bpt_x, distro_bpt_y, match_dist_bpt, valid_bpt])

                else:
                    bpt_set.append([i,j,mid_x[i], mid_y[j],0, 0, 0, 0, [], [],[], []])
                    #goodinds_pt.append(i*j)
                
                if valid_plus.size > 10:
                    distro_plus_x = np.copy(self.agnplus_df.offset_niiha.iloc[valid_plus])
                    distro_plus_y = np.copy(self.agnplus_df.offset_oiiihb.iloc[valid_plus])

                    mn_x_plus = np.mean(distro_plus_x)
                    mn_y_plus = np.mean(distro_plus_y)
                    med_x_plus = np.median(distro_plus_x)
                    med_y_plus = np.median(distro_plus_y)
                    plus_set.append([i,j, mid_x[i], mid_y[j],mn_x_plus, mn_y_plus, med_x_plus, med_y_plus, distro_plus_x, distro_plus_y, match_dist_plus, valid_plus])

                else:
                    plus_set.append([i,j,mid_x[i], mid_y[j], 0, 0, 0, 0, [], [], [], []])
                
                
                if valid_neither.size >10:
                    distro_neither_x = np.copy(self.neither_agn_df.offset_niiha.iloc[valid_neither])
                    distro_neither_y = np.copy(self.neither_agn_df.offset_oiiihb.iloc[valid_neither])

                    mn_x_neither = np.mean(distro_neither_x)
                    mn_y_neither = np.mean(distro_neither_y)
                    med_x_neither = np.median(distro_neither_x)
                    med_y_neither = np.median(distro_neither_y)
                    neither_set.append([i,j, mid_x[i], mid_y[j], mn_x_neither, mn_y_neither, med_x_neither, med_y_neither, distro_neither_x, distro_neither_y, match_dist_neither, valid_neither])
                else:
                    neither_set.append([i,j,mid_x[i], mid_y[j], 0, 0, 0, 0, [], [], [],[]])


                coordpairs[i*(len(mid_x)-1)+j] = ((i,j,mid_x[i], mid_y[j]))
                #print(i*(len(mid_x)-1)+j+1)



        self.coordpairs = coordpairs
        self.bpt_set = bpt_set
        self.plus_set = plus_set
        self.neither_set = neither_set
        self.sing_set = sing_set
        self.low_set = low_set
        self.high_set = high_set 
        self.sy2_set = sy2_set
        self.clustering_set = clustering_set
        self.clust1_set = clust1_set
        self.clust2_set = clust2_set
        self.clust3_set = clust3_set
        
        self.liner_high_set = liner_high_set
        self.liner_low_set = liner_low_set
        self.mesh_x = mesh_x
        self.mesh_y = mesh_y
        self.binx_vals = xvals
        self.biny_vals = yvals

    def correct_av(self,reg, x_test, av_balm, hb_sn):
        #x_test = av_gsw.transpose()
        av_balm_fixed = []
        for i in range(len(hb_sn)):

            if hb_sn[i] <10:
                x_samp = x_test[i].reshape(1,-1)
                av_fix = np.float64(reg.predict(x_samp))
                if av_fix<0:
                    av_fix=0
            elif av_balm[i] <0:
                av_fix = 0
            elif av_balm[i]>3:
                av_fix = 3
            else:
                av_fix = av_balm[i]
            av_balm_fixed.append(av_fix)
        return np.array(av_balm_fixed)


    def get_highsn_match_only_d4000(self, agn_inds, sf_inds, sf_plus_inds, 
                              agnplus_inds, sncut=2, load=False, fname='', with_av=True):
        '''
        Matches BPT AGN to BPT SF, BPT+ SF, unclassified of similar d4000, M*, Mfib, z
        Ensures that the match has a high S/N in subtracted fluxes
        '''
        self.bptsf_inds = sf_inds
        self.bptagn_inds = agn_inds
        self.bptplus_sf_inds = sf_plus_inds
        self.bptplus_agn_inds = agnplus_inds 
        if with_av:
            subfold = 'matching_with_av/'
        else:
            subfold = 'matching_without_av/'        
        if not load:
            #setting up lists/arrays for storing info
            agns_selfmatch = []
            agns_selfmatch_other = []
            
            agnsplus_selfmatch = []
            agnsplus_selfmatch_other = []
            
            #lists for storing agn indices
            agns = []
            agns_plus = []
            neither_agn = []
            #lists for storing match indices         
            sfs = []
            sfs_plus = []
            neither_matches = []
            numpassed = np.zeros((3, len(agn_inds)))
            numpassed_best = []
            mininds = np.zeros((3, len(agn_inds)))
            mindists = np.zeros((3, len(agn_inds)))
            minids = np.zeros((3, len(agn_inds)), dtype=np.int64)
            mininds_agn = np.zeros((2, len(agn_inds)))
            mindists_agn = np.zeros((2, len(agn_inds)))
            minids_agn = np.zeros((2, len(agn_inds)), dtype=np.int64)
            
            for i, agn_ind in enumerate(agn_inds):
                if i%100 == 0:
                    print(i)
                    t = time.localtime()
                    current_time = time.strftime("%H:%M:%S", t)
                    print(current_time)
                #for self-matching, don't want to compare to itself
                otheragns = np.where(agn_inds != agn_ind)[0]
                '''
                BPT AGN self-matching
                '''
                #computing differences for self matching agn
                diffd4000_agn = (self.bpt_eldiag.d4000[agn_ind]-self.bpt_eldiag.d4000[agn_inds][otheragns])**2
                diffmass_agn = (self.bpt_eldiag.mass[agn_ind]-self.bpt_eldiag.mass[agn_inds][otheragns])**2
                difffibmass_agn = (self.bpt_eldiag.fibmass[agn_ind]-self.bpt_eldiag.fibmass[agn_inds][otheragns])**2
                diffz_agn = (self.bpt_eldiag.z[agn_ind]-self.bpt_eldiag.z[agn_inds][otheragns])**2/np.std(redshift_m2)
                diff_av_agn = (self.bpt_eldiag.av_gsw[agn_ind]-self.bpt_eldiag.av_gsw[agn_inds][otheragns])**2
                diffs_agn = np.sqrt(diffd4000_agn+diffmass_agn+difffibmass_agn+diffz_agn+diff_av_agn)
                
                mindiff_ind_agn = np.where(diffs_agn == np.min(diffs_agn) )[0]
                '''
                BPT Plus self-agn matching
                '''
                #computing differences for self matching agn to bpt+ agn
                diffd4000_agnplus = (self.bpt_eldiag.d4000[agn_ind]-self.plus_eldiag.d4000[agnplus_inds])**2
                diffmass_agnplus = (self.bpt_eldiag.mass[agn_ind]-self.plus_eldiag.mass[agnplus_inds])**2
                difffibmass_agnplus = (self.bpt_eldiag.fibmass[agn_ind]-self.plus_eldiag.fibmass[agnplus_inds])**2
                diffz_agnplus = (self.bpt_eldiag.z[agn_ind]-self.plus_eldiag.z[agnplus_inds])**2/np.std(redshift_m2)
                diff_av_agnplus = (self.bpt_eldiag.av_gsw[agn_ind]-self.plus_eldiag.av_gsw[agnplus_inds])**2
                diffs_agnplus = np.sqrt(diffd4000_agnplus+diffmass_agnplus+difffibmass_agnplus+diffz_agnplus+diff_av_agnplus)
                mindiff_ind_agnplus = np.where(diffs_agnplus == np.min(diffs_agnplus) )[0]
                '''
                BPT AGN-BPT SF matching
                '''
                #computing differences for bpt SF 
                diffd4000_bpt = (self.bpt_eldiag.d4000[agn_ind] - self.bpt_eldiag.d4000[sf_inds])**2
                diffmass_bpt = (self.bpt_eldiag.mass[agn_ind] - self.bpt_eldiag.mass[sf_inds])**2
                difffibmass_bpt = (self.bpt_eldiag.fibmass[agn_ind] - self.bpt_eldiag.fibmass[sf_inds])**2
                diffz_bpt = (self.bpt_eldiag.z[agn_ind]-self.bpt_eldiag.z[sf_inds])**2/np.std(redshift_m2)
                diff_av_bpt = (self.bpt_eldiag.av_gsw[agn_ind]-self.bpt_eldiag.av_gsw[sf_inds])**2
                diffs_bpt = np.sqrt(diffd4000_bpt+diffmass_bpt+difffibmass_bpt+diffz_bpt+diff_av_bpt)
                #mindiff_ind_bpt = np.where(diffs_bpt == np.min(diffs_bpt) )[0]
                bptdistrat = (np.array(cosmo.luminosity_distance(self.bpt_eldiag.z[sf_inds]))/np.array(cosmo.luminosity_distance(self.eldiag.z[agn_ind])))**2 

                oiiiflux_sub_bpt = self.bpt_eldiag.oiiiflux[agn_ind]-self.bpt_eldiag.oiiiflux[sf_inds]*bptdistrat
                niiflux_sub_bpt = self.bpt_eldiag.niiflux[agn_ind]-self.bpt_eldiag.niiflux[sf_inds]*bptdistrat
                hbetaflux_sub_bpt =self.bpt_eldiag.hbetaflux[agn_ind]-self.bpt_eldiag.hbetaflux[sf_inds]*bptdistrat
                halpflux_sub_bpt  = self.bpt_eldiag.halpflux[agn_ind]- self.bpt_eldiag.halpflux[sf_inds]*bptdistrat
                
                oiiiflux_err_sub_bpt = np.sqrt(self.bpt_eldiag.oiii_err[agn_ind]**2 + (self.bpt_eldiag.oiii_err[sf_inds]*bptdistrat)**2)
                niiflux_err_sub_bpt =  np.sqrt(self.bpt_eldiag.nii_err[agn_ind]**2 + (self.bpt_eldiag.nii_err[sf_inds]*bptdistrat)**2)
                hbetaflux_err_sub_bpt = np.sqrt(self.bpt_eldiag.hbeta_err[agn_ind]**2 + (self.bpt_eldiag.hbeta_err[sf_inds]*bptdistrat)**2)
                halpflux_err_sub_bpt  = np.sqrt(self.bpt_eldiag.halp_err[agn_ind]**2 + (self.bpt_eldiag.halp_err[sf_inds]*bptdistrat)**2)
                
                oiiiflux_sn_sub_bpt = oiiiflux_sub_bpt/ oiiiflux_err_sub_bpt
                niiflux_sn_sub_bpt = niiflux_sub_bpt/niiflux_err_sub_bpt
                hbetaflux_sn_sub_bpt = hbetaflux_sub_bpt/hbetaflux_err_sub_bpt
                halpflux_sn_sub_bpt = halpflux_sub_bpt/halpflux_err_sub_bpt
                diffs_bpt_sort = np.argsort(diffs_bpt)    
                
                inds_high_sn_bpt = np.where((oiiiflux_sn_sub_bpt[diffs_bpt_sort]>sncut) & (niiflux_sn_sub_bpt[diffs_bpt_sort]>sncut) &
                                            (hbetaflux_sn_sub_bpt[diffs_bpt_sort]>sncut) &(halpflux_sn_sub_bpt[diffs_bpt_sort] >sncut) )[0]
                #print(inds_high_sn_bpt)
                #print(inds_high_sn_bpt)
                if len(inds_high_sn_bpt) >0:    
                    mindiff_ind_bpt = diffs_bpt_sort[inds_high_sn_bpt[0]]
                    n_pass_bpt = len(inds_high_sn_bpt)
                else:
                    mindiff_ind_bpt = -1
                    n_pass_bpt = len(diffs_bpt_sort)
                #computing differences for bpt+ SF 
                '''
                BPT AGN-BPT PLUS SF matching
                '''
                diffd4000_bptplus = (self.bpt_eldiag.d4000[agn_ind] - self.plus_eldiag.d4000[sf_plus_inds])**2
                diffmass_bptplus = (self.bpt_eldiag.mass[agn_ind] - self.plus_eldiag.mass[sf_plus_inds])**2
                difffibmass_bptplus = (self.bpt_eldiag.fibmass[agn_ind] - self.plus_eldiag.fibmass[sf_plus_inds])**2
                diffz_bptplus = (self.bpt_eldiag.z[agn_ind]-self.plus_eldiag.z[sf_plus_inds])**2/np.std(redshift_m2)
                diff_av_bptplus = (self.bpt_eldiag.av_gsw[agn_ind]-self.plus_eldiag.av_gsw[sf_plus_inds])**2
                diffs_bptplus = np.sqrt(diffd4000_bptplus+diffmass_bptplus+difffibmass_bptplus+diffz_bptplus+diff_av_bptplus)

                bptdistrat_plus = (np.array(cosmo.luminosity_distance(self.plus_eldiag.z[sf_plus_inds]))/np.array(cosmo.luminosity_distance(self.bpt_eldiag.z[agn_ind])))**2 

                oiiiflux_sub_plus = self.bpt_eldiag.oiiiflux[agn_ind]-self.plus_eldiag.oiiiflux[sf_plus_inds]*bptdistrat_plus
                niiflux_sub_plus = self.bpt_eldiag.niiflux[agn_ind]-self.plus_eldiag.niiflux[sf_plus_inds]*bptdistrat_plus
                hbetaflux_sub_plus =self.bpt_eldiag.hbetaflux[agn_ind]-self.plus_eldiag.hbetaflux[sf_plus_inds]*bptdistrat_plus
                halpflux_sub_plus  = self.bpt_eldiag.halpflux[agn_ind]- self.plus_eldiag.halpflux[sf_plus_inds]*bptdistrat_plus
                
                oiiiflux_err_sub_plus = np.sqrt(self.bpt_eldiag.oiii_err[agn_ind]**2 + (self.plus_eldiag.oiii_err[sf_plus_inds]*bptdistrat_plus)**2)
                niiflux_err_sub_plus =  np.sqrt(self.bpt_eldiag.nii_err[agn_ind]**2 + (self.plus_eldiag.nii_err[sf_plus_inds]*bptdistrat_plus)**2)
                hbetaflux_err_sub_plus = np.sqrt(self.bpt_eldiag.hbeta_err[agn_ind]**2 + (self.plus_eldiag.hbeta_err[sf_plus_inds]*bptdistrat_plus)**2)
                halpflux_err_sub_plus  = np.sqrt(self.bpt_eldiag.halp_err[agn_ind]**2 + (self.plus_eldiag.halp_err[sf_plus_inds]*bptdistrat_plus)**2)
                
                oiiiflux_sn_sub_plus = oiiiflux_sub_plus/ oiiiflux_err_sub_plus
                niiflux_sn_sub_plus = niiflux_sub_plus/niiflux_err_sub_plus
                hbetaflux_sn_sub_plus = hbetaflux_sub_plus/hbetaflux_err_sub_plus
                halpflux_sn_sub_plus = halpflux_sub_plus/halpflux_err_sub_plus
  
                diffs_bptplus_sort = np.argsort(diffs_bptplus)    
                
                inds_high_sn_bptplus = np.where((oiiiflux_sn_sub_plus[diffs_bptplus_sort]>sncut) & (niiflux_sn_sub_plus[diffs_bptplus_sort]>sncut) &
                                            (hbetaflux_sn_sub_plus[diffs_bptplus_sort]>sncut) &(halpflux_sn_sub_plus[diffs_bptplus_sort] >sncut)&
                                                (self.plus_eldiag.oiii_err[diffs_bptplus_sort]*bptdistrat_plus[diffs_bptplus_sort] != 0 ) &
                                                (self.plus_eldiag.hbeta_err[diffs_bptplus_sort]*bptdistrat_plus[diffs_bptplus_sort] != 0 ) )[0]
                #print(inds_high_sn_bptplus)

                if len(inds_high_sn_bptplus) >0:    
                    mindiff_ind_bptplus = diffs_bptplus_sort[inds_high_sn_bptplus[0]]
                    n_pass_plus = len(inds_high_sn_bptplus)
                else:
                    mindiff_ind_bptplus = -1
                    n_pass_plus = len(diffs_bptplus_sort)
                '''
                BPT AGN-Neither match
                '''
                #computing differences for unclassifiable 
                diffd4000_neither = (self.bpt_eldiag.d4000[agn_ind] - self.neither_eldiag.d4000_neither)**2
                diffmass_neither = (self.bpt_eldiag.mass[agn_ind] - self.neither_eldiag.mass_neither)**2
                difffibmass_neither = (self.bpt_eldiag.fibmass[agn_ind] - self.neither_eldiag.fibmass_neither)**2
                diffz_neither = (self.bpt_eldiag.z[agn_ind]-self.neither_eldiag.z_neither)**2/np.std(redshift_m2)
                diff_av_agn_neither = (self.bpt_eldiag.av_gsw[agn_ind]-self.neither_eldiag.av_gsw_neither)**2
                diffs_neither = np.sqrt(diffd4000_neither+diffmass_neither+difffibmass_neither+diffz_neither+diff_av_agn_neither)

                bptdistrat_neither = (np.array(cosmo.luminosity_distance(self.neither_eldiag.z))/np.array(cosmo.luminosity_distance(self.bpt_eldiag.z[agn_ind])))**2 
                        

                oiiiflux_sub_neither = self.bpt_eldiag.oiiiflux[agn_ind]-self.neither_eldiag.oiiiflux_neither*bptdistrat_neither
                niiflux_sub_neither = self.bpt_eldiag.niiflux[agn_ind]-self.neither_eldiag.niiflux_neither*bptdistrat_neither
                hbetaflux_sub_neither =self.bpt_eldiag.hbetaflux[agn_ind]-self.neither_eldiag.hbetaflux_neither*bptdistrat_neither
                halpflux_sub_neither  = self.bpt_eldiag.halpflux[agn_ind]- self.neither_eldiag.halpflux_neither*bptdistrat_neither
                
                oiiiflux_err_sub_neither = np.sqrt(self.bpt_eldiag.oiii_err[agn_ind]**2 + (self.neither_eldiag.oiii_err*bptdistrat_neither)**2)
                niiflux_err_sub_neither =  np.sqrt(self.bpt_eldiag.nii_err[agn_ind]**2 + (self.neither_eldiag.nii_err*bptdistrat_neither)**2)
                hbetaflux_err_sub_neither = np.sqrt(self.bpt_eldiag.hbeta_err[agn_ind]**2 + (self.neither_eldiag.hbeta_err*bptdistrat_neither)**2)
                halpflux_err_sub_neither  = np.sqrt(self.bpt_eldiag.halp_err[agn_ind]**2 + (self.neither_eldiag.halp_err*bptdistrat_neither)**2)
                
                oiiiflux_sn_sub_neither = oiiiflux_sub_neither/ oiiiflux_err_sub_neither
                niiflux_sn_sub_neither = niiflux_sub_neither/niiflux_err_sub_neither
                hbetaflux_sn_sub_neither = hbetaflux_sub_neither/hbetaflux_err_sub_neither
                halpflux_sn_sub_neither = halpflux_sub_neither/halpflux_err_sub_neither
  
                diffs_neither_sort = np.argsort(diffs_neither)    
  
                inds_high_sn_neither = np.where((oiiiflux_sn_sub_neither[diffs_neither_sort]>sncut) & (niiflux_sn_sub_neither[diffs_neither_sort]>sncut) &
                                                (hbetaflux_sn_sub_neither[diffs_neither_sort]>sncut) &(halpflux_sn_sub_neither[diffs_neither_sort] >sncut) &
                                                (self.neither_eldiag.oiii_err[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) &
                                                (self.neither_eldiag.hbeta_err[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) &
                                                (self.neither_eldiag.halp_err[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) &
                                                (self.neither_eldiag.nii_err[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) )[0]
                if len(inds_high_sn_neither)>0:
                    mindiff_ind_neither = diffs_neither_sort[inds_high_sn_neither[0]]
                    n_pass_neither = len(inds_high_sn_neither)
                else:
                    mindiff_ind_neither = -1
                    n_pass_neither = len(diffs_neither_sort)
                #print(inds_high_sn_neither)                            
                mindiffinds = [mindiff_ind_bpt, mindiff_ind_bptplus, mindiff_ind_neither]
                
                
                mindist_out = [diffs_bpt[mindiff_ind_bpt], diffs_bptplus[mindiff_ind_bptplus],  
                               diffs_neither[mindiff_ind_neither]]
                bad_dists = np.where(mindiffinds == -1)[0]
                if bad_dists.size >0:
                    mindist_out[bad_dists]=99999
                #print(mindist_out)
                n_pass_out = [n_pass_bpt, n_pass_plus, n_pass_neither]
                 
                #assigning the ids, inds, dists to be saved
                minid_out =[ self.bpt_eldiag.ids[sf_inds[mindiff_ind_bpt]],  
                            self.plus_eldiag.ids[sf_plus_inds[mindiff_ind_bptplus]],
                            self.neither_eldiag.ids[mindiff_ind_neither]]
                minind_out =[ sf_inds[mindiff_ind_bpt],  
                            sf_plus_inds[mindiff_ind_bptplus],
                            mindiff_ind_neither]
                #assigning the ids, inds, dists to be saved for self-matching

                minid_outagn = [self.bpt_eldiag.ids[agn_inds[mindiff_ind_agn]],  
                            self.plus_eldiag.ids[agnplus_inds[mindiff_ind_agnplus]]]
                minind_outagn =[ agn_inds[mindiff_ind_agn],  
                            agnplus_inds[mindiff_ind_bptplus]]
                mindist_outagn = [diffs_agn[mindiff_ind_agn], diffs_agnplus[mindiff_ind_agnplus]]
                numpassed[:, i] = n_pass_out
                
                #saving the relevant info 
                mindists[:, i] = mindist_out
                minids[:, i] = minid_out
                mininds[:, i] = minind_out
    
                mindists_agn[:, i] = mindist_outagn
                minids_agn[:, i] = minid_outagn
                mininds_agn[:, i] = minind_outagn
                mindist_ind = np.int(np.where(mindist_out == np.min(mindist_out))[0])
                #getting the best one, 0 = BPT SF, 1 = BPT SF+, 2 = Unclassifiable
                if mindist_ind ==0:
                    sfs.append(sf_inds[mindiff_ind_bpt])
                    agns.append(agn_ind)
                    numpassed_best.append(n_pass_bpt)
                    
                elif mindist_ind==1:
                    sfs_plus.append(sf_plus_inds[mindiff_ind_bptplus])
                    agns_plus.append(agn_ind)
                    numpassed_best.append(n_pass_plus)
                    
                else:
                    neither_matches.append(mindiff_ind_neither)
                    neither_agn.append(agn_ind)
                    numpassed_best.append(n_pass_neither)
                    
                mindistagn_ind = np.int(np.where(mindist_outagn==np.min(mindist_outagn))[0])
                if mindistagn_ind ==0:
                    agns_selfmatch.append(agn_ind)
                    agns_selfmatch_other.append(mindiff_ind_agn[0])
                else:
                    agnsplus_selfmatch.append(agn_ind)
                    agnsplus_selfmatch_other.append(agnplus_inds[mindiff_ind_agnplus[0]])  
                            

            #converting lists to arrays and saving to class attributes
            self.numpassed = numpassed
            self.numpassed_best = np.array(numpassed_best)
            self.agns = np.array(agns)
            self.sfs = np.array(sfs)
            self.agns_plus = np.array(agns_plus)
            self.sfs_plus = np.array(sfs_plus)
            self.neither_matches = np.array(neither_matches)
            self.neither_agn = np.array(neither_agn)
            self.mindists = mindists
            self.mininds= mininds
            self.minids = minids
            self.mindists_best = np.min(self.mindists, axis=0)
    
            self.agns_selfmatch = np.array(agns_selfmatch)
            self.agns_selfmatch_other = np.array(agns_selfmatch_other)
            self.agnsplus_selfmatch = np.array(agnsplus_selfmatch)
            self.agnsplus_selfmatch_other = np.array(agnsplus_selfmatch_other)
            self.mindists_agn = mindists_agn
            self.minids_agn = minids_agn
            self.mininds_agn = mininds_agn
            self.mindistsagn_best = np.min(self.mindists_agn, axis=0)
           
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_n_pass_best_highsn'+str(sncut)+fname+'.txt',self.numpassed_best, fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_n_pass_highsn'+str(sncut)+fname+'.txt',self.numpassed.transpose(), fmt='%6.d')
            
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_agns_selfmatch_highsn'+str(sncut)+fname+'.txt',self.agns_selfmatch, fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_agns_selfmatch_other_highsn'+str(sncut)+fname+'.txt',self.agns_selfmatch_other, fmt='%6.d')
            
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_agnsplus_selfmatch_highsn'+str(sncut)+fname+'.txt',self.agnsplus_selfmatch, fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_agnsplus_selfmatch_other_highsn'+str(sncut)+fname+'.txt',self.agnsplus_selfmatch_other, fmt='%6.d')
            
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_agns_highsn'+str(sncut)+fname+'.txt',self.agns, fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_sfs_highsn'+str(sncut)+fname+'.txt',self.sfs, fmt='%6.d')
            
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_agns_plus_highsn'+str(sncut)+fname+'.txt',self.agns_plus, fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_sfs_plus_highsn'+str(sncut)+fname+'.txt',self.sfs_plus, fmt='%6.d')
            
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_neither_matches_highsn'+str(sncut)+fname+'.txt',self.neither_matches, fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_neither_agn_highsn'+str(sncut)+fname+'.txt',self.neither_agn, fmt='%6.d')
            
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_mindists_best_highsn'+str(sncut)+fname+'.txt',self.mindists_best, fmt='%8.6f')
            
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_mindists_highsn'+str(sncut)+fname+'.txt',self.mindists.transpose(), fmt='%8.6f')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_mininds_highsn'+str(sncut)+fname+'.txt',self.mininds.transpose(), fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_minids_highsn'+str(sncut)+fname+'.txt',self.minids.transpose(), fmt='%18.d')
            
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_mindists_agn_highsn'+str(sncut)+fname+'.txt',self.mindists_agn.transpose(), fmt='%8.6f')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_mininds_agn_highsn'+str(sncut)+fname+'.txt',self.mininds_agn.transpose(), fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_minids_agn_highsn'+str(sncut)+fname+'.txt',self.minids_agn.transpose(), fmt='%18.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_mindistsagn_best_highsn'+str(sncut)+fname+'.txt',self.mindistsagn_best, fmt='%8.6f')
                
        else:
            #once the matching is already done just need to load items in
            self.numpassed =  np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_n_pass_highsn'+str(sncut)+fname+'.txt',dtype=np.int64, unpack=True)
            
            self.numpassed_best = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_n_pass_best_highsn'+str(sncut)+fname+'.txt',dtype=np.int64)
            
            self.agns_selfmatch = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_agns_selfmatch_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            self.agns_selfmatch_other = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_agns_selfmatch_other_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            
            self.agnsplus_selfmatch = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_agnsplus_selfmatch_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            self.agnsplus_selfmatch_other = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_agnsplus_selfmatch_other_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            
            self.agns = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_agns_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            self.sfs = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_sfs_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            
            self.agns_plus = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_agns_plus_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            self.sfs_plus = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_sfs_plus_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)

            self.neither_agn = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_neither_agn_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)            
            self.neither_matches = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_neither_matches_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            
            
            self.mindists = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_mindists_highsn'+str(sncut)+fname+'.txt', unpack=True)
            self.mininds = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_mininds_highsn'+str(sncut)+fname+'.txt', dtype=np.int64, unpack=True)
            self.minids = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_minids_highsn'+str(sncut)+fname+'.txt', dtype=np.int64, unpack=True)
            self.mindists_best = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_mindists_best_highsn'+str(sncut)+fname+'.txt')
            
            self.mindists_agn = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_mindists_agn_highsn'+str(sncut)+fname+'.txt', unpack=True)
            self.minids_agn = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_mininds_agn_highsn'+str(sncut)+fname+'.txt', dtype=np.int64, unpack=True)
            self.mininds_agn = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_minids_agn_highsn'+str(sncut)+fname+'.txt', dtype=np.int64, unpack=True)
            self.mindistsagn_best = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_mindistsagn_best_highsn'+str(sncut)+fname+'.txt')
            
            agn_dist_inds = []
            agn_plus_dist_inds = [] 
            agn_neither_dist_inds = []
            agn_selfdist_inds = []
            agn_selfplus_dist_inds = [] 
            for i in range(len(self.mindists_best)):
                mn = np.where(self.mindists[:,i] == self.mindists_best[i])[0]
                if len(mn) ==1:
                        
                    if mn == 0:
                        agn_dist_inds.append(i)
                    elif mn==1: 
                        agn_plus_dist_inds.append(i)
                    else:
                        agn_neither_dist_inds.append(i)
                else:
                    if 0 in mn:
                        agn_dist_inds.append(i)
                    elif 1 in mn:
                        agn_plus_dist_inds.append(i)
                    else:
                        agn_neither_dist_inds.append(i)
                mnself = np.where(self.mindists_agn[:,i] == self.mindistsagn_best[i])[0]
                if len(mnself)==1:
                    if mnself==0:
                        agn_selfdist_inds.append(i)
                    else:
                        agn_selfplus_dist_inds.append(i)
                else:
                    if 0 in mnself:
                        agn_selfdist_inds.append(i)
                    else:
                        agn_selfplus_dist_inds.append(i)
                
            self.agn_dist_inds = np.array(agn_dist_inds)
            self.agn_plus_dist_inds = np.array(agn_plus_dist_inds)
            self.agn_neither_dist_inds = np.array(agn_neither_dist_inds)
            self.agn_ind_mapping = combine_arrs([self.agn_dist_inds, self.agn_plus_dist_inds, self.agn_neither_dist_inds])

            self.agn_selfdist_inds = np.array(agn_selfdist_inds)
            self.agn_selfplus_dist_inds = np.array(agn_selfplus_dist_inds)
            self.agn_selfind_mapping = combine_arrs([self.agn_selfdist_inds, self.agn_selfplus_dist_inds])


            self.mindists_best_sing_ord = combine_arrs([self.mindists_best[self.agn_dist_inds], self.mindists_best[self.agn_plus_dist_inds], self.mindists_best[self.agn_neither_dist_inds] ])




    def get_highsn_match_only_d4000_hdelta(self, agn_inds, sf_inds, sf_plus_inds, 
                              agnplus_inds, sncut=2, load=False, fname='', with_av=True):
        '''
        Matches BPT AGN to BPT SF, BPT+ SF, unclassified of similar d4000, M*, Mfib, z
        Ensures that the match has a high S/N in subtracted fluxes
        '''
        self.bptsf_inds = sf_inds
        self.bptagn_inds = agn_inds
        self.bptplus_sf_inds = sf_plus_inds
        self.bptplus_agn_inds = agnplus_inds 
        if with_av:
            subfold = 'matching_with_av/'
        else:
            subfold = 'matching_mwithout_av/'        
        if not load:
            #setting up lists/arrays for storing info
            agns_selfmatch = []
            agns_selfmatch_other = []
            
            agnsplus_selfmatch = []
            agnsplus_selfmatch_other = []
            
            #lists for storing agn indices
            agns = []
            agns_plus = []
            neither_agn = []
            
            second_agns = []
            second_agns_plus = []
            second_neither_agn = []
            #lists for storing match indices         
            sfs = []
            sfs_plus = []
            neither_matches = []
            
            second_sfs = []
            second_sfs_plus = []
            second_neither_matches = []
            
            numpassed = np.zeros((3, len(agn_inds)))
            numpassed_best = []
            mininds = np.zeros((3, len(agn_inds)))
            second_mininds = np.zeros((3, len(agn_inds)))

            mindists = np.zeros((3, len(agn_inds)))
            second_mindists = np.zeros((3, len(agn_inds)))

            minids = np.zeros((3, len(agn_inds)), dtype=np.int64)
            mininds_agn = np.zeros((2, len(agn_inds)))

            mindists_agn = np.zeros((2, len(agn_inds)))
            minids_agn = np.zeros((2, len(agn_inds)), dtype=np.int64)
            
            
            for i, agn_ind in enumerate(agn_inds):
                if i%100 == 0:
                    print(i)
                    t = time.localtime()
                    current_time = time.strftime("%H:%M:%S", t)
                    print(current_time)
                #for self-matching, don't want to compare to itself
                otheragns = np.where(agn_inds != agn_ind)[0]
                '''
                BPT AGN self-matching
                '''
                #computing differences for self matching agn
                diffd4000_agn = (self.bpt_eldiag.d4000.iloc[agn_ind]-self.bpt_eldiag.d4000.iloc[agn_inds[otheragns]])**2#/np.var(sdssobj.alld4000)
                diffhd_agn = (self.bpt_eldiag.hdelta_lick.iloc[agn_ind]-self.bpt_eldiag.hdelta_lick.iloc[agn_inds[otheragns]])**2#/np.var(sdssobj.hdelta_lick)
                diffsfr_agn = (self.bpt_eldiag.sfr.iloc[agn_ind]-self.bpt_eldiag.sfr.iloc[agn_inds[otheragns]])**2
                
                diffmass_agn = (self.bpt_eldiag.mass.iloc[agn_ind]-self.bpt_eldiag.mass.iloc[agn_inds[otheragns]])**2
                difffibmass_agn = (self.bpt_eldiag.fibmass.iloc[agn_ind]-self.bpt_eldiag.fibmass.iloc[agn_inds[otheragns]])**2
                diffz_agn = (self.bpt_eldiag.z.iloc[agn_ind]-self.bpt_eldiag.z.iloc[agn_inds[otheragns]])**2/np.var(redshift_m2)
                diff_av_agn = (self.bpt_eldiag.tauv_cont.iloc[agn_ind]-self.bpt_eldiag.tauv_cont.iloc[agn_inds[otheragns]])**2
                diffs_agn = np.array(np.sqrt(diffsfr_agn+diffd4000_agn+diffhd_agn+diffmass_agn+difffibmass_agn+diffz_agn+diff_av_agn))
                
                mindiff_ind_agn = np.where(diffs_agn == np.min(diffs_agn) )[0]
                '''
                BPT Plus self-agn matching
                '''
                #computing differences for self matching agn to bpt+ agn
                diffd4000_agnplus = (self.bpt_eldiag.d4000.iloc[agn_ind]-self.plus_eldiag.d4000.iloc[agnplus_inds])**2#/np.var(sdssobj.alld4000)
                diffhd_agnplus = (self.bpt_eldiag.hdelta_lick.iloc[agn_ind]-self.plus_eldiag.hdelta_lick.iloc[agnplus_inds])**2#/np.var(sdssobj.hdelta_lick)
                diffsfr_agnplus = (self.bpt_eldiag.sfr.iloc[agn_ind]-self.plus_eldiag.sfr.iloc[agnplus_inds])**2

                diffmass_agnplus = (self.bpt_eldiag.mass.iloc[agn_ind]-self.plus_eldiag.mass.iloc[agnplus_inds])**2
                difffibmass_agnplus = (self.bpt_eldiag.fibmass.iloc[agn_ind]-self.plus_eldiag.fibmass.iloc[agnplus_inds])**2
                diffz_agnplus = (self.bpt_eldiag.z.iloc[agn_ind]-self.plus_eldiag.z.iloc[agnplus_inds])**2/np.var(redshift_m2)
                diff_av_agnplus = (self.bpt_eldiag.tauv_cont.iloc[agn_ind]-self.plus_eldiag.tauv_cont.iloc[agnplus_inds])**2
                diffs_agnplus = np.array(np.sqrt(diffsfr_agnplus+diffd4000_agnplus+diffhd_agnplus+diffmass_agnplus+difffibmass_agnplus+diffz_agnplus+diff_av_agnplus))
                mindiff_ind_agnplus = np.where(diffs_agnplus == np.min(diffs_agnplus) )[0]
                '''
                BPT AGN-BPT SF matching
                '''
                #computing differences for bpt SF 
                diffd4000_bpt = (self.bpt_eldiag.d4000.iloc[agn_ind] - self.bpt_eldiag.d4000.iloc[sf_inds])**2#/np.var(sdssobj.alld4000)
                diffhd_bpt = (self.bpt_eldiag.hdelta_lick.iloc[agn_ind]-self.bpt_eldiag.hdelta_lick.iloc[sf_inds])**2#/np.var(sdssobj.hdelta_lick)
                diffsfr_bpt = (self.bpt_eldiag.sfr.iloc[agn_ind] - self.bpt_eldiag.sfr.iloc[sf_inds])**2

                diffmass_bpt = (self.bpt_eldiag.mass.iloc[agn_ind] - self.bpt_eldiag.mass.iloc[sf_inds])**2
                difffibmass_bpt = (self.bpt_eldiag.fibmass.iloc[agn_ind] - self.bpt_eldiag.fibmass.iloc[sf_inds])**2
                diffz_bpt = (self.bpt_eldiag.z.iloc[agn_ind]-self.bpt_eldiag.z.iloc[sf_inds])**2/np.var(redshift_m2)
                diff_av_bpt = (self.bpt_eldiag.tauv_cont.iloc[agn_ind]-self.bpt_eldiag.tauv_cont.iloc[sf_inds])**2
                diffs_bpt = np.array(np.sqrt(diffsfr_bpt+diffd4000_bpt+diffhd_bpt+diffmass_bpt+difffibmass_bpt+diffz_bpt+diff_av_bpt))
                #mindiff_ind_bpt = np.where(diffs_bpt == np.min(diffs_bpt) )[0]
                bptdistrat = (np.array(cosmo.luminosity_distance(self.bpt_eldiag.z.iloc[sf_inds]))/np.array(cosmo.luminosity_distance(self.bpt_eldiag.z.iloc[agn_ind])))**2 

                oiiiflux_sub_bpt = self.bpt_eldiag.oiiiflux.iloc[agn_ind]-self.bpt_eldiag.oiiiflux.iloc[sf_inds]*bptdistrat
                niiflux_sub_bpt = self.bpt_eldiag.niiflux.iloc[agn_ind]-self.bpt_eldiag.niiflux.iloc[sf_inds]*bptdistrat
                hbetaflux_sub_bpt =self.bpt_eldiag.hbetaflux.iloc[agn_ind]-self.bpt_eldiag.hbetaflux.iloc[sf_inds]*bptdistrat
                halpflux_sub_bpt  = self.bpt_eldiag.halpflux.iloc[agn_ind]- self.bpt_eldiag.halpflux.iloc[sf_inds]*bptdistrat
                
                oiiiflux_err_sub_bpt = np.sqrt(self.bpt_eldiag.oiii_err.iloc[agn_ind]**2 + (self.bpt_eldiag.oiii_err.iloc[sf_inds]*bptdistrat)**2)
                niiflux_err_sub_bpt =  np.sqrt(self.bpt_eldiag.nii_err.iloc[agn_ind]**2 + (self.bpt_eldiag.nii_err.iloc[sf_inds]*bptdistrat)**2)
                hbetaflux_err_sub_bpt = np.sqrt(self.bpt_eldiag.hbeta_err.iloc[agn_ind]**2 + (self.bpt_eldiag.hbeta_err.iloc[sf_inds]*bptdistrat)**2)
                halpflux_err_sub_bpt  = np.sqrt(self.bpt_eldiag.halp_err.iloc[agn_ind]**2 + (self.bpt_eldiag.halp_err.iloc[sf_inds]*bptdistrat)**2)
                
                oiiiflux_sn_sub_bpt = oiiiflux_sub_bpt/ oiiiflux_err_sub_bpt
                niiflux_sn_sub_bpt = niiflux_sub_bpt/niiflux_err_sub_bpt
                hbetaflux_sn_sub_bpt = hbetaflux_sub_bpt/hbetaflux_err_sub_bpt
                halpflux_sn_sub_bpt = halpflux_sub_bpt/halpflux_err_sub_bpt
                diffs_bpt_sort = np.argsort(diffs_bpt)    
                
                inds_high_sn_bpt = np.where((oiiiflux_sn_sub_bpt.iloc[diffs_bpt_sort]>sncut) & (niiflux_sn_sub_bpt.iloc[diffs_bpt_sort]>sncut) &
                                            (hbetaflux_sn_sub_bpt.iloc[diffs_bpt_sort]>sncut) &(halpflux_sn_sub_bpt.iloc[diffs_bpt_sort] >sncut) )[0]
                #print(inds_high_sn_bpt)
                #print(inds_high_sn_bpt)
                if len(inds_high_sn_bpt) >0:    
                    mindiff_ind_bpt = diffs_bpt_sort[inds_high_sn_bpt[0]]
                    if len(inds_high_sn_bpt)>1:
                        second_best_ind_bpt = diffs_bpt_sort[inds_high_sn_bpt[1]]
                    else:
                        second_best_ind_bpt=mindiff_ind_bpt+1
                    n_pass_bpt = len(inds_high_sn_bpt)
                else:
                    mindiff_ind_bpt = -1
                    second_best_ind_bpt = -2
                    n_pass_bpt = len(diffs_bpt_sort)
                #computing differences for bpt+ SF 
                #computing differences for bpt+ SF 
                '''
                BPT AGN-BPT PLUS SF matching
                '''
                diffd4000_bptplus = (self.bpt_eldiag.d4000.iloc[agn_ind] - self.plus_eldiag.d4000.iloc[sf_plus_inds])**2#/np.var(sdssobj.alld4000)
                diffhd_bptplus = (self.bpt_eldiag.hdelta_lick.iloc[agn_ind]-self.plus_eldiag.hdelta_lick.iloc[sf_plus_inds])**2#/np.var(sdssobj.hdelta_lick)
                diffsfr_bptplus = (self.bpt_eldiag.sfr.iloc[agn_ind] - self.plus_eldiag.sfr.iloc[sf_plus_inds])**2
                diffmass_bptplus = (self.bpt_eldiag.mass.iloc[agn_ind] - self.plus_eldiag.mass.iloc[sf_plus_inds])**2
                difffibmass_bptplus = (self.bpt_eldiag.fibmass.iloc[agn_ind] - self.plus_eldiag.fibmass.iloc[sf_plus_inds])**2
                diffz_bptplus = (self.bpt_eldiag.z.iloc[agn_ind]-self.plus_eldiag.z.iloc[sf_plus_inds])**2/np.var(redshift_m2)
                diff_av_bptplus = (self.bpt_eldiag.tauv_cont.iloc[agn_ind]-self.plus_eldiag.tauv_cont.iloc[sf_plus_inds])**2
                diffs_bptplus =np.array( np.sqrt(diffsfr_bptplus+diffd4000_bptplus+diffhd_bptplus+diffmass_bptplus+difffibmass_bptplus+diffz_bptplus+diff_av_bptplus))

                bptdistrat_plus = (np.array(cosmo.luminosity_distance(self.plus_eldiag.z.iloc[sf_plus_inds]))/np.array(cosmo.luminosity_distance(self.bpt_eldiag.z.iloc[agn_ind])))**2 

                oiiiflux_sub_plus = self.bpt_eldiag.oiiiflux.iloc[agn_ind]-self.plus_eldiag.oiiiflux.iloc[sf_plus_inds]*bptdistrat_plus
                niiflux_sub_plus = self.bpt_eldiag.niiflux.iloc[agn_ind]-self.plus_eldiag.niiflux.iloc[sf_plus_inds]*bptdistrat_plus
                hbetaflux_sub_plus =self.bpt_eldiag.hbetaflux.iloc[agn_ind]-self.plus_eldiag.hbetaflux.iloc[sf_plus_inds]*bptdistrat_plus
                halpflux_sub_plus  = self.bpt_eldiag.halpflux.iloc[agn_ind]- self.plus_eldiag.halpflux.iloc[sf_plus_inds]*bptdistrat_plus
                
                oiiiflux_err_sub_plus = np.sqrt(self.bpt_eldiag.oiii_err.iloc[agn_ind]**2 + (self.plus_eldiag.oiii_err.iloc[sf_plus_inds]*bptdistrat_plus)**2)
                niiflux_err_sub_plus =  np.sqrt(self.bpt_eldiag.nii_err.iloc[agn_ind]**2 + (self.plus_eldiag.nii_err.iloc[sf_plus_inds]*bptdistrat_plus)**2)
                hbetaflux_err_sub_plus = np.sqrt(self.bpt_eldiag.hbeta_err.iloc[agn_ind]**2 + (self.plus_eldiag.hbeta_err.iloc[sf_plus_inds]*bptdistrat_plus)**2)
                halpflux_err_sub_plus  = np.sqrt(self.bpt_eldiag.halp_err.iloc[agn_ind]**2 + (self.plus_eldiag.halp_err.iloc[sf_plus_inds]*bptdistrat_plus)**2)
                
                oiiiflux_sn_sub_plus = oiiiflux_sub_plus/ oiiiflux_err_sub_plus
                niiflux_sn_sub_plus = niiflux_sub_plus/niiflux_err_sub_plus
                hbetaflux_sn_sub_plus = hbetaflux_sub_plus/hbetaflux_err_sub_plus
                halpflux_sn_sub_plus = halpflux_sub_plus/halpflux_err_sub_plus
  
                diffs_bptplus_sort = np.argsort(diffs_bptplus)    
                
                inds_high_sn_bptplus = np.where((oiiiflux_sn_sub_plus.iloc[diffs_bptplus_sort]>sncut) & (niiflux_sn_sub_plus.iloc[diffs_bptplus_sort]>sncut) &
                                            (hbetaflux_sn_sub_plus.iloc[diffs_bptplus_sort]>sncut) &(halpflux_sn_sub_plus.iloc[diffs_bptplus_sort] >sncut)&
                                                (self.plus_eldiag.oiii_err.iloc[diffs_bptplus_sort]*bptdistrat_plus[diffs_bptplus_sort] != 0 ) &
                                                (self.plus_eldiag.hbeta_err.iloc[diffs_bptplus_sort]*bptdistrat_plus[diffs_bptplus_sort] != 0 ) )[0]
                #print(inds_high_sn_bptplus)

                if len(inds_high_sn_bptplus) >0:    
                    mindiff_ind_bptplus = diffs_bptplus_sort[inds_high_sn_bptplus[0]]
                    if len(inds_high_sn_bptplus)>1:
                        
                        second_best_ind_bptplus = diffs_bptplus_sort[inds_high_sn_bptplus[1]]
                    else:
                        second_best_ind_bptplus = mindiff_ind_bptplus+1
                    n_pass_plus = len(inds_high_sn_bptplus)
                else:
                    mindiff_ind_bptplus = -1
                    second_best_ind_bptplus = -2
                    
                    n_pass_plus = len(diffs_bptplus_sort)
                '''
                BPT AGN-Neither match
                '''
                #computing differences for unclassifiable 
                diffd4000_neither = (self.bpt_eldiag.d4000.iloc[agn_ind] - self.neither_eldiag.d4000)**2#/np.var(sdssobj.alld4000)
                diffhd_neither = (self.bpt_eldiag.hdelta_lick.iloc[agn_ind] - self.neither_eldiag.hdelta_lick)**2#/np.var(sdssobj.hdelta_lick)
                diffsfr_neither = (self.bpt_eldiag.sfr.iloc[agn_ind] - self.neither_eldiag.sfr)**2

                diffmass_neither = (self.bpt_eldiag.mass.iloc[agn_ind] - self.neither_eldiag.mass)**2
                difffibmass_neither = (self.bpt_eldiag.fibmass.iloc[agn_ind] - self.neither_eldiag.fibmass)**2
                diffz_neither = (self.bpt_eldiag.z.iloc[agn_ind]-self.neither_eldiag.z)**2/np.var(redshift_m2)
                diff_av_agn_neither = (self.bpt_eldiag.tauv_cont.iloc[agn_ind]-self.neither_eldiag.tauv_cont)**2
                diffs_neither = np.array(np.sqrt(diffsfr_neither+diffd4000_neither+diffhd_neither+diffmass_neither+difffibmass_neither+diffz_neither+diff_av_agn_neither))

                bptdistrat_neither = (np.array(cosmo.luminosity_distance(self.neither_eldiag.z))/np.array(cosmo.luminosity_distance(self.bpt_eldiag.z.iloc[agn_ind])))**2 
                        

                oiiiflux_sub_neither = self.bpt_eldiag.oiiiflux.iloc[agn_ind]-self.neither_eldiag.oiiiflux*bptdistrat_neither
                niiflux_sub_neither = self.bpt_eldiag.niiflux.iloc[agn_ind]-self.neither_eldiag.niiflux*bptdistrat_neither
                hbetaflux_sub_neither =self.bpt_eldiag.hbetaflux.iloc[agn_ind]-self.neither_eldiag.hbetaflux*bptdistrat_neither
                halpflux_sub_neither  = self.bpt_eldiag.halpflux.iloc[agn_ind]- self.neither_eldiag.halpflux*bptdistrat_neither
                
                oiiiflux_err_sub_neither = np.sqrt(self.bpt_eldiag.oiii_err.iloc[agn_ind]**2 + (self.neither_eldiag.oiii_err*bptdistrat_neither)**2)
                niiflux_err_sub_neither =  np.sqrt(self.bpt_eldiag.nii_err.iloc[agn_ind]**2 + (self.neither_eldiag.nii_err*bptdistrat_neither)**2)
                hbetaflux_err_sub_neither = np.sqrt(self.bpt_eldiag.hbeta_err.iloc[agn_ind]**2 + (self.neither_eldiag.hbeta_err*bptdistrat_neither)**2)
                halpflux_err_sub_neither  = np.sqrt(self.bpt_eldiag.halp_err.iloc[agn_ind]**2 + (self.neither_eldiag.halp_err*bptdistrat_neither)**2)
                
                oiiiflux_sn_sub_neither = oiiiflux_sub_neither/oiiiflux_err_sub_neither
                niiflux_sn_sub_neither = niiflux_sub_neither/niiflux_err_sub_neither
                hbetaflux_sn_sub_neither = hbetaflux_sub_neither/hbetaflux_err_sub_neither
                halpflux_sn_sub_neither = halpflux_sub_neither/halpflux_err_sub_neither
  
                diffs_neither_sort = np.argsort(diffs_neither)    
  
                inds_high_sn_neither = np.where((oiiiflux_sn_sub_neither.iloc[diffs_neither_sort]>sncut) & (niiflux_sn_sub_neither.iloc[diffs_neither_sort]>sncut) &
                                                (hbetaflux_sn_sub_neither.iloc[diffs_neither_sort]>sncut) &(halpflux_sn_sub_neither.iloc[diffs_neither_sort] >sncut) &
                                                (self.neither_eldiag.oiii_err.iloc[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) &
                                                (self.neither_eldiag.hbeta_err.iloc[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) &
                                                (self.neither_eldiag.halp_err.iloc[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) &
                                                (self.neither_eldiag.nii_err.iloc[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) )[0]
                if len(inds_high_sn_neither)>0:
                    mindiff_ind_neither = diffs_neither_sort[inds_high_sn_neither[0]]
                    if len(inds_high_sn_neither)>1:
                        second_best_ind_neither = diffs_neither_sort[inds_high_sn_neither[1]]
                    else:
                        second_best_ind_neither = mindiff_ind_neither + 1
                    n_pass_neither = len(inds_high_sn_neither)
                else:
                    mindiff_ind_neither = -1
                    second_best_ind_neither = -2
                    n_pass_neither = len(diffs_neither_sort)
                #print(inds_high_sn_neither)                            
                mindiffinds = [mindiff_ind_bpt, mindiff_ind_bptplus, mindiff_ind_neither]
                second_mindiffinds = [second_best_ind_bpt, second_best_ind_bptplus, second_best_ind_neither]
                
                
                mindist_out = [diffs_bpt[mindiff_ind_bpt], diffs_bptplus[mindiff_ind_bptplus],  
                               diffs_neither[mindiff_ind_neither]]
                secondmindist_out = [diffs_bpt[second_best_ind_bpt], diffs_bptplus[second_best_ind_bptplus],  
                               diffs_neither[second_best_ind_neither]]

                bad_dists = np.where(mindiffinds == -1)[0]
                bad_dists2 = np.where(mindiffinds == -2)[0]

                if bad_dists.size >0:
                    mindist_out[bad_dists]=99999
                if bad_dists2.size >0:
                    secondmindist_out[bad_dists2]=99999
                                        
                #print(mindist_out)
                n_pass_out = [n_pass_bpt, n_pass_plus, n_pass_neither]
                 
                #assigning the ids, inds, dists to be saved
                minid_out =[ self.bpt_eldiag.ids.iloc[sf_inds[mindiff_ind_bpt]],  
                            self.plus_eldiag.ids.iloc[sf_plus_inds[mindiff_ind_bptplus]],
                            self.neither_eldiag.ids.iloc[mindiff_ind_neither]]
                minind_out =[ sf_inds[mindiff_ind_bpt],  
                            sf_plus_inds[mindiff_ind_bptplus],
                            mindiff_ind_neither]
                secondminind_out =[ sf_inds[second_best_ind_bpt],  
                            sf_plus_inds[second_best_ind_bptplus],
                            second_best_ind_neither]

                #assigning the ids, inds, dists to be saved for self-matching

                minid_outagn = [self.bpt_eldiag.ids.iloc[agn_inds[mindiff_ind_agn]],  
                            self.plus_eldiag.ids.iloc[agnplus_inds[mindiff_ind_agnplus]]]
                minind_outagn =[ agn_inds[mindiff_ind_agn],  
                            agnplus_inds[mindiff_ind_bptplus]]
                mindist_outagn = [diffs_agn[mindiff_ind_agn], diffs_agnplus[mindiff_ind_agnplus]]
                numpassed[:, i] = n_pass_out
                
                #saving the relevant info 
                mindists[:, i] = mindist_out
                minids[:, i] = minid_out
                mininds[:, i] = minind_out
                
                second_mininds[:,i] = secondminind_out
                second_mindists[:, i] = secondmindist_out
    
                mindists_agn[:, i] = mindist_outagn
                minids_agn[:, i] = minid_outagn
                mininds_agn[:, i] = minind_outagn

                mindist_ind = np.int(np.where(mindist_out == np.min(mindist_out))[0])

                #getting the best one, 0 = BPT SF, 1 = BPT SF+, 2 = Unclassifiable
                if mindist_ind ==0:
                    sfs.append(sf_inds[mindiff_ind_bpt])
                    agns.append(agn_ind)
                    numpassed_best.append(n_pass_bpt)
                    
                elif mindist_ind==1:
                    sfs_plus.append(sf_plus_inds[mindiff_ind_bptplus])
                    agns_plus.append(agn_ind)
                    numpassed_best.append(n_pass_plus)
                    
                else:
                    neither_matches.append(mindiff_ind_neither)
                    neither_agn.append(agn_ind)
                    numpassed_best.append(n_pass_neither)

                comb_first_sec = np.hstack([mindist_out, secondmindist_out])
                sort_comb_first_sec = np.argsort(comb_first_sec)
                sec_best_ind = np.where(comb_first_sec == comb_first_sec[sort_comb_first_sec[1]])[0]
                
                if sec_best_ind ==0:
                    second_sfs.append(sf_inds[mindiff_ind_bpt])
                    second_agns.append(agn_ind)
                elif sec_best_ind==1:
                    second_sfs_plus.append(sf_plus_inds[mindiff_ind_bptplus])
                    second_agns_plus.append(agn_ind)        
                elif sec_best_ind==2:
                    second_neither_matches.append(mindiff_ind_neither)
                    second_neither_agn.append(agn_ind)
                elif sec_best_ind ==3:
                    second_sfs.append(sf_inds[second_best_ind_bpt])
                    second_agns.append(agn_ind)
                elif sec_best_ind==4:
                    second_sfs_plus.append(sf_plus_inds[second_best_ind_bptplus])
                    second_agns_plus.append(agn_ind)        
                elif sec_best_ind==5:
                    second_neither_matches.append(second_best_ind_neither)
                    second_neither_agn.append(agn_ind)


                                        
                mindistagn_ind = np.int(np.where(mindist_outagn==np.min(mindist_outagn))[0])
                if mindistagn_ind ==0:
                    agns_selfmatch.append(agn_ind)
                    agns_selfmatch_other.append(mindiff_ind_agn[0])
                else:
                    agnsplus_selfmatch.append(agn_ind)
                    agnsplus_selfmatch_other.append(agnplus_inds[mindiff_ind_agnplus[0]])  

                            

            #converting lists to arrays and saving to class attributes
            self.numpassed = numpassed
            self.numpassed_best = np.array(numpassed_best)
            self.agns = np.array(agns)
            self.sfs = np.array(sfs)
            self.agns_plus = np.array(agns_plus)
            self.sfs_plus = np.array(sfs_plus)
            self.neither_matches = np.array(neither_matches)
            self.neither_agn = np.array(neither_agn)
            
            self.second_agns=np.array(second_agns)
            self.second_sfs = np.array(second_sfs)
            self.second_agns_plus = np.array(second_agns_plus)
            self.second_sfs_plus = np.array(second_sfs_plus)
            self.second_neither_matches = np.array(second_neither_matches)
            self.second_neither_agn = np.array(second_neither_agn)


            self.mindists = mindists
            self.mininds= mininds
            
            self.second_mininds = second_mininds
            
            self.minids = minids
            self.mindists_best = np.min(self.mindists, axis=0)
            self.second_mindists = np.array(second_mindists)
            self.second_mindists_best = np.min(self.second_mindists, axis=0)
    
            self.agns_selfmatch = np.array(agns_selfmatch)
            self.agns_selfmatch_other = np.array(agns_selfmatch_other)
            self.agnsplus_selfmatch = np.array(agnsplus_selfmatch)
            self.agnsplus_selfmatch_other = np.array(agnsplus_selfmatch_other)
            self.mindists_agn = mindists_agn
            self.minids_agn = minids_agn
            self.mininds_agn = mininds_agn
            self.mindistsagn_best = np.min(self.mindists_agn, axis=0)
           
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_n_pass_best_highsn'+str(sncut)+fname+'.txt',self.numpassed_best, fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_n_pass_highsn'+str(sncut)+fname+'.txt',self.numpassed.transpose(), fmt='%6.d')
            
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_agns_selfmatch_highsn'+str(sncut)+fname+'.txt',self.agns_selfmatch, fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_agns_selfmatch_other_highsn'+str(sncut)+fname+'.txt',self.agns_selfmatch_other, fmt='%6.d')
            
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_agnsplus_selfmatch_highsn'+str(sncut)+fname+'.txt',self.agnsplus_selfmatch, fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_agnsplus_selfmatch_other_highsn'+str(sncut)+fname+'.txt',self.agnsplus_selfmatch_other, fmt='%6.d')
            
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_agns_highsn'+str(sncut)+fname+'.txt',self.agns, fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_sfs_highsn'+str(sncut)+fname+'.txt',self.sfs, fmt='%6.d')
            
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_agns_plus_highsn'+str(sncut)+fname+'.txt',self.agns_plus, fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_sfs_plus_highsn'+str(sncut)+fname+'.txt',self.sfs_plus, fmt='%6.d')
            
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_neither_matches_highsn'+str(sncut)+fname+'.txt',self.neither_matches, fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_neither_agn_highsn'+str(sncut)+fname+'.txt',self.neither_agn, fmt='%6.d')
            
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_mindists_best_highsn'+str(sncut)+fname+'.txt',self.mindists_best, fmt='%8.6f')
            
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_mindists_highsn'+str(sncut)+fname+'.txt',self.mindists.transpose(), fmt='%8.6f')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_mininds_highsn'+str(sncut)+fname+'.txt',self.mininds.transpose(), fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_minids_highsn'+str(sncut)+fname+'.txt',self.minids.transpose(), fmt='%18.d')
            
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_mindists_agn_highsn'+str(sncut)+fname+'.txt',self.mindists_agn.transpose(), fmt='%8.6f')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_mininds_agn_highsn'+str(sncut)+fname+'.txt',self.mininds_agn.transpose(), fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_minids_agn_highsn'+str(sncut)+fname+'.txt',self.minids_agn.transpose(), fmt='%18.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_mindistsagn_best_highsn'+str(sncut)+fname+'.txt',self.mindistsagn_best, fmt='%8.6f')

            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_mindists_best_highsn'+str(sncut)+fname+'.txt',self.second_mindists_best, fmt='%8.6f')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_mindists_highsn'+str(sncut)+fname+'.txt',self.second_mindists.transpose(), fmt='%8.6f')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_mininds_highsn'+str(sncut)+fname+'.txt',self.second_mininds.transpose(), fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_neither_matches_highsn'+str(sncut)+fname+'.txt',self.second_neither_matches, fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_neither_agn_highsn'+str(sncut)+fname+'.txt',self.second_neither_agn, fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_agns_plus_highsn'+str(sncut)+fname+'.txt',self.second_agns_plus, fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_sfs_plus_highsn'+str(sncut)+fname+'.txt',self.second_sfs_plus, fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_agns_highsn'+str(sncut)+fname+'.txt',self.second_agns, fmt='%6.d')
            np.savetxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_sfs_highsn'+str(sncut)+fname+'.txt',self.second_sfs, fmt='%6.d')                
        else:
            #once the matching is already done just need to load items in
            self.numpassed =  np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_n_pass_highsn'+str(sncut)+fname+'.txt',dtype=np.int64, unpack=True)
            
            self.numpassed_best = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_n_pass_best_highsn'+str(sncut)+fname+'.txt',dtype=np.int64)
            
            self.agns_selfmatch = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_agns_selfmatch_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            self.agns_selfmatch_other = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_agns_selfmatch_other_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            
            self.agnsplus_selfmatch = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_agnsplus_selfmatch_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            self.agnsplus_selfmatch_other = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_agnsplus_selfmatch_other_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            
            self.agns = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_agns_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            self.sfs = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_sfs_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            
            self.agns_plus = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_agns_plus_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            self.sfs_plus = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_sfs_plus_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)

            self.neither_agn = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_neither_agn_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)            
            self.neither_matches = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_neither_matches_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            
            
            self.mindists = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_mindists_highsn'+str(sncut)+fname+'.txt', unpack=True)
            self.mininds = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_mininds_highsn'+str(sncut)+fname+'.txt', dtype=np.int64, unpack=True)
            self.minids = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_minids_highsn'+str(sncut)+fname+'.txt', dtype=np.int64, unpack=True)
            self.mindists_best = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_mindists_best_highsn'+str(sncut)+fname+'.txt')
            
            self.mindists_agn = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_mindists_agn_highsn'+str(sncut)+fname+'.txt', unpack=True)
            self.minids_agn = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_mininds_agn_highsn'+str(sncut)+fname+'.txt', dtype=np.int64, unpack=True)
            self.mininds_agn = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_minids_agn_highsn'+str(sncut)+fname+'.txt', dtype=np.int64, unpack=True)
            self.mindistsagn_best = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_mindistsagn_best_highsn'+str(sncut)+fname+'.txt')
            self.second_agns = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_agns_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            self.second_sfs = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_sfs_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            
            self.second_agns_plus = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_agns_plus_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)
            self.second_sfs_plus = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_sfs_plus_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)

            self.second_neither_agn = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_neither_agn_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)            
            self.second_neither_matches = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_neither_matches_highsn'+str(sncut)+fname+'.txt', dtype=np.int64)

            self.second_mindists = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_mindists_highsn'+str(sncut)+fname+'.txt', unpack=True)
            self.second_mininds = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_mininds_highsn'+str(sncut)+fname+'.txt', dtype=np.int64, unpack=True)

            self.second_mindists_best = np.loadtxt(catfold+'d4000m/'+subfold+'d4000m_hd_second_mindists_best_highsn'+str(sncut)+fname+'.txt')
                        
            agn_dist_inds = []
            agn_plus_dist_inds = [] 
            agn_neither_dist_inds = []
            agn_selfdist_inds = []
            agn_selfplus_dist_inds = [] 
            for i in range(len(self.mindists_best)):
                mn = np.where(self.mindists[:,i] == self.mindists_best[i])[0]
                if len(mn) ==1:
                        
                    if mn == 0:
                        agn_dist_inds.append(i)
                    elif mn==1: 
                        agn_plus_dist_inds.append(i)
                    else:
                        agn_neither_dist_inds.append(i)
                else:
                    if 0 in mn:
                        agn_dist_inds.append(i)
                    elif 1 in mn:
                        agn_plus_dist_inds.append(i)
                    else:
                        agn_neither_dist_inds.append(i)
                mnself = np.where(self.mindists_agn[:,i] == self.mindistsagn_best[i])[0]
                if len(mnself)==1:
                    if mnself==0:
                        agn_selfdist_inds.append(i)
                    else:
                        agn_selfplus_dist_inds.append(i)
                else:
                    if 0 in mnself:
                        agn_selfdist_inds.append(i)
                    else:
                        agn_selfplus_dist_inds.append(i)
                
            self.agn_dist_inds = np.array(agn_dist_inds)
            self.agn_plus_dist_inds = np.array(agn_plus_dist_inds)
            self.agn_neither_dist_inds = np.array(agn_neither_dist_inds)
            self.agn_ind_mapping = combine_arrs([self.agn_dist_inds, self.agn_plus_dist_inds, self.agn_neither_dist_inds])

            self.agn_selfdist_inds = np.array(agn_selfdist_inds)
            self.agn_selfplus_dist_inds = np.array(agn_selfplus_dist_inds)
            self.agn_selfind_mapping = combine_arrs([self.agn_selfdist_inds, self.agn_selfplus_dist_inds])


            self.mindists_best_sing_ord = combine_arrs([self.mindists_best[self.agn_dist_inds], self.mindists_best[self.agn_plus_dist_inds], self.mindists_best[self.agn_neither_dist_inds] ])

