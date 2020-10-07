import numpy as np
from ast_func import *
catfold='catalogs/'
import astropy.cosmology as apc
from loaddata_m2 import redshift_m2
cosmo = apc.Planck15
from setops import *
import time
from astropy.stats import bootstrap as apy_bootstrap
from demarcations import *
m_ssfr = -0.4597
b_ssfr = -5.2976
class SFRMatch:
    def __init__(self, eldiag):
        self.eldiag=eldiag

        
    def get_highsn_match_only(self, agn_inds, sf_inds, sf_plus_inds, 
                              agnplus_inds, sncut=2, load=False, fname='', with_av=True):
        '''
        Matches BPT AGN to BPT SF, BPT+ SF, unclassified of similar SFR, M*, Mfib, z
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
                diffsfr_agn = (self.eldiag.sfr[agn_ind]-self.eldiag.sfr[agn_inds][otheragns])**2
                diffmass_agn = (self.eldiag.mass[agn_ind]-self.eldiag.mass[agn_inds][otheragns])**2
                difffibmass_agn = (self.eldiag.fibmass[agn_ind]-self.eldiag.fibmass[agn_inds][otheragns])**2
                diffz_agn = (self.eldiag.z[agn_ind]-self.eldiag.z[agn_inds][otheragns])**2/np.std(redshift_m2)
                diff_av_agn = (self.eldiag.av_gsw[agn_ind]-self.eldiag.av_gsw[agn_inds][otheragns])**2
                diffs_agn = np.sqrt(diffsfr_agn+diffmass_agn+difffibmass_agn+diffz_agn+diff_av_agn)
                
                mindiff_ind_agn = np.where(diffs_agn == np.min(diffs_agn) )[0]
                '''
                BPT Plus self-agn matching
                '''
                #computing differences for self matching agn to bpt+ agn
                diffsfr_agnplus = (self.eldiag.sfr[agn_ind]-self.eldiag.sfr_plus[agnplus_inds])**2
                diffmass_agnplus = (self.eldiag.mass[agn_ind]-self.eldiag.mass_plus[agnplus_inds])**2
                difffibmass_agnplus = (self.eldiag.fibmass[agn_ind]-self.eldiag.fibmass_plus[agnplus_inds])**2
                diffz_agnplus = (self.eldiag.z[agn_ind]-self.eldiag.z_plus[agnplus_inds])**2/np.std(redshift_m2)
                diff_av_agnplus = (self.eldiag.av_gsw[agn_ind]-self.eldiag.av_gsw_plus[agnplus_inds])**2
                diffs_agnplus = np.sqrt(diffsfr_agnplus+diffmass_agnplus+difffibmass_agnplus+diffz_agnplus+diff_av_agnplus)
                mindiff_ind_agnplus = np.where(diffs_agnplus == np.min(diffs_agnplus) )[0]
                '''
                BPT AGN-BPT SF matching
                '''
                #computing differences for bpt SF 
                diffsfr_bpt = (self.eldiag.sfr[agn_ind] - self.eldiag.sfr[sf_inds])**2
                diffmass_bpt = (self.eldiag.mass[agn_ind] - self.eldiag.mass[sf_inds])**2
                difffibmass_bpt = (self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass[sf_inds])**2
                diffz_bpt = (self.eldiag.z[agn_ind]-self.eldiag.z[sf_inds])**2/np.std(redshift_m2)
                diff_av_bpt = (self.eldiag.av_gsw[agn_ind]-self.eldiag.av_gsw[sf_inds])**2
                diffs_bpt = np.sqrt(diffsfr_bpt+diffmass_bpt+difffibmass_bpt+diffz_bpt+diff_av_bpt)
                #mindiff_ind_bpt = np.where(diffs_bpt == np.min(diffs_bpt) )[0]
                bptdistrat = (np.array(cosmo.luminosity_distance(self.eldiag.z[sf_inds]))/np.array(cosmo.luminosity_distance(self.eldiag.z[agn_ind])))**2 

                oiiiflux_sub_bpt = self.eldiag.oiiiflux[agn_ind]-self.eldiag.oiiiflux[sf_inds]*bptdistrat
                niiflux_sub_bpt = self.eldiag.niiflux[agn_ind]-self.eldiag.niiflux[sf_inds]*bptdistrat
                hbetaflux_sub_bpt =self.eldiag.hbetaflux[agn_ind]-self.eldiag.hbetaflux[sf_inds]*bptdistrat
                halpflux_sub_bpt  = self.eldiag.halpflux[agn_ind]- self.eldiag.halpflux[sf_inds]*bptdistrat
                
                oiiiflux_err_sub_bpt = np.sqrt(self.eldiag.oiii_err_bpt[agn_ind]**2 + (self.eldiag.oiii_err_bpt[sf_inds]*bptdistrat)**2)
                niiflux_err_sub_bpt =  np.sqrt(self.eldiag.nii_err_bpt[agn_ind]**2 + (self.eldiag.nii_err_bpt[sf_inds]*bptdistrat)**2)
                hbetaflux_err_sub_bpt = np.sqrt(self.eldiag.hbeta_err_bpt[agn_ind]**2 + (self.eldiag.hbeta_err_bpt[sf_inds]*bptdistrat)**2)
                halpflux_err_sub_bpt  = np.sqrt(self.eldiag.halp_err_bpt[agn_ind]**2 + (self.eldiag.halp_err_bpt[sf_inds]*bptdistrat)**2)
                
                oiiiflux_sn_sub_bpt = oiiiflux_sub_bpt*1e17/ oiiiflux_err_sub_bpt
                niiflux_sn_sub_bpt = niiflux_sub_bpt*1e17/niiflux_err_sub_bpt
                hbetaflux_sn_sub_bpt = hbetaflux_sub_bpt*1e17/hbetaflux_err_sub_bpt
                halpflux_sn_sub_bpt = halpflux_sub_bpt*1e17/halpflux_err_sub_bpt
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
                diffsfr_bptplus = (self.eldiag.sfr[agn_ind] - self.eldiag.sfr_plus[sf_plus_inds])**2
                diffmass_bptplus = (self.eldiag.mass[agn_ind] - self.eldiag.mass_plus[sf_plus_inds])**2
                difffibmass_bptplus = (self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass_plus[sf_plus_inds])**2
                diffz_bptplus = (self.eldiag.z[agn_ind]-self.eldiag.z_plus[sf_plus_inds])**2/np.std(redshift_m2)
                diff_av_bptplus = (self.eldiag.av_gsw[agn_ind]-self.eldiag.av_gsw_plus[sf_plus_inds])**2
                diffs_bptplus = np.sqrt(diffsfr_bptplus+diffmass_bptplus+difffibmass_bptplus+diffz_bptplus+diff_av_bptplus)

                bptdistrat_plus = (np.array(cosmo.luminosity_distance(self.eldiag.z_plus[sf_plus_inds]))/np.array(cosmo.luminosity_distance(self.eldiag.z[agn_ind])))**2 

                oiiiflux_sub_plus = self.eldiag.oiiiflux[agn_ind]-self.eldiag.oiiiflux_plus[sf_plus_inds]*bptdistrat_plus
                niiflux_sub_plus = self.eldiag.niiflux[agn_ind]-self.eldiag.niiflux_plus[sf_plus_inds]*bptdistrat_plus
                hbetaflux_sub_plus =self.eldiag.hbetaflux[agn_ind]-self.eldiag.hbetaflux_plus[sf_plus_inds]*bptdistrat_plus
                halpflux_sub_plus  = self.eldiag.halpflux[agn_ind]- self.eldiag.halpflux_plus[sf_plus_inds]*bptdistrat_plus
                
                oiiiflux_err_sub_plus = np.sqrt(self.eldiag.oiii_err_bpt[agn_ind]**2 + (self.eldiag.oiii_err_plus[sf_plus_inds]*bptdistrat_plus)**2)
                niiflux_err_sub_plus =  np.sqrt(self.eldiag.nii_err_bpt[agn_ind]**2 + (self.eldiag.nii_err_plus[sf_plus_inds]*bptdistrat_plus)**2)
                hbetaflux_err_sub_plus = np.sqrt(self.eldiag.hbeta_err_bpt[agn_ind]**2 + (self.eldiag.hbeta_err_plus[sf_plus_inds]*bptdistrat_plus)**2)
                halpflux_err_sub_plus  = np.sqrt(self.eldiag.halp_err_bpt[agn_ind]**2 + (self.eldiag.halp_err_plus[sf_plus_inds]*bptdistrat_plus)**2)
                
                oiiiflux_sn_sub_plus = oiiiflux_sub_plus*1e17/ oiiiflux_err_sub_plus
                niiflux_sn_sub_plus = niiflux_sub_plus*1e17/niiflux_err_sub_plus
                hbetaflux_sn_sub_plus = hbetaflux_sub_plus*1e17/hbetaflux_err_sub_plus
                halpflux_sn_sub_plus = halpflux_sub_plus*1e17/halpflux_err_sub_plus
  
                diffs_bptplus_sort = np.argsort(diffs_bptplus)    
                
                inds_high_sn_bptplus = np.where((oiiiflux_sn_sub_plus[diffs_bptplus_sort]>sncut) & (niiflux_sn_sub_plus[diffs_bptplus_sort]>sncut) &
                                            (hbetaflux_sn_sub_plus[diffs_bptplus_sort]>sncut) &(halpflux_sn_sub_plus[diffs_bptplus_sort] >sncut)&
                                                (self.eldiag.oiii_err_plus[diffs_bptplus_sort]*bptdistrat_plus[diffs_bptplus_sort] != 0 ) &
                                                (self.eldiag.hbeta_err_plus[diffs_bptplus_sort]*bptdistrat_plus[diffs_bptplus_sort] != 0 ) )[0]
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
                diffsfr_neither = (self.eldiag.sfr[agn_ind] - self.eldiag.sfr_neither)**2
                diffmass_neither = (self.eldiag.mass[agn_ind] - self.eldiag.mass_neither)**2
                difffibmass_neither = (self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass_neither)**2
                diffz_neither = (self.eldiag.z[agn_ind]-self.eldiag.z_neither)**2/np.std(redshift_m2)
                diff_av_agn_neither = (self.eldiag.av_gsw[agn_ind]-self.eldiag.av_gsw_neither)**2
                diffs_neither = np.sqrt(diffsfr_neither+diffmass_neither+difffibmass_neither+diffz_neither+diff_av_agn_neither)

                bptdistrat_neither = (np.array(cosmo.luminosity_distance(self.eldiag.z_neither))/np.array(cosmo.luminosity_distance(self.eldiag.z[agn_ind])))**2 
                        

                oiiiflux_sub_neither = self.eldiag.oiiiflux[agn_ind]-self.eldiag.oiiiflux_neither*bptdistrat_neither
                niiflux_sub_neither = self.eldiag.niiflux[agn_ind]-self.eldiag.niiflux_neither*bptdistrat_neither
                hbetaflux_sub_neither =self.eldiag.hbetaflux[agn_ind]-self.eldiag.hbetaflux_neither*bptdistrat_neither
                halpflux_sub_neither  = self.eldiag.halpflux[agn_ind]- self.eldiag.halpflux_neither*bptdistrat_neither
                
                oiiiflux_err_sub_neither = np.sqrt(self.eldiag.oiii_err_bpt[agn_ind]**2 + (self.eldiag.oiii_err_neither*bptdistrat_neither)**2)
                niiflux_err_sub_neither =  np.sqrt(self.eldiag.nii_err_bpt[agn_ind]**2 + (self.eldiag.nii_err_neither*bptdistrat_neither)**2)
                hbetaflux_err_sub_neither = np.sqrt(self.eldiag.hbeta_err_bpt[agn_ind]**2 + (self.eldiag.hbeta_err_neither*bptdistrat_neither)**2)
                halpflux_err_sub_neither  = np.sqrt(self.eldiag.halp_err_bpt[agn_ind]**2 + (self.eldiag.halp_err_neither*bptdistrat_neither)**2)
                
                oiiiflux_sn_sub_neither = oiiiflux_sub_neither*1e17/ oiiiflux_err_sub_neither
                niiflux_sn_sub_neither = niiflux_sub_neither*1e17/niiflux_err_sub_neither
                hbetaflux_sn_sub_neither = hbetaflux_sub_neither*1e17/hbetaflux_err_sub_neither
                halpflux_sn_sub_neither = halpflux_sub_neither*1e17/halpflux_err_sub_neither
  
                diffs_neither_sort = np.argsort(diffs_neither)    
  
                inds_high_sn_neither = np.where((oiiiflux_sn_sub_neither[diffs_neither_sort]>sncut) & (niiflux_sn_sub_neither[diffs_neither_sort]>sncut) &
                                                (hbetaflux_sn_sub_neither[diffs_neither_sort]>sncut) &(halpflux_sn_sub_neither[diffs_neither_sort] >sncut) &
                                                (self.eldiag.oiii_err_neither[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) &
                                                (self.eldiag.hbeta_err_neither[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) &
                                                (self.eldiag.halp_err_neither[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) &
                                                (self.eldiag.nii_err_neither[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) )[0]
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
                minid_out =[ self.eldiag.ids[sf_inds[mindiff_ind_bpt]],  
                            self.eldiag.ids_plus[sf_plus_inds[mindiff_ind_bptplus]],
                            self.eldiag.ids_neither[mindiff_ind_neither]]
                minind_out =[ sf_inds[mindiff_ind_bpt],  
                            sf_plus_inds[mindiff_ind_bptplus],
                            mindiff_ind_neither]
                #assigning the ids, inds, dists to be saved for self-matching

                minid_outagn = [self.eldiag.ids[agn_inds[mindiff_ind_agn]],  
                            self.eldiag.ids_plus[agnplus_inds[mindiff_ind_agnplus]]]
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
            
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_mindists_highsn'+str(sncut)+fname+'.txt',self.mindists.transpose(), fmt='%8.6f')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_mininds_highsn'+str(sncut)+fname+'.txt',self.mininds.transpose(), fmt='%6.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_minids_highsn'+str(sncut)+fname+'.txt',self.minids.transpose(), fmt='%18.d')
            
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_mindists_agn_highsn'+str(sncut)+fname+'.txt',self.mindists_agn.transpose(), fmt='%8.6f')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_mininds_agn_highsn'+str(sncut)+fname+'.txt',self.mininds_agn.transpose(), fmt='%6.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_minids_agn_highsn'+str(sncut)+fname+'.txt',self.minids_agn.transpose(), fmt='%18.d')
            np.savetxt(catfold+'sfrm/'+subfold+'sfrm_mindistsagn_best_highsn'+str(sncut)+fname+'.txt',self.mindistsagn_best, fmt='%8.6f')
                
        else:
            #once the matching is already done just need to load items in
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
            
            self.mindists_agn = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_mindists_agn_highsn'+str(sncut)+fname+'.txt', unpack=True)
            self.minids_agn = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_mininds_agn_highsn'+str(sncut)+fname+'.txt', dtype=np.int64, unpack=True)
            self.mininds_agn = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_minids_agn_highsn'+str(sncut)+fname+'.txt', dtype=np.int64, unpack=True)
            self.mindistsagn_best = np.loadtxt(catfold+'sfrm/'+subfold+'sfrm_mindistsagn_best_highsn'+str(sncut)+fname+'.txt')
            
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
                diffd4000_agn = (self.eldiag.d4000[agn_ind]-self.eldiag.d4000[agn_inds][otheragns])**2
                diffmass_agn = (self.eldiag.mass[agn_ind]-self.eldiag.mass[agn_inds][otheragns])**2
                difffibmass_agn = (self.eldiag.fibmass[agn_ind]-self.eldiag.fibmass[agn_inds][otheragns])**2
                diffz_agn = (self.eldiag.z[agn_ind]-self.eldiag.z[agn_inds][otheragns])**2/np.std(redshift_m2)
                diff_av_agn = (self.eldiag.av_gsw[agn_ind]-self.eldiag.av_gsw[agn_inds][otheragns])**2
                diffs_agn = np.sqrt(diffd4000_agn+diffmass_agn+difffibmass_agn+diffz_agn+diff_av_agn)
                
                mindiff_ind_agn = np.where(diffs_agn == np.min(diffs_agn) )[0]
                '''
                BPT Plus self-agn matching
                '''
                #computing differences for self matching agn to bpt+ agn
                diffd4000_agnplus = (self.eldiag.d4000[agn_ind]-self.eldiag.d4000_plus[agnplus_inds])**2
                diffmass_agnplus = (self.eldiag.mass[agn_ind]-self.eldiag.mass_plus[agnplus_inds])**2
                difffibmass_agnplus = (self.eldiag.fibmass[agn_ind]-self.eldiag.fibmass_plus[agnplus_inds])**2
                diffz_agnplus = (self.eldiag.z[agn_ind]-self.eldiag.z_plus[agnplus_inds])**2/np.std(redshift_m2)
                diff_av_agnplus = (self.eldiag.av_gsw[agn_ind]-self.eldiag.av_gsw_plus[agnplus_inds])**2
                diffs_agnplus = np.sqrt(diffd4000_agnplus+diffmass_agnplus+difffibmass_agnplus+diffz_agnplus+diff_av_agnplus)
                mindiff_ind_agnplus = np.where(diffs_agnplus == np.min(diffs_agnplus) )[0]
                '''
                BPT AGN-BPT SF matching
                '''
                #computing differences for bpt SF 
                diffd4000_bpt = (self.eldiag.d4000[agn_ind] - self.eldiag.d4000[sf_inds])**2
                diffmass_bpt = (self.eldiag.mass[agn_ind] - self.eldiag.mass[sf_inds])**2
                difffibmass_bpt = (self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass[sf_inds])**2
                diffz_bpt = (self.eldiag.z[agn_ind]-self.eldiag.z[sf_inds])**2/np.std(redshift_m2)
                diff_av_bpt = (self.eldiag.av_gsw[agn_ind]-self.eldiag.av_gsw[sf_inds])**2
                diffs_bpt = np.sqrt(diffd4000_bpt+diffmass_bpt+difffibmass_bpt+diffz_bpt+diff_av_bpt)
                #mindiff_ind_bpt = np.where(diffs_bpt == np.min(diffs_bpt) )[0]
                bptdistrat = (np.array(cosmo.luminosity_distance(self.eldiag.z[sf_inds]))/np.array(cosmo.luminosity_distance(self.eldiag.z[agn_ind])))**2 

                oiiiflux_sub_bpt = self.eldiag.oiiiflux[agn_ind]-self.eldiag.oiiiflux[sf_inds]*bptdistrat
                niiflux_sub_bpt = self.eldiag.niiflux[agn_ind]-self.eldiag.niiflux[sf_inds]*bptdistrat
                hbetaflux_sub_bpt =self.eldiag.hbetaflux[agn_ind]-self.eldiag.hbetaflux[sf_inds]*bptdistrat
                halpflux_sub_bpt  = self.eldiag.halpflux[agn_ind]- self.eldiag.halpflux[sf_inds]*bptdistrat
                
                oiiiflux_err_sub_bpt = np.sqrt(self.eldiag.oiii_err_bpt[agn_ind]**2 + (self.eldiag.oiii_err_bpt[sf_inds]*bptdistrat)**2)
                niiflux_err_sub_bpt =  np.sqrt(self.eldiag.nii_err_bpt[agn_ind]**2 + (self.eldiag.nii_err_bpt[sf_inds]*bptdistrat)**2)
                hbetaflux_err_sub_bpt = np.sqrt(self.eldiag.hbeta_err_bpt[agn_ind]**2 + (self.eldiag.hbeta_err_bpt[sf_inds]*bptdistrat)**2)
                halpflux_err_sub_bpt  = np.sqrt(self.eldiag.halp_err_bpt[agn_ind]**2 + (self.eldiag.halp_err_bpt[sf_inds]*bptdistrat)**2)
                
                oiiiflux_sn_sub_bpt = oiiiflux_sub_bpt*1e17/ oiiiflux_err_sub_bpt
                niiflux_sn_sub_bpt = niiflux_sub_bpt*1e17/niiflux_err_sub_bpt
                hbetaflux_sn_sub_bpt = hbetaflux_sub_bpt*1e17/hbetaflux_err_sub_bpt
                halpflux_sn_sub_bpt = halpflux_sub_bpt*1e17/halpflux_err_sub_bpt
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
                diffd4000_bptplus = (self.eldiag.d4000[agn_ind] - self.eldiag.d4000_plus[sf_plus_inds])**2
                diffmass_bptplus = (self.eldiag.mass[agn_ind] - self.eldiag.mass_plus[sf_plus_inds])**2
                difffibmass_bptplus = (self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass_plus[sf_plus_inds])**2
                diffz_bptplus = (self.eldiag.z[agn_ind]-self.eldiag.z_plus[sf_plus_inds])**2/np.std(redshift_m2)
                diff_av_bptplus = (self.eldiag.av_gsw[agn_ind]-self.eldiag.av_gsw_plus[sf_plus_inds])**2
                diffs_bptplus = np.sqrt(diffd4000_bptplus+diffmass_bptplus+difffibmass_bptplus+diffz_bptplus+diff_av_bptplus)

                bptdistrat_plus = (np.array(cosmo.luminosity_distance(self.eldiag.z_plus[sf_plus_inds]))/np.array(cosmo.luminosity_distance(self.eldiag.z[agn_ind])))**2 

                oiiiflux_sub_plus = self.eldiag.oiiiflux[agn_ind]-self.eldiag.oiiiflux_plus[sf_plus_inds]*bptdistrat_plus
                niiflux_sub_plus = self.eldiag.niiflux[agn_ind]-self.eldiag.niiflux_plus[sf_plus_inds]*bptdistrat_plus
                hbetaflux_sub_plus =self.eldiag.hbetaflux[agn_ind]-self.eldiag.hbetaflux_plus[sf_plus_inds]*bptdistrat_plus
                halpflux_sub_plus  = self.eldiag.halpflux[agn_ind]- self.eldiag.halpflux_plus[sf_plus_inds]*bptdistrat_plus
                
                oiiiflux_err_sub_plus = np.sqrt(self.eldiag.oiii_err_bpt[agn_ind]**2 + (self.eldiag.oiii_err_plus[sf_plus_inds]*bptdistrat_plus)**2)
                niiflux_err_sub_plus =  np.sqrt(self.eldiag.nii_err_bpt[agn_ind]**2 + (self.eldiag.nii_err_plus[sf_plus_inds]*bptdistrat_plus)**2)
                hbetaflux_err_sub_plus = np.sqrt(self.eldiag.hbeta_err_bpt[agn_ind]**2 + (self.eldiag.hbeta_err_plus[sf_plus_inds]*bptdistrat_plus)**2)
                halpflux_err_sub_plus  = np.sqrt(self.eldiag.halp_err_bpt[agn_ind]**2 + (self.eldiag.halp_err_plus[sf_plus_inds]*bptdistrat_plus)**2)
                
                oiiiflux_sn_sub_plus = oiiiflux_sub_plus*1e17/ oiiiflux_err_sub_plus
                niiflux_sn_sub_plus = niiflux_sub_plus*1e17/niiflux_err_sub_plus
                hbetaflux_sn_sub_plus = hbetaflux_sub_plus*1e17/hbetaflux_err_sub_plus
                halpflux_sn_sub_plus = halpflux_sub_plus*1e17/halpflux_err_sub_plus
  
                diffs_bptplus_sort = np.argsort(diffs_bptplus)    
                
                inds_high_sn_bptplus = np.where((oiiiflux_sn_sub_plus[diffs_bptplus_sort]>sncut) & (niiflux_sn_sub_plus[diffs_bptplus_sort]>sncut) &
                                            (hbetaflux_sn_sub_plus[diffs_bptplus_sort]>sncut) &(halpflux_sn_sub_plus[diffs_bptplus_sort] >sncut)&
                                                (self.eldiag.oiii_err_plus[diffs_bptplus_sort]*bptdistrat_plus[diffs_bptplus_sort] != 0 ) &
                                                (self.eldiag.hbeta_err_plus[diffs_bptplus_sort]*bptdistrat_plus[diffs_bptplus_sort] != 0 ) )[0]
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
                diffd4000_neither = (self.eldiag.d4000[agn_ind] - self.eldiag.d4000_neither)**2
                diffmass_neither = (self.eldiag.mass[agn_ind] - self.eldiag.mass_neither)**2
                difffibmass_neither = (self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass_neither)**2
                diffz_neither = (self.eldiag.z[agn_ind]-self.eldiag.z_neither)**2/np.std(redshift_m2)
                diff_av_agn_neither = (self.eldiag.av_gsw[agn_ind]-self.eldiag.av_gsw_neither)**2
                diffs_neither = np.sqrt(diffd4000_neither+diffmass_neither+difffibmass_neither+diffz_neither+diff_av_agn_neither)

                bptdistrat_neither = (np.array(cosmo.luminosity_distance(self.eldiag.z_neither))/np.array(cosmo.luminosity_distance(self.eldiag.z[agn_ind])))**2 
                        

                oiiiflux_sub_neither = self.eldiag.oiiiflux[agn_ind]-self.eldiag.oiiiflux_neither*bptdistrat_neither
                niiflux_sub_neither = self.eldiag.niiflux[agn_ind]-self.eldiag.niiflux_neither*bptdistrat_neither
                hbetaflux_sub_neither =self.eldiag.hbetaflux[agn_ind]-self.eldiag.hbetaflux_neither*bptdistrat_neither
                halpflux_sub_neither  = self.eldiag.halpflux[agn_ind]- self.eldiag.halpflux_neither*bptdistrat_neither
                
                oiiiflux_err_sub_neither = np.sqrt(self.eldiag.oiii_err_bpt[agn_ind]**2 + (self.eldiag.oiii_err_neither*bptdistrat_neither)**2)
                niiflux_err_sub_neither =  np.sqrt(self.eldiag.nii_err_bpt[agn_ind]**2 + (self.eldiag.nii_err_neither*bptdistrat_neither)**2)
                hbetaflux_err_sub_neither = np.sqrt(self.eldiag.hbeta_err_bpt[agn_ind]**2 + (self.eldiag.hbeta_err_neither*bptdistrat_neither)**2)
                halpflux_err_sub_neither  = np.sqrt(self.eldiag.halp_err_bpt[agn_ind]**2 + (self.eldiag.halp_err_neither*bptdistrat_neither)**2)
                
                oiiiflux_sn_sub_neither = oiiiflux_sub_neither*1e17/ oiiiflux_err_sub_neither
                niiflux_sn_sub_neither = niiflux_sub_neither*1e17/niiflux_err_sub_neither
                hbetaflux_sn_sub_neither = hbetaflux_sub_neither*1e17/hbetaflux_err_sub_neither
                halpflux_sn_sub_neither = halpflux_sub_neither*1e17/halpflux_err_sub_neither
  
                diffs_neither_sort = np.argsort(diffs_neither)    
  
                inds_high_sn_neither = np.where((oiiiflux_sn_sub_neither[diffs_neither_sort]>sncut) & (niiflux_sn_sub_neither[diffs_neither_sort]>sncut) &
                                                (hbetaflux_sn_sub_neither[diffs_neither_sort]>sncut) &(halpflux_sn_sub_neither[diffs_neither_sort] >sncut) &
                                                (self.eldiag.oiii_err_neither[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) &
                                                (self.eldiag.hbeta_err_neither[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) &
                                                (self.eldiag.halp_err_neither[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) &
                                                (self.eldiag.nii_err_neither[diffs_neither_sort]*bptdistrat_neither[diffs_neither_sort] != 0 ) )[0]
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
                minid_out =[ self.eldiag.ids[sf_inds[mindiff_ind_bpt]],  
                            self.eldiag.ids_plus[sf_plus_inds[mindiff_ind_bptplus]],
                            self.eldiag.ids_neither[mindiff_ind_neither]]
                minind_out =[ sf_inds[mindiff_ind_bpt],  
                            sf_plus_inds[mindiff_ind_bptplus],
                            mindiff_ind_neither]
                #assigning the ids, inds, dists to be saved for self-matching

                minid_outagn = [self.eldiag.ids[agn_inds[mindiff_ind_agn]],  
                            self.eldiag.ids_plus[agnplus_inds[mindiff_ind_agnplus]]]
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



    def subtract_elflux(self, sncut=2, halphbeta_sncut=2):
    
        '''
        Subtracting the SF component and setting a number of attributes that relate.
        '''
        #######################
        #BPT fluxes
        #######################
        
        self.bptdistrat = (np.array(cosmo.luminosity_distance(self.eldiag.z[self.sfs]))/np.array(cosmo.luminosity_distance(self.eldiag.z[self.agns])))**2 
        self.sdssids_bpt = self.eldiag.ids[self.agns]
        self.oiiiflux_sub  =  np.copy(self.eldiag.oiiiflux[self.agns] - self.eldiag.oiiiflux[self.sfs]*self.bptdistrat)
        self.oiiiflux_sfratio = np.copy(self.eldiag.oiiiflux[self.sfs]*self.bptdistrat / self.eldiag.oiiiflux[self.agns])
        self.niiflux_sub  = np.copy(self.eldiag.niiflux[self.agns] - self.eldiag.niiflux[self.sfs]*self.bptdistrat)
        self.niiflux_sfratio = np.copy(self.eldiag.niiflux[self.sfs]*self.bptdistrat / self.eldiag.niiflux[self.agns])
        self.oiflux_sub  = np.copy(self.eldiag.oiflux[self.agns] - self.eldiag.oiflux[self.sfs]*self.bptdistrat)
        self.oiflux_sfratio = np.copy(self.eldiag.oiflux[self.sfs]*self.bptdistrat / self.eldiag.oiflux[self.agns])
        self.siiflux_sub  = np.copy(self.eldiag.siiflux[self.agns] - self.eldiag.siiflux[self.sfs]*self.bptdistrat)
        self.siiflux_sfratio = np.copy(self.eldiag.siiflux[self.sfs]*self.bptdistrat / self.eldiag.siiflux[self.agns])
        self.oiiflux_sub  = np.copy(self.eldiag.oiiflux[self.agns] - self.eldiag.oiiflux[self.sfs]*self.bptdistrat)
        self.oiiflux_sfratio = np.copy(self.eldiag.oiiflux[self.sfs]*self.bptdistrat / self.eldiag.oiiflux[self.agns])       
        self.hbetaflux_sub = np.copy(self.eldiag.hbetaflux[self.agns] - self.eldiag.hbetaflux[self.sfs]*self.bptdistrat)
        self.hbetaflux_sfratio = np.copy(self.eldiag.hbetaflux[self.sfs]*self.bptdistrat / self.eldiag.hbetaflux[self.agns])
        self.halpflux_sub = np.copy(self.eldiag.halpflux[self.agns] - self.eldiag.halpflux[self.sfs]* self.bptdistrat)
        self.halpflux_sfratio = np.copy(self.eldiag.halpflux[self.sfs]*self.bptdistrat / self.eldiag.halpflux[self.agns])

        self.oiii_eqw = np.copy(self.eldiag.oiii_eqw[self.agns])
        self.oiiiflux  =  np.copy(self.eldiag.oiiiflux[self.agns])
        self.niiflux  = np.copy(self.eldiag.niiflux[self.agns])
        self.hbetaflux = np.copy(self.eldiag.hbetaflux[self.agns]) 
        self.halpflux = np.copy(self.eldiag.halpflux[self.agns]) 
        self.oiflux  = np.copy(self.eldiag.oiflux[self.agns] )
        self.siiflux  = np.copy(self.eldiag.siiflux[self.agns])
        self.oiiflux = np.copy(self.eldiag.oiiflux[self.agns])
        self.hbetaflux_err = np.copy(self.eldiag.hbeta_err_bpt[self.agns])
        self.halpflux_err = np.copy(self.eldiag.halp_err_bpt[self.agns])
        self.niiflux_err = np.copy(self.eldiag.nii_err_bpt[self.agns])
        self.oiiiflux_err = np.copy(self.eldiag.oiii_err_bpt[self.agns])
        self.oiiflux_err = np.copy(self.eldiag.oii_err_bpt[self.agns])
        self.oiflux_err  = np.copy(self.eldiag.oi_err_bpt[self.agns])
        self.siiflux_err = np.copy(self.eldiag.sii_err_bpt[self.agns])

        self.oiiiflux_match  =  np.copy(self.eldiag.oiiiflux[self.sfs])
        self.niiflux_match  = np.copy(self.eldiag.niiflux[self.sfs])
        self.hbetaflux_match = np.copy(self.eldiag.hbetaflux[self.sfs]) 
        self.halpflux_match = np.copy(self.eldiag.halpflux[self.sfs]) 
        self.hbetaflux_err_match = np.copy(self.eldiag.hbeta_err_bpt[self.sfs])
        self.halpflux_err_match = np.copy(self.eldiag.halp_err_bpt[self.sfs])
        self.halpflux_sn_match = np.copy(self.halpflux_match*1e17/self.halpflux_err_match)
        self.hbetaflux_sn_match = np.copy(self.hbetaflux_match*1e17/self.hbetaflux_err_match)

        self.halphbeta_ratio = np.copy(self.halpflux/self.hbetaflux)
        self.halphbeta_ratio_match = np.copy(self.halpflux_match/self.hbetaflux_match)
        
        self.oiiiflux_sub_err = np.copy(np.sqrt(self.eldiag.oiii_err_bpt[self.agns]**2 +(self.eldiag.oiii_err_bpt[self.sfs]*self.bptdistrat)**2))
        self.oiiflux_sub_err = np.copy(np.sqrt(self.eldiag.oii_err_bpt[self.agns]**2 +(self.eldiag.oii_err_bpt[self.sfs]*self.bptdistrat)**2))
        self.oiflux_sub_err = np.copy(np.sqrt(self.eldiag.oi_err_bpt[self.agns]**2 +(self.eldiag.oi_err_bpt[self.sfs]*self.bptdistrat)**2))
        self.siiflux_sub_err = np.copy(np.sqrt(self.eldiag.sii_err_bpt[self.agns]**2 +(self.eldiag.sii_err_bpt[self.sfs]*self.bptdistrat)**2))
        self.niiflux_sub_err = np.copy(np.sqrt(self.eldiag.nii_err_bpt[self.agns]**2+(self.eldiag.nii_err_bpt[self.sfs]*self.bptdistrat)**2))        
        self.hbetaflux_sub_err = np.copy(np.sqrt(self.eldiag.hbeta_err_bpt[self.agns]**2+(self.eldiag.hbeta_err_bpt[self.sfs]*self.bptdistrat)**2))
        self.halpflux_sub_err = np.copy(np.sqrt(self.eldiag.halp_err_bpt[self.agns]**2+(self.eldiag.halp_err_bpt[self.sfs]*self.bptdistrat)**2))
        self.halphbeta_ratio_err = np.copy(self.halphbeta_ratio*np.sqrt((self.halpflux_err/(self.halpflux*1e17))**2 + 
                                                                (self.hbetaflux_err/(self.hbetaflux*1e17))**2) )
        self.halphbeta_ratio_err_match = np.copy(self.halphbeta_ratio_match*np.sqrt((self.halpflux_err_match/(self.halpflux_match*1e17))**2 + 
                                                                (self.hbetaflux_err_match/(self.hbetaflux_match*1e17))**2) )
        
        self.halphbeta_sub_ratio  = np.copy(self.halpflux_sub/self.hbetaflux_sub)
        self.halphbeta_ratio_sub_err = np.copy(self.halphbeta_sub_ratio*np.sqrt((self.halpflux_sub_err/(self.halpflux_sub*1e17))**2 + 
                                                                (self.hbetaflux_sub_err/(self.hbetaflux_sub*1e17))**2) )
        self.halphbeta_sub_sn = np.copy(self.halphbeta_sub_ratio/self.halphbeta_ratio_sub_err)
        self.halphbeta_sn_match = np.copy(self.halphbeta_ratio_match/self.halphbeta_ratio_err_match)
        self.halphbeta_sub_sn_filt_bpt = np.where(self.halphbeta_sub_sn>halphbeta_sncut)[0]
        self.halphbeta_sub_and_match_sn_filt_bpt = np.where((self.halphbeta_sub_sn > halphbeta_sncut) &(self.halphbeta_sn_match > halphbeta_sncut))

        self.halpflux_sn = np.copy(self.halpflux*1e17/self.halpflux_err)
        self.hbetaflux_sn = np.copy(self.hbetaflux*1e17/self.hbetaflux_err)
        self.halphbeta_sn = np.copy(self.halphbeta_ratio/self.halphbeta_ratio_err)
        self.oiiiflux_sn = np.copy(self.oiiiflux*1e17/self.oiiiflux_err)
        self.oiiflux_sn = np.copy(self.oiiflux*1e17/self.oiiflux_err)
        self.oiflux_sn = np.copy(self.oiflux*1e17/self.oiflux_err)

        self.niiflux_sn = np.copy(self.niiflux*1e17/self.niiflux_err)
        self.siiflux_sn = np.copy(self.hbetaflux*1e17/self.siiflux_err)
        
        #1e17 factor because flux converted to ergs/s in ELObj.py
        self.oiiflux_sub_sn = np.copy(self.oiiflux_sub*1e17/self.oiiflux_sub_err)
        self.oiflux_sub_sn = np.copy(self.oiflux_sub*1e17/self.oiflux_sub_err)
        self.siiflux_sub_sn = np.copy(self.siiflux_sub*1e17/self.siiflux_sub_err)
        self.oiiiflux_sub_sn = np.copy(self.oiiiflux_sub*1e17/self.oiiiflux_sub_err)
        self.niiflux_sub_sn = np.copy(self.niiflux_sub*1e17/self.niiflux_sub_err)
        self.hbetaflux_sub_sn = np.copy(self.hbetaflux_sub*1e17/self.hbetaflux_sub_err)
        self.halpflux_sub_sn = np.copy(self.halpflux_sub*1e17/self.halpflux_sub_err)
        #######################
        #bpt filter
        #######################
        self.bpt_sn_filt_bool = ((self.oiiiflux_sub_sn > sncut)& (self.niiflux_sub_sn > sncut)& 
                                    (self.hbetaflux_sub_sn > sncut)& (self.halpflux_sub_sn > sncut))
        
        self.bpt_sn_filt_bool_intermed = ((self.oiiiflux_sub_sn > sncut-1)& (self.niiflux_sub_sn > sncut-1)& 
                                    (self.hbetaflux_sub_sn > sncut-1)& (self.halpflux_sub_sn > sncut-1)&
                                    ((self.oiiiflux_sub_sn < sncut) | (self.niiflux_sub_sn < sncut) | 
                                    (self.hbetaflux_sub_sn < sncut) | (self.halpflux_sub_sn < sncut)))
        self.bpt_sn_filt = np.where(self.bpt_sn_filt_bool)[0]
        self.bpt_sn_filt_intermed = np.where(self.bpt_sn_filt_bool_intermed)[0]

        self.bpt_not_sn_filt_bool = np.logical_not(self.bpt_sn_filt_bool)
        self.bpt_not_sn_filt = np.where(self.bpt_not_sn_filt_bool)[0]
        
        self.av_sub_bpt_agn = self.eldiag.extinction(ha=self.halpflux_sub, hb=self.hbetaflux_sub, agn=True)
        self.av_sub_bpt_sf = self.eldiag.extinction(ha=self.halpflux_sub, hb=self.hbetaflux_sub, agn=False)
        self.av_bpt_agn = self.eldiag.extinction(ha=self.halpflux, hb= self.hbetaflux, agn=True)
        self.av_bpt_sf = np.copy(self.eldiag.av[self.sfs])
        
        self.oiiiflux_sub_dered_bpt = self.eldiag.dustcorrect(self.oiiiflux_sub, av=self.av_sub_bpt_agn)
        self.oiiiflux_dered_bpt = self.eldiag.dustcorrect(self.oiiiflux, av= self.av_bpt_agn)
        self.oiflux_sub_dered_bpt = self.eldiag.dustcorrect(self.oiflux_sub, av=self.av_sub_bpt_agn)
        self.oiflux_dered_bpt = self.eldiag.dustcorrect(self.oiflux, av= self.av_bpt_agn)
        self.halpflux_sub_dered_bpt = self.eldiag.dustcorrect(self.halpflux_sub, av=self.av_sub_bpt_agn)
        self.halpflux_dered_bpt = self.eldiag.dustcorrect(self.halpflux, av= self.av_bpt_agn)
        self.hbetaflux_sub_dered_bpt = self.eldiag.dustcorrect(self.hbetaflux_sub, av=self.av_sub_bpt_agn)
        self.hbetaflux_dered_bpt = self.eldiag.dustcorrect(self.hbetaflux, av= self.av_bpt_agn)

        self.oiiflux_sub_dered_bpt = self.eldiag.dustcorrect(self.oiiflux_sub, av=self.av_sub_bpt_agn)
        self.oiiflux_dered_bpt = self.eldiag.dustcorrect(self.oiiflux, av=self.av_bpt_agn)
        self.niiflux_sub_dered_bpt = self.eldiag.dustcorrect(self.niiflux_sub, av=self.av_sub_bpt_agn)
        self.niiflux_dered_bpt = self.eldiag.dustcorrect(self.niiflux, av=self.av_bpt_agn)

        self.good_oiii_oii_nii_sub_bpt = np.where((self.oiiiflux_sub_sn >2) &(self.oiiflux_sub_sn>2)&
                                          (self.niiflux_sub_sn>2))[0]

        self.good_oi_sub_bpt = np.where((self.oiflux_sub_sn >2))[0]
        self.good_oi_bpt = np.where((self.oiflux_sn >2))[0]

        self.good_oiii_oii_nii_bpt = np.where((self.oiiiflux_sn >2) &(self.oiiflux_sn>2)&
                                          (self.niiflux_sn>2))[0]
        self.good_oiii_hb_sii_ha_sub_bpt = np.where((self.oiiiflux_sub_sn >2) &(self.hbetaflux_sub_sn>2)&
                                          (self.siiflux_sub_sn>2)&(self.halpflux_sub_sn>2))[0]
        self.good_oiii_hb_sii_ha_bpt = np.where((self.oiiiflux_sn >2) &(self.hbetaflux_sn>2)&
                                          (self.siiflux_sn>2) &(self.halpflux_sn>2))[0]


        self.z_bpt = np.copy(self.eldiag.z[self.agns]) 
        self.mass_bpt = np.copy(self.eldiag.mass[self.agns])
        self.fibmass_bpt = np.copy(self.eldiag.fibmass[self.agns])
        
        self.av_gsw_bpt_agn = np.copy(self.eldiag.av_gsw[self.agns])
        self.av_gsw_bpt_sf = np.copy(self.eldiag.av_gsw[self.sfs])

        self.filt_by_av_bpt = np.where(self.halphbeta_sn_match > halphbeta_sncut)[0]
        
        self.av_bpt_sf_bn_edges, self.av_bpt_sf_bncenters, self.av_bpt_sf_bns, self.av_bpt_sf_bn_inds, self.av_bpt_sf_valbns = self.bin_quantity(self.av_bpt_sf[self.filt_by_av_bpt], 0.3, -6, 6, threshold=100)
        self.av_sub_bpt_agn_binned  = self.bin_by_ind(self.av_sub_bpt_agn, self.av_bpt_sf_bn_inds, self.av_bpt_sf_bncenters[self.av_bpt_sf_valbns])
        self.hbeta_flux_sub_bpt_agn_binned = self.bin_by_ind(self.hbetaflux_sub[self.filt_by_av_bpt], self.av_bpt_sf_bn_inds, self.av_bpt_sf_bncenters[self.av_bpt_sf_valbns])
        self.halpha_flux_sub_bpt_agn_binned = self.bin_by_ind(self.halpflux_sub[self.filt_by_av_bpt], self.av_bpt_sf_bn_inds, self.av_bpt_sf_bncenters[self.av_bpt_sf_valbns])
        self.bootstrapped_av_bpt_sf_inds = [self.bootstrap(np.array(self.av_bpt_sf_bn_inds[i]), 1000, data_only=True) for i in range(len(self.av_bpt_sf_bn_inds))]
        self.bootstrapped_halpha_mean_sums = []
        self.bootstrapped_halpha_std_sums = []
        self.bootstrapped_hbeta_mean_sums = []
        self.bootstrapped_hbeta_std_sums = []
        
        for i in range(len(self.bootstrapped_av_bpt_sf_inds)): 
            halp_flux_sums = np.sum(self.halpflux_sub[self.filt_by_av_bpt][np.int64(self.bootstrapped_av_bpt_sf_inds[i])], axis=1)
            mean_halp_flux_sums = np.mean(halp_flux_sums)
            std_halp_flux_sums = np.std(halp_flux_sums)
            self.bootstrapped_halpha_mean_sums.append(mean_halp_flux_sums)
            self.bootstrapped_halpha_std_sums.append(std_halp_flux_sums)
            
            hbeta_flux_sums = np.sum(self.hbetaflux_sub[self.filt_by_av_bpt][np.int64(self.bootstrapped_av_bpt_sf_inds[i])], axis=1)
            mean_hbeta_flux_sums = np.mean(hbeta_flux_sums)
            std_hbeta_flux_sums = np.std(hbeta_flux_sums)
            self.bootstrapped_hbeta_mean_sums.append(mean_hbeta_flux_sums)
            self.bootstrapped_hbeta_std_sums.append(std_hbeta_flux_sums)
            
        self.bootstrapped_halpha_mean_sums = np.array(self.bootstrapped_halpha_mean_sums)
        self.bootstrapped_halpha_std_sums = np.array(self.bootstrapped_halpha_std_sums)
        self.bootstrapped_hbeta_mean_sums = np.array(self.bootstrapped_hbeta_mean_sums)
        self.bootstrapped_hbeta_std_sums = np.array(self.bootstrapped_hbeta_std_sums)
        self.bootstrapped_halphbeta = self.bootstrapped_halpha_mean_sums/self.bootstrapped_hbeta_mean_sums
        self.bootstrapped_halphbeta_err = self.bootstrapped_halphbeta*np.sqrt( (self.bootstrapped_halpha_std_sums/self.bootstrapped_halpha_mean_sums)**2+
                                                                              (self.bootstrapped_hbeta_std_sums/self.bootstrapped_hbeta_mean_sums)**2)
        
            
        
        self.d4000_bpt = np.copy(self.eldiag.d4000[self.agns])
        self.ssfr_bpt = np.copy(self.eldiag.ssfr[self.agns])
        self.sfr_bpt = np.copy(self.eldiag.sfr[self.agns])
        self.massfrac_bpt = np.copy(self.eldiag.massfracgsw[self.agns])
        self.fibsfr_bpt = np.copy(self.sfr_bpt + np.log10(self.massfrac_bpt))
        self.fibsfr_mpa_bpt = np.copy(self.eldiag.fibsfr_mpa[self.agns])
        self.fibssfr_mpa_bpt = np.copy(self.eldiag.fibssfr_mpa[self.agns])
        self.fibsfr_mpa_match_bpt = np.copy(self.eldiag.fibsfr_mpa[self.sfs])
        self.fibssfr_mpa_match_bpt = np.copy(self.eldiag.fibssfr_mpa[self.sfs])

        self.sfr_match_bpt = np.copy(self.eldiag.sfr[self.sfs])
        self.mass_match_bpt = np.copy(self.eldiag.mass[self.sfs])
        self.ssfr_match_bpt = self.sfr_match_bpt-self.mass_match_bpt
        self.massfrac_match_bpt = np.copy(self.eldiag.massfracgsw[self.sfs])
        self.z_match_bpt = np.copy(self.eldiag.z[self.sfs])
        self.fibmass_match_bpt = np.copy(self.eldiag.fibmass[self.sfs])
        self.fibsfr_match_bpt = np.copy(self.sfr_match_bpt + np.log10(self.massfrac_match_bpt))
        self.halpfibsfr_match_bpt = np.copy(self.eldiag.halpfibsfr[self.sfs])
        self.halpfibssfr_match_bpt = np.copy(self.halpfibsfr_match_bpt-self.fibmass_match_bpt)

        self.halplum_bpt = getlumfromflux(self.halpflux, self.z_bpt)
        self.oiiilum_bpt = getlumfromflux(self.oiiiflux, self.z_bpt)
        self.oilum_bpt = getlumfromflux(self.oiflux, self.z_bpt)
        self.oiiilum_dered_bpt = getlumfromflux(self.oiiiflux_dered_bpt, self.z_bpt)
        self.oilum_dered_bpt = getlumfromflux(self.oiflux_dered_bpt, self.z_bpt)
        self.halplum_dered_bpt = getlumfromflux(self.halpflux_dered_bpt, self.z_bpt)

        self.oiiilum_sub_bpt = getlumfromflux(self.oiiiflux_sub, self.z_bpt)
        self.oilum_sub_bpt = getlumfromflux(self.oiflux_sub, self.z_bpt)
        self.halplum_sub_bpt = getlumfromflux(self.halpflux_sub, self.z_bpt)

        self.oiiilum_sub_dered_bpt = getlumfromflux(self.oiiiflux_sub_dered_bpt, self.z_bpt)
        self.oilum_sub_dered_bpt = getlumfromflux(self.oiflux_sub_dered_bpt, self.z_bpt)
        self.halplum_sub_dered_bpt = getlumfromflux(self.halpflux_sub_dered_bpt, self.z_bpt)

        self.vdisp_bpt = np.copy(self.eldiag.vdisp[self.agns])
        
        self.log_oiii_hbeta_sub = np.copy(np.log10(self.oiiiflux_sub/self.hbetaflux_sub))
        self.log_nii_halpha_sub = np.copy(np.log10(self.niiflux_sub/self.halpflux_sub))
        self.log_sii_halpha_sub = np.copy(np.log10(self.siiflux_sub/self.halpflux_sub))
        self.log_oi_halpha_sub = np.copy(np.log10(self.oiflux_sub/self.halpflux_sub))

        self.log_oiii_hbeta = np.copy(np.log10(self.oiiiflux/self.hbetaflux))
        self.log_nii_halpha = np.copy(np.log10(self.niiflux/self.halpflux))
        self.log_sii_halpha = np.copy(np.log10(self.siiflux/self.halpflux))
        self.log_oi_halpha = np.copy(np.log10(self.oiflux/self.halpflux))

        self.offset_oiii_hbeta = np.copy(self.log_oiii_hbeta_sub - self.log_oiii_hbeta) 
        self.offset_nii_halpha = np.copy(self.log_nii_halpha_sub - self.log_nii_halpha )
        
        #computing values for bpt plus matches
        self.bptdistrat_plus = (np.array(cosmo.luminosity_distance(self.eldiag.z_plus[self.sfs_plus]))/np.array(cosmo.luminosity_distance(self.eldiag.z[self.agns_plus])))**2 #Mpc to cm

        self.sdssids_plus = self.eldiag.ids[self.agns_plus]
        
        self.oiiiflux_sub_plus = np.copy(self.eldiag.oiiiflux[self.agns_plus] - self.eldiag.oiiiflux_plus[self.sfs_plus]*self.bptdistrat_plus)
        self.oiiiflux_sfratio_plus = np.copy(self.eldiag.oiiiflux_plus[self.sfs_plus]*self.bptdistrat_plus / self.eldiag.oiiiflux[self.agns_plus])
        self.niiflux_sub_plus = np.copy(self.eldiag.niiflux[self.agns_plus] - self.eldiag.niiflux_plus[self.sfs_plus]*self.bptdistrat_plus)
        self.niiflux_sfratio_plus = np.copy(self.eldiag.niiflux_plus[self.sfs_plus]*self.bptdistrat_plus / self.eldiag.niiflux[self.agns_plus])
        self.hbetaflux_sub_plus =  np.copy(self.eldiag.hbetaflux[self.agns_plus] - self.eldiag.hbetaflux_plus[self.sfs_plus]*self.bptdistrat_plus)
        self.hbetaflux_sfratio_plus = np.copy(self.eldiag.hbetaflux_plus[self.sfs_plus]*self.bptdistrat_plus / self.eldiag.hbetaflux[self.agns_plus])
        self.halpflux_sub_plus = np.copy(self.eldiag.halpflux[self.agns_plus] - self.eldiag.halpflux_plus[self.sfs_plus]*self.bptdistrat_plus)
        self.halpflux_sfratio_plus = np.copy(self.eldiag.halpflux_plus[self.sfs_plus]*self.bptdistrat_plus / self.eldiag.halpflux[self.agns_plus])
        self.siiflux_sub_plus  = np.copy(self.eldiag.siiflux[self.agns_plus] - self.eldiag.siiflux[self.sfs_plus]*self.bptdistrat_plus)
        self.siiflux_sfratio_plus = np.copy(self.eldiag.siiflux_plus[self.sfs_plus]*self.bptdistrat_plus / self.eldiag.siiflux[self.agns_plus])
        self.oiiflux_sub_plus  = np.copy(self.eldiag.oiiflux[self.agns_plus] - self.eldiag.oiiflux[self.sfs_plus]*self.bptdistrat_plus)
        self.oiiflux_sfratio_plus = np.copy(self.eldiag.oiiflux_plus[self.sfs_plus]*self.bptdistrat_plus / self.eldiag.oiiflux[self.agns_plus])
        self.oiflux_sub_plus  = np.copy(self.eldiag.oiflux[self.agns_plus] - self.eldiag.oiflux[self.sfs_plus]*self.bptdistrat_plus)
        self.oiflux_sfratio_plus = np.copy(self.eldiag.oiflux_plus[self.sfs_plus]*self.bptdistrat_plus / self.eldiag.oiflux[self.agns_plus])
        self.oiiiflux_plus = np.copy(self.eldiag.oiiiflux[self.agns_plus])
        self.niiflux_plus = np.copy(self.eldiag.niiflux[self.agns_plus] )
        self.hbetaflux_plus = np.copy(self.eldiag.hbetaflux[self.agns_plus]) 
        self.halpflux_plus = np.copy(self.eldiag.halpflux[self.agns_plus] )
        self.siiflux_plus  = np.copy(self.eldiag.siiflux[self.agns_plus])
        self.oiiflux_plus  = np.copy(self.eldiag.oiiflux[self.agns_plus])
        self.oiflux_plus  = np.copy(self.eldiag.oiflux[self.agns_plus])
                                     
        self.hbetaflux_err_plus = np.copy(self.eldiag.hbeta_err_bpt[self.agns_plus])
        self.halpflux_err_plus = np.copy(self.eldiag.halp_err_bpt[self.agns_plus])
        
        self.oiiiflux_err_plus = np.copy(self.eldiag.oiii_err_bpt[self.agns_plus])
        self.oiiflux_err_plus = np.copy(self.eldiag.oii_err_bpt[self.agns_plus])
        self.oiflux_err_plus = np.copy(self.eldiag.oi_err_bpt[self.agns_plus])
        
        self.niiflux_err_plus = np.copy(self.eldiag.nii_err_bpt[self.agns_plus])
        self.siiflux_err_plus = np.copy(self.eldiag.sii_err_bpt[self.agns_plus])
        
        self.hbetaflux_match_plus = np.copy(self.eldiag.hbetaflux_plus[self.sfs_plus]) 
        self.halpflux_match_plus = np.copy(self.eldiag.halpflux_plus[self.sfs_plus] )
        self.hbetaflux_err_match_plus = np.copy(self.eldiag.hbeta_err_plus[self.sfs_plus])
        self.halpflux_err_match_plus = np.copy(self.eldiag.halp_err_plus[self.sfs_plus])
        self.halpflux_sn_match_plus = np.copy(self.halpflux_match_plus*1e17/self.halpflux_err_match_plus)
        self.hbetaflux_sn_match_plus = np.copy(self.hbetaflux_match_plus*1e17/self.hbetaflux_err_match_plus)        
        
        self.halphbeta_ratio_plus = np.copy(self.halpflux_plus/self.hbetaflux_plus)
        self.halphbeta_ratio_err_plus = np.copy(self.halphbeta_ratio_plus*np.sqrt( (self.halpflux_err_plus/(self.halpflux_plus*1e17))**2 
                                                                          +(self.hbetaflux_err_plus/(self.hbetaflux_plus*1e17))**2))
        self.halphbeta_ratio_match_plus = np.copy(self.halpflux_match_plus/self.hbetaflux_match_plus)
        self.halphbeta_ratio_err_match_plus = np.copy(self.halphbeta_ratio_match_plus*np.sqrt( (self.halpflux_err_match_plus/(self.halpflux_match_plus*1e17))**2 
                                                                          +(self.hbetaflux_err_match_plus/(self.hbetaflux_match_plus*1e17))**2))
        
        self.halpflux_sn_plus = np.copy(self.halpflux_plus*1e17/self.halpflux_err_plus)
        self.hbetaflux_sn_plus = np.copy(self.hbetaflux_plus*1e17/self.hbetaflux_err_plus)
        self.oiiiflux_sn_plus = np.copy(self.oiiiflux_plus*1e17/self.oiiiflux_err_plus)
        self.oiiflux_sn_plus = np.copy(self.oiiflux_plus*1e17/self.oiiflux_err_plus)
        self.oiflux_sn_plus = np.copy(self.oiflux_plus*1e17/self.oiflux_err_plus)

        self.siiflux_sn_plus = np.copy(self.siiflux_plus*1e17/self.siiflux_err_plus)
        self.niiflux_sn_plus = np.copy(self.niiflux_plus*1e17/self.niiflux_err_plus)

        self.halphbeta_sn_plus = np.copy(self.halphbeta_ratio_plus/self.halphbeta_ratio_err_plus)


        
        self.oiiiflux_sub_err_plus = np.copy(np.sqrt(self.eldiag.oiii_err_bpt[self.agns_plus]**2 +(self.eldiag.oiii_err_plus[self.sfs_plus]*self.bptdistrat_plus)**2))
        self.oiiflux_sub_err_plus = np.copy(np.sqrt(self.eldiag.oii_err_bpt[self.agns_plus]**2 +(self.eldiag.oii_err_plus[self.sfs_plus]*self.bptdistrat_plus)**2))
        self.oiflux_sub_err_plus = np.copy(np.sqrt(self.eldiag.oi_err_bpt[self.agns_plus]**2 +(self.eldiag.oi_err_plus[self.sfs_plus]*self.bptdistrat_plus)**2))
        self.siiflux_sub_err_plus = np.copy(np.sqrt(self.eldiag.sii_err_bpt[self.agns_plus]**2 +(self.eldiag.sii_err_plus[self.sfs_plus]*self.bptdistrat_plus)**2))
        self.niiflux_sub_err_plus = np.copy(np.sqrt(self.eldiag.nii_err_bpt[self.agns_plus]**2+(self.eldiag.nii_err_plus[self.sfs_plus]*self.bptdistrat_plus)**2))
        self.hbetaflux_sub_err_plus = np.copy(np.sqrt(self.eldiag.hbeta_err_bpt[self.agns_plus]**2+(self.eldiag.hbeta_err_plus[self.sfs_plus]*self.bptdistrat_plus)**2))
        self.halpflux_sub_err_plus = np.copy(np.sqrt(self.eldiag.halp_err_bpt[self.agns_plus]**2+(self.eldiag.halp_err_plus[self.sfs_plus]*self.bptdistrat_plus)**2))
        
        self.oiiiflux_sub_sn_plus = np.copy(self.oiiiflux_sub_plus*1e17/self.oiiiflux_sub_err_plus)
        self.oiiflux_sub_sn_plus = np.copy(self.oiiflux_sub_plus*1e17/self.oiiflux_sub_err_plus)
        self.oiflux_sub_sn_plus = np.copy(self.oiflux_sub_plus*1e17/self.oiflux_sub_err_plus)

        self.siiflux_sub_sn_plus = np.copy(self.siiflux_sub_plus*1e17/self.siiflux_sub_err_plus)
        
        self.niiflux_sub_sn_plus = np.copy(self.niiflux_sub_plus*1e17/self.niiflux_sub_err_plus)
        self.hbetaflux_sub_sn_plus = np.copy(self.hbetaflux_sub_plus*1e17/self.hbetaflux_sub_err_plus)
        self.halpflux_sub_sn_plus= np.copy(self.halpflux_sub_plus*1e17/self.halpflux_sub_err_plus)
        self.halphbeta_sub_ratio_plus  = np.copy(self.halpflux_sub_plus/self.hbetaflux_sub_plus)
        self.halphbeta_ratio_sub_err_plus = np.copy(self.halphbeta_sub_ratio_plus*np.sqrt((self.halpflux_sub_err_plus/(self.halpflux_sub_plus*1e17))**2 + 
                                                                (self.hbetaflux_sub_err_plus/(self.hbetaflux_sub_plus*1e17))**2) )
        self.halphbeta_sub_sn_plus = np.copy(self.halphbeta_sub_ratio_plus/self.halphbeta_ratio_sub_err_plus)
        self.halphbeta_sn_match_plus = np.copy(self.halphbeta_ratio_match_plus/self.halphbeta_ratio_err_match_plus)
        self.halphbeta_sub_sn_filt_plus = np.where(self.halphbeta_sub_sn_plus>halphbeta_sncut)[0]
        self.halphbeta_sub_and_match_sn_filt_plus = np.where((self.halphbeta_sub_sn_plus > halphbeta_sncut) & (self.halphbeta_sn_match_plus > halphbeta_sncut))    
        self.bpt_plus_sn_filt_bool = ((self.oiiiflux_sub_sn_plus > sncut)& (self.niiflux_sub_sn_plus > sncut)& 
                                    (self.hbetaflux_sub_sn_plus > sncut)& (self.halpflux_sub_sn_plus > sncut))
        self.bpt_plus_sn_filt_bool_intermed = ((self.oiiiflux_sub_sn_plus > sncut-1)& (self.niiflux_sub_sn_plus > sncut-1)& 
                                    (self.hbetaflux_sub_sn_plus > sncut-1)& (self.halpflux_sub_sn_plus > sncut-1)&
                                    ((self.oiiiflux_sub_sn_plus < sncut) | (self.niiflux_sub_sn_plus < sncut) | 
                                    (self.hbetaflux_sub_sn_plus < sncut) | (self.halpflux_sub_sn_plus < sncut)))
        self.bpt_plus_sn_filt = np.where(self.bpt_plus_sn_filt_bool)[0]
        self.bpt_plus_sn_filt_intermed = np.where(self.bpt_plus_sn_filt_bool_intermed)[0]

        self.bpt_plus_not_sn_filt_bool = np.logical_not(self.bpt_plus_sn_filt_bool)
        self.bpt_plus_not_sn_filt = np.where(self.bpt_plus_not_sn_filt_bool)[0]

        
        
        self.av_plus_sf = np.copy(self.eldiag.av_plus[self.sfs_plus])
        self.av_plus_agn = self.eldiag.extinction(ha=self.halpflux_plus, hb=self.hbetaflux_plus, agn=True)
        self.av_sub_plus_agn = self.eldiag.extinction(ha=self.halpflux_sub_plus, hb=self.hbetaflux_sub_plus,agn=True)
        self.av_sub_plus_sf = self.eldiag.extinction(ha=self.halpflux_sub_plus, hb=self.hbetaflux_sub_plus, agn=False)
        self.av_gsw_plus_agn = np.copy(self.eldiag.av_gsw[self.agns_plus])
        self.av_gsw_plus_sf = np.copy(self.eldiag.av_gsw_plus[self.sfs_plus])
        
        self.filt_by_av_plus = np.where(self.halphbeta_sn_match_plus > halphbeta_sncut)[0]
        
        self.av_plus_sf_bn_edges, self.av_plus_sf_bncenters, self.av_plus_sf_bns, self.av_plus_sf_bn_inds, self.av_plus_sf_valbns = self.bin_quantity(self.av_plus_sf[self.filt_by_av_plus], 0.3, -6, 6, threshold=100)
        self.av_sub_plus_agn_binned  = self.bin_by_ind(self.av_sub_plus_agn, self.av_plus_sf_bn_inds, self.av_plus_sf_bncenters[self.av_plus_sf_valbns])
        self.hbeta_flux_sub_plus_agn_binned = self.bin_by_ind(self.hbetaflux_sub_plus[self.filt_by_av_plus], self.av_plus_sf_bn_inds, self.av_plus_sf_bncenters[self.av_plus_sf_valbns])
        self.halpha_flux_sub_plus_agn_binned = self.bin_by_ind(self.halpflux_sub_plus[self.filt_by_av_plus], self.av_plus_sf_bn_inds, self.av_plus_sf_bncenters[self.av_plus_sf_valbns])
        self.bootstrapped_av_plus_sf_inds = [self.bootstrap(np.array(self.av_plus_sf_bn_inds[i]), 1000, data_only=True) for i in range(len(self.av_plus_sf_bn_inds))]
        self.bootstrapped_halpha_mean_sums_plus = []
        self.bootstrapped_halpha_std_sums_plus = []
        self.bootstrapped_hbeta_mean_sums_plus = []
        self.bootstrapped_hbeta_std_sums_plus = []
        
        for i in range(len(self.bootstrapped_av_plus_sf_inds)): 
            halp_flux_sums_plus = np.sum(self.halpflux_sub_plus[self.filt_by_av_plus][np.int64(self.bootstrapped_av_plus_sf_inds[i])], axis=1)
            mean_halp_flux_sums_plus = np.mean(halp_flux_sums_plus)
            std_halp_flux_sums_plus = np.std(halp_flux_sums_plus)
            self.bootstrapped_halpha_mean_sums_plus.append(mean_halp_flux_sums_plus)
            self.bootstrapped_halpha_std_sums_plus.append(std_halp_flux_sums_plus)
            
            hbeta_flux_sums_plus = np.sum(self.hbetaflux_sub_plus[self.filt_by_av_plus][np.int64(self.bootstrapped_av_plus_sf_inds[i])], axis=1)
            mean_hbeta_flux_sums_plus = np.mean(hbeta_flux_sums_plus)
            std_hbeta_flux_sums_plus = np.std(hbeta_flux_sums_plus)
            self.bootstrapped_hbeta_mean_sums_plus.append(mean_hbeta_flux_sums_plus)
            self.bootstrapped_hbeta_std_sums_plus.append(std_hbeta_flux_sums_plus)
            
        self.bootstrapped_halpha_mean_sums_plus = np.array(self.bootstrapped_halpha_mean_sums_plus)
        self.bootstrapped_halpha_std_sums_plus = np.array(self.bootstrapped_halpha_std_sums_plus)
        self.bootstrapped_hbeta_mean_sums_plus = np.array(self.bootstrapped_hbeta_mean_sums_plus)
        self.bootstrapped_hbeta_std_sums_plus = np.array(self.bootstrapped_hbeta_std_sums_plus)
        self.bootstrapped_halphbeta_plus = self.bootstrapped_halpha_mean_sums_plus/self.bootstrapped_hbeta_mean_sums_plus
        self.bootstrapped_halphbeta_err_plus = self.bootstrapped_halphbeta_plus*np.sqrt( (self.bootstrapped_halpha_std_sums_plus/self.bootstrapped_halpha_mean_sums_plus)**2+
                                                                              (self.bootstrapped_hbeta_std_sums_plus/self.bootstrapped_hbeta_mean_sums_plus)**2)
        
        
        self.oiiiflux_sub_dered_plus = self.eldiag.dustcorrect(self.oiiiflux_sub_plus, av=self.av_sub_plus_agn)
        self.oiiiflux_dered_plus = self.eldiag.dustcorrect(self.oiiiflux_plus, av=self.av_plus_agn)
        self.oiflux_sub_dered_plus = self.eldiag.dustcorrect(self.oiflux_sub_plus, av=self.av_sub_plus_agn)
        self.oiflux_dered_plus = self.eldiag.dustcorrect(self.oiflux_plus, av=self.av_plus_agn)

        self.halpflux_sub_dered_plus = self.eldiag.dustcorrect(self.halpflux_sub_plus, av=self.av_sub_plus_agn)
        self.halpflux_dered_plus = self.eldiag.dustcorrect(self.halpflux_plus, av=self.av_plus_agn)

        self.oiiflux_sub_dered_plus = self.eldiag.dustcorrect(self.oiiflux_sub_plus, av=self.av_sub_plus_agn)
        self.oiiflux_dered_plus = self.eldiag.dustcorrect(self.oiiflux_plus, av=self.av_plus_agn)

        self.niiflux_sub_dered_plus = self.eldiag.dustcorrect(self.niiflux_sub_plus, av=self.av_sub_plus_agn)
        self.niiflux_dered_plus = self.eldiag.dustcorrect(self.niiflux_plus, av=self.av_plus_agn)

        self.good_oiii_oii_nii_sub_plus = np.where((self.oiiiflux_sub_sn_plus >2) &(self.oiiflux_sub_sn_plus>2)&
                                          (self.niiflux_sub_sn_plus>2))[0]
        self.good_oiii_oii_nii_plus = np.where((self.oiiiflux_sn_plus >2) &(self.oiiflux_sn_plus>2)&
                                          (self.niiflux_sn_plus>2))[0]
        self.good_oi_sub_plus = np.where((self.oiflux_sub_sn_plus >2))[0]

        self.good_oi_plus = np.where((self.oiflux_sn_plus >2))[0]

        self.good_oiii_hb_sii_ha_sub_plus = np.where((self.oiiiflux_sub_sn_plus >2) &(self.hbetaflux_sub_sn_plus>2)&
                                          (self.siiflux_sub_sn_plus>2) &(self.halpflux_sub_sn_plus>2))[0]
        self.good_oiii_hb_sii_ha_plus = np.where((self.oiiiflux_sn_plus >2) &(self.hbetaflux_sn_plus>2)&
                                          (self.niiflux_sn_plus>2)&(self.halpflux_sn_plus>2))[0]
        self.z_plus = np.copy(self.eldiag.z[self.agns_plus])
        self.mass_plus = np.copy(self.eldiag.mass[self.agns_plus])
        self.fibmass_plus = np.copy(self.eldiag.fibmass[self.agns_plus])

        self.d4000_plus = np.copy(self.eldiag.d4000[self.agns_plus])
        
        self.ssfr_plus = np.copy(self.eldiag.ssfr[self.agns_plus])
        self.sfr_plus = np.copy(self.eldiag.sfr[self.agns_plus])
        self.massfrac_plus = self.eldiag.massfracgsw[self.agns_plus]
        self.fibsfr_plus = np.copy(self.sfr_plus + np.log10(self.massfrac_plus))
        self.sfr_match_plus = np.copy(self.eldiag.sfr_plus[self.sfs_plus])
        self.mass_match_plus = np.copy(self.eldiag.mass_plus[self.sfs_plus])
        self.ssfr_match_plus = self.sfr_match_plus-self.mass_match_plus
        self.fibmass_match_plus = np.copy(self.eldiag.fibmass_plus[self.sfs_plus])

        self.halpfibsfr_match_plus = np.copy(self.eldiag.halpfibsfr_plus[self.sfs_plus])
        self.halpfibssfr_match_plus = np.copy(self.halpfibsfr_match_plus-self.fibmass_match_plus)
        
        self.z_match_plus = np.copy(self.eldiag.z_plus[self.sfs_plus])
        self.massfrac_match_plus = np.copy(self.eldiag.massfracgsw_plus[self.sfs_plus])
        self.fibsfr_match_plus = np.copy(self.sfr_match_plus + np.log10(self.massfrac_match_plus))
        self.fibsfr_mpa_plus = np.copy(self.eldiag.fibsfr_mpa[self.agns_plus])
        self.fibssfr_mpa_plus = np.copy(self.eldiag.fibssfr_mpa[self.agns_plus])
        self.fibsfr_mpa_match_plus = np.copy(self.eldiag.fibsfr_mpa_plus[self.sfs_plus])
        self.fibssfr_mpa_match_plus = np.copy(self.eldiag.fibssfr_mpa_plus[self.sfs_plus])

        self.oiiilum_sub_plus = getlumfromflux(self.oiiiflux_sub_plus, self.z_plus)        
        self.oiiilum_sub_dered_plus = getlumfromflux(self.oiiiflux_sub_dered_plus, self.z_plus)
        self.oiiilum_plus = getlumfromflux(self.oiiiflux_plus, self.z_plus)
        self.oiiilum_dered_plus = getlumfromflux(self.oiiiflux_dered_plus, self.z_plus)
        self.oilum_sub_plus = getlumfromflux(self.oiflux_sub_plus, self.z_plus)        
        self.oilum_sub_dered_plus = getlumfromflux(self.oiflux_sub_dered_plus, self.z_plus)
        self.oilum_plus = getlumfromflux(self.oiflux_plus, self.z_plus)
        self.oilum_dered_plus = getlumfromflux(self.oiflux_dered_plus, self.z_plus)

        self.vdisp_plus = np.copy(self.eldiag.vdisp[self.agns_plus])
                
        
        self.log_oiii_hbeta_sub_plus = np.copy(np.log10(self.oiiiflux_sub_plus/self.hbetaflux_sub_plus))
        self.log_nii_halpha_sub_plus = np.copy(np.log10(self.niiflux_sub_plus/self.halpflux_sub_plus))
        self.log_sii_halpha_sub_plus = np.copy(np.log10(self.siiflux_sub_plus/self.halpflux_sub_plus))
        self.log_oi_halpha_sub_plus = np.copy(np.log10(self.oiflux_sub_plus/self.halpflux_sub_plus))
        
        self.log_oiii_hbeta_plus = np.copy(np.log10(self.oiiiflux_plus/self.hbetaflux_plus))
        self.log_nii_halpha_plus = np.copy(np.log10(self.niiflux_plus/self.halpflux_plus))
        self.log_sii_halpha_plus = np.copy(np.log10(self.siiflux_plus/self.halpflux_plus))
        self.log_oi_halpha_plus = np.copy(np.log10(self.oiflux_plus/self.halpflux_plus))

        self.offset_oiii_hbeta_plus = np.copy(self.log_oiii_hbeta_sub_plus - self.log_oiii_hbeta_plus)
        self.offset_nii_halpha_plus = np.copy(self.log_nii_halpha_sub_plus - self.log_nii_halpha_plus )
        


        #computing values for neither
        self.bptdistrat_neither = (np.array(cosmo.luminosity_distance(self.eldiag.z_neither[self.neither_matches]))/np.array(cosmo.luminosity_distance(self.eldiag.z[self.neither_agn])))**2 #Mpc to cm
        self.sdssids_neither = self.eldiag.ids[self.neither_agn]

        self.oiiiflux_sub_neither = np.copy( self.eldiag.oiiiflux[self.neither_agn] - self.eldiag.oiiiflux_neither[self.neither_matches]*self.bptdistrat_neither)
        self.oiiiflux_sfratio_neither = np.copy(self.eldiag.oiiiflux_neither[self.neither_matches]*self.bptdistrat_neither / self.eldiag.oiiiflux[self.neither_agn])

        self.niiflux_sub_neither = np.copy(self.eldiag.niiflux[self.neither_agn]- self.eldiag.niiflux_neither[self.neither_matches]*self.bptdistrat_neither)
        self.niiflux_sfratio_neither = np.copy(self.eldiag.niiflux_neither[self.neither_matches]*self.bptdistrat_neither / self.eldiag.niiflux[self.neither_agn])

        self.hbetaflux_sub_neither = np.copy(self.eldiag.hbetaflux[self.neither_agn]- self.eldiag.hbetaflux_neither[self.neither_matches]*self.bptdistrat_neither)
        self.hbetaflux_sfratio_neither = np.copy(self.eldiag.hbetaflux_neither[self.neither_matches]*self.bptdistrat_neither / self.eldiag.hbetaflux[self.neither_agn])

        self.halpflux_sub_neither = np.copy(self.eldiag.halpflux[self.neither_agn]- self.eldiag.halpflux_neither[self.neither_matches]*self.bptdistrat_neither)
        self.halpflux_sfratio_neither = np.copy(self.eldiag.halpflux_neither[self.neither_matches]*self.bptdistrat_neither / self.eldiag.halpflux[self.neither_agn])

        self.siiflux_sub_neither  = np.copy(self.eldiag.siiflux[self.neither_agn] - self.eldiag.siiflux[self.neither_matches]*self.bptdistrat_neither)
        self.siiflux_sfratio_neither = np.copy(self.eldiag.siiflux_neither[self.neither_matches]*self.bptdistrat_neither / self.eldiag.siiflux[self.neither_agn])

        self.oiiflux_sub_neither  = np.copy(self.eldiag.oiiflux[self.neither_agn] - self.eldiag.oiiflux[self.neither_matches]*self.bptdistrat_neither)
        self.oiiflux_sfratio_neither = np.copy(self.eldiag.oiiflux_neither[self.neither_matches]*self.bptdistrat_neither / self.eldiag.oiiflux[self.neither_agn])

        self.oiflux_sub_neither  = np.copy(self.eldiag.oiflux[self.neither_agn] - self.eldiag.oiflux[self.neither_matches]*self.bptdistrat_neither)
        self.oiflux_sfratio_neither = np.copy(self.eldiag.oiflux_neither[self.neither_matches]*self.bptdistrat_neither / self.eldiag.oiflux[self.neither_agn])


        self.oiiiflux_neither = np.copy(self.eldiag.oiiiflux[self.neither_agn] )
        self.niiflux_neither = np.copy(self.eldiag.niiflux[self.neither_agn])
        self.hbetaflux_neither = np.copy(self.eldiag.hbetaflux[self.neither_agn])
        self.halpflux_neither = np.copy(self.eldiag.halpflux[self.neither_agn])
        self.siiflux_neither  = np.copy(self.eldiag.siiflux[self.neither_agn])
        self.oiiflux_neither  = np.copy(self.eldiag.oiiflux[self.neither_agn])
        self.oiflux_neither  = np.copy(self.eldiag.oiflux[self.neither_agn])

        self.halpflux_err_neither = np.copy(self.eldiag.halp_err_bpt[self.neither_agn])
        self.hbetaflux_err_neither = np.copy(self.eldiag.hbeta_err_bpt[self.neither_agn])
        self.oiiiflux_err_neither = np.copy(self.eldiag.oiii_err_bpt[self.neither_agn])
        self.oiiflux_err_neither = np.copy(self.eldiag.oii_err_bpt[self.neither_agn])
        self.oiflux_err_neither = np.copy(self.eldiag.oi_err_bpt[self.neither_agn])
        self.niiflux_err_neither = np.copy(self.eldiag.nii_err_bpt[self.neither_agn])
        self.siiflux_err_neither = np.copy(self.eldiag.sii_err_bpt[self.neither_agn])

        self.halphbeta_ratio_neither  = np.copy(self.halpflux_neither/self.hbetaflux_neither)
        self.halphbeta_ratio_err_neither = np.copy(self.halphbeta_ratio_neither*np.sqrt((self.halpflux_err_neither/(self.halpflux_neither*1e17))**2 + 
                                                                                (self.hbetaflux_err_neither/(self.hbetaflux_neither*1e17))**2))
        self.hbetaflux_match_neither = np.copy(self.eldiag.hbetaflux_neither[self.neither_matches])
        self.halpflux_match_neither = np.copy(self.eldiag.halpflux_neither[self.neither_matches])
        self.halpflux_err_match_neither = np.copy(self.eldiag.halp_err_neither[self.neither_matches])
        self.hbetaflux_err_match_neither = np.copy(self.eldiag.hbeta_err_neither[self.neither_matches])
        self.halphbeta_ratio_match_neither  = np.copy(self.halpflux_match_neither/self.hbetaflux_match_neither)
        self.halphbeta_ratio_err_match_neither = np.copy(self.halphbeta_ratio_match_neither*np.sqrt((self.halpflux_err_match_neither/(self.halpflux_match_neither*1e17))**2 + 
                                                                                (self.hbetaflux_err_match_neither/(self.hbetaflux_match_neither*1e17))**2))
        self.halpflux_sn_match_neither = np.copy(self.halpflux_match_neither*1e17/self.halpflux_err_match_neither)
        self.hbetaflux_sn_match_neither = np.copy(self.hbetaflux_match_neither*1e17/self.hbetaflux_err_match_neither)

        self.halpflux_sn_neither = np.copy(self.halpflux_neither*1e17/self.halpflux_err_neither)
        self.hbetaflux_sn_neither = np.copy(self.hbetaflux_neither*1e17/self.hbetaflux_err_neither)
        self.oiiiflux_sn_neither = np.copy(self.oiiiflux_neither*1e17/self.oiiiflux_err_neither)
        self.oiiflux_sn_neither = np.copy(self.oiiflux_neither*1e17/self.oiiflux_err_neither)
        self.oiflux_sn_neither = np.copy(self.oiflux_neither*1e17/self.oiflux_err_neither)
        self.siiflux_sn_neither = np.copy(self.siiflux_neither*1e17/self.siiflux_err_neither)
        self.niiflux_sn_neither = np.copy(self.niiflux_neither*1e17/self.niiflux_err_neither)
        self.halphbeta_sn_neither = np.copy(self.halphbeta_ratio_neither/self.halphbeta_ratio_err_neither)
        self.halphbeta_sn_match_neither = np.copy(self.halphbeta_ratio_match_neither/self.halphbeta_ratio_err_match_neither)
 

        self.oiiiflux_sub_err_neither = np.copy(np.sqrt(self.eldiag.oiii_err_bpt[self.neither_agn]**2 +(self.eldiag.oiii_err_neither[self.neither_matches]*self.bptdistrat_neither)**2))
        self.oiiflux_sub_err_neither = np.copy(np.sqrt(self.eldiag.oii_err_bpt[self.neither_agn]**2 +(self.eldiag.oii_err_neither[self.neither_matches]*self.bptdistrat_neither)**2))
        self.oiflux_sub_err_neither = np.copy(np.sqrt(self.eldiag.oi_err_bpt[self.neither_agn]**2 +(self.eldiag.oi_err_neither[self.neither_matches]*self.bptdistrat_neither)**2))
        self.siiflux_sub_err_neither = np.copy(np.sqrt(self.eldiag.sii_err_bpt[self.neither_agn]**2 +(self.eldiag.sii_err_neither[self.neither_matches]*self.bptdistrat_neither)**2))
        self.niiflux_sub_err_neither = np.copy(np.sqrt(self.eldiag.nii_err_bpt[self.neither_agn]**2+(self.eldiag.nii_err_neither[self.neither_matches]*self.bptdistrat_neither)**2)) 
        self.hbetaflux_sub_err_neither = np.copy(np.sqrt(self.eldiag.hbeta_err_bpt[self.neither_agn]**2+(self.eldiag.hbeta_err_neither[self.neither_matches]*self.bptdistrat_neither)**2))
        self.halpflux_sub_err_neither = np.copy(np.sqrt(self.eldiag.halp_err_bpt[self.neither_agn]**2+(self.eldiag.halp_err_neither[self.neither_matches]*self.bptdistrat_neither)**2))
 
        self.oiiiflux_sub_sn_neither = np.copy(self.oiiiflux_sub_neither*1e17/self.oiiiflux_sub_err_neither)
        self.oiiflux_sub_sn_neither = np.copy(self.oiiflux_sub_neither*1e17/self.oiiflux_sub_err_neither)
        self.oiflux_sub_sn_neither = np.copy(self.oiflux_sub_neither*1e17/self.oiflux_sub_err_neither)

        self.siiflux_sub_sn_neither = np.copy(self.siiflux_sub_neither*1e17/self.siiflux_sub_err_neither)
        self.niiflux_sub_sn_neither = np.copy(self.niiflux_sub_neither*1e17/self.niiflux_sub_err_neither)
        self.hbetaflux_sub_sn_neither = np.copy(self.hbetaflux_sub_neither*1e17/self.hbetaflux_sub_err_neither)
        self.halpflux_sub_sn_neither = np.copy(self.halpflux_sub_neither*1e17/self.halpflux_sub_err_neither)
        self.halphbeta_sub_ratio_neither  = np.copy(self.halpflux_sub_neither/self.hbetaflux_sub_neither)
        self.halphbeta_ratio_sub_err_neither = np.copy(self.halphbeta_sub_ratio_neither*np.sqrt((self.halpflux_sub_err_neither/(self.halpflux_sub_neither*1e17))**2 + 
                                                                (self.hbetaflux_sub_err_neither/(self.hbetaflux_sub_neither*1e17))**2) )
        self.halphbeta_sub_sn_neither = np.copy(self.halphbeta_sub_ratio_neither/self.halphbeta_ratio_sub_err_neither)
        self.halphbeta_sub_sn_filt_neither = np.where(self.halphbeta_sub_sn_neither>halphbeta_sncut)[0]
        self.halphbeta_sub_and_match_sn_filt_neither = np.where((self.halphbeta_sub_sn_neither>halphbeta_sncut)&(self.halphbeta_sn_match_neither > halphbeta_sncut))[0]        
        
        
        

        self.bpt_neither_sn_filt_bool = ((self.oiiiflux_sub_sn_neither > sncut)& (self.niiflux_sub_sn_neither > sncut)& 
                                    (self.hbetaflux_sub_sn_neither > sncut)& (self.halpflux_sub_sn_neither > sncut))
        self.bpt_neither_sn_filt_bool_intermed = ((self.oiiiflux_sub_sn_neither > sncut-1)& (self.niiflux_sub_sn_neither > sncut-1)& 
                                    (self.hbetaflux_sub_sn_neither > sncut-1)& (self.halpflux_sub_sn_neither > sncut-1)&
                                    ((self.oiiiflux_sub_sn_neither < sncut) | (self.niiflux_sub_sn_neither < sncut) | 
                                    (self.hbetaflux_sub_sn_neither < sncut) | (self.halpflux_sub_sn_neither < sncut)))
        self.bpt_neither_sn_filt = np.where(self.bpt_neither_sn_filt_bool)[0]
        self.bpt_neither_sn_filt_intermed = np.where(self.bpt_neither_sn_filt_bool_intermed)[0]

        self.bpt_neither_not_sn_filt_bool = np.logical_not(self.bpt_neither_sn_filt_bool)
        self.bpt_neither_not_sn_filt = np.where(self.bpt_neither_not_sn_filt_bool)[0]
                                      
        self.av_sub_neither_agn = self.eldiag.extinction(ha=self.halpflux_sub_neither, hb=self.hbetaflux_sub_neither, agn=True)
        self.av_sub_neither_sf = self.eldiag.extinction(ha=self.halpflux_sub_neither, hb=self.hbetaflux_sub_neither, agn=False)
        self.av_neither_sf = self.eldiag.av_neither[self.neither_matches]
        self.av_neither_agn = self.eldiag.extinction(ha=self.halpflux_neither, hb=self.hbetaflux_neither, agn=True)
        self.av_gsw_neither_agn = np.copy(self.eldiag.av_gsw[self.neither_agn])
        self.av_gsw_neither_sf = np.copy(self.eldiag.av_gsw_neither[self.neither_matches])
        
        self.filt_by_av_neither = np.where(self.halphbeta_sn_match_neither > halphbeta_sncut)[0]
        
        self.av_neither_sf_bn_edges, self.av_neither_sf_bncenters, self.av_neither_sf_bns, self.av_neither_sf_bn_inds, self.av_neither_sf_valbns = self.bin_quantity(self.av_neither_sf[self.filt_by_av_neither], 0.3, -6, 6, threshold=100)
        self.av_sub_neither_agn_binned  = self.bin_by_ind(self.av_sub_neither_agn, self.av_neither_sf_bn_inds, self.av_neither_sf_bncenters[self.av_neither_sf_valbns])
        self.hbeta_flux_sub_neither_agn_binned = self.bin_by_ind(self.hbetaflux_sub_neither[self.filt_by_av_neither], self.av_neither_sf_bn_inds, self.av_neither_sf_bncenters[self.av_neither_sf_valbns])
        self.halpha_flux_sub_neither_agn_binned = self.bin_by_ind(self.halpflux_sub_neither[self.filt_by_av_neither], self.av_neither_sf_bn_inds, self.av_neither_sf_bncenters[self.av_neither_sf_valbns])
        self.bootstrapped_av_neither_sf_inds = [self.bootstrap(np.array(self.av_neither_sf_bn_inds[i]), 1000, data_only=True) for i in range(len(self.av_neither_sf_bn_inds))]
        self.bootstrapped_halpha_mean_sums_neither = []
        self.bootstrapped_halpha_std_sums_neither = []
        self.bootstrapped_hbeta_mean_sums_neither = []
        self.bootstrapped_hbeta_std_sums_neither = []
        
        for i in range(len(self.bootstrapped_av_neither_sf_inds)): 
            halp_flux_sums_neither = np.sum(self.halpflux_sub_neither[self.filt_by_av_neither][np.int64(self.bootstrapped_av_neither_sf_inds[i])], axis=1)
            mean_halp_flux_sums_neither = np.mean(halp_flux_sums_neither)
            std_halp_flux_sums_neither = np.std(halp_flux_sums_neither)
            self.bootstrapped_halpha_mean_sums_neither.append(mean_halp_flux_sums_neither)
            self.bootstrapped_halpha_std_sums_neither.append(std_halp_flux_sums_neither)
            
            hbeta_flux_sums_neither = np.sum(self.hbetaflux_sub_neither[self.filt_by_av_neither][np.int64(self.bootstrapped_av_neither_sf_inds[i])], axis=1)
            mean_hbeta_flux_sums_neither = np.mean(hbeta_flux_sums_neither)
            std_hbeta_flux_sums_neither = np.std(hbeta_flux_sums_neither)
            self.bootstrapped_hbeta_mean_sums_neither.append(mean_hbeta_flux_sums_neither)
            self.bootstrapped_hbeta_std_sums_neither.append(std_hbeta_flux_sums_neither)
            
        self.bootstrapped_halpha_mean_sums_neither = np.array(self.bootstrapped_halpha_mean_sums_neither)
        self.bootstrapped_halpha_std_sums_neither = np.array(self.bootstrapped_halpha_std_sums_neither)
        self.bootstrapped_hbeta_mean_sums_neither = np.array(self.bootstrapped_hbeta_mean_sums_neither)
        self.bootstrapped_hbeta_std_sums_neither = np.array(self.bootstrapped_hbeta_std_sums_neither)
        self.bootstrapped_halphbeta_neither = self.bootstrapped_halpha_mean_sums_neither/self.bootstrapped_hbeta_mean_sums_neither
        self.bootstrapped_halphbeta_err_neither = self.bootstrapped_halphbeta_neither*np.sqrt( (self.bootstrapped_halpha_std_sums_neither/self.bootstrapped_halpha_mean_sums_neither)**2+
                                                                              (self.bootstrapped_hbeta_std_sums_neither/self.bootstrapped_hbeta_mean_sums_neither)**2)
                        
        self.oiiiflux_sub_dered_neither = self.eldiag.dustcorrect(self.oiiiflux_sub_neither, av=self.av_sub_neither_agn)
        self.oiiiflux_dered_neither = self.eldiag.dustcorrect(self.oiiiflux_neither, av=self.av_neither_agn)
        self.oiiflux_sub_dered_neither = self.eldiag.dustcorrect(self.oiiflux_sub_neither, av=self.av_sub_neither_agn)
        self.oiiflux_dered_neither = self.eldiag.dustcorrect(self.oiiflux_neither, av=self.av_neither_agn)
        self.niiflux_sub_dered_neither = self.eldiag.dustcorrect(self.niiflux_sub_neither, av=self.av_sub_neither_agn)
        self.niiflux_dered_neither = self.eldiag.dustcorrect(self.niiflux_neither, av=self.av_neither_agn)
        self.oiflux_sub_dered_neither = self.eldiag.dustcorrect(self.oiflux_sub_neither, av=self.av_sub_neither_agn)
        self.oiflux_dered_neither = self.eldiag.dustcorrect(self.oiflux_neither, av=self.av_neither_agn)
        
        self.good_oiii_oii_nii_sub_neither = np.where((self.oiiiflux_sub_sn_neither >2) &(self.oiiflux_sub_sn_neither>2)&
                                          (self.niiflux_sub_sn_neither>2))[0]    
        self.good_oiii_oii_nii_neither = np.where((self.oiiiflux_sn_neither >2) &(self.oiiflux_sn_neither>2)&
                                          (self.niiflux_sn_neither>2))[0]    
        self.good_oi_neither = np.where((self.oiflux_sn_neither >2))[0]    
        self.good_oi_sub_neither = np.where((self.oiflux_sub_sn_neither >2))[0]    
        self.good_oiii_hb_sii_ha_sub_neither = np.where((self.oiiiflux_sub_sn_neither >2) &(self.hbetaflux_sub_sn_neither>2)&
                                          (self.siiflux_sub_sn_neither>2)&(self.halpflux_sub_sn_neither>2))[0]    
        self.good_oiii_hb_sii_ha_neither = np.where((self.oiiiflux_sn_neither >2) &(self.hbetaflux_sn_neither>2)&
                                          (self.siiflux_sn_neither>2)&(self.halpflux_sn_neither>2))[0]    
        
        self.z_neither = np.copy(self.eldiag.z[self.neither_agn])
        self.mass_neither = np.copy(self.eldiag.mass[self.neither_agn])
        self.fibmass_neither = np.copy(self.eldiag.fibmass[self.neither_agn])

        self.d4000_neither = np.copy(self.eldiag.d4000[self.neither_agn])        
        self.ssfr_neither = np.copy(self.eldiag.ssfr[self.neither_agn])
        self.sfr_neither = np.copy(self.eldiag.sfr[self.neither_agn])
        self.massfrac_neither = self.eldiag.massfracgsw[self.neither_agn]
        self.fibsfr_neither = np.copy(self.sfr_neither + np.log10(self.massfrac_neither))
        self.sfr_match_neither = np.copy(self.eldiag.sfr_neither[self.neither_matches])
        self.mass_match_neither = np.copy(self.eldiag.mass_neither[self.neither_matches])
        self.ssfr_match_neither = self.sfr_match_neither-self.mass_match_neither
        self.z_match_neither = np.copy(self.eldiag.z_neither[self.neither_matches])
        self.fibmass_match_neither = np.copy(self.eldiag.fibmass_neither[self.neither_matches])
        self.halpfibsfr_match_neither = np.copy(self.eldiag.halpfibsfr_neither[self.neither_matches])
        self.halpfibssfr_match_neither = np.copy(self.halpfibsfr_match_neither-self.fibmass_match_neither)
        self.massfrac_match_neither = np.copy(self.eldiag.massfracgsw_neither[self.neither_matches])
        self.fibsfr_match_neither = np.copy(self.sfr_match_neither + np.log10(self.massfrac_match_neither))
        self.fibsfr_mpa_neither = np.copy(self.eldiag.fibsfr_mpa[self.neither_agn])
        self.fibssfr_mpa_neither = np.copy(self.eldiag.fibssfr_mpa[self.neither_agn])

        self.fibsfr_mpa_match_neither = np.copy(self.eldiag.fibsfr_mpa_neither[self.neither_matches])
        self.fibssfr_mpa_match_neither = np.copy(self.eldiag.fibssfr_mpa_neither[self.neither_matches])

        self.oiiilum_sub_neither = getlumfromflux(self.oiiiflux_sub_neither, self.z_neither)
        self.oiiilum_neither = getlumfromflux(self.oiiiflux_neither, self.z_neither)
        self.oiiilum_dered_neither = getlumfromflux(self.oiiiflux_dered_neither, self.z_neither)      
        self.oiiilum_sub_dered_neither = getlumfromflux(self.oiiiflux_sub_dered_neither, self.z_neither)
        self.oilum_sub_neither = getlumfromflux(self.oiflux_sub_neither, self.z_neither)
        self.oilum_neither = getlumfromflux(self.oiflux_neither, self.z_neither)
        self.oilum_dered_neither = getlumfromflux(self.oiflux_dered_neither, self.z_neither)      
        self.oilum_sub_dered_neither = getlumfromflux(self.oiflux_sub_dered_neither, self.z_neither)

        self.vdisp_neither = np.copy(self.eldiag.vdisp[self.neither_agn])
        
        self.log_oiii_hbeta_sub_neither = np.copy(np.log10(self.oiiiflux_sub_neither/self.hbetaflux_sub_neither))
        self.log_nii_halpha_sub_neither = np.copy(np.log10(self.niiflux_sub_neither/self.halpflux_sub_neither))
        self.log_sii_halpha_sub_neither = np.copy(np.log10(self.siiflux_sub_neither/self.halpflux_sub_neither))
        self.log_oi_halpha_sub_neither = np.copy(np.log10(self.oiflux_sub_neither/self.halpflux_sub_neither))

        self.log_oiii_hbeta_neither = np.copy(np.log10(self.oiiiflux_neither/self.hbetaflux_neither))
        self.log_nii_halpha_neither = np.copy(np.log10(self.niiflux_neither/self.halpflux_neither))
        self.log_sii_halpha_neither = np.copy(np.log10(self.siiflux_neither/self.halpflux_neither))
        self.log_oi_halpha_neither = np.copy(np.log10(self.oiflux_neither/self.halpflux_neither))

        self.offset_oiii_hbeta_neither = np.copy(self.log_oiii_hbeta_sub_neither - self.log_oiii_hbeta_neither) 
        self.offset_nii_halpha_neither = np.copy(self.log_nii_halpha_sub_neither - self.log_nii_halpha_neither ) 
        

        self.log_oiii_hbeta_sing = combine_arrs([self.log_oiii_hbeta, self.log_oiii_hbeta_plus, self.log_oiii_hbeta_neither])
        self.log_nii_halpha_sing = combine_arrs([self.log_nii_halpha, self.log_nii_halpha_plus, self.log_nii_halpha_neither])
        self.log_sii_halpha_sing = combine_arrs([self.log_sii_halpha, self.log_sii_halpha_plus, self.log_sii_halpha_neither])
        self.log_oi_halpha_sing = combine_arrs([self.log_oi_halpha, self.log_oi_halpha_plus, self.log_oi_halpha_neither])

        self.log_oiii_hbeta_sf_match =  np.copy(np.log10(self.eldiag.oiiiflux[self.sfs]/self.eldiag.hbetaflux[self.sfs]))
        self.log_nii_halpha_sf_match =  np.copy(np.log10(self.eldiag.niiflux[self.sfs]/self.eldiag.halpflux[self.sfs]))

        self.log_oiii_hbeta_sf =  np.copy(np.log10(self.eldiag.oiiiflux[self.bptsf_inds]/self.eldiag.hbetaflux[self.bptsf_inds]))
        self.log_nii_halpha_sf =  np.copy(np.log10(self.eldiag.niiflux[self.bptsf_inds]/self.eldiag.halpflux[self.bptsf_inds]))

        self.mass_sf = np.copy(self.eldiag.mass[self.bptsf_inds])
        self.sfr_sf = np.copy(self.eldiag.sfr[self.bptsf_inds])
        self.ssfr_sf = np.copy(self.eldiag.ssfr[self.bptsf_inds])
        

        self.log_oiii_hbeta_agns_sf= combine_arrs([self.log_oiii_hbeta, self.log_oiii_hbeta_plus, self.log_oiii_hbeta_neither, self.log_oiii_hbeta_sf_match])
        self.log_nii_halpha_agns_sf = combine_arrs([self.log_nii_halpha, self.log_nii_halpha_plus, self.log_nii_halpha_neither, self.log_nii_halpha_sf_match])
        
        self.log_oiii_hbeta_full_bpt= combine_arrs([self.log_oiii_hbeta, self.log_oiii_hbeta_plus, self.log_oiii_hbeta_neither, self.log_oiii_hbeta_sf])
        self.log_nii_halpha_full_bpt = combine_arrs([self.log_nii_halpha, self.log_nii_halpha_plus, self.log_nii_halpha_neither, self.log_nii_halpha_sf])
        

        self.log_oiii_hbeta_sub_sing = combine_arrs([self.log_oiii_hbeta_sub, self.log_oiii_hbeta_sub_plus, self.log_oiii_hbeta_sub_neither])
        self.log_nii_halpha_sub_sing = combine_arrs([self.log_nii_halpha_sub, self.log_nii_halpha_sub_plus, self.log_nii_halpha_sub_neither])
        self.log_sii_halpha_sub_sing = combine_arrs([self.log_sii_halpha_sub, self.log_sii_halpha_sub_plus, self.log_sii_halpha_sub_neither])
        self.log_oi_halpha_sub_sing = combine_arrs([self.log_oi_halpha_sub, self.log_oi_halpha_sub_plus, self.log_oi_halpha_sub_neither])

        
        self.log_oiii_hbeta_sub_sing_agns_sf_match = combine_arrs([self.log_oiii_hbeta_sub, self.log_oiii_hbeta_sub_plus, self.log_oiii_hbeta_sub_neither, self.log_oiii_hbeta_sf_match])
        self.log_nii_halpha_sub_sing_agns_sf_match = combine_arrs([self.log_nii_halpha_sub, self.log_nii_halpha_sub_plus, self.log_nii_halpha_sub_neither, self.log_nii_halpha_sf_match])
        
        self.log_oiii_hbeta_sub_sing_agns_sf = combine_arrs([self.log_oiii_hbeta_sub, self.log_oiii_hbeta_sub_plus, self.log_oiii_hbeta_sub_neither, self.log_oiii_hbeta_sf])
        self.log_nii_halpha_sub_sing_agns_sf = combine_arrs([self.log_nii_halpha_sub, self.log_nii_halpha_sub_plus, self.log_nii_halpha_sub_neither, self.log_nii_halpha_sf])
        
        self.oiiilum_sing = combine_arrs([self.oiiilum_bpt, self.oiiilum_plus, self.oiiilum_neither])
        self.oiiilum_dered_sing = combine_arrs([self.oiiilum_dered_bpt, self.oiiilum_dered_plus, self.oiiilum_dered_neither])

        self.oiflux_sing = combine_arrs([self.oiflux, self.oiflux_plus, self.oiflux_neither])
        self.oiflux_sn_sing = combine_arrs([self.oiflux_sn, self.oiflux_sn_plus, self.oiflux_sn_neither])
        self.oiflux_sub_sing = combine_arrs([self.oiflux_sub, self.oiflux_sub_plus, self.oiflux_sub_neither])
        self.oiflux_sub_sn_sing = combine_arrs([self.oiflux_sub_sn, self.oiflux_sub_sn_plus, self.oiflux_sub_sn_neither])
        self.oiflux_sfratio_sing = combine_arrs([self.oiflux_sfratio, self.oiflux_sfratio_plus, self.oiflux_sfratio_neither])

        self.oiiflux_sing = combine_arrs([self.oiiflux, self.oiiflux_plus, self.oiiflux_neither])
        self.oiiflux_sn_sing = combine_arrs([self.oiiflux_sn, self.oiiflux_sn_plus, self.oiiflux_sn_neither])
        self.oiiflux_dered_sing = combine_arrs([self.oiiflux_dered_bpt, self.oiiflux_dered_plus, self.oiiflux_dered_neither])
        self.oiiflux_sub_sing = combine_arrs([self.oiiflux_sub, self.oiiflux_sub_plus, self.oiiflux_sub_neither])
        self.oiiflux_sub_dered_sing = combine_arrs([self.oiiflux_sub_dered_bpt, self.oiiflux_sub_dered_plus, self.oiiflux_sub_dered_neither])
        self.oiiflux_sub_sn_sing = combine_arrs([self.oiiflux_sub_sn, self.oiiflux_sub_sn_plus, self.oiiflux_sub_sn_neither])
        self.oiiflux_sfratio_sing = combine_arrs([self.oiiflux_sfratio, self.oiiflux_sfratio_plus, self.oiiflux_sfratio_neither])

        self.oiiiflux_sing = combine_arrs([self.oiiiflux, self.oiiiflux_plus, self.oiiiflux_neither])
        self.oiiiflux_sn_sing = combine_arrs([self.oiiiflux_sn, self.oiiiflux_sn_plus, self.oiiiflux_sn_neither])
        self.oiiiflux_dered_sing = combine_arrs([self.oiiiflux_dered_bpt, self.oiiiflux_dered_plus, self.oiiiflux_dered_neither])
        self.oiiiflux_sub_sing = combine_arrs([self.oiiiflux_sub, self.oiiiflux_sub_plus, self.oiiiflux_sub_neither])
        self.oiiiflux_sub_dered_sing = combine_arrs([self.oiiiflux_sub_dered_bpt, self.oiiiflux_sub_dered_plus, self.oiiiflux_sub_dered_neither])
        self.oiiiflux_sub_sn_sing = combine_arrs([self.oiiiflux_sub_sn, self.oiiiflux_sub_sn_plus, self.oiiiflux_sub_sn_neither])
        self.oiiiflux_sfratio_sing = combine_arrs([self.oiiiflux_sfratio, self.oiiiflux_sfratio_plus, self.oiiiflux_sfratio_neither])

        
        self.niiflux_sing = combine_arrs([self.niiflux, self.niiflux_plus, self.niiflux_neither])
        self.niiflux_dered_sing = combine_arrs([self.niiflux_dered_bpt, self.niiflux_dered_plus, self.niiflux_dered_neither])
        self.niiflux_sn_sing = combine_arrs([self.niiflux_sn, self.niiflux_sn_plus, self.niiflux_sn_neither])
        self.niiflux_sub_sing = combine_arrs([self.niiflux_sub, self.niiflux_sub_plus, self.niiflux_sub_neither])
        self.niiflux_sub_dered_sing = combine_arrs([self.niiflux_sub_dered_bpt, self.niiflux_sub_dered_plus, self.niiflux_sub_dered_neither])
        self.niiflux_sub_sn_sing = combine_arrs([self.niiflux_sub_sn, self.niiflux_sub_sn_plus, self.niiflux_sub_sn_neither])
        self.niiflux_sfratio_sing = combine_arrs([self.niiflux_sfratio, self.niiflux_sfratio_plus, self.niiflux_sfratio_neither])

        self.siiflux_sing = combine_arrs([self.siiflux, self.siiflux_plus, self.siiflux_neither])
        self.siiflux_sn_sing = combine_arrs([self.siiflux_sn, self.siiflux_sn_plus, self.siiflux_sn_neither])
        self.siiflux_sub_sing = combine_arrs([self.siiflux_sub, self.siiflux_sub_plus, self.siiflux_sub_neither])
        self.siiflux_sub_sn_sing = combine_arrs([self.siiflux_sub_sn, self.siiflux_sub_sn_plus, self.siiflux_sub_sn_neither])
        self.siiflux_sfratio_sing = combine_arrs([self.siiflux_sfratio, self.siiflux_sfratio_plus, self.siiflux_sfratio_neither])

        self.halpflux_sing = combine_arrs([self.halpflux, self.halpflux_plus, self.halpflux_neither])
        self.halpflux_sn_sing = combine_arrs([self.halpflux_sn, self.halpflux_sn_plus, self.halpflux_sn_neither])
        self.halpflux_sub_sing = combine_arrs([self.halpflux_sub, self.halpflux_sub_plus, self.halpflux_sub_neither])
        self.halpflux_sub_sn_sing = combine_arrs([self.halpflux_sub_sn, self.halpflux_sub_sn_plus, self.halpflux_sub_sn_neither])
        self.halpflux_sfratio_sing = combine_arrs([self.halpflux_sfratio, self.halpflux_sfratio_plus, self.halpflux_sfratio_neither])

        self.hbetaflux_sing = combine_arrs([self.hbetaflux, self.hbetaflux_plus, self.hbetaflux_neither])
        self.hbetaflux_sub_sn_sing = combine_arrs([self.hbetaflux_sn, self.hbetaflux_sn_plus, self.hbetaflux_sn_neither])
        self.hbetaflux_sub_sing = combine_arrs([self.hbetaflux_sub, self.hbetaflux_sub_plus, self.hbetaflux_sub_neither])
        self.hbetaflux_sub_sn_sing = combine_arrs([self.hbetaflux_sub_sn, self.hbetaflux_sub_sn_plus, self.hbetaflux_sub_sn_neither])
        self.hbetaflux_sfratio_sing = combine_arrs([self.hbetaflux_sfratio, self.hbetaflux_sfratio_plus, self.hbetaflux_sfratio_neither])

        self.halpflux_sn_match_sing = combine_arrs([self.halpflux_sn_match, self.halpflux_sn_match_plus, self.halpflux_sn_match_neither])
        self.hbetaflux_sn_match_sing = combine_arrs([self.hbetaflux_sn_match, self.hbetaflux_sn_match_plus, self.hbetaflux_sn_match_neither])

        
        self.oiiilum_sub_sing = combine_arrs([self.oiiilum_sub_bpt, self.oiiilum_sub_plus, self.oiiilum_sub_neither])
        self.oiiilum_sub_dered_sing = combine_arrs([self.oiiilum_sub_dered_bpt, self.oiiilum_sub_dered_plus, self.oiiilum_sub_dered_neither])
        self.vdisp_sing = combine_arrs([self.vdisp_bpt, self.vdisp_plus, self.vdisp_neither])
        self.fibsfr_sing = combine_arrs([self.fibsfr_bpt, self.fibsfr_plus, self.fibsfr_neither])
        self.fibsfr_match_sing = combine_arrs([self.fibsfr_match_bpt, self.fibsfr_match_plus, self.fibsfr_match_neither])                
        self.fibsfr_mpa_sing = combine_arrs([self.fibsfr_mpa_bpt, self.fibsfr_mpa_plus, self.fibsfr_mpa_neither])

        self.fibsfr_mpa_match_sing = combine_arrs([self.fibsfr_match_bpt, self.fibsfr_match_plus, self.fibsfr_match_neither])                
        self.fibssfr_mpa_sing = combine_arrs([self.fibssfr_mpa_bpt, self.fibssfr_mpa_plus, self.fibssfr_mpa_neither])
        self.fibssfr_mpa_match_sing = combine_arrs([self.fibssfr_mpa_match_bpt, self.fibssfr_mpa_match_plus, self.fibssfr_mpa_match_neither])                

        self.halpfibsfr_match_sing = combine_arrs([self.halpfibsfr_match_bpt, self.halpfibsfr_match_plus, self.halpfibsfr_match_neither])
        self.halpfibssfr_match_sing = combine_arrs([self.halpfibssfr_match_bpt, self.halpfibssfr_match_plus, self.halpfibssfr_match_neither])

        self.sfr_sing = combine_arrs([self.sfr_bpt, self.sfr_plus, self.sfr_neither])
        self.mass_sing = combine_arrs([self.mass_bpt, self.mass_plus, self.mass_neither])
        self.mass_match_sing = combine_arrs([self.mass_match_bpt, self.mass_match_plus, self.mass_match_neither])
        self.fibmass_sing = combine_arrs([self.fibmass_bpt, self.fibmass_plus, self.fibmass_neither])
        self.fibmass_match_sing = combine_arrs([self.fibmass_match_bpt, self.fibmass_match_plus, self.fibmass_match_neither])

        self.sdssids_sing = combine_arrs([self.sdssids_bpt, self.sdssids_plus, self.sdssids_neither])
        self.sfr_match_sing = combine_arrs([self.sfr_match_bpt, self.sfr_match_plus, self.sfr_match_neither])
        self.av_gsw_sing_sf = combine_arrs([self.av_gsw_bpt_sf, self.av_gsw_plus_sf, self.av_gsw_neither_sf])
        self.av_gsw_sing_agn = combine_arrs([self.av_gsw_bpt_agn, self.av_gsw_plus_agn, self.av_gsw_neither_agn])
        self.av_sub_sing = combine_arrs([self.av_sub_bpt_agn, self.av_sub_plus_agn, self.av_sub_neither_agn])

        self.z_sing= combine_arrs([self.z_bpt, self.z_plus, self.z_neither])
        self.z_match_sing= combine_arrs([self.z_match_bpt, self.z_match_plus, self.z_match_neither])
        
        self.ssfr_sing = combine_arrs([self.ssfr_bpt, self.ssfr_plus, self.ssfr_neither])
        self.ssfr_match_sing = combine_arrs([self.ssfr_match_bpt, self.ssfr_match_plus, self.ssfr_match_neither])

        self.delta_ssfr_bpt = self.get_deltassfr(self.mass_bpt, self.ssfr_bpt)
        self.delta_ssfr_plus = self.get_deltassfr(self.mass_plus, self.ssfr_plus)
        self.delta_ssfr_neither = self.get_deltassfr(self.mass_neither, self.ssfr_neither)        
        self.delta_ssfr_sing = self.get_deltassfr(self.mass_sing, self.ssfr_sing)
        
        self.delta_ssfr_match_sing = self.get_deltassfr(self.mass_match_sing, self.ssfr_match_sing)
        self.d4000_sing = combine_arrs([self.d4000_bpt, self.d4000_plus, self.d4000_neither])

        self.logoh_sub_sing = self.nii_oii_to_oh(self.niiflux_sub_dered_sing, self.oiiflux_sub_dered_sing)
        self.logoh_sing = self.nii_oii_to_oh(self.niiflux_dered_sing, self.oiiflux_dered_sing)

        self.offset_oiii_hbeta_sing = combine_arrs([self.offset_oiii_hbeta, self.offset_oiii_hbeta_plus, self.offset_oiii_hbeta_neither])
        self.offset_nii_halpha_sing = combine_arrs([self.offset_nii_halpha, self.offset_nii_halpha_plus, self.offset_nii_halpha_neither])
        
        '''
        Everything below are filters
        '''
        minmass = 10.2
        maxmass = 10.4
        d4000_cut = 1.6
        self.high_ssfr_obj = np.where(self.delta_ssfr_sing>-0.7)[0]
        self.low_ssfr_obj = np.where(self.delta_ssfr_sing<=-0.7)[0]
        self.old_d4000 = np.where(self.d4000_sing>d4000_cut)[0]
        self.young_d4000 = np.where(self.d4000_sing<=d4000_cut)[0]

        ''' 
        mass filt and ssfr
        '''
        self.mass_filt = np.where((self.mass_sing>minmass) &
                                  (self.mass_sing<maxmass))[0]                
        self.high_ssfr_mass_filt_obj = np.where((self.delta_ssfr_sing>-0.7) &
                                                (self.mass_sing>minmass) &
                                                (self.mass_sing<maxmass))[0]
        self.low_ssfr_mass_filt_obj = np.where((self.delta_ssfr_sing<=-0.7) &
                                               (self.mass_sing>minmass) &
                                               (self.mass_sing<maxmass))[0]

        '''
        liner filt and ssfr
        '''
        self.liner_sub_filt = np.where( (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                   (self.siiflux_sub_sn_sing>2))[0]
        self.high_ssfr_liner_sub_filt_obj = np.where((self.delta_ssfr_sing>-0.7) &
                                                (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                (self.siiflux_sub_sn_sing>2))[0]
        self.low_ssfr_liner_sub_filt_obj = np.where((self.delta_ssfr_sing<=-0.7) &
                                               (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                               (self.siiflux_sub_sn_sing>2))[0]

        '''
        seyfert filt and ssfr
        '''
        self.sy2_sub_filt =  np.where( (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                   (self.siiflux_sub_sn_sing>2))[0]
        self.high_ssfr_sy2_sub_filt_obj = np.where((self.delta_ssfr_sing>-0.7) &
                                                (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                (self.siiflux_sub_sn_sing>2))[0]
        self.low_ssfr_sy2_sub_filt_obj = np.where((self.delta_ssfr_sing<=-0.7) &
                                               (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                               (self.siiflux_sub_sn_sing>2))[0]        
        '''
        high ssfr post sub
        '''
        
        self.high_ssfr_sii_sub_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                               (self.siiflux_sub_sn_sing>2))[0]
        self.high_ssfr_ha_match_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                (self.halpflux_sn_match_sing>2) &
                                                (self.hbetaflux_sn_match_sing>2))[0]
        self.high_ssfr_oi_sub_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                              (self.oiflux_sub_sn_sing>2))[0]
        self.high_ssfr_oii_sub_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                               (self.oiiflux_sub_sn_sing>2))[0]
        self.high_ssfr_oii_sii_sub_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                   (self.siiflux_sub_sn_sing>2) &
                                                   (self.oiiflux_sub_sn_sing>2))[0]
        self.high_ssfr_oi_sii_sub_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                  (self.siiflux_sub_sn_sing>2) &
                                                  (self.oiflux_sub_sn_sing>2))[0]
        ''' 
        high ssfr post sub mass filt
        '''
        self.high_ssfr_sii_sub_mass_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                         (self.siiflux_sub_sn_sing>2) &
                                                         (self.mass_sing>minmass) &
                                                         (self.mass_sing<maxmass))[0]
        self.high_ssfr_ha_match_mass_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                          (self.halpflux_sn_match_sing>2)&
                                                          (self.hbetaflux_sn_match_sing>2) &
                                                          (self.mass_sing>minmass) &
                                                          (self.mass_sing<maxmass))[0]
        self.high_ssfr_oi_sub_mass_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                        (self.oiflux_sub_sn_sing>2) &
                                                        (self.mass_sing>minmass) &
                                                        (self.mass_sing<maxmass))[0]
        self.high_ssfr_oii_sub_mass_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                         (self.oiiflux_sub_sn_sing>2) &
                                                         (self.mass_sing>minmass) &
                                                         (self.mass_sing<maxmass))[0]
        self.high_ssfr_oii_sii_sub_mass_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                             (self.siiflux_sub_sn_sing>2) &
                                                             (self.oiiflux_sub_sn_sing>2) &
                                                             (self.mass_sing>minmass) &
                                                             (self.mass_sing<maxmass))[0]
        self.high_ssfr_oi_sii_sub_mass_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                            (self.siiflux_sub_sn_sing>2) &
                                                            (self.oiflux_sub_sn_sing>2) &
                                                            (self.mass_sing>minmass) &
                                                            (self.mass_sing<maxmass))[0]

        '''
        high ssfr post sub liner
        '''
        
        self.high_ssfr_sii_sub_liner_sub_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                               (self.siiflux_sub_sn_sing>2) &
                                               (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) 
                                                )[0]
        self.high_ssfr_ha_match_liner_sub_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                (self.halpflux_sn_match_sing>2) &
                                                (self.hbetaflux_sn_match_sing>2) &
                                                (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                (self.siiflux_sub_sn_sing>2))[0]
        self.high_ssfr_oi_sub_liner_sub_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                              (self.oiflux_sub_sn_sing>2) &
                                              (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                              (self.siiflux_sub_sn_sing>2))[0]
        self.high_ssfr_oii_sub_liner_sub_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                               (self.oiiflux_sub_sn_sing>2) &
                                               (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                               (self.siiflux_sub_sn_sing>2))[0]
        self.high_ssfr_oii_sii_sub_liner_sub_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                   (self.oiiflux_sub_sn_sing>2) &
                                                   (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                   (self.siiflux_sub_sn_sing>2))[0]
        self.high_ssfr_oi_sii_sub_liner_sub_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                  (self.oiflux_sub_sn_sing>2) &
                                                (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                (self.siiflux_sub_sn_sing>2))[0]

        '''
        post sub liner
        '''
        
        self.sii_sub_liner_sub_filt_sing = np.where((self.siiflux_sub_sn_sing>2) &
                                                    (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) 
                                                )[0]
        self.ha_match_liner_sub_filt_sing = np.where((self.halpflux_sn_match_sing>2) &
                                                     (self.hbetaflux_sn_match_sing>2) &
                                                     (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                     (self.siiflux_sub_sn_sing>2))[0]
        self.oi_sub_liner_sub_filt_sing = np.where((self.oiflux_sub_sn_sing>2) &
                                                   (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                   (self.siiflux_sub_sn_sing>2))[0]
        self.oii_sub_liner_sub_filt_sing = np.where((self.oiiflux_sub_sn_sing>2) &
                                                    (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                    (self.siiflux_sub_sn_sing>2))[0]
        self.oii_sii_sub_liner_sub_filt_sing = np.where((self.oiiflux_sub_sn_sing>2) &
                                                        (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                        (self.siiflux_sub_sn_sing>2))[0]
        self.oi_sii_sub_liner_sub_filt_sing = np.where((self.oiflux_sub_sn_sing>2) &
                                                       (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                       (self.siiflux_sub_sn_sing>2))[0]

        '''
        high ssfr post sub seyfert
        '''
        
        self.high_ssfr_sii_sub_sy2_sub_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                               (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                               (self.siiflux_sub_sn_sing>2))[0]
        self.high_ssfr_ha_match_sy2_sub_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                (self.halpflux_sn_match_sing>2) &
                                                (self.hbetaflux_sn_match_sing>2) &
                                                (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                (self.siiflux_sub_sn_sing>2))[0]
        self.high_ssfr_oi_sub_sy2_sub_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                              (self.oiflux_sub_sn_sing>2) &
                                              (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                              (self.siiflux_sub_sn_sing>2))[0]
        self.high_ssfr_oii_sub_sy2_sub_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                               (self.oiiflux_sub_sn_sing>2) &
                                               (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                               (self.siiflux_sub_sn_sing>2))[0]
        self.high_ssfr_oii_sii_sub_sy2_sub_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                   (self.oiiflux_sub_sn_sing>2)&
                                                   (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                   (self.siiflux_sub_sn_sing>2))[0]
        self.high_ssfr_oi_sii_sub_sy2_sub_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                  (self.oiflux_sub_sn_sing>2) &
                                                  (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                  (self.siiflux_sub_sn_sing>2))[0]       
        '''
        sy2 postsub
        '''
        
        self.sii_sub_sy2_sub_filt_sing = np.where((self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                               (self.siiflux_sub_sn_sing>2))[0]
        self.ha_match_sy2_sub_filt_sing = np.where((self.halpflux_sn_match_sing>2) &
                                                (self.hbetaflux_sn_match_sing>2) &
                                                (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                (self.siiflux_sub_sn_sing>2))[0]
        self.oi_sub_sy2_sub_filt_sing = np.where((self.oiflux_sub_sn_sing>2) &
                                              (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                              (self.siiflux_sub_sn_sing>2))[0]
        self.oii_sub_sy2_sub_filt_sing = np.where((self.oiiflux_sub_sn_sing>2) &
                                               (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                               (self.siiflux_sub_sn_sing>2))[0]
        self.oii_sii_sub_sy2_sub_filt_sing = np.where((self.oiiflux_sub_sn_sing>2)&
                                                                (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                                (self.siiflux_sub_sn_sing>2))[0]
        self.oi_sii_sub_sy2_sub_filt_sing = np.where((self.oiflux_sub_sn_sing>2) &
                                                     (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                     (self.siiflux_sub_sn_sing>2))[0]       

        '''
        low ssfr group post sub
        '''
        self.low_ssfr_ha_match_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                               (self.halpflux_sn_match_sing>2) &
                                               (self.hbetaflux_sn_match_sing>2))[0]
        self.low_ssfr_sii_sub_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                              (self.siiflux_sub_sn_sing>2))[0]
        self.low_ssfr_oi_sub_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                             (self.oiflux_sub_sn_sing>2))[0]        
        self.low_ssfr_oii_sub_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                              (self.oiiflux_sub_sn_sing>2))[0]
        self.low_ssfr_oii_sii_sub_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                  (self.oiiflux_sub_sn_sing>2) &
                                                (self.siiflux_sub_sn_sing>2))[0]
        self.low_ssfr_oi_sii_sub_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                 (self.oiflux_sub_sn_sing>2)&
                                                (self.siiflux_sub_sn_sing>2))[0]
        '''
        low ssfr post sub mass filt
        '''
        self.low_ssfr_ha_match_mass_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                         (self.halpflux_sn_match_sing>2) &
                                                         (self.hbetaflux_sn_match_sing>2) &
                                                         (self.mass_sing>minmass) &
                                                         (self.mass_sing<maxmass))[0]
        self.low_ssfr_sii_sub_mass_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                        (self.siiflux_sub_sn_sing>2) &
                                                        (self.mass_sing>minmass) &
                                                        (self.mass_sing<maxmass))[0]
        self.low_ssfr_oi_sub_mass_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                       (self.oiflux_sub_sn_sing>2) &
                                                       (self.mass_sing>minmass) &
                                                       (self.mass_sing<maxmass))[0]    
        self.low_ssfr_oii_sub_mass_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                        (self.oiiflux_sub_sn_sing>2) &
                                                        (self.mass_sing>minmass) &
                                                        (self.mass_sing<maxmass))[0]
        self.low_ssfr_oii_sii_sub_mass_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                            (self.siiflux_sub_sn_sing>2) &
                                                            (self.oiiflux_sub_sn_sing>2) &
                                                            (self.mass_sing>minmass) &
                                                            (self.mass_sing<maxmass))[0]
        self.low_ssfr_oi_sii_sub_mass_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                           (self.siiflux_sub_sn_sing>2) &
                                                           (self.oiflux_sub_sn_sing>2) &
                                                           (self.mass_sing>minmass) &
                                                           (self.mass_sing<maxmass))[0]
        '''
        low ssfr group post sub liner
        '''


        self.low_ssfr_ha_match_liner_sub_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                               (self.halpflux_sn_match_sing>2) &
                                               (self.hbetaflux_sn_match_sing>2) &
                                                (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                (self.siiflux_sub_sn_sing>2))[0]
        self.low_ssfr_sii_sub_liner_sub_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                (self.siiflux_sub_sn_sing>2))[0]
        self.low_ssfr_oi_sub_liner_sub_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                             (self.oiflux_sub_sn_sing>2) &
                                                (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                (self.siiflux_sub_sn_sing>2))[0]        
        self.low_ssfr_oii_sub_liner_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                              (self.oiiflux_sub_sn_sing>2) &
                                                (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                (self.siiflux_sub_sn_sing>2))[0]
        self.low_ssfr_oii_sii_sub_liner_sub_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                  (self.oiiflux_sub_sn_sing>2) &
                                                (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                (self.siiflux_sub_sn_sing>2))[0]
        self.low_ssfr_oi_sii_sub_liner_sub_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                 (self.oiflux_sub_sn_sing>2) &
                                                (self.log_oiii_hbeta_sub_sing < np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                (self.siiflux_sub_sn_sing>2))[0]

        '''
        low ssfr group post sub sy2
        '''
        self.low_ssfr_ha_match_sy2_sub_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                        (self.halpflux_sn_match_sing>2) &
                                                        (self.hbetaflux_sn_match_sing>2)&
                                                        (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                        (self.siiflux_sub_sn_sing>2))
        self.low_ssfr_sii_sub_sy2_sub_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                       (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                       (self.siiflux_sub_sn_sing>2))
        self.low_ssfr_oi_sub_sy2_sub_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                      (self.oiflux_sub_sn_sing>2)&
                                                      (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                      (self.siiflux_sub_sn_sing>2))        
        self.low_ssfr_oii_sub_sy2_sub_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                       (self.oiiflux_sub_sn_sing>2)&
                                                       (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                       (self.siiflux_sub_sn_sing>2))
        self.low_ssfr_oii_sii_sub_sy2_sub_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                           (self.oiiflux_sub_sn_sing>2)&
                                                           (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                           (self.siiflux_sub_sn_sing>2))
        self.low_ssfr_oi_sii_sub_sy2_sub_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                          (self.oiflux_sub_sn_sing>2) &
                                                          (self.log_oiii_hbeta_sub_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sub_sing)) ) &
                                                          (self.siiflux_sub_sn_sing>2))

        
        '''
        high ssfr group presub
        '''
        self.high_ssfr_sii_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                           (self.siiflux_sn_sing>2))
        self.high_ssfr_oii_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                           (self.oiiflux_sn_sing>2))
        self.high_ssfr_oii_sii_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                               (self.siiflux_sn_sing>2) &
                                               (self.oiiflux_sn_sing>2))
        self.high_ssfr_oi_sii_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                              (self.siiflux_sn_sing>2) &
                                              (self.oiflux_sn_sing>2))

        '''
        high ssfr group presub mass restricted
        '''
        self.high_ssfr_sii_mass_filt_sing = np.where((self.delta_ssfr_sing>-0.7) & 
                                                     (self.siiflux_sn_sing>2) &
                                                     (self.mass_sing>minmass) &
                                                     (self.mass_sing<maxmass))
        self.high_ssfr_oii_mass_filt_sing = np.where((self.delta_ssfr_sing>-0.7) & 
                                                     (self.oiiflux_sn_sing>2) &
                                                     (self.mass_sing>minmass) &
                                                     (self.mass_sing<maxmass))
        self.high_ssfr_oii_sii_mass_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                         (self.siiflux_sn_sing>2) &
                                                         (self.oiiflux_sn_sing>2) &
                                                         (self.mass_sing>minmass) &
                                                         (self.mass_sing<maxmass))
        self.high_ssfr_oi_sii_mass_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                        (self.siiflux_sn_sing>2) &
                                                        (self.oiflux_sn_sing>2)&
                                                        (self.mass_sing>minmass) &
                                                        (self.mass_sing<maxmass))


        '''
        high ssfr group presub liner
        '''
        self.high_ssfr_sii_liner_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                      (self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                      (self.siiflux_sn_sing>2))
        self.high_ssfr_oii_liner_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                      (self.oiiflux_sn_sing>2)&
                                                      (self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                      (self.siiflux_sn_sing>2))
        self.high_ssfr_oii_sii_liner_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                          (self.oiiflux_sn_sing>2)&
                                                          (self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                          (self.siiflux_sn_sing>2))
        self.high_ssfr_oi_sii_liner_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                         (self.oiflux_sn_sing>2)&
                                                         (self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                         (self.siiflux_sn_sing>2))
        '''
        presub liner
        '''
        self.sii_liner_filt_sing = np.where((self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                            (self.siiflux_sn_sing>2))
        self.oii_liner_filt_sing = np.where((self.oiiflux_sn_sing>2)&
                                            (self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                            (self.siiflux_sn_sing>2))
        self.oii_sii_liner_filt_sing = np.where((self.oiiflux_sn_sing>2)&
                                                (self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                (self.siiflux_sn_sing>2))
        self.oi_sii_liner_filt_sing = np.where((self.oiflux_sn_sing>2)&
                                               (self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                               (self.siiflux_sn_sing>2))
        '''
        high ssfr group presub sy2
        '''
        self.high_ssfr_sii_sy2_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                    (self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                    (self.siiflux_sn_sing>2))[0]
        self.high_ssfr_oii_sy2_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                    (self.oiiflux_sn_sing>2) &
                                                    (self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                    (self.siiflux_sn_sing>2))[0]
        self.high_ssfr_oii_sii_sy2_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                        (self.oiiflux_sn_sing>2) &
                                                        (self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                        (self.siiflux_sn_sing>2))[0]
        self.high_ssfr_oi_sii_sy2_filt_sing = np.where((self.delta_ssfr_sing>-0.7) &
                                                       (self.oiflux_sn_sing>2) &
                                                       (self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                       (self.siiflux_sn_sing>2))[0]

        '''
        high ssfr group presub sy2
        '''
        self.sii_sy2_filt_sing = np.where((self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                          (self.siiflux_sn_sing>2))[0]
        self.oii_sy2_filt_sing = np.where((self.oiiflux_sn_sing>2) &
                                          (self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                          (self.siiflux_sn_sing>2))[0]
        self.oii_sii_sy2_filt_sing = np.where((self.oiiflux_sn_sing>2) &
                                              (self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                              (self.siiflux_sn_sing>2))[0]
        self.oi_sii_sy2_filt_sing = np.where((self.oiflux_sn_sing>2) &
                                             (self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                             (self.siiflux_sn_sing>2))[0]


        '''
        low ssfr group presub
        '''
        self.low_ssfr_sii_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                          (self.siiflux_sn_sing>2))[0]
        self.low_ssfr_oi_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                         (self.oiflux_sn_sing>2))[0]
        self.low_ssfr_oii_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                          (self.oiiflux_sn_sing>2))[0]
        self.low_ssfr_oii_sii_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                              (self.siiflux_sn_sing>2) &
                                              (self.oiiflux_sn_sing>2))[0]
        self.low_ssfr_oi_sii_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                             (self.siiflux_sn_sing>2) &
                                             (self.oiflux_sn_sing>2))[0]

        ''' 
        low ssfr group mass filt presub
        '''
        self.low_ssfr_sii_mass_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                    (self.siiflux_sn_sing>2) &
                                                    (self.mass_sing>minmass) &
                                                    (self.mass_sing<maxmass))[0]
        self.low_ssfr_oi_mass_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                   (self.oiflux_sn_sing>2) &
                                                   (self.mass_sing>minmass) &
                                                   (self.mass_sing<maxmass))[0]
        self.low_ssfr_oii_mass_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                    (self.oiiflux_sn_sing>2) &
                                                    (self.mass_sing>minmass) &
                                                    (self.mass_sing<maxmass))[0]
        self.low_ssfr_oii_sii_mass_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                        (self.siiflux_sn_sing>2) &
                                                        (self.oiiflux_sn_sing>2) &
                                                        (self.mass_sing>minmass) &
                                                        (self.mass_sing<maxmass))[0]
        self.low_ssfr_oi_sii_mass_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                       (self.siiflux_sn_sing>2) &
                                                       (self.oiflux_sn_sing>2) &
                                                       (self.mass_sing>minmass) &
                                                       (self.mass_sing<maxmass))[0]


        '''
        low ssfr group presub liner
        '''
        self.low_ssfr_sii_liner_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                     (self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                     (self.siiflux_sn_sing>2))[0]
        self.low_ssfr_oi_liner_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                    (self.oiflux_sn_sing>2)&
                                                    (self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                    (self.siiflux_sn_sing>2))[0]
        self.low_ssfr_oii_liner_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                     (self.oiiflux_sn_sing>2) &
                                                     (self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                     (self.siiflux_sn_sing>2))[0]
        self.low_ssfr_oii_sii_liner_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                         (self.siiflux_sn_sing>2) &
                                                         (self.oiiflux_sn_sing>2) &
                                                         (self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                         (self.siiflux_sn_sing>2))[0]
        self.low_ssfr_oi_sii_liner_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                        (self.oiflux_sn_sing>2) &
                                                        (self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                        (self.siiflux_sn_sing>2))[0]



        '''
        low ssfr group presub sy2
        '''
        self.low_ssfr_sii_sy2_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                   (self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                   (self.siiflux_sn_sing>2))[0]
        self.low_ssfr_oi_sy2_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                  (self.oiflux_sn_sing>2)&
                                                  (self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                  (self.siiflux_sn_sing>2))[0]
        self.low_ssfr_oii_sy2_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                   (self.oiiflux_sn_sing>2) &
                                                   (self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                   (self.siiflux_sn_sing>2))[0]
        self.low_ssfr_oii_sii_sy2_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                       (self.oiiflux_sn_sing>2)&
                                                       (self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                       (self.siiflux_sn_sing>2))[0]
        self.low_ssfr_oi_sii_sy2_filt_sing = np.where((self.delta_ssfr_sing<=-0.7) &
                                                      (self.oiflux_sn_sing>2)&
                                                      (self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                                      (self.siiflux_sn_sing>2))[0]


        '''
        full group filtering post sub
        '''
        self.ha_match_sing = np.where( (self.halpflux_sn_match_sing>2) &(self.hbetaflux_sn_match_sing>2))
        self.sii_sub_sing = np.where(self.siiflux_sub_sn_sing>2)
        self.oi_sub_sing = np.where(self.oiflux_sub_sn_sing>2)
        self.oii_sub_sing = np.where(self.oiiflux_sub_sn_sing>2)
        self.oii_sii_sub_sing = np.where((self.siiflux_sub_sn_sing>2) &(self.oiiflux_sub_sn_sing>2))
        self.oi_sii_sub_sing = np.where((self.siiflux_sub_sn_sing>2) &(self.oiflux_sub_sn_sing>2))
        ''' 
        full group filtering +mass restriction post sub
        '''
        
        self.ha_match_mass_filt_sing = np.where( (self.halpflux_sn_match_sing>2) &(self.hbetaflux_sn_match_sing>2)&(self.mass_sing>minmass) &(self.mass_sing<maxmass))
        self.sii_sub_mass_filt_sing = np.where((self.siiflux_sub_sn_sing>2)&(self.mass_sing>minmass) &(self.mass_sing<maxmass))
        self.oi_sub_mass_filt_sing = np.where((self.oiflux_sub_sn_sing>2)&(self.mass_sing>minmass) &(self.mass_sing<maxmass))
        self.oii_sub_mass_filt_sing = np.where((self.oiiflux_sub_sn_sing>2)&(self.mass_sing>minmass) &(self.mass_sing<maxmass))
        self.oii_sii_sub_mass_filt_sing = np.where((self.siiflux_sub_sn_sing>2) &(self.oiiflux_sub_sn_sing>2)&(self.mass_sing>minmass) &(self.mass_sing<maxmass))
        self.oi_sii_sub_mass_filt_sing = np.where((self.siiflux_sub_sn_sing>2) &(self.oiflux_sub_sn_sing>2)&(self.mass_sing>minmass) &(self.mass_sing<maxmass))

        ''' 
        full group filtering pre sub
        '''

        self.oi_sing = np.where(self.oiflux_sn_sing>2)
        self.oii_sing = np.where(self.oiiflux_sn_sing>2)
        self.oii_sii_sing = np.where((self.siiflux_sn_sing>2) &
                                     (self.oiiflux_sn_sing>2))
        self.oi_sii_sing = np.where((self.siiflux_sn_sing>2) &
                                    (self.oiflux_sn_sing>2))
        self.sii_sing = np.where(self.siiflux_sn_sing>2)[0]
        self.liner_filt = np.where((self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                         (self.siiflux_sn_sing>2))[0]
        self.sy2_filt = np.where((self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                         (self.siiflux_sn_sing>2))[0]
        '''
        full group filtering pre sub liner
        '''
        self.oi_liner_filt_sing = np.where((self.oiflux_sn_sing>2)&
                                           (self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                           (self.siiflux_sn_sing>2))[0]
        self.oii_liner_filt_sing = np.where((self.oiiflux_sn_sing>2)&
                                            (self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                            (self.siiflux_sn_sing>2))[0]
        self.oii_sii_liner_filt_sing = np.where((self.siiflux_sn_sing>2) &
                                                (self.oiiflux_sn_sing>2)&
                                                (self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ))[0]
        self.oi_sii_liner_filt_sing = np.where((self.siiflux_sn_sing>2) &
                                               (self.oiflux_sn_sing>2)&
                                               (self.log_oiii_hbeta_sing < np.log10(y2_linersy2(self.log_sii_halpha_sing)) ))[0]
        '''
        full group filtering pre sub sy2
        '''
        self.oi_sy2_filt_sing = np.where((self.oiflux_sn_sing>2)&
                                         (self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                         (self.siiflux_sn_sing>2))[0]
        self.oii_sy2_filt_sing = np.where((self.oiiflux_sn_sing>2)&
                                          (self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                          (self.siiflux_sn_sing>2))[0]
        self.oii_sii_sy2_filt_sing = np.where((self.siiflux_sn_sing>2) &
                                              (self.oiiflux_sn_sing>2)&
                                              (self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ))[0]
        self.oi_sii_sy2_filt_sing = np.where((self.siiflux_sn_sing>2) &
                                             (self.oiflux_sn_sing>2)&
                                             (self.log_oiii_hbeta_sing >= np.log10(y2_linersy2(self.log_sii_halpha_sing)) ) &
                                             (self.siiflux_sn_sing>2))[0]

        ''' 
        full group filtering +mass restriction post sub
        '''

        self.sii_sub_mass_filt_sing = np.where((self.siiflux_sub_sn_sing>2)  &(self.mass_sing>minmass) &(self.mass_sing<maxmass))
        self.oi_mass_filt_sing = np.where((self.oiflux_sn_sing>2) &(self.mass_sing>minmass) &(self.mass_sing<maxmass))
        self.oii_mass_filt_sing = np.where((self.oiiflux_sn_sing>2) &(self.mass_sing>minmass) &(self.mass_sing<maxmass))
        self.oii_sii_mass_filt_sing = np.where((self.siiflux_sn_sing>2) &(self.oiiflux_sn_sing>2) &(self.mass_sing>minmass) &(self.mass_sing<maxmass))
        self.oi_sii_mass_filt_sing = np.where((self.siiflux_sn_sing>2) &(self.oiflux_sn_sing>2) &(self.mass_sing>minmass) &(self.mass_sing<maxmass))

    def bin_by_bpt(self, binsize=0.1):
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
        sing_set = []
        low_set = []
        high_set = []
        mesh_x, mesh_y = np.meshgrid((xvals[:-1] +xvals[1:])/2, (yvals[:-1]+yvals[1:])/2) 
        coordpairs = {}
        for i in range(len(xvals)-1):
            for j in range(len(yvals)-1):


                valid_sing = np.where((self.log_oiii_hbeta_sing >= yvals[j]) &
                                      (self.log_oiii_hbeta_sing < yvals[j+1]) &
                                      (self.log_nii_halpha_sing >= xvals[i]) &
                                      (self.log_nii_halpha_sing < xvals[i+1] )
                                      )[0]
                
                valid_bpt = np.where((self.log_oiii_hbeta >= yvals[j] ) &
                                     (self.log_oiii_hbeta < yvals[j+1]) &
                                     (self.log_nii_halpha >= xvals[i] ) &
                                     (self.log_nii_halpha < xvals[i+1])
                                     )[0]
                valid_plus = np.where((self.log_oiii_hbeta_plus >= yvals[j] ) &
                                     (self.log_oiii_hbeta_plus < yvals[j+1]) &
                                     (self.log_nii_halpha_plus >= xvals[i] ) &
                                     (self.log_nii_halpha_plus < xvals[i+1])
                                     )[0]
                valid_neither = np.where((self.log_oiii_hbeta_neither >= yvals[j] ) &
                                     (self.log_oiii_hbeta_neither < yvals[j+1]) &
                                     (self.log_nii_halpha_neither >= xvals[i] ) &
                                     (self.log_nii_halpha_neither < xvals[i+1])
                                     )[0]
                valid_high = np.where((self.log_oiii_hbeta_sing[self.high_ssfr_obj] >= yvals[j] ) &
                                     (self.log_oiii_hbeta_sing[self.high_ssfr_obj] < yvals[j+1]) &
                                     (self.log_nii_halpha_sing[self.high_ssfr_obj] >= xvals[i] ) &
                                     (self.log_nii_halpha_sing[self.high_ssfr_obj] < xvals[i+1])
                                     )[0]
                valid_low = np.where((self.log_oiii_hbeta_sing[self.low_ssfr_obj] >= yvals[j] ) &
                                     (self.log_oiii_hbeta_sing[self.low_ssfr_obj] < yvals[j+1]) &
                                     (self.log_nii_halpha_sing[self.low_ssfr_obj] >= xvals[i] ) &
                                     (self.log_nii_halpha_sing[self.low_ssfr_obj] < xvals[i+1])
                                     )[0]
                
                match_dist_bpt = np.copy(self.mindists_best[self.agn_dist_inds])[valid_bpt]
                match_dist_plus = np.copy(self.mindists_best[self.agn_plus_dist_inds])[valid_plus]
                match_dist_neither = np.copy(self.mindists_best[self.agn_neither_dist_inds])[valid_neither]
                match_dist_sing = np.copy(self.mindists_best_sing_ord)[valid_sing]
                match_dist_high = np.copy(self.mindists_best_sing_ord)[valid_high]
                match_dist_low = np.copy(self.mindists_best_sing_ord)[valid_low]
                
                
                distro_full_x = np.copy(self.offset_nii_halpha_sing[valid_sing])
                distro_full_y = np.copy(self.offset_oiii_hbeta_sing[valid_sing])
                
                distro_high_x = np.copy(self.offset_nii_halpha_sing[valid_high])
                distro_high_y = np.copy(self.offset_oiii_hbeta_sing[valid_high])
                
                distro_low_x = np.copy(self.offset_nii_halpha_sing[valid_low])
                distro_low_y = np.copy(self.offset_oiii_hbeta_sing[valid_low])
                
                distro_bpt_x = np.copy(self.offset_nii_halpha[valid_bpt])
                distro_bpt_y = np.copy(self.offset_oiii_hbeta[valid_bpt])
                distro_plus_x = np.copy(self.offset_nii_halpha_plus[valid_plus])
                distro_plus_y = np.copy(self.offset_oiii_hbeta_plus[valid_plus])
                distro_neither_x = np.copy(self.offset_nii_halpha_neither[valid_neither])
                distro_neither_y = np.copy(self.offset_oiii_hbeta_neither[valid_neither])
                coordpairs[i*(len(mid_x)-1)+j] = ((i,j,mid_x[i], mid_y[j]))
                #print(i*(len(mid_x)-1)+j+1)
                if valid_sing.size > 10:
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
                    mn_x_bpt = np.mean(distro_bpt_x)
                    mn_y_bpt = np.mean(distro_bpt_y)
                    med_x_bpt = np.median(distro_bpt_x)
                    med_y_bpt = np.median(distro_bpt_y)
                    bpt_set.append([i,j, mid_x[i], mid_y[j], mn_x_bpt, mn_y_bpt, med_x_bpt, med_y_bpt, distro_bpt_x, distro_bpt_y, match_dist_bpt, valid_bpt])

                else:
                    bpt_set.append([i,j,mid_x[i], mid_y[j],0, 0, 0, 0, [], [],[], []])
                    #goodinds_pt.append(i*j)
                if valid_plus.size > 10:
                    mn_x_plus = np.mean(distro_plus_x)
                    mn_y_plus = np.mean(distro_plus_y)
                    med_x_plus = np.median(distro_plus_x)
                    med_y_plus = np.median(distro_plus_y)
                    plus_set.append([i,j, mid_x[i], mid_y[j],mn_x_plus, mn_y_plus, med_x_plus, med_y_plus, distro_plus_x, distro_plus_y, match_dist_plus, valid_plus])

                else:
                    plus_set.append([i,j,mid_x[i], mid_y[j], 0, 0, 0, 0, [], [], [], []])
                if valid_neither.size >10:
                    mn_x_neither = np.mean(distro_neither_x)
                    mn_y_neither = np.mean(distro_neither_y)
                    med_x_neither = np.median(distro_neither_x)
                    med_y_neither = np.median(distro_neither_y)
                    neither_set.append([i,j, mid_x[i], mid_y[j], mn_x_neither, mn_y_neither, med_x_neither, med_y_neither, distro_neither_x, distro_neither_y, match_dist_neither, valid_neither])
                else:
                    neither_set.append([i,j,mid_x[i], mid_y[j], 0, 0, 0, 0, [], [], [],[]])
        self.coordpairs = coordpairs
        self.bpt_set = bpt_set
        self.plus_set = plus_set
        self.neither_set = neither_set
        self.sing_set = sing_set
        self.low_set = low_set
        self.high_set = high_set
        self.mesh_x = mesh_x
        self.mesh_y = mesh_y
        self.binx_vals = xvals
        self.biny_vals = yvals
    def bin_quantity(self, quantity, binsize, mn, mx, threshold=0):
        bn_edges = np.arange(mn, mx, binsize)
        bncenters = (bn_edges[1:]+bn_edges[:-1])/2
        bns = []
        bn_inds = []
        valid_bns = []
        for i in range(len(bn_edges)-1):
            val = np.where((quantity>bn_edges[i]) & (quantity <= bn_edges[i+1]))[0]
            if val.size > threshold:
                bns.append(quantity[val])
                bn_inds.append(val)
                valid_bns.append(i)
        return bn_edges, bncenters, bns, bn_inds, valid_bns
    def bin_by_ind(self, quantity, inds, bncenters):
        binned_quantity = []
        for ind_set in inds:
            binned_quantity.append(quantity[ind_set])
        return binned_quantity
    def bootstrap(self, data, bootnum, data_only=False):
        bootstrap_results = apy_bootstrap(data, bootnum=bootnum)
        if data_only:
            return bootstrap_results
        means = np.mean(bootstrap_results, axis=1)
        std_mean = np.std(means)
        mean_means = np.mean(means)
        return bootstrap_results, means, std_mean, mean_means
    def get_deltassfr(self, mass, ssfr):
        delta_ssfr = ssfr-(mass*m_ssfr+b_ssfr)
        return delta_ssfr
    def nii_oii_to_oh(self, nii, oii):
        z = 1.08*(np.log10(nii/oii)**2) + 1.78*(np.log10(nii/oii)) + 1.24
        logoh = 12+np.log10(z*10**(-3.31))
        return logoh