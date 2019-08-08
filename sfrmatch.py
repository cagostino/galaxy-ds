import numpy as np
from ast_func import *
catfold='catalogs/'
import astropy.cosmology as apc
from loaddata_m2 import redshift_m2
cosmo = apc.Planck15
from setops import *
import time

class SFRMatch:
    def __init__(self, eldiag):
        self.eldiag=eldiag
        
    def get_highsn_match_only(self, agn_inds, sf_inds, sf_plus_inds, agnplus_inds, sncut=2, load=False):
        '''
        Matches BPT AGN to BPT SF, BPT+ SF, unclassified of similar SFR, M*, Mfib, z
        Ensures that the match has a high S/N in subtracted fluxes
        '''
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
                
                #computing differences for self matching agn
                diffsfr_agn = (self.eldiag.sfr[agn_ind]-self.eldiag.sfr[agn_inds][otheragns])**2
                diffmass_agn = (self.eldiag.mass[agn_ind]-self.eldiag.mass[agn_inds][otheragns])**2
                difffibmass_agn = (self.eldiag.fibmass[agn_ind]-self.eldiag.fibmass[agn_inds][otheragns])**2
                diffz_agn = (self.eldiag.z[agn_ind]-self.eldiag.z[agn_inds][otheragns])**2/np.std(redshift_m2)
                diffs_agn = np.sqrt(diffsfr_agn+diffmass_agn+difffibmass_agn+diffz_agn)
                

                
                mindiff_ind_agn = np.where(diffs_agn == np.min(diffs_agn) )[0]
    
                #computing differences for self matching agn to bpt+ agn
                diffsfr_agnplus = (self.eldiag.sfr[agn_ind]-self.eldiag.sfr_plus[agnplus_inds])**2
                diffmass_agnplus = (self.eldiag.mass[agn_ind]-self.eldiag.mass_plus[agnplus_inds])**2
                difffibmass_agnplus = (self.eldiag.fibmass[agn_ind]-self.eldiag.fibmass_plus[agnplus_inds])**2
                diffz_agnplus = (self.eldiag.z[agn_ind]-self.eldiag.z_plus[agnplus_inds])**2/np.std(redshift_m2)
                diffs_agnplus = np.sqrt(diffsfr_agnplus+diffmass_agnplus+difffibmass_agnplus+diffz_agnplus)
                mindiff_ind_agnplus = np.where(diffs_agnplus == np.min(diffs_agnplus) )[0]
                
                #computing differences for bpt SF 
                diffsfr_bpt = (self.eldiag.sfr[agn_ind] - self.eldiag.sfr[sf_inds])**2
                diffmass_bpt = (self.eldiag.mass[agn_ind] - self.eldiag.mass[sf_inds])**2
                difffibmass_bpt = (self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass[sf_inds])**2
                diffz_bpt = (self.eldiag.z[agn_ind]-self.eldiag.z[sf_inds])**2/np.std(redshift_m2)
                diffs_bpt = np.sqrt(diffsfr_bpt+diffmass_bpt+difffibmass_bpt+diffz_bpt)

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
                if len(inds_high_sn_bpt) >0:    
                    mindiff_ind_bpt = diffs_bpt_sort[inds_high_sn_bpt[0]]
                    n_pass_bpt = inds_high_sn_bpt[0]
                else:
                    mindiff_ind_bpt = -1
                    n_pass_bpt = len(diffs_bpt_sort)
                #computing differences for bpt+ SF 
                diffsfr_bptplus = (self.eldiag.sfr[agn_ind] - self.eldiag.sfr_plus[sf_plus_inds])**2
                diffmass_bptplus = (self.eldiag.mass[agn_ind] - self.eldiag.mass_plus[sf_plus_inds])**2
                difffibmass_bptplus = (self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass_plus[sf_plus_inds])**2
                diffz_bptplus = (self.eldiag.z[agn_ind]-self.eldiag.z_plus[sf_plus_inds])**2/np.std(redshift_m2)
                diffs_bptplus = np.sqrt(diffsfr_bptplus+diffmass_bptplus+difffibmass_bptplus+diffz_bptplus)

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
                                            (hbetaflux_sn_sub_plus[diffs_bptplus_sort]>sncut) &(halpflux_sn_sub_plus[diffs_bptplus_sort] >sncut) )[0]
                if len(inds_high_sn_bptplus) >0:    
                    mindiff_ind_bptplus = diffs_bptplus_sort[inds_high_sn_bptplus[0]]
                    n_pass_plus = inds_high_sn_bptplus[0]
                else:
                    mindiff_ind_neither = -1
                    n_pass_plus = len(diffs_bptplus_sort)

                #computing differences for unclassifiable 
                diffsfr_neither = (self.eldiag.sfr[agn_ind] - self.eldiag.sfr_neither)**2
                diffmass_neither = (self.eldiag.mass[agn_ind] - self.eldiag.mass_neither)**2
                difffibmass_neither = (self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass_neither)**2
                diffz_neither = (self.eldiag.z[agn_ind]-self.eldiag.z_neither)**2/np.std(redshift_m2)
                diffs_neither = np.sqrt(diffsfr_neither+diffmass_neither+difffibmass_neither+diffz_neither)

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
                                                (hbetaflux_sn_sub_neither[diffs_neither_sort]>sncut) &(halpflux_sn_sub_neither[diffs_neither_sort] >sncut) )[0]
                if len(inds_high_sn_neither)>0:
                    mindiff_ind_neither = diffs_neither_sort[inds_high_sn_neither[0]]
                    n_pass_neither = inds_high_sn_neither[0]
                else:
                    mindiff_ind_neither = -1
                    n_pass_neither = len(diffs_neither_sort)
                            
                                                      
                mindist_out = [diffs_bpt[mindiff_ind_bpt], diffs_bptplus[mindiff_ind_bptplus],  
                               diffs_neither[mindiff_ind_neither]]
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
            
            np.savetxt(catfold+'sfrm/sfrm_n_pass_best_highsn'+str(sncut)+'.txt',self.numpassed_best, fmt='%6.d')
            np.savetxt(catfold+'sfrm/sfrm_n_pass_highsn'+str(sncut)+'.txt',self.numpassed.transpose(), fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/sfrm_agns_selfmatch_highsn'+str(sncut)+'.txt',self.agns_selfmatch, fmt='%6.d')
            np.savetxt(catfold+'sfrm/sfrm_agns_selfmatch_other_highsn'+str(sncut)+'.txt',self.agns_selfmatch_other, fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/sfrm_agnsplus_selfmatch_highsn'+str(sncut)+'.txt',self.agnsplus_selfmatch, fmt='%6.d')
            np.savetxt(catfold+'sfrm/sfrm_agnsplus_selfmatch_other_highsn'+str(sncut)+'.txt',self.agnsplus_selfmatch_other, fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/sfrm_agns_highsn'+str(sncut)+'.txt',self.agns, fmt='%6.d')
            np.savetxt(catfold+'sfrm/sfrm_sfs_highsn'+str(sncut)+'.txt',self.sfs, fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/sfrm_agns_plus_highsn'+str(sncut)+'.txt',self.agns_plus, fmt='%6.d')
            np.savetxt(catfold+'sfrm/sfrm_sfs_plus_highsn'+str(sncut)+'.txt',self.sfs_plus, fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/sfrm_neither_matches_highsn'+str(sncut)+'.txt',self.neither_matches, fmt='%6.d')
            np.savetxt(catfold+'sfrm/sfrm_neither_agn_highsn'+str(sncut)+'.txt',self.neither_agn, fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/sfrm_mindists_best_highsn'+str(sncut)+'.txt',self.mindists_best, fmt='%8.6f')
            
            np.savetxt(catfold+'sfrm/sfrm_mindists_highsn'+str(sncut)+'.txt',self.mindists.transpose(), fmt='%8.6f')
            np.savetxt(catfold+'sfrm/sfrm_mininds_highsn'+str(sncut)+'.txt',self.mininds.transpose(), fmt='%6.d')
            np.savetxt(catfold+'sfrm/sfrm_minids_highsn'+str(sncut)+'.txt',self.minids.transpose(), fmt='%18.d')
            
            np.savetxt(catfold+'sfrm/sfrm_mindists_agn_highsn'+str(sncut)+'.txt',self.mindists_agn.transpose(), fmt='%8.6f')
            np.savetxt(catfold+'sfrm/sfrm_mininds_agn_highsn'+str(sncut)+'.txt',self.mininds_agn.transpose(), fmt='%6.d')
            np.savetxt(catfold+'sfrm/sfrm_minids_agn_highsn'+str(sncut)+'.txt',self.minids_agn.transpose(), fmt='%18.d')
            np.savetxt(catfold+'sfrm/sfrm_mindistsagn_best_highsn'+str(sncut)+'.txt',self.mindistsagn_best, fmt='%8.6f')
                
        else:
            #once the matching is already done just need to load items in
            self.numpassed =  np.loadtxt(catfold+'sfrm/sfrm_n_pass_highsn'+str(sncut)+'.txt',dtype=np.int64, unpack=True)
            
            self.numpassed_best = np.loadtxt(catfold+'sfrm/sfrm_n_pass_best_highsn'+str(sncut)+'.txt',dtype=np.int64)
            
            self.agns_selfmatch = np.loadtxt(catfold+'sfrm/sfrm_agns_selfmatch_highsn'+str(sncut)+'.txt', dtype=np.int64)
            self.agns_selfmatch_other = np.loadtxt(catfold+'sfrm/sfrm_agns_selfmatch_other_highsn'+str(sncut)+'.txt', dtype=np.int64)
            
            self.agnsplus_selfmatch = np.loadtxt(catfold+'sfrm/sfrm_agnsplus_selfmatch_highsn'+str(sncut)+'.txt', dtype=np.int64)
            self.agnsplus_selfmatch_other = np.loadtxt(catfold+'sfrm/sfrm_agnsplus_selfmatch_other_highsn'+str(sncut)+'.txt', dtype=np.int64)
            
            self.agns = np.loadtxt(catfold+'sfrm/sfrm_agns_highsn'+str(sncut)+'.txt', dtype=np.int64)
            self.sfs = np.loadtxt(catfold+'sfrm/sfrm_sfs_highsn'+str(sncut)+'.txt', dtype=np.int64)
            
            self.agns_plus = np.loadtxt(catfold+'sfrm/sfrm_agns_plus_highsn'+str(sncut)+'.txt', dtype=np.int64)
            self.sfs_plus = np.loadtxt(catfold+'sfrm/sfrm_sfs_plus_highsn'+str(sncut)+'.txt', dtype=np.int64)

            self.neither_agn = np.loadtxt(catfold+'sfrm/sfrm_neither_agn_highsn'+str(sncut)+'.txt', dtype=np.int64)            
            self.neither_matches = np.loadtxt(catfold+'sfrm/sfrm_neither_matches_highsn'+str(sncut)+'.txt', dtype=np.int64)
            
            
            self.mindists = np.loadtxt(catfold+'sfrm/sfrm_mindists_highsn'+str(sncut)+'.txt', unpack=True)
            self.mininds = np.loadtxt(catfold+'sfrm/sfrm_mininds_highsn'+str(sncut)+'.txt', dtype=np.int64, unpack=True)
            self.minids = np.loadtxt(catfold+'sfrm/sfrm_minids_highsn'+str(sncut)+'.txt', dtype=np.int64, unpack=True)
            self.mindists_best = np.loadtxt(catfold+'sfrm/sfrm_mindists_best_highsn'+str(sncut)+'.txt')
            
            self.mindists_agn = np.loadtxt(catfold+'sfrm/sfrm_mindists_agn_highsn'+str(sncut)+'.txt', unpack=True)
            self.minids_agn = np.loadtxt(catfold+'sfrm/sfrm_mininds_agn_highsn'+str(sncut)+'.txt', dtype=np.int64, unpack=True)
            self.mininds_agn = np.loadtxt(catfold+'sfrm/sfrm_minids_agn_highsn'+str(sncut)+'.txt', dtype=np.int64, unpack=True)
            self.mindistsagn_best = np.loadtxt(catfold+'sfrm/sfrm_mindistsagn_best_highsn'+str(sncut)+'.txt')

    def getmatch(self, agn_inds, sf_inds, sf_plus_inds, agnplus_inds, load=False):
        '''
        Matches BPT AGN to BPT SF, BPT+ SF of same SFR/M*
        '''
        
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
            
            mininds = np.zeros((3, len(agn_inds)))
            mindists = np.zeros((3, len(agn_inds)))
            minids = np.zeros((3, len(agn_inds)), dtype=np.int64)
            mininds_agn = np.zeros((2, len(agn_inds)))
            mindists_agn = np.zeros((2, len(agn_inds)))
            minids_agn = np.zeros((2, len(agn_inds)), dtype=np.int64)
            
            for i, agn_ind in enumerate(agn_inds):
                if i%10000 == 0:
                    print(i)
                    t = time.localtime()
                    current_time = time.strftime("%H:%M:%S", t)
                    print(current_time)                #for self-matching, don't want to compare to itself
                otheragns = np.where(agn_inds != agn_ind)[0]
                
                #computing differences for self matching agn
                diffsfr_agn = (self.eldiag.sfr[agn_ind]-self.eldiag.sfr[agn_inds][otheragns])**2
                diffmass_agn = (self.eldiag.mass[agn_ind]-self.eldiag.mass[agn_inds][otheragns])**2
                difffibmass_agn = (self.eldiag.fibmass[agn_ind]-self.eldiag.fibmass[agn_inds][otheragns])**2
                diffz_agn = (self.eldiag.z[agn_ind]-self.eldiag.z[agn_inds][otheragns])**2/np.std(redshift_m2)
                diffs_agn = np.sqrt(diffsfr_agn+diffmass_agn+difffibmass_agn+diffz_agn)
                
                mindiff_ind_agn = np.where(diffs_agn == np.min(diffs_agn) )[0]
    
                #computing differences for self matching agn to bpt+ agn
                diffsfr_agnplus = (self.eldiag.sfr[agn_ind]-self.eldiag.sfr_plus[agnplus_inds])**2
                diffmass_agnplus = (self.eldiag.mass[agn_ind]-self.eldiag.mass_plus[agnplus_inds])**2
                difffibmass_agnplus = (self.eldiag.fibmass[agn_ind]-self.eldiag.fibmass_plus[agnplus_inds])**2
                diffz_agnplus = (self.eldiag.z[agn_ind]-self.eldiag.z_plus[agnplus_inds])**2/np.std(redshift_m2)
                diffs_agnplus = np.sqrt(diffsfr_agnplus+diffmass_agnplus+difffibmass_agnplus+diffz_agnplus)
                mindiff_ind_agnplus = np.where(diffs_agnplus == np.min(diffs_agnplus) )[0]
                
                #computing differences for bpt SF 
                diffsfr_bpt = (self.eldiag.sfr[agn_ind] - self.eldiag.sfr[sf_inds])**2
                diffmass_bpt = (self.eldiag.mass[agn_ind] - self.eldiag.mass[sf_inds])**2
                difffibmass_bpt = (self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass[sf_inds])**2
                diffz_bpt = (self.eldiag.z[agn_ind]-self.eldiag.z[sf_inds])**2/np.std(redshift_m2)
                diffs_bpt = np.sqrt(diffsfr_bpt+diffmass_bpt+difffibmass_bpt+diffz_bpt)
                mindiff_ind_bpt = np.where(diffs_bpt == np.min(diffs_bpt) )[0]

                #if there happen to be multiple matches with the same min diff, just take first
                if mindiff_ind_bpt.size > 1: 
                    mindiff_ind_bpt = np.array([mindiff_ind_bpt[0]])

                #computing differences for bpt+ SF 
                diffsfr_bptplus = (self.eldiag.sfr[agn_ind] - self.eldiag.sfr_plus[sf_plus_inds])**2
                diffmass_bptplus = (self.eldiag.mass[agn_ind] - self.eldiag.mass_plus[sf_plus_inds])**2
                difffibmass_bptplus = (self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass_plus[sf_plus_inds])**2
                diffz_bptplus = (self.eldiag.z[agn_ind]-self.eldiag.z_plus[sf_plus_inds])**2/np.std(redshift_m2)
                diffs_bptplus = np.sqrt(diffsfr_bptplus+diffmass_bptplus+difffibmass_bptplus+diffz_bptplus)
                mindiff_ind_bptplus = np.where(diffs_bptplus == np.min(diffs_bptplus))[0]

                if mindiff_ind_bptplus.size > 1: 
                    mindiff_ind_bptplus = np.array([mindiff_ind_bptplus[0]])  

                #computing differences for unclassifiable 
                diffsfr_neither = (self.eldiag.sfr[agn_ind] - self.eldiag.sfr_neither)**2
                diffmass_neither = (self.eldiag.mass[agn_ind] - self.eldiag.mass_neither)**2
                difffibmass_neither = (self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass_neither)**2
                diffz_neither = (self.eldiag.z[agn_ind]-self.eldiag.z_neither)**2/np.std(redshift_m2)
                diffs_neither = np.sqrt(diffsfr_neither+diffmass_neither+difffibmass_neither+diffz_neither)
                        
                                      
                mindiff_ind_neither = np.where(diffs_neither == np.min(diffs_neither))[0]
                if mindiff_ind_neither.size > 1: 
                    mindiff_ind_neither = np.array([mindiff_ind_neither[0]])  
                #assigning the ids, inds, dists to be saved
                minid_out =[ self.eldiag.ids[sf_inds[mindiff_ind_bpt]],  
                            self.eldiag.ids_plus[sf_plus_inds[mindiff_ind_bptplus]],
                            self.eldiag.ids_neither[mindiff_ind_neither]]
                minind_out =[ sf_inds[mindiff_ind_bpt],  
                            sf_plus_inds[mindiff_ind_bptplus],
                            mindiff_ind_neither]
                
                mindist_out = [diffs_bpt[mindiff_ind_bpt], diffs_bptplus[mindiff_ind_bptplus],  
                               diffs_neither[mindiff_ind_neither]]
                
                #assigning the ids, inds, dists to be saved for self-matching

                minid_outagn = [self.eldiag.ids[agn_inds[mindiff_ind_agn]],  
                            self.eldiag.ids_plus[agnplus_inds[mindiff_ind_agnplus]]]
                minind_outagn =[ agn_inds[mindiff_ind_agn],  
                            agnplus_inds[mindiff_ind_bptplus]]
                mindist_outagn = [diffs_agn[mindiff_ind_agn], diffs_agnplus[mindiff_ind_agnplus]]
                
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
                    sfs.append(sf_inds[mindiff_ind_bpt][0])
                    agns.append(agn_ind)
                elif mindist_ind==1:
                    sfs_plus.append(sf_plus_inds[mindiff_ind_bptplus][0])
                    agns_plus.append(agn_ind)
                else:
                    neither_matches.append(mindiff_ind_neither[0])
                    neither_agn.append(agn_ind)
                    
                mindistagn_ind = np.int(np.where(mindist_outagn==np.min(mindist_outagn))[0])
                if mindistagn_ind ==0:
                    agns_selfmatch.append(agn_ind)
                    agns_selfmatch_other.append(mindiff_ind_agn[0])
                else:
                    agnsplus_selfmatch.append(agn_ind)
                    agnsplus_selfmatch_other.append(agnplus_inds[mindiff_ind_agnplus[0]])  
            #converting lists to arrays and saving to class attributes
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
            
            np.savetxt(catfold+'sfrm/sfrm_agns_selfmatch.txt',self.agns_selfmatch, fmt='%6.d')
            np.savetxt(catfold+'sfrm/sfrm_agns_selfmatch_other.txt',self.agns_selfmatch_other, fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/sfrm_agnsplus_selfmatch.txt',self.agnsplus_selfmatch, fmt='%6.d')
            np.savetxt(catfold+'sfrm/sfrm_agnsplus_selfmatch_other.txt',self.agnsplus_selfmatch_other, fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/sfrm_agns.txt',self.agns, fmt='%6.d')
            np.savetxt(catfold+'sfrm/sfrm_sfs.txt',self.sfs, fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/sfrm_agns_plus.txt',self.agns_plus, fmt='%6.d')
            np.savetxt(catfold+'sfrm/sfrm_sfs_plus.txt',self.sfs_plus, fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/sfrm_neither_matches.txt',self.neither_matches, fmt='%6.d')
            np.savetxt(catfold+'sfrm/sfrm_neither_agn.txt',self.neither_agn, fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/sfrm_mindists_best.txt',self.mindists_best, fmt='%8.6f')
            
            np.savetxt(catfold+'sfrm/sfrm_mindists.txt',self.mindists.transpose(), fmt='%8.6f')
            np.savetxt(catfold+'sfrm/sfrm_mininds.txt',self.mininds.transpose(), fmt='%6.d')
            np.savetxt(catfold+'sfrm/sfrm_minids.txt',self.minids.transpose(), fmt='%18.d')
            
            np.savetxt(catfold+'sfrm/sfrm_mindists_agn.txt',self.mindists_agn.transpose(), fmt='%8.6f')
            np.savetxt(catfold+'sfrm/sfrm_mininds_agn.txt',self.mininds_agn.transpose(), fmt='%6.d')
            np.savetxt(catfold+'sfrm/sfrm_minids_agn.txt',self.minids_agn.transpose(), fmt='%18.d')
            np.savetxt(catfold+'sfrm/sfrm_mindistsagn_best.txt',self.mindistsagn_best, fmt='%8.6f')
                
        else:
            #once the matching is already done just need to load items in
            
            self.agns_selfmatch = np.loadtxt(catfold+'sfrm/sfrm_agns_selfmatch.txt', dtype=np.int64)
            self.agns_selfmatch_other = np.loadtxt(catfold+'sfrm/sfrm_agns_selfmatch_other.txt', dtype=np.int64)
            
            self.agnsplus_selfmatch = np.loadtxt(catfold+'sfrm/sfrm_agnsplus_selfmatch.txt', dtype=np.int64)
            self.agnsplus_selfmatch_other = np.loadtxt(catfold+'sfrm/sfrm_agnsplus_selfmatch_other.txt', dtype=np.int64)
            
            self.agns = np.loadtxt(catfold+'sfrm/sfrm_agns.txt', dtype=np.int64)
            self.sfs = np.loadtxt(catfold+'sfrm/sfrm_sfs.txt', dtype=np.int64)
            
            self.agns_plus = np.loadtxt(catfold+'sfrm/sfrm_agns_plus.txt', dtype=np.int64)
            self.sfs_plus = np.loadtxt(catfold+'sfrm/sfrm_sfs_plus.txt', dtype=np.int64)

            self.neither_agn = np.loadtxt(catfold+'sfrm/sfrm_neither_agn.txt', dtype=np.int64)            
            self.neither_matches = np.loadtxt(catfold+'sfrm/sfrm_neither_matches.txt', dtype=np.int64)
            
            
            self.mindists = np.loadtxt(catfold+'sfrm/sfrm_mindists.txt', unpack=True)
            self.mininds = np.loadtxt(catfold+'sfrm/sfrm_mininds.txt', dtype=np.int64, unpack=True)
            self.minids = np.loadtxt(catfold+'sfrm/sfrm_minids.txt', dtype=np.int64, unpack=True)
            self.mindists_best = np.loadtxt(catfold+'sfrm/sfrm_mindists_best.txt')
            
            self.mindists_agn = np.loadtxt(catfold+'sfrm/sfrm_mindists_agn.txt', unpack=True)
            self.minids_agn = np.loadtxt(catfold+'sfrm/sfrm_mininds_agn.txt', dtype=np.int64, unpack=True)
            self.mininds_agn = np.loadtxt(catfold+'sfrm/sfrm_minids_agn.txt', dtype=np.int64, unpack=True)
            self.mindistsagn_best = np.loadtxt(catfold+'sfrm/sfrm_mindistsagn_best.txt')

    def getmatch_lindist(self, agn_inds, sf_inds, sf_plus_inds, agnplus_inds, load=False):
        '''
        Matches BPT AGN to BPT SF, BPT+ SF of same SFR/M*
        '''
        
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
            
            mininds = np.zeros((3, len(agn_inds)))
            mindists = np.zeros((3, len(agn_inds)))
            minids = np.zeros((3, len(agn_inds)), dtype=np.int64)
            mininds_agn = np.zeros((2, len(agn_inds)))
            mindists_agn = np.zeros((2, len(agn_inds)))
            minids_agn = np.zeros((2, len(agn_inds)), dtype=np.int64)
            
            for i, agn_ind in enumerate(agn_inds):
                if i%10000 == 0:
                    print(i)
                    t = time.localtime()
                    current_time = time.strftime("%H:%M:%S", t)
                    print(current_time)                #for self-matching, don't want to compare to itself
                otheragns = np.where(agn_inds != agn_ind)[0]
                
                #computing differences for self matching agn
                diffsfr_agn = abs(self.eldiag.sfr[agn_ind]-self.eldiag.sfr[agn_inds][otheragns])
                diffmass_agn = abs(self.eldiag.mass[agn_ind]-self.eldiag.mass[agn_inds][otheragns])
                difffibmass_agn = abs(self.eldiag.fibmass[agn_ind]-self.eldiag.fibmass[agn_inds][otheragns])
                diffz_agn = abs(self.eldiag.z[agn_ind]-self.eldiag.z[agn_inds][otheragns])/np.std(redshift_m2)
                diffs_agn = (diffsfr_agn+diffmass_agn+difffibmass_agn+diffz_agn)
                
                mindiff_ind_agn = np.where(diffs_agn == np.min(diffs_agn) )[0]
    
                #computing differences for self matching agn to bpt+ agn
                diffsfr_agnplus = abs(self.eldiag.sfr[agn_ind]-self.eldiag.sfr_plus[agnplus_inds])
                diffmass_agnplus = abs(self.eldiag.mass[agn_ind]-self.eldiag.mass_plus[agnplus_inds])
                difffibmass_agnplus = abs(self.eldiag.fibmass[agn_ind]-self.eldiag.fibmass_plus[agnplus_inds])
                diffz_agnplus = abs(self.eldiag.z[agn_ind]-self.eldiag.z_plus[agnplus_inds])/np.std(redshift_m2)
                diffs_agnplus = (diffsfr_agnplus+diffmass_agnplus+difffibmass_agnplus+diffz_agnplus)
                mindiff_ind_agnplus = np.where(diffs_agnplus == np.min(diffs_agnplus) )[0]
                
                #computing differences for bpt SF 
                diffsfr_bpt = abs(self.eldiag.sfr[agn_ind] - self.eldiag.sfr[sf_inds])
                diffmass_bpt = abs(self.eldiag.mass[agn_ind] - self.eldiag.mass[sf_inds])
                difffibmass_bpt = abs(self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass[sf_inds])
                diffz_bpt = abs(self.eldiag.z[agn_ind]-self.eldiag.z[sf_inds])/np.std(redshift_m2)
                diffs_bpt = (diffsfr_bpt+diffmass_bpt+difffibmass_bpt+diffz_bpt)
                mindiff_ind_bpt = np.where(diffs_bpt == np.min(diffs_bpt) )[0]

                #if there happen to be multiple matches with the same min diff, just take first
                if mindiff_ind_bpt.size > 1: 
                    mindiff_ind_bpt = np.array([mindiff_ind_bpt[0]])

                #computing differences for bpt+ SF 
                diffsfr_bptplus = abs(self.eldiag.sfr[agn_ind] - self.eldiag.sfr_plus[sf_plus_inds])
                diffmass_bptplus = abs(self.eldiag.mass[agn_ind] - self.eldiag.mass_plus[sf_plus_inds])
                difffibmass_bptplus = abs(self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass_plus[sf_plus_inds])
                diffz_bptplus = abs(self.eldiag.z[agn_ind]-self.eldiag.z_plus[sf_plus_inds])/np.std(redshift_m2)
                diffs_bptplus = (diffsfr_bptplus+diffmass_bptplus+difffibmass_bptplus+diffz_bptplus)
                mindiff_ind_bptplus = np.where(diffs_bptplus == np.min(diffs_bptplus))[0]

                if mindiff_ind_bptplus.size > 1: 
                    mindiff_ind_bptplus = np.array([mindiff_ind_bptplus[0]])  

                #computing differences for unclassifiable 
                diffsfr_neither = abs(self.eldiag.sfr[agn_ind] - self.eldiag.sfr_neither)
                diffmass_neither = abs(self.eldiag.mass[agn_ind] - self.eldiag.mass_neither)
                difffibmass_neither = abs(self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass_neither)
                diffz_neither = abs(self.eldiag.z[agn_ind]-self.eldiag.z_neither)/np.std(redshift_m2)
                diffs_neither = (diffsfr_neither+diffmass_neither+difffibmass_neither+diffz_neither)
                        
                                      
                mindiff_ind_neither = np.where(diffs_neither == np.min(diffs_neither))[0]
                if mindiff_ind_neither.size > 1: 
                    mindiff_ind_neither = np.array([mindiff_ind_neither[0]])  
                #assigning the ids, inds, dists to be saved
                minid_out =[ self.eldiag.ids[sf_inds[mindiff_ind_bpt]],  
                            self.eldiag.ids_plus[sf_plus_inds[mindiff_ind_bptplus]],
                            self.eldiag.ids_neither[mindiff_ind_neither]]
                minind_out =[ sf_inds[mindiff_ind_bpt],  
                            sf_plus_inds[mindiff_ind_bptplus],
                            mindiff_ind_neither]
                
                mindist_out = [diffs_bpt[mindiff_ind_bpt], diffs_bptplus[mindiff_ind_bptplus],  
                               diffs_neither[mindiff_ind_neither]]
                
                #assigning the ids, inds, dists to be saved for self-matching

                minid_outagn = [self.eldiag.ids[agn_inds[mindiff_ind_agn]],  
                            self.eldiag.ids_plus[agnplus_inds[mindiff_ind_agnplus]]]
                minind_outagn =[ agn_inds[mindiff_ind_agn],  
                            agnplus_inds[mindiff_ind_bptplus]]
                mindist_outagn = [diffs_agn[mindiff_ind_agn], diffs_agnplus[mindiff_ind_agnplus]]
                
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
                    sfs.append(sf_inds[mindiff_ind_bpt][0])
                    agns.append(agn_ind)
                elif mindist_ind==1:
                    sfs_plus.append(sf_plus_inds[mindiff_ind_bptplus][0])
                    agns_plus.append(agn_ind)
                else:
                    neither_matches.append(mindiff_ind_neither[0])
                    neither_agn.append(agn_ind)
                    
                mindistagn_ind = np.int(np.where(mindist_outagn==np.min(mindist_outagn))[0])
                if mindistagn_ind ==0:
                    agns_selfmatch.append(agn_ind)
                    agns_selfmatch_other.append(mindiff_ind_agn[0])
                else:
                    agnsplus_selfmatch.append(agn_ind)
                    agnsplus_selfmatch_other.append(agnplus_inds[mindiff_ind_agnplus[0]])  
            #converting lists to arrays and saving to class attributes
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
            
            np.savetxt(catfold+'sfrm/lindist/sfrm_agns_selfmatch.txt',self.agns_selfmatch, fmt='%6.d')
            np.savetxt(catfold+'sfrm/lindist/sfrm_agns_selfmatch_other.txt',self.agns_selfmatch_other, fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/lindist/sfrm_agnsplus_selfmatch.txt',self.agnsplus_selfmatch, fmt='%6.d')
            np.savetxt(catfold+'sfrm/lindist/sfrm_agnsplus_selfmatch_other.txt',self.agnsplus_selfmatch_other, fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/lindist/sfrm_agns.txt',self.agns, fmt='%6.d')
            np.savetxt(catfold+'sfrm/lindist/sfrm_sfs.txt',self.sfs, fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/lindist/sfrm_agns_plus.txt',self.agns_plus, fmt='%6.d')
            np.savetxt(catfold+'sfrm/lindist/sfrm_sfs_plus.txt',self.sfs_plus, fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/lindist/sfrm_neither_matches.txt',self.neither_matches, fmt='%6.d')
            np.savetxt(catfold+'sfrm/lindist/sfrm_neither_agn.txt',self.neither_agn, fmt='%6.d')
            
            np.savetxt(catfold+'sfrm/lindist/sfrm_mindists_best.txt',self.mindists_best, fmt='%8.6f')
            
            np.savetxt(catfold+'sfrm/lindist/sfrm_mindists.txt',self.mindists.transpose(), fmt='%8.6f')
            np.savetxt(catfold+'sfrm/lindist/sfrm_mininds.txt',self.mininds.transpose(), fmt='%6.d')
            np.savetxt(catfold+'sfrm/lindist/sfrm_minids.txt',self.minids.transpose(), fmt='%18.d')
            
            np.savetxt(catfold+'sfrm/lindist/sfrm_mindists_agn.txt',self.mindists_agn.transpose(), fmt='%8.6f')
            np.savetxt(catfold+'sfrm/lindist/sfrm_mininds_agn.txt',self.mininds_agn.transpose(), fmt='%6.d')
            np.savetxt(catfold+'sfrm/lindist/sfrm_minids_agn.txt',self.minids_agn.transpose(), fmt='%18.d')
            np.savetxt(catfold+'sfrm/lindist/sfrm_mindistsagn_best.txt',self.mindistsagn_best, fmt='%8.6f')
                
        else:
            #once the matching is already done just need to load items in
            
            self.agns_selfmatch = np.loadtxt(catfold+'sfrm/lindist/sfrm_agns_selfmatch.txt', dtype=np.int64)
            self.agns_selfmatch_other = np.loadtxt(catfold+'sfrm/lindist/sfrm_agns_selfmatch_other.txt', dtype=np.int64)
            
            self.agnsplus_selfmatch = np.loadtxt(catfold+'sfrm/lindist/sfrm_agnsplus_selfmatch.txt', dtype=np.int64)
            self.agnsplus_selfmatch_other = np.loadtxt(catfold+'sfrm/lindist/sfrm_agnsplus_selfmatch_other.txt', dtype=np.int64)
            
            self.agns = np.loadtxt(catfold+'sfrm/lindist/sfrm_agns.txt', dtype=np.int64)
            self.sfs = np.loadtxt(catfold+'sfrm/lindist/sfrm_sfs.txt', dtype=np.int64)
            
            self.agns_plus = np.loadtxt(catfold+'sfrm/lindist/sfrm_agns_plus.txt', dtype=np.int64)
            self.sfs_plus = np.loadtxt(catfold+'sfrm/lindist/sfrm_sfs_plus.txt', dtype=np.int64)

            self.neither_agn = np.loadtxt(catfold+'sfrm/lindist/sfrm_neither_agn.txt', dtype=np.int64)            
            self.neither_matches = np.loadtxt(catfold+'sfrm/lindist/sfrm_neither_matches.txt', dtype=np.int64)
            
            
            self.mindists = np.loadtxt(catfold+'sfrm/lindist/sfrm_mindists.txt', unpack=True)
            self.mininds = np.loadtxt(catfold+'sfrm/lindist/sfrm_mininds.txt', dtype=np.int64, unpack=True)
            self.minids = np.loadtxt(catfold+'sfrm/lindist/sfrm_minids.txt', dtype=np.int64, unpack=True)
            self.mindists_best = np.loadtxt(catfold+'sfrm/lindist/sfrm_mindists_best.txt')
            
            self.mindists_agn = np.loadtxt(catfold+'sfrm/lindist/sfrm_mindists_agn.txt', unpack=True)
            self.minids_agn = np.loadtxt(catfold+'sfrm/lindist/sfrm_mininds_agn.txt', dtype=np.int64, unpack=True)
            self.mininds_agn = np.loadtxt(catfold+'sfrm/lindist/sfrm_minids_agn.txt', dtype=np.int64, unpack=True)
            self.mindistsagn_best = np.loadtxt(catfold+'sfrm/lindist/sfrm_mindistsagn_best.txt')


    def subtract_elflux(self, sncut=2):
    
        self.bptdistrat = (np.array(cosmo.luminosity_distance(self.eldiag.z[self.sfs]))/np.array(cosmo.luminosity_distance(self.eldiag.z[self.agns])))**2 
        self.oiiiflux_sub  =  self.eldiag.oiiiflux[self.agns] - self.eldiag.oiiiflux[self.sfs]*self.bptdistrat
        self.niiflux_sub  = self.eldiag.niiflux[self.agns] - self.eldiag.niiflux[self.sfs]*self.bptdistrat
        self.hbetaflux_sub = self.eldiag.hbetaflux[self.agns] - self.eldiag.hbetaflux[self.sfs]*self.bptdistrat
        self.halpflux_sub = self.eldiag.halpflux[self.agns] - self.eldiag.halpflux[self.sfs]* self.bptdistrat

        self.oiiiflux_sub_err = np.sqrt(self.eldiag.oiii_err_bpt[self.agns]**2 +(self.eldiag.oiii_err_bpt[self.sfs]*self.bptdistrat)**2)
        self.niiflux_sub_err = np.sqrt(self.eldiag.nii_err_bpt[self.agns]**2+(self.eldiag.nii_err_bpt[self.sfs]*self.bptdistrat)**2)        
        self.hbetaflux_sub_err = np.sqrt(self.eldiag.hbeta_err_bpt[self.agns]**2+(self.eldiag.hbeta_err_bpt[self.sfs]*self.bptdistrat)**2)
        self.halpflux_sub_err = np.sqrt(self.eldiag.halp_err_bpt[self.agns]**2+(self.eldiag.halp_err_bpt[self.sfs]*self.bptdistrat)**2)

        #1e17 factor because flux converted to ergs/s in ELObj.py
        self.oiiiflux_sub_sn = self.oiiiflux_sub*1e17/self.oiiiflux_sub_err
        self.niiflux_sub_sn = self.niiflux_sub*1e17/self.niiflux_sub_err
        self.hbetaflux_sub_sn = self.hbetaflux_sub*1e17/self.hbetaflux_sub_err
        self.halpflux_sub_sn = self.halpflux_sub*1e17/self.halpflux_sub_err
        
        #bpt filter
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
        
        self.av_sub = self.eldiag.extinction(ha=self.halpflux_sub, hb=self.hbetaflux_sub)
        self.av_agn = self.eldiag.av[self.agns]
        self.av_sf = self.eldiag.av[self.sfs]
        
        self.oiiiflux_sub_dered = self.eldiag.dustcorrect(self.oiiiflux_sub, av=self.av_sub)
        self.z = np.copy(self.eldiag.z[self.agns]) 
        self.mass = np.copy(self.eldiag.mass[self.agns])
        self.ssfr = np.copy(self.eldiag.ssfr[self.agns])
        
        self.oiiilum_sub_dered = getlumfromflux(self.oiiiflux_sub_dered, self.z)
        
        self.log_oiii_hbeta_sub = np.log10(self.oiiiflux_sub/self.hbetaflux_sub)
        self.log_nii_halpha_sub = np.log10(self.niiflux_sub/self.halpflux_sub)


        #computing values for bpt plus matches
        self.bptdistrat_plus = (np.array(cosmo.luminosity_distance(self.eldiag.z_plus[self.sfs_plus]))/np.array(cosmo.luminosity_distance(self.eldiag.z[self.agns_plus])))**2 #Mpc to cm

        
        self.oiiiflux_sub_plus = self.eldiag.oiiiflux[self.agns_plus] - self.eldiag.oiiiflux_plus[self.sfs_plus]*self.bptdistrat_plus
        self.niiflux_sub_plus = self.eldiag.niiflux[self.agns_plus] - self.eldiag.niiflux_plus[self.sfs_plus]*self.bptdistrat_plus
        self.hbetaflux_sub_plus = self.eldiag.hbetaflux[self.agns_plus] - self.eldiag.hbetaflux_plus[self.sfs_plus]*self.bptdistrat_plus
        self.halpflux_sub_plus = self.eldiag.halpflux[self.agns_plus] - self.eldiag.halpflux_plus[self.sfs_plus]*self.bptdistrat_plus
        
        self.oiiiflux_sub_err_plus = np.sqrt(self.eldiag.oiii_err_bpt[self.agns_plus]**2 +(self.eldiag.oiii_err_plus[self.sfs_plus]*self.bptdistrat_plus)**2)
        self.niiflux_sub_err_plus = np.sqrt(self.eldiag.nii_err_bpt[self.agns_plus]**2+(self.eldiag.nii_err_plus[self.sfs_plus]*self.bptdistrat_plus)**2)        
        self.hbetaflux_sub_err_plus = np.sqrt(self.eldiag.hbeta_err_bpt[self.agns_plus]**2+(self.eldiag.hbeta_err_plus[self.sfs_plus]*self.bptdistrat_plus)**2)
        self.halpflux_sub_err_plus = np.sqrt(self.eldiag.halp_err_bpt[self.agns_plus]**2+(self.eldiag.halp_err_plus[self.sfs_plus]*self.bptdistrat_plus)**2)

        self.oiiiflux_sub_sn_plus = self.oiiiflux_sub_plus*1e17/self.oiiiflux_sub_err_plus
        self.niiflux_sub_sn_plus = self.niiflux_sub_plus*1e17/self.niiflux_sub_err_plus
        self.hbetaflux_sub_sn_plus = self.hbetaflux_sub_plus*1e17/self.hbetaflux_sub_err_plus
        self.halpflux_sub_sn_plus= self.halpflux_sub_plus*1e17/self.halpflux_sub_err_plus

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

        
        
        self.av_plus_sf = self.eldiag.av_plus[self.sfs_plus]
        self.av_plus_agn = self.eldiag.av[self.agns_plus]
        self.av_sub_plus = self.eldiag.extinction(ha=self.halpflux_sub_plus, hb=self.hbetaflux_sub_plus)
        
        self.oiiiflux_sub_plus_dered = self.eldiag.dustcorrect(self.oiiiflux_sub_plus, av=self.av_sub_plus)
        self.z_plus = np.copy(self.eldiag.z[self.agns_plus])
        self.mass_plus = np.copy(self.eldiag.mass[self.agns_plus])
        self.ssfr_plus = np.copy(self.eldiag.ssfr[self.agns_plus])
        
        self.oiiilum_sub_plus_dered = getlumfromflux(self.oiiiflux_sub_plus_dered, self.z_plus)
        
        self.log_oiii_hbeta_sub_plus = np.log10(self.oiiiflux_sub_plus/self.hbetaflux_sub_plus)
        self.log_nii_halpha_sub_plus = np.log10(self.niiflux_sub_plus/self.halpflux_sub_plus)


        #computing values for neither
        self.bptdistrat_neither = (np.array(cosmo.luminosity_distance(self.eldiag.z_neither[self.neither_matches]))/np.array(cosmo.luminosity_distance(self.eldiag.z[self.neither_agn])))**2 #Mpc to cm

        self.oiiiflux_sub_neither = self.eldiag.oiiiflux[self.neither_agn] - self.eldiag.oiiiflux_neither[self.neither_matches]*self.bptdistrat_neither
        self.niiflux_sub_neither = self.eldiag.niiflux[self.neither_agn]- self.eldiag.niiflux_neither[self.neither_matches]*self.bptdistrat_neither
        self.hbetaflux_sub_neither = self.eldiag.hbetaflux[self.neither_agn]- self.eldiag.hbetaflux_neither[self.neither_matches]*self.bptdistrat_neither
        self.halpflux_sub_neither = self.eldiag.halpflux[self.neither_agn]- self.eldiag.halpflux_neither[self.neither_matches]*self.bptdistrat_neither

        self.oiiiflux_sub_err_neither = np.sqrt(self.eldiag.oiii_err_bpt[self.neither_agn]**2 +(self.eldiag.oiii_err_neither[self.neither_matches]*self.bptdistrat_neither)**2)
        self.niiflux_sub_err_neither = np.sqrt(self.eldiag.nii_err_bpt[self.neither_agn]**2+(self.eldiag.nii_err_neither[self.neither_matches]*self.bptdistrat_neither)**2)        
        self.hbetaflux_sub_err_neither = np.sqrt(self.eldiag.hbeta_err_bpt[self.neither_agn]**2+(self.eldiag.hbeta_err_neither[self.neither_matches]*self.bptdistrat_neither)**2)
        self.halpflux_sub_err_neither = np.sqrt(self.eldiag.halp_err_bpt[self.neither_agn]**2+(self.eldiag.halp_err_neither[self.neither_matches]*self.bptdistrat_neither)**2)
 
        self.oiiiflux_sub_sn_neither = self.oiiiflux_sub_neither*1e17/self.oiiiflux_sub_err_neither
        self.niiflux_sub_sn_neither = self.niiflux_sub_neither*1e17/self.niiflux_sub_err_neither
        self.hbetaflux_sub_sn_neither = self.hbetaflux_sub_neither*1e17/self.hbetaflux_sub_err_neither
        self.halpflux_sub_sn_neither = self.halpflux_sub_neither*1e17/self.halpflux_sub_err_neither
        
        self.oiiiflux_sub_sn_sing = combine_arrs([self.oiiiflux_sub_sn, self.oiiiflux_sub_sn_plus, self.oiiiflux_sub_sn_neither])
        self.niiflux_sub_sn_sing = combine_arrs([self.niiflux_sub_sn, self.niiflux_sub_sn_plus, self.niiflux_sub_sn_neither])
        self.hbetaflux_sub_sn_sing = combine_arrs([self.hbetaflux_sub_sn, self.hbetaflux_sub_sn_plus, self.hbetaflux_sub_sn_neither])
        self.halpflux_sub_sn_sing = combine_arrs([self.halpflux_sub_sn, self.halpflux_sub_sn_plus, self.halpflux_sub_sn_neither])
        
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

        self.bpt_sing_sn_filt_bool = ((self.oiiiflux_sub_sn_sing > sncut)& (self.niiflux_sub_sn_sing > sncut)& 
                                    (self.hbetaflux_sub_sn_sing > sncut)& (self.halpflux_sub_sn_sing > sncut))
        self.bpt_sing_sn_filt = np.where(self.bpt_sing_sn_filt_bool)[0]
        self.bpt_sing_not_sn_filt_bool = np.logical_not(self.bpt_sing_sn_filt_bool)
        self.bpt_sing_not_sn_filt = np.where(self.bpt_sing_not_sn_filt_bool)[0]

                              
        self.av_sub_neither = self.eldiag.extinction(ha=self.halpflux_sub_neither, hb=self.hbetaflux_sub_neither)
        self.av_neither_sf = self.eldiag.av_neither[self.neither_matches]
        self.av_neither_agn = self.eldiag.av[self.neither_agn]
        
        self.oiiiflux_sub_neither_dered = self.eldiag.dustcorrect(self.oiiiflux_sub_neither, av=self.av_sub_neither)
        self.z_neither = np.copy(self.eldiag.z[self.neither_agn])
        self.mass_neither = np.copy(self.eldiag.mass[self.neither_agn])
        self.ssfr_neither = np.copy(self.eldiag.ssfr[self.neither_agn])
        
        self.oiiilum_sub_neither_dered = getlumfromflux(self.oiiiflux_sub_neither_dered, self.z_neither)
        
        self.log_oiii_hbeta_sub_neither = np.log10(self.oiiiflux_sub_neither/self.hbetaflux_sub_neither)
        self.log_nii_halpha_sub_neither = np.log10(self.niiflux_sub_neither/self.halpflux_sub_neither)
        
        self.log_oiii_hbeta_sub_sing = combine_arrs([self.log_oiii_hbeta_sub, self.log_oiii_hbeta_sub_plus, self.log_oiii_hbeta_sub_neither])
        self.log_nii_halpha_sub_sing = combine_arrs([self.log_nii_halpha_sub, self.log_nii_halpha_sub_plus, self.log_nii_halpha_sub_neither])
        
    def getmatch_avg(self, agn_inds, sf_inds, sf_plus_inds, agnplus_inds, load=False, n_avg=10):
        '''
        Matches BPT AGN to n nearest BPT SF, BPT+ SF of same SFR/M*
        '''
        self.n_avg = n_avg
            
        self.z_nonagn = combine_arrs([self.eldiag.z[sf_inds], self.eldiag.z_plus[sf_plus_inds], self.eldiag.z_neither])
        self.sfr_nonagn = combine_arrs([self.eldiag.sfr[sf_inds], self.eldiag.sfr_plus[sf_plus_inds], self.eldiag.sfr_neither])
        self.fibmass_nonagn = combine_arrs([self.eldiag.fibmass[sf_inds], self.eldiag.fibmass_plus[sf_plus_inds], self.eldiag.fibmass_neither])
        self.mass_nonagn = combine_arrs([self.eldiag.mass[sf_inds], self.eldiag.mass_plus[sf_plus_inds], self.eldiag.mass_neither])

        if not load:
            agns_selfmatch = []
            agnsplus_selfmatch = []
            agns_selfmatch_other = np.zeros((n_avg,len(agn_inds)))
            agnsplus_selfmatch_other = np.zeros((n_avg,len(agn_inds)))
            agns = []
            mindists = np.zeros(len(agn_inds))
            self.dists_agn = []
            self.dists_matches = []
            self.dists_best = []
            self.best_types = []
            n_agn_selfmatch = 0
            n_agnplus_selfmatch = 0
            mindists_agn = np.zeros((2, len(agn_inds)))
            
            sfs = np.zeros((n_avg, len(agn_inds)))            

            for i, agn_ind in enumerate(agn_inds):
                if i%1000 == 0:
                    print(i)
                    t = time.localtime()
                    current_time = time.strftime("%H:%M:%S", t)
                    print(current_time)                
                otheragns = np.where(agn_inds != agn_ind)[0]
                
                diffsfr_agn = abs(self.eldiag.sfr[agn_ind]-self.eldiag.sfr[agn_inds][otheragns])
                diffmass_agn = abs(self.eldiag.mass[agn_ind]-self.eldiag.mass[agn_inds][otheragns])
                difffibmass_agn = abs(self.eldiag.fibmass[agn_ind]-self.eldiag.fibmass[agn_inds][otheragns])
                diffz_agn = abs(self.eldiag.z[agn_ind]-self.eldiag.z[agn_inds][otheragns])/np.std(redshift_m2)
                diffs_agn = diffsfr_agn+diffmass_agn+difffibmass_agn+diffz_agn
                diffs_agn_sort = np.argsort(diffs_agn)[:self.n_avg]    
        
                avg_diffs_agn = np.mean(diffs_agn[diffs_agn_sort])
                
                diffsfr_agnplus = abs(self.eldiag.sfr[agn_ind]-self.eldiag.sfr_plus[agnplus_inds])
                diffmass_agnplus = abs(self.eldiag.mass[agn_ind]-self.eldiag.mass_plus[agnplus_inds])
                difffibmass_agnplus = abs(self.eldiag.fibmass[agn_ind]-self.eldiag.fibmass_plus[agnplus_inds])
                diffz_agnplus = abs(self.eldiag.z[agn_ind]-self.eldiag.z_plus[agnplus_inds])/np.std(redshift_m2)
                diffs_agnplus = diffsfr_agnplus+diffmass_agnplus+difffibmass_agnplus+diffz_agnplus
                diffs_agnplus_sort = np.argsort(diffs_agnplus)[:self.n_avg]                                
                avg_diffs_agnplus = np.mean(diffs_agnplus[diffs_agnplus_sort])
                
                
                diffsfr_bpt = abs(self.eldiag.sfr[agn_ind] - self.sfr_nonagn)
                diffmass_bpt = abs(self.eldiag.mass[agn_ind] - self.mass_nonagn)
                difffibmass_bpt = abs(self.eldiag.fibmass[agn_ind] - self.fibmass_nonagn)
                diffz_bpt = abs(self.eldiag.z[agn_ind]-self.z_nonagn)/np.std(redshift_m2)
                diffs_bpt = diffsfr_bpt+diffmass_bpt+difffibmass_bpt+diffz_bpt
                diffs_bpt_sort = np.argsort(diffs_bpt)[:self.n_avg]
                avg_diffs_bpt = np.mean(diffs_bpt[diffs_bpt_sort])
                
                mindist_outagn = [avg_diffs_agn, avg_diffs_agnplus]
                mindists[i] = avg_diffs_bpt
                mindists_agn[:, i] = mindist_outagn            
                agns.append(agn_ind)
                sfs[:, i] = diffs_bpt_sort
                self.dists_best.append(diffs_bpt[diffs_bpt_sort])
                self.best_types.append(0)
                
                mindistagn_ind = np.int(np.where(mindist_outagn==np.min(mindist_outagn))[0])
                if mindistagn_ind ==0:
                    agns_selfmatch.append(agn_ind)
                    agns_selfmatch_other[:,i] = diffs_agn_sort
                    n_agn_selfmatch +=1
                else:
                    agnsplus_selfmatch.append(agn_ind)
                    agnsplus_selfmatch_other[:,i] = diffs_agnplus_sort
                    n_agnplus_selfmatch += 1

            self.mindists_avg = np.copy(mindists)
            self.mindists_agn_avg = np.copy(mindists_agn)
            self.mindistsagn_best_avg = np.min(self.mindists_agn_avg, axis=0)    
            np.savetxt(catfold+'sfrm/sfrm_mindists_avg' +str(self.n_avg)+'.txt',self.mindists_avg.transpose(), fmt='%8.6f')               
            np.savetxt(catfold+'sfrm/sfrm_mindists_agn_avg' +str(self.n_avg)+'.txt',self.mindists_agn_avg.transpose(), fmt='%8.6f')
            np.savetxt(catfold+'sfrm/sfrm_mindistsagn_best_avg' +str(self.n_avg)+'.txt',self.mindistsagn_best_avg.transpose(), fmt='%8.6f')
            np.savetxt(catfold+'sfrm/sfrm_agns_avg' +str(self.n_avg)+'.txt', np.array(agns), fmt='%6.d')

            np.savetxt(catfold+'sfrm/sfrm_sfs_avg' +str(self.n_avg)+'.txt', np.array(sfs), fmt='%6.d')

            np.savetxt(catfold+'sfrm/sfrm_agns_selfmatch_avg' +str(self.n_avg)+'.txt', np.array(agns_selfmatch), fmt='%6.d') 
            np.savetxt(catfold+'sfrm/sfrm_agnsplus_selfmatch_avg' +str(self.n_avg)+'.txt', np.array(agnsplus_selfmatch), fmt='%6.d') 
            np.savetxt(catfold+'sfrm/sfrm_agns_selfmatch_other_avg' +str(self.n_avg)+'.txt', np.array(agns_selfmatch_other[0:n_agn_selfmatch]), fmt='%6.d') 
            np.savetxt(catfold+'sfrm/sfrm_agnsplus_selfmatch_other_avg' +str(self.n_avg)+'.txt', np.array(agnsplus_selfmatch_other[0:n_agnplus_selfmatch]), fmt='%6.d') 

        self.agns_selfmatch_avg = np.loadtxt(catfold+'sfrm/sfrm_agns_selfmatch_avg' +str(self.n_avg)+'.txt', dtype=np.int64)
        self.agns_selfmatch_other_avg = np.loadtxt(catfold+'sfrm/sfrm_agns_selfmatch_other_avg' +str(self.n_avg)+'.txt', dtype=np.int64)
        self.agnsplus_selfmatch_avg = np.loadtxt(catfold+'sfrm/sfrm_agnsplus_selfmatch_avg' +str(self.n_avg)+'.txt', dtype=np.int64)
        self.agnsplus_selfmatch_other_avg = np.loadtxt(catfold+'sfrm/sfrm_agnsplus_selfmatch_other_avg' +str(self.n_avg)+'.txt', dtype=np.int64)
        self.agns_avg = np.loadtxt(catfold+'sfrm/sfrm_agns_avg' +str(self.n_avg)+'.txt', dtype=np.int64)
        self.sfs_avg = np.loadtxt(catfold+'sfrm/sfrm_sfs_avg' +str(self.n_avg)+'.txt', dtype=np.int64, unpack=True)
        self.mindists_avg = np.loadtxt(catfold+'sfrm/sfrm_mindists_avg' +str(self.n_avg)+'.txt', unpack=True)
        self.mindists_agn_avg = np.loadtxt(catfold+'sfrm/sfrm_mindists_agn_avg' +str(self.n_avg)+'.txt', unpack=True)
        self.mindistsagn_best_avg = np.loadtxt(catfold+'sfrm/sfrm_mindistsagn_best_avg' +str(self.n_avg)+'.txt')

 
    def subtract_elflux_avg(self, sf_inds, sf_plus_inds, sncut=2):

        self.oiiiflux_nonagn = combine_arrs([self.eldiag.oiiiflux[sf_inds], self.eldiag.oiiiflux_plus[sf_plus_inds], self.eldiag.oiiiflux_neither])
        self.niiflux_nonagn = combine_arrs([self.eldiag.niiflux[sf_inds], self.eldiag.niiflux_plus[sf_plus_inds], self.eldiag.niiflux_neither])
        self.hbetaflux_nonagn = combine_arrs([self.eldiag.hbetaflux[sf_inds], self.eldiag.oiiiflux_plus[sf_plus_inds], self.eldiag.hbetaflux_neither])
        self.halpflux_nonagn  = combine_arrs([self.eldiag.halpflux[sf_inds], self.eldiag.halpflux_plus[sf_plus_inds], self.eldiag.halpflux_neither]) 
        
        self.oiiiflux_err_nonagn = combine_arrs([self.eldiag.oiii_err_bpt[sf_inds], self.eldiag.oiii_err_plus[sf_plus_inds], self.eldiag.oiii_err_neither])
        self.niiflux_err_nonagn = combine_arrs([self.eldiag.nii_err_bpt[sf_inds], self.eldiag.nii_err_plus[sf_plus_inds], self.eldiag.nii_err_neither])
        self.hbetaflux_err_nonagn = combine_arrs([self.eldiag.hbeta_err_bpt[sf_inds], self.eldiag.hbeta_err_plus[sf_plus_inds], self.eldiag.hbeta_err_neither])
        self.halpflux_err_nonagn  = combine_arrs([self.eldiag.halp_err_bpt[sf_inds], self.eldiag.halp_err_plus[sf_plus_inds], self.eldiag.halp_err_neither])
        self.av_nonagn = combine_arrs([self.eldiag.av[sf_inds], self.eldiag.av_plus[sf_plus_inds], self.eldiag.av_neither])
        
        self.bptdistrat_avg = (np.array(np.mean(cosmo.luminosity_distance(self.z_nonagn[self.sfs_avg]), 
                                                axis=1))/np.array(cosmo.luminosity_distance(self.eldiag.z[self.agns_avg])))**2 #Mpc to cm
        self.oiiiflux_sub_avg  =  self.eldiag.oiiiflux[self.agns_avg] - np.mean(self.oiiiflux_nonagn[self.sfs_avg], axis=1)*self.bptdistrat_avg
        self.niiflux_sub_avg  = self.eldiag.niiflux[self.agns_avg] - np.mean(self.niiflux_nonagn[self.sfs_avg], axis=1)*self.bptdistrat_avg
        self.hbetaflux_sub_avg = self.eldiag.hbetaflux[self.agns_avg] - np.mean(self.hbetaflux_nonagn[self.sfs_avg],axis=1)*self.bptdistrat_avg
        self.halpflux_sub_avg = self.eldiag.halpflux[self.agns_avg] - np.mean(self.halpflux_nonagn[self.sfs_avg], axis=1)* self.bptdistrat_avg

        self.oiiiflux_sub_err_avg = np.sqrt(self.eldiag.oiii_err_bpt[self.agns_avg]**2 +(np.mean(self.oiiiflux_err_nonagn[self.sfs_avg], axis=1)/np.sqrt(self.n_avg)*self.bptdistrat_avg)**2)
        self.niiflux_sub_err_avg = np.sqrt(self.eldiag.nii_err_bpt[self.agns_avg]**2+(np.mean(self.niiflux_err_nonagn[self.sfs_avg], axis=1)/np.sqrt(self.n_avg)*self.bptdistrat_avg)**2)        
        self.hbetaflux_sub_err_avg = np.sqrt(self.eldiag.hbeta_err_bpt[self.agns_avg]**2+(np.mean(self.hbetaflux_err_nonagn[self.sfs_avg],axis=1)/np.sqrt(self.n_avg)*self.bptdistrat_avg)**2)
        self.halpflux_sub_err_avg = np.sqrt(self.eldiag.halp_err_bpt[self.agns_avg]**2+(np.mean(self.halpflux_err_nonagn[self.sfs_avg], axis=1)/np.sqrt(self.n_avg)*self.bptdistrat_avg)**2)
 
        self.oiiiflux_sub_sn_avg = self.oiiiflux_sub_avg*1e17/self.oiiiflux_sub_err_avg
        self.niiflux_sub_sn_avg = self.niiflux_sub_avg*1e17/self.niiflux_sub_err_avg
        self.hbetaflux_sub_sn_avg = self.hbetaflux_sub_avg*1e17/self.hbetaflux_sub_err_avg
        self.halpflux_sub_sn_avg = self.halpflux_sub_avg*1e17/self.halpflux_sub_err_avg
        
        
        self.bpt_sn_filt_avg_bool = ((self.oiiiflux_sub_sn_avg > sncut)& (self.niiflux_sub_sn_avg > sncut)& 
                                    (self.hbetaflux_sub_sn_avg > sncut)& (self.halpflux_sub_sn_avg > sncut))
        self.bpt_sn_filt_avg_bool_intermed = ((self.oiiiflux_sub_sn_avg > sncut-1)& (self.niiflux_sub_sn_avg > sncut-1)& 
                                    (self.hbetaflux_sub_sn_avg > sncut-1)& (self.halpflux_sub_sn_avg > sncut-1)&
                                    (self.oiiiflux_sub_sn_avg < sncut)& (self.niiflux_sub_sn_avg < sncut)& 
                                    (self.hbetaflux_sub_sn_avg < sncut)& (self.halpflux_sub_sn_avg < sncut))
        
        
        self.bpt_sn_filt_avg = np.where(self.bpt_sn_filt_avg_bool)[0]
        self.bpt_sn_filt_avg_intermed = np.where(self.bpt_sn_filt_avg_bool_intermed)

        self.bpt_not_sn_filt_avg_bool = np.logical_not(self.bpt_sn_filt_avg_bool)
        self.bpt_not_sn_filt_avg = np.where(self.bpt_not_sn_filt_avg_bool)

        

        self.av_sub_avg = self.eldiag.extinction(ha=self.halpflux_sub_avg, hb=self.hbetaflux_sub_avg)
        self.av_agn_avg = self.eldiag.av[self.agns_avg]
        self.av_sf_avg = np.mean(self.av_nonagn[self.sfs_avg], axis = 1)
        
        self.oiiiflux_sub_dered_avg = self.eldiag.dustcorrect(self.oiiiflux_sub_avg, av=self.av_sub_avg)
        self.z_avg = np.copy(self.eldiag.z[self.agns_avg]) 
        self.mass_avg = np.copy(self.eldiag.mass[self.agns_avg])
        self.ssfr_avg = np.copy(self.eldiag.ssfr[self.agns_avg])
        
        self.oiiilum_sub_dered_avg = getlumfromflux(self.oiiiflux_sub_dered_avg, self.z_avg)
        
        self.log_oiii_hbeta_sub_avg = np.log10(self.oiiiflux_sub_avg/self.hbetaflux_sub_avg)
        self.log_nii_halpha_sub_avg = np.log10(self.niiflux_sub_avg/self.halpflux_sub_avg)
    
    def getmatch_avg_sep(self, agn_inds, sf_inds, sf_plus_inds, agnplus_inds, load=False, n_avg=3):
        '''
        Matches BPT AGN to n nearest BPT SF, BPT+ SF of same SFR/M*
        '''
        self.n_avg = n_avg
        if not load:

            agns_selfmatch = []
            
            agnsplus_selfmatch = []
            
            agns = []
            agns_plus = []                   
            neither_agn = []
            #massive_agns = np.where(self.eldiag.mass[agn_inds]>=10)[0]
            mindists = np.zeros((3, len(agn_inds)))
            self.dists_agn = []
            self.dists_bpt = []
            self.dists_agnplus = []
            self.dists_bptplus = []
            self.dists_neither = [] 
            self.dists_best = []
            self.best_types = []
            mindists_agn = np.zeros((2, len(agn_inds)))
            #store agns in agns_plus when they match to bpt+ sf best

            selfmatchother_fil = open(catfold+'sfrm/sfrm_agns_selfmatch_other_avg' +str(self.n_avg)+'.txt','w')
            selfmatch_agnsplusother_fil = open(catfold+'sfrm/sfrm_agnsplus_selfmatch_other_avg' +str(self.n_avg)+'.txt', 'w')
            
            sfs_fil = open(catfold+'sfrm/sfrm_sfs_avg' +str(self.n_avg)+'.txt','w')
            sfs_plus_fil = open(catfold+'sfrm/sfrm_sfs_plus_avg' +str(self.n_avg)+'.txt','w')
            
            neither_match_fil = open(catfold+'sfrm/sfrm_neither_matches_avg' +str(self.n_avg)+'.txt','w')
            
            for i, agn_ind in enumerate(agn_inds):
                if i%1000 == 0:
                    print(i)
                    t = time.localtime()
                    current_time = time.strftime("%H:%M:%S", t)
                    print(current_time)
                otheragns = np.where(agn_inds != agn_ind)[0]
                
                diffsfr_agn = abs(self.eldiag.sfr[agn_ind]-self.eldiag.sfr[agn_inds][otheragns])
                diffmass_agn = abs(self.eldiag.mass[agn_ind]-self.eldiag.mass[agn_inds][otheragns])
                difffibmass_agn = abs(self.eldiag.fibmass[agn_ind]-self.eldiag.fibmass[agn_inds][otheragns])
                diffz_agn = abs(self.eldiag.z[agn_ind]-self.eldiag.z[agn_inds][otheragns])/np.std(redshift_m2)
                diffs_agn = diffsfr_agn+diffmass_agn+difffibmass_agn+diffz_agn
                diffs_agn_sort = np.argsort(diffs_agn)[:self.n_avg]    
                
                avg_diffs_agn = np.mean(diffs_agn[diffs_agn_sort])
                
                diffsfr_agnplus = abs(self.eldiag.sfr[agn_ind]-self.eldiag.sfr_plus[agnplus_inds])
                diffmass_agnplus = abs(self.eldiag.mass[agn_ind]-self.eldiag.mass_plus[agnplus_inds])
                difffibmass_agnplus = abs(self.eldiag.fibmass[agn_ind]-self.eldiag.fibmass_plus[agnplus_inds])
                diffz_agnplus = abs(self.eldiag.z[agn_ind]-self.eldiag.z_plus[agnplus_inds])/np.std(redshift_m2)
                diffs_agnplus = diffsfr_agnplus+diffmass_agnplus+difffibmass_agnplus+diffz_agnplus
                diffs_agnplus_sort = np.argsort(diffs_agnplus)[:self.n_avg]                                
                avg_diffs_agnplus = np.mean(diffs_agnplus[diffs_agnplus_sort])
                
                
                diffsfr_bpt = abs(self.eldiag.sfr[agn_ind] - self.eldiag.sfr[sf_inds])
                diffmass_bpt = abs(self.eldiag.mass[agn_ind] - self.eldiag.mass[sf_inds])
                difffibmass_bpt = abs(self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass[sf_inds])
                diffz_bpt = abs(self.eldiag.z[agn_ind]-self.eldiag.z[sf_inds])/np.std(redshift_m2)
                diffs_bpt = diffsfr_bpt+diffmass_bpt+difffibmass_bpt+diffz_bpt
                diffs_bpt_sort = np.argsort(diffs_bpt)[:self.n_avg]
                avg_diffs_bpt = np.mean(diffs_bpt[diffs_bpt_sort])
                    
                diffsfr_bptplus = abs(self.eldiag.sfr[agn_ind] - self.eldiag.sfr_plus[sf_plus_inds])
                diffmass_bptplus = abs(self.eldiag.mass[agn_ind] - self.eldiag.mass_plus[sf_plus_inds])
                difffibmass_bptplus = abs(self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass_plus[sf_plus_inds])
                diffz_bptplus = abs(self.eldiag.z[agn_ind]-self.eldiag.z_plus[sf_plus_inds])/np.std(redshift_m2)
                diffs_bptplus = diffsfr_bptplus+diffmass_bptplus+difffibmass_bptplus+diffz_bptplus
                diffs_bptplus_sort = np.argsort(diffs_bptplus)[:self.n_avg]      
                avg_diffs_bptplus = np.mean(diffs_bptplus[diffs_bptplus_sort])
                
                    
                diffsfr_neither = abs(self.eldiag.sfr[agn_ind] - self.eldiag.sfr_neither)
                diffmass_neither = abs(self.eldiag.mass[agn_ind] - self.eldiag.mass_neither)
                difffibmass_neither = abs(self.eldiag.fibmass[agn_ind] - self.eldiag.fibmass_neither)
                diffz_neither = abs(self.eldiag.z[agn_ind]-self.eldiag.z_neither)/np.std(redshift_m2)
                diffs_neither = diffsfr_neither+diffmass_neither+difffibmass_neither+diffz_neither
                diffs_neither_sort = np.argsort(diffs_neither)[:self.n_avg]                        
                avg_diffs_neither = np.mean(diffs_neither[diffs_neither_sort])                      

                mindist_out = [avg_diffs_bpt, avg_diffs_bptplus, avg_diffs_neither]
                
                mindist_outagn = [avg_diffs_agn, avg_diffs_agnplus]
                mindists[:, i] = mindist_out
                mindists_agn[:, i] = mindist_outagn            

                mindist_ind = np.int(np.where(mindist_out == np.min(mindist_out))[0])
                if mindist_ind ==0:
                    agns.append(agn_ind)
                    sav = ''
                    for sf_ind in range(len(diffs_bpt_sort)-1):
                        sav+=str(sf_inds[diffs_bpt_sort[sf_ind]])+' '
                    sav+=str(sf_inds[diffs_bpt_sort[-1]])+'\n'
                    sfs_fil.write(sav)
                    self.dists_best.append(diffs_bpt[diffs_bpt_sort])
                    self.best_types.append(0)
                elif mindist_ind==1:
                    agns_plus.append(agn_ind)
                    sav = ''
                    for plus_ind in range(len(diffs_bptplus_sort)-1):
                        sav+=str(sf_plus_inds[diffs_bptplus_sort[plus_ind]])+' '
                    sav+=str(sf_plus_inds[diffs_bptplus_sort[-1]])+'\n'
                    sfs_plus_fil.write(sav)                
                    self.dists_best.append(diffs_bptplus[diffs_bptplus_sort])
                    self.best_types.append(1)
                else:
                    neither_agn.append(agn_ind)
                    sav = ''
                    for neither_ind in range(len(diffs_neither_sort)-1):
                        sav+=str(diffs_neither_sort[neither_ind])+' '
                    sav+=str(diffs_neither_sort[-1])+'\n'
                    neither_match_fil.write(sav) 
                    self.dists_best.append(diffs_neither[diffs_neither_sort])
                    self.best_types.append(2)
                mindistagn_ind = np.int(np.where(mindist_outagn==np.min(mindist_outagn))[0])
                if mindistagn_ind ==0:
                    agns_selfmatch.append(agn_ind)
                    sav = ''
                    for agn_ind in range(len(diffs_agn_sort)-1):
                        sav+=str(diffs_agn_sort[agn_ind])+' '
                    sav+=str(diffs_agn_sort[-1])+'\n'
                    selfmatchother_fil.write(sav) 
                else:
                    agnsplus_selfmatch.append(agn_ind)
                    sav = ''
                    for agnplus_ind in range(len(diffs_agnplus_sort)-1):
                        sav+=str(diffs_agnplus_sort[agnplus_ind])+' '
                    sav+=str(diffs_agnplus_sort[-1])+'\n'
                    selfmatch_agnsplusother_fil.write(sav) 
                
            selfmatchother_fil.close()
            selfmatch_agnsplusother_fil.close()
            
            sfs_fil.close()
            sfs_plus_fil.close()
            
            neither_match_fil.close()
            
            self.mindists_avg = np.copy(mindists)
            self.mindists_agn_avg = np.copy(mindists_agn)
            self.mindists_best_avg = np.min(self.mindists_avg, axis=0)
            self.mindistsagn_best_avg = np.min(self.mindists_agn_avg, axis=0)      

            np.savetxt(catfold+'sfrm/sfrm_mindists_avg' +str(self.n_avg)+'.txt',self.mindists_avg.transpose(), fmt='%8.6f')               
            np.savetxt(catfold+'sfrm/sfrm_mindists_agn_avg' +str(self.n_avg)+'.txt',self.mindists_agn_avg.transpose(), fmt='%8.6f')
            
            np.savetxt(catfold+'sfrm/sfrm_mindists_best_avg' +str(self.n_avg)+'.txt',self.mindists_best_avg, fmt='%8.6f')               
            np.savetxt(catfold+'sfrm/sfrm_mindistsagn_best_avg' +str(self.n_avg)+'.txt',self.mindistsagn_best_avg, fmt='%8.6f')
            
            np.savetxt(catfold+'sfrm/sfrm_agns_avg' +str(self.n_avg)+'.txt', np.array(agns), fmt='%6.d')
            np.savetxt(catfold+'sfrm/sfrm_agns_plus_avg' +str(self.n_avg)+'.txt', np.array(agns_plus), fmt='%6.d')
            np.savetxt(catfold+'sfrm/sfrm_neither_agn_avg' +str(self.n_avg)+'.txt', np.array(neither_agn), fmt='%6.d')

            np.savetxt(catfold+'sfrm/sfrm_agns_selfmatch_avg' +str(self.n_avg)+'.txt', np.array(agns_selfmatch), fmt='%6.d') 
            np.savetxt(catfold+'sfrm/sfrm_agnsplus_selfmatch_avg' +str(self.n_avg)+'.txt', np.array(agnsplus_selfmatch), fmt='%6.d') 
    
        self.agns_selfmatch_avg = np.loadtxt(catfold+'sfrm/sfrm_agns_selfmatch_avg' +str(self.n_avg)+'.txt', dtype=np.int64)
        self.agns_selfmatch_other_avg = np.loadtxt(catfold+'sfrm/sfrm_agns_selfmatch_other_avg' +str(self.n_avg)+'.txt', dtype=np.int64)
        
        self.agnsplus_selfmatch_avg = np.loadtxt(catfold+'sfrm/sfrm_agnsplus_selfmatch_avg' +str(self.n_avg)+'.txt', dtype=np.int64)
        self.agnsplus_selfmatch_other_avg = np.loadtxt(catfold+'sfrm/sfrm_agnsplus_selfmatch_other_avg' +str(self.n_avg)+'.txt', dtype=np.int64)
        
        self.agns_avg = np.loadtxt(catfold+'sfrm/sfrm_agns_avg' +str(self.n_avg)+'.txt', dtype=np.int64)
        self.sfs_avg = np.loadtxt(catfold+'sfrm/sfrm_sfs_avg' +str(self.n_avg)+'.txt', dtype=np.int64)
        
        self.agns_plus_avg = np.loadtxt(catfold+'sfrm/sfrm_agns_plus_avg' +str(self.n_avg)+'.txt', dtype=np.int64)
        self.sfs_plus_avg = np.loadtxt(catfold+'sfrm/sfrm_sfs_plus_avg' +str(self.n_avg)+'.txt', dtype=np.int64)

        self.neither_agn_avg = np.loadtxt(catfold+'sfrm/sfrm_neither_agn_avg' +str(self.n_avg)+'.txt', dtype=np.int64)            
        self.neither_matches_avg = np.loadtxt(catfold+'sfrm/sfrm_neither_matches_avg' +str(n_avg)+'.txt', dtype=np.int64)
        
        
        self.mindists_avg = np.loadtxt(catfold+'sfrm/sfrm_mindists_avg' +str(self.n_avg)+'.txt', unpack=True)
        self.mindists_best_avg = np.loadtxt(catfold+'sfrm/sfrm_mindists_best_avg' +str(self.n_avg)+'.txt')
        
        self.mindists_agn_avg = np.loadtxt(catfold+'sfrm/sfrm_mindists_agn_avg' +str(self.n_avg)+'.txt', unpack=True)
        self.mindistsagn_best_avg = np.loadtxt(catfold+'sfrm/sfrm_mindistsagn_best_avg' +str(self.n_avg)+'.txt')

    def subtract_elflux_avg_sep(self, sncut=2):

        self.bptdistrat_avg = (np.array(np.mean(cosmo.luminosity_distance(self.eldiag.z[self.sfs_avg]), axis=1))/np.array(cosmo.luminosity_distance(self.eldiag.z[self.agns_avg])))**2 #Mpc to cm
        self.oiiiflux_sub_avg  =  self.eldiag.oiiiflux[self.agns_avg] - np.mean(self.eldiag.oiiiflux[self.sfs_avg], axis=1)*self.bptdistrat_avg
        self.niiflux_sub_avg  = self.eldiag.niiflux[self.agns_avg] - np.mean(self.eldiag.niiflux[self.sfs_avg], axis=1)*self.bptdistrat_avg
        self.hbetaflux_sub_avg = self.eldiag.hbetaflux[self.agns_avg] - np.mean(self.eldiag.hbetaflux[self.sfs_avg],axis=1)*self.bptdistrat_avg
        self.halpflux_sub_avg = self.eldiag.halpflux[self.agns_avg] - np.mean(self.eldiag.halpflux[self.sfs_avg], axis=1)* self.bptdistrat_avg

        self.oiiiflux_sub_err_avg = np.sqrt(self.eldiag.oiii_err_bpt[self.agns_avg]**2 +(np.mean(self.eldiag.oiii_err_bpt[self.sfs_avg], axis=1)/np.sqrt(self.n_avg)*self.bptdistrat_avg)**2)
        self.niiflux_sub_err_avg = np.sqrt(self.eldiag.nii_err_bpt[self.agns_avg]**2+(np.mean(self.eldiag.nii_err_bpt[self.sfs_avg], axis=1)/np.sqrt(self.n_avg)*self.bptdistrat_avg)**2)        
        self.hbetaflux_sub_err_avg = np.sqrt(self.eldiag.hbeta_err_bpt[self.agns_avg]**2+(np.mean(self.eldiag.hbeta_err_bpt[self.sfs_avg],axis=1)/np.sqrt(self.n_avg)*self.bptdistrat_avg)**2)
        self.halpflux_sub_err_avg = np.sqrt(self.eldiag.halp_err_bpt[self.agns_avg]**2+(np.mean(self.eldiag.halp_err_bpt[self.sfs_avg], axis=1)/np.sqrt(self.n_avg)*self.bptdistrat_avg)**2)
 
        self.oiiiflux_sub_sn_avg = self.oiiiflux_sub_avg*1e17/self.oiiiflux_sub_err_avg
        self.niiflux_sub_sn_avg = self.niiflux_sub_avg*1e17/self.niiflux_sub_err_avg
        self.hbetaflux_sub_sn_avg = self.hbetaflux_sub_avg*1e17/self.hbetaflux_sub_err_avg
        self.halpflux_sub_sn_avg = self.halpflux_sub_avg*1e17/self.halpflux_sub_err_avg   
        
        self.bpt_sn_filt_avg_bool = ((self.oiiiflux_sub_sn_avg > sncut)& (self.niiflux_sub_sn_avg > sncut)& 
                                    (self.hbetaflux_sub_sn_avg > sncut)& (self.halpflux_sub_sn_avg > sncut))
        self.bpt_sn_filt_avg_bool_intermed = ((self.oiiiflux_sub_sn_avg > sncut-1)& (self.niiflux_sub_sn_avg > sncut-1)& 
                                    (self.hbetaflux_sub_sn_avg > sncut-1)& (self.halpflux_sub_sn_avg > sncut-1)&
                                    (self.oiiiflux_sub_sn_avg < sncut)& (self.niiflux_sub_sn_avg < sncut)& 
                                    (self.hbetaflux_sub_sn_avg < sncut)& (self.halpflux_sub_sn_avg < sncut))      
        
        self.bpt_sn_filt_avg = np.where(self.bpt_sn_filt_avg_bool)[0]
        self.bpt_sn_filt_avg_intermed = np.where(self.bpt_sn_filt_avg_bool_intermed)

        self.bpt_not_sn_filt_avg_bool = np.logical_not(self.bpt_sn_filt_avg_bool)
        self.bpt_not_sn_filt_avg = np.where(self.bpt_not_sn_filt_avg_bool)        

        self.av_sub_avg = self.eldiag.extinction(ha=self.halpflux_sub_avg, hb=self.hbetaflux_sub_avg)
        self.av_agn_avg = self.eldiag.av[self.agns_avg]
        self.av_sf_avg = self.eldiag.av[self.sfs_avg]
        
        self.oiiiflux_sub_dered_avg = self.eldiag.dustcorrect(self.oiiiflux_sub_avg, av=self.av_sub_avg)
        self.z_avg = np.copy(self.eldiag.z[self.agns_avg]) 
        self.mass_avg = np.copy(self.eldiag.mass[self.agns_avg])
        self.ssfr_avg = np.copy(self.eldiag.ssfr[self.agns_avg])
        
        self.oiiilum_sub_dered_avg = getlumfromflux(self.oiiiflux_sub_dered_avg, self.z_avg)
        
        self.log_oiii_hbeta_sub_avg = np.log10(self.oiiiflux_sub_avg/self.hbetaflux_sub_avg)
        self.log_nii_halpha_sub_avg = np.log10(self.niiflux_sub_avg/self.halpflux_sub_avg)

        self.bptdistrat_plus_avg = (np.array(np.mean(cosmo.luminosity_distance(self.eldiag.z_plus[self.sfs_plus_avg]), axis=1 ))/np.array(cosmo.luminosity_distance(self.eldiag.z[self.agns_plus_avg])))**2 #Mpc to cm

        
        self.oiiiflux_sub_plus_avg = self.eldiag.oiiiflux[self.agns_plus_avg] - np.mean(self.eldiag.oiiiflux_plus[self.sfs_plus_avg],axis=1)*self.bptdistrat_plus_avg
        self.niiflux_sub_plus_avg = self.eldiag.niiflux[self.agns_plus_avg] - np.mean(self.eldiag.niiflux_plus[self.sfs_plus_avg], axis=1)*self.bptdistrat_plus_avg
        self.hbetaflux_sub_plus_avg = self.eldiag.hbetaflux[self.agns_plus_avg] - np.mean(self.eldiag.hbetaflux_plus[self.sfs_plus_avg], axis=1)*self.bptdistrat_plus_avg
        self.halpflux_sub_plus_avg = self.eldiag.halpflux[self.agns_plus_avg] - np.mean(self.eldiag.halpflux_plus[self.sfs_plus_avg], axis=1)*self.bptdistrat_plus_avg
        
        self.oiiiflux_sub_err_plus_avg = np.sqrt(self.eldiag.oiii_err_bpt[self.agns_plus_avg]**2 +(np.mean(self.eldiag.oiii_err_plus[self.sfs_plus_avg], axis=1)/np.sqrt(self.n_avg)*self.bptdistrat_plus_avg)**2)
        self.niiflux_sub_err_plus_avg = np.sqrt(self.eldiag.nii_err_bpt[self.agns_plus_avg]**2+(np.mean(self.eldiag.nii_err_plus[self.sfs_plus_avg], axis=1)/np.sqrt(self.n_avg)*self.bptdistrat_plus_avg)**2)        
        self.hbetaflux_sub_err_plus_avg = np.sqrt(self.eldiag.hbeta_err_bpt[self.agns_plus_avg]**2+(np.mean(self.eldiag.hbeta_err_plus[self.sfs_plus_avg], axis=1)/np.sqrt(self.n_avg)*self.bptdistrat_plus_avg)**2)
        self.halpflux_sub_err_plus_avg = np.sqrt(self.eldiag.halp_err_bpt[self.agns_plus_avg]**2+(np.mean(self.eldiag.halp_err_plus[self.sfs_plus_avg], axis=1)/np.sqrt(self.n_avg)*self.bptdistrat_plus_avg)**2)

        self.oiiiflux_sub_sn_plus_avg = self.oiiiflux_sub_plus_avg*1e17/self.oiiiflux_sub_err_plus_avg
        self.niiflux_sub_sn_plus_avg = self.niiflux_sub_plus_avg*1e17/self.niiflux_sub_err_plus_avg
        self.hbetaflux_sub_sn_plus_avg = self.hbetaflux_sub_plus_avg*1e17/self.hbetaflux_sub_err_plus_avg
        self.halpflux_sub_sn_plus_avg = self.halpflux_sub_plus_avg*1e17/self.halpflux_sub_err_plus_avg
        self.bpt_plus_sn_filt_avg = np.where((self.oiiiflux_sub_sn_plus_avg > sncut)& (self.niiflux_sub_sn_plus_avg > sncut)& 
                                    (self.hbetaflux_sub_sn_plus_avg > sncut)& (self.halpflux_sub_sn_plus_avg > sncut))[0]
        self.av_plus_sf_avg = self.eldiag.av_plus[self.sfs_plus_avg]
        self.av_plus_agn_avg = self.eldiag.av[self.agns_plus_avg]
        self.av_sub_plus_avg = self.eldiag.extinction(ha=self.halpflux_sub_plus_avg, hb=self.hbetaflux_sub_plus_avg)
        
        self.oiiiflux_sub_plus_dered_avg = self.eldiag.dustcorrect(self.oiiiflux_sub_plus_avg, av=self.av_sub_plus_avg)
        self.z_plus_avg = np.copy(self.eldiag.z[self.agns_plus_avg])
        self.mass_plus_avg = np.copy(self.eldiag.mass[self.agns_plus_avg])
        self.ssfr_plus_avg = np.copy(self.eldiag.ssfr[self.agns_plus_avg])
        
        self.oiiilum_sub_plus_dered_avg = getlumfromflux(self.oiiiflux_sub_plus_dered_avg, self.z_plus_avg)
        
        self.log_oiii_hbeta_sub_plus_avg = np.log10(self.oiiiflux_sub_plus_avg/self.hbetaflux_sub_plus_avg)
        self.log_nii_halpha_sub_plus_avg = np.log10(self.niiflux_sub_plus_avg/self.halpflux_sub_plus_avg)

        self.bptdistrat_neither_avg = (np.array(np.mean(cosmo.luminosity_distance(self.eldiag.z_neither[self.neither_matches_avg]), axis=1))/np.array(cosmo.luminosity_distance(self.eldiag.z[self.neither_agn_avg])))**2 #Mpc to cm

        self.oiiiflux_sub_neither_avg = self.eldiag.oiiiflux[self.neither_agn_avg] - np.mean(self.eldiag.oiiiflux_neither[self.neither_matches_avg], axis=1)*self.bptdistrat_neither_avg
        self.niiflux_sub_neither_avg = self.eldiag.niiflux[self.neither_agn_avg]- np.mean(self.eldiag.niiflux_neither[self.neither_matches_avg], axis=1)*self.bptdistrat_neither_avg
        self.hbetaflux_sub_neither_avg = self.eldiag.hbetaflux[self.neither_agn_avg]- np.mean(self.eldiag.hbetaflux_neither[self.neither_matches_avg], axis=1)*self.bptdistrat_neither_avg
        self.halpflux_sub_neither_avg = self.eldiag.halpflux[self.neither_agn_avg]- np.mean(self.eldiag.halpflux_neither[self.neither_matches_avg], axis=1)*self.bptdistrat_neither_avg

        self.oiiiflux_sub_err_neither_avg = np.sqrt(self.eldiag.oiii_err_bpt[self.neither_agn_avg]**2 +(np.mean(self.eldiag.oiii_err_neither[self.neither_matches_avg], axis=1)/np.sqrt(self.n_avg)*self.bptdistrat_neither_avg)**2)
        self.niiflux_sub_err_neither_avg = np.sqrt(self.eldiag.nii_err_bpt[self.neither_agn_avg]**2+(np.mean(self.eldiag.nii_err_neither[self.neither_matches_avg], axis=1)/np.sqrt(self.n_avg)*self.bptdistrat_neither_avg)**2)        
        self.hbetaflux_sub_err_neither_avg = np.sqrt(self.eldiag.hbeta_err_bpt[self.neither_agn_avg]**2+(np.mean(self.eldiag.hbeta_err_neither[self.neither_matches_avg], axis=1)/np.sqrt(self.n_avg)*self.bptdistrat_neither_avg)**2)
        self.halpflux_sub_err_neither_avg = np.sqrt(self.eldiag.halp_err_bpt[self.neither_agn_avg]**2+(np.mean(self.eldiag.halp_err_neither[self.neither_matches_avg], axis=1)/np.sqrt(self.n_avg)*self.bptdistrat_neither_avg)**2)
 
        self.oiiiflux_sub_sn_neither_avg = self.oiiiflux_sub_neither_avg*1e17/self.oiiiflux_sub_err_neither_avg
        self.niiflux_sub_sn_neither_avg = self.niiflux_sub_neither_avg*1e17/self.niiflux_sub_err_neither_avg
        self.hbetaflux_sub_sn_neither_avg = self.hbetaflux_sub_neither_avg*1e17/self.hbetaflux_sub_err_neither_avg
        self.halpflux_sub_sn_neither_avg = self.halpflux_sub_neither_avg*1e17/self.halpflux_sub_err_neither_avg
        
        self.bpt_neither_sn_filt_avg = np.where((self.oiiiflux_sub_sn_neither_avg > sncut)& (self.niiflux_sub_sn_neither_avg > sncut)& 
                                    (self.hbetaflux_sub_sn_neither_avg > sncut)& (self.halpflux_sub_sn_neither_avg > sncut))[0]        
                              
        self.av_sub_neither_avg = self.eldiag.extinction(ha=self.halpflux_sub_neither_avg, hb=self.hbetaflux_sub_neither_avg)
        self.av_neither_sf_avg = self.eldiag.av_neither[self.neither_matches_avg]
        self.av_neither_agn_avg = self.eldiag.av[self.neither_agn_avg]
        
        self.oiiiflux_sub_neither_dered_avg = self.eldiag.dustcorrect(self.oiiiflux_sub_neither_avg, av=self.av_sub_neither_avg)
        self.z_neither_avg = np.copy(self.eldiag.z[self.neither_agn_avg])
        self.mass_neither_avg = np.copy(self.eldiag.mass[self.neither_agn_avg])
        self.ssfr_neither_avg = np.copy(self.eldiag.ssfr[self.neither_agn_avg])
        
        self.oiiilum_sub_neither_dered_avg = getlumfromflux(self.oiiiflux_sub_neither_dered_avg, self.z_neither_avg)
        
        self.log_oiii_hbeta_sub_neither_avg = np.log10(self.oiiiflux_sub_neither_avg/self.hbetaflux_sub_neither_avg)
        self.log_nii_halpha_sub_neither_avg = np.log10(self.niiflux_sub_neither_avg/self.halpflux_sub_neither_avg)