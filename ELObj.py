import numpy as np
from ast_func import *
from demarcations import *
class ELObj:
    def __init__(self, sdssinds, sdss, make_spec, gswcat, gsw = False, xr=False):
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
        self.oII_sn = np.reshape(sdss.alloII[sdssinds]/sdss.alloIIerr[sdssinds],-1)
        #BPT filt used for everything
        self.bpt_sn_filt_bool = (self.halp_sn>2) & (self.hbeta_sn > 2) & (self.oiii_sn > 2) & (self.nii_sn > 2)
        self.bpt_sn_filt = np.where(self.bpt_sn_filt_bool)[0]
        
        self.not_bpt_sn_filt_bool  = np.logical_not(self.bpt_sn_filt_bool)

        self.halp_nii_filt_bool = ( (self.halp_sn > 2) & (self.nii_sn > 2) &( (self.oiii_sn<=2) | (self.hbeta_sn<=2) ) )
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

        if gsw:
            self.allmass = gswcat.allmass[make_spec]
            self.allsfr =gswcat.allsfr[make_spec]
            self.allz =  gswcat.allz[make_spec]
            self.mass = gswcat.allmass[make_spec][self.bpt_sn_filt]
            self.mass_plus = gswcat.allmass[make_spec][self.halp_nii_filt]
            self.mass_neither = gswcat.allmass[make_spec][self.neither_filt]

            self.weakmass = gswcat.allmass[make_spec][self.not_bpt_sn_filt]
            
            self.sfr =gswcat.allsfr[make_spec][self.bpt_sn_filt]
            self.sfr_plus = gswcat.allsfr[make_spec][self.halp_nii_filt]
            self.sfr_neither = gswcat.allsfr[make_spec][self.neither_filt]
            
            self.weaksfr = gswcat.allsfr[make_spec][self.not_bpt_sn_filt]
            self.ssfr= self.sfr-self.mass
            self.weakssfr = self.weaksfr-self.weakmass
            self.z =  gswcat.allz[make_spec][self.bpt_sn_filt]
            self.z_plus =  gswcat.allz[make_spec][self.halp_nii_filt]
            self.z_neither =  gswcat.allz[make_spec][self.neither_filt]
            
            self.tbt_filtall = np.where((self.neIII_sn>5)&(self.oII_sn >5  ) &
                                 ( gswcat.allz[make_spec]>0.02))[0]   
            self.ssfrtbt = self.allsfr-self.allmass 
            self.ids = gswcat.allids[make_spec][self.bpt_sn_filt]
            self.ids_plus= gswcat.allids[make_spec][self.halp_nii_filt]
            self.ids_neither = gswcat.allids[make_spec][self.neither_filt]
            
            self.ra = gswcat.allra[make_spec][self.bpt_sn_filt]
            self.dec = gswcat.alldec[make_spec][self.bpt_sn_filt]
            
        else:
            self.allmass = gswcat.matchmass[make_spec]
            self.allsfr =gswcat.matchsfr[make_spec]
            self.allz =  gswcat.allz[make_spec]

            self.mass = gswcat.matchmass[make_spec][self.bpt_sn_filt]
            self.mass_plus = gswcat.matchmass[make_spec][self.halp_nii_filt]
            self.mass_neither = gswcat.matchmass[make_spec][self.neither_filt]

            self.weakmass = gswcat.matchmass[make_spec][self.not_bpt_sn_filt]
            self.sfr = gswcat.matchsfr[make_spec][self.bpt_sn_filt]
            self.sfr_plus = gswcat.matchsfr[make_spec][self.halp_nii_filt]

            self.weaksfr = gswcat.matchsfr[make_spec][self.not_bpt_sn_filt]
            self.ssfr = self.sfr-self.mass
            self.weakssfr = self.weaksfr-self.weakmass
            self.z =  gswcat.z[make_spec][self.bpt_sn_filt]
            self.weakz = gswcat.z[make_spec][self.not_bpt_sn_filt]
            self.distances=np.array(cosmo.luminosity_distance(self.z))* (3.086e+24) #Mpc to cm
            if len(self.weakz) !=0:
                self.weakdistances = np.array(cosmo.luminosity_distance(self.weakz))*(3.086e+24)
            self.ra = gswcat.matchra[make_spec][self.bpt_sn_filt]
            self.dec = gswcat.matchdec[make_spec][self.bpt_sn_filt]
            self.xrayra = gswcat.matchxrayra[make_spec][self.bpt_sn_filt]
            self.xraydec = gswcat.matchxraydec[make_spec][self.bpt_sn_filt]
            self.xrayfulllum = gswcat.fulllumsrf[make_spec][self.bpt_sn_filt]
            self.ids = gswcat.ids[make_spec][self.bpt_sn_filt]
            self.tbt_filtall = np.where((self.neIII_sn>5)&(self.oII_sn >5  ) &
                                 ( gswcat.z[make_spec]>0.02))[0]   
            self.ssfrtbt = self.allsfr-self.allmass
            
        if xr:
            self.exptimes = gswcat.exptimes[make_spec][self.bpt_sn_filt]
        self.yvals_bpt = sdss.alloIII[sdssinds]/sdss.allhbeta[sdssinds]
        self.xvals1_bpt = sdss.allnII[sdssinds]/sdss.allhalpha[sdssinds]
        self.xvals2_bpt = sdss.allSII[sdssinds]/sdss.allhalpha[sdssinds]
        self.xvals3_bpt = sdss.alloI[sdssinds]/sdss.allhalpha[sdssinds]
        
        self.tbtx = sdss.allneIII[sdssinds]/sdss.alloII[sdssinds]
        self.tbt_filt = np.where((self.neIII_sn[self.bpt_sn_filt]>5)&(self.oII_sn[self.bpt_sn_filt] >5  ) &
                                 (self.z>0.02))[0]      
   
        
        self.neiiioii = np.log10(np.copy(self.tbtx))[self.bpt_sn_filt]
        
        self.niiha = np.log10(np.copy(self.xvals1_bpt))[self.bpt_sn_filt]
        self.niihaplus = np.log10(np.copy(self.xvals1_bpt))[self.halp_nii_filt]
        self.siiha = np.log10(np.copy(self.xvals2_bpt))[self.bpt_sn_filt][self.vo87_1_filt]
        self.oiha = np.log10(np.copy(self.xvals3_bpt))[self.bpt_sn_filt][self.vo87_2_filt]
        self.oiiihb = np.log10(np.copy(self.yvals_bpt))[self.bpt_sn_filt]

        self.massfrac = np.copy(10**(sdss.all_fibmass[self.sdss_filt]))/np.copy(10**(sdss.all_sdss_avgmasses[self.sdss_filt]))
        self.massfrac_plus = np.copy(10**(sdss.all_fibmass[self.sdss_filt_plus]))/np.copy(10**(sdss.all_sdss_avgmasses[self.sdss_filt_plus]))
        self.massfrac_neither = np.copy(10**(sdss.all_fibmass[self.sdss_filt_neither]))/np.copy(10**(sdss.all_sdss_avgmasses[self.sdss_filt_neither]))

        self.massfracgsw = np.copy(10**(sdss.all_fibmass[self.sdss_filt]))/np.copy(10**(self.mass))
        self.massfracgsw_plus = np.copy(10**(sdss.all_fibmass[self.sdss_filt_plus]))/np.copy(10**(self.mass_plus))
        self.massfracgsw_neither = np.copy(10**(sdss.all_fibmass[self.sdss_filt_neither]))/np.copy(10**(self.mass_neither))

        self.weakmassfracgsw = np.copy(10**(sdss.all_fibmass[self.sdss_filt_weak]))/np.copy(10**(self.weakmass))

        self.weakmassfrac = np.copy(10**(sdss.all_fibmass[self.sdss_filt_weak])/np.copy(10**sdss.all_sdss_avgmasses[self.sdss_filt_weak]))
        self.ohabund = np.reshape(sdss.fiboh[self.sdss_filt],-1)

        self.oiiiflux = np.copy(sdss.alloIII[self.sdss_filt])/1e17
        self.hbetaflux = np.copy(sdss.allhbeta[self.sdss_filt])/1e17
        self.halpflux = np.copy(sdss.allhalpha[self.sdss_filt])/1e17
        self.niiflux = np.copy(sdss.allnII[self.sdss_filt])/1e17

        self.oiiiflux_plus = np.copy(sdss.alloIII[self.sdss_filt_plus])/1e17
        self.hbetaflux_plus = np.copy(sdss.allhbeta[self.sdss_filt_plus])/1e17
        self.halpflux_plus = np.copy(sdss.allhalpha[self.sdss_filt_plus])/1e17
        self.niiflux_plus = np.copy(sdss.allnII[self.sdss_filt_plus])/1e17
        
        self.oiiiflux_neither = np.copy(sdss.alloIII[self.sdss_filt_neither])/1e17
        self.hbetaflux_neither = np.copy(sdss.allhbeta[self.sdss_filt_neither])/1e17
        self.halpflux_neither = np.copy(sdss.allhalpha[self.sdss_filt_neither])/1e17
        self.niiflux_neither = np.copy(sdss.allnII[self.sdss_filt_neither])/1e17
 
        self.oiii_err_bpt = np.copy(sdss.alloIII_err[self.sdss_filt])
        self.hbeta_err_bpt = np.copy(sdss.allhbeta_err[self.sdss_filt])
        self.halp_err_bpt= np.copy(sdss.allhalpha_err[self.sdss_filt])
        self.nii_err_bpt = np.copy(sdss.allnII_err[self.sdss_filt])

        self.oiii_err_plus = np.copy(sdss.alloIII_err[self.sdss_filt_plus])
        self.hbeta_err_plus = np.copy(sdss.allhbeta_err[self.sdss_filt_plus])
        self.halp_err_plus = np.copy(sdss.allhalpha_err[self.sdss_filt_plus])
        self.nii_err_plus = np.copy(sdss.allnII_err[self.sdss_filt_plus])
        
        self.oiii_err_neither = np.copy(sdss.alloIII_err[self.sdss_filt_neither])
        self.hbeta_err_neither = np.copy(sdss.allhalpha_err[self.sdss_filt_neither])
        self.halp_err_neither = np.copy(sdss.allhalpha_err[self.sdss_filt_neither])
        self.nii_err_neither = np.copy(sdss.allnII[self.sdss_filt_neither])
                

               
        self.av = self.extinction()
        self.av_plus = self.extinction(ha = self.halpflux_plus, hb = self.hbetaflux_plus)
        self.av_neither = self.extinction(ha = self.halpflux_neither, hb = self.hbetaflux_neither)
        
        
        self.halpflux_corr = self.dustcorrect(self.halpflux)
        self.oiiiflux_corr = self.dustcorrect(self.oiiiflux)
        self.vdisp = np.copy(sdss.allvdisp[self.sdss_filt])
        self.balmerfwhm = np.copy(sdss.allbalmerdisp[self.sdss_filt]*2 * np.sqrt(2*np.log(2)))
        self.forbiddenfwhm = np.copy(sdss.allforbiddendisp[self.sdss_filt]*2 * np.sqrt(2*np.log(2)))
        self.forbiddenfwhmerr = np.copy(sdss.allforbiddendisperr[self.sdss_filt]*2*np.sqrt(2*np.log(2)))
        self.balmerfwhmerr = np.copy(sdss.allbalmerdisperr[self.sdss_filt]*2*np.sqrt(2*np.log(2)))

        self.gswfilt = self.bpt_sn_filt
        self.oiiilum = getlumfromflux(self.oiiiflux_corr,self.z)
        self.halplum = getlumfromflux(self.halpflux_corr, self.z)
        self.oiiilum_uncorr = getlumfromflux(self.oiiiflux,self.z)
        self.halplum_uncorr = getlumfromflux(self.halpflux, self.z) #units from mpa/jhu

        self.fibmass = sdss.all_fibmass[self.sdss_filt]
        self.fibmass_plus = sdss.all_fibmass[self.sdss_filt_plus]
        self.fibmass_neither = sdss.all_fibmass[self.sdss_filt_neither]

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
            xvals = np.copy(np.log10(self.xvals1_bpt[filt]))
            yvals = np.copy(np.log10(self.yvals_bpt[filt]))
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
    def get_bptplus_groups(self, filt=[]):
        groups =[]
        if len(filt) != 0:
            xvals = np.copy(np.log10(self.xvals1_bpt[filt]))
            yvals = np.copy(np.log10(self.yvals_bpt[filt]))
        else:
            xvals = np.copy(self.niiha)
            yvals = np.copy(self.oiiihb)
        for i in range(len(xvals)):
            if xvals[i] < -0.4:
                if yvals[i] < np.log10(y1_kauffman(xvals[i])):
                    groups.append('HII')
                else:
                    groups.append('AGN')
            else:
                groups.append('AGN')
        groups=np.array(groups)
        agn = np.where(groups == 'AGN')[0]
        nonagn = np.where(groups == 'HII')[0]
        return groups,nonagn, agn
    def get_bptplus_niigroups(self, filt=[]):
        groups =[]
        if len(filt) != 0:
            xvals = np.copy(np.log10(self.xvals1_bpt[filt]))
        else:
            xvals = np.copy(self.niihaplus)
        for i in range(len(xvals)):
            if xvals[i] < -0.4:
                groups.append('HII')
            else:
                groups.append('AGN')
        groups=np.array(groups)
        agn = np.where(groups == 'AGN')[0]
        nonagn = np.where(groups == 'HII')[0]
        return groups,nonagn, agn
    
    def extinction(self, ha=[], hb=[]):
        if len(ha) == 0  and len(hb) == 0:
            ha = self.halpflux
            hb = self.hbetaflux
        av = 7.23*np.log10((ha/hb) / 2.86) # A(V) mag
        return av
    def dustcorrect(self,flux, av=[]):
        if len(av)==0:
            av=self.av
        return flux*10**(0.4*av*1.120)
    def halptofibsfr_corr(self):
        logsfrfib = np.log10(7.9e-42)+np.log10(self.halplum)-0.24#+0.4*A(H_alpha)
        return logsfrfib
    def halptofibsfr_uncorr(self):
        logsfrfib = np.log10(7.9e-42)+np.log10(self.halplum_uncorr)-0.24+0.4*self.av
        return logsfrfib
