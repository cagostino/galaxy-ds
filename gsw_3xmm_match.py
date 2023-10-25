import numpy as np
from ast_func import *
import pandas as pd
raind = 0
decind = 1
sfrind = 2
massind = 3
plateind = 4
fiberind = 5
mjdind = 6
sedind = 7
av_ind = 8
av_err_ind = 9
a_uv_ind=10
sfrerrorind = 11
masserrorind = 12

sigma1_ind=13
nyuenv_ind=14
baldenv_ind=15
irx_ind=16
axisrat_ind =17
nuv_ind=18
fuv_ind=19

class GSWCat:
    def __init__(self, goodinds, sfrplus, sedflag=0):
        self.inds = goodinds
        self.sedfilt = np.where(sfrplus['flag_sed'][self.inds]==sedflag)[0]
        self.gsw_df = sfrplus.iloc[self.inds].iloc[self.sedfilt]
        
        

class GSWCatmatch3xmm(GSWCat):
    def __init__(self, goodinds, sfrplus, xrayflag, fullflux_filt, 
                 efullflux_filt ,hardflux_filt, ehardflux_filt, hardflux2_filt, softflux_filt, esoftflux_filt, ext, hr1, hr2, hr3, hr4, sedflag=0):
        super().__init__(goodinds, sfrplus, sedflag=sedflag)
        
        self.gsw_df['softflux'] = softflux_filt[self.sedfilt]
        self.gsw_df['hardflux'] = hardflux_filt[self.sedfilt]
        self.gsw_df['fullflux'] = fullflux_filt[self.sedfilt]        
        self.gsw_df['esoftflux'] = esoftflux_filt[self.sedfilt]
        self.gsw_df['ehardflux'] = ehardflux_filt[self.sedfilt]

        self.gsw_df['efullflux'] = efullflux_filt[self.sedfilt]
        self.gsw_df['fullflux_sn'] = self.gsw_df.fullflux/self.gsw_df.efullflux
        self.gsw_df['hardflux_sn'] = self.gsw_df.hardflux/self.gsw_df.ehardflux
        self.gsw_df['softflux_sn'] = self.gsw_df.softflux/self.gsw_df.esoftflux
        
        self.gsw_df['softlums'] =getlumfromflux(softflux_filt[self.sedfilt], self.z)
        self.gsw_df['hardlums'] =getlumfromflux(hardflux_filt[self.sedfilt], self.z)
        self.gsw_df['hardlums2'] =getlumfromflux(hardflux2_filt[self.sedfilt], self.z)
        self.gsw_df['fulllums'] = getlumfromflux(fullflux_filt[self.sedfilt], self.z)
        self.gsw_df['efulllums_up'] =  getlumfromflux(fullflux_filt[self.sedfilt]+efullflux_filt[self.sedfilt],self.z)
        self.gsw_df['efulllums_down'] = getlumfromflux(fullflux_filt[self.sedfilt]-efullflux_filt[self.sedfilt],self.z)
        self.gsw_df['softlumsrf'] = np.log10(self.gsw_df.softlums*(1+self.z)**(1.7-2))
        self.gsw_df['hardlumsrf'] = np.log10(self.gsw_df.hardlums*(1+self.z)**(1.7-2))
        self.gsw_df['ehardlums_up'] =  getlumfromflux(hardflux_filt[self.sedfilt]+ehardflux_filt[self.sedfilt],self.z)
        self.gsw_df['ehardlums_down'] = getlumfromflux(hardflux_filt[self.sedfilt]-ehardflux_filt[self.sedfilt],self.z)

        self.gsw_df['hardlums2rf'] = np.log10(self.gsw_df.hardlums2*(1+self.z)**(1.7-2))
        self.gsw_df['hr1'] = hr1[self.sedfilt]
        self.gsw_df['hr2'] = hr2[self.sedfilt]
        self.gsw_df['hr3'] = hr3[self.sedfilt]
        self.gsw_df['hr4'] = hr4[self.sedfilt]
        
        self.gsw_df['fulllumsrf'] = np.log10(self.gsw_df.fulllums*(1+self.z)**(1.7-2))
        self.gsw_df['efulllumsrf_down'] = np.log10(self.gsw_df.efulllums_down*(1+self.z)**(1.7-2))
        self.gsw_df['efulllumsrf_up'] = np.log10(self.gsw_df.efulllums_up*(1+self.z)**(1.7-2))
        
        self.gsw_df['ehardlumsrf_down'] = np.log10(self.gsw_df.ehardlums_down*(1+self.z)**(1.7-2))
        self.gsw_df['ehardlumsrf_up'] = np.log10(self.gsw_df.ehardlums_up*(1+self.z)**(1.7-2))
        
        self.ext = ext[self.sedfilt]
        self.xrayflag = xrayflag[self.sedfilt]
        
class GSWCatmatch_CSC(GSWCat):
    def __init__(self, goodinds, sfrplus,  fullflux_filt, 
                  hardflux_filt, softflux_filt, sedflag=0): #, ext, hr1, hr2, hr3, hr4,efullflux_filtxrayflag
        super().__init__(goodinds, sfrplus, sedflag=sedflag)
        
        self.gsw_df['softflux'] = softflux_filt[self.sedfilt]
        self.gsw_df['hardflux'] = hardflux_filt[self.sedfilt]
        self.gsw_df['fullflux'] = fullflux_filt[self.sedfilt]
        
        #self.efullflux = efullflux_filt[self.sedfilt]
        
        self.gsw_df['softlums'] =getlumfromflux(softflux_filt[self.sedfilt], self.z)
        self.gsw_df['hardlums'] =getlumfromflux(hardflux_filt[self.sedfilt], self.z)
        #self.hardlums2 =getlumfromflux(hardflux2_filt[self.sedfilt], self.z)
        self.gsw_df['fulllums'] = getlumfromflux(fullflux_filt[self.sedfilt], self.z)
        #self.efulllums_up =  getlumfromflux(fullflux_filt[self.sedfilt]+efullflux_filt[self.sedfilt],self.z)
        #self.efulllums_down = getlumfromflux(fullflux_filt[self.sedfilt]-efullflux_filt[self.sedfilt],self.z)
        self.gsw_df['softlumsrf'] = np.log10(self.gsw_df.softlums*(1+self.z)**(1.7-2))
        self.gsw_df['hardlumsrf'] = np.log10(self.gsw_df.hardlums*(1+self.z)**(1.7-2))
        #self.hardlums2rf = np.log10(self.hardlums2*(1+self.z)**(1.7-2))
        '''self.hr1 = hr1[self.sedfilt]
        self.hr2 = hr2[self.sedfilt]
        self.hr3 = hr3[self.sedfilt]
        self.hr4 = hr4[self.sedfilt]
        
        self.xrayflag = xrayflag[self.sedfilt]
        '''
        self.gsw_df['fulllumsrf'] = np.log10(self.gsw_df.fulllums*(1+self.z)**(1.7-2))
        '''
        self.efulllumsrf_down = np.log10(self.efulllums_down*(1+self.z)**(1.7-2))
        self.efulllumsrf_up = np.log10(self.efulllums_up*(1+self.z)**(1.7-2))
        self.ext = ext[self.sedfilt]
        '''
        
class GSWCatmatch_radio(GSWCat):
    def __init__(self, goodinds,  sfrplus,  nvss_flux,
            first_flux , wenss_flux ,vlss_flux, sedflag=0): 
        super().__init__(goodinds, sfrplus, sedflag=sedflag)
        
        self.gsw_df['nvss_flux'] = nvss_flux[self.sedfilt]
        self.gsw_df['first_flux'] = first_flux[self.sedfilt]
        self.gsw_df['wenss_flux'] = wenss_flux[self.sedfilt]
        self.gsw_df['vlss_flux'] = vlss_flux[self.sedfilt]        

        self.gsw_df['nvss_lums'] =getlumfromflux(nvss_flux[self.sedfilt], self.z)
        self.gsw_df['firstlums'] =getlumfromflux(first_flux[self.sedfilt], self.z)
        self.gsw_df['wensslums'] = getlumfromflux(wenss_flux[self.sedfilt], self.z)
        self.gsw_df['vlsslums'] = getlumfromflux(vlss_flux[self.sedfilt], self.z)
        
