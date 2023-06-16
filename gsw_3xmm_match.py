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

sigma1_ind=11
nyuenv_ind=12
baldenv_ind=13
irx_ind=14
axisrat_ind =15
nuv_ind=16
fuv_ind=17

class GSWCat:
    def __init__(self, goodinds, gswlcids, redshift, sfrplus, sedflag=0):
        self.inds = goodinds
        self.sedfilt = np.where(sfrplus[sedind][self.inds]==sedflag)[0]
        self.z = redshift[self.inds][self.sedfilt]
        self.sedflags = sfrplus[sedind][self.inds][self.sedfilt]
        self.gids = gswlcids[1][self.inds][self.sedfilt]
        self.ids = gswlcids[0][self.inds][self.sedfilt]
        self.ra =sfrplus[raind][self.inds][self.sedfilt]
        self.dec = sfrplus[decind][self.inds][self.sedfilt]

        self.allra =sfrplus[raind]
        self.alldec = sfrplus[decind]

        self.sfr=sfrplus[sfrind][self.inds][self.sedfilt]
        self.mass = sfrplus[massind][self.inds][self.sedfilt]
        self.plate = sfrplus[plateind][self.inds][self.sedfilt]
        self.fiber = sfrplus[fiberind][self.inds][self.sedfilt]
        self.mjd = sfrplus[mjdind][self.inds][self.sedfilt]        
        self.av = sfrplus[av_ind][self.inds][self.sedfilt]
        self.av_err = sfrplus[av_err_ind][self.inds][self.sedfilt]
        self.a_uv = sfrplus[a_uv_ind][self.inds][self.sedfilt]
        
        self.sigma1 = sfrplus[sigma1_ind][self.inds][self.sedfilt]
        self.nyuenv = sfrplus[nyuenv_ind][self.inds][self.sedfilt]
        self.baldenv = sfrplus[baldenv_ind][self.inds][self.sedfilt]
        self.irx = sfrplus[irx_ind][self.inds][self.sedfilt]
        self.axisrat = sfrplus[axisrat_ind][self.inds][self.sedfilt]
        self.nuv = sfrplus[nuv_ind][self.inds][self.sedfilt]
        self.fuv = sfrplus[fuv_ind][self.inds][self.sedfilt]
        
        self.gsw_dict = {'z':self.z,
                         'sedflags':self.sedflags,
                         'ids':self.ids,
                         'ra': self.ra,
                         'dec': self.dec,
                         'sfr':self.sfr,
                         'mass':self.mass,
                         'plate':self.plate,
                         'fiber': self.fiber,
                         'mjd':self.mjd,
                         'av':self.av,
                         'sigma1':self.sigma1,
                         'baldenv':self.baldenv,
                         'nyuenv':self.nyuenv,
                         'irx':self.irx,
                         'axisrat':self.axisrat,
                         'av_err':self.av_err,
                         'a_uv':self.a_uv,
                         'nuv':self.nuv,
                         'fuv':self.fuv
                         }
        self.gsw_df = pd.DataFrame(self.gsw_dict)
        

class GSWCatmatch3xmm(GSWCat):
    def __init__(self, goodinds, gswlcids, redshift, sfrplus, xrayflag, fullflux_filt, 
                 efullflux_filt ,hardflux_filt, ehardflux_filt, hardflux2_filt, softflux_filt, esoftflux_filt, ext, hr1, hr2, hr3, hr4, sedflag=0):
        super().__init__(goodinds, gswlcids, redshift, sfrplus, sedflag=sedflag)
        
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
    def __init__(self, goodinds, gswlcids, redshift, sfrplus,  fullflux_filt, 
                  hardflux_filt, softflux_filt, sedflag=0): #, ext, hr1, hr2, hr3, hr4,efullflux_filtxrayflag
        super().__init__(goodinds, gswlcids, redshift, sfrplus, sedflag=sedflag)
        
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
    def __init__(self, goodinds, gswlcids, redshift, sfrplus,  nvss_flux,
            first_flux , wenss_flux ,vlss_flux, sedflag=0): 
        super().__init__(goodinds, gswlcids, redshift, sfrplus, sedflag=sedflag)
        
        self.gsw_df['nvss_flux'] = nvss_flux[self.sedfilt]
        self.gsw_df['first_flux'] = first_flux[self.sedfilt]
        self.gsw_df['wenss_flux'] = wenss_flux[self.sedfilt]
        self.gsw_df['vlss_flux'] = vlss_flux[self.sedfilt]        

        self.gsw_df['nvss_lums'] =getlumfromflux(nvss_flux[self.sedfilt], self.z)
        self.gsw_df['firstlums'] =getlumfromflux(first_flux[self.sedfilt], self.z)
        self.gsw_df['wensslums'] = getlumfromflux(wenss_flux[self.sedfilt], self.z)
        self.gsw_df['vlsslums'] = getlumfromflux(vlss_flux[self.sedfilt], self.z)
        
