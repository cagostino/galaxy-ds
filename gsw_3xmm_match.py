import numpy as np
from ast_func import *
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
        self.sfr=sfrplus[sfrind][self.inds][self.sedfilt]
        self.mass = sfrplus[massind][self.inds][self.sedfilt]
        self.plate = sfrplus[plateind][self.inds][self.sedfilt]
        self.fiber = sfrplus[fiberind][self.inds][self.sedfilt]
        self.mjd = sfrplus[mjdind][self.inds][self.sedfilt]        
        self.av = sfrplus[av_ind][self.inds][self.sedfilt]
        self.av_err = sfrplus[av_err_ind][self.inds][self.sedfilt]
        

class GSWCatmatch3xmm(GSWCat):
    def __init__(self, goodinds, gswlcids, redshift, sfrplus, xrayflag, fullflux_filt, 
                 efullflux_filt ,hardflux_filt, hardflux2_filt, softflux_filt, ext,sedflag=0):
        super().__init__(goodinds, gswlcids, redshift, sfrplus, sedflag=sedflag)
        self.softlums =getlumfromflux(softflux_filt[self.sedfilt], self.z)
        self.hardlums =getlumfromflux(hardflux_filt[self.sedfilt], self.z)
        self.hardlums2 =getlumfromflux(hardflux2_filt[self.sedfilt], self.z)
        self.fulllums = getlumfromflux(fullflux_filt[self.sedfilt], self.z)
        self.efulllums_up =  getlumfromflux(fullflux_filt[self.sedfilt]+efullflux_filt[self.sedfilt],self.z)
        self.efulllums_down = getlumfromflux(fullflux_filt[self.sedfilt]-efullflux_filt[self.sedfilt],self.z)
        self.softlumsrf = np.log10(self.softlums*(1+self.z)**(1.7-2))
        self.hardlumsrf = np.log10(self.hardlums*(1+self.z)**(1.7-2))
        self.hardlums2rf = np.log10(self.hardlums2*(1+self.z)**(1.7-2))
        self.xrayflag = xrayflag[self.sedfilt]
        self.fulllumsrf = np.log10(self.fulllums*(1+self.z)**(1.7-2))
        self.efulllumsrf_down = np.log10(self.efulllums_down*(1+self.z)**(1.7-2))
        self.efulllumsrf_up = np.log10(self.efulllums_up*(1+self.z)**(1.7-2))
        self.ext = ext[self.sedfilt]