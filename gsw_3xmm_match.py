import numpy as np
raind = 0
decind = 1
sfrind = 2
massind = 3
plateind = 4
fiberind = 5
mjdind = 6
sedind = 7
class GSWCatmatch3xmm:
    def __init__(self, goodinds, gswlcids, redshift, sfrplus, fullflux_filt=[], hardflux_filt=[], hardflux2_filt=[], softflux_filt=[], sedflag=0):
        self.sedflags = sfrplus[sedind]
        self.sedflag = sedflag
        self.inds = goodinds
        self.sedfilt = np.where(self.sedflags[self.inds] == sedflag)[0]
        self.allids = gswlcids[0]
        self.ids = gswlcids[0][self.inds][self.sedfilt]
        self.matchsedflags = sfrplus[sedind][self.inds][self.sedfilt]
        self.gids = gswlcids[1][self.inds][self.sedfilt]
        self.z = redshift[self.inds][self.sedfilt]
        self.allz = redshift
        self.sfr= np.transpose(sfrplus[2][self.inds][self.sedfilt])[0]
        self.allra = sfrplus[raind]
        self.alldec = sfrplus[decind]
        self.matchra =sfrplus[raind][self.inds][self.sedfilt]
        self.matchdec = sfrplus[decind][self.inds][self.sedfilt]
        self.allsfr = sfrplus[sfrind]
        self.matchsfr=sfrplus[sfrind][self.inds][self.sedfilt]
        self.allmass = sfrplus[massind]
        self.matchmass = sfrplus[massind][self.inds][self.sedfilt]
        self.allplate = sfrplus[plateind]
        self.matchplate = sfrplus[plateind][self.inds][self.sedfilt]
        self.allfiber = sfrplus[fiberind]
        self.matchfiber = sfrplus[fiberind][self.inds][self.sedfilt]
        self.allmjds = sfrplus[mjdind]
        self.matchmjd = sfrplus[mjdind][self.inds][self.sedfilt]
        if len(fullflux_filt) != 0:
            self.softlums =getlumfromflux(softflux_filt[self.sedfilt], self.z)
            self.hardlums =getlumfromflux(hardflux_filt[self.sedfilt], self.z)
            self.hardlums2 =getlumfromflux(hardflux2_filt[self.sedfilt], self.z)
    
            self.fulllums =getlumfromflux(fullflux_filt[self.sedfilt], self.z)
    
            self.softlumsrf = np.log10(self.softlums*(1+self.z)**(1.7-2))
            self.hardlumsrf = np.log10(self.hardlums*(1+self.z)**(1.7-2))
            self.hardlums2rf = np.log10(self.hardlums2*(1+self.z)**(1.7-2))
    
            self.fulllumsrf = np.log10(self.fulllums*(1+self.z)**(1.7-2))
