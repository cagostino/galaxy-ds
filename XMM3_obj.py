import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coords
import astropy.cosmology as apc
cosmo = apc.Planck15

class XMM3:
    def __init__(self, xmmcat, xmmcatobs):
        self.sourceids = xmmcat.getcol(0)
        self.ra = xmmcat.getcol(2)
        self.radeg = self.ra*u.degree
        self.obs_ra = xmmcatobs.getcol('RAJ2000')
        self.obs_dec = xmmcatobs.getcol('DEJ2000')
        
        self.dec = xmmcat.getcol(3)
        self.decdeg = self.dec*u.degree
        self.obsids = xmmcatobs.getcol(1)
        self.skyco = SkyCoord(ra=self.radeg, dec=self.decdeg)
        self.tpn = xmmcatobs.getcol('t_PN')
        self.tmos1 = xmmcatobs.getcol('t_M1')
        self.tmos2 = xmmcatobs.getcol('t_M2')
        self.flux1 = xmmcat.getcol(5) #0.2-0.5 keV
        self.flux2 = xmmcat.getcol(7) #0.5-1 keV
        self.flux3 = xmmcat.getcol(9) #1-2 keV
        self.flux4  = xmmcat.getcol(11) #2-4.5
        self.flux5 = xmmcat.getcol(13) #4.5-12
        self.flux8 = xmmcat.getcol(15) #0.2-12
        self.hardflux2 = (self.flux4 + self.flux5)*0.87 #2.547/2.92097
        self.hardflux = (self.flux8 - (self.flux1 + self.flux2 + self.flux3))*0.87 #2.547/2.92097
        self.softflux = self.flux2 + self.flux3
        self.fullflux = (self.flux8-self.flux1)*0.91    #3.94337/4.31728
        #self.flux8 = xmmcat.getcol
    def obsmatching(self, filtinds):
        matchedsetinds = []
        matchedsetids = []
        for i, obsid in enumerate(self.obsids):
            #print(i)
            currentsetinds = []
            currentsetids= []
            for j, sourceid in enumerate(self.sourceids[filtinds]):
                if str(obsid) == str(sourceid)[1:11]:
                    currentsetinds.append(j)
                    currentsetids.append(sourceid)
                else:
                    currentsetids.append(-999)

            matchedsetinds.append(len(currentsetinds))
            matchedsrcids = np.where(currentsetids != -999)[0]
            matchedsetids.append(currentsetids)
        return np.array(matchedsetinds), np.array(matchedsetids)
    def singtimearr(self, matchedsetinds):
        exptimes = []
        exptimesbyobs =[]
        val = np.where(matchedsetinds != 0)[0]
        for ind in val:
            exp = np.array([self.tpn[ind], self.tmos1[ind], self.tmos2[ind]])
            for k in range(matchedsetinds[ind]):
                exptimes.append(np.max(exp))
            exptimesbyobs.append(np.max(exp))
        return np.array(exptimes), np.array(exptimesbyobs)
