import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coords
import astropy.cosmology as apc
cosmo = apc.Planck15

class XMM3:
    def __init__(self, xmmcat, xmmcatobs):
        self.sourceids = xmmcat.getcol('Source')
        self.ra = xmmcat.getcol('RAJ2000')
        self.radeg = self.ra*u.degree
        self.obs_ra = xmmcatobs.getcol('RAJ2000')
        self.obs_dec = xmmcatobs.getcol('DEJ2000')
        
        self.dec = xmmcat.getcol('DEJ2000')
        self.decdeg = self.dec*u.degree
        self.obsids = xmmcatobs.getcol(1)
        self.skyco = SkyCoord(ra=self.radeg, dec=self.decdeg)
        self.tpn = xmmcatobs.getcol('t_PN')
        self.tmos1 = xmmcatobs.getcol('t_M1')
        self.tmos2 = xmmcatobs.getcol('t_M2')
        self.flux1 = xmmcat.getcol('Flux1') #0.2-0.5 keV
        self.eflux1 = xmmcat.getcol('e_Flux1')
        self.flux2 = xmmcat.getcol('Flux2') #0.5-1 keV
        self.eflux2 = xmmcat.getcol('e_Flux2')
        self.flux3 = xmmcat.getcol('Flux3') #1-2 keV
        self.eflux3 = xmmcat.getcol('e_Flux3')
        self.flux4  = xmmcat.getcol('Flux4') #2-4.5
        self.eflux4 = xmmcat.getcol('e_Flux4')
        self.flux5 = xmmcat.getcol('Flux5') #4.5-12
        self.eflux5 = xmmcat.getcol('e_Flux5')
        self.flux8 = xmmcat.getcol('Flux8') #0.2-12
        self.eflux8  = xmmcat.getcol('e_Flux8')
        self.HR1 = xmmcat.getcol('HR1')
        self.eHR1 = xmmcat.getcol('e_HR1')
        self.HR2 = xmmcat.getcol('HR2')
        self.eHR2 = xmmcat.getcol('e_HR2')
        self.HR3 = xmmcat.getcol('HR3')
        self.eHR3 = xmmcat.getcol('e_HR3')
        self.HR4 = xmmcat.getcol('HR4')
        self.eHR4 = xmmcat.getcol('e_HR4')
        self.qualflag = xmmcat.getcol('S')
        self.ext = xmmcat.getcol('ext')
        self.extML = xmmcat.getcol('ext')
        self.hardflux2 = (self.flux4 + self.flux5)*0.87 #2.547/2.92097
        self.ehardflux2 =  np.sqrt(self.eflux4**2+self.eflux5**2)*0.87
        self.hardflux = (self.flux8 - (self.flux1 + self.flux2 + self.flux3))*0.87 #2.547/2.92097
        self.ehardflux = np.sqrt(self.eflux8**2+self.eflux1**2+self.eflux2**2+self.eflux3**2)*0.87
        self.softflux = self.flux2 + self.flux3
        self.esoftflux = np.sqrt(self.eflux2**2+self.eflux3**2)
        self.fullflux = (self.flux8-self.flux1)*0.91    #3.94337/4.31728
        self.efullflux = np.sqrt(self.eflux8**2+self.eflux1**2)*0.91
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
        matchedobsinds = []
        matchedobsids = []
        matchedexps = []
        for k, source in enumerate(self.sourceids[filtinds]):
            for l, obsid in enumerate(self.obsids):
                if str(obsid)==str(source)[1:11]:
                    matchedobsinds.append(l)
                    matchedobsids.append(obsid)
                    exp = np.array([self.tpn[l], self.tmos1[l], self.tmos2[l]])
                    matchedexps.append(np.max(exp))
                    continue
        return np.array(matchedsetinds), np.array(matchedsetids), np.array(matchedobsinds), np.array(matchedobsids), np.array(matchedexps)
    def singtimearr(self, matchedsetinds):
        exptimes = []
        exptimesbyobs =[]
        obsid = []
        val = np.where(matchedsetinds != 0)[0]
        for ind in val:
            exp = np.array([self.tpn[ind], self.tmos1[ind], self.tmos2[ind]])
            for k in range(matchedsetinds[ind]):
                exptimes.append(np.max(exp))
                obsid.append(self.obsids[ind])
            exptimesbyobs.append(np.max(exp))
        return np.array(exptimes), np.array(exptimesbyobs), np.array(obsid)
