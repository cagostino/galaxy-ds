import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coords
import astropy.cosmology as apc
cosmo = apc.Planck15

class XMM:
    def __init__(self, xmmcat):
        self.sourceids = xmmcat.getcol('Source')
        
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
        
        self.hardflux_sn = self.hardflux/self.ehardflux
        self.fullflux_sn = self.fullflux/self.efullflux
        self.softflux_sn = self.softflux/self.esoftflux
        #self.flux8 = xmmcat.getcol

    def get_texp(self, matchinds):
        texps = []
        for k, source in enumerate(self.sourceids[matchinds]):
            source_obs = str(np.int64(str(source)[2:11]))
            obsids = np.array(np.array(self.obsids,dtype=np.int64), dtype='str')
            obs_ = np.where(obsids == source_obs)[0]
            #print(source_obs, obsids[obs_])
            exp = np.array([self.tpn[obs_], self.tmos1[obs_], self.tmos2[obs_]])
            texps.append(np.max(exp))
        return np.array(texps)
class XMM3obs(XMM):
    def __init__(self, xmmcat, xmmcatobs):
        super().__init__(xmmcat)
        self.obsids = xmmcatobs.getcol(1)
        self.ra = xmmcat.getcol('RAJ2000')

        self.dec = xmmcat.getcol('DEJ2000')
        self.obs_ra = xmmcatobs.getcol('RAJ2000')
        self.obs_dec = xmmcatobs.getcol('DEJ2000')
        self.tpn = xmmcatobs.getcol('t_PN')
        self.tmos1 = xmmcatobs.getcol('t_M1')
        self.tmos2 = xmmcatobs.getcol('t_M2')
class XMM4obs(XMM):
    def __init__(self, xmmcat, xmmcatobs):
        super().__init__(xmmcat)
        self.ra = xmmcat.getcol('RA_ICRS')
        self.dec = xmmcat.getcol('DE_ICRS')
        self.obs_ra = xmmcatobs.RAJ2000
        self.obs_dec = xmmcatobs.DEJ2000
        self.obsids = np.array(xmmcatobs.ObsID, dtype=np.int64)
        self.tpn = np.array(xmmcatobs['t.PN'])
        self.tmos1 = np.array(xmmcatobs['t.M1'])
        self.tmos2 = np.array(xmmcatobs['t.M2'])
        exp = np.vstack([self.tpn, self.tmos1, self.tmos2])
        self.texps  = np.max(exp, axis=0)
        

class CSC:
    def __init__(self, csccat):

        self.fluxu = csccat['Fluxu'] #0.2-0.5 keV
        #self.eflux1 = xmmcat['e_Flux1']
        self.fluxs = csccat['Fluxs']#0.5-1.2 keV
        #self.eflux2 = xmmcat.getcol('e_Flux2')
        self.fluxm = csccat['Fluxm'] #1.2-2 keV
        #self.eflux3 = xmmcat.getcol('e_Flux3')
        self.fluxh  = csccat['Fluxh'] #2-7
        #self.eflux4 = xmmcat.getcol('e_Flux4')
        self.fluxb = csccat['Fluxb'] #0.5-7
        #self.eflux5 = xmmcat.getcol('e_Flux5')
        '''
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
        '''
        self.hardflux = (self.fluxh)*1.15 #convert from 2-7 to 2-10 
        #self.ehardflux = np.sqrt(self.eflux8**2+self.eflux1**2+self.eflux2**2+self.eflux3**2)*0.87
        self.softflux = self.fluxs + self.fluxm
        #self.esoftflux = np.sqrt(self.eflux2**2+self.eflux3**2)
        self.fullflux = (self.fluxb)*1.04    #3.94337/4.31728
        #self.efullflux = np.sqrt(self.eflux8**2+self.eflux1**2)*0.91
        #self.flux8 = xmmcat.getcol
    