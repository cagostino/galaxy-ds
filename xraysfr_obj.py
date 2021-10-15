import numpy as np

lum_arr = np.logspace(34,46,1000)
loglum_arr = np.log10(lum_arr)

sfrsoft = 2.2e-40*lum_arr
logsfrsoft = np.log10(sfrsoft)#from salpeter IMF
logsfrsoft = logsfrsoft - 0.2 #converted to Chabrier IMF

sfrhard = 2.0e-40*lum_arr
logsfrhard = np.log10(sfrhard)#from salpeter IMF
logsfrhard =logsfrhard - 0.2 #converted to Chabrier IMF
#combine the supposed sfr form soft and hard to get the full
sfrfull= 1.05e-40*lum_arr
logsfrfull = np.log10(sfrfull) -0.2
xrayranallidict = {'soft':1/(1.39e-40),'hard':1/(1.26e-40),'full':1/(0.66e-40)}


class Xraysfr:
    def __init__(self, xraylums, gswcat, filt, agn, nonagn, typ):
        self.typ = typ
        self.filt = filt
        #self.unclassifiable = np.array([i for i in range(len(xraylums)) if i not in self.filt   ])
        self.mass = gswcat.mass
        self.z = gswcat.z
        self.z_filt = self.z[self.filt]
        self.ra = gswcat.ra
        self.ra_filt = gswcat.ra[self.filt]
        self.dec = gswcat.dec
        self.dec_filt = gswcat.dec[self.filt]
        #self.xrayra = gswcat.matchxrayra
        #self.xrayra_filt = gswcat.matchxrayra[self.filt]
        #self.xraydec = gswcat.matchxraydec
        #self.xraydec_filt = gswcat.matchxraydec[self.filt]
        self.lum_mass =  xraylums-np.array(gswcat.gsw_df.mass) #mass_m2_match
        self.lum = xraylums
        self.sfr_mass = np.array(gswcat.gsw_df.sfr)-np.array(gswcat.gsw_df.mass)# sfr_m2_match-mass_m2_match
        self.sfr = np.array(gswcat.gsw_df.sfr)
        self.lxsfr = np.log10(xrayranallidict[typ]*10**(self.sfr))
        self.lxsfr_filt = np.log10(xrayranallidict[typ]*10**(self.sfr[self.filt]))
 
        self.notagn = np.where(self.lum[self.filt]==0)
        self.mass_filt = self.mass[self.filt]
        self.lum_mass_filt = self.lum_mass[self.filt]
        self.sfr_filt = self.sfr[self.filt]
        self.sfr_mass_filt = self.sfr_mass[self.filt]
        self.lum_filt = self.lum[self.filt]
        #self.mess = np.where((self.lum_mass_val >29) & (self.lum_mass_val<30) & (self.softsfr_mass_val >-11))[0]
        self.valid = np.where(self.lum[self.filt] >0)[0]
        self.validagn = np.where(self.lum[self.filt][agn]>0)[0]
        self.validnoagn = np.where(self.lum[self.filt][nonagn]>0)[0]
        #valid overall
        self.lum_mass_val_filt =  self.lum_mass[self.filt][self.valid] #mass_m2_match
        self.lum_val_filt = self.lum[self.filt][self.valid]
        self.sfr_mass_val_filt = self.sfr_mass[self.filt][self.valid]
        self.sfr_val_filt = self.sfr[self.filt][self.valid]
        self.mass_val_filt = self.mass[self.filt][self.valid]
        #valid bpt agn
        self.lum_mass_val_filtagn =  self.lum_mass[self.filt][agn][self.validagn]
        self.lum_val_filtagn = self.lum[self.filt][agn][self.validagn]
        self.sfr_mass_val_filtagn = self.sfr_mass[self.filt][agn][self.validagn]
        self.sfr_val_filtagn = self.sfr[self.filt][agn][self.validagn]
        self.mass_filtagn = self.mass[self.filt][agn][self.validagn]

        #valid bpt hii
        self.lum_mass_val_filtnoagn =  self.lum_mass[self.filt][nonagn][self.validnoagn]
        self.lum_val_filtnoagn = self.lum[self.filt][nonagn][self.validnoagn]
        self.sfr_mass_val_filtnoagn = self.sfr_mass[self.filt][nonagn][self.validnoagn]
        self.sfr_val_filtnoagn = self.sfr[self.filt][nonagn][self.validnoagn]
        self.mass_filtnoagn = self.mass[self.filt][nonagn][self.validnoagn]

        self.likelyagn_xr = np.where((self.lxsfr[self.filt][self.valid] < self.lum[self.filt][self.valid] - 0.6) &
                                     (self.lum[self.filt][self.valid] > 0))[0]
        self.likelyagnbpthii = np.where((self.lxsfr[self.filt][nonagn][self.validnoagn] < self.lum[self.filt][nonagn][self.validnoagn]-0.6) &
                                        (self.lum[self.filt][nonagn][self.validnoagn] >0))[0]
        self.likelyagnbptagn = np.where((self.lxsfr[self.filt][agn][self.validagn] < self.lum[self.filt][agn][self.validagn]-0.6) &
                                        (self.lum[self.filt][agn][self.validagn] >0))[0]
        self.likelysf = np.where((abs(self.lxsfr[self.filt][self.valid] - self.lum[self.filt][self.valid]) < 0.3) &
                                     (self.lum[self.filt][self.valid] > 0))[0]
        self.likelysfbpthii = np.where((abs(self.lxsfr[self.filt][nonagn][self.validnoagn] - self.lum[self.filt][nonagn][self.validnoagn])<0.3) &
                                        (self.lum[self.filt][nonagn][self.validnoagn] >0))[0]
        self.likelysfbptagn = np.where((abs(self.lxsfr[self.filt][agn][self.validagn] - self.lum[self.filt][agn][self.validagn])<0.3) &
                                        (self.lum[self.filt][agn][self.validagn] >0))[0]
        #valid xray agn
        self.lum_mass_val_filt_xrayagn =  self.lum_mass[self.filt][self.valid][self.likelyagn_xr]
        self.lum_val_filt_xrayagn = self.lum[self.filt][self.valid][self.likelyagn_xr ]
        self.sfr_mass_val_filt_xrayagn = self.sfr_mass[self.filt][self.valid][self.likelyagn_xr]
        self.sfr_val_filt_xrayagn = self.sfr[self.filt][self.valid][self.likelyagn_xr ]
        self.mass_filt_xrayagn = self.mass[self.filt][self.valid][self.likelyagn_xr ]
        self.z_filt_xrayagn = self.z[self.filt][self.valid][self.likelyagn_xr]
        
        #valid bpt agn

        self.lum_mass_val_filtagn_xrayagn =  self.lum_mass[self.filt][agn][self.validagn][self.likelyagnbptagn]
        self.lum_val_filtagn_xrayagn = self.lum[self.filt][agn][self.validagn][self.likelyagnbptagn]
        self.sfr_mass_val_filtagn_xrayagn = self.sfr_mass[self.filt][agn][self.validagn][self.likelyagnbptagn]
        self.sfr_val_filtagn_xrayagn = self.sfr[self.filt][agn][self.validagn][self.likelyagnbptagn]
        self.mass_filtagn_xrayagn = self.mass[self.filt][agn][self.validagn][self.likelyagnbptagn]
        self.lxsfr_val_filtagn_xrayagn = self.lxsfr[self.filt][agn][self.validagn][self.likelyagnbptagn]
        #valid bpt hii
        self.lum_mass_val_filtnoagn_xrayagn =  self.lum_mass[self.filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.lum_val_filtnoagn_xrayagn = self.lum[self.filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.sfr_mass_val_filtnoagn_xrayagn = self.sfr_mass[self.filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.sfr_val_filtnoagn_xrayagn = self.sfr[self.filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.mass_filtnoagn_xrayagn = self.mass[self.filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.lxsfr_val_filtnoagn_xrayagn = self.lxsfr[self.filt][nonagn][self.validnoagn][self.likelyagnbpthii]