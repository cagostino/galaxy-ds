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
        self.mass = gswcat.matchmass
        self.z = gswcat.z
        self.z_filt = self.z[filt]
        self.lum_mass =  xraylums-gswcat.matchmass #mass_m2_match
        self.lum = xraylums
        self.sfr_mass = gswcat.matchsfr-gswcat.matchmass# sfr_m2_match-mass_m2_match
        self.sfr = gswcat.matchsfr
        self.lxsfr = np.log10(xrayranallidict[typ]*10**(self.sfr))
        self.lxsfr_filt = np.log10(xrayranallidict[typ]*10**(self.sfr[filt]))

        self.notagn = np.where(self.lum[filt]==0)
        self.mass_filt = self.mass[filt]
        self.lum_mass_filt = self.lum_mass[filt]
        self.sfr_filt = self.sfr[filt]
        self.sfr_mass_filt = self.sfr_mass[filt]
        self.lum_filt = self.lum[filt]
        #self.mess = np.where((self.lum_mass_val >29) & (self.lum_mass_val<30) & (self.softsfr_mass_val >-11))[0]
        self.valid = np.where(self.lum[filt] >0)[0]
        self.validagn = np.where(self.lum[filt][agn]>0)[0]
        self.validnoagn = np.where(self.lum[filt][nonagn]>0)[0]
        #valid overall
        self.lum_mass_val_filt =  self.lum_mass[filt][self.valid] #mass_m2_match
        self.lum_val_filt = self.lum[filt][self.valid]
        self.sfr_mass_val_filt = self.sfr_mass[filt][self.valid]
        self.sfr_val_filt = self.sfr[filt][self.valid]
        self.mass_val_filt = self.mass[filt][self.valid]
        #valid bpt agn
        self.lum_mass_val_filtagn =  self.lum_mass[filt][agn][self.validagn]
        self.lum_val_filtagn = self.lum[filt][agn][self.validagn]
        self.sfr_mass_val_filtagn = self.sfr_mass[filt][agn][self.validagn]
        self.sfr_val_filtagn = self.sfr[filt][agn][self.validagn]
        self.mass_filtagn = self.mass[filt][agn][self.validagn]

        #valid bpt hii
        self.lum_mass_val_filtnoagn =  self.lum_mass[filt][nonagn][self.validnoagn]
        self.lum_val_filtnoagn = self.lum[filt][nonagn][self.validnoagn]
        self.sfr_mass_val_filtnoagn = self.sfr_mass[filt][nonagn][self.validnoagn]
        self.sfr_val_filtnoagn = self.sfr[filt][nonagn][self.validnoagn]
        self.mass_filtnoagn = self.mass[filt][nonagn][self.validnoagn]

        self.likelyagn_xr = np.where((self.lxsfr[filt][self.valid] < self.lum[filt][self.valid] - 0.6) &
                                     (self.lum[filt][self.valid] > 0))[0]
        self.likelyagnbpthii = np.where((self.lxsfr[filt][nonagn][self.validnoagn] < self.lum[filt][nonagn][self.validnoagn]-0.6) &
                                        (self.lum[filt][nonagn][self.validnoagn] >0))[0]
        self.likelyagnbptagn = np.where((self.lxsfr[filt][agn][self.validagn] < self.lum[filt][agn][self.validagn]-0.6) &
                                        (self.lum[filt][agn][self.validagn] >0))[0]
        self.likelysf = np.where((abs(self.lxsfr[filt][self.valid] - self.lum[filt][self.valid]) < 0.3) &
                                     (self.lum[filt][self.valid] > 0))[0]
        self.likelysfbpthii = np.where((abs(self.lxsfr[filt][nonagn][self.validnoagn] - self.lum[filt][nonagn][self.validnoagn])<0.3) &
                                        (self.lum[filt][nonagn][self.validnoagn] >0))[0]
        self.likelysfbptagn = np.where((abs(self.lxsfr[filt][agn][self.validagn] - self.lum[filt][agn][self.validagn])<0.3) &
                                        (self.lum[filt][agn][self.validagn] >0))[0]
        #valid xray agn
        self.lum_mass_val_filt_xrayagn =  self.lum_mass[filt][self.valid][self.likelyagn_xr]
        self.lum_val_filt_xrayagn = self.lum[filt][self.valid][self.likelyagn_xr ]
        self.sfr_mass_val_filt_xrayagn = self.sfr_mass[filt][self.valid][self.likelyagn_xr]
        self.sfr_val_filt_xrayagn = self.sfr[filt][self.valid][self.likelyagn_xr ]
        self.mass_filt_xrayagn = self.mass[filt][self.valid][self.likelyagn_xr ]
        self.z_filt_xrayagn = self.z[filt][self.valid][self.likelyagn_xr]
        
        #valid bpt agn

        self.lum_mass_val_filtagn_xrayagn =  self.lum_mass[filt][agn][self.validagn][self.likelyagnbptagn]
        self.lum_val_filtagn_xrayagn = self.lum[filt][agn][self.validagn][self.likelyagnbptagn]
        self.sfr_mass_val_filtagn_xrayagn = self.sfr_mass[filt][agn][self.validagn][self.likelyagnbptagn]
        self.sfr_val_filtagn_xrayagn = self.sfr[filt][agn][self.validagn][self.likelyagnbptagn]
        self.mass_filtagn_xrayagn = self.mass[filt][agn][self.validagn][self.likelyagnbptagn]

        #valid bpt hii
        self.lum_mass_val_filtnoagn_xrayagn =  self.lum_mass[filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.lum_val_filtnoagn_xrayagn = self.lum[filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.sfr_mass_val_filtnoagn_xrayagn = self.sfr_mass[filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.sfr_val_filtnoagn_xrayagn = self.sfr[filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.mass_filtnoagn_xrayagn = self.mass[filt][nonagn][self.validnoagn][self.likelyagnbpthii]
