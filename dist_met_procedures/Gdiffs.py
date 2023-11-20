#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
To be used within the main program, matchgal
"""
import numpy as np
from sklearn import linear_model
from scipy.interpolate import griddata

class Gdiffs:
    def __init__(self, xrids, gswids, xray, gsw, quantities, filt=[]):
        '''
        xrids and gswids are lists of SDSS ids, used to make sure that the 
        X-ray fractions are not including the galaxy itself
        '''
        self.xrids = xrids
        self.gswids = gswids
        self.xray = xray
        self.gsw = gsw
        self.quantities = quantities
        self.filt = filt
        
        self.dists, self.q_diffs = self.galaxy_diffs_bpt()
    def galaxy_diffs_bpt(self):
        dists = []
        #ra dec
        #mass redshift
        lenloop = len(self.xray['mass'])
        var_qs = [np.var(self.xray[quantity]) for quantity in self.quantities]

        quantity_diffs =  [ [] for i in range(len(self.quantities))]
        
        dists = []
        for i in range(lenloop):
            distsq_i = 0
            for j, quantity in enumerate(self.quantities):
                q_diff = (self.xray[quantity].iloc[i] - self.gsw[quantity])**2/var_qs[j]
                quantity_diffs[j].append(q_diff)
                distsq_i+=q_diff
            dists.append(np.sqrt(distsq_i))
        return np.array(dists), np.array(quantity_diffs)
    def get_filt(self,filt):
        self.filt = filt
        self.dists_filt = np.copy(self.dists[:, filt])
        self.q_diffs_filt = np.copy(quantity[:, filt] for quantity in self.quantities)
    def nearbyx(self, comminds):
        nclones = []
        mindists = []
        alldists = []
        allfracs = []
        allfracx = []
        allfracgsw = []
        bins = np.linspace(0, 10, 201)
        self.bins = bins
        for diff in self.dists_filt:
            allothers = [diff[comminds] > 0] #don't want to find itself
            xrdiff = diff[comminds][allothers]
            nearestx = np.min(xrdiff)

            fin = np.where((diff < 10) & (diff > 0) & (np.isfinite(diff)))
            hist = np.histogram(diff[fin], bins=bins)
            frac = []
            fracx = []
            fracgsw = []
            for bn in bins:
                xrinc = np.where(xrdiff < bn)[0]
                gswinc = np.where(diff[fin] < bn)[0]
                if xrinc.size == 0 or gswinc.size == 0:
                    frac.append(0)
                    fracx.append(0)
                    fracgsw.append(gswinc.size/diff[fin].size)
                    continue
                fracxr = xrinc.size/gswinc.size
                frac.append(fracxr)
                fracx.append(xrinc.size/xrdiff.size)
                fracgsw.append(gswinc.size/diff[fin].size)
            within = np.where((diff < nearestx) & (diff > 0))[0] #how many galaxies are
            nwithin = within.size
            nclones.append(nwithin)
            mindists.append(nearestx)
            allfracs.append(np.array(frac))
            allfracx.append(np.array(fracx))
            allfracgsw.append(np.array(fracgsw))
            alldists.append(diff[comminds])
        self.nrx = np.array(nclones)
        self.mindx = np.array(mindists)
        self.alldists = np.array(alldists)
        self.xrgswfracs = np.array(allfracs)
        self.xrfrac = np.array(allfracx)
        self.gswfrac = np.array(allfracgsw)
    def getdist_by_thresh(self, threshold):
        allvalid_gals = []
        for dist in self.dists_filt:
            valid_gals = np.where(dist < threshold)[0]
            allvalid_gals.append(valid_gals)
        self.passinggals =  np.array(allvalid_gals)
    def interpdistgrid(self, nx, ny, binnum,method='linear', xrfracs=False):
        if xrfracs:
            self.xrfracs=xrfracs.grid
            self.x = xrfracs.meshx
            self.y = xrfracs.meshy
        else:
            self.xrfracs = self.xrgswfracs[:,binnum].reshape((self.x_xr.size, 1))
        x, y = self.x_xr, self.y_xr
        mnx, mxx = np.min(x), np.max(x)
        mny, mxy = np.min(y), np.max(y)
        X = np.vstack([x, y]).transpose()#, x*x, y*y
        self.regr = linear_model.LinearRegression()
        self.regr.fit(X, self.xrfracs.reshape(-1))
        self.predxrfrac = self.regr.predict(X)

        self.rangex = np.linspace(mnx-1*(mxx-mnx)/nx, mxx+1*(mxx-mnx)/nx, nx)#
        self.rangey = np.linspace(mny-1*(mxy-mny)/ny, mxy+1*(mxy-mny)/ny, ny)#
        self.meshx, self.meshy = np.meshgrid(self.rangex, self.rangey)
        points = np.vstack([x, y]).transpose()
        self.grid = griddata(points, self.xrfracs, (self.meshx, self.meshy),method=method).reshape((nx, ny))
        #self.grid = griddata(points, self.predxrfrac, (self.meshx, self.meshy),method='linear').reshape((nx, ny))
