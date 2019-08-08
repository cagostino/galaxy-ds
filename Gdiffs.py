#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
To be used within the main program, matchgal.py
"""
import numpy as np
from sklearn import linear_model
from scipy.interpolate import griddata

class Gdiffs:
    def __init__(self, m1ids, gswids, xray, gsw):
        self.m1ids = m1ids
        self.gswids = gswids
        self.xray = xray
        self.gsw = gsw
        self.dists, self.massd, self.zd, self.bptxd, self.bptyd = self.galaxy_diffs()
    def galaxy_diffs(self, filt = []):
        dists = []
        #ra dec
        #mass redshift
        lenloop = len(self.xray.mass)
        varmass = np.var(self.xray.mass)#(np.max(self.xray.mass)-np.min(self.xray.mass))**2
        varz = np.var(self.xray.z)#(np.max(self.xray.z)-np.min(self.xray.z))**2
        varbptx = np.var(self.xray.niiha)#(np.max(self.xray.niiha)-np.min(self.xray.niiha))**2
        varbpty = np.var(self.xray.oiiihb)#(np.max(self.xray.oiiihb)-np.min(self.xray.oiiihb))**2
        bptxd =[]
        bptyd = []
        massd =[]
        zd =[]
        for i in range(lenloop):
            val = np.where(self.gswids == self.m1ids[i])[0]
            massdiffs = (self.xray.mass[i] - self.gsw.mass)**2/varmass #massdiff
            zdiffs  =   (self.xray.z[i] - self.gsw.z)**2/varz
            bptxdiffs = (self.xray.niiha[i] - self.gsw.niiha)**2/varbptx
            bptydiffs = (self.xray.oiiihb[i] - self.gsw.oiiihb)**2/varbpty
            dist = np.sqrt(massdiffs + zdiffs + bptxdiffs + bptydiffs)
            dists.append(dist)
            massd.append(massdiffs)
            bptxd.append(bptxdiffs)
            bptyd.append(bptydiffs)
            zd.append(zdiffs)
        return np.array(dists), np.array(massd), np.array(zd), np.array(bptxd), np.array(bptyd)
    def get_filt(self,filt):
        self.filt = filt
        self.dists_filt = np.copy(self.dists[:, filt])
        self.massd_filt = np.copy(self.massd[:, filt])
        self.bptxd_filt = np.copy(self.bptxd[:, filt])
        self.bptyd_filt = np.copy(self.bptyd[:, filt])
        self.zd_filt = np.copy(self.zd[:,filt])
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
        self.alls82dists = np.array(alldists)
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
            self.xrfracs = self.xrgswfracs[:,binnum].reshape((self.xray.niiha.size, 1))
        x, y = self.xray.niiha, self.xray.oiiihb
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
    def maskgrid(self):
        x = self.xray.niiha
        y = self.xray.oiiihb
        interdistx = self.rangex[1]-self.rangex[0]
        interdisty = self.rangey[1]-self.rangey[0]
        invaly = []
        invalx = []
        for i in range(len(self.rangex)):
            distx =(self.rangex[i] -x)

            for j in range(len(self.rangey)):
                disty = (self.rangey[j] - y)
                dists = np.sqrt(distx**2+disty**2)
                ad = np.where(dists < np.sqrt((interdistx**2+interdisty**2))/np.sqrt(2))[0]
                if len(ad) == 0:
                    invaly.append(j)
                    invalx.append(i)
        return invalx,invaly