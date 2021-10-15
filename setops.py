#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from astropy.stats import bootstrap as apy_bootstrap

def commonpts1d(arr1, arr2):
    '''
    """
    Created to accompany the info from np.intersect1d
    Implemented into numpy but until I update that, I use this.

    Find the indices of points that are common among two arrays.

    Return the indices for the common points in both arrays.

    Parameters
    ----------
    arr1,arr2 : array_like

    Returns
    -------
    commvals : ndarray
        Result from np.intersect1d on arr1,arr2
    comm1 : ndarry
        Indices for common points in first array.
    comm2 : ndarray
        Indices for common points in second array

    Examples
    --------
    >>> commonpts1d([1,2,3,4],[2,1,4,6])
    (array(1,2,4),array([0,1,3]),array([1,0,2]))
    '''
    first = []
    second = []
    commvals = np.intersect1d(arr1, arr2)
    for i in range(len(commvals)):
        val1 = np.where(arr1 == commvals[i])[0][0]
        val2 = np.where(arr2 == commvals[i])[0][0]
        first.append(val1)
        second.append(val2) #shouldn't happen
    comm1 = np.reshape(np.array(first),-1)
    comm2 = np.reshape(np.array(second),-1)
    return commvals, comm1, comm2

def combine_arrs(lst_arr):
    '''
    This function combines multiple arrays into a single array
    '''
    #creating single arrays by combining them
    new_arr = np.array([], dtype=lst_arr[0].dtype)
    for arr in lst_arr:
        new_arr =np.append(new_arr, arr)
    return new_arr

def bin_quantity( quantity, binsize, mn, mx, threshold=0):
    '''
    bins a quantity based on min, max, binsize
    -if a threshold is specified, only bins with a certain minimum number of objects will be produced
    '''
    bn_edges = np.arange(mn, mx, binsize)
    bncenters = (bn_edges[1:]+bn_edges[:-1])/2
    bns = []
    bn_inds = []
    valid_bns = []
    for i in range(len(bn_edges)-1):
        val = np.where((quantity>bn_edges[i]) & (quantity <= bn_edges[i+1]))[0]
        if val.size > threshold:
            bns.append(quantity[val])
            bn_inds.append(val)
            valid_bns.append(i)
    return bn_edges, bncenters, bns, bn_inds, valid_bns
def bin_by_ind( quantity, inds, bncenters):
    '''
    putting other quantities into bins specified by some quantity produced by bin_quantity^
    '''
    binned_quantity = []
    for ind_set in inds:
        binned_quantity.append(quantity[ind_set])
    return binned_quantity
def bootstrap( data, bootnum, data_only=False):
    '''
    bootstraping a data set using astropy.bootstrap
    return stats about it if desired
    '''
    bootstrap_results = apy_bootstrap(data, bootnum=bootnum)
    if data_only:
        return bootstrap_results
    means = np.mean(bootstrap_results, axis=1)
    std_mean = np.std(means)
    mean_means = np.mean(means)
    return bootstrap_results, means, std_mean, mean_means