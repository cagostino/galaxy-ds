#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
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