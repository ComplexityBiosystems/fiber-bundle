#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Stefan Hiemer
"""

import os
import time
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import h5py



def cluster_list(lst,n):
    return [lst[i:(i + n)] for i in np.arange(0, len(lst), n)]

def read_h5(fibers,
            load,
            temperature,
            k,
            distribution,
            h5file=None,
            subset=None,
            thresh=False):
    """
    Read data created by run.py.
    
    Parameters
    ----------
    fibers : int 
        number of fibers in fiber bundle.
    load : float 
        load per fiber (symbol f0 in paper).
    temperature : float 
        temperature (symbol T in paper).
    k : float
        distribution parameter. For "uniform" it is the width of distribution, 
        for "weibull" the exponent (unit mean) and "identical" the threshold 
        value.
    dists : list
        distribution name. Currently only "uniform", "weibull" and "identical" 
        possible.
    h5file : str
        name of h5file where to store results
    subset : int
        draw only first "subset" entries.
    thresh : bool
        if True, returns also thresholds for the specific trajectory.
    
    Returns
    ------- timeseries,n_fibers,aval
    thresholds : list
        list of np.ndarrays. Each array contains the thresholds for each fiber 
        bundle.
    timeseries : list
        list of np.ndarrays. Each array contains the times at which thermally 
        activated events occured for an individual trajectory.
    n_fibers : list
        list of np.ndarrays. Each array contains the number of fibers intact at 
        a thermal activated event.
    aval : list
        list of np.ndarrays. Each array contains the avalanche size of events 
        for an individual trajectory.
    """

    path = os.path.join(distribution, "fibers-"+str(fibers), "k-"+str(k),
                        "t-"+str(temperature), "i-"+str(load))
    if h5file is None:
        raise ValueError("h5file is None.")

    t = time.time()

    with h5py.File(h5file,'r') as f:

        if subset is None:
            subset = f[path+"/seeds"].shape[0]

        # random seeds
        seeds = f[path+"/seeds"][:subset,0]

        if thresh:
            # thresholds
            thresholds = f[path+"/thresholds"][:subset*fibers,0]

        # calculate number of timeseries
        n_timeseries = int(f[path+"/length"].shape[0]/f[path+"/seeds"].shape[0])

        # load length of individual time series
        length = f[path+"/length"][:subset*n_timeseries,0]

        subset = np.sum(length)

        # load number of fibers which are still alive at that moment in time
        n_fibers = f[path+"/n_fibers"][:subset,0]

        #
        timeseries = f[path+"/time"][:subset,0]

    print("Data loaded:",time.time()-t)

    if thresh:
        # split thresholds for individual systems
        thresholds = np.split(thresholds,indices_or_sections=seeds.shape[0])
        print("Thresholds splitted:",time.time()-t)

    # split into individual timeseries
    split = np.cumsum(length)
    timeseries = np.split(timeseries,indices_or_sections = split)[:-1]
    n_fibers = np.split(n_fibers,indices_or_sections = split)[:-1]
    print("Split data into individual timeseries:",time.time()-t)

    # calculate avalanches from n_fibers
    with Pool(cpu_count()) as p:
        aval = p.map(partial(_aval,fibers=fibers),n_fibers)
    print("Avalanches calculated:",time.time()-t)

    # cluster timeseries which belong to one system
    timeseries = cluster_list(timeseries,n_timeseries)
    n_fibers = cluster_list(n_fibers,n_timeseries)
    aval = cluster_list(aval,n_timeseries)
    print("Clustered for individual timeseries:",time.time()-t)

    if thresh:
        return thresholds,timeseries,n_fibers,aval
    else:
        return timeseries,n_fibers,aval

def _aval(n_fiber,fibers):
    return np.insert(n_fiber[:-1],0,fibers) - n_fiber

if __name__ == "__main__":

    thresholds,timeseries,n_fibers,aval = read_h5(fibers=15,
                                                  load=0.175,
                                                  temperature=0.05,
                                                  k=1,
                                                  distribution="uniform",
                                                  subset=2)

    # extract lifetimes
    lifetimes = [t[-1] for t in timeseries[1]]

    import matplotlib.pyplot as plt

    fig,ax = plt.subplots(1,1)

    ax.hist(lifetimes,bins=125)
    plt.show()
