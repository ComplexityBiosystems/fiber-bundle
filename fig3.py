import numpy as np
from scipy.special import expi

import matplotlib.pyplot as plt

from read_h5 import read_h5

def plot(load=0.125,temp=0.05,thresh_dist="uniform",mode="cdf"):
    """
    Plot Fig. 3 in paper.

    Parameters
    ----------
    load : float
        load per fiber (f0 in paper).
    temp : float
        temperature (symbol T in paper).
    thresh_dist : str
        name of distribution (only "uniform" allowed). 
    mode : str
        either "pdf" or "cdf". Only the latter is shown in the paper.

    Returns
    -------
    None

    """
    
    timeseries,n_fibers,aval = read_h5(fibers=1,
                                       load=load,
                                       temperature=temp,
                                       k=1,
                                       distribution=thresh_dist,
                                       subset=None,
                                       h5file="fiber-bundles.h5")

    # extract lifetimes
    lifetimes = np.array([t[0][-1] for t in timeseries])
    #
    mask = ~np.isinf(lifetimes)
    lifetimes = lifetimes[mask]
    #
    fig,ax = plt.subplots(1,1,figsize=(15,10))
    font=35

    # exclude zero
    x = np.linspace(0,np.max(lifetimes),1000)
    x[0] = 1e-9

    if mode == "pdf":


        # histogram of data
        ax.hist(lifetimes,
                bins=500,
                density=True)

        # theoretical prediction
        ax.plot(x,pdf(x,
                      load = load,
                      temp = temp,
                      thresh_dist=thresh_dist),
                color='k')
        ax.set_ylabel(r"p($\tau$)",fontsize=font)
        ax.set_ylim(bottom=0)

    elif mode == "cdf":

        # histogram of data
        ax.hist(lifetimes,
                bins=lifetimes.shape[0],
                density=True,
                cumulative=True,
                alpha=1.)
        ax.plot(x,cdf(x,
                      load = load,
                      temp = temp,
                      thresh_dist=thresh_dist),
                lw=5,linestyle="--",
                color='k')
        ax.set_ylabel(r"P($\tau$)",fontsize=font)
        ax.set_ylim(0,1.05)

    ax.set_xlabel(r"$\tau$",fontsize=font)
    #ax.set_xlim(,np.max(x))

    ax.tick_params(axis='both', which='both', labelsize=font)

    param = '-'.join(["i",str(load),"temp",str(temp)])
    plt.savefig(mode+"-comparison_single-fiber_"+param+".pdf",
                format="pdf",
                bbox_inches="tight")

    return

def pdf(time,load,temp,
        thresh_dist="uniform",k=1):
    """
    Calculate Eq. B6 in the paper depending on the distribution.

    Parameters
    ----------
    time : np.ndarray
        time at which to calculate (symbol tau in paper).
    load : float
        load per fiber (f0 in paper).
    temp : float
        temperature (symbol T in paper).
    thresh_dist : str
        name of distribution (only "uniform" allowed).
    k: float
        width of uniform distribution.

    Returns
    -------
    pdf : np.ndarray
        probability density function of lifetimes.

    """
    if thresh_dist=="uniform":
        pdf =  load/k * temp/(k*time) *(np.exp(-np.exp(-(k-load)/temp)*time)-np.exp(-time))
    else:
        raise NotImplementedError("Distribution not implemented: ",thresh_dist)

    # mask for dirac delta
    mask = time == 0

    if np.any(mask):
        pdf[mask]=np.inf

    return pdf

def cdf(time,load,temp,
        thresh_dist="uniform",k=1):
    """
    Calculate Eq. B7 in the paper depending on the distribution.

    Parameters
    ----------
    time : np.ndarray
        time at which to calculate (symbol tau in paper).
    load : float
        load per fiber (f0 in paper).
    temp : float
        temperature (symbol T in paper).
    thresh_dist : str
        name of distribution (only "uniform" allowed).
    k: float
        width of uniform distribution.

    Returns
    -------
    cdf : np.ndarray
        cumulative distribution function of lifetimes.

    """
    
    if thresh_dist=="uniform":
        cdf = 1 +  temp/k *(expi(-np.exp(-(k-load)/temp)*time)-expi(-time))
    else:
        raise NotImplementedError("Distribution not implemented: ",thresh_dist)

    return cdf

if __name__ == "__main__":
    
    #
    plot(load=0.15,temp=0.1,thresh_dist="uniform",mode="cdf")
    
