import numpy as np

import matplotlib.pyplot as plt

from read_h5 import read_h5

def moment_plot(loads=[0.5,0.7,0.9],
                temps=[0.05,0.1,0.15],
                fibers=[1,10,100,1000,10000]):
    """
    Plot Fig. 1 in paper.

    Parameters
    ----------
    loads : list
        list of float with load per fiber (f0 in paper).
    temps : list
        list of floats with temperature (symbol T in paper).
    fibers : list
        list of integers for number of fibers (symbol N in paper).

    Returns
    -------
    None.
    """

    from itertools import product
    
    # options for plotting
    colors = plt.get_cmap("tab10")
    font=18
    combs = list(product(loads,temps,fibers))
    mean = []
    var = []

    for load,temp,fiber in combs:
        timeseries,n_fibers,aval = read_h5(fibers=fiber,
                                           load=load,
                                           temperature=temp,
                                           k=1,
                                           distribution="identical",
                                           subset=None,
                                           h5file="fiber-bundles.h5")

        # extract lifetimes
        lifetimes = np.array([t[0][-1] for t in timeseries])

        #
        mean.append(np.mean(lifetimes))
        var.append(np.var(lifetimes))

    combs = np.array(combs).astype(float)

    #
    _loads,_temps,_fibers = combs[:,0],combs[:,1],combs[:,2]
    mean,var = np.array(mean), np.array(var)

    #
    N = np.arange(1,np.max(fibers)*5 + 1)

    # create two fig
    fig,axs = plt.subplots(1,2,figsize=(12, 8), 
                           sharex=True)

    # each temperature-load combination gets its own color
    i = 0
    for load,temp in product(loads,temps):

        # create mask
        mask = (_temps == temp) & (_loads==load)

        # plot data
        axs[0].scatter(_fibers[mask],mean[mask],color=colors(i))
        axs[1].scatter(_fibers[mask],var[mask],color=colors(i))

        # exact calculation as - line
        axs[0].plot(N,homog_lifetime_mean(f0=load,T=temp,N=N,t=1,
                                        approximations=None),color=colors(i),
                  label=str(temp)+", "+str(load))
        axs[1].plot(N,homog_lifetime_var(f0=load,T=temp,N=N,t=1,
                                        approximations=None),color=colors(i),
                  label=str(temp)+", "+str(load))

        # approximation N>> and F>>t as horizontal dotted lines
        axs[0].axhline(y=homog_lifetime_mean(f0=load,T=temp,N=N,t=1,
                                        approximations=['N>>','F>>t']),
                       xmin=0,xmax=np.max(N),
                     color=colors(i),linestyle=':')
        axs[1].plot(N,homog_lifetime_var(f0=load,T=temp,N=N,t=1,
                                        approximations=['N>>','F>>t']),
                    color=colors(i),linestyle=':')

        i += 1

    #
    axs[0].set_xlim(left=0.5,right=np.max(fibers)*5)
    #axs[0].set_ylim(bottom=0)
    axs[1].set_xlim(left=0.5,right=np.max(fibers)*5)
    #axs[1].set_ylim(bottom=0)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[0].set_xlabel("N",fontsize=font)
    axs[1].set_xlabel("N",fontsize=font)
    axs[0].set_ylabel(r"$\langle\tau\rangle$",fontsize=font)
    axs[1].set_ylabel(r"$\sigma_{\tau}^{2}$",fontsize=font)
    #
    axs[0].set_xticks(fibers)

    axs[1].legend(loc='center left',
         bbox_to_anchor=(1, 0.5),
         fontsize=font,
         title=r"T, $f_{0}$",
         title_fontsize=font,
         frameon=False,
         borderpad=0)

    axs[0].tick_params(axis='both', which='both', labelsize=font)
    axs[1].tick_params(axis='both', which='both', labelsize=font)
    plt.subplots_adjust(wspace=0.35)

    plt.savefig("moment-comparison_identical.pdf",
                format="pdf",
                bbox_inches="tight")

    return

from scipy.special import expi

def homog_lifetime_mean(f0,T,N,t=1,
                        approximations=None):
    """
    Calculate average lifetime with Equations. 13, 15 and 17 in the paper 
    depending on the chosen approximation.

    Parameters
    ----------
    f0 : float
        load per fiber (f0 in paper).
    T : float
        temperature (symbol T in paper).
    N : np.ndarray
        integers for number of fibers (symbol N in paper). Must be of dimension 
        0 or 1
    t : float
        value of threshold
    approximations : list or None
        "N>>" and "F>>t" possible entries

    Returns
    -------
    means : np.ndarray
        average lifetime
    """

    #
    if len(np.shape(N))==0:
        N = np.array([N])
    elif len(np.shape(N))==1:
        pass
    else:
        raise ValueError("N has inadequate shape: ",np.shape(N))

    # number of events
    N_av = np.ceil(N*(1 - f0/t))

    # this limit ignores N and N_av
    if approximations is not None:
        if "N>>" in approximations and "F>>t" in approximations:
            return np.exp(t/T)*(expi(-t/T)-expi(-f0/T))

    #
    mean = []
    for _N,_N_av in zip(N,N_av):
        n = np.arange(_N-_N_av+1,_N+1)

        if approximations is None:
            mean.append(np.exp(t/T)*np.sum(np.exp(-_N*f0/(n*T))/n))

        elif "N>>" in approximations and "F>>t" not in approximations:
            mean.append(np.exp(t/T)*(expi(-(_N*f0*t)/((_N*f0+t)*T) )-expi(-f0/T)))
        else:
            return ValueError(approximations)

    return np.array(mean)

def homog_lifetime_var(f0,T,N,t=1,
                        approximations=None):
    """
    Calculate variance of the lifetime with Equations. 14, 16 and 18 in the 
    paper depending on the chosen approximation.

    Parameters
    ----------
    f0 : float
        load per fiber (f0 in paper).
    T : float
        temperature (symbol T in paper).
    N : np.ndarray
        integers for number of fibers (symbol N in paper). Must be of dimension 
        0 or 1
    t : float
        value of threshold
    approximations : list or None
        "N>>" and "F>>t" possible entries

    Returns
    -------
    variances : np.ndarray
        average lifetime
    """

    #
    if len(np.shape(N))==0:
        N = np.array([N])
    elif len(np.shape(N))==1:
        pass
    else:
        raise ValueError("N has inadequate shape: ",np.shape(N))

    # number of events
    N_av = np.ceil(N*(1 - f0/t))

    #
    var = []

    for _N,_N_av in zip(N,N_av):
        n = np.arange(_N-_N_av+1,_N+1)

        if approximations is None:
            var.append(np.exp(2*t/T)*np.sum(np.exp(-2*_N*f0/(n*T))/(n**2)))

        elif "N>>" in approximations and "F>>t" not in approximations:
            var.append(T/(2*_N*f0)*np.exp(2*t/T)*(np.exp(-2*f0/T)-np.exp(-2*(_N*f0*t)/((_N*f0+t)*T))))

        elif "N>>" in approximations and "F>>t" in approximations:
            var.append(T/(2*_N*f0)*np.exp(2*t/T)*(np.exp(-2*f0/T)-np.exp(-2*t/T)))
        else:
            return ValueError(approximations)

    return np.array(var)

if __name__ == "__main__":
    
    moment_plot()
