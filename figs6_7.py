import os
import json
from itertools import product

import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from read_h5 import read_h5

def plot_aval_distribution(fibers = [100000],
                           loads=[0.125,0.15,0.175],
                           temperatures=[0.05,0.1,0.15],
                           ks=[1],
                           distributions=["uniform"],
                           individ_mode="hist",
                           basis=1.125,
                           power_exp=None):
    """
    Reads the avalanche data previously created by get_aval_count and creates 
    Figures like Fig. 6 and 7 in the paper while fitting the curves.
    
    Parameters
    ----------
    fibers : list
        number of fibers. 
    loads : list
        load per fiber (f0 in paper).
    temps : list
        temperature (symbol T in paper).
    ks : list
        distribution parameter. For weibull the exponent,
        for uniform the span.
    distributions : list 
        list of names of distributions. Currently only "weibull" and "uniform"
        possible.
    individ_mode : str
        whether to plot curves as scatter plot or logarithmically binned 
        histogram. In paper "hist" was used.
    basis : float 
        size of logarithmic bin.
    power_exp : float or None 
        can be used to fix power law exponent.

    Returns
    -------
    None
    """

    #
    font=28
    small_font=26
    markers = [".","o","^","v","<"]
    linestyles=["-",":","--","-."]
    bound=10

    #
    colors = plt.get_cmap("inferno")

    list_prefac = []
    cs = []
    ns = []

    for distribution,fiber in product(distributions,fibers):

        fig,axs = plt.subplots(len(temperatures),1,
                               sharex=True,sharey=False,
                               figsize=(9,12))

        combinations = list(product(loads,temperatures,ks))
        j = 0
        ncomb = len(combinations)
        for load,t,k in combinations:

            print(f"load: {load}, temp: {t}, k: {k}")

            #
            name='-'.join(["fiber"+str(fiber),"load"+str(load),"temp"+str(t),
                           "k"+str(k),distribution])

            #
            avsize,count = np.split(np.loadtxt("avalanchesize-counts/avalanchesize-count_"+name+".csv",
                                              delimiter=","),
                                   indices_or_sections=2,axis=1)

            # get rid of avalanches of size one
            #avsize, count = avsize[1:,0],count[1:,0]
            probs = count/np.sum(count)

            row = temperatures.index(t)

            #
            if individ_mode=="hist":

                # create logarithmic binning
                maxbin = np.log(np.max(avsize))/np.log(basis)
                bins = np.insert(np.cumsum(basis**np.arange(0,maxbin)),0,0)
                bins = np.min(avsize)+bins
                hist, edges = np.histogram(a=avsize,
                                           bins = bins,
                                           weights=count,density=False)

                # kick out empty bins
                ind = np.argmin(hist)
                if hist[ind] == 0:
                    hist = hist[:ind]
                    edges = edges[:ind+1]

                # normalize by bin width and that everything adds up to one
                hist = hist/(np.ceil(edges[1:])-np.ceil(edges[:-1]))
                hist = hist/np.sum(hist)

                # stuff needed for the scales of the plot later
                if loads.index(load) == 0:
                    min = 1
                    max = 1

                _min = np.min(hist)
                if min > _min:
                    min = _min.copy()

                _max = np.max(edges)
                if max < _max:
                    max = _max

                axs[row].scatter(x = (edges[1:] + edges[:-1])/2,
                                 y = hist,
                                 s=10.,
                                 color = colors(loads.index(load)/(len(loads)-1) * 0.9),
                                 label=str(np.round(load,3)))

                # use only a subportion of the data for fitting
                x = (edges[1:] + edges[:-1])/2
                mask = (x > bound)
                x,hist = x[mask],hist[mask]

                if power_exp is None:
                    x = np.column_stack((x,np.log(x)))
                    y=-np.log(hist)
                else:
                    y = -np.log(hist * x**power_exp)
                    x = x[:,None]

                # precalculate for intercept
                x_offset = np.mean(x,axis=0)
                y_offset = np.average(y, axis=0)

                # solve non negative least squares to get weights
                weights,residual = nnls(x-x_offset,
                                        y-y_offset)
                
                # calculate final parameters
                prefac = y_offset - np.dot(x_offset,weights)
                prefac = np.exp(-prefac)
                if power_exp is None:
                    x = x[:,0]
                    c,n = weights
                else:
                    c,n = weights[0],power_exp
                    
                #
                axs[row].plot(x,prefac*np.exp(-c*x)*x**(-n),
                              color = colors(loads.index(load)/(len(loads)-1) * 0.9),
                              linestyle="--")
                list_prefac.append(prefac),ns.append(n),cs.append(c)
                print(f"prefac.: {prefac}, exp. decay {c}, pow. law exp. {n}\n")

            elif individ_mode=="scatter":

                if loads.index(load) == 0:
                    max = 1

                _max = np.max(avsize)
                if max < _max:
                    max = _max

                axs[row].scatter(avsize,probs,
                                 s=0.5,
                                 label=r"$T$ "+str(t)+r" ,$\sigma_{0}$ "+str(np.round(load,3))) 

            #
            axs[row].set_xscale("log")
            axs[row].set_yscale("log")
            axs[row].set_ylim(top=1.1)
            if individ_mode=="hist":
                axs[row].set_xlim(left=1.,right=max*2)
            elif individ_mode=="scatter":
                axs[row].set_xlim(left=1.,right=max*2)
            else:
                raise ValueError

            if row == len(temperatures)-1:
                axs[row].set_xlabel(r"aval. size $\Delta$",fontsize=font)
            if loads.index(load) == 0:
                axs[row].set_ylabel(r"P($\Delta$)",fontsize=font)

            #
            j += 1
            if j%(len(temperatures)*len(loads))==0 and j!=0:

                # create custom legend
                leg = []
                for l in loads:
                    leg.append(mlines.Line2D([], [],
                                             color=colors(l/(len(loads)-1) * 0.95),
                                             label="$f_{0}$ "+str(np.round(l,3))))
                for temp in temperatures:

                    axs[temperatures.index(temp)].grid()
                    axs[temperatures.index(temp)].set_yticks(np.logspace(-10,0,11)[::2].tolist())
                    axs[temperatures.index(temp)].text(x=0.025,y=0.025,
                                  s=r"$T$ "+str(temp),
                                  transform=axs[temperatures.index(temp)].transAxes,
                                  fontsize=small_font)

                    if temperatures.index(temp) == int(len(temperatures)/2):
                        axs[temperatures.index(temp)].legend(loc='center left',
                             bbox_to_anchor=(1, 0.5),
                             fontsize=font, markerscale=3,
                             title=r"$f_{0}$",
                             title_fontsize=font,
                             frameon=False)

                    axs[temperatures.index(temp)].tick_params(axis='both',
                                                              which='major',
                                                              labelsize=small_font)
                    leg.append(mlines.Line2D([], [],
                               linestyle=linestyles[temperatures.index(temp)],
                               marker=markers[temperatures.index(temp)],
                               color="black",
                               label="T "+str(temp)))

                # save huge plot
                if individ_mode=="hist":
                    plt.savefig("global-aval-dist-fit-hist-basis"+str(basis)+"_dist_"+name+".pdf",
                                format="pdf",bbox_inches="tight")
                elif individ_mode=="scatter":
                    plt.savefig("global-aval-dist-fit-scatter_dist_"+name+".pdf",
                                format="pdf",bbox_inches="tight")
                else:
                    ValueError("individ_mod must be either 'hist' or scatter.load mode is: ",individ_mode)
                plt.show()

    with open("avalanche-parameter.json","w") as f:
        json.dump({"params":combinations,
                   "a":list_prefac,
                   "n":ns,
                   "c":cs},f)

    return

def get_aval_count(fibers = [100000],
                   loads=[0.125,0.15,0.175],
                   temperatures=[0.05,0.1,0.15],
                   ks=[1],
                   distributions=["uniform"],
                   h5file="fiber-bundles.h5"):
    """
    Reads the hdf5 file and extracts the avalanches. Stores information in the 
    directory avalanchesize-counts which is created if it not already exists.
    
    Parameters
    ----------
    fibers : list
        number of fibers. 
    loads : list
        load per fiber (f0 in paper).
    temps : list
        temperature (symbol T in paper).
    ks : list
        distribution parameter. For weibull the exponent,
        for uniform the span.
    distributions : list 
        list of names of distributions. Currently only "weibull" and "uniform"
        possible.
    h5file : str
        hdf5 file in which to find the raw data.

    Returns
    -------
    None
    """

    if not os.path.isdir("avalanchesize-counts"):
        os.mkdir("avalanchesize-counts")

    combinations = list(product(distributions,
                                fibers,
                                ks,
                                temperatures,
                                loads))
    j = 0
    ncomb = len(combinations)
    for distribution,fiber,k,t,load in combinations:
        print(distribution,fiber,k,t,load)
        #
        timeseries,n_fibers,avalanches = read_h5(fiber,load,t,k,distribution,
                                                 h5file,None,False)

        #
        name='-'.join(["fiber"+str(fiber),"load"+str(load),
                       "temp"+str(t),"k"+str(k),distribution])

        # select first resample throw out samples that instantly fail
        avalanches = [a[0] for a in avalanches if a[0].shape[0]!=1]

        # extract
        a0 = np.hstack([aval[0] for aval in avalanches])
        a_inf = np.hstack([aval[-1] for aval in avalanches])
        avalanches = np.hstack([aval[1:-1] for aval in avalanches])

        #
        np.savetxt("avalanchesize-counts/avalanchesize-count_"+name+".csv",
                   np.column_stack((np.unique(avalanches,return_counts=True))),
                   delimiter=",")

        np.savetxt("avalanchesize-counts/a0-count_"+name+".csv",
                   np.column_stack((np.unique(a0,return_counts=True))),
                   delimiter=",")

        np.savetxt("avalanchesize-counts/ainf-count_"+name+".csv",
                   np.column_stack((np.unique(a_inf,return_counts=True))),
                   delimiter=",")

        j +=1
        print(j,ncomb)

    return

if __name__ == "__main__":
    #
    for loads,distributions in zip([(np.array([0.5,0.6,0.7,0.8,0.9])*np.exp(-1)).tolist(), 
                                    [0.125,0.15,0.175,0.2,0.225]],
                                   [ ["weibull"],["uniform"] ]):
        
        get_aval_count(loads=loads, 
                       distributions=distributions)
        plot_aval_distribution(fibers = [100000],
                               loads=loads,
                               temperatures=[0.05,0.1,0.15],
                               ks=[1],
                               distributions=distributions,
                               individ_mode="hist",
                               power_exp=None) # set this to 2.5 to recreate plots in paper.
