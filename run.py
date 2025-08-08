import os
from multiprocessing import Pool
from functools import partial
from itertools import product

import numpy as np
import h5py

from bundle_creep import fiber_bundle_creep

def submit_jobs(nbundles,
                dists,
                fibers,
                ks,
                temperatures,
                loads,
                h5file="fiber-bundles.h5",
                n_samples=1,
                offset = 0,
                save_thresholds=False,
                _seed = 0):
    """
    Perform creep fiber bundle simulations for all combinations of parameters.
    
    Parameters
    ----------
    nbundles : int
        number of different threshold realizations
    dists : list
        list of distribution names. Currently only "uniform", "weibull" and 
        "identical" possible.
    fibers : list or np.ndarray 
        number of fibers in fiber bundle.
    ks : list or np.ndarray
        distribution parameter. For "uniform" it is the width of distribution, 
        for "weibull" the exponent (unit mean) and "identical" the threshold 
        value.
    temperatures : list 
        list of temperatures (symbol T in paper).
    loads : list 
        list of loads per fiber (symbol f0 in paper).
    h5file : str
        name of h5file where to store results
    n_samples : int
        number of samplings performed on each fiber bundle.
    offset : int
        number of seeds discarded as they've already been done
    _seed : int
        seed for the random number generator
    
    Returns
    -------
    None
    """

    # set seed of random number generator
    np.random.seed(_seed)

    # sample seeds
    seeds = np.random.randint(0,99999999,int(nbundles+offset))
    seeds = seeds[offset:].tolist()

    # check if hdf5 file exists
    if not os.path.isfile(h5file):
        with h5py.File(h5file,'w') as f:
            pass

    # iterate over different parameters
    for dist,fib,k,t,i in product(dists,fibers,ks,temperatures,loads):

        #
        prefix = os.path.join(dist, "fibers-"+str(fib), "k-"+str(k),
                              "t-"+str(t), "i-"+str(i))
        print(dist,fib,k,t,i)

        with Pool() as p:
            data = p.map(partial(fiber_bundle_creep,
                                 n_fibers=fib,
                                 n_samples=n_samples,
                                 load=i,
                                 temperature=t,
                                 k=k,
                                 distribution=dist),
                                 seeds)

        # extract time and n_fibers
        thresholds, time, n_fibers, length = [], [], [], []
        [(thresholds.append(th),time.append(ti),n_fibers.append(n),
          length.append(l)) for th,ti,n,l in data]

        # accumulate data
        thresholds = np.hstack(thresholds)
        time = np.hstack(time)
        n_fibers = np.hstack(n_fibers)
        length = np.hstack(length)
        seeds = np.array(seeds)

        # create parameter hdf5 group if it does not exist
        with h5py.File(h5file,'r+') as f:

            if prefix in f:

                if save_thresholds:
                    # save thresholds
                    old = f[prefix+'/thresholds'].shape[0]
                    f[prefix+'/thresholds'].resize((int(old+thresholds.shape[0])),
                                                   axis=0)
                    f[prefix+'/thresholds'][-thresholds.shape[0]:] = thresholds[:,None]

                # save times
                old = f[prefix+'/time'].shape[0]
                f[prefix+'/time'].resize((int(old+time.shape[0])),
                                         axis=0)
                f[prefix+'/time'][-time.shape[0]:] = time[:,None]

                # save number fo fibers still unbroken
                old = f[prefix+'/n_fibers'].shape[0]
                f[prefix+'/n_fibers'].resize((int(old+n_fibers.shape[0])),
                                             axis=0)
                f[prefix+'/n_fibers'][-n_fibers.shape[0]:] = n_fibers[:, None]

                # save length of timeseries
                old = f[prefix+'/length'].shape[0]
                f[prefix+'/length'].resize((int(old+length.shape[0])),
                                           axis=0)
                f[prefix+'/length'][-length.shape[0]:] = length[:,None]

                # save seeds
                old = f[prefix+'/seeds'].shape[0]
                f[prefix+'/seeds'].resize((int(old+seeds.shape[0])), axis=0)
                f[prefix+'/seeds'][-seeds.shape[0]:] = seeds[:,None]

            else:

                if save_thresholds:
                    # save thresholds
                    f.create_dataset(prefix+"/thresholds",
                                    data = np.expand_dims(thresholds,axis=-1),
                                    maxshape = (None, 1),
                                    dtype = 'float32',
                                    compression="gzip",
                                    compression_opts=9,
                                    chunks=True)

                # save times
                f.create_dataset(prefix+"/time",
                                data = np.expand_dims(time,axis=-1),
                                maxshape = (None, 1),
                                dtype = 'float32',
                                compression="gzip",
                                compression_opts=9,
                                chunks=True)

                # save number fo fibers still unbroken
                f.create_dataset(prefix+"/n_fibers",
                                data = np.expand_dims(n_fibers,axis=-1),
                                maxshape = (None, 1),
                                dtype = 'int32',
                                compression="gzip",
                                compression_opts=9,
                                chunks=True)

                # save length of timeseries
                f.create_dataset(prefix+"/length",
                                data = np.expand_dims(length,axis=-1),
                                maxshape = (None, 1),
                                dtype = 'int32',
                                compression="gzip",
                                compression_opts=9,
                                chunks=True)

                # save seeds
                f.create_dataset(prefix+"/seeds",
                                data = seeds[:,None],
                                maxshape = (None, 1),
                                dtype = 'int64',
                                compression="gzip",
                                compression_opts=9,
                                chunks=True)

    return

if __name__ == "__main__":

    submit_jobs(nbundles = 10000,
                dists = ["uniform"],
                fibers = [2,3,4,5,6,7,8,9],
                ks = [1],
                temperatures = [0.05,0.1,0.15],
                loads = [0.125,0.15,0.175],
                h5file="fiber-bundles.h5",
                n_samples=1,
                offset = 0 ,
                _seed = 0)
