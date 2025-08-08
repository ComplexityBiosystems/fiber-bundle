import numpy as np
from scipy.special import gamma

def fiber_bundle_creep(seed,
                       n_fibers,
                       n_samples,
                       load,
                       temperature,
                       k,
                       distribution):
    """
    Checks thresholds for critical, immediate failure and deletes thresholds 
    that lie below the load. The load is redistributed until no fibers fail 
    critically anymore and the remaining thresholds as well as the final load 
    are returned.

    Parameters
    ----------
    seed : int 
        seed for random number generator to generate thresholds
    n_fibers : int 
        number of fibers
    n_samples : int 
        number of samples from the same draw of thresholds
    load : float 
        constant load already normalized by inital number of fibers
    temperature : float 
        temperature 
    k: float 
       some parameter for the threshold distribution. For uniform dist. it's
       the width of the distribution, for the weibull distribution it is the
       exponent (the scale parameter is chosen to yield a unity mean).
    distribution: str 
       name of distribution. Currently only "weibull", "uniform" and 
       "identical" are implemented.
      
    Returns
    -------
    thresholds : np.ndarray
        thresholds of fiber bundle with shape (n_fibers).
    fail_time : np.ndarray
        times at which failure events occured.
    nfiber : list
        number of fibers intact at that time.
    length : 
        if nsamples >1, then this is the length for each individual sample 
        trajectory.
    """
    np.random.seed(seed)

    # draw thresholds from uniform distribution in interval [0,1].
    # from that you generate other distributions later on.
    if not distribution == "identical":
        thresholds = np.random.uniform(size=n_fibers)
        thresholds = np.sort(thresholds)
    else:
        thresholds = np.full(n_fibers,k)

    # convert to other distribution
    if distribution == "weibull" and not k is None:
        thresholds = (- np.log(1-thresholds))**(1/k) /gamma(1+1/k)
    elif distribution == "uniform":
        thresholds = k*thresholds
    elif distribution=="identical":
        pass
    else:
        raise ValueError("Unknown distribution:", distribution)

    # fail thresholds below load immediatly until all thresholds above load
    _thresholds, load = immediate_failure(thresholds,load)

    # sample trajectories from the same thresholds multiple times.
    fail_time, nfibers = [],[]
    for j in range(n_samples):

        t,n = trajectory(index=j,
                         thresholds=_thresholds,
                         load=load,
                         temperature=temperature,
                         seed=seed)
        fail_time.append(t),nfibers.append(n)

    # needed later for splitting the individual trajectories
    length = [len(t) for t in fail_time]

    return thresholds, np.hstack(fail_time), np.hstack(nfibers), np.hstack(length)


def trajectory(index,thresholds, load, temperature, seed):
    """
    Checks thresholds for critical, immediate failure and deletes thresholds 
    that lie below the load. The load is redistributed until no fibers fail 
    critically anymore and the remaining thresholds as well as the final load 
    are returned.

    Parameters
    ----------
    index : int 
        index of trajectory, that creates a new seed. This is only important 
        if you want to create different time evolutions for the same set of 
        thresholds like in Fig. 3 of the paper.
    thresholds : np.ndarray
        fiber thresholds.
    load : float
        current load.
    temperature : float
        temperature for the stochastic time evolution.
    seed : int
        random seed for the stochastic time evolution. 
        
    Returns
    -------
    fail_time : np.ndarray
        times at which failure events occured.
    nfiber : list
        number of fibers intact at that time.

    """

    # create seed for the drawing of the failure times
    np.random.seed(seed+index)

    # initialize time
    fail_time = [0]
    n_fibers = [thresholds.shape[0]]

    while thresholds.shape[0] != 0:

        # calculate failure rates
        rate = np.exp(- (thresholds - load)/temperature)
        tot_rate = np.sum(rate)

        # delete random fiber
        thresholds = np.delete(thresholds,
                               np.digitize(np.random.uniform(),
                               bins = np.cumsum(rate)/tot_rate))

        # calculate new load
        try:
            load = load * n_fibers[-1]/thresholds.shape[0]
        except ZeroDivisionError:
            load = np.inf

        # fail thresholds below load immediatly until all thresholds above load
        thresholds, load = immediate_failure(thresholds,
                                 load = load)

        # update time and number of fibers
        fail_time.append(fail_time[-1] + (-1)*np.log(np.random.uniform())/tot_rate)
        n_fibers.append(thresholds.shape[0])

    return fail_time, n_fibers

def immediate_failure(thresholds,load):
    """
    Checks thresholds for critical, immediate failure and deletes thresholds 
    that lie below the load. The load is redistributed until no fibers fail 
    critically anymore and the remaining thresholds as well as the final load 
    are returned.

    Parameters
    ----------
    thresholds : np.ndarray
        thresholds intact so far.
    load : float
        current load.

    Returns
    -------
    thresholds : np.ndarray
        thresholds with failed thresholds excluded.
    load : float
        updated load.

    """

    # check if
    above_load = thresholds > load

    while not np.all(above_load):

        n_old = thresholds.shape[0]

        # sort out all fibers below load/n_fibers
        thresholds = thresholds[above_load]

        # calculate new load
        try:
            load = load * n_old/thresholds.shape[0]
        except ZeroDivisionError:
            load = np.inf

        # check if
        above_load = thresholds > load

    return thresholds, load

if __name__ == "__main__":
    thresholds, t, n_fibers, length = fiber_bundle_creep(seed=1,
                           n_fibers=10000,
                           n_samples=1,
                           load=0.15,
                           temperature=0.1,
                           k=1,
                           distribution='uniform')
