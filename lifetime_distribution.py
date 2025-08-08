from itertools import combinations,chain

import numpy as np
from scipy.sparse import lil_matrix
from scipy.special import comb

def create_phasetype(t,load,temp,debug=False):
    """
    Calculate Eq. 24 in the paper. We evaluate it as a phase-type distribution.
    If one wants to understand the inner workings of this code, one should 
    first familiarize itself with the section characterization of
    
    https://en.wikipedia.org/wiki/Phase-type_distribution
        
    Parameters
    ----------
    t : np.ndarray
        threshold values.
    load : float
        load per fiber (f0 in paper).
    temp : float
        temperature (symbol T in paper).
    debug : bool
        if True, print out additional information for debugging.

    Returns
    -------
    S : scipy.sparse.matrix 
        subgenerator matrix of a phase-type distribution. 
        shape (nstates,nstates).
    s0 : np.ndarray or scipy.sparse.matrix.
        sum over rows of S multiplied by (-1).
    alpha : np.ndarray
        probability row vector of a phase-type distribution.
    """

    # initial number of fibers
    nfibers = t.shape[0]
    indices = np.arange(nfibers)

    # calculate load after each event
    loads = load * nfibers/np.arange(nfibers,0,-1)

    if debug:
        print("t",t)
        print("loads",loads)

    # determine when each fiber will have to fail
    fail_stage = np.digitize(t,bins = loads)
    fstage,nfail = np.unique(fail_stage,return_counts=True)
    nfail = np.cumsum(nfail)

    # convert nfail to an array of same shape as t
    _nfail = np.zeros(t.shape)
    for i in np.arange(nfail.shape[0]-1):
        _nfail[fstage[i]:fstage[i+1]] = nfail[i]
    _nfail[fstage[-1]:nfibers] = nfail[-1]

    nfail = _nfail.copy().astype(int)
    del _nfail

    # mask used to kick out all which have failed in the initial big break
    mask = nfail < np.arange(1,nfibers+1)

    if not np.any(mask):
        return None,None,None

    #
    stages = np.arange(0,nfibers)[mask]
    n_pos = (nfibers - nfail[mask]).astype(int)
    nstates = comb(n_pos,nfibers - stages).astype(int)
    cum_nstates = np.cumsum(nstates).astype(int)

    if debug:
        print("fail_stage:",fail_stage)
        print("fstage:",fstage)
        print("nfail:",nfail)
        print("mask",mask)
        print("stages",stages)
        print("n_pos",n_pos)
        print("nstates",nstates)

    # concstruct transition matrix. The +1 is the absorbing state.
    Q = lil_matrix((int(cum_nstates[-1]+1),int(cum_nstates[-1]+1)))

    #
    for i in np.arange(stages.shape[0]):
        if debug:
            print("##############################################################\n",
                  i+1,stages[-1]+1)
        #
        draw = nfibers - stages[i]

        ind_comb = np.fromiter(chain.from_iterable(combinations(
                                           indices[-n_pos[i]:],
                                           draw)),
                                         int,
                                         count=nstates[i]*draw).reshape(-1, draw)

        # calculate rates
        rates = (np.exp((loads[stages[i]] - t[ind_comb])/temp)).flatten()

        if debug:
            print("ind_comb",ind_comb)
            print("draw",draw)

        # determine wether after time event avalanche is triggered.
        # Then we need to determine to which superstate the transition jumps.
        if stages[i] != stages[-1]:

            # generate index combinations as if all events would happen
            # without triggering any immediate fails
            jump_comb = np.hstack([np.delete(ind_comb,j,axis=1)[:,None,:] \
                                  for j in np.arange(draw)])
            jump_comb = jump_comb.reshape((nstates[i]*draw,jump_comb.shape[2]))

            # check for immediate fails of the newly generated combinations
            no_break = t[jump_comb] > loads[stages[i]+1:]

            # determine wether any combination leads to complete failure
            broken = ~np.any(no_break,axis=1)

            # technically it would be better to bifurcate here between
            # combinations that fail immediatly and the ones that don't

            # count number of immediate breaks
            nbreaks = np.argmax(no_break,axis=-1)

            if debug:
                print("jump_comb",jump_comb)
                print("no_break",no_break)
                print("broken",broken)
                print("nbreaks",nbreaks)

            # generate index combinations after immediate breaks
            jump_comb = [jump_comb[j,nbreaks[j]:] for j in np.arange(nstates[i]*draw)]

            # get column index. different ks and
            k = [None if broken[j] else jump_comb[j].shape[0] for j in np.arange(nstates[i]*draw)]
            index = [None if broken[j] else np.where(stages==nfibers-k[j])[0][0]\
                     for j in np.arange(nstates[i]*draw)]
            n = [None if broken[j] else n_pos[index[j]] for j in np.arange(nstates[i]*draw)]

            if debug:
                print("jump_comb",jump_comb)
                print("k",k)
                print("index",index)
                print("n",n)

            cols = [cum_nstates[-1] if broken[j] else \
                    get_comb_index(jump_comb[j],n[j],k[j],nfibers) \
                    for j in np.arange(nstates[i]*draw)]

            cols = np.array([int(cols[j]) if broken[j] else \
                             int(cols[j]+cum_nstates[index[j]-1]) \
                             for j in np.arange(nstates[i]*draw)])

        else:
            cols = np.full(rates.shape,cum_nstates[-1])

        #
        if i == 0:
            rows = np.arange(nstates[i]).repeat(draw)
        else:
            rows = np.arange(nstates[i]).repeat(draw) + cum_nstates[i-1]

        # fill Markov transition matrix
        for row,col,rate in zip(rows,cols,rates):
            Q[row,col] += rate

        if debug:
            print("rows",rows)
            print("cols",cols)
            print("rates",rates)

    # sum over rows to construct diagonal
    Q.setdiag(-Q.sum(axis=1))

    # draw S and s0 from Q
    S = Q[:cum_nstates[-1],:cum_nstates[-1]].tocsc()
    s0 = Q[:cum_nstates[-1],-1].tocsc()
    # draw S0

    # set up alpha
    alpha = lil_matrix((cum_nstates[-1],1))
    alpha[0,0] = 1
    alpha.tocsc()

    if debug:
        print("S\n",S,"\n")
        print("s0\n",s0,"\n")
        print("alpha\n",alpha,"\n")

    return S, s0, alpha

def get_comb_index(c,n,k,nfibers):
    """
    c: np.array, combination of indices starting from 0
    n: scalar
    k: scalar
    """

    c = c - nfibers + n
    if len(c.shape)==1:
        return comb(n,k) - comb(n-c-1,np.arange(k,0,-1)).sum()-1
    elif len(c.shape)==2:
        return comb(n,k) - comb(n-c-1,np.arange(k,0,-1)).sum(axis=1)-1
    else:
        raise NotImplementedError
