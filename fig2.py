from multiprocessing import Pool
from functools import partial

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from scipy.linalg import expm,eig,inv,solve_triangular

import matplotlib.pyplot as plt

from lifetime_distribution import create_phasetype
from read_h5 import read_h5

def cond_heterog_lifetime(x,
                          alpha,
                          S,
                          s0=None,
                          k = None,
                          quantity="pdf",
                          mode="dense-brute"):
    """
    Calculate Eq. 24 in the paper. We evaluate it as a phase-type distribution.
    If one wants to understand the inner workings of this code, one should 
    first familiarize itself with the section characterization of
    
    https://en.wikipedia.org/wiki/Phase-type_distribution
        
    Parameters
    ----------
    x : np.ndarray
        threshold values.
    alpha : np.ndarray
        probability row vector of a phase-type distribution.
    S : np.array or scipy.sparse.matrix 
        subgenerator matrix of a phase-type distribution. 
        shape (nstates,nstates).
    s0 : np.ndarray or scipy.sparse.matrix.
        sum over rows of S multiplied by (-1).
    k : None or int
        number of eigenvalues to approximate matrix exponential. If None, all
        eigenvalues are calculated, so it is exact.
    quantity : str 
        either "pdf" or "cdf" which will return either cumulative prob. 
        function or the probability density.
    mode : str
        different ways of calculation. Only these have been tested to give 
        consistent results: "dense-eig-square","dense-eig-triangular",
        "sparse-brute","sparse-eig"

    Returns
    -------
    pdf or cdf : np.ndarray
        probability density function or cumul. prob. function of lifetimes.

    """

    if s0 is None and quantity == "pdf":
        s0 = -S.sum(axis=1)
    elif s0 is not None and quantity != "pdf":
        s0 = None

    if quantity not in ["pdf","cdf"]:
        raise NotImplementedError("Quantity not implmeneted:",quantity)
    if k is None:
        k = S.shape[0]

    # eigendecomposition
    if "dense" in mode:

        if sparse.issparse(S):
            S = S.toarray()

        if s0 is not None:
            if sparse.issparse(s0) and "brute" in mode:
                s0 = s0.toarray()
            elif sparse.issparse(s0) and "eig" in mode:
                s0 = s0.toarray()#.flatten()

        if sparse.issparse(alpha):
            if sparse.issparse(alpha) and "brute" in mode:
                alpha = alpha.toarray()
            elif sparse.issparse(alpha) and "eig" in mode:
                alpha = alpha.toarray()#.flatten()

    elif "sparse" in mode:
         if not sparse.issparse(S):
             raise TypeError("S not sparse.")
         if s0 is not None:
             if not sparse.issparse(s0):
                 raise TypeError("s0 not sparse.")
         if not sparse.issparse(alpha):
             raise TypeError("alpha not sparse.")

    if mode == "dense-brute":

        with Pool() as p:
            if quantity == "pdf":
                res = p.map(partial(_pdf_dense_brute,
                                    alpha=alpha,S=S,s0=s0),
                            x)
            elif quantity == "cdf":
                res = p.map(partial(_cdf_dense_brute,
                                    alpha=alpha,S=S),
                            x)

        return np.array(res)

    elif "dense-eig" in mode:

        # eigenvalues, eigenvectors
        if "square" in mode:
            D,U = eig(S)
            print("Eigendecomposition done.")
            U_inv = inv(U)
            print("Inversion done.")
        elif "triangular" in mode:
            print()
            D,U = eig_triangular(S,S.shape[0],order="largest")
            print("Eigendecomposition done.")
            U_inv = inv(U)
            #D,U_inv = eig_triangular(S,S.shape[0],order="largest",
            #                         right=False,left=True)
            #U_inv = np.flip(U_inv,axis=1)
            print("Inversion done.")
        else:
            raise ValueError("mode does not define mode of eigenvalue decomposition.")

        res = []
        if quantity == "pdf":
            for _x in x:
                res.append((alpha.T@U@(np.exp(D*_x)*np.eye(D.shape[0]))@U_inv@s0)[0,0])
        elif quantity == "cdf":
            for _x in x:
                res.append((1 - alpha.T@U@(np.exp(D*_x)*np.eye(D.shape[0]))@U_inv@np.ones(S.shape[0]))[0])

        return np.array(res)

        #return (np.ravel(alpha.T.dot(U)) * np.exp(x[:,None]*D[None,:])).dot(U.T.dot(s0))
    elif "sparse-brute" in mode:

        if quantity == "pdf":
            with Pool() as p:
                res = p.map(partial(_pdf_sparse_brute,
                                    alpha=alpha,S=S,s0=s0),
                            x)
        elif quantity == "cdf":
            with Pool() as p:
                res = p.map(partial(_cdf_sparse_brute,
                                    alpha=alpha,S=S),
                            x)

        return np.array(res)

    elif "sparse-eig" in mode:

        # eigenvalues, eigenvectors
        D,U = eig_triangular(S,S.shape[0],order="largest")
        print("Eigendecomposition done.")
        #D,U_inv = eig_triangular(S,S.shape[0],order="largest",
        #                         right=False,left=True)
        U_inv = sparse.linalg.inv(U)
        print("Inversion done.")

        res = []
        if quantity == "pdf":
            for _x in x:
                res.append(alpha.T@U@sparse.diags(np.exp(D*_x),format='csc')@U_inv@s0)
        elif quantity == "cdf":
            for _x in x:
                res.append(1 - (alpha.T@U@sparse.diags(np.exp(D*_x),format='csc')@U_inv).sum(axis=1))

        return np.array([r[0,0] for r in res])

    else:
        raise NotImplementedError("Mode not known")

def _pdf_dense_brute(_x,alpha,S,s0):
    return alpha.T.dot(expm(_x * S)).dot(s0)[0,0]

def _pdf_sparse_brute(_x,alpha,S,s0):
    return alpha.T.dot(linalg.expm(_x * S)).dot(s0)[0,0]

def _cdf_dense_brute(_x,alpha,S):
    return 1-alpha.T.dot(expm(_x * S)).sum()

def _cdf_sparse_brute(_x,alpha,S):
    return 1-alpha.T.dot(linalg.expm(_x * S)).sum()

def eig_triangular(a,k,order,right=True,left=False):
    """
    a: scipy sparse matrix, size (n,n)
    k: number of eigenvectors desired
    order: str, either "smallest" or "largest"
    """

    if left and right:
        raise ValueError("Left and right cannot be True at the same time.")
    elif not left and not right:
        raise ValueError("Left and right cannot be False at the same time.")

    # extract eigvalues
    if sparse.issparse(a):
        eigvalues = np.asarray(a[np.arange(a.shape[0]),np.arange(a.shape[0])])[0]
    else:
        eigvalues = a[np.arange(a.shape[0]),np.arange(a.shape[0])]

    # get positions of eigenvalues
    if order == "largest":
        indices = np.flip(np.argpartition(eigvalues,-k)[-k:][np.argsort(eigvalues)])

    elif order == "smallest":
        indices = np.argpartition(eigvalues,k-1)[:k][np.argsort(eigvalues)]

    else:
        raise ValueError("Unknown order:",order)

    #
    eigvecs = []
    for i in indices:

        # setup matrix that has to be solved
        if right:
            _a = a[:i+1,:i+1].copy()

            # set last row to one in order to avoid a singular matrix
            _b = np.zeros(i+1)
            _a[i,i],_b[i] = 1,1

            # calculate eigenvectors and normalize
            if sparse.issparse(a):
                eigvec = linalg.spsolve_triangular(_a-sparse.eye(_a.shape[0])*a[i,i],
                                                   b=_b,
                                                   lower=False)
                eigvec = sparse.hstack([sparse.csc_matrix(eigvec),
                                        sparse.csc_matrix((1,a.shape[0]-_a.shape[0]))])
                eigvecs.append(eigvec/linalg.norm(eigvec))

            else:
                eigvec = solve_triangular(_a-np.eye(_a.shape[0])*a[i,i],
                                          _b,
                                          lower=False)
                eigvecs.append(np.append(eigvec/np.linalg.norm(eigvec),
                                         np.zeros(a.shape[0]-_a.shape[0])))
        elif left:
            _a = a.T[-a.shape[0]+i:,-a.shape[0]+i:].copy()

            # set first row to one in order to avoid a singular matrix
            _b = np.zeros(a.shape[0]-i)
            _a[0,0],_b[0] = 1,1

            # calculate eigenvectors and normalize
            if sparse.issparse(a):
                eigvec = linalg.spsolve_triangular(_a-sparse.eye(_a.shape[0])*a[i,i],
                                                   _b,
                                                   lower=True)

                eigvec = sparse.hstack([sparse.csc_matrix((1,a.shape[0]-_a.shape[0])),
                                        sparse.csc_matrix(eigvec)])
                eigvecs.append(eigvec/linalg.norm(eigvec))

            else:
                eigvec = solve_triangular(_a-np.eye(_a.shape[0])*a[i,i],
                                          _b,
                                          lower=True)
                eigvecs.append(np.append(np.zeros(a.shape[0]-_a.shape[0]),
                                         eigvec/np.linalg.norm(eigvec)))

        else:
            raise ValueError("Either left or right must be true.")

    # return in same format as scipy.sparse.linalg.eigs
    if sparse.issparse(a):
        return eigvalues[indices],sparse.vstack(eigvecs).T
    else:
        return eigvalues[indices],np.vstack(eigvecs).T

def test_cond_heterog_lifetime():
    """
    Simple assert that check that different ways of calculating the 
    probability distribution function give the same result.
    """

    methods = ["dense-eig-square","dense-eig-triangular",
               "sparse-brute","sparse-eig"]

    load = 0.125
    fibers = 5
    temp = 0.05
    np.random.seed(0)
    t = np.random.rand(fibers)
    t.sort()

    S, s0, alpha = create_phasetype(t = t,
                                    load=load,
                                    temp=temp,
                                    debug=False)

    x = np.linspace(0,1000,1000)

    try:

        benchmark = cond_heterog_lifetime(x,alpha,S,s0=s0,
                                          mode="dense-brute",
                                          quantity="pdf")

        for method in methods:
            pred = cond_heterog_lifetime(x,alpha,S,s0=s0,mode=method,
                                         quantity="pdf")
            assert np.allclose(benchmark,pred)

    except AssertionError as err:
        print("pdf failed with this method: ",method,"\n")
        print(err,"\n")
        print(np.column_stack((benchmark,pred)))
        return

    try:

        benchmark = cond_heterog_lifetime(x,alpha,S,s0=s0,
                                          mode="dense-brute",
                                          quantity="cdf")

        for method in methods:
            pred = cond_heterog_lifetime(x,alpha,S,s0=s0,mode=method,
                                         quantity="cdf")
            assert np.allclose(benchmark,pred)

    except AssertionError as err:
        print("cdf failed with this method: ",method,"\n")
        print(err,"\n")
        print(np.column_stack((benchmark,pred)))
        return

    print("Test passed!")

    return

def plot_conditionallifetime(load=0.175,temp=0.15,fibers=5,
                             mode="pdf"):
    """
    Plots Fig. 3 in paper for 10 different realizations.

    Parameters
    ----------
    load : float
        load per fiber (f0 in paper).
    temp : float
        temperature (symbol T in paper).
    fibers : int
        number of fibers. 
    mode : str
        either "pdf" or "cdf". Only the latter is shown in the paper.

    Returns
    -------
    None

    """

    thresholds,timeseries,n_fibers,aval = read_h5(fibers=fibers,
                                                  load=load,
                                                  temperature=temp,
                                                  k=1,
                                                  h5file="fiber-bundles.h5",
                                                  thresh=True,
                                                  distribution="uniform",
                                                  subset=10)

    font=35

    # exclude samples which have immediatly failed
    for s in np.arange(0,10):
        if timeseries[s][0][-1] == 0:
            continue

        S, s0, alpha = create_phasetype(t = thresholds[s],
                                        load=load,
                                        temp=temp,
                                        debug=False)

        # extract heterog_lifetimes
        heterog_lifetimes = np.array([t[-1] for t in timeseries[s]])

        fig,ax = plt.subplots(1,1,figsize=(15, 10))
        if mode == "pdf":
            x = np.linspace(0,np.max(heterog_lifetimes),1000)

            ax.hist(heterog_lifetimes,bins=500,density=True)
            ax.plot(x,cond_heterog_lifetime(x, alpha = alpha,S = S,s0=s0,
                                            mode="sparse-eig",
                               quantity="pdf"),
                    color='k')

            ax.set_ylabel(r"p($\tau$)",fontsize=font)
            ax.set_ylim(bottom=0)

        elif mode == "cdf":
            x = np.linspace(0,np.max(heterog_lifetimes),1000)

            ax.hist(heterog_lifetimes,bins=heterog_lifetimes.shape[0],
                    density=True,cumulative=True)
            ax.plot(x,cond_heterog_lifetime(x, alpha = alpha,S = S,
                                            mode="sparse-eig",quantity="cdf"),
                    color='k',
                    lw=5,linestyle="--",)

            ax.set_ylabel(r"P($\tau$)",fontsize=font)
            ax.set_ylim(0,1.05)

        ax.set_xlim(0,np.max(x))

        ax.set_xlabel(r"$\tau$",fontsize=font)
        ax.tick_params(axis='both', which='both', labelsize=font)

        param = '-'.join(["fibers",str(fibers),"i",str(load),"temp",str(temp)])
        plt.savefig(mode+"-comparison_"+param+"_"+str(s)+".pdf",
                    format="pdf",
                    bbox_inches="tight")
        #plt.show()
    return

if __name__ == "__main__":
    #test_cond_heterog_lifetime()
    plot_conditionallifetime(load=0.175,temp=0.15,fibers=10,mode="cdf")
