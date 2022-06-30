
import numpy as np
from scipy.optimize import nnls 


def nnlsq_model(bvals, signal):
    
    D = np.array([0,.5,1,1.5,2,2.5]) * 10**-3
    Model = np.zeros((bvals.shape[0], len(D)))
    for i in range(len(D)):
        Model[:,i] = np.exp(-D[i] * bvals)
    
    cs = nnls(Model, signal)[0]

    cs = np.array(cs)
    fractions = 1/np.sum(cs) * cs[cs > 0]
    Ds = 1000 * D[cs > 0]

    print('Multi-Tensor')
    print(fractions)
    print(Ds)
    return fractions, Ds


def lstq(bvals, signal):
    A = np.ones((len(bvals),2))
    A[:,1] = -1/1000 * bvals
    cs = nnls(A, np.log(signal))[0]
    print(cs)
    return cs