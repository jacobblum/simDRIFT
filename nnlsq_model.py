
import numpy as np
from scipy.optimize import nnls 


def nnlsq_model(bvals, signal):
    num_params = 140
    D = np.linspace(0, 3.5*10**-3, num_params)
    Model = np.zeros((bvals.shape[0], num_params))
    for i in range(num_params):
        Model[:,i] = np.exp(-D[i] * bvals)
    cs = nnls(Model, signal)[0]
    cs = np.array(cs)
    fractions = 1/np.sum(cs) * cs[cs > 0]
    Ds = 1000 * D[cs > 0]

    print('Multi-Tensor')
    print(fractions)
    print(Ds)

    return fractions, Ds
