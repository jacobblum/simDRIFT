import numpy as np
import os
import cupy as cp


def _dbsi_99():
    path, fname = os.path.split(os.path.realpath(__file__)) 
    
    try:
        bvals = np.loadtxt(path + os.sep + 'bval99')
        bvecs = np.loadtxt(path + os.sep + 'bvec99').T


    except:
        bvals = np.loadtxt(path + os.sep + 'bval99', delimiter=',')
        bvecs = np.loadtxt(path + os.sep + 'bvec99', delimiter=',').T


 
    div_safe = [not np.all(bvecs[i,:] == 0) for i in range(bvecs.shape[0])]
    bvecs[div_safe] = bvecs[div_safe] / np.linalg.norm(bvecs[div_safe], ord = 2, axis = 1)[:, None]

    return bvals, bvecs



