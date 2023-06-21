import numpy as np
import os

def get(*args):

    for arg in args:
        if arg == 'DBSI_99':
            return _DBSI_99() 
        elif arg == 'ABCD_102':
            return _ABCD_102()
        elif arg == 'NODDI':
            return _NODDI()


def _DBSI_99():
    path, _ = os.path.split(os.path.realpath(__file__)) 
    
    try:
        bvals = np.loadtxt(path + os.sep + 'bval99')
        bvecs = np.loadtxt(path + os.sep + 'bvec99').T

    except:
        bvals = np.loadtxt(path + os.sep + 'bval99', delimiter=',')
        bvecs = np.loadtxt(path + os.sep + 'bvec99', delimiter=',').T

    div_safe = [not np.all(bvecs[i,:] == 0) for i in range(bvecs.shape[0])]
    bvecs[div_safe] = bvecs[div_safe] / np.linalg.norm(bvecs[div_safe], ord = 2, axis = 1)[:, None]

    return bvals, bvecs


def _ABCD_102():
    path, _ = os.path.split(os.path.realpath(__file__)) 
    
    try:
        bvals = np.loadtxt(path + os.sep + 'bval_ABCD')
        bvecs = np.loadtxt(path + os.sep + 'bvec_ABCD').T

    except:
        bvals = np.loadtxt(path + os.sep + 'bval_ABCD', delimiter=',')
        bvecs = np.loadtxt(path + os.sep + 'bvec_ABCD', delimiter=',').T

    div_safe = [not np.all(bvecs[i,:] == 0) for i in range(bvecs.shape[0])]
    bvecs[div_safe] = bvecs[div_safe] / np.linalg.norm(bvecs[div_safe], ord = 2, axis = 1)[:, None]

    return bvals, bvecs

def _NODDI():
    path, _ = os.path.split(os.path.realpath(__file__)) 
    
    try:
        bvals = np.loadtxt(path + os.sep + 'bval_NODDI')
        bvecs = np.loadtxt(path + os.sep + 'bvec_NODDI').T

    except:
        bvals = np.loadtxt(path + os.sep + 'bval_NODDI', delimiter=',')
        bvecs = np.loadtxt(path + os.sep + 'bvec_NODDI', delimiter=',').T

    div_safe = [not np.all(bvecs[i,:] == 0) for i in range(bvecs.shape[0])]
    bvecs[div_safe] = bvecs[div_safe] / np.linalg.norm(bvecs[div_safe], ord = 2, axis = 1)[:, None]

    return bvals, bvecs



