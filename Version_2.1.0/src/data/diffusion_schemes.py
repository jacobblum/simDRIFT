import numpy as np
import os
import logging

DATA_DIR = os.path.dirname(os.path.relpath(__file__)) 

def normalize_bvecs(bvecs):
    div_safe = [not np.all(bvecs[i,:] == 0) for i in range(bvecs.shape[0])]
    bvecs[div_safe] = bvecs[div_safe] / np.linalg.norm(bvecs[div_safe], ord = 2, axis = 1)[:, None]
    return bvecs

def get_from_default(fname): 
    return diff_scheme_opts[fname]()

def get_from_custom(bvals_path, bvecs_path):
    bvals = np.loadtxt(bvals_path)
    bvecs = np.loadtxt(bvecs_path).T
    return bvals, normalize_bvecs(bvecs)

    
def _DBSI_99():

    bvals = np.loadtxt(os.path.join(DATA_DIR, 'bval99'))
    bvecs = np.loadtxt(os.path.join(DATA_DIR, 'bvec99')).T

    return bvals, normalize_bvecs(bvecs)


def _ABCD_102():
   
    bvals = np.loadtxt(os.path.join(DATA_DIR, 'bval_ABCD'))
    bvecs = np.loadtxt(os.path.join(DATA_DIR, 'bvec_ABCD')).T

    return bvals, normalize_bvecs(bvecs)

def _NODDI():

    bvals = np.loadtxt(os.path.join(DATA_DIR, 'bval_NODDI'))
    bvecs = np.loadtxt(os.path.join(DATA_DIR, 'bvec_NODDI')).T

    return bvals, normalize_bvecs(bvecs)

diff_scheme_opts = {'DBSI_99': _DBSI_99,
                    'ABCD_102': _ABCD_102,
                    'NODDI': _NODDI}



