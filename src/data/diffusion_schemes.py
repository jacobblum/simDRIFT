import numpy as np
import os
import logging

def normalize_bvecs(bvecs):
    """Function to normalize b-vectors (custom or included)

    :param bvecs: b-vectors to normalize
    :type bvecs: np.ndarray
    :return: Normalized b-vectors
    :rtype: np.ndarray
    """    
    div_safe = [not np.all(bvecs[i,:] == 0) for i in range(bvecs.shape[0])]
    bvecs[div_safe] = bvecs[div_safe] / np.linalg.norm(bvecs[div_safe], ord = 2, axis = 1)[:, None]
    return bvecs

def get_from_default(fname): 
    """Returns included or default diffusion scheme files

    :param fname: Input file name
    :type fname: str
    :return: b-values and (normalized) b-vectors
    :rtype: np.ndarray
    """    
    return diff_scheme_opts[fname]()

def get_from_custom(bvals_path, bvecs_path):
    """Retrieves custom b-values and imports and normalizes b-vectors

    :param bvals_path: Path to custom b-values file
    :type bvals_path: str
    :param bvecs_path: Path to custom b-vectors file
    :type bvecs_path: str
    :return: b-values and (normalized) b-vectors
    :rtype: np.ndarray
    """    
    bvals = np.loadtxt(bvals_path)
    bvecs = np.loadtxt(bvecs_path).T
    return bvals, normalize_bvecs(bvecs)

    
def _DBSI_99():
    """Returns b-values and (normalized) b-vectors for the 99-Direction scheme used for DBSI

    :return: b-values and (normalized) b-vectors
    :rtype: np.ndarray
    """    

    DATA_DIR = os.path.dirname(os.path.relpath(__file__)) 
    bvals = np.loadtxt(os.path.join(DATA_DIR, 'bval99'))
    bvecs = np.loadtxt(os.path.join(DATA_DIR, 'bvec99')).T
    return bvals, normalize_bvecs(bvecs)


def _ABCD():
    """Returns b-values and (normalized) b-vectors for the popular aABCD scheme

    :return: b-values and (normalized) b-vectors
    :rtype: np.ndarray
    """    
    DATA_DIR = os.path.dirname(os.path.relpath(__file__)) 
    bvals = np.loadtxt(os.path.join(DATA_DIR, 'bval_ABCD'))
    bvecs = np.loadtxt(os.path.join(DATA_DIR, 'bvec_ABCD')).T
    return bvals, normalize_bvecs(bvecs)

def _NODDI():
    """Returns b-values and (normalized) b-vectors for the native NODDI diffusion scheme

    :return: b-values and (normalized) b-vectors
    :rtype: np.ndarray
    """
    DATA_DIR = os.path.dirname(os.path.relpath(__file__)) 
    bvals = np.loadtxt(os.path.join(DATA_DIR, 'bval_NODDI'))
    bvecs = np.loadtxt(os.path.join(DATA_DIR, 'bvec_NODDI')).T
    return bvals, normalize_bvecs(bvecs)

diff_scheme_opts = {'DBSI_99': _DBSI_99,
                    'ABCD': _ABCD,
                    'NODDI_145': _NODDI}



