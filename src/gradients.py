import numpy as np 
import matplotlib.pyplot as plt 
from typing import Union, Tuple, Type, Dict
import os 
import argparse
import importlib
import logging
import sys

DEFAULT_DIFFUSION_SCHEME_LIST = ['ABCD', 'NODDI', '99']


logger = logging.getLogger('simDRIFT')

def generate_diffusion_scheme_dictionary() -> Dict[str, Dict[str, np.ndarray]]:

    base_diffusion_scheme_dir = os.path.join(
                                    os.path.dirname(os.path.realpath(__file__)),
                                    f"data{os.sep}diffusion_schemes"
                                    )
    

    scheme_dicts = dict(keys = DEFAULT_DIFFUSION_SCHEME_LIST)

    for scheme_name in DEFAULT_DIFFUSION_SCHEME_LIST:

        scheme_dict = {}

        try:
            # Load bval file
            bvals = np.loadtxt(os.path.join(
                                            base_diffusion_scheme_dir, 
                                            f"bval{scheme_name}"
                                            )
                                )

            # Load bvec file                                
            bvecs = np.loadtxt(os.path.join(
                                            base_diffusion_scheme_dir, 
                                            f"bvec{scheme_name}"
                                            ), 
                                ndmin=2
                                )
        
         
        
        except (ValueError, FileNotFoundError):
            logger.info(f"Failed to load file: [bval/bvec]{scheme_name}!" + 
                           "\nPlease ensure the file exists and is correctly formatted")
            sys.exit(1)
     
        # Reshape bvec file if axis = -1 is not aligned with the bval file
        if bvecs.shape[0] != bvals.shape[0]:
            bvecs = bvecs.T

        
        scheme_dict['bvals'] = bvals
        scheme_dict['bvecs'] = bvecs
        
        # Write to output scheme
        scheme_dicts[scheme_name] = scheme_dict

    return scheme_dicts


GAMMA = 267.513e6 # (sT)^-1

def calc_q(gradient, dt):
    """Calculate the q-vector array corresponding to the gradient array.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array with shape (n of measurements, n of time points, 3).
    dt : float
        Duration of a time step in the gradient array.

    Returns
    -------
    q : numpy.ndarray
        q-vector array.
    """
    q = GAMMA * np.concatenate(
        (
            np.zeros((gradient.shape[0], 1, 3)),
            np.cumsum( dt * (gradient[:, 1::, :] + gradient[:, 0:-1, :]) / 2, axis=1),
        ),
        axis=1,
    )
    return q


def calc_b(gradient, dt):
    """Calculate b-values of the gradient array.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array with shape (n of measurements, n of time points, 3).
    dt : float
        Duration of a time step in gradient array.

    Returns
    -------
    b : numpy.ndarray
        b-values.
    """
    q = calc_q(gradient, dt)
    b = np.trapz(np.linalg.norm(q, axis=2) ** 2, axis=1, dx= dt)

    return b


def set_b(gradient, dt, b):
    """Scale the gradient array magnitude to correspond to given b-values.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array with shape (n of measurements, n of time points, 3).
    dt : float
        Duration of a time step in gradient array.
    b : float or numpy.ndarray
        b-value or an array of b-values with length equal to n of measurements.

    Returns
    -------
    scaled_g : numpy.ndarray
        Scaled gradient array.
    """
    b = np.asarray(b)
    if np.any(np.isclose(calc_b(gradient, dt), 0)):
        raise Exception("b-value can not be changed for measurements with b = 0")
    ratio = b / calc_b(gradient, dt)
    scaled_g = gradient * np.sqrt(ratio)[:, np.newaxis, np.newaxis]
    return scaled_g

def vec2vec_rotmat(v, k):
    """Return a rotation matrix defining a rotation that aligns v with k.

    Parameters
    -----------
    v : numpy.ndarray
        1D array with length 3.
    k : numpy.ndarray
        1D array with length 3.

    Returns
    ---------
    R : numpy.ndarray
        3 by 3 rotation matrix.
    """
    v = v / np.linalg.norm(v)
    k = k / np.linalg.norm(k)
    axis = np.cross(v, k)
    if np.linalg.norm(axis) < np.finfo(float).eps:
        if np.linalg.norm(v - k) > np.linalg.norm(v):
            return -np.eye(3)
        else:
            return np.eye(3)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.dot(v, k))
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    R = (
        np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.matmul(K, K)
    )  # Rodrigues' rotation formula
    return R

def rotate_gradient(gradient, Rs):
    """Rotate the gradient array of each measurement according to the
    corresponding rotation matrix.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array with shape (n of measurements, n of time points, 3).
    Rs : numpy.ndarray
        Rotation matrix array with shape (n of measurements, 3, 3).

    Returns
    -------
    g : numpy.ndarray
        Rotated gradient array.
    """
    g = np.zeros(gradient.shape)
    for i, R in enumerate(Rs):
        if not np.isclose(np.linalg.det(R), 1) or not np.all(
            np.isclose(R.T, np.linalg.inv(R))
        ):
            raise ValueError(f"Rs[{i}] ({R}) is not a valid rotation matrix")
        g[i, :, :] = np.matmul(R, gradient[i, :, :].T).T
    return g

def interpolate_gradient(waveforms: np.ndarray, TE : float, dt : float) -> np.ndarray:
    r"""Interpolate the gradient array to have ` TE / dt` time points.

    Parameters
    ----------
    waveforms : numpy.ndarray
        Gradient array with shape (n of measurements, n of time points, 3).
    
    TE : float
        Duration of the imaging experiment
    
    dt : float
        Duration of a time step in the gradient array.
 
    Returns
    -------
    interp_g : numpy.ndarray
        Interpolated gradient array.
    """

    interp_g = np.zeros((waveforms.shape[0], int (TE / dt), 3))

    for k in range(3):
        interp_g[..., k] = np.interp(
                                    np.concatenate([np.linspace(0, TE, int (TE / dt)     ) for _ in range(waveforms.shape[0])]),
                                    np.concatenate([np.linspace(0, TE, waveforms.shape[1]) for _ in range(waveforms.shape[0])]),
                                    waveforms.reshape((waveforms.shape[0] * waveforms.shape[1], 3))[:, k]
                                    ).reshape((waveforms.shape[0], int (TE / dt )))

    return interp_g




class PuledGradientSpinEchoDataset:
    """ Object for storing the PGSE sequence and basic manipulations
    and pre-processing (e.g., interpolating the bvals/bvecs to a gradient waveform) 
    
    Args:
        bval_path : Input bval data file
        bvec_path : Input bvec data file
        TE : Duration of the experiment [sec.]
        Delta : The Diffusion Time [sec.]
        delta : the pulse width [sec.]
        dt : the timestep parameter [sec.]
        USE_DEFAULT_DSCHEME : A flag to determine if custom diffusion data is to be used
        
    References
    ----------
    .. [1] Kerkelä et al., (2020).
        Disimpy: A massively parallel Monte Carlo simulator for generating diffusion-weighted MRI data in Python. 
        Journal of Open Source Software, 5(52), 2527. https://doi.org/10.21105/joss.02527
        
    """

    def __init__(self,
                 input_bval_file : str,
                 input_bvec_file : str,
                 input_base_diffusion_scheme : str,
                 TE : float,
                 Delta : float,
                 delta : float,
                 dt : float,
                 USE_DEFAULT_DSCHEME : bool = False
                ) -> None:
        
        self.input_bval_file = input_bval_file
        self.input_bvec_file = input_bvec_file
        self.input_base_diffusion_scheme = input_base_diffusion_scheme
        self.TE = TE
        self.Delta = Delta
        self.delta = delta
        self.dt = dt
        self.USE_DEFAULT_DSCHEME = USE_DEFAULT_DSCHEME
        
        if not (self.USE_DEFAULT_DSCHEME):
            self.bvals, self.bvecs = self._load_custom_diffusion_scheme()

        else:
            d_scheme_dict = generate_diffusion_scheme_dictionary()      
            self.bvals = d_scheme_dict[self.input_base_diffusion_scheme]['bvals']
            self.bvecs = d_scheme_dict[self.input_base_diffusion_scheme]['bvecs']

          

        self.bvals, self.bvecs = self._normalize_diffusion_scheme()
        
        self.G = np.zeros( (self.bvals.shape[0], int( self.TE / self.dt ), 3)) 
        
        # Positive Pulse
        self.G[:,  1:int(delta / dt),    0] =  1
        
        # Negative Pulse
        self.G[:, -1*int(delta / dt):-1, 0] = -1
        
        self.G = set_b(self.G, dt, self.bvals)

        Rs = np.zeros((self.bvals.shape[0], 3, 3))
        for i, bvec in enumerate(self.bvecs):
            Rs[i] = vec2vec_rotmat(np.array([1.0, 0.0, 0.0]), bvec) 
        
        self.G = rotate_gradient(self.G, Rs)

        return 
    

    def _load_custom_diffusion_scheme(self) -> None:
        """Try to load the custom provided diffusion schemes. 
        Note that FileNotFoundError is not possible here because of the existence
        of the file is garunteed while parsing the configuration.ini file. 
        """
        try: 
            # load bvals first. Having access to the expected shape should help with loading
            # the bvec file.
            bvals = np.loadtxt(self.input_bval_file)

            # load bvecs.
            bvecs = np.loadtxt(self.input_bvec_file, ndmin=2)

        except (ValueError):
            logger.info(f"unable to load one of {self.input_bval_file} or {self.input_bvec_file}" + "\n"
                        f"Please make sure the input files are readable to np.loadtxt"
                        )
            sys.exit(1)
        
        if bvecs.shape[0] != bvals.shape[0]:
            bvecs = bvecs.T
        
        return bvals, bvecs

    def _normalize_diffusion_scheme(self):
        """Normalize the bvecs so that each row is unit normal
        """
        # normalize the bvectors
        self.bvecs[~(self.bvals == 0), :] /= np.linalg.norm(self.bvecs[~(self.bvals == 0), :], ord = 2, axis = -1 )[:, None]        
        
        # make bvecs rowwise operations divison safe
        self.bvecs[(self.bvecs == 0).all(axis = -1)] = 1e-7
        
        return self.bvals, self.bvecs
        
def get_gradient_obj(args : argparse.Namespace) -> PuledGradientSpinEchoDataset:
    return PuledGradientSpinEchoDataset(
        input_bval_file=args.bvals,
        input_bvec_file=args.bvecs,
        input_base_diffusion_scheme=args.dscheme,
        TE=args.TE,
        Delta=args.Delta,
        delta=args.delta,
        dt=args.dt,
        USE_DEFAULT_DSCHEME=args.custom_diff_scheme_flag
    )
