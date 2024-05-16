import numpy as np 
import matplotlib.pyplot as plt 
from typing import Union, Tuple
import os 
from src.data import diffusion_schemes

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

def pgse(sim_class) -> np.ndarray:
    """Generate a pulsed gradient spin echo gradient array.

    Parameters
    ----------
    delta (ms) : float 
        Diffusion encoding time.
    DELTA (ms) : float
        Diffusion time.
    dt (ms) : float
        Duration of a timestep in the simulation
    bvals ( s / mm^{2} ) : float or numpy.ndarray
        b-value or an array of b-values.
    bvecs : numpy.ndarray
        b-vector or array of b-vectors.

    Returns
    -------
    gradient : numpy.ndarray
        Gradient array.

    References
    ----------
    .. [1] Kerkelä et al., (2020).
        Disimpy: A massively parallel Monte Carlo simulator for generating diffusion-weighted MRI data in Python. 
        Journal of Open Source Software, 5(52), 2527. https://doi.org/10.21105/joss.02527
    
    """

    Delta = sim_class.Delta 
    delta = sim_class.delta  
    dt    = sim_class.dt    
    TE    = sim_class.TE

    if sim_class.custom_diff_scheme_flag:
        if all([type(sim_class.bvals) is str, type(sim_class.bvecs) is str]):
            bvals, bvecs = load_diffusion_scheme_from_txt_file(sim_class.bvals, sim_class.bvecs)

    else:
        bvals, bvecs = diffusion_schemes.get_from_default(sim_class.diff_scheme)
        
    gradient = np.zeros((bvals.shape[0], int( TE / dt ), 3)) 
    gradient[:,  1:int(delta / dt),    0] =  1
    gradient[:, -1*int(delta / dt):-1, 0] = -1
    gradient = set_b(gradient, dt, bvals)

    Rs = np.zeros((len(bvals), 3, 3))
    for i, bvecs in enumerate(bvecs):
        Rs[i] = vec2vec_rotmat(np.array([1.0, 0.0, 0.0]), bvecs) 
    gradient = rotate_gradient(gradient, Rs)
    return gradient

def load_diffusion_scheme_from_txt_file(bvals : str, bvecs : str) -> Tuple[np.ndarray, np.ndarray]:
    if all([os.path.exists(path) for path in [bvals, bvecs]]):
        bvals_np = np.loadtxt(bvals).astype(np.float32)
        bvecs_np = np.loadtxt(bvecs).astype(np.float32)

        if bvecs_np.shape[-1] != 3:
            bvecs_np = bvecs_np.T
        
        bvecs_np[(bvecs_np == 0).all(axis = 1)] = 1e-5
        bvecs_np /= np.linalg.norm(bvecs_np, ord = 2, axis = 1)[:, None]
        bvals_np = bvals_np * 1e6 # convert to s / m^2 
        
        return bvals_np, bvecs_np
    else:
        raise Exception(
            "One of the bval/bvec paths do not exist!" \
            "Please make sure that the bval / bvec paths are correct!"
        )
        
