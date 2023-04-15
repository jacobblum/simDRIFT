import numpy as np 
import numba 
from numba import jit, cuda, int32, float32
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
import jp as jp
import matplotlib.pyplot as plt
import time 
import os  
from tqdm import tqdm
import nibabel as nb
import glob as glob 
import configparser
from ast import literal_eval
from multiprocessing import Process
import shutil
import sys
from data import diffusion_schemes
import cupy as cp

def _add_noise(signal, snr, noise_type = 'Rician'):
    sigma = 1.0/snr 
    real_channel_noise = np.random.normal(0, sigma, signal.shape[0])
    complex_channel_noise = np.random.normal(0, sigma, signal.shape[0])

    if noise_type == 'Gaussian':
        return signal + real_channel_noise

    if noise_type == 'Rician':
        return np.sqrt((signal + real_channel_noise)**2 + complex_channel_noise**2)
  

def _signal(spins: list, bvals: np.ndarray, bvecs: np.ndarray, Delta: float, dt: float, SNR = None, noise_type = 'Gaussian') -> np.ndarray:
   
    """ use CuPy to execute the einstein summations on the gpu """
    
    gamma = 42.58 # MHz/T - The Gyromatnetic Ratio of a Proton
    delta = dt    # ms - From the narrow pulse approximation. More convient for GPU memory managment 

    trajectory_t1m = cp.array([spin._get_position_t1m() for spin in spins])
    trajectory_t2p = cp.array([spin._get_position_t2p() for spin in spins])
    bvals_cp = cp.array(bvals)
    bvecs_cp = cp.array(bvecs)

    scaled_gradients = np.einsum('i, ij -> ij', (np.sqrt( (bvals_cp * 1e-3)/ (gamma**2*delta**2*(Delta - delta/3)))), bvecs_cp)
    phase_shifts = gamma * np.einsum('ij, kj -> ik', scaled_gradients, (trajectory_t1m - trajectory_t2p))*dt
    signal = np.abs(1/trajectory_t1m.shape[0]) * np.sum(np.exp(-(0+1j)*phase_shifts), axis = 1)

    if SNR != None:
        noised_signal = _add_noise(signal, SNR, noise_type)
        return noised_signal
    else:
        return signal


def _save_data(spins: list, Delta: float, dt: float):

    """
    fiber_1
    fiber_2
    fiber_total
    cells
    total_signal
    """

    bvals, bvecs = diffusion_schemes._dbsi_99()

    fiber1 = np.array([spin._get_fiber_index() if spin._get_bundle_index() == 1 else -1 for spin in spins])
    fiber2 = np.array([spin._get_fiber_index() if spin._get_bundle_index() == 2 else -1 for spin in spins])
    cells  = np.array([spin._get_cell_index() for spin in spins])

    """ fiber 1 signal """


    sys.stdout.write(' \n Computing fiber 1 signal \n ')

    fiber_1_spins = np.array(spins)[fiber1 > -1]
    fiber_1_signal = _signal(fiber_1_spins,
                             bvals,
                             bvecs,
                             Delta, 
                             dt)

 
    """ fiber 2 signal """
    sys.stdout.write(' \n Computing fiber 2 signal \n ')
    fiber_2_spins = np.array(spins)[fiber2 > -1]
    fiber_2_signal = _signal(fiber_2_spins,
                            bvals,
                            bvecs,
                            Delta, 
                            dt)


    """ total fiber signal """
    sys.stdout.write(' \n Computing total fiber signal \n ')
    total_fiber_spins = np.hstack([fiber_1_spins, fiber_2_spins])
    total_fiber_signal = _signal(total_fiber_spins,
                                 bvals,
                                 bvecs,
                                 Delta, 
                                 dt)
    
    
    
    """ Cell Spins """
    sys.stdout.write(' \n Computing cell signal \n ')
    cell_spins = np.array(spins)[cells > -1]
    cell_signal = _signal(cell_spins,
                          bvals,
                          bvecs,
                          Delta, 
                          dt)
    
    """ Total Signal """
    sys.stdout.write(' \n Computing total signal \n ')
    total_signal = _signal(spins,
                           bvals,
                           bvecs,
                           Delta,
                           dt)


    
    return
