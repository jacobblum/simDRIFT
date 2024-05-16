import numpy as np  
import os  
import sys
from src.data import diffusion_schemes
import nibabel as nb
import time
import logging
import torch 
from datetime import datetime
import shutil

def _add_noise(signal, snr):
    """Add Gaussian noise to the forward simulated signal [2]_

    :param signal: The forward simulated signal
    :type signal: np.ndarray
    :param snr: The desired signal to noise ratio of the b0 image
    :type snr: float
    :return: The noise-added signal
    :rtype: np.ndarray
    """ 
    sigma = 1.0/snr 
    real_channel_noise = np.random.normal(0, sigma, signal.shape[0])
    return signal + real_channel_noise

   
def _signal(phase : np.ndarray, SNR : int = None) -> np.ndarray:
    """Calculates the PGSE signal from the forward simulated spin trajectories [3]_. Note that this computation is executed on the GPU using PyTorch.

    :param spins: A list of each objects.spin instance corresponding to a spin in the ensemble of random walkers
    :type spins: list
    :param bvals: The supplied b-values (diffusion weighting factors)
    :type bvals: np.ndarray
    :param bvecs: The supplied diffusion gradients 
    :type bvecs: np.ndarray
    :param Delta: The diffusion time, in milliseconds
    :type Delta: float
    :param dt: The time step parameter, also equal to delta because of the narrow pulse approximation
    :type dt: float
    :param SNR: The snr of the b0 image. If a value is not entered, the SNR of the signal is infinite. (Defaults to ``None``)
    :type SNR: float, optional
    :return: the forward simulated PGSE signal (``signal``), the initial spin positions (``trajectory_t1m``), and the final spin positions ``trajectory_t2p``
    :rtype: np.ndarray
    """
    
    signal = (1/phase.shape[0]) * np.real(np.nansum(np.exp(1j * phase), axis = 0))

    if SNR != None:
        noised_signal = _add_noise(signal, SNR)
        return noised_signal
    else:
        return signal
    
def _generate_signals_and_trajectories(self):
    """Helper function to organize and store compartment specific and combined trajectories and their incident signals

    :return: signals with associated labels (``signals_dict``), trajectories with associated labels (``trajectories_dict``)
    :rtype: dictionaries
    """
    
    signals_dict = {}
    trajectories_dict = {}
    
    trajectory_t1m = []
    trajectory_t2p = []
    fiber_spins    = []
    cells          = []

    for spin in self.spins:
        fiber_spins.append([-1 if spin._get_bundle_index() is None else spin._get_bundle_index()])
        cells.append([spin._get_cell_index()])
        trajectory_t1m.append(spin.position_t1m)
        trajectory_t2p.append(spin.position_t2p)

    fiber_spins    = np.concatenate(fiber_spins)
    cells          = np.concatenate(cells)
    trajectory_t1m = np.array(trajectory_t1m)
    trajectory_t2p = np.array(trajectory_t2p)

    water = self.water_key

    logging.info('------------------------------')
    logging.info(' Signal Generation') 
    logging.info('------------------------------')
    

    for i in range(1, int(np.amax(fiber_spins))+1):
        """ ith Fiber Signal """
        ith_fiber_phase = self.phase[fiber_spins == i, :]
        if ith_fiber_phase.shape[0] > 0:
            logging.info(f" Computing fiber {i} signal...") 
            Start = time.time()
            ith_fiber_signal = _signal(ith_fiber_phase)
            End = time.time()
            logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
            signals_dict[f"fiber_{i}_signal"] = ith_fiber_signal
            trajectories_dict[f"fiber_{i}_trajectories"] = (trajectory_t1m[fiber_spins == i, :], trajectory_t2p[fiber_spins == i, :])

    """ Total Fiber Signal """ 
    total_fiber_phase = self.phase[fiber_spins > -1, :]
    if total_fiber_phase.shape[0] > 0:
        logging.info(' Computing total fiber signal...')
        Start = time.time()
        total_fiber_signal = _signal(total_fiber_phase)
        End = time.time()
        logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
        signals_dict['total_fiber_signal'] = total_fiber_signal
        trajectories_dict['total_fiber_trajectories'] = (trajectory_t1m[fiber_spins > -1, :], trajectory_t2p[fiber_spins > -1, :])

    """ Cell Signal """
 
    cell_phase = self.phase[cells > -1, :]

    if cell_phase.shape[0] > 0:
        logging.info(' Computing cell signal...')
        Start = time.time()
        cell_signal = _signal(cell_phase)
        End = time.time()
        logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
        signals_dict['cell_signal'] = cell_signal
        trajectories_dict['cell_trajectories'] = (trajectory_t1m[cells > -1, :], trajectory_t2p[cells > -1, :])

    """ Water Signal """

    total_water_phase = self.phase[water > -1, :]
    if total_water_phase.shape[0] > 0:
        logging.info(' Computing water signal...')
        Start = time.time()
        total_water_signal = _signal(total_water_phase)
        End = time.time()
        logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
        signals_dict['water_signal'] = total_water_signal
        trajectories_dict['water_trajectories'] = (trajectory_t1m[water > -1, :], trajectory_t2p[water > -1, :])

    
    """ Total Signal """
    logging.info(' Computing total signal...')
    Start = time.time()
    total_signal =  _signal(self.phase)    
    End = time.time()
    logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
    signals_dict['total_signal'] = total_signal
    trajectories_dict['total_trajectories'] = (trajectory_t1m, trajectory_t2p)
    return signals_dict, trajectories_dict

def _save_data(self):
    """Helper function that saves signals and trajectories to the current directory.
    """

    RESULTS_DIR = self.results_directory
    SIGNALS_DIR = os.path.join(RESULTS_DIR, 'signals')
    TRAJ_DIR    = os.path.join(RESULTS_DIR, 'trajectories')

    shutil.copyfile(src = self.cfg_path, dst = os.path.join(RESULTS_DIR, 'input_simulation_parameters.ini'))
    shutil.copyfile(src = os.path.join(os.getcwd(),'log'), dst = os.path.join(RESULTS_DIR, 'log'))

    signals_dict, trajectories_dict = _generate_signals_and_trajectories(self)
    logging.info('------------------------------')
    logging.info(' Saving outputs to {} ...'.format(RESULTS_DIR))

    if not os.path.exists(SIGNALS_DIR): os.mkdir(SIGNALS_DIR)
    for key in signals_dict.keys():        
        Nifti = nb.Nifti1Image(signals_dict[key], affine = np.eye(4))
        nb.save(Nifti, os.path.join(SIGNALS_DIR, '{}.nii'.format(key)))

    if not os.path.exists(TRAJ_DIR): os.mkdir(TRAJ_DIR)
    for key in trajectories_dict.keys():        
        np.save(os.path.join(TRAJ_DIR, '{}_t1m.npy'.format(key)), trajectories_dict[key][0])
        np.save(os.path.join(TRAJ_DIR, '{}_t2p.npy'.format(key)), trajectories_dict[key][1])
    logging.info(' Program complete!')
    logging.info('------------------------------')
    return
