import numpy as np  
import os  
import sys
from data import diffusion_schemes
import cupy as cp
import nibabel as nb
import time
import logging

""""
def spins_in_voxel(trajectoryT1m, trajectoryT2p):
   
        Helper function to ensure that the spins at time T2p are wtihin the self.voxelDims x self.voxelDims x inf imaging voxel

    Parameters
    ----------
    trajectoryT1m: N_{spins} x 3 ndarray
        The initial spin position at time t1m

    trajectoryT2p: N_{spins} x 3 ndarray
        The spin position at time t2p

    Returns
    -------
    traj1_vox: (N, 3) ndarray
        Position at T1m of the spins which stay within the voxel
    traj2_vox: (N, 3) ndarray
        Position at T2p of the spins which stay within the voxel

    Notes
    -----
    None

    References
    ----------
    None

    
    traj1_vox = []
    traj2_vox = []

    for i in range(trajectoryT1m.shape[0]):
        if np.amin(trajectoryT2p[i, 0:2]) >= 0 + 0.5*self.buffer and np.amax(trajectoryT2p[i, 0:2]) <= self.voxelDims + 0.5*self.buffer:
            traj1_vox.append(trajectoryT1m[i, :])
            traj2_vox.append(trajectoryT2p[i, :])
    return np.array(traj1_vox), np.array(traj2_vox)
"""


def _add_noise(signal, snr):
    sigma = 1.0/snr 
    real_channel_noise = np.random.normal(0, sigma, signal.shape[0])
    return signal + real_channel_noise

   
def _signal(spins: list, bvals: np.ndarray, bvecs: np.ndarray, Delta: float, dt: float, SNR = None) -> np.ndarray:
   
    """ Use CuPy to execute the einstein summations on the GPU """
    
    gamma = 42.58 # MHz/T - The proton gyromagnetic ratio
    delta = dt    # ms - From the narrow pulse approximation. More convient for GPU memory managment 

    trajectory_t1m = cp.array([spin._get_position_t1m() for spin in spins])
    trajectory_t2p = cp.array([spin._get_position_t2p() for spin in spins])
    bvals_cp       = cp.array(bvals)
    bvecs_cp       = cp.array(bvecs)

    scaled_gradients = np.einsum('i, ij -> ij', (np.sqrt( (bvals_cp * 1e-3)/ (gamma**2*delta**2*(Delta - delta/3)))), bvecs_cp)
    phase_shifts = gamma * np.einsum('ij, kj -> ik', scaled_gradients, (trajectory_t1m - trajectory_t2p))*dt
    signal = 1/trajectory_t1m.shape[0] * np.abs(np.sum(np.exp(-(0+1j)*phase_shifts), axis = 1))


    if SNR != None:
        noised_signal = _add_noise(signal, SNR)
        return noised_signal, trajectory_t1m, trajectory_t2p
    else:
        return signal, trajectory_t1m, trajectory_t2p


def _generate_signals_and_trajectories(spins: list, Delta: float, dt: float, diff_schemes: str, SNR = None):
    signals_dict = {}
    trajectories_dict = {}

    bvals, bvecs = diffusion_schemes.get('DBSI_99')

    fiber1 = np.array([spin._get_fiber_index() if spin._get_bundle_index() == 1 else -1 for spin in spins])
    fiber2 = np.array([spin._get_fiber_index() if spin._get_bundle_index() == 2 else -1 for spin in spins])
    cells  = np.array([spin._get_cell_index() for spin in spins])
    iSpin = 0
    water = np.empty(len(spins))
    for spin in spins:
        if spin._in_water:
            watInd[iSpin] = iSpin
        else:
            watInd[iSpin] = -1
        iSpin = iSpin + 1
    
    """ fiber 1 signal """
    logging.info('------------------------------')
    logging.info('Computing fiber 1 signal...')  
    Start = time.time()
    fiber_1_spins = np.array(spins)[fiber1 > -1]
    fiber_1_signal, fiber_1_trajectory_t1m, fiber_1_trajectory_t2p = _signal(fiber_1_spins,
                                                                             bvals,
                                                                             bvecs,
                                                                             Delta, 
                                                                             dt)
    End = time.time()
    logging.info('Fiber 1 signal computed in {} sec'.format(round(End-Start),4))
    logging.info('------------------------------')  
    signals_dict['fiber_1_signal'] = fiber_1_signal
    trajectories_dict['fiber_1_trajectories'] = (fiber_1_trajectory_t1m, fiber_1_trajectory_t2p)

    """ fiber 2 signal """
    logging.info('Computing fiber 2 signal...')
    Start = time.time()
    fiber_2_spins = np.array(spins)[fiber2 > -1]
    fiber_2_signal, fiber_2_trajectory_t1m, fiber_2_trajectory_t2p = _signal(fiber_2_spins,
                                                                             bvals,
                                                                             bvecs,
                                                                             Delta, 
                                                                             dt)
    End = time.time()
    logging.info('Fiber 2 signal computed in {} sec'.format(round(End-Start),4))
    logging.info('------------------------------')
    signals_dict['fiber_2_signal'] = fiber_2_signal
    trajectories_dict['fiber_2_trajectories'] = (fiber_2_trajectory_t1m, fiber_2_trajectory_t2p)


    """ total fiber signal """ 
    logging.info('Computing total fiber signal...')
    Start = time.time()
    total_fiber_spins = np.hstack([fiber_1_spins, fiber_2_spins])
    total_fiber_signal, total_fiber_trajectory_t1m, total_fiber_trajectory_t2p = _signal(total_fiber_spins,
                                                                                         bvals,
                                                                                         bvecs,
                                                                                         Delta, 
                                                                                         dt) 
    End = time.time()
    logging.info('Total fiber signal computed in {} sec'.format(round(End-Start),4))
    logging.info('------------------------------')  
    signals_dict['total_fiber_signal'] = total_fiber_signal
    trajectories_dict['total_fiber_trajectories'] = (total_fiber_trajectory_t1m, total_fiber_trajectory_t2p)

    """ Cell Signal """
    logging.info('Computing cell signal...')
    Start = time.time()
    cell_spins = np.array(spins)[cells > -1]
    cell_signal, cell_trajectory_t1m, cell_trajectory_t2p = _signal(cell_spins,
                                                                        bvals,
                                                                        bvecs,
                                                                        Delta, 
                                                                        dt)
    
    End = time.time()
    logging.info('Cell signal computed in {} sec'.format(round(End-Start),4))
    logging.info('------------------------------')  
    signals_dict['cell_signal'] = cell_signal
    trajectories_dict['cell_trajectories'] = (cell_trajectory_t1m, cell_trajectory_t2p)


    """ Water Signal (Added KLU 04.26.23)"""
    
    logging.info('Computing water signal...')
    Start = time.time()
    water_spins = np.array(spins)[water > -1]
    water_signal, water_trajectory_t1m, water_trajectory_t2p = _signal(water_spins,
                                                                        bvals,
                                                                        bvecs,
                                                                        Delta, 
                                                                        dt)
    
    End = time.time()
    logging.info('Water signal computed in {} sec'.format(round(End-Start),4))
    logging.info('------------------------------')  
    signals_dict['water_signal'] = water_signal
    trajectories_dict['water_trajectories'] = (water_trajectory_t1m, water_trajectory_t2p)
                                                    
    """ Total Signal """
    logging.info('Computing total signal...')
    Start = time.time()
    total_signal, total_trajectory_t1m, total_trajectory_t2p = _signal(spins,
                                                                       bvals,
                                                                       bvecs,
                                                                       Delta,
                                                                       dt)
    
    End = time.time()
    logging.info('Total signal computed in {} sec'.format(round(End-Start),4))
    logging.info('------------------------------')  
    signals_dict['total_signal'] = total_signal
    trajectories_dict['total_trajectories'] = (total_fiber_trajectory_t1m, total_fiber_trajectory_t2p)
    return signals_dict, trajectories_dict

def _save_data(self, spins: list, Delta: float, dt: float, diff_scheme: str):

    signals_dict, trajectories_dict = _generate_signals_and_trajectories(self.spins,
                                                                         self.parameters['Delta'],
                                                                         self.parameters['dt'],
                                                                         diff_scheme)
    logging.info('Saving outputs to {} ...'.format(os.getcwd()))
    if not os.path.exists(r'./signals'): os.mkdir(r'./signals')
    for key in signals_dict.keys():        
        Nifti = nb.Nifti1Image(signals_dict[key].get(), affine = np.eye(4))
        nb.save(Nifti, r'./signals/{}.nii'.format(key))

    if not os.path.exists(r'./trajectories'): os.mkdir(r'./trajectories')
    for key in trajectories_dict.keys():        
        np.save(r'./trajectories/{}_t1m.npy'.format(key), trajectories_dict[key][0].get())
        np.save(r'./trajectories/{}_t2p.npy'.format(key), trajectories_dict[key][1].get())
    logging.info('Simulation Complete!')
    logging.info('------------------------------')
    return
