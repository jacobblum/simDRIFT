import numpy as np  
import os  
import sys
from src.data import diffusion_schemes
import nibabel as nb
import time
import logging
import torch 

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
    r"""
    Add Gaussian Noise to the forward simulated signal

    Args:
        signal (np.ndarray): The forward simulated signal
        snr (float): The signal to noise ratio of the b0 image

    Shapes:
        signal: (n_bvals,) where n_bvals is the number of b-values in the supplied bval file

    Returns:
        noised_signal (np.ndarray): The noised signal
    
    References:
        [1] Garyfallidis E, Brett M, Amirbekian B, Rokem A, van der Walt S, Descoteaux M, Nimmo-Smith I and Dipy Contributors (2014). DIPY, a library for the analysis of diffusion MRI data. Frontiers in Neuroinformatics, vol.8, no.8.
    
    
    """
    sigma = 1.0/snr 
    real_channel_noise = np.random.normal(0, sigma, signal.shape[0])
    return signal + real_channel_noise

   
def _signal(spins: list, bvals: np.ndarray, bvecs: np.ndarray, Delta: float, dt: float, SNR = None) -> np.ndarray:
    r"""
    Calculates the PGSE signal from the forward simulated spin trajectories. Note that this computation is executed on the GPU using CuPy.
    
    Args:
        spins (list): A list of each objects.spin instance corresponding to a spin in the ensemble of random walkers
        bvals (np.ndarray): The supplied b-values (diffusion weighting factors)
        bvecs (np.ndarray): The supplied diffusion gradients 
        Delta (float): The diffusion time (ms)
        dt (float): The time step parameter, also equal to delta because of the narrow pulse approximation
        SNR (float, optional): The snr of the b0 image. If a value is not entered, the snr of the signal is infinite. 

    Shapes:
        spins: (n_walkers,) where n_walkers is an input parameter denoting the n umber of spins in the ensemble
        bvals: (n_bvals,) where n_bvals is the number of b-values in the supplied bval file
        bvecs: (n_bvals, 3) where n_bvals is the number of b-values in the supplied bval file)I

    Returns:
        signal (np.ndarray): the forward simulated PGSE signal
        trajectory_t1m (np.ndarray): the initial spin positions 
        trajectory_t2p (np.ndarray): the final spin positions

    References:
        [1] Hall, M. G., and Alexander, D. C. (2009). Convergence and parameter choice for monte-carlo simulations of diffusion MRI. IEEE Trans. Med. Imaging 28, 1354–1364. doi: 10.1109/TMI.2009.2015756
    """    



    """ Use CuPy to execute the einstein summations on the GPU """
    
    gamma = 42.58 # MHz/T - The proton gyromagnetic ratio
    delta = dt    # ms - From the narrow pulse approximation. More convient for GPU memory managment 

    #Possibly call finite voxel helper

    trajectory_t1m = torch.from_numpy(np.array([spin._get_position_t1m() for spin in spins])).float().to('cuda')
    trajectory_t2p = torch.from_numpy(np.array([spin._get_position_t2p() for spin in spins])).float().to('cuda')
    bvals_cu       = torch.from_numpy(np.array(bvals)).float().to('cuda')
    bvecs_cu       = torch.from_numpy(np.array(bvecs)).float().to('cuda')

    scaled_gradients = torch.einsum('i, ij -> ij', (torch.sqrt( (bvals_cu * 1e-3)/ (gamma**2*delta**2*(Delta - delta/3)))), bvecs_cu)
    phase_shifts = gamma * torch.einsum('ij, kj -> ik', scaled_gradients, (trajectory_t1m - trajectory_t2p))*dt
    signal =  torch.abs(torch.sum(torch.exp(-(0+1j)*phase_shifts), axis = 1)).cpu().numpy()
    signal /= trajectory_t1m.shape[0]


    if SNR != None:
        noised_signal = _add_noise(signal, SNR)
        return noised_signal, trajectory_t1m, trajectory_t2p
    else:
        return signal, trajectory_t1m, trajectory_t2p


def _generate_signals_and_trajectories(self):
    r"""
    Helper function to organize and store compartment specific and combined trajectories and their incident signals
    """
    
    signals_dict = {}
    trajectories_dict = {}

    if not self.custom_diff_scheme_flag:
        bvals, bvecs = diffusion_schemes.get_from_default(self.diff_scheme)
    else:
        bvals, bvecs = diffusion_schemes.get_from_custom(self.bvals, self.bvecs)
  
    fiber_spins = np.array([-1 if spin._get_bundle_index() is None else spin._get_bundle_index() for spin in self.spins])
    cells  = np.array([spin._get_cell_index() for spin in self.spins])
    water  = np.array([spin._get_water_index() for spin in self.spins])


    logging.info('------------------------------')
    logging.info(' Signal Generation') 
    logging.info('------------------------------')
    

    for i in range(1, int(np.amax(fiber_spins))+1):
        """ Fiber i Signal """

        fiber_i_spins = np.array(self.spins)[fiber_spins == i]

        if any(fiber_i_spins):
            logging.info(f" Computing fiber {i} signal...") 
            Start = time.time()
            fiber_i_signal, fiber_i_trajectory_t1m, fiber_i_trajectory_t2p = _signal(fiber_i_spins,
                                                                                    bvals,
                                                                                    bvecs,
                                                                                    self.Delta, 
                                                                                    self.dt)
            End = time.time()
            logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
            signals_dict[f"fiber_{i}_signal"] = fiber_i_signal
            trajectories_dict[f"fiber_{i}_trajectories"] = (fiber_i_trajectory_t1m, fiber_i_trajectory_t2p)

  
    """ Total Fiber Signal """ 

    total_fiber_spins = np.array(self.spins)[fiber_spins > -1]

    if any(total_fiber_spins):
        logging.info(' Computing total fiber signal...')
        Start = time.time()
        total_fiber_signal, total_fiber_trajectory_t1m, total_fiber_trajectory_t2p = _signal(total_fiber_spins,
                                                                                            bvals,
                                                                                            bvecs,
                                                                                            self.Delta, 
                                                                                            self.dt) 
        End = time.time()
        logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
        signals_dict['total_fiber_signal'] = total_fiber_signal
        trajectories_dict['total_fiber_trajectories'] = (total_fiber_trajectory_t1m, total_fiber_trajectory_t2p)


    """ Cell Signal """
 
    cell_spins = np.array(self.spins)[cells > -1]

    if any(cell_spins):
        logging.info(' Computing cell signal...')
        Start = time.time()
        cell_signal, cell_trajectory_t1m, cell_trajectory_t2p = _signal(cell_spins,
                                                                            bvals,
                                                                            bvecs,
                                                                            self.Delta, 
                                                                            self.dt)
        
        End = time.time()
        logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
        signals_dict['cell_signal'] = cell_signal
        trajectories_dict['cell_trajectories'] = (cell_trajectory_t1m, cell_trajectory_t2p)


    """ Water Signal """

    water_spins = np.array(self.spins)[water > -1]
    
    if any(water_spins):
        logging.info(' Computing water signal...')
        Start = time.time()
        water_signal, water_trajectory_t1m, water_trajectory_t2p = _signal(water_spins,
                                                                            bvals,
                                                                            bvecs,
                                                                            self.Delta, 
                                                                            self.dt)
        
        End = time.time()
        logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
        signals_dict['water_signal'] = water_signal
        trajectories_dict['water_trajectories'] = (water_trajectory_t1m, water_trajectory_t2p)

    """ Total Signal """
    logging.info(' Computing total signal...')
    Start = time.time()
    total_signal, total_trajectory_t1m, total_trajectory_t2p = _signal(self.spins,
                                                                       bvals,
                                                                       bvecs,
                                                                       self.Delta,
                                                                       self.dt)
    
    End = time.time()
    logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
    signals_dict['total_signal'] = total_signal
    trajectories_dict['total_trajectories'] = (total_trajectory_t1m, total_trajectory_t2p)
    return signals_dict, trajectories_dict

    # """ Fiber 1 Plus Water Signal"""
    # f1_water_spins = np.hstack([fiber_1_spins, water_spins])
    # if any(f1_water_spins) & any(fiber_1_spins) & any(water_spins):
    #     logging.info(' Computing fiber 1 + water signal...')
    #     Start = time.time()
    #     f1_water_signal, f1_water_trajectory_t1m, f1_water_trajectory_t2p = _signal(f1_water_spins, bvals, bvecs, Delta, dt)
    #     End = time.time()
    #     logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
    #     signals_dict['f1_water_signal'] = f1_water_signal
    #     trajectories_dict['f1_water_trajectories'] = (f1_water_trajectory_t1m, f1_water_trajectory_t2p)

    # """ Fiber 1 Plus Cell Signal"""
    # f1_cell_spins = np.hstack([fiber_1_spins, cell_spins])
    # if any(f1_cell_spins) & any(fiber_1_spins) & any(cell_spins):
    #     logging.info(' Computing fiber 1 + cell signal...')
    #     Start = time.time()
    #     f1_cell_spins = np.hstack([fiber_1_spins, cell_spins])
    #     f1_cell_signal, f1_cell_trajectory_t1m, f1_cell_trajectory_t2p = _signal(f1_cell_spins, bvals, bvecs, Delta, dt)
    #     End = time.time()
    #     logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
    #     signals_dict['f1_cell_signal'] = f1_cell_signal
    #     trajectories_dict['f1_cell_trajectories'] = (f1_cell_trajectory_t1m, f1_cell_trajectory_t2p)

    # """ Fiber 1 Plus Cell Plus Water Signal"""
    # f1_cell_water_spins = np.hstack([fiber_1_spins, cell_spins, water_spins])
    # if any(f1_cell_water_spins) & any(fiber_1_spins) & any(cell_spins):
    #     logging.info(' Computing fiber 1 + cell + water signal...')
    #     Start = time.time()
    #     f1_cell_water_signal, f1_cell_water_trajectory_t1m, f1_cell_water_trajectory_t2p = _signal(f1_cell_water_spins, bvals, bvecs, Delta, dt)
    #     End = time.time()
    #     logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
    #     signals_dict['f1_cell_water_signal'] = f1_cell_water_signal
    #     trajectories_dict['f1_cell_water_trajectories'] = (f1_cell_water_trajectory_t1m, f1_cell_water_trajectory_t2p)

    # """ Fiber 2 Plus Water Signal"""
    # f2_water_spins = np.hstack([fiber_2_spins, water_spins])
    # if any(f2_water_spins) & any(fiber_2_spins):
    #     logging.info(' Computing fiber 2 + water signal...')
    #     Start = time.time()
    #     f2_water_signal, f2_water_trajectory_t1m, f2_water_trajectory_t2p = _signal(f2_water_spins, bvals, bvecs, Delta, dt)
    #     End = time.time()
    #     logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
    #     signals_dict['f2_water_signal'] = f2_water_signal
    #     trajectories_dict['f2_water_trajectories'] = (f2_water_trajectory_t1m, f2_water_trajectory_t2p)


    # """ Fiber 2 Plus Cell Signal"""
    # f2_cell_spins = np.hstack([fiber_2_spins, cell_spins])
    # if any(f2_cell_spins) & any(fiber_2_spins) & any(cell_spins):
    #     logging.info(' Computing fiber 2 + cell signal...')
    #     Start = time.time()
    #     f2_cell_signal, f2_cell_trajectory_t1m, f2_cell_trajectory_t2p = _signal(f2_cell_spins, bvals, bvecs, Delta, dt)
    #     End = time.time()
    #     logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
    #     signals_dict['f2_cell_signal'] = f2_cell_signal
    #     trajectories_dict['f2_cell_trajectories'] = (f2_cell_trajectory_t1m, f2_cell_trajectory_t2p)

    # """ Fiber 2 Plus Cell Plus Water Signal"""
    # f2_cell_water_spins = np.hstack([fiber_2_spins, cell_spins, water_spins])
    # if any(f2_cell_water_spins) & any(fiber_2_spins) & any(cell_spins):
    #     logging.info(' Computing fiber 2 + cell + water signal...')
    #     Start = time.time()
    #     f2_cell_water_signal, f2_cell_water_trajectory_t1m, f2_cell_water_trajectory_t2p = _signal(f2_cell_water_spins, bvals, bvecs, Delta, dt)
    #     End = time.time()
    #     logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
    #     signals_dict['f2_cell_water_signal'] = f2_cell_water_signal
    #     trajectories_dict['f2_cell_water_trajectories'] = (f2_cell_water_trajectory_t1m, f2_cell_water_trajectory_t2p)

    # """ Both Fibers + Water Signal""" 
    # total_fiber_water_spins = np.hstack([total_fiber_spins, water_spins])
    # if any(total_fiber_water_spins) & any(fiber_1_spins) & any(fiber_2_spins):
    #     logging.info(' Computing total fiber + water signal...')
    #     Start = time.time()
    #     total_fiber_water_signal, total_fiber_water_trajectory_t1m, total_fiber_water_trajectory_t2p = _signal(total_fiber_water_spins, bvals, bvecs, Delta, dt) 
    #     End = time.time()
    #     logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
    #     signals_dict['total_fiber_water_signal'] = total_fiber_water_signal
    #     trajectories_dict['total_fiber_water_trajectories'] = (total_fiber_water_trajectory_t1m, total_fiber_water_trajectory_t2p)

    # """ Both Fibers + Cell Signal""" 
    # total_fiber_cell_spins = np.hstack([total_fiber_spins, cell_spins])
    # if any(total_fiber_cell_spins) & any(fiber_1_spins) & any(fiber_2_spins) & any(cell_spins):
    #     logging.info(' Computing total fiber + cell signal...')
    #     Start = time.time()
    #     total_fiber_cell_signal, total_fiber_cell_trajectory_t1m, total_fiber_cell_trajectory_t2p = _signal(total_fiber_cell_spins, bvals, bvecs, Delta, dt)
    #     End = time.time()
    #     logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
    #     signals_dict['total_fiber_cell_signal'] = total_fiber_cell_signal
    #     trajectories_dict['total_fiber_cell_trajectories'] = (total_fiber_cell_trajectory_t1m, total_fiber_cell_trajectory_t2p)

    # """ Cell + Water Signal""" 
    # water_cell_spins = np.hstack([total_fiber_spins, cell_spins])
    # if any(water_cell_spins) & any(cell_spins):
    #     logging.info(' Computing water + cell signal...')
    #     Start = time.time()
    #     water_cell_signal, water_cell_trajectory_t1m, water_cell_trajectory_t2p = _signal(water_cell_spins, bvals, bvecs, Delta, dt)
    #     End = time.time()
    #     logging.info('     Done! Signal computed in {} sec'.format(round(End-Start),4))
    #     signals_dict['water_cell_signal'] = water_cell_signal
    #     trajectories_dict['water_cell_trajectories'] = (water_cell_trajectory_t1m, water_cell_trajectory_t2p)

def _save_data(self):
    r"""
    Helper function that saves signals and trajectories
    """


    signals_dict, trajectories_dict = _generate_signals_and_trajectories(self)
    logging.info('------------------------------')
    logging.info(' Saving outputs to {} ...'.format(os.getcwd()))
    if not os.path.exists(r'./signals'): os.mkdir(r'./signals')
    for key in signals_dict.keys():        
        Nifti = nb.Nifti1Image(signals_dict[key], affine = np.eye(4))
        nb.save(Nifti, r'./signals/{}.nii'.format(key))

    if not os.path.exists(r'./trajectories'): os.mkdir(r'./trajectories')
    for key in trajectories_dict.keys():        
        np.save(r'./trajectories/{}_t1m.npy'.format(key), trajectories_dict[key][0].cpu().numpy())
        np.save(r'./trajectories/{}_t2p.npy'.format(key), trajectories_dict[key][1].cpu().numpy())
    logging.info(' Program complete!')
    logging.info('------------------------------')
    return
