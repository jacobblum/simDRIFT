import numpy as np  
import os  
import sys
from src.data import diffusion_schemes
import nibabel as nb
import time
import logging
import torch 

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

   
def _signal(spins: list, bvals: np.ndarray, bvecs: np.ndarray, Delta: float, dt: float, SNR = None) -> np.ndarray:
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
    
    gamma = 42.58 # MHz/T - The proton gyromagnetic ratio
    delta = dt    # ms - From the narrow pulse approximation. More convenient for GPU memory management 

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
    """Helper function to organize and store compartment specific and combined trajectories and their incident signals

    :return: signals with associated labels (``signals_dict``), trajectories with associated labels (``trajectories_dict``)
    :rtype: dictionaries
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

def _save_data(self):
    """Helper function that saves signals and trajectories to the current directory.
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
