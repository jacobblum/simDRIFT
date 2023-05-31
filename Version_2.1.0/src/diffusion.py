from ast import Del
import multiprocessing as mp
from multiprocessing.sharedctypes import Value
import numpy as np 
import numba 
from numba import jit, njit, cuda
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
import time
from physics import walk_in_fiber, walk_in_cell, walk_in_water
import sys
import operator
import logging


def _caclulate_volumes(spins):
    
    fiber1 = np.array([spin._get_fiber_index() if spin._get_bundle_index() == 1 else -1 for spin in spins])
    fiber2 = np.array([spin._get_fiber_index() if spin._get_bundle_index() == 2 else -1 for spin in spins])
    cells  = np.array([spin._get_cell_index() for spin in spins])
    water  = np.array([spin._get_water_index() for spin in spins])

    logging.info('------------------------------')  
    logging.info(' Empirical Volume Fractions')
    logging.info('------------------------------')   
    
    
    logging.info(' Fiber 1 Volume: {}'.format(
        len(fiber1[fiber1 > -1]) / len(spins)))
    logging.info(' Fiber 2 Volume: {}'.format(
        len(fiber2[fiber2 > -1]) / len(spins)))
    logging.info('    Cell Volume: {} '.format(
        len(cells[cells > -1]) / len(spins)))
    logging.info('   Water Volume: {} '.format(
        len(water[water > -1]) / len(spins)))
    logging.info('          TOTAL: {} '.format(
        (len(water[water > -1])+len(cells[cells > -1])+len(fiber1[fiber1 > -1])+len(fiber2[fiber2 > -1]))/len(spins)))
        
def _simulate_diffusion(self, spins:  list, cells:  list, fibers: list, Delta : float, dt : float, water_diffusivity : float) -> None:
    _caclulate_volumes(spins)

    random_states_cuda            = cuda.to_device(create_xoroshiro128p_states(len(spins), seed = 42))    
    fiber_centers_cuda            = cuda.to_device(np.array([fiber._get_center() for fiber in fibers], dtype= np.float32))
    fiber_directions_cuda         = cuda.to_device(np.array([fiber._get_direction() for fiber in fibers], dtype= np.float32))
    fiber_step_cuda               = cuda.to_device(np.array([math.sqrt(6*fiber._get_diffusivity()*dt) for fiber in fibers], dtype= np.float32))
    fiber_radii_cuda              = cuda.to_device(np.array([fiber._get_radius() for fiber in fibers], dtype= np.float32))
    spin_positions_cuda           = cuda.to_device(np.array([spin._get_position_t1m() for spin in spins], dtype= np.float32))
    spin_in_fiber_1_at_index_cuda = cuda.to_device(np.array([spin._get_fiber_index() if spin._get_bundle_index() == 1 else -1 for spin in spins]))
    spin_in_fiber_2_at_index_cuda = cuda.to_device(np.array([spin._get_fiber_index() if spin._get_bundle_index() == 2 else -1 for spin in spins]))

    cell_centers_cuda             = cuda.to_device(np.array([cell._get_center() for cell in cells], dtype=np.float32))
    spin_in_cell_at_index_cuda    = cuda.to_device(np.array([spin._get_cell_index() for spin in spins]))
    cell_step_cuda                = cuda.to_device(np.array([math.sqrt(6*cell._get_diffusivity()*dt) for cell in cells], dtype= np.float32))
    cell_radii_cuda               = cuda.to_device(np.array([cell._get_radius() for cell in cells], dtype=np.float32)) 
    water_step                    = cuda.to_device(np.array([math.sqrt(6*water_diffusivity*dt)])) 


    Start = time.time()
    threads_per_block = 320
    blocks_per_grid = (len(spins) + (threads_per_block-1)) // threads_per_block
    logging.info('------------------------------')  
    logging.info(' Beginning Simulation...')
    logging.info('------------------------------')    
    for i in range(int(Delta/dt)):
        sys.stdout.write('\r' + 'dMRI-SIM:  Step ' +  str(i+1) + '/' + str(int(Delta/dt)))
        sys.stdout.flush()
        _diffusion_context_manager[blocks_per_grid,threads_per_block](random_states_cuda, 
                                                                      spin_positions_cuda, 
                                                                      spin_in_fiber_1_at_index_cuda, 
                                                                      spin_in_fiber_2_at_index_cuda, 
                                                                      fiber_centers_cuda,
                                                                      fiber_step_cuda,
                                                                      fiber_radii_cuda,
                                                                      fiber_directions_cuda, 
                                                                      spin_in_cell_at_index_cuda, 
                                                                      cell_centers_cuda, 
                                                                      cell_step_cuda,
                                                                      cell_radii_cuda,
                                                                      water_step, 
                                                                      (self.parameters['fiber_configuration'] == 'Void'))
        
        cuda.synchronize()

    End = time.time()
    sys.stdout.write('\n')
    logging.info(' Simulation complete!')
    logging.info(' Elapsed time: {} seconds'.format(round((End-Start)),3))
    spin_positions_t2p = spin_positions_cuda.copy_to_host()
    for ii, spin in enumerate(spins):
        spin._set_position_t2p(spin_positions_t2p[ii,:])
    
    self.spins = spins
    return 

@numba.cuda.jit
def _diffusion_context_manager(random_states, 
                               spin_positions, 
                               spin_in_fiber_1_at_index, 
                               spin_in_fiber_2_at_index, 
                               fiber_centers,
                               fiber_step,
                               fiber_radii,
                               fiber_directions, 
                               spin_in_cell_at_index, 
                               cell_centers,
                               cell_step,
                               cell_radii,
                               water_step,  
                               void):
    """
    Parameters:

    rng_states:
    spinPositions: (N_spins, 3) numpy array
    spin_in_fiber_key: (N_spins, ) numpy array, the spin_in_fiber[i] is the fiber index of the i-th spin. -1 if spin not in fiber.
    """
    i = cuda.grid(1)
    if i > spin_positions.shape[0]:
        return
    
    if spin_in_fiber_1_at_index[i] > -1:
        walk_in_fiber._diffusion_in_fiber(i, 
                                          random_states,
                                          fiber_centers[spin_in_fiber_1_at_index[i],:],
                                          fiber_radii[spin_in_fiber_1_at_index[i]],
                                          fiber_directions[spin_in_fiber_1_at_index[i],:],
                                          fiber_step[spin_in_fiber_1_at_index[i]], 
                                          spin_positions)
        
        return
    
    
    
    elif spin_in_fiber_2_at_index[i] > -1:
        walk_in_fiber._diffusion_in_fiber(i, 
                                          random_states,
                                          fiber_centers[spin_in_fiber_2_at_index[i],:],
                                          fiber_radii[spin_in_fiber_2_at_index[i]],
                                          fiber_directions[spin_in_fiber_2_at_index[i],:],
                                          fiber_step[spin_in_fiber_2_at_index[i]], 
                                          spin_positions)
        
        return

    elif spin_in_cell_at_index[i] > -1:
        walk_in_cell._diffusion_in_cell(i, 
                                        random_states, 
                                        cell_centers[spin_in_cell_at_index[i], :], 
                                        cell_radii[spin_in_cell_at_index[i]],
                                        cell_step[spin_in_cell_at_index[i]], 
                                        fiber_centers,
                                        fiber_radii,
                                        fiber_directions, 
                                        spin_positions,  
                                        void)
        
        return
        
    else:
        walk_in_water._diffusion_in_water(i,
                                          random_states,
                                          fiber_centers,
                                          fiber_directions, 
                                          fiber_radii,
                                          cell_centers,
                                          cell_radii, 
                                          spin_positions, 
                                          water_step[0])
    return
