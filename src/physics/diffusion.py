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
import sys
import operator
import logging
from src.physics import walk_in_fiber, walk_in_cell, walk_in_water


def _caclulate_volumes(spins):
    """Calculates empirical volume fraction for each simulated compartment (i.e., fibers, cells, and water)

    :param spins: A one-dimensional list (length = ``n_walkers``) containing each instance of ``objects.spin()``. Each entry corresponds to one spin from the spin ensemble. 
    :type spins: list
    """
    fiber_spins = np.array([-1 if spin._get_bundle_index() is None else spin._get_bundle_index() for spin in spins])
    cells  = np.array([spin._get_cell_index() for spin in spins])

    water_spins = np.array([1 if np.logical_and(spin._get_bundle_index() is None, spin._get_cell_index() == -1) else -1 for spin in spins])

    logging.info('------------------------------')  
    logging.info(' Empirical Volume Fractions')
    logging.info('------------------------------')   
    
    for i in range(1, int(np.amax(fiber_spins))+1):
        logging.info(f" Fiber {i} Volume: {len(fiber_spins[np.where(fiber_spins == i)]) / len(fiber_spins)}")

    logging.info(' Cell Volume:    {} '.format(
        len(cells[cells > -1]) / len(spins))
        )
    
    logging.info(' Water Volume:   {} '.format(
        len(water_spins[water_spins > -1]) / len(spins))
        )
    
    v = 0

    for i in range(1, int(np.amax(fiber_spins))+1): v+= len(fiber_spins[np.where(fiber_spins == i)]) / len(fiber_spins)
    v += len(cells[cells > -1]) / len(spins)
    v +=  len(water_spins[water_spins > -1]) / len(spins)

    logging.info(' Total Volume:   {}'.format(v))

    return
    
def _simulate_diffusion(self) -> None:
    """Iterates over the range :math:`t \in [0, \Delta ]` with a step size of :math:`\dd{t}`.

    :param self: the ``dmri_simulation`` object
    :type self: class object
    :param spins: A one-dimensional list (length = ``n_walkers``) containing each instance of ``objects.spin()``. Each entry corresponds to one spin from the spin ensemble. 
    :type spins: list
    :param cells: A one-dimensional list (length = ``n_cells``) containing each instance of ``objects.cell()``. Each entry corresponds to one cell within the simulated imaging voxel. 
    :type cells: list
    :param fibers: A list (size = ``n_fibers``\ :math:`\\times`\ ``n_fibers``) containing each instance of ``objects.fiber()``. Each entry corresponds to one fiber within the simulated imaging voxel. 
    :type fibers: list
    :param Delta: The diffusion time supplied by the user in units of milliseconds.
    :type Delta: float
    :param dt: The user-supplied duration for each time step in units of milliseconds. Assumed to be equal to :math:`\delta` under the narrow-pulse approximation.
    :type dt: float
    :param water_diffusivity: The user-supplied diffusivity for free water, in units of :math:`{\mathrm{μm}^2}\\, \mathrm{ms}^{-1}`.
    :type water_diffusivity: float
    :return: Updated spin trajectories within each instance of the spin object in the spin list

    .. note::
        At each iteration, the updated spin position is written to ``spin_positions_cuda``
    """

    _caclulate_volumes(self.spins)

    random_states_cuda            = cuda.to_device(create_xoroshiro128p_states(len(self.spins), seed = 42))    
    fiber_centers_cuda            = cuda.to_device(np.array([fiber._get_center() for fiber in self.fibers], dtype= np.float32))
    fiber_directions_cuda         = cuda.to_device(np.array([fiber._get_direction() for fiber in self.fibers], dtype= np.float32))
    fiber_step_cuda               = cuda.to_device(np.array([math.sqrt(6*fiber._get_diffusivity()*self.dt) for fiber in self.fibers], dtype= np.float32))
    fiber_radii_cuda              = cuda.to_device(np.array([fiber._get_radius() for fiber in self.fibers], dtype= np.float32))
    spin_positions_cuda           = cuda.to_device(np.array([spin._get_position_t1m() for spin in self.spins], dtype= np.float32))
    spin_in_fiber_at_index_cuda   = cuda.to_device(np.array([-1 if spin._get_bundle_index() is None else spin._get_fiber_index() for spin in self.spins]))
    cell_centers_cuda             = cuda.to_device(np.array([cell._get_center() for cell in self.cells], dtype=np.float32))
    spin_in_cell_at_index_cuda    = cuda.to_device(np.array([spin._get_cell_index() for spin in self.spins]))
    cell_step_cuda                = cuda.to_device(np.array([math.sqrt(6*cell._get_diffusivity()*self.dt) for cell in self.cells], dtype= np.float32))
    cell_radii_cuda               = cuda.to_device(np.array([cell._get_radius() for cell in self.cells], dtype=np.float32)) 
    spin_in_water_at_index_cuda   = cuda.to_device(spin_in_cell_at_index_cuda)  

    spin_in_water_at_index_cuda = []
    water_step = []
    for ii in range(len(self.spins)):
        if np.logical_and(spin_in_cell_at_index_cuda[ii] == -1, spin_in_fiber_at_index_cuda[ii] == -1):
            spin_in_water_at_index_cuda.append(1)
            
            if np.random.rand() > 0.5:
                water_step.append(math.sqrt(6*self.water_diffusivity*self.dt))
            else: 
                spin_in_water_at_index_cuda[ii] = 2
                water_step.append(math.sqrt(6*10.0*self.dt))

        else:
            spin_in_water_at_index_cuda.append(-1)
            water_step.append(math.sqrt(6*self.water_diffusivity*self.dt))

    spin_in_water_at_index_cuda = cuda.to_device(np.array(spin_in_water_at_index_cuda))
    water_step                  = cuda.to_device(np.array(water_step))


    Start = time.time()
    threads_per_block = 320
    blocks_per_grid = (len(self.spins) + (threads_per_block-1)) // threads_per_block
    logging.info('------------------------------')  
    logging.info(' Beginning Simulation...')
    logging.info('------------------------------')    
    for i in range(int(self.Delta/self.dt)):
        sys.stdout.write('\r' + 'simDRIFT:  Step ' +  str(i+1) + '/' + str(int(self.Delta/self.dt)))
        sys.stdout.flush()
        
        _diffusion_context_manager[blocks_per_grid,threads_per_block](random_states_cuda, 
                                                                      spin_positions_cuda, 
                                                                      spin_in_fiber_at_index_cuda, 
                                                                      fiber_centers_cuda,
                                                                      fiber_step_cuda,
                                                                      fiber_radii_cuda,
                                                                      fiber_directions_cuda, 
                                                                      spin_in_cell_at_index_cuda, 
                                                                      cell_centers_cuda, 
                                                                      cell_step_cuda,
                                                                      cell_radii_cuda,
                                                                      spin_in_water_at_index_cuda,
                                                                      water_step, 
                                                                      self.fiber_configuration == 'Void'
                                                                      )
        
        cuda.synchronize()
    End = time.time()
    sys.stdout.write('\n')
    logging.info(' Simulation complete!')
    logging.info(' Elapsed time: {} seconds'.format(round((End-Start)),3))
    
    self.water_key = spin_in_water_at_index_cuda.copy_to_host()
    
    spin_positions_t2p = spin_positions_cuda.copy_to_host()
    for ii, spin in enumerate(self.spins):
        spin._set_position_t2p(spin_positions_t2p[ii,:])
    
    return 


@numba.cuda.jit
def _diffusion_context_manager(random_states, 
                               spin_positions, 
                               spin_in_fiber_at_index, 
                               fiber_centers,
                               fiber_step,
                               fiber_radii,
                               fiber_directions, 
                               spin_in_cell_at_index, 
                               cell_centers,
                               cell_step,
                               cell_radii,
                               spin_in_water_at_index,
                               water_step,  
                               void):
    """Helper function to segment each spin into the relevant ``physics`` module for its resident compartment

    :param random_states: Randomized states
    :param spin_positions: Initial spin positions
    :param spin_in_fiber_at_index: Spin indices for each fiber
    :param fiber_centers: Geometric centers of each fiber
    :param fiber_step: Length of diffusion step to take for each fiber type, in units of micrometers
    :param fiber_radii: Radius of each fiber, in units of micrometers
    :param fiber_directions: Relative fiber rotations
    :param spin_in_cell_at_index: Spin indices for each cell
    :param cell_centers: Geometric centers for each cell
    :param cell_step: Length of diffusion step to take for each cell type, in units of micrometers
    :param cell_radii: Radius of each cell, in units of micrometers
    :param water_step: Length of diffusion step to take in free water, in units of micrometers
    :param void: Boolean argument, True if fiber configuration is void, False otherwise
    """
    i = cuda.grid(1)
    if i > spin_positions.shape[0]:
        return
    
    if spin_in_fiber_at_index[i] > -1:
        walk_in_fiber._diffusion_in_fiber(i, 
                                          random_states,
                                          fiber_centers[spin_in_fiber_at_index[i],:],
                                          fiber_radii[spin_in_fiber_at_index[i]],
                                          fiber_directions[spin_in_fiber_at_index[i],:],
                                          fiber_step[spin_in_fiber_at_index[i]], 
                                          spin_positions)
        
        return
    
    if spin_in_cell_at_index[i] > -1:
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
        
    if spin_in_water_at_index[i] > -1:
        walk_in_water._diffusion_in_water(i,
                                          random_states,
                                          fiber_centers,
                                          fiber_directions, 
                                          fiber_radii,
                                          cell_centers,
                                          cell_radii, 
                                          spin_positions, 
                                          water_step[i])
        
    else:
        raise Exception('Spin not in fiber, cell, or water? Not good... if this happens call me @ 612-214-6025')
        

    return



