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
from tqdm import tqdm
from typing import Dict,Type


GAMMA = 267.513e6 

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
        logging.info(" Fiber {} Volume: {:.3f}".format(i, len(fiber_spins[np.where(fiber_spins == i)]) / len(fiber_spins)))

    logging.info(' Cell Volume:    {:.3f} '.format(
        len(cells[cells > -1]) / len(spins))
        )
    
    logging.info(' Water Volume:   {:.3f} '.format(
        len(water_spins[water_spins > -1]) / len(spins))
        )
    
    v = 0

    for i in range(1, int(np.amax(fiber_spins))+1): v+= len(fiber_spins[np.where(fiber_spins == i)]) / len(fiber_spins)
    v += len(cells[cells > -1]) / len(spins)
    v +=  len(water_spins[water_spins > -1]) / len(spins)

    logging.info(' Total Volume:   {:.3f}'.format(v))

    return
    

def _package_data(self) -> Dict[str, Dict[str, Type[numba.cuda.cudadrv.devicearray.DeviceNDArray]]]:
    outputArgs = {'fiber_centers'    : {'data' : [], 'dtype' : np.float32},
                 'fiber_directions'  : {'data' : [], 'dtype' : np.float32},
                 'fiber_step'        : {'data' : [], 'dtype' : np.float32}, 
                 'fiber_radii'       : {'data' : [], 'dtype' : np.float32},
                 'fiber_theta'       : {'data' : [], 'dtype' : np.float32},
                 'curvature_params'  : {'data' : [], 'dtype' : np.float32},
                 'spin_positions_t0' : {'data' : [], 'dtype' : np.float32},
                 'spins_fiber_index' : {'data' : [], 'dtype' : np.int32  },
                 'cell_centers'      : {'data' : [], 'dtype' : np.float32},
                 'spins_cell_index'  : {'data' : [], 'dtype' : np.int32  },
                 'cell_step'         : {'data' : [], 'dtype' : np.float32},
                 'cell_radii'        : {'data' : [], 'dtype' : np.float32},
                 'spin_water_index'  : {'data' : [], 'dtype' : np.int32  },
                 'water_step'        : {'data' : [], 'dtype' : np.float32},
                 'gradient'          : {
                                        'data' : self.G, 
                                        'dtype': np.float32
                                        },
                 'phase'             : {
                                        'data' : np.zeros((len(self.spins), self.G.shape[0])), 
                                        'dtype': np.float32
                                        }
                 }
    
    #####################################################################################
    #                                       Package Data                                #
    #####################################################################################
    for fiber in self.fibers:
        outputArgs['fiber_centers'   ]['data'].append(fiber.center)                           
        outputArgs['fiber_directions']['data'].append(fiber.direction)     
        outputArgs['fiber_step'      ]['data'].append(np.sqrt(6.0*fiber.diffusivity*self.dt)) 
        outputArgs['fiber_radii'     ]['data'].append(fiber.radius)                             
        outputArgs['fiber_theta'     ]['data'].append(fiber.theta)
        outputArgs['curvature_params']['data'].append(
                                                      [fiber.__dict__['kappa'], fiber.__dict__['L'], fiber.__dict__['A'], fiber.__dict__['P']]
                                                      )
    for cell in self.cells:
        outputArgs['cell_centers']['data'].append(cell.center)
        outputArgs['cell_radii'  ]['data'].append(cell.radius)
        outputArgs['cell_step'   ]['data'].append(np.sqrt(6.0 * cell.diffusivity*self.dt))

    for spin in self.spins:
        outputArgs['spin_positions_t0']['data'].append(spin.position_t1m)
        outputArgs['spins_fiber_index']['data'].append(-1 if spin._get_bundle_index() is None else spin._get_fiber_index())
        outputArgs['spins_cell_index' ]['data'].append(spin._get_cell_index())
        outputArgs['spin_water_index' ]['data'].append(1 if np.logical_and(spin._get_bundle_index() is None, spin._get_cell_index() == -1) else -1)

    outputArgs['water_step']['data'].append(math.sqrt(6*self.water_diffusivity*self.dt))

    
    #####################################################################################
    #                              Send Data to GPU                                     #
    #####################################################################################
    for k, v in outputArgs.items():
        outputArgs[k]['data'] = cuda.to_device(
                                               np.array(v['data'], dtype = v['dtype'])
                                              )
    return outputArgs

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
    simulation_data    = _package_data(self)
    random_states_cuda = cuda.to_device(create_xoroshiro128p_states(len(self.spins), seed = 42))  

    Start = time.time()
    threads_per_block = 320
    blocks_per_grid = (len(self.spins) + (threads_per_block-1)) // threads_per_block
    logging.info('------------------------------')  
    logging.info(' Beginning Simulation...')
    logging.info('------------------------------')    


    for i in range(int(self.TE/self.dt)):
        sys.stdout.write('\r' + 'simDRIFT:  Step {:05d}'.format(i + 1)
                         + '/{:5d}'.format(  int(self.TE/self.dt) + 1) 
                         + ' | t = {:.5f}'.format(  1e3*(i+1)*self.dt)
                         + ' (ms) | TE = {:5f} (ms)'.format(1e3*self.TE)
                         )
        sys.stdout.flush()
        if i == 1:
            sys.stdout.write('\n')
            start = time.time()
            _diffusion_context_manager[blocks_per_grid,threads_per_block](random_states_cuda, 
                                                                          simulation_data['spin_positions_t0']['data'], 
                                                                          simulation_data['spins_fiber_index']['data'],
                                                                          simulation_data['fiber_centers'    ]['data'],
                                                                          simulation_data['fiber_step'       ]['data'],
                                                                          simulation_data['fiber_radii'      ]['data'],
                                                                          simulation_data['fiber_directions' ]['data'],
                                                                          simulation_data['spins_cell_index' ]['data'],
                                                                          simulation_data['cell_centers'     ]['data'],
                                                                          simulation_data['cell_step'        ]['data'],
                                                                          simulation_data['cell_radii'       ]['data'],
                                                                          simulation_data['water_step'       ]['data'],
                                                                          simulation_data['fiber_theta'      ]['data'],
                                                                          self.fiber_configuration == 'Void', 
                                                                          simulation_data['curvature_params' ]['data']
                                                                        )
            cuda.synchronize()
          
            _calculate_phase[blocks_per_grid, threads_per_block](simulation_data['phase'            ]['data'],
                                                                 simulation_data['spin_positions_t0']['data'],
                                                                 simulation_data['gradient'         ]['data'],
                                                                 self.dt,
                                                                 i
                                                                 )
            
            cuda.synchronize()
            end = time.time()

            logging.info(' Step 2 elapsed in {:.4f} (sec.) ... projected total simulation time is {:.4f} (sec.)'.format(end - start, (end - start ) * (self.Delta / self.dt) ))
        else:
            _diffusion_context_manager[blocks_per_grid,threads_per_block](random_states_cuda, 
                                                                          simulation_data['spin_positions_t0']['data'], 
                                                                          simulation_data['spins_fiber_index']['data'],
                                                                          simulation_data['fiber_centers'    ]['data'],
                                                                          simulation_data['fiber_step'       ]['data'],
                                                                          simulation_data['fiber_radii'      ]['data'],
                                                                          simulation_data['fiber_directions' ]['data'],
                                                                          simulation_data['spins_cell_index' ]['data'],
                                                                          simulation_data['cell_centers'     ]['data'],
                                                                          simulation_data['cell_step'        ]['data'],
                                                                          simulation_data['cell_radii'       ]['data'],
                                                                          simulation_data['water_step'       ]['data'],
                                                                          simulation_data['fiber_theta'      ]['data'],
                                                                          self.fiber_configuration == 'Void', 
                                                                          simulation_data['curvature_params' ]['data']
                                                                        )


            cuda.synchronize()
            
            _calculate_phase[blocks_per_grid, threads_per_block](simulation_data['phase'            ]['data'],
                                                                 simulation_data['spin_positions_t0']['data'],
                                                                 simulation_data['gradient'         ]['data'],
                                                                 self.dt,
                                                                 i
                                                                 )
            
            cuda.synchronize()
        
    End = time.time()
    sys.stdout.write('\n')
    logging.info(' Simulation complete!')
    logging.info(' Elapsed time: {} seconds'.format(round((End-Start)),3))

    self.water_key     = simulation_data['spin_water_index' ]['data'].copy_to_host()
    spin_positions_t2p = simulation_data['spin_positions_t0']['data'].copy_to_host()
    self.phase         = simulation_data['phase']['data'].copy_to_host()

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
                               water_step,  
                               theta_cuda,
                               void, 
                               curvature_params):
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
       
    if spin_in_fiber_at_index[i] > -1 or spin_in_cell_at_index[i] > -1:       
        if spin_in_fiber_at_index[i] > -1:
            walk_in_fiber._diffusion_in_fiber(i, 
                                            random_states,
                                            fiber_centers[spin_in_fiber_at_index[i],:],
                                            fiber_radii[spin_in_fiber_at_index[i]],
                                            fiber_directions[spin_in_fiber_at_index[i],:],
                                            fiber_step[spin_in_fiber_at_index[i]], 
                                            theta_cuda[spin_in_fiber_at_index[i]],
                                            spin_positions,
                                            curvature_params[spin_in_fiber_at_index[i], :]
                                            )
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
                                            theta_cuda, 
                                            void,
                                            curvature_params
                                        )
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
                                          water_step[0],
                                          theta_cuda,
                                          curvature_params
                                          )
        return
            

@numba.cuda.jit
def _calculate_phase(phases,
                     positions,
                     G,
                     dt,
                     t
                     ):
    
    i = cuda.grid(1)
    if i > positions.shape[0]:
        return
    
    for m in range(G.shape[0]):
        phases[i, m] += ( 
                        GAMMA 
                        * dt
                        * (
                            (G[m, t, 0]   * positions[i, 0])
                            + (G[m, t, 1] * positions[i, 1])
                            + (G[m, t, 2] * positions[i, 2]) 
                        )
                    ) 
    return
