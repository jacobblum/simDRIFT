import os  
import sys
import multiprocessing as mp
from multiprocessing.sharedctypes import Value
from scipy.io import savemat
import numpy as np 
import numba 
from numba import jit, njit, cuda
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
from jp import linalg
import math

@numba.cuda.jit
def _find_spin_locations_kernel(resident_fiber_indxs_cuda: numba.cuda.cudadrv.devicearray.DeviceNDArray, 
                                resident_cell_indxs_cuda:  numba.cuda.cudadrv.devicearray.DeviceNDArray , 
                                fiber_centers_cuda:        numba.cuda.cudadrv.devicearray.DeviceNDArray,
                                fiber_directions_cuda:     numba.cuda.cudadrv.devicearray.DeviceNDArray,
                                fiber_radii_cuda:          numba.cuda.cudadrv.devicearray.DeviceNDArray,
                                cell_centers_cuda:         numba.cuda.cudadrv.devicearray.DeviceNDArray,
                                cell_radii_cuda:           numba.cuda.cudadrv.devicearray.DeviceNDArray,
                                spin_positions_cuda:       numba.cuda.cudadrv.devicearray.DeviceNDArray
                                ) -> None:
    i = cuda.grid(1)
  
    if i > spin_positions_cuda.shape[0]:
        return
    
    u3 = cuda.local.array(shape = 3, dtype= numba.float32)
    for j in range(u3.shape[0]):
        u3[j] = 1/math.sqrt(3) * 1.
        
    for j in range(fiber_centers_cuda.shape[0]):
        dFv = linalg.dL2(spin_positions_cuda[i,:], fiber_centers_cuda[j,:], fiber_directions_cuda[j,:],True)
        if dFv < fiber_radii_cuda[j]:
            resident_fiber_indxs_cuda[i] = j
            break
    
    for j in range(cell_centers_cuda.shape[0]):
       dC = linalg.dL2(spin_positions_cuda[i,:], cell_centers_cuda[j,:], u3, False)
       if dC < cell_radii_cuda[j]:
           resident_cell_indxs_cuda[i] = j
           break

    return

def _find_spin_locations(self, spins, cells, fibers):
   
    """ Unfortunately it is not possible to pass python objects to the GPU. This function will make the relevant arrays for the GPU to operate on 
        and then update the python object classes accordingly

        Fiber Attributes
         - self._center: (3,) np.ndarray
         - self._direction: (3,) np.ndarray
         - self._radius: float

        Cell Attributes
         - None Currently

        Spin Attributes: 
         - self._position_t1m: (3,) np.ndarray
        
        the _find_spin_locations_kernel will compute: 
                                        
         - d(spin[i]._position_t1m(), fiber[j]._center()) =  || spin[i]._position_t1m() - fiber[j]._center() ||_{L2} 
                                                                                           -  proj_{fiber[j]._direction()} || spin[i]._position_t1m() - fiber[j]._center() ||_{L2} for i = 1,2,... n_walkers, j = 1,2,... n_fibers

         - d(spin[i]._position_t1m(), cell[j]._center()) = || spin[i]._position_t1m() - cell[j]._center() ||_{L2} for i = 1,2,... n_walkers, j = 1,2,... n_cells
                                        
        """
    
    
    """ Declare Cuda Device Arrays """

    resident_fiber_indxs_cuda = cuda.to_device( -1 * np.ones(shape = (len(spins),), dtype= np.int32))
    resident_cell_indxs_cuda  = cuda.to_device( -1 * np.ones(shape = (len(spins),), dtype= np.int32))
    fiber_centers_cuda        = cuda.to_device(np.array([fiber._get_center() for fiber in fibers], dtype= np.float32))
    fiber_directions_cuda     = cuda.to_device(np.array([fiber._get_direction() for fiber in fibers], dtype= np.float32))
    fiber_radii_cuda          = cuda.to_device(np.array([fiber._get_radius() for fiber in fibers], dtype= np.float32))
    cell_centers_cuda         = cuda.to_device(np.array([cell._get_center() for cell in cells], dtype= np.float32))
    cell_radii_cuda           = cuda.to_device(np.array([cell._get_radius() for cell in cells], dtype=np.float32))
    spin_positions_cuda       = cuda.to_device(np.array([spin._get_position_t1m() for spin in spins], dtype= np.float32))
    threads_per_block = 128
    blocks_per_grid = (len(spins) + (threads_per_block-1)) // threads_per_block

    _find_spin_locations_kernel[blocks_per_grid,threads_per_block](resident_fiber_indxs_cuda,
                                                                   resident_cell_indxs_cuda,
                                                                   fiber_centers_cuda,
                                                                   fiber_directions_cuda,
                                                                   fiber_radii_cuda,
                                                                   cell_centers_cuda,
                                                                   cell_radii_cuda,
                                                                   spin_positions_cuda)
    


    resident_fiber_indxs = resident_fiber_indxs_cuda.copy_to_host()
    resident_cell_indxs  = resident_cell_indxs_cuda.copy_to_host()
    spinInds = range(len(spins))
    negSpinInds = -1 * np.ones(shape = (len(spins),), dtype= np.int32)
    resident_water_indxs = np.where(np.logical_and(resident_fiber_indxs < 0, resident_cell_indxs < 0),spinInds,negSpinInds)
    
    for i, spin in enumerate(spins):
        spin._set_fiber_index(resident_fiber_indxs[i])
        spin._set_fiber_bundle(fibers[resident_fiber_indxs[i]]._get_bundle())
        spin._set_cell_index(resident_cell_indxs[i])
        spin._set_water_index(resident_water_indxs[i])

    self.spins = spins

    return
