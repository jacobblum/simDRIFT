import os  
import sys
import multiprocessing as mp
from multiprocessing.sharedctypes import Value
import numpy as np 
import numba 
from numba import jit, njit, cuda
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
from src.jp import linalg
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
    r"""
    Locate spins within resident microstructural elements
    
    Args:
        resident_fiber_indxs_cuda (numba.cuda.cudadrv.devicearray.DeviceNDArray): Array to write resident fiber indices to
        resident_cell_indxs_cuda (numba.cuda.cudadrv.devicearray.DeviceNDArray): Array to write resident cell indicies to
        fiber_centers_cuda (numba.cuda.cudadrv.devicearray.DeviceNDArray): coordinates of the fiber centers
        fiber_directions_cuda (numba.cuda.cudadrv.devicearray.DeviceNDArray): directions of the fibers
        fiber_radii_cuda (numba.cuda.cudadrv.devicearray.DeviceNDArray): fiber radii
        cell_centers_cuda (numba.cuda.cudadrv.devicearray.DeviceNDArray): coordinates of the cell centers
        cell_radii_cuda (numba.cuda.cudadrv.devicearray.DeviceNDArray): cell radii
        spin_positions_cuda (numba.cuda.cudadrv.devicearray.DeviceNDArray): initial spin positions
    """
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

def _find_spin_locations(self):
    r"""
    helper function to find initial spin locations
    """
    resident_fiber_indxs_cuda = cuda.to_device( -1 * np.ones(shape = (len(self.spins),), dtype= np.int32))
    resident_cell_indxs_cuda  = cuda.to_device( -1 * np.ones(shape = (len(self.spins),), dtype= np.int32))
    fiber_centers_cuda        = cuda.to_device(np.array([fiber._get_center() for fiber in self.fibers], dtype= np.float32))
    fiber_directions_cuda     = cuda.to_device(np.array([fiber._get_direction() for fiber in self.fibers], dtype= np.float32))
    fiber_radii_cuda          = cuda.to_device(np.array([fiber._get_radius() for fiber in self.fibers], dtype= np.float32))
    cell_centers_cuda         = cuda.to_device(np.array([cell._get_center() for cell in self.cells], dtype= np.float32))
    cell_radii_cuda           = cuda.to_device(np.array([cell._get_radius() for cell in self.cells], dtype=np.float32))
    spin_positions_cuda       = cuda.to_device(np.array([spin._get_position_t1m() for spin in self.spins], dtype= np.float32))
    threads_per_block = 128
    blocks_per_grid = (len(self.spins) + (threads_per_block-1)) // threads_per_block

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
    spinInds = range(len(self.spins))
    negSpinInds = -1 * np.ones(shape = (len(self.spins),), dtype= np.int32)
    resident_water_indxs = np.where(np.logical_and(resident_fiber_indxs < 0, resident_cell_indxs < 0),spinInds,negSpinInds)
    
    for i, spin in enumerate(self.spins):
        spin._set_fiber_index(resident_fiber_indxs[i])
        spin._set_fiber_bundle(self.fibers[resident_fiber_indxs[i]]._get_bundle())
        spin._set_cell_index(resident_cell_indxs[i])
        spin._set_water_index(resident_water_indxs[i])

    return
