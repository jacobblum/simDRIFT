import os  
import sys
import shutil
import multiprocessing as mp
from multiprocessing.sharedctypes import Value
from scipy.io import savemat
import numpy as np 
import numba 
from numba import jit, njit, cuda
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
import time
import jp as jp

@numba.cuda.jit(fastmath=True)
def _find_spin_locations_kernel(spinInFiber1_i, spinInFiber2_i, spinInCell_i, initialSpinPositions, fiberCenters, cellCenters, fiberRotationReference):
    i = cuda.grid(1)
    if i > initialSpinPositions.shape[0]:
        return 
    
    """ Find the position of each of the ensemble's spins within the imaging voxel

        Parameters
        ----------
        spinInFiber_i: 1-d ndarray
            The index of the fiber a spin is located within; -1 if False, 0... N_fibers if True.
        spinInCell_i: 1-d ndarray
            The index of the cell a spin is located within; -1 if False, 0...N_cells if True
        initialSpinPositions : N_spins x 3 ndarray
            The initialSpinPositions[i,:] are the 3 spatial positions of the spin at its initial position
        fiberCenters: N_{fibers} x 6 ndarray
            The spatial position, rotmat index, intrinsic diffusivity, and radius of the i-th fiber
        cellCenters: N_{cells} x 4 ndarray
            The 3-spatial dimensions and radius of the i-th cell
        fiberRotationReference: 2 x 3 ndarray
            The Ry(Theta_{i}).dot([0,0,1]) vector  

        Returns
        -------
        spinInFiber_i : 1-d ndarray
            See parameters note
        spinInCell_i: 1-d ndarray
            See parameters note             

        Notes
        -----
        None

        References
        ----------
        None

        Examples
        --------
        >>> self.find_spin_locations.forall(self.numSpins)(spinInFiber_i_GPU, spinInCell_i_GPU, spinInitialPositions_GPU, fiberCenters_GPU, cellCenters_GPU, self.fiberRotationReference)
    """
    KeyFiber1 = numba.int32(-1)
    KeyFiber2 = numba.int32(-1)
    KeyCell = numba.int32(-1)
    spinPosition = cuda.local.array(shape = 3, dtype = numba.float32)
    fiber1Distance = numba.float32(0.0)
    fiber2Distance = numba.float32(0.0)
    cellDistance = numba.float32(0.0)
    for k in range(spinPosition.shape[0]): 
        spinPosition[k] = initialSpinPositions[i,k]
    for j in range(fiberCenters.shape[0]): 
        rotationIndex = int(fiberCenters[j,4])
        if int(fiberCenters[j,5]) == int(fiberCenters[0,5]):
            fiberDistance = jp.euclidean_distance(spinPosition, fiberCenters[j,0:3], fiberRotationReference[rotationIndex,:], 'fiber')
            if fiberDistance < fiberCenters[j,3]: 
                KeyFiber1 = j
                break
        else:
            fiberDistance = jp.euclidean_distance(spinPosition, fiberCenters[j,0:3], fiberRotationReference[rotationIndex,:], 'fiber')
            if fiberDistance < fiberCenters[j,3]: 
                KeyFiber2 = j
                break
    
    for j in range(cellCenters.shape[0]):
        cellDistance = jp.euclidean_distance(spinPosition, cellCenters[j,0:3], fiberRotationReference[0,:], 'cell')
        if cellDistance < cellCenters[j,3]:
            KeyCell = j
            break
    spinInCell_i[i] = KeyCell
    spinInFiber1_i[i] = KeyFiber1
    spinInFiber2_i[i] = KeyFiber2
    return

def _find_spin_locations(initial_spin_positions, fiber_centers, cell_centers, rotation_reference, savePath, cfg_path):
    data_dir = savePath
    if not os.path.exists(data_dir): os.mkdir(data_dir)
    path, file = os.path.split(cfg_path)  
    if not os.path.exists(data_dir + os.sep + file): shutil.move(cfg_path, data_dir + os.sep + file)

    spin_in_fiber1_at_index = -1.0*np.ones(initial_spin_positions.shape[0])
    spin_in_fiber2_at_index = -1.0*np.ones(initial_spin_positions.shape[0])
    spin_in_cell_at_index  = -1.0*np.ones(initial_spin_positions.shape[0])
    spin_in_fiber1_at_index_gpu, spin_in_fiber2_at_index_gpu, spin_in_cell_at_index_gpu = cuda.to_device(spin_in_fiber1_at_index), cuda.to_device(spin_in_fiber1_at_index), cuda.to_device(spin_in_cell_at_index)
    fiber_centers_gpu, cell_centers_gpu = cuda.to_device(fiber_centers), cuda.to_device(cell_centers)
    initial_spin_positions_gpu = cuda.to_device(initial_spin_positions)
    
    threads_per_block = 64
    blocks_per_grid = (initial_spin_positions.shape[0] + (threads_per_block-1)) // threads_per_block
    
    _find_spin_locations_kernel[blocks_per_grid,threads_per_block](spin_in_fiber1_at_index_gpu, spin_in_fiber2_at_index_gpu, spin_in_cell_at_index_gpu, initial_spin_positions_gpu, fiber_centers_gpu, cell_centers_gpu, rotation_reference)
    ## Return These
    spin_in_fiber1_at_index_output, spin_in_fiber2_at_index_output, spin_in_cell_at_index_output = spin_in_fiber1_at_index_gpu.copy_to_host(), spin_in_fiber2_at_index_gpu.copy_to_host(), spin_in_cell_at_index_gpu.copy_to_host()
    np.save(data_dir + os.sep + "indFiber1Spins.npy", spin_in_fiber1_at_index_output)
    np.save(data_dir + os.sep + "indFiber2Spins.npy", spin_in_fiber2_at_index_output)
    np.save(data_dir + os.sep + "indCellsSpins.npy", spin_in_cell_at_index_output)
    return spin_in_fiber1_at_index_output, spin_in_fiber2_at_index_output, spin_in_cell_at_index_output

