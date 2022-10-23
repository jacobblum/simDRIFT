import numpy as np 
import numba 
from numba import jit, cuda
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
import jp
import time 

def _find_spin_locations(initial_spin_positions, fiber_centers, cell_centers, rotation_reference):
    spin_in_fiber_at_index = -1.0*np.ones(initial_spin_positions.shape[0])
    spin_in_cell_at_index  = -1.0*np.ones(initial_spin_positions.shape[0])
    spin_in_fiber_at_index_gpu, spin_in_cell_at_index_gpu = cuda.to_device(spin_in_fiber_at_index), cuda.to_device(spin_in_cell_at_index)
    fiber_centers_gpu, cell_centers_gpu = cuda.to_device(fiber_centers), cuda.to_device(cell_centers)
    initial_spin_positions_gpu = cuda.to_device(initial_spin_positions)
    
    threads_per_block = 256
    blocks_per_grid = (initial_spin_positions.shape[0] + (threads_per_block-1)) // threads_per_block
    
    _find_spin_locations_kernel[2048,512](spin_in_fiber_at_index_gpu, spin_in_cell_at_index_gpu, initial_spin_positions_gpu, fiber_centers_gpu, cell_centers_gpu, rotation_reference)
    ## Return These
    spin_in_fiber_at_index_output, spin_in_cell_at_index_output = spin_in_fiber_at_index_gpu.copy_to_host(), spin_in_cell_at_index_gpu.copy_to_host()
    return spin_in_fiber_at_index_output, spin_in_cell_at_index_output
@cuda.jit 
def _find_spin_locations_kernel(spinInFiber_i, spinInCell_i, initialSpinPositions, fiberCenters, cellCenters, fiberRotationReference):
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
    KeyFiber = numba.int32(-1)
    KeyCell = numba.int32(-1)
    spinPosition = cuda.local.array(shape = 3, dtype = numba.float32)
    fiberDistance = numba.float32(0.0)
    cellDistance = numba.float32(0.0)
    for k in range(spinPosition.shape[0]): spinPosition[k] = initialSpinPositions[i,k]
    for j in range(fiberCenters.shape[0]): 
        rotationIndex = int(fiberCenters[j,4])
        fiberDistance = jp.euclidean_distance(spinPosition, fiberCenters[j,0:3], fiberRotationReference[rotationIndex,:], 'fiber')
        if fiberDistance < fiberCenters[j,3]: 
            KeyFiber = j
            break
    
    for j in range(cellCenters.shape[0]):
        cellDistance = jp.euclidean_distance(spinPosition, cellCenters[j,0:3], fiberRotationReference[0,:], 'cell')
        if cellDistance < cellCenters[j,3]:
            KeyCell = j
            break
    spinInCell_i[i] = KeyCell
    spinInFiber_i[i] = KeyFiber
    return
