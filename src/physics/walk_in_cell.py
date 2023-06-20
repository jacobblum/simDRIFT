import numba 
from numba import jit, cuda
import math
from jp import random, linalg
import operator


@numba.cuda.jit(nopython=True)
def _diffusion_in_cell(i, random_states, cell_center, cell_radii, cell_step, fiber_centers, fiber_radii, fiber_directions, spin_positions, void):
    r"""
    Rejection-sample random steps of a spin residing within a cell in the imaging voxel. Note that the rejection criteria is if the spin steps outside of the cell. Thus, the 
    spin is confined to the cell. This is reasonable so long as the diffusion times simulated does not exceed the intra-cellular pre-exchangle lifetime. 
    
    Args:
        i (int): The absolute position of the current thread in the entire grid of blocks
        random_states (numba.cuda.cudadrv.devicearray.DeviceNDArray): xoroshiro128p random states
        cell_center (numba.cuda.cudadrv.devicearray.DeviceNDArray): cordinates of the cell center
        cell_radii (float): the radius of the cell (um)
        cell_step (float): step size
        fiber_centers (numba.cuda.cudadrv.devicearray.DeviceNDArray): cordinates of the fiber centers
        fiber_radii (numba.cuda.cudadrv.devicearray.DeviceNDArray): radii of the fibers
        fiber_directions (numba.cuda.cudadrv.devicearray.DeviceNDArray): directions of the fiber centers
        spin_positions (numba.cuda.cudadrv.devicearray.DeviceNDArray: array to write updated spin positions to
        void (bool): void configuration 
    
    Shapes:
        random_states: (n_walkers,) where n_walkers is an input parameter denoting the number of spins in the ensemble
        cell_center: (3,)
        fiber_centers: (n_fibers x n_fibers, 3) where n_fibers is computed in a manner such that the fibers occupy the supplied fiber fraction of the imaging voxel
        fiber_radii (n_fibers x n_fibers, ) where n_fibers is computed in a manner such that the fibers occupy the supplied fiber fraction of the imaging voxel
        fiber_directions: (n_fibers x n_fibers, 3) where n_fibers is computed in a manner such that the fibers occupy the supplied fiber fraction of the imaging voxel
        spin_positions: (n_walkers, 3) where n_walkers is an input parameter denoting the number of spins in the ensemble  

    References:
    [1] Yang DM, Huettner JE, Bretthorst GL, Neil JJ, Garbow JR, Ackerman JJH. Intracellular water preexchange lifetime in neurons and astrocytes. Magn Reson Med. 2018 Mar;79(3):1616-1627. doi: 10.1002/mrm.26781. Epub 2017 Jul 4. PMID: 28675497; PMCID: PMC5754269.
    """

    previous_position = cuda.local.array(shape = 3, dtype = numba.float32)
    proposed_new_position = cuda.local.array(shape = 3, dtype = numba.float32)
    u3 = cuda.local.array(shape = 3, dtype= numba.float32)

    for j in range(u3.shape[0]):
        u3[j] = 1/math.sqrt(3) * 1.

    invalid_step = True
    while invalid_step:
        is_in_fiber = False
        is_not_in_cell = False
        proposed_new_position = random.random_on_S2(random_states, 
                                                    proposed_new_position, 
                                                    i)
        
        for k in range(proposed_new_position.shape[0]):
            previous_position[k] = spin_positions[i, k]
            proposed_new_position[k] = previous_position[k] + (cell_step*proposed_new_position[k])
        dC = linalg.dL2(proposed_new_position, cell_center, u3, False)
        if dC > cell_radii:
            is_not_in_cell = True
        
        if operator.and_(not(is_not_in_cell), not(void)):
            for k in range(fiber_centers.shape[0]):
                dFv = linalg.dL2(proposed_new_position, fiber_centers[k,:], fiber_directions[k,:], True)
                if dFv < fiber_radii[k]:
                    is_in_fiber = True
                    break
        if operator.and_(not(is_in_fiber), not(is_not_in_cell)):
            invalid_step = False 
    
    for k in range(proposed_new_position.shape[0]): spin_positions[i,k] = proposed_new_position[k]

        



    

