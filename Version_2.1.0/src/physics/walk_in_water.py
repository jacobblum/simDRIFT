import numpy as np
import numba
from numba import jit, cuda
from jp import random, linalg
import math
import operator


@numba.cuda.jit(nopython=True, parallel=True)
def _diffusion_in_water(i,random_states,fiber_centers,fiber_directions,fiber_radii,cell_centers,cell_radii,spin_positions,step):
    r"""
    
    Rejection-sample random steps of a spin residing within the extra cellular and extra fiber compartment (water). Note that the rejection criteria is if the spin steps into any of the voxels constituent microstructure.
    Thus, the spin is confined to the water. This is reasonable so long as the diffusion times simulated does not exceed the intra-cellular pre-exchangle lifetime of the fibers and cells. 
    
    Args:
        i (int): The absolute position of the current thread in the entire grid of blocks
        random_states (numba.cuda.cudadrv.devicearray.DeviceNDArray): xoroshiro128p random states
        fiber_centers (numba.cuda.cudadrv.devicearray.DeviceNDArray): coordinates of the fiber centers
        fiber_radii (numba.cuda.cudadrv.devicearray.DeviceNDArray): radii of the fibers
        fiber_directions (numba.cuda.cudadrv.devicearray.DeviceNDArray): directions of the fibers
        cell_centers (numba.cuda.cudadrv.devicearray.DeviceNDArray): coordinates of the cell centers
        cell_radii (numba.cuda.cudadrv.devicearray.DeviceNDArray): radii of the cells
        spin_positions (numba.cuda.cudadrv.devicearray.DeviceNDArray): array to write updated positions to   
        step (float): step size in the water
       
    Shapes:
        random_states: (n_walkers,) where n_walkers is an input parameter denoting the number of spins in the ensemble
        fiber_centers: (n_fibers x n_fibers, 3) where n_fibers is computed in a manner such that the fibers occupy the supplied fiber fraction of the imaging voxel
        fiber_radii: (n_fibers x n_fibers, ) where n_fibers is computed in a manner such that the fibers occupy the supplied fiber fraction of the imaging voxel
        fiber_directions: (n_fibers x n_fibers, 3) where n_fibers is computed in a manner such that the fibers occupy the supplied fiber fraction of the imaging voxel
        cell_centers: (n_cells, 3) where n_cells is computed in a manner such that the fibers occupy the supplied fiber fraction of the imaging voxel
        cell_radii: (n_cells, ) where n_cells is computed in a manner such that the fibers occupy the supplied fiber fraction of the imaging voxel
        spin_positions: (n_walkers, 3) where n_walkers is an input parameter denoting the number of spins in the ensemble  

    References:
    [1] Yang DM, Huettner JE, Bretthorst GL, Neil JJ, Garbow JR, Ackerman JJH. Intracellular water preexchange lifetime in neurons and astrocytes. Magn Reson Med. 2018 Mar;79(3):1616-1627. doi: 10.1002/mrm.26781. Epub 2017 Jul 4. PMID: 28675497; PMCID: PMC5754269.
    
    """

    previous_position = cuda.local.array(shape=3, dtype=numba.float32)
    proposed_new_position = cuda.local.array(shape=3, dtype=numba.float32)

    u3 = cuda.local.array(shape=3, dtype=numba.float32)

    for j in range(u3.shape[0]):
        u3[j] = 1/math.sqrt(3) * 1.

    invalid_step = True
    while invalid_step:

        stepped_into_fiber = False
        stepped_into_cell = False
        proposed_new_position = random.random_on_S2(random_states,
                                                    proposed_new_position,
                                                    i)

        for k in range(proposed_new_position.shape[0]):
            previous_position[k] = spin_positions[i, k]
            proposed_new_position[k] = previous_position[k] + (step*proposed_new_position[k])

        for k in range(fiber_centers.shape[0]):
            dFv = linalg.dL2(proposed_new_position, fiber_centers[k, :], fiber_directions[k, :], True)
            if dFv < fiber_radii[k]:
                stepped_into_fiber = True
                break

        if not (stepped_into_fiber):
            for k in range(cell_centers.shape[0]):
                dC = linalg.dL2(proposed_new_position, cell_centers[k, :], u3, False)
                if dC < cell_radii[k]:
                    stepped_into_cell = True
                    break

        if operator.and_(not (stepped_into_cell), not (stepped_into_fiber)):
            invalid_step = False

    for k in range(proposed_new_position.shape[0]):
        spin_positions[i, k] = proposed_new_position[k]
    return
