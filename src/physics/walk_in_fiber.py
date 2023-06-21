import numba 
from numba import jit, cuda
from src.jp import random, linalg


@numba.cuda.jit(nopython=True,parallel=True)
def _diffusion_in_fiber(i, random_states, fiber_center, fiber_radius, fiber_direction, fiber_step, spin_positions):
    r"""
    
    Rejection-sample random steps of a spin residing within a fiber in the imaging voxel. Note that the rejection criteria is if the spin steps outside of the fiber. Thus, the 
    spin is confined to the fiber. This is reasonable so long as the diffusion times simulated does not exceed the intra-cellular pre-exchangle lifetime of the fiber. 
    
    Args:
        i (int): The absolute position of the current thread in the entire grid of blocks
        random_states (numba.cuda.cudadrv.devicearray.DeviceNDArray): xoroshiro128p random states
        fiber_center (numba.cuda.cudadrv.devicearray.DeviceNDArray): cordinates of the fiber centers
        fiber_radius (float): radii of the fibers
        fiber_direction (numba.cuda.cudadrv.devicearray.DeviceNDArray): directions of the fiber centers
        fiber_step (float): step size
        spin_positions (numba.cuda.cudadrv.devicearray.DeviceNDArray): array to write updated spin positions to   
        void (bool): void configuration 
    
    Shapes:
        random_states: (n_walkers,) where n_walkers is an input parameter denoting the number of spins in the ensemble
        fiber_center: (3,)
        fiber_direction: (3,)
        spin_positions: (n_walkers, 3) where n_walkers is an input parameter denoting the number of spins in the ensemble  

    References:
    [1] Yang DM, Huettner JE, Bretthorst GL, Neil JJ, Garbow JR, Ackerman JJH. Intracellular water preexchange lifetime in neurons and astrocytes. Magn Reson Med. 2018 Mar;79(3):1616-1627. doi: 10.1002/mrm.26781. Epub 2017 Jul 4. PMID: 28675497; PMCID: PMC5754269.
    
    """

    eps = numba.float32(1e-3)
    previous_position = cuda.local.array(shape = 3, dtype = numba.float32)
    proposed_new_position = cuda.local.array(shape = 3, dtype = numba.float32)

    distance = fiber_radius + eps
    while(distance > fiber_radius):
        proposed_new_position = random.random_on_S2(random_states, 
                                                     proposed_new_position, 
                                                     i)
        
        for k in range(proposed_new_position.shape[0]):
            previous_position[k] = spin_positions[i, k]
            proposed_new_position[k] = previous_position[k] + (fiber_step * proposed_new_position[k])
        distance = linalg.dL2(proposed_new_position, fiber_center, fiber_direction, True)
    for k in range(proposed_new_position.shape[0]): spin_positions[i,k] = proposed_new_position[k]







