import numpy as np
import numba
from numba import jit, cuda
from src.jp import random, linalg
import math
import operator


@numba.cuda.jit(nopython=True, parallel=True)
def _diffusion_in_water(i,
                        random_states,
                        fiber_centers,
                        fiber_directions,
                        fiber_radii,
                        cell_centers,
                        cell_radii,
                        spin_positions,
                        step, 
                        thetas,
                        curvature_params):
    """Simulated Brownian motion of a spin confined to the extra-cellular/axonal water, implemented via random walk with rejection sampling for proposed steps into cells or fibers. Note that this implementation assumes zero exchange between compartments and is therefore only pysically-accurate for :math:`\Delta < {\\tau_{i}(\text{cells})}` and :math:`\Delta < {\\tau_{i}(\text{fibers})}` [1]_. 

    :param i: Absolute index of the current thread within the block grid
    :type i: int
    :param random_states: ``xoroshiro128p`` random states
    :type random_states: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param fiber_centers: Coordinates of the fiber centers
    :type fiber_centers: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param fiber_directions: Orientation of each fiber type
    :type fiber_directions: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param fiber_radii: Radii of each fiber type
    :type fiber_radii: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param cell_centers: Coordinates of the cell centers
    :type cell_centers: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param cell_radii: Cell radius, in units of :math:`{\mathrm{μm}}`
    :type cell_radii: float
    :param spin_positions: Array containing the updated spin positions
    :type spin_positions: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param step: Distance travelled by resident spins for each time step :math:`\dd{t}`
    :type step: float
       
    **Shapes**
        :random_states: 
            (n_walkers,) where n_walkers is an input parameter denoting the number of spins in the ensemble
        :cell_center: 
            (3,)
        :fiber_centers: 
            (n_fibers x n_fibers, 3) where n_fibers is computed in a manner such that the fibers occupy the supplied fiber fraction of the imaging voxel
        :fiber_radii: 
            (n_fibers x n_fibers, ) where n_fibers is computed in a manner such that the fibers occupy the supplied fiber fraction of the imaging voxel
        :fiber_directions: 
            (n_fibers x n_fibers, 3) where n_fibers is computed in a manner such that the fibers occupy the supplied fiber fraction of the imaging voxel
        :spin_positions: 
            (n_walkers, 3) where n_walkers is an input parameter denoting the number of spins in the ensemble  
    """

    previous_position = cuda.local.array(shape=3, dtype=numba.float32)
    proposed_new_position = cuda.local.array(shape=3, dtype=numba.float32)
    dynamic_fiber_center    = cuda.local.array(shape = 3, dtype = numba.float32)
    dynamic_fiber_direction = cuda.local.array(shape = 3, dtype = numba.float32)

    u3 = cuda.local.array(shape=3, dtype=numba.float32)

    for j in range(u3.shape[0]):
        u3[j] = 1/math.sqrt(3) * 1.

    invalid_step = True
    while invalid_step:
        stepped_into_fiber = False
        stepped_into_cell  = False
        proposed_new_position = random.random_on_S2(random_states,
                                                    proposed_new_position,
                                                    i)

        for k in range(proposed_new_position.shape[0]):
            previous_position[k] = spin_positions[i, k]
            proposed_new_position[k] = previous_position[k] + (step*proposed_new_position[k])

        for j in range(fiber_centers.shape[0]):
                dynamic_fiber_center = linalg.gamma(proposed_new_position, 
                                                    fiber_directions[j,:], 
                                                    thetas[j],
                                                    dynamic_fiber_center,
                                                    curvature_params[j, :]
                                                    )
        
                dynamic_fiber_direction = linalg.d_gamma__d_t(proposed_new_position,
                                                              fiber_directions[j,:], 
                                                              thetas[j],
                                                              dynamic_fiber_direction,
                                                              curvature_params[j, :]
                                                              )
        
                for k in range(dynamic_fiber_center.shape[0]): 
                    dynamic_fiber_center[k] = dynamic_fiber_center[k] + fiber_centers[j, k]

                dFv = linalg.dL2(proposed_new_position, 
                                 dynamic_fiber_center, 
                                 dynamic_fiber_direction, 
                                 True
                                )

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
