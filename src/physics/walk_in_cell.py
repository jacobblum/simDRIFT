import numba 
from numba import jit, cuda
import math
from src.jp import random, linalg
import operator


@numba.cuda.jit(nopython=True)
def _diffusion_in_cell(i, 
                       random_states, 
                       cell_center, 
                       cell_radii, 
                       cell_step, 
                       fiber_centers, 
                       fiber_radii, 
                       fiber_directions, 
                       spin_positions, 
                       thetas, 
                       void,
                       A
                       ):
    """Simulated Brownian motion of a spin confined to within in a cell, implemented via random walk with rejection sampling for proposed steps beyond the cell membrane. Note that this implementation assumes zero exchange between compartments and is therefore only pysically-accurate for :math:`\Delta < {\\tau_{i}}` [1]_. 

    :param i: Absolute index of the current thread within the block grid
    :type i: int
    :param random_states: ``xoroshiro128p`` random states
    :type random_states: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param cell_center: Coordinates of the cell centers
    :type cell_center: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param cell_radii: Cell radius, in units of :math:`{\mathrm{μm}}`
    :type cell_radii: float
    :param cell_step: Distance travelled by resident spins for each time step :math:`\dd{t}`
    :type cell_step: float
    :param fiber_centers: Coordinates of the fiber centers
    :type fiber_centers: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param fiber_radii: Radii of each fiber type
    :type fiber_radii: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param fiber_directions: Orientation of each fiber type
    :type fiber_directions: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param spin_positions: Array containing the updated spin positions
    :type spin_positions: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param void: Logical condition that is ``True`` if ``fiber_configuration`` = ``Void`` and ``False`` otherwise
    :type void: bool
    
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

    previous_position = cuda.local.array(shape = 3, dtype = numba.float32)
    proposed_new_position = cuda.local.array(shape = 3, dtype = numba.float32)
    u3 = cuda.local.array(shape = 3, dtype= numba.float32)
    dynamic_fiber_center    = cuda.local.array(shape = 3, dtype = numba.float32)
    dynamic_fiber_direction = cuda.local.array(shape = 3, dtype = numba.float32)

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
                dynamic_fiber_center    = linalg.gamma(proposed_new_position, 
                                               fiber_directions[k,:], 
                                               thetas[k],
                                               dynamic_fiber_center,
                                               A[k]
                                               )
        
                dynamic_fiber_direction = linalg.d_gamma__d_t(proposed_new_position,
                                                      fiber_directions[k,:], 
                                                      thetas[k],
                                                      dynamic_fiber_direction,
                                                      A[k]
                                                      )
        
                for kk in range(dynamic_fiber_center.shape[0]): 
                    dynamic_fiber_center[kk] = dynamic_fiber_center[kk] + fiber_centers[k, kk]

                dFv = linalg.dL2(proposed_new_position, 
                                 dynamic_fiber_center, 
                                 dynamic_fiber_direction, 
                                 True
                                )

                if dFv < fiber_radii[k]:
                    is_in_fiber = True
                    break
        if operator.and_(not(is_in_fiber), not(is_not_in_cell)):
            invalid_step = False 
    
    for k in range(proposed_new_position.shape[0]): spin_positions[i,k] = proposed_new_position[k]

        



    

