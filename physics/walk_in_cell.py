import numba 
from numba import jit, cuda
import math
from jp import random, linalg
import operator


@numba.cuda.jit(nopython=True,parallel=True)
def _diffusion_in_cell(i, 
                       random_states, 
                       cell_center, 
                       cell_radii, 
                       cell_step, 
                       fiber_centers, 
                       fiber_radii,
                       fiber_directions, 
                       spin_positions, 
                       void):

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

        



    

