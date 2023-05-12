import numba 
from numba import jit, cuda
from jp import random, linalg


@numba.cuda.jit(nopython=True,parallel=True)
def _diffusion_in_fiber(i, 
                        random_states, 
                        fiber_center, 
                        fiber_radius, 
                        fiber_direction, 
                        fiber_step,
                        spin_positions):
    

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







