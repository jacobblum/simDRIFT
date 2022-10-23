import numpy as np 
import numba 
from numba import jit, cuda
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
import jp

@cuda.jit(device=True)
def _diffusion_in_extra_environment(gpu_index, rng_states, spin_positions, fiber_centers, cell_centers, rotation_reference, dt):
    D = numba.float32(3.0)
    Step = numba.float32(math.sqrt(6*D*dt))
    previous_position = cuda.local.array(shape = 3, dtype = numba.float32)
    proposed_new_position = cuda.local.array(shape = 3, dtype = numba.float32)
    distance_cell = numba.float32(0.0)
    distance_fiber = numba.float32(0.0)
    invalid_step = True
    while invalid_step:
        
        
        stepped_into_fiber = False
        stepped_into_cell = False

        proposed_new_position = jp.randomDirection(rng_states, proposed_new_position, gpu_index)
        
        for k in range(proposed_new_position.shape[0]):
            previous_position[k] = spin_positions[gpu_index,k]
            proposed_new_position[k] = previous_position[k] + (Step*proposed_new_position[k])
        
     
        for k in range(fiber_centers.shape[0]):
            fiber_rotation_index = int(fiber_centers[k,4])
            distance_fiber = jp.euclidean_distance(proposed_new_position, fiber_centers[k,0:3], rotation_reference[fiber_rotation_index,:], 'fiber')
            if distance_fiber < fiber_centers[k,3]:
                stepped_into_fiber = True
                break
        if not(stepped_into_fiber):
            for k in range(cell_centers.shape[0]):
                distance_cell = jp.euclidean_distance(proposed_new_position, cell_centers[k,0:3], rotation_reference[0,:], 'cell')
                if distance_cell < cell_centers[k,3]:
                    stepped_into_cell = True
                    break
        if (not(stepped_into_cell)) & (not(stepped_into_fiber)):
            invalid_step = False
    for k in range(proposed_new_position.shape[0]): spin_positions[gpu_index, k] = proposed_new_position[k]
    return


        