import numpy as np 
import numba 
from numba import jit, cuda
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
import jp


@cuda.jit(device=True)
def _diffusion_in_cell(gpu_index, rng_states, spin_positions, spin_in_cell_at_index, cell_centers, fiber_centers, rotation_reference, dt, void):
    D = numba.float32(3.0)
    Step = numba.float32(math.sqrt(6*D*dt))
    previous_position = cuda.local.array(shape = 3, dtype = numba.float32)
    proposed_new_position = cuda.local.array(shape = 3, dtype = numba.float32)
  
    distance_cell = numba.float32(0.0)
    distance_fiber = numba.float32(0.0)
    invalid_step = True
    while invalid_step:
        is_in_fiber = False
        is_not_in_cell = False
        proposed_new_position = jp.randomDirection(rng_states, proposed_new_position, gpu_index)
        for k in range(proposed_new_position.shape[0]):
            previous_position[k] = spin_positions[gpu_index, k]
            proposed_new_position[k] = previous_position[k] + (Step*proposed_new_position[k])
        distance_cell = jp.euclidean_distance(proposed_new_position, cell_centers[spin_in_cell_at_index,0:3], rotation_reference[0,:], 'cell')
       
        if distance_cell > cell_centers[spin_in_cell_at_index,3]:
            is_not_in_cell = True

        if (not(is_not_in_cell)) & (not(void)):
            for k in range(fiber_centers.shape[0]):
                rotation_index = int(fiber_centers[k, 4])
                distance_fiber = jp.euclidean_distance(proposed_new_position, fiber_centers[k,0:3], rotation_reference[rotation_index,:], 'fiber')
                if distance_fiber < fiber_centers[k,3]:
                    is_in_fiber = True
                    break
        if (not(is_in_fiber)) & (not(is_not_in_cell)):
            invalid_step = False 
    for k in range(proposed_new_position.shape[0]): spin_positions[gpu_index,k] = proposed_new_position[k]

        



    

