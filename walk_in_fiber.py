from concurrent.futures import thread
from operator import xor
import numpy as np 
import numba 
from numba import jit, cuda
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
import jp as jp



@numba.cuda.jit(nopython=True,parallel=True)
def _diffusion_in_fiber(gpu_index, rng_states, spin_in_fiber_at_index, spin_positions, fiber_centers, fiber_rotation_reference, dt):
    D = numba.float32(fiber_centers[spin_in_fiber_at_index,5])
    Step = numba.float32(math.sqrt(6.0*D*dt))
    previous_position = cuda.local.array(shape = 3, dtype = numba.float32)
    proposed_new_position = cuda.local.array(shape = 3, dtype = numba.float32)

    current_fiber_radius = numba.float32(fiber_centers[spin_in_fiber_at_index,3])
    fiber_rotation_index = int(fiber_centers[spin_in_fiber_at_index,4])
    
    distance = current_fiber_radius + .001
    while(distance > current_fiber_radius):
        proposed_new_position = jp.randomDirection(rng_states, proposed_new_position, gpu_index)
        for k in range(proposed_new_position.shape[0]):
            previous_position[k] = spin_positions[gpu_index, k]
            proposed_new_position[k] = previous_position[k] + (Step * proposed_new_position[k])
        distance = jp.euclidean_distance(proposed_new_position, fiber_centers[spin_in_fiber_at_index, 0:3], fiber_rotation_reference[fiber_rotation_index,:], 'fiber')
    for k in range(proposed_new_position.shape[0]): spin_positions[gpu_index,k] = proposed_new_position[k]







