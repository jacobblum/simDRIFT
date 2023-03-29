from concurrent.futures import thread
from operator import xor
import numpy as np 
import numba 
from numba import jit, cuda
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math

@cuda.jit(device = True)
def randomDirection(rng_states, an_array, thread_id):
    sum = 0
    for i in range(an_array.shape[0]):
        an_array[i] = xoroshiro128p_normal_float32(rng_states, thread_id)
        sum = sum + math.pow(an_array[i], 2)
    for i in range(an_array.shape[0]):
        an_array[i] = an_array[i]/math.sqrt(sum)
    return an_array


@cuda.jit(device = True)
def euclidean_distance(an_array1, an_array2, rotation_reference, fiber):
    dist = 0
    proj = 0
    
    if fiber == 'fiber':  
        for i in range(an_array1.shape[0]):
            dist += math.pow(an_array1[i]-an_array2[i],2)
            proj += ((an_array1[i] - an_array2[i])*rotation_reference[i])    
        return math.sqrt(dist - math.pow(proj,2))
    if fiber == 'cell':
        for i in range(an_array1.shape[0]):
            dist += math.pow(an_array1[i]-an_array2[i],2)
        return math.sqrt(dist)

@cuda.jit(device = True)
def diffusion_step(an_array, step):
    i = cuda.grid(1)
    if i > an_array.shape[0]:
        return
    
    D = numba.float32(3.0)
    Step = numba.float32(D)
    prevPosition = cuda.local.array(shape = 3, dtype = numba.float32)
    newPosition = cuda.local.array(shape = 3, dtype = numba.float32)
    for j in range(prevPosition.shape[0]): 
        prevPosition[j] = an_array[i,j]
        newPosition[j] = prevPosition[j]+Step
        for _ in range(1000): 3.8*math.sqrt(_)
        an_array[i,j] = newPosition[j]

    
    